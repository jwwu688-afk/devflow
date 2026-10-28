"""Issue 分诊 Agent。

职责边界：输入一个 GitHub Issue（标题/正文/标签/assignees）+ 检索到的上下文
（相似 Issue、相关代码、项目文档、会话证据、团队成员画像），输出分诊结论：
分类、优先级、复杂度、建议负责人、重复候选、证据、行动清单和可回填的草稿。

在编排链里的位置：Planner 认为需要 Issue 视角时调度（task_type = "issue_analysis"）。
它的 conclusion 会进 synthesis 的「下一步」，suggested_owner 会被 Observer 校验
（是否在团队成员名单里，不在就发 owner_needs_review 警告）。

**双轨设计**（与 pr_review / ci_debug 同构）：
先算规则 fallback → 再调 LLM → 合并 → 校验，失败则静默退回 fallback。

本文件里最值得注意的不是模型调用，而是 **_merge_with_fallback 里的几个「安全阀」**：
模型很爱编造一个不存在的负责人、或把明显的 Bug 说成 "Question"，
这些错误会直接影响派活，所以合并阶段要用规则强行纠正。
"""

import re
from typing import Any

from app.schemas.analysis import IssueAnalysisOutput, IssueDrafts
from app.services.agents.base import BaseAgent
from app.services.llm.client import LLMClient
from app.services.llm.prompts import ISSUE_ANALYSIS_PROMPT


class IssueAgent(BaseAgent):
    """Issue 分诊：规则兜底 + LLM 增强 + 负责人校验。"""

    name = "issue_analyst_agent"  # 注意：类名是 IssueAgent，但 Agent 标识是 issue_analyst_agent，
    # Planner 里的 agent_name 也是这个值，两者必须一致，否则任务找不到执行者
    description = "分析 GitHub Issue 的分诊结论、优先级、复杂度、负责人、证据和下一步动作。"
    available_tools = ["get_issue_detail", "search_similar_issues", "search_related_code", "search_project_docs", "read_thread_context"]

    def __init__(self, llm: LLMClient | None = None) -> None:
        """构造，可注入 LLM 客户端。"""
        # 注入式 LLM，单测可替换为假客户端
        self.llm = llm or LLMClient()

    async def run(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """规则兜底 → LLM 增强 → 合并校验。

        输入：input_data 为 Issue 详情（number/title/body/labels/assignees/state）；
              context 提供 duplicate_candidates、code_references、project_documents、
              conversation_evidence、team_members。
        输出：IssueAnalysisOutput 的 dict。
        失败路径：LLM 为空或校验失败 → 返回 fallback，不抛异常。
        """
        fallback = self._fallback_analysis(input_data, context)
        # 唯一一次 LLM 调用（花钱点）：Issue 正文 + 上下文一起喂给模型
        llm_output = await self.llm.chat_json(ISSUE_ANALYSIS_PROMPT, {"issue": input_data, "context": context})
        if llm_output:
            try:
                # 合并不是简单覆盖，还包含负责人合法性校验等修正逻辑
                normalized = self._merge_with_fallback(llm_output, fallback, input_data, context)
                return IssueAnalysisOutput.model_validate(normalized).model_dump()
            except Exception:
                # 模型输出结构不可控，校验失败等同于「没答」
                pass
        return fallback

    def _fallback_analysis(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """纯规则分诊，产出完整 IssueAnalysisOutput。

        顺序即依赖：先分类/定级/估复杂度（都基于文本+标签），
        再据此推负责人、判重复、下结论，最后由结论生成清单与草稿。
        """
        # 分类与定级都只看「标题+正文」+ 标签，转小写便于关键词匹配
        text = f"{input_data.get('title', '')} {input_data.get('body', '')}".lower()
        body = str(input_data.get("body") or "")
        labels = [label.lower() for label in input_data.get("labels", [])]
        category = self._classify_category(text, labels)
        priority = self._classify_priority(text, labels)
        complexity = self._estimate_complexity(text, labels)
        owner, owner_reason = self._suggest_owner(input_data, text, context)
        # 重复候选来自检索层，这里只做「结构校验 + 分数归一」，不重新算相似度
        duplicates = self._validated_duplicates(context.get("duplicate_candidates", []))
        conclusion, conclusion_reason = self._choose_conclusion(input_data, text, labels, category, complexity, duplicates)
        checklist = self._build_checklist(conclusion, category)
        drafts = self._build_drafts(input_data, conclusion, checklist)
        evidence = self._collect_evidence(input_data, context)
        confidence = self._estimate_confidence(body, evidence, duplicates, owner)
        return IssueAnalysisOutput(
            summary=self._build_summary(input_data, conclusion),
            conclusion=conclusion,
            conclusion_reason=conclusion_reason,
            category=category,
            priority=priority,
            complexity=complexity,
            suggested_owner=owner,
            owner_reason=owner_reason,
            duplicate_candidates=duplicates,
            evidence=evidence,
            checklist=checklist,
            # action_items 与 checklist 同源：历史调用方读的是 action_items，新代码读 checklist，
            # 两个字段都填是为了不破坏任何一侧
            action_items=checklist,
            drafts=drafts,
            confidence=confidence,
        ).model_dump()

    def _merge_with_fallback(self, output: dict[str, Any], fallback: dict[str, Any], input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """合并模型输出与规则结果，并纠正几类高风险错误。

        安全阀清单（每条都对应一个真实踩过的坑）：
        1. checklist / action_items 互为兜底 —— 模型常只填一个；
        2. suggested_owner 必须存在于 assignees 或 team_members，否则退回规则建议
           （派给不存在的人比不派人更糟）；
        3. 模型给了泛称（backend/frontend/待分配）而规则给出了真人时，用规则的
           （泛称无法执行，等于没派）；
        4. 模型把明显 Bug 判成 "Question" 时，采信规则的分类。
        """
        merged = {**fallback, **output}
        # 阀 1：两个同义字段互相补齐
        if not merged.get("checklist") and merged.get("action_items"):
            merged["checklist"] = merged["action_items"]
        if not merged.get("action_items") and merged.get("checklist"):
            merged["action_items"] = merged["checklist"]
        if not merged.get("evidence"):
            merged["evidence"] = fallback["evidence"]
        merged["evidence"] = self._dedupe_evidence(merged.get("evidence"))
        if not merged.get("duplicate_candidates"):
            merged["duplicate_candidates"] = fallback["duplicate_candidates"]
        # drafts 结构可能被模型写成字符串，校验前先确认是 dict，否则整份输出会被 pydantic 拒掉
        if not isinstance(merged.get("drafts"), dict):
            merged["drafts"] = fallback["drafts"]
        # 阀 2：负责人白名单校验（"待分配"/空串视为合法，因为那代表「确实没人」）
        if not self._is_allowed_owner(str(merged.get("suggested_owner") or ""), input_data, context):
            merged["suggested_owner"] = fallback["suggested_owner"]
            merged["owner_reason"] = fallback["owner_reason"]
        # 阀 3：模型给泛称、规则给了真人 → 用真人
        if self._should_prefer_fallback_owner(str(merged.get("suggested_owner") or ""), str(fallback.get("suggested_owner") or "")):
            merged["suggested_owner"] = fallback["suggested_owner"]
            merged["owner_reason"] = fallback["owner_reason"]
        # 阀 4：Question 是最「安全」的分类，模型倾向往这里躲；规则若能明确分类就覆盖它。
        # 用 in 集合而不是 != "Question"，是为了只在规则给出具体类型时才覆盖
        if merged.get("category") == "Question" and fallback.get("category") in {"Bug", "Feature", "Documentation", "Ops", "Refactor", "Test"}:
            merged["category"] = fallback["category"]
        return merged

    def _should_prefer_fallback_owner(self, current_owner: str, fallback_owner: str) -> bool:
        """模型给的是「岗位名」而不是人名时，改用规则给出的具体成员。

        generic_owners 里的都是「方向名」，写进 Issue 也无法指派给具体的人。
        """
        generic_owners = {"", "待分配", "unassigned", "backend", "frontend", "ci", "docs"}
        return fallback_owner not in generic_owners and current_owner.lower() in generic_owners

    def _is_allowed_owner(self, owner: str, input_data: dict[str, Any], context: dict[str, Any]) -> bool:
        """校验负责人是否「真实存在」。

        合法来源有两个：Issue 上已有的 assignees（现成的分配事实），
        以及配置好的团队成员名单。两者都没有的名字视为模型编造。
        「待分配」/空串算合法——它表达的是「确实没人」，不是错误答案。
        """
        normalized_owner = owner.strip().lower()
        if normalized_owner in {"", "待分配", "unassigned"}:
            return True
        # 用集合而不是列表做 in 判断：成员可能有几十个，但这里也只是一次性构建，主要图清晰
        allowed = {str(item).strip().lower() for item in input_data.get("assignees") or [] if str(item).strip()}
        members = context.get("team_members") or []
        if isinstance(members, list):
            for member in members:
                if isinstance(member, dict) and str(member.get("name") or "").strip():
                    allowed.add(str(member.get("name")).strip().lower())
        return normalized_owner in allowed

    def _build_summary(self, input_data: dict[str, Any], conclusion: str) -> str:
        """一句话摘要：编号 + 标题 + 结论。摘要会被 Observer 和 synthesis 直接引用。"""
        number = input_data.get("number")
        # 标题缺失时用 "Untitled Issue"，避免出现 "Issue #12  的建议结论是" 这种双空格
        title = str(input_data.get("title") or "Untitled Issue").strip()
        prefix = f"Issue #{number}" if number else "Issue"
        return f"{prefix} {title} 的建议结论是：{conclusion}。"

    def _classify_category(self, text: str, labels: list[str]) -> str:
        """按关键词判类型。规则**有序**，顺序即优先级。

        Bug 排第一：回归/报错类信息最需要准确识别，误判成 Feature 会漏掉修复优先级。
        Question 排最后：它是「兜底型」分类，放在前面会吞掉其他判断。
        全不命中时默认 Feature（新建 Issue 大多是需求）。
        """
        # 标签和正文拼在一起匹配：标签是人工标注的高置信信号，正文是补充
        joined = " ".join(labels) + " " + text
        rules = [
            ("Bug", ["bug", "error", "fail", "500", "crash", "exception", "broken", "regression"]),
            ("Documentation", ["doc", "documentation", "readme", "guide", "文档", "说明"]),
            ("Test", ["test", "pytest", "coverage", "单测", "测试"]),
            ("Ops", ["ops", "ci", "workflow", "deploy", "docker", "infra", "环境", "权限"]),
            ("Refactor", ["refactor", "cleanup", "重构", "technical debt"]),
            ("Feature", ["feature", "enhancement", "proposal", "需求", "新增"]),
            ("Question", ["question", "how to", "why does", "咨询", "疑问"]),
        ]
        for category, tokens in rules:
            if any(token in joined for token in tokens):
                return category
        return "Feature"

    def _classify_priority(self, text: str, labels: list[str]) -> str:
        """按关键词定 P0/P1/P3，默认 P2。

        只判「极端情况」：P0（数据丢失/全量不可用）和 P3（typo/文案）有明确信号，
        中间地带统一 P2。理由是规则无法可靠区分 P1 与 P2 的细微差别，
        与其猜错不如给个中间值，让人工调。
        """
        joined = " ".join(labels) + " " + text
        if any(token in joined for token in ["p0", "sev0", "critical", "security", "data loss", "数据丢失", "全量不可用"]):
            return "P0"
        if any(token in joined for token in ["p1", "sev1", "crash", "cannot login", "500", "生产", "阻塞", "blocker"]):
            return "P1"
        if any(token in joined for token in ["p3", "minor", "polish", "typo", "文案", "低优"]):
            return "P3"
        return "P2"

    def _estimate_complexity(self, text: str, labels: list[str]) -> str:
        """估工作量档位 S/M/L/XL。

        两个依据：关键词（架构/迁移直接 XL）和正文长度（280 / 1400 字符是经验分界）。
        长度阈值只是粗略代理——描述越详细通常意味着改动越大，但长描述也可能只是
        bug 报告写得细，所以长度只在中段起决定作用。
        """
        joined = " ".join(labels) + " " + text
        if any(token in joined for token in ["xl", "architecture", "migration", "rewrite", "跨模块", "架构"]):
            return "XL"
        if any(token in joined for token in ["large", "complex", "integration", "schema", "多模块"]):
            return "L"
        # 短文本 + 文案类关键词 = S（改个错别字/文档）
        if len(text) < 280 and any(token in joined for token in ["typo", "copy", "文案", "doc"]):
            return "S"
        # 兜底按长度分：1400 字符以下算 M，之上算 L
        return "M" if len(text) < 1400 else "L"

    def _suggest_owner(self, input_data: dict[str, Any], issue_text: str, context: dict[str, Any]) -> tuple[str, str]:
        """负责人建议，按「确定性从高到低」逐级降级。

        顺序：已分配的人 → 团队画像匹配 → 按方向给出「待分配 + 原因」。
        关键点：**拿不准时宁可返回「待分配」+ 解释**，也不要瞎猜一个人。
        因为 owner_reason 会展示给用户看，说清「为什么没派」比给个错误的人有用。
        """
        assignees = input_data.get("assignees") or []
        # 已经有负责人就直接沿用，不覆盖人的既有决定
        if assignees:
            owner = str(assignees[0])
            return owner, f"Issue 已分配给 {owner}，优先沿用现有负责人。"

        # signal_text 比 issue_text 更宽：除了 Issue 正文，还把检索到的代码/文档/会话证据拼进来，
        # 这样「改动涉及 electron 相关代码」也能影响负责人判断
        signal_text = self._owner_signal_text(issue_text, context)
        team_owner, team_reason = self._suggest_owner_from_team(issue_text, signal_text, context)
        if team_owner:
            return team_owner, team_reason or "团队画像与 Issue 关键词匹配。"

        # 走到这里说明匹配失败，只能按方向解释原因。四种措辞对应四种不同的用户动作
        area = self._infer_area(signal_text)
        members = context.get("team_members") or []
        if not isinstance(members, list) or not members:
            if area != "待分配":
                return "待分配", f"当前项目还没有添加团队成员；这个 Issue 更像 {area} 方向，请先在团队面板添加对应开发成员后再分配。"
            return "待分配", "当前项目还没有添加团队成员，请先在团队面板添加开发成员后再分配。"
        if area != "待分配":
            return "待分配", f"未找到与 {area} 方向匹配的团队成员，请先补充团队成员画像后再分配。"
        return "待分配", "当前 Issue 与团队画像、代码/文档证据的匹配度不足，需要人工确认负责人。"

    def _suggest_owner_from_team(self, issue_text: str, signal_text: str, context: dict[str, Any]) -> tuple[str | None, str | None]:
        """用团队画像给每个成员打分，取最高分（必须 > 0）。

        打分权重刻意拉开差距：
        - 单个词命中 +1（技术栈词，如 "react"），零散、噪声大；
        - 领域规则命中 +8（如「Issue 提到 electron」且「成员擅长 electron」），
          这种是强信号，一条足以超过零散词命中。
        同分时保留先出现的成员（把成员顺序当作默认优先级，不引入额外随机性）。
        """
        members = context.get("team_members") or []
        if not isinstance(members, list):
            return None, None

        best_name = None
        best_score = 0
        best_reasons: list[str] = []
        for member in members:
            if not isinstance(member, dict):
                continue
            name = str(member.get("name") or "").strip()
            # techStack 和 tech_stack 两种命名都支持：前端表单和后端模型的历史差异
            profile = " ".join(
                str(member.get(field) or "").lower()
                for field in ["role", "strengths", "techStack", "tech_stack"]
            )
            score = 0
            reasons: list[str] = []
            for term in self._profile_terms(profile):
                # len(term) >= 2 过滤掉单字（如 "js" 会被切成... 其实 js 是 2 字符），
                # 单字中文命中率太高，几乎人人都匹配
                if len(term) >= 2 and term in signal_text:
                    score += 1
                    reasons.append(term)
            for domain, issue_terms, profile_terms in self._owner_domain_rules():
                # 双向匹配：Issue 侧命中该领域 且 画像侧也命中该领域，才算领域匹配
                if any(term in signal_text for term in issue_terms) and any(term in profile for term in profile_terms):
                    score += 8
                    reasons.append(domain)
            # 严格大于：并列时保留先出现的（best_score 初始 0，所以必须有分才会被选中）
            if name and score > best_score:
                best_name = name
                best_score = score
                best_reasons = reasons
        if best_name and best_score > 0:
            # 只取前 4 个匹配理由：理由太多反而看不出重点，去重避免同一词重复出现
            reason_text = "、".join(dict.fromkeys(best_reasons[:4])) or "团队画像"
            return best_name, f"团队成员画像与 Issue/代码/文档信号匹配：{reason_text}，匹配分 {best_score}。"
        return None, None

    def _infer_area(self, issue_text: str) -> str:
        """粗判 Issue 属于哪个方向，仅用于在「无人可派」时解释原因。"""
        area_keywords = {
            # frontend 词表明显更长：桌面化/electron 相关需求曾经常被误判，特意补了大量同义词
            "backend": ["api", "database", "token", "server", "fastapi", "postgres", "auth", "jwt"],
            "frontend": ["ui", "page", "button", "react", "next", "css", "layout", "browser", "页面", "按钮", "electron", "desktop", "桌面化", "桌面端"],
            "ci": ["ci", "workflow", "actions", "pytest", "lint", "build", "deploy"],
            "docs": ["readme", "doc", "文档", "说明"],
        }
        for area, keywords in area_keywords.items():
            if any(keyword in issue_text for keyword in keywords):
                return area
        return "待分配"

    def _owner_signal_text(self, issue_text: str, context: dict[str, Any]) -> str:
        """拼出「用于匹配负责人的文本」：Issue 正文 + 检索到的代码/文档/会话片段。

        每个来源只取前 6 条：这是匹配负责人用的，不需要全量，取太多会让
        泛化词（如 "api"）到处命中，反而稀释信号。
        """
        parts = [issue_text]
        for key in ["code_references", "project_documents", "conversation_evidence"]:
            items = context.get(key)
            if not isinstance(items, list):
                continue
            for item in items[:6]:
                if isinstance(item, dict):
                    # 标题、片段、正文都拼进去：不同检索源填的字段不一样
                    parts.extend([str(item.get("title") or ""), str(item.get("snippet") or ""), str(item.get("content") or "")])
        text = " ".join(parts).lower()
        # 领域扩展：出现「桌面化」时补一串 electron 生态关键词。
        # 原因是团队画像里写的是 "electron/node"，而 Issue 里写的是中文「桌面化」，
        # 不扩展就永远匹配不上（这是一次真实误判的补丁）
        if any(term in text for term in ["桌面化", "桌面端", "桌面应用", "electron"]):
            text += " electron desktop windows macos javascript node frontend"
        return text

    def _profile_terms(self, profile: str) -> list[str]:
        """把成员画像切成可匹配的词元。

        先把「擅长/熟悉/技术栈」这类修饰词替换成空格（它们本身没有区分度，
        留着会让每个成员都命中），再用正则切出英文词/数字/中文串。
        """
        normalized = profile.replace("擅长", " ").replace("熟悉", " ").replace("技术栈", " ")
        return re.findall(r"[a-zA-Z0-9_+#.\u4e00-\u9fff]+", normalized.lower())

    def _owner_domain_rules(self) -> list[tuple[str, list[str], list[str]]]:
        """领域匹配规则：(领域名, Issue 侧关键词, 画像侧关键词)。

        与单个技术词命中的区别在于「双向确认」：只有 Issue 和画像同时落在同一领域
        才给高分，避免「Issue 全是技术词、成员画像也全是技术词」导致的假匹配。
        """
        return [
            (
                "桌面化/electron",
                ["桌面化", "桌面端", "桌面应用", "electron", "desktop", "windows", "macos", "native app"],
                ["electron", "desktop", "桌面", "windows", "macos", "tauri", "javascript", "js", "node", "frontend", "前端"],
            ),
            (
                "后端/API",
                ["api", "database", "token", "server", "fastapi", "postgres", "auth", "jwt"],
                ["api", "database", "server", "fastapi", "postgres", "auth", "jwt", "后端"],
            ),
            (
                "前端/UI",
                ["ui", "page", "button", "react", "next", "css", "layout", "browser", "页面", "按钮"],
                ["ui", "react", "next", "css", "browser", "frontend", "前端"],
            ),
            (
                "CI/工程化",
                ["ci", "workflow", "actions", "pytest", "lint", "build", "deploy"],
                ["ci", "workflow", "actions", "pytest", "lint", "build", "deploy", "工程化"],
            ),
        ]

    def _validated_duplicates(self, candidates: Any) -> list[dict[str, Any]]:
        """校验并规整重复 Issue 候选，最多 5 条。

        相似度由检索层算好（score），这里只做三件事：夹到 0~1、
        丢掉缺 id 或标题的残条、统一成固定三个字段。
        """
        if not isinstance(candidates, list):
            return []
        rows = []
        for item in candidates[:5]:
            if not isinstance(item, dict):
                continue
            # issue_id / source_id 两种字段名都兼容（不同检索来源的命名差异）
            issue_id = str(item.get("issue_id") or item.get("source_id") or "").strip()
            title = str(item.get("title") or "").strip()
            try:
                # 夹紧：检索层偶尔会给出 >1 或负数的分数
                score = max(0.0, min(1.0, float(item.get("score") or 0)))
            except (TypeError, ValueError):
                score = 0.0
            # id 和标题缺一不可：缺 id 无法关联回原 Issue，缺标题无法展示
            if issue_id and title:
                rows.append({"issue_id": issue_id, "title": title, "score": score})
        return rows

    def _choose_conclusion(
        self,
        input_data: dict[str, Any],
        text: str,
        labels: list[str],
        category: str,
        complexity: str,
        duplicates: list[dict[str, Any]],
    ) -> tuple[str, str]:
        """给出分诊结论，返回 (结论, 理由)。

        顺序即优先级，且**越靠前越不可协商**：
        已关闭/无效 → 高相似重复 → 范围过大拆分为 → 信息不足待澄清 → 进入开发。
        这条顺序决定了用户看到的最终建议，改动它会直接改变产品行为。
        """
        state = str(input_data.get("state") or "").lower()
        joined = " ".join(labels) + " " + text
        if state == "closed" or any(token in joined for token in ["invalid", "wontfix", "won't fix", "not planned", "无效", "不处理"]):
            return "关闭", "Issue 已关闭或带有无效/不处理信号，建议归档并保留关闭说明。"
        # 0.86 是「高度相似」的经验阈值：低于它只能算参考，高于它才敢建议合并
        if any(float(item.get("score") or 0) >= 0.86 for item in duplicates):
            return "合并到已有 Issue", "相似历史 Issue 的匹配度较高，建议合并上下文，避免重复推进。"
        if complexity in {"XL"} or self._looks_like_multi_scope(text):
            return "拆分", "Issue 范围偏大或包含多个独立目标，拆分后更容易排期和验收。"
        if self._needs_clarification(text, category):
            return "等待澄清", "当前描述缺少影响范围、复现/验收条件或期望结果，直接进入开发风险较高。"
        return "进入开发", "Issue 信息基本可执行，优先确认验收口径后进入开发。"

    def _looks_like_multi_scope(self, text: str) -> bool:
        """判断 Issue 是否包含多个独立目标。

        两个信号取或：列表项数量（>=5 个 bullet/顿号/分号说明列了一堆事），
        或出现「同时/以及/跨模块」这类并列词。
        """
        separators = text.count("\n-") + text.count("\n*") + text.count("、") + text.count(";")
        broad_terms = ["同时", "以及", "并且", "多个", "all of", "migration", "architecture", "跨模块"]
        return separators >= 5 or any(term in text for term in broad_terms)

    def _needs_clarification(self, text: str, category: str) -> bool:
        """判断信息是否不足以开工。

        按类型给不同标准，因为「够不够」本身取决于要做什么：
        - 通用：正文 < 80 字符基本等于没说清；
        - Bug：要有复现步骤或影响范围，二者有一即可（有些 bug 一眼能看出影响，不必写复现）；
        - Feature：要有验收标准/价值说明，或正文够长（< 220 字符且无验收标准 → 待澄清）。
        """
        if len(text.strip()) < 80:
            return True
        if category == "Bug":
            has_repro = any(token in text for token in ["reproduce", "steps", "复现", "步骤", "expected", "actual", "期望", "实际"])
            has_impact = any(token in text for token in ["impact", "影响", "scope", "范围", "blocking", "阻塞"])
            return not (has_repro or has_impact)
        if category == "Feature":
            has_acceptance = any(token in text for token in ["acceptance", "验收", "criteria", "why", "value", "价值", "场景"])
            return not has_acceptance and len(text.strip()) < 220
        return False  # 其他类型（文档/重构/运维）不设门槛，避免过度拦截

    def _collect_evidence(self, input_data: dict[str, Any], context: dict[str, Any]) -> list[dict[str, Any]]:
        """汇总证据列表，最多 10 条。

        Observer 会用 evidence 是否为空来判断「这个结论有没有依据」（evidence_gap 警告），
        所以即使什么都没检索到，也要把 Issue 自身作为一条证据塞进去——
        Issue 正文本身就是最基础的依据。
        """
        evidence = [
            {
                "source_type": "issue",
                "title": f"Issue #{input_data.get('number')}: {input_data.get('title')}",
                # 标题+正文拼起来截到 320 字符，够判断主题又不至于让证据体膨胀
                "snippet": self._clip("\n".join(filter(None, [str(input_data.get("title") or ""), str(input_data.get("body") or "")])), 320),
                "reference": str(input_data.get("id") or input_data.get("number") or ""),
                "confidence": 1.0,  # Issue 自身是原始数据而非推断，置信度固定满值
            }
        ]
        # 相似 Issue 作为证据时，confidence 直接用相似度分数：让读者知道这条参考有多可靠
        for duplicate in self._validated_duplicates(context.get("duplicate_candidates", []))[:3]:
            evidence.append(
                {
                    "source_type": "similar_issue",
                    "title": duplicate["title"],
                    "snippet": f"相似度 {duplicate['score']:.2f}，可作为重复/历史处理经验参考。",
                    "reference": duplicate["issue_id"],
                    "confidence": duplicate["score"],
                }
            )
        # 三类检索结果映射到统一的 source_type，下游不必关心它来自哪个检索器
        for key, source_type in [
            ("code_references", "code"),
            ("project_documents", "document"),
            ("conversation_evidence", "conversation"),
        ]:
            evidence.extend(self._evidence_from_context(context.get(key), source_type))
        evidence.extend(self._team_evidence(context.get("team_members")))
        # 先整体去重再截断：同一个片段可能从多个来源返回，去重后再取 10 条才不浪费名额
        return self._dedupe_evidence(evidence)[:10]

    def _dedupe_evidence(self, items: Any) -> list[dict[str, Any]]:
        """按 (来源类型, 引用, 标题, 片段前 180 字) 四元组去重。

        片段只取前 180 字参与比较：不同检索器返回的同一段内容，
        头部一致而尾部（比如截断位置）可能不同，用全长比较反而去不掉重。
        没有 snippet 的条目直接丢弃——证据的核心就是那段引用内容。
        """
        if not isinstance(items, list):
            return []
        rows: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str, str]] = set()
        for item in items:
            if not isinstance(item, dict) or not item.get("snippet"):
                continue
            key = (
                str(item.get("source_type") or ""),
                str(item.get("reference") or ""),
                str(item.get("title") or ""),
                str(item.get("snippet") or "")[:180],
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append(item)
        return rows

    def _evidence_from_context(self, items: Any, default_source_type: str) -> list[dict[str, Any]]:
        """把一类检索结果转成统一证据结构，每类最多 4 条。

        字段名做多重兜底（title/path/source_id、snippet/content），
        因为不同检索器（代码搜索、文档库、会话历史）返回的字段并不统一。
        """
        if not isinstance(items, list):
            return []
        rows = []
        for item in items[:4]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or item.get("path") or item.get("source_id") or default_source_type).strip()
            snippet = self._clip(str(item.get("snippet") or item.get("content") or ""), 320)
            # 没有正文片段的检索结果没有证据价值，跳过
            if not snippet:
                continue
            score = item.get("score")
            try:
                # 有分数就夹到 0~1；没分数保持 None（表示「不知道相似度」，
                # 而不是 0，否则会被误读成「完全不相关」）
                confidence = max(0.0, min(1.0, float(score))) if score is not None else None
            except (TypeError, ValueError):
                confidence = None
            rows.append(
                {
                    "source_type": str(item.get("source_type") or default_source_type),
                    "title": title,
                    "snippet": snippet,
                    "reference": str(item.get("reference") or item.get("source_id") or item.get("id") or ""),
                    "confidence": confidence,
                }
            )
        return rows

    def _team_evidence(self, members: Any) -> list[dict[str, Any]]:
        """把团队成员画像也当证据：它是「负责人建议」这个结论的依据。

        confidence 为 None：画像不是「相似度匹配」意义上的证据，
        给个数字反而会误导（让人以为算过匹配度，其实匹配分在 owner_reason 里）。
        """
        if not isinstance(members, list) or not members:
            return []
        rows = []
        # 只取前 3 个成员：证据列表上限 10 条，团队成员不该占太多名额
        for member in members[:3]:
            if not isinstance(member, dict):
                continue
            name = str(member.get("name") or "").strip()
            if not name:
                continue
            # 用「；」拼接多个画像字段，过滤空字段避免出现连续分隔符
            profile = "；".join(
                str(member.get(field) or "").strip()
                for field in ["role", "strengths", "techStack", "tech_stack"]
                if str(member.get(field) or "").strip()
            )
            rows.append(
                {
                    "source_type": "team_member",
                    "title": f"团队成员：{name}",
                    "snippet": profile or "团队成员资料可作为负责人分配依据。",
                    "reference": name,
                    "confidence": None,
                }
            )
        return rows

    def _build_checklist(self, conclusion: str, category: str) -> list[str]:
        """按结论生成行动清单。

        结论不同，清单完全是不同的事（澄清要问信息、拆分要切任务、关闭要写说明），
        所以没有统一模板，只能按结论分支。只有「进入开发」走基础清单。
        """
        if conclusion == "等待澄清":
            return [
                "请补充问题背景、影响范围和用户场景。",
                "Bug 类问题补充复现步骤、期望结果、实际结果和日志/截图。",
                "Feature 类需求补充验收标准和非目标范围。",
                "澄清后重新运行 Issue Agent 分析。",
            ]
        if conclusion == "合并到已有 Issue":
            return [
                "确认相似 Issue 是否覆盖当前诉求。",
                "把当前 Issue 的新增信息整理为评论追加到已有 Issue。",
                "为当前 Issue 标记 duplicate/关联关系。",
                "经人工确认后关闭当前重复 Issue。",
            ]
        if conclusion == "拆分":
            return [
                "按用户价值、模块边界或交付顺序拆成 2-5 个子任务。",
                "为每个子任务补充验收标准、依赖关系和建议负责人。",
                "确认最小可交付范围，优先排第一个可独立验证的子任务。",
                "把拆分说明作为评论回填到原 Issue。",
            ]
        if conclusion == "关闭":
            return [
                "确认关闭原因是无效、重复、已完成还是不计划处理。",
                "补充一条简短关闭评论，说明依据和替代路径。",
                "经人工确认后更新标签或关闭 Issue。",
            ]
        # 以下为「进入开发」的通用清单
        baseline = [
            "确认 Issue 验收标准和不做范围。",
            "定位相关代码/文档位置并补充设计备注。",
            "创建开发分支并实现最小修复/功能增量。",
            "补充自动化测试或手工验证步骤。",
            "提交 PR 并在描述中关联该 Issue。",
        ]
        # Bug 额外插入一条「先写失败测试」：修复类工作必须先能复现，否则改了也不知道对不对。
        # insert(2, ...) 把它放在「定位代码」之后、「创建分支」之前
        if category == "Bug":
            baseline.insert(2, "先写一个能复现问题的失败测试或最小复现脚本。")
        return baseline

    def _build_drafts(self, input_data: dict[str, Any], conclusion: str, checklist: list[str]) -> IssueDrafts:
        """按结论生成可回填的草稿文本。

        草稿只是「文本」，不会自动发到 GitHub——真正的写操作要过 SafetyAgent 与人工确认。
        """
        number = input_data.get("number")
        title = str(input_data.get("title") or "该 Issue")
        clarification = None
        task_breakdown = None
        if conclusion == "等待澄清":
            clarification = (
                f"针对 Issue #{number} {title}，目前信息还不足以直接进入开发。"
                "请补充：影响范围、复现步骤/使用场景、期望结果、实际结果，以及是否有日志、截图或相关版本信息。"
            )
        # 拆分和直接开发都要给任务拆解，但内容来源不同：都用清单前 5 条
        if conclusion in {"拆分", "进入开发"}:
            task_breakdown = "建议下一步拆解：\n" + "\n".join(f"- {item}" for item in checklist[:5])
        # 重复 Issue 不需要拆解，需要的是「合并 + 留痕 + 关闭」的说明
        if conclusion == "合并到已有 Issue":
            task_breakdown = (
                f"建议将 Issue #{number} 的新增信息合并到相似历史 Issue，并在当前 Issue 留下 duplicate 说明后再关闭。"
            )
        return IssueDrafts(clarification_comment=clarification, task_breakdown=task_breakdown)

    def _estimate_confidence(self, body: str, evidence: list[dict[str, Any]], duplicates: list[dict[str, Any]], owner: str) -> float:
        """按信息充分度估置信度，区间 0.35~0.86。

        四个加分项对应四种「信息变多」的情形：正文足够详细、证据条数够、有重复参照、
        负责人能落到具体人（说明画像匹配成功）。上限 0.86 与 PR/CI 侧保持一致，
        让 synthesis 里不同 Agent 的置信度可以横向比较。
        """
        score = 0.45
        if len(body.strip()) >= 120:
            score += 0.12  # 120 字符是「描述算详细」的经验线
        if len(evidence) >= 3:
            score += 0.14  # 三条以上证据才算「有交叉验证」
        if duplicates:
            score += 0.08  # 有历史相似 Issue 可参考
        if owner != "待分配":
            score += 0.08  # 负责人能落地
        return round(max(0.35, min(0.86, score)), 2)

    def _clip(self, text: str, limit: int) -> str:
        """截断文本并加省略号。

        limit - 1 而不是 limit：把省略号也算进长度，保证结果不超过 limit。
        rstrip 避免截断处留下空格再接省略号（看起来像两个空格）。
        """
        text = (text or "").strip()
        return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
