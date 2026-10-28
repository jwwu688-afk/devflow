"""PR 审查 Agent。

职责边界：输入一个 PR 的元信息 + 变更文件（含 patch）+ 已有 review 评论 + CI 摘要，
输出风险点、阻塞项、检查清单、测试建议和一句「合入建议」。

在编排链里的位置：Planner 在 PR 相关诉求下调度它；它的 merge_recommendation 会被
synthesis_agent 直接引用为总体判断，blocking_issues 会被 Observer 升级为 blocker。

**双轨设计**（与 ci_debug_agent 同构）：
1. 先用纯规则算出一份 fallback（_fallback_review），保证永远有结果；
2. 再调 LLM 拿更细致的判断，用 _merge_with_fallback 与 fallback 合并；
3. LLM 返回空 / 校验失败 → 静默退回 fallback，不抛异常。

**LLM 成本控制**：diff 走 chunk_pr_files 切块后再进 prompt（见 diff_chunks），
不直接塞原始 files。大 PR 的 patch 动辄几万字符，不分块会顶爆上下文窗口。
"""

from typing import Any

from app.schemas.analysis import PRReviewOutput
from app.services.agents.base import BaseAgent
from app.services.llm.client import LLMClient
from app.services.llm.prompts import PR_REVIEW_PROMPT
from app.services.rag.chunking import chunk_pr_files


class PRReviewAgent(BaseAgent):
    """PR diff 审查：规则兜底 + LLM 增强。"""

    name = "pr_review_agent"
    description = "总结 PR diff，并生成风险点、检查清单和测试建议。"
    available_tools = ["get_pull_request_detail", "get_pull_request_files", "chunk_diff"]

    def __init__(self, llm: LLMClient | None = None) -> None:
        """构造，可注入 LLM 客户端。"""
        # 注入式 LLM：单测可传假客户端，避免真实调用
        self.llm = llm or LLMClient()

    async def run(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """切块 → 规则兜底 → LLM 增强 → 合并校验。

        输入：input_data 含 number/title/body/state/author/files/comments，
              context 可含 ci_summary、code_references、project_documents、conversation_evidence。
        输出：PRReviewOutput 的 dict。
        失败路径：LLM 为空或校验失败直接返回 fallback（规则结果），工作流不中断。
        """
        # 先切块：一份 PR 的 patch 可能上千行，切块既控制 prompt 长度，
        # 也让规则侧能按块数判断「这个 PR 是否体量过大」
        diff_chunks = chunk_pr_files(input_data.get("files", []))
        fallback = self._fallback_review(input_data, context, diff_chunks)
        # 唯一一次 LLM 调用（花钱点）：把 PR 元信息、切好的 diff 块、检索到的上下文一起喂进去
        llm_output = await self.llm.chat_json(PR_REVIEW_PROMPT, {"pull_request": input_data, "diff_chunks": diff_chunks, "context": context})
        if llm_output:
            try:
                return PRReviewOutput.model_validate(self._merge_with_fallback(llm_output, fallback)).model_dump()
            except Exception:
                # 模型输出结构不可控，校验失败等同于「没答」，静默走 fallback
                pass
        return fallback

    def _fallback_review(self, input_data: dict[str, Any], context: dict[str, Any], diff_chunks: list[dict[str, Any]]) -> dict[str, Any]:
        """纯规则审查，产出一份结构完整的 PRReviewOutput。

        规则只做「可机械判定」的事：敏感路径、测试文件缺失、diff 体量、失败 CI、未处理评论。
        语义级判断（这段逻辑到底对不对）交给 LLM。
        """
        files = input_data.get("files", [])
        comments = input_data.get("comments", [])
        filenames = [item.get("filename", "") for item in files]
        # 增删行数用于判断「改动量」，注意用 int(... or 0)：GitHub 偶尔给 None
        total_additions = sum(int(item.get("additions") or 0) for item in files)
        total_deletions = sum(int(item.get("deletions") or 0) for item in files)
        # 敏感路径词表：这些目录/文件一旦改动就要重点看，是「风险启发式」而非精确规则。
        # 宁可多标（多标只是多让 reviewer 看一眼），不能漏标安全/支付/迁移这类致命路径
        risky_files = [
            name
            for name in filenames
            if any(token in name.lower() for token in ["auth", "security", "migration", "schema", "payment", "permission", "config", "ci", "workflow"])
        ]
        # 测试文件词表：用来判断「这次改动有没有配套测试」
        test_files = [
            name
            for name in filenames
            if any(token in name.lower() for token in ["test", "spec", "__tests__", "pytest", "playwright"])
        ]
        # patch 总字符数：diff 体量的另一个度量（行数之外），用于触发「建议分块 review」
        total_patch_chars = sum(len(item.get("patch") or "") for item in files)
        risk_points = []
        blocking_issues = []
        review_findings = []
        # ci_summary 来自上游（不是本 Agent 自己去查）：这里只消费「关联仓库还有几个失败 run」
        ci_summary = context.get("ci_summary") or {}
        failed_ci = int(ci_summary.get("failed_runs") or 0)
        if failed_ci:
            # P1 = 最严重：失败 CI 直接挡合并，所以同时进 blocking_issues（会被 Observer 升级为 blocker）
            message = f"关联仓库仍有 {failed_ci} 个失败 CI，需要先确认是否由当前 PR 引入或阻塞合并。"
            blocking_issues.append(message)
            review_findings.append(self._finding("P1", "失败 CI 阻塞合并", message, "先修复或解释失败 CI，再重新评估合入。"))
        if risky_files:
            # P2：敏感文件只是「需要重点验证」，不必然挡合并，所以进 risk_points 不进 blocking_issues
            message = f"高敏感文件需要重点审查：{', '.join(risky_files[:5])}"
            risk_points.append(message)
            review_findings.append(self._finding("P2", "敏感文件变更需要重点验证", message, "补充代码审查记录、回归测试或人工验证说明。"))
        if comments:
            # 已有评论未确认 = 可能还有未解决的讨论，保守当阻塞处理
            message = f"已有 {len(comments)} 条 review comment，需要确认是否都已处理。"
            risk_points.append(message)
            blocking_issues.append(message)
            review_findings.append(self._finding("P2", "Review 评论尚需确认", message, "逐条回应并确认评论已解决后再放行。"))
        # 双阈值取或：12000 字符偏「文本量」，8 个 chunk 偏「文件分散度」，
        # 两个都可能意味着 reviewer 会看不全，任一触发就提示
        if total_patch_chars > 12000 or len(diff_chunks) > 8:
            message = "Diff 较大，建议按模块分块 Review，避免遗漏跨文件影响。"
            risk_points.append(message)
            review_findings.append(self._finding("P3", "Diff 体量较大", message, "按模块拆分审查路径，必要时补充分阶段验证。", blocking=False))
        # 缺测试的判定带三个前提：确实有文件改动、没有测试文件、且（碰了敏感路径 或 改动超过 250 行）。
        # 小改动没带测试是常态，一律报会变成噪音；250 行是「改得够多，值得要求测试」的经验线
        if files and not test_files and (risky_files or total_additions + total_deletions > 250):
            message = "当前变更缺少明显测试文件，合并前需要补充自动化测试或明确手工验证记录。"
            blocking_issues.append(message)
            review_findings.append(self._finding("P2", "缺少测试覆盖证明", message, "补充自动化测试，或留下可复核的手工验证记录。"))
        # 两个兜底：保证前端渲染时「风险点」和「发现项」永远不为空，
        # 否则界面会出现空板块，用户会以为是加载失败
        if not risk_points:
            risk_points.append("需要确认边界条件和异常路径是否有测试覆盖")
        if not review_findings:
            review_findings.append(
                self._finding(
                    "P3",
                    "未发现明确阻断项",
                    "基于当前 diff、评论和 CI 摘要，未发现明确 P1/P2 阻断项。",
                    "按 checklist 完成最终人工确认。",
                    blocking=False,
                )
            )

        merge_recommendation, recommendation_reason = self._choose_recommendation(input_data, files, risk_points, blocking_issues, failed_ci)
        # plan / executed_steps 是「透明化」字段：告诉用户我做了哪几步、实际拿到了多少数据，
        # 与计算逻辑无关，方便用户判断结论够不够扎实
        plan = [
            "理解 PR 标题、描述、分支、变更文件和关联上下文。",
            "按模块拆分 diff 风险，识别敏感路径和大体量改动。",
            "读取相关源代码、项目文档和历史会话证据。",
            "检查 CI 状态、失败记录和现有 review comments。",
            "形成合入建议、阻塞项、测试建议和 review checklist。",
        ]
        executed_steps = [
            f"已读取 PR #{input_data.get('number')}，状态 {input_data.get('state') or 'unknown'}，作者 {input_data.get('author') or 'unknown'}。",
            f"已扫描 {len(files)} 个变更文件，合计 +{total_additions} / -{total_deletions}。",
            f"已识别 {len(risky_files)} 个敏感文件、{len(test_files)} 个测试相关文件、{len(comments)} 条 review comment。",
            f"已纳入 {len(context.get('code_references') or [])} 条代码线索、{len(context.get('project_documents') or [])} 条项目文档、{failed_ci} 个失败 CI 信号。",
            f"已输出结论：{merge_recommendation}。",
        ]
        return PRReviewOutput(
            summary=f"PR #{input_data.get('number')} 修改了 {len(files)} 个文件，主要集中在 {', '.join(filenames[:3]) or '未提供文件'}。",
            plan=plan,
            executed_steps=executed_steps,
            merge_recommendation=merge_recommendation,
            recommendation_reason=recommendation_reason,
            review_findings=review_findings,
            # 只列前 5 个文件：key_changes 是摘要，列全量没有阅读价值
            key_changes=[f"更新 {name}" for name in filenames[:5]],
            # 无文件时换一句话：否则会出现「未发现风险」这种基于零信息的误导性结论
            risk_points=risk_points if files else ["缺少 changed files，风险判断有限"],
            blocking_issues=blocking_issues,
            review_checklist=["确认接口兼容性", "确认错误处理路径", "确认测试覆盖关键分支", "确认 review comments 已回应"],
            test_suggestions=self._test_suggestions(filenames, risky_files, failed_ci),
            # 优先展示敏感文件；没有敏感文件时退回变更文件列表
            files_need_attention=(risky_files or filenames)[:5],
            # 评论正文只取前 5 条且过滤空 body：空评论（纯 reaction/系统提示）没有信息量
            review_comments=[str(comment.get("body") or "").strip() for comment in comments[:5] if str(comment.get("body") or "").strip()],
            confidence=self._estimate_confidence(files, context, failed_ci, blocking_issues),
        ).model_dump()

    def _merge_with_fallback(self, output: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
        """合并 LLM 输出与规则兜底，并把两者捏成自洽的结论。

        三件事依次做：
        1. 空字段用 fallback 补（模型漏答的部分不丢）；
        2. 把 review_findings 规整成统一结构，再由它反推 blocking_issues；
        3. **一致性修正**：如果存在阻塞项但结论仍是「建议合入」，强制降级为「修改后合入」。
           第 3 步是防止模型自相矛盾的关键安全阀——结论段和阻塞项段冲突时，
           用户只会看结论，所以必须让结论反映阻塞项的存在。
        """
        merged = {**fallback, **output}
        # 列表字段：空列表一律视为「模型没答」，用规则结果顶上
        for key in [
            "plan",
            "executed_steps",
            "review_findings",
            "key_changes",
            "risk_points",
            "blocking_issues",
            "review_checklist",
            "test_suggestions",
            "files_need_attention",
            "review_comments",
        ]:
            if not merged.get(key):
                merged[key] = fallback[key]
        # 文本字段：空白串视为没答
        for key in ["merge_recommendation", "recommendation_reason", "summary"]:
            if not str(merged.get(key) or "").strip():
                merged[key] = fallback[key]
        # 模型的 review_findings 字段名/结构可能不规范，统一走一次规范化
        merged["review_findings"] = self._normalize_review_findings(merged.get("review_findings"), fallback)
        # 由 findings 反推 blocking_issues，并与模型给的那份取并集去重：
        # 这样「模型只说没问题但 findings 里有 P1」的情况也不会漏掉阻塞项
        merged["blocking_issues"] = self._blocking_issues_from_findings(merged.get("blocking_issues"), merged["review_findings"])
        # 安全阀：结论与阻塞项冲突时，以阻塞项为准
        if merged["blocking_issues"] and merged.get("merge_recommendation") == "建议合入":
            merged["merge_recommendation"] = "修改后合入"
            merged["recommendation_reason"] = "存在 P1/P2 审查发现，完成修复或确认后才能放行。"
        return merged

    def _finding(
        self,
        severity: str,
        title: str,
        evidence: str,
        required_action: str,
        *,
        blocking: bool | None = None,
    ) -> dict[str, Any]:
        """构造一条 review finding。

        blocking 默认由 severity 推导（P1/P2 即阻塞），但允许显式覆盖：
        像「Diff 体量较大」这种 P3 提醒不该挡合并，而某些 P2 也可能只是想引起注意。
        blocking 用 keyword-only（*）传参，避免位置参数误传成 required_action。
        """
        return {
            "severity": severity,
            "title": title,
            "evidence": evidence,
            "required_action": required_action,
            "blocking": severity in {"P1", "P2"} if blocking is None else blocking,
        }

    def _normalize_review_findings(self, items: Any, fallback: dict[str, Any]) -> list[dict[str, Any]]:
        """把形态不定的 findings 规整成统一结构。

        LLM 输出的字段名不固定（title/summary/message，evidence/reason/description，
        required_action/action/recommendation 都见过），这里用 or 串联逐个试，
        保证不同措辞都能落到同一个 schema 上，下游不必再做兼容。
        """
        # 模型给的不是非空 list 时直接用 fallback 的 findings，避免后面遍历出错
        source = items if isinstance(items, list) and items else fallback.get("review_findings") or []
        normalized: list[dict[str, Any]] = []
        for item in source:
            if not isinstance(item, dict):
                continue
            # severity 归一：大小写统一，非法值一律降为 P3（宁可低估严重度，也不要误报 P1）
            severity = str(item.get("severity") or "").strip().upper()
            if severity not in {"P1", "P2", "P3"}:
                severity = "P3"
            title = str(item.get("title") or item.get("summary") or item.get("message") or "").strip()
            evidence = str(item.get("evidence") or item.get("reason") or item.get("description") or "").strip()
            required_action = str(item.get("required_action") or item.get("action") or item.get("recommendation") or "").strip()
            # 标题和证据都空的条目没有任何信息量，直接丢弃
            if not title and not evidence:
                continue
            raw_blocking = item.get("blocking")
            # 模型给的 blocking 只在明确为 true 时采信；给字符串 "false" 之类一律按 severity 推导
            blocking = severity in {"P1", "P2"} if raw_blocking is None else raw_blocking is True
            normalized.append(
                self._finding(
                    severity,
                    # title/evidence 互相兜底：只有其一时另一个顶上；标题过长时用证据前 80 字
                    title or evidence[:80],
                    evidence or title,
                    required_action or "请补充可复核的处理说明。",
                    blocking=blocking,
                )
            )
        # 一条都没规整出来时退回 fallback，保证 findings 永不为空
        return normalized or fallback.get("review_findings") or []

    def _blocking_issues_from_findings(self, blocking_issues: Any, review_findings: list[dict[str, Any]]) -> list[str]:
        """把 findings 里的阻塞项拍平成字符串列表，与已有 blocking_issues 合并去重。

        为什么要拍平：Observer 和 synthesis 消费的是 blocking_issues（字符串列表），
        而不是结构化的 findings；两条链路并存时必须在合并点对齐格式。
        """
        rows = [str(item).strip() for item in blocking_issues or [] if str(item).strip()]
        for finding in review_findings:
            # 双重条件：P1/P2 或显式 blocking=true 都算阻塞（后者覆盖「P3 但人为标为阻塞」的情况）
            if finding.get("severity") in {"P1", "P2"} or finding.get("blocking"):
                action = str(finding.get("required_action") or "").strip()
                title = str(finding.get("title") or "").strip()
                # 拼成「标题：动作」，缺一半时只保留有的那半，避免出现孤零零的冒号
                message = f"{title}：{action}" if title and action else title or action
                if message:
                    rows.append(message)
        # dict.fromkeys 去重且保序：模型与规则的表述可能重复，但先出现的那条通常更具体
        return list(dict.fromkeys(rows))

    def _choose_recommendation(
        self,
        input_data: dict[str, Any],
        files: list[dict[str, Any]],
        risk_points: list[str],
        blocking_issues: list[str],
        failed_ci: int,
    ) -> tuple[str, str]:
        """决定合入建议，返回 (结论, 理由)。

        判定是有序短路，顺序即优先级，**改动顺序会改变对外结论**：
        PR 已关闭 → 描述含 WIP/blocked → 失败 CI → 没有 diff → 有阻塞项 → 有风险点 → 默认可合。
        越靠前的条件越「不可协商」：一个已关闭的 PR 无论 diff 多干净都不该建议合入。
        """
        state = str(input_data.get("state") or "").lower()
        # 只把标题+描述拼起来做关键词扫描，不扫 diff：diff 里出现 "wip" 这种词太正常了
        text = f"{input_data.get('title', '')} {input_data.get('body', '')}".lower()
        # 已关闭且无 merged_at = 被拒绝/放弃，直接给「拒绝」
        if state == "closed" and not input_data.get("merged_at"):
            return "拒绝", "PR 已关闭且没有合并记录，当前不建议继续合入。"
        # WIP/draft 等信号：作者自己说了还没准备好，即使技术上没问题也要等
        if any(token in text for token in ["do not merge", "wip", "draft", "blocked", "暂缓", "不要合并"]):
            return "暂缓", "PR 描述中存在 WIP/blocked/暂缓信号，需要等待作者更新。"
        # 失败 CI 优先于「有没有阻塞项」：CI 是客观门禁，不因代码看起来没问题而放行
        if failed_ci:
            return "暂缓", "存在失败 CI 信号，合并前需要先确认根因并恢复质量门禁。"
        # 没有 diff 就无法判断影响范围，只能暂缓（而不是默认放行）
        if not files:
            return "暂缓", "缺少 changed files，无法判断真实影响范围。"
        # 到这里问题都是「可修复的」，所以是「修改后合入」而不是「暂缓」
        if blocking_issues:
            return "修改后合入", "存在可修复的阻塞项，完成测试或说明补充后可重新评估。"
        # 从风险点文案里反查是否出现关键风险词——这里依赖上面生成 message 时的措辞
        # （"高敏感"/"较大"/"review comment"），改文案就要同步改这里
        if risk_points and any(token in " ".join(risk_points) for token in ["高敏感", "较大", "review comment"]):
            return "修改后合入", "PR 有明确风险点，建议先完成重点 review 和验证再合入。"
        return "建议合入", "未发现明显阻塞项，按 checklist 完成最终确认后可以合入。"

    def _test_suggestions(self, filenames: list[str], risky_files: list[str], failed_ci: int) -> list[str]:
        """按变更文件的路径特征给测试建议。

        用路径关键词猜技术栈（有 .tsx/next 就补前端检查，有 .py/api 就补后端测试），
        比读文件内容便宜得多；代价是猜错（比如纯文档 PR 也会命中 .md 里的关键词），
        所以第一条「运行相关单元测试和集成测试」永远存在，作为通用兜底。
        """
        suggestions = ["运行相关单元测试和集成测试"]
        # 把所有文件名拼成一个串再匹配，比逐个文件遍历更省事，也避免同一类建议重复添加
        lowered = " ".join(filenames).lower()
        if any(token in lowered for token in ["frontend", ".tsx", ".ts", "next", "react"]):
            suggestions.append("运行前端类型检查/构建，并覆盖关键页面交互")
        if any(token in lowered for token in ["backend", ".py", "api", "db"]):
            suggestions.append("运行后端 pytest，并覆盖接口或数据库边界")
        if risky_files:
            suggestions.append("对敏感文件补充回归测试或人工验证记录")
        if failed_ci:
            suggestions.append("先修复失败 CI，再重新运行完整流水线")
        # 去重保序：避免前端+后端都命中时产生重复条目
        return list(dict.fromkeys(suggestions))

    def _estimate_confidence(self, files: list[dict[str, Any]], context: dict[str, Any], failed_ci: int, blocking_issues: list[str]) -> float:
        """按「有多少可用信息」估置信度，区间 0.38~0.88。

        注意这里的信息量与结论好坏无关：发现一堆阻塞项也会略微加分（说明判断有依据），
        置信度衡量的是「这个结论有多少证据支撑」，不是「代码质量如何」。
        """
        score = 0.48
        if files:
            score += 0.12  # 拿到了 diff，这是最关键的信息
        if context.get("code_references"):
            score += 0.08  # 检索到了相关源码
        if context.get("project_documents"):
            score += 0.06  # 有项目文档作为规范依据
        if context.get("conversation_evidence"):
            score += 0.05  # 有历史会话上下文
        if failed_ci:
            score += 0.04  # CI 提供了客观信号
        if blocking_issues:
            score += 0.04  # 找到了具体问题点，说明分析有落点
        # 上限 0.88：纯启发式 + 模型判断不足以支撑更高确定性，留出人工复核空间
        return round(max(0.38, min(0.88, score)), 2)
