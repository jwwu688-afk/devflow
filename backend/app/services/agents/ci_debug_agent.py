"""CI 失败诊断 Agent。

职责边界：输入一次失败的 workflow run（job 列表 + 原始日志文本），输出失败类型、
根因判断、修复步骤，以及「是否阻塞合并」的结论。

在编排链里的位置：通常由 Planner 在检测到 failed CI 时调度，产出会被 synthesis_agent
的「下一步」直接引用（is_merge_blocking / debug_steps 两个字段最常被用到）。

**降级设计（本文件最重要的部分）**：
run() 先执行纯规则的 _fallback_debug()，再调 LLM，最后把两者合并。
这样无论模型超时、返回非 JSON、还是字段缺失，都有一份结构完整、内容可信的结果可以返回。
代价是每次都会先做一遍规则分析（开销很小，纯字符串处理），换来确定性兜底。

**LLM 调用点（花钱的一步）**：一次 chat_json，prompt 是 CI_DEBUG_PROMPT，
输入是整份 workflow_run（含日志）。日志通常很长，所以下面 _extract_first_error
会主动截断——这是本文件里唯一影响 token 成本的地方。
"""

import re
from typing import Any

from app.schemas.analysis import CIDebugOutput
from app.services.agents.base import BaseAgent
from app.services.llm.client import LLMClient
from app.services.llm.prompts import CI_DEBUG_PROMPT


class CIDebugAgent(BaseAgent):
    """分析 workflow 失败日志，给出根因与排查步骤（规则 + LLM 混合）。"""

    name = "ci_debug_agent"
    description = "分析 workflow 失败日志，并提出排查步骤。"
    available_tools = ["get_workflow_run", "get_workflow_logs", "extract_failed_log_sections"]

    def __init__(self, llm: LLMClient | None = None) -> None:
        """构造，可注入 LLM 客户端。"""
        # 允许注入 LLMClient：测试里可以塞假客户端，避免单测真的打模型、花钱且不确定
        self.llm = llm or LLMClient()

    async def run(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """先算规则兜底，再尝试 LLM，最后合并。

        输入：input_data 为 workflow run 详情（name/conclusion/status/jobs/logs_text 等），
              context 可带 recent_prs、code_references 作为额外线索。
        输出：CIDebugOutput 的 dict。
        失败路径：LLM 返回空、或返回内容过不了 pydantic 校验 → 直接返回 fallback，
                  **不抛异常**，保证编排层不会因为一次模型抽风而中断整条工作流。
        """
        # 先算兜底：这一步不依赖网络，任何时候都能产出完整结果
        fallback = self._fallback_debug(input_data, context)
        # 唯一一次 LLM 调用（花钱点）。把整份 run 数据喂进去，让模型自己挑重点
        llm_output = await self.llm.chat_json(CI_DEBUG_PROMPT, {"workflow_run": input_data, "context": context})
        if llm_output:
            try:
                # 用 fallback 补齐模型漏掉的字段后整体校验，校验不过就当模型没答
                return CIDebugOutput.model_validate(self._merge_with_fallback(llm_output, fallback)).model_dump()
            except Exception:
                # 这里刻意裸吞异常：模型输出不可控（少字段、类型错），
                # 任何校验失败都等价于「模型没给出可用答案」，落到下面的 fallback 即可
                pass
        return fallback

    def _fallback_debug(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """纯规则诊断，产出与 LLM 版本同构的完整结果。

        流程：找失败 job → 找失败 step → 分类失败类型 → 抽首个错误块 →
              抽相关文件 → 推根因 → 给修复步骤 → 判是否阻塞合并。
        """
        # 分类用的日志转小写（关键字匹配不区分大小写）；原始大小写版本留给错误块提取，
        # 因为错误信息里的路径/类名大小写是有意义的
        logs = (input_data.get("logs_text") or "").lower()
        raw_logs = str(input_data.get("logs_text") or "")
        # 先过滤出失败的 job：日志里所有 job 都会打印，只有失败的才值得分析
        failed_jobs = [
            job
            for job in input_data.get("jobs", [])
            if isinstance(job, dict) and job.get("conclusion") == "failure"
        ]
        # 再下钻到 step 级别：失败 step 名比 job 名更能指向具体命令
        failed_steps = [
            step.get("name")
            for job in failed_jobs
            for step in job.get("steps", [])
            if isinstance(step, dict) and step.get("conclusion") == "failure" and step.get("name")
        ]
        failure_type = self._classify_failure(logs)
        first_error = self._extract_first_error(raw_logs)
        related_files = self._extract_related_files(raw_logs, context)
        # 名字列表只取前 3 个用于展示：完整的失败 job 清单会把 summary 撑爆
        failed_job_names = [str(job.get("name")) for job in failed_jobs if job.get("name")]
        job_hint = f"失败 job：{', '.join(failed_job_names[:3])}。" if failed_job_names else ""
        step_hint = f"失败 step：{', '.join(failed_steps[:3])}。" if failed_steps else ""
        root_cause = self._infer_root_cause(failure_type, first_error, failed_job_names, failed_steps)
        fix_steps = self._fix_steps(failure_type, related_files)
        is_merge_blocking = self._is_merge_blocking(input_data, failure_type)
        # 阻塞与否必须给出理由：synthesis_agent 会把这句话直接写进备忘录
        blocking_reason = (
            "当前 workflow run 结论为 failure，属于合并前需要恢复的质量门禁。"
            if is_merge_blocking
            else "当前 workflow run 没有失败结论，不阻塞合并。"
        )
        # plan / executed_steps 是「给用户看我在干什么」的透明化字段，
        # 与真正的计算逻辑无关，改动它们不影响结论，只影响可读性
        plan = [
            "找出失败 job 和失败 step。",
            "提取首个关键错误块。",
            "判断失败类型：依赖、测试、构建、权限、环境变量或未知。",
            "搜索相关代码、测试文件和 workflow 配置。",
            "关联最近 PR 或变更模块。",
            "输出根因、修复步骤和是否阻塞合并。",
        ]
        # executed_steps 用「已…」句式回填实际拿到的数量，让用户能判断证据是否充分
        executed_steps = [
            f"已识别失败 job {len(failed_jobs)} 个：{', '.join(failed_job_names[:3]) or '未提供 job 明细'}。",
            f"已识别失败 step {len(failed_steps)} 个：{', '.join(failed_steps[:3]) or '未提供 step 明细'}。",
            f"已提取首个关键错误块：{first_error or '日志中未找到明确 error 块'}。",
            f"已分类失败类型为 {failure_type}，关联文件 {len(related_files)} 个。",
            f"已结合最近 PR 信号 {len(context.get('recent_prs') or [])} 条和代码/配置线索 {len(context.get('code_references') or [])} 条。",
            "已生成根因、修复步骤和阻塞判断。",
        ]
        return CIDebugOutput(
            # name 缺失时用 "CI run" 兜底，避免 summary 以 " 执行失败" 开头（空名字 + 空格）
            failure_summary=input_data.get("name", "CI run") + f" 执行失败。{job_hint}{step_hint}需要结合日志定位首个错误块。",
            failure_type=failure_type,
            plan=plan,
            executed_steps=executed_steps,
            first_error=first_error,
            root_cause=root_cause,
            possible_causes=self._possible_causes(failure_type),
            fix_steps=fix_steps,
            # debug_steps 固定四条：这是给用户的「动手顺序」，不随失败类型变化，
            # 具体类型的差异体现在 fix_steps 里
            debug_steps=["查看失败 job 的首个错误块", "本地复现对应命令", "检查最近 PR 修改的相关文件", "修复后重新运行失败 job 和完整 workflow"],
            related_files=related_files,
            is_merge_blocking=is_merge_blocking,
            blocking_reason=blocking_reason,
            confidence=self._estimate_confidence(first_error, failed_jobs, related_files, context),
        ).model_dump()

    def _merge_with_fallback(self, output: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
        """用 fallback 填补 LLM 输出的空缺，而不是整体二选一。

        合并策略有意区分字段类型：
        - 列表/字典字段：空列表视为「模型没答」，用 fallback 覆盖；
        - 文本字段：空白字符串视为没答；
        - failure_type：模型答 "unknown" 但规则能分类时，采信规则（分类是硬信号）。
        这样能保住模型写得好的部分，只补它漏的部分。
        """
        merged = {**fallback, **output}
        # 第一类：结构性字段，只判断「是否为空」
        for key in ["plan", "executed_steps", "possible_causes", "fix_steps", "debug_steps", "related_files"]:
            if not merged.get(key):
                merged[key] = fallback[key]
        # 第二类：文本字段，空串和纯空格都算没答
        for key in ["failure_summary", "failure_type", "root_cause", "blocking_reason"]:
            if not str(merged.get(key) or "").strip():
                merged[key] = fallback[key]
        # 第三类：模型退化成 unknown 时，规则分类通常更准（毕竟它是基于日志字面匹配的）
        if merged.get("failure_type") == "unknown" and fallback.get("failure_type") != "unknown":
            merged["failure_type"] = fallback["failure_type"]
        # first_error 是「证据」，模型经常不给；而 synthesis 会引用它，所以必须补上
        if not merged.get("first_error"):
            merged["first_error"] = fallback["first_error"]
        return merged

    def _classify_failure(self, logs: str) -> str:
        """按关键字把失败归类，返回固定的几个类型之一。

        规则列表**有序**，顺序即优先级，这一点很关键：
        权限/环境类问题常伴随 test 失败（例如测试因为没权限读文件而断言失败），
        因此 permission/environment 必须排在 test 前面，否则会被误判成测试问题。
        """
        ordered_rules = [
            ("permission", ["permission", "denied", "forbidden", "eacces", "权限"]),
            ("environment", ["missing env", "environment variable", "secret", "not set", "环境变量"]),
            ("dependency", ["npm err", "pip install", "dependency", "module not found", "no matching distribution", "cannot find module", "依赖"]),
            ("lint", ["eslint", "lint", "ruff", "flake8", "biome"]),
            ("build", ["build failed", "compilation", "webpack", "vite", "next build", "tsc", "type error", "构建"]),
            ("test", ["pytest", "assert", "test failed", "tests failed", "failed tests", "断言", "测试"]),
        ]
        for failure_type, tokens in ordered_rules:
            if any(token in logs for token in tokens):
                return failure_type
        # 兜底统一返回 "unknown"：下面那个 if 目前和 return 结果相同，
        # 保留它是为了将来想在「有 error 但没分类」和「完全没有信号」之间做区分时有个落点
        if "failed" in logs or "error" in logs:
            return "unknown"
        return "unknown"

    def _extract_first_error(self, logs: str) -> str | None:
        """从日志里截出「第一处错误」及上下文，最多 900 字符。

        为什么要截断：这段文本会进 prompt，CI 日志动辄上万行，
        不截断会直接顶爆上下文窗口、显著推高成本，而根因通常就在第一处错误附近。
        """
        if not logs.strip():
            return None
        lines = [line.strip() for line in logs.splitlines() if line.strip()]
        for index, line in enumerate(lines):
            lowered = line.lower()
            if any(token in lowered for token in ["error", "failed", "failure", "exception", "traceback", "assertionerror", "npm err", "fatal"]):
                # 前后各取一点上下文：只给错误行往往看不出是哪个命令/文件触发的；
                # 前 2 后 5 是折中，通常足够看到「命令 + 错误」两句
                start = max(0, index - 2)
                end = min(len(lines), index + 5)
                return "\n".join(lines[start:end])[:900]
        # 一条匹配都没有时，退回日志开头 6 行：CI 日志开头常是 setup 信息，
        # 依赖/权限问题往往在那里就露头了
        return "\n".join(lines[:6])[:900]

    def _extract_related_files(self, logs: str, context: dict[str, Any]) -> list[str]:
        """从日志文本里正则捞文件路径，再叠加检索到的代码引用。"""
        candidates = re.findall(r"[\w./\\-]+\.(?:py|ts|tsx|js|jsx|json|ya?ml|toml|ini|cfg|md)", logs)
        # 日志里未必有路径（比如只是 "npm ERR! code ELIFECYCLE"），
        # 所以再把 context 里已检索到的代码引用也算进来
        for item in context.get("code_references") or []:
            if isinstance(item, dict):
                title = str(item.get("title") or item.get("reference") or "").strip()
                # 只收带扩展名的（像文件），避免把 "PR #12" 之类的标题当文件
                if "." in title:
                    candidates.append(title)
        normalized = []
        for item in candidates:
            # 日志里的路径常被引号/括号/逗号包着，去掉这些包裹字符；统一分隔符便于去重比较；
            # 显示时仍保留原路径形态（只把 Windows 反斜杠归一化）
            cleaned = item.strip("`'\"()[]{}:,;").replace("\\", "/")
            if cleaned and cleaned not in normalized:
                normalized.append(cleaned)
        # 最多 8 个：再多对定位没帮助，反而会稀释 fix_steps 里的重点提示
        return normalized[:8]

    def _infer_root_cause(self, failure_type: str, first_error: str | None, failed_job_names: list[str], failed_steps: list[str]) -> str:
        """按失败类型生成一句根因描述，并带上失败范围（job/step 名）。"""
        # scope 是「失败集中在哪个 job/step」的简写；两段用逗号连接后整体过滤空段，
        # 因此全都缺失时 scope 为空串，模板里的括号部分会被省略
        scope = ", ".join(filter(None, [", ".join(failed_job_names[:2]), ", ".join(failed_steps[:2])]))
        if failure_type == "test":
            return f"失败集中在测试执行阶段{f'（{scope}）' if scope else ''}，首个错误指向断言、用例或被测行为不一致。"
        if failure_type == "build":
            return f"失败集中在构建/类型检查阶段{f'（{scope}）' if scope else ''}，首个错误通常来自编译、类型或打包配置问题。"
        if failure_type == "dependency":
            return "日志显示依赖安装或模块解析失败，优先检查 lockfile、包版本和 CI 缓存。"
        if failure_type == "permission":
            return "日志显示权限不足或访问被拒绝，优先检查 token、文件权限和 workflow 权限配置。"
        if failure_type == "environment":
            return "日志显示环境变量或 secret 缺失，优先检查 workflow env 配置和仓库 secrets。"
        if failure_type == "lint":
            return "日志显示 lint/格式检查失败，优先定位对应文件的规则违规并本地运行 lint。"
        # unknown 时没有分类结论，只能把首个错误块当线索原样带出（再截 160 字符防止结论过长）
        return f"日志缺少足够明确的分类信号，首个错误块是当前最可靠的排查入口：{first_error[:160] if first_error else '未提供日志'}"

    def _possible_causes(self, failure_type: str) -> list[str]:
        """按类型给 3 条常见成因，作为「还没确定根因」时的候选假设。"""
        mapping = {
            "test": ["测试断言不一致", "最近变更改变了接口行为", "测试数据或环境初始化不完整"],
            "build": ["类型错误或编译错误", "构建配置与代码变更不匹配", "生成产物或路径引用缺失"],
            "lint": ["格式或静态规则违规", "新增代码未经过本地 lint", "配置升级导致规则变化"],
            "dependency": ["依赖版本变化", "lockfile 未同步", "CI 缓存或 registry 配置异常"],
            "permission": ["workflow 权限不足", "token scope 不足", "脚本访问受限目录"],
            "environment": ["环境变量缺失", "secret 未配置", "CI 与本地默认配置不一致"],
            "unknown": ["首个错误块信息不足", "上游命令吞掉了真实错误", "需要查看完整 job 日志"],
        }
        # 直接下标取值（不用 get）：failure_type 只可能来自 _classify_failure 的固定枚举，
        # 一旦越界说明代码内部不一致，让它抛错比悄悄返回空列表更好排查
        return mapping[failure_type]

    def _fix_steps(self, failure_type: str, related_files: list[str]) -> list[str]:
        """按类型给 3 条修复动作；有相关文件时把文件名塞进第一条，让建议落到具体位置。"""
        # focus 只是拼进句子的补充短语，没有相关文件时整段消失，句子仍然通顺
        focus = f"，重点检查 {', '.join(related_files[:3])}" if related_files else ""
        mapping = {
            "test": [f"本地运行失败测试命令并复现{focus}", "对照最近 PR 修改确认预期行为", "修正代码或测试数据后补充回归用例"],
            "build": [f"本地运行构建/类型检查命令{focus}", "修复首个编译或类型错误", "确认生成路径、导入路径和配置文件同步更新"],
            "lint": [f"本地运行 lint/format 命令{focus}", "修复首个规则违规", "必要时调整规则并说明原因"],
            "dependency": ["重新安装依赖并确认 lockfile 更新", "检查 package/requirements 与 CI 版本是否一致", "清理 CI 缓存后重跑"],
            "permission": ["检查 workflow permissions 配置", "确认 token/secrets 的 scope", "避免脚本写入受限目录或使用受限 API"],
            "environment": ["补齐 workflow env 或 repository secrets", "为本地和 CI 提供一致默认配置", "在日志中避免泄露 secret 明文"],
            "unknown": ["展开完整 job 日志", "找到最早失败命令的 stderr", "必要时在 workflow 中增加诊断输出后重跑"],
        }
        return mapping[failure_type]

    def _is_merge_blocking(self, input_data: dict[str, Any], failure_type: str) -> bool:
        """判断这次 CI 是否应阻断合并。

        两个条件取或：
        1) run 结论明确是 failure —— 最直接的信号；
        2) run 已完成但类型不是 unknown —— 说明虽然没标 failure，但日志已经能看出问题，
           保守起见当作阻塞（宁可拦住也不要放过一个已知有错的构建）。
        """
        conclusion = str(input_data.get("conclusion") or "").lower()
        status = str(input_data.get("status") or "").lower()
        return conclusion == "failure" or (status == "completed" and failure_type != "unknown")

    def _estimate_confidence(self, first_error: str | None, failed_jobs: list[dict[str, Any]], related_files: list[str], context: dict[str, Any]) -> float:
        """按「证据多少」给一个 0.36~0.86 的置信度。

        基线 0.42 表示「只凭日志做了分类」；每多一类证据加一点分。
        上下限是刻意收窄的：规则判断拿不到 1.0（不是确定性结论），
        也不允许低于 0.36（否则 synthesis 里的置信度排序会失去区分度）。
        """
        score = 0.42
        if first_error:
            score += 0.16  # 有明确错误块，可信度提升最大
        if failed_jobs:
            score += 0.12  # 知道具体哪个 job 失败
        if related_files:
            score += 0.08  # 定位到了具体文件
        if context.get("recent_prs"):
            score += 0.05  # 有近期 PR 作为变更线索
        if context.get("code_references"):
            score += 0.05  # 有检索到的代码引用
        return round(max(0.36, min(0.86, score)), 2)
