"""Planner Agent（编排链的第一环）。

职责边界：读懂用户的目标 + 当前可用的仓库信号，产出一份 **WorkflowSpec**——
即「要跑哪些 Agent、各自负责什么、依赖谁、允许访问哪些数据源」。
它不执行任何分析，也不调 LLM：任务选择完全由关键词和上下文开关决定。

为什么是规则而不是让模型规划：
- 规划结果决定整条链跑哪些 Agent，进而决定成本。规则版可预测、可测试、不会「今天多点两个任务」；
- 每个 task 都带 WorkflowClaim（target/allowed_sources），这是**权限边界**：
  下游 Agent 只能碰 claim 里声明过的数据源，规则化构造比模型生成更可控。

关键设计点：
- task 的选择条件都是「上下文有该信号」AND「用户提到了相关词 或 问题很宏观」，
  两者缺一不可——光有 PR 但用户问的是 Issue，就不该跑去审查 PR（省一次 LLM 调用）。
- 全都没命中时，用 _fallback_*_task 至少挂一个有数据的任务，避免空工作流。

replan() 是第二轮：Observer 发现问题后，由它补任务（补安全闸门、补 RAG 取证据、补 CI 检查）。
"""

from __future__ import annotations

from typing import Any

from app.schemas.workflow import WorkflowClaim, WorkflowObservation, WorkflowSpec, WorkflowTask, WorkflowTaskResult
from app.services.agents.base import BaseAgent


class PlannerAgent(BaseAgent):
    """把用户目标编译成有边界、可执行的多 Agent 工作流规格。"""

    name = "planner_agent"
    description = "根据用户目标和可用仓库信号，构建有边界的工程工作流规格。"
    available_tools = ["create_workflow_spec"]

    async def run(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """首轮规划：按上下文信号 + 关键词决定任务图。

        输入：input_data["message"] 或 ["goal"] 是用户原话；
              context 是仓库信号快照（has_repo / has_issue / has_pr / has_ci_run /
              issue_number / pr_number / workflow_run_number / requires_safety /
              enable_rag_task 等）。
        输出：WorkflowSpec 的 json dict，交给 workflow_orchestrator 执行。
        """
        # message 与 goal 二选一：chat 链路传 message，内部调用可能只传 goal
        message = str(input_data.get("message") or input_data.get("goal") or "").strip()
        lowered = message.lower()
        tasks: list[WorkflowTask] = []
        # 三个 label 用于把「Issue #123」这类具体编号写进 objective，
        # 编号缺失时退化成「当前 Issue」，保证目标描述始终可读
        issue_label = f"Issue #{context['issue_number']}" if context.get("issue_number") is not None else "当前 Issue"
        pr_label = f"PR #{context['pr_number']}" if context.get("pr_number") is not None else "当前 PR"
        ci_label = (
            f"CI/Workflow Run #{context['workflow_run_number']}"
            if context.get("workflow_run_number") is not None
            else "最新失败的 CI 运行"
        )
        # 兼容新旧两个字段名：has_ci_run 是当前口径，has_failed_ci 是历史调用方仍在用的
        has_ci_run = context.get("has_ci_run", context.get("has_failed_ci"))

        # repo_health 是「基线任务」：先拿一份仓库快照，后续 Agent 才有共享证据可依赖。
        # 默认 True 是为了让「没有 context 的裸调用」也能跑到一个有意义的工作流
        if context.get("has_repo", True):
            tasks.append(
                WorkflowTask(
                    id="repo_health",
                    agent_name="repo_health_agent",
                    task_type="repo_health",
                    objective="在专用分析前捕获仓库基线。",
                    claim=WorkflowClaim(
                        target_type="repository",
                        target_id=context.get("repo_id"),
                        # 只读三类列表数据，不涉及写操作
                        allowed_sources=["issues", "pull_requests", "workflow_runs"],
                        reason="共享仓库快照可以让后续 Agent 结论有证据支撑。",
                    ),
                )
            )

        # broad 是「宏观问题」开关：用户问「整体状态/下一步」时，即使没点名 PR/CI，
        # 也该把所有可用信号都跑一遍；否则会漏掉关键上下文
        broad = self._is_broad_engineering_question(lowered)
        # 双重条件：有 Issue 数据 AND（宏观问题 或 明确提到 issue 相关词）。
        # 单有数据不构成理由——用户问 PR 时不该白跑一次 Issue 分析（那是真金白银的模型调用）
        if context.get("has_issue") and (broad or self._mentions_issue(lowered)):
            tasks.append(
                WorkflowTask(
                    id="issue_analysis",
                    agent_name="issue_analyst_agent",
                    task_type="issue_analysis",
                    objective=f"分析 {issue_label} 的分诊结论、优先级、负责人、证据和下一步动作。",
                    # 依赖只在 repo_health 确实存在时挂上，避免引用不存在的 task id
                    dependencies=["repo_health"] if self._has_task(tasks, "repo_health") else [],
                    claim=WorkflowClaim(
                        target_type="issue",
                        target_id=context.get("issue_id"),
                        allowed_sources=["issue", "similar_issues", "code", "documents", "team_members"],
                        reason="判断负责人和优先级需要明确 Issue 范围。",
                    ),
                    input_hint={"issue_number": context.get("issue_number")},
                )
            )

        if context.get("has_pr") and (broad or self._mentions_pr(lowered)):
            tasks.append(
                WorkflowTask(
                    id="pr_review",
                    agent_name="pr_review_agent",
                    task_type="pr_review",
                    objective=f"审查 {pr_label} diff、风险、阻塞问题和测试建议。",
                    dependencies=["repo_health"] if self._has_task(tasks, "repo_health") else [],
                    claim=WorkflowClaim(
                        target_type="pull_request",
                        target_id=context.get("pr_id"),
                        allowed_sources=["pull_request", "diff", "review_comments", "ci_summary", "documents"],
                        reason="合并就绪度和风险判断需要明确 PR 范围。",
                    ),
                    input_hint={"pr_number": context.get("pr_number")},
                    # critical 只在用户问到合并时置真：表示这个任务的结论是决策的硬依赖，
                    # 它失败时工作流不该继续给出「可以合」的结论
                    critical=self._mentions_merge(lowered),
                )
            )

        # CI 调试多一个触发条件（提到 merge 也触发）：合并判断必须建立在 CI 状态之上
        if has_ci_run and (broad or self._mentions_ci(lowered) or self._mentions_merge(lowered)):
            tasks.append(
                WorkflowTask(
                    id="ci_debug",
                    agent_name="ci_debug_agent",
                    task_type="ci_debug",
                    objective=f"分析 {ci_label}，并判断是否阻塞合并或发布。",
                    dependencies=["repo_health"] if self._has_task(tasks, "repo_health") else [],
                    claim=WorkflowClaim(
                        target_type="workflow_run",
                        target_id=context.get("workflow_run_id"),
                        allowed_sources=["workflow_run", "jobs", "logs", "recent_prs", "code"],
                        reason="任何合并或就绪度结论都需要先看 CI 失败上下文。",
                    ),
                    input_hint={"workflow_run_number": context.get("workflow_run_number")},
                    # CI 固定 critical：它的结论直接决定能不能合，不能因为「没跑成」而被跳过
                    critical=True,
                )
            )

        # RAG 检索任务：上下文显式要求（enable_rag_task）或用户提到历史/知识库时补上。
        # 它是唯一能「补证据」的任务，也是 replan 阶段最常被追加的一个
        if context.get("enable_rag_task") or self._mentions_history(lowered):
            tasks.append(
                WorkflowTask(
                    id="rag_search",
                    agent_name="rag_context_agent",
                    task_type="rag_search",
                    objective="检索与目标相关的历史 Issue、PR、失败 CI 日志、项目文档和已批准知识。",
                    dependencies=["repo_health"] if self._has_task(tasks, "repo_health") else [],
                    claim=WorkflowClaim(
                        target_type="knowledge",
                        target_id=context.get("repo_id"),
                        # 只能读已入库/已批准的知识，不能读任意文件——这是检索侧的权限边界
                        allowed_sources=["rag_documents", "approved_memory", "uploaded_knowledge"],
                        reason="历史证据可以避免重复分析和遗漏回归风险。",
                    ),
                )
            )

        # 安全审查任务：只有上下文声明「这次请求涉及写操作」才加，避免只读问答也弹确认框
        if context.get("requires_safety"):
            tasks.append(
                WorkflowTask(
                    id="safety_review",
                    agent_name="safety_agent",
                    task_type="safety_review",
                    objective="判断写入或外部发送风险，并生成安全草稿而不是执行副作用。",
                    claim=WorkflowClaim(
                        target_type="external_action",
                        target_id=context.get("repo_id"),
                        allowed_sources=["user_request", "repository_snapshot"],
                        # write_intent=True 是可审计标记：trace 里能看出这次链路包含写意图
                        write_intent=True,
                        reason="写入类操作必须经过人工确认。",
                    ),
                    critical=True,
                )
            )

        # 退化保护：如果关键词一个都没命中，tasks 里只会剩 repo_health，
        # 那样的工作流等于「查了一遍仓库」就给结论，几乎没有信息量。
        # 这里按「手头有什么数据」挑一个最相关的任务补上，优先级 PR > CI > Issue
        if len(tasks) == 1 and tasks[0].task_type == "repo_health":
            if context.get("has_pr"):
                tasks.append(self._fallback_pr_task(context))
            elif has_ci_run:
                tasks.append(self._fallback_ci_task(context))
            elif context.get("has_issue"):
                tasks.append(self._fallback_issue_task(context))

        spec = WorkflowSpec(
            # goal 兜底成一句固定描述：spec.goal 会被 synthesis 写进备忘录标题区，不能为空
            goal=message or "运行一次工程协作工作流。",
            trigger_message=message,
            # acceptance_criteria 是「写给下游和用户看的契约」，Observer/Synthesis 的行为
            # 与这些条目是对应的（claim 边界、证据要求、P1/P2 分级、草稿而非直接执行）
            acceptance_criteria=[
                "所有被选中的 Agent 都必须停留在声明的 claim 范围内。",
                "每个专用结果都应包含证据、置信度，或在证据不足时说明明确原因。",
                "Observer 必须在最终综合前标出跨 Agent 冲突。",
                "最终综合必须保留 What、Why、Tradeoff、Open Questions、Next Action。",
                "PR 审查必须按 P1/P2/P3 分级；P1/P2 未解决前不能建议直接合入。",
                "外部写操作必须返回草稿，并要求人工确认。",
            ],
            tasks=tasks,
            # 与 safety_review 任务同源：有安全任务就意味着这份 spec 需要人确认
            requires_human_confirmation=bool(context.get("requires_safety")),
        )
        # mode="json" 让结果可直接进 trace / 消息队列，datetime 等类型会被序列化
        return spec.model_dump(mode="json")

    async def replan(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """第二轮规划：读 Observer 的 findings，补上缺失的任务。

        只做「追加」，不删改已有任务——第一轮已经跑过的结果还有效，
        重排会让 trace 与结果对不上。三类补充各有触发条件：
        1) missing_safety_review → 补安全闸门（流程漏洞，必须补）；
        2) 证据类问题（task_error/evidence_gap/low_confidence 等）→ 补 RAG 取证据；
        3) 结论可能依赖 CI 但没跑 → 补 CI 检查。
        """
        previous_spec = WorkflowSpec.model_validate(input_data["spec"])
        results = [WorkflowTaskResult.model_validate(item) for item in input_data.get("task_results") or []]
        observation = WorkflowObservation.model_validate(input_data["observation"])
        # deep copy：直接改 previous_spec.tasks 会污染调用方持有的对象（它可能还要拿去落库）
        tasks = [task.model_copy(deep=True) for task in previous_spec.tasks]
        task_types = {task.task_type for task in tasks}
        # 只有成功的任务才配当依赖：把失败任务的 id 挂成 dependencies 会让新任务直接跑不起来
        successful_task_ids = {result.task_id for result in results if result.status == "success"}
        finding_types = {finding.finding_type for finding in observation.findings}

        # 情况 1：Observer 说「该有人确认却没跑安全审查」→ 补上。
        # 用 _unique_task_id 防止 id 冲突（虽然正常不会重复，但 replan 可以被多次调用）
        if "missing_safety_review" in finding_types and "safety_review" not in task_types:
            tasks.append(
                WorkflowTask(
                    id=self._unique_task_id(tasks, "safety_review"),
                    agent_name="safety_agent",
                    task_type="safety_review",
                    objective="Review write or external-send risk before producing a final recommendation.",
                    dependencies=self._dependency_if_successful(successful_task_ids, "repo_health"),
                    claim=WorkflowClaim(
                        target_type="external_action",
                        target_id=context.get("repo_id"),
                        allowed_sources=["user_request", "repository_snapshot"],
                        write_intent=True,
                        reason="Replan added a missing safety gate after Observer flagged it.",
                    ),
                    critical=True,
                )
            )

        # 情况 2：这些 finding 都指向「证据不足或某步没跑成」，最有效的补救是去检索历史资料
        needs_context_recovery = bool(
            finding_types
            & {
                "task_error",
                "task_skipped",
                "evidence_gap",
                "low_confidence",
                "owner_needs_review",
                "team_profile_missing",
            }
        )
        if needs_context_recovery and "rag_search" not in task_types:
            tasks.append(
                WorkflowTask(
                    id=self._unique_task_id(tasks, "rag_search"),
                    agent_name="rag_context_agent",
                    task_type="rag_search",
                    objective="Recover missing evidence by searching historical issues, pull requests, failed CI logs, project documents, and approved knowledge.",
                    dependencies=self._dependency_if_successful(successful_task_ids, "repo_health"),
                    claim=WorkflowClaim(
                        target_type="knowledge",
                        target_id=context.get("repo_id"),
                        allowed_sources=["rag_documents", "approved_memory", "uploaded_knowledge"],
                        reason="Replan added historical context because Observer found weak evidence or confidence.",
                    ),
                )
            )

        # 情况 3：没跑过 CI 检查，且「当前讨论内容涉及 CI/合并」→ 补一次。
        # _replan_needs_ci 会把 goal、各 Agent 摘要、Observer findings 全串起来做关键词扫描，
        # 因此即使原始提问没提 CI，只要中间结论提到了，也能触发补跑
        if context.get("has_ci_run", context.get("has_failed_ci")) and "ci_debug" not in task_types and self._replan_needs_ci(previous_spec, results, observation):
            tasks.append(
                WorkflowTask(
                    id=self._unique_task_id(tasks, "ci_debug"),
                    agent_name="ci_debug_agent",
                    task_type="ci_debug",
                    objective="Check the latest failed CI run before final merge or release guidance.",
                    dependencies=self._dependency_if_successful(successful_task_ids, "repo_health"),
                    claim=WorkflowClaim(
                        target_type="workflow_run",
                        target_id=context.get("workflow_run_id"),
                        allowed_sources=["workflow_run", "jobs", "logs", "recent_prs", "code"],
                        reason="Replan added CI debugging because the current decision may depend on failed automation.",
                    ),
                    input_hint={"workflow_run_number": context.get("workflow_run_number")},
                    critical=True,
                )
            )

        replanned = previous_spec.model_copy(deep=True)
        replanned.tasks = tasks
        return replanned.model_dump(mode="json")

    def _fallback_issue_task(self, context: dict[str, Any]) -> WorkflowTask:
        """退化路径：用户没提具体诉求，但仓库里有 Issue —— 拿它当「最强的可用信号」。

        与上面常规分支的区别：allowed_sources 更窄（只给 issue + team_members），
        因为这是「顺手分析一下」，不该消耗完整的检索预算。
        """
        label = f"Issue #{context['issue_number']}" if context.get("issue_number") is not None else "最新 Issue"
        return WorkflowTask(
            id="issue_analysis",
            agent_name="issue_analyst_agent",
            task_type="issue_analysis",
            objective=f"分析 {label}，因为它是当前最强的可用工程信号。",
            dependencies=["repo_health"],
            claim=WorkflowClaim(target_type="issue", target_id=context.get("issue_id"), allowed_sources=["issue", "team_members"]),
            input_hint={"issue_number": context.get("issue_number")},
        )

    def _fallback_pr_task(self, context: dict[str, Any]) -> WorkflowTask:
        """退化路径：兜底分析 PR（优先级最高，因为它最接近「要不要合」这个真实决策）。"""
        label = f"PR #{context['pr_number']}" if context.get("pr_number") is not None else "最新 PR"
        return WorkflowTask(
            id="pr_review",
            agent_name="pr_review_agent",
            task_type="pr_review",
            objective=f"审查 {label}，因为它是当前最强的可用工程信号。",
            dependencies=["repo_health"],
            claim=WorkflowClaim(target_type="pull_request", target_id=context.get("pr_id"), allowed_sources=["pull_request", "diff", "ci_summary"]),
            input_hint={"pr_number": context.get("pr_number")},
        )

    def _fallback_ci_task(self, context: dict[str, Any]) -> WorkflowTask:
        """退化路径：兜底分析失败 CI。critical=True 因为它天然代表「流水线是红的」。"""
        label = (
            f"CI/Workflow Run #{context['workflow_run_number']}"
            if context.get("workflow_run_number") is not None
            else "最新失败的 CI 运行"
        )
        return WorkflowTask(
            id="ci_debug",
            agent_name="ci_debug_agent",
            task_type="ci_debug",
            objective=f"分析 {label}，因为它是当前最强的可用工程信号。",
            dependencies=["repo_health"],
            claim=WorkflowClaim(target_type="workflow_run", target_id=context.get("workflow_run_id"), allowed_sources=["workflow_run", "jobs", "logs"]),
            input_hint={"workflow_run_number": context.get("workflow_run_number")},
            critical=True,
        )

    def _has_task(self, tasks: list[WorkflowTask], task_id: str) -> bool:
        """判断某个 task id 是否已存在，用于决定要不要挂依赖。"""
        return any(task.id == task_id for task in tasks)

    def _unique_task_id(self, tasks: list[WorkflowTask], base: str) -> str:
        """生成不冲突的 task id：base 被占用则依次试 base_2、base_3…

        为什么要唯一：results 是按 task_id 与任务对应的，id 重复会让结果串到错的任务上。
        """
        existing = {task.id for task in tasks}
        if base not in existing:
            return base
        index = 2
        while f"{base}_{index}" in existing:
            index += 1
        return f"{base}_{index}"

    def _dependency_if_successful(self, successful_task_ids: set[str], task_id: str) -> list[str]:
        """只在依赖任务成功时才返回依赖，否则返回空列表（让它独立执行）。

        这个取舍很重要：repo_health 失败时，如果仍然挂依赖，新补的任务会被调度器判为
        「依赖未满足」而直接跳过，replan 就白做了；不挂依赖至少还能跑一次。
        """
        return [task_id] if task_id in successful_task_ids else []

    def _replan_needs_ci(
        self,
        spec: WorkflowSpec,
        results: list[WorkflowTaskResult],
        observation: WorkflowObservation,
    ) -> bool:
        """判断 replan 时是否值得补一个 CI 分析任务。

        判定依据不是用户原话，而是「当前所有已知文本」：目标、触发消息、
        各 Agent 摘要、Observer 的每条 finding。任一提到 CI 或合并即认为相关。
        这样能把「PR 分析里说 CI 失败了」这类间接信号也纳入判断。
        """
        text = " ".join(
            [
                spec.goal,
                spec.trigger_message,
                *(result.summary for result in results),
                *(finding.message for finding in observation.findings),
            ]
        ).lower()
        return self._mentions_ci(text) or self._mentions_merge(text)

    # 以下五个 _mentions_* 是规划阶段的关键词识别器。
    # 它们是整条链「跑哪些 Agent」的开关，词表就是产品行为的一部分：
    # 加词 = 让某类提问更容易触发对应 Agent（更贵但更全），删词反之。
    # 注意 "pr" 这种短词也会命中 "process"/"prompt" 等，属于可接受误报（多跑一次审查而已）。

    def _mentions_issue(self, text: str) -> bool:
        """是否提到 Issue 相关诉求（含「需求/问题/负责人/优先级」等中文说法）。"""
        return any(token in text for token in ["issue", "bug", "需求", "问题", "负责人", "分配", "优先级"])

    def _mentions_pr(self, text: str) -> bool:
        """是否提到 PR / 代码审查相关诉求。"""
        return any(token in text for token in ["pr", "pull request", "diff", "review", "合并", "代码审查", "风险"])

    def _mentions_ci(self, text: str) -> bool:
        """是否提到 CI / 构建 / 测试失败相关诉求。"""
        return any(token in text for token in ["ci", "workflow", "action", "build", "test", "失败", "流水线", "构建"])

    def _mentions_merge(self, text: str) -> bool:
        """是否在问「能不能合 / 能不能发」——这类判断必须带上 CI 视角。"""
        return any(token in text for token in ["merge", "合并", "能不能上", "能不能发", "readiness", "release"])

    def _mentions_history(self, text: str) -> bool:
        """是否在问历史 / 相似案例 / 知识库——命中就补一个 RAG 检索任务。"""
        return any(token in text for token in ["history", "similar", "rag", "knowledge", "历史", "相似", "知识库", "记忆"])

    def _is_broad_engineering_question(self, text: str) -> bool:
        """判断是不是「宏观工程问题」。

        命中时所有可用信号都会跑（broad 分支），因为「整体状态怎么样」这种问题
        缺少具体指向，只能靠全量分析来回答；代价是这类提问最贵。
        """
        return any(
            token in text
            for token in [
                "multi-agent",
                "workflow",
                "协作",
                "工程状态",
                "阻塞",
                "风险",
                "优先处理",
                "下一步",
                "整体",
                "全局",
                "当前仓库",
                "能不能合并",
            ]
        )
