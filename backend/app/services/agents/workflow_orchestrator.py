"""工作流编排器：把 Planner → 专用 Agent → Observer → Synthesis 串成一条可重试的链。

这是整个 Agent 层的中枢，理解它就能理解其余所有 Agent 的位置：
1. **Planner** 产出 WorkflowSpec（任务图）；
2. **专用 Agent** 按 DAG 并行执行，各自产出 WorkflowTaskResult；
3. **Observer** 审查全部结果，产出 findings + 总体置信度；
4. 若 Observer 发现「可修复」的问题，回调 Planner.replan 补任务，再跑一轮（最多 2 次）；
5. **Synthesis** 汇总成最终备忘录，打包成 WorkflowReport。

设计要点：
- 执行器由外部以 `task_runners`（task_type → runner）注入，编排器本身不 import 具体 Agent，
  这样新增 Agent 只需注册一个 runner，不用改编排逻辑；
- 任何单任务失败都被转成 status="error"/"skipped" 的结果而不是抛出去，
  **一条链不会因为一个 Agent 挂了而整体失败**——代价是下游要自己看 status；
- 全程记录 trace（workflow_spec / workflow_task_result / workflow_observer_result /
  workflow_replan 四类事件），前端的时间线与调试都靠它。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

from app.core.config import settings
from app.schemas.workflow import WorkflowObservation, WorkflowReport, WorkflowSpec, WorkflowTask, WorkflowTaskResult
from app.services.agents.base import BaseAgent
from app.services.agents.observer_agent import ObserverAgent
from app.services.agents.planner_agent import PlannerAgent
from app.services.agents.synthesis_agent import SynthesisAgent


# 任务执行器的签名：(任务定义, 所属 spec, 共享 context) → 结果。
# 返回类型允许是 dict，方便 runner 直接返回 pydantic 模型序列化的结果，
# 编排器在 _run_single_task 里统一校验成 WorkflowTaskResult。
WorkflowTaskRunner = Callable[[WorkflowTask, WorkflowSpec, dict[str, Any]], Awaitable[WorkflowTaskResult | dict[str, Any]]]


class WorkflowOrchestrator(BaseAgent):
    """多 Agent 工作流的调度者与 trace 记录者。"""

    name = "workflow_orchestrator"
    description = "协调 Planner、专用 Agent、Observer 和 Synthesis 完成工程决策。"
    available_tools = ["run_engineering_workflow"]

    def __init__(
        self,
        task_runners: dict[str, WorkflowTaskRunner],
        planner: PlannerAgent | None = None,
        observer: ObserverAgent | None = None,
        synthesizer: SynthesisAgent | None = None,
    ) -> None:
        """注入任务执行器与三个内置角色。"""
        # task_runners 必须外部传入（本类不持有任何具体 Agent 的知识）；
        # 三个内置角色允许注入，是为了单测能替换成假实现，避免真实 LLM 调用
        self.task_runners = task_runners
        self.planner = planner or PlannerAgent()
        self.observer = observer or ObserverAgent()
        self.synthesizer = synthesizer or SynthesisAgent()

    async def run(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """完整入口：先规划，再执行。

        拆成两步是为了让「只执行已有 spec」的场景（前端重跑、调试）能直接走 run_spec。
        """
        spec_data = await self.planner.run(input_data, context)
        return await self.run_spec(spec_data, context)

    async def run_spec(
        self,
        spec_data: WorkflowSpec | dict[str, Any],
        context: dict[str, Any],
        *,
        allow_replan: bool = True,
    ) -> dict[str, Any]:
        """执行一份 spec，含最多 N 轮 replan，最后综合成 WorkflowReport。

        allow_replan 是 keyword-only：重跑/调试场景应显式传 False，
        避免为了「再看一遍结果」而无谓地多跑一轮 Agent（每轮都是真金白银）。
        """
        started = time.perf_counter()
        # spec 可以是对象也可以是 dict：前者来自 planner 内部调用，后者来自 API/数据库
        spec = spec_data if isinstance(spec_data, WorkflowSpec) else WorkflowSpec.model_validate(spec_data)
        current_spec = spec
        # task_results 跨轮累积（不只保留本轮）：Observer 和 Synthesis 要看全量历史，
        # 否则 replan 后会把第一轮的结论丢掉
        task_results: list[WorkflowTaskResult] = []
        observations: list[WorkflowObservation] = []
        # round_records 保留每轮的 (spec, 本轮结果, observation)，用于生成 trace 时间线；
        # 注意 results 只存本轮，与累积的 task_results 不同
        round_records: list[dict[str, Any]] = []
        replan_events: list[dict[str, Any]] = []
        max_replans = self._max_replans(context, allow_replan)

        # 循环上界是 max_replans + 1：第一次是原始执行，之后每次都是重规划
        for attempt in range(max_replans + 1):
            round_results = await self._run_task_graph(current_spec, context, previous_results=task_results)
            task_results.extend(round_results)
            # Observer 看的是「累积的全部结果」，所以 replan 不会丢掉上一轮的发现
            observation = await self._observe(current_spec, task_results, context)
            observations.append(observation)
            round_records.append({"spec": current_spec, "results": round_results, "observation": observation})

            # 已达重规划上限：直接退出，不再问 planner
            if attempt >= max_replans:
                break

            # 两个提前退出条件：Observer 认为没什么可修的；或 planner 没有 replan 能力
            replan_reason = self._replan_reason(task_results, observation)
            if replan_reason is None:
                break
            if not hasattr(self.planner, "replan"):
                break

            # 把 spec/结果/observation 全量传给 planner —— 它靠 finding_type 决定补什么任务。
            # 都转成 json 形式，保证跨进程/落库时口径一致
            next_spec_data = await self.planner.replan(
                {
                    "spec": current_spec.model_dump(mode="json"),
                    "task_results": [item.model_dump(mode="json") for item in task_results],
                    "observation": observation.model_dump(mode="json"),
                    "reason": replan_reason,
                    "replan_iteration": attempt + 1,
                },
                context,
            )
            next_spec = self._normalize_replanned_spec(
                original_spec=spec,
                current_spec=current_spec,
                next_spec=WorkflowSpec.model_validate(next_spec_data),
            )
            # 任务图没变就停：否则会「同一份 spec 反复跑」直到耗尽 max_replans，
            # 白白多花几轮模型调用。这是防止空转的关键判断
            if self._same_task_graph(current_spec, next_spec):
                break

            # trace 事件：记录这次重规划从几个任务变成几个、新增了哪些
            replan_events.append(
                {
                    "type": "workflow_replan",
                    "workflow_id": spec.workflow_id,
                    "iteration": attempt + 1,
                    "reason": replan_reason,
                    "from_task_count": len(current_spec.tasks),
                    "to_task_count": len(next_spec.tasks),
                    # 差集即新增任务；只记 id 不记详情，详情在后面 workflow_spec 事件里
                    "added_tasks": [
                        task.id
                        for task in next_spec.tasks
                        if task.id not in {existing.id for existing in current_spec.tasks}
                    ],
                }
            )
            current_spec = next_spec

        # 只在理论情况下（observations 为空）才会走默认分支；正常至少有一轮
        observation = observations[-1] if observations else WorkflowObservation(summary="Workflow did not run.")
        # 综合用的是**最后一轮**的 spec（含 replan 新增任务）和累积的全部结果
        synthesis = await self.synthesizer.run(
            {
                "spec": current_spec.model_dump(mode="json"),
                "task_results": [item.model_dump(mode="json") for item in task_results],
                "observation": observation.model_dump(mode="json"),
            },
            context,
        )
        report = WorkflowReport(
            spec=current_spec,
            task_results=task_results,
            observation=observation,
            final_answer=str(synthesis.get("answer") or ""),
            # 以下三个是「可审计」字段：调了哪些工具、引用了哪些资料、跑了哪些轮次
            tool_calls=self._tool_calls(current_spec, task_results, replan_events),
            citations=self._citations(task_results),
            metrics={
                # duration_ms 覆盖全流程（含 replan），是成本分析的主要指标
                "duration_ms": int((time.perf_counter() - started) * 1000),
                "task_count": len(task_results),  # 实际执行的任务数（含各轮累积）
                "plan_task_count": len(current_spec.tasks),  # 最终计划里的任务数
                "execution_rounds": len(observations),  # 轮次数 = 1 + 实际 replan 次数
                "replan_count": len(replan_events),
                "success_count": len([item for item in task_results if item.status == "success"]),
                "error_count": len([item for item in task_results if item.status == "error"]),
                "skipped_count": len([item for item in task_results if item.status == "skipped"]),
                "observer_findings": len(observation.findings),
            },
            agent_events=self._agent_events(round_records, replan_events),
        )
        return report.model_dump(mode="json")

    async def _observe(
        self,
        spec: WorkflowSpec,
        task_results: list[WorkflowTaskResult],
        context: dict[str, Any],
    ) -> WorkflowObservation:
        """调 Observer 做审查，并把结果转回强类型对象。

        Observer 返回 dict，这里立刻 model_validate：后续要频繁读
        observation.findings / overall_confidence，用对象比反复取字典安全。
        """
        observation_data = await self.observer.run(
            {
                "spec": spec.model_dump(mode="json"),
                "task_results": [item.model_dump(mode="json") for item in task_results],
            },
            context,
        )
        return WorkflowObservation.model_validate(observation_data)

    async def _run_task_graph(
        self,
        spec: WorkflowSpec,
        context: dict[str, Any],
        previous_results: list[WorkflowTaskResult] | None = None,
    ) -> list[WorkflowTaskResult]:
        """按依赖关系分层执行任务图（DAG 调度）。

        调度规则：
        1. 依赖里有失败/跳过的任务 → 本任务直接判 skipped（依赖阻断会沿链传播）；
        2. 依赖全部成功的任务才是 ready，可并行执行，每批数量受 max_parallel_tasks 限制；
        3. 死锁检测：如果还有 pending 但没有任何 ready（循环依赖或依赖 id 写错），
           把剩余任务全标成 error 并退出，避免 while 死循环。

        previous_results 用于 replan 后的第二轮：已成功的任务不再重跑（省钱），
        失败的依赖仍会阻断下游（不重试，因为 replan 是「补任务」而不是「重试」）。
        返回值只包含本轮产生的结果，不含 previous_results。
        """
        pending = {task.id: task for task in spec.tasks}
        # 上一轮已成功的任务直接视为完成：pending 里会被 pop 掉，不会重复执行
        completed: dict[str, WorkflowTaskResult] = {
            result.task_id: result
            for result in previous_results or []
            if result.status == "success"
        }
        # 失败/跳过的任务单独记一份，作为「阻断源」向下游传播
        blocked_dependencies: dict[str, WorkflowTaskResult] = {
            result.task_id: result
            for result in previous_results or []
            if result.status in {"error", "skipped"}
        }
        for task_id in completed:
            pending.pop(task_id, None)
        results: list[WorkflowTaskResult] = []

        while pending:
            # 第一优先级：先把「依赖已经废了」的任务清掉，它们连跑都不用跑。
            # 每轮都重新扫，所以阻断会一层层往下传（repo_health 挂了 → 所有下游都 skipped）
            blocked = [
                task
                for task in pending.values()
                if any(dependency in blocked_dependencies for dependency in task.dependencies)
            ]
            for task in blocked:
                failed_dependencies = [
                    dependency
                    for dependency in task.dependencies
                    if dependency in blocked_dependencies
                ]
                result = self._dependency_blocked_result(task, failed_dependencies)
                blocked_dependencies[task.id] = result
                results.append(result)
                pending.pop(task.id, None)
            if not pending:
                break

            # 第二优先级：找出所有依赖都满足的任务
            ready = [
                task
                for task in pending.values()
                if all(dependency in completed for dependency in task.dependencies)
            ]
            # 还有任务但没有一个可跑 = 循环依赖 / 依赖指向不存在的 id。
            # 全标 error 后 break，绝不能让 while 空转（这是最容易被写出的死循环 bug）
            if not ready:
                for task in pending.values():
                    result = self._error_result(task, "任务依赖存在循环或缺失依赖。", error_kind="dependency_cycle_or_missing")
                    blocked_dependencies[task.id] = result
                    results.append(result)
                break

            # 截断到 max_parallel_tasks：并发上限主要防两类问题——模型 API 限流、
            # 以及同时拉取大量 GitHub 数据时打满外部配额
            batch = ready[: spec.max_parallel_tasks]
            # return_exceptions=False：_run_single_task 内部已把异常转成 error 结果，
            # 所以这里不该再出现异常；真出现说明编排层自身有 bug，让它炸出来
            batch_results = await asyncio.gather(
                *(self._run_single_task(task, spec, context) for task in batch),
                return_exceptions=False,
            )
            for task, result in zip(batch, batch_results):
                if result.status == "success":
                    completed[task.id] = result
                else:
                    blocked_dependencies[task.id] = result
                results.append(result)
                pending.pop(task.id, None)

        return results

    async def _run_single_task(
        self,
        task: WorkflowTask,
        spec: WorkflowSpec,
        context: dict[str, Any],
    ) -> WorkflowTaskResult:
        """执行单个任务，并保证「任何失败都变成结果对象」而不是异常。

        四类结局：
        - 没有注册 runner → skipped（配置缺失，不算错误）；
        - 正常返回 → success（dict 会被校验成 WorkflowTaskResult）；
        - 超时 → error(kind=timeout)；
        - 抛异常 → error(kind=task_runtime_error)。
        """
        started_at = datetime.now(timezone.utc)
        runner = self.task_runners.get(task.task_type)
        # 没有执行器时返回 skipped 而不是 error：这通常是「该功能未上线」，
        # 标成 error 会让 Observer 报阻塞，从而误伤整条链
        if runner is None:
            return WorkflowTaskResult(
                task_id=task.id,
                agent_name=task.agent_name,
                task_type=task.task_type,
                status="skipped",
                summary=f"没有为 {task.task_type} 注册任务执行器。",
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
            )
        try:
            # 超时保护必须有：Agent 内部要调 LLM，模型不响应时若没有 wait_for，
            # 整个 HTTP 请求会一直挂着，前端只能等到网关超时
            raw_result = await asyncio.wait_for(
                runner(task, spec, context),
                timeout=self._task_timeout_seconds(context),
            )
            # runner 允许返回 dict（更省事），这里统一成模型
            result = raw_result if isinstance(raw_result, WorkflowTaskResult) else WorkflowTaskResult.model_validate(raw_result)
            # 时间戳兜底：runner 自己填了就尊重它的值，没填用编排层记录的时间，
            # 保证 trace 里每个任务都有耗时可比
            result.started_at = result.started_at or started_at
            result.completed_at = result.completed_at or datetime.now(timezone.utc)
            return result
        except TimeoutError:
            return self._error_result(
                task,
                f"{task.agent_name} 执行超过 {self._task_timeout_seconds(context):g} 秒。",
                started_at=started_at,
                error_kind="timeout",
            )
        except Exception as exc:
            # 兜住所有异常（含 LLM 报错、网络错误、代码 bug）：编排层不能让一条链整体失败
            return self._error_result(task, str(exc), started_at=started_at, error_kind="task_runtime_error")

    def _error_result(
        self,
        task: WorkflowTask,
        error: str,
        started_at: datetime | None = None,
        *,
        error_kind: str = "task_runtime_error",
    ) -> WorkflowTaskResult:
        """构造统一的失败结果。

        output 里塞 error_kind 而不是原始异常：下游/trace 只需要区分
        「超时 / 运行时错误 / 依赖循环」，具体堆栈在日志里。
        """
        return WorkflowTaskResult(
            task_id=task.id,
            agent_name=task.agent_name,
            task_type=task.task_type,
            status="error",
            summary=f"{task.agent_name} 执行失败。",
            output={"error_kind": error_kind},
            error=error,
            started_at=started_at or datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        )

    def _dependency_blocked_result(self, task: WorkflowTask, failed_dependencies: list[str]) -> WorkflowTaskResult:
        """构造「因依赖失败而跳过」的结果。

        confidence 显式给 0.0（而不是留 None）：让下游一眼看出这条结果完全不可用；
        Observer 的置信度计算会跳过 skipped，所以这个 0 不会污染整体均值。
        """
        return WorkflowTaskResult(
            task_id=task.id,
            agent_name=task.agent_name,
            task_type=task.task_type,
            status="skipped",
            summary=f"{task.agent_name} 未执行，因为依赖任务未成功：{', '.join(failed_dependencies)}。",
            output={"error_kind": "dependency_failed", "failed_dependencies": failed_dependencies},
            confidence=0.0,
            completed_at=datetime.now(timezone.utc),
        )

    def _task_timeout_seconds(self, context: dict[str, Any]) -> float:
        """取单任务超时时间：context 优先（可按请求调大），否则用全局配置。

        双重兜底（float 转换失败、以及下限 0.001）是为了防止 0 或负数被传给
        asyncio.wait_for——那样会立刻超时，整个工作流等于全废。
        """
        try:
            value = float(context.get("workflow_task_timeout_seconds", settings.workflow_task_timeout_seconds))
        except (TypeError, ValueError):
            value = settings.workflow_task_timeout_seconds
        return max(0.001, value)

    def _max_replans(self, context: dict[str, Any], allow_replan: bool) -> int:
        """重规划次数上限，夹在 0~2、默认 1。

        为什么硬封顶 2：每次 replan 都是一整轮 Agent 执行（含多次 LLM 调用），
        允许无限重规划会让一次请求的成本失控，且收益在第 3 轮后基本为零。
        """
        if not allow_replan or context.get("disable_replan"):
            return 0
        try:
            requested = int(context.get("max_replans", 1))
        except (TypeError, ValueError):
            requested = 1
        return max(0, min(2, requested))

    def _replan_reason(self, results: list[WorkflowTaskResult], observation: WorkflowObservation) -> str | None:
        """判断要不要再跑一轮，返回原因字符串（None 表示不需要）。

        判定顺序体现优先级：先看 Observer 的「可修复」类 finding（最具体、最可行动），
        再看是否有任务报错，最后才用置信度兜底。
        不可修复的类型（如 ci_merge_gate、cross_agent_conflict）**不在列表里**——
        它们需要人改代码，重跑一遍也还是同样的结论，纯属浪费。
        """
        fixable_findings = {
            "task_error",
            "task_skipped",
            "evidence_gap",
            "low_confidence",
            "missing_safety_review",
            "owner_needs_review",
            "team_profile_missing",
        }
        for finding in observation.findings:
            if finding.finding_type in fixable_findings:
                # 优先用 recommendation（可执行建议），没有才退回 message
                return f"{finding.finding_type}: {finding.recommendation or finding.message}"
        if any(result.status == "error" for result in results):
            return "task_error: at least one workflow task failed"
        # 0.45 是「结论太弱、值得补证据」的阈值，与 planner 侧的低置信度口径保持一致
        low_confidence_results = [
            result
            for result in results
            if result.status == "success" and result.confidence is not None and result.confidence < 0.45
        ]
        if low_confidence_results:
            return "low_confidence: one or more successful task results have weak confidence"
        # results 非空才判整体置信度：空结果集时 overall_confidence 是默认值，不该触发 replan
        if results and observation.overall_confidence < 0.45:
            return "low_confidence: observer confidence is below the workflow threshold"
        return None

    def _normalize_replanned_spec(
        self,
        *,
        original_spec: WorkflowSpec,
        current_spec: WorkflowSpec,
        next_spec: WorkflowSpec,
    ) -> WorkflowSpec:
        """把 replan 返回的 spec 拉回「不许变」的几条约束。

        Planner 只被允许**改任务列表**，以下字段一律以旧值为准，理由各不相同：
        - workflow_id：必须沿用最初的值，否则整条 trace 会断成两条互不相关的记录；
        - goal / trigger_message / acceptance_criteria：允许补空，但不允许改内容，
          否则「重规划」就变成了「换目标」；
        - max_parallel_tasks：沿用当前值，防止 replan 把并发调到很大绕过资源限制；
        - requires_human_confirmation：只能用 or 累加（原本要确认的，永远不会被改成不用确认），
          这是安全方向的单向棘轮。
        """
        normalized = next_spec.model_copy(deep=True)
        normalized.workflow_id = original_spec.workflow_id
        normalized.goal = normalized.goal or current_spec.goal
        normalized.trigger_message = normalized.trigger_message or current_spec.trigger_message
        if not normalized.acceptance_criteria:
            normalized.acceptance_criteria = list(current_spec.acceptance_criteria)
        normalized.max_parallel_tasks = current_spec.max_parallel_tasks
        # 单向棘轮：只有「从不需要变成需要」，没有「从需要变成不需要」
        normalized.requires_human_confirmation = (
            current_spec.requires_human_confirmation or normalized.requires_human_confirmation
        )
        return normalized

    def _same_task_graph(self, left: WorkflowSpec, right: WorkflowSpec) -> bool:
        """比较两版任务图是否等价（只看 id / 类型 / 依赖，不看描述文案）。

        只比结构不比文案：planner 每次 replan 都可能微调 objective 措辞，
        若把文案纳入比较，会被判定为「有变化」从而继续空转一轮。
        """
        def signature(spec: WorkflowSpec) -> list[tuple[str, str, tuple[str, ...]]]:
            """取出任务图的「结构签名」（id / 类型 / 依赖），用于等价比较。"""
            return [(task.id, task.task_type, tuple(task.dependencies)) for task in spec.tasks]

        return signature(left) == signature(right)

    def _tool_calls(
        self,
        spec: WorkflowSpec,
        results: list[WorkflowTaskResult],
        replan_events: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """生成「回放用」的工具调用清单。

        严格说这里不是真实 tool 调用，而是把每个任务与每次 replan 记成
        统一的 tool_call 形状，好让前端/审计用同一套组件渲染。
        """
        calls = [
            {
                # 命名成 workflow.<task_type>：与真实工具（如 list_issues）区分开，
                # 一眼能看出这是工作流内部步骤
                "tool_name": f"workflow.{result.task_type}",
                "arguments": {
                    "workflow_id": spec.workflow_id,
                    "task_id": result.task_id,
                    "agent": result.agent_name,
                },
                "status": result.status,
                "error": result.error,
            }
            for result in results
        ]
        # 重规划也算一次「调用」：它确实是 planner 的一次额外计算
        calls.extend(
            {
                "tool_name": "planner_agent.replan_workflow_spec",
                "arguments": {
                    "workflow_id": spec.workflow_id,
                    "iteration": event.get("iteration"),
                    "reason": event.get("reason"),
                },
                "status": "success",
                "error": None,
            }
            for event in replan_events
        )
        return calls

    def _citations(self, results: list[WorkflowTaskResult]) -> list[dict[str, Any]]:
        """汇总所有任务的引用，按 (type, id, title) 去重。

        多个 Agent 可能引用同一份 PR / 同一条日志（例如 pr_review 和 ci_debug
        都引用了同一个 CI run），合并后只保留一份，前端参考资料列表才不会重复。
        """
        citations: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for result in results:
            for citation in result.citations:
                # id 兼容 id / source_id 两种字段名
                key = (
                    str(citation.get("type") or ""),
                    str(citation.get("id") or citation.get("source_id") or ""),
                    str(citation.get("title") or ""),
                )
                if key in seen:
                    continue
                seen.add(key)
                citations.append(citation)
        return citations

    def _agent_events(
        self,
        round_records: list[dict[str, Any]],
        replan_events: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """生成完整的 trace 事件流，这是前端时间线与问题排查的唯一数据源。

        每轮产出三类事件，顺序即发生顺序：
        1. workflow_replan（如果是 replan 轮）+ workflow_spec：本轮要跑什么；
        2. workflow_task_result：每个任务的结局（含置信度、证据数、skill 运行时信息）；
        3. workflow_observer_result：本轮审查结论，最后一轮带 is_final=True。
        """
        events: list[dict[str, Any]] = []
        for index, record in enumerate(round_records):
            # 第 0 轮是初次执行，之后每轮前面都对应一次 replan 事件。
            # 用 index-1 对齐：replan_events[0] 引发的是 index=1 的那一轮
            if index > 0 and index - 1 < len(replan_events):
                events.append(replan_events[index - 1])
            spec = record["spec"]
            events.append(
                {
                    "type": "workflow_spec",
                    "workflow_id": spec.workflow_id,
                    "goal": spec.goal,
                    "iteration": index,
                    "is_replan": index > 0,
                    # 任务详情连同 claim 一起记录：事后能复盘「这一步被允许访问哪些数据源」
                    "tasks": [
                        {
                            "id": task.id,
                            "agent_name": task.agent_name,
                            "task_type": task.task_type,
                            "dependencies": task.dependencies,
                            "claim": task.claim.model_dump(mode="json"),
                        }
                        for task in spec.tasks
                    ],
                }
            )
            for result in record["results"]:
                # _skill_runtime 是部分 Agent 挂在 output 里的技能运行时元信息。
                # isinstance 判空是必需的：output 不保证是 dict
                skill_runtime = result.output.get("_skill_runtime") if isinstance(result.output, dict) else None
                events.append(
                    {
                        "type": "workflow_task_result",
                        "workflow_id": spec.workflow_id,
                        "iteration": index,
                        "task_id": result.task_id,
                        "agent_name": result.agent_name,
                        "task_type": result.task_type,
                        "status": result.status,
                        "summary": result.summary,
                        "confidence": result.confidence,
                        "evidence_count": result.evidence_count,
                        # 白名单展平 skill 运行时字段：只透出这 8 个，
                        # 避免把整个 skill runtime（可能很大）写进 trace
                        **(
                            {
                                key: value
                                for key, value in skill_runtime.items()
                                if key
                                in {
                                    "skill_name",
                                    "skill_title",
                                    "skill_version",
                                    "skill_steps",
                                    "skill_activation_mode",
                                    "skill_activation_reason",
                                    "skill_instruction_digest",
                                    "validation",
                                }
                            }
                            if isinstance(skill_runtime, dict)
                            else {}
                        ),
                    }
                )
            observation = record["observation"]
            events.append(
                {
                    "type": "workflow_observer_result",
                    "workflow_id": spec.workflow_id,
                    "iteration": index,
                    # 只有最后一轮是最终结论：前面各轮的 finding 可能已被后续修复
                    "is_final": index == len(round_records) - 1,
                    "overall_confidence": observation.overall_confidence,
                    "human_review_required": observation.human_review_required,
                    "findings": [finding.model_dump(mode="json") for finding in observation.findings],
                }
            )
        return events
