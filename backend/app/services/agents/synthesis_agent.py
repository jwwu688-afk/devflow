"""工作流综合 Agent（编排链的最后一环）。

位置：Planner → 专用 Agent → Observer → **Synthesis**。
输入是三份结构化对象（spec 任务图、每个 Agent 的结果、Observer 的跨域检查结论），
输出是一份人读的 Markdown 决策备忘录。

设计取舍：这里**不调用 LLM**，完全是规则 + 模板拼接。
理由：综合层要做的是「忠实汇总别人的结论」，一旦引入模型就会改写数字、编造证据，
而且同输入结果不稳定，无法用于对外汇报。措辞灵活性让位给可复现性。
副作用是排版与措辞比较机械，调整文风只能改这里的代码。
"""

from __future__ import annotations

from typing import Any

from app.schemas.workflow import WorkflowObservation, WorkflowSpec, WorkflowTaskResult
from app.services.agents.base import BaseAgent


class SynthesisAgent(BaseAgent):
    """把多 Agent 工作流结果综合成工程决策备忘录。"""

    name = "synthesis_agent"
    description = "将多 Agent 工作流综合成一份工程决策备忘录。"
    available_tools = ["synthesize_workflow_report"]

    async def run(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """入口：把 dict 输入转成强类型对象后渲染备忘录。

        输入：input_data["spec"]（任务图）、["task_results"]（各 Agent 结果列表，
              可能为空）、["observation"]（Observer 结论），全部按 pydantic 校验，
              字段缺失/类型不符会在这一步直接抛错，而不是在模板拼装时才炸。
        输出：{"answer": str}，与 chat 链路的字段名保持一致，方便前端复用渲染逻辑。
        """
        spec = WorkflowSpec.model_validate(input_data["spec"])
        # task_results 允许为空列表：全部 Agent 失败时也要能给出一份「什么都没跑成」的备忘录
        results = [WorkflowTaskResult.model_validate(item) for item in input_data.get("task_results") or []]
        observation = WorkflowObservation.model_validate(input_data["observation"])
        answer = self._build_answer(spec, results, observation)
        return {"answer": answer}

    def _build_answer(self, spec: WorkflowSpec, results: list[WorkflowTaskResult], observation: WorkflowObservation) -> str:
        """按固定章节顺序拼出备忘录。

        章节顺序是「结论先行」：总体判断在第二行，读者不必翻到最后才知道能不能合。
        """
        # 按严重度拆开：blocker 决定能不能推进，warning 只影响「是否需要补充证据」
        blockers = [item for item in observation.findings if item.severity == "blocker"]
        warnings = [item for item in observation.findings if item.severity == "warning"]
        decision = self._decision(blockers, warnings, results)

        lines = [
            "## 多 Agent 协作结论",
            "",
            f"目标：{spec.goal}",
            f"总体判断：{decision}",
            # 置信度保留两位小数：Observer 给的是 0~1 浮点，直接打印会出现一长串尾数
            f"Observer 置信度：{observation.overall_confidence:.2f}",
            "",
            "## 协作任务图",
        ]
        # 先讲「谁负责什么、依赖谁」，读者才能看懂后面各 Agent 的输出为何是这个顺序
        for task in spec.tasks:
            deps = f"，依赖：{', '.join(task.dependencies)}" if task.dependencies else ""
            lines.append(f"- {task.id}: {task.agent_name} 负责 {task.objective}{deps}")

        # 摘要段落里失败/跳过的任务也会列出（带 status），因为「哪个 Agent 没跑成」本身是信息
        lines.extend(["", "## Agent 输出摘要"])
        for result in results:
            lines.append(f"- {result.agent_name} [{result.status}]: {result.summary or '无摘要。'}")

        lines.extend(["", "## Observer 检查"])
        if observation.findings:
            for finding in observation.findings:
                # severity 大写是为了在纯文本 Markdown 里更醒目（没有颜色可用）
                lines.append(f"- {finding.severity.upper()}: {finding.message} 建议：{finding.recommendation}")
        else:
            # 「没发现问题」也要显式写出来，否则读者无法区分「检查过且没问题」和「没检查」
            lines.append("- 未发现跨 Agent 结论冲突。")

        lines.extend(["", "## 决策依据与待确认项"])
        lines.extend(f"- {item}" for item in self._decision_basis(results, observation))

        next_steps = self._next_steps(blockers, warnings, results)
        lines.extend(["", "## 下一步"])
        lines.extend(f"- {item}" for item in next_steps)
        return "\n".join(lines).strip()

    def _decision(
        self,
        blockers: list[Any],
        warnings: list[Any],
        results: list[WorkflowTaskResult],
    ) -> str:
        """把「阻塞项 + warning + 各 Agent 结果」压成一句总体判断。

        判定是有优先级的，顺序即重要性：先看有没有硬阻塞，再看 PR Agent 的具体建议，
        最后才看 warning。改这里的顺序会直接改变对外结论，属于高风险改动。
        """
        if blockers:
            # CI merge gate / 跨 Agent 冲突这两类 blocker 性质不同：前者是机器判定，
            # 后者是两个 Agent 结论矛盾，都需要「先停手」，所以单独给一句更具体的措辞
            if any(item.finding_type in {"ci_merge_gate", "cross_agent_conflict"} for item in blockers):
                return "暂缓合并或发布，先处理 CI/跨 Agent 冲突。"
            # 需要人工确认的情况单独提示：这类阻塞不是「有问题」，而是「流程还没走完」
            if any(item.finding_type == "human_confirmation_required" for item in blockers):
                return "只生成草稿，等待人工确认后再执行外部写操作。"
            return "存在阻塞项，当前不建议直接推进。"
        # 没有阻塞时优先复用 PR Agent 的合入建议，避免综合层自己另造一套判断口径
        pr_result = self._by_type(results, "pr_review")
        if pr_result and pr_result.output.get("merge_recommendation"):
            return f"按 PR Agent 建议推进：{pr_result.output.get('merge_recommendation')}。"
        if warnings:
            # warning 不阻断，但要在结论里点明「还欠东西」，避免读者以为已经完全就绪
            return "可以继续推进，但需要补齐 Observer 标出的证据或团队信息。"
        return "可以继续推进，未发现明显跨域阻塞。"

    def _next_steps(
        self,
        blockers: list[Any],
        warnings: list[Any],
        results: list[WorkflowTaskResult],
    ) -> list[str]:
        """汇总可执行的下一步动作，去重后最多 6 条。

        顺序按「离动手最近」排：CI 修复步骤 → PR 阻塞项/测试建议 → Issue 清单 → Observer 建议。
        每条来源都截断（[:3] / [:2]），因为备忘录的下一步清单过长就没人看了。
        """
        steps: list[str] = []
        # 通过 task_type 找结果而不是按 Agent 类名，避免改名/换实现后这里静默失效
        ci_result = self._by_type(results, "ci_debug")
        pr_result = self._by_type(results, "pr_review")
        issue_result = self._by_type(results, "issue_analysis")

        # 只有 CI 明确判定「阻塞合入」时才把调试步骤顶上来，否则会淹没其他动作
        if ci_result and ci_result.output.get("is_merge_blocking"):
            # if str(item).strip() 过滤空串：模型有时会返回空条目，留着会渲染出空 bullet
            debug_steps = [str(item) for item in ci_result.output.get("debug_steps") or [] if str(item).strip()]
            # 有具体步骤就取前 3 条；一条都没有时给一句兜底，不能让 CI 阻塞却没有下一步
            steps.extend(debug_steps[:3] or ["先定位并修复最新失败 CI。"])
        if pr_result:
            blocking = [str(item) for item in pr_result.output.get("blocking_issues") or [] if str(item).strip()]
            tests = [str(item) for item in pr_result.output.get("test_suggestions") or [] if str(item).strip()]
            # 阻塞项优先于测试建议：先能合，再谈覆盖
            steps.extend(blocking[:2])
            steps.extend(tests[:2])
        if issue_result:
            # checklist 与 action_items 是 Issue Agent 不同版本/不同路径的字段名，两个都认，
            # 用 or 串联可以在旧结果上继续工作（question: 兼容性兜底）
            actions = [str(item) for item in issue_result.output.get("checklist") or issue_result.output.get("action_items") or [] if str(item).strip()]
            steps.extend(actions[:2])
        # Observer 的建议放最后：它们是「元层面」的提醒（证据不足、信息缺失），不直接可执行
        for finding in [*blockers, *warnings]:
            if finding.recommendation:
                steps.append(str(finding.recommendation))
        if not steps:
            steps.append("把本次 workflow 结论回填到 PR/Issue 描述或团队同步中。")
        # dict.fromkeys 去重且保持插入顺序；最后再截到 6 条，防止多来源叠加后清单爆炸
        return list(dict.fromkeys(steps))[:6]

    def _decision_basis(self, results: list[WorkflowTaskResult], observation: WorkflowObservation) -> list[str]:
        """列出「结论是基于什么得出的」，让读者能回溯每条判断的来源。

        每条都带证据条数与置信度（有才写），这是备忘录可信度的支撑材料。
        """
        basis: list[str] = []
        for result in results:
            # 置信度/证据数可能为 None 或 0，分别处理：None 说明该 Agent 没给，0 说明确实没证据
            confidence = f"，置信度 {result.confidence:.2f}" if result.confidence is not None else ""
            evidence = f"，证据 {result.evidence_count} 条" if result.evidence_count else ""
            basis.append(f"{result.agent_name}：{result.summary or '无摘要'}{evidence}{confidence}。")
        if observation.findings:
            for finding in observation.findings:
                basis.append(f"待确认：{finding.message} 处理建议：{finding.recommendation}")
        else:
            # 即使 Observer 没提问题，也要提醒「外部写操作仍需人工确认」——这是流程约束，
            # 不能因为自动检查通过了就默认可以对外发东西
            basis.append("待确认：最终合入、评论或外部发送仍需要人工按团队流程确认。")
        # 只留 8 条：依据段落是辅助阅读的，太长会把正文挤下去
        return basis[:8]

    def _by_type(self, results: list[WorkflowTaskResult], task_type: str) -> WorkflowTaskResult | None:
        """按 task_type 取第一个**成功**的结果。

        只认 status == "success"：失败结果里的 output 可能缺字段或只有半份，
        拿它做决策会得出错误结论。失败任务仍会在「Agent 输出摘要」里列出，不会丢失信息。
        """
        return next((item for item in results if item.task_type == task_type and item.status == "success"), None)
