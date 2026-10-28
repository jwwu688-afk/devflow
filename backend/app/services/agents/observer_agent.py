"""Observer 审查 Agent（编排链的质检环节）。

位置：Planner → 专用 Agent → **Observer** → Synthesis。
它不生产业务结论，只做「挑毛病」：把各 Agent 的输出当作证据来审，
输出一组 findings（每条带 severity / finding_type / recommendation）和一个总体置信度。

为什么这一层必须是纯规则、不调 LLM：
- 它要在**所有** Agent 之后跑，如果再调一次模型，成本与延迟都翻倍；
- 更重要的是它的判断要被信任。规则化的判定（「CI 说阻塞合并，PR 说可以合」）
  比模型的「我觉得有冲突」更可解释、更可复现，也更容易被测。

finding_type 的取值不是随意的字符串——synthesis_agent 的 _decision 会按
"ci_merge_gate" / "cross_agent_conflict" / "human_confirmation_required" 做分支，
改动这里等于改动最终对外结论。
"""

from __future__ import annotations

from typing import Any

from app.schemas.workflow import WorkflowFinding, WorkflowObservation, WorkflowSpec, WorkflowTaskResult
from app.services.agents.base import BaseAgent


class ObserverAgent(BaseAgent):
    """对工作流输出做六类检查，产出 findings 与整体置信度。"""

    name = "observer_agent"
    description = "审查工作流任务输出中的冲突、证据缺口、低置信度和人工确认门槛。"
    available_tools = ["inspect_workflow_outputs"]

    async def run(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """跑六个独立检查器，汇总成 WorkflowObservation。

        输入：input_data["spec"]（任务图，提供 requires_human_confirmation 等元信息）、
              ["task_results"]（各 Agent 结果，可为空）；context 里读 team_members。
        输出：observation.model_dump(mode="json") —— 这里用 json 模式是因为结果要直接
              落库/进 trace，datetime 等类型必须变成可序列化值。
        """
        spec = WorkflowSpec.model_validate(input_data["spec"])
        results = [WorkflowTaskResult.model_validate(item) for item in input_data.get("task_results") or []]
        findings: list[WorkflowFinding] = []

        # 六个检查器按「从硬到软」排列：先看任务是否跑成，再看证据、置信度，
        # 最后才是跨 Agent 冲突与流程合规。每个检查器只负责一类问题，互不依赖，方便单独测试。
        findings.extend(self._status_findings(results))
        findings.extend(self._evidence_findings(results))
        findings.extend(self._confidence_findings(results))
        findings.extend(self._merge_gate_findings(results))
        findings.extend(self._ownership_findings(results, context))
        findings.extend(self._safety_findings(results, spec))

        # 整体置信度 = 成功任务的自报置信度均值，再按问题数量扣分。
        # 只用 status == "success" 的：失败任务没给置信度，跳过而不是按 0 计入，
        # 否则一次失败就会把整体拉到最低，失去区分度
        confidence_values = [item.confidence for item in results if item.confidence is not None and item.status == "success"]
        # 没有任何成功任务时给 0.55：这是个「中性偏低」的默认值，表示结论不可信但也不至于完全否掉
        base_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.55
        # 扣分系数：blocker 每个 -0.2（一次就能显著压低），warning -0.05（需要累积才明显）
        blocker_penalty = 0.2 * len([item for item in findings if item.severity == "blocker"])
        warning_penalty = 0.05 * len([item for item in findings if item.severity == "warning"])
        # 夹在 [0.1, 0.95]：下限保证不会出现 0（0 会被前端当成「无数据」），
        # 上限永远不到 1，因为规则审查本身不构成确定性结论
        overall_confidence = round(max(0.1, min(0.95, base_confidence - blocker_penalty - warning_penalty)), 2)

        observation = WorkflowObservation(
            findings=findings,
            overall_confidence=overall_confidence,
            # 两个来源取或：spec 显式声明需要人确认，或本次审查发现了任一 blocker。
            # 这条会直接决定前端是否弹确认框，宁可多弹也不能漏
            human_review_required=spec.requires_human_confirmation or any(item.severity == "blocker" for item in findings),
            summary=self._summary(findings, overall_confidence),
        )
        return observation.model_dump(mode="json")

    def _status_findings(self, results: list[WorkflowTaskResult]) -> list[WorkflowFinding]:
        """检查任务执行状态：error 是阻塞，skipped 只是警告。

        区别对待的理由：skipped 往往是「没有相关数据」（比如仓库没配 CI），
        属于信息缺失；error 是「本该跑却跑挂了」，属于确定性缺陷。
        """
        findings = []
        for result in results:
            if result.status == "error":
                findings.append(
                    WorkflowFinding(
                        finding_type="task_error",
                        severity="blocker",
                        message=f"{result.agent_name} 执行失败：{result.error or '未知错误'}",
                        task_ids=[result.task_id],
                        recommendation="先修复任务失败，或用更窄范围重新运行，再信任综合结论。",
                    )
                )
            elif result.status == "skipped":
                findings.append(
                    WorkflowFinding(
                        finding_type="task_skipped",
                        severity="warning",
                        message=f"{result.agent_name} 被跳过，因此工作流覆盖不完整。",
                        task_ids=[result.task_id],
                        recommendation="如果该领域会影响决策，请先连接或同步缺失的仓库信号。",
                    )
                )
        return findings

    def _evidence_findings(self, results: list[WorkflowTaskResult]) -> list[WorkflowFinding]:
        """检查「结论有没有证据支撑」。

        只审成功任务——失败任务的问题已经由 _status_findings 报过了，重复报会重复扣分。
        repo_health / safety_review 两类豁免：它们的产出本身就是统计口径或流程判定，
        天然不带 evidence_count / citations，要求它们给证据属于误伤。
        """
        findings = []
        for result in results:
            if result.status != "success" or result.task_type in {"repo_health", "safety_review"}:
                continue
            # 两个字段都为空才算「无证据」：只要给了引用，即使没数 evidence_count 也算有据可查
            if result.evidence_count <= 0 and not result.citations:
                findings.append(
                    WorkflowFinding(
                        finding_type="evidence_gap",
                        severity="warning",
                        message=f"{result.agent_name} 没有返回明确证据或引用。",
                        task_ids=[result.task_id],
                        recommendation="在做高风险决策前，优先同步相关 Issue、PR 文件、CI 日志，并实时搜索当前代码。",
                    )
                )
        return findings

    def _confidence_findings(self, results: list[WorkflowTaskResult]) -> list[WorkflowFinding]:
        """标出置信度低于 0.5 的任务。

        0.5 是「基本靠猜」的分界线：低于它对高风险决策不够用，但对低风险场景还能接受，
        所以只给 warning 而不是 blocker。
        """
        findings = []
        for result in results:
            if result.status == "success" and result.confidence is not None and result.confidence < 0.5:
                findings.append(
                    WorkflowFinding(
                        finding_type="low_confidence",
                        severity="warning",
                        message=f"{result.agent_name} 的置信度偏低（{result.confidence:.2f}）。",
                        task_ids=[result.task_id],
                        recommendation="行动前请补充更多上下文，或检查底层证据。",
                    )
                )
        return findings

    def _merge_gate_findings(self, results: list[WorkflowTaskResult]) -> list[WorkflowFinding]:
        """合入门禁检查：CI 阻塞、PR 阻塞问题、以及两者结论互相矛盾。

        这是 Observer 最有价值的一段——单个 Agent 各自都对，但结论冲突时只有
        跨 Agent 的视角才能发现，所以冲突判定放在这一层。
        """
        findings = []
        pr_result = self._by_type(results, "pr_review")
        ci_result = self._by_type(results, "ci_debug")
        # CI 侧的阻塞判定直接复用 ci_debug_agent 的结论，不在这里重新算，避免两处口径不一致
        if ci_result and bool(ci_result.output.get("is_merge_blocking")):
            findings.append(
                WorkflowFinding(
                    finding_type="ci_merge_gate",
                    severity="blocker",
                    message="CI Agent 将最新失败 workflow 标记为阻塞合并。",
                    task_ids=[ci_result.task_id],
                    recommendation="批准合并或发布前，请先修复或解释 CI 失败。",
                )
            )
        # PR 侧只要有 blocking_issues 就是阻塞，哪怕模型自己在 merge_recommendation 里说可以合
        if pr_result and pr_result.output.get("blocking_issues"):
            findings.append(
                WorkflowFinding(
                    finding_type="pr_blocking_issues",
                    severity="blocker",
                    message="PR 审查发现了阻塞问题。",
                    task_ids=[pr_result.task_id],
                    recommendation="解决 PR 阻塞问题后重新运行工作流。",
                )
            )
        # 跨 Agent 冲突：PR 倾向合并 + CI 判阻塞。两个 task_id 都挂上，方便前端定位到具体两条结果
        if pr_result and ci_result and bool(ci_result.output.get("is_merge_blocking")):
            if self._looks_like_merge_approval(str(pr_result.output.get("merge_recommendation") or "")):
                findings.append(
                    WorkflowFinding(
                        finding_type="cross_agent_conflict",
                        severity="blocker",
                        message="PR Agent 倾向合并，但 CI Agent 报告了阻塞合并的失败。",
                        task_ids=[pr_result.task_id, ci_result.task_id],
                        recommendation="在失败解决前，让 CI 门禁优先于 PR 合并建议。",
                    )
                )
        return findings

    def _ownership_findings(self, results: list[WorkflowTaskResult], context: dict[str, Any]) -> list[WorkflowFinding]:
        """检查 Issue 负责人建议是否可信（是否在已配置的团队成员里）。

        仅在「有 Issue 分析结果」时才有意义，没有就直接跳过，避免产出无关噪音。
        """
        issue_result = self._by_type(results, "issue_analysis")
        if not issue_result:
            return []
        # 小写化后比较，避免 "Alice" / "alice" 被判成不存在
        owner = str(issue_result.output.get("suggested_owner") or "").strip().lower()
        if not owner:
            return []
        team_members = context.get("team_members") or []
        # 先把成员名集合建好再比较：成员列表通常十几条，逐次遍历没必要
        known_names = {
            str(member.get("name") or "").strip().lower()
            for member in team_members
            if isinstance(member, dict) and str(member.get("name") or "").strip()
        }
        # 这两个是模型在「不知道该给谁」时的常见占位输出，等同于没给建议
        unassigned_markers = {"unassigned", "待分配"}
        # 注意条件顺序：先判「给了个不存在的名字」，再判「根本没有任何成员配置」，
        # 两种情况措辞不同——前者是名字错了，后者是配置缺失，用户要做的动作不一样
        if owner in unassigned_markers or (known_names and owner not in known_names):
            return [
                WorkflowFinding(
                    finding_type="owner_needs_review",
                    severity="warning",
                    message="Issue 负责人建议为空，或不在已配置团队成员中。",
                    task_ids=[issue_result.task_id],
                    recommendation="依赖自动负责人分配前，请先更新团队成员画像。",
                )
            ]
        if not known_names:
            return [
                WorkflowFinding(
                    finding_type="team_profile_missing",
                    severity="warning",
                    message="当前还没有配置用于负责人路由的 DevFlow 团队成员。",
                    task_ids=[issue_result.task_id],
                    recommendation="请添加团队成员画像，让负责人推荐更可解释。",
                )
            ]
        return []

    def _safety_findings(self, results: list[WorkflowTaskResult], spec: WorkflowSpec) -> list[WorkflowFinding]:
        """安全合规检查：该跑 SafetyAgent 却没跑，或跑了但要求人工确认。

        这两种都是 blocker：前者是流程缺失（等于没上闸门），后者是闸门确实拦住了。
        """
        safety_result = self._by_type(results, "safety_review")
        # 声明需要人确认但压根没跑安全审查 —— 属于流程漏洞，不能当作「没问题」放行
        if not safety_result and spec.requires_human_confirmation:
            return [
                WorkflowFinding(
                    finding_type="missing_safety_review",
                    severity="blocker",
                    message="工作流需要人工确认，但没有运行 SafetyAgent 任务。",
                    recommendation="在起草评论、标签、创建/关闭 Issue 或外部发送前，请先运行 SafetyAgent。",
                )
            ]
        if safety_result and safety_result.output.get("requires_confirmation"):
            return [
                WorkflowFinding(
                    finding_type="human_confirmation_required",
                    severity="blocker",
                    message="SafetyAgent 要求用户对本次写入类操作进行人工确认。",
                    task_ids=[safety_result.task_id],
                    recommendation="只返回草稿；MVP 模式下不要执行外部动作。",
                )
            ]
        return []

    def _by_type(self, results: list[WorkflowTaskResult], task_type: str) -> WorkflowTaskResult | None:
        """按 task_type 取第一个成功的结果；失败结果不参与门禁判定（没有可信 output）。"""
        return next((item for item in results if item.task_type == task_type and item.status == "success"), None)

    def _looks_like_merge_approval(self, value: str) -> bool:
        """粗判一句 merge_recommendation 是不是「同意合并」。

        两个条件同时成立才算：命中正向词，且**没有**命中条件性否定的词。
        例如 "可以合并，但需修复后" 同时含 "合并" 和 "修复后"，判为否——
        这正是我们要的：带条件的同意不构成与 CI 的冲突。
        这是字符串启发式，措辞大改（比如模型换成英文长句）时需要同步维护词表。
        """
        lowered = value.lower()
        return any(token in lowered for token in ["approve", "merge", "建议合并", "建议合入"]) and not any(
            token in lowered for token in ["after", "blocked", "暂缓", "修复后", "修改后"]
        )

    def _summary(self, findings: list[WorkflowFinding], confidence: float) -> str:
        """一句话总结，供 synthesis 的「Agent 输出摘要」直接引用。"""
        blockers = len([item for item in findings if item.severity == "blocker"])
        warnings = len([item for item in findings if item.severity == "warning"])
        # 分三档措辞：有阻塞 / 只有警告 / 干净。置信度都带上，方便读者判断结论强度
        if blockers:
            return f"Observer 发现 {blockers} 个阻塞项和 {warnings} 个警告；置信度 {confidence:.2f}。"
        if warnings:
            return f"Observer 发现 {warnings} 个警告；置信度 {confidence:.2f}。"
        return f"Observer 没有发现跨 Agent 阻塞项；置信度 {confidence:.2f}。"
