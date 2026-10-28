"""多 Agent 工程工作流（WorkflowOrchestrator）的编排测试。

验证行为：一次"能不能合并"的请求会被拆成规划（Planner）→ 并行执行（各 Agent）
→ 观测（Observer）→ 汇总四步，本文件逐条锁住这条链路上的关键约束：
1) 规划要覆盖 Issue / PR / CI 多个域，并给出验收标准；
2) **目标锁定**：用户在消息里点名的 #142 / #86 / CI #9001 必须被严格遵守，
   即使之后数据库里出现了更新的记录，也不能被"最新记录"顶替；
3) 点名了不存在的目标时要报错，绝不静默回退到最新一条；
4) 跨仓库的 claim（篡改过的 target_id）必须被拒绝；
5) 互不依赖的任务要真并行（用耗时断言）、冲突时 Observer 要拦住合并；
6) 证据不足时触发一次 replan 去补 RAG 检索；
7) 已确认的计划不再重规划；卡死任务要超时；依赖失败要级联跳过；
8) 需要人工确认的动作，Observer 必须报 missing_safety_review；
9) 聊天工具调用要把 workflow / task 运行记录落库。
"""

import asyncio
import time
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.api.routes.chat import WorkflowTargetResolutionError, _build_conversation_tools, _build_engineering_workflow_runtime
from app.core.config import settings
from app.db.models import AgentTaskRun, AgentWorkflowRun, Conversation, Issue, PRFile, PullRequest, Repository, WorkflowRun
from app.db.session import Base
from app.schemas.workflow import WorkflowClaim, WorkflowSpec, WorkflowTask, WorkflowTaskResult
from app.services.agents.observer_agent import ObserverAgent
from app.services.agents.planner_agent import PlannerAgent
from app.services.agents.pr_review_agent import PRReviewAgent
from app.services.agents.workflow_orchestrator import WorkflowOrchestrator
from app.services.skills import get_skill_registry


@pytest.mark.asyncio
async def test_planner_builds_cross_domain_workflow_for_merge_readiness() -> None:
    """测的是规划阶段的任务拆分能力。

    场景：用户问「这个 PR 能不能合并」，并给出上下文：有 issue、有 PR、
    有失败 CI。
    期望：生成的任务里同时包含 repo_health、issue_analysis、pr_review、
    ci_debug 四类，且带非空的 acceptance_criteria。
    重要性：合并判断本质是多域信息的综合题，少了任何一个域都可能放过风险
    （例如只看代码不看 CI）；acceptance_criteria 则让"做完没做完"可判定，
    否则 Agent 容易交一份看似完整实则无结论的报告。
    """
    result = await PlannerAgent().run(
        {"message": "这个 PR 能不能合并？请做多 agent 协作 review，检查 Issue、PR 和 CI 风险。"},
        {
            "repo_id": "repo-1",
            "has_issue": True,
            "issue_id": "issue-1",
            "has_pr": True,
            "pr_id": "pr-1",
            "has_failed_ci": True,
            "workflow_run_id": "run-1",
        },
    )

    task_types = [task["task_type"] for task in result["tasks"]]
    assert "repo_health" in task_types
    assert "issue_analysis" in task_types
    assert "pr_review" in task_types
    # 有失败 CI 时才会规划 ci_debug，说明规划是依据上下文条件推导的
    assert "ci_debug" in task_types
    # 缺了验收标准，报告就无法判断"是否真的完成了"
    assert result["acceptance_criteria"]


@pytest.mark.asyncio
async def test_confirmed_workflow_keeps_explicit_issue_pr_and_ci_claims_after_newer_records_arrive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """测的是"目标锁定"：用户点名的目标不能被后来的新记录顶替。

    场景：库里先有 Issue #142 / PR #86 / CI #9001；用户消息里明确点名这三个；
    规划完成后，又插入了更"新"的 Issue #999 / PR #999 / CI #9999。
    期望：
    - 规划出的 claim 仍指向 #142 / #86 / #9001，objective 里也保留号码；
    - 但执行上下文（新的对话轮次）会解析到最新的三条（这是"确认执行"时
      的默认目标解析行为，与已确认的 spec 无关）；
    - 真正执行已确认的 spec 时，各任务摘要仍是 #142 / #86 / requested-ci，
      即 spec 里的 claim 覆盖了执行上下文的默认目标；
    - PR 审查收到的 number 为 86、ci_summary 准确、技能为 pr-review
      且正文含 P1/P2 判定标准；输出标记 entrypoint_completed。
    重要性：这是整个编排最关键的一致性保证。若不锁定目标，用户在确认框里
    看到的是"审查 PR #86"，实际却被换成了 #999——报告看起来正常，结论却
    完全错位，且极难被发现。这也解释了为什么 execution_context 变化不影响结果。
    """
    # 关闭 LLM，走确定性路径：规划与审查都走本地规则，结果可复现
    monkeypatch.setattr(settings, "llm_api_key", None)
    # 捕获 PR 审查实际收到的入参，用于验证"确实审的是 #86"
    captured_pr_input: dict[str, object] = {}

    async def capture_pr_run(self, input_data, context):
        """替换 PRReviewAgent.run：只记录关键入参，返回一份固定结构的结论。"""
        captured_pr_input.update(
            {
                "number": input_data["number"],
                "ci_summary": context["ci_summary"],
                "skill_name": context["skill_runtime"]["skill_name"],
                "skill_instructions": context["skill_instructions"],
            }
        )
        return {
            "merge_recommendation": "暂缓合并",
            "risk_points": [],
            "blocking_issues": [],
            "test_suggestions": [],
            "confidence": 0.8,
        }

    monkeypatch.setattr(PRReviewAgent, "run", capture_pr_run)
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    now = datetime.now(timezone.utc)
    with Session(engine) as db:
        repo = Repository(owner="acme", name="targets", full_name="acme/targets")
        db.add(repo)
        db.flush()
        # 三条"被点名"的记录，created_at 都设为 now，稍后会被 newer_at 的记录盖过
        requested_issue = Issue(
            repo_id=repo.id,
            number=142,
            title="Requested issue",
            body="Repro steps and user impact are documented.",
            labels=["bug"],
            state="open",
            created_at=now,
        )
        requested_pr = PullRequest(
            repo_id=repo.id,
            number=86,
            title="Requested PR",
            body="Fixes the requested issue.",
            state="open",
            author="alice",
            created_at=now,
        )
        requested_run = WorkflowRun(
            repo_id=repo.id,
            github_run_id=9001,
            name="requested-ci",
            status="completed",
            conclusion="failure",
            jobs=[{"name": "pytest", "conclusion": "failure"}],
            logs_text="requested test failed",
            created_at=now,
        )
        db.add_all([requested_issue, requested_pr, requested_run])
        db.flush()
        # PR 需要一个文件改动，否则审查可能因"空 diff"直接跳过
        db.add(PRFile(pr_id=requested_pr.id, filename="backend/app/target.py", status="modified", additions=3, deletions=1, patch="target fix"))
        db.commit()

        message = "请用多 Agent 检查 Issue #142、PR #86 和 CI #9001，判断能不能合并。"
        skill_context = {"_skill_registry": get_skill_registry()}
        # 第一轮：从消息里解析出被点名的目标，用于规划
        planning_context, _ = _build_engineering_workflow_runtime(repo, db, message, skill_context)
        spec = WorkflowSpec.model_validate(await PlannerAgent().run({"message": message}, planning_context))
        claims = {task.task_type: task.claim.target_id for task in spec.tasks}
        # 三条 claim 都必须精确指向被点名的记录（内部用 uuid 而非 #number）
        assert claims["issue_analysis"] == str(requested_issue.id)
        assert claims["pr_review"] == str(requested_pr.id)
        assert claims["ci_debug"] == str(requested_run.id)
        # objective 里保留人类可读的号码，用户确认时看的就是这段文案
        assert "Issue #142" in next(task.objective for task in spec.tasks if task.task_type == "issue_analysis")
        assert "PR #86" in next(task.objective for task in spec.tasks if task.task_type == "pr_review")

        # 插入更"新"的记录，制造"用最新记录替换"的诱惑
        newer_at = now + timedelta(hours=1)
        newer_issue = Issue(repo_id=repo.id, number=999, title="Newer issue", state="open", created_at=newer_at)
        newer_pr = PullRequest(repo_id=repo.id, number=999, title="Newer PR", state="open", created_at=newer_at)
        newer_run = WorkflowRun(
            repo_id=repo.id,
            github_run_id=9999,
            name="newer-ci",
            status="completed",
            conclusion="failure",
            jobs=[],
            created_at=newer_at,
        )
        db.add_all([newer_issue, newer_pr, newer_run])
        db.commit()

        # 第二轮：用户说"确认执行"，此时若按"最新"解析会得到 #999——这正是要防的
        execution_context, runners = _build_engineering_workflow_runtime(repo, db, "确认执行这个计划", skill_context)
        assert execution_context["issue_id"] == str(newer_issue.id)
        assert execution_context["pr_id"] == str(newer_pr.id)
        assert execution_context["workflow_run_id"] == str(newer_run.id)

        # 但执行已确认的 spec 时，结论必须仍基于 #142 / #86 / requested-ci
        report = await WorkflowOrchestrator(runners).run_spec(spec, execution_context)
        summaries = {result["task_type"]: result["summary"] for result in report["task_results"]}
        assert summaries["issue_analysis"].startswith("Issue #142:")
        assert summaries["pr_review"].startswith("PR #86:")
        assert summaries["ci_debug"].startswith("CI requested-ci:")
        # PR 审查实际收到的就是 86，而不是 999
        assert captured_pr_input["number"] == 86
        # CI 汇总统计的是整个仓库的失败数（2 条都失败），但"最新失败"锁定为 requested-ci
        assert captured_pr_input["ci_summary"] == {"failed_runs": 2, "latest_failed_run": "requested-ci"}
        # 审查过程注入了 pr-review 技能及其 P1/P2 判定标准
        assert captured_pr_input["skill_name"] == "pr-review"
        assert "P1/P2" in str(captured_pr_input["skill_instructions"])
        pr_output = next(result["output"] for result in report["task_results"] if result["task_type"] == "pr_review")
        # 技能入口被真正执行过，validation 才会是 completed
        assert pr_output["_skill_runtime"]["validation"] == "entrypoint_completed"


def test_workflow_rejects_missing_explicit_target_instead_of_using_latest() -> None:
    """测的是"点名了不存在目标"时的硬失败，而不是静默回退。

    场景：库里只有 PR #1，用户却要求审查 PR #404。
    期望：抛 WorkflowTargetResolutionError，错误信息里同时含 "PR #404"
    与「不会回退」（明确告知用户不会拿别的 PR 顶替）。
    重要性：静默回退是最危险的"友好"设计——用户以为在审 #404，
    实际拿到的是 #1 的结论。错误信息里声明"不会回退"，是为了让用户明白
    这是有意为之而非 bug。
    """
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        repo = Repository(owner="acme", name="missing", full_name="acme/missing")
        db.add(repo)
        db.flush()
        # 库里确实有数据，确保失败原因是"目标不存在"而不是"没有数据"
        db.add(PullRequest(repo_id=repo.id, number=1, title="Only PR", state="open"))
        db.commit()

        with pytest.raises(WorkflowTargetResolutionError, match=r"PR #404.*不会回退"):
            _build_engineering_workflow_runtime(repo, db, "请审查 PR #404", {})


@pytest.mark.asyncio
async def test_workflow_rejects_claim_target_from_another_repository() -> None:
    """测的是 claim 的租户隔离：不能拿别的仓库的 target_id 来执行。

    场景：当前仓库 acme/one 里只有 Issue #1；另一个仓库 acme/two 有 Issue #142。
    手工构造一个 claim 指向 #142 的 id（模拟被篡改或被注入的 spec），
    再直接调用 issue_analysis 的 runner。
    期望：抛 WorkflowTargetResolutionError，信息含「不属于仓库 acme/one」。
    重要性：spec 可能是模型生成的、也可能来自客户端回传，不能信任其中的
    target_id。少了这道校验，一个仓库的对话就能读到另一个仓库的 issue 内容，
    属于越权数据访问。
    """
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        repo = Repository(owner="acme", name="one", full_name="acme/one")
        other_repo = Repository(owner="acme", name="two", full_name="acme/two")
        db.add_all([repo, other_repo])
        db.flush()
        local_issue = Issue(repo_id=repo.id, number=1, title="Local issue", state="open")
        foreign_issue = Issue(repo_id=other_repo.id, number=142, title="Foreign issue", state="open")
        db.add_all([local_issue, foreign_issue])
        db.commit()

        # 先拿到正常构建出的 runner，再用篡改过的 claim 去打它
        workflow_context, runners = _build_engineering_workflow_runtime(repo, db, "检查 Issue", {})
        task = WorkflowTask(
            id="issue_analysis",
            agent_name="issue_analyst_agent",
            task_type="issue_analysis",
            objective="Analyze a tampered claim",
            claim=WorkflowClaim(target_type="issue", target_id=str(foreign_issue.id)),
        )
        spec = WorkflowSpec(goal="tampered", trigger_message="tampered", tasks=[task])

        with pytest.raises(WorkflowTargetResolutionError, match="不属于仓库 acme/one"):
            await runners["issue_analysis"](task, spec, workflow_context)


@pytest.mark.asyncio
async def test_orchestrator_runs_ready_tasks_in_parallel_and_observer_blocks_merge() -> None:
    """测的是真并行执行 + Observer 的合并门禁。

    场景：PR 审查与 CI 排查两个任务互不依赖，各 sleep 0.12s；
    PR 侧结论"建议合并"，CI 侧结论"失败且阻塞合并"。
    期望：总耗时应小于 0.22s（串行会是 0.24s 以上，这个阈值能区分串行/并行）；
    success_count 为 2 且没有触发 replan；
    findings 里同时出现 ci_merge_gate 与 cross_agent_conflict；
    最终答案改为「暂缓合并」。
    重要性：并行是把多 Agent 延迟压下来的关键（否则 4 个任务就是 4 倍等待）；
    而 cross_agent_conflict 检查的是"PR 说能合、CI 说不能合"这类跨 Agent 矛盾——
    若简单地取多数或取后者，就会把失败 CI 的结论吞掉，直接导致带病上线。
    """

    class StaticPlanner:
        """固定返回两个互不依赖的任务，避免依赖 LLM 规划的不确定性。"""

        async def run(self, input_data, context):
            return WorkflowSpec(
                goal="Check merge readiness",
                trigger_message="merge?",
                tasks=[
                    WorkflowTask(
                        id="pr_review",
                        agent_name="pr_review_agent",
                        task_type="pr_review",
                        objective="Review PR",
                        claim=WorkflowClaim(target_type="pull_request", target_id="pr-1"),
                    ),
                    WorkflowTask(
                        id="ci_debug",
                        agent_name="ci_debug_agent",
                        task_type="ci_debug",
                        objective="Review CI",
                        claim=WorkflowClaim(target_type="workflow_run", target_id="run-1"),
                    ),
                ],
            ).model_dump(mode="json")

    async def run_pr(task, spec, context):
        # 0.12s 是刻意选的可测延迟：两个并行约 0.12s，串行约 0.24s
        await asyncio.sleep(0.12)
        return WorkflowTaskResult(
            task_id=task.id,
            agent_name=task.agent_name,
            task_type=task.task_type,
            status="success",
            summary="PR looks mergeable.",
            output={"merge_recommendation": "建议合并", "blocking_issues": []},
            evidence_count=1,
            confidence=0.8,
        )

    async def run_ci(task, spec, context):
        await asyncio.sleep(0.12)
        return WorkflowTaskResult(
            task_id=task.id,
            agent_name=task.agent_name,
            task_type=task.task_type,
            status="success",
            summary="CI is failing.",
            output={"is_merge_blocking": True, "failure_type": "test", "debug_steps": ["Fix failing tests"]},
            evidence_count=1,
            confidence=0.8,
        )

    started = time.perf_counter()
    report = await WorkflowOrchestrator(
        {"pr_review": run_pr, "ci_debug": run_ci},
        planner=StaticPlanner(),  # type: ignore[arg-type]
    ).run({"message": "merge?"}, {})
    duration = time.perf_counter() - started

    # 并行证明：若退化为串行，这条断言会稳定失败
    assert duration < 0.22
    assert report["metrics"]["success_count"] == 2
    # 两个任务都成功且证据充足，不应触发重规划
    assert report["metrics"]["replan_count"] == 0
    finding_types = [item["finding_type"] for item in report["observation"]["findings"]]
    # CI 门禁 + 跨 Agent 冲突两类发现都必须报出来
    assert "ci_merge_gate" in finding_types
    assert "cross_agent_conflict" in finding_types
    # 冲突时最终结论必须偏向保守
    assert "暂缓合并" in report["final_answer"]


@pytest.mark.asyncio
async def test_orchestrator_replans_with_rag_when_evidence_is_missing() -> None:
    """测的是证据不足时的单一重规划：补一次 RAG 检索。

    场景：PR 审查任务虽然 success，但 evidence_count=0（结论没有证据支撑）。
    期望：replan_count 为 1、execution_rounds 为 2；
    task_results 顺序为 ["pr_review", "rag_search"]；
    spec.tasks 里新增了 rag_search；agent_events 里有 workflow_replan 事件。
    重要性：没有证据的"看起来能合并"是最危险的输出。重规划把"再找证据"
    变成显式的一步，而不是让模型硬着头皮下结论；同时把重规划次数限为 1，
    防止证据永远找不齐时陷入无限循环。
    """

    class StaticPlanner(PlannerAgent):
        """继承真实 Planner 以便复用内部依赖，但 run 固定返回单任务。"""

        async def run(self, input_data, context):
            return WorkflowSpec(
                goal="Check merge readiness",
                trigger_message="merge?",
                tasks=[
                    WorkflowTask(
                        id="pr_review",
                        agent_name="pr_review_agent",
                        task_type="pr_review",
                        objective="Review PR",
                        claim=WorkflowClaim(target_type="pull_request", target_id="pr-1"),
                    )
                ],
            ).model_dump(mode="json")

    async def run_pr(task, spec, context):
        # evidence_count=0 是触发重规划的关键信号
        return WorkflowTaskResult(
            task_id=task.id,
            agent_name=task.agent_name,
            task_type=task.task_type,
            status="success",
            summary="PR looks mergeable, but no concrete evidence was attached.",
            output={"merge_recommendation": "approve"},
            evidence_count=0,
            confidence=0.72,
        )

    async def run_rag(task, spec, context):
        # 补检回来的证据：1 条 citation，确认"补齐"动作真的产出了东西
        return WorkflowTaskResult(
            task_id=task.id,
            agent_name=task.agent_name,
            task_type=task.task_type,
            status="success",
            summary="Recovered related historical context.",
            output={"answer": "Found similar PR history."},
            evidence_count=1,
            confidence=0.68,
            citations=[{"type": "pull_request", "id": "pr-0", "title": "Similar PR"}],
        )

    report = await WorkflowOrchestrator(
        {"pr_review": run_pr, "rag_search": run_rag},
        planner=StaticPlanner(),
    ).run({"message": "merge?"}, {"repo_id": "repo-1"})

    assert report["metrics"]["replan_count"] == 1
    assert report["metrics"]["execution_rounds"] == 2
    # rag_search 只跑在第二轮，排在原任务之后
    assert [item["task_id"] for item in report["task_results"]] == ["pr_review", "rag_search"]
    assert "rag_search" in [task["task_type"] for task in report["spec"]["tasks"]]
    # 重规划要留下显式事件，前端才能提示"已补充检索"
    assert any(event["type"] == "workflow_replan" for event in report["agent_events"])


@pytest.mark.asyncio
async def test_orchestrator_runs_confirmed_spec_without_replanning() -> None:
    """测的是"已确认的计划"必须原样执行，不得再规划。

    场景：传入一份用户已确认的 spec，planner 换成"一被调用就 AssertionError"的哨兵。
    期望：不抛异常；返回的 workflow_id 与传入 spec 一致；task_count 为 1；
    执行结果就是那个 repo_health 任务。
    重要性：重规划意味着"用户确认的方案被悄悄改了"。用哨兵 planner 能确保
    任何一次多余的规划调用都会立刻暴露，这比断言"结果看起来对"更可靠。
    """

    class FailingPlanner:
        """哨兵：run_spec 路径绝不该调用 planner。"""

        async def run(self, input_data, context):
            raise AssertionError("confirmed specs must not be replanned")

    async def run_repo_health(task, spec, context):
        return WorkflowTaskResult(
            task_id=task.id,
            agent_name=task.agent_name,
            task_type=task.task_type,
            status="success",
            summary="Repo baseline collected.",
            evidence_count=1,
            confidence=0.8,
        )

    spec = WorkflowSpec(
        goal="Execute a confirmed plan",
        trigger_message="plan",
        tasks=[
            WorkflowTask(
                id="repo_health",
                agent_name="repo_health_agent",
                task_type="repo_health",
                objective="Collect repo baseline",
                claim=WorkflowClaim(target_type="repository", target_id="repo-1"),
            )
        ],
    )

    report = await WorkflowOrchestrator(
        {"repo_health": run_repo_health},
        planner=FailingPlanner(),  # type: ignore[arg-type]
    ).run_spec(spec.model_dump(mode="json"), {})

    # workflow_id 不变，说明执行的就是同一份 spec 而非重新生成
    assert report["spec"]["workflow_id"] == spec.workflow_id
    assert report["metrics"]["task_count"] == 1
    assert report["task_results"][0]["task_id"] == "repo_health"


@pytest.mark.asyncio
async def test_orchestrator_times_out_stuck_task() -> None:
    """测的是卡死任务的超时兜底。

    场景：任务体 sleep 1 秒，但把 workflow_task_timeout_seconds 设为 0.01，
    并关闭重规划（避免超时又触发一轮）。
    期望：该任务结果为 status=error、output.error_kind=timeout。
    重要性：单个 Agent 卡住若不设超时，整条工作流会一直挂起，
    用户看到的是一直转圈。超时后明确标记 error 而不是"成功但无结论"，
    才能让上层正确降级。
    """

    class StaticPlanner:
        """固定返回一个必然超时的任务。"""

        async def run(self, input_data, context):
            return WorkflowSpec(
                goal="Check stuck worker",
                trigger_message="run",
                tasks=[
                    WorkflowTask(
                        id="slow",
                        agent_name="slow_agent",
                        task_type="slow_task",
                        objective="Never finish quickly",
                        claim=WorkflowClaim(target_type="repository", target_id="repo-1"),
                    )
                ],
            ).model_dump(mode="json")

    async def run_slow(task, spec, context):
        # 1 秒远大于 0.01 秒超时，用来稳定触发超时分支
        await asyncio.sleep(1)
        return WorkflowTaskResult(
            task_id=task.id,
            agent_name=task.agent_name,
            task_type=task.task_type,
            status="success",
            summary="too late",
        )

    report = await WorkflowOrchestrator(
        {"slow_task": run_slow},
        planner=StaticPlanner(),  # type: ignore[arg-type]
    ).run({"message": "run"}, {"workflow_task_timeout_seconds": 0.01, "disable_replan": True})

    result = report["task_results"][0]
    assert result["status"] == "error"
    # 必须区分"超时"与普通失败，前端提示与重试策略都不同
    assert result["output"]["error_kind"] == "timeout"


@pytest.mark.asyncio
async def test_orchestrator_marks_dependents_skipped_when_dependency_fails() -> None:
    """测的是依赖失败的级联跳过。

    场景：root 任务必定失败，child 声明 dependencies=["root"]。
    期望：root 为 error；child 为 skipped 且 error_kind 为 dependency_failed；
    child 的函数体根本不会执行（里面是 AssertionError）。
    重要性：依赖任务在前置失败后继续跑，等于在错误前提上产出结论，
    会把错误放大成"看起来正常的报告"。标记 skipped 而非 error，
    也让用户能区分"自己跑挂了"和"被上游拖累"。
    """

    class StaticPlanner:
        """固定返回 root → child 的依赖链。"""

        async def run(self, input_data, context):
            return WorkflowSpec(
                goal="Check dependency propagation",
                trigger_message="run",
                tasks=[
                    WorkflowTask(
                        id="root",
                        agent_name="root_agent",
                        task_type="root_task",
                        objective="Fail first",
                        claim=WorkflowClaim(target_type="repository", target_id="repo-1"),
                    ),
                    WorkflowTask(
                        id="child",
                        agent_name="child_agent",
                        task_type="child_task",
                        objective="Requires root",
                        # 依赖关系是这条用例的核心
                        dependencies=["root"],
                        claim=WorkflowClaim(target_type="repository", target_id="repo-1"),
                    ),
                ],
            ).model_dump(mode="json")

    async def run_root(task, spec, context):
        return WorkflowTaskResult(
            task_id=task.id,
            agent_name=task.agent_name,
            task_type=task.task_type,
            status="error",
            summary="root failed",
            error="boom",
        )

    async def run_child(task, spec, context):  # pragma: no cover - should not execute
        # 一旦被执行就说明跳过逻辑失效
        raise AssertionError("dependent task should be skipped")

    report = await WorkflowOrchestrator(
        {"root_task": run_root, "child_task": run_child},
        planner=StaticPlanner(),  # type: ignore[arg-type]
    ).run({"message": "run"}, {"disable_replan": True})

    statuses = {result["task_id"]: result for result in report["task_results"]}
    assert statuses["root"]["status"] == "error"
    assert statuses["child"]["status"] == "skipped"
    assert statuses["child"]["output"]["error_kind"] == "dependency_failed"


@pytest.mark.asyncio
async def test_observer_flags_missing_safety_review_when_required() -> None:
    """测的是 Observer 的安全门禁：需要人工确认却没有安全审查时必须报出。

    场景：一份 requires_human_confirmation=True 的计划（要发 GitHub 评论），
    但任务列表为空——即没有任何 safety 审查环节。
    期望：human_review_required 为 True；
    findings[0].finding_type 为 missing_safety_review。
    重要性：对外发言（评论、回复）是不可撤回的副作用，必须经安全审查。
    这条检查是"忘了规划 safety 任务"时的兜底网；若缺失，Agent 可能直接
    把带敏感信息的草稿发到公开 PR 上。
    """
    spec = WorkflowSpec(
        goal="Draft a GitHub comment",
        trigger_message="comment",
        # 关键：声明需要人工确认
        requires_human_confirmation=True,
        tasks=[],
    )

    observation = await ObserverAgent().run({"spec": spec.model_dump(mode="json"), "task_results": []}, {})

    assert observation["human_review_required"] is True
    assert observation["findings"][0]["finding_type"] == "missing_safety_review"


@pytest.mark.asyncio
async def test_chat_workflow_tool_persists_workflow_and_task_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    """测的是端到端：聊天里触发的工作流要落库，且工具集覆盖 PR 与 CI。

    场景：准备一个仓库（1 issue + 1 PR + 1 条失败流水线），
    通过 _build_conversation_tools 拿到工具集，调用 workflow.run_engineering_review。
    期望：route 为 engineering_workflow；答案里含「多 Agent 协作结论」；
    tool_calls 里出现 workflow.pr_review 与 workflow.ci_debug；
    AgentWorkflowRun 恰好 1 条、AgentTaskRun 至少 3 条。
    重要性：运行记录是"为什么给出这个结论"的唯一事后凭据，
    也是前端工作流视图的数据来源；只跑不落库等于过程完全不可追溯。
    """
    # 关闭 LLM，让工作流走确定性分支
    monkeypatch.setattr(settings, "llm_api_key", None)
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    now = datetime.now(timezone.utc)
    with Session(engine) as db:
        repo = Repository(owner="acme", name="demo", full_name="acme/demo")
        db.add(repo)
        db.flush()
        conversation = Conversation(repo_id=repo.id, title="Workflow")
        db.add(conversation)
        # issue 正文刻意写成"步骤 + 期望/实际 + 影响"，满足三域规划的前置条件
        issue = Issue(
            repo_id=repo.id,
            number=7,
            title="Login API returns 500",
            body="Steps: call login. Expected 401, actual 500. Impact: users cannot login.",
            labels=["bug"],
            state="open",
            created_at=now,
        )
        pr = PullRequest(
            repo_id=repo.id,
            number=8,
            title="Fix login error handling",
            body="Fixes login status handling.",
            state="open",
            author="alice",
            created_at=now,
        )
        db.add_all([issue, pr])
        db.flush()
        db.add(PRFile(pr_id=pr.id, filename="backend/app/auth.py", status="modified", additions=40, deletions=8, patch="handle login"))
        db.add(
            WorkflowRun(
                repo_id=repo.id,
                name="backend-tests",
                status="completed",
                conclusion="failure",
                # jobs 与 logs 都要给，CI 排查 Agent 需要据此定位失败步骤
                jobs=[{"name": "pytest", "conclusion": "failure", "steps": [{"name": "Run tests", "conclusion": "failure"}]}],
                logs_text="pytest tests/test_auth.py FAILED AssertionError",
                created_at=now,
            )
        )
        db.commit()

        tools = _build_conversation_tools(repo, conversation, db)
        result = await tools["workflow.run_engineering_review"](
            {"message": "这个 PR 能不能合并？请用多 agent 协作检查 Issue、PR 和 CI。"},
            {"team_members": [{"name": "Alice", "role": "Backend", "strengths": "FastAPI auth", "techStack": "Python"}]},
        )

        assert result["route"] == "engineering_workflow"
        # 答案必须是汇总式的"协作结论"，而不是单个 Agent 的片段
        assert "多 Agent 协作结论" in result["answer"]
        tool_names = [item["tool_name"] for item in result["tool_calls"]]
        assert "workflow.pr_review" in tool_names
        assert "workflow.ci_debug" in tool_names
        # 一次工作流对应一条主记录
        assert db.query(AgentWorkflowRun).count() == 1
        # 至少 3 个任务（repo_health / issue / pr / ci 中的多数），允许以后拆得更细
        assert db.query(AgentTaskRun).count() >= 3
