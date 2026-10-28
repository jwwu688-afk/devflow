"""RAG / Agent 评测体系（Eval）的测试。

评测是这个项目"能不能放心改 Prompt 和检索策略"的基础设施，本文件覆盖：
- 基础评测（run_basic_eval）能产出真实指标与报告；
- ragas 兼容层：新版本 ragas 只暴露 metrics.collections 接口，需要做适配；
- 长答案切块做 faithfulness 打分时不能丢内容；
- 评测集来源：无文件时退回结构化数据（Issue），有固定评测集时能按文件名解析到具体 Document；
- 硬规则（必须包含 / 禁止出现 / 正则格式）的判定；
- 评测用的项目上下文（团队成员、记忆笔记）字段形状要与前端一致；
- 只补跑失败指标（rescore）时不重复跑 Agent；
- 两次评测对比：基线不一致时不可比，指标下降要判为回归并卡住发布门禁；
- 失败自动生成 Issue 草稿（复现环境 / 验收标准）；
- 真实 ChatAgent 端到端评测，以及评测会话用完必须回滚；
- 异步接口：先入队，后台任务跑完后指标落库可查。
"""

import re
import json
from pathlib import Path

import pytest
from fastapi import BackgroundTasks
from langchain_core.messages import AIMessage
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.db.models import Base, Conversation, Document, EvalRun, Repository
from app.api.routes.evaluations import _eval_comparison, _issue_markdown
from app.schemas.evals import EvalRunRequest
from app.services.agents.chat_agent import ChatAgent
from app.services.evals.ragas_runner import (
    CORE_RAGAS_METRICS,
    RagEvalCase,
    _default_cases,
    _hard_rules,
    _evaluation_project_context,
    _ensure_ragas_compatibility,
    _split_faithfulness_response,
    rescore_rag_result,
    run_rag_quality_eval,
)
from app.services.evals.suites import list_eval_suites, load_eval_suite
from app.services.evals.runner import run_basic_eval
from app.services.rag.vector_store import reset_milvus_store_cache


@pytest.mark.asyncio
async def test_basic_eval_runner_returns_real_metrics() -> None:
    """场景：直接跑一次内置的基础评测（不注入任何假数据）。

    期望：Issue 分类准确率与工具调用准确率都大于 0，RAG 命中率@3 和
    CI pytest 失败检出率都是 1.0，报告 markdown 含标题与指标名。
    为什么重要：这是"评测本身是否真的在跑"的冒烟测试；
    如果指标恒为 0 或报告为空，后面所有质量门禁都是自欺欺人。
    """
    result = await run_basic_eval()
    assert result["issue_classification_accuracy"] > 0
    assert result["tool_call_accuracy"] > 0
    assert result["rag_hit_rate_at_3"] == 1.0
    assert result["ci_pytest_failure_detection"] == 1.0
    assert "Eval Report" in result["report_markdown"]
    assert "CI pytest failure detection" in result["report_markdown"]


def test_ragas_collections_api_imports_with_current_langchain() -> None:
    """场景：先调用兼容层 _ensure_ragas_compatibility()，再按新版路径导入四个核心指标类。

    期望：四个类都能成功导入。
    为什么重要：ragas 0.2/0.3/0.4 的模块路径反复变动，
    项目必须能在当前锁定的 langchain 版本下成功导入，否则评测功能在启动阶段就崩。
    """
    _ensure_ragas_compatibility()
    from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

    assert all([AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness])


def test_faithfulness_chunking_keeps_all_answer_content() -> None:
    """场景：一段很长（中间一句重复 90 次）的答案，按 max_chars=120 切块送去做忠实度打分。

    期望：切成多于 2 块、每块长度在 (0, 120] 内，且把块拼回去后
    （忽略空白差异）与原文完全一致。
    为什么重要：faithfulness 要逐句核对答案是否有出处，
    长答案必须分块送模型；切块时只要丢一个字，打分就会误判成"编造内容"。
    """
    response = "第一条事实。\n" + ("第二条很长的事实内容" * 90) + "！\n最后一条事实。"

    chunks = _split_faithfulness_response(response, max_chars=120)

    assert len(chunks) > 2
    assert all(0 < len(chunk) <= 120 for chunk in chunks)
    # 去掉所有空白后再比较，避免换行位置差异影响判断，但绝不放过丢字
    assert re.sub(r"\s+", "", "".join(chunks)) == re.sub(r"\s+", "", response)


def test_default_ragas_cases_fall_back_to_structured_repository_data() -> None:
    """场景：仓库里没有任何评测集文件，只有一条 Issue。

    期望：自动用这条 Issue 生成 1 个评测用例，id 为 "issue-7"，
    source_type 为 issue，期望命中的就是这条 Issue 的 id。
    为什么重要：新接入的仓库通常没有人工评测集；
    能自动从已有数据造用例，才能"接入即可评测"，否则评测功能永远用不上。
    """
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    repo = Repository(owner="local", name="fallback", full_name="local/fallback")
    db.add(repo)
    db.flush()
    # 局部导入：Issue 只在这个用例里用到，放在这里可以避免污染模块顶部的导入列表
    from app.db.models import Issue

    issue = Issue(repo_id=repo.id, number=7, title="Refresh token fails", body="Retry once with an idempotency key", state="open", labels=["bug"])
    db.add(issue)
    db.flush()

    cases = _default_cases(db, str(repo.id))

    # 一个 Issue 对应一个用例，id 里带 Issue 编号便于排查
    assert len(cases) == 1
    assert cases[0].id == "issue-7"
    assert cases[0].source_type == "issue"
    assert cases[0].expected_source_ids == [str(issue.id)]


def test_fixed_eval_suite_resolves_repository_sources(tmp_path: Path) -> None:
    """场景：仓库里有一份知识文件 01_repository_facts.md，磁盘上有一份"固定基线"评测集
    （用例通过 source_file 引用该文件，并约束必须含 main、答案需完全匹配 ^main$）。

    期望：评测集列表显示 ready=True、recommended=True 且带 sha256；
    加载后用例的 expected_source_ids 同时包含 Document.id 与原始 source_id，
    must_include / answer_regex 原样保留。
    为什么重要：固定基线让"这次改动有没有让效果变差"可比；
    source_file 到 Document 的解析一旦失败，用例找不到证据源，命中率会恒为 0。
    """
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    repo = Repository(owner="DemoOrg", name="DevFlow-AI", full_name="DemoOrg/DevFlow-AI")
    db.add(repo)
    db.flush()
    source_id = __import__("uuid").uuid4()
    document = Document(
        repo_id=repo.id,
        source_type="knowledge_file",
        source_id=source_id,
        title="Facts",
        content="默认分支是 main。",
        # filename 是与评测集 source_file 字段对应的关键，用于反查文档
        meta={"filename": "01_repository_facts.md"},
    )
    db.add(document)
    db.flush()
    suite_dir = tmp_path / "evals"
    suite_dir.mkdir()
    # 手写一份最小评测集 JSON：schema_version=2 是当前格式版本
    (suite_dir / "baseline.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "name": "fixed-baseline",
                "label": "固定基线",
                "repository_names": ["DevFlow-AI"],
                "recommended": True,
                "cases": [
                    {
                        "id": "default-branch",
                        "question": "默认分支是什么？",
                        "reference": "main",
                        "source_file": "01_repository_facts.md",
                        "must_include": ["main"],
                        "answer_regex": "^main$",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    suites = list_eval_suites(db, repo.id, suite_dir=suite_dir)
    loaded = load_eval_suite(db, repo.id, "fixed-baseline", suite_dir=suite_dir)

    assert suites[0]["ready"] is True
    assert suites[0]["recommended"] is True
    # sha256 是评测集的"版本指纹"，用于判断两次评测是否可比
    assert suites[0]["sha256"]
    # 同时给出 Document.id（库内主键）和原始 source_id（外部引用），两种引用方式都要能命中
    assert loaded.cases[0].expected_source_ids == [str(document.id), str(source_id)]
    assert loaded.cases[0].must_include == ["main"]
    assert loaded.cases[0].answer_regex == "^main$"


def test_hard_rules_enforce_required_forbidden_and_output_format() -> None:
    """场景：用例要求答案必须提"当前 Git checkout"、禁止提"GitHub 仓库"、
    且整段答案只能形如"当前 Git checkout。"；而实际答案恰好是"GitHub 仓库"。

    期望：硬规则判为不通过（passed=False），缺失短语、违规短语、格式不匹配三项分别被记录。
    为什么重要：硬规则是零成本、零幻觉的确定性检查，
    对"数据来源必须说实话"这类合规要求，比让模型自己打分可靠得多。
    """
    case = RagEvalCase(
        id="authority",
        question="权威来源？",
        reference="当前 Git checkout。",
        expected_source_ids=["source-1"],
        must_include=["当前 Git checkout"],
        must_not_include=["GitHub 仓库"],
        answer_regex=r"^当前 Git checkout[。.]?$",
    )
    result = {"answer": "GitHub 仓库", "tool_calls": [], "citations": []}
    # 第三个参数是检索到的来源，第四是证据列表；这里只需保证结构正确
    hard = _hard_rules(case, result, [{"source_id": "source-1"}], ["evidence"])

    assert hard["passed"] is False
    assert hard["missing_phrases"] == ["当前 Git checkout"]
    assert hard["forbidden_phrases"] == ["GitHub 仓库"]
    assert hard["answer_pattern_matched"] is False


def test_eval_project_context_matches_frontend_team_and_memory_shape() -> None:
    """场景：库里有团队成员文档（meta 用 snake_case 的 tech_stack）和记忆笔记文档。

    期望：_evaluation_project_context 输出的 team_members 用 camelCase 的 techStack，
    并带上 name/role/strengths；project_memory 只含 title + content。
    为什么重要：评测时要把"项目上下文"喂给 Agent，形状必须与前端/线上接口一致，
    否则评测环境与真实环境的输入不同，评测分数就失去参考价值。
    """
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    repo = Repository(owner="local", name="context", full_name="local/context")
    db.add(repo)
    db.flush()
    db.add_all(
        [
            Document(
                repo_id=repo.id,
                source_type="team_member",
                source_id=__import__("uuid").uuid4(),
                title="团队成员：星哥",
                content="name: 星哥",
                meta={
                    "name": "星哥",
                    "role": "前端开发工程师",
                    "strengths": "擅长 Electron",
                    "tech_stack": "JavaScript、Electron、TypeScript",
                },
            ),
            Document(
                repo_id=repo.id,
                source_type="memory_note",
                source_id=__import__("uuid").uuid4(),
                title="桌面化上下文",
                content="Issue #1 是桌面化需求。",
                meta={},
            ),
        ]
    )
    db.flush()

    context = _evaluation_project_context(db, repo.id)

    assert context["team_members"] == [
        {
            "name": "星哥",
            "role": "前端开发工程师",
            "strengths": "擅长 Electron",
            "techStack": "JavaScript、Electron、TypeScript",
        }
    ]
    assert context["project_memory"] == [
        {"title": "桌面化上下文", "content": "Issue #1 是桌面化需求。"}
    ]


class EvalAnswerModel:
    """假模型：返回一段固定答案，用于端到端评测时不真的调用大模型。"""

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        return AIMessage(content="Retries use an idempotency key and only retry once.")


class RecordingScorer:
    """假评分器：固定返回一组高分，并记录每次调用参数。

    用它替代真实 ragas 打分：既能避免 LLM 成本与波动，
    又能断言"传给评分器的输入（答案、检索上下文）是不是评测真正拿到的东西"。
    """

    version = "test-ragas-0.4"
    judge_model = "recording-judge"

    def __init__(self) -> None:
        self.calls = []

    async def score(self, **kwargs):
        # 记录完整入参，供用例断言 response / retrieved_contexts
        self.calls.append(kwargs)
        return (
            {
                "context_precision": 0.95,
                "context_recall": 0.96,
                "faithfulness": 0.97,
                "answer_relevancy": 0.94,
                "agent_goal_accuracy": 1.0,
            },
            {},
        )


class RetryScorer:
    """假评分器：只关心"这次要求它算哪些指标"，用来验证重算时只补缺失项。"""

    version = "test-ragas-0.4"
    judge_model = "retry-judge"

    def __init__(self) -> None:
        self.metric_names = None

    async def score(self, **kwargs):
        # 抓下 metric_names，断言调用方只请求了失败的那个指标
        self.metric_names = kwargs["metric_names"]
        return ({"faithfulness": 0.99}, {})


@pytest.mark.asyncio
async def test_rescore_reuses_saved_agent_evidence_and_only_retries_missing_metrics() -> None:
    """场景：上一次评测里 faithfulness 因为 TimeoutError 失败，其它四项指标都有值。

    期望：重算时只请求 {"faithfulness"} 一个指标；
    结果标记为 run_mode=rescore、rescore_of 指向原评测 id、agent_reused=True；
    该用例的 ragas_errors 清空、faithfulness 补上 0.99，整体变为完整且通过。
    为什么重要：Agent 跑一次很贵，重算只应补失败指标；
    如果整轮重跑，既浪费成本，又因为重新生成答案导致两次结果不可比。
    """
    scorer = RetryScorer()
    source_result = {
        "target": {"repo_id": "repo-1"},
        # 评测集指纹 + pipeline 配置，决定这次结果能否与其它评测比较
        "evaluation_set": {"name": "fixed", "sha256": "abc"},
        "pipeline": {"top_k": 5},
        "retrieval": {"recall_at_k": 1.0, "precision_at_k": 1.0, "mrr": 1.0, "ndcg": 1.0},
        "performance": {"cases": 1, "avg_latency_ms": 100},
        "cases": [
            {
                "id": "case-1",
                "question": "问题",
                "reference": "答案",
                "response": "答案",
                "retrieved_contexts": ["答案"],
                "retrieved_sources": [{"source_id": "source-1"}],
                "tool_calls": [],
                "tool_observations": [],
                "expected_tool_calls": [],
                "hard_rules": {"passed": True, "source_hit": True, "contexts_present": True, "answer_present": True},
                "ragas": {
                    "context_precision": 1.0,
                    "context_recall": 1.0,
                    "answer_relevancy": 1.0,
                    "agent_goal_accuracy": 1.0,
                },
                "ragas_errors": {"faithfulness": "TimeoutError: "},
            }
        ],
    }

    rescored = await rescore_rag_result(source_result, eval_id="eval-1", scorer=scorer)

    assert scorer.metric_names == {"faithfulness"}
    assert rescored["run_mode"] == "rescore"
    assert rescored["rescore_of"] == "eval-1"
    assert rescored["performance"]["agent_reused"] is True
    assert rescored["cases"][0]["ragas_errors"] == {}
    assert rescored["cases"][0]["ragas"]["faithfulness"] == 0.99
    assert rescored["evaluation_complete"] is True
    assert rescored["status"] == "passed"


def test_eval_comparison_requires_same_baseline_and_reports_regressions() -> None:
    """场景：基线 faithfulness=0.95 且用例全通过；候选只有 0.80 且用例硬规则不通过。

    期望：可比（comparable=True）、faithfulness 差值 -0.15、回归用例数 1、
    发布门禁不通过；随后把候选的评测集 sha256 改掉，则变为不可比，
    原因里出现"评测集 Hash 不一致"。
    为什么重要：拿了不同版本的评测集去比分数是无效对比（题目都变了）；
    而门禁判断必须基于"有回归就拦"，否则质量只会一路下滑。
    """
    shared = {
        "evaluation_set": {"name": "fixed", "sha256": "abc"},
        "ragas": {"version": "0.4.3", "judge_model": "judge"},
        "pipeline": {"top_k": 5},
        "evaluation_complete": True,
    }
    base = EvalRun(
        name="base",
        result_json={
            **shared,
            "passed": True,
            "quality_status": "passed",
            "summary_metrics": {"faithfulness": 0.95},
            "cases": [{"id": "case-1", "hard_rules": {"passed": True}, "ragas_errors": {}}],
        },
    )
    candidate = EvalRun(
        name="candidate",
        result_json={
            **shared,
            "passed": False,
            "quality_status": "failed",
            "summary_metrics": {"faithfulness": 0.80},
            "cases": [
                {"id": "case-1", "hard_rules": {"passed": False}, "ragas_errors": {}},
            ],
        },
    )

    comparison = _eval_comparison(base, candidate)

    assert comparison["comparable"] is True
    # delta 统一按 candidate - base 计算，负值即退化
    assert comparison["metric_deltas"] == [
        {"name": "faithfulness", "base": 0.95, "candidate": 0.8, "delta": -0.15}
    ]
    assert comparison["regressed_cases"] == 1
    assert comparison["release_gate_passed"] is False

    # 篡改评测集指纹后，两次结果的"题目"不再相同，必须拒绝比较
    candidate.result_json = {
        **candidate.result_json,
        "evaluation_set": {"name": "fixed", "sha256": "different"},
    }
    incompatible = _eval_comparison(base, candidate)
    assert incompatible["comparable"] is False
    assert "评测集 Hash 不一致" in incompatible["incompatibility_reasons"]


def test_issue_draft_markdown_contains_reproduction_and_actions() -> None:
    """场景：一次评测因评分器超时（TimeoutError）整体失败，需要自动生成 Issue 草稿。

    期望：markdown 里有"复现环境"和"验收标准"两节，含 TimeoutError 与诊断建议"仅重试异常评分"；
    而没有拿到分数的工具指标（tool_call_accuracy）不应出现在草稿里。
    为什么重要：失败草稿要能直接贴到 Issue 里让工程师复现；
    把"没有分数"的指标写进去只会制造噪音，反而掩盖真正的失败原因。
    """
    eval_run = EvalRun(
        name="failed-run",
        result_json={
            "status": "error",
            "quality_status": "failed",
            "evaluation_set": {"name": "fixed", "sha256": "abc"},
            "ragas": {
                "version": "0.4.3",
                "judge_model": "judge",
                "metrics": {
                    "faithfulness": {"score": None, "threshold": 0.9},
                    "tool_call_accuracy": {"score": None, "threshold": 0.9},
                },
            },
            "cases": [
                {
                    "id": "case-1",
                    "question": "问题",
                    "reference": "标准答案",
                    "response": "错误答案",
                    "ragas_errors": {"faithfulness": "TimeoutError"},
                    "diagnoses": [
                        {"category": "judge", "title": "Judge 异常", "action": "仅重试异常评分"}
                    ],
                }
            ],
        },
    )

    markdown = _issue_markdown(eval_run)

    # 两节固定结构：环境信息（便于复现）+ 验收标准（便于关闭 Issue）
    assert "## 复现环境" in markdown
    assert "TimeoutError" in markdown
    assert "仅重试异常评分" in markdown
    # 没有分数的指标不写进草稿，避免把噪音当成失败原因
    assert "tool_call_accuracy" not in markdown
    assert "## 验收标准" in markdown


@pytest.mark.asyncio
async def test_rag_quality_eval_runs_real_chatagent_and_rolls_back_eval_conversation(monkeypatch) -> None:
    """场景：用真实 ChatAgent（模型换成假模型）+ 假评分器跑一个 RAG 质量评测用例。

    期望：结果 kind=chatagent_ragas、目标为 ChatAgent、状态 passed、硬门禁通过、
    recall@3 = 1.0；所有核心 ragas 指标都有分数；
    评分器收到的 answer 与检索上下文确实来自这次 Agent 执行；
    进度事件严格按 preparing → chatagent → ragas → case_complete → finalizing 顺序推送；
    最后库里不残留 status=evaluation 的会话。
    为什么重要：评测必须跑"真实链路"才有意义；
    而评测产生的会话若不回滚，会污染用户的历史列表与统计（用户会看到一堆莫名其妙的会话）。
    """
    monkeypatch.setattr(settings, "milvus_enabled", False)
    reset_milvus_store_cache()
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    repo = Repository(owner="local", name="ragas", full_name="local/ragas")
    db.add(repo)
    db.flush()
    source_id = __import__("uuid").uuid4()
    document = Document(
        repo_id=repo.id,
        source_type="project_doc",
        source_id=source_id,
        title="Retry policy",
        content="Authentication retries use an idempotency key and retry only once.",
        meta={},
    )
    db.add(document)
    db.commit()
    scorer = RecordingScorer()
    progress_events = []

    result = await run_rag_quality_eval(
        db,
        str(repo.id),
        cases=[
            RagEvalCase(
                id="retry-policy",
                question="How should authentication retries work?",
                reference="Use an idempotency key and retry only once.",
                expected_source_ids=[str(document.id), str(source_id)],
                source_type="project_doc",
            )
        ],
        scorer=scorer,
        # agent_factory 用 lambda 延迟构造，模拟生产里"每次评测新建一个 Agent"的写法
        agent_factory=lambda: ChatAgent(model=EvalAnswerModel()),  # type: ignore[arg-type]
        progress_callback=lambda event: progress_events.append(event),
    )

    assert result["kind"] == "chatagent_ragas"
    assert result["target"]["name"] == "ChatAgent"
    assert result["status"] == "passed"
    assert result["hard_gate"]["passed"] is True
    assert result["retrieval"]["recall_at_k"] == 1.0
    assert result["ragas"]["version"] == "test-ragas-0.4"
    assert all(result["ragas"]["metrics"][name]["score"] is not None for name in CORE_RAGAS_METRICS)
    assert scorer.calls[0]["response"].startswith("Retries use an idempotency key")
    # 检索上下文必须非空，否则 faithfulness 这类指标根本无从判断
    assert scorer.calls[0]["retrieved_contexts"]
    # 进度事件顺序是前端进度条的依据，顺序错乱会让 UI 停在错误阶段
    assert [event["stage"] for event in progress_events] == [
        "preparing",
        "chatagent",
        "ragas",
        "case_complete",
        "finalizing",
    ]
    # 评测会话必须被清理干净，不能以 evaluation 状态残留在库里
    assert db.query(Conversation).filter(Conversation.status == "evaluation").count() == 0


@pytest.mark.asyncio
async def test_ragas_api_persists_summary_metrics(monkeypatch) -> None:
    """场景：调用异步评测接口，后台任务被替换成假的评测实现（会推一条进度事件后返回失败结果）。

    期望：接口立即返回 status=queued、历史列表里也是 queued；
    手动执行 BackgroundTasks 后，历史与详情的状态变为 failed，
    且 summary_metrics 被完整落库（faithfulness/context_recall）。
    为什么重要：长评测必须异步化，否则请求会超时；
    而"后台跑完后指标能查到"是用户唯一能看到结果的途径——落库断了，评测就白跑了。
    """
    # 局部导入：避免模块级导入顺序影响其它用例对路由模块的 monkeypatch
    from app.api.routes import evaluations as evaluation_routes

    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    repo = Repository(owner="local", name="api", full_name="local/api")
    db.add(repo)
    db.commit()

    async def fake_run(db_arg, repo_id, **kwargs):
        # 断言后台任务拿到的是同一个 session 和正确的仓库 id（用后即弃的 session 会导致写入丢失）
        assert db_arg is db
        assert repo_id == str(repo.id)
        assert kwargs["run_judge"] is True
        await kwargs["progress_callback"](
            {
                "stage": "ragas",
                "message": "scoring",
                "completed_cases": 0,
                "total_cases": 1,
            }
        )
        return {
            "kind": "chatagent_ragas",
            "status": "failed",
            "passed": False,
            "summary_metrics": {"faithfulness": 0.62, "context_recall": 0.91},
            "ragas": {"available": True, "metrics": {}},
            "cases": [],
        }

    monkeypatch.setattr(evaluation_routes, "run_rag_quality_eval", fake_run)
    # 后台任务会另开 session，这里替换成同一个，保证断言能看到写入结果
    monkeypatch.setattr(evaluation_routes, "SessionLocal", lambda: db)
    background_tasks = BackgroundTasks()
    response = await evaluation_routes.run_rag_eval(
        EvalRunRequest(name="api-ragas", repo_id=repo.id, run_judge=True),
        background_tasks=background_tasks,
        db=db,
        _user=object(),
    )
    queued_history = await evaluation_routes.list_evals(db=db, _user=object())

    assert response.status == "queued"
    assert queued_history[0]["status"] == "queued"

    # 手动触发后台任务，相当于 worker 把评测跑完
    await background_tasks()
    history = await evaluation_routes.list_evals(db=db, _user=object())
    detail = await evaluation_routes.get_eval(response.eval_id, db=db, _user=object())

    assert db.query(EvalRun).count() == 1
    assert history[0]["status"] == "failed"
    assert history[0]["metrics"] == {"faithfulness": 0.62, "context_recall": 0.91}
    assert detail["status"] == "failed"
