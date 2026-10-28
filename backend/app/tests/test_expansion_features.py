"""「扩展能力」的一组集成测试（动作草稿 / 多仓库周报 / 代码图谱 / PR 与 CI 上下文构建 / 快照清理）。

这些能力共同支撑"研发助手能真正动手、能跨仓库看全局、能给 PR 和 CI 提供精准上下文"：
1. 动作草稿：AI 生成待确认操作 → 用户确认后落执行结果与审计 id；
2. 多仓库周报：按工作区聚合 issue/PR/CI 指标并生成 markdown；
3. 代码图谱：从源码抽符号与调用关系，并支持按符号检索；
4. PR 快照：PR 分支的代码索引只作用于该 PR，不污染主分支代码库；
5. 上下文安全：diff patch 只用于本地代码定位、绝不能进语义检索；CI 日志里的 token
   进语义查询前必须打码，但本地代码检索仍需原文（否则定位不到报错）；
6. 快照清理：过期目录与对应图谱数据要一起删干净。
"""

import os
from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")

# 以下 import 必须在设置 DATABASE_URL 之后，db 模块导入时就会读取该变量
from app.api.routes.action_drafts import confirm_action_draft, create_action_draft
from app.api.routes.ci import _augment_ci_context_with_snapshot, _build_ci_analysis_context
from app.api.routes.code_graph import search_code_graph
from app.api.routes.pull_requests import _augment_pr_context_with_snapshot, _build_pr_analysis_context
from app.api.routes.workspaces import create_workspace, multi_repo_report
from app.core.config import settings
from app.db.models import Base, CodeRelation, CodeSymbol, Issue, PRFile, PRReviewComment, PullRequest, Repository, User, WorkflowRun
from app.schemas.action_drafts import ActionDraftCreateRequest
from app.schemas.workspaces import MultiRepoReportRequest, WorkspaceCreateRequest
from app.services.code_analysis import _sync_code_graph
from app.services.code_search import search_repository_code
from app.services.rag.vector_store import reset_milvus_store_cache
from app.services.worktree_manager import _sync_pr_snapshot_analysis, cleanup_pr_snapshots


def _db_session():
    """每个用例一套独立内存库，避免跨用例的数据串味。"""
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _user(db):
    """建一个 owner 角色的用户：多数写接口都要求有操作人，用于审计。"""
    user = User(name="owner", role="owner")
    db.add(user)
    db.flush()
    return user


def _repo(db, name: str = "demo") -> Repository:
    """建一个 managed 模式的仓库：clone_url 必填，后续快照/索引路径会用到。"""
    repo = Repository(owner="local", name=name, full_name=f"local/{name}", checkout_mode="managed", clone_url=f"https://github.com/local/{name}.git")
    db.add(repo)
    db.flush()
    return repo


@pytest.mark.asyncio
async def test_send_report_action_draft_confirm_records_execution() -> None:
    """场景：AI 生成一份"发送周报"草稿，用户确认执行。

    期望：草稿状态变为 executed，execution_result.status 为 recorded，并返回审计 id。
    为什么重要：AI 不能自己把消息发出去，必须"先草稿、人确认、再执行"；
    审计 id 是出问题后追责与回放的唯一凭据。
    """
    db = _db_session()
    user = _user(db)
    repo = _repo(db)

    draft = await create_action_draft(
        ActionDraftCreateRequest(repo_id=repo.id, draft_type="send_report", target_type="repository", title="Weekly report", content="Report body"),
        db=db,
        user=user,
    )

    result = await confirm_action_draft(draft.id, db=db, user=user)

    assert result["draft"].status == "executed"
    # execution_result 记录的是"这次执行发生了什么"，status=recorded 表示已留痕
    assert result["draft"].execution_result["status"] == "recorded"
    assert result["audit_id"]


@pytest.mark.asyncio
async def test_multi_repo_report_aggregates_workspace_repositories(monkeypatch) -> None:
    """场景：工作区里有两个仓库，api 有 1 个 open issue，web 有 1 个 open PR 和 1 次失败的 CI，
    生成今天的多仓库周报。

    期望：metrics 中 repos=2、open_issues=1、open_prs=1、failed_ci=1，
    report_markdown 含"多仓库研发周报"标题。
    为什么重要：周报的每个数字都要能对应到具体仓库数据；
    统计口径错了（比如把已关闭的 issue 也算进去），团队就会被误导去做无用功。
    """
    # 关掉 milvus：周报不需要向量检索，避免测试依赖外部服务
    monkeypatch.setattr(settings, "milvus_enabled", False)
    reset_milvus_store_cache()
    db = _db_session()
    user = _user(db)
    repo_a = _repo(db, "api")
    repo_b = _repo(db, "web")
    db.add(Issue(repo_id=repo_a.id, number=1, title="Fix auth", state="open", created_at=datetime.now(timezone.utc)))
    db.add(PullRequest(repo_id=repo_b.id, number=2, title="UI update", state="open", created_at=datetime.now(timezone.utc)))
    db.add(WorkflowRun(repo_id=repo_b.id, github_run_id=3, name="test", status="completed", conclusion="failure", created_at=datetime.now(timezone.utc)))
    db.flush()

    workspace = await create_workspace(WorkspaceCreateRequest(name="team", repo_ids=[repo_a.id, repo_b.id]), db=db, user=user)
    result = await multi_repo_report(
        MultiRepoReportRequest(
            workspace_id=workspace.id,
            start_date=datetime.now(timezone.utc).date(),
            end_date=datetime.now(timezone.utc).date(),
        ),
        db=db,
        user=user,
    )

    assert result["metrics"]["repos"] == 2
    assert result["metrics"]["open_issues"] == 1
    assert result["metrics"]["open_prs"] == 1
    assert result["metrics"]["failed_ci"] == 1
    assert "多仓库研发周报" in result["report_markdown"]


@pytest.mark.asyncio
async def test_code_graph_indexes_symbols_and_relations(tmp_path) -> None:
    """场景：一个 Python 文件里，handler() 调用了从 lib 导入的 helper()，另有一个空类 Worker。

    期望：索引出 2 个符号（handler、Worker）；按 "handler" 检索时第一个符号是 handler；
    关系里存在 type=calls 的边。
    为什么重要：调用关系是"改动影响面分析"的基础——
    没有 calls 边，就答不出"改 handler 会影响谁"。
    """
    db = _db_session()
    user = _user(db)
    repo = _repo(db)
    # 用临时目录当源码根，避免依赖真实 clone
    project = tmp_path / "repo"
    project.mkdir()
    (project / "service.py").write_text(
        "from lib import helper\n\n"
        "def handler():\n"
        "    return helper()\n\n"
        "class Worker:\n"
        "    pass\n",
        encoding="utf-8",
    )

    count = _sync_code_graph(db, repo, project, branch="main", commit_sha="abc123")
    db.commit()
    result = await search_code_graph(repo.id, q="handler", limit=20, db=db, _user=user)

    # 2 个符号：函数 handler + 类 Worker
    assert count == 2
    assert result["symbols"][0].name == "handler"
    # calls 边来自函数体里对 helper() 的调用
    assert any(item["relation_type"] == "calls" for item in result["relations"])


def test_pr_snapshot_live_search_is_pr_scoped(tmp_path) -> None:
    """场景：在 PR 快照检出目录里搜代码，并传入 PR 相关的 metadata。

    期望：结果 metadata 原样带回 branch / commit_sha / pr_id / pr_number / checkout_kind=pr_snapshot，
    source_type 仍是 workspace_file。
    为什么重要：同一仓库同时有主干和多个 PR 的代码副本，
    丢掉这些标记就无法判断"这段代码来自哪个 PR 的哪个提交"，引用会张冠李戴。
    """
    db = _db_session()
    repo = _repo(db)
    project = tmp_path / "repo"
    project.mkdir()
    (project / "feature.py").write_text("def feature_flag():\n    return True\n", encoding="utf-8")

    snapshot_results = search_repository_code(
        repo,
        "feature_flag",
        # 显式传 checkout_path：直接指向 PR 快照目录，而不是仓库主 checkout
        checkout_path=project,
        metadata={
            "branch": "feature/pr-snapshot",
            "commit_sha": "abc123",
            "pr_id": "pr-123",
            "pr_number": 12,
            "checkout_kind": "pr_snapshot",
        },
    )

    assert snapshot_results
    metadata = snapshot_results[0]["metadata"]
    # 逐项校验来源标记被完整透传，前端靠这些字段做"在 PR diff 里打开"的跳转
    assert metadata["checkout_kind"] == "pr_snapshot"
    assert metadata["branch"] == "feature/pr-snapshot"
    assert metadata["commit_sha"] == "abc123"
    assert metadata["pr_id"] == "pr-123"
    assert metadata["pr_number"] == 12
    assert snapshot_results[0]["source_type"] == "workspace_file"


def test_pr_snapshot_analysis_writes_code_graph_without_documents(tmp_path, monkeypatch) -> None:
    """场景：对一个 PR 的快照目录做分析（milvus 关闭，也就是没有文档向量库）。

    期望：仍然写入 2 个 CodeSymbol，且都带 pr_id 与 pr_number；
    整个过程不需要文档向量库（代码图谱只走数据库）。
    为什么重要：PR 审查常常要"只看图谱不建向量"，快速给出改动影响面；
    若强依赖向量库，PR 审查就会在索引未就绪时整体不可用。
    """
    monkeypatch.setattr(settings, "milvus_enabled", False)
    reset_milvus_store_cache()
    db = _db_session()
    repo = _repo(db)
    pr = PullRequest(repo_id=repo.id, number=12, title="Feature flag", state="open", head_branch="feature/pr-snapshot")
    db.add(pr)
    db.flush()
    project = tmp_path / "repo"
    project.mkdir()
    (project / "feature.py").write_text("def handler():\n    return helper()\n\ndef helper():\n    return True\n", encoding="utf-8")

    indexed = _sync_pr_snapshot_analysis(db, repo, pr, project, "feature/pr-snapshot", "abc123")
    db.commit()

    assert indexed == 2
    # 关键：符号被标记为该 PR 所有，主干索引查询时不会被误算进来
    assert db.query(CodeSymbol).filter(CodeSymbol.pr_id == pr.id, CodeSymbol.pr_number == 12).count() == 2


def test_pr_analysis_context_prefers_snapshot_docs_and_graph(tmp_path, monkeypatch) -> None:
    """场景：PR 改动了 service.py，快照已就绪（indexed_symbols=2），要构建"分析这个 PR"的上下文。

    期望：counts.snapshot_code > 0（确实用上了快照代码）；
    code_references 第一条带 pr_id，证明引用的代码来自该 PR 快照；
    code_graph_impact 里第一个符号是 handler，给出改动影响面。
    为什么重要：PR 分析若读的是主干代码，就会评论"这段代码不存在"或者给出已经过时的建议——
    只看快照才能基于改动后的真实代码评审。
    """
    monkeypatch.setattr(settings, "milvus_enabled", False)
    reset_milvus_store_cache()
    db = _db_session()
    repo = _repo(db)
    pr = PullRequest(repo_id=repo.id, number=4, title="Use handler", state="open", head_branch="feature/handler")
    db.add(pr)
    db.flush()
    db.add(PRFile(pr_id=pr.id, filename="service.py", status="modified", additions=4, deletions=1, patch="handler()"))
    db.flush()
    project = tmp_path / "repo"
    project.mkdir()
    (project / "service.py").write_text("def handler():\n    return helper()\n\ndef helper():\n    return True\n", encoding="utf-8")
    _sync_pr_snapshot_analysis(db, repo, pr, project, "feature/handler", "abc123")
    db.commit()
    db.refresh(pr)

    context = {"code_references": []}
    counts = {"code": 0}
    # snapshot 结构模仿快照服务真实返回：带索引符号数与快照路径
    snapshot = {"status": "ready", "snapshot_name": "PR 分支快照", "pr_id": str(pr.id), "pr_number": pr.number, "branch": pr.head_branch, "commit_sha": "abc123", "indexed_symbols": 2, "snapshot_path": str(project)}
    _augment_pr_context_with_snapshot(db, pr, "handler service.py", context, counts, snapshot)

    # 用快照而不是慢速二次检索，所以单独计数
    assert counts["snapshot_code"] > 0
    assert context["code_references"][0]["metadata"]["pr_id"] == str(pr.id)
    assert context["code_graph_impact"]["symbols"][0]["name"] == "handler"


def test_pr_analysis_keeps_patch_out_of_semantic_queries(monkeypatch) -> None:
    """场景：PR 文件 patch 里放了一个唯一标记串，另有一条评审评论。

    期望：标记串出现在 code_query 和本地代码检索里（用于定位改动的代码），
    但绝不出现在语义检索（向量库）的 query 中；语义检索应使用评论等"可共享文本"。
    为什么重要：diff patch 可能包含内部代码/密钥，进入向量库就等于长期留存并可能被别的仓库检索到；
    而本地代码检索是进程内的，可以放心带上 patch 来精确定位。
    """
    monkeypatch.setattr(settings, "milvus_enabled", False)
    reset_milvus_store_cache()
    db = _db_session()
    repo = _repo(db)
    pr = PullRequest(
        repo_id=repo.id,
        number=8,
        title="Retry token refresh",
        body="Retry once after a refresh failure.",
        state="open",
    )
    db.add(pr)
    db.flush()
    db.add_all(
        [
            PRFile(
                pr_id=pr.id,
                filename="auth.py",
                status="modified",
                additions=2,
                deletions=1,
                # patch 里放唯一标记串，用来区分"本地代码检索"和"语义检索"两条链路的入参
                patch="SENSITIVE_PATCH_ONLY_FOR_CODE_LOOKUP",
            ),
            PRReviewComment(
                pr_id=pr.id,
                body="Keep the retry idempotent.",
                path="auth.py",
                line=42,
                author="reviewer",
            ),
        ]
    )
    db.commit()
    db.refresh(pr)
    semantic_calls: list[str] = []
    code_calls: list[str] = []

    def fake_search(_db, _repo_id, query, source_type=None, _limit=5, **kwargs):
        # 只记录 query，不真正检索；用空结果让流程继续
        semantic_calls.append(query)
        return []

    def fake_code_search(_repo, query, **_kwargs):
        # 同样只记录 query，用于对比两条链路拿到的文本差异
        code_calls.append(query)
        return []

    # 在路由模块上打桩（服务内部按模块属性名调用），避免真的连向量库/扫盘
    monkeypatch.setattr("app.api.routes.pull_requests.search_similar_documents", fake_search)
    monkeypatch.setattr("app.api.routes.pull_requests.search_repository_code", fake_code_search)

    code_query, _context, _counts = _build_pr_analysis_context(db, pr)

    assert "SENSITIVE_PATCH_ONLY_FOR_CODE_LOOKUP" in code_query
    # 本地代码检索允许带 patch（用于精确定位改动位置）
    assert any("SENSITIVE_PATCH_ONLY_FOR_CODE_LOOKUP" in query for query in code_calls)
    assert semantic_calls
    # 语义检索绝不能带上 patch 原文
    assert all("SENSITIVE_PATCH_ONLY_FOR_CODE_LOOKUP" not in query for query in semantic_calls)
    # 语义检索改用评论内容，既能召回相关文档又不泄露 diff
    assert any("Keep the retry idempotent." in query for query in semantic_calls)


def test_ci_context_uses_related_pr_snapshot_docs_and_graph(tmp_path, monkeypatch) -> None:
    """场景：CI 失败日志里提到 service.py 的 test_target，仓库有一个关联 PR 快照已建好索引。

    期望：counts.snapshot_code > 0；引用元数据 checkout_kind=pr_snapshot；
    图谱影响面第一个符号是 test_target。
    为什么重要：CI 失败时最需要的是"这次改动动了什么"，
    用 PR 快照 + 图谱能直接指向失败符号，而不是让用户在主干代码里瞎找。
    """
    monkeypatch.setattr(settings, "milvus_enabled", False)
    reset_milvus_store_cache()
    db = _db_session()
    repo = _repo(db)
    pr = PullRequest(repo_id=repo.id, number=7, title="Fix service", state="open", head_branch="feature/ci")
    db.add(pr)
    db.flush()
    project = tmp_path / "repo"
    project.mkdir()
    (project / "service.py").write_text("def test_target():\n    return build()\n\ndef build():\n    return True\n", encoding="utf-8")
    _sync_pr_snapshot_analysis(db, repo, pr, project, "feature/ci", "abc123")
    run = WorkflowRun(repo_id=repo.id, github_run_id=99, name="tests", status="completed", conclusion="failure", logs_text="ERROR service.py test_target failed")
    db.add(run)
    db.commit()

    # related_files_from_logs 由日志解析得到，是本次要重点解释的文件
    context = {"code_references": [], "related_files_from_logs": ["service.py"]}
    counts = {"code": 0}
    snapshot = {"status": "ready", "snapshot_name": "PR 分支快照", "pr_id": str(pr.id), "pr_number": pr.number, "branch": pr.head_branch, "commit_sha": "abc123", "indexed_symbols": 2, "snapshot_path": str(project)}
    _augment_ci_context_with_snapshot(db, run, "service.py test_target failed", context, counts, snapshot)

    assert counts["snapshot_code"] > 0
    # 来源标记必须是 pr_snapshot，便于前端把引用定位到 PR 版本而非主干
    assert context["code_references"][0]["metadata"]["checkout_kind"] == "pr_snapshot"
    assert context["code_graph_impact"]["symbols"][0]["name"] == "test_target"


def test_ci_analysis_redacts_semantic_queries_but_keeps_local_code_context(monkeypatch) -> None:
    """场景：CI 日志里混入了一个形如 GitHub token 的字符串（ghp_ + 32 位），
    同时还有真实的报错行 auth.py test_refresh failed。

    期望：本地代码检索的 query 里保留 token 原文（否则报错行上下文会被破坏），
    但所有进语义检索的 query 里 token 已被替换成 [REDACTED]。
    为什么重要：CI 日志经常意外打印密钥；把它送进向量库等于永久留存并可能被检索出来，
    而本地检索是临时的，可以保留原文以保证定位精度。
    """
    monkeypatch.setattr(settings, "milvus_enabled", False)
    reset_milvus_store_cache()
    db = _db_session()
    repo = _repo(db)
    # 现场拼出一个"看起来像真 token"的字符串，避免在仓库里硬编码真实凭证
    secret = "ghp_" + "a" * 32
    run = WorkflowRun(
        repo_id=repo.id,
        github_run_id=101,
        name="tests",
        status="completed",
        conclusion="failure",
        logs_text=f"TOKEN={secret}\nERROR auth.py test_refresh failed",
    )
    db.add(run)
    db.commit()
    semantic_calls: list[str] = []
    code_calls: list[str] = []

    def fake_search(_db, _repo_id, query, source_type=None, _limit=5, **kwargs):
        semantic_calls.append(query)
        return []

    def fake_code_search(_repo, query, **_kwargs):
        code_calls.append(query)
        return []

    monkeypatch.setattr("app.api.routes.ci.search_similar_documents", fake_search)
    monkeypatch.setattr("app.api.routes.ci.search_repository_code", fake_code_search)

    code_query, _context, _counts = _build_ci_analysis_context(db, run)

    assert secret in code_query
    # 本地代码检索保留原文，才能定位到真实报错行
    assert code_calls and secret in code_calls[0]
    assert semantic_calls
    # 语义检索三条断言：不含明文、且确实做了打码标记
    assert all(secret not in query for query in semantic_calls)
    assert all("[REDACTED]" in query for query in semantic_calls)


def test_cleanup_pr_snapshots_removes_old_dirs_and_code_graph(tmp_path, monkeypatch) -> None:
    """场景：一个 30 天前的 PR 快照目录（带 .devflow-pr-snapshot.json 元数据），
    库里有属于该 PR 的符号与调用关系；按 max_age_hours=1 清理。

    期望：removed=1、目录被删、该仓库的 CodeSymbol 与 CodeRelation 都被清空。
    为什么重要：PR 快照是完整代码副本，长期不清理会吃满磁盘；
    只删目录不删图谱数据则会留下指向"已不存在的代码"的脏节点，让影响面分析给出错误结果。
    """
    monkeypatch.setattr(settings, "milvus_enabled", False)
    reset_milvus_store_cache()
    checkout_root = tmp_path / "checkouts"
    monkeypatch.setattr(settings, "repo_checkout_dir", str(checkout_root))
    db = _db_session()
    repo = _repo(db)
    snapshot_dir = checkout_root / "worktrees" / str(repo.id) / "pr-1-local__demo"
    snapshot_dir.mkdir(parents=True)
    # 元数据文件里写 30 天前的 updated_at，清理逻辑靠它判断是否过期
    (snapshot_dir / ".devflow-pr-snapshot.json").write_text(
        '{"updated_at": "'
        + (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        + '", "pr_id": "00000000-0000-0000-0000-000000000123"}',
        encoding="utf-8",
    )
    pr_id = UUID("00000000-0000-0000-0000-000000000123")
    symbol = CodeSymbol(
        repo_id=repo.id,
        path="service.py",
        name="handler",
        kind="function",
        language="py",
        start_line=1,
        pr_id=pr_id,
    )
    db.add(symbol)
    db.flush()
    db.add(CodeRelation(repo_id=repo.id, source_symbol_id=symbol.id, source_name="handler", target_name="helper", relation_type="calls", path="service.py", pr_id=pr_id))
    db.commit()

    result = cleanup_pr_snapshots(db, repo, max_age_hours=1)

    assert result["removed"] == 1
    assert not snapshot_dir.exists()
    # 目录删了，图谱数据也必须跟着删，避免指向已删除代码的脏节点
    assert db.query(CodeSymbol).filter(CodeSymbol.repo_id == repo.id).count() == 0
    assert db.query(CodeRelation).filter(CodeRelation.repo_id == repo.id).count() == 0
