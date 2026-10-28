"""检索来源策略（source policy）与知识库索引边界的测试。

验证行为：这个项目把数据分成两类，本文件是这条分界线的"宪法"：
- rag（进向量库、可语义检索）：issue / PR / 流水线 / 项目文档 / 知识文件 / 记忆笔记；
- direct（不进向量库、直接查库）：源码、成员、周报、聊天会话、各类记忆。
任何越界都会造成实际问题，因此逐条覆盖：
1) retrieval_mode / should_vectorize 的分类表，含"未知类型默认走 direct"；
2) 知识库默认开启 0.8 分阈值，低于阈值的结果不进入上下文；
3) 源码类文档被拒绝向量化（会引导用户去"工作区实时搜索"）；
4) embedding 条数与文档数不一致时必须整体回滚，不能写半批；
5) 纯向量检索在 Milvus 关闭时报错而不是偷偷退回数据库查询，
   而 hybrid 检索允许降级为全文（这是有意的差异）；
6) 重索引只处理含非结构化历史的来源，且会清理历史脏数据、脱敏 CI 日志；
7) 会话封存（seal）产生的原始聊天/记忆绝不能回灌到项目 RAG 里。
"""

import os
import uuid

# 必须在导入 app.db.models 之前设置：模型/引擎在导入期就会读取 DATABASE_URL，
# 这里指向内存 SQLite，保证测试不碰真实库。
os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.db.models import (
    Base,
    ChatMessage,
    ChatSession,
    Conversation,
    ConversationMemory,
    Document,
    EvidenceItem,
    Issue,
    KnowledgeBaseConfig,
    PRFile,
    PRReviewComment,
    PullRequest,
    Repository,
    WorkflowRun,
)
from app.services.chat_memory import ContextAssembler, EvidenceStore, SessionSealer
from app.services.rag.indexing import CI_LOG_CHUNK_CHARS, index_repository_documents
from app.services.rag.knowledge_base import get_or_create_config, retrieve_knowledge
from app.services.rag.qa import rag_status, reindex_documents
from app.services.rag.retrieval import search_similar_documents
from app.services.rag.source_policy import retrieval_mode, should_vectorize
from app.services.rag.vector_store import DocumentPayload, MilvusUnavailableError, add_documents, reset_milvus_store_cache


def _db_session():
    """建一个纯内存库并建表，返回 Session。"""
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _repo(db):
    """造一个仓库记录。

    full_name 固定为 local/policy：本文件每个用例都用全新内存库，
    不会跨用例撞名，固定值反而让失败信息更好读。
    """
    repo = Repository(owner="local", name="policy", full_name="local/policy")
    db.add(repo)
    db.flush()
    return repo


def _vector() -> list[float]:
    """造一个与全局配置维度一致的伪向量。

    首维为 1、其余全 0：既满足维度契约，又能让"是否被替换/是否被截断"
    这类问题一眼看出来。
    """
    return [1.0, *([0.0] * (settings.embedding_dimensions - 1))]


class _FakeMilvusStore:
    """极简 Milvus 替身：只记录 upsert 入参，不做任何真实 IO。

    这样断言的粒度可以细到"写进向量库的到底是哪几条文档、哪些向量"，
    而这正是本文件要守住的边界（不该向量化的来源绝不能出现在 upserts 里）。
    """

    def __init__(self) -> None:
        self.collection_name = "test_collection"
        self.upserts: list[tuple[list[Document], list[list[float]]]] = []

    def upsert_documents(self, documents, vectors) -> None:
        self.upserts.append((list(documents), list(vectors)))

    def delete_documents(self, **_kwargs) -> None:
        return None


def _install_fake_milvus(monkeypatch) -> _FakeMilvusStore:
    """把 vector_store 内部的 store 工厂替换成假实现，并返回该实例供断言。

    替换的是 _milvus_store 工厂函数而不是具体方法，因此连接、健康检查等
    所有路径都会被一并绕开，测试完全不依赖本机是否有 Milvus。
    """
    store = _FakeMilvusStore()
    monkeypatch.setattr("app.services.rag.vector_store._milvus_store", lambda: store)
    return store


@pytest.mark.parametrize(
    ("source_type", "expected"),
    [
        # 以下六类属于"非结构化文本"，值得做语义检索，走 rag
        ("issue", "rag"),
        ("pull_request", "rag"),
        ("workflow_run", "rag"),
        ("project_doc", "rag"),
        ("knowledge_file", "rag"),
        ("memory_note", "rag"),
        # 以下这些要么本身是结构化数据（成员/周报/清单），要么是源码（应走实时搜索），
        # 要么是私密原始记录（聊天/记忆），一律不进向量库
        ("code_file", "direct"),
        ("project_overview", "direct"),
        ("project_manifest", "direct"),
        ("team_member", "direct"),
        ("weekly_report", "direct"),
        ("chat_session", "direct"),
        ("conversation_memory", "direct"),
        ("thread_memory", "direct"),
        # 未登记的新类型必须默认 direct：宁可漏掉语义检索，也不能把隐私数据向量化
        ("future_unknown_type", "direct"),
    ],
)
def test_source_policy_separates_rag_and_direct_data(source_type, expected) -> None:
    """测的是来源分类表本身，以及两个入口函数的一致性。

    场景：遍历全部已登记来源类型 + 一个未来的未知类型。
    期望：retrieval_mode 返回预期分类；should_vectorize 与"是否等于 rag"严格一致。
    重要性：这是整个 RAG 的准入名单。默认策略必须是"未知 → direct"——
    新增来源类型时若忘记登记，最坏结果是搜不到，而不是把隐私数据灌进向量库。
    """
    assert retrieval_mode(source_type) == expected
    # 两个函数必须同步，否则会出现"判定为 rag 却不向量化"的隐藏 bug
    assert should_vectorize(source_type) is (expected == "rag")


def test_knowledge_base_defaults_enable_high_score_threshold() -> None:
    """测的是知识库配置的默认阈值：默认开启且为 0.8。

    场景：仓库首次访问知识库配置，触发自动创建。
    期望：score_threshold_enabled 为 True，阈值约等于 0.8。
    重要性：默认关闭阈值等于"任何低分结果都会被塞进提示词"，
    是幻觉的主要来源之一；把它做成默认开启属于产品级决策，必须锁住。
    """
    db = _db_session()
    repo = _repo(db)

    config = get_or_create_config(db, repo.id)

    assert config.score_threshold_enabled is True
    assert config.score_threshold == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_default_score_threshold_filters_results_below_point_eight(monkeypatch) -> None:
    """测的是阈值边界：0.81 留下、0.79 丢掉。

    场景：伪造两条检索结果，分数紧贴 0.8 两侧（0.81 / 0.79）。
    期望：只剩 high-confidence；trace 里如实回报阈值已启用且为 0.8。
    重要性：用贴边的数值而非 0.9/0.1 这种极端值，才能真正验证比较用的是
    ">=" 还是 ">"，以及阈值是否被正确加载（而不是被忽略后全放行）。
    """
    db = _db_session()
    repo = _repo(db)
    monkeypatch.setattr(
        "app.services.rag.knowledge_base.search_similar_documents",
        lambda *_args, **_kwargs: [
            {"id": "high-confidence", "score": 0.81},
            {"id": "low-confidence", "score": 0.79},
        ],
    )

    items, trace, _duration_ms = await retrieve_knowledge(
        db,
        repo.id,
        "authentication token",
        retrieval_method="fulltext",
        rerank_enabled=False,
    )

    # 只用一条窄缝区分，任何比较符改动都会让本断言失败
    assert [item["id"] for item in items] == ["high-confidence"]
    assert trace["score_threshold_enabled"] is True
    assert trace["score_threshold"] == pytest.approx(0.8)


def test_add_documents_only_embeds_rag_sources(monkeypatch) -> None:
    """测的是写入侧的准入：只有 rag 来源会被算 embedding 并写进 Milvus。

    场景：一次提交三条不同类型文档（knowledge_file / team_member / chat_session）。
    期望：三条都落库且 meta 里标注了各自的 retrieval_mode；
    但 Milvus 只收到 1 次 upsert，且里面只有 knowledge_file 那一条；
    同时 Document 实体上没有 embedding 属性（向量不存关系库）。
    重要性：如果把成员或聊天记录也送去 embedding，既浪费成本，
    又会让私密内容进入可被语义检索的空间——这是数据合规问题。
    """
    store = _install_fake_milvus(monkeypatch)
    # 假 embedding：按文本条数返回等量伪向量，保持"条数一致"这个前提成立
    monkeypatch.setattr(
        "app.services.rag.vector_store.embed_texts",
        lambda texts, dimensions: [_vector() for _text in texts],
    )
    db = _db_session()
    repo = _repo(db)

    docs = add_documents(
        db,
        repo.id,
        [
            DocumentPayload(source_type="knowledge_file", title="Runbook", content="Rotate the token."),
            DocumentPayload(source_type="team_member", title="Alice", content="Backend engineer"),
            DocumentPayload(source_type="chat_session", title="Session", content="Raw transcript"),
        ],
    )

    by_type = {doc.source_type: doc for doc in docs}
    # 元数据要如实记录来源的检索模式，供前端展示和后续排查
    assert by_type["knowledge_file"].meta["retrieval_mode"] == "rag"
    assert by_type["team_member"].meta["retrieval_mode"] == "direct"
    # 向量只存在于 Milvus，关系库实体上不应有该属性
    assert not hasattr(by_type["knowledge_file"], "embedding")
    # 只 upsert 一次，且只包含 rag 来源
    assert len(store.upserts) == 1
    assert [doc.source_type for doc in store.upserts[0][0]] == ["knowledge_file"]
    assert store.upserts[0][1] == [_vector()]


def test_add_documents_rejects_source_code_documents(monkeypatch) -> None:
    """测的是源码入库的明确拒绝（并给出可操作指引）。

    场景：传入 source_type="code_file" 的文档。
    期望：抛 ValueError，错误信息含「工作区实时搜索」。
    重要性：源码应当在工作区里实时检索（保证与当前分支一致），
    若被切块写进向量库，用户会拿到过期的代码片段。错误信息里给出替代方案，
    是为了让调用方知道"该改用哪个接口"而不是只看到一个失败。
    """
    # 关掉 Milvus 并清缓存，确保失败原因只能是"策略拒绝"，而不是连不上向量库
    monkeypatch.setattr(settings, "milvus_enabled", False)
    reset_milvus_store_cache()
    db = _db_session()
    repo = _repo(db)

    with pytest.raises(ValueError, match="工作区实时搜索"):
        add_documents(
            db,
            repo.id,
            [DocumentPayload(source_type="code_file", title="auth.py", content="def rotate_token(): pass")],
        )


def test_add_documents_rejects_partial_embedding_batches(monkeypatch) -> None:
    """测的是 embedding 条数校验 + 事务回滚。

    场景：提交 1 条文档，但 embed_texts 返回空列表（模拟服务端静默丢批）。
    期望：抛 ValueError（信息含 "embedding count"），且 Document 表里一行都没有。
    重要性：条数不匹配若不拦，会出现"文档已入库但没有向量"的幽灵记录——
    全文搜得到、语义搜不到，且难以定位。这里同时断言了回滚，
    防止只抛异常却已写入半批数据。
    """
    monkeypatch.setattr(settings, "milvus_enabled", False)
    reset_milvus_store_cache()
    monkeypatch.setattr("app.services.rag.vector_store.embed_texts", lambda _texts, _dimensions: [])
    db = _db_session()
    repo = _repo(db)

    with pytest.raises(ValueError, match="embedding count"):
        add_documents(
            db,
            repo.id,
            [DocumentPayload(source_type="knowledge_file", title="Runbook", content="Rotate the token.")],
        )

    # 关键：失败后不能留下半截数据
    assert db.query(Document).count() == 0


def test_general_rag_search_excludes_code_and_direct_records(monkeypatch) -> None:
    """测的是读取侧的准入：默认检索与显式指定来源时的过滤行为。

    场景：库里同时存在 knowledge_file 与 team_member 两种文档，然后分别做：
    默认检索、显式查 code_file（fulltext 与 vector 两种方式）、
    显式查 team_member、以及同时请求 knowledge_file + code_file 的混合检索。
    期望：默认检索只返回 knowledge_file；code_file 两种方式都返回空；
    team_member 返回空；混合检索的结果是 knowledge_file 的子集（code_file 被过滤掉）。
    重要性：写入侧拦住了不代表读取侧安全——历史脏数据、旧版本索引都可能残留
    源码/成员记录，读取侧必须再拦一道，否则"direct"策略形同虚设。
    """
    monkeypatch.setattr(settings, "milvus_enabled", False)
    reset_milvus_store_cache()
    db = _db_session()
    repo = _repo(db)
    add_documents(
        db,
        repo.id,
        [
            DocumentPayload(
                source_type="knowledge_file",
                title="Authentication runbook",
                content="Rotate the access token after authentication failure.",
            ),
            DocumentPayload(
                source_type="team_member",
                title="Token owner",
                content="Maintains authentication token services.",
            ),
        ],
        # 直接给现成向量并跳过向量库同步，让本用例只考察检索侧过滤
        vectors=[_vector(), _vector()],
        sync_vector_store=False,
    )
    # 关掉向量召回（返回空），确保 code_file 的检索只可能走关键词路径
    monkeypatch.setattr("app.services.rag.retrieval.search_milvus_documents", lambda *_args, **_kwargs: [])

    # ① 不指定来源的普通检索
    general = search_similar_documents(
        db,
        repo.id,
        "authentication token",
        retrieval_method="fulltext",
        apply_rerank=False,
    )
    # ② 明确点名要 code_file（hybrid），应被策略拦下
    code = search_similar_documents(
        db,
        repo.id,
        "rotate_access_token",
        source_type="code_file",
        retrieval_method="hybrid",
        apply_rerank=False,
    )
    # ③ 明确点名 team_member（direct 类）
    team = search_similar_documents(
        db,
        repo.id,
        "token",
        source_type="team_member",
        retrieval_method="fulltext",
        apply_rerank=False,
    )
    # ④ 同样的 code_file 但走纯 vector，验证两种检索方式都被拦
    code_with_vector_method = search_similar_documents(
        db,
        repo.id,
        "rotate_access_token",
        source_type="code_file",
        retrieval_method="vector",
        apply_rerank=False,
    )
    # ⑤ 混合来源：合法来源与非法来源同时请求，非法的那部分必须被静默滤掉
    mixed = search_similar_documents(
        db,
        repo.id,
        "rotate_access_token",
        source_types=["knowledge_file", "code_file"],
        retrieval_method="hybrid",
        limit=2,
        apply_rerank=False,
    )

    # 普通检索只能看见 rag 来源
    assert {item["source_type"] for item in general} == {"knowledge_file"}
    # code_file 无论 hybrid 还是 vector 都必须为空
    assert code == []
    assert code_with_vector_method == []
    # 用子集断言（而不是相等）：核心是"结果里绝不能出现 code_file"
    assert {item["source_type"] for item in mixed} <= {"knowledge_file"}
    assert team == []


def test_vector_search_does_not_fall_back_to_database(monkeypatch) -> None:
    """测的是纯向量检索在 Milvus 不可用时必须报错，而不是退回 SQL 暴力扫描。

    场景：milvus_enabled=False，库里却有一条内容完全匹配的文档。
    期望：抛 MilvusUnavailableError（信息含 "disabled"）。
    重要性：这条"不降级"是刻意设计——静默退回数据库全表扫描会在大仓库上
    拖垮服务，且让运维误以为向量库工作正常；显式报错才能暴露配置问题。
    """
    monkeypatch.setattr(settings, "milvus_enabled", False)
    reset_milvus_store_cache()
    db = _db_session()
    repo = _repo(db)
    # 故意放一条"如果降级就一定能搜到"的文档，确保报错不是因为没有数据
    db.add(
        Document(
            repo_id=repo.id,
            source_type="knowledge_file",
            source_id=uuid.uuid4(),
            title="Authentication runbook",
            content="Rotate the access token.",
            meta={"retrieval_mode": "rag"},
        )
    )
    db.commit()

    with pytest.raises(MilvusUnavailableError, match="disabled"):
        search_similar_documents(
            db,
            repo.id,
            "authentication token",
            retrieval_method="vector",
            apply_rerank=False,
        )


def test_hybrid_search_falls_back_to_fulltext_when_milvus_is_disabled(monkeypatch) -> None:
    """测的是 hybrid 检索的降级：向量路不可用时退化为纯全文，而不是整体失败。

    场景：milvus_enabled=False，用 hybrid 方式检索一条内容匹配的文档。
    期望：仍返回该文档（走 BM25 全文路径）。
    重要性：与上一个用例形成对照——纯 vector 必须报错（用户明确只要语义检索，
    偷偷换算法等于骗人），而 hybrid 本身包含全文路，降级后结果仍然可信，
    因此允许继续服务。两者的差异必须被测试固定，否则很容易被"统一处理"改坏。
    """
    monkeypatch.setattr(settings, "milvus_enabled", False)
    reset_milvus_store_cache()
    db = _db_session()
    repo = _repo(db)
    db.add(
        Document(
            repo_id=repo.id,
            source_type="knowledge_file",
            source_id=uuid.uuid4(),
            title="Authentication runbook",
            content="Rotate the access token after authentication failure.",
            meta={"retrieval_mode": "rag"},
        )
    )
    db.commit()

    results = search_similar_documents(
        db,
        repo.id,
        "authentication token",
        retrieval_method="hybrid",
        apply_rerank=False,
    )

    assert [item["title"] for item in results] == ["Authentication runbook"]


@pytest.mark.asyncio
async def test_code_file_retrieval_is_rejected_without_query_embedding(monkeypatch) -> None:
    """测的是"拒绝"发生在算 query 向量之前，且来源参数会被规范化。

    场景：请求 source_types=[" code_file ", "code_file"]（带多余空格 + 重复）。
    embed_knowledge_query 被替换成 pytest.fail，一旦被调用测试立即失败。
    期望：返回空列表、trace.source_types 被规整为 ["code_file"]，
    且全程没有为查询算过 embedding。
    重要性：若先算 embedding 再判断策略，等于白花一次（可能很贵的）模型调用；
    来源参数不规范化则会出现"带空格就绕过了黑名单"的过滤漏洞。
    """
    monkeypatch.setattr(settings, "milvus_enabled", False)
    reset_milvus_store_cache()
    # 哨兵：源码检索绝不应触发查询向量化
    monkeypatch.setattr(
        "app.services.rag.knowledge_base.embed_knowledge_query",
        lambda *_args, **_kwargs: pytest.fail("code retrieval must not embed the query"),
    )
    db = _db_session()
    repo = _repo(db)
    items, trace, _duration_ms = await retrieve_knowledge(
        db,
        repo.id,
        "rotate_access_token",
        # 空格 + 重复都故意留着，用于验证入参规范化
        source_types=[" code_file ", "code_file"],
        rerank_enabled=False,
    )

    assert items == []
    assert trace["source_types"] == ["code_file"]


def test_reindex_uses_the_global_embedding_contract(monkeypatch) -> None:
    """测的是重建索引时用的是「知识库配置里的」embedding 契约，而非全局默认值。

    场景：仓库配置一份 KnowledgeBaseConfig，库里放 1 条 issue；
    vector_store.embed_texts（全局默认 provider 的入口）被替换成 pytest.fail，
    同时记录 embed_knowledge_texts 实际收到的 provider/model/dimensions。
    期望：vector_store.embed_texts 从未被调用；实际调用参数与配置完全一致，
    且只批量处理 1 条文本；Milvus 收到 1 次 upsert。
    重要性：先用全局 provider 预计算再写库，会在仓库自定义了 embedding 模型时
    产出维度/空间不一致的向量；这个用例正是为了把"两套入口只留一套"钉死。
    """
    store = _install_fake_milvus(monkeypatch)
    # 哨兵：不允许走"用全局配置预embed"的老路径
    monkeypatch.setattr(
        "app.services.rag.vector_store.embed_texts",
        lambda *_args, **_kwargs: pytest.fail("repository refresh must not pre-embed with global settings"),
    )
    calls: list[tuple[str, str, int, int]] = []

    def fake_embed(texts, *, provider, model, dimensions):
        """记录每次 embedding 调用的契约参数与批量大小。"""
        calls.append((provider, model, dimensions, len(texts)))
        return [_vector() for _text in texts]

    monkeypatch.setattr("app.services.rag.qa.embed_knowledge_texts", fake_embed)
    db = _db_session()
    repo = _repo(db)
    # 配置里显式写入与全局一致的值，这样断言可以逐个字段比对
    db.add(
        KnowledgeBaseConfig(
            repo_id=repo.id,
            embedding_provider=settings.embedding_provider,
            embedding_model=settings.embedding_model,
            embedding_dimensions=settings.embedding_dimensions,
        )
    )
    db.add(Issue(repo_id=repo.id, number=3, title="Token expiry", body="Refresh once.", state="open"))
    db.commit()

    indexed = reindex_documents(db, repo.id)

    assert indexed == 1
    # 三元组 + 批量条数都必须精确匹配，任何一项不同即视为契约被破坏
    assert calls == [(settings.embedding_provider, settings.embedding_model, settings.embedding_dimensions, 1)]
    assert len(store.upserts) == 1


def test_reindex_rejects_partial_embedding_batches(monkeypatch) -> None:
    """测的是重索引路径同样要校验 embedding 条数。

    场景：库里有一条 memory_note 文档，但 embed_texts 返回空列表。
    期望：抛 ValueError（信息含 "embedding count"）。
    重要性：重索引是"修复数据"的手段，如果它自己悄悄写残缺数据，
    用户会以为索引已修好，问题反而被掩盖。
    """
    monkeypatch.setattr(settings, "milvus_enabled", False)
    reset_milvus_store_cache()
    db = _db_session()
    repo = _repo(db)
    db.add(
        Document(
            repo_id=repo.id,
            source_type="memory_note",
            source_id=uuid.uuid4(),
            title="Decision",
            content="Use short-lived tokens.",
            meta={},
        )
    )
    db.commit()
    # 返回空批次，模拟 embedding 服务静默丢数据
    monkeypatch.setattr("app.services.rag.qa.embed_texts", lambda _texts, _dimensions: [])

    with pytest.raises(ValueError, match="embedding count"):
        reindex_documents(db, repo.id)


def test_reindex_clears_legacy_metadata_and_status_only_counts_rag(monkeypatch) -> None:
    """测的是重索引的"清理"职责：老脏数据被删、遗留 embedding 元数据被抹掉、
    统计口径只算 rag 来源。

    场景：库里预置四类历史遗留：
    - 一条 memory_note（合法 rag 来源，应保留并重建）；
    - 一条 weekly_report，meta 里带着旧 provider/model/dimensions（direct 来源，
      元数据应被清空且不再"假装"有向量）；
    - chat_session 原始聊天、pull_request 带敏感 patch、通过的流水线、
      code_file 源码分块——这四类都是不该留在索引里的历史脏数据。
    期望：重新索引后 indexed==1；weekly 的 retrieval_mode 变回 direct 且
    embedding_* 键消失；rag_status 只统计 1 条（memory_note）；
    四类脏数据全部被删除，尤其 SENSITIVE_PATCH 这类内容绝不能残留。
    重要性：版本升级后旧索引格式不一致是必然的，重索引若只"补"不"清"，
    敏感补丁和私密聊天会长期潜伏在检索范围内；统计口径混乱还会让用户
    看到"知识库有 6 条"但实际只有 1 条可检索。
    """
    _install_fake_milvus(monkeypatch)
    # 两条路径的 embed 都要伪造：add_documents 走 vector_store，reindex 走 qa
    monkeypatch.setattr(
        "app.services.rag.vector_store.embed_texts",
        lambda texts, dimensions: [_vector() for _text in texts],
    )
    monkeypatch.setattr(
        "app.services.rag.qa.embed_texts",
        lambda texts, dimensions: [_vector() for _text in texts],
    )
    db = _db_session()
    repo = _repo(db)
    docs = add_documents(
        db,
        repo.id,
        [
            DocumentPayload(source_type="memory_note", title="Decision", content="Use short-lived tokens."),
            DocumentPayload(source_type="weekly_report", title="Weekly", content="Two pull requests merged."),
        ],
    )
    # 给 direct 来源的 weekly_report 人为塞入遗留 embedding 元数据，
    # 模拟"旧版本曾错误地把它向量化过"的历史状态
    weekly = next(doc for doc in docs if doc.source_type == "weekly_report")
    weekly.meta = {
        **(weekly.meta or {}),
        "embedding_provider": "legacy",
        "embedding_model": "legacy",
        "embedding_dimensions": 2,
    }
    db.add_all(
        [
            # 私密原始聊天记录：属于 direct，绝不能进索引
            Document(
                repo_id=repo.id,
                source_type="chat_session",
                source_id=uuid.uuid4(),
                title="Legacy raw chat",
                content="A private raw conversation transcript.",
                meta={},
            ),
            # 带敏感补丁的 PR：用显眼的占位串便于断言"已彻底删除"
            Document(
                repo_id=repo.id,
                source_type="pull_request",
                source_id=uuid.uuid4(),
                title="Legacy PR",
                content="SENSITIVE_PATCH_FROM_OLD_INDEX",
                meta={},
            ),
            Document(
                repo_id=repo.id,
                source_type="workflow_run",
                source_id=uuid.uuid4(),
                title="Legacy passing run",
                content="all tests passed",
                meta={"conclusion": "success"},
            ),
            Document(
                repo_id=repo.id,
                source_type="code_file",
                source_id=uuid.uuid4(),
                title="Legacy source chunk",
                content="def stale_handler(): pass",
                meta={"retrieval_mode": "lexical", "path": "service.py"},
            ),
        ]
    )
    db.commit()

    indexed = reindex_documents(db, repo.id)
    # reindex 后需要 refresh 才能看到外部写入的 meta 变更
    db.refresh(weekly)
    status = rag_status(db, repo.id)

    # 只有 memory_note 这一条真正被重建索引
    assert indexed == 1
    assert not hasattr(weekly, "embedding")
    # 遗留的 embedding 元数据必须被抹掉，且检索模式被纠正为 direct
    assert weekly.meta["retrieval_mode"] == "direct"
    assert "embedding_provider" not in weekly.meta
    # 统计口径只数 rag 来源，用户看到的数字才是"真正可检索"的数量
    assert status["document_count"] == 1
    assert status["source_types"] == {"memory_note": 1}
    # 库里只剩 2 条（memory_note + weekly_report），其余脏数据都被清了
    assert db.query(Document).filter(Document.repo_id == repo.id).count() == 2
    # 逐类确认删除：私密聊天
    assert db.query(Document).filter(Document.title == "Legacy raw chat").count() == 0
    # 逐类确认删除：敏感补丁（按内容查，防止改名绕过）
    assert db.query(Document).filter(Document.content == "SENSITIVE_PATCH_FROM_OLD_INDEX").count() == 0
    # 逐类确认删除：通过的流水线（没有失败信息，索引它没有价值）
    assert db.query(Document).filter(Document.title == "Legacy passing run").count() == 0
    # 逐类确认删除：源码分块
    assert db.query(Document).filter(Document.title == "Legacy source chunk").count() == 0


def test_repository_index_only_embeds_unstructured_historical_fields(monkeypatch) -> None:
    """测的是仓库索引的字段取舍：只把「非结构化历史文本」写进可检索内容。

    场景：一条 issue（带 labels/author/assignees 等结构化字段）、
    一条 PR（含文件 patch 与 review 评论）、两条流水线（一成功一失败）。
    期望：indexed == 3（issue 1 + PR 1 + 失败的流水线 1，成功的被跳过）；
    issue 正文里不出现 "state:" / "labels:" 这类结构化前缀，但 state 存进 meta；
    PR 的 patch（SENSITIVE_PATCH...）不进正文，review 评论反而要进；
    PR 的 files 只保留文件名列表；只有失败的那条流水线生成文档。
    重要性：把结构化字段写进正文会污染 BM25 词频并让提示词充满噪声；
    而 patch 体积大且含敏感代码，一旦被向量化，几乎无法从索引里彻底清除；
    成功流水线的日志对排障没有价值，索引它只会稀释检索精度。
    """
    _install_fake_milvus(monkeypatch)
    monkeypatch.setattr(
        "app.services.rag.vector_store.embed_texts",
        lambda texts, dimensions: [_vector() for _text in texts],
    )
    db = _db_session()
    repo = _repo(db)
    # issue 同时带了结构化字段，用来验证它们不会混进正文
    db.add(
        Issue(
            repo_id=repo.id,
            number=7,
            title="Authentication occasionally fails",
            body="Users see an expired-session message.",
            state="open",
            labels=["backend"],
            author="alice",
            assignees=["bob"],
        )
    )
    # PR 里塞了两类"不该被索引"和"该被索引"的内容做对比：
    # patch 含敏感代码（不该进正文），review 评论是人工经验（应当进正文）
    db.add(
        PullRequest(
            repo_id=repo.id,
            number=9,
            title="Retry token refresh",
            body="Retry once after refresh.",
            state="open",
            author="carol",
            base_branch="main",
            head_branch="fix/retry",
            files=[
                PRFile(
                    filename="auth.py",
                    status="modified",
                    additions=4,
                    deletions=1,
                    patch="SENSITIVE_PATCH_SHOULD_NOT_BE_EMBEDDED",
                )
            ],
            review_comments=[
                PRReviewComment(
                    body="Keep the retry idempotent.",
                    path="auth.py",
                    line=42,
                    author="reviewer",
                )
            ],
        )
    )
    # 一成功一失败：只有失败的那条应当被索引
    db.add_all(
        [
            WorkflowRun(
                repo_id=repo.id,
                name="passing build",
                status="completed",
                conclusion="success",
                logs_text="all tests passed",
            ),
            WorkflowRun(
                repo_id=repo.id,
                name="failing build",
                status="completed",
                conclusion="failure",
                logs_text="test_refresh_retry failed with 401",
            ),
        ]
    )
    db.flush()

    indexed = index_repository_documents(db, repo.id)
    docs = db.query(Document).filter(Document.repo_id == repo.id).all()
    by_type = {doc.source_type: doc for doc in docs}

    # 3 = issue + PR + 失败的流水线（成功的那条不计）
    assert indexed == 3
    # 结构化字段不进正文
    assert "state:" not in by_type["issue"].content
    assert "labels:" not in by_type["issue"].content
    # 但它们要以 meta 的形式保留，供过滤使用
    assert by_type["issue"].meta["state"] == "open"
    # patch 不进正文
    assert "SENSITIVE_PATCH_SHOULD_NOT_BE_EMBEDDED" not in by_type["pull_request"].content
    # review 经验要进正文
    assert "Keep the retry idempotent." in by_type["pull_request"].content
    # 文件只留文件名，不留 patch
    assert by_type["pull_request"].meta["files"] == ["auth.py"]
    # 只有失败流水线生成了文档，且日志正文被保留
    workflow_docs = [doc for doc in docs if doc.source_type == "workflow_run"]
    assert len(workflow_docs) == 1
    assert "test_refresh_retry failed with 401" in workflow_docs[0].content


def test_repository_sync_respects_global_embedding_contract(monkeypatch) -> None:
    """测的是同步触发的索引同样遵守仓库的 embedding 契约。

    场景：仓库配置 KnowledgeBaseConfig，放 1 条 issue；
    vector_store.embed_texts 换成 pytest.fail，记录 embed_knowledge_texts 的入参。
    期望：indexed == 1；入参三元组与配置完全一致；Document 实体无 embedding 属性；
    meta 里记录本次使用的 provider；Milvus 收到 1 次 upsert。
    重要性：与重索引用例互为补充——两条入口（手动重建 / 同步自动索引）
    若有一条漏改，都会产生与其他数据不可比的向量，检索结果就会莫名错乱。
    """
    store = _install_fake_milvus(monkeypatch)
    # 哨兵：不允许用全局 provider 预先算向量
    monkeypatch.setattr(
        "app.services.rag.vector_store.embed_texts",
        lambda *_args, **_kwargs: pytest.fail("sync must not use the global embedding provider"),
    )
    calls: list[tuple[str, str, int]] = []

    def fake_embed(texts, *, provider, model, dimensions):
        """记录同步路径实际使用的 embedding 契约三元组。"""
        calls.append((provider, model, dimensions))
        return [_vector() for _text in texts]

    # 注意替换的是 indexing 模块里的引用，说明同步索引走的是 indexing 这条通道
    monkeypatch.setattr("app.services.rag.indexing.embed_knowledge_texts", fake_embed)
    db = _db_session()
    repo = _repo(db)
    db.add(
        KnowledgeBaseConfig(
            repo_id=repo.id,
            embedding_provider=settings.embedding_provider,
            embedding_model=settings.embedding_model,
            embedding_dimensions=settings.embedding_dimensions,
        )
    )
    db.add(Issue(repo_id=repo.id, number=5, title="Token expiry", body="Refresh once.", state="open"))
    db.flush()

    indexed = index_repository_documents(db, repo.id)
    # .one() 同时验证"恰好一条 issue 文档"，防止重复索引
    doc = db.query(Document).filter(Document.repo_id == repo.id, Document.source_type == "issue").one()

    assert indexed == 1
    # 契约三元组必须与仓库配置一致
    assert calls == [(settings.embedding_provider, settings.embedding_model, settings.embedding_dimensions)]
    # 向量只写 Milvus，不落到关系库实体上
    assert not hasattr(doc, "embedding")
    # 但要记住用的是哪个 provider，便于日后判断是否需要整体重建
    assert doc.meta["embedding_provider"] == settings.embedding_provider
    assert len(store.upserts) == 1


def test_failed_ci_logs_are_redacted_and_bounded_before_embedding(monkeypatch) -> None:
    """测的是 CI 日志在入库前必须「脱敏 + 限长」。

    场景：一条失败流水线，日志开头带真实格式的 GitHub token
    （Authorization: Bearer ... 与 TOKEN=... 两种写法各一次），
    后面跟 1200 行重复内容把长度撑爆。
    期望：被切成多块（len > 1）；任何一块里都不含原始 secret；
    出现 "[REDACTED" 标记；每块长度不超过 CI_LOG_CHUNK_CHARS 加标题开销；
    每块的 meta.secrets_redacted 均为 True。
    重要性：CI 日志是最容易夹带真实凭据的数据源，一旦被向量化，
    secret 会永久留在向量库并可被语义检索出来——这是最高危的泄露路径。
    限长则防止单条超长日志撑爆 embedding 上下文长度。
    """
    _install_fake_milvus(monkeypatch)
    monkeypatch.setattr(
        "app.services.rag.vector_store.embed_texts",
        lambda texts, dimensions: [_vector() for _text in texts],
    )
    db = _db_session()
    repo = _repo(db)
    # 用 ghp_ + 32 个 a 构造形似真实 PAT 的假 secret（不是真凭据）
    secret = "ghp_" + "a" * 32
    db.add(
        WorkflowRun(
            repo_id=repo.id,
            name="failing build",
            status="completed",
            conclusion="failure",
            # 两种常见泄露写法都覆盖，后面再堆 1200 行内容触发分块
            logs_text=f"Authorization: Bearer {secret}\nTOKEN={secret}\n" + ("failure context\n" * 1200),
        )
    )
    db.flush()

    index_repository_documents(db, repo.id)
    # 按标题排序，让分块顺序稳定，断言不依赖数据库返回顺序
    docs = (
        db.query(Document)
        .filter(Document.repo_id == repo.id, Document.source_type == "workflow_run")
        .order_by(Document.title.asc())
        .all()
    )

    # 长日志必然被切成多块
    assert len(docs) > 1
    # 任何一块都不能残留原文 secret
    assert all(secret not in doc.content for doc in docs)
    # 必须留下脱敏痕迹（便于审计"这里原本有敏感信息"）
    assert "[REDACTED" in "\n".join(doc.content for doc in docs)
    # 每块长度受控（+1 是换行符开销，标题也会计入正文）
    assert all(len(doc.content) <= CI_LOG_CHUNK_CHARS + len("failing build") + 1 for doc in docs)
    # 每块都要打上脱敏标记，避免下游误当成"干净数据"
    assert all(doc.meta["secrets_redacted"] is True for doc in docs)


def test_evidence_store_redacts_failed_ci_logs_before_chat_context() -> None:
    """测的是证据库（EvidenceStore）这条链路同样要脱敏。

    场景：失败流水线日志里写 MY_API_KEY=<secret>，然后重建证据库。
    期望：唯一一条 workflow_run 证据里不含 secret，且被替换为
    "MY_API_KEY=[REDACTED]"；meta.secrets_redacted 为 True。
    重要性：证据库是直接拼进聊天上下文给大模型看的，比向量库更"近"，
    这里漏了脱敏等于把密钥直接发给外部模型服务。
    """
    db = _db_session()
    repo = _repo(db)
    secret = "ghp_" + "a" * 32
    db.add(
        WorkflowRun(
            repo_id=repo.id,
            name="failing build",
            status="completed",
            conclusion="failure",
            logs_text=f"MY_API_KEY={secret}\nERROR authentication failed",
        )
    )
    db.flush()

    EvidenceStore(db).rebuild_repo(repo)
    # 恰好一条 workflow_run 证据，且带 error 字段供后续断言
    evidence = (
        db.query(EvidenceItem)
        .filter(EvidenceItem.repo_id == repo.id, EvidenceItem.source_type == "workflow_run")
        .one()
    )

    assert secret not in evidence.content
    # 保留键名、替换值，用户仍能看出"这里配了个 API key"
    assert "MY_API_KEY=[REDACTED]" in evidence.content
    assert evidence.meta["secrets_redacted"] is True


def test_context_assembler_prefetches_rag_evidence_before_chat_agent(monkeypatch) -> None:
    """测的是上下文装配会「先把用户问题喂给检索」，再把证据注入系统提示。

    场景：把 EvidenceStore.search 替换成固定返回一条 project_doc 证据的假实现，
    然后装配一次没有 conversation 的上下文。
    期望：检索恰好用用户原问题查了一次（queries 精确等于问题原文）；
    assembled.evidence 里有该证据；system_context 里出现 "[Related Evidence]" 段
    以及证据正文。
    重要性：如果只把历史消息塞给模型而不预取证据，模型就完全靠内部知识回答，
    等于白做了 RAG；"用原问题查一次"也防止有人把改写后的查询带进去导致召回漂移。
    """
    db = _db_session()
    repo = _repo(db)
    # 记录检索被调用时的 query，用于验证"用的是原问题"
    queries: list[str] = []

    def fake_search(self, repo_id, conversation_id, query, limit=5):
        """签名必须与 EvidenceStore.search 一致（含 self），否则替换无效。"""
        queries.append(query)
        return [
            {
                "title": "Authentication retry policy",
                "source_type": "project_doc",
                "source_id": "doc-1",
                "snippet": "Refresh retries must use an idempotency key.",
                "score": 0.91,
                "metadata": {},
                "retrieval": {"sources": ["vector", "keyword"]},
            }
        ]

    monkeypatch.setattr(EvidenceStore, "search", fake_search)
    # conversation 传 None：模拟新会话第一轮，没有历史可参考
    assembled = ContextAssembler(db).assemble(
        repo,
        None,
        "How should refresh retries work?",
    )

    assert queries == ["How should refresh retries work?"]
    assert assembled.evidence[0]["source_id"] == "doc-1"
    # 证据段落标题固定，前端/提示词模板都依赖这个标记
    assert "[Related Evidence]" in assembled.system_context
    assert "Refresh retries must use an idempotency key." in assembled.system_context


def test_context_assembler_reports_exact_recent_history_coverage() -> None:
    """测的是历史覆盖度的精确统计：3 条消息完整进提示词，一条不少。

    场景：一段 3 轮消息（用户 hi → 助手回复 → 用户追问"我刚刚说了什么？"），
    装配上下文后读取 compression_stats["history_coverage"]。
    期望：complete_and_exact 为 True；prompt_messages、exact_suffix_messages
    均为 3；omitted 为 0；没有压缩边界（has_compaction_boundary 为 False）。
    重要性：用户问"我刚刚说了什么"时，只要历史被截断就会答错，而这类错误
    看起来像模型幻觉、实际是上下文装配问题。这套指标是区分两者的唯一依据。
    """
    db = _db_session()
    repo = _repo(db)
    conversation = Conversation(repo_id=repo.id, title="Recall recent messages")
    db.add(conversation)
    db.flush()
    # 三条消息只求短小清晰，重点在于"数量恰好能被完整覆盖"
    db.add_all(
        [
            ChatMessage(repo_id=repo.id, conversation_id=conversation.id, role="user", content="hi"),
            ChatMessage(
                repo_id=repo.id,
                conversation_id=conversation.id,
                role="assistant",
                content="你好！有什么可以帮你？",
            ),
            ChatMessage(
                repo_id=repo.id,
                conversation_id=conversation.id,
                role="user",
                content="我刚刚说了什么？",
            ),
        ]
    )
    db.flush()

    assembled = ContextAssembler(db).assemble(repo, conversation, "我刚刚说了什么？")
    coverage = assembled.compression_stats["history_coverage"]

    # 完整且逐字未改
    assert coverage["complete_and_exact"] is True
    assert coverage["prompt_messages"] == 3
    assert coverage["exact_suffix_messages"] == 3
    # 一条都没被丢
    assert coverage["omitted_messages"] == 0
    # 没有发生压缩，自然也就没有压缩分界点
    assert coverage["has_compaction_boundary"] is False


@pytest.mark.asyncio
async def test_session_sealing_keeps_raw_memory_out_of_project_rag() -> None:
    """测的是会话封存（seal）的写入边界：原始聊天只进会话表，绝不进 RAG 索引。

    场景：一段正常问答，用假 LLM 生成摘要与结构化记忆，然后执行 seal。
    期望：ChatSession 恰好 1 条、ConversationMemory 恰好 1 条（结构化记忆有落库），
    但 source_type 属于 chat_session / conversation_memory / thread_memory 的
    Document 数量为 0——也就是没有任何原始/记忆类数据被写成可检索文档。
    重要性：这是"个人对话"与"项目知识"的隔离墙。一旦聊天原文进入项目向量库，
    任何有该项目访问权限的同事都可能通过提问检索到别人的私密对话。
    """

    class FakeLLM:
        """假 LLM：同时实现对话与结构化抽取两个入口。

        chat_json 的返回必须把所有字段都填全（哪怕为空列表），
        因为封存逻辑会逐字段读取，缺键会抛 KeyError 而不是走到断言。
        """

        async def chat_text(self, system_prompt, user_prompt):
            return "The team chose short-lived access tokens."

        async def chat_json(self, system_prompt, payload):
            return {
                "summary": "Use short-lived access tokens.",
                "facts": ["Tokens expire after fifteen minutes."],
                "decisions": ["Refresh once before asking the user to sign in."],
                "open_questions": [],
                "tasks": [],
                "user_preferences": [],
                "repo_context": [],
                "citations": [],
            }

    db = _db_session()
    repo = _repo(db)
    conversation = Conversation(repo_id=repo.id, title="Authentication")
    db.add(conversation)
    db.flush()
    # 一问一答：封存通常以"一轮完整对话"为单位
    user_message = ChatMessage(
        repo_id=repo.id,
        conversation_id=conversation.id,
        role="user",
        content="How should token refresh work?",
    )
    assistant_message = ChatMessage(
        repo_id=repo.id,
        conversation_id=conversation.id,
        role="assistant",
        content="Refresh once, then require sign-in.",
    )
    db.add_all([user_message, assistant_message])
    db.flush()

    await SessionSealer(db, llm=FakeLLM()).seal(
        repo=repo,
        conversation=conversation,
        user_message=user_message,
        assistant_message=assistant_message,
        route="chat",
        tool_calls=[],
    )
    db.commit()

    # 会话与结构化记忆要正常落库
    assert db.query(ChatSession).filter(ChatSession.conversation_id == conversation.id).count() == 1
    assert db.query(ConversationMemory).filter(ConversationMemory.conversation_id == conversation.id).count() == 1
    # 核心断言：三类私密来源都不应产出任何 Document（=没进 RAG）
    assert (
        db.query(Document)
        .filter(Document.source_type.in_(["chat_session", "conversation_memory", "thread_memory"]))
        .count()
        == 0
    )
