"""RAG 检索流水线的端到端测试。

验证行为：一条查询从进来到被判定「能不能答」，中间要经过
关键词(BM25)召回 → 元数据过滤 → 双路融合 → 重排 → 父块扩展 →
分数阈值 → 上下文压缩 → 答案闸门 这几个环节，本文件逐个环节锁住行为：
- BM25 要能抓出稀有错误名，并把命中证据（trace）暴露出来；
- 元数据过滤必须在打分前生效，不能"先算后筛"；
- 融合公式与权重必须是项目既定值，不允许随手调参；
- 父块扩展只能读同一来源、同一版本的分块，不能串到旧分支；
- 答案闸门覆盖 4 种决策（无证据/需澄清/冲突/可作答）；
- Qwen 重排拿到的必须是**压缩前**的原文，且失败时要降级到启发式并记录原因。
"""

import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import Base, Document, Repository
from app.services.rag.answer_gate import evaluate_answer_gate
from app.services.rag.knowledge_base import retrieve_knowledge
from app.services.rag.retrieval import (
    _merge_candidates,
    expand_parent_context,
    search_similar_documents,
)


def _db_session():
    """建一个内存 SQLite 会话，各用例数据互不影响。"""
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _repo(db):
    """造一个仓库记录。

    full_name 加随机 uuid 后缀：避免多个用例共用同一内存库约定或撞唯一索引。
    """
    repo = Repository(owner="local", name="retrieval", full_name=f"local/retrieval-{uuid.uuid4()}")
    db.add(repo)
    db.flush()
    return repo


def _document(db, repo, *, title, content, source_id=None, metadata=None):
    """造一条知识库文档（模拟已切分好的 chunk）。

    source_id 可复用：同一文件的多个分块共享一个 source_id，父块扩展正是靠它聚合；
    metadata 里通常放 chunk_id / parent_id / branch 等结构信息。
    """
    item = Document(
        repo_id=repo.id,
        source_type="knowledge_file",
        source_id=source_id or uuid.uuid4(),
        title=title,
        content=content,
        meta=metadata or {},
    )
    db.add(item)
    db.flush()
    return item


def test_bm25_prefers_rare_error_name_and_exposes_trace() -> None:
    """测的是 BM25 的「稀有词加权」：罕见错误名应压过高频词的堆砌文档。

    场景：一篇文档反复出现 failure/retry/token（词频高但都是常见词），
    另一篇只出现一次 ExpiredSignatureError（极稀有，逆文档频率极高）。
    期望：含错误名的那篇排第一，bm25_score > 0、keyword_rank == 1、
    keyword_backend 标记为 bm25、且 bm25_matched_terms 里能查到命中的小写词项。
    重要性：排查线上问题时用户输入的往往就是报错类名，若 BM25 被高频词带偏，
    真正有用的那篇会被埋掉；trace 字段则是线上定位"为什么没搜到"的唯一线索。
    """
    db = _db_session()
    repo = _repo(db)
    # 干扰项：failure/retry/token 各重复两次，词频高但全是通用词
    _document(db, repo, title="General retry notes", content="failure retry token failure retry token")
    # 目标项：含唯一且罕见的异常类名，metadata 里还标了源码路径
    rare = _document(
        db,
        repo,
        title="JWT failure",
        content="ExpiredSignatureError is raised when the access token has expired.",
        metadata={"path": "backend/app/auth/token.py"},
    )
    db.commit()

    # apply_rerank=False：只看 BM25 本身的能力，不让重排掩盖问题
    results = search_similar_documents(
        db,
        repo.id,
        "ExpiredSignatureError",
        retrieval_method="fulltext",
        apply_rerank=False,
    )

    assert results[0]["id"] == str(rare.id)
    assert results[0]["bm25_score"] > 0
    assert results[0]["keyword_rank"] == 1
    assert results[0]["retrieval"]["keyword_backend"] == "bm25"
    # 命中词项统一小写化后记录，便于排查分词/大小写问题
    assert "expiredsignatureerror" in results[0]["retrieval"]["bm25_matched_terms"]


def test_metadata_state_filter_is_applied_before_bm25_scoring() -> None:
    """测的是元数据过滤发生在打分之前，而不是打完分再筛。

    场景：两篇内容完全相同的文档（authentication token failure），
    只有 metadata.state 一个是 open、一个是 closed；查询带 state=open 过滤。
    期望：结果里只剩 open 那篇。
    重要性：若先打分后过滤，BM25 的 idf 统计会把 closed 文档也算进语料，
    而且分页/limit 会被不可见文档占位，导致"明明有结果却返回空"。
    """
    db = _db_session()
    repo = _repo(db)
    open_doc = _document(
        db,
        repo,
        title="Open authentication issue",
        content="authentication token failure",
        metadata={"state": "open"},
    )
    _document(
        db,
        repo,
        title="Closed authentication issue",
        content="authentication token failure",
        metadata={"state": "closed"},
    )
    db.commit()

    results = search_similar_documents(
        db,
        repo.id,
        "authentication token",
        retrieval_method="fulltext",
        metadata_filters={"state": "open"},
        apply_rerank=False,
    )

    # 严格断言"只剩一条"，而不是"第一条是 open"——后者无法证明 closed 被排除了
    assert [item["id"] for item in results] == [str(open_doc.id)]


def test_weighted_fusion_uses_project_formula_and_dual_recall_bonus() -> None:
    """测的是融合公式的具体数值：权重改动必须让本用例失败。

    场景：shared 文档同时被向量路和关键词路召回（双路命中），
    vector-only 只被向量路召回，分数 0.8。
    期望：shared 融合分恰为 1.0（向量 0.62 + 关键词 0.30 + 双路加成 0.08），
    vector-only 为 0.496（0.8 × 0.62），fusion_weights 明确回报三项权重。
    重要性：融合权重是检索质量的"总阀门"，一次不经意的调参就会让所有
    排序结果漂移且难以归因；把公式钉死在测试里，是有意的保护性设计。
    """
    # 伪造向量路召回结果：分数与排名都要齐备，融合阶段会读 retrieval 里的细项
    vector = [
        {
            "id": "shared",
            "score": 1.0,
            "vector_score": 1.0,
            "metadata": {"chunk_id": "chunk-shared"},
            "retrieval": {"sources": ["vector"], "vector_score": 1.0, "vector_rank": 1},
        },
        {
            "id": "vector-only",
            "score": 0.8,
            "vector_score": 0.8,
            "metadata": {"chunk_id": "chunk-vector"},
            "retrieval": {"sources": ["vector"], "vector_score": 0.8, "vector_rank": 2},
        },
    ]
    # 伪造关键词路结果：只有 shared，构成"双路命中"的对比组
    keyword = [
        {
            "id": "shared",
            "score": 1.0,
            "keyword_score": 1.0,
            "bm25_score": 8.2,
            "metadata": {"chunk_id": "chunk-shared"},
            "retrieval": {"sources": ["keyword"], "keyword_score": 1.0, "keyword_rank": 1},
        }
    ]

    results = _merge_candidates(vector, keyword)

    assert results[0]["id"] == "shared"
    # 0.62*1.0 + 0.30*1.0 + 0.08（双路加成）= 1.0
    assert results[0]["combined_score"] == pytest.approx(1.0)
    assert results[0]["retrieval"]["fusion_method"] == "weighted"
    # 权重字典要能回报给前端/trace，任何数值变更都会在这里被拦下
    assert results[0]["retrieval"]["fusion_weights"] == {
        "vector": 0.62,
        "keyword": 0.30,
        "dual_recall_bonus": 0.08,
    }
    # 单路命中没有加成：0.8 * 0.62 = 0.496
    assert results[1]["combined_score"] == pytest.approx(0.496)


def test_parent_expansion_reads_only_same_parent_source_and_version() -> None:
    """测的是父块扩展的边界：只能聚合同一 source、同一 parent、同一分支的分块。

    场景：同一 source_id 下建 3 个 main 分支分块（parent-auth，index 0/1/2），
    再加 1 个 legacy 分支分块且 parent_id、child_index 都相同（=2）作为陷阱。
    然后拿 index=1 的分块去扩展。
    期望：把 main 的 0/1/2 按顺序拼成一段（\n\n 分隔），legacy 的内容
    绝不能被包含进来；stage.additional_chunks == 2（除自身外新增 2 块）。
    重要性：父块扩展是"用上下文补全被切碎的段落"，一旦版本/来源判断写漏，
    就会把旧分支的过期内容混进答案，这是最难被用户察觉的错误——答案读起来通顺，
    但依据的是几个月前的规范。
    """
    db = _db_session()
    repo = _repo(db)
    source_id = uuid.uuid4()
    rows = [
        _document(
            db,
            repo,
            title=f"auth.md#{index + 1}",
            content=content,
            source_id=source_id,
            metadata={
                "chunk_id": f"chunk-{index}",
                "parent_id": "parent-auth",
                "child_index": index,
                "branch": "main",
            },
        )
        for index, content in enumerate(["token setup", "refresh token failure", "retry once"])
    ]
    # 陷阱数据：parent_id 与 child_index 都能对上，只有 branch 不同，用来验证版本过滤
    _document(
        db,
        repo,
        title="old branch",
        content="must not be expanded",
        source_id=source_id,
        metadata={"chunk_id": "old", "parent_id": "parent-auth", "child_index": 2, "branch": "legacy"},
    )
    db.commit()

    # 只传入中间那块（index=1），扩展应向前后补齐同一 parent 的兄弟块
    items, stage = expand_parent_context(
        db,
        repo.id,
        [
            {
                "id": str(rows[1].id),
                "source_id": str(source_id),
                "source_type": "knowledge_file",
                "metadata": rows[1].meta,
                "_raw_content": rows[1].content,
                "retrieval": {},
            }
        ],
    )

    assert items[0]["parent_expanded"] is True
    # 按 child_index 顺序拼接，块之间用空行分隔
    assert items[0]["_raw_content"] == "token setup\n\nrefresh token failure\n\nretry once"
    # 核心断言：legacy 分支内容被排除
    assert "must not be expanded" not in items[0]["_raw_content"]
    assert stage["additional_chunks"] == 2


@pytest.mark.parametrize(
    ("query", "items", "expected"),
    [
        # 场景一：没有任何证据 → 直接判"证据不足"，不允许模型凭空作答
        ("token refresh", [], "insufficient_evidence"),
        (
            # 场景二：证据都相关但彼此指向不同处理方式，且问句指代模糊（"这个"）
            # → 求助澄清比给一个可能错一半的答案更合适
            "这个怎么办",
            [
                {"id": "a", "source_id": "one", "score": 0.9, "snippet": "第一种处理方式"},
                {"id": "b", "source_id": "two", "score": 0.88, "snippet": "第二种处理方式"},
            ],
            "ask_clarification",
        ),
        (
            # 场景三：两条高分证据结论互斥（允许 vs 禁止部署）→ 冲突
            # 这是最危险的情况：任意选一条都会造成生产事故
            "是否允许部署",
            [
                {"id": "a", "source_id": "one", "score": 0.92, "snippet": "当前版本允许部署。"},
                {"id": "b", "source_id": "two", "score": 0.9, "snippet": "当前版本禁止部署。"},
            ],
            "conflict",
        ),
        (
            # 场景四：单一高分证据且直接作答 → 正常放行
            "token refresh policy",
            [{"id": "a", "source_id": "one", "score": 0.92, "snippet": "Refresh the token once."}],
            "answer",
        ),
    ],
)
def test_answer_gate_exposes_all_project_decisions(query, items, expected) -> None:
    """测的是答案闸门的四种决策分支全覆盖。

    场景：用参数化覆盖 无证据 / 需澄清 / 冲突 / 可作答 四条路径，
    阈值统一设为 0.8 且启用。
    期望：decision 与预期一致，且每种决策都必须附 reason 说明理由。
    重要性：闸门是防幻觉的最后一道关；漏掉任何一个分支（比如冲突没被识别），
    模型就会自信地给出错误结论。要求 reason 则保证前端能向用户解释"为什么不答"。
    """
    gate = evaluate_answer_gate(query, items, score_threshold=0.8, threshold_enabled=True)

    assert gate["decision"] == expected
    # 必须给出可展示的理由，不能只返回一个枚举值
    assert "reason" in gate


@pytest.mark.asyncio
async def test_qwen_rerank_receives_raw_content_before_compression(monkeypatch) -> None:
    """测的是流水线顺序：重排看到原文，压缩发生在重排之后。

    场景：候选内容是「大量无关铺垫 + 末尾才出现关键句」，snippet 只有一句
    压缩前的短摘要。fake_qwen 内部断言收到的 _raw_content 以 "irrelevant setup"
    开头、且 items[0] 里没有 compression 字段。
    期望：重排后走压缩，compression.mode 为 query_aware_extractive，
    snippet 里出现查询关键实体；pipeline trace 的阶段名严格等于既定顺序；
    闸门最终判 answer。
    重要性：若先压缩再重排，重排模型只看得到被截断的片段，关键句可能已被丢掉，
    等于用残缺信息做排序决策；阶段顺序错乱也会让 trace 无法用于性能归因。
    """
    db = _db_session()
    repo = _repo(db)
    # 长前缀是为了模拟"关键信息在文末"的真实长文档，压缩若不感知查询就会丢掉它
    candidate = {
        "id": "auth",
        "source_id": str(uuid.uuid4()),
        "source_type": "knowledge_file",
        "title": "auth.md",
        "_raw_content": "irrelevant setup. " * 80 + "Refresh the token once after ExpiredSignatureError.",
        "snippet": "short pre-rerank snippet",
        "score": 0.91,
        "metadata": {},
        "retrieval": {"fusion_method": "weighted"},
    }

    # 伪造检索：需要通过 kwargs 把 trace 字典回填，模拟真实实现的副作用式追踪
    def fake_search(*_args, **kwargs):
        kwargs["pipeline_trace"].update({"fusion_method": "weighted", "stages": []})
        return [candidate]

    async def fake_qwen(_query, items, _config, _limit):
        # 关键断言①：重排拿到的是完整原文（未压缩）
        assert items[0]["_raw_content"].startswith("irrelevant setup")
        # 关键断言②：此时还没生成 compression 字段
        assert "compression" not in items[0]
        return [{**items[0], "score": 0.95, "rerank_score": 0.95, "retrieval": {**items[0]["retrieval"], "rerank_mode": "qwen_api"}}]

    monkeypatch.setattr("app.services.rag.knowledge_base.search_similar_documents", fake_search)
    monkeypatch.setattr("app.services.rag.knowledge_base._qwen_rerank", fake_qwen)

    items, trace, _duration = await retrieve_knowledge(
        db,
        repo.id,
        "ExpiredSignatureError token refresh",
        retrieval_method="fulltext",
        score_threshold_enabled=False,
    )

    # 压缩确实执行了，且用的是查询感知的抽取式压缩
    assert items[0]["compression"]["mode"] == "query_aware_extractive"
    # 压缩后仍保留了错名这个关键实体，说明压缩没有把答案句切掉
    assert "ExpiredSignatureError" in items[0]["snippet"]
    stage_names = [stage["name"] for stage in trace["pipeline"]["stages"]]
    # 阶段顺序即契约：重排 → 父块扩展 → 阈值 → 压缩 → 闸门
    assert stage_names == ["rerank", "parent_expansion", "score_threshold", "context_compression", "answer_gate"]
    assert trace["answer_gate"]["decision"] == "answer"


@pytest.mark.asyncio
async def test_qwen_failure_is_traced_and_falls_back_to_heuristic(monkeypatch) -> None:
    """测的是重排服务故障时的降级：超时不能让整个检索失败。

    场景：_qwen_rerank 抛 TimeoutError("rerank timeout")。
    期望：结果照常返回，rerank_mode 退化为 heuristic；
    rerank 阶段标记 fallback=True 且 fallback_reason 为 "TimeoutError"。
    重要性：外部模型 API 超时是常态，若不做降级，用户会看到整个问答报错；
    而记录降级原因又能避免"检索质量悄悄下降却没人发现"。
    """
    db = _db_session()
    repo = _repo(db)

    def fake_search(*_args, **kwargs):
        kwargs["pipeline_trace"].update({"fusion_method": "weighted", "stages": []})
        return [
            {
                "id": "auth",
                "source_id": str(uuid.uuid4()),
                "source_type": "knowledge_file",
                "title": "auth middleware",
                "_raw_content": "auth middleware refreshes the token",
                "snippet": "auth middleware refreshes the token",
                "score": 0.9,
                "metadata": {},
                "retrieval": {"sources": ["vector", "keyword"]},
            }
        ]

    async def fail_qwen(*_args, **_kwargs):
        raise TimeoutError("rerank timeout")

    monkeypatch.setattr("app.services.rag.knowledge_base.search_similar_documents", fake_search)
    monkeypatch.setattr("app.services.rag.knowledge_base._qwen_rerank", fail_qwen)

    items, trace, _duration = await retrieve_knowledge(
        db,
        repo.id,
        "auth middleware",
        retrieval_method="fulltext",
        score_threshold_enabled=False,
    )

    # 从 trace 里精确定位 rerank 阶段，断言降级信息被如实记录
    rerank_stage = next(stage for stage in trace["pipeline"]["stages"] if stage["name"] == "rerank")
    assert items[0]["retrieval"]["rerank_mode"] == "heuristic"
    assert rerank_stage["fallback"] is True
    # 记录异常类型而非泛化文案，便于区分超时/限流/鉴权失败
    assert rerank_stage["fallback_reason"] == "TimeoutError"


@pytest.mark.asyncio
async def test_retrieval_trace_contains_every_selected_pipeline_stage() -> None:
    """测的是完整流水线的阶段清单与 trace 字段齐备性。

    场景：真实跑一遍 retrieve_knowledge（关闭重排与阈值），数据只有一篇文档。
    期望：pipeline.stages 的 name 序列完整且顺序固定（9 个阶段全在），
    trace.fusion_method 为 fulltext，闸门判 answer，且结果里
    keyword_backend == bm25、selected_for_context == True。
    重要性：trace 是检索可观测性的唯一来源，前端"检索详情"面板和线上排查
    都依赖它；阶段缺失往往意味着代码里某段被条件跳过却没留下痕迹。
    """
    db = _db_session()
    repo = _repo(db)
    _document(
        db,
        repo,
        title="Authentication runbook",
        content="The auth middleware refreshes expired tokens once.",
        metadata={"path": "backend/app/auth/middleware.py"},
    )
    db.commit()

    items, trace, _duration = await retrieve_knowledge(
        db,
        repo.id,
        "auth middleware token",
        retrieval_method="fulltext",
        rerank_enabled=False,
        score_threshold_enabled=False,
    )

    stage_names = [stage["name"] for stage in trace["pipeline"]["stages"]]
    # 即使单路检索，也要把"未启用的一路"和"单检索器选择"作为阶段记录下来
    assert stage_names == [
        "vector_recall",
        "keyword_recall",
        "single_retriever_selection",
        "deduplication_and_parent_aggregation",
        "rerank",
        "parent_expansion",
        "score_threshold",
        "context_compression",
        "answer_gate",
    ]
    assert trace["fusion_method"] == "fulltext"
    assert trace["answer_gate"]["decision"] == "answer"
    # 走的是 BM25 而非退化的朴素关键词匹配
    assert items[0]["retrieval"]["keyword_backend"] == "bm25"
    # 标记该条被选入最终上下文（前端据此高亮）
    assert items[0]["retrieval"]["selected_for_context"] is True
