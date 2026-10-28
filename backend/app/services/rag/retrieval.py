"""检索（recall + fusion）：从向量库和关键词索引两路召回，再融合成一个排序。

位置：切块 → 向量化 → **检索** → 重排 → 生成 → 答案门禁。
本文件负责「检索」以及检索内部的融合/去重/父块扩展，输出一批带分数的候选，
交给 knowledge_base.retrieve_knowledge 去做重排与后续阶段。

为什么要双路召回，而不是只用向量：
- 向量擅长语义相近（「登录失败」能召回「认证不通过」），
  但对**精确 token** 很弱——错误码 `E5021`、函数名 `_merge_candidates`、
  配置键 `MILVUS_URI` 这类内容，向量化之后几乎没有区分度；
- BM25 恰好相反：精确 token 命中极准，但不懂同义改写。

两路互补，所以默认 retrieval_method="hybrid" 同时跑，然后按权重融合：
    combined = 0.62 * vector_score + 0.30 * keyword_score + 0.08 * 双路命中奖励
权重和为 1.0，最后再 min(1.0, ...) 兜一层。向量权重更高，是因为
大多数自然语言问题的语义信号比词面信号更可靠。

**降级策略**：Milvus 不可用时，hybrid 会静默退化为纯关键词检索
（vector_status="unavailable"），只有用户显式要求 method="vector" 时才抛错——
因为「明确要求向量检索」意味着他知道自己在要什么，静默给别的结果是欺骗。

去重与父块占位限制（MAX_CHUNKS_PER_PARENT=2）都是为了**证据多样性**：
同一份文档的 5 个分块全进上下文，等于用 5 个名额只讲了一件事。
"""

import heapq
import hashlib
import math
import re
import uuid
from collections import Counter
from datetime import datetime, time, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.db.models import Document
from app.services.context_compression import compress_retrieval_results
from app.services.rag.rerank import rerank_documents
from app.services.rag.source_policy import RAG_SOURCE_TYPES, is_searchable_source, normalize_source_types
from app.services.rag.vector_store import MilvusUnavailableError, search_milvus_documents


# BM25 的两个标准调参常量：
# K1 控制词频饱和速度（1.2~2.0 是常见区间，1.5 是通用默认值）
# B 控制文档长度归一化强度（0.75 是经典默认，0 表示不归一化）
BM25_K1 = 1.5
BM25_B = 0.75
# 融合权重。三者之和恰为 1.0，便于把 combined_score 解释成「加权相关度」
VECTOR_FUSION_WEIGHT = 0.62
KEYWORD_FUSION_WEIGHT = 0.30
# 双路都召回时的奖励分：两套独立算法都认为相关，可信度显著更高
DUAL_RECALL_BONUS = 0.08
# 同一父文档最多贡献 2 个分块。设成 2 而不是 1，是为了保留「相邻两句」
# 这种必要的上下文；但绝不能更多，否则一份长文档会霸占整个上下文窗口
MAX_CHUNKS_PER_PARENT = 2
# 近似重复的判定阈值（Jaccard 相似度）。0.9 很保守，只去掉几乎一字不差的片段，
# 避免误删「相似但说的事不同」的内容
NEAR_DUPLICATE_JACCARD = 0.9


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """余弦相似度。任一向量为空时返回 0.0。

    末位的 `or 1.0` 防止零向量导致除零——虽然 embeddings 都做了归一化，
    但这里作为通用工具函数仍需自保。
    """
    if not a or not b:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / ((math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))) or 1.0)


def _as_uuid(value: str | uuid.UUID) -> uuid.UUID:
    """字符串/UUID 归一化。"""
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


def _parse_datetime(value: str | None, end_of_day: bool = False) -> datetime | None:
    """解析 ISO 时间；纯日期（10 字符）按天补齐到当天起点或终点。

    end_of_day 的作用：用户说「截止到 3 月 1 日」时，意图是**包含** 3 月 1 日全天。
    如果用 00:00 作上界，3 月 1 日当天创建的数据会被排除掉，这是一个很常见的
    边界 bug。所以上界用 time.max。

    时区统一补 UTC：比较的双方必须有时区，否则 Python 会抛
    「can't compare offset-naive and offset-aware datetimes」。解析失败返回 None，
    由调用方决定是「忽略这个条件」还是「过滤掉全部」。
    """
    if not value:
        return None
    try:
        if len(value) == 10:
            parsed_date = datetime.fromisoformat(value).date()
            return datetime.combine(parsed_date, time.max if end_of_day else time.min, tzinfo=timezone.utc)
        # Z 要转成 +00:00（早期 Python 的 fromisoformat 不认 Z）
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _metadata_time(doc_meta: dict) -> datetime | None:
    """取文档的时间戳用于范围过滤，按「最能代表新鲜度」的顺序尝试。

    updated_at 优先：一条 issue 被更新后，旧版本的结论可能已失效，
    以最后更新时间为准更符合直觉。
    """
    for key in ["updated_at", "created_at", "merged_at", "closed_at"]:
        parsed = _parse_datetime(doc_meta.get(key))
        if parsed:
            return parsed
    return None


def _matches_metadata(doc_meta: dict, filters: dict | None) -> bool:
    """按 metadata 过滤候选（在 Python 侧做，不是 SQL）。

    这样做的原因：metadata 是 JSON 字段，跨数据库（PG / SQLite）的 JSON 查询
    语法不一致，而且 Milvus 返回的结果本来就已经在内存里了，再过一遍 SQL 反而更绕。

    支持的条件：
    - label：标签包含（labels 是列表，用 in 判断）
    - state：精确相等
    - path / path_prefix / module：模糊包含。有 files 列表时只看 files，
      没有时退化为全文搜索（覆盖 workflow_run 这类没有 files 的来源）
    - start_date / updated_after、end_date / updated_before：时间范围
    - branch：依次匹配 branch / base_branch / head_branch
    - 其他任意键：列表用 in，标量用 ==

    时间条件的处理有一个刻意的选择：**文档没有时间就视为不匹配**
    （`doc_time is None` 直接 False）。因为用户说「最近一个月」时，
    一条不知道是什么时候的数据不该被当成符合条件。
    """
    if not filters:
        return True

    label = filters.get("label")
    if label and label not in (doc_meta.get("labels") or []):
        return False

    state = filters.get("state")
    if state and doc_meta.get("state") != state:
        return False

    # 三个键是同一语义的不同叫法，取第一个有值的
    path = (filters.get("path") or filters.get("path_prefix") or filters.get("module") or "").lower()
    if path:
        files = [str(item).lower() for item in doc_meta.get("files") or []]
        if files and not any(path in filename for filename in files):
            return False
        # 没有 files 字段（如 CI 运行）时，把所有 metadata 值拼成文本再找
        if not files and path not in " ".join(str(value).lower() for value in doc_meta.values()):
            return False

    # 上界用 end_of_day=True，让「截止某天」包含当天
    start_at = _parse_datetime(filters.get("start_date") or filters.get("updated_after"))
    end_at = _parse_datetime(filters.get("end_date") or filters.get("updated_before"), end_of_day=True)
    doc_time = _metadata_time(doc_meta)
    if start_at and (doc_time is None or doc_time < start_at):
        return False
    if end_at and (doc_time is None or doc_time > end_at):
        return False

    for key, value in filters.items():
        # 跳过已专门处理过的键，以及空值（空值表示「不限制」）
        if key in {
            "label",
            "path",
            "path_prefix",
            "module",
            "start_date",
            "end_date",
            "updated_after",
            "updated_before",
        } or value in (None, ""):
            continue
        if key == "branch":
            # PR 的分支信息可能落在三个不同字段里
            current = doc_meta.get("branch") or doc_meta.get("base_branch") or doc_meta.get("head_branch")
            if current != value:
                return False
            continue
        current = doc_meta.get(key)
        if isinstance(current, list):
            if value not in current:
                return False
        elif current != value:
            return False
    return True


def _tokens(text: str) -> list[str]:
    """切词：英文按符号切，中文用单字 + 二字组。

    与 rerank / embeddings 保持一致的分词策略，避免三处对「什么算一个词」
    的理解不同而互相打架。
    """
    lowered = text.lower()
    tokens = re.findall(r"[a-z0-9_\-]+", lowered)
    for segment in re.findall(r"[\u4e00-\u9fff]+", lowered):
        tokens.extend(segment)
        tokens.extend(segment[index : index + 2] for index in range(max(0, len(segment) - 1)))
    return tokens


def _clip(text: str, limit: int = 260) -> str:
    """压空白 + 截断，超长时用省略号收尾。

    截到 limit - 1 再加 "…"，保证总长仍不超过 limit。
    """
    cleaned = " ".join((text or "").split())
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1].rstrip() + "…"


def _normalise_scores(items: list[dict[str, Any]], field: str) -> None:
    """把某个分数字段**原地**按最大值缩放到 [0, 1]。

    为什么必须归一化再融合：BM25 的分数是无上界的（可能是 3.7、也可能是 42），
    而向量相似度在 COSINE 下约在 [0, 1]。直接加权求和的话，BM25 会
    凭数值量级碾压向量分，融合权重就形同虚设。

    除以 max 而不是做 min-max：min-max 会把最低分压成 0，
    让「本来就弱相关」的候选显得完全无关，反而放大了噪音之间的差异。
    最大值只有 0 时用 1.0 兜底，避免除零。
    """
    if not items:
        return
    values = [max(0.0, float(item.get(field) or 0.0)) for item in items]
    max_value = max(values) or 1.0
    for item, value in zip(items, values):
        item[field] = round(value / max_value, 4)


def _document_result(doc: Document, score: float, source: str, score_field: str) -> dict[str, Any]:
    """把 ORM 文档转成统一的候选字典。

    统一结构的价值：无论来自向量路还是关键词路，后面所有阶段（融合、去重、
    重排、门禁）都只用这一套字段，不必关心数据从哪来。

    score_field 是动态键名（"vector_score" 或 "keyword_score"），
    用参数化落键，避免写两个几乎一样的函数。

    retrieval 子字典记录「这条候选的来历」，贯穿整条链路不断被补充
    （后面会陆续加入 rerank_mode、answer_gate 等），是排查问题的核心线索。
    """
    metadata = doc.meta or {}
    result = {
        "id": str(doc.id),
        "title": doc.title,
        "source_type": doc.source_type,
        "source_id": str(doc.source_id),
        "score": round(max(0.0, score), 4),
        "metadata": metadata,
        "snippet": _clip(doc.content),
        "_raw_content": doc.content,
        score_field: round(max(0.0, score), 4),
        "retrieval": {"sources": [source], score_field: round(max(0.0, score), 4)},
    }
    return result


def _milvus_vector_candidates(
    repo_id: str | uuid.UUID,
    query: str,
    source_types: list[str],
    fetch_limit: int,
    metadata_filters: dict | None,
    query_embedding: list[float] | None = None,
) -> list[dict[str, Any]]:
    """向量路召回。

    source_type / source_types 的传法有个细节：单类型时用精确匹配（source_type ==），
    多类型时才用 in 列表。这样生成的 Milvus 过滤表达式更简单，
    单类型查询也能用上更高效的过滤路径。

    metadata 过滤在 Python 侧做（Milvus 的标量过滤能力有限），
    所以这里取回 fetch_limit 条之后再筛，实际可能少于 fetch_limit。

    vector_score_raw 保留归一化前的原始相似度：归一化后的分数只适合做加权，
    但排查「这条到底有多像」时需要看原始值（比如 COSINE 下 0.83 才有参考意义）。
    """
    milvus_results = search_milvus_documents(
        repo_id,
        query,
        source_type=source_types[0] if len(source_types) == 1 else None,
        source_types=source_types if len(source_types) > 1 else None,
        limit=fetch_limit,
        query_embedding=query_embedding,
    )
    candidates = []
    for item in milvus_results:
        if not _matches_metadata(item.get("metadata") or {}, metadata_filters):
            continue
        score = max(0.0, float(item.get("score") or 0.0))
        candidates.append(
            {
                **item,
                # 兜底链：Milvus 的 document_id → source_id → 空串
                "id": str(item.get("id") or item.get("source_id") or ""),
                "score": score,
                "vector_score": score,
                "vector_score_raw": score,
                "retrieval": {
                    "sources": ["vector"],
                    "vector_score_raw": score,
                    "vector_backend": "milvus",
                },
            }
        )
    _normalise_scores(candidates, "vector_score")
    for rank, item in enumerate(candidates, start=1):
        # 归一化之后把 score 也同步成归一化值，供还没有重排的调用方直接用
        item["score"] = item["vector_score"]
        item["retrieval"]["vector_score"] = item["vector_score"]
        item["vector_rank"] = rank
        item["retrieval"]["vector_rank"] = rank
    return candidates


def _bm25_document_stats(doc: Document, query_terms: list[str]) -> dict[str, Any]:
    """预计算单文档的 BM25 中间量（词频、字段权重、文档长度）。

    为什么要分字段加权：BM25 原版把整篇文档当成一袋词，但检索场景里
    「命中在标题」和「命中在正文第 300 行」的价值天差地别。所以这里先按字段
    算加权词频，再统一喂给 BM25 公式：
    - symbol 4.5（函数名/类名命中几乎必然强相关，权重最高）
    - title 4.0
    - path 3.0（文件路径命中说明找对文件了）
    - labels 2.5
    - content 1.0（正文是基线）

    先算好 stats 再统一算分，是为了能在**全语料**上先统计 document_frequency
    （IDF 需要全局信息），而不是边遍历边算——那样 IDF 就只能用局部近似。
    """
    metadata = doc.meta or {}
    field_tokens = {
        "title": _tokens(doc.title or ""),
        "path": _tokens(str(metadata.get("path") or "")),
        "symbol": _tokens(str(metadata.get("symbol") or metadata.get("name") or "")),
        "labels": _tokens(" ".join(str(item) for item in metadata.get("labels") or [])),
        "content": _tokens(doc.content or ""),
    }
    field_weights = {"title": 4.0, "path": 3.0, "symbol": 4.5, "labels": 2.5, "content": 1.0}
    counters = {field: Counter(tokens) for field, tokens in field_tokens.items()}
    # 加权词频：一个词在不同字段出现，贡献累加
    weighted_tf = {
        term: sum(float(counters[field].get(term, 0)) * field_weights[field] for field in field_tokens)
        for term in query_terms
    }
    matched_terms = [term for term in query_terms if weighted_tf.get(term, 0.0) > 0.0]
    # 文档长度用全部字段的 token 总数；max(1, ...) 防止空文档导致除零
    document_length = max(1, sum(len(tokens) for tokens in field_tokens.values()))
    return {
        "doc": doc,
        "weighted_tf": weighted_tf,
        "matched_terms": matched_terms,
        "document_length": document_length,
    }


def _bm25_score(
    stats: dict[str, Any],
    *,
    document_count: int,
    document_frequency: Counter[str],
    average_document_length: float,
) -> float:
    """标准 BM25 打分，只是把词频换成上面算好的加权词频。

    公式：sum over terms of
        idf(t) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))

    IDF 用 log1p 形式：log(1 + (N - df + 0.5) / (df + 0.5))，
    这个变体保证结果恒为非负（原始 log 形式在 df > N/2 时会变负，
    出现在绝大多数文档里的词反而拉低分数，不符合直觉）。

    length_norm 是长度归一化：长文档天然更容易命中词，需要打折。
    平均长度用 max(..., 1.0) 兜底，防止空语料除零。

    最后 max(0.0, score)：理论上不该为负，但浮点误差或极端参数下仍可能，
    负分会影响后面的 top-N 堆比较。
    """
    score = 0.0
    document_length = float(stats["document_length"])
    length_norm = BM25_K1 * (1.0 - BM25_B + BM25_B * document_length / max(average_document_length, 1.0))
    for term, weighted_tf in stats["weighted_tf"].items():
        if weighted_tf <= 0:
            continue
        df = float(document_frequency.get(term, 0))
        idf = math.log1p((document_count - df + 0.5) / (df + 0.5))
        score += idf * (weighted_tf * (BM25_K1 + 1.0)) / (weighted_tf + length_norm)
    return max(0.0, score)


def _keyword_candidates(
    db: Session,
    repo_id: str | uuid.UUID,
    query: str,
    source_types: list[str],
    fetch_limit: int,
    metadata_filters: dict | None,
) -> list[dict[str, Any]]:
    """关键词路召回：在业务库上全量扫一遍算 BM25。

    为什么是「全量扫」而不是走 SQL 索引：BM25 需要全语料的 document_frequency
    （IDF），无法用单条 SQL 的 WHERE 精确表达。当前数据量（单仓库几千条文档）
    下，全量扫一遍只需几十毫秒，可以接受；数据量再上一个数量级就应该换成
    Elasticsearch / PostgreSQL 的全文检索。

    两个性能/内存技巧：
    - yield_per(500)：游标分批取，避免一次性把所有行读进内存（几万条文档时
      差别很大）；
    - heapq 维护大小为 fetch_limit 的最小堆：只保留 Top-N，而不是把所有
      候选（可能上万条）都留在内存里再排序。heapreplace 的语义是
      「弹出最小、压入新的」，正好维持「堆里始终是当前最大的 N 个」。

    state 条件在 SQL 层就用 JSON 表达式过滤了（这是唯一下推到 SQL 的 metadata
    条件），因为 state 是最常用、最有选择性的过滤项，能显著减少扫描行数。

    document_frequency.update(set(matched_terms)) 用 set：一个词在同一文档里
    出现 10 次，df 也只算 1（df 的定义就是「包含该词的文档数」）。
    """
    terms = list(dict.fromkeys(_tokens(query)))
    # 查询切不出词（全是标点）：无法算 BM25，直接返回空
    if not terms:
        return []
    q = db.query(Document).filter(
        Document.repo_id == _as_uuid(repo_id),
        Document.source_type.in_(source_types),
    )
    state = (metadata_filters or {}).get("state")
    if state:
        q = q.filter(Document.meta["state"].as_string() == str(state))

    corpus: list[dict[str, Any]] = []
    document_frequency: Counter[str] = Counter()
    total_document_length = 0
    # 第一遍：收集统计量（此时还不能算分，因为 IDF 需要全局 df）
    for doc in q.yield_per(500):
        if not _matches_metadata(doc.meta or {}, metadata_filters):
            continue
        stats = _bm25_document_stats(doc, terms)
        corpus.append(stats)
        total_document_length += int(stats["document_length"])
        document_frequency.update(set(stats["matched_terms"]))

    if not corpus:
        return []

    average_document_length = total_document_length / len(corpus)
    # 第二遍：算分并维护 Top-N 最小堆。
    # 元组第二项 sequence 是「插入顺序」，用来让同分候选的排序稳定可复现
    top_candidates: list[tuple[float, int, dict[str, Any]]] = []
    for sequence, stats in enumerate(corpus):
        doc = stats["doc"]
        score = _bm25_score(
            stats,
            document_count=len(corpus),
            document_frequency=document_frequency,
            average_document_length=average_document_length,
        )
        # 0 分表示一个查询词都没命中，直接丢弃
        if score <= 0:
            continue
        item = _document_result(doc, score, "keyword", "keyword_score")
        item["bm25_score"] = round(score, 4)
        item["bm25_matched_terms"] = list(stats["matched_terms"])
        item["retrieval"].update(
            {
                "keyword_backend": "bm25",
                "bm25_score": round(score, 4),
                "bm25_matched_terms": list(stats["matched_terms"]),
                # 把 k1/b 也记下来，便于「同样的结果为什么分数不同」的排查
                "bm25_k1": BM25_K1,
                "bm25_b": BM25_B,
            }
        )
        entry = (score, sequence, item)
        if len(top_candidates) < fetch_limit:
            heapq.heappush(top_candidates, entry)
        elif score > top_candidates[0][0]:
            # 堆顶是当前 N 个里最小的，比它大才值得替换
            heapq.heapreplace(top_candidates, entry)
    # 堆里是乱序的，按分数降序、同分按插入顺序重排
    candidates = [
        entry[2]
        for entry in sorted(top_candidates, key=lambda entry: (entry[0], entry[1]), reverse=True)
    ]
    _normalise_scores(candidates, "keyword_score")
    for rank, item in enumerate(candidates, start=1):
        item["score"] = item["keyword_score"]
        item["retrieval"]["keyword_score"] = item["keyword_score"]
        item["keyword_rank"] = rank
        item["retrieval"]["keyword_rank"] = rank
    return candidates


def _merge_candidates(vector_items: list[dict[str, Any]], keyword_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把向量路与关键词路的结果按「同一条数据」合并，并算融合分。

    合并键的选择有优先级：
    1. metadata.chunk_id —— 最精确，一个 chunk 一个 id；
    2. item.id —— 退而求其次（文档级 id）；
    3. source_type:source_id:title —— 最后兜底。
    之所以优先 chunk_id，是因为同一份文档被切成多个 chunk 时，
    只有 chunk_id 能区分「是不是同一段文字」。

    合并时的取值规则：
    - 各种 score 取 max：两路都给出分数时，取更乐观的那个。取平均会
      惩罚「只有一路召回」的候选，而那恰恰可能是另一路漏掉的好结果；
    - snippet / _raw_content 取**更长**的那个：内容更完整总是更好；
    - sources 累积并去重（["vector", "keyword"]），这是双路命中奖励的依据。

    融合分：
        combined = 0.62*vector + 0.30*keyword + 0.08*(双路都命中)
    双路奖励只在两路分数都非零时给，且用 min(1.0, ...) 封顶。
    """
    merged: dict[str, dict[str, Any]] = {}
    for source_name, items in [("vector", vector_items), ("keyword", keyword_items)]:
        for item in items:
            metadata = item.get("metadata") or {}
            key = str(
                metadata.get("chunk_id")
                or item.get("id")
                or f"{item.get('source_type')}:{item.get('source_id')}:{item.get('title')}"
            )
            current = merged.get(key)
            if current is None:
                # 首次遇到：深拷贝一层 retrieval，避免后续修改污染入参
                current = {**item, "retrieval": {**(item.get("retrieval") or {})}}
                current["retrieval"]["sources"] = list(dict.fromkeys(current["retrieval"].get("sources") or [source_name]))
                merged[key] = current
            else:
                # 已存在：各项取 max（见 docstring 的说明）
                current["vector_score"] = max(float(current.get("vector_score") or 0.0), float(item.get("vector_score") or 0.0))
                current["keyword_score"] = max(float(current.get("keyword_score") or 0.0), float(item.get("keyword_score") or 0.0))
                current["vector_score_raw"] = max(
                    float(current.get("vector_score_raw") or 0.0), float(item.get("vector_score_raw") or 0.0)
                )
                current["bm25_score"] = max(float(current.get("bm25_score") or 0.0), float(item.get("bm25_score") or 0.0))
                current["snippet"] = max([str(current.get("snippet") or ""), str(item.get("snippet") or "")], key=len)
                current["_raw_content"] = max(
                    [str(current.get("_raw_content") or ""), str(item.get("_raw_content") or "")], key=len
                )
                current["retrieval"]["sources"] = list(dict.fromkeys([*(current["retrieval"].get("sources") or []), source_name]))
                # 把另一路的 trace 字段也合并进来（sources 已单独处理，跳过）
                for trace_key, trace_value in (item.get("retrieval") or {}).items():
                    if trace_key == "sources":
                        continue
                    current["retrieval"][trace_key] = trace_value
    output = []
    for item in merged.values():
        vector_score = float(item.get("vector_score") or 0.0)
        keyword_score = float(item.get("keyword_score") or 0.0)
        both_bonus = DUAL_RECALL_BONUS if vector_score and keyword_score else 0.0
        combined = min(1.0, VECTOR_FUSION_WEIGHT * vector_score + KEYWORD_FUSION_WEIGHT * keyword_score + both_bonus)
        item["combined_score"] = round(combined, 4)
        item["fusion_score"] = round(combined, 4)
        item["score"] = round(combined, 4)
        item["retrieval"].update(
            {
                "fusion_method": "weighted",
                # 把权重写进 trace：换了权重之后，历史记录仍能解释「当时的分数怎么来的」
                "fusion_weights": {
                    "vector": VECTOR_FUSION_WEIGHT,
                    "keyword": KEYWORD_FUSION_WEIGHT,
                    "dual_recall_bonus": DUAL_RECALL_BONUS,
                },
                "dual_recall_bonus": both_bonus,
                "combined_score": item["combined_score"],
                "fusion_score": item["fusion_score"],
            }
        )
        output.append(item)
    output.sort(key=lambda item: float(item.get("combined_score") or 0.0), reverse=True)
    for rank, item in enumerate(output, start=1):
        item["fusion_rank"] = rank
        item["retrieval"]["fusion_rank"] = rank
    return output


def _normalised_fingerprint(text: str) -> str:
    """文本指纹：只保留字母/数字/下划线/汉字，再取 sha256。

    这样「登录 失败！」与「登录失败」会得到同一个指纹，能识别「仅标点或空白
    不同」的重复。用哈希而不是原串做 key，是为了让内存占用与文本长度无关。
    归一化后为空则返回空串（调用方会把空指纹视为「不可比较」）。
    """
    normalised = re.sub(r"[^a-z0-9_\-\u4e00-\u9fff]+", "", str(text or "").lower())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest() if normalised else ""


def _candidate_term_set(item: dict[str, Any]) -> set[str]:
    """取候选的词集合，用于算 Jaccard 相似度。优先用完整正文而不是 snippet。"""
    text = str(item.get("_raw_content") or item.get("snippet") or "")
    return set(_tokens(text))


def _same_duplicate_scope(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """判断两个候选是否属于「可以互相判定为重复」的同一范围。

    为什么要限定范围：不同来源里出现相同句子是完全正常的（比如两个 issue
    都在引用同一段报错日志），它们各自都有价值，不能当成重复去掉。

    判定优先级：都有 parent_id 就只比 parent_id（同一个父文档的子块）；
    否则比较 (source_type, source_id)，即同一条上游数据。
    """
    left_meta = left.get("metadata") or {}
    right_meta = right.get("metadata") or {}
    left_parent = str(left_meta.get("parent_id") or "")
    right_parent = str(right_meta.get("parent_id") or "")
    if left_parent and right_parent:
        return left_parent == right_parent
    return (
        str(left.get("source_type") or "") == str(right.get("source_type") or "")
        and str(left.get("source_id") or "") == str(right.get("source_id") or "")
    )


def _deduplicate_candidates(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """两级去重：先精确指纹，再近似（Jaccard）。

    精确那级用指纹字典 O(1) 命中；近似那级需要与已保留的每个候选比一次，
    所以只对「指纹没命中」的候选做，实际开销可控。

    被删掉的候选不会被静默丢弃：它的 id 会追加到保留者的
    retrieval.duplicates_removed 里，removed 列表也会返回给 trace。
    这样「为什么这条没出现」永远有据可查。

    注意近似判定在 current_terms 或 existing_terms 为空时直接跳过（continue）：
    空集合算 Jaccard 会得到 0/0，没有意义。
    """
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    fingerprints: dict[str, dict[str, Any]] = {}
    term_sets: dict[str, set[str]] = {}
    for item in items:
        item_id = str(item.get("id") or "")
        raw_text = str(item.get("_raw_content") or item.get("snippet") or "")
        fingerprint = _normalised_fingerprint(raw_text)
        duplicate_of: dict[str, Any] | None = fingerprints.get(fingerprint) if fingerprint else None
        # 指纹相同但不在同一范围 → 不算重复（见 _same_duplicate_scope 的说明）
        if duplicate_of is not None and not _same_duplicate_scope(item, duplicate_of):
            duplicate_of = None
        if duplicate_of is None:
            current_terms = _candidate_term_set(item)
            for existing in kept:
                if not _same_duplicate_scope(item, existing):
                    continue
                existing_id = str(existing.get("id") or "")
                # 缓存已算过的词集合，避免重复分词
                existing_terms = term_sets.get(existing_id) or _candidate_term_set(existing)
                if not current_terms or not existing_terms:
                    continue
                union = current_terms | existing_terms
                similarity = len(current_terms & existing_terms) / max(len(union), 1)
                if similarity >= NEAR_DUPLICATE_JACCARD:
                    duplicate_of = existing
                    break
            term_sets[item_id] = current_terms
        if duplicate_of is not None:
            duplicate_id = str(duplicate_of.get("id") or "")
            duplicate_trace = duplicate_of.setdefault("retrieval", {})
            duplicate_trace.setdefault("duplicates_removed", []).append(item_id)
            removed.append({"id": item_id, "duplicate_of": duplicate_id})
            continue
        kept.append(item)
        if fingerprint:
            fingerprints[fingerprint] = item
    return kept, removed


def _limit_parent_occupancy(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """限制每个父文档最多贡献 MAX_CHUNKS_PER_PARENT 个候选。

    输入必须已经按分数降序排好（调用方保证了这点），这样「先到先得」
    等价于「保留分数最高的 N 个」。

    没有 parent_id 的候选不受限制（它们本来就是独立的文档，不是某个父文档的分块）。
    """
    counts: Counter[str] = Counter()
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for item in items:
        metadata = item.get("metadata") or {}
        parent_id = str(metadata.get("parent_id") or "")
        if parent_id and counts[parent_id] >= MAX_CHUNKS_PER_PARENT:
            removed.append({"id": str(item.get("id") or ""), "parent_id": parent_id})
            continue
        kept.append(item)
        if parent_id:
            counts[parent_id] += 1
    return kept, removed


def _same_version(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """判断两条 metadata 是否属于「同一个版本」。

    逐个比较 branch / commit_sha / version / checksum / knowledge_document_id，
    只要有一项双方都有值却不相等，就认为版本不同。

    关键规则：**只有双方都有值时才比较**。如果一方缺这个字段（老数据没有
    checksum、issue 没有 commit_sha），不能因此判定版本不同——那会把大量
    正常候选排除掉。宁可放过，不可错杀。
    """
    for key in ("branch", "commit_sha", "version", "checksum", "knowledge_document_id"):
        left_value = left.get(key)
        right_value = right.get(key)
        if left_value is not None and right_value is not None and left_value != right_value:
            return False
    return True


def expand_parent_context(
    db: Session,
    repo_id: str | uuid.UUID,
    items: list[dict[str, Any]],
    *,
    neighbor_window: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """父块扩展：把命中分块的**相邻分块**合并进来，补全被切断的上下文。

    为什么需要：切块时一段话可能被切成 3 片，命中中间那片时，前因后果都在
    另外两片里。只把这半句话给模型，它只能猜。

    做法：对每个候选，按 (repo_id, source_type, source_id) 找出同一来源的所有
    分块，再筛出 parent_id 相同、版本相同、child_index 距离 <= neighbor_window 的，
    按 child_index 排序后拼成一个完整文本（覆盖 _raw_content）。

    四个「不满足就原样返回」的条件（都是常见情况，不是错误）：
    - 缺 parent_id / child_index / source_id / source_type：老数据或非切块来源，
      没有父块概念；
    - source_id 不是合法 UUID；
    - 找不到匹配的兄弟分块；
    - 兄弟分块内容全为空。

    每种情况都会写一个明确的 reason 到 retrieval.parent_expansion，
    这样「为什么这条没扩展」可以直接从返回结果看出来。

    parent_expanded 的判定是「扩出来的东西确实比原来多」：
    len(expanded_ids) > 1 或 原 id 不在列表里。前者是拿到了多个分块，
    后者说明连命中的那条都被去重丢掉了（内容重复）。

    注意这里**不改变 score**：扩展只影响内容长度，不影响相关性排序。
    """
    expanded: list[dict[str, Any]] = []
    expanded_items = 0
    expanded_chunks = 0
    for item in items:
        metadata = dict(item.get("metadata") or {})
        parent_id = str(metadata.get("parent_id") or "")
        child_index = metadata.get("child_index")
        source_id = item.get("source_id")
        source_type = str(item.get("source_type") or "")
        retrieval = dict(item.get("retrieval") or {})
        if not parent_id or child_index is None or not source_id or not source_type:
            retrieval["parent_expansion"] = {"expanded": False, "reason": "parent metadata unavailable"}
            expanded.append({**item, "retrieval": retrieval})
            continue
        try:
            source_uuid = _as_uuid(source_id)
        except (TypeError, ValueError):
            retrieval["parent_expansion"] = {"expanded": False, "reason": "invalid source id"}
            expanded.append({**item, "retrieval": retrieval})
            continue
        # 一次取回该来源的全部分块，再在内存里筛（分块总数一般很小）
        siblings = (
            db.query(Document)
            .filter(
                Document.repo_id == _as_uuid(repo_id),
                Document.source_type == source_type,
                Document.source_id == source_uuid,
            )
            .all()
        )
        selected: list[Document] = []
        current_index = int(child_index)
        for sibling in siblings:
            sibling_meta = sibling.meta or {}
            # 同一个 parent 才算「同一节」，否则会把别的章节也拼进来
            if str(sibling_meta.get("parent_id") or "") != parent_id:
                continue
            # 版本不同（比如代码已在另一个分支改过）就不能混着拼
            if not _same_version(metadata, sibling_meta):
                continue
            sibling_index = int(sibling_meta.get("child_index") or 0)
            if abs(sibling_index - current_index) <= max(0, neighbor_window):
                selected.append(sibling)
        # 按 child_index 升序，保证拼接后的顺序与原文一致
        selected.sort(key=lambda row: int((row.meta or {}).get("child_index") or 0))
        if not selected:
            retrieval["parent_expansion"] = {"expanded": False, "reason": "no matching siblings"}
            expanded.append({**item, "retrieval": retrieval})
            continue
        parts: list[str] = []
        seen_parts: set[str] = set()
        expanded_ids: list[str] = []
        for sibling in selected:
            content = str(sibling.content or "").strip()
            fingerprint = _normalised_fingerprint(content)
            # 相邻分块在重叠区会有重复内容，拼接前先去重
            if not content or fingerprint in seen_parts:
                continue
            seen_parts.add(fingerprint)
            parts.append(content)
            expanded_ids.append(str(sibling.id))
        if not parts:
            retrieval["parent_expansion"] = {"expanded": False, "reason": "empty sibling content"}
            expanded.append({**item, "retrieval": retrieval})
            continue
        parent_expanded = len(expanded_ids) > 1 or str(item.get("id") or "") not in expanded_ids
        retrieval["parent_expansion"] = {
            "expanded": parent_expanded,
            "parent_id": parent_id,
            "chunk_ids": expanded_ids,
            "neighbor_window": max(0, neighbor_window),
        }
        # expanded_chunk_ids 写进 metadata：下游可以据此知道这个证据由哪几块组成
        next_metadata = {**metadata, "expanded_chunk_ids": expanded_ids}
        expanded.append(
            {
                **item,
                "metadata": next_metadata,
                # 只覆盖 _raw_content，不动 snippet——snippet 用于列表展示，
                # 保持短小；_raw_content 才是喂给模型的完整内容
                "_raw_content": "\n\n".join(parts),
                "parent_expanded": parent_expanded,
                "retrieval": retrieval,
            }
        )
        if parent_expanded:
            expanded_items += 1
            # len - 1：减去原本命中的那一条，得到「额外补充了几块」
            expanded_chunks += max(0, len(expanded_ids) - 1)
    return expanded, {
        "name": "parent_expansion",
        "input_count": len(items),
        "output_count": len(expanded),
        "expanded_items": expanded_items,
        "additional_chunks": expanded_chunks,
    }


def search_similar_documents(
    db: Session,
    repo_id: str | uuid.UUID,
    query: str,
    source_type: str | None = None,
    limit: int = 5,
    metadata_filters: dict | None = None,
    query_embedding: list[float] | None = None,
    retrieval_method: str = "hybrid",
    apply_rerank: bool = True,
    apply_compression: bool = True,
    source_types: list[str] | None = None,
    pipeline_trace: dict[str, Any] | None = None,
) -> list[dict]:
    """检索主函数：召回 → 融合 → 去重 → 限父占用 → 重排 → 压缩。

    参数里三个 apply_* 是给上层编排用的开关：
    - apply_rerank=False：只做召回和融合，重排交给调用方（retrieve_knowledge
      就是这样用的，因为它要先拿到更多候选，再用 Qwen 或启发式统一重排）；
    - apply_compression=False：不做上下文压缩，同理。

    默认全开是为了让「直接调用本函数」的简单场景也能拿到可用结果。

    类型过滤分两层：
    1. normalize_source_types 合并 source_type（单个）与 source_types（列表）并去重；
       两者都没给时，用**全部 RAG 来源类型**作为默认；
    2. is_searchable_source 再过一遍策略白名单——直连类型（team_member 等）
       根本不在向量库/关键词索引里，传进来也只会浪费一次查询。

    fetch_limit = limit * 8 并夹到 [40, 200]：召回要远多于最终需要的条数，
    才有去重、限占用、重排的余地。上限 200 是延迟与内存的权衡——
    再大对结果质量的边际收益很小，但每次查询的代价线性上升。

    Milvus 失败的降级逻辑：hybrid 模式下静默跳过向量路（vector_status
    记为 unavailable，trace 里能看到）；vector 模式下直接 raise，
    因为调用方明确只要向量结果，给别的等于撒谎。
    """
    requested_types = normalize_source_types([source_type, *(source_types or [])]) or sorted(RAG_SOURCE_TYPES)
    # dict.fromkeys 保序去重：让同一组输入产生稳定的查询计划与 trace
    searchable_types = list(dict.fromkeys(item for item in requested_types if is_searchable_source(item)))
    if not searchable_types:
        return []

    fetch_limit = min(max(limit * 8, 40), 200)
    method = retrieval_method.strip().lower()
    vector_types = searchable_types
    vector_candidates: list[dict[str, Any]] = []
    keyword_candidates: list[dict[str, Any]] = []
    stages: list[dict[str, Any]] = []
    vector_status = "skipped"
    if method in {"vector", "hybrid"} and vector_types:
        try:
            vector_candidates = _milvus_vector_candidates(
                repo_id,
                query,
                vector_types,
                fetch_limit,
                metadata_filters,
                query_embedding=query_embedding,
            )
            vector_status = "success"
        except MilvusUnavailableError:
            # hybrid 可以没有向量路，但纯 vector 模式不能
            vector_status = "unavailable"
            if method == "vector":
                raise
    stages.append(
        {
            "name": "vector_recall",
            "status": vector_status,
            "candidate_count": len(vector_candidates),
            "backend": "milvus",
        }
    )
    # 关键词路只在 fulltext / hybrid 时跑：vector 模式下用户明确不要词面匹配
    keyword_types = searchable_types if method in {"fulltext", "hybrid"} else []
    if keyword_types:
        keyword_candidates = _keyword_candidates(
            db,
            repo_id,
            query,
            keyword_types,
            fetch_limit,
            metadata_filters,
        )
    stages.append(
        {
            "name": "keyword_recall",
            "status": "success" if keyword_types else "skipped",
            "candidate_count": len(keyword_candidates),
            "backend": "bm25",
        }
    )
    if method == "vector":
        merged = vector_candidates
    elif method == "fulltext":
        merged = keyword_candidates
    else:
        merged = _merge_candidates(vector_candidates, keyword_candidates)
    stages.append(
        {
            # hybrid 之外的两种模式没有「融合」这一步，但仍要记一条 stage 保持结构一致
            "name": "weighted_fusion" if method == "hybrid" else "single_retriever_selection",
            "method": "weighted" if method == "hybrid" else method,
            "candidate_count": len(merged),
            "weights": {
                "vector": VECTOR_FUSION_WEIGHT,
                "keyword": KEYWORD_FUSION_WEIGHT,
                "dual_recall_bonus": DUAL_RECALL_BONUS,
            },
        }
    )
    # 去重必须在限占用之前：先去重再做父块计数，才不会把重复项也算进配额
    deduplicated, duplicates_removed = _deduplicate_candidates(merged)
    aggregated, parent_overflow_removed = _limit_parent_occupancy(deduplicated)
    stages.append(
        {
            "name": "deduplication_and_parent_aggregation",
            "input_count": len(merged),
            "output_count": len(aggregated),
            "duplicates_removed": duplicates_removed,
            "parent_overflow_removed": parent_overflow_removed,
        }
    )
    # 重排的输入是「已去重、已限占用」的候选；不开重排时直接截前 limit 条
    ranked = rerank_documents(query, aggregated, limit) if apply_rerank else aggregated[:limit]
    if apply_rerank:
        stages.append(
            {
                "name": "rerank",
                "input_count": len(aggregated),
                "output_count": len(ranked),
                # 从第一条的 trace 里取实际用的重排方式（qwen_api / heuristic）
                "mode": ranked[0].get("retrieval", {}).get("rerank_mode") if ranked else None,
            }
        )
    if pipeline_trace is not None:
        # update 而不是覆盖：pipeline_trace 可能已被上层预置了字段
        pipeline_trace.update(
            {
                "retrieval_method": method,
                "fusion_method": "weighted" if method == "hybrid" else method,
                "stages": stages,
            }
        )
    if not apply_compression:
        return ranked
    # 这里的压缩没有传 max_total_tokens，用压缩模块的默认预算
    compressed = compress_retrieval_results(query, ranked, max_tokens_per_item=260)
    if pipeline_trace is not None:
        pipeline_trace["stages"].append(
            {
                "name": "context_compression",
                "input_count": len(ranked),
                "output_count": len(compressed),
            }
        )
    return compressed
