"""向量库：Milvus 集合定义、双写、检索入口。

这是 RAG 链路里「检索」环节的存储底座。写入路径与读取路径的职责完全不同：

写入侧（add_document / add_documents / replace_*）
    一次写入要同时落到两个地方——
    1. Postgres 的 Document 表：保留全文与元数据，是**唯一事实来源**；
    2. Milvus 集合：只存向量 + 用于检索/展示的少量字段。
    两者必须在同一个事务语义下推进（见 add_documents 的 commit 参数），
    否则会出现「向量库有、业务库没有」或反过来的脏数据。

读取侧（search_milvus_documents）
    用查询向量在 Milvus 做 ANN 检索，按 repo_id 强隔离，返回带 score 的候选列表。
    注意它返回的只是**候选**，后面还要经过 rerank 与 answer_gate 才能变成答案。

集合与索引策略（_ensure_collection）：
- 单集合多租户：所有仓库共用一个 collection，靠 repo_id 标量字段过滤隔离。
  这样避免「每个仓库建一个集合」导致集合数量爆炸、资源碎片化。
- 索引类型 AUTOINDEX：交给 Milvus 按数据量自动选索引（小数据量走 IVF 系，
  大数据量自动切 HNSW 之类），运维成本最低；代价是不能精细调参。
- 度量 COSINE：与 embeddings 里的 normalize_embeddings=True 匹配，
  归一化后的向量用余弦最直观（取值范围 -1~1，阈值好定）。
- enable_dynamic_field=True：允许写入未在 schema 里声明的字段，
  代价是这些字段没有索引、不能用于过滤，只能读出来看看。

一个硬约束：集合的向量维度**一旦建成就不能改**。所以启动时会比对
配置的 dimensions 与集合实际维度，不一致直接报错并提示换集合名重建——
绝不做「自动迁移」，因为那意味着所有历史向量都要重算。
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import Document
from app.services.rag.embeddings import embed_text, embed_texts, embedding_contract, embedding_provider_name
from app.services.rag.source_policy import policy_metadata, should_vectorize

logger = logging.getLogger(__name__)


class MilvusUnavailableError(RuntimeError):
    """Milvus 未启用、依赖缺失或连接失败时的统一异常。

    单独一个异常类型，是为了让上层能区分「检索功能不可用（可降级到关键词检索）」
    和「业务逻辑出错（必须失败）」。继承 RuntimeError 表示这是运行环境问题，
    不是调用方的用法问题。
    """
    pass

try:  # pymilvus 在测试和本地演示模式中是可选依赖。
    from pymilvus import DataType, MilvusClient
except Exception:  # pragma: no cover - 依赖缺失时会走到这里。
    # 置 None 而不是让 import 直接失败：不启用向量的部署（只用直连查询）
    # 也应该能正常启动，真正调用向量功能时再由 _milvus_store 报错
    DataType = None  # type: ignore[assignment]
    MilvusClient = None  # type: ignore[assignment]


@dataclass
class DocumentPayload:
    """写入一条 RAG 文档的入参（与 ORM 的 Document 解耦）。

    用独立的 data class 而不是直接传 Document 对象：调用方（如文档入库管道）
    在写库之前就需要构造 payload 并批量提交，此时还没有 ORM 实例；
    而且这样也便于在写入前做校验、打日志。
    """

    source_type: str
    title: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    # source_id 指「这条文档属于哪个上游对象」（issue id / 文档 id 等），
    # 用来按来源批量删除；为空时随机生成一个
    source_id: uuid.UUID | str | None = None


def _as_uuid(value: uuid.UUID | str | None) -> uuid.UUID:
    """把可能为 str / None 的值归一成 UUID。

    None 时生成随机 UUID 而不是报错：调用方可能只是「顺手写一条文档」，
    并不关心 source_id。字符串则做一次严格解析，格式非法就抛错（这属于调用方 bug）。
    """
    if value is None:
        return uuid.uuid4()
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


def stable_source_id(repo_id: uuid.UUID | str, source_type: str, key: str) -> uuid.UUID:
    """由「仓库 + 类型 + 业务键」生成确定性的 UUID（uuid5）。

    用途：同一条上游数据（例如同一个 issue）每次同步都应该映射到同一个
    source_id，这样再次导入时是覆盖而不是新增。
    uuid5 是「命名空间 + 名称」的哈希，天然满足这个需求；
    命名空间用 NAMESPACE_URL 只是取一个现成的、稳定的命名空间，无其他含义。
    """
    return uuid.uuid5(uuid.NAMESPACE_URL, f"devflow:{repo_id}:{source_type}:{key}")


def _escape_filter_value(value: str) -> str:
    """转义要拼进 Milvus 过滤表达式字符串字面量里的值。

    Milvus 的 filter 是一段**表达式文本**（不是参数化查询），所以值里如果
    含双引号或反斜杠，就能跳出字符串字面量、篡改整个表达式。
    这里按「先转义反斜杠、再转义双引号」的顺序处理——顺序反了会把刚插入的
    反斜杠再转义一遍。
    """
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _clip_utf8(value: str, max_bytes: int) -> str:
    """按 **UTF-8 字节数**截断字符串，并保证不切坏多字节字符。

    Milvus 的 VARCHAR max_length 是按字节算的，而 Python 的 len() 是字符数。
    一个汉字占 3 字节，若不按字节判断，中文内容会超限报错。
    截断后用 errors="ignore" 解码：末尾被切断的半个汉字会被丢掉，
    而不是产生替换字符或抛异常。
    """
    data = value.encode("utf-8")
    if len(data) <= max_bytes:
        return value
    return data[:max_bytes].decode("utf-8", errors="ignore")


def _json_dumps(value: dict[str, Any], max_chars: int = 4096) -> str:
    """把 metadata 序列化成受长度限制的 JSON 字符串。

    default=str 让 datetime / UUID 等非 JSON 原生类型退化成字符串而不是抛错——
    metadata 是"尽力保存"的信息，丢一点精度远好过整条写入失败。
    """
    text = json.dumps(value or {}, ensure_ascii=False, default=str)
    return _clip_utf8(text, max_chars)


class MilvusVectorStore:
    """Milvus 集合的薄封装：负责建集合、写入、删除、检索。

    实例在创建时就会校验/建立集合，所以拿到实例即代表「集合可用」。
    通过 _milvus_store() 的 lru_cache 保证进程内单例（连接可复用）。
    """

    def __init__(self) -> None:
        """连接 Milvus 并确保集合存在。

        dimensions 取自全局配置而非构造参数：向量维度是全局契约，
        不允许某个调用点私自指定，否则同一个集合里会混入不同维度的向量。
        """
        if MilvusClient is None or DataType is None:
            raise RuntimeError("pymilvus is not installed")
        self.collection_name = settings.milvus_collection
        self.dimensions = settings.embedding_dimensions
        # token 为空时传空串而不是 None：MilvusClient 对 None 的处理因版本而异
        self.client = MilvusClient(uri=settings.milvus_uri, token=settings.milvus_token or "")
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """集合已存在则校验维度，不存在则按固定 schema 创建。

        已存在时的维度校验是这里最关键的一步：维度不符意味着当前配置的
        embedding 模型与集合里历史数据不属于同一个向量空间，检索结果会完全
        错乱。报错信息特意给出「换一个集合名并重建索引」的指引——
        因为原地改维度在 Milvus 里是不可能的，只能新建集合重灌数据。
        """
        if self.client.has_collection(self.collection_name):
            description = self.client.describe_collection(self.collection_name)
            actual_dimensions = _collection_dimensions(description)
            if actual_dimensions != self.dimensions:
                raise RuntimeError(
                    f"Milvus collection '{self.collection_name}' has dimension {actual_dimensions}; "
                    f"the configured embedding contract requires {self.dimensions}. "
                    "Use a new collection name and rebuild the index."
                )
            return

        # auto_id=False：主键由我们自己给（= Document.id），这样才能和业务库对齐
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
        # pk 用 VARCHAR 存 UUID 字符串；64 足够容纳带连字符的 UUID（36 字符）
        schema.add_field(field_name="pk", datatype=DataType.VARCHAR, is_primary=True, max_length=64)
        # repo_id 是租户隔离键，所有查询都会带上它。没有为它单独建标量索引，
        # 因为当前数据量下过滤的开销可以忽略
        schema.add_field(field_name="repo_id", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="document_id", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="source_type", datatype=DataType.VARCHAR, max_length=80)
        schema.add_field(field_name="source_id", datatype=DataType.VARCHAR, max_length=64)
        # 以下 max_length 全部按字节计；写入时必须用 _clip_utf8 预裁剪，
        # 否则超长会整条写入失败
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=500)
        # content 上限 8192 字节（约 2700 个汉字）：这是一次检索能带回的正文体量，
        # 超过部分在入库时就被截掉了，原文完整版在 Postgres 里
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=8192)
        schema.add_field(field_name="metadata_json", datatype=DataType.VARCHAR, max_length=4096)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=self.dimensions)

        # AUTOINDEX 让 Milvus 按数据量自选最优索引；COSINE 与归一化向量配套
        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE")
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )

    def upsert_documents(self, documents: list[Document], vectors: list[list[float]]) -> None:
        """按主键 upsert（存在则覆盖，不存在则插入）。

        这里再过滤一次 should_vectorize：调用方可能已经把不该向量化的类型传进来，
        双重保险避免直连类型的数据意外进向量库。
        upsert 而非 insert 是幂等性的基础——同一条文档重复导入不会产生副本。
        """
        if len(documents) != len(vectors):
            raise ValueError("document/vector count mismatch")
        rows = []
        for doc, vector in zip(documents, vectors):
            if not should_vectorize(doc.source_type):
                continue
            # 维度不符直接拒绝，不做截断/补齐（理由见 embeddings 模块）
            if len(vector) != self.dimensions:
                raise ValueError(
                    f"document {doc.id} has {len(vector)} embedding dimensions; expected {self.dimensions}"
                )
            rows.append(
                {
                    "pk": str(doc.id),
                    "repo_id": str(doc.repo_id),
                    "document_id": str(doc.id),
                    "source_type": doc.source_type,
                    "source_id": str(doc.source_id),
                    "title": _clip_utf8(doc.title, 500),
                    "content": _clip_utf8(doc.content or "", 8192),
                    "metadata_json": _json_dumps(doc.meta or {}),
                    "embedding": vector,
                }
            )
        if rows:
            self.client.upsert(collection_name=self.collection_name, data=rows)

    def delete_documents(
        self,
        *,
        repo_id: uuid.UUID | str | None = None,
        source_types: list[str] | None = None,
        source_id: uuid.UUID | str | None = None,
        document_ids: list[uuid.UUID | str] | None = None,
    ) -> None:
        """按条件批量删除。

        多个条件是 **and** 关系（用 " and " 连接），必须全部满足才删——
        这是安全设计：比如同时给 repo_id 和 source_id，就只会删本仓库的
        这一条来源，不会误删其他仓库的数据。

        所有参数都为空的极端情况下不会执行任何删除（expressions 为空即跳过），
        避免「不传条件 = 清空集合」这种灾难。
        """
        expressions: list[str] = []
        if document_ids:
            # in [...] 列表形式；每个值单独转义，防止表达式注入
            ids = ", ".join(f'"{_escape_filter_value(str(item))}"' for item in document_ids)
            expressions.append(f"document_id in [{ids}]")
        if repo_id is not None:
            expressions.append(f'repo_id == "{_escape_filter_value(str(repo_id))}"')
        if source_types:
            values = ", ".join(f'"{_escape_filter_value(item)}"' for item in source_types)
            expressions.append(f"source_type in [{values}]")
        if source_id is not None:
            expressions.append(f'source_id == "{_escape_filter_value(str(source_id))}"')
        if expressions:
            self.client.delete(collection_name=self.collection_name, filter=" and ".join(expressions))

    def search(
        self,
        *,
        repo_id: uuid.UUID | str,
        query_embedding: list[float],
        source_type: str | None = None,
        source_types: list[str] | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """向量检索，返回统一结构的候选列表。

        过滤条件：repo_id 强隔离（必不可少），来源类型可选（source_type 精确
        匹配优先于 source_types 列表，两者只能生效一个）。

        返回字段的设计：
        - snippet：截断到 260 字符，供前端列表展示，避免响应体过大；
        - _raw_content：完整正文（仍受入库时 8192 字节限制），供重排与生成使用。
          下划线前缀表示「内部字段，不要直接给用户看」。
        - score 保留 4 位小数，够用且便于日志对比。

        metadata_json 解析失败时退化成 {} 而不是抛错：元数据只是辅助信息，
        因为一条记录格式坏了就让整个检索失败不值得。

        注意 limit 是**召回数**不是最终答案引用数：默认 20 是因为后面还有 rerank
        会再筛一遍，召回要适当放宽。
        """
        expressions = [f'repo_id == "{_escape_filter_value(str(repo_id))}"']
        if source_type:
            expressions.append(f'source_type == "{_escape_filter_value(source_type)}"')
        elif source_types:
            values = ", ".join(f'"{_escape_filter_value(item)}"' for item in source_types)
            expressions.append(f"source_type in [{values}]")
        results = self.client.search(
            collection_name=self.collection_name,
            # data 必须是二维（支持批量查询），这里只查一条
            data=[query_embedding],
            filter=" and ".join(expressions),
            limit=limit,
            # 显式列出 output_fields：不列的话默认只返回主键与距离
            output_fields=["document_id", "source_type", "source_id", "title", "content", "metadata_json"],
        )
        # 返回结构是「每个查询向量一个结果列表」，这里只有一个查询，取第 0 项
        hits = results[0] if results else []
        output: list[dict[str, Any]] = []
        for hit in hits:
            entity = hit.get("entity") or {}
            metadata_text = entity.get("metadata_json") or "{}"
            try:
                metadata = json.loads(metadata_text)
            except json.JSONDecodeError:
                metadata = {}
            output.append(
                {
                    "id": str(entity.get("document_id") or hit.get("id")),
                    "title": str(entity.get("title") or ""),
                    "source_type": str(entity.get("source_type") or ""),
                    "source_id": str(entity.get("source_id") or ""),
                    # 距离字段名是 distance，COSINE 度量下越大越相似
                    "score": round(float(hit.get("distance") or 0.0), 4),
                    "metadata": metadata,
                    "snippet": str(entity.get("content") or "")[:260],
                    "_raw_content": str(entity.get("content") or ""),
                }
            )
        return output


def _collection_dimensions(description: dict[str, Any]) -> int:
    """从 describe_collection 的返回里挖出 embedding 字段的维度。

    这里做了三层兼容（params / type_params / 顶层 dim），因为 pymilvus 不同
    大版本返回的字段结构不一样。找不到时抛 RuntimeError 而不是返回 0：
    拿不到维度就说明无法验证契约，必须让部署失败，而不是带着风险继续跑。
    """
    for schema_field in description.get("fields") or []:
        if schema_field.get("name") != "embedding":
            continue
        params = schema_field.get("params") or schema_field.get("type_params") or {}
        value = params.get("dim") or schema_field.get("dim")
        if value is not None:
            return int(value)
    raise RuntimeError("Milvus collection does not expose an embedding dimension")


@lru_cache(maxsize=1)
def _milvus_store() -> MilvusVectorStore:
    """进程内单例的存储实例。

    lru_cache(maxsize=1) 同时承担两个职责：单例缓存 + 失败后可重置
    （reset_milvus_store_cache）。

    首次创建时如果 Milvus 挂了，会把异常统一包成 MilvusUnavailableError，
    让上层可以 `except MilvusUnavailableError` 走降级路径，而不必处理
    pymilvus 的各种底层异常。

    注意：失败不会被缓存（异常会穿透 lru_cache），所以后续调用会重试连接。
    """
    if not settings.milvus_enabled:
        raise MilvusUnavailableError("Milvus is disabled; vector and hybrid RAG are unavailable")
    try:
        return MilvusVectorStore()
    except Exception as exc:  # pragma: no cover - 依赖本地基础设施。
        raise MilvusUnavailableError(f"Milvus is unavailable: {exc}") from exc


def reset_milvus_store_cache() -> None:
    """清空单例缓存，下次调用会重新连接并重新校验集合。

    典型用途：改了配置（集合名、维度、URI）后热生效，以及在测试里隔离状态。
    """
    _milvus_store.cache_clear()


def milvus_runtime_status() -> dict[str, Any]:
    """给健康检查接口用的运行状态快照。

    不抛异常，始终返回一个字典；不可用时 status="degraded" 并带上错误原因。
    同时返回 embedding contract，因为「维度/模型对不对」是排查向量问题的
    第一要素，放在一起能省一次接口调用。
    """
    try:
        store = _milvus_store()
    except MilvusUnavailableError as exc:
        return {
            "backend": "milvus",
            "status": "degraded",
            "available": False,
            "error": str(exc),
            "collection": settings.milvus_collection,
            "embedding": embedding_contract(),
        }
    return {
        "backend": "milvus",
        "status": "ready",
        "available": True,
        "error": None,
        "collection": store.collection_name,
        "embedding": embedding_contract(),
    }


def upsert_milvus_documents(documents: list[Document], vectors: list[list[float]]) -> bool:
    """把 ORM 文档写入 Milvus（过滤掉不该向量化的类型）。

    与 MilvusVectorStore.upsert_documents 的分工：本函数站在「全局配置」视角
    做前置校验（维度必须等于 settings.embedding_dimensions）与类型过滤，
    然后把真正的工作交给 store。

    返回值恒为 True（无可用文档时也返回 True）：语义是「本次同步顺利完成」，
    而不是「写了多少条」。调用方通常不需要区分。
    """
    if len(documents) != len(vectors):
        raise ValueError("document/vector count mismatch")
    # 全量校验而不是只校验将要写入的部分：维度错误说明整批数据都不可信，
    # 早失败比写一半再失败更好
    for vector in vectors:
        if len(vector) != settings.embedding_dimensions:
            raise ValueError(
                f"embedding has {len(vector)} dimensions; expected {settings.embedding_dimensions}. "
                "Refusing to truncate or pad vectors."
            )
    pairs = [(doc, vector) for doc, vector in zip(documents, vectors) if should_vectorize(doc.source_type)]
    eligible_documents = [item[0] for item in pairs]
    eligible_vectors = [item[1] for item in pairs]
    # 没有任何需要向量化的文档：直接返回，连 Milvus 连接都不用建立
    if not eligible_documents:
        return True
    store = _milvus_store()
    try:
        store.upsert_documents(eligible_documents, eligible_vectors)
    except Exception as exc:  # pragma: no cover - 依赖本地基础设施。
        raise MilvusUnavailableError("failed to write documents to Milvus") from exc
    return True


def delete_milvus_documents(
    *,
    repo_id: uuid.UUID | str | None = None,
    source_types: list[str] | None = None,
    source_id: uuid.UUID | str | None = None,
    document_ids: list[uuid.UUID | str] | None = None,
    strict: bool = True,
) -> bool:
    """从 Milvus 删除文档，strict 控制「删不掉时是否抛错」。

    为什么要有 strict：
    - 业务删除（用户主动删知识库）必须用 strict=True——删不干净会留下幽灵数据，
      以后检索还能搜到「已删除」的内容，这是严重问题；
    - 清理类操作（重建索引前的清场）用 strict=False，Milvus 本来就不可用时
      没必要阻断流程，记一条 warning 即可。

    Milvus 不可用时：strict=True 直接把 MilvusUnavailableError 抛出去，
    strict=False 返回 False。
    """
    try:
        store = _milvus_store()
    except MilvusUnavailableError:
        if strict:
            raise
        return False
    try:
        store.delete_documents(repo_id=repo_id, source_types=source_types, source_id=source_id, document_ids=document_ids)
    except Exception as exc:  # pragma: no cover - 依赖本地基础设施。
        if not strict:
            logger.warning("从 Milvus 删除文档失败：%s", exc)
            return False
        raise MilvusUnavailableError("failed to delete documents from Milvus") from exc
    return True


def search_milvus_documents(
    repo_id: str | uuid.UUID,
    query: str,
    source_type: str | None = None,
    limit: int = 20,
    query_embedding: list[float] | None = None,
    source_types: list[str] | None = None,
) -> list[dict[str, Any]]:
    """检索入口。

    query_embedding 可选：不传就现场对 query 做向量化。允许外部传入是为了
    让上层「一次向量化、多处复用」（例如同时检索向量库和做关键词打分），
    避免同一个查询被编码两次，既省一次 API 调用也保证两处用的是同一个向量。

    注意这里用的是 embed_text（文档侧语义），不是 embed_knowledge_query。
    对非对称模型（Qwen3）来说这会让精度略降，是一个已知的待优化点。

    整个函数把任何异常都包成 MilvusUnavailableError，方便上层统一降级。
    """
    store = _milvus_store()
    try:
        vector = query_embedding or embed_text(query, settings.embedding_dimensions)
        return store.search(
            repo_id=repo_id,
            query_embedding=vector,
            source_type=source_type,
            source_types=source_types,
            limit=limit,
        )
    except Exception as exc:  # pragma: no cover - 依赖本地基础设施。
        raise MilvusUnavailableError("failed to search documents in Milvus") from exc


def add_document(
    db: Session,
    repo_id: str | uuid.UUID,
    source_type: str,
    source_id: str | uuid.UUID | None,
    title: str,
    content: str,
    metadata: dict[str, Any] | None = None,
    *,
    commit: bool = True,
) -> Document:
    """写入单条文档：业务库 + （必要时）向量库。

    为什么显式禁止 code_file：源码必须通过工作区实时搜索获取。
    如果把源码也做成向量快照，代码一改快照就过期，检索到的会是已经不存在的
    代码行——这是 RAG 用在代码场景上最典型的错误，所以这里硬性拦住。

    meta 的合并顺序有讲究：调用方 metadata → 策略元数据 → embedding 元数据。
    后两者由系统生成，必须覆盖调用方可能传进来的同名错误值。

    向量化用的是 "标题\\n正文" 的拼接形式，与文档入库管道保持一致，
    保证「同一种内容在两条写入路径下的向量语义相同」。
    """
    if source_type == "code_file":
        raise ValueError("源码必须通过工作区实时搜索，不能保存为 RAG Document")
    vectorized = should_vectorize(source_type)
    # 非向量化类型不生成 embedding，vectors 为空列表，后续也不会写 Milvus
    vectors = [embed_text(f"{title}\n{content}", settings.embedding_dimensions)] if vectorized else []
    embedding_metadata = (
        {
            "embedding_provider": embedding_provider_name(),
            "embedding_model": settings.embedding_model,
            "embedding_dimensions": settings.embedding_dimensions,
        }
        if vectorized
        else {}
    )
    doc = Document(
        repo_id=_as_uuid(repo_id),
        source_type=source_type,
        source_id=_as_uuid(source_id),
        # 与 Milvus schema 的 500 字节限制对齐（这里按字符截，实际写入时还会再按字节裁）
        title=title[:500],
        content=content,
        meta={
            **(metadata or {}),
            **policy_metadata(source_type),
            **embedding_metadata,
        },
    )
    db.add(doc)
    # flush 而不 commit：先让数据库分配主键（doc.id），Milvus 的 pk 需要它
    db.flush()
    if vectorized and settings.milvus_enabled:
        upsert_milvus_documents([doc], vectors)
    if commit:
        db.commit()
        db.refresh(doc)
    return doc


def add_documents(
    db: Session,
    repo_id: str | uuid.UUID,
    payloads: list[DocumentPayload],
    *,
    commit: bool = True,
    vectors: list[list[float]] | None = None,
    embedding_info: dict[str, Any] | None = None,
    generate_embeddings: bool = True,
    sync_vector_store: bool = True,
) -> list[Document]:
    """批量写入（文档入库管道的主入口）。

    三个可选开关的用途：
    - vectors：外部已经算好的向量。给出时直接复用，不再调用 embedding API
      （文档入库管道就是这样做的，它分批算完再一次性传进来）。
    - generate_embeddings：False 表示「只写业务库、不生成向量」，
      用于回填元数据等不需要向量的场景。
    - sync_vector_store：False 表示跳过 Milvus 双写，用于批量迁移时先写库后统一同步。

    参数校验的三道关卡：
    1. 空内容 payload 直接丢弃（向量化空串没有意义，还会污染检索）；
    2. code_file 一律拒绝（同上）；
    3. vectors 条数必须与**过滤后**的 payload 数一致——注意是过滤后，
      因为调用方通常是先按原列表算向量再传进来，这里要求它自己先对齐。

    「RAG 文档必须有向量才能同步」这个检查（函数中段的 sync_vector_store 分支）很关键：
    如果某个该向量化的文档最终 embedding 仍是 None 却要写 Milvus，
    就会写进一条没有向量的记录，或者直接崩在 Milvus 侧。
    这里提前用明确的错误信息拦住。
    """
    # 过滤空内容；strip 判断而不是 len，避免只含空白字符的"假内容"
    clean_payloads = [payload for payload in payloads if payload.content.strip()]
    if any(payload.source_type == "code_file" for payload in clean_payloads):
        raise ValueError("源码必须通过工作区实时搜索，不能保存为 RAG Document")
    if vectors is not None and len(vectors) != len(clean_payloads):
        raise ValueError("embedding count does not match document payload count")

    # 先占位成 None，之后只给「需要向量化」的下标填值
    embeddings: list[list[float] | None] = [None] * len(clean_payloads)
    # 只对这些下标求向量；直连类型（team_member 等）保持 None
    vector_indexes = [
        index for index, payload in enumerate(clean_payloads) if should_vectorize(payload.source_type)
    ]
    if vectors is not None:
        # 外部传入：按下标逐一取，非向量化位置即使传了也会被忽略
        for index in vector_indexes:
            embeddings[index] = vectors[index]
    elif vector_indexes and generate_embeddings:
        # 内部生成：只对需要向量化的文本批量编码，顺序与 vector_indexes 一致
        generated = embed_texts(
            [f"{clean_payloads[index].title}\n{clean_payloads[index].content}" for index in vector_indexes],
            settings.embedding_dimensions,
        )
        if len(generated) != len(vector_indexes):
            raise ValueError("embedding count does not match vectorized document payload count")
        for index, embedding in zip(vector_indexes, generated):
            embeddings[index] = embedding
    # 要同步向量库却没有向量 → 提前失败，避免把残缺数据写进去
    if sync_vector_store and any(embeddings[index] is None for index in vector_indexes):
        raise ValueError("RAG documents require embeddings before Milvus synchronization")

    docs: list[Document] = []
    for payload, embedding in zip(clean_payloads, embeddings):
        # 用「有没有向量」反推是否向量化，而不是再查一次 should_vectorize，
        # 保证 meta 里记录的 embedding 信息与实际情况一致
        vectorized = embedding is not None
        embedding_metadata = (
            {
                "embedding_provider": embedding_provider_name(),
                "embedding_model": settings.embedding_model,
                "embedding_dimensions": settings.embedding_dimensions,
                # embedding_info 放最后：调用方显式传入的信息优先
                **(embedding_info or {}),
            }
            if vectorized
            else {}
        )
        docs.append(
            Document(
                repo_id=_as_uuid(repo_id),
                source_type=payload.source_type,
                source_id=_as_uuid(payload.source_id),
                title=payload.title[:500],
                content=payload.content,
                meta={
                    **(payload.metadata or {}),
                    **policy_metadata(payload.source_type),
                    **embedding_metadata,
                },
            )
        )
    if not docs:
        return []
    db.add_all(docs)
    # flush 让 ORM 分配 id，供 Milvus 的 pk 使用；此时尚未 commit
    db.flush()
    if sync_vector_store and settings.milvus_enabled:
        vector_documents = [docs[index] for index in vector_indexes]
        vector_values = [embeddings[index] for index in vector_indexes]
        # 这里再过滤一次 None 只是为了让类型检查器满意（上面已经保证不为 None）
        upsert_milvus_documents(
            vector_documents,
            [vector for vector in vector_values if vector is not None],
        )
    if commit:
        db.commit()
        for doc in docs:
            db.refresh(doc)
    return docs


def replace_documents(
    db: Session,
    repo_id: str | uuid.UUID,
    source_types: list[str],
    payloads: list[DocumentPayload],
    *,
    commit: bool = True,
    vectors: list[list[float]] | None = None,
    embedding_info: dict[str, Any] | None = None,
    generate_embeddings: bool = True,
    sync_vector_store: bool = True,
) -> list[Document]:
    """按来源类型整体替换：先删该仓库下这些类型的全部文档，再写入新的。

    这是「重新同步某个数据源」的标准做法（例如重新拉一遍所有 issue）。
    先删后加的顺序很重要：如果先加后删，新数据会被一起删掉。

    synchronize_session=False：跳过 ORM 的会话同步（不逐条标记对象状态），
    这在大批量删除时能快很多，代价是会话里可能残留已删除对象的引用——
    本函数之后立刻写入新对象，不依赖旧对象，所以安全。

    注意业务库删除和 Milvus 删除是**两次独立操作**，不在同一个事务里
    （Milvus 不支持跨库事务）。极端情况下可能出现业务库已删、Milvus 删除失败，
    此时向量库会残留数据。这是当前架构的已知限制。
    """
    repo_uuid = _as_uuid(repo_id)
    db.query(Document).filter(Document.repo_id == repo_uuid, Document.source_type.in_(source_types)).delete(
        synchronize_session=False
    )
    # flush 让删除先落到数据库，避免与新插入的数据在会话里冲突
    db.flush()
    if sync_vector_store and settings.milvus_enabled:
        delete_milvus_documents(repo_id=repo_uuid, source_types=source_types)
    docs = add_documents(
        db,
        repo_uuid,
        payloads,
        # 这里必须 commit=False：由本函数统一提交，保证「删 + 加」是一个事务
        commit=False,
        vectors=vectors,
        embedding_info=embedding_info,
        generate_embeddings=generate_embeddings,
        sync_vector_store=sync_vector_store,
    )
    if commit:
        db.commit()
    return docs


def replace_documents_by_metadata(
    db: Session,
    repo_id: str | uuid.UUID,
    source_type: str,
    metadata_filters: dict[str, Any],
    payloads: list[DocumentPayload],
    *,
    commit: bool = True,
) -> list[Document]:
    """按 metadata 精确匹配替换（比 replace_documents 粒度更细）。

    场景：同一个 source_type 下有多批数据（例如不同 PR 的评论），
    只想替换某一批，就靠 metadata_filters 圈定范围。

    实现上是在 Python 里过滤而不是下推到 SQL：因为 meta 是 JSON 字段，
    跨数据库（PG / SQLite）做 JSON 查询的写法不一致。代价是全表扫描，
    在当前数据量下可以接受；数据量上去后应该改成 JSONB 索引查询。

    判断条件是 `==` 严格相等，所以 filters 的值类型必须与 meta 里存的完全一致
    （比如 int 和 str 不相等）。

    注意最后调用 add_documents 时没有传 vectors，所以会走「内部生成向量」路径。
    """
    repo_uuid = _as_uuid(repo_id)
    rows = db.query(Document).filter(Document.repo_id == repo_uuid, Document.source_type == source_type).all()
    document_ids = [
        row.id
        for row in rows
        # all(...)：所有 filter 条件都满足才算命中（and 语义）
        if all((row.meta or {}).get(key) == value for key, value in metadata_filters.items())
    ]
    if document_ids:
        db.query(Document).filter(Document.id.in_(document_ids)).delete(synchronize_session=False)
        # 只有该类型确实需要向量化时才去删 Milvus，避免无谓的连接
        if should_vectorize(source_type) and settings.milvus_enabled:
            delete_milvus_documents(repo_id=repo_uuid, document_ids=document_ids)
    db.flush()
    docs = add_documents(db, repo_uuid, payloads, commit=False)
    if commit:
        db.commit()
    return docs


def delete_document_group(
    db: Session,
    repo_id: str | uuid.UUID,
    source_id: str | uuid.UUID,
    source_type: str | None = None,
    *,
    commit: bool = True,
) -> int:
    """删除某个上游对象产出的全部 RAG 文档（例如「删掉这份知识文件的所有分块」）。

    这是文档入库管道「重传覆盖」的一半：先按 source_id 删干净，再写新的。

    实现细节：
    - source_type 可选，给了就同时按类型过滤，避免不同来源共用同一个 source_id
      时误删（理论上不该发生，但多一层保险）。
    - 先把 id 全查出来再 delete：因为 delete 之后就拿不到 id 了，
      而 Milvus 需要用 document_ids 精确删除。
    - Milvus 的三个条件（repo_id + source_id + document_ids）都传，是 and 关系，
      多重约束确保只删目标数据。
    - commit=False 时由调用方控制事务边界，这样才能与后续的写入组成一个事务。

    返回被删除的业务库记录数。
    """
    repo_uuid = _as_uuid(repo_id)
    source_uuid = _as_uuid(source_id)
    query = db.query(Document).filter(Document.repo_id == repo_uuid, Document.source_id == source_uuid)
    if source_type:
        query = query.filter(Document.source_type == source_type)
    document_ids = [row.id for row in query.all()]
    count = query.delete(synchronize_session=False)
    # 若指定了类型且该类型不向量化，就不必碰 Milvus；未指定类型时保守地都删
    if settings.milvus_enabled and (source_type is None or should_vectorize(source_type)):
        delete_milvus_documents(repo_id=repo_uuid, source_id=source_uuid, document_ids=document_ids)
    if commit:
        db.commit()
    return count
