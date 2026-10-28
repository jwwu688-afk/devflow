"""RAG（检索增强生成）相关模型（业务对象：知识库配置、文档与分块、检索与问答结果）。

这一组模型覆盖 RAG 的完整链路，读代码时建议按链路顺序看：
配置（KnowledgeBaseConfigUpdate）→ 数据（KnowledgeDocumentResponse / KnowledgeChunkResponse）
→ 检索（RAGSearchResponse / RetrievalTest*）→ 生成（RAGAnswerResponse）→ 体检（RAGStatusResponse）。

贯穿全模块的一个约定：所有配置更新模型的字段都是「可选」，
None 表示「本次不修改」。服务端据此做部分更新（PATCH），
避免前端为了改一个 top_k 而回传整份配置、把其他字段意外覆盖成默认值。
"""

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field


class KnowledgeBaseConfigUpdate(BaseModel):
    """知识库配置的部分更新请求（业务对象：一次调参）。

    每个字段都带范围约束，原因不只是「防手抖」：
    embedding_dimensions 必须与向量库集合维度一致，chunk_size 过大则单块超模型上下文，
    这些值一旦越界会让整条检索链路静默失效（能查但查不准），比直接报错更难排查。
    """
    embedding_provider: str | None = None  # 向量化方式；切换后需全量重建索引
    embedding_model: str | None = None  # 模型名；与维度绑定
    # 64~4096：覆盖当前主流 embedding 模型的实际输出维度区间，越界几乎必然是填错
    embedding_dimensions: int | None = Field(default=None, ge=64, le=4096)
    # 检索方式用枚举收口：vector=纯向量，fulltext=关键词，hybrid=两者融合
    retrieval_method: Literal["vector", "fulltext", "hybrid"] | None = None
    rerank_enabled: bool | None = None  # 精排开关（更准但更慢更贵）
    rerank_provider: str | None = None
    rerank_model: str | None = None
    # 1~20：上限 20 是为了控制进入 prompt 的 token 量，再多收益递减
    top_k: int | None = Field(default=None, ge=1, le=20)
    score_threshold_enabled: bool | None = None
    score_threshold: float | None = Field(default=None, ge=0, le=1)  # 相似度阈值，0~1
    # 200~4000 字符：太小的块语义不完整，太大的块会稀释向量并挤占上下文
    chunk_size: int | None = Field(default=None, ge=200, le=4000)
    # 0~1000：重叠不能超过块大小，具体上限由服务层再校验一次
    chunk_overlap: int | None = Field(default=None, ge=0, le=1000)


class KnowledgeBaseResponse(BaseModel):
    """知识库概览（业务对象：一个仓库的知识库「卡片」，含统计与运行状态）。

    注意这里的 id/name/full_name 其实来自 repository：知识库与仓库是 1:1 的，
    没有独立实体，复用它可以让前端少一次查询。
    """
    id: UUID  # 即 repo_id
    name: str  # 仓库名
    full_name: str  # owner/name
    description: str = ""
    provider: str = "github"
    document_count: int = 0  # 已入库文档（源文件）数
    chunk_count: int = 0  # 文本块总数，直接影响检索粒度
    character_count: int = 0  # 总字符数，用于估算存储与向量成本
    config: dict[str, Any] = Field(default_factory=dict)  # 当前生效的检索/切分配置
    runtime: dict[str, Any] = Field(default_factory=dict)  # 运行态信息（向量库连通性、实际使用的模型）
    created_at: datetime | None = None
    updated_at: datetime | None = None


class KnowledgeDocumentResponse(BaseModel):
    """知识库文档（业务对象：一个已上传/已抓取的源文件及其处理状态）。"""
    id: UUID
    repo_id: UUID
    name: str  # 文件名
    source_type: str  # upload_file / url / repo_path ...
    status: str  # queued / parsing / indexed / failed
    content_type: str | None = None  # MIME 类型
    file_size: int = 0  # 字节数
    character_count: int = 0  # 解析出的字符数
    chunk_count: int = 0  # 切块数
    error_message: str | None = None  # 处理失败原因，直接在列表里展示
    created_at: datetime | None = None
    updated_at: datetime | None = None
    completed_at: datetime | None = None  # 完成时间，用于算处理耗时


class KnowledgeChunkResponse(BaseModel):
    """文档分块（业务对象：切分后的一块文本，检索的最小单位）。

    enabled 允许单独禁用某块：某些块可能是目录、版权声明等噪音，
    关掉它比整份文档不索引更精细。
    """
    id: UUID
    position: int  # 块在文档中的序号，决定展示与拼接顺序
    title: str  # 块标题（通常由所在小节/函数名生成）
    content: str  # 块正文
    character_count: int  # 字符数，用于估算 token
    enabled: bool = True  # 是否参与检索；默认开启


class RAGQueryRequest(BaseModel):
    """问答式检索请求（业务对象：用户直接抛给知识库的一个问题）。

    与 RetrievalTestRequest 的区别：这里的问题会再交给 LLM 生成答案，
    因此长度上限更宽松（4000），而 top_k 上限更严（10）以控制生成时的上下文成本。
    """
    # 1~4000 字符：空问题无意义，过长问题多半是误把整篇文档粘进来了
    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=10)  # 送入生成阶段的证据条数
    # 限定来源类型（如只看 upload_file）；用 max_length 限制元素个数，防止超长列表拖慢查询
    source_types: list[str] = Field(default_factory=list, max_length=20)
    metadata_filters: dict[str, Any] = Field(default_factory=dict)  # 元数据过滤条件（标签、路径等）


class RAGSourceResponse(BaseModel):
    """一条引用来源（业务对象：答案所依据的具体文本块）。

    citation 是给用户看的角标编号（[1][2]），与 answer 文本中的标记一一对应；
    document_id/source_id 用字符串而非 UUID，是因为来源可能不止数据库文档（含 Milvus 主键）。
    """
    citation: int  # 引用序号，从 1 开始
    document_id: str  # 文档块 ID
    source_id: str  # 上游来源对象 ID
    source_type: str  # 来源类型
    title: str  # 展示标题（文件名/符号名）
    snippet: str  # 引用摘录
    score: float  # 相关性分数（融合/精排后的最终分）
    metadata: dict[str, Any] = Field(default_factory=dict)  # 行号、路径等
    retrieval: dict[str, Any] = Field(default_factory=dict)  # 检索过程细节（向量分/关键词分/精排分），用于调参


class RAGSearchResponse(BaseModel):
    """纯检索结果（业务对象：只检索不生成，供调试与知识库预览）。"""
    query: str
    results: list[RAGSourceResponse]
    result_count: int
    took_ms: int  # 耗时毫秒；检索调试页要展示，便于判断是否超预算
    retrieval_config: dict[str, Any] = Field(default_factory=dict)  # 本次实际生效的配置快照（含默认值填充结果）


class RetrievalTestRequest(BaseModel):
    """检索调试请求（业务对象：知识库页「试一次检索」的表单）。

    所有可调参数都是可选的 None：不传就沿用知识库当前配置，
    这样可以先「按现状跑一次」建立基线，再逐项改参数做对比。
    """
    # 调试场景问题通常较短，限制 250 字符，避免把整段文档当 query 测出无意义的结果
    query: str = Field(min_length=1, max_length=250)
    top_k: int | None = Field(default=None, ge=1, le=20)  # 调试允许比问答更大的 top_k
    retrieval_method: Literal["vector", "fulltext", "hybrid"] | None = None  # 临时切换检索方式
    rerank_enabled: bool | None = None  # 临时开关精排
    score_threshold_enabled: bool | None = None
    score_threshold: float | None = Field(default=None, ge=0, le=1)
    metadata_filters: dict[str, Any] = Field(default_factory=dict)


class RetrievalTestResponse(BaseModel):
    """检索调试结果（业务对象：一次试验的完整留档，对应 retrieval_test_runs 表）。"""
    id: UUID  # 记录 ID，便于把结果链接分享给同事
    query: str
    results: list[RAGSourceResponse]
    result_count: int
    took_ms: int
    retrieval_config: dict[str, Any] = Field(default_factory=dict)  # 本次参数快照
    created_at: datetime | None = None


class RetrievalTestHistoryResponse(BaseModel):
    """检索历史条目（业务对象：调试记录列表中的一行，不含结果详情以省带宽）。"""
    id: UUID
    query: str
    result_count: int  # 命中数，快速看出「参数改动是否提高了召回」
    took_ms: int
    created_at: datetime | None = None


class ModelCatalogResponse(BaseModel):
    """可用模型清单（业务对象：配置页下拉框的选项来源）。

    把 embedding / rerank / 生成模型分三组返回：
    它们的可选集合与计费方式完全不同，混在一个列表里会让前端难以区分。
    """
    embedding_models: list[dict[str, Any]]  # 向量模型（含维度等元信息）
    rerank_models: list[dict[str, Any]]  # 精排模型
    deepseek: dict[str, Any]  # 生成模型信息（当前只接了 DeepSeek）


class RAGAnswerResponse(BaseModel):
    """RAG 问答结果（业务对象：一个问题 + 答案 + 引用 + 生成方式）。

    generation_mode 是本模型最关键的字段，它暴露了「降级链」走到了哪一步：
    llm=正常由大模型生成；extractive=模型不可用时退化为抽取式拼接；
    no_evidence=没检索到证据，直接如实告知，而不是让模型硬编一个答案。
    前端据此可以给「非 llm」的结果加提示标记，避免用户误以为是完整生成。
    """
    question: str  # 原始问题
    answer: str  # 答案正文
    sources: list[RAGSourceResponse]  # 引用来源
    retrieval_count: int  # 检索到的证据条数
    generation_mode: Literal["llm", "extractive", "no_evidence"]  # 生成方式（降级链）
    model: str | None = None  # 实际使用的模型；抽取式/无证据时为空
    answer_gate: dict[str, Any] = Field(default_factory=dict)  # 答案门禁检查结果（是否引用了证据、是否越界）
    retrieval_config: dict[str, Any] = Field(default_factory=dict)  # 检索参数快照
    took_ms: int  # 端到端耗时


class RAGStatusResponse(BaseModel):
    """RAG 系统体检（业务对象：知识库设置页的状态面板）。

    分块返回（embedding / vector_store / generation）而不是平铺：
    三个子系统可能各自异常（模型加载失败、Milvus 连不上、LLM Key 缺失），
    分开才能定位到底哪一环坏了。
    """
    repo_id: UUID
    document_count: int  # 文档数
    source_count: int  # 来源条目数
    source_types: dict[str, int] = Field(default_factory=dict)  # 按来源类型统计，如 {"upload_file": 3}
    embedding: dict[str, Any] = Field(default_factory=dict)  # 向量化组件状态
    vector_store: dict[str, Any] = Field(default_factory=dict)  # 向量库状态（集合、条数、连通性）
    generation: dict[str, Any] = Field(default_factory=dict)  # 生成模型状态（是否配置 Key、模型名）
    supported_files: list[str] = Field(default_factory=list)  # 支持上传的扩展名，供上传控件过滤


class RAGReindexResponse(BaseModel):
    """重建索引结果（业务对象：一次全量重建的回执）。"""
    repo_id: UUID
    indexed_documents: int  # 成功重建的文档数
    embedding_provider: str  # 本次使用的向量化方式
    embedding_model: str  # 本次使用的模型；换模型后必须重建，故一并回显
