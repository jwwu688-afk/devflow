"""知识库与记忆相关的响应/请求模型（业务对象：项目知识、团队信息、知识图谱、召回与记忆候选）。

本模块是「知识管理页」的数据契约，涵盖四类东西：
- 人工录入的知识：备忘、团队成员；
- 由文档切块而来的知识条目与上传回执；
- 知识图谱（节点 + 边）；
- 可观测性数据：召回事件、记忆状态、待确认的记忆候选。

注意 metadata 字段在这里直接用原名（而不是 ORM 里的 meta 别名）：
这些模型多数不是直接从 ORM 转换（from_attributes），而是服务层手工组装的 DTO，
因此没有 SQLAlchemy 保留字冲突的问题，也就能用更直白的对外字段名。
"""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class MemoryNoteRequest(BaseModel):
    """手动添加一条知识备忘（业务对象：用户随手记的项目背景）。

    两个字段都给默认空串，是为了让前端可以先提交再补内容（草稿式录入）。
    """
    title: str = ""
    content: str = ""


class TeamMemberRequest(BaseModel):
    """新增/更新团队成员（业务对象：人工维护的人员画像）。

    这份画像用于「建议负责人」：模型根据 role / strengths / tech_stack 判断谁更合适。
    tech_stack 用别名是为了兼容前端 camelCase，同时后端保持 snake_case。
    """
    name: str
    role: str = ""  # 角色（后端/前端/测试...）
    strengths: str = ""  # 擅长领域，自然语言描述
    tech_stack: str = Field(default="", alias="techStack")  # 技术栈；前端传 techStack

    model_config = {"populate_by_name": True}  # 允许同时用别名与字段名传参


class TeamMemberResponse(BaseModel):
    """团队成员详情（业务对象：一条人员画像记录）。"""
    id: UUID
    name: str
    role: str = ""
    strengths: str = ""
    tech_stack: str = Field(default="", alias="techStack")
    created_at: datetime | None = None

    model_config = {"populate_by_name": True}


class KnowledgeItemResponse(BaseModel):
    """知识条目（业务对象：一次上传/一次抓取聚合出的「一份知识」，由若干块组成）。

    document_ids 指向切分后的多个 Document：一个文件会被切成多块，
    但用户视角看到的是「一个条目」，因此这里把块 ID 聚合到一起并给出预览与块数。
    """
    id: UUID
    document_ids: list[UUID]  # 该条目对应的所有文本块 ID
    source_type: str  # 来源类型：upload_file / url / repo_path ...
    title: str  # 条目标题（通常是文件名）
    content_preview: str  # 正文预览，列表页只渲染这段，避免拉全量正文
    metadata: dict[str, Any] = Field(default_factory=dict)  # 附加信息（大小、页数、标签等）
    chunk_count: int  # 切块数量，侧面反映内容体量
    created_at: datetime | None = None


class KnowledgeUploadResponse(BaseModel):
    """上传回执（业务对象：一次文件上传的处理结果）。

    duplicate 单独给一个字段而不是用错误码表达：
    重复上传不是错误，而是「命中已有内容」，前端应提示「已存在，已跳过」而非报错。
    """
    id: UUID
    filename: str
    chunks: int  # 切出的块数；0 表示解析失败或内容为空
    indexed: bool  # 是否已完成向量入库
    duplicate: bool = False  # 是否因 checksum 重复而复用已有条目


class KnowledgeGraphNodeResponse(BaseModel):
    """知识图谱节点（业务对象：图上的一个概念/文档/符号）。

    id 用 str 而非 UUID：节点可能来自不同实体（文档块、符号、issue），
    甚至包括虚拟节点（如「项目」本身），统一成字符串最省事。
    """
    id: str
    type: str  # 节点类型，决定前端图标与配色
    title: str  # 主标题
    subtitle: str = ""  # 副标题（路径、编号等）
    metadata: dict[str, Any] = Field(default_factory=dict)
    highlighted: bool = False  # 是否高亮（当前搜索命中的节点）


class KnowledgeGraphEdgeResponse(BaseModel):
    """知识图谱边（业务对象：两个节点之间的一条关系）。"""
    id: str
    from_type: str  # 起点类型
    from_id: str  # 起点 ID
    to_type: str  # 终点类型
    to_id: str  # 终点 ID
    relation: str  # 关系语义：depends_on / implements / references ...
    confidence: float = 1.0  # 置信度；模型推断的边通常 <1，前端可弱化显示
    source: str = "explicit"  # 边来源：explicit=显式声明，其余为推断
    metadata: dict[str, Any] = Field(default_factory=dict)


class KnowledgeGraphResponse(BaseModel):
    """知识图谱查询结果（业务对象：以某个节点为中心、限定深度的子图）。"""
    repo_id: UUID
    center_type: str | None = None  # 中心节点类型；为空表示是全局概览图
    center_id: str | None = None  # 中心节点 ID
    depth: int = 1  # 展开层数；默认为 1，层数越大节点数指数增长，故由前端显式指定
    nodes: list[KnowledgeGraphNodeResponse]
    edges: list[KnowledgeGraphEdgeResponse]
    stats: dict[str, Any] = Field(default_factory=dict)  # 统计信息（总节点数、被裁剪的节点数等）


class KnowledgeGraphRebuildResponse(BaseModel):
    """图谱重建结果（业务对象：一次全量重建的产出规模）。"""
    repo_id: UUID
    edges: int  # 重建后的边数
    nodes: int  # 重建后的节点数


class RecallEventResponse(BaseModel):
    """召回事件详情（业务对象：一次检索调用的可观测记录）。

    与 RecallEvent ORM 模型的差异：results_json / citations_json / meta
    在这里被重命名为 results / citations / metadata（服务层做映射），
    对外语义更清晰，也避免把「_json」这种存储细节暴露给前端。
    """
    id: UUID
    repo_id: UUID | None = None
    conversation_id: UUID | None = None
    tool_name: str  # 调用的检索工具
    query: str = ""  # 实际查询语句
    scope: str = "project"  # 检索范围
    mode: str = "hybrid"  # 检索模式
    status: str = "success"
    result_count: int = 0  # 命中条数；为 0 是排查「答不准」的第一线索
    results: list[dict[str, Any]] = Field(default_factory=list)  # 命中结果与分数
    citations: list[dict[str, Any]] = Field(default_factory=list)  # 最终被引用的条目
    metadata: dict[str, Any] = Field(default_factory=dict)  # 耗时等运行信息
    created_at: datetime | None = None


class MemorySearchResponse(BaseModel):
    """记忆检索结果（业务对象：一次「翻记忆」的返回）。"""
    query: str  # 回显查询，便于用户确认自己搜的是什么
    results: list[dict[str, Any]]  # 命中的记忆条目（来源多样，保持宽松结构）
    result_count: int  # 命中数，与 len(results) 一致但在前端判断空态时更方便


class MemoryStatusResponse(BaseModel):
    """记忆系统状态总览（业务对象：知识库概览页的一组统计卡片）。

    每个字段对应一张表/一个子系统的统计，分成独立 dict 而不是平铺数字，
    是为了让各子系统能各自扩展统计项而不影响其他卡片。
    """
    repo_id: UUID
    index: dict[str, Any] = Field(default_factory=dict)  # 索引状态（来自 project_index_states）
    documents: dict[str, Any] = Field(default_factory=dict)  # 文档/切块统计
    evidence: dict[str, Any] = Field(default_factory=dict)  # 证据条目统计
    recall: dict[str, Any] = Field(default_factory=dict)  # 召回情况统计（成功率、平均条数）
    candidates: dict[str, Any] = Field(default_factory=dict)  # 记忆候选统计（待确认数量等）
    vector: dict[str, Any] = Field(default_factory=dict)  # 向量库状态（集合是否存在、条数）


class MemoryCandidateResponse(BaseModel):
    """记忆候选详情（业务对象：一条等待确认的记忆条目）。"""
    id: UUID
    repo_id: UUID | None = None
    conversation_id: UUID | None = None
    session_id: UUID | None = None  # 来自哪个封存片段
    kind: str  # fact / decision / question / task / preference
    title: str
    content: str
    status: str  # pending / accepted / rejected
    source: str  # 产生方式
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None
    reviewed_at: datetime | None = None  # 审核时间；未审核为空


class MemoryCandidateActionResponse(BaseModel):
    """记忆候选的确认/驳回结果（业务对象：审核动作的回执）。

    document_id 仅在「确认」时可能有值：确认后该记忆会被写入知识库成为一个文档块，
    前端可据此跳转到对应知识条目。
    """
    id: UUID
    status: str  # 处理后的状态：accepted / rejected
    document_id: UUID | None = None  # 落库生成的文档 ID；驳回时为 None
