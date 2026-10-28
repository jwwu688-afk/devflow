"""代码图谱搜索的响应模型（业务对象：按关键词/语义查到的符号、关系与文档）。

这个接口一次返回三份结果，是刻意的：用户搜一个函数名，通常同时想知道
「它定义在哪（symbols）」「它调用了谁（relations）」「有没有相关文档（documents）」，
合并返回可省掉三次往返，也避免前端出现三段不同步的加载状态。
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class CodeSymbolResponse(BaseModel):
    """代码符号（业务对象：一个函数/类/方法的定义位置）。"""
    id: UUID
    repo_id: UUID
    path: str  # 文件路径
    name: str  # 符号名
    kind: str  # function / class / method ...
    language: str
    start_line: int | None = None  # 起始行；解析不精确时可能为空
    end_line: int | None = None
    branch: str | None = None  # 所属分支，用于区分不同版本的同一符号
    commit_sha: str | None = None  # 所属提交
    pr_id: UUID | None = None  # PR 场景下的临时符号
    pr_number: int | None = None
    # ORM 侧属性名是 meta（metadata 被 SQLAlchemy 占用），对外 JSON 暴露为 metadata
    metadata: dict = Field(default_factory=dict, alias="meta")
    created_at: datetime

    model_config = {"from_attributes": True, "populate_by_name": True}


class CodeGraphSearchResponse(BaseModel):
    """代码图谱搜索结果（业务对象：一次搜索的符号 + 关系 + 文档三元组）。"""
    query: str  # 原始查询词，回显给前端做「结果对不上」的排查
    symbols: list[CodeSymbolResponse]  # 命中的符号（强类型，前端要渲染代码跳转）
    relations: list[dict]  # 符号间关系边；结构随语言解析器不同，故用 dict
    documents: list[dict]  # 相关文档片段（含分数、来源），同样保持宽松结构
