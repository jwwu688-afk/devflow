"""统一搜索的出入参模型（业务对象：跨 issue/PR/CI/文档的一次带过滤条件的检索）。

过滤器全部设为可选且互相「与」：调用方可以只给 query，也可以叠加标签、路径、模块、
状态、时间等维度收窄范围。这样一套接口能同时服务「自由搜索」和「条件筛选」两种 UI。
"""

from uuid import UUID

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """搜索请求（业务对象：一次检索的查询词 + 各类过滤条件）。"""
    repo_id: UUID  # 必填：所有检索都被仓库边界限制，防止跨项目串数据
    query: str  # 查询词；可为空，此时等价于「纯条件筛选」
    source_type: str | None = None  # 限定来源类型（code/document/issue/pr/ci），为空表示全类型
    label: str | None = None  # 按标签过滤（issue/PR 的 labels）
    path: str | None = None  # 按文件路径过滤（代码类结果）
    module: str | None = None  # 按模块过滤（目录/包名）——比 path 更粗的粒度
    state: str | None = None  # 按状态过滤（open/closed/success 等）
    # 日期用 str 而不是 date：前端可能传 "2026-10" 这类非完整日期或相对区间，
    # 由检索层自行解析，schema 层不强行收紧格式以免误杀合法输入
    start_date: str | None = None
    end_date: str | None = None
    # 1~20 条：既控制上下文长度，也避免一次返回过多结果让用户无从选择
    limit: int = Field(default=5, ge=1, le=20)


class SearchResponse(BaseModel):
    """搜索结果（业务对象：一组异构的命中项）。

    results 保持 list[dict]：不同 source_type 的字段差异很大（代码有行号、PR 有分支、
    CI 有日志），强类型化需要一层带 tag 的联合体，收益不抵复杂度。
    """
    results: list[dict]
