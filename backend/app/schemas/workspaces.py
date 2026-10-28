"""工作区与多仓报表模型（业务对象：把多个仓库组合成一个项目视角，并据此出跨仓周报）。

注意请求与响应里 repo_ids 的类型不一致：
- 请求用 list[UUID]：入口处就做强校验，非法 ID 直接 422；
- 响应用 list[str]：数据库里存的是 JSON 字符串数组（见 Workspace.repo_ids），
  保持原样返回可避免每次读都做一次 UUID 转换，失败解析反而会让整个列表 500。
"""

from datetime import date, datetime
from uuid import UUID

from pydantic import BaseModel, Field


class WorkspaceCreateRequest(BaseModel):
    """新建工作区请求（业务对象：用户圈定的一组仓库）。"""
    name: str  # 工作区名，必填
    description: str | None = None  # 用途说明
    repo_ids: list[UUID] = Field(default_factory=list)  # 成员仓库；允许为空以便先建后加


class WorkspaceResponse(BaseModel):
    """工作区详情（业务对象：一个多仓组合的配置）。"""
    id: UUID
    name: str
    description: str | None = None
    repo_ids: list[str] = Field(default_factory=list)  # 与 DB 中 JSON 列一致，保持字符串
    created_at: datetime
    updated_at: datetime | None = None

    model_config = {"from_attributes": True}


class MultiRepoReportRequest(BaseModel):
    """多仓报表请求（业务对象：按工作区或直接指定仓库生成跨仓周报）。

    workspace_id 与 repo_ids 互为兜底：
    给了工作区就用工作区成员，没给工作区则用显式传入的 repo_ids。
    """
    workspace_id: UUID | None = None  # 用已保存的工作区
    repo_ids: list[UUID] = Field(default_factory=list)  # 临时指定仓库，覆盖工作区成员
    start_date: date  # 统计区间起（含）
    end_date: date  # 统计区间止（含）


class MultiRepoReportResponse(BaseModel):
    """多仓报表结果（业务对象：一份可读报告 + 可计算的指标 + 各仓摘要）。

    同时给 Markdown（给人看）和 metrics（给图表/告警用），
    是因为同一份数据要同时满足「贴到周会文档」和「喂给可视化」两种用途。
    """
    report_markdown: str  # 渲染好的报告正文
    metrics: dict  # 汇总指标（总量、按类型分布等）
    repo_summaries: list[dict]  # 每个仓库的分项摘要
