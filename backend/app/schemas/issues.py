"""Issue 相关响应模型（业务对象：同步下来的 issue，以及 AI 对它的分析结论）。

只读模型：issue 内容以 GitHub 为准，本地不允许编辑，避免双向同步冲突。
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

from app.schemas.analysis import IssueAnalysisOutput


class IssueResponse(BaseModel):
    """Issue 详情（业务对象：一条 issue 的展示字段，直接映射 Issue ORM 模型）。"""
    id: UUID
    repo_id: UUID
    number: int  # 仓库内编号（#123），列表页展示用
    title: str
    body: str | None = None  # 正文 Markdown；历史数据可能缺失
    state: str  # open / closed
    labels: list[str] = []  # 标签（JSON 列原样返回）
    author: str | None = None  # 作者登录名
    assignees: list[str] = []  # 指派人列表
    created_at: datetime | None = None  # 来自 GitHub，可能为空
    updated_at: datetime | None = None

    model_config = {"from_attributes": True}


class IssueAnalysisResponse(BaseModel):
    """Issue 分析结果（业务对象：一次分析请求的产出 + 记录 ID）。

    analysis_id 用 UUID | str 联合类型，兼容早期以字符串保存的分析记录。
    """
    analysis_id: UUID | str
    result: IssueAnalysisOutput  # 见 schemas.analysis 中的严格结构定义
