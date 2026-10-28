"""PR 相关响应模型（业务对象：PR 详情及其文件变更、评审评论、AI 评审结论）。

PullRequestResponse 用嵌套模型带出 files / review_comments，
前端打开 PR 详情页时一次拿全，无需再发两个请求；
代价是响应体较大，因此列表接口通常不应复用这个模型（见各列表接口的简化输出）。
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

from app.schemas.analysis import PRReviewOutput


class PRFileResponse(BaseModel):
    """PR 中的单个变更文件（业务对象：一处文件改动 + 其 diff）。"""
    id: UUID
    filename: str  # 仓库内相对路径
    status: str  # added / modified / removed / renamed
    additions: int  # 新增行数
    deletions: int  # 删除行数
    patch: str | None = None  # 统一 diff；二进制文件或超大文件可能为空

    model_config = {"from_attributes": True}


class PRReviewCommentResponse(BaseModel):
    """PR 上的一条评审评论（业务对象：行内或整体评论，可能来自人或 AI）。"""
    id: UUID
    body: str | None = None  # 评论正文 Markdown
    path: str | None = None  # 行内评论所属文件；整体评论为空
    line: int | None = None  # 行号；整体评论为空
    author: str | None = None  # 评论者
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = {"from_attributes": True}


class PullRequestResponse(BaseModel):
    """PR 详情（业务对象：一个 PR 的完整展示数据，含嵌套子资源）。"""
    id: UUID
    repo_id: UUID
    number: int  # 仓库内编号（#456）
    title: str
    body: str | None = None  # PR 描述
    state: str  # open / closed；是否合并看 merged_at
    author: str | None = None
    base_branch: str | None = None  # 目标分支
    head_branch: str | None = None  # 源分支
    merged_at: datetime | None = None  # 非空即已合并
    created_at: datetime | None = None
    updated_at: datetime | None = None
    files: list[PRFileResponse] = []  # 嵌套的变更文件列表（关系已预加载，避免 N+1）
    review_comments: list[PRReviewCommentResponse] = []  # 嵌套的评论列表

    model_config = {"from_attributes": True}


class PRAnalysisResponse(BaseModel):
    """PR 评审结果（业务对象：一次 AI Code Review 的产出）。

    analysis_id 用 UUID | str，兼容历史记录与其他分析链路的统一返回形状。
    """
    analysis_id: UUID | str
    result: PRReviewOutput  # 见 schemas.analysis 中的严格结构定义
