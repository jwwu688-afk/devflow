"""CI 相关响应模型（业务对象：流水线运行记录，以及 AI 对失败原因的诊断结果）。

只定义「读」模型：CI 数据全部由 GitHub 同步而来，前端不会写入，
因此没有 Create/Update 请求体，减少攻击面。
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

from app.schemas.analysis import CIDebugOutput


class WorkflowRunResponse(BaseModel):
    """流水线运行详情（业务对象：GitHub Actions 的一次 workflow run）。

    直接对应 WorkflowRun ORM 模型（from_attributes），用于列表与详情页。
    """
    id: UUID
    repo_id: UUID
    name: str  # workflow 名称
    status: str  # queued / in_progress / completed
    conclusion: str | None = None  # success / failure / cancelled；运行中为 None
    html_url: str | None = None  # 跳转 GitHub 的链接
    jobs: list[dict] = []  # 各 job 明细；结构来自 GitHub API，用 dict 直传避免频繁改模型
    logs_text: str | None = None  # 原始日志，可能很大，前端按需展开
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = {"from_attributes": True}


class CIAnalysisResponse(BaseModel):
    """CI 失败诊断结果（业务对象：一次诊断请求的产出）。

    analysis_id 复用 UUID | str 的联合类型，兼容历史记录里以字符串形式保存的 ID。
    """
    analysis_id: UUID | str  # 对应 analysis_results 表的记录 ID
    result: CIDebugOutput  # 结构化的诊断结论（见 schemas.analysis）
