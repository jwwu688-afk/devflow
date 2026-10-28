"""项目索引相关模型（业务对象：仓库代码索引的进度、触发与「稍后提醒」设置）。

索引是重量级操作（要遍历文件、切块、向量化），因此接口被设计成
「查询状态 + 异步触发 + 可延后提醒」三件套，避免前端反复触发导致并发索引。
"""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class ProjectIndexStateResponse(BaseModel):
    """索引状态快照（业务对象：某仓库当前索引进度，前端进度条的数据源）。"""
    repo_id: UUID
    status: str  # missing / indexing / ready / failed
    fingerprint: str = ""  # 内容指纹，前端可比对判断是否需要重新索引
    docs_indexed: int = 0  # 已索引数（分子）
    docs_total: int = 0  # 总数（分母）；为 0 时前端不要算百分比，避免除零
    error_message: str | None = None  # 失败原因
    summary_json: dict[str, Any] | None = None  # 索引摘要（语言分布、文件数等）
    snoozed_until: datetime | None = None  # 在此之前不再提醒用户
    last_scan_at: datetime | None = None  # 上次扫描时间


class ProjectIndexScanResponse(BaseModel):
    """手动触发索引的响应（业务对象：一次「开始索引」的受理回执）。

    started 表示是否真的启动了任务：已有任务在跑时会返回 False，
    这样前端能区分「刚启动」和「本来就在跑」，避免重复提示。
    """
    repo_id: UUID
    started: bool
    status: str = "building"  # 受理后的初始状态


class ProjectIndexSnoozeRequest(BaseModel):
    """「稍后提醒」请求（业务对象：用户希望延后多久再提示索引）。"""
    # 1~30 天：太短没意义，太长等于永久关闭，因此用范围约束代替开关
    days: int = Field(default=7, ge=1, le=30)


class ProjectIndexSnoozeResponse(BaseModel):
    """「稍后提醒」响应（业务对象：延后到的具体时间点）。

    返回计算好的绝对时间而不是天数，让前端无需重复做时间运算，也避免时区理解不一致。
    """
    repo_id: UUID
    snoozed_until: datetime
