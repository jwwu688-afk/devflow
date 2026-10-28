"""回答反馈模型（业务对象：用户对某条助手回答的赞/踩评价，以及后台的处理流程）。

把评分与原因定义成 Literal 别名而不是裸 str：
1. 入参非法枚举会被 pydantic 直接挡成 422，避免脏值进库；
2. OpenAPI 会生成枚举列表，前端可直接据此渲染按钮，不必硬编码文案；
3. 复查时一眼能看出「系统认可的全部取值」，不会漏统计某个值。
"""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


# 评分只有「有帮助/没帮助」两档：刻意不做 5 星，降低用户表态成本，也便于二分类统计
FeedbackRating = Literal["helpful", "unhelpful"]
# 负反馈原因码：覆盖问答链路上最常见的几类失败，便于聚合出「主要问题在哪一环」
FeedbackReason = Literal[
    "inaccurate",  # 答案不准
    "not_relevant",  # 答非所问
    "missing_context",  # 缺少必要上下文
    "unreliable_citation",  # 引用不可靠
    "tool_error",  # 工具调用出错
    "other",
]
# 后台处理状态：open（待处理）→ in_review（处理中）→ resolved / dismissed（已解决 / 不处理）
ReviewStatus = Literal["open", "in_review", "resolved", "dismissed"]


class ChatFeedbackCreate(BaseModel):
    """提交反馈的请求体（业务对象：用户点下赞/踩时发送的数据）。"""
    repo_id: UUID
    conversation_id: UUID
    assistant_message_id: UUID  # 被评价的那条助手消息；库里有唯一约束，重复提交即为覆盖
    rating: FeedbackRating
    reason: FeedbackReason | None = None  # 仅负反馈需要填；正反馈留空
    # 限制 1000 字：反馈是给人看的短文本，过长往往是把日志粘进来了，说明该走 issue 渠道
    comment: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def validate_reason(self) -> "ChatFeedbackCreate":
        """跨字段校验：点赞时不允许带负反馈原因。

        用 mode="after" 是因为要等字段类型都转换完（rating 已确定是合法枚举）再判断，
        mode="before" 阶段拿到的可能是未校验的原始值。
        """
        if self.rating == "helpful" and self.reason is not None:
            raise ValueError("有帮助反馈不需要填写负反馈原因")
        return self


class ChatFeedbackReview(BaseModel):
    """人工处理反馈的请求体（业务对象：运营/开发对一条反馈的处理结论）。"""
    review_status: ReviewStatus  # 处理后的状态
    review_note: str | None = Field(default=None, max_length=2000)  # 处理说明，留档供复盘


class ChatFeedbackResponse(BaseModel):
    """反馈详情（业务对象：反馈列表中的一条记录，含通知与处理状态）。

    notification_status / notification_error 单独暴露，是为了让运营能看出
    「飞书通知没发出去」这类问题，而不是只看到反馈本身。
    """
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    repo_id: UUID
    conversation_id: UUID
    assistant_message_id: UUID
    run_id: UUID | None = None  # 关联的 Agent 运行记录；轨迹被清理后为 None
    rating: FeedbackRating
    reason: FeedbackReason | None = None
    comment: str | None = None
    review_status: ReviewStatus
    review_note: str | None = None
    notification_status: str  # pending / sent / failed
    notification_error: str | None = None  # 推送失败原因
    notified_at: datetime | None = None
    reviewed_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
