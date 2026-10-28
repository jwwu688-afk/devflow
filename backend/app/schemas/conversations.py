"""会话相关模型（业务对象：左侧会话列表中的一条对话线程）。"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class ConversationCreateRequest(BaseModel):
    """新建会话请求（业务对象：用户手动开一个新对话）。

    title 允许为空：为空时服务端会给出默认标题（"默认会话"），
    并在首轮问答后可能根据内容自动改写，避免用户被迫先想标题。
    """
    title: str | None = None


class ConversationResponse(BaseModel):
    """会话详情/列表项（业务对象：一条会话的展示信息）。

    message_count 是数据库里的冗余计数（见 Conversation 模型），
    前端用它显示条数而不必额外请求消息列表。
    """
    id: UUID
    repo_id: UUID
    title: str
    status: str = "active"  # active / archived
    message_count: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = {"from_attributes": True}
