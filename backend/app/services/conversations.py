"""会话（Conversation）的增删查与历史数据回填。

一个仓库下可以有多个会话；但任何时刻都必须至少存在一个 status="active" 的会话，
因为前端一进入项目页就会拉会话列表，没有会话时无从发起提问。
本模块所有函数都只 flush 不 commit，把提交时机交给上层请求统一控制。
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.db.models import (
    AgentRun,
    AgentTaskRun,
    AgentWorkflowRun,
    ChatFeedback,
    ChatMessage,
    ChatSession,
    Conversation,
    ConversationMemory,
    EvidenceItem,
    MemoryCandidate,
    RecallEvent,
    Repository,
)


def _default_title(db: Session, repo_id: uuid.UUID) -> str:
    """按仓库已有会话数生成「新会话 N」式的标题。

    用总数递增而不是取最大序号：删过会话后可能出现重名，
    但标题只是给人看的展示字段，重名无副作用，换来的是实现简单。
    """
    existing = db.query(Conversation).filter(Conversation.repo_id == repo_id).count()
    return "默认会话" if existing == 0 else f"新会话 {existing + 1}"


def ensure_default_conversation(db: Session, repo_id: uuid.UUID) -> Conversation:
    """返回该仓库最早的活跃会话，没有就新建一个「默认会话」。

    取 created_at 最早的那个（而不是最新的）：保证默认会话在整个仓库生命周期内稳定，
    前端保存的「当前会话」不会因为新建会话而漂移。
    """
    conversation = (
        db.query(Conversation)
        .filter(Conversation.repo_id == repo_id, Conversation.status == "active")
        .order_by(Conversation.created_at.asc())
        .first()
    )
    if conversation:
        return conversation
    conversation = Conversation(repo_id=repo_id, title="默认会话", status="active")
    db.add(conversation)
    # flush 才能拿到 id，调用方（如回填逻辑）需要用它的 id 去更新关联表。
    db.flush()
    return conversation


def ensure_conversation(db: Session, repo_id: uuid.UUID, conversation_id: uuid.UUID | None = None) -> Conversation:
    """校验 conversation_id 归属该仓库且未被删除，否则回落到默认会话。

    刻意不抛 404：前端可能持有已被删除的会话 id，此时静默切到默认会话
    比抛错体验更好，也更符合「会话只是容器」的语义。
    """
    if conversation_id:
        conversation = (
            db.query(Conversation)
            .filter(Conversation.id == conversation_id, Conversation.repo_id == repo_id, Conversation.status == "active")
            .one_or_none()
        )
        if conversation:
            return conversation
    return ensure_default_conversation(db, repo_id)


def list_conversations(db: Session, repo_id: uuid.UUID) -> list[Conversation]:
    """列出仓库下的活跃会话，最近活跃的排前面。

    读取时顺手做两件事：确保默认会话存在、把历史数据里 conversation_id 为空的消息归位。
    属于「读时自愈」：老版本升级上来的数据不用单独跑迁移脚本也能正常展示。
    """
    ensure_default_conversation(db, repo_id)
    _backfill_repo_conversation(db, repo_id)
    db.flush()
    return (
        db.query(Conversation)
        .filter(Conversation.repo_id == repo_id, Conversation.status == "active")
        # 先按最近更新时间倒序；同一时间的（批量导入常见）再用创建时间正序，保证顺序稳定。
        .order_by(Conversation.updated_at.desc(), Conversation.created_at.asc())
        .all()
    )


def create_conversation(db: Session, repo_id: uuid.UUID, title: str | None = None) -> Conversation:
    """新建会话；标题为空时自动生成一个。"""
    # strip 后为空也走自动命名，避免出现只含空格的标题。
    conversation = Conversation(repo_id=repo_id, title=(title or "").strip() or _default_title(db, repo_id))
    db.add(conversation)
    db.flush()
    return conversation


def delete_conversation(db: Session, repo_id: uuid.UUID, conversation_id: uuid.UUID) -> Conversation:
    """删除会话并返回「删除后应该展示给前端的那个会话」。

    两段式删除：关联的业务数据（消息、运行记录、反馈、记忆）**物理删除**，
    会话本身只做 status="deleted" 的软删除——保留行是为了让审计/统计还能看到
    这个会话曾经存在过；而消息等明细没有保留价值，硬删可以避免无限膨胀。
    返回替换会话而不是空值，前端删完立刻就有可显示的当前会话。
    """
    conversation = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id, Conversation.repo_id == repo_id, Conversation.status == "active")
        .one_or_none()
    )
    if conversation is None:
        # 已被删过或不属于该仓库：幂等处理，直接给一个可用会话。
        return ensure_default_conversation(db, repo_id)

    # 先收集子表的 id，后面按 in_ 批量删二级/三级表（任务运行挂在 workflow 下）。
    session_ids = [
        item.id
        for item in db.query(ChatSession.id).filter(ChatSession.conversation_id == conversation.id).all()
    ]
    workflow_ids = [
        item.id
        for item in db.query(AgentWorkflowRun.id).filter(AgentWorkflowRun.conversation_id == conversation.id).all()
    ]

    # 删除顺序 = 从叶子到根，避免外键约束报错。
    # synchronize_session=False：这些对象不打算在当前会话里继续用，跳过 ORM 同步更快。
    db.query(ChatFeedback).filter(ChatFeedback.conversation_id == conversation.id).delete(synchronize_session=False)
    db.query(MemoryCandidate).filter(MemoryCandidate.conversation_id == conversation.id).delete(synchronize_session=False)
    # 记忆候选既可能挂会话也可能只挂 session，两种都要清，否则会留下孤儿数据。
    if session_ids:
        db.query(MemoryCandidate).filter(MemoryCandidate.session_id.in_(session_ids)).delete(synchronize_session=False)
    db.query(RecallEvent).filter(RecallEvent.conversation_id == conversation.id).delete(synchronize_session=False)
    if workflow_ids:
        db.query(AgentTaskRun).filter(AgentTaskRun.workflow_id.in_(workflow_ids)).delete(synchronize_session=False)
    db.query(AgentWorkflowRun).filter(AgentWorkflowRun.conversation_id == conversation.id).delete(synchronize_session=False)
    db.query(ChatMessage).filter(ChatMessage.conversation_id == conversation.id).delete(synchronize_session=False)
    db.query(ChatSession).filter(ChatSession.conversation_id == conversation.id).delete(synchronize_session=False)
    db.query(AgentRun).filter(AgentRun.conversation_id == conversation.id).delete(synchronize_session=False)
    db.query(ConversationMemory).filter(ConversationMemory.conversation_id == conversation.id).delete(synchronize_session=False)
    db.query(EvidenceItem).filter(EvidenceItem.conversation_id == conversation.id).delete(synchronize_session=False)
    # 会话本身软删除：保留 id 与标题，方便事后追溯。
    conversation.status = "deleted"
    conversation.updated_at = datetime.now(timezone.utc)
    db.flush()

    # 排序规则和 list_conversations 保持一致，保证「删除后回到的那个会话」就是列表第一条。
    replacement = (
        db.query(Conversation)
        .filter(Conversation.repo_id == repo_id, Conversation.status == "active")
        .order_by(Conversation.updated_at.desc(), Conversation.created_at.asc())
        .first()
    )
    return replacement or create_conversation(db, repo_id, "默认会话")


def touch_conversation(
    db: Session,
    conversation: Conversation,
    *,
    user_message: str | None = None,
    increment: int = 0,
) -> None:
    """更新会话的活跃时间、消息计数，并在首条消息时自动用问题当标题。

    自动改标题的触发条件很严格：只有「默认会话/新会话」这类系统名、
    且此前一条消息都没有时才改，避免把用户手动改过的标题覆盖掉。
    """
    if user_message and conversation.message_count == 0 and conversation.title.startswith(("默认会话", "新会话")):
        # 换行换成空格并截到 40 字：标题是单行展示，太长会把侧边栏撑坏。
        title = user_message.strip().replace("\n", " ")
        if title:
            conversation.title = title[:40]
    # message_count 兼容历史 NULL 数据（老记录可能没初始化）。
    conversation.message_count = (conversation.message_count or 0) + increment
    conversation.updated_at = datetime.now(timezone.utc)
    db.flush()


def backfill_all_default_conversations(db: Session) -> None:
    """启动时给所有仓库补齐默认会话，并归位历史遗留的无归属数据。

    这是一次性/幂等的修复动作，所以这里是少数几个自己 commit 的函数之一：
    它在启动阶段被调用，没有外层请求事务可依赖。
    """
    for repo in db.query(Repository).all():
        ensure_default_conversation(db, repo.id)
        _backfill_repo_conversation(db, repo.id)
    db.commit()


def _backfill_repo_conversation(db: Session, repo_id: uuid.UUID) -> None:
    """把 repo_id 下 conversation_id 为 NULL 的老数据挂到默认会话上。

    这三张表在建表初期还没有 conversation_id 字段，升级后历史行是空的；
    这里用一条 UPDATE 批量补，比逐行改快得多。
    """
    conversation = ensure_default_conversation(db, repo_id)
    for model in [ChatMessage, ChatSession, AgentRun]:
        db.query(model).filter(model.repo_id == repo_id, model.conversation_id.is_(None)).update(
            {model.conversation_id: conversation.id},
            synchronize_session=False,
        )
    # 回填后重算消息数：刚才挂进去的消息也要计入，否则列表上的未读数/计数会偏小。
    conversation.message_count = db.query(ChatMessage).filter(ChatMessage.conversation_id == conversation.id).count()
