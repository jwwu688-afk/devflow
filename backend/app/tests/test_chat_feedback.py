"""聊天回答「反馈（点赞/点踩）」接口的测试。

覆盖三条行为：
1. 点踩（unhelpful）时应落库、触发通知、并把答案关联到 AgentRun / ChatSession 以便追溯；
2. 反馈统计（feedback_metrics）能算出负面反馈率与原因分布；
3. 同一条 assistant 消息重复提交反馈是幂等的（不会重复落库、不会重复通知）。
"""

import os
import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# 必须在 import app.* 之前设置：app.db 里的 engine 在模块导入时就会读取该环境变量，
# 这里指向内存 SQLite，保证测试不碰真实数据库
os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")

from app.api.routes import feedback as feedback_routes
from app.db.models import AgentRun, Base, ChatFeedback, ChatMessage, ChatSession, Conversation, Repository
from app.schemas.feedback import ChatFeedbackCreate, ChatFeedbackReview


def _db_session():
    """建一个内存库 + 建表，返回 Session。每个用例独立，互不污染。"""
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _feedback_fixture(db):
    """伪造一条「用户提问 → 助手答复 → AgentRun → ChatSession」的完整链路。

    反馈接口需要顺着 assistant_message 反查 run_id、会话摘要、工具调用事件，
    所以这里把相关的表都填上最少的必需数据，避免测试里出现空关联。
    """
    # full_name 加随机后缀：同一进程内多次建表/用例之间避免唯一约束冲突
    repo = Repository(owner="devflow", name="demo", full_name=f"devflow/demo-{uuid.uuid4().hex[:8]}")
    db.add(repo)
    db.flush()
    conversation = Conversation(repo_id=repo.id, title="反馈测试")
    db.add(conversation)
    db.flush()
    run_id = uuid.uuid4()
    # 用户消息：点踩反馈要能定位到「回答的是哪个问题」
    user_message = ChatMessage(
        repo_id=repo.id,
        conversation_id=conversation.id,
        role="user",
        content="为什么 CI 失败？",
    )
    assistant_message = ChatMessage(
        repo_id=repo.id,
        conversation_id=conversation.id,
        role="assistant",
        content="CI 是因为测试失败。",
        route="ci_debug",
        # tool_calls / meta 都写上 ci_debug，模拟真实回答：反馈追溯接口要读这些字段
        tool_calls=[{"tool_name": "ci_debug", "status": "success"}],
        meta={"run_id": str(run_id), "agent_events": [{"type": "tool_use", "tool_name": "ci_debug"}]},
    )
    db.add_all([user_message, assistant_message])
    db.flush()
    # AgentRun 记录本次 Agent 执行的输入输出，反馈靠 meta.run_id 关联到它
    db.add(
        AgentRun(
            id=run_id,
            repo_id=repo.id,
            conversation_id=conversation.id,
            user_message=user_message.content,
            route="ci_debug",
            tool_calls=assistant_message.tool_calls,
            final_answer=assistant_message.content,
            status="success",
        )
    )
    # ChatSession 是「已封存」的会话摘要，反馈追溯接口会读它的 agent_events
    db.add(
        ChatSession(
            repo_id=repo.id,
            conversation_id=conversation.id,
            user_message_id=user_message.id,
            assistant_message_id=assistant_message.id,
            route="ci_debug",
            transcript_json={
                "agent_events": [{"type": "tool_use", "tool_name": "ci_debug"}],
                "tool_use_results": [{"type": "tool_result", "status": "success"}],
                "memory_update": {"facts": ["CI failed"]},
            },
            digest="排查 CI 失败",
            status="sealed",
        )
    )
    db.commit()
    return repo, conversation, assistant_message, run_id


@pytest.mark.asyncio
async def test_negative_feedback_persists_notifies_and_traces(monkeypatch):
    """场景：用户对一条 ci_debug 回答点踩，并填写原因 inaccurate + 补一句说明。

    期望：反馈成功落库并返回 run_id / 通知状态 sent / 待处理状态 open；
    feedback_metrics 统计出负面反馈率 100%、原因计数 {inaccurate: 1}；
    feedback_trace 能把 run、提问、回答、会话事件整条链路串起来；
    最后人工复核可把状态改成 resolved 并记录复核时间。
    为什么重要：负面反馈是 Prompt 迭代的唯一真实信号源，链路任何一环断了都无法定位问题。
    """
    db = _db_session()
    repo, conversation, assistant_message, run_id = _feedback_fixture(db)
    notifications = []

    async def fake_notification(**kwargs):
        """替换真实通知发送（飞书/企业微信等外部依赖），只记录调用参数。"""
        notifications.append(kwargs)
        return "sent", None

    # 用 monkeypatch 打桩，避免测试真的往外部 IM 推消息
    monkeypatch.setattr(feedback_routes, "send_negative_feedback_notification", fake_notification)
    result = await feedback_routes.create_feedback(
        ChatFeedbackCreate(
            repo_id=repo.id,
            conversation_id=conversation.id,
            assistant_message_id=assistant_message.id,
            rating="unhelpful",
            reason="inaccurate",
            comment="真正失败的是类型检查。",
        ),
        db,
    )

    assert result.run_id == run_id
    assert result.notification_status == "sent"
    assert result.review_status == "open"
    # 只通知一次，防止重复打扰客服/负责人
    assert len(notifications) == 1

    # 统计口径校验：分母是 assistant 消息数，分子是点踩数
    metrics = await feedback_routes.feedback_metrics(repo.id, db)
    assert metrics["assistant_messages"] == 1
    assert metrics["unhelpful"] == 1
    assert metrics["negative_feedback_rate"] == 1.0
    assert metrics["reason_counts"] == {"inaccurate": 1}

    # 追溯接口要用 run_id 把整条执行链路还原出来
    trace = await feedback_routes.feedback_trace(run_id, db)
    assert trace["trace_id"] == str(run_id)
    assert trace["run"]["route"] == "ci_debug"
    assert trace["messages"]["assistant"]["id"] == str(assistant_message.id)
    assert trace["session"]["agent_events"][0]["tool_name"] == "ci_debug"

    reviewed = await feedback_routes.review_feedback(
        result.id,
        ChatFeedbackReview(review_status="resolved", review_note="已修复 Prompt"),
        db,
    )
    assert reviewed.review_status == "resolved"
    assert reviewed.reviewed_at is not None


@pytest.mark.asyncio
async def test_feedback_is_idempotent_per_assistant_message(monkeypatch):
    """场景：同一条 assistant 消息，用户连点两次「没用」。

    期望：两次返回同一个反馈 id、库里只有 1 条记录、只发 1 次通知；
    之后再提交「有用（helpful）」时，记录被更新为 helpful 且原因被清空、不再需要通知。
    为什么重要：用户手抖/网络重试很容易重复提交，幂等做不好会刷屏通知并污染统计。
    """
    db = _db_session()
    repo, conversation, assistant_message, _run_id = _feedback_fixture(db)
    notifications = []

    async def fake_notification(**kwargs):
        """同样打桩外部通知，只累计调用次数用于断言。"""
        notifications.append(kwargs)
        return "sent", None

    monkeypatch.setattr(feedback_routes, "send_negative_feedback_notification", fake_notification)
    payload = ChatFeedbackCreate(
        repo_id=repo.id,
        conversation_id=conversation.id,
        assistant_message_id=assistant_message.id,
        rating="unhelpful",
    )
    first = await feedback_routes.create_feedback(payload, db)
    # 完全相同的 payload 再提交一次，应命中已有记录而不是新建
    second = await feedback_routes.create_feedback(payload, db)

    assert first.id == second.id
    assert db.query(ChatFeedback).count() == 1
    assert len(notifications) == 1

    # 同一消息改成「有用」：复用同一条记录，rating 被覆盖
    helpful = await feedback_routes.create_feedback(
        ChatFeedbackCreate(
            repo_id=repo.id,
            conversation_id=conversation.id,
            assistant_message_id=assistant_message.id,
            rating="helpful",
        ),
        db,
    )
    assert helpful.rating == "helpful"
    assert helpful.reason is None
    assert helpful.notification_status == "not_applicable"
