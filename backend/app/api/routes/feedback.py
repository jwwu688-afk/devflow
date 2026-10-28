"""聊天回答的点赞/点踩反馈接口，挂载前缀 /api/chat/feedback。

前端在每个 assistant 气泡下放「有用 / 没用」，点踩时可以选原因并留评论；
后台再对点踩记录做人工复盘（review_status），并把严重问题通过通知渠道发出去。

接口清单：
    GET   /metrics?repo_id=...        反馈质量指标（覆盖率、好评率、待复盘数、原因分布）
    GET   /trace/{run_id}             一次回答的完整执行轨迹（消息、工具调用、Agent 事件）
    GET   ?repo_id=...                反馈列表
    POST  /?                          提交/更新反馈（按 assistant_message_id 幂等 upsert）
    PATCH /{feedback_id}              复盘结论（resolved / dismissed / in_review）

说明：本模块没有挂 require_permission，用户身份统一走 ensure_demo_user——
课程项目是单用户 demo，等接入真实登录后这里需要补上权限校验。
"""

import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.models import AgentRun, ChatFeedback, ChatMessage, ChatSession, Conversation, Repository
from app.db.session import get_db
from app.schemas.feedback import ChatFeedbackCreate, ChatFeedbackResponse, ChatFeedbackReview
from app.services.feedback_notifications import send_negative_feedback_notification
from app.services.permissions import ensure_demo_user

router = APIRouter()


def _run_id_from_message(message: ChatMessage) -> uuid.UUID | None:
    """从消息的 meta.run_id 里取出关联的 AgentRun ID。

    入参：message 为 assistant 消息。
    返回：合法 UUID，或 None（没记录、格式非法都算没有）。
    说明：
        run_id 没有独立外键列，是塞在 meta(JSON) 里的；而 JSON 里的值可能是字符串、
        也可能是脏数据，所以必须 try/except 兜住，不能让一条坏数据把整个轨迹接口打挂。
    """
    raw_run_id = (message.meta or {}).get("run_id")
    if not raw_run_id:
        return None
    try:
        return uuid.UUID(str(raw_run_id))
    except (TypeError, ValueError):
        return None


def _message_or_404(db: Session, payload: ChatFeedbackCreate) -> ChatMessage:
    """按 (消息ID, 仓库, 会话, role=assistant) 四元组定位被评价的回答。

    入参：payload 提供 assistant_message_id / repo_id / conversation_id。
    返回：ChatMessage。
    异常：任一项对不上 → 404。
    说明：
        刻意把 repo_id 和 conversation_id 也放进过滤条件，而不是只按主键查：
        否则知道某个 message_id 的人可以对别的仓库/会话里的回答刷反馈，
        属于越权写入。多带两个条件是成本最低的隔离手段。
    """
    message = (
        db.query(ChatMessage)
        .filter(
            ChatMessage.id == payload.assistant_message_id,
            ChatMessage.repo_id == payload.repo_id,
            ChatMessage.conversation_id == payload.conversation_id,
            # 只允许评价 AI 的回答；顺手挡住「给自己（user 消息）点赞」这种脏数据。
            ChatMessage.role == "assistant",
        )
        .one_or_none()
    )
    if message is None:
        raise HTTPException(status_code=404, detail="未找到对应的 Agent 回答")
    return message


def _serialize_message(message: ChatMessage | None) -> dict[str, Any] | None:
    """把 ChatMessage 转成轨迹接口里用的精简 dict；None 原样返回 None。

    不直接用 pydantic 模型是因为这里只需要几个字段，且字段名要对齐前端轨迹面板的约定。
    """
    if message is None:
        return None
    return {
        "id": str(message.id),
        "role": message.role,
        "content": message.content,
        "route": message.route,
        # tool_calls / meta 都可能为 NULL，统一兜成空列表/空字典，前端免判空。
        "tool_calls": message.tool_calls or [],
        "metadata": message.meta or {},
        "created_at": message.created_at.isoformat() if message.created_at else None,
    }


@router.get("/metrics")
async def feedback_metrics(
    repo_id: uuid.UUID = Query(...),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """GET /api/chat/feedback/metrics?repo_id=...：反馈质量指标。

    入参：repo_id 为必填查询参数（Query(...) 表示不给就 422）。
    返回：dict，含覆盖率、好评率、点踩率、待复盘数、点踩原因分布等。
    异常：无（repo 不存在时各项为 0）。
    副作用：只读。注意把该仓库的反馈全量拉进内存统计，数据量大时应改成聚合 SQL。
    """
    rows = db.query(ChatFeedback).filter(ChatFeedback.repo_id == repo_id).all()
    assistant_messages = db.query(ChatMessage).filter(ChatMessage.repo_id == repo_id, ChatMessage.role == "assistant").count()
    # sum(bool) 是「数 True 的个数」的惯用写法，比 len([...]) 少一次中间列表。
    helpful = sum(row.rating == "helpful" for row in rows)
    unhelpful = sum(row.rating == "unhelpful" for row in rows)
    rated = len(rows)
    reasons: dict[str, int] = {}
    for row in rows:
        if row.reason:
            reasons[row.reason] = reasons.get(row.reason, 0) + 1
    return {
        "repo_id": str(repo_id),
        "assistant_messages": assistant_messages,
        "rated_messages": rated,
        # 三个比率的分子分母不同，看指标时别混：
        #   feedback_coverage      = 有评价的 / 全部回答（有多少回答被评过）
        #   helpful_rate           = 好评 / 有评价的（评价里的好评占比）
        #   negative_feedback_rate = 差评 / 全部回答（把「没人评」也算进分母，
        #                            用来衡量负面反馈在整体回答中的绝对占比）
        # 所有除法都先判分母，避免 0 除；round 到 4 位是给前端百分比展示用。
        "feedback_coverage": round(rated / assistant_messages, 4) if assistant_messages else 0.0,
        "helpful": helpful,
        "unhelpful": unhelpful,
        "helpful_rate": round(helpful / rated, 4) if rated else 0.0,
        "unhelpful_rate": round(unhelpful / rated, 4) if rated else 0.0,
        "negative_feedback_rate": round(unhelpful / assistant_messages, 4) if assistant_messages else 0.0,
        # 待复盘 = 点踩 且 还没处理完的（open / in_review 都算待办）。
        "open_reviews": sum(row.rating == "unhelpful" and row.review_status in {"open", "in_review"} for row in rows),
        "reason_counts": reasons,
    }


@router.get("/trace/{run_id}")
async def feedback_trace(run_id: uuid.UUID, db: Session = Depends(get_db)) -> dict[str, Any]:
    """GET /api/chat/feedback/trace/{run_id}：一次回答的完整执行轨迹。

    入参：run_id 为 AgentRun 的 ID（路径参数）。
    返回：dict，含仓库/会话概要、反馈、run 记录、用户与 AI 两条消息、会话 transcript
        （agent_events / tool_use_results / memory_update）。
    异常：
        AgentRun 和 feedback 都不存在 → 404。只要有一个存在就尽力返回，
        因为 AgentRun 可能已被清理，但反馈记录还留着。
    副作用：只读。
    """
    run = db.get(AgentRun, run_id)
    feedback = db.query(ChatFeedback).filter(ChatFeedback.run_id == run_id).one_or_none()
    if run is None and feedback is None:
        raise HTTPException(status_code=404, detail="未找到对应的执行轨迹")

    assistant_message = db.get(ChatMessage, feedback.assistant_message_id) if feedback else None
    if assistant_message is None and run is not None:
        # 兜底路径：没有反馈（或反馈没记 message id）时，按 run 所在会话把 assistant 消息
        # 全捞出来，再逐条比对 meta.run_id 找出对应的那条。
        # 之所以不直接查 meta JSON 字段，是因为 SQLite/MySQL 对 JSON 查询的语法不一致。
        candidates = (
            db.query(ChatMessage)
            .filter(ChatMessage.conversation_id == run.conversation_id, ChatMessage.role == "assistant")
            .order_by(ChatMessage.created_at.desc())
            .all()
        )
        assistant_message = next((item for item in candidates if _run_id_from_message(item) == run_id), None)

    # ChatSession 通过 assistant_message_id 反查：一个 session 记录了「一轮问答」的完整过程，
    # 包括用户消息 ID 和 transcript（Agent 事件流）。
    session = (
        db.query(ChatSession).filter(ChatSession.assistant_message_id == assistant_message.id).one_or_none()
        if assistant_message
        else None
    )
    user_message = db.get(ChatMessage, session.user_message_id) if session and session.user_message_id else None
    # 仓库/会话 ID 优先取反馈上的（用户操作时记下的），没有再用 run 上的。
    repo_id = feedback.repo_id if feedback else run.repo_id if run else None
    conversation_id = feedback.conversation_id if feedback else run.conversation_id if run else None
    repo = db.get(Repository, repo_id) if repo_id else None
    conversation = db.get(Conversation, conversation_id) if conversation_id else None
    transcript = session.transcript_json if session else {}

    return {
        "trace_id": str(run_id),
        "repository": {"id": str(repo.id), "full_name": repo.full_name} if repo else None,
        "conversation": {"id": str(conversation.id), "title": conversation.title} if conversation else None,
        # 复用响应模型做序列化，保证这里返回的 feedback 结构和列表接口完全一致。
        "feedback": ChatFeedbackResponse.model_validate(feedback).model_dump(mode="json") if feedback else None,
        "run": {
            "id": str(run.id),
            "status": run.status,
            "route": run.route,
            "user_message": run.user_message,
            "final_answer": run.final_answer,
            "tool_calls": run.tool_calls or [],
            "created_at": run.created_at.isoformat() if run.created_at else None,
        }
        if run
        else None,
        "messages": {
            "user": _serialize_message(user_message),
            "assistant": _serialize_message(assistant_message),
        },
        "session": {
            "id": str(session.id),
            "status": session.status,
            "digest": session.digest,
            # get 带默认值：transcript 可能是旧版本写入的，缺某个键时返回空而不是 KeyError。
            "agent_events": transcript.get("agent_events", []),
            "tool_use_results": transcript.get("tool_use_results", []),
            "memory_update": transcript.get("memory_update", {}),
            "sealed_at": session.sealed_at.isoformat() if session.sealed_at else None,
        }
        if session
        else None,
    }


@router.get("", response_model=list[ChatFeedbackResponse])
async def list_feedback(
    repo_id: uuid.UUID = Query(...),
    conversation_id: uuid.UUID | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=500),
    db: Session = Depends(get_db),
) -> list[ChatFeedback]:
    """GET /api/chat/feedback?repo_id=...：反馈列表。

    入参：repo_id 必填；conversation_id 可选（只看某个会话）；limit 1~500，默认 200。
    返回：按创建时间倒序的反馈列表。
    异常：参数非法 → 422。
    副作用：只读。
    """
    query = db.query(ChatFeedback).filter(ChatFeedback.repo_id == repo_id)
    if conversation_id:
        query = query.filter(ChatFeedback.conversation_id == conversation_id)
    return query.order_by(ChatFeedback.created_at.desc()).limit(limit).all()


@router.post("", response_model=ChatFeedbackResponse)
async def create_feedback(payload: ChatFeedbackCreate, db: Session = Depends(get_db)) -> ChatFeedback:
    """POST /api/chat/feedback：提交反馈（同一个回答重复提交 = 覆盖更新）。

    入参：payload 含 repo_id / conversation_id / assistant_message_id / rating / reason / comment。
    返回：最新的 ChatFeedback。
    异常：
        仓库、会话不存在或二者不匹配 → 404；
        消息不是该仓库该会话里的 assistant 消息 → 404（见 _message_or_404）。
    副作用：
        写/更新 chat_feedback；点踩时可能还会发一条通知（外部 HTTP）并把通知结果写回。
        这里分两次 commit：先落反馈，再落通知结果——即使通知服务挂了，
        用户的评价也不会丢，只是 notification_status 记成 failed。
    """
    repo = db.get(Repository, payload.repo_id)
    conversation = db.get(Conversation, payload.conversation_id)
    # conversation.repo_id != repo.id 也要拦：否则可以拿 A 仓库的会话 + B 仓库的 ID 拼出越权写入。
    if repo is None or conversation is None or conversation.repo_id != repo.id:
        raise HTTPException(status_code=404, detail="未找到对应的项目会话")
    message = _message_or_404(db, payload)
    run_id = _run_id_from_message(message)
    run = db.get(AgentRun, run_id) if run_id else None
    # demo 环境没有真实登录态，统一取/建一个演示用户来充当 feedback.user_id。
    reporter = ensure_demo_user(db)

    # 一个回答只保留一条反馈（按 assistant_message_id 查），所以这里是 upsert 语义：
    # 用户从「赞」改成「踩」时走的是更新分支，而不是插入第二条。
    feedback = db.query(ChatFeedback).filter(ChatFeedback.assistant_message_id == message.id).one_or_none()
    # notify 的判定要覆盖三种情况，避免重复打扰：
    #   1) 之前没反馈过；2) 之前是 helpful；3) 之前就是 unhelpful 但通知还没发出去（notified_at 为空）。
    notify = payload.rating == "unhelpful" and (feedback is None or feedback.rating != "unhelpful" or feedback.notified_at is None)
    if feedback is None:
        feedback = ChatFeedback(
            repo_id=repo.id,
            conversation_id=conversation.id,
            assistant_message_id=message.id,
            run_id=run_id,
            user_id=reporter.id,
            rating=payload.rating,
        )
        db.add(feedback)

    # 必须在覆盖之前记下旧值：下面要靠它判断「是不是刚从好评变成差评」。
    previous_rating = feedback.rating
    feedback.rating = payload.rating
    # reason 只在点踩时有意义，改回好评就清掉，避免历史原因残留在统计里。
    feedback.reason = payload.reason if payload.rating == "unhelpful" else None
    # 空字符串统一转 None：否则统计里会出现一个原因为 "" 的桶。
    feedback.comment = (payload.comment or "").strip() or None
    feedback.run_id = run_id
    feedback.user_id = reporter.id
    if payload.rating == "unhelpful":
        # 只有「新出现的差评」才重置复盘状态；否则用户重复点踩会把已经处理完的记录打回 open。
        if previous_rating != "unhelpful":
            feedback.review_status = "open"
            feedback.review_note = None
            feedback.reviewed_at = None
        if notify:
            # 先标 pending，表示「即将发通知」；真正结果在下面发完之后再回写。
            feedback.notification_status = "pending"
            feedback.notification_error = None
    else:
        # 改成好评视为问题已解决：自动置 resolved，省掉一次人工复盘。
        feedback.review_status = "resolved"
        feedback.review_note = None
        feedback.reviewed_at = datetime.now(timezone.utc)
        feedback.notification_status = "not_applicable"
        feedback.notification_error = None
    db.commit()
    db.refresh(feedback)

    if notify:
        notification_status, notification_error = await send_negative_feedback_notification(
            feedback=feedback,
            repo=repo,
            message=message,
            run=run,
            reporter=reporter,
        )
        feedback.notification_status = notification_status
        feedback.notification_error = notification_error
        # 只有确实发成功了才记 notified_at，这个字段是上面 notify 判重的关键依据。
        feedback.notified_at = datetime.now(timezone.utc) if notification_status == "sent" else None
        db.commit()
        db.refresh(feedback)
    return feedback


@router.patch("/{feedback_id}", response_model=ChatFeedbackResponse)
async def review_feedback(
    feedback_id: uuid.UUID,
    payload: ChatFeedbackReview,
    db: Session = Depends(get_db),
) -> ChatFeedback:
    """PATCH /api/chat/feedback/{feedback_id}：填写复盘结论。

    入参：
        feedback_id: 路径参数。
        payload: review_status（如 resolved / dismissed / in_review）+ 可选 review_note。
    返回：更新后的 ChatFeedback。
    异常：反馈不存在 → 404。
    副作用：更新复盘字段（不改 rating，用户评价保持原样，只叠加处理结果）。
    """
    feedback = db.get(ChatFeedback, feedback_id)
    if feedback is None:
        raise HTTPException(status_code=404, detail="未找到反馈记录")
    feedback.review_status = payload.review_status
    feedback.review_note = (payload.review_note or "").strip() or None
    # 只有「已处理」类状态才盖时间戳；退回 in_review 时清空，方便区分「处理过又重开」的记录。
    feedback.reviewed_at = datetime.now(timezone.utc) if payload.review_status in {"resolved", "dismissed"} else None
    db.commit()
    db.refresh(feedback)
    return feedback
