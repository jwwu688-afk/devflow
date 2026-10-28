"""持久化对话记忆（时间线 / 精确召回复原 / 压缩落库 / 会话封存）的测试。

聊天记忆有三层，本文件逐层验证：
1. MessageStore.timeline：跨过压缩边界仍能拿到完整时间线，且不重复显示"兼容保留副本"；
   同一时间戳的消息用 id 做稳定游标分页；
2. MemoryTools：压缩丢掉的原文仍可"逐字召回"（并写 RecallEvent 审计），封存会话能读 transcript；
3. ProgressiveContextManager / SessionSealer：正式压缩复用原始行、能从工具调用里还原
   "当前正在改哪些文件"；记忆抽取失败时必须保留原始 transcript 而不是丢数据。
4. ContextAssembler：最近的历史必须走 history 通道，不能混进 system_context（否则会重复注入）。
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db.models import ChatMessage, ChatSession, Conversation, RecallEvent, Repository
from app.db.session import Base
from app.services.chat_memory import (
    ContextAssembler,
    MemoryTools,
    MessageStore,
    ProgressiveContextManager,
    SessionSealer,
)


def _db() -> Session:
    """每个用例一套独立的内存库，避免用例之间互相看到对方的数据。"""
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def _scope(db: Session) -> tuple[Repository, Conversation]:
    """建一个仓库 + 一个会话，作为所有消息的归属（记忆按 仓库/会话 维度隔离）。"""
    repo = Repository(owner="acme", name="context", full_name="acme/context")
    db.add(repo)
    db.flush()
    conversation = Conversation(repo_id=repo.id, title="Long context")
    db.add(conversation)
    db.flush()
    return repo, conversation


def _message(
    db: Session,
    repo: Repository,
    conversation: Conversation,
    *,
    role: str,
    content: str,
    at: datetime,
    route: str | None = None,
    tool_calls: list[dict] | None = None,
    meta: dict | None = None,
) -> ChatMessage:
    """插入一条聊天消息。at 必须显式传入，因为很多用例要构造"同一秒内"或"跨压缩边界"的时序。"""
    row = ChatMessage(
        repo_id=repo.id,
        conversation_id=conversation.id,
        role=role,
        content=content,
        route=route,
        tool_calls=tool_calls or [],
        meta=meta or {},
        created_at=at,
    )
    db.add(row)
    db.flush()
    return row


def test_durable_timeline_crosses_boundary_and_hides_compatibility_copies() -> None:
    """场景：一段对话中间插入了压缩边界（compact_boundary system 消息），
    边界后还有一条"内容与旧答案相同"的兼容保留副本（preserved_after_compact）。

    期望：时间线返回 [first, answer, latest]，边界消息和那条副本都不出现；
    也就是说读历史时用户看到的仍是连续、无重复的对话。
    为什么重要：压缩后若把边界和副本一起渲染，用户会看到"同一句话说了两遍"，
    以及一条莫名其妙的 summary 系统消息。
    """
    db = _db()
    repo, conversation = _scope(db)
    start = datetime.now(timezone.utc)
    first = _message(db, repo, conversation, role="user", content="first exact request", at=start)
    answer = _message(db, repo, conversation, role="assistant", content="first exact answer", at=start + timedelta(seconds=1))
    # 压缩边界：meta.subtype 标记类型，时间线读取时据此隐藏
    boundary = _message(
        db,
        repo,
        conversation,
        role="system",
        content="summary",
        route="context_compaction",
        meta={"subtype": "compact_boundary"},
        at=start + timedelta(seconds=2),
    )
    # 兼容用的"保留副本"：内容与 answer 完全相同，靠 meta 标记识别后跳过
    _message(
        db,
        repo,
        conversation,
        role="assistant",
        content=answer.content,
        meta={"subtype": "preserved_after_compact", "preserved_from": str(answer.id)},
        at=start + timedelta(seconds=3),
    )
    latest = _message(db, repo, conversation, role="user", content="latest request", at=start + timedelta(seconds=4))

    timeline = MessageStore(db).timeline(repo.id, conversation.id, limit=100)

    # 三条真实对话按时间顺序保留
    assert [row.id for row in timeline] == [first.id, answer.id, latest.id]
    # 压缩边界本身不作为"对话内容"展示
    assert boundary.id not in {row.id for row in timeline}


def test_durable_timeline_cursor_is_stable_when_timestamps_are_equal() -> None:
    """场景：同一秒内写入 3 条消息（created_at 完全相同），按每页 2 条翻页。

    期望：翻页拼起来的结果与"按 id 排序"完全一致，既不漏也不重。
    为什么重要：批量写入/快速连发会让多条消息时间戳相同；
    只按时间排序的游标在边界处会重复或跳条，用户就会看到消息凭空消失。
    """
    db = _db()
    repo, conversation = _scope(db)
    at = datetime.now(timezone.utc)
    rows = [
        _message(db, repo, conversation, role="user", content=f"same-time-{index}", at=at)
        for index in range(3)
    ]
    # 预期顺序：同一时间戳时按 id 兜底排序，作为游标稳定性的基准
    expected = sorted(rows, key=lambda row: row.id)

    newest_page = MessageStore(db).timeline(repo.id, conversation.id, limit=2)
    # 用"最新页第一条"作为游标向前翻，拿更早的一页
    older_page = MessageStore(db).timeline(
        repo.id,
        conversation.id,
        limit=2,
        before_message_id=newest_page[0].id,
    )

    assert [row.id for row in [*older_page, *newest_page]] == [row.id for row in expected]


@pytest.mark.asyncio
async def test_exact_thread_recall_reads_pre_boundary_text_and_is_audited() -> None:
    """场景：一条用户原话已被压缩边界"压缩掉"（不再出现在活跃上下文里），
    用户随后要求逐字召回它；另外还按 message_id + offset/max_chars 取一个字符区间。

    期望：关键词能命中压缩前的原文，答案里带上消息 id；
    按 id 取的第 7~19 个字符精确出现在返回中；
    两次调用都写入 RecallEvent，mode 都是 exact_transcript。
    为什么重要：压缩是为了省 token，但"用户要看原话"是刚需，
    所以原文必须留在库里可按需精确取回；审计记录则用于分析召回行为与成本。
    """
    db = _db()
    repo, conversation = _scope(db)
    start = datetime.now(timezone.utc)
    # 用一个唯一标记串，确保命中的是这条原文而不是任何摘要/副本
    old = _message(
        db,
        repo,
        conversation,
        role="user",
        content="UNIQUE-PRE-BOUNDARY original wording",
        at=start,
    )
    _message(
        db,
        repo,
        conversation,
        role="system",
        content="compressed summary",
        route="context_compaction",
        # preserved_message_ids 为空，表示这条边界之后"什么都没原样保留"，最容易丢原文的情况
        meta={"subtype": "compact_boundary", "preserved_message_ids": []},
        at=start + timedelta(seconds=1),
    )
    _message(db, repo, conversation, role="assistant", content="new answer", at=start + timedelta(seconds=2))

    tools = MemoryTools(db, repo, conversation)
    # 两种召回方式：按关键词搜，按 message_id 取字符区间（offset=7, max_chars=12）
    result = await tools.get_thread_context({"keyword": "UNIQUE-PRE-BOUNDARY", "limit": 10}, {})
    page = await tools.get_thread_context(
        {"message_id": str(old.id), "offset": 7, "max_chars": 12},
        {},
    )

    assert "UNIQUE-PRE-BOUNDARY original wording" in result["answer"]
    assert str(old.id) in result["answer"]
    # 字符级切片必须精确：offset=7、长度 12 对应 old.content[7:19]
    assert old.content[7:19] in page["answer"]
    # 每次召回都要留审计记录，mode 固定为"精确原文"
    events = db.query(RecallEvent).filter(RecallEvent.tool_name == "devflow_get_thread_context").all()
    assert len(events) == 2
    assert all(event.mode == "exact_transcript" for event in events)


@pytest.mark.asyncio
async def test_session_event_tool_returns_transcript_and_records_recall() -> None:
    """场景：一个已封存的旧会话，用户读取它的事件记录，第二次还带上 offset 做续读。

    期望：能读到封存的原始 transcript；续读结果的 metadata.offset 为 5；
    两次调用各写一条 RecallEvent，mode 为 sealed_transcript，result_count 均为 1。
    为什么重要："上次会话我们决定了什么"依赖封存数据可读；
    offset 生效则保证长 transcript 不会被单次截断而永远看不到后半段。
    """
    db = _db()
    repo, conversation = _scope(db)
    # 手工造一条 sealed 会话：transcript_json 里放一句唯一文本，便于断言读到的就是它
    session = ChatSession(
        repo_id=repo.id,
        conversation_id=conversation.id,
        route="chat",
        digest="short digest",
        transcript_json={"messages": [{"role": "user", "content": "exact sealed text"}]},
        status="sealed",
        sealed_at=datetime.now(timezone.utc),
    )
    db.add(session)
    db.flush()

    result = await MemoryTools(db, repo, conversation).read_session_events(
        {"session_id": str(session.id), "include_transcript": True},
        {},
    )
    continuation = await MemoryTools(db, repo, conversation).read_session_events(
        # 带 offset 的续读请求：验证长 transcript 可以从中间接着看
        {"session_id": str(session.id), "include_transcript": True, "offset": 5, "max_chars": 400},
        {},
    )

    assert "exact sealed text" in result["answer"]
    assert continuation["results"][0]["metadata"]["offset"] == 5
    events = db.query(RecallEvent).filter(RecallEvent.tool_name == "devflow_read_session_events").all()
    assert len(events) == 2
    assert all(event.mode == "sealed_transcript" for event in events)
    assert all(event.result_count == 1 for event in events)


@pytest.mark.asyncio
async def test_formal_compaction_reuses_original_rows_and_rehydrates_working_state() -> None:
    """场景：6 条长消息（其中一条用过 workspace.read_file 读 backend/app/auth.py），
    用假 LLM（因为真模型只负责写摘要，这里不关心摘要质量）执行一次正式压缩。

    期望：压缩成功；rehydration.working_files 还原出 ["backend/app/auth.py"]；
    库里不新增任何 preserved_after_compact 副本（旧做法会给每条消息复制一份）；
    压缩边界记录的 preserved_message_ids 恰好等于当前活跃的非 system 消息，且都来自原始行。
    为什么重要：「复用原有行 + 用边界表记录保留范围」避免了每次压缩都膨胀数据；
    而"当前在改哪些文件"必须在压缩后还能还原，否则模型会丢失正在进行的任务上下文。
    """
    class FakeLLM:
        """假 LLM：只回一句像样的摘要，重点是不引入网络依赖与随机性。"""

        async def chat_text(self, system_prompt: str, user_prompt: str) -> str:
            return "User intent, decisions, errors, pending task and next step are preserved."

    db = _db()
    repo, conversation = _scope(db)
    start = datetime.now(timezone.utc)
    rows: list[ChatMessage] = []
    # 交替造 user/assistant，每条都足够长（80 次 "details "），确保会触发压缩
    for index in range(6):
        rows.append(
            _message(
                db,
                repo,
                conversation,
                role="user" if index % 2 == 0 else "assistant",
                content=f"message {index} " + ("details " * 80),
                route="workspace.read_file" if index == 3 else None,
                # 只在第 4 条上挂一个 read_file 调用，作为"正在编辑的工作文件"来源
                tool_calls=[
                    {
                        "tool_name": "workspace.read_file",
                        "arguments": {"path": "backend/app/auth.py"},
                        "status": "success",
                    }
                ]
                if index == 3
                else [],
                at=start + timedelta(seconds=index),
            )
        )

    manager = ProgressiveContextManager(db, llm=FakeLLM())  # type: ignore[arg-type]
    # 触发一次 auto_compact 阶段的压缩
    result = await manager._compact(
        repo,
        conversation,
        "continue",
        {"stage": "auto_compact"},
        "auto_compact",
    )

    assert result["compacted"] is True
    # 从工具调用参数里还原出"当前工作文件"
    assert result["rehydration"]["working_files"] == ["backend/app/auth.py"]
    # 关键：不应再生成任何"保留副本"行，否则压缩一次数据就翻倍
    assert not any(
        (row.meta or {}).get("subtype") == "preserved_after_compact"
        for row in db.query(ChatMessage).all()
    )
    boundary = db.get(ChatMessage, uuid.UUID(result["boundary_id"]))
    assert boundary is not None
    # 边界里记录"保留了哪些原始消息 id"，活跃上下文就是按这份清单筛出来的
    preserved_ids = [uuid.UUID(value) for value in boundary.meta["preserved_message_ids"]]
    active = MessageStore(db).active_for_context(repo.id, conversation.id, limit=100)
    # 两项校验：清单顺序一致；且每条都确实来自原始行（没有新建副本）
    assert [row.id for row in active if row.role != "system"] == preserved_ids
    assert all(row.id in {item.id for item in rows} for row in active if row.role != "system")


@pytest.mark.asyncio
async def test_session_sealer_keeps_raw_transcript_when_extraction_fails() -> None:
    """场景：封存会话时，负责抽取结构化记忆的模型超时了。

    期望：会话仍被封存，状态为 sealed_with_warnings；
    原始对话内容（用户输入与助手回答）完整保留；errors 里记下 memory_extraction_failed。
    为什么重要：记忆抽取只是"加分项"，原始 transcript 才是不可再生的数据；
    一旦因为抽取失败而丢对话，用户就永久失去了那段记录。
    """
    class FailingLLM:
        """假 LLM：chat_json 直接超时，模拟记忆抽取服务不可用。"""

        async def chat_json(self, system_prompt: str, payload: dict) -> dict:
            raise TimeoutError("memory model timeout")

    db = _db()
    repo, conversation = _scope(db)
    now = datetime.now(timezone.utc)
    user = _message(db, repo, conversation, role="user", content="remember exact input", at=now)
    assistant = _message(db, repo, conversation, role="assistant", content="completed answer", at=now + timedelta(seconds=1))

    sealed = await SessionSealer(db, llm=FailingLLM()).seal(  # type: ignore[arg-type]
        repo=repo,
        conversation=conversation,
        user_message=user,
        assistant_message=assistant,
        route="chat",
        tool_calls=[],
    )

    assert sealed.status == "sealed_with_warnings"
    # 原文必须逐字保留（这是"抽取失败也不能丢数据"的核心断言）
    assert sealed.transcript_json["messages"][0]["content"] == "remember exact input"
    assert sealed.transcript_json["messages"][1]["content"] == "completed answer"
    assert "memory_extraction_failed" in sealed.transcript_json["sealing"]["errors"][0]


def test_context_assembler_keeps_recent_history_out_of_system_context() -> None:
    """场景：库里已有一条用户消息，且它就是当前正在处理的消息。

    期望：system_context 里不含这条消息文本（否则会重复注入），
    而是出现在 history_messages 末尾；pending_user_estimated_tokens 为 0
    （当前消息不算"待补"，不该再计一次）；组装后的 prompt 估算 token 不小于历史本身。
    为什么重要：同一条消息既进系统上下文又进历史，是 token 浪费与"鸡同鸭讲"的常见来源。
    """
    db = _db()
    repo, conversation = _scope(db)
    now = datetime.now(timezone.utc)
    _message(db, repo, conversation, role="user", content="UNIQUE-RECENT-MESSAGE", at=now)

    assembled = ContextAssembler(db).assemble(repo, conversation, "UNIQUE-RECENT-MESSAGE")

    assert "UNIQUE-RECENT-MESSAGE" not in assembled.system_context
    assert assembled.history_messages[-1]["content"] == "UNIQUE-RECENT-MESSAGE"
    assert assembled.compression_stats["pending_user_estimated_tokens"] == 0
    assert assembled.compression_stats["assembled_prompt_estimated_tokens"] >= assembled.compression_stats["history_estimated_tokens"]
