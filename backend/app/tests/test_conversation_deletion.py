"""删除会话的级联清理测试。

删会话远不止删一条 Conversation：消息、Agent 运行记录、封存会话、反馈、
记忆候选、召回事件、工作流与任务、证据项都以 conversation_id 挂靠。
这个测试把这些表全填上，验证删除后：
1. 目标会话被软删（status=deleted）而不是物理消失，便于审计；
2. 所有从属记录被清空，不留孤儿数据（外键在 SQLite 下必须开启才能生效）；
3. 同一仓库里的其他会话不受影响，并被选为新的"当前会话"。
"""

import os
import uuid

# 先设环境变量再 import app.*：db 模块在导入时就会读取它
os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app.db.models import (
    AgentRun,
    AgentTaskRun,
    AgentWorkflowRun,
    Base,
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
from app.services.conversations import delete_conversation


def _db_session():
    """内存库 + 显式打开外键约束。

    SQLite 默认 FOREIGN KEY 是关闭的（PRAGMA foreign_keys=OFF），
    不打开的话 ON DELETE CASCADE / 外键校验都不会生效，测试会"假通过"。
    """
    engine = create_engine("sqlite+pysqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, _connection_record):
        # 每个新连接都要单独设置，sqlite 的这个开关是连接级的
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_delete_conversation_removes_session_memory_and_workflow_dependencies() -> None:
    """场景：一个仓库下有两个会话，其中目标会话把「所有可能挂靠的子表」都塞了一遍。

    期望：删除后返回剩下那个会话（作为当前活跃会话）；目标会话 status 变成 deleted；
    10 张从属表的记录数全为 0。
    为什么重要：漏删任何一张表都会留下孤儿数据——例如残留的 MemoryCandidate
    会让下一轮对话"记得"已经删掉的会话内容，残留任务运行记录还会继续被调度执行。
    """
    db = _db_session()
    # full_name 加随机后缀，避免重复建库时的唯一约束冲突
    repo = Repository(owner="devflow", name="conversation-delete", full_name=f"devflow/{uuid.uuid4().hex}")
    db.add(repo)
    db.flush()
    target = Conversation(repo_id=repo.id, title="hi，你好")
    # 对照组：同仓库的另一个会话，删除操作不应该波及它
    replacement = Conversation(repo_id=repo.id, title="保留会话")
    db.add_all([target, replacement])
    db.flush()

    run = AgentRun(
        repo_id=repo.id,
        conversation_id=target.id,
        user_message="hi，你好",
        route="test",
        tool_calls=[],
        final_answer="你好",
    )
    user_message = ChatMessage(repo_id=repo.id, conversation_id=target.id, role="user", content="hi，你好")
    assistant_message = ChatMessage(repo_id=repo.id, conversation_id=target.id, role="assistant", content="你好")
    session = ChatSession(repo_id=repo.id, conversation_id=target.id, route="test", transcript_json={}, digest="你好")
    workflow = AgentWorkflowRun(repo_id=repo.id, conversation_id=target.id, goal="test")
    db.add_all([run, user_message, assistant_message, session, workflow])
    db.flush()

    # 这一批是"间接挂靠"的子记录：反馈、记忆候选、召回事件、工作流任务、长期记忆、证据
    db.add_all(
        [
            ChatFeedback(
                repo_id=repo.id,
                conversation_id=target.id,
                assistant_message_id=assistant_message.id,
                run_id=run.id,
                rating="helpful",
            ),
            MemoryCandidate(
                repo_id=repo.id,
                conversation_id=target.id,
                session_id=session.id,
                kind="fact",
                title="candidate",
                content="must be deleted before the session",
            ),
            RecallEvent(repo_id=repo.id, conversation_id=target.id, tool_name="memory.search"),
            AgentTaskRun(
                workflow_id=workflow.id,
                task_id="task-1",
                agent_name="test-agent",
                task_type="test",
            ),
            ConversationMemory(repo_id=repo.id, conversation_id=target.id),
            EvidenceItem(
                repo_id=repo.id,
                conversation_id=target.id,
                source_type="test",
                source_id="conversation-delete",
                title="evidence",
                content="delete regression",
            ),
        ]
    )
    db.commit()

    active = delete_conversation(db, repo.id, target.id)
    db.commit()

    # 返回的是"删除后剩余的会话"，前端据此跳转
    assert active.id == replacement.id
    # 软删：记录保留但标记 deleted，保留可审计痕迹
    assert db.get(Conversation, target.id).status == "deleted"
    # 逐表检查没有孤儿记录
    for model in [
        ChatFeedback,
        MemoryCandidate,
        RecallEvent,
        AgentTaskRun,
        AgentWorkflowRun,
        ChatMessage,
        ChatSession,
        AgentRun,
        ConversationMemory,
        EvidenceItem,
    ]:
        assert db.query(model).count() == 0
