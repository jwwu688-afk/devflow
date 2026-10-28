"""删除仓库（delete_repo）的级联清理回归测试。

验证行为：用户删除一个仓库时，所有「挂在这个仓库上的」派生数据必须被彻底清掉，
既包括关系库里的会话/记忆/工作流/证据/知识图谱/索引状态，也包括 Milvus 里的向量。
本用例刻意把十来张关联表全部塞满数据再删，就是为了防住「新增一张表但忘了
加清理逻辑」导致的孤儿数据回归——这类漏删在 SQLite 下不报错，只在生产库留垃圾。
"""

import os
import uuid

# 必须在导入 app.db.models 之前设置：模型/引擎在导入期就会读取 DATABASE_URL，
# 这里指向内存 SQLite，保证测试不碰真实库。
os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app.api.routes import repos as repo_routes
from app.db.models import (
    AgentRun,
    AgentTaskRun,
    AgentWorkflowRun,
    Base,
    ChatMessage,
    ChatSession,
    Conversation,
    EvidenceItem,
    KnowledgeGraphEdge,
    MemoryCandidate,
    ProjectIndexState,
    RecallEvent,
    Repository,
    Workspace,
)


def _db_session():
    """建内存库并显式打开 SQLite 外键约束，返回 Session。

    SQLite 默认不强制外键，必须用 PRAGMA foreign_keys=ON 打开，
    否则本用例想验证的「外键顺序/级联」问题在测试里根本不会暴露
    （生产环境 PostgreSQL 是强制的，测试必须对齐才有意义）。
    """
    engine = create_engine("sqlite+pysqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, _connection_record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


@pytest.mark.asyncio
async def test_delete_repo_removes_all_conversation_and_workflow_dependencies(monkeypatch) -> None:
    """测的是删除仓库时的全量级联：关联表清空 + 向量删除被调用一次 + 工作区解除引用。

    场景：构造一个仓库，并在其下挂上工作区、会话、聊天会话、工作流、
    任务运行、Agent 运行、聊天消息、证据、记忆候选、召回事件、知识图谱边、
    项目索引状态等全量依赖数据，然后调用删除接口。
    期望：
    - 返回 deleted=True、cleanup.milvus_deleted=True；
    - Milvus 删除恰好被调用一次，参数为 {repo_id, strict: False}
      （strict=False 表示向量库不可用时不阻塞主流程）；
    - 所有关联表计数归零（逐表断言，避免只测了汇总）；
    - 仍被保留的 Workspace 的 repo_ids 里必须移除该仓库，不能留悬空 id。
    重要性：漏删任何一张表都会留下孤儿记录，之后同名仓库重新接入时
    会把旧数据串出来（串号），属于难以排查的数据污染。
    """
    db = _db_session()
    # 仓库名带随机后缀，避免任何潜在的全局唯一约束冲突
    repo = Repository(owner="devflow", name="delete-me", full_name=f"devflow/delete-{uuid.uuid4().hex[:8]}")
    db.add(repo)
    db.flush()  # flush 但不 commit：先拿到 repo.id 供后续外键引用

    # 工作区通过 repo_ids 数组弱引用仓库（无外键），会话则用 repo_id 强引用
    workspace = Workspace(name="deletion-test", repo_ids=[str(repo.id)])
    conversation = Conversation(repo_id=repo.id, title="delete regression")
    db.add_all([workspace, conversation])
    db.flush()

    # 注意：下面这些子表故意把 repo_id 置为 None，只通过 conversation_id 间接连到仓库，
    # 用来验证「间接关联」的数据也能被清理（只看 repo_id 的实现会漏掉它们）。
    session = ChatSession(
        repo_id=None,
        conversation_id=conversation.id,
        route="test",
        transcript_json={},
        digest="delete regression",
    )
    workflow = AgentWorkflowRun(
        repo_id=None,
        conversation_id=conversation.id,
        goal="delete regression",
    )
    db.add_all([session, workflow])
    db.flush()  # 同样先拿 id，供下游 MemoryCandidate / AgentTaskRun 引用

    # 各表故意按「删除难度」排列：有外键指向 session / workflow 的放在前面，
    # 若删除顺序写反，PRAGMA foreign_keys=ON 会直接报错。
    db.add_all(
        [
            MemoryCandidate(
                repo_id=None,
                conversation_id=conversation.id,
                session_id=session.id,
                kind="fact",
                title="candidate",
                content="must be deleted before the session",
            ),
            RecallEvent(
                repo_id=None,
                conversation_id=conversation.id,
                tool_name="memory.search",
            ),
            AgentTaskRun(
                workflow_id=workflow.id,
                task_id="task-1",
                agent_name="test-agent",
                task_type="test",
            ),
            AgentRun(
                repo_id=None,
                conversation_id=conversation.id,
                user_message="delete",
                route="test",
                tool_calls=[],
                final_answer="deleted",
            ),
            ChatMessage(
                repo_id=None,
                conversation_id=conversation.id,
                role="user",
                content="delete this project",
            ),
            EvidenceItem(
                repo_id=None,
                conversation_id=conversation.id,
                source_type="test",
                source_id="delete-test",
                title="evidence",
                content="delete regression",
            ),
            # 这两张直接挂 repo_id，验证「直接关联」路径
            KnowledgeGraphEdge(
                repo_id=repo.id,
                from_type="issue",
                from_id="1",
                to_type="pull_request",
                to_id="2",
                relation="related",
            ),
            ProjectIndexState(repo_id=repo.id),
        ]
    )
    db.commit()

    # 记录 Milvus 删除调用的参数，用于断言「调了几次、传了什么」
    vector_calls = []

    def fake_delete_milvus_documents(**kwargs):
        """假向量删除：不真连 Milvus，只记录入参并声称成功。"""
        vector_calls.append(kwargs)
        return True

    # 只替换 repo_routes 命名空间里的引用，避免真实网络调用
    monkeypatch.setattr(repo_routes, "delete_milvus_documents", fake_delete_milvus_documents)

    result = await repo_routes.delete_repo(repo.id, db)

    assert result["deleted"] is True
    assert result["cleanup"]["milvus_deleted"] is True
    # 恰好一次，且 strict=False（向量库异常不得阻塞删除主流程）
    assert vector_calls == [{"repo_id": repo.id, "strict": False}]
    # 仓库本体与各关联表全部清空
    assert db.get(Repository, repo.id) is None
    assert db.query(Conversation).count() == 0
    assert db.query(ChatSession).count() == 0
    assert db.query(MemoryCandidate).count() == 0
    assert db.query(RecallEvent).count() == 0
    assert db.query(AgentWorkflowRun).count() == 0
    assert db.query(AgentTaskRun).count() == 0
    assert db.query(AgentRun).count() == 0
    assert db.query(ChatMessage).count() == 0
    assert db.query(EvidenceItem).count() == 0
    assert db.query(KnowledgeGraphEdge).count() == 0
    assert db.query(ProjectIndexState).count() == 0
    # 工作区保留，但必须把被删仓库从 repo_ids 中摘掉（不留悬空 id）
    assert db.get(Workspace, workspace.id).repo_ids == []
