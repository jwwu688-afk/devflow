"""仓库数据同步（sync）的测试。

验证行为：sync_repository 只负责把 GitHub 上的「事实数据」（issue / PR /
流水线）落进关系库，**不负责**向量化与 RAG 建索引。本用例通过在 vector_store
上装「一碰就炸」的假实现，来证明同步流程与 embedding/Milvus 完全解耦——
同步应当是快操作，不能因为向量服务不可用就失败。
"""

import os

# 必须在导入 app.db.models 之前设置：模型/引擎在导入期就会读取 DATABASE_URL，
# 这里指向内存 SQLite，保证测试不碰真实库。
os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import Base, Document, Issue, Repository
from app.schemas.repos import RepoSyncRequest
from app.services.repos_sync import sync_repository


def _db_session():
    """建一个纯内存库并建表，返回一个 Session。

    每个测试都用独立的内存引擎，互不污染，也不依赖本机数据库。
    """
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


@pytest.mark.asyncio
async def test_sync_repository_persists_facts_without_triggering_rag(monkeypatch) -> None:
    """测的是「同步只写事实、不触发 RAG」这条边界。

    场景：把 list_issues 替换成返回一条真实结构 issue 的假函数；同时把
    embed_texts / upsert_milvus_documents / delete_milvus_documents 全部替换成
    「调用即抛 AssertionError」的哨兵。若同步流程里任何一处偷偷调用了向量层，
    本测试会立刻失败。
    期望：status 为 completed；synced 计数为 issues=1、其余两类为 0；
    issue 字段正确入库；Document 表行数为 0（说明没有产生待索引文档）。
    重要性：把同步与索引解耦，才能让「接入仓库」秒级完成、向量服务故障时
    也不阻塞基础功能；否则一个 embedding 超时就会拖垮整个同步任务。
    """
    # 假数据刻意贴近 GitHub 真实响应结构（含 labels/user/时间戳等嵌套字段），
    # 这样字段映射逻辑（嵌套 dict → 扁平列）也能被真实覆盖到，而不是只测了空值路径。
    async def fake_list_issues(*_args, **_kwargs):
        return [
            {
                "id": 7800,
                "number": 78,
                "title": "Error: Could not import 'app'.",
                "body": "Realistic issue body",
                "state": "open",
                "labels": [{"name": "bug"}],
                "user": {"login": "contributor"},
                "assignees": [],
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-02T00:00:00Z",
                "closed_at": None,
            }
        ]

    def fail_if_vector_runtime_is_used(*_args, **_kwargs):
        """哨兵函数：被调用即说明同步路径误用了向量层。"""
        raise AssertionError("GitHub fact sync must not depend on embeddings or Milvus")

    # 只替换 issue 拉取；向量层的三个出口全部换成哨兵，任何一条被触发都算失败
    monkeypatch.setattr("app.services.repos_sync.list_issues", fake_list_issues)
    monkeypatch.setattr("app.services.rag.vector_store.embed_texts", fail_if_vector_runtime_is_used)
    monkeypatch.setattr("app.services.rag.vector_store.delete_milvus_documents", fail_if_vector_runtime_is_used)
    monkeypatch.setattr("app.services.rag.vector_store.upsert_milvus_documents", fail_if_vector_runtime_is_used)

    db = _db_session()
    # 先落一条 Repository，因为 Issue.repo_id 是外键，需要一个父记录
    repo = Repository(
        owner="openai",
        name="openai-quickstart-python",
        full_name="openai/openai-quickstart-python",
        provider="github",
        api_base_url="https://api.github.com",
    )
    db.add(repo)
    db.commit()

    # 关掉 PR 与流水线两类，只留 issue，把断言范围收窄到单一数据源
    result = await sync_repository(
        db,
        repo,
        RepoSyncRequest(sync_pull_requests=False, sync_workflow_runs=False, limit=1),
    )

    # 用 .one() 同时验证「恰好一条」：多了说明重复写入，少了说明没入库
    issue = db.query(Issue).filter(Issue.repo_id == repo.id).one()
    assert result.status == "completed"
    # 计数如实反映只同步了 issue 一类
    assert result.synced == {"issues": 1, "pull_requests": 0, "workflow_runs": 0}
    # 字段级核对：编号与标题必须原样保留（标题里带引号，顺带验证转义没问题）
    assert issue.number == 78
    assert issue.title == "Error: Could not import 'app'."
    # 核心断言：同步不产出 Document（即不进入 RAG 索引流程）
    assert db.query(Document).filter(Document.repo_id == repo.id).count() == 0
