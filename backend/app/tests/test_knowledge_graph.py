"""知识图谱重建与查询的测试。

图谱把散落在各表里的实体（Issue、PR、被改动的代码文件、团队成员、记忆笔记）
连成一张关系网，供前端"项目全景图"展示、也供检索时做多跳扩展。
这里验证：重建时能从 PR 正文 / 改动文件 / Issue 指派人 / 笔记引用等线索中
自动推导出显式边（resolves、touches_file、assigned_to、cites），
并且按中心节点 + 深度查询时能取到正确的子图。
"""

import os
import uuid

# 必须先于 app.* 的导入设置，db 模块导入时会读取该变量
os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import Base, Document, Issue, PRFile, PullRequest, Repository
from app.services.knowledge_graph import knowledge_graph_payload, rebuild_knowledge_graph


def _db_session():
    """独立内存库 + 建表，保证每个用例的数据互不可见。"""
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_rebuild_knowledge_graph_creates_explicit_project_edges() -> None:
    """场景：一个 PR 正文写着 "Fixes #12"、改动了 backend/app/auth.py；
    对应 Issue #12 指派给 alice，团队文档里也有 alice；另有一条记忆笔记正文提到 "See #12"。

    期望：重建后至少 4 条边，且必须包含
    pull_request→issue(resolves)、pull_request→code_file(touches_file)、
    issue→team_member(assigned_to)，以及笔记→某实体(cites)；
    以 PR 为中心、深度 1 查询时能取到该 PR 节点。
    为什么重要：这些边全靠文本线索（"Fixes #12"、"See #12"）和字段关联（assignees、PRFile）推导，
    漏掉任何一种，"为什么这个 PR 修了哪个 Issue"这类跨实体问题就答不出来。
    """
    db = _db_session()
    repo = Repository(owner="local", name="demo", full_name="local/demo")
    db.add(repo)
    db.flush()

    # 用显式 uuid 作为 source_id，保证文档与后续节点能对上
    member_id = uuid.uuid4()
    note_id = uuid.uuid4()
    # Issue 的 assignees 是导出 issue→team_member(assigned_to) 边的来源
    issue = Issue(repo_id=repo.id, number=12, title="Auth refresh fails", state="open", assignees=["alice"])
    # PR 正文里的 "Fixes #12" 是导出 pull_request→issue(resolves) 边的来源
    pr = PullRequest(repo_id=repo.id, number=4, title="Refresh tokens", state="closed", body="Fixes #12", author="alice")
    db.add_all(
        [
            Document(
                repo_id=repo.id,
                source_type="team_member",
                source_id=member_id,
                title="Alice",
                content="name: alice",
                # meta 里的 name 用于和 Issue.assignees 里的 "alice" 对上号
                meta={"name": "alice", "role": "Backend"},
            ),
            Document(
                repo_id=repo.id,
                source_type="memory_note",
                source_id=note_id,
                title="Auth convention",
                # 正文里同时出现文件路径和 "#12"，用来验证笔记会分别 cite 代码文件与 Issue
                content="Keep token refresh in backend/app/auth.py. See #12.",
                meta={"display_title": "Auth convention"},
            ),
            issue,
            pr,
        ]
    )
    db.flush()
    db.add(PRFile(pr_id=pr.id, filename="backend/app/auth.py", status="modified"))
    db.commit()

    result = rebuild_knowledge_graph(db, repo)
    # 以 PR 为中心查深度 1 的子图（前端点开某个节点时的视角）
    graph = knowledge_graph_payload(db, repo, center_type="pull_request", center_id=str(pr.id), depth=1)
    # 不带中心的完整全景图，用于整体统计
    overview = knowledge_graph_payload(db, repo)
    # 用三元组做集合，断言"某种类型的边存在"而不依赖顺序
    edge_keys = {(edge["from_type"], edge["to_type"], edge["relation"]) for edge in graph["edges"]}

    assert result["edges"] >= 4
    # PR 正文 "Fixes #12" → 解决关系
    assert ("pull_request", "issue", "resolves") in edge_keys
    # PRFile 表 → 改动文件关系
    assert ("pull_request", "code_file", "touches_file") in edge_keys
    # Issue.assignees → 指派关系
    assert ("issue", "team_member", "assigned_to") in edge_keys
    # 笔记引用关系在全景图里也要存在（不局限于 PR 中心子图）
    assert any(edge["from_type"] == "memory_note" and edge["relation"] == "cites" for edge in overview["edges"])
    # 中心节点本身必须出现在返回的 nodes 里，否则前端画不出焦点
    assert any(node["type"] == "pull_request" and node["id"] == str(pr.id) for node in graph["nodes"])
