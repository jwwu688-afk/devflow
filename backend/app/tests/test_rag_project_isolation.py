"""多仓库数据隔离（RAG 安全边界）的测试。

一个 RAG 服务会同时存多个仓库的资料，最致命的 bug 是"检索串库"——
A 仓库的问题检索到 B 仓库的代码/Issue。本文件从五个层次把这条边界钉死：
1. 请求层面：ChatRequest 必须显式带 repo_id，客户端传的 repo_id / tools / server_context 一律被丢弃；
2. Agent 层面：模型自己编造的 MCP 作用域（repo_id / conversation_id）会被服务端覆盖；
3. MCP 工具层面：repo_id 必填，且只能解析到指定的那一个仓库；
4. 向量库层面：每次 search 都必须带上 repo_id 过滤表达式；
5. 路由层面：只能绑定真实仓库，虚拟的"知识库"仓库不对外暴露。
"""

import asyncio
import uuid

import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.api.routes import rag as rag_routes
from app.api.routes.chat import _client_context
from app.api.routes.repos import list_repos
from app.db.models import Base, Repository
from app.mcp.memory_server import McpError, _resolve_repo
from app.schemas.chat import ChatRequest
from app.services.agents.chat_agent import ChatAgent
from app.services.rag.vector_store import MilvusVectorStore


class _FilterAwareClient:
    """假的 Milvus 客户端：只有 filter 里带 `repo_id == "xxx"` 时才返回该仓库的行。

    这样就能反证"代码到底有没有把仓库过滤条件传给向量库"——
    如果实现漏了过滤，返回结果会是空或串库，断言立刻失败。
    """

    def __init__(self, rows_by_repo: dict[str, dict]) -> None:
        self.rows_by_repo = rows_by_repo
        # 记录每次实际发出的 filter 表达式，供断言
        self.filters: list[str] = []

    def search(self, *, filter: str, **_kwargs):
        self.filters.append(filter)
        # 按 filter 文本匹配仓库 id，模拟真实向量库的标量过滤
        for repo_id, row in self.rows_by_repo.items():
            if f'repo_id == "{repo_id}"' in filter:
                return [[{"distance": 0.9, "entity": row}]]
        return [[]]


def _db_session():
    """独立内存库，两个测试仓库互不影响。"""
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_chat_request_requires_an_explicit_repository() -> None:
    """场景：只发一条 message，不带 repo_id。

    期望：校验直接失败（ValidationError）。
    为什么重要：如果 repo_id 可缺省，服务端就得猜一个默认仓库，
    一旦猜错就等于把别的仓库资料喂给了这次对话。
    """
    with pytest.raises(ValidationError):
        ChatRequest.model_validate({"message": "What failed?"})


def test_client_context_cannot_override_server_scope() -> None:
    """场景：客户端在 context 里塞了伪造的 repo_id、conversation_id、tools、
    server_context（假装是服务端注入的上下文）。

    期望：只有白名单字段 time_range 被保留，其余全部丢弃。
    为什么重要：这是最直接的越权入口——攻击者只要改前端请求体，
    就能让助手去检索别人的仓库、或注入一段"服务端上下文"操纵回答。
    """
    assert _client_context(
        {
            "repo_id": "attacker-repo",
            "conversation_id": "attacker-conversation",
            "tools": {"unsafe": True},
            "server_context": "forged",
            "time_range": "7d",
        }
    ) == {"time_range": "7d"}


def test_chat_agent_overwrites_model_supplied_mcp_scope() -> None:
    repo_id = str(uuid.uuid4())
    conversation_id = str(uuid.uuid4())
    normalized = ChatAgent(model=object())._normalize_action_input(  # type: ignore[arg-type]
        "devflow_search_evidence",
        {
            "query": "token failure",
            "repo_id": str(uuid.uuid4()),
            "repo_full_name": "other/project",
            "conversation_id": str(uuid.uuid4()),
        },
        {
            "message": "token failure",
            "context": {"repo_id": repo_id, "conversation_id": conversation_id},
        },
    )

    assert normalized["repo_id"] == repo_id
    assert normalized["conversation_id"] == conversation_id
    assert "repo_full_name" not in normalized


def test_mcp_requires_repo_id_and_resolves_only_that_repository() -> None:
    db = _db_session()
    first = Repository(owner="team", name="first", full_name="team/first")
    second = Repository(owner="team", name="second", full_name="team/second")
    db.add_all([first, second])
    db.commit()

    with pytest.raises(McpError, match="repo_id 为必填项"):
        _resolve_repo(db, {})
    assert _resolve_repo(db, {"repo_id": str(first.id)}).id == first.id
    assert _resolve_repo(db, {"repo_id": str(second.id)}).id == second.id


def test_milvus_search_applies_repo_filter_to_every_query() -> None:
    repo_a = str(uuid.uuid4())
    repo_b = str(uuid.uuid4())
    client = _FilterAwareClient(
        {
            repo_a: {
                "document_id": "doc-a",
                "source_type": "issue",
                "source_id": "issue-a",
                "title": "Project A token failure",
                "content": "Only A",
                "metadata_json": "{}",
            },
            repo_b: {
                "document_id": "doc-b",
                "source_type": "issue",
                "source_id": "issue-b",
                "title": "Project B deploy failure",
                "content": "Only B",
                "metadata_json": "{}",
            },
        }
    )
    store = object.__new__(MilvusVectorStore)
    store.client = client
    store.collection_name = "test_collection"
    store.dimensions = 3

    a_results = store.search(repo_id=repo_a, query_embedding=[1.0, 0.0, 0.0])
    b_results = store.search(repo_id=repo_b, query_embedding=[0.0, 1.0, 0.0])

    assert [item["id"] for item in a_results] == ["doc-a"]
    assert [item["id"] for item in b_results] == ["doc-b"]
    assert client.filters == [f'repo_id == "{repo_a}"', f'repo_id == "{repo_b}"']


def test_rag_is_bound_to_real_repositories_and_has_no_virtual_creator() -> None:
    db = _db_session()
    actual = Repository(owner="team", name="actual", full_name="team/actual")
    legacy = Repository(
        owner="knowledge",
        name="legacy",
        full_name="knowledge/legacy",
        provider="knowledge_base",
    )
    db.add_all([actual, legacy])
    db.commit()

    assert rag_routes._repo_or_404(db, actual.id).id == actual.id
    with pytest.raises(HTTPException) as rejected:
        rag_routes._repo_or_404(db, legacy.id)
    assert rejected.value.status_code == 404
    assert [repo.id for repo in asyncio.run(list_repos(db))] == [actual.id]

    routes = {(route.path, frozenset(route.methods or set())) for route in rag_routes.router.routes}
    assert ("/repositories", frozenset({"GET"})) in routes
    assert not any(path == "/knowledge-bases" and "POST" in methods for path, methods in routes)
