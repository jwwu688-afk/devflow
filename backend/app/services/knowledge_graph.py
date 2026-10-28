"""项目知识图谱：把 issue / PR / CI / 文档 / 记忆 / 代码文件连成一张关系网。

建图方式是完全**规则化**的（不做 LLM 抽取）：
- 结构化关系直接连，例如 PR 改了哪个文件、issue 指派给谁；
- 文本关系靠正则在正文里找引用，例如 "fixes #12"、"PR #34"、文件路径出现。
好处是结果确定、可解释、可重算；代价是文本里没写清楚的关系就抽不到，
所以每条边的 confidence 都标了来源可靠度（显式字段 1.0 > 正则命中 0.75~0.85）。

图谱是「重建式」的：每次 rebuild 先删该仓库所有边再全量生成，
因为关系是纯派生的，没有需要保留的用户修改。
"""

import json
import re
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from app.db.models import (
    AgentRun,
    ConversationMemory,
    Document,
    Issue,
    KnowledgeGraphEdge,
    PRFile,
    PullRequest,
    Repository,
    ThreadMemory,
    WorkflowRun,
)

# 前端一次最多渲染这么多节点：再多力导向图会卡，交互也没法看清。
# 超出部分会被截断，并在 stats.truncated 里告知前端「还有更多」。
GRAPH_NODE_LIMIT = 72
# "#123" 形式的 issue 引用。前置否定断言 (?<![A-Za-z]) 避免把 "abc#12" 或十六进制色值误判。
ISSUE_REF_RE = re.compile(r"(?<![A-Za-z])#(\d+)\b")
# "fixes #12" / "closes #12" / "resolved #12" 这类「会关闭 issue」的关键词，
# 比普通提及更强，用来区分 resolves 和 mentions 两种关系。
CLOSING_ISSUE_RE = re.compile(
    r"\b(?:fix(?:e[sd])?|close[sd]?|resolve[sd]?)\s+(?:issue\s+)?#(\d+)\b",
    re.IGNORECASE,
)
# "PR #34" / "pull request 34" 形式的 PR 引用。
PR_REF_RE = re.compile(r"\b(?:PR|pull request)\s*#?(\d+)\b", re.IGNORECASE)

# 节点类型别名归一化表：历史数据/不同来源用了各种叫法（pr、ci、weekly_report…），
# 统一收敛成少数几种标准类型，否则前端要处理十几套类型判断。
NODE_TYPE_ALIASES = {
    "pr": "pull_request",
    "pull_request": "pull_request",
    "workflow_run": "ci_run",
    "ci": "ci_run",
    "ci_run": "ci_run",
    "memory": "memory_note",
    "memory_note": "memory_note",
    "knowledge_file": "document",
    "weekly_report": "document",
    "project_doc": "document",
    "project_manifest": "document",
    "project_overview": "document",
    "document": "document",
    "code_file": "code_file",
    "team_member": "team_member",
    "issue": "issue",
    "session": "session_decision",
    "session_decision": "session_decision",
}

# 这几种 source_type 都渲染成统一的 document 节点（只是副标题不同）。
DOCUMENT_NODE_TYPES = {"knowledge_file", "weekly_report", "project_doc", "project_manifest", "project_overview"}


@dataclass
class DocumentGroup:
    """同一 (source_type, source_id) 下多个分块合并成的一个「文档节点」。

    Document 表是按块存的（长文档会被切成多段），建图时需要先按逻辑文档聚合，
    否则同一份文档会在图上出现 N 个节点，边也会重复。
    """

    source_type: str
    source_id: str
    title: str
    content: str
    metadata: dict[str, Any]
    created_at: Any = None


def normalize_node_type(value: str | None) -> str:
    """把各种别名统一成标准节点类型；未知类型原样返回（不丢弃，便于排查新数据）。"""
    key = (value or "").strip()
    return NODE_TYPE_ALIASES.get(key, key)


def rebuild_knowledge_graph(db: Session, repo: Repository) -> dict[str, int]:
    """全量重建某仓库的图谱，返回 {edges, nodes} 统计。

    先 DELETE 再重建：图谱是派生数据，重算比「做增量 diff」简单可靠得多。
    注意这里自己 commit（不等外层）：这是独立的批量作业，语义上就该立即落地。
    """
    db.query(KnowledgeGraphEdge).filter(KnowledgeGraphEdge.repo_id == repo.id).delete(synchronize_session=False)
    edges = _generate_edges(db, repo.id)
    db.add_all(
        KnowledgeGraphEdge(
            repo_id=repo.id,
            from_type=edge["from_type"],
            from_id=edge["from_id"],
            to_type=edge["to_type"],
            to_id=edge["to_id"],
            relation=edge["relation"],
            confidence=edge["confidence"],
            source=edge["source"],
            meta=edge["metadata"],
        )
        for edge in edges
    )
    db.commit()
    # 节点数在 commit 之后统计：_collect_nodes 会把边引用到的「游离节点」也补进来。
    nodes = _collect_nodes(db, repo.id)
    return {"edges": len(edges), "nodes": len(nodes)}


def knowledge_graph_payload(
    db: Session,
    repo: Repository,
    *,
    center_type: str | None = None,
    center_id: str | None = None,
    depth: int = 1,
) -> dict[str, Any]:
    """组装给前端的图谱数据（可只取某个节点周围的子图）。

    center_type/center_id 为空时返回全局概览（按优先级取前 GRAPH_NODE_LIMIT 个节点）；
    指定中心节点时以它为中心做 BFS，取 depth 层内的邻居。
    """
    edge_count = db.query(KnowledgeGraphEdge).filter(KnowledgeGraphEdge.repo_id == repo.id).count()
    if edge_count == 0:
        # 懒建图：仓库刚接入、还没人点过图谱时，第一次访问就地生成，
        # 省掉一个「必须手动触发建图」的前置步骤。
        rebuild_knowledge_graph(db, repo)

    nodes_by_key = _collect_nodes(db, repo.id)
    edge_rows = db.query(KnowledgeGraphEdge).filter(KnowledgeGraphEdge.repo_id == repo.id).all()
    edges = [_edge_payload(row) for row in edge_rows]
    center_key = None
    resolved_center_type = normalize_node_type(center_type)
    if resolved_center_type and center_id:
        center_key = _node_key(resolved_center_type, center_id)
        # 中心节点可能不在节点表里（例如前端拿了一个已删除的 id），
        # 这里补一个占位节点，避免「点了却什么都没显示」。
        nodes_by_key.setdefault(
            center_key,
            _fallback_node(resolved_center_type, center_id, highlighted=True),
        )

    # depth 夹在 1..2：3 层以上子图会迅速膨胀到几百个节点，前端渲染不动。
    selected_keys, truncated = _select_subgraph(nodes_by_key, edges, center_key, max(1, min(depth, 2)))
    # 只保留两端都在选中集合里的边，否则会出现「连到画布外节点」的悬空线。
    selected_edges = [
        edge for edge in edges if _node_key(edge["from_type"], edge["from_id"]) in selected_keys and _node_key(edge["to_type"], edge["to_id"]) in selected_keys
    ]
    for edge in selected_edges:
        if edge["relation"] == "used_as_evidence":
            # 被 AI 回答当作证据引用过的节点高亮：这是用户最关心的「哪些资料真的被用了」。
            nodes_by_key[_node_key(edge["to_type"], edge["to_id"])]["highlighted"] = True

    return {
        "repo_id": repo.id,
        "center_type": resolved_center_type or None,
        "center_id": center_id,
        "depth": depth,
        # 按 selected_keys 的顺序输出（BFS/优先级顺序），前端可以据此做布局初始位置。
        "nodes": [nodes_by_key[key] for key in selected_keys if key in nodes_by_key],
        "edges": selected_edges,
        "stats": {
            "total_nodes": len(nodes_by_key),
            "total_edges": len(edges),
            "visible_nodes": len(selected_keys),
            "visible_edges": len(selected_edges),
            # 截断标记：前端据此提示「当前仅展示部分关系」。
            "truncated": truncated,
        },
    }


def _generate_edges(db: Session, repo_id: uuid.UUID) -> list[dict[str, Any]]:
    """扫描该仓库的所有实体，生成边列表（此时还没写库）。

    返回的是 dict 列表而不是 ORM 对象：中间要用 dict 做去重和合并置信度，
    等去重完再统一转成 ORM 行更省事。
    """
    issues = db.query(Issue).filter(Issue.repo_id == repo_id).all()
    prs = db.query(PullRequest).filter(PullRequest.repo_id == repo_id).all()
    runs = db.query(WorkflowRun).filter(WorkflowRun.repo_id == repo_id).all()
    # PRFile 没有 repo_id 字段，只能 join 到 PullRequest 上再过滤。
    pr_files = db.query(PRFile).join(PullRequest, PRFile.pr_id == PullRequest.id).filter(PullRequest.repo_id == repo_id).all()
    docs = _document_groups(db, repo_id)

    # 建几张查找表，把「文本里的编号/名字/路径」映射到实体 id，避免每次匹配都查库。
    issues_by_number = {issue.number: issue for issue in issues}
    prs_by_number = {pr.number: pr for pr in prs}
    team_by_name = _team_members_by_name(docs)
    code_path_to_id = _code_path_to_node_id(pr_files)
    # 用 (from_type, from_id, to_type, to_id, relation) 做键去重：
    # 同一对实体可能被多条规则命中，只保留一条边并把置信度取最大。
    edges: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}

    def add_edge(
        from_type: str,
        from_id: str,
        to_type: str,
        to_id: str,
        relation: str,
        *,
        confidence: float = 1.0,
        source: str = "explicit",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """登记一条边：自动归一化类型、丢弃自环与空 id、并做去重合并。

        默认 confidence=1.0 / source="explicit"：调用方只在「靠正则猜出来的」关系上
        显式降低置信度并标注来源，这样默认值就代表「来自结构化字段，可信」。
        """
        from_type = normalize_node_type(from_type)
        to_type = normalize_node_type(to_type)
        # 自环（issue → 自己）没有信息量，直接丢弃，否则图上会出现小圈圈。
        if not from_id or not to_id or (from_type == to_type and from_id == to_id):
            return
        key = (from_type, str(from_id), to_type, str(to_id), relation)
        existing = edges.get(key)
        if existing:
            # 重复命中：保留更高的置信度（多个规则都说有这条关系 → 更可信），
            # 并把 metadata 合并，行号/编号等线索都不丢。
            existing["confidence"] = max(existing["confidence"], confidence)
            existing["metadata"].update(metadata or {})
            return
        edges[key] = {
            "id": _edge_id(from_type, str(from_id), to_type, str(to_id), relation),
            "from_type": from_type,
            "from_id": str(from_id),
            "to_type": to_type,
            "to_id": str(to_id),
            "relation": relation,
            "confidence": confidence,
            "source": source,
            "metadata": metadata or {},
        }

    for issue in issues:
        # 指派关系来自结构化字段，置信度默认 1.0。
        for assignee in issue.assignees or []:
            member_id = team_by_name.get(str(assignee).strip().lower())
            if member_id:
                add_edge("issue", str(issue.id), "team_member", member_id, "assigned_to", metadata={"assignee": assignee})
        # 正文里提到的 issue/PR/文件，属于弱关系，由 _add_text_reference_edges 统一处理。
        _add_text_reference_edges(
            add_edge,
            "issue",
            str(issue.id),
            f"{issue.title}\n{issue.body or ''}",
            issues_by_number,
            prs_by_number,
            code_path_to_id,
            relation_for_issue="mentions",
            relation_for_pr="mentions",
            relation_for_code="mentions",
        )

    for pr in prs:
        # 分支名也算进文本：很多人会在分支名里写 issue 号（如 fix/123-login）。
        text = f"{pr.title}\n{pr.body or ''}\n{pr.head_branch or ''}\n{pr.base_branch or ''}"
        closing_refs = _closing_issue_refs(text)
        # 带 fix/close/resolve 关键词的引用是「会关闭该 issue」的强关系，单独建 resolves。
        for number in closing_refs:
            issue = issues_by_number.get(number)
            if issue:
                add_edge("pull_request", str(pr.id), "issue", str(issue.id), "resolves", metadata={"issue_number": number})
        # 用集合差集排除掉已经算作 resolves 的编号，避免同一对实体既 resolves 又 mentions。
        for number in _issue_refs(text) - closing_refs:
            issue = issues_by_number.get(number)
            if issue:
                add_edge("pull_request", str(pr.id), "issue", str(issue.id), "mentions", metadata={"issue_number": number})
        author_member = team_by_name.get(str(pr.author or "").strip().lower())
        if author_member:
            # 作者关系在结构化字段里没有直接外键，只能靠名字匹配到 team_member 文档。
            add_edge("pull_request", str(pr.id), "team_member", author_member, "mentions", metadata={"author": pr.author})

    for pr_file in pr_files:
        # 优先复用 code_path_to_id 里已算好的节点 id，没有就现算（结果一样，只是省一次函数调用）。
        code_id = code_path_to_id.get(pr_file.filename) or _path_node_id(pr_file.filename)
        add_edge("pull_request", str(pr_file.pr_id), "code_file", code_id, "touches_file", metadata={"path": pr_file.filename, "status": pr_file.status})

    for run in runs:
        # 只有失败的 CI 才建关系：成功的 run 对「出问题查根源」没有价值，
        # 全部建边会让图谱被大量无关节点淹没。
        if (run.conclusion or "").lower() != "failure":
            continue
        text = f"{run.name}\n{run.logs_text or ''}"
        for number in _pr_refs(text):
            pr = prs_by_number.get(number)
            if pr:
                # 日志里提到 PR 号，是较强的信号（CI 基本就是为这个 PR 跑的），给 0.85。
                add_edge("ci_run", str(run.id), "pull_request", str(pr.id), "failed_in_ci", confidence=0.85, metadata={"pr_number": number})
        for number in _issue_refs(text):
            issue = issues_by_number.get(number)
            if issue:
                # 日志里出现 issue 号更像「顺手提到」，可靠性低于 PR，给 0.75。
                add_edge("ci_run", str(run.id), "issue", str(issue.id), "failed_in_ci", confidence=0.75, metadata={"issue_number": number})

    for doc in docs:
        # team_member 文档本身是「人」，上面已经作为目标节点用过，不需要再从它出发建边。
        if doc.source_type == "team_member":
            continue
        # 记忆笔记用 cites（引用），普通文档用 mentions（提及）——关系语义不同，前端展示也不同。
        from_type = "memory_note" if doc.source_type == "memory_note" else "document"
        relation = "cites" if doc.source_type == "memory_note" else "mentions"
        # metadata 也参与文本匹配：有些来源路径/标题只记在 metadata 里。
        # default=str 兜住不可序列化的值（如 datetime），否则 json.dumps 会抛异常。
        text = f"{doc.title}\n{doc.content}\n{json.dumps(doc.metadata, ensure_ascii=False, default=str)}"
        _add_text_reference_edges(
            add_edge,
            from_type,
            doc.source_id,
            text,
            issues_by_number,
            prs_by_number,
            code_path_to_id,
            relation_for_issue=relation,
            relation_for_pr=relation,
            relation_for_code="cites" if doc.source_type == "memory_note" else "documents",
        )

    for decision in _decision_nodes(db, repo_id):
        # 会话里沉淀的「决策」也要参与建图：它是人做的结论，往往同时在讲 issue 和文件。
        text = str(decision["metadata"].get("content") or decision["title"])
        _add_text_reference_edges(
            add_edge,
            "session_decision",
            decision["id"],
            text,
            issues_by_number,
            prs_by_number,
            code_path_to_id,
            relation_for_issue="cites",
            relation_for_pr="cites",
            relation_for_code="cites",
        )
        # 记忆里自带的 citations 是结构化引用（不是文本匹配），置信度给 0.9。
        for citation in decision["metadata"].get("citations") or []:
            target_type = normalize_node_type(str(citation.get("type") or citation.get("source_type") or ""))
            target_id = str(citation.get("id") or citation.get("source_id") or "")
            if target_type and target_id:
                add_edge("session_decision", decision["id"], target_type, target_id, "cites", confidence=0.9, source="memory_citation")

    # 只取最近 25 次 Agent 运行：历史太久的工具调用轨迹对当前项目状态参考价值低，
    # 而且全量参与会让每次重建都要解析大量 JSON。
    for run in db.query(AgentRun).filter(AgentRun.repo_id == repo_id).order_by(AgentRun.created_at.desc()).limit(25):
        source_id = f"agent_run:{run.id}"
        # tool_calls 是嵌套结构，用递归找出所有「像引用」的 dict（同时有 type/source_type 和 id/source_id）。
        for citation in _extract_citation_like_values(run.tool_calls or []):
            target_type = normalize_node_type(str(citation.get("type") or citation.get("source_type") or ""))
            target_id = str(citation.get("id") or citation.get("source_id") or "")
            if target_type and target_id:
                # used_as_evidence：这条资料真的被 AI 用过，前端会高亮它。
                add_edge("session_decision", source_id, target_type, target_id, "used_as_evidence", confidence=0.75, source="tool_call_trace")

    # 用 dict 去重后取出所有边；顺序不保证，但边本身没有顺序语义。
    return list(edges.values())


def _add_text_reference_edges(
    add_edge: Any,
    from_type: str,
    from_id: str,
    text: str,
    issues_by_number: dict[int, Issue],
    prs_by_number: dict[int, PullRequest],
    code_path_to_id: dict[str, str],
    *,
    relation_for_issue: str,
    relation_for_pr: str,
    relation_for_code: str,
) -> None:
    """在一段自由文本里找 issue / PR / 文件路径引用，并建成边。

    三种关系的名字由调用方传入（mentions/cites/documents），因为「同一处引用」
    在不同来源里语义不同：文档提到是 mentions，记忆笔记提到是 cites。

    add_edge 以参数形式传入（而不是模块级函数），是为了复用 _generate_edges 里的
    去重字典和置信度合并逻辑——它是闭包，只有传进来才能写进同一个 edges 字典。
    """
    for number in _issue_refs(text):
        issue = issues_by_number.get(number)
        if issue:
            add_edge(from_type, from_id, "issue", str(issue.id), relation_for_issue, confidence=0.75, metadata={"issue_number": number})
    for number in _pr_refs(text):
        pr = prs_by_number.get(number)
        if pr:
            # PR 引用形式（"PR #12"）比裸 #12 更明确，所以置信度略高（0.8 vs 0.75）。
            add_edge(from_type, from_id, "pull_request", str(pr.id), relation_for_pr, confidence=0.8, metadata={"pr_number": number})
    for path, code_id in code_path_to_id.items():
        # 长度 >= 6 才匹配：避免 "a.py"、"go.mod" 这类短路径在任意文本里误命中。
        # 这里用子串包含而不是正则，因为路径里的 / 和 . 在正则里都得转义，不划算。
        if len(path) >= 6 and path in text:
            add_edge(from_type, from_id, "code_file", code_id, relation_for_code, confidence=0.8, metadata={"path": path})


def _collect_nodes(db: Session, repo_id: uuid.UUID) -> dict[str, dict[str, Any]]:
    """收集该仓库的全部图节点，返回 key("类型:id") → 节点 dict。

    节点是从多张表实时拼出来的（不单独存节点表）：这样节点永远和源数据一致，
    不会出现「实体删了但节点还在」的脏数据；代价是每次看图都要多查几次库。
    """
    nodes: dict[str, dict[str, Any]] = {}

    def put(node_type: str, node_id: str, title: str, subtitle: str = "", metadata: dict[str, Any] | None = None, highlighted: bool = False) -> None:
        """登记一个节点；已存在则合并 metadata 并保留高亮标记（不会被后来的 False 覆盖）。"""
        normalized = normalize_node_type(node_type)
        key = _node_key(normalized, str(node_id))
        if key in nodes:
            nodes[key]["metadata"].update(metadata or {})
            nodes[key]["highlighted"] = nodes[key]["highlighted"] or highlighted
            return
        nodes[key] = {
            "id": str(node_id),
            "type": normalized,
            # 标题为空时兜底显示 id，保证图上不会出现无标签的节点。
            "title": title or str(node_id),
            "subtitle": subtitle or "",
            "metadata": metadata or {},
            "highlighted": highlighted,
        }

    # 节点标题里带上编号（"#12 登录失败"），因为同一标题可能对应多个 issue，编号才是唯一标识。
    for issue in db.query(Issue).filter(Issue.repo_id == repo_id).all():
        put("issue", str(issue.id), f"#{issue.number} {issue.title}", issue.state, {"number": issue.number, "labels": issue.labels or [], "assignees": issue.assignees or []})
    for pr in db.query(PullRequest).filter(PullRequest.repo_id == repo_id).all():
        # 已合并的 PR 用 merged 覆盖 state（GitHub 的已合并 PR 仍返回 state=closed，看不出合并）。
        subtitle = "merged" if pr.merged_at else pr.state
        put("pull_request", str(pr.id), f"#{pr.number} {pr.title}", subtitle, {"number": pr.number, "author": pr.author, "head_branch": pr.head_branch})
    for run in db.query(WorkflowRun).filter(WorkflowRun.repo_id == repo_id).all():
        # 结论（success/failure）比状态（completed）更有信息量，优先展示结论。
        put("ci_run", str(run.id), run.name, run.conclusion or run.status, {"status": run.status, "conclusion": run.conclusion, "html_url": run.html_url})

    docs = _document_groups(db, repo_id)
    for doc in docs:
        # 人、记忆、普通文档三类节点的展示字段不同，所以分开处理；其余类型（如 code_file）忽略。
        if doc.source_type == "team_member":
            put("team_member", doc.source_id, doc.metadata.get("name") or doc.title, doc.metadata.get("role") or "", doc.metadata)
        elif doc.source_type == "memory_note":
            put("memory_note", doc.source_id, doc.title, "memory note", doc.metadata)
        elif doc.source_type in DOCUMENT_NODE_TYPES:
            # `dict | dict` 生成新字典：不能就地改 doc.metadata，否则会影响后续调用方的数据。
            put("document", doc.source_id, doc.metadata.get("display_title") or doc.title, doc.source_type, doc.metadata | {"source_type": doc.source_type})

    for pr_file in db.query(PRFile).join(PullRequest, PRFile.pr_id == PullRequest.id).filter(PullRequest.repo_id == repo_id).all():
        # 同一文件被多个 PR 改过时 put 会合并 metadata，节点仍然只有一个。
        put("code_file", _path_node_id(pr_file.filename), pr_file.filename, "PR touched file", {"path": pr_file.filename, "status": pr_file.status})

    for decision in _decision_nodes(db, repo_id):
        put("session_decision", decision["id"], decision["title"], decision["subtitle"], decision["metadata"])

    for run in db.query(AgentRun).filter(AgentRun.repo_id == repo_id).order_by(AgentRun.created_at.desc()).limit(25):
        # Agent 运行也复用 session_decision 类型：它们同属「会话里产出的结论」这一类。
        put("session_decision", f"agent_run:{run.id}", f"AI answer: {run.route}", "tool evidence path", {"route": run.route, "created_at": run.created_at.isoformat() if run.created_at else None})

    for edge in db.query(KnowledgeGraphEdge).filter(KnowledgeGraphEdge.repo_id == repo_id).all():
        # 兜底：边指向的实体可能已经不在任何源表里（例如 PR 被删但边还没重建），
        # setdefault 补一个只有 id 的占位节点，防止前端画出「指向不存在节点」的边。
        from_key = _node_key(edge.from_type, edge.from_id)
        to_key = _node_key(edge.to_type, edge.to_id)
        nodes.setdefault(from_key, _fallback_node(edge.from_type, edge.from_id))
        nodes.setdefault(to_key, _fallback_node(edge.to_type, edge.to_id))

    return nodes


def _select_subgraph(
    nodes_by_key: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
    center_key: str | None,
    depth: int,
) -> tuple[list[str], bool]:
    """挑选要返回给前端的节点，返回 (key 列表, 是否被截断)。

    两种模式：
    - 有中心节点：以它为中心做 BFS，取 depth 层内的邻居，节点的先后顺序即「离中心由近到远」；
    - 无中心节点：全局概览，先按「是否参与任何边」排序，再按类型优先级，最后按标题。
    共同约束是 GRAPH_NODE_LIMIT。
    """
    if center_key and center_key in nodes_by_key:
        # 邻接表按无向图构建：看 PR 的关联时，既想看它改的文件，也想看改过这些文件的其它 PR。
        adjacency: dict[str, set[str]] = defaultdict(set)
        for edge in edges:
            a = _node_key(edge["from_type"], edge["from_id"])
            b = _node_key(edge["to_type"], edge["to_id"])
            adjacency[a].add(b)
            adjacency[b].add(a)
        selected: list[str] = []
        seen = {center_key}
        queue: deque[tuple[str, int]] = deque([(center_key, 0)])
        while queue and len(selected) < GRAPH_NODE_LIMIT:
            key, distance = queue.popleft()
            selected.append(key)
            # 到最深一层就不再扩邻居，但当前节点仍然要输出。
            if distance >= depth:
                continue
            # 邻居按标题排序后再入队：同一层内的展示顺序稳定，不会每次刷新都变。
            for nxt in sorted(adjacency.get(key, []), key=lambda item: nodes_by_key.get(item, {}).get("title", item)):
                if nxt in seen:
                    continue
                seen.add(nxt)
                queue.append((nxt, distance + 1))
        # 队列还有剩余 = 因为节点上限被提前截断。
        return selected, bool(queue)

    touched: set[str] = set()
    for edge in edges:
        touched.add(_node_key(edge["from_type"], edge["from_id"]))
        touched.add(_node_key(edge["to_type"], edge["to_id"]))
    # 排序字典序说明：0/1 表示「有没有边」（有边的优先，孤立节点最后），
    # 其次是类型优先级，最后是标题——整体是一个确定性排序，便于前端缓存和对比。
    ranked = sorted(
        nodes_by_key,
        key=lambda key: (0 if key in touched else 1, _node_priority(nodes_by_key[key]["type"]), nodes_by_key[key]["title"]),
    )
    selected = ranked[:GRAPH_NODE_LIMIT]
    return selected, len(ranked) > GRAPH_NODE_LIMIT


def _document_groups(db: Session, repo_id: uuid.UUID) -> list[DocumentGroup]:
    """把分块的 Document 按 (source_type, source_id) 聚合成逻辑文档。

    同类文档可能有多个块，这里按 chunk_index 排序后取前 4 块拼成正文：
    建图只需要「够用来正则匹配引用」的文本，取全部块会让处理时间和内存无谓膨胀。
    """
    groups: dict[tuple[str, str], list[Document]] = defaultdict(list)
    for doc in db.query(Document).filter(Document.repo_id == repo_id).order_by(Document.created_at.asc()).all():
        groups[(doc.source_type, str(doc.source_id))].append(doc)

    output: list[DocumentGroup] = []
    for (source_type, source_id), docs in groups.items():
        # 缺 chunk_index 的历史数据按 0 处理，再用 created_at 兜底排序，保证顺序稳定。
        docs.sort(key=lambda item: ((item.meta or {}).get("chunk_index", 0), item.created_at))
        first = docs[0]
        # 标题/元信息都取第一个块：同一文档的各块 metadata 基本一致，没必要合并。
        metadata = dict(first.meta or {})
        title = str(metadata.get("display_title") or first.title)
        content = "\n".join(doc.content or "" for doc in docs[:4])
        output.append(DocumentGroup(source_type, source_id, title, content, metadata, first.created_at))
    return output


def _decision_nodes(db: Session, repo_id: uuid.UUID) -> list[dict[str, Any]]:
    """把会话记忆里的 decisions 数组摊平成「决策节点」列表。

    两种记忆都要覆盖：ThreadMemory（会话内）和 ConversationMemory（跨会话汇集），
    所以用列表推导拼上前缀一起遍历，靠 prefix 区分节点 id 的来源，避免 id 撞车。
    """
    output: list[dict[str, Any]] = []
    thread_memories = db.query(ThreadMemory).filter(ThreadMemory.repo_id == repo_id).all()
    conversation_memories = db.query(ConversationMemory).filter(ConversationMemory.repo_id == repo_id).all()
    for prefix, memory in [("thread", item) for item in thread_memories] + [("conversation", item) for item in conversation_memories]:
        # decisions 是字符串数组（不是对象），enumerate 的 index 用来构造稳定 id。
        for index, decision in enumerate(memory.decisions or []):
            text = str(decision).strip()
            if not text:
                continue
            # id 形如 "thread:<memory_id>:decision:3"：memory 不变时 id 稳定，
            # 但 decisions 数组顺序一变（插入/删除）后面的 id 会整体偏移——这是可接受的，
            # 因为节点是每次重建生成的，不跨版本引用。
            output.append(
                {
                    "id": f"{prefix}:{memory.id}:decision:{index}",
                    "title": _clip(text, 72),
                    "subtitle": "session decision",
                    "metadata": {
                        "content": text,
                        "memory_id": str(memory.id),
                        "citations": memory.citations or [],
                        "sessions_incorporated": memory.sessions_incorporated,
                    },
                }
            )
    return output


def _team_members_by_name(docs: list[DocumentGroup]) -> dict[str, str]:
    """建立「人名（小写）→ team_member 节点 id」的映射，用于给 issue/PR 关联到人。

    同时索引 metadata.name 和标题：有些数据只填了其中一个，
    两个都收录可以提高匹配率（大小写不敏感是因为 GitHub 用户名本身不区分大小写）。
    """
    output: dict[str, str] = {}
    for doc in docs:
        if doc.source_type != "team_member":
            continue
        for value in [doc.metadata.get("name"), doc.title]:
            text = str(value or "").strip().lower()
            if text:
                output[text] = doc.source_id
    return output


def _code_path_to_node_id(pr_files: list[PRFile]) -> dict[str, str]:
    """建立「文件路径 → code_file 节点 id」。

    setdefault 而非直接赋值：同一路径被多个 PR 改过时保留第一次的映射，
    但节点 id 只由路径决定（_path_node_id），所以值其实是相同的，写不写回都无差别。
    """
    output: dict[str, str] = {}
    for pr_file in pr_files:
        output.setdefault(pr_file.filename, _path_node_id(pr_file.filename))
    return output


def _extract_citation_like_values(value: Any) -> list[dict[str, Any]]:
    """递归挖出嵌套结构（tool_calls 的任意层级）里所有「像引用」的 dict。

    判定条件很宽松：有 type/source_type 且 有 id/source_id 就算引用。
    为什么不按已知 schema 精确取值：工具调用的结构随工具不同而不同，
    递归找特征字段比维护一堆分类型解析逻辑更稳、更容易扩。
    """
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if ("source_type" in value or "type" in value) and ("source_id" in value or "id" in value):
            found.append(value)
        # 命中后仍然继续往子节点递归：一个引用对象里可能还嵌着别的引用（如 citations 数组）。
        for child in value.values():
            found.extend(_extract_citation_like_values(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_extract_citation_like_values(child))
    return found


def _edge_payload(row: KnowledgeGraphEdge) -> dict[str, Any]:
    """把边记录转成前端结构（id 转字符串，meta 空值归一成 {}）。"""
    return {
        "id": str(row.id),
        "from_type": row.from_type,
        "from_id": row.from_id,
        "to_type": row.to_type,
        "to_id": row.to_id,
        "relation": row.relation,
        "confidence": row.confidence,
        "source": row.source,
        "metadata": row.meta or {},
    }


def _fallback_node(node_type: str, node_id: str, highlighted: bool = False) -> dict[str, Any]:
    """给「只在边里出现、没有对应实体」的 id 造一个占位节点，保证边两端都有节点。"""
    return {
        "id": str(node_id),
        "type": normalize_node_type(node_type),
        # 标题直接用 id：至少能让用户看到它指向什么，点击后可以去后端查。
        "title": str(node_id),
        "subtitle": "referenced item",
        "metadata": {},
        "highlighted": highlighted,
    }


def _node_key(node_type: str, node_id: str) -> str:
    """节点唯一键："类型:id"。

    必须带类型，因为不同表的自增 id 会重复（issue 1 和 pr 1 是两回事）。
    """
    return f"{normalize_node_type(node_type)}:{node_id}"


def _edge_id(from_type: str, from_id: str, to_type: str, to_id: str, relation: str) -> str:
    """用 uuid5 生成确定性边 id。

    选 uuid5（基于命名空间 + 名字的哈希）而不是 uuid4：
    同样的关系每次重建都得到同一个 id，前端做增量更新/动画过渡时不会「整图重画」。
    当前实现里 id 主要给前端当 key 用，重复建边时也能天然对齐。
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"devflow-graph:{from_type}:{from_id}:{relation}:{to_type}:{to_id}"))


def _path_node_id(path: str) -> str:
    """文件节点的 id 统一加 path: 前缀，避免和 issue/PR 的数字 id 混淆。"""
    return f"path:{path}"


def _issue_refs(text: str) -> set[int]:
    """抽取文本里所有 issue 编号（含 "PR #12" 里的 12，属于有意的宽松匹配）。"""
    return {int(match.group(1)) for match in ISSUE_REF_RE.finditer(text or "")}


def _closing_issue_refs(text: str) -> set[int]:
    """抽取「会关闭 issue」的编号（fixes/closes/resolves 后跟 #N）。"""
    return {int(match.group(1)) for match in CLOSING_ISSUE_RE.finditer(text or "")}


def _pr_refs(text: str) -> set[int]:
    """抽取文本里的 PR 编号（"PR #34" / "pull request 34"）。"""
    return {int(match.group(1)) for match in PR_REF_RE.finditer(text or "")}


def _node_priority(node_type: str) -> int:
    """节点类型在全局概览里的展示优先级（数字越小越靠前）。

    顺序大致按「业务关注度」排：issue/PR 最常看，代码文件最碎所以放最后。
    未知类型给 20，排在所有已知类型之后但不丢弃。
    """
    order = {
        "issue": 0,
        "pull_request": 1,
        "ci_run": 2,
        "team_member": 3,
        "memory_note": 4,
        "session_decision": 5,
        "document": 6,
        "code_file": 7,
    }
    return order.get(node_type, 20)


def _clip(text: str, limit: int) -> str:
    """截断到 limit（含省略号）并去掉尾部空格。

    先 [:limit-1] 再拼 "..."：这样最终长度不会超过 limit，
    避免把 72 字的标题撑成 75 字。
    """
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "..."
