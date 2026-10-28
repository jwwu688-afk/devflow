"""记忆中心：召回审计 + 记忆候选的沉淀与审核。

分成两条独立链路：
1. 召回事件（RecallEvent）：每次工具检索都留一条「查了什么、命中几条」的记录，
   用于事后复盘「Agent 为什么没找到答案」——是没检索、还是检索到但没用上。
2. 记忆候选（MemoryCandidate）：会话结束时把结构化的决策/事实/任务抽出来，
   先落成 pending 待人工审核，**审核通过后才会写进知识库参与检索**。
   加这道人工闸门的原因：自动抽取难免把模型幻觉或临时讨论当成结论，
   直接入库会污染后续所有召回结果，且很难被发现和清理。
"""

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.models import (
    Document,
    EvidenceItem,
    MemoryCandidate,
    ProjectIndexState,
    RecallEvent,
    Repository,
)
from app.services.rag.vector_store import add_document, milvus_runtime_status


def record_recall_event(
    db: Session,
    *,
    repo_id: uuid.UUID | None,
    conversation_id: uuid.UUID | None,
    tool_name: str,
    query: str,
    results: list[dict[str, Any]] | None = None,
    citations: list[dict[str, Any]] | None = None,
    scope: str = "project",
    mode: str = "hybrid",
    status: str = "success",
    metadata: dict[str, Any] | None = None,
) -> RecallEvent:
    """记录一次检索（或一次检索失败）事件。

    只保留前 12 条结果明细：审计要的是「大致命中了什么」，
    存全量会让这张表迅速膨胀，而且检索结果本身在别处已有原始记录。
    result_count 记的是**真实命中总数**，不是截断后的 12——两者含义不同。
    只 flush 不 commit：审计要和调用方的业务事务同生共死。
    """
    # 逐条裁剪成固定字段，避免把内部字段（如 _raw_content、embedding）写进审计表。
    safe_results = [_recall_result_payload(item) for item in (results or [])[:12]]
    event = RecallEvent(
        repo_id=repo_id,
        conversation_id=conversation_id,
        tool_name=tool_name,
        # 去掉首尾空白：同一 query 因为多敲了空格而看起来不同，会给统计添麻烦。
        query=query.strip(),
        scope=scope,
        mode=mode,
        status=status,
        result_count=len(results or []),
        results_json=safe_results,
        citations_json=(citations or [])[:12],
        meta=metadata or {},
    )
    db.add(event)
    db.flush()
    return event


def list_recall_events(
    db: Session,
    *,
    repo_id: uuid.UUID,
    conversation_id: uuid.UUID | None = None,
    limit: int = 30,
) -> list[dict[str, Any]]:
    """列出最近的召回事件（按时间倒序）。

    limit 用 min(max(limit,1),100) 夹紧：既防前端传 0/负数导致空结果或 SQL 异常，
    也防传 10000 把整张表拉出来打爆内存。
    """
    query = db.query(RecallEvent).filter(RecallEvent.repo_id == repo_id)
    if conversation_id:
        query = query.filter(RecallEvent.conversation_id == conversation_id)
    rows = query.order_by(RecallEvent.created_at.desc()).limit(min(max(limit, 1), 100)).all()
    return [_recall_event_payload(row) for row in rows]


def memory_status(db: Session, repo: Repository) -> dict[str, Any]:
    """汇总该仓库的记忆/索引健康度，供前端「记忆」面板展示。

    一次性把索引状态、各类来源的文档数、证据数、候选数、最近一次召回、
    向量库运行状况都取出来：这些都是小结果集，合并成一次请求比前端分别拉更省。
    """
    # 索引状态是单行记录（每仓库一条），没有就视为尚未建索引。
    index_state = db.query(ProjectIndexState).filter(ProjectIndexState.repo_id == repo.id).one_or_none()
    # 三条 group by 统计：按来源类型/状态分组，前端直接拿来画分布。
    # dict(...) 把 (key, count) 行转成映射，key 可能为 NULL（历史数据），故下面求和时统一转 int。
    source_counts = dict(
        db.query(Document.source_type, func.count(Document.id))
        .filter(Document.repo_id == repo.id)
        .group_by(Document.source_type)
        .all()
    )
    evidence_counts = dict(
        db.query(EvidenceItem.source_type, func.count(EvidenceItem.id))
        .filter(EvidenceItem.repo_id == repo.id)
        .group_by(EvidenceItem.source_type)
        .all()
    )
    candidate_counts = dict(
        db.query(MemoryCandidate.status, func.count(MemoryCandidate.id))
        .filter(MemoryCandidate.repo_id == repo.id)
        .group_by(MemoryCandidate.status)
        .all()
    )
    latest_recall = (
        db.query(RecallEvent)
        .filter(RecallEvent.repo_id == repo.id)
        .order_by(RecallEvent.created_at.desc())
        .first()
    )
    return {
        "repo_id": repo.id,
        "index": {
            # 没有索引记录时状态给 "missing"，前端据此提示「先去建索引」。
            "status": index_state.status if index_state else "missing",
            "docs_indexed": index_state.docs_indexed if index_state else 0,
            "docs_total": index_state.docs_total if index_state else 0,
            # 时间统一转 isoformat 字符串，避免 JSON 序列化 datetime 的格式在不同环境下不一致。
            "last_scan_at": index_state.last_scan_at.isoformat() if index_state and index_state.last_scan_at else None,
            "error_message": index_state.error_message if index_state else None,
            "summary_json": index_state.summary_json if index_state else None,
        },
        "documents": {
            # 分组的 key 可能为 None，int(value) 只对计数值做转换；sum 的空值是 0。
            "total": sum(int(value) for value in source_counts.values()),
            "by_source_type": source_counts,
        },
        "evidence": {
            "total": sum(int(value) for value in evidence_counts.values()),
            "by_source_type": evidence_counts,
        },
        "recall": {
            "total": db.query(RecallEvent).filter(RecallEvent.repo_id == repo.id).count(),
            "latest_at": latest_recall.created_at.isoformat() if latest_recall and latest_recall.created_at else None,
        },
        "candidates": {
            "total": sum(int(value) for value in candidate_counts.values()),
            "by_status": candidate_counts,
            # 单独把待审核数拎出来，前端可以在导航上做红点提醒。
            "pending": int(candidate_counts.get("pending") or 0),
        },
        # 向量库状态是运行时探测（Milvus 可能没起来），失败也不该让整个接口失败。
        "vector": milvus_runtime_status(),
    }


def capture_memory_candidates(
    db: Session,
    *,
    repo_id: uuid.UUID | None,
    conversation_id: uuid.UUID | None,
    session_id: uuid.UUID | None,
    memory_update: dict[str, Any],
) -> list[MemoryCandidate]:
    """把一次记忆抽取的结果落成待审核候选，返回本次新建的那些。

    repo_id 为空直接返回空列表：记忆是挂在项目维度上的，
    没有项目归属就无处存放（也无法在审核时按项目过滤）。
    """
    if repo_id is None:
        return []
    # 五类记忆分开存：kind 决定审核页的分组和展示，也方便以后按类型做不同的保留策略。
    specs = [
        ("decision", memory_update.get("decisions") or []),
        ("fact", memory_update.get("facts") or []),
        ("task", memory_update.get("tasks") or []),
        ("preference", memory_update.get("user_preferences") or []),
        ("repo_context", memory_update.get("repo_context") or []),
    ]
    created: list[MemoryCandidate] = []
    for kind, values in specs:
        for value in values:
            content = str(value or "").strip()
            # 空内容跳过；已存在同内容的候选也跳过，避免同一结论被反复抽出来刷屏。
            if not content or _candidate_exists(db, repo_id, kind, content):
                continue
            candidate = MemoryCandidate(
                repo_id=repo_id,
                conversation_id=conversation_id,
                session_id=session_id,
                kind=kind,
                title=_candidate_title(kind, content),
                content=content,
                # 统一以 pending 落库，必须人工审核后才会入库。
                status="pending",
                source="session_sealer",
                # meta 里留来源 key 和抓取时间，排查「这条记忆是哪次会话产生的」时有据可查。
                meta={"memory_update_key": kind, "captured_at": datetime.now(timezone.utc).isoformat()},
            )
            db.add(candidate)
            created.append(candidate)
    # 没有新建就不 flush：避免为一个空操作平白触发一次 SQL。
    if created:
        db.flush()
    return created


def list_memory_candidates(
    db: Session,
    *,
    repo_id: uuid.UUID,
    status: str | None = "pending",
    limit: int = 50,
) -> list[dict[str, Any]]:
    """按状态列出记忆候选，默认只看 pending（审核页最常见的视图）。

    status 传 "all" 或空值时不过滤；limit 同样夹在 1..100。
    """
    query = db.query(MemoryCandidate).filter(MemoryCandidate.repo_id == repo_id)
    if status and status != "all":
        query = query.filter(MemoryCandidate.status == status)
    rows = query.order_by(MemoryCandidate.created_at.desc()).limit(min(max(limit, 1), 100)).all()
    return [_candidate_payload(row) for row in rows]


def approve_memory_candidate(db: Session, repo: Repository, candidate_id: uuid.UUID) -> tuple[MemoryCandidate, Document | None]:
    """审核通过：把候选内容写进知识库文档，使其今后可被检索。

    返回 (候选, 新文档)：幂等场景下（本来就已通过）第二个元素为 None——
    此时并没有新建文档，调用方不该拿它去刷缓存。
    add_document 传 commit=False，把提交权交给外层请求，
    保证「状态改为 approved」和「文档入库」在同一个事务里，不会出现一半成功。
    """
    candidate = _candidate_or_none(db, repo.id, candidate_id)
    if candidate is None:
        # 用固定错误码而不是中文句子：路由层负责翻译成用户可读文案。
        raise ValueError("candidate_not_found")
    if candidate.status == "approved":
        # 重复点击「通过」不应产生第二份文档，直接返回。
        return candidate, None
    doc = add_document(
        db,
        repo.id,
        # 来源类型记为 memory_note，后续检索/统计时能把「人工沉淀的记忆」和原始资料区分开。
        "memory_note",
        # 每次生成新的随机 source_id：同一条记忆将来若被重新抽取，
        # 也会是一条独立文档，而不是覆盖旧文档（便于对比与追溯）。
        uuid.uuid4(),
        candidate.title,
        candidate.content,
        {
            "display_title": candidate.title,
            "candidate_id": str(candidate.id),
            "candidate_kind": candidate.kind,
            "source": "approved_memory_candidate",
            # 把候选原有的 meta（抓取时间等）一并带上，保持来源链路完整。
            **(candidate.meta or {}),
        },
        commit=False,
    )
    candidate.status = "approved"
    candidate.reviewed_at = datetime.now(timezone.utc)
    # 反向记下生成的文档 id：以后想追「这条记忆在知识库里是哪篇」可以直接跳转。
    candidate.meta = {**(candidate.meta or {}), "approved_document_id": str(doc.id)}
    db.flush()
    return candidate, doc


def reject_memory_candidate(db: Session, repo: Repository, candidate_id: uuid.UUID) -> MemoryCandidate:
    """审核驳回：只把状态改成 rejected，不动知识库。

    保留记录而不删除，是为了让「这条结论曾被否决过」这件事留痕；
    _candidate_exists 也只把 pending/approved 视为重复，所以驳回后同名内容还能再次被抽取。
    """
    candidate = _candidate_or_none(db, repo.id, candidate_id)
    if candidate is None:
        raise ValueError("candidate_not_found")
    candidate.status = "rejected"
    candidate.reviewed_at = datetime.now(timezone.utc)
    db.flush()
    return candidate


def _candidate_or_none(db: Session, repo_id: uuid.UUID, candidate_id: uuid.UUID) -> MemoryCandidate | None:
    """按 (仓库, id) 取候选。

    带上 repo_id 是防止越权：只凭 candidate_id 就能操作别的项目的记忆。
    """
    return (
        db.query(MemoryCandidate)
        .filter(MemoryCandidate.repo_id == repo_id, MemoryCandidate.id == candidate_id)
        .one_or_none()
    )


def _candidate_exists(db: Session, repo_id: uuid.UUID, kind: str, content: str) -> bool:
    """判断同仓库同类型同内容的候选是否已存在。

    只把 pending/approved 算作重复：rejected 的允许重新出现（用户可能改主意），
    这样既去重又不会把被否决过的结论永久拉黑。
    只 select id 且带 first()，不取整行、不算总数，纯粹为了「存在性」做最省资源的判断。
    """
    return (
        db.query(MemoryCandidate.id)
        .filter(
            MemoryCandidate.repo_id == repo_id,
            MemoryCandidate.kind == kind,
            MemoryCandidate.content == content,
            MemoryCandidate.status.in_(["pending", "approved"]),
        )
        .first()
        is not None
    )


def _candidate_title(kind: str, content: str) -> str:
    """给候选生成「类型：内容摘要」形式的标题，便于审核时快速扫读。"""
    labels = {
        "decision": "决策",
        "fact": "事实",
        "task": "任务",
        "preference": "偏好",
        "repo_context": "项目上下文",
    }
    # 未知 kind 时退化成原样输出，保证将来新增类型不会直接报错或显示空白。
    return f"{labels.get(kind, kind)}：{_clip(content, 80)}"


def _recall_result_payload(item: dict[str, Any]) -> dict[str, Any]:
    """把一条检索结果裁剪成审计用的固定结构（丢内部字段、截断片段）。"""
    return {
        "title": str(item.get("title") or ""),
        "source_type": str(item.get("source_type") or ""),
        "source_id": str(item.get("source_id") or ""),
        # score 原样保留（可能是 None/float），不做转换，避免精度或类型语义丢失。
        "score": item.get("score"),
        "snippet": _clip(str(item.get("snippet") or ""), 260),
        "metadata": item.get("metadata") or {},
        "retrieval": item.get("retrieval") or {},
    }


def _recall_event_payload(row: RecallEvent) -> dict[str, Any]:
    """把 RecallEvent 行转成 API 输出结构（JSON 列空值统一成空容器）。"""
    return {
        "id": row.id,
        "repo_id": row.repo_id,
        "conversation_id": row.conversation_id,
        "tool_name": row.tool_name,
        "query": row.query,
        "scope": row.scope,
        "mode": row.mode,
        "status": row.status,
        "result_count": row.result_count,
        "results": row.results_json or [],
        "citations": row.citations_json or [],
        "metadata": row.meta or {},
        # created_at 原样返回 datetime，由 FastAPI 序列化，省去这里格式化。
        "created_at": row.created_at,
    }


def _candidate_payload(row: MemoryCandidate) -> dict[str, Any]:
    """把 MemoryCandidate 行转成 API 输出结构。"""
    return {
        "id": row.id,
        "repo_id": row.repo_id,
        "conversation_id": row.conversation_id,
        "session_id": row.session_id,
        "kind": row.kind,
        "title": row.title,
        "content": row.content,
        "status": row.status,
        "source": row.source,
        "metadata": row.meta or {},
        "created_at": row.created_at,
        "reviewed_at": row.reviewed_at,
    }


def _clip(text: str, limit: int) -> str:
    """压平空白并截断，超出时保留 limit-1 个字符再加省略号。

    先 rstrip 再拼 "..."：避免截断点正好落在空格上，
    出现「xxx ...」这种看起来像多了一个空格的脏文案。
    """
    cleaned = " ".join((text or "").split())
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1].rstrip() + "..."
