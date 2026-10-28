"""会话记忆的核心实现：消息存取、证据检索、上下文组装、渐进式压缩、会话封存。

五个组件的分工（按一次对话的时间顺序）：
1. MessageStore：消息的读写。「持久化全量记录」和「模型当前可见的投影」是两套读法：
   timeline() 是完整历史（给 UI 和精确检索），recent() 是压缩后的有效上下文（给模型）。
2. EvidenceStore：把仓库数据 + 向量检索结果反范式化成一张 EvidenceItem 表，
   让记忆工具能做「关键词 + 向量」的混合检索，而不必每次跨多张表 join。
3. ContextAssembler：按 token 预算把 系统提示/记忆/证据/近期对话 拼成最终 prompt。
4. ProgressiveContextManager：压力大时主动压缩（写一条 compact_boundary 分界消息）。
5. SessionSealer：一轮问答结束后封存快照、抽取记忆增量、生成会话摘要。

最需要理解的一点：**压缩不删除原始消息**。
压缩只是插入一条带摘要的 system 消息作为「分界线」，模型从分界线之后开始看，
原始消息仍可通过 devflow_get_thread_context 精确取回（所以叫「可恢复的压缩」）。
"""

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import and_, or_
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import (
    AgentRun,
    ChatMessage,
    ChatSession,
    ChatSession as ChatSessionModel,
    Conversation,
    ConversationMemory,
    Document,
    EvidenceItem,
    Issue,
    PullRequest,
    Repository,
    ThreadMemory,
    WorkflowRun,
)
from app.services.context_compression import (
    COMPACT_BOUNDARY_SUBTYPE,
    COMPACT_SUMMARY_SYSTEM_PROMPT,
    adjust_keep_start_to_preserve_api_invariants,
    build_compaction_summary_payload,
    calculate_context_pressure,
    clip_to_token_budget,
    compress_messages,
    default_context_budget,
    dumps_compact,
    ensure_compaction_summary_preserves_key_facts,
    estimate_tokens,
    extract_relevant_snippet,
    fallback_memory_update,
    fallback_compaction_summary,
    format_compaction_summary,
    is_compact_boundary_message,
    messages_after_compact_boundary,
    memory_document_text,
    merge_memory,
    microcompact_messages,
    render_structured_memory,
    structured_memory_from_model,
)
from app.services.llm.client import LLMClient
from app.services.memory_hub import capture_memory_candidates, record_recall_event
from app.services.rag.indexing import CI_LOG_CHUNK_CHARS, sanitize_ci_log
from app.services.rag.retrieval import search_similar_documents
from app.services.rag.source_policy import RAG_SOURCE_TYPES
from app.services.rag.vector_store import MilvusUnavailableError


def _repo_filter(model: Any, repo_id: uuid.UUID | None) -> Any:
    """构造 repo 过滤条件，关键在 None 要翻译成 IS NULL。

    库里「没有仓库归属」的行（如全局记忆）存的是 NULL 而不是某个特殊 id，
    所以 repo_id 为 None 时必须用 is_(None)，用 == None 在部分方言下匹配不到。
    """
    return model.repo_id.is_(None) if repo_id is None else model.repo_id == repo_id


def _conversation_filter(model: Any, conversation_id: uuid.UUID | None) -> Any:
    """同 _repo_filter：None 表示「属于整个仓库而非某个会话」，对应 IS NULL。"""
    return model.conversation_id.is_(None) if conversation_id is None else model.conversation_id == conversation_id


def _tokens(text: str) -> list[str]:
    """切成检索用词（小写，保留中文连续段和带下划线/横线的标识符）。"""
    return re.findall(r"[a-zA-Z0-9_\-\u4e00-\u9fff]+", text.lower())


def _clip(text: str, limit: int) -> str:
    """截断到 limit 长度（含省略号），空值安全。"""
    text = (text or "").strip()
    # limit-3 给 "..." 留位置；max(0, ...) 防止 limit 太小时出现负数切片。
    return text if len(text) <= limit else text[: max(0, limit - 3)].rstrip() + "..."


class MessageStore:
    """消息读写门面。

    只持有 db 会话，不缓存任何状态：所有查询都即时打库，
    保证同一请求内多次读取看到的是最新数据（压缩流程会边读边写）。
    """

    def __init__(self, db: Session) -> None:
        self.db = db

    def append(
        self,
        *,
        repo_id: uuid.UUID | None,
        conversation_id: uuid.UUID | None,
        role: str,
        content: str,
        route: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> ChatMessage:
        """追加一条消息并 flush（不 commit）。

        flush 是为了拿到自增 id —— 调用方经常要立刻把这个 id 写进别的记录
        （例如 ChatSession 的 user_message_id）。提交时机由外层请求决定。
        tool_calls/meta 用 `or []` / `or {}` 兜底，避免库里出现 NULL 后到处判空。
        """
        message = ChatMessage(
            repo_id=repo_id,
            conversation_id=conversation_id,
            role=role,
            content=content,
            route=route,
            tool_calls=tool_calls or [],
            meta=meta or {},
        )
        self.db.add(message)
        self.db.flush()
        return message

    def recent(self, repo_id: uuid.UUID | None, conversation_id: uuid.UUID | None, limit: int = 20) -> list[ChatMessage]:
        """Return the active context projection, not the complete transcript.

        New compact boundaries keep references to the original recent messages
        instead of copying them.  Older databases can still contain
        ``preserved_after_compact`` copies, so this method understands both
        layouts during the migration period.

        译文：返回「当前有效上下文」的投影，而不是完整历史。

        新版压缩边界用**引用**（preserved_message_ids）指向原始近期消息，而不是复制一份；
        老数据库里还残留着 subtype=preserved_after_compact 的**副本**，
        因此这个读取方法在迁移期要同时兼容两种布局：
        - 引用布局：按 id 精确取出被保留的消息，按原顺序排好；
        - 副本布局：那些副本消息要排除掉，否则同一句话会出现两次。
        """
        boundary = self.last_compact_boundary(repo_id, conversation_id)
        if boundary is None:
            # 还没压缩过：直接返回时间线（包含 system 消息，因为可能有别的系统提示）。
            return self.timeline(repo_id, conversation_id, limit=limit, include_system=True)

        boundary_meta = boundary.meta or {}
        preserved_ids = [
            value
            for value in boundary_meta.get("preserved_message_ids") or []
            if value
        ]
        preserved_rows: list[ChatMessage] = []
        if preserved_ids:
            parsed_ids: list[uuid.UUID] = []
            for value in preserved_ids:
                try:
                    parsed_ids.append(uuid.UUID(str(value)))
                except ValueError:
                    # 历史数据里可能混入非 UUID 的脏值，跳过而不是让整次读取失败。
                    continue
            if parsed_ids:
                by_id = {
                    row.id: row
                    for row in self.db.query(ChatMessage)
                    .filter(
                        _repo_filter(ChatMessage, repo_id),
                        _conversation_filter(ChatMessage, conversation_id),
                        ChatMessage.id.in_(parsed_ids),
                    )
                    .all()
                }
                # 按 parsed_ids 的顺序还原（而不是按查询结果顺序），保证保留消息的原始先后关系。
                preserved_rows = [by_id[item] for item in parsed_ids if item in by_id]

        # 分界线之后的消息 + 分界线本身，构成压缩后的有效上下文。
        rows_after_boundary = (
            self.db.query(ChatMessage)
            .filter(
                _repo_filter(ChatMessage, repo_id),
                _conversation_filter(ChatMessage, conversation_id),
                ChatMessage.created_at >= boundary.created_at,
            )
            .order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc())
            .all()
        )
        projected = [boundary, *preserved_rows]
        projected.extend(
            row
            for row in rows_after_boundary
            if row.id != boundary.id
            # 已经通过 preserved_ids 取到的不重复加入。
            and str(row.id) not in preserved_ids
            # 只在新布局下排除旧副本：如果没有 preserved_ids（纯旧数据），
            # 这些副本就是唯一的内容来源，不能排除，否则上下文会空掉。
            and not (
                preserved_ids
                and (row.meta or {}).get("subtype") == "preserved_after_compact"
            )
        )
        # 只保留最后 limit 条：越靠后越新，与 token 预算的取舍方向一致。
        return projected[-max(1, limit) :]

    def timeline(
        self,
        repo_id: uuid.UUID | None,
        conversation_id: uuid.UUID | None,
        *,
        limit: int | None = 500,
        include_system: bool = False,
        before_message_id: uuid.UUID | None = None,
        after_message_id: uuid.UUID | None = None,
    ) -> list[ChatMessage]:
        """Return the durable transcript while hiding compatibility copies.

        Compact boundaries change the model's active projection; they do not
        erase the transcript.  This method is the authoritative read path for
        UI history and exact-memory tools.

        译文：返回「持久化的完整记录」，同时隐藏兼容用的副本消息。

        压缩分界线只改变模型看到的投影，**不会抹掉原始记录**。
        所以这个方法才是 UI 历史记录和精确记忆工具的权威读取路径。

        分页参数说明：before/after 都用 (created_at, id) 组合比较而不是只比 id，
        因为 id 顺序不一定等于时间顺序（并发写入、导入数据都可能乱序），
        组合条件能保证分页边界严格且不重不漏。
        """
        query = self.db.query(ChatMessage).filter(
            _repo_filter(ChatMessage, repo_id),
            _conversation_filter(ChatMessage, conversation_id),
        )
        if not include_system:
            # 默认过滤 system：压缩摘要/失败记录这些内部消息不该出现在用户可见的历史里。
            query = query.filter(ChatMessage.role != "system")

        if before_message_id:
            before = (
                self.db.query(ChatMessage)
                .filter(
                    _repo_filter(ChatMessage, repo_id),
                    _conversation_filter(ChatMessage, conversation_id),
                    ChatMessage.id == before_message_id,
                )
                .one_or_none()
            )
            # 锚点消息不存在时**不加任何条件**（相当于从头取），
            # 这比报错更宽容：前端拿着已删除的消息 id 翻页时不至于白屏。
            if before and before.created_at:
                query = query.filter(
                    or_(
                        ChatMessage.created_at < before.created_at,
                        and_(
                            ChatMessage.created_at == before.created_at,
                            # 同一时间戳时用 id 做二级排序键，保证边界确定。
                            ChatMessage.id < before.id,
                        ),
                    )
                )
        if after_message_id:
            after = (
                self.db.query(ChatMessage)
                .filter(
                    _repo_filter(ChatMessage, repo_id),
                    _conversation_filter(ChatMessage, conversation_id),
                    ChatMessage.id == after_message_id,
                )
                .one_or_none()
            )
            if after and after.created_at:
                query = query.filter(
                    or_(
                        ChatMessage.created_at > after.created_at,
                        and_(
                            ChatMessage.created_at == after.created_at,
                            ChatMessage.id > after.id,
                        ),
                    )
                )

        rows = query.order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc()).all()
        durable = [
            row
            for row in rows
            if (row.meta or {}).get("subtype") != "preserved_after_compact"
        ]
        if limit is None:
            # None 表示「不分页，全都要」（精确检索场景会用到）。
            return durable
        if after_message_id and not before_message_id:
            # 向后翻页取**开头** limit 条：调用方是从该点往后读。
            return durable[: max(1, limit)]
        # 其它情况（首屏、向前翻页）取**最后** limit 条：优先给最新的内容。
        return durable[-max(1, limit) :]

    def active_for_context(
        self,
        repo_id: uuid.UUID | None,
        conversation_id: uuid.UUID | None,
        limit: int = 1000,
    ) -> list[ChatMessage]:
        """给上下文组装用的读取入口（语义化命名，实现就是 recent）。

        默认 limit 很大（1000）：压缩流程需要看到足够多的历史才能决定压什么。
        """
        return self.recent(repo_id, conversation_id, limit=limit)

    def all_for_context(self, repo_id: uuid.UUID | None, conversation_id: uuid.UUID | None, limit: int = 200) -> list[ChatMessage]:
        """取最近 limit 条原始记录（含 system），返回时恢复成时间正序。

        先按时间倒序取再 reverse：这样「limit」限制的是最新的一段，
        而返回给调用方的顺序仍然是可读的正序。
        """
        rows = (
            self.db.query(ChatMessage)
            .filter(_repo_filter(ChatMessage, repo_id), _conversation_filter(ChatMessage, conversation_id))
            .order_by(ChatMessage.created_at.desc())
            .limit(limit)
            .all()
        )
        return list(reversed(rows))

    def last_compact_boundary(self, repo_id: uuid.UUID | None, conversation_id: uuid.UUID | None) -> ChatMessage | None:
        """找出最后一条压缩分界线。

        只在最近 50 条 system 消息里找：分界线本质上「越新越有用」，
        限定窗口避免一次会话压缩过几十次后还要扫全表。
        """
        rows = (
            self.db.query(ChatMessage)
            .filter(_repo_filter(ChatMessage, repo_id), _conversation_filter(ChatMessage, conversation_id), ChatMessage.role == "system")
            .order_by(ChatMessage.created_at.desc())
            .limit(50)
            .all()
        )
        for row in rows:
            # 复用 context_compression 的判定，避免两处对 subtype 的理解不一致。
            if is_compact_boundary_message({"role": row.role, "meta": row.meta or {}}):
                return row
        return None


class EvidenceStore:
    """证据仓库：把散落在多张表里的「可检索内容」摊平到 EvidenceItem 一张表。

    为什么要反规范化（把内容复制一份）：记忆检索要同时看 issue/PR/CI/文档/记忆/会话摘要，
    如果每次都跨表 join + 拼接，查询会非常重；摊平后一条 SQL 就能做关键词匹配，
    也能和向量检索结果合并排序。代价是数据可能滞后，所以每次索引/写入后要 rebuild。
    """

    def __init__(self, db: Session) -> None:
        self.db = db

    def rebuild_repo(self, repo: Repository | None, conversation: Conversation | None = None) -> None:
        """重建证据表（先删后插）。

        删除范围分两种：
        - 指定了 conversation：只清「本会话的 + 无会话归属的（repo 级共享）」，
          保留其它会话的证据，避免一个会话的写入把整仓证据清空；
        - 未指定会话：清掉整个 repo 的证据（repo 级重建）。
        repo 为 None 时按「无仓库归属」处理，对应 repo_id IS NULL 的那批数据。
        """
        repo_id = repo.id if repo else None
        conversation_id = conversation.id if conversation else None
        if conversation_id:
            self.db.query(EvidenceItem).filter(
                _repo_filter(EvidenceItem, repo_id),
                # 注意这里用 or_ 而不是只匹配本会话：repo 级共享证据也要一起刷新，
                # 否则仓库数据更新后，会话内看到的仍是旧证据。
                or_(EvidenceItem.conversation_id == conversation_id, EvidenceItem.conversation_id.is_(None)),
            ).delete(synchronize_session=False)
        else:
            self.db.query(EvidenceItem).filter(_repo_filter(EvidenceItem, repo_id)).delete(synchronize_session=False)

        if repo:
            # 仓库本身作为一条证据：回答「这个项目是什么」时最先用到它。
            self._add(
                repo_id,
                None,
                "repository",
                str(repo.id),
                repo.full_name,
                # filter(None, [...]) 顺手去掉空字符串，避免正文里出现空行。
                "\n".join(filter(None, [repo.description or "", f"默认分支：{repo.default_branch or '未知'}"])),
                {"anchor": f"repo:{repo.full_name}"},
            )
            for issue in self.db.query(Issue).filter(Issue.repo_id == repo.id).all():
                self._add(
                    repo.id,
                    None,
                    "issue",
                    str(issue.id),
                    f"Issue #{issue.number}: {issue.title}",
                    "\n".join(filter(None, [issue.title, issue.body or "", f"state: {issue.state}", f"labels: {', '.join(issue.labels or [])}"])),
                    {"number": issue.number, "state": issue.state},
                )
            for pr in self.db.query(PullRequest).filter(PullRequest.repo_id == repo.id).all():
                self._add(
                    repo.id,
                    None,
                    "pull_request",
                    str(pr.id),
                    f"PR #{pr.number}: {pr.title}",
                    "\n".join(filter(None, [pr.title, pr.body or "", f"state: {pr.state}", f"base: {pr.base_branch}", f"head: {pr.head_branch}"])),
                    {"number": pr.number, "state": pr.state},
                )
            for run in self.db.query(WorkflowRun).filter(WorkflowRun.repo_id == repo.id).all():
                # 只收「失败且有日志」的 CI：成功的 run 对排障没有价值，
                # 全部收进来会让证据表被大量无用记录挤占。
                if run.conclusion != "failure" or not (run.logs_text or "").strip():
                    continue
                # 日志里可能含 token/密码，入库前必须脱敏；再截断到单块大小，避免一条证据过长。
                sanitized_logs = sanitize_ci_log(run.logs_text or "")
                safe_logs = sanitized_logs[:CI_LOG_CHUNK_CHARS]
                self._add(
                    repo.id,
                    None,
                    "workflow_run",
                    str(run.id),
                    f"CI: {run.name}",
                    "\n".join(filter(None, [run.name, run.status, run.conclusion or "", safe_logs])),
                    {
                        "status": run.status,
                        "conclusion": run.conclusion,
                        # 记录是否被截断 + 已脱敏，便于使用者判断这份日志能不能作为完整依据。
                        "logs_truncated": len(sanitized_logs) > CI_LOG_CHUNK_CHARS,
                        "secrets_redacted": True,
                    },
                )
            for doc in (
                self.db.query(Document)
                .filter(Document.repo_id == repo.id, Document.source_type.in_(RAG_SOURCE_TYPES))
                .all()
            ):
                # 文档用 source_id 而不是主键当证据 id：一个文档有多个分块，
                # 用 source_id 才能把它们视为同一来源（检索去重时用得上）。
                self._add(repo.id, None, doc.source_type, str(doc.source_id), doc.title, doc.content, doc.meta or {})

        # 记忆的证据化：有会话就取会话记忆，否则取仓库级的线程记忆。
        memories = (
            self.db.query(ConversationMemory).filter(ConversationMemory.conversation_id == conversation_id).all()
            if conversation_id
            else self.db.query(ThreadMemory).filter(_repo_filter(ThreadMemory, repo_id)).all()
        )
        for memory in memories:
            self._add(
                repo_id,
                conversation_id,
                "memory",
                str(memory.id),
                "Thread memory",
                memory_document_text(memory),
                {"sessions_incorporated": memory.sessions_incorporated},
            )
        # 会话摘要和 Agent 运行记录各取最近 30 条：更早的通常已被记忆/摘要覆盖，
        # 全量收进来只会增加噪声。limit(30) 也能控制重建耗时。
        session_query = self.db.query(ChatSession).filter(_repo_filter(ChatSession, repo_id))
        run_query = self.db.query(AgentRun).filter(_repo_filter(AgentRun, repo_id))
        if conversation_id:
            session_query = session_query.filter(ChatSession.conversation_id == conversation_id)
            run_query = run_query.filter(AgentRun.conversation_id == conversation_id)
        for session in session_query.order_by(ChatSession.created_at.desc()).limit(30):
            self._add(repo_id, conversation_id, "session", str(session.id), f"Session digest: {session.route}", session.digest, {"route": session.route})
        for run in run_query.order_by(AgentRun.created_at.desc()).limit(30):
            self._add(repo_id, conversation_id, "agent_run", str(run.id), f"Agent run: {run.route}", f"{run.user_message}\n{run.final_answer}", {"route": run.route})
        # 统一 flush：整批证据作为一个事务写入，调用方决定何时 commit。
        self.db.flush()

    def search(self, repo_id: uuid.UUID | None, conversation_id: uuid.UUID | None, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """混合检索：先走向量检索，再用证据表做关键词兜底，结果去重合并。

        两段式的理由：
        1. 向量检索能处理语义相近但用词不同的情况，质量更好；但 Milvus 可能不可用，
           所以整体用 try/except 兜住，失败就退化成纯关键词。
        2. 关键词检索覆盖「精确标识符」（函数名、错误码）这类向量检索容易忽略的情况。
        seen 集合按 (source_type, source_id) 去重，保证同一来源不会出现两次。
        """
        query_terms = _tokens(query)
        if not query_terms:
            # 没有任何有效词（纯标点等）就无法打分，直接返回空而不是返回随机内容。
            return []
        results: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        if repo_id:
            try:
                rag_results = search_similar_documents(
                    self.db,
                    repo_id,
                    query,
                    source_type=None,
                    # 多取一些（至少 8）：后面要按 limit 截断，多取才能保证去重后还有足够候选。
                    limit=max(limit, 8),
                )
            except MilvusUnavailableError:
                # 向量库不可用属于可预期情况，静默降级到关键词检索。
                rag_results = []
            for item in rag_results:
                key = (str(item.get("source_type") or ""), str(item.get("source_id") or ""))
                if key in seen:
                    continue
                seen.add(key)
                results.append(
                    {
                        "title": item.get("title") or "",
                        "source_type": item.get("source_type") or "",
                        "source_id": item.get("source_id") or "",
                        "snippet": item.get("snippet") or "",
                        "score": item.get("score") or 0,
                        "metadata": item.get("metadata") or {},
                        "retrieval": item.get("retrieval") or {},
                    }
                )
                if len(results) >= limit:
                    # 向量结果已够数就直接返回，不再走关键词（省一次全表扫描）。
                    return results
        # 关键词兜底：只在 500 条最近证据里找，避免大仓库全表扫描。
        candidates = (
            self.db.query(EvidenceItem)
            .filter(
                _repo_filter(EvidenceItem, repo_id),
                or_(EvidenceItem.conversation_id == conversation_id, EvidenceItem.conversation_id.is_(None)),
            )
            .order_by(EvidenceItem.created_at.desc())
            .limit(500)
            .all()
        )
        ranked: list[tuple[int, EvidenceItem]] = []
        for item in candidates:
            key = (item.source_type, item.source_id)
            if key in seen:
                continue
            haystack = f"{item.title}\n{item.content}".lower()
            # 朴素词频打分：能覆盖精确匹配场景，不追求语义（语义交给上面的向量检索）。
            score = sum(haystack.count(term) for term in query_terms)
            if score:
                ranked.append((score, item))
        # 同分时按创建时间倒序：新证据更可能反映当前状态。
        ranked.sort(key=lambda pair: (pair[0], pair[1].created_at), reverse=True)
        for score, item in ranked:
            key = (item.source_type, item.source_id)
            if key in seen:
                continue
            seen.add(key)
            results.append(
                {
                    "title": item.title,
                    "source_type": item.source_type,
                    "source_id": item.source_id,
                    "snippet": extract_relevant_snippet(query, item.content, max_tokens=90),
                    # 这里的 score 是词频（整数），与向量检索的相似度不同量纲，
                    # 靠 retrieval.sources 标记来源，让上层知道分数怎么来的。
                    "score": score,
                    "metadata": item.meta or {},
                    "retrieval": {"sources": ["evidence_item_keyword"]},
                }
            )
            if len(results) >= limit:
                break
        return results

    def _add(self, repo_id: uuid.UUID | None, conversation_id: uuid.UUID | None, source_type: str, source_id: str, title: str, content: str, meta: dict[str, Any]) -> None:
        """插入一条证据；内容为空的直接跳过。

        跳空内容是必要的：EvidenceItem.content 不设空值约束，但空证据会污染检索
        （命中后 snippet 为空，等于给模型一条空结果）。
        """
        if not content.strip():
            return
        self.db.add(EvidenceItem(repo_id=repo_id, conversation_id=conversation_id, source_type=source_type, source_id=source_id, title=title, content=content, meta=meta))


@dataclass
class AssembledContext:
    """一次组装的结果：给模型的两部分输入 + 证据 + 诊断统计。

    system_context 把所有「非对话」内容（仓库、会话、记忆、压缩摘要、证据）
    拼成一段带标签的文本；history_messages 是真正的多轮对话。
    分开是因为前者通常要作为 system 消息、且更适合做 prompt 缓存。
    """

    system_context: str
    history_messages: list[dict[str, str]]
    evidence: list[dict[str, Any]]
    # 压缩统计只用于诊断展示，不影响对话内容，所以给个默认空字典。
    compression_stats: dict[str, Any] = field(default_factory=dict)


class ContextAssembler:
    """按 token 预算组装一次请求的完整上下文。

    设计目标是「各部分都有配额」：系统提示、记忆、证据、近期对话各自有上限，
    任何一个都不能把整个窗口吃光。这样才能保证最关键的「用户当前问题」永远放得下。
    """

    def __init__(self, db: Session) -> None:
        self.db = db

    def assemble(self, repo: Repository | None, conversation: Conversation | None, current_message: str, limit: int = 16) -> AssembledContext:
        """组装上下文：取记忆/证据/历史 → 必要时先 microcompact → 压缩 → 估算压力 → 可能再压一次。

        「可能再压一次」的逻辑值得注意：第一次压缩后如果整体（含系统提示和证据）
        仍然超预算，会用 microcompact 再压一遍历史。因为历史消息是最容易无损压缩的部分，
        先动它比重写系统提示或丢证据更划算。
        """
        budget = default_context_budget()
        repo_id = repo.id if repo else None
        conversation_id = conversation.id if conversation else None
        message_store = MessageStore(self.db)
        # 先确认有没有压缩分界线：有的话历史只需要从分界线之后取。
        boundary = message_store.last_compact_boundary(repo_id, conversation_id)
        memory = (
            self.db.query(ConversationMemory).filter(ConversationMemory.conversation_id == conversation_id).first()
            if conversation_id
            else self.db.query(ThreadMemory).filter(_repo_filter(ThreadMemory, repo_id)).first()
        )
        # 只取「已封存」的会话摘要：sealing 状态说明摘要还没写好，取到会是不完整内容。
        latest_session = (
            self.db.query(ChatSession)
            .filter(
                _repo_filter(ChatSession, repo_id),
                _conversation_filter(ChatSession, conversation_id),
                ChatSession.status.in_(["sealed", "sealed_with_warnings"]),
            )
            .order_by(ChatSession.created_at.desc())
            .first()
        )
        # 证据只取 8 条：再多会挤占对话历史的预算，且边际收益递减。
        evidence = EvidenceStore(self.db).search(repo_id, conversation_id, current_message, limit=8)
        # 多取一些历史（至少 30 条）再压缩：压缩需要余量，只取 limit 条就没什么可压的了。
        history_fetch_limit = max(limit, 30)
        recent = message_store.recent(repo_id, conversation_id, limit=history_fetch_limit)
        # 只把 user/assistant 交给模型：system（压缩摘要）已经在 system_context 里，
        # tool 消息由工具调用配对逻辑单独处理，混进对话历史会破坏 API 的消息结构。
        raw_history = [
            {
                "role": item.role,
                "content": item.content,
                "route": item.route,
                "tool_calls": item.tool_calls or [],
                "meta": item.meta or {},
            }
            for item in recent
            if item.role in {"user", "assistant"}
        ]
        micro_stats: dict[str, Any] | None = None
        # 估算历史本身的 token（含 tool_calls 的 JSON 体积，工具参数也可能很大）。
        raw_history_tokens = sum(
            estimate_tokens(str(item.get("role") or ""))
            + estimate_tokens(str(item.get("content") or ""))
            + (
                estimate_tokens(dumps_compact(item.get("tool_calls")))
                if item.get("tool_calls")
                else 0
            )
            for item in raw_history
        )
        # 判断「当前用户消息是否已经落库成最后一条历史」：
        # 如果已经在了，就不能再重复计入 pending 预算，否则会高估用量、过早触发压缩。
        current_message_in_history = bool(
            raw_history
            and raw_history[-1].get("role") == "user"
            and " ".join(str(raw_history[-1].get("content") or "").split())
            == " ".join(str(current_message or "").split())
        )
        pending_user_tokens = 0 if current_message_in_history else estimate_tokens(current_message)
        raw_history_pressure = calculate_context_pressure(raw_history_tokens + pending_user_tokens)
        history_source = raw_history
        if raw_history_pressure.should_warn:
            # 先做无损（可恢复）的 microcompact：清掉能重跑的工具输出，比删对话更安全。
            history_source, micro_stats = microcompact_messages(raw_history, current_message)
        history, history_stats = compress_messages(history_source, budget.recent_tokens)

        def build_context(next_history: list[dict[str, str]], next_evidence: list[dict[str, Any]]) -> str:
            """把「仓库/会话标识 + 记忆 + 证据」拼成一段带标签的 system 文本。

            用 [Tag]...[/Tag] 包裹各段：模型能明确区分数据边界，
            也方便日志里定位是哪一段占了多少 token。
            每个部分都单独 clip 到各自预算，再整体拼接（不再整体截断），
            否则后拼接的部分会被整体截断吃掉。
            """
            system_sections: list[str] = []
            if repo:
                system_sections.append(f"[Current Repository]\n{repo.full_name}\n[/Current Repository]")
            if conversation:
                system_sections.append(f"[Current Conversation]\n{conversation.title}\n[/Current Conversation]")

            sections: list[str] = []
            # 仓库/会话标识是稳定的短文本，放最前面，有利于 prompt 缓存命中。
            stable_system = clip_to_token_budget("\n\n".join(system_sections), budget.system_tokens)
            if stable_system:
                sections.append(stable_system)

            memory_sections: list[str] = []
            if boundary and boundary.content:
                transcript_ref = (boundary.meta or {}).get("transcript_ref") or {}
                rehydration = (boundary.meta or {}).get("rehydration") or {}
                # 告诉模型「精确原文还能取回来」：否则它可能以为早期内容已经永久丢失，
                # 从而不敢回答或凭空编造。
                recovery_hint = (
                    "\nExact transcript recovery: call devflow_get_thread_context with this conversation"
                    if transcript_ref
                    else ""
                )
                if rehydration:
                    # 重水合状态：压缩时保存的「正在改哪些文件、用了哪些技能/工具」，
                    # 让模型压缩后仍知道当前工作面。
                    memory_sections.append(
                        "[Rehydrated Working State]\n"
                        + dumps_compact(rehydration)
                        + "\n[/Rehydrated Working State]"
                    )
                memory_sections.append(
                    f"[Compaction Boundary]\n{boundary.content}{recovery_hint}\n[/Compaction Boundary]"
                )
            if memory and memory_document_text(memory).strip():
                # 记忆只用一半预算：另一半留给压缩摘要和会话摘要，避免记忆挤掉更近的信息。
                memory_text = render_structured_memory(memory, max_tokens=max(256, budget.memory_tokens // 2))
                memory_sections.append(f"[Thread Memory: {memory.sessions_incorporated} sessions]\n{memory_text}\n[/Thread Memory]")
            if latest_session and latest_session.digest:
                memory_sections.append(f"[Previous Session Digest]\n{latest_session.digest}\n[/Previous Session Digest]")
            memory_context = clip_to_token_budget("\n\n".join(memory_sections), budget.memory_tokens)
            if memory_context:
                sections.append(memory_context)
            if next_evidence:
                # 证据预算按条数均分（至少 120 token/条），保证每条都能露出一点内容，
                # 而不是前几条占满、后面的全被截空。
                evidence_budget = max(120, budget.evidence_tokens // max(len(next_evidence), 1))
                lines = ["[Related Evidence]"]
                for item in next_evidence:
                    snippet = clip_to_token_budget(item["snippet"], evidence_budget)
                    lines.append(f"- [{item['source_type']}] {item['title']}: {snippet}")
                lines.append("[/Related Evidence]")
                sections.append(clip_to_token_budget("\n".join(lines), budget.evidence_tokens))
            return "\n\n".join(sections)

        system_context = build_context(history, evidence)
        history_prompt_tokens = sum(
            estimate_tokens(item.get("role")) + estimate_tokens(item.get("content"))
            for item in history
        )
        initial_prompt_tokens = estimate_tokens(system_context) + history_prompt_tokens + pending_user_tokens
        initial_pressure = calculate_context_pressure(initial_prompt_tokens)
        if initial_pressure.should_warn and micro_stats is None:
            # 第二次机会：首次评估时历史本身没超（所以没做 microcompact），
            # 但加上系统提示和证据后总量超了。此时再压历史一遍，通常就能腾出空间。
            micro_raw_history, micro_stats = microcompact_messages(raw_history, current_message)
            history, history_stats = compress_messages(micro_raw_history, budget.recent_tokens)
            system_context = build_context(history, evidence)

        history_prompt_tokens = sum(
            estimate_tokens(item.get("role")) + estimate_tokens(item.get("content"))
            for item in history
        )
        assembled_prompt_tokens = estimate_tokens(system_context) + history_prompt_tokens + pending_user_tokens
        final_pressure = calculate_context_pressure(assembled_prompt_tokens)
        microcompacted_messages = int((micro_stats or {}).get("compacted_messages") or 0)
        # 覆盖度统计：告诉上层「模型看到的这段历史有多完整」。
        # 任何一项为真都意味着有信息没进上下文，用户追问细节时可能需要 rehydration。
        history_coverage = {
            "source_messages": len(raw_history),
            "prompt_messages": len(history),
            "omitted_messages": int(history_stats.get("omitted_messages") or 0),
            "truncated_messages": int(history_stats.get("truncated_messages") or 0),
            "exact_suffix_messages": int(history_stats.get("exact_suffix_messages") or 0),
            # 取满了抓取上限说明可能还有更早的消息没被读到（不是被压缩掉的）。
            "source_limit_reached": len(recent) >= history_fetch_limit,
            "has_compaction_boundary": boundary is not None,
            "microcompacted_messages": microcompacted_messages,
        }
        # 只有五项全为 0/False，才算「完整且逐字保留」。
        # 这个布尔值被评估模块用来判断「这次回答是否基于完整上下文」，很重要。
        history_coverage["complete_and_exact"] = not any(
            [
                history_coverage["omitted_messages"],
                history_coverage["truncated_messages"],
                history_coverage["source_limit_reached"],
                history_coverage["has_compaction_boundary"],
                history_coverage["microcompacted_messages"],
            ]
        )

        return AssembledContext(
            system_context=system_context,
            history_messages=history,
            evidence=evidence,
            # 这份诊断非常有用：出问题时能立刻判断是预算太小、历史被压掉了，
            # 还是证据/记忆占了太多空间。前端「上下文诊断」面板直接展示它。
            compression_stats={
                # budget 是 dataclass，用 __dict__ 直接摊平成可序列化的字段。
                "budget": budget.__dict__,
                "history": history_stats,
                "history_coverage": history_coverage,
                "microcompact": micro_stats,
                "pressure": final_pressure.as_dict(),
                "initial_pressure": initial_pressure.as_dict(),
                "raw_history_pressure": raw_history_pressure.as_dict(),
                "compact_boundary": {
                    "id": str(boundary.id) if boundary else None,
                    "created_at": boundary.created_at.isoformat() if boundary and boundary.created_at else None,
                    "mode": (boundary.meta or {}).get("mode") if boundary else None,
                },
                "system_context_estimated_tokens": estimate_tokens(system_context),
                "history_estimated_tokens": history_prompt_tokens,
                "pending_user_estimated_tokens": pending_user_tokens,
                "assembled_prompt_estimated_tokens": assembled_prompt_tokens,
                # 分段占比：一眼看出哪一块最占地方，便于针对性调预算。
                "prompt_breakdown": {
                    "system_memory_evidence": estimate_tokens(system_context),
                    "recent_history": history_prompt_tokens,
                    "pending_user": pending_user_tokens,
                    "tool_observation_budget": budget.tool_observation_tokens,
                    "reserved_response": budget.reserved_response_tokens,
                },
                "evidence_items": len(evidence),
                "mode": "progressive_budgeted_context",
            },
        )


class ProgressiveContextManager:
    """在上下文快满之前主动压缩历史，为后续对话腾出空间。

    触发条件是三重判断（见 ensure_headroom）：
    1. 压力阶段达到 auto_compact 及以上；
    2. 自动压缩开关打开；
    3. 熔断器未打开（连续失败太多次就先放弃，避免每轮都白烧一次模型调用）。
    压缩动作本身是幂等增量的：每压一次插入一条分界线，历史由后续的 recent() 自动截断。
    """

    def __init__(self, db: Session, llm: LLMClient | None = None) -> None:
        self.db = db
        # 允许注入 llm 便于测试；默认自己建一个（懒加载，不会在构造时就发请求）。
        self.llm = llm or LLMClient()

    async def ensure_headroom(self, repo: Repository | None, conversation: Conversation | None, current_message: str) -> dict[str, Any]:
        """检查并（必要时）执行压缩，返回结构化结果（attempted/compacted/reason 等）。

        这个函数永远不抛异常：压缩失败不应影响用户的正常对话，
        而是记录一条失败消息并打开熔断器，让接下来的几轮直接跳过压缩尝试。
        """
        if repo is None or conversation is None:
            # 没有仓库/会话就没有可压缩的归属，直接跳过（这种情况多见于工具调用或全局问答）。
            return {"attempted": False, "reason": "no_repo_or_conversation"}
        assembled = ContextAssembler(self.db).assemble(repo, conversation, current_message)
        assembled_pressure = assembled.compression_stats.get("pressure") or {}
        micro_stats = assembled.compression_stats.get("microcompact") or {}
        # 同时看两个口径：
        # - raw_segment_tokens：分界线之后原始历史的总量（真实占用）；
        # - assembled_tokens：实际送进模型 prompt 的量（已经压过）。
        # 只用后者会漏判「原始历史已经巨大、只是这次恰好压得下去」的情况。
        raw_segment_tokens = self._raw_segment_tokens(repo.id, conversation.id) + estimate_tokens(current_message)
        assembled_tokens = int(
            assembled.compression_stats.get("assembled_prompt_estimated_tokens")
            or assembled.compression_stats.get("system_context_estimated_tokens")
            or 0
        )
        raw_pressure_state = calculate_context_pressure(raw_segment_tokens)
        assembled_pressure_state = calculate_context_pressure(assembled_tokens)
        pressure_state = calculate_context_pressure(max(raw_segment_tokens, assembled_tokens))
        pressure = {
            **pressure_state.as_dict(),
            "raw_segment_estimated_tokens": raw_segment_tokens,
            "assembled_context_estimated_tokens": assembled_tokens,
            "raw_stage": raw_pressure_state.stage,
            "assembled_stage": assembled_pressure.get("stage") or assembled_pressure_state.stage,
            "microcompact_estimated_tokens_saved": micro_stats.get("estimated_tokens_saved", 0),
            "microcompact_compacted_messages": micro_stats.get("compacted_messages", 0),
        }
        raw_stage = raw_pressure_state.stage
        prompt_stage = str(pressure.get("assembled_stage") or assembled_pressure_state.stage)
        stage = str(pressure.get("stage") or "normal")
        if not settings.context_auto_compact_enabled:
            # 关闭自动压缩：只上报压力，交给人工或上层决定。
            return {"attempted": False, "reason": "auto_compact_disabled", "pressure": pressure}
        if (
            prompt_stage in {"normal", "warning"}
            and raw_stage in {"auto_compact", "aggressive", "manual_path"}
            and int(micro_stats.get("compacted_messages") or 0) > 0
        ):
            # 关键优化：原始历史虽然很大，但 microcompact 之后送进 prompt 的量已经正常，
            # 那就没必要再做昂贵的 LLM 摘要压缩（那会丢信息且慢）。
            return {
                "attempted": False,
                "reason": "microcompact_sufficient",
                "pressure": pressure,
                "microcompact": micro_stats,
            }
        if stage in {"normal", "warning"}:
            # 还没到需要压缩的程度，直接返回当前阶段名作为原因（便于观测）。
            return {"attempted": False, "reason": stage, "pressure": pressure}

        failures = self._consecutive_failures(repo.id, conversation.id)
        if failures >= settings.context_max_compaction_failures:
            # 熔断：连续失败说明模型或环境有问题，再试也是白花时间与 token。
            return {
                "attempted": False,
                "reason": "circuit_breaker_open",
                "consecutive_failures": failures,
                "pressure": pressure,
            }

        try:
            result = await self._compact(repo, conversation, current_message, pressure, stage)
            # 成功则把失败计数清零返回，让调用方知道熔断器已恢复。
            return {"attempted": True, **result, "pressure": pressure, "consecutive_failures": 0}
        except Exception as exc:  # pragma: no cover - defensive circuit breaker path
            # 先回滚本次压缩的半成品写入，否则会留下不完整的边界/摘要数据。
            self.db.rollback()
            failure = self._record_failure(repo.id, conversation.id, current_message, stage, exc)
            self.db.flush()
            return {
                "attempted": True,
                "compacted": False,
                "reason": "compaction_failed",
                "failure_id": str(failure.id),
                "error": str(exc),
                "pressure": pressure,
                "consecutive_failures": failures + 1,
            }

    async def _compact(
        self,
        repo: Repository,
        conversation: Conversation,
        current_message: str,
        pressure: dict[str, Any],
        stage: str,
    ) -> dict[str, Any]:
        """执行一次压缩：按「逐级加严」的多个方案尝试，选第一个压到安全线以内的。

        为什么要有多个方案：一次激进的压缩会丢掉太多细节，而温和的压缩可能压不到位。
        所以从当前阶段开始，依次尝试更激进的计划（keep_recent 更少、摘要更短），
        一旦压完压力回到 normal/warning 就停手 —— 用最小的代价达到目的。

        只压缩「分界线之后」的消息：分界线之前的早已被摘要覆盖，重复压缩没有意义。
        """
        store = MessageStore(self.db)
        all_rows = store.active_for_context(repo.id, conversation.id, limit=500)
        all_messages = [self._row_to_message(row) for row in all_rows]
        # include_boundary=False：分界线本身不参与「待摘要」的候选，
        # 否则它会被反复摘要进新摘要里，越滚越长。
        segment_messages = messages_after_compact_boundary(all_messages, include_boundary=False)
        message_ids = {str(item.get("id")) for item in segment_messages if item.get("id")}
        segment_rows = [row for row in all_rows if str(row.id) in message_ids]
        # 只把「对话类」消息纳入压缩范围：system 消息（含旧的失败记录）不算正文。
        candidate_rows = [
            row
            for row in segment_rows
            if row.role in {"user", "assistant", "tool", "tool_use", "tool_result"}
        ]
        # 压力越大，允许尝试的方案越多（auto → aggressive → manual）。
        plan_stages = [stage]
        if stage == "auto_compact":
            plan_stages.extend(["aggressive", "manual_path"])
        elif stage == "aggressive":
            plan_stages.append("manual_path")
        plans = [self._compaction_plan_for_stage(item) for item in plan_stages]
        # 消息太少时压缩没有收益（可能压完只剩一两条），直接放弃。
        # +1 是给「当前这条用户消息」留的位置。
        if len(candidate_rows) <= min(int(item["keep_recent"]) for item in plans) + 1:
            return {"compacted": False, "reason": "not_enough_messages"}

        attempts: list[dict[str, Any]] = []
        selected: tuple[
            dict[str, Any],
            list[ChatMessage],
            list[ChatMessage],
            str,
            str,
            dict[str, Any],
            int,
        ] | None = None
        # 重水合状态只算一次，所有方案共用（它描述的是「当前工作状态」，与压缩力度无关）。
        rehydration = self._rehydration_state(candidate_rows)
        for index, plan in enumerate(plans, start=1):
            keep_rows = self._messages_to_keep(candidate_rows, keep_count=int(plan["keep_recent"]))
            keep_ids = {row.id for row in keep_rows}
            summarize_rows = [row for row in candidate_rows if row.id not in keep_ids]
            if not summarize_rows:
                # 这个方案不需要摘要任何东西，换下一个（通常意味着 keep 定得过大）。
                continue
            summary, mode, preservation_report = await self._summary_for(
                repo,
                conversation,
                summarize_rows,
                current_message,
                max_summary_tokens=int(plan["summary_tokens"]),
            )
            # mode_prefix 让报告里能看出「这条摘要是 aggressive 方案产生的」。
            mode_prefix = str(plan.get("mode_prefix") or "")
            if mode_prefix:
                mode = f"{mode_prefix}_{mode}"
            post_compact_tokens = self._post_compact_tokens(
                summary,
                keep_rows,
                current_message=current_message,
                rehydration=rehydration,
            )
            post_pressure = calculate_context_pressure(post_compact_tokens)
            # 每次尝试都记录，最终一并返回：出问题时能看到「试了几档、每档压到多少」。
            attempts.append(
                {
                    "attempt": index,
                    "plan": plan,
                    "post_compact_estimated_tokens": post_compact_tokens,
                    "post_compact_stage": post_pressure.stage,
                }
            )
            # 先假定这次结果可用，再判断是否需要继续加严（后面的方案会覆盖它）。
            selected = (
                plan,
                keep_rows,
                summarize_rows,
                summary,
                mode,
                preservation_report,
                post_compact_tokens,
            )
            if post_pressure.stage in {"normal", "warning"}:
                break

        if selected is None:
            # 所有方案都没能摘要出内容（例如每条消息都被判定为「必须保留」）。
            return {"compacted": False, "reason": "nothing_to_summarize"}
        # 解包：最后一次尝试的方案（可能是最激进的那档）就是最终采用的结果。
        plan, keep_rows, summarize_rows, summary, mode, preservation_report, post_compact_tokens = selected
        boundary = self._insert_boundary(
            repo,
            conversation,
            summary,
            mode,
            pressure,
            summarize_rows,
            keep_rows,
            preservation_report,
            rehydration,
        )
        # 返回的字段既是给调用方用的，也是排查压缩效果的现场记录。
        return {
            "compacted": True,
            "mode": mode,
            "stage": stage,
            "boundary_id": str(boundary.id),
            "messages_summarized": len(summarize_rows),
            "messages_preserved": len(keep_rows),
            "post_compact_estimated_tokens": post_compact_tokens,
            "post_compact_pressure": calculate_context_pressure(post_compact_tokens).as_dict(),
            "compaction_plan": plan,
            "compaction_attempts": attempts,
            "rehydration": rehydration,
        }

    def _compaction_plan_for_stage(self, stage: str) -> dict[str, Any]:
        """按压力等级给出「保留多少条 + 摘要多少 token」的方案。

        等级越高保留越少、摘要越短（1/3 与 1/2、1/2 与 0.67）：
        这是「丢细节换空间」的量化表达。所有下限都用 max(2/3/256) 夹住，
        因为保留少于 2 条对话、或摘要短于 256 token，模型基本无法接着干活。
        """
        default_keep = max(2, settings.context_compact_keep_recent_messages)
        default_summary_tokens = max(256, settings.context_compact_summary_tokens)
        if stage == "manual_path":
            # 最激进：只留最近 1/3 的消息，摘要压到一半。
            return {
                "keep_recent": max(2, min(default_keep, max(2, default_keep // 3))),
                "summary_tokens": min(default_summary_tokens, max(256, int(default_summary_tokens * 0.5))),
                "mode_prefix": "manual_path",
            }
        if stage == "aggressive":
            # 次激进：留一半消息，摘要压到 2/3。
            return {
                "keep_recent": max(2, min(default_keep, max(3, default_keep // 2))),
                "summary_tokens": min(default_summary_tokens, max(256, int(default_summary_tokens * 0.67))),
                "mode_prefix": "aggressive",
            }
        # auto_compact（默认档）：完全按配置来。
        return {"keep_recent": default_keep, "summary_tokens": default_summary_tokens, "mode_prefix": ""}

    def _messages_to_keep(self, rows: list[ChatMessage], keep_count: int | None = None) -> list[ChatMessage]:
        """算出「保留哪些消息」，并修正两个容易被忽略的配对问题。

        1. 用 adjust_keep_start_to_preserve_api_invariants 把切点前移，
           保证不会出现「有 tool_result 却没有对应 tool_use」；
        2. 如果保留段的第一条是 assistant，而它的前一条是 user（那就是它要回答的问题），
           把这条 user 也带进来 —— 否则模型会看到一段没有提问的回答。
        """
        keep = max(2, keep_count if keep_count is not None else settings.context_compact_keep_recent_messages)
        start_index = max(0, len(rows) - keep)
        start_index = adjust_keep_start_to_preserve_api_invariants(
            [self._row_to_message(row) for row in rows],
            start_index,
        )
        keep_rows = rows[start_index:]
        if (
            keep_rows
            and keep_rows[0].role == "assistant"
            and start_index > 0
            and rows[start_index - 1].role == "user"
        ):
            keep_rows = [rows[start_index - 1], *keep_rows]
        return keep_rows

    async def _summary_for(
        self,
        repo: Repository,
        conversation: Conversation,
        rows: list[ChatMessage],
        current_message: str,
        max_summary_tokens: int,
    ) -> tuple[str, str, dict[str, Any]]:
        """生成压缩摘要，返回 (摘要, 模式, 保真报告)。

        三级策略，优先选代价最低且质量够用的：
        1. session_memory：已有结构化会话记忆时直接拿它当摘要 —— 零模型调用、零成本，
           而且记忆本身就是「提炼过的状态」，比临时摘要更稳定；
        2. full_compaction：调模型按 COMPACT_SUMMARY_SYSTEM_PROMPT 写摘要；
        3. fallback_extractive：模型返回空（或失败）时用规则抽取兜底，
           保证压缩流程一定能产出东西，不会卡死。
        三种模式返回的 mode 字符串会写进分界线，事后可统计各模式的触发比例。
        """
        messages = [self._row_to_message(row) for row in rows]
        memory = self.db.query(ConversationMemory).filter(ConversationMemory.conversation_id == conversation.id).first()
        if memory and memory_document_text(memory).strip():
            summary = render_structured_memory(memory, max_tokens=max_summary_tokens)
            # 加一句承上启下的说明，让模型知道「之前的对话已被这段记忆概括」，
            # 而不是以为上下文莫名断掉了。
            continuation = (
                "This conversation was compacted using structured session memory. "
                "Continue from the preserved recent messages and the current user request.\n\n"
                f"{summary}"
            )
            summary, report = ensure_compaction_summary_preserves_key_facts(messages, continuation, max_summary_tokens)
            return summary, "session_memory", report

        payload = build_compaction_summary_payload(messages)
        # 把「触发压缩的当前请求」也告诉模型：摘要要服务于这个请求，而不是泛泛总结。
        llm_summary = await self.llm.chat_text(
            COMPACT_SUMMARY_SYSTEM_PROMPT,
            f"Current user request that triggered pressure:\n{current_message}\n\nConversation segment to compact:\n{payload}",
        )
        summary = format_compaction_summary(llm_summary)
        if not summary:
            # 模型没给出可用摘要（返回空/只返回标签）：退化成规则摘要。
            summary = fallback_compaction_summary(messages, max_tokens=max_summary_tokens)
            summary, report = ensure_compaction_summary_preserves_key_facts(messages, summary, max_summary_tokens)
            return summary, "fallback_extractive", report
        summary, report = ensure_compaction_summary_preserves_key_facts(messages, summary, max_summary_tokens)
        return summary, "full_compaction", report

    def _insert_boundary(
        self,
        repo: Repository,
        conversation: Conversation,
        summary: str,
        mode: str,
        pressure: dict[str, Any],
        summarized_rows: list[ChatMessage],
        keep_rows: list[ChatMessage],
        preservation_report: dict[str, Any],
        rehydration: dict[str, Any],
    ) -> ChatMessage:
        """插入压缩分界线（一条 role=system 的消息），把本次压缩的全部元信息记在 meta 里。

        时间戳处理是这里最微妙的一段：分界线必须排在所有被摘要/被保留的消息**之后**，
        否则 MessageStore.recent() 按 created_at >= boundary.created_at 取「分界线之后」时，
        会把同时间戳的消息漏掉或多算。所以当 now 不大于已有消息时间时，硬加 1 微秒。
        """
        now = datetime.now(timezone.utc)
        latest_source_time = max(
            (row.created_at for row in [*summarized_rows, *keep_rows] if row.created_at),
            default=None,
        )
        if latest_source_time:
            # 数据库里可能是 naive datetime（SQLite），比较前要统一成同一「aware 性」，
            # 否则 Python 会直接抛 can't compare offset-naive and offset-aware。
            comparable_now = now if latest_source_time.tzinfo else now.replace(tzinfo=None)
            if latest_source_time >= comparable_now:
                now = latest_source_time + timedelta(microseconds=1)
        boundary = ChatMessage(
            repo_id=repo.id,
            conversation_id=conversation.id,
            role="system",
            content=summary,
            # route 固定成 context_compaction：_consecutive_failures 靠它区分压缩类系统消息。
            route="context_compaction",
            tool_calls=[],
            meta={
                "subtype": COMPACT_BOUNDARY_SUBTYPE,
                "trigger": "auto",
                "mode": mode,
                # 记录当时的压力详情，事后能复现「为什么这次触发了压缩」。
                "pressure": pressure,
                "pre_compact_message_id": str(summarized_rows[-1].id) if summarized_rows else None,
                # 三个 id 精确描述「摘要覆盖了哪一段消息」，是审计压缩范围的依据。
                "summarized_from_message_id": str(summarized_rows[0].id) if summarized_rows else None,
                "summarized_to_message_id": str(summarized_rows[-1].id) if summarized_rows else None,
                "preserved_start_message_id": str(keep_rows[0].id) if keep_rows else None,
                # 保留消息用 id 引用而不是复制内容：这样原始消息只有一份，
                # 也避免了「副本与原文不一致」的隐患。
                "preserved_message_ids": [str(row.id) for row in keep_rows],
                # transcript_ref 让前端/模型知道「精确原文可以通过这个工具取回」。
                "transcript_ref": {
                    "repo_id": str(repo.id),
                    "conversation_id": str(conversation.id),
                    "tool": "devflow_get_thread_context",
                },
                "messages_summarized": len(summarized_rows),
                "messages_preserved": len(keep_rows),
                "summary_estimated_tokens": estimate_tokens(summary),
                # 保真报告：哪些关键片段被保住了、哪些是另外补进去的。
                "preservation": preservation_report,
                "rehydration": rehydration,
            },
            created_at=now,
        )
        self.db.add(boundary)
        self.db.flush()
        return boundary

    def _post_compact_tokens(
        self,
        summary: str,
        keep_rows: list[ChatMessage],
        *,
        current_message: str = "",
        rehydration: dict[str, Any] | None = None,
    ) -> int:
        """估算压缩之后的 token 总量（用于判断是否已降到安全线）。

        四部分相加：摘要 + 保留消息 + 当前用户消息 + 重水合状态。
        容易漏的是最后一项——重水合本身也占上下文，不把它算进来会低估压缩后的大小。
        """
        return (
            estimate_tokens(summary)
            + sum(estimate_tokens(row.content) for row in keep_rows)
            + estimate_tokens(current_message)
            + estimate_tokens(dumps_compact(rehydration or {}))
        )

    def _rehydration_state(self, rows: list[ChatMessage]) -> dict[str, Any]:
        """从历史里提取「工作状态快照」，压缩后重新注入，避免模型丢失现场感。

        抽取三类信息：正在改哪些文件（来自工具参数里的 path/file_path）、
        激活过哪些技能、用过哪些工具。都取自 tool_calls 和 agent_events 两条来源，
        因为不同 Agent 版本记录的字段不同，两条都扫才不会漏。
        每类都有条数上限（5/6/8），按「新的覆盖旧的」方式裁剪，避免状态无限增长。
        """
        working_files: list[str] = []
        active_skills: list[str] = []
        recent_tools: list[str] = []

        def remember(target: list[str], value: Any, limit: int) -> None:
            """去重追加，超限时从头丢弃最旧的一条（保持「最近的在后」）。"""
            clean = str(value or "").strip()
            if not clean or clean in target:
                return
            target.append(clean)
            if len(target) > limit:
                # 切片删除而不是 pop(0) 循环：一次删掉多余的头部，O(n) 只走一次。
                del target[0 : len(target) - limit]

        for row in rows:
            calls = [item for item in (row.tool_calls or []) if isinstance(item, dict)]
            events = [item for item in (row.meta or {}).get("agent_events") or [] if isinstance(item, dict)]
            for call in calls:
                remember(recent_tools, call.get("tool_name"), 8)
                arguments = call.get("arguments") or {}
                if isinstance(arguments, dict):
                    # path 和 file_path 两种命名都兼容（不同工具的参数名不一致）。
                    remember(working_files, arguments.get("path") or arguments.get("file_path"), 5)
            for event in events:
                event_type = str(event.get("type") or "")
                if event_type in {"tool_use", "tool_result"}:
                    remember(recent_tools, event.get("tool_name"), 8)
                arguments = event.get("arguments") or {}
                if isinstance(arguments, dict):
                    remember(working_files, arguments.get("path") or arguments.get("file_path"), 5)
                if event_type == "skill_activated":
                    # 技能名可能只给了 title 没给 name，两个字段都取。
                    remember(active_skills, event.get("skill_name") or event.get("skill_title"), 6)

        # recovery 里写明「用什么工具能取回精确内容」，和分界线里的提示呼应。
        return {
            "working_files": working_files,
            "active_skills": active_skills,
            "recent_tools": recent_tools,
            "recovery": {
                "exact_transcript_tool": "devflow_get_thread_context",
                "sealed_session_tool": "devflow_read_session_events",
            },
        }

    def _raw_segment_tokens(self, repo_id: uuid.UUID, conversation_id: uuid.UUID) -> int:
        """统计「当前有效段落」（最后一条分界线之后）的原始 token 总量。

        这是判断「真实占用」的口径：只有它够大，才说明历史确实需要压缩，
        而不是仅因为系统提示/证据占地方导致的总量偏高。
        """
        rows = MessageStore(self.db).active_for_context(repo_id, conversation_id, limit=1000)
        messages = messages_after_compact_boundary([self._row_to_message(row) for row in rows], include_boundary=False)
        total = 0
        for message in messages:
            total += estimate_tokens(str(message.get("role") or ""))
            total += estimate_tokens(str(message.get("content") or ""))
            # 工具调用的参数（尤其是写文件的内容）可能比正文还大，必须计入。
            if message.get("tool_calls"):
                total += estimate_tokens(dumps_compact(message.get("tool_calls")))
        return total

    def _row_to_message(self, row: ChatMessage) -> dict[str, Any]:
        """把 ORM 行转成压缩模块使用的普通 dict（id 转成字符串便于 JSON 化）。"""
        return {
            "id": str(row.id),
            "role": row.role,
            "content": row.content,
            "route": row.route,
            "tool_calls": row.tool_calls or [],
            "meta": row.meta or {},
        }

    def _consecutive_failures(self, repo_id: uuid.UUID, conversation_id: uuid.UUID) -> int:
        """数「最近连续失败了几次压缩」，用于熔断判断。

        只统计最后一条分界线之后的失败记录，并且一旦遇到非失败消息就停止计数——
        这是「连续」的定义：中间成功过一次就重新计数。
        limit 取熔断阈值本身即可，没必要把历史失败全查出来。
        """
        boundary = MessageStore(self.db).last_compact_boundary(repo_id, conversation_id)
        query = self.db.query(ChatMessage).filter(
            ChatMessage.repo_id == repo_id,
            ChatMessage.conversation_id == conversation_id,
            ChatMessage.role == "system",
            ChatMessage.route == "context_compaction",
        )
        if boundary and boundary.created_at:
            query = query.filter(ChatMessage.created_at > boundary.created_at)
        count = 0
        for row in query.order_by(ChatMessage.created_at.desc()).limit(settings.context_max_compaction_failures).all():
            if (row.meta or {}).get("subtype") == "compact_failure":
                count += 1
            else:
                # 遇到成功记录或其它系统消息就中断，保证只统计「最近的连续失败」。
                break
        return count

    def _record_failure(self, repo_id: uuid.UUID, conversation_id: uuid.UUID, current_message: str, stage: str, exc: Exception) -> ChatMessage:
        """记录一次压缩失败（也是一条 system 消息，route 同样是 context_compaction）。

        把错误信息截断到 240 token 再存：异常栈可能非常长，
        原样入库既没必要也会反过来占上下文。
        current_message 也只留 120 token，够定位是哪个请求触发的即可。
        """
        failure = ChatMessage(
            repo_id=repo_id,
            conversation_id=conversation_id,
            role="system",
            content=clip_to_token_budget(f"Context compaction failed during {stage}: {exc}", 240),
            route="context_compaction",
            tool_calls=[],
            meta={
                "subtype": "compact_failure",
                "stage": stage,
                "current_message": clip_to_token_budget(current_message, 120),
            },
            created_at=datetime.now(timezone.utc),
        )
        self.db.add(failure)
        return failure


class SessionSealer:
    """一轮问答结束后「封存」这一轮：存快照、抽取记忆增量、更新会话记忆。

    设计原则是**先保证原始记录落库，再做增强**：
    先把对话原文写进 ChatSession（status=sealing），再去调模型抽记忆。
    这样即使模型/网络失败，这一轮对话也不会丢，只是记忆更新缺失（状态标成 sealed_with_warnings）。
    这正是「聊天已经成功了，不能因为后处理失败而让用户看到报错」的落地方式。
    """

    def __init__(self, db: Session, llm: LLMClient | None = None) -> None:
        self.db = db
        self.llm = llm or LLMClient()

    async def seal(
        self,
        *,
        repo: Repository | None,
        conversation: Conversation | None,
        user_message: ChatMessage,
        assistant_message: ChatMessage,
        route: str,
        tool_calls: list[dict[str, Any]],
        agent_events: list[dict[str, Any]] | None = None,
        model_usage: list[dict[str, Any]] | None = None,
        context_diagnostics: dict[str, Any] | None = None,
    ) -> ChatSessionModel:
        """封存一轮对话，返回 ChatSession 记录。

        三个 ORM 参数（user_message/assistant_message/conversation）都可能为空，
        因为工具调用等场景下并不总是有完整的对话上下文；本函数对所有可选值都做了兜底。
        """
        agent_events = agent_events or []
        model_usage = model_usage or []
        context_diagnostics = context_diagnostics or {}
        # 先算一份「规则版」记忆和摘要：它是后面所有失败路径的兜底，
        # 也保证 session.digest 从创建那一刻起就不为空。
        fallback_update = fallback_memory_update(
            user_message.content,
            assistant_message.content,
            route,
            tool_calls,
        )
        fallback_digest = str(fallback_update.get("summary") or "")
        session = ChatSession(
            repo_id=repo.id if repo else None,
            conversation_id=conversation.id if conversation else None,
            user_message_id=user_message.id,
            assistant_message_id=assistant_message.id,
            route=route,
            # transcript_json 把这一轮的所有现场都装进去：
            # 原文、工具调用、Agent 事件、模型用量、上下文诊断、记忆更新内容。
            # sealed session 是「精确记忆」的数据源，所以尽量存全。
            transcript_json={
                "messages": [
                    {"role": "user", "content": user_message.content, "created_at": self._iso(user_message.created_at)},
                    {"role": "assistant", "content": assistant_message.content, "created_at": self._iso(assistant_message.created_at)},
                ],
                "tool_calls": tool_calls,
                "agent_events": agent_events,
                # 只挑工具相关事件单独存一份，读取时不用在全部事件里过滤。
                "tool_use_results": [
                    event for event in agent_events if event.get("type") in {"tool_use", "tool_result"}
                ],
                "model_usage": model_usage,
                "context_diagnostics": context_diagnostics,
                "memory_update": fallback_update,
                # phase=raw_persisted 表示「原文已落库、增强还没做完」。
                "sealing": {"phase": "raw_persisted", "errors": []},
            },
            digest=fallback_digest,
            status="sealing",
            sealed_at=None,
        )
        self.db.add(session)
        self.db.flush()

        errors: list[str] = []
        memory_update = fallback_update
        digest = fallback_digest
        try:
            memory_update = await self._build_memory_update(
                user_message.content,
                assistant_message.content,
                route,
                tool_calls,
                fallback_digest,
                repo_id=repo.id if repo else None,
                conversation_id=conversation.id if conversation else None,
            )
            # _digest 是内部字段（pop 掉，不写进最终记忆），只用来更新 session.digest。
            digest = str(memory_update.pop("_digest", "") or memory_update.get("summary") or fallback_digest)
        except Exception as exc:  # model/network failures must not invalidate the completed chat turn
            # 译文：模型/网络故障绝不能让已经完成的这轮对话作废。
            errors.append(f"memory_extraction_failed: {exc}")
            memory_update = fallback_update
            digest = fallback_digest

        try:
            # 用嵌套事务包住记忆写入：失败只回滚这一段，
            # 上面已经 flush 的 ChatSession 原文不受影响。
            with self.db.begin_nested():
                self._update_thread_memory(
                    repo.id if repo else None,
                    conversation.id if conversation else None,
                    digest,
                    memory_update,
                )
                # 同时把决策/任务等沉淀成待审核候选（人工确认后才真正进知识库）。
                capture_memory_candidates(
                    self.db,
                    repo_id=repo.id if repo else None,
                    conversation_id=conversation.id if conversation else None,
                    session_id=session.id,
                    memory_update=memory_update,
                )
                self.db.flush()
        except Exception as exc:  # keep raw ChatSession even when enrichment fails
            # 译文：增强步骤失败时，保留原始的 ChatSession（不要删）。
            errors.append(f"memory_persistence_failed: {exc}")

        transcript = dict(session.transcript_json or {})
        # 下划线开头的字段是内部元数据（如 _digest/_update_mode），不写进 transcript。
        transcript["memory_update"] = {
            key: value for key, value in memory_update.items() if not key.startswith("_")
        }
        transcript["sealing"] = {
            "phase": "completed",
            "errors": errors,
            # 证据刷新不在这一步做：推迟到下次请求（由 EvidenceStore 重建），
            # 避免封存阶段再做一次全仓扫描。
            "evidence_refresh": "deferred_to_next_request",
        }
        session.transcript_json = transcript
        session.digest = digest
        # 有错误就标 sealed_with_warnings，而不是 failed：
        # 数据本身是可用的，只是不完整，调用方/前端据此降级展示。
        session.status = "sealed_with_warnings" if errors else "sealed"
        session.sealed_at = datetime.now(timezone.utc)
        self.db.flush()
        return session

    async def _build_memory_update(
        self,
        user_message: str,
        assistant_message: str,
        route: str,
        tool_calls: list[dict[str, Any]],
        digest: str,
        *,
        repo_id: uuid.UUID | None,
        conversation_id: uuid.UUID | None,
    ) -> dict[str, Any]:
        """调模型做「增量记忆更新」，返回合并后的记忆快照。

        关键设计：让模型返回**更新后的完整快照**而不是增量 diff。
        原因是记忆有多个列表字段，做 diff 需要可靠的合并语义，模型很容易给错格式；
        直接要全量快照，落库时整体覆盖即可（对应 _update_mode="snapshot"）。
        模型返回空/失败时退回规则版（fallback），保证调用方永远拿到可用的结构。
        """
        fallback = fallback_memory_update(user_message, assistant_message, route, tool_calls)
        # 有会话用会话级记忆，否则用仓库级线程记忆；两者字段一致，只是作用域不同。
        if repo_id and conversation_id:
            current_memory: ConversationMemory | ThreadMemory | None = (
                self.db.query(ConversationMemory)
                .filter(ConversationMemory.conversation_id == conversation_id)
                .first()
            )
        else:
            current_memory = (
                self.db.query(ThreadMemory)
                .filter(_repo_filter(ThreadMemory, repo_id))
                .first()
            )
        existing_memory = self._memory_state(current_memory) if current_memory else {}
        # 提示词里明确三点：输入包含已有记忆、返回完整快照、只返回 JSON。
        # 这三点缺一个都会让模型输出难以稳定解析。
        prompt = (
            "请增量更新软件工程 Agent 的数据库会话记忆。"
            "输入包含已有记忆和本轮原始记录；返回更新后的完整快照，而不是只返回本轮增量。"
            "只返回 JSON，包含 digest、summary、facts、decisions、open_questions、tasks、"
            "user_preferences、repo_context、citations。summary 必须描述当前工作状态和下一步；"
            "保留重要文件、函数、命令、错误修复和用户纠正，不要编造。"
        )
        parsed = await self.llm.chat_json(
            prompt,
            {
                "route": route,
                "digest": digest,
                "existing_memory": existing_memory,
                "user_message": user_message,
                "assistant_message": assistant_message,
                "tool_calls": tool_calls,
            },
        )
        update = structured_memory_from_model(parsed) if parsed else fallback
        if not update.get("summary"):
            # 模型给了结构但漏了摘要：用规则版的摘要补上，
            # 否则记忆里会出现「有事实但没概述」的割裂状态。
            update["summary"] = fallback["summary"]
        if parsed:
            # 标记为快照模式：_update_thread_memory 会据此「整体覆盖」而不是「合并追加」。
            update["_update_mode"] = "snapshot"
            update["_digest"] = str(parsed.get("digest") or parsed.get("summary") or digest)
        return update

    def _memory_state(self, memory: ConversationMemory | ThreadMemory) -> dict[str, Any]:
        """把记忆 ORM 行转成普通 dict（缺字段用 getattr 兜底，兼容新旧表结构）。"""
        return {
            "summary": memory.summary or "",
            "facts": getattr(memory, "facts", None) or [],
            "decisions": memory.decisions or [],
            "open_questions": memory.open_questions or [],
            "tasks": getattr(memory, "tasks", None) or [],
            "user_preferences": getattr(memory, "user_preferences", None) or [],
            "repo_context": getattr(memory, "repo_context", None) or [],
            "citations": getattr(memory, "citations", None) or [],
        }

    def _new_memory(self, repo_id: uuid.UUID | None, conversation_id: uuid.UUID | None) -> ConversationMemory | ThreadMemory:
        """新建一条空记忆（会话级优先），并 flush 拿到 id。

        字段全部显式初始化成空容器而不是留 NULL：
        后续的合并逻辑可以直接 extend/append，不用到处判空。
        """
        if repo_id and conversation_id:
            memory = ConversationMemory(
                repo_id=repo_id,
                conversation_id=conversation_id,
                summary="",
                facts=[],
                decisions=[],
                open_questions=[],
                tasks=[],
                user_preferences=[],
                repo_context=[],
                citations=[],
                sessions_incorporated=0,
            )
        else:
            # 没有会话归属时退化成仓库级线程记忆（没有 conversation_id 字段）。
            memory = ThreadMemory(
                repo_id=repo_id,
                summary="",
                facts=[],
                decisions=[],
                open_questions=[],
                tasks=[],
                user_preferences=[],
                repo_context=[],
                citations=[],
                sessions_incorporated=0,
            )
        self.db.add(memory)
        self.db.flush()
        return memory

    def _update_thread_memory(
        self,
        repo_id: uuid.UUID | None,
        conversation_id: uuid.UUID | None,
        digest: str,
        memory_update: dict[str, Any],
    ) -> ConversationMemory | ThreadMemory | None:
        """把这次记忆更新写进记忆行，返回更新后的行。

        两种写入方式：
        - snapshot（模型给了完整快照）：整体覆盖，只把 summary 按 1100 token 截断；
        - 否则（规则版增量）：用 merge_memory 合并追加、去重、限长。
        sessions_incorporated 每次都 +1，用来体现「这条记忆吸收了多少轮会话」，
        它也会显示在上下文里（[Thread Memory: N sessions]），让模型知道记忆的覆盖广度。
        """
        if repo_id and conversation_id:
            memory: ConversationMemory | ThreadMemory | None = self.db.query(ConversationMemory).filter(ConversationMemory.conversation_id == conversation_id).first()
        else:
            memory = self.db.query(ThreadMemory).filter(_repo_filter(ThreadMemory, repo_id)).first()
        if not memory:
            memory = self._new_memory(repo_id, conversation_id)
        if memory_update.get("_update_mode") == "snapshot":
            # 快照模式下直接采用模型给的全量内容（_digest/_update_mode 等下划线字段会被过滤掉）。
            merged = {
                **structured_memory_from_model(memory_update),
                "summary": clip_to_token_budget(str(memory_update.get("summary") or digest), 1100),
            }
        else:
            merged = merge_memory(self._memory_state(memory), memory_update, digest, max_tokens=1100)
        # 逐字段赋值而不是整体替换对象：保留 ORM 身份，也避免误改其它列。
        memory.summary = merged["summary"]
        memory.facts = merged["facts"]
        memory.decisions = merged["decisions"]
        memory.open_questions = merged["open_questions"]
        memory.tasks = merged["tasks"]
        memory.user_preferences = merged["user_preferences"]
        memory.repo_context = merged["repo_context"]
        memory.citations = merged["citations"]
        # 兼容历史 NULL：老行可能没有这个计数。
        memory.sessions_incorporated = (memory.sessions_incorporated or 0) + 1
        return memory

    def _iso(self, value: datetime | None) -> str | None:
        """datetime → ISO 字符串（None 安全），用于写进 JSON 快照。"""
        return value.isoformat() if value else None


class MemoryTools:
    """暴露给 Agent 的三个「记忆类工具」。

    共同点：都是**只读**（不写业务数据，只记一条召回审计），
    并且都返回统一的 {route, tool_calls, answer, citations, results} 结构，
    这样 Agent 侧不用为每个工具写不同的结果解析逻辑。

    为什么单独做工具而不是直接塞进上下文：这些内容可能非常大（完整历史、封存会话），
    让模型按需调用、分批读取，比一次性全塞进 prompt 便宜得多也不会超窗口。
    """

    def __init__(self, db: Session, repo: Repository | None, conversation: Conversation | None = None) -> None:
        self.db = db
        self.repo = repo
        self.repo_id = repo.id if repo else None
        self.conversation = conversation
        self.conversation_id = conversation.id if conversation else None

    async def search_evidence(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """语义/关键词混合检索证据（对应 Agent 的 devflow_search_evidence）。

        context 参数目前未使用，但保留是为了和其它工具保持统一签名
        （Agent 的工具调度器会统一传 context）。
        """
        query = str(input_data.get("query") or input_data.get("message") or "")
        results = EvidenceStore(self.db).search(self.repo_id, self.conversation_id, query, limit=int(input_data.get("limit") or 5))
        # 没有结果时返回一句明确的话而不是空字符串：模型能据此判断「要不要换关键词再试」，
        # 而不是误以为工具调用失败。
        answer = "No evidence matched this query." if not results else "\n".join(
            f"- [{item['source_type']}] {item['title']}: {item['snippet']}" for item in results
        )
        # citations 只带最小定位信息（类型/来源 id/标题），供前端渲染引用角标。
        citations = [{"type": item["source_type"], "id": item["source_id"], "title": item["title"]} for item in results]
        # 记一条召回事件：这是事后分析「Agent 到底查了什么、有没有查到」的唯一依据。
        record_recall_event(
            self.db,
            repo_id=self.repo_id,
            conversation_id=self.conversation_id,
            tool_name="devflow_search_evidence",
            query=query,
            results=results,
            citations=citations,
            # 有会话就是会话级检索，否则是项目级（作用域不同，统计时要区分）。
            scope="conversation" if self.conversation_id else "project",
            mode="hybrid",
            metadata={"limit": int(input_data.get("limit") or 5)},
        )
        return {
            "route": "memory_search",
            # tool_calls 回填一条「我调过这个工具」的记录，便于上层统一统计工具使用情况。
            "tool_calls": [{"tool_name": "search_evidence", "arguments": {"query": query}, "status": "success"}],
            "answer": answer,
            "citations": citations,
            "results": results,
        }

    async def get_thread_context(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """按需精确读取原始对话记录（压缩后恢复细节的关键工具）。

        三种取法：
        - 给 message_id：只取这一条，并支持 offset 分段读（长消息可以多次调用）；
        - 给 keyword：在完整时间线里做包含匹配，再取最后 limit 条；
        - 都不给：取时间线末尾 limit 条。
        max_chars 夹在 200..16000：太小读不到东西，太大一次返回会撑爆上下文。
        """
        keyword = str(input_data.get("keyword") or input_data.get("query") or "").strip().lower()
        # limit 夹在 1..100，max_chars 夹在 200..16000，都是防呆上限。
        limit = min(max(int(input_data.get("limit") or 30), 1), 100)
        message_id = self._uuid_or_none(input_data.get("message_id"))
        before_message_id = self._uuid_or_none(input_data.get("before_message_id"))
        after_message_id = self._uuid_or_none(input_data.get("after_message_id"))
        # offset 只对「指定 message_id」有效（用于分批读完一条超长消息）。
        offset = max(int(input_data.get("offset") or 0), 0)
        max_chars = min(max(int(input_data.get("max_chars") or 6000), 200), 16000)

        store = MessageStore(self.db)
        if message_id:
            message = (
                self.db.query(ChatMessage)
                .filter(
                    _repo_filter(ChatMessage, self.repo_id),
                    _conversation_filter(ChatMessage, self.conversation_id),
                    ChatMessage.id == message_id,
                    # 只允许读对话类消息：system 里可能有内部诊断信息，不该暴露给模型。
                    ChatMessage.role.in_(["user", "assistant", "tool", "tool_result"]),
                )
                .one_or_none()
            )
            # 排除 preserved_after_compact 副本：读到副本会造成内容重复。
            messages = [message] if message and (message.meta or {}).get("subtype") != "preserved_after_compact" else []
        else:
            # limit=None 表示要全量（这里需要先全量筛关键词，再截断）。
            messages = store.timeline(
                self.repo_id,
                self.conversation_id,
                limit=None,
                before_message_id=before_message_id,
                after_message_id=after_message_id,
            )
            if keyword:
                terms = _tokens(keyword)
                # 朴素包含匹配：精确检索场景下（找某个报错原文）比向量检索更可靠。
                messages = [msg for msg in messages if any(term in msg.content.lower() for term in terms)]
            messages = messages[-limit:]

        lines: list[str] = []
        results: list[dict[str, Any]] = []
        # remaining_chars 是全局预算：所有消息共享，用完就停，保证总量可控。
        remaining_chars = max_chars
        for message in messages:
            content = message.content or ""
            # 只有指定了单条消息时才应用 offset，批量模式从每条的开头读。
            chunk_offset = offset if message_id else 0
            chunk = content[chunk_offset : chunk_offset + remaining_chars]
            if not chunk and content:
                # 预算已用完（剩余 0）或已读到末尾，跳过这条继续看下一条。
                continue
            chunk_end = chunk_offset + len(chunk)
            truncated = chunk_end < len(content)
            created_at = message.created_at.isoformat() if message.created_at else ""
            # 头部带 message_id/role/时间/字符区间：模型后续要「续读」时能直接复用这些值。
            lines.append(
                f"[message_id={message.id} role={message.role} created_at={created_at} chars={chunk_offset}:{chunk_end}/{len(content)}]"
            )
            lines.append(chunk)
            if truncated:
                # 明确告诉模型「还没读完 + 怎么继续」，这是分段读取能work的关键提示。
                lines.append(
                    f"[内容未读完；继续调用 message_id={message.id}, offset={chunk_end}]"
                )
            results.append(
                {
                    "title": f"{message.role} message",
                    "source_type": "chat_message",
                    "source_id": str(message.id),
                    "snippet": chunk,
                    "metadata": {
                        "role": message.role,
                        "created_at": created_at,
                        "offset": chunk_offset,
                        # next_offset 供程序化续读（前端/工具直接取用，不用解析上面那句提示）。
                        "next_offset": chunk_end if truncated else None,
                        "total_chars": len(content),
                        "truncated": truncated,
                    },
                }
            )
            remaining_chars -= len(chunk)
            if remaining_chars <= 0:
                break

        citations = [
            {"type": "chat_message", "id": item["source_id"], "title": item["title"]}
            for item in results
        ]
        # 精确读取同样记审计：能看出模型什么时候「回查原文」，以及查了多少。
        record_recall_event(
            self.db,
            repo_id=self.repo_id,
            conversation_id=self.conversation_id,
            tool_name="devflow_get_thread_context",
            query=keyword or str(message_id or ""),
            results=results,
            citations=citations,
            scope="conversation",
            mode="exact_transcript",
            metadata={
                "limit": limit,
                "message_id": str(message_id) if message_id else None,
                "before_message_id": str(before_message_id) if before_message_id else None,
                "after_message_id": str(after_message_id) if after_message_id else None,
                "offset": offset,
                "max_chars": max_chars,
            },
        )
        return {
            "route": "thread_context",
            "tool_calls": [{"tool_name": "get_thread_context", "arguments": {"keyword": keyword, "message_id": str(message_id or "")}, "status": "success"}],
            "answer": "\n".join(lines) if lines else "当前会话还没有匹配的消息。",
            "citations": citations,
            "results": results,
        }

    async def read_session_events(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """读取已封存的会话记录（含可选的完整 transcript）。

        两条路径：
        - 给了 session_id：只读那一条（并支持 offset 分段读）；
        - 没给：读最近几条（limit 夹在 1..20）。
        include_transcript 默认 True：会话摘要（digest）通常很短，
        但排障时往往需要原始 transcript，所以默认带上，需要省 token 时再关掉。
        """
        session_id = input_data.get("session_id")
        include_transcript = bool(input_data.get("include_transcript", True))
        offset = max(int(input_data.get("offset") or 0), 0)
        # max_chars 上下限比线程工具更宽（400..20000）：会话快照通常更大。
        max_chars = min(max(int(input_data.get("max_chars") or 8000), 400), 20000)
        query = self.db.query(ChatSession).filter(
            _repo_filter(ChatSession, self.repo_id),
            _conversation_filter(ChatSession, self.conversation_id),
        ).order_by(ChatSession.created_at.desc())
        if session_id:
            parsed_session_id = self._uuid_or_none(session_id)
            if parsed_session_id is None:
                # 非法 id 直接返回空，而不是抛异常：模型可能传了非 UUID 的字符串，
                # 报错会让它以为工具坏了。
                sessions = []
            else:
                sessions = query.filter(ChatSession.id == parsed_session_id).limit(1).all()
        else:
            sessions = query.limit(min(max(int(input_data.get("limit") or 5), 1), 20)).all()
        lines: list[str] = []
        results: list[dict[str, Any]] = []
        remaining_chars = max_chars
        for session in sessions:
            # 头部带 route 和封存时间，方便判断这条会话属于哪条链路、什么时候结束的。
            header = f"Session {session.id} route={session.route} sealed_at={session.sealed_at}"
            payload = session.digest
            if include_transcript:
                # 紧凑序列化（无空格）以尽量省字符预算。
                payload += "\ntranscript=" + dumps_compact(session.transcript_json or {})
            chunk_offset = offset if session_id else 0
            chunk = payload[chunk_offset : chunk_offset + remaining_chars]
            chunk_end = chunk_offset + len(chunk)
            truncated = chunk_end < len(payload)
            lines.append(header)
            lines.append(chunk)
            if truncated:
                # 同线程工具：给出续读参数，让模型可以分批读完一大段 transcript。
                lines.append(
                    f"[会话记录未读完；继续调用 session_id={session.id}, offset={chunk_end}。]"
                )
            results.append(
                {
                    "title": f"Session {session.route}",
                    "source_type": "session",
                    "source_id": str(session.id),
                    "snippet": chunk,
                    "metadata": {
                        "sealed_at": session.sealed_at.isoformat() if session.sealed_at else None,
                        "include_transcript": include_transcript,
                        "offset": chunk_offset,
                        "next_offset": chunk_end if truncated else None,
                        "total_chars": len(payload),
                        "truncated": truncated,
                    },
                }
            )
            remaining_chars -= len(chunk)
            if remaining_chars <= 0:
                break
        citations = [
            {"type": "session", "id": item["source_id"], "title": item["title"]}
            for item in results
        ]
        record_recall_event(
            self.db,
            repo_id=self.repo_id,
            conversation_id=self.conversation_id,
            tool_name="devflow_read_session_events",
            query=str(session_id or "latest sessions"),
            results=results,
            citations=citations,
            scope="conversation",
            mode="sealed_transcript",
            metadata={"include_transcript": include_transcript, "offset": offset, "max_chars": max_chars},
        )
        return {
            "route": "session_events",
            "tool_calls": [{"tool_name": "read_session_events", "arguments": {"session_id": str(session_id or "")}, "status": "success"}],
            "answer": "\n\n".join(lines) if lines else "No sealed sessions found yet.",
            "citations": citations,
            "results": results,
        }

    @staticmethod
    def _uuid_or_none(value: Any) -> uuid.UUID | None:
        """把任意输入宽松解析成 UUID，解析不了返回 None。

        模型传进来的 id 可能是空串、None、带引号的字符串甚至随意编造的内容，
        这里统一转成 None 让调用方按「参数无效」处理，而不是让整个工具调用抛异常。
        """
        if not value:
            return None
        try:
            return uuid.UUID(str(value))
        except (TypeError, ValueError):
            return None
