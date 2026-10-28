"""仓库「知识库 / 项目记忆」接口，挂在 /api/repos 前缀下。

这里是 RAG 的数据入口，功能可归为四块：
    1. 知识条目：列表、手写记忆笔记、上传文档（解析→切块→向量化→入库）、删除；
    2. 记忆检索：按关键字走混合检索，并记录 recall 事件（谁搜了什么、命中了什么）；
    3. 记忆候选：对话中自动抽出的「待确认记忆」，人工 approve / reject；
    4. 知识图谱与团队画像：图谱查询/重建，团队成员的增删查。

关键设计：所有内容最终都落在 documents 表——一个逻辑文档会被切成多个 chunk 行，
它们共享 (source_type, source_id)，读取时再按这个键聚合回一个条目（见 _group_documents）。
"""

import asyncio
import hashlib
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import Document, Repository
from app.db.session import get_db
from app.schemas.knowledge import (
    KnowledgeGraphRebuildResponse,
    KnowledgeGraphResponse,
    KnowledgeItemResponse,
    KnowledgeUploadResponse,
    MemoryCandidateActionResponse,
    MemoryCandidateResponse,
    MemorySearchResponse,
    MemoryStatusResponse,
    MemoryNoteRequest,
    RecallEventResponse,
    TeamMemberRequest,
    TeamMemberResponse,
)
from app.services.knowledge_graph import knowledge_graph_payload, rebuild_knowledge_graph
from app.services.chat_memory import EvidenceStore
from app.services.memory_hub import (
    approve_memory_candidate,
    list_memory_candidates,
    list_recall_events,
    memory_status,
    record_recall_event,
    reject_memory_candidate,
)
from app.services.rag.chunking import chunk_document
from app.services.rag.document_extraction import MAX_DOCUMENT_BYTES, extract_document_text
from app.services.rag.document_processing import SUPPORTED_KNOWLEDGE_SUFFIXES
from app.services.rag.embeddings import embed_texts, embedding_provider_name
from app.services.rag.vector_store import DocumentPayload, add_document, add_documents, delete_document_group, stable_source_id

router = APIRouter()

# 会出现在「知识库」列表里的 source_type 白名单。
# 刻意是白名单而非黑名单：代码符号、issue 之类的数据也存在 documents 表里，
# 但它们是索引产物，不该混进用户维护的知识列表。
USER_KNOWLEDGE_SOURCE_TYPES = [
    "memory_note",
    "knowledge_file",
    "team_member",
    "weekly_report",
    "project_overview",
    "project_doc",
    "project_manifest",
]
# 调 embedding 接口时每批 64 条：单次请求体过大容易被服务端拒绝或超时，
# 分批还能让进度/失败重试的粒度更细。
EMBEDDING_BATCH_SIZE = 64


def _repo_or_404(db: Session, repo_id: uuid.UUID) -> Repository:
    """取仓库，不存在则 404。几乎所有接口都先过这一关，防止给幽灵仓库写数据。"""
    repo = db.get(Repository, repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到仓库")
    return repo


def _clip(text: str, limit: int = 360) -> str:
    """把文本压成列表页用的预览串。

    入参：text 原文；limit 结果最大长度（默认 360）。
    返回：超长时截断并补 "..."，总长不超过 limit。
    说明：
        先做 CRLF→LF 归一，否则 Windows 文本里的 \\r 会在前端显示成多余空行；
        截断时留 1 个字符给 "..."，再用 rstrip 去掉断点处的空格，
        避免出现「xxx ...」这种尾巴带空格的难看效果。
    """
    text = (text or "").strip().replace("\r\n", "\n")
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "..."


def _safe_filename(filename: str) -> str:
    """清洗上传文件名，防目录穿越、防非法字符。

    入参：浏览器传来的原始 filename（可能含 ../、路径分隔符）。
    返回：安全的裸文件名。
    说明：
        先 Path(...).name 只保留最后一段（干掉 ../../etc/passwd 这类路径），
        再把白名单外的字符（字母/数字/点/下划线/连字符/中文）替换成下划线。
        留中文是为了让用户下载时还能看出文件是什么。
    """
    value = Path(filename or "upload.txt").name
    return re.sub(r"[^a-zA-Z0-9._\-\u4e00-\u9fff]+", "_", value).strip("._") or "upload.txt"


def _upload_root(repo_id: uuid.UUID) -> Path:
    """返回（并确保存在）某仓库的上传目录。

    副作用：mkdir(parents=True, exist_ok=True) 会真实创建目录。
    相对路径按当前工作目录解析（而不是按代码文件位置），
    这样同一份配置在 uvicorn / pytest 下行为一致，取决于启动目录。
    """
    root = Path(settings.upload_dir)
    if not root.is_absolute():
        root = Path.cwd() / root
    path = root / str(repo_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _extract_upload_text(filename: str, data: bytes) -> str:
    """从上传文件里抽文本，失败时返回空串而不是抛异常。

    返回空串后由调用方统一报 400「无法提取文本」——这样「解析失败」和
    「文件里确实没字」对用户是同一种反馈，不用区分底层是 PDF 加密还是编码错误。
    """
    try:
        return extract_document_text(filename, data)
    except Exception:
        return ""


def _embed_default_in_batches(texts: list[str]) -> list[list[float]]:
    """分批调用 embedding，返回与输入一一对应的向量列表。

    入参：texts 待向量化文本。
    返回：向量列表，顺序与 texts 严格一致（后续要用它和 chunk 一一对应）。
    说明：切片用 range(0, len, BATCH)，最后一批可能不足 BATCH_SIZE，切片会自动截断。
    """
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        vectors.extend(
            embed_texts(
                texts[start : start + EMBEDDING_BATCH_SIZE],
                dimensions=settings.embedding_dimensions,
            )
        )
    return vectors


def _group_documents(documents: list[Document]) -> list[KnowledgeItemResponse]:
    """把「多行 chunk」聚合成「一个知识条目」，供前端列表展示。

    入参：documents 里的若干行（同一逻辑文档的多个切片会重复出现）。
    返回：按创建时间倒序的条目列表。
    说明：
        1. 用 (source_type, source_id) 作分组键：source_id 是逻辑文档 ID，
           同一文档的所有 chunk 共用它；
        2. 组内按 chunk_index 排序，保证预览正文是「从开头连续的两段」而不是随机切片；
        3. 预览只拼前 2 个 chunk（再多列表就太重了），完整内容要看详情；
        4. 时间倒序排序时用 datetime.min.replace(tzinfo=utc) 兜底 created_at 为空，
           且必须是 aware 时间，否则和数据库里的 aware 时间比较会抛 TypeError。
    """
    groups: dict[tuple[str, uuid.UUID], list[Document]] = {}
    for doc in documents:
        groups.setdefault((doc.source_type, doc.source_id), []).append(doc)

    items: list[KnowledgeItemResponse] = []
    for (source_type, source_id), docs in groups.items():
        docs.sort(key=lambda item: ((item.meta or {}).get("chunk_index", 0), item.created_at))
        first = docs[0]
        content = "\n".join(doc.content for doc in docs[:2])
        metadata = dict(first.meta or {})
        # 兜底写入 source_type：部分历史数据的 meta 里没有这个键，前端靠它选图标。
        metadata.setdefault("source_type", source_type)
        items.append(
            KnowledgeItemResponse(
                id=source_id,
                document_ids=[doc.id for doc in docs],
                source_type=source_type,
                title=metadata.get("display_title") or first.title,
                content_preview=_clip(content),
                metadata=metadata,
                chunk_count=len(docs),
                created_at=first.created_at,
            )
        )
    items.sort(key=lambda item: item.created_at or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    return items


def _team_from_document(doc: Document) -> TeamMemberResponse:
    """把一条 team_member 文档转成团队成响应。

    名称优先取 meta.name；老数据没有 meta 时从标题「团队成员：张三」里剥出来，
    兼容两种历史写法（中文冒号 / 英文 Team member: ）。
    """
    meta = doc.meta or {}
    return TeamMemberResponse(
        id=doc.source_id,
        name=str(meta.get("name") or doc.title.replace("团队成员：", "").replace("Team member: ", "")),
        role=str(meta.get("role") or ""),
        strengths=str(meta.get("strengths") or ""),
        # tech_stack / techStack 两种键都认：不同时期写入的字段名不一致。
        techStack=str(meta.get("tech_stack") or meta.get("techStack") or ""),
        created_at=doc.created_at,
    )


@router.get("/{repo_id}/knowledge", response_model=list[KnowledgeItemResponse])
async def list_knowledge(repo_id: uuid.UUID, db: Session = Depends(get_db)) -> list[KnowledgeItemResponse]:
    """GET /api/repos/{repo_id}/knowledge：知识库列表（已按逻辑文档聚合）。

    入参：repo_id（路径）；db 由 get_db 注入。
    返回：KnowledgeItemResponse 列表，未分页。
    异常：仓库不存在 → 404。
    副作用：只读。内部会拉出该仓库全部相关文档行（含所有 chunk），数据量大时较重。
    """
    _repo_or_404(db, repo_id)
    docs = (
        db.query(Document)
        .filter(Document.repo_id == repo_id, Document.source_type.in_(USER_KNOWLEDGE_SOURCE_TYPES))
        .order_by(Document.created_at.desc())
        .all()
    )
    return _group_documents(docs)


@router.get("/{repo_id}/memory/search", response_model=MemorySearchResponse)
async def search_memory(
    repo_id: uuid.UUID,
    q: str = Query(default=""),
    limit: int = Query(default=8, ge=1, le=30),
    db: Session = Depends(get_db),
) -> MemorySearchResponse:
    """GET /api/repos/{repo_id}/memory/search：手动检索项目记忆。

    入参：
        q: 关键字；空串时不检索（直接返回空结果），避免把整个库都倒出来。
        limit: 1~30，默认 8。
    返回：MemorySearchResponse(query, results, result_count)。
    异常：仓库不存在 → 404。
    副作用：**这是一个会写库的 GET**——无论结果多少都会记一条 recall 事件
        （tool_name=manual_memory_search），用于分析「用户到底在找什么」；
        所以这里必须 commit。
    """
    repo = _repo_or_404(db, repo_id)
    query = q.strip()
    results = EvidenceStore(db).search(repo.id, None, query, limit=limit) if query else []
    record_recall_event(
        db,
        repo_id=repo.id,
        conversation_id=None,
        tool_name="manual_memory_search",
        query=query,
        results=results,
        citations=[{"type": item["source_type"], "id": item["source_id"], "title": item["title"]} for item in results],
        scope="project",
        mode="hybrid",
        metadata={"limit": limit, "surface": "workspace_memory"},
    )
    db.commit()
    return MemorySearchResponse(query=query, results=results, result_count=len(results))


@router.get("/{repo_id}/memory/recall-events", response_model=list[RecallEventResponse])
async def get_recall_events(
    repo_id: uuid.UUID,
    conversation_id: uuid.UUID | None = Query(default=None),
    limit: int = Query(default=30, ge=1, le=100),
    db: Session = Depends(get_db),
) -> list[RecallEventResponse]:
    """GET /api/repos/{repo_id}/memory/recall-events：查看记忆召回历史。

    入参：conversation_id 可选（只看某会话的召回）；limit 1~100，默认 30。
    返回：按时间倒序的召回事件列表。
    异常：仓库不存在 → 404。
    副作用：只读。
    """
    _repo_or_404(db, repo_id)
    return [RecallEventResponse.model_validate(item) for item in list_recall_events(db, repo_id=repo_id, conversation_id=conversation_id, limit=limit)]


@router.get("/{repo_id}/memory/status", response_model=MemoryStatusResponse)
async def get_memory_status(repo_id: uuid.UUID, db: Session = Depends(get_db)) -> MemoryStatusResponse:
    """GET /api/repos/{repo_id}/memory/status：项目记忆的整体状态概览。

    入参：repo_id。
    返回：MemoryStatusResponse（各类记忆/候选的数量、覆盖率等，具体由 memory_status 组装）。
    异常：仓库不存在 → 404。
    副作用：只读。
    """
    repo = _repo_or_404(db, repo_id)
    return MemoryStatusResponse.model_validate(memory_status(db, repo))


@router.get("/{repo_id}/memory/candidates", response_model=list[MemoryCandidateResponse])
async def get_memory_candidates(
    repo_id: uuid.UUID,
    status: str | None = Query(default="pending"),
    limit: int = Query(default=50, ge=1, le=100),
    db: Session = Depends(get_db),
) -> list[MemoryCandidateResponse]:
    """GET /api/repos/{repo_id}/memory/candidates：查看「待确认记忆」候选。

    入参：
        status: 默认 "pending"（只看待审的）；传 None/空则不按状态过滤。
        limit: 1~100，默认 50。
    返回：候选记忆列表。
    异常：仓库不存在 → 404。
    副作用：只读。
    背景：对话结束后系统会从对话里抽取「值得长期记住」的事实，但不敢直接入库，
        先存成候选，由人在这里确认（approve）后才成为正式记忆。
    """
    _repo_or_404(db, repo_id)
    return [MemoryCandidateResponse.model_validate(item) for item in list_memory_candidates(db, repo_id=repo_id, status=status, limit=limit)]


@router.post("/{repo_id}/memory/candidates/{candidate_id}/approve", response_model=MemoryCandidateActionResponse)
async def approve_candidate(repo_id: uuid.UUID, candidate_id: uuid.UUID, db: Session = Depends(get_db)) -> MemoryCandidateActionResponse:
    """POST /api/repos/{repo_id}/memory/candidates/{candidate_id}/approve：采纳候选记忆。

    入参：repo_id、candidate_id（路径）。
    返回：MemoryCandidateActionResponse(id, status, document_id)。
    异常：候选不存在（service 抛 ValueError）→ 404，detail 为机器可读的 "candidate_not_found"。
    副作用：
        把候选写成正式记忆文档，随后**重建证据库与知识图谱**（两者都依赖全部文档，
        所以必须整体重建而不是增量更新），最后统一 commit。
        仓库大时这一步会比较慢。
    """
    repo = _repo_or_404(db, repo_id)
    try:
        candidate, doc = approve_memory_candidate(db, repo, candidate_id)
    except ValueError:
        # from None：屏蔽掉 ValueError 的上下文，避免 500 风格的堆栈混进 404 的响应日志。
        raise HTTPException(status_code=404, detail="candidate_not_found") from None
    EvidenceStore(db).rebuild_repo(repo)
    rebuild_knowledge_graph(db, repo)
    db.commit()
    return MemoryCandidateActionResponse(id=candidate.id, status=candidate.status, document_id=doc.id if doc else None)


@router.post("/{repo_id}/memory/candidates/{candidate_id}/reject", response_model=MemoryCandidateActionResponse)
async def reject_candidate(repo_id: uuid.UUID, candidate_id: uuid.UUID, db: Session = Depends(get_db)) -> MemoryCandidateActionResponse:
    """POST /api/repos/{repo_id}/memory/candidates/{candidate_id}/reject：拒绝候选记忆。

    入参：repo_id、candidate_id（路径）。
    返回：MemoryCandidateActionResponse（document_id 为 None，因为没有生成文档）。
    异常：候选不存在 → 404 "candidate_not_found"。
    副作用：只改候选状态，**不重建**证据库/图谱——拒绝不会改变已有数据。
    """
    repo = _repo_or_404(db, repo_id)
    try:
        candidate = reject_memory_candidate(db, repo, candidate_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="candidate_not_found") from None
    db.commit()
    return MemoryCandidateActionResponse(id=candidate.id, status=candidate.status)


@router.post("/{repo_id}/knowledge/notes", response_model=KnowledgeItemResponse)
async def create_memory_note(
    repo_id: uuid.UUID,
    payload: MemoryNoteRequest,
    db: Session = Depends(get_db),
) -> KnowledgeItemResponse:
    """POST /api/repos/{repo_id}/knowledge/notes：手写一条记忆笔记。

    入参：payload 含 title / content。
    返回：KnowledgeItemResponse（包装成和列表项一致的结构，前端可直接插进列表）。
    异常：仓库不存在 → 404；标题与内容同时为空 → 400
        （实际上 title 已兜底为「未命名记忆」，这条分支很少触发，保留作为最后防线）。
    副作用：写一条 memory_note 文档并重建知识图谱。
        注意 add_document 内部会 commit，所以这里没有再显式提交。
    """
    _repo_or_404(db, repo_id)
    content = payload.content.strip()
    title = payload.title.strip() or "未命名记忆"
    if not content and not title:
        raise HTTPException(status_code=400, detail="记忆标题或内容至少需要填写一项")
    source_id = uuid.uuid4()
    doc = add_document(
        db,
        repo_id,
        "memory_note",
        source_id,
        title,
        # 只写标题、没写正文时，用标题当正文：documents.content 不能为空（要做向量化）。
        content or title,
        {"display_title": title, "created_by": "user"},
    )
    rebuild_knowledge_graph(db, _repo_or_404(db, repo_id))
    # 复用分组逻辑把单条文档包成列表项，保证返回结构和 GET 列表页完全一致。
    return _group_documents([doc])[0]


@router.post("/{repo_id}/knowledge/upload", response_model=KnowledgeUploadResponse)
async def upload_knowledge_file(
    repo_id: uuid.UUID,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> KnowledgeUploadResponse:
    """POST /api/repos/{repo_id}/knowledge/upload：上传文档进知识库。

    入参：
        file: multipart 文件字段，必填。
    返回：
        KnowledgeUploadResponse(id, filename, chunks, indexed, duplicate)；
        命中相同内容时 duplicate=True 且直接返回已有条目，不重复入库。
    异常：
        空文件 → 400；超过 MAX_DOCUMENT_BYTES（12MB）→ 400；
        后缀不在白名单 → 415（Unsupported Media Type，比 400 更准确地表达「类型不支持」）；
        抽不出文本 / 切不出块 → 400。
    副作用：
        1. 原文件落盘到 upload_dir/{repo_id}/{source_id}-{filename}；
        2. documents 表写入 N 条 chunk；
        3. 重建知识图谱。
    实现要点：
        解析、切块、embedding 都是 CPU/网络密集操作，全部丢进 asyncio.to_thread，
        否则会阻塞事件循环，导致同时进行的聊天请求卡住。
    """
    repo = _repo_or_404(db, repo_id)
    # 多读 1 个字节：既能判断「是否超限」，又不必把超大文件整个读进内存。
    data = await file.read(MAX_DOCUMENT_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(data) > MAX_DOCUMENT_BYTES:
        raise HTTPException(status_code=400, detail="Uploaded file is too large; max size is 12 MB")

    filename = _safe_filename(file.filename or "upload.txt")
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_KNOWLEDGE_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_KNOWLEDGE_SUFFIXES))
        raise HTTPException(status_code=415, detail=f"Unsupported file type. Supported: {supported}")
    # 用内容哈希做去重键（而不是文件名）：同一份文件改名重传也不会产生第二份索引。
    checksum = hashlib.sha256(data).hexdigest()
    existing = (
        db.query(Document)
        .filter(Document.repo_id == repo_id, Document.source_type == "knowledge_file")
        .order_by(Document.created_at.asc())
        .all()
    )
    duplicate_docs = [doc for doc in existing if (doc.meta or {}).get("checksum") == checksum]
    if duplicate_docs:
        # 命中重复：返回已有条目的信息，chunks 用「已存在的行数」而不是本次的上传结果。
        return KnowledgeUploadResponse(
            id=duplicate_docs[0].source_id,
            filename=str((duplicate_docs[0].meta or {}).get("filename") or filename),
            chunks=len(duplicate_docs),
            indexed=True,
            duplicate=True,
        )
    # 解析可能很耗时（PDF/Word），放线程池里跑，避免阻塞事件循环。
    text = await asyncio.to_thread(_extract_upload_text, filename, data)
    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from this file")

    source_id = uuid.uuid4()
    stored_path = _upload_root(repo_id) / f"{source_id}-{filename}"
    # 1800/180：每块约 1800 字符、相邻块重叠 180 字符。
    # 重叠是为了防止「一句话被切在边界上」导致检索时两半都不完整。
    chunks = await asyncio.to_thread(chunk_document, text, filename, 1800, 180)
    if not chunks:
        raise HTTPException(status_code=400, detail="Document did not produce indexable chunks")
    payloads = [
        DocumentPayload(
            source_type="knowledge_file",
            source_id=source_id,
            # 每个 chunk 的 title 带序号，便于在检索结果里定位是文档的哪一段。
            title=f"{filename}#{index + 1}",
            content=str(chunk["content"]),
            metadata={
                # 把 chunk 自带的元信息（标题层级等）展开进来；content 单独作为正文字段，
                # 值为 None 的键直接剔掉，避免元数据里塞一堆 null。
                **{key: value for key, value in chunk.items() if key != "content" and value is not None},
                "display_title": filename,
                "filename": filename,
                "stored_path": str(stored_path),
                "content_type": file.content_type,
                "checksum": checksum,
                "file_size": len(data),
                "chunk_index": index,
                "chunk_chars": len(str(chunk["content"])),
                "total_chunks": len(chunks),
                # 记录切块策略版本：将来换算法后，新旧数据可以区分，便于重建索引。
                "chunk_strategy": "structure_aware_v1",
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
                # 记录当时用的 embedding 提供方/模型/维度：换模型后能识别出哪些向量已过期。
                "embedding_provider": embedding_provider_name(),
                "embedding_model": settings.embedding_model,
                "embedding_dimensions": settings.embedding_dimensions,
            },
        )
        for index, chunk in enumerate(chunks)
    ]
    # 向量化输入带上标题前缀：chunk 脱离了文档上下文时，标题能显著提升召回准确性。
    embedding_inputs = [f"{payload.title}\n{payload.content}" for payload in payloads]
    vectors = await asyncio.to_thread(_embed_default_in_batches, embedding_inputs)
    try:
        await asyncio.to_thread(stored_path.write_bytes, data)
        docs = add_documents(db, repo_id, payloads, commit=False, vectors=vectors)
        rebuild_knowledge_graph(db, repo)
    except Exception:
        # 补偿式回滚：先回滚数据库，再删已经落盘的文件。
        # 不删的话会留下孤儿文件；而且下次重传同名内容不会被去重逻辑命中（库里没记录），
        # 会再写一份，磁盘上就越积越多。
        db.rollback()
        try:
            stored_path.unlink(missing_ok=True)
        except OSError:
            # 删文件失败（被占用等）不该掩盖真正的业务异常，所以直接忽略。
            pass
        raise
    return KnowledgeUploadResponse(id=source_id, filename=filename, chunks=len(docs), indexed=bool(docs))


@router.get("/{repo_id}/knowledge/graph", response_model=KnowledgeGraphResponse)
async def get_knowledge_graph(
    repo_id: uuid.UUID,
    center_type: str | None = Query(default=None),
    center_id: str | None = Query(default=None),
    depth: int = Query(default=1, ge=1, le=2),
    db: Session = Depends(get_db),
) -> KnowledgeGraphResponse:
    """GET /api/repos/{repo_id}/knowledge/graph：查询知识图谱。

    入参：
        center_type / center_id: 以某个节点为中心取子图；不传则返回全图。
        depth: 展开层数，1~2（限制为 2 是因为层数越大节点数指数增长，前端也画不下）。
    返回：KnowledgeGraphResponse（nodes + edges）。
    异常：仓库不存在 → 404。
    副作用：只读。
    """
    repo = _repo_or_404(db, repo_id)
    return KnowledgeGraphResponse.model_validate(
        knowledge_graph_payload(db, repo, center_type=center_type, center_id=center_id, depth=depth)
    )


@router.post("/{repo_id}/knowledge/graph/rebuild", response_model=KnowledgeGraphRebuildResponse)
async def rebuild_knowledge_graph_route(repo_id: uuid.UUID, db: Session = Depends(get_db)) -> KnowledgeGraphRebuildResponse:
    """POST /api/repos/{repo_id}/knowledge/graph/rebuild：手动重建知识图谱。

    入参：repo_id。
    返回：KnowledgeGraphRebuildResponse（重建后的边数、节点数）。
    异常：仓库不存在 → 404。
    副作用：同步全量重建图谱（遍历该仓库全部文档），会比较慢；日常由增删文档时自动触发，
        这个接口是给「图谱看着不对」时手动兜底用的。
    """
    repo = _repo_or_404(db, repo_id)
    result = rebuild_knowledge_graph(db, repo)
    return KnowledgeGraphRebuildResponse(repo_id=repo.id, edges=result["edges"], nodes=result["nodes"])


@router.delete("/{repo_id}/knowledge/{source_id}")
async def delete_knowledge_item(
    repo_id: uuid.UUID,
    source_id: uuid.UUID,
    source_type: str | None = Query(default=None),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """DELETE /api/repos/{repo_id}/knowledge/{source_id}：删除一条知识（含其全部 chunk）。

    入参：
        source_id: 逻辑文档 ID（不是单条 chunk 的主键）。
        source_type: 可选，进一步限定类型；不传则删该 source_id 下所有类型的行。
    返回：{"repo_id", "source_id", "deleted"}，deleted 为实际删除的 chunk 行数。
    异常：仓库不存在 → 404。source_id 不存在时不报错，deleted=0。
    副作用：删库 + 重建知识图谱。**磁盘上的原始上传文件不会被删**（只有 DB 记录会清掉），
        需要清理存储的话得另外处理。
    """
    _repo_or_404(db, repo_id)
    count = delete_document_group(db, repo_id, source_id, source_type=source_type)
    rebuild_knowledge_graph(db, _repo_or_404(db, repo_id))
    return {"repo_id": str(repo_id), "source_id": str(source_id), "deleted": count}


@router.get("/{repo_id}/team-members", response_model=list[TeamMemberResponse])
async def list_team_members(repo_id: uuid.UUID, db: Session = Depends(get_db)) -> list[TeamMemberResponse]:
    """GET /api/repos/{repo_id}/team-members：列出团队成员。

    入参：repo_id。
    返回：按创建时间正序（先配的排前面）的成员列表。
    异常：仓库不存在 → 404。
    副作用：只读。团队成员本质是 Document(source_type="team_member")，
        所以 Issue 分析里能直接把它们当「团队画像」证据用。
    """
    _repo_or_404(db, repo_id)
    docs = (
        db.query(Document)
        .filter(Document.repo_id == repo_id, Document.source_type == "team_member")
        .order_by(Document.created_at.asc())
        .all()
    )
    return [_team_from_document(doc) for doc in docs]


@router.post("/{repo_id}/team-members", response_model=TeamMemberResponse)
async def create_team_member(
    repo_id: uuid.UUID,
    payload: TeamMemberRequest,
    db: Session = Depends(get_db),
) -> TeamMemberResponse:
    """POST /api/repos/{repo_id}/team-members：新增/更新团队成员。

    入参：payload 含 name（必填）、role、strengths、tech_stack。
    返回：TeamMemberResponse。
    异常：仓库不存在 → 404；姓名为空 → 400。
    副作用：写一条 team_member 文档并重建知识图谱。
    幂等设计：
        source_id 由 stable_source_id(repo_id, "team_member", 姓名小写) 生成——
        同一个人重复提交会落到同一个 source_id 上，于是「先删旧行再插新行」就等价于更新，
        不会出现两个「张三」。
    """
    _repo_or_404(db, repo_id)
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="团队成员姓名为必填项")
    # 正文用 key: value 行格式：既方便人读，也方便检索/模型按字段取用。
    content = "\n".join(
        [
            f"name: {name}",
            f"role: {payload.role.strip()}",
            f"strengths: {payload.strengths.strip()}",
            f"tech_stack: {payload.tech_stack.strip()}",
        ]
    )
    source_id = stable_source_id(repo_id, "team_member", name.lower())
    # commit=False：删旧 + 插新必须在一个事务里，否则中间失败会出现「人没了」的空窗。
    delete_document_group(db, repo_id, source_id, source_type="team_member", commit=False)
    doc = add_document(
        db,
        repo_id,
        "team_member",
        source_id,
        f"团队成员：{name}",
        content,
        {
            "display_title": name,
            "name": name,
            "role": payload.role.strip(),
            "strengths": payload.strengths.strip(),
            "tech_stack": payload.tech_stack.strip(),
        },
    )
    rebuild_knowledge_graph(db, _repo_or_404(db, repo_id))
    return _team_from_document(doc)


@router.delete("/{repo_id}/team-members/{member_id}")
async def delete_team_member(repo_id: uuid.UUID, member_id: uuid.UUID, db: Session = Depends(get_db)) -> dict[str, Any]:
    """DELETE /api/repos/{repo_id}/team-members/{member_id}：删除团队成员。

    入参：member_id 就是该成员文档的 source_id。
    返回：{"repo_id", "member_id", "deleted"}。
    异常：仓库不存在 → 404；成员不存在时 deleted=0，不报错。
    副作用：删文档 + 重建知识图谱，之后 Issue 分析就不会再推荐这个人。
    """
    _repo_or_404(db, repo_id)
    count = delete_document_group(db, repo_id, member_id, source_type="team_member")
    rebuild_knowledge_graph(db, _repo_or_404(db, repo_id))
    return {"repo_id": str(repo_id), "member_id": str(member_id), "deleted": count}
