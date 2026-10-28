"""RAG 知识库管理接口，挂载前缀 /api/rag。

这是「知识库引擎」的控制台，和 knowledge.py 的区别：
    knowledge.py   面向「仓库项目记忆」，数据直接进 documents 表；
    rag.py         面向「可配置的 RAG 知识库」——多一层 KnowledgeSourceDocument 做文档管理，
                   支持切片参数、embedding 模型、检索阈值等配置，以及检索效果测试。

接口分组：
    模型目录      GET  /models
    知识库配置    GET  /repositories、GET/PATCH /repositories/{id}[/config]
    文档管理      GET/POST /repositories/{id}/documents、chunks、retry、DELETE
    检索测试      POST/GET /repositories/{id}/retrieval-tests
    运行时        GET  /{repo_id}/status、POST /{repo_id}/search、/ask、/reindex

路径声明顺序有讲究：静态段（/models、/repositories）必须写在 /{repo_id}/... 之前，
否则会被当成 UUID 参数去解析，直接 422。
"""

import asyncio
import hashlib
import re
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import (
    Document,
    KnowledgeSourceDocument,
    Repository,
    RetrievalTestRun,
)
from app.db.session import get_db
from app.schemas.rag import (
    KnowledgeBaseConfigUpdate,
    KnowledgeBaseResponse,
    KnowledgeChunkResponse,
    KnowledgeDocumentResponse,
    ModelCatalogResponse,
    RAGAnswerResponse,
    RAGQueryRequest,
    RAGReindexResponse,
    RAGSearchResponse,
    RetrievalTestHistoryResponse,
    RetrievalTestRequest,
    RetrievalTestResponse,
)
from app.services.rag.document_extraction import MAX_DOCUMENT_BYTES
from app.services.rag.document_processing import SUPPORTED_KNOWLEDGE_SUFFIXES, process_knowledge_document
from app.services.rag.embeddings import validate_embedding_contract
from app.services.rag.knowledge_base import (
    config_dict,
    get_or_create_config,
    model_catalog,
    record_retrieval_test,
    retrieve_knowledge,
    runtime_status,
)
from app.services.rag.qa import answer_question, build_sources, rag_status, reindex_documents
from app.services.rag.vector_store import delete_document_group

router = APIRouter()


def _repo_or_404(db: Session, repo_id: uuid.UUID) -> Repository:
    """取仓库，且拒绝 provider="knowledge_base" 的记录。

    为什么多这一层判断：知识库本身也以 Repository 行的形式存在（provider 标记），
    如果在知识库上再开知识库，会出现「库中库」的套娃和递归统计，直接当 404 处理最省事。
    """
    repo = db.get(Repository, repo_id)
    if repo is None or repo.provider == "knowledge_base":
        raise HTTPException(status_code=404, detail="未找到仓库")
    return repo


def _safe_filename(filename: str) -> str:
    """清洗上传文件名：只留最后一段，非法字符换成下划线（保留中文），防目录穿越。"""
    value = Path(filename or "upload.txt").name
    return re.sub(r"[^a-zA-Z0-9._\-\u4e00-\u9fff]+", "_", value).strip("._") or "upload.txt"


def _upload_root(repo_id: uuid.UUID) -> Path:
    """返回该知识库的上传目录（不存在则创建）。相对 upload_dir 按进程 cwd 解析。"""
    root = Path(settings.upload_dir)
    if not root.is_absolute():
        root = Path.cwd() / root
    path = root / str(repo_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _validate_chunking(chunk_size: int, overlap: int) -> None:
    """校验切片参数：重叠必须严格小于切片长度。

    否则切片窗口原地踏步（步长 ≤ 0），切块循环会死循环或产出无限重复块。
    """
    if overlap >= chunk_size:
        raise HTTPException(status_code=400, detail="切片重叠必须小于切片长度")


def _document_response(item: KnowledgeSourceDocument) -> KnowledgeDocumentResponse:
    """ORM → 响应模型的字段映射（逐个显式列出，避免把内部字段意外暴露给前端）。"""
    return KnowledgeDocumentResponse(
        id=item.id,
        repo_id=item.repo_id,
        name=item.name,
        source_type=item.source_type,
        status=item.status,
        content_type=item.content_type,
        file_size=item.file_size,
        character_count=item.character_count,
        chunk_count=item.chunk_count,
        error_message=item.error_message,
        created_at=item.created_at,
        updated_at=item.updated_at,
        completed_at=item.completed_at,
    )


def _knowledge_base(db: Session, repo: Repository) -> KnowledgeBaseResponse:
    """组装一个知识库的完整视图：基础信息 + 统计 + 配置 + 运行时状态。

    统计口径：
        document_count / character_count 来自 KnowledgeSourceDocument（原始文档）；
        chunk_count 来自 Document（实际切片行数）。
        前者是「用户上传了几个文件」，后者是「切成了多少块」，两者不等是正常的。
    副作用：get_or_create_config 可能顺手创建默认配置（写库）。
    """
    config = get_or_create_config(db, repo.id)
    # coalesce 保证 SUM 在无数据时返回 0 而不是 NULL，省去 Python 侧判空。
    document_count, character_count = (
        db.query(func.count(KnowledgeSourceDocument.id), func.coalesce(func.sum(KnowledgeSourceDocument.character_count), 0))
        .filter(KnowledgeSourceDocument.repo_id == repo.id)
        .one()
    )
    chunk_count = db.query(func.count(Document.id)).filter(Document.repo_id == repo.id).scalar() or 0
    return KnowledgeBaseResponse(
        id=repo.id,
        name=repo.name,
        full_name=repo.full_name,
        description=repo.description or "",
        provider=repo.provider or "github",
        document_count=int(document_count or 0),
        chunk_count=int(chunk_count),
        character_count=int(character_count or 0),
        config=config_dict(config),
        runtime=runtime_status(config),
        created_at=repo.created_at,
        updated_at=repo.updated_at,
    )


@router.get("/models", response_model=ModelCatalogResponse)
async def get_model_catalog() -> ModelCatalogResponse:
    """GET /api/rag/models：可选 embedding 模型目录。

    入参：无。
    返回：ModelCatalogResponse，供前端配置页下拉选择。
    异常：无。
    副作用：只读（纯静态配置）。
    """
    return ModelCatalogResponse.model_validate(model_catalog())


@router.get("/repositories", response_model=list[KnowledgeBaseResponse])
async def list_rag_repositories(db: Session = Depends(get_db)) -> list[KnowledgeBaseResponse]:
    """GET /api/rag/repositories：列出所有可当知识库用的仓库，附统计与配置。

    入参：无。
    返回：按 updated_at 倒序、排除 provider="knowledge_base" 的列表。
    异常：无。
    副作用：只读（但每个仓库都会 get_or_create_config，可能补建配置行），
        且对每个仓库各做两次聚合查询——仓库多时是 N+1 查询，见性能备注。
    """
    repos = (
        db.query(Repository)
        .filter(Repository.provider != "knowledge_base")
        .order_by(Repository.updated_at.desc())
        .all()
    )
    return [_knowledge_base(db, repo) for repo in repos]


@router.get("/repositories/{repo_id}", response_model=KnowledgeBaseResponse)
async def get_rag_repository(repo_id: uuid.UUID, db: Session = Depends(get_db)) -> KnowledgeBaseResponse:
    """GET /api/rag/repositories/{repo_id}：单个知识库详情（含统计/配置/运行时状态）。

    入参：repo_id。
    返回：KnowledgeBaseResponse。
    异常：仓库不存在或是 knowledge_base 类型 → 404。
    副作用：只读（可能补建默认配置）。
    """
    return _knowledge_base(db, _repo_or_404(db, repo_id))


@router.patch("/repositories/{repo_id}/config", response_model=KnowledgeBaseResponse)
async def update_rag_config(
    repo_id: uuid.UUID,
    payload: KnowledgeBaseConfigUpdate,
    db: Session = Depends(get_db),
) -> KnowledgeBaseResponse:
    """PATCH /api/rag/repositories/{repo_id}/config：局部更新知识库配置。

    入参：payload 只带要改的字段（PATCH 语义）。
    返回：更新后的 KnowledgeBaseResponse。
    异常：
        重叠 ≥ 切片长度 → 400；
        embedding 三元组（provider/model/dimensions）不被支持或与现有向量不兼容 → 409
        （用 409 是因为这属于「与当前资源状态冲突」，不是请求格式错误）。
    副作用：写 config 行并提交。
    注意：这里只改配置，**不会**自动重建已有向量；要让历史数据用新模型，得再调 /reindex。
    """
    repo = _repo_or_404(db, repo_id)
    # commit=False：先攒着，等所有校验都通过再统一提交，避免校验失败留下半套配置。
    config = get_or_create_config(db, repo_id, commit=False)
    # exclude_unset 是 PATCH 的关键：只取用户显式传了的字段，
    # 未传的字段保持原值（否则会被默认值覆盖成空）。
    changes = payload.model_dump(exclude_unset=True)
    # 切片长度/重叠可能只改了一个，另一个要用「即将生效的值」参与校验，所以先算出组合后的结果。
    next_size = int(changes.get("chunk_size", config.chunk_size))
    next_overlap = int(changes.get("chunk_overlap", config.chunk_overlap))
    _validate_chunking(next_size, next_overlap)
    try:
        validate_embedding_contract(
            str(changes.get("embedding_provider", config.embedding_provider)),
            str(changes.get("embedding_model", config.embedding_model)),
            int(changes.get("embedding_dimensions", config.embedding_dimensions)),
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    for key, value in changes.items():
        setattr(config, key, value)
    db.commit()
    db.refresh(repo)
    return _knowledge_base(db, repo)


@router.get("/repositories/{repo_id}/documents", response_model=list[KnowledgeDocumentResponse])
async def list_documents(repo_id: uuid.UUID, db: Session = Depends(get_db)) -> list[KnowledgeDocumentResponse]:
    """GET /api/rag/repositories/{repo_id}/documents：文档列表（含处理状态）。

    入参：repo_id。
    返回：按上传时间倒序的文档列表，status 可能是 queued/processing/completed/failed。
    异常：仓库不存在 → 404。
    副作用：只读。前端靠轮询这个接口来展示异步处理进度。
    """
    _repo_or_404(db, repo_id)
    items = (
        db.query(KnowledgeSourceDocument)
        .filter(KnowledgeSourceDocument.repo_id == repo_id)
        .order_by(KnowledgeSourceDocument.created_at.desc())
        .all()
    )
    return [_document_response(item) for item in items]


@router.post("/repositories/{repo_id}/documents", response_model=KnowledgeDocumentResponse, status_code=202)
async def upload_document(
    repo_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> KnowledgeDocumentResponse:
    """POST /api/rag/repositories/{repo_id}/documents：上传文档（异步处理）。

    入参：file 为 multipart 文件字段。
    返回：202 + 文档记录（status=queued），真正的解析/切片/向量化在后台做。
    异常：
        空文件 / 超 12MB → 400；
        扩展名不支持 → 415；
        （内容重复不算异常，直接返回已有文档。）
    副作用：
        落盘原文件、写 KnowledgeSourceDocument（status=queued），
        然后通过 BackgroundTasks 异步执行 process_knowledge_document——
        这个后台任务会自己开新 session 更新状态，所以不会用到本次请求的 db。
    """
    _repo_or_404(db, repo_id)
    # 多读 1 字节即可判断是否超限，不必把超大文件整个读进内存。
    data = await file.read(MAX_DOCUMENT_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="上传文件为空")
    if len(data) > MAX_DOCUMENT_BYTES:
        raise HTTPException(status_code=400, detail="文件不能超过 12 MB")
    filename = _safe_filename(file.filename or "upload.txt")
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_KNOWLEDGE_SUFFIXES:
        raise HTTPException(status_code=415, detail=f"暂不支持 {suffix or '无扩展名'} 文件")
    checksum = hashlib.sha256(data).hexdigest()
    # 去重时排除 failed：之前处理失败的文档不该阻止用户重传同一份文件修好它。
    duplicate = (
        db.query(KnowledgeSourceDocument)
        .filter(
            KnowledgeSourceDocument.repo_id == repo_id,
            KnowledgeSourceDocument.checksum == checksum,
            KnowledgeSourceDocument.status != "failed",
        )
        .order_by(KnowledgeSourceDocument.created_at.desc())
        .first()
    )
    if duplicate is not None:
        return _document_response(duplicate)
    document = KnowledgeSourceDocument(
        repo_id=repo_id,
        name=filename,
        content_type=file.content_type,
        checksum=checksum,
        file_size=len(data),
        # stored_path 是 NOT NULL 列，但此时还不知道最终文件名（要用文档 ID 命名），
        # 先用 "pending" 占位，落盘成功后立刻回填。
        stored_path="pending",
        status="queued",
    )
    db.add(document)
    # flush 而非 commit：只为拿到数据库生成的 document.id 用于拼文件名。
    db.flush()
    stored_path = _upload_root(repo_id) / f"{document.id}-{filename}"
    try:
        # 写盘放线程池：大文件同步写会阻塞事件循环。
        await asyncio.to_thread(stored_path.write_bytes, data)
        document.stored_path = str(stored_path)
        db.commit()
    except Exception:
        # 与 knowledge.py 上传一致：回滚 DB + 删除刚落盘的文件，避免产生孤儿文件。
        db.rollback()
        try:
            await asyncio.to_thread(stored_path.unlink, missing_ok=True)
        except OSError:
            pass
        raise
    db.refresh(document)
    background_tasks.add_task(process_knowledge_document, document.id)
    return _document_response(document)


@router.get("/repositories/{repo_id}/documents/{document_id}/chunks", response_model=list[KnowledgeChunkResponse])
async def list_document_chunks(
    repo_id: uuid.UUID,
    document_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> list[KnowledgeChunkResponse]:
    """GET /api/rag/repositories/{repo_id}/documents/{document_id}/chunks：查看文档切片。

    入参：repo_id、document_id。
    返回：按创建时间正序的切片列表，position 从 1 开始。
    异常：仓库不存在 → 404；文档不存在时返回空列表（不报错）。
    副作用：只读。用于排查「切得好不好」。
    """
    _repo_or_404(db, repo_id)
    rows = (
        db.query(Document)
        .filter(Document.repo_id == repo_id, Document.source_id == document_id)
        .order_by(Document.created_at.asc())
        .all()
    )
    return [
        KnowledgeChunkResponse(
            id=row.id,
            # meta 里没有 chunk_index 时（老数据）退回 enumerate 的顺序号；
            # +1 是为了给用户看 1-based 的序号。
            position=int((row.meta or {}).get("chunk_index", index)) + 1,
            title=row.title,
            content=row.content,
            character_count=len(row.content),
        )
        for index, row in enumerate(rows)
    ]


@router.post("/repositories/{repo_id}/documents/{document_id}/retry", response_model=KnowledgeDocumentResponse)
async def retry_document(
    repo_id: uuid.UUID,
    document_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> KnowledgeDocumentResponse:
    """POST /api/rag/repositories/{repo_id}/documents/{document_id}/retry：重试失败的文档。

    入参：repo_id、document_id。
    返回：状态已重置为 queued 的文档。
    异常：仓库不存在 → 404；文档不存在或不属于该仓库 → 404。
    副作用：清空 error_message、状态回 queued，并重新排后台处理任务。
        不校验当前状态——已完成/处理中的文档也能「重试」，
        等于强制重跑一次解析与向量化（旧的切片会被处理逻辑覆盖）。
    """
    _repo_or_404(db, repo_id)
    document = db.get(KnowledgeSourceDocument, document_id)
    # 同时校验 repo_id：防止拿 A 库的 document_id 去操作 B 库的文档。
    if document is None or document.repo_id != repo_id:
        raise HTTPException(status_code=404, detail="未找到文档")
    document.status = "queued"
    document.error_message = None
    db.commit()
    db.refresh(document)
    background_tasks.add_task(process_knowledge_document, document.id)
    return _document_response(document)


@router.delete("/repositories/{repo_id}/documents/{document_id}")
async def delete_document(repo_id: uuid.UUID, document_id: uuid.UUID, db: Session = Depends(get_db)) -> dict[str, Any]:
    """DELETE /api/rag/repositories/{repo_id}/documents/{document_id}：删除文档。

    入参：repo_id、document_id。
    返回：{"id", "deleted": true}。
    异常：仓库不存在 → 404；文档不存在或跨仓库 → 404。
    副作用：
        删除该文档的全部切片（documents 表）、删除磁盘原文件、删除文档记录，最后统一提交。
        注意这里是物理删除（不是软删），删完无法恢复。
    """
    _repo_or_404(db, repo_id)
    source = db.get(KnowledgeSourceDocument, document_id)
    if source is None or source.repo_id != repo_id:
        raise HTTPException(status_code=404, detail="未找到文档")
    # commit=False：切片删除和文档记录删除要在一个事务里，
    # 否则中途失败会出现「文档还在但切片没了」的半残状态。
    delete_document_group(db, repo_id, document_id, source_type="knowledge_file", commit=False)
    # 文件在提交前删：这里没有回滚保护（与上传逻辑不同），
    # 若后续 commit 失败，会留下「记录还在、文件已丢」的情况——属于已知取舍，换取逻辑简单。
    path = Path(source.stored_path)
    if path.exists() and path.is_file():
        path.unlink()
    db.delete(source)
    db.commit()
    return {"id": str(document_id), "deleted": True}


@router.post("/repositories/{repo_id}/retrieval-tests", response_model=RetrievalTestResponse)
async def run_retrieval_test(
    repo_id: uuid.UUID,
    payload: RetrievalTestRequest,
    db: Session = Depends(get_db),
) -> RetrievalTestResponse:
    """POST /api/rag/repositories/{repo_id}/retrieval-tests：跑一次检索测试。

    入参：payload 含 query、top_k、检索方式（向量/关键词/混合）、是否 rerank、
        是否启用分数阈值及阈值、元数据过滤。
    返回：RetrievalTestResponse，含命中结果、耗时、本次实际使用的检索配置（trace）。
    异常：仓库不存在 → 404；参数非法 → 422。
    副作用：
        会把这次测试**持久化**（record_retrieval_test），用于对比不同配置的召回效果；
        检索本身也会调 embedding。
    用途：给调参用的实验台——改完阈值/重排后跑几条典型 query，看结果是否变好。
    """
    _repo_or_404(db, repo_id)
    items, trace, duration_ms = await retrieve_knowledge(
        db,
        repo_id,
        payload.query.strip(),
        top_k=payload.top_k,
        retrieval_method=payload.retrieval_method,
        rerank_enabled=payload.rerank_enabled,
        score_threshold_enabled=payload.score_threshold_enabled,
        score_threshold=payload.score_threshold,
        metadata_filters=payload.metadata_filters,
    )
    results = build_sources(items)
    run = record_retrieval_test(db, repo_id, payload.query.strip(), trace, results, duration_ms)
    return RetrievalTestResponse(
        id=run.id,
        query=run.query,
        results=results,
        result_count=len(results),
        took_ms=duration_ms,
        retrieval_config=trace,
        created_at=run.created_at,
    )


@router.get("/repositories/{repo_id}/retrieval-tests", response_model=list[RetrievalTestHistoryResponse])
async def list_retrieval_tests(
    repo_id: uuid.UUID,
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> list[RetrievalTestHistoryResponse]:
    """GET /api/rag/repositories/{repo_id}/retrieval-tests：检索测试历史。

    入参：limit 1~100，默认 20。
    返回：按时间倒序的历史记录（只带结果条数，不带完整结果，减小响应体）。
    异常：仓库不存在 → 404。
    副作用：只读。
    """
    _repo_or_404(db, repo_id)
    runs = (
        db.query(RetrievalTestRun)
        .filter(RetrievalTestRun.repo_id == repo_id)
        .order_by(RetrievalTestRun.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        RetrievalTestHistoryResponse(
            id=run.id,
            query=run.query,
            # results_json 可能为 NULL，兜成空列表再取长度。
            result_count=len(run.results_json or []),
            took_ms=run.duration_ms,
            created_at=run.created_at,
        )
        for run in runs
    ]


@router.get("/{repo_id}/status")
async def get_rag_status(repo_id: uuid.UUID, db: Session = Depends(get_db)) -> dict[str, Any]:
    """GET /api/rag/{repo_id}/status：RAG 运行时状态。

    入参：repo_id。
    返回：dict，{...rag_status 的字段, config, runtime}——
        含文档/切片数量、向量库连通性、当前 embedding 配置等，供前端健康面板展示。
    异常：仓库不存在 → 404。
    副作用：只读（可能补建默认配置）。
    """
    _repo_or_404(db, repo_id)
    config = get_or_create_config(db, repo_id)
    return {**rag_status(db, repo_id), "config": config_dict(config), "runtime": runtime_status(config)}


@router.post("/{repo_id}/search", response_model=RAGSearchResponse)
async def search_rag(repo_id: uuid.UUID, payload: RAGQueryRequest, db: Session = Depends(get_db)) -> RAGSearchResponse:
    """POST /api/rag/{repo_id}/search：只检索不生成（返回相关片段）。

    入参：payload.question 查询语句、top_k、source_types、metadata_filters。
    返回：RAGSearchResponse（命中片段 + 耗时 + 实际检索配置）。
    异常：仓库不存在 → 404。
    副作用：只读；会调 embedding + 向量检索。
    与 /ask 的区别：这个不调大模型生成答案，前端可用来做「先看依据」的模式。
    """
    _repo_or_404(db, repo_id)
    items, trace, duration_ms = await retrieve_knowledge(
        db,
        repo_id,
        payload.question.strip(),
        top_k=payload.top_k,
        source_types=payload.source_types,
        metadata_filters=payload.metadata_filters,
    )
    results = build_sources(items)
    return RAGSearchResponse(
        query=payload.question.strip(),
        results=results,
        result_count=len(results),
        took_ms=duration_ms,
        retrieval_config=trace,
    )


@router.post("/{repo_id}/ask", response_model=RAGAnswerResponse)
async def ask_rag(repo_id: uuid.UUID, payload: RAGQueryRequest, db: Session = Depends(get_db)) -> RAGAnswerResponse:
    """POST /api/rag/{repo_id}/ask：检索 + 生成答案（经典 RAG 问答）。

    入参：payload.question（**不 strip**，原样交给 answer_question，便于保留换行等格式）、
        top_k、source_types、metadata_filters。
    返回：RAGAnswerResponse（答案 + 引用来源）。
    异常：仓库不存在 → 404；大模型失败向上抛。
    副作用：调用 LLM，耗时较 search 长；除检索外不写库。
    """
    _repo_or_404(db, repo_id)
    result = await answer_question(
        db,
        repo_id,
        payload.question,
        payload.top_k,
        payload.source_types,
        payload.metadata_filters,
    )
    return RAGAnswerResponse.model_validate(result)


@router.post("/{repo_id}/reindex", response_model=RAGReindexResponse)
async def reindex_rag(repo_id: uuid.UUID, db: Session = Depends(get_db)) -> RAGReindexResponse:
    """POST /api/rag/{repo_id}/reindex：按当前配置重建全部向量。

    入参：repo_id。
    返回：RAGReindexResponse（重建的文档数 + 本次使用的 embedding provider/model）。
    异常：仓库不存在 → 404。
    副作用：**重活且同步执行**——会重新切块/重新调用 embedding/重写向量库，
        大库可能耗时很久，请求期间前端需要自行处理超时；中途失败可能留下部分重建的数据。
    什么时候需要：改了 embedding 模型或切片参数之后（配置改动不会自动生效）。
    """
    _repo_or_404(db, repo_id)
    config = get_or_create_config(db, repo_id)
    count = reindex_documents(db, repo_id)
    return RAGReindexResponse(
        repo_id=repo_id,
        indexed_documents=count,
        embedding_provider=config.embedding_provider,
        embedding_model=config.embedding_model,
    )
