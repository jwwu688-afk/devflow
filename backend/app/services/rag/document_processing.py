"""知识文件入库管道：把一次上传走完「抽取 → 切块 → 向量化 → 写库」。

这是 RAG 写入侧的总编排，也是「切块 → 向量化」两个环节真正串起来的地方。
整条链路在 process_knowledge_document 里按状态机推进：

    parsing   → 读盘 + 抽取纯文本
    splitting → 切块（用知识库配置里的 chunk_size / chunk_overlap）
    embedding → 批量向量化
    completed → 落库成功
    failed    → 任一步出错，记录错误信息

**状态是写进数据库的**（不只在内存里），因为这是一个可能跑几十秒的长任务：
前端靠轮询 status 给用户进度条，任务进程崩溃时也能看到停在哪个阶段。

两个关键的一致性设计：
1. delete_document_group(commit=False) + add_documents(commit=False) + 最后一次
   commit：先删旧向量再写新向量，两者在同一个事务里。这样重复上传同一文件
   不会留下两份向量，中途失败也能整体回滚，不会出现「删了旧的但新的没写进去」。
2. 向量与原文分开存：向量进 Milvus，content 也作为 payload 一起存（见
   DocumentPayload），避免检索后还要回查原文件。
"""

import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.db.models import KnowledgeSourceDocument
from app.db.session import SessionLocal
from app.services.rag.chunking import chunk_document
from app.services.rag.document_extraction import MAX_DOCUMENT_BYTES, extract_document_text
from app.services.rag.embeddings import embed_knowledge_texts
from app.services.rag.knowledge_base import get_or_create_config
from app.services.rag.vector_store import DocumentPayload, add_documents, delete_document_group

# 允许上传的后缀白名单。注意包含 .log：CI 日志是重要的排障知识来源。
# 白名单而非黑名单——上传是外部输入，默认拒绝比默认接受安全
SUPPORTED_KNOWLEDGE_SUFFIXES = {".txt", ".md", ".markdown", ".mdx", ".pdf", ".docx", ".json", ".csv", ".log"}
# 每批 64 条向量化：兼顾 API 单次请求的 payload 上限与吞吐。
# 太小则请求数暴增、延迟变高；太大则单次请求超时风险上升、失败重试代价更大
EMBEDDING_BATCH_SIZE = 64


def extract_text(filename: str, data: bytes) -> str:
    """抽取纯文本的薄封装。

    单独包一层是为了让调用方（含测试）不必直接依赖 document_extraction 模块，
    以后换抽取实现时改这里即可。
    """
    return extract_document_text(filename, data)


def _embed_in_batches(
    texts: list[str],
    *,
    provider: str,
    model: str,
    dimensions: int,
) -> list[list[float]]:
    """把长文本列表分批向量化，再按原顺序拼回一个列表。

    extend 而不是 append：embed_knowledge_texts 返回的是向量列表，
    要保持「输入第 i 条 ↔ 输出第 i 个向量」的对应关系，顺序绝不能乱。
    分批的副作用是：某一批失败会抛异常，整个文档入库失败（而不是部分成功），
    这是有意的——半个文档入库比不入库更危险。
    """
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        vectors.extend(
            embed_knowledge_texts(
                texts[start : start + EMBEDDING_BATCH_SIZE],
                provider=provider,
                model=model,
                dimensions=dimensions,
            )
        )
    return vectors


def _set_status(document_id: uuid.UUID, status: str, *, error: str | None = None) -> None:
    """独立开一个短事务更新文档状态。

    为什么单独开会话：状态更新要尽可能独立于主流程的事务。
    主流程最后可能回滚，但「已经进入 parsing 阶段」这个事实应该被记录下来。
    用 finally 保证无论如何都关会话，避免连接泄漏。
    """
    db = SessionLocal()
    try:
        document = db.get(KnowledgeSourceDocument, document_id)
        if document is None:
            return
        document.status = status
        document.error_message = error
        document.updated_at = datetime.now(timezone.utc)
        db.commit()
    finally:
        db.close()


def process_knowledge_document(document_id: uuid.UUID) -> None:
    """一次完整的知识文件入库（由后台任务调用，同步阻塞）。

    入参只有 document_id：所有业务数据（路径、仓库、校验和）都从数据库读，
    因为这是跨进程的任务入口，不能依赖调用方传复杂对象。

    几个值得注意的实现选择：
    - 读文件时只读 MAX_DOCUMENT_BYTES + 1 字节，用「多读 1 字节」来判断是否超限。
      这样即使文件有几十 GB 也不会被一次性读进内存。
    - 向量化的输入是 "标题\\n正文"，而不是纯正文：标题（文件名 + 章节名）
      在语义上很有信息量，能显著改善「按文件名找内容」的召回。
    - metadata 里把 chunk 的所有非 content 字段平铺进去（chunk_id、parent_id、
      start_line 等），查询时就能直接展示来源位置，不用回查。
    - metadata 过滤掉 None 值：Milvus 的标量字段对 null 支持不好，空值会导致
      表达式过滤行为不确定。
    - chunk_strategy 记录切块策略版本，为将来换算法留出区分新旧数据的依据。
    """
    _set_status(document_id, "parsing")
    db = SessionLocal()
    try:
        source = db.get(KnowledgeSourceDocument, document_id)
        if source is None:
            return
        # 多读 1 字节，用长度判断是否超限，避免大文件占满内存
        with Path(source.stored_path).open("rb") as handle:
            data = handle.read(MAX_DOCUMENT_BYTES + 1)
        if len(data) > MAX_DOCUMENT_BYTES:
            raise ValueError(f"knowledge file exceeds the {MAX_DOCUMENT_BYTES} byte limit")
        text = extract_text(source.name, data)
        # 抽不出任何文本（如纯图片 PDF）：没有可检索内容，明确失败而不是静默成功
        if not text.strip():
            raise ValueError("文档中没有可索引的文本")

        source.status = "splitting"
        source.character_count = len(text)
        db.commit()
        # 取该仓库的切块配置；没有就用默认值创建一份
        config = get_or_create_config(db, source.repo_id)
        chunks = chunk_document(
            text,
            source.name,
            max_chars=config.chunk_size,
            overlap=config.chunk_overlap,
        )
        if not chunks:
            raise ValueError("文档切分后没有有效片段")

        source.status = "embedding"
        source.chunk_count = len(chunks)
        db.commit()
        titles = []
        for index, chunk in enumerate(chunks):
            # 标题优先级：章节标题 > 页码 > 兜底序号，保证每条都有可读标题
            label = chunk.get("section_title")
            if not label and chunk.get("page"):
                label = f"Page {chunk['page']}"
            label = str(label or f"Chunk {index + 1}")
            # child_index 为 0 时不加后缀（第一个子块没必要标 "(1)"）
            child_suffix = f" ({int(chunk.get('child_index') or 0) + 1})" if chunk.get("child_index") else ""
            titles.append(f"{source.name} · {label}{child_suffix}")
        contents = [str(chunk["content"]) for chunk in chunks]
        # 注意这里向量化的是「标题 + 正文」，而 content 字段存的是纯正文。
        # 差异是刻意的：检索时标题能提升召回，展示时不需要重复标题
        vectors = _embed_in_batches(
            [f"{title}\n{content}" for title, content in zip(titles, contents)],
            provider=config.embedding_provider,
            model=config.embedding_model,
            dimensions=config.embedding_dimensions,
        )
        payloads = [
            DocumentPayload(
                source_type="knowledge_file",
                source_id=source.id,
                title=title,
                content=content,
                metadata={
                    # 平铺 chunk 的其他字段（chunk_id / parent_id / start_line / page…）；
                    # 过滤掉 content（已单独存）与 None（Milvus 对 null 不友好）
                    **{key: value for key, value in chunk.items() if key != "content" and value is not None},
                    "display_title": source.name,
                    "filename": source.name,
                    "knowledge_document_id": str(source.id),
                    "checksum": source.checksum,
                    "file_size": source.file_size,
                    "content_type": source.content_type,
                    "chunk_index": index,
                    "chunk_chars": len(content),
                    "total_chunks": len(chunks),
                    "chunk_strategy": "structure_aware_v1",
                },
            )
            for index, (title, content, chunk) in enumerate(zip(titles, contents, chunks))
        ]
        # 先删后加、且都 commit=False，与下面最后一次 db.commit() 组成同一事务，
        # 保证「重传同一文件」是覆盖而不是追加
        delete_document_group(db, source.repo_id, source.id, source_type="knowledge_file", commit=False)
        add_documents(
            db,
            source.repo_id,
            payloads,
            commit=False,
            vectors=vectors,
            embedding_info={
                "embedding_provider": config.embedding_provider,
                "embedding_model": config.embedding_model,
                "embedding_dimensions": config.embedding_dimensions,
            },
        )
        source.status = "completed"
        source.error_message = None
        source.completed_at = datetime.now(timezone.utc)
        source.updated_at = source.completed_at
        db.commit()
    except Exception as exc:
        # 先回滚（丢弃半成品向量），再用新查询把状态改成 failed。
        # 必须重新 get：rollback 之后原对象的属性可能已过期
        db.rollback()
        source = db.get(KnowledgeSourceDocument, document_id)
        if source is not None:
            source.status = "failed"
            # 错误信息截断到 1000 字符：异常文本可能很长（如整段 HTML 响应），
            # 直接入库会撑爆字段
            source.error_message = str(exc)[:1000]
            source.updated_at = datetime.now(timezone.utc)
            db.commit()
    finally:
        db.close()
