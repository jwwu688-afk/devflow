"""GitHub 数据源的索引构建：把 issue / PR / CI 运行记录转成 RAG 文档。

位置：位于切块之后、向量化之前。与 document_processing 的区别在于数据来源——
那边处理用户上传的文件（走完整的切块管道），这里处理的是**结构化记录**
（一条 issue 就是一个文档，不需要再切块）。

三类数据的处理策略各不相同，原因都是「这条记录的自然单位是什么」：
- issue / PR：一条记录天然就是一个语义单位，标题 + 正文 + 评论拼成一个文档；
- workflow_run：日志可能极长，所以按固定字符数切成多片，只有失败的运行才入库
  （成功的 CI 没有排障价值，入库只会稀释检索精度）。

入库方式统一走 replace_documents：按 source_type 整体替换，
保证「重新同步一次」的结果是幂等的，不会累积重复文档。

本文件里最重要的一段是 sanitize_ci_log——CI 日志是最容易泄漏密钥的地方
（构建脚本常常会 echo 环境变量），所以必须在入库前把各类凭证刷掉。
"""

import re

from sqlalchemy.orm import Session, selectinload

from app.db.models import Issue, KnowledgeBaseConfig, PullRequest, Repository, WorkflowRun
from app.services.rag.embeddings import embed_knowledge_texts
from app.services.rag.vector_store import DocumentPayload, replace_documents

# 参与同步的三种来源类型。与 source_policy 里的 RAG_SOURCE_TYPES 取交集，
# 这里只列出本同步器负责的那部分（project_doc / knowledge_file 由别的入口写入）
SYNCED_REPO_SOURCE_TYPES = ["issue", "pull_request", "workflow_run"]
# CI 日志单片 8000 字符：比文档切块的 1600 大很多，因为日志是连续文本、
# 上下文依赖强，切太碎反而丢失堆栈的连续性
CI_LOG_CHUNK_CHARS = 8000
# 最多 6 片（约 48000 字符）：再多的日志对排障帮助有限，
# 而向量的存储与调用成本是线性增长的
CI_LOG_MAX_CHUNKS = 6


def sanitize_ci_log(text: str) -> str:
    """清洗 CI 日志中的凭证。**顺序敏感，不要随意调整。**

    逐条规则及其针对的目标：
    1. PEM 私钥整块替换（DOTALL 让 . 跨行匹配，因为私钥是多行的）；
    2. GitHub token（ghp_/gho_/ghu_/ghs_/ghr_ 前缀 + 20 位以上）；
    3. AWS Access Key（AKIA + 16 位大写字母数字）；
    4. JWT（三段点分的 base64url）；
    5. `Authorization: Bearer xxx` 头；
    6. GitHub Actions 的 `::add-mask::` 指令后跟的值；
    7. 通用兜底：键名里含 token/secret/password/api_key 等的一律替换值。

    为什么 7 放最后：它是启发式的、最激进的一条，会把形如
    `SOME_TOKEN=abc` 的内容全部抹掉。如果放在前面，可能会把某些更具体的
    模式（比如 JWT 本体）提前吃掉，导致替换后的标签不准确（日志里看不出
    究竟是哪类凭证泄漏，排查时反而少了一条线索）。

    所有替换都是「保留键名、只抹掉值」，这样日志里还能看出「这里有个
    API_KEY 参数」，排障信息不会完全丢失。

    注意：这是**尽力而为**的脱敏，不可能覆盖所有凭证形态。真正的防护
    应该是在 CI 侧用 GitHub Secrets，本函数只是入向量库前的最后一道防线。
    """
    sanitized = re.sub(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
        "[REDACTED_PRIVATE_KEY]",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    # gh[pousr]_ 覆盖 personal/oauth/user/server/refresh 五类 token
    sanitized = re.sub(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b", "[REDACTED_GITHUB_TOKEN]", sanitized)
    sanitized = re.sub(r"\bAKIA[A-Z0-9]{16}\b", "[REDACTED_AWS_ACCESS_KEY]", sanitized)
    sanitized = re.sub(
        r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b",
        "[REDACTED_JWT]",
        sanitized,
    )
    # \1 保留 "Authorization: Bearer" 前缀，只替换后面的凭证本体
    sanitized = re.sub(
        r"(?i)\b(authorization\s*:\s*bearer)\s+[^\s,;]+",
        r"\1 [REDACTED]",
        sanitized,
    )
    # ::add-mask:: 是 Actions 自己标记敏感值的语法，后面的值必须刷掉
    sanitized = re.sub(r"(?i)(::add-mask::)[^\r\n]+", r"\1[REDACTED]", sanitized)
    # 通用键值兜底：键名可含前后缀（如 DB_PASSWORD、my_api_key），
    # 值是「非空白、非逗号、非分号」的连续串
    return re.sub(
        r"(?i)\b([a-z0-9_]*(?:api[_-]?key|access[_-]?key|access[_-]?token|auth[_-]?token|"
        r"token|secret|password|passwd|private[_-]?key|client[_-]?secret|account[_-]?key)[a-z0-9_]*)"
        r"(\s*[:=]\s*)([^\s,;]+)",
        r"\1\2[REDACTED]",
        sanitized,
    )


def _ci_log_chunks(text: str) -> tuple[list[str], bool]:
    """把 CI 日志先脱敏再定长切片，并返回「是否发生了截断」。

    顺序很重要：**先脱敏再截断**。反过来的话，一个被切断的 token
    可能只剩半截，正则就匹配不到了，等于把密钥漏进向量库。

    先按 CI_LOG_CHUNK_CHARS * CI_LOG_MAX_CHUNKS 截总长，再切片，
    这样能保证切片数量上限，不会先生成上百个片再丢弃。

    返回的 bool 会写进 metadata.logs_truncated：检索结果里能提示用户
    「这段日志不完整，请去原始链接看全文」，避免基于残缺日志下错结论。

    过滤掉纯空白片：日志里常有大量连续空行，切出来的空片没有向量化价值。
    """
    sanitized = sanitize_ci_log(text.strip())
    max_chars = CI_LOG_CHUNK_CHARS * CI_LOG_MAX_CHUNKS
    bounded = sanitized[:max_chars]
    chunks = [
        bounded[start : start + CI_LOG_CHUNK_CHARS]
        for start in range(0, len(bounded), CI_LOG_CHUNK_CHARS)
        if bounded[start : start + CI_LOG_CHUNK_CHARS].strip()
    ]
    return chunks, len(sanitized) > max_chars


def _iso(value) -> str | None:
    """datetime → ISO 字符串；None 保持 None。

    统一转成字符串是为了能进 JSON 序列化的 metadata（Milvus 的
    metadata_json 字段），同时也让 rerank 的 _freshness_bonus 能解析时间。
    """
    return value.isoformat() if value else None


def _payload(source_type: str, source_id, title: str, content: str, metadata: dict) -> DocumentPayload | None:
    """构造一条待入库文档；内容为空时返回 None（调用方据此跳过）。

    用 None 而不是抛异常：很多记录确实没有正文（只有标题的 issue、
    body 为空的 PR），跳过是正常业务，不是错误。

    title 截到 500 字符，与 Milvus schema 的字段上限对齐。
    """
    if not content.strip():
        return None
    return DocumentPayload(
        source_type=source_type,
        source_id=source_id,
        # 注意这里只做字符截断，真正的字节级裁剪在 vector_store 写入时进行
        title=title[:500],
        content=content,
        metadata=metadata,
    )


def index_repository_documents(
    db: Session,
    repo_id,
    *,
    generate_embeddings: bool = True,
    sync_vector_store: bool = True,
) -> int:
    """重建某个仓库的三类 RAG 文档，返回写入的文档条数。

    执行顺序：先收集全部 payload（纯内存操作，不碰外部服务），
    再一次性批量向量化，最后一次性替换入库。

    这个「先全部构造、再统一提交」的顺序有两个好处：
    - 向量化可以一次批量调用，比分批快得多；
    - replace_documents 会先把该类型的旧数据全删掉，如果中途失败，
      至少不会留下「删了一半」的状态（业务库侧靠调用方的事务兜住）。

    注意 PR 查询用了 selectinload 预加载 files 与 review_comments：
    否则在循环里访问 pr.files 会触发 N+1 查询（每个 PR 两条额外 SQL），
    仓库有几百个 PR 时性能会塌方式下降。

    workflow_run 只索引 conclusion == "failure" 的运行：成功日志没有排障价值，
    入库只会占空间并稀释检索质量。

    config 为 None（该仓库还没建知识库配置）时，vectors 保持 None，
    交给 replace_documents 走「内部生成」或纯业务库写入的路径。
    """
    repo = db.get(Repository, repo_id)
    if repo is None:
        return 0

    payloads: list[DocumentPayload] = []

    # —— 1. Issue ——
    issues = db.query(Issue).filter(Issue.repo_id == repo_id).all()
    for issue in issues:
        # 标题 + 正文拼成一条文档。标签/状态/负责人放 metadata，
        # 检索时可用作过滤或展示，不参与向量化
        content = "\n".join([issue.title, issue.body or ""])
        item = _payload(
            "issue",
            issue.id,
            issue.title,
            content,
            {
                "number": issue.number,
                "labels": issue.labels or [],
                "state": issue.state,
                "author": issue.author,
                "assignees": issue.assignees or [],
                "created_at": _iso(issue.created_at),
                "updated_at": _iso(issue.updated_at),
                "closed_at": _iso(issue.closed_at),
            },
        )
        if item:
            payloads.append(item)

    # —— 2. Pull Request ——
    # selectinload 预加载关联表，避免 N+1 查询
    prs = (
        db.query(PullRequest)
        .options(selectinload(PullRequest.files), selectinload(PullRequest.review_comments))
        .filter(PullRequest.repo_id == repo_id)
        .all()
    )
    for pr in prs:
        # 行内评论按 "路径:行号 作者: 内容" 拼接，保留定位信息，
        # 这样检索到某条评论时还能知道它在哪个文件的哪一行
        comments_text = "\n".join(
            f"{comment.path or ''}:{comment.line or ''} {comment.author or ''}: {comment.body or ''}"
            for comment in pr.review_comments
        )
        content = "\n".join(
            [
                pr.title,
                pr.body or "",
                "review comments:",
                comments_text,
            ]
        )
        item = _payload(
            "pull_request",
            pr.id,
            pr.title,
            content,
            {
                "number": pr.number,
                "state": pr.state,
                "author": pr.author,
                "files": [file.filename for file in pr.files],
                "base_branch": pr.base_branch,
                "head_branch": pr.head_branch,
                "created_at": _iso(pr.created_at),
                "updated_at": _iso(pr.updated_at),
                "merged_at": _iso(pr.merged_at),
            },
        )
        if item:
            payloads.append(item)

    # —— 3. CI 运行记录 ——
    runs = db.query(WorkflowRun).filter(WorkflowRun.repo_id == repo_id).all()
    for run in runs:
        # 只索引失败且有日志的运行；跳过时不做任何记录（静默，
        # 因为这是常规情况，不是异常）
        if run.conclusion != "failure" or not (run.logs_text or "").strip():
            continue
        log_chunks, logs_truncated = _ci_log_chunks(run.logs_text or "")
        for chunk_index, log_chunk in enumerate(log_chunks):
            # 每个分片都以 run.name 开头：日志正文里往往看不出是哪个流水线，
            # 把名字带上才能让向量「知道」这段日志属于哪条 job
            content = "\n".join([run.name, log_chunk])
            item = _payload(
                "workflow_run",
                run.id,
                run.name if len(log_chunks) == 1 else f"{run.name}#{chunk_index + 1}",
                content,
                {
                    "status": run.status,
                    "conclusion": run.conclusion,
                    "html_url": run.html_url,
                    # jobs 只保留 name，控制 metadata 体积
                    "jobs": [job.get("name") for job in (run.jobs or []) if isinstance(job, dict)],
                    "created_at": _iso(run.created_at),
                    "updated_at": _iso(run.updated_at),
                    "chunk_index": chunk_index,
                    "total_chunks": len(log_chunks),
                    # 让下游知道证据是不完整的
                    "logs_truncated": logs_truncated,
                    # 明确标注已脱敏，前端可据此提示用户
                    "secrets_redacted": True,
                },
            )
            if item:
                payloads.append(item)

    vectors = None
    embedding_info = None
    # 只有需要生成向量时才去查配置；不需要则跳过这次查询
    config = (
        db.query(KnowledgeBaseConfig).filter(KnowledgeBaseConfig.repo_id == repo_id).one_or_none()
        if generate_embeddings
        else None
    )
    if config is not None and payloads:
        # 与文档入库管道一致：向量化的是 "标题\n正文"，
        # 而 content 字段保存纯正文
        vectors = embed_knowledge_texts(
            [f"{payload.title}\n{payload.content}" for payload in payloads],
            provider=config.embedding_provider,
            model=config.embedding_model,
            dimensions=config.embedding_dimensions,
        )
        embedding_info = {
            "embedding_provider": config.embedding_provider,
            "embedding_model": config.embedding_model,
            "embedding_dimensions": config.embedding_dimensions,
        }

    # commit=False：事务边界交给调用方（同步任务），便于与上游的状态更新一起提交
    docs = replace_documents(
        db,
        repo_id,
        SYNCED_REPO_SOURCE_TYPES,
        payloads,
        commit=False,
        vectors=vectors,
        embedding_info=embedding_info,
        generate_embeddings=generate_embeddings,
        sync_vector_store=sync_vector_store,
    )
    return len(docs)
