"""CI（GitHub Actions）相关接口，挂在 /api 前缀下。

核心能力是「失败 CI 根因分析」：把 workflow 的失败 job/step、日志里的首个错误块、
从日志里抽出的文件路径、相关代码与 workflow 配置、最近 PR、项目文档和团队上下文
一起交给 CIDebugAgent，产出失败类型、根因、修复步骤，以及「是否阻塞合并」的判断。

接口：
    GET  /api/repos/{repo_id}/workflow-runs            run 列表
    GET  /api/workflow-runs/{run_id}                   run 详情
    GET  /api/workflow-runs/{run_id}/failed-jobs       只看失败的 job/step（不进模型，快速排查）
    POST /api/workflow-runs/{run_id}/analyze           同步分析
    POST /api/workflow-runs/{run_id}/analyze/stream    SSE 流式分析（前端主用）

一个关键细节：CI 日志里常带 token、内网地址等敏感信息，所以凡是「要进检索或进模型」
的日志文本都先过 sanitize_ci_log 脱敏，而给代码检索用的原文则不做处理（只用于关键词匹配）。
"""

import asyncio
import json
import re
from collections.abc import AsyncIterator
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import or_
from sqlalchemy.orm import Session, selectinload

from app.db.models import AnalysisResult, CodeRelation, CodeSymbol, Conversation, Document, PullRequest, Repository, WorkflowRun
from app.db.session import get_db
from app.schemas.ci import CIAnalysisResponse, WorkflowRunResponse
from app.services.agents.ci_debug_agent import CIDebugAgent
from app.services.chat_memory import ContextAssembler
from app.services.code_search import search_repository_code
from app.services.rag.indexing import sanitize_ci_log
from app.services.rag.retrieval import search_similar_documents
from app.services.worktree_manager import prepare_pr_worktree

router = APIRouter()

# 与 issues.py / pull_requests.py 保持一致的项目文档类型集合（三个 Agent 的口径要统一）。
PROJECT_DOC_TYPES = {"knowledge_file", "memory_note", "project_doc"}
# 让模型输出「公开排查过程」的提示词。这段文字会直接展示给用户，
# 所以明确禁止私有推理链，也禁止编造日志里没有的 job/step/文件。
PUBLIC_REASONING_PROMPT = """
你是 DevFlow AI 的 CI 根因分析 Agent。请用中文流式输出一段“公开排查过程”，帮助用户理解你如何按计划定位失败 CI。
要求：
- 只输出面向用户的可审计分析摘要，不输出隐藏链路思维或私有推理链。
- 按计划说明你检查了失败 job、失败 step、首个错误块、失败类型、相关代码/配置、最近 PR，并如何判断是否阻塞合并。
- 不要编造输入中没有的 job、step、文件、PR 或错误日志。
- 不要输出最终 JSON。
"""


def _encode_sse(event: str, data: dict) -> str:
    """编码一条 SSE 帧；结尾空行是帧分隔符，缺了浏览器不会派发。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _trace_item(kind: str, title: str, content: str, status: str = "done") -> dict:
    """构造前端时间线条目：kind 决定图标，status=running 时前端显示进行中。"""
    return {"kind": kind, "title": title, "content": content, "status": status}


def _document_evidence(items: list[dict], default_source_type: str) -> list[dict]:
    """规整证据结构；缺标题或缺摘要的条目直接丢弃（前端无法展示，也会干扰模型）。"""
    evidence = []
    for item in items:
        snippet = str(item.get("snippet") or "").strip()
        title = str(item.get("title") or "").strip()
        if not snippet or not title:
            continue
        evidence.append(
            {
                "source_type": item.get("source_type") or default_source_type,
                "title": title,
                "snippet": snippet,
                "source_id": str(item.get("source_id") or item.get("id") or ""),
                "score": item.get("score"),
                "metadata": item.get("metadata") or {},
            }
        )
    return evidence


def _dedupe_evidence(items: list[dict], limit: int) -> list[dict]:
    """按 (来源类型, 来源ID, 标题) 去重并截断。

    快照代码与主干代码可能命中同一处，去重可避免同一段代码出现两次（浪费 token、
    还会让模型误判有两处问题）。顺序保持「先到先得」，即快照优先。
    """
    output = []
    seen = set()
    for item in items:
        key = (item.get("source_type"), item.get("source_id"), item.get("title"))
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
        if len(output) >= limit:
            break
    return output


def _fallback_project_documents(db: Session, repo_id: UUID) -> list[dict]:
    """语义检索空结果时的兜底：取最近 4 篇项目文档，保证模型有基本项目背景。"""
    rows = (
        db.query(Document)
        .filter(Document.repo_id == repo_id, Document.source_type.in_(PROJECT_DOC_TYPES))
        .order_by(Document.created_at.desc())
        .limit(4)
        .all()
    )
    return [
        {
            "source_type": doc.source_type,
            "title": doc.title,
            # 概览级截断，全文会撑爆 prompt。
            "snippet": doc.content[:260],
            "source_id": str(doc.source_id),
            # 兜底数据没有相似度分数。
            "score": None,
            "metadata": doc.meta or {},
        }
        for doc in rows
    ]


def _conversation_evidence(db: Session, run: WorkflowRun, conversation_id: UUID | None, query: str) -> list[dict]:
    """把当前会话上下文作为证据（最多 5 条）；会话不属于本仓库时返回空，避免越权与噪音。"""
    if not conversation_id:
        return []
    conversation = db.get(Conversation, conversation_id)
    if conversation is None or conversation.repo_id != run.repo_id:
        return []
    repo = db.get(Repository, run.repo_id)
    assembled = ContextAssembler(db).assemble(repo, conversation, query, limit=8)
    evidence = _document_evidence(assembled.evidence, "conversation")
    if assembled.system_context.strip():
        evidence.append(
            {
                "source_type": "conversation",
                "title": f"会话上下文：{conversation.title}",
                "snippet": assembled.system_context[:320],
                "source_id": str(conversation.id),
                "score": None,
            }
        )
    return evidence[:5]


def _ci_plan() -> list[str]:
    """固定的六步排查计划。

    固定而非让模型自拟：步骤可枚举，SSE 才能把「执行 N/6」的进度准确推给前端，
    用户也能对照检查「AI 到底查没查这一步」。
    """
    return [
        "找出失败 job 和失败 step。",
        "提取首个关键错误块。",
        "判断失败类型：依赖、测试、构建、权限、环境变量或未知。",
        "搜索相关代码、测试文件和配置文件。",
        "关联最近 PR 或变更模块。",
        "输出根因、修复步骤和阻塞判断。",
    ]


def _workflow_payload(run: WorkflowRun) -> dict:
    """把 WorkflowRun 行转成给 Agent 的输入快照。

    jobs 是 JSON 列（列表），为空时兜成 []；logs_text 是完整日志，由 Agent 侧自行截断。
    """
    return {
        "id": str(run.id),
        "name": run.name,
        "status": run.status,
        "conclusion": run.conclusion,
        "jobs": run.jobs or [],
        "logs_text": run.logs_text,
        "html_url": run.html_url,
    }


def _failed_jobs(run: WorkflowRun) -> list[dict]:
    """筛出失败的 job，并在每个 job 上补一个 failed_steps 字段。

    为什么补这一步：GitHub 的原始数据里 step 的结论散落在 job.steps 里，
    模型直接读容易漏；预先算好「哪些 step 失败了」能显著提升定位准确率。
    用 **job 展开保留原有字段，前端/模型仍能拿到 job 名、id 等信息。
    """
    return [
        {
            **job,
            "failed_steps": [step for step in job.get("steps", []) if step.get("conclusion") == "failure"],
        }
        for job in (run.jobs or [])
        if job.get("conclusion") == "failure"
    ]


def _first_error(logs: str | None) -> str | None:
    """从日志里截出「第一个错误块」。

    入参：logs 原始日志文本（可能为 None）。
    返回：一段最多 900 字符的错误上下文；日志里没有任何错误关键词时返回开头 6 行。
    说明：
        取上下文而不是单行——单行「Error: xxx」往往缺上下文，模型只能猜；
        这里向上取 2 行、向下取 5 行，正好覆盖堆栈开头。
        找不到关键词时退回前 6 行（编译类失败通常错误就在开头），
        这样模型永远有东西可看，不会因为关键词不匹配而完全失去线索。
    """
    if not logs or not logs.strip():
        return None
    lines = [line.strip() for line in logs.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        lowered = line.lower()
        # 关键词覆盖了 Python/Node/通用 CLI 的常见报错形态（含 npm ERR! 这种带空格的写法）。
        if any(token in lowered for token in ["error", "failed", "failure", "exception", "traceback", "assertionerror", "npm err", "fatal"]):
            start = max(0, index - 2)
            end = min(len(lines), index + 5)
            return "\n".join(lines[start:end])[:900]
    return "\n".join(lines[:6])[:900]


def _related_files_from_logs(logs: str | None) -> list[str]:
    """用正则从日志里捞出「像文件路径」的字符串，作为代码检索的线索。

    入参：logs 原始日志。
    返回：去重后的文件路径，最多 8 个。
    说明：
        正则限定常见源码/配置后缀，避免把日志里的随机单词当文件名；
        strip 掉两侧的引号/括号/冒号（堆栈里常见 "at foo (bar.ts:12:3)" 这种包裹），
        反斜杠转正斜杠，让 Windows 风格路径也能和索引里的 posix 路径对上。
    """
    if not logs:
        return []
    candidates = re.findall(r"[\w./\\-]+\.(?:py|ts|tsx|js|jsx|json|ya?ml|toml|ini|cfg|md)", logs)
    normalized = []
    for item in candidates:
        cleaned = item.strip("`'\"()[]{}:,;").replace("\\", "/")
        if cleaned and cleaned not in normalized:
            normalized.append(cleaned)
    return normalized[:8]


def _related_pr_for_run(db: Session, run: WorkflowRun) -> PullRequest | None:
    """猜这条 CI run 对应哪个 PR（WorkflowRun 上没有 PR 外键，只能启发式匹配）。

    策略：
        1. 先看最近更新的 8 个 PR，哪个 head_branch 的名字出现在 run 名或日志里；
        2. 都对不上就退回「最近更新的那个 PR」。
    为什么可以「猜错也没关系」：这个结果只用于准备 PR 快照、增强证据，
        猜错最多是证据相关性下降，不会写坏数据；但猜对了能显著提升定位精度。
    """
    candidates = (
        db.query(PullRequest)
        .options(selectinload(PullRequest.files))
        .filter(PullRequest.repo_id == run.repo_id)
        .order_by(PullRequest.updated_at.desc(), PullRequest.created_at.desc())
        .limit(8)
        .all()
    )
    if not candidates:
        return None
    logs = f"{run.name}\n{run.logs_text or ''}".lower()
    for pr in candidates:
        if pr.head_branch and pr.head_branch.lower() in logs:
            return pr
    return candidates[0]


def _snapshot_summary(snapshot: dict) -> dict:
    """裁剪快照结果给前端展示用；比 PR 版多一个 pull_request 字段（关联到的 PR 摘要）。"""
    return {
        "status": snapshot.get("status"),
        "message": snapshot.get("message"),
        "name": snapshot.get("snapshot_name") or "PR 分支快照",
        "branch": snapshot.get("branch"),
        "commit_sha": snapshot.get("commit_sha"),
        "pr_id": snapshot.get("pr_id"),
        "pr_number": snapshot.get("pr_number"),
        "indexed_symbols": snapshot.get("indexed_symbols"),
        "pull_request": snapshot.get("pull_request"),
    }


def _prepare_ci_pr_snapshot(db: Session, run: WorkflowRun, enabled: bool) -> dict:
    """尝试为这条 CI run 关联的 PR 准备分支快照。

    入参：enabled=False 直接跳过。
    返回：状态字典，status="ready" 才算可用；其余（skipped / missing_repo /
        missing_pr / 失败）都只是降级，不抛异常——CI 分析不依赖快照也能跑。
    副作用：会 clone/worktree + 建索引，比较重。
    说明：多了一步「先找到关联 PR」，因为 CI 没有直接指向 PR 的字段。
    """
    if not enabled:
        return {"status": "skipped", "message": "未请求 PR 分支快照。"}
    repo = db.get(Repository, run.repo_id)
    if repo is None:
        return {"status": "missing_repo", "message": "未找到仓库。"}
    pr = _related_pr_for_run(db, run)
    if pr is None:
        return {"status": "missing_pr", "message": "未找到可关联的 PR。"}
    snapshot = prepare_pr_worktree(db, repo, pr)
    # 把 PR 摘要塞回快照结果，前端能直接展示「本次分析基于哪个 PR」。
    snapshot["pull_request"] = {
        "id": str(pr.id),
        "number": pr.number,
        "title": pr.title,
        "head_branch": pr.head_branch,
        "state": pr.state,
    }
    return snapshot


def _ci_code_graph_impact(db: Session, pr_id: UUID, related_files: list[str]) -> dict:
    """算「这条失败 CI 关联 PR」的代码影响面。

    入参：
        pr_id: 关联 PR 的 ID（决定查哪一份 PR 级索引）。
        related_files: 从日志里抽出的文件路径，用于收窄范围。
    返回：{"related_files", "symbols", "relations"}。
    说明：
        这里只按 pr_id 过滤、不带 repo_id——PR 级索引里的 pr_id 本身已唯一确定一份快照，
        多加条件反而要再查一次仓库；上限仍是 30 符号 / 60 关系，避免撑爆上下文。
    """
    paths = list(dict.fromkeys(related_files))
    symbol_query = db.query(CodeSymbol).filter(CodeSymbol.pr_id == pr_id)
    if paths:
        symbol_query = symbol_query.filter(CodeSymbol.path.in_(paths))
    symbols = symbol_query.order_by(CodeSymbol.path.asc(), CodeSymbol.start_line.asc()).limit(30).all()
    symbol_names = [symbol.name for symbol in symbols]
    relation_query = db.query(CodeRelation).filter(CodeRelation.pr_id == pr_id)
    relation_filters = []
    if paths:
        relation_filters.append(CodeRelation.path.in_(paths))
    if symbol_names:
        # 调用方可能不在变更文件里，所以按符号名再捞一批关系边。
        relation_filters.extend([CodeRelation.source_name.in_(symbol_names), CodeRelation.target_name.in_(symbol_names)])
    if relation_filters:
        # 先判空再 or_(*filters)，否则空参数会生成非法条件。
        relation_query = relation_query.filter(or_(*relation_filters))
    relations = relation_query.limit(60).all()
    return {
        "related_files": paths[:30],
        "symbols": [
            {
                "name": symbol.name,
                "kind": symbol.kind,
                "path": symbol.path,
                "start_line": symbol.start_line,
                "branch": symbol.branch,
                "commit_sha": symbol.commit_sha,
            }
            for symbol in symbols
        ],
        "relations": [
            {
                "source": relation.source_name,
                "target": relation.target_name,
                "type": relation.relation_type,
                "path": relation.path,
                "line": (relation.meta or {}).get("line"),
            }
            for relation in relations
        ],
    }


def _augment_ci_context_with_snapshot(db: Session, run: WorkflowRun, query: str, context: dict, counts: dict, snapshot: dict) -> None:
    """把 PR 快照的证据并入 CI 分析上下文（原地修改 context/counts）。

    入参：query 为 _build_ci_analysis_context 返回的检索词；snapshot 为快照结果。
    副作用：只读库与快照目录，但会改写传入的两个字典。
    说明：
        除 status 外还要求 pr_id 存在——后续代码图谱查询必须有 PR 才能进行；
        不满足时只塞一个空的 impact 结构，保证 context 形状稳定，下游不用判空。
    """
    context["pr_snapshot"] = _snapshot_summary(snapshot)
    # 先置 0，保证即使提前 return，counts 的键也齐全（SSE 文案会直接读这些键）。
    counts["snapshot_code"] = 0
    counts["impacted_symbols"] = 0
    counts["impacted_relations"] = 0
    if snapshot.get("status") != "ready" or not snapshot.get("pr_id"):
        context["code_graph_impact"] = {"related_files": context.get("related_files_from_logs") or [], "symbols": [], "relations": []}
        return

    repo = db.get(Repository, run.repo_id)
    snapshot_docs = _document_evidence(
        search_repository_code(
            repo,
            query,
            limit=8,
            # 指向 PR 快照目录：日志里报错的代码多半只在 PR 分支上存在。
            checkout_path=snapshot.get("snapshot_path"),
            metadata={
                "checkout_kind": "pr_snapshot",
                "pr_id": str(snapshot["pr_id"]),
                "pr_number": snapshot.get("pr_number"),
                "branch": snapshot.get("branch"),
                "commit_sha": snapshot.get("commit_sha"),
            },
        )
        if repo and snapshot.get("snapshot_path")
        else [],
        "code",
    )
    # 快照代码优先，并与主干代码去重合并，最多 10 条。
    context["code_references"] = _dedupe_evidence([*snapshot_docs, *(context.get("code_references") or [])], 10)
    # snapshot["pr_id"] 可能是字符串（service 返回值未规范化），这里显式转 UUID 供查询使用。
    pr_uuid = UUID(str(snapshot["pr_id"]))
    impact = _ci_code_graph_impact(db, pr_uuid, context.get("related_files_from_logs") or [])
    context["code_graph_impact"] = impact
    counts["snapshot_code"] = len(snapshot_docs)
    counts["impacted_symbols"] = len(impact["symbols"])
    counts["impacted_relations"] = len(impact["relations"])
    # 覆盖原值：合并去重后条数可能变少，进度文案要用新数字。
    counts["code"] = len(context["code_references"])


def _build_ci_analysis_context(db: Session, run: WorkflowRun, conversation_id: UUID | None = None) -> tuple[str, dict, dict]:
    """为 CI 根因分析收集全部证据。

    入参：
        run: 目标 workflow run。
        conversation_id: 可选，补充会话证据。
    返回：(检索词, context, counts)。
    副作用：只读，但会跑代码检索与向量检索。
    设计要点：
        三个不同的查询各司其职——
            code_query     用原始日志（含堆栈/路径），目标是「找到报错的代码」；
            semantic_query 用**脱敏后的**日志，进向量库/模型，避免泄露 token；
            workflow 检索   专门在 .github/workflows 下按 run 名找配置，定位「哪个 workflow 定义有问题」。
    """
    repo = db.get(Repository, run.repo_id)
    first_error = _first_error(run.logs_text)
    failed_jobs = _failed_jobs(run)
    # 把各 job 的失败 step 摊平成一维列表；isinstance 判一下是因为 jobs 是 JSON 列，
    # 历史数据里可能混入非 dict 元素，直接 .get 会抛 AttributeError。
    failed_steps = [
        step
        for job in failed_jobs
        for step in job.get("failed_steps", [])
        if isinstance(step, dict)
    ]
    related_files = _related_files_from_logs(run.logs_text)
    code_query = "\n".join(
        filter(None, [run.name, run.conclusion or "", first_error or "", run.logs_text[:5000] if run.logs_text else ""])
    )
    semantic_query = sanitize_ci_log(
        "\n".join(filter(None, [run.name, first_error or "", run.logs_text[:3000] if run.logs_text else ""]))
    )
    code_docs = _document_evidence(
        search_repository_code(repo, code_query, limit=6) if repo else [],
        "code",
    )
    project_docs = _document_evidence(
        search_similar_documents(
            db,
            str(run.repo_id),
            semantic_query,
            limit=4,
            source_types=sorted(PROJECT_DOC_TYPES),
        ),
        "document",
    )
    if not project_docs:
        project_docs = _fallback_project_documents(db, run.repo_id)
    workflow_docs = _document_evidence(
        # relative_path 限定目录：workflow YAML 和业务代码混在一起检索时容易被淹没。
        search_repository_code(repo, run.name, limit=4, relative_path=".github/workflows") if repo else [],
        "workflow",
    )
    recent_prs = [
        {
            "number": pr.number,
            "title": pr.title,
            "state": pr.state,
            "author": pr.author,
            "updated_at": pr.updated_at.isoformat() if pr.updated_at else None,
        }
        for pr in (
            db.query(PullRequest)
            .filter(PullRequest.repo_id == run.repo_id)
            .order_by(PullRequest.updated_at.desc())
            .limit(5)
            .all()
        )
    ]
    chat_evidence = _conversation_evidence(db, run, conversation_id, semantic_query)
    context = {
        "repo_id": str(run.repo_id),
        "plan": _ci_plan(),
        "failed_jobs": failed_jobs,
        "failed_steps": failed_steps,
        "first_error": first_error,
        "related_files_from_logs": related_files,
        "code_references": code_docs,
        "workflow_references": workflow_docs,
        "project_documents": project_docs,
        "recent_prs": recent_prs,
        "conversation_evidence": chat_evidence,
    }
    counts = {
        "failed_jobs": len(failed_jobs),
        "failed_steps": len(failed_steps),
        "code": len(code_docs),
        "workflow": len(workflow_docs),
        "documents": len(project_docs),
        "recent_prs": len(recent_prs),
        "conversation": len(chat_evidence),
        "related_files": len(related_files),
    }
    return code_query, context, counts


def _persist_ci_analysis(
    db: Session,
    run: WorkflowRun,
    result: dict,
    model_name: str,
    conversation_id: UUID | None,
    evidence_counts: dict,
) -> AnalysisResult:
    """把 CI 分析结果落库（target_type="workflow_run", analysis_type="ci_debug"）。

    返回：已持久化的 AnalysisResult。
    副作用：**内部 commit**，调用方不要把它包进需要回滚的事务。
    """
    analysis = AnalysisResult(
        target_type="workflow_run",
        target_id=run.id,
        analysis_type="ci_debug",
        input_snapshot={
            "run_id": str(run.id),
            "conversation_id": str(conversation_id) if conversation_id else None,
            "evidence_counts": evidence_counts,
        },
        result_json=result,
        model_name=model_name,
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    return analysis


def _public_reasoning_fallback(run: WorkflowRun, context: dict, counts: dict) -> list[str]:
    """LLM 无输出时的兜底「排查过程」（数字都来自真实统计，内容依然准确）。"""
    first_error = context.get("first_error") or "日志中没有明确错误块"
    return [
        f"- 我先制定 CI 排查计划，再读取 workflow「{run.name}」的状态、结论、job 和日志。\n",
        f"- 当前找到失败 job {counts['failed_jobs']} 个、失败 step {counts['failed_steps']} 个，首个关键错误块是：{str(first_error)[:180]}。\n",
        f"- 我从日志中提取到 {counts['related_files']} 个相关文件，并检索代码/配置线索 {counts['code'] + counts['workflow']} 条，其中 PR 快照代码 {counts.get('snapshot_code', 0)} 条。\n",
        f"- 同时参考最近 PR {counts['recent_prs']} 条、项目文档/记忆 {counts['documents']} 条、会话证据 {counts['conversation']} 条。\n",
        "- 最后输出失败类型、根因、修复步骤，以及该 CI 是否阻塞合并。\n",
    ]


async def _stream_public_reasoning(agent: CIDebugAgent, run: WorkflowRun, payload: dict, context: dict, counts: dict) -> AsyncIterator[str]:
    """流式产出「公开排查过程」，模型无输出时退化为模板文案。

    入参：agent 复用其 llm 客户端；context_preview 每类只截前几条
        （公开过程是摘要，喂全量会让模型把日志原文抄一遍）。
    返回：异步字符串生成器。
    副作用：调用大模型（流式）。
    """
    # streamed 标记模型是否真的产出过内容，False 才走兜底。
    streamed = False
    user_payload = {
        "workflow_run": payload,
        "plan": context.get("plan") or [],
        "evidence_counts": counts,
        "context_preview": {
            "failed_jobs": context.get("failed_jobs", [])[:3],
            "first_error": context.get("first_error"),
            "related_files_from_logs": context.get("related_files_from_logs", [])[:5],
            "recent_prs": context.get("recent_prs", [])[:3],
            "pr_snapshot": context.get("pr_snapshot"),
            "code_graph_impact": context.get("code_graph_impact"),
            "code_references": context.get("code_references", [])[:3],
        },
    }
    async for chunk in agent.llm.chat_text_stream(
        PUBLIC_REASONING_PROMPT,
        [{"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}],
    ):
        streamed = True
        yield chunk
    if streamed:
        return
    for line in _public_reasoning_fallback(run, context, counts):
        # 逐行节流：兜底文案是瞬时的，不分段会被一次性刷屏。
        await asyncio.sleep(0.06)
        yield line


async def _ci_analysis_sse(run_id: UUID, conversation_id: UUID | None, db: Session) -> AsyncIterator[str]:
    """SSE 主流程：按六步计划推进度，最后给出结构化根因结论。

    事件序列：trace(计划) → trace(各步骤) → trace(思考中) → thinking_delta* →
        trace(结论) → final{analysis_id, result}；异常一律转成 error 事件。
    异常处理说明：
        响应头在生成器启动前就已发出，无法改 HTTP 状态码，
        所以「run 不存在」「分析失败」都只能作为 SSE error 事件下发。
    副作用：可能准备 PR 快照（clone + 建索引）、调用大模型、写 AnalysisResult。
    备注：第 1/6 步会推两条 trace——先一条 running 表示开始，查完后再一条带具体
        job 名的结果，前端据 status 切换「进行中/完成」。
    """
    run = db.get(WorkflowRun, run_id)
    if run is None:
        yield _encode_sse("error", {"message": "未找到 workflow 运行"})
        return
    agent = CIDebugAgent()
    payload = _workflow_payload(run)
    plan = _ci_plan()
    try:
        # 先把完整计划整条推给前端，后续文案里的 1/6、2/6 与之对应。
        yield _encode_sse("trace", _trace_item("step", "制定排查计划", "\n".join(f"{index + 1}. {item}" for index, item in enumerate(plan)), "done"))
        yield _encode_sse("trace", _trace_item("step", "执行 1/6：定位失败 job/step", f"读取 workflow「{run.name}」的 jobs、steps、状态和结论。", "running"))
        snapshot = _prepare_ci_pr_snapshot(db, run, enabled=True)
        yield _encode_sse(
            "trace",
            _trace_item(
                "tool",
                "关联 PR 分支快照",
                # 优先展示 service 给的 message（能说明失败原因），其次才是状态兜底。
                str(snapshot.get("message") or f"状态：{snapshot.get('status')}；代码图符号：{snapshot.get('indexed_symbols') or 0}。"),
            ),
        )
        _query, context, counts = _build_ci_analysis_context(db, run, conversation_id)
        _augment_ci_context_with_snapshot(db, run, _query, context, counts, snapshot)
        # 再写一次 plan：_build 里也放了一份，这里覆盖以确保与本次 SSE 使用的计划一致。
        context["plan"] = plan
        failed_job_names = [str(job.get("name")) for job in context.get("failed_jobs", []) if job.get("name")]
        yield _encode_sse("trace", _trace_item("tool", "执行 1/6：定位失败 job/step", f"失败 job：{', '.join(failed_job_names[:3]) or '未提供'}；失败 step 数：{counts['failed_steps']}。"))
        yield _encode_sse("trace", _trace_item("tool", "执行 2/6：提取首个错误块", str(context.get("first_error") or "日志中没有提取到明确 error 块。")))
        yield _encode_sse("trace", _trace_item("tool", "执行 3/6：判断失败类型", "根据日志关键词、失败命令和 job/step 名称进行分类。"))
        yield _encode_sse("trace", _trace_item("tool", "执行 4/6：搜索代码和配置", f"找到 {counts['code']} 条代码线索，其中 PR 快照代码 {counts.get('snapshot_code', 0)} 条；workflow 配置线索 {counts['workflow']} 条、日志文件路径 {counts['related_files']} 个。"))
        yield _encode_sse("trace", _trace_item("tool", "执行 5/6：关联最近 PR", f"读取最近 PR {counts['recent_prs']} 条，会话证据 {counts['conversation']} 条。"))
        yield _encode_sse("trace", _trace_item("thinking", "LLM 公开排查过程", "开始综合失败日志、代码配置、最近 PR 和项目文档。", "running"))
        async for chunk in _stream_public_reasoning(agent, run, payload, context, counts):
            yield _encode_sse("thinking_delta", {"content": chunk})
        result = await agent.run(payload, context)
        analysis = _persist_ci_analysis(db, run, result, agent.llm.model, conversation_id, counts)
        # is_merge_blocking 是布尔值，前端要的是「是/否」，所以在后端就翻成中文。
        yield _encode_sse("trace", _trace_item("result", "执行 6/6：输出根因和阻塞判断", f"失败类型：{result.get('failure_type')}；阻塞合并：{'是' if result.get('is_merge_blocking') else '否'}。"))
        yield _encode_sse("final", {"analysis_id": str(analysis.id), "result": result})
    except Exception as exc:
        # 回滚可能残留的未提交写入；错误以事件下发（抛异常会让前端只看到连接中断）。
        db.rollback()
        yield _encode_sse("error", {"message": f"分析失败：{exc}"})


@router.get("/repos/{repo_id}/workflow-runs", response_model=list[WorkflowRunResponse])
async def list_workflow_runs(repo_id: UUID, db: Session = Depends(get_db)) -> list[WorkflowRun]:
    """GET /api/repos/{repo_id}/workflow-runs：CI run 列表。

    入参：repo_id（路径）。
    返回：按创建时间倒序的完整列表，未分页。
    异常：无（仓库不存在时返回空列表）。
    副作用：只读。注意返回体里带 logs_text（完整日志），列表项多时响应会很大。
    """
    return db.query(WorkflowRun).filter(WorkflowRun.repo_id == repo_id).order_by(WorkflowRun.created_at.desc()).all()


@router.get("/workflow-runs/{run_id}", response_model=WorkflowRunResponse)
async def get_workflow_run(run_id: UUID, db: Session = Depends(get_db)) -> WorkflowRun:
    """GET /api/workflow-runs/{run_id}：单次 run 详情（含 jobs 与日志）。

    入参：run_id。
    返回：WorkflowRunResponse。
    异常：不存在 → 404。
    副作用：只读。
    """
    run = db.get(WorkflowRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="未找到 workflow 运行")
    return run


@router.get("/workflow-runs/{run_id}/failed-jobs")
async def get_failed_jobs(run_id: UUID, db: Session = Depends(get_db)) -> dict:
    """GET /api/workflow-runs/{run_id}/failed-jobs：只看失败的 job 和其中的失败 step。

    入参：run_id。
    返回：{"run_id", "workflow", "failed_jobs"}。
    异常：不存在 → 404。
    副作用：只读，**不调用大模型**——排查页首屏用它快速定位，比跑 AI 分析便宜得多。
    说明：这段筛选逻辑与 _failed_jobs 重复（没复用），改动时两处要一起改。
    """
    run = db.get(WorkflowRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="未找到 workflow 运行")
    failed_jobs = [
        {
            **job,
            "failed_steps": [step for step in job.get("steps", []) if step.get("conclusion") == "failure"],
        }
        for job in (run.jobs or [])
        if job.get("conclusion") == "failure"
    ]
    return {"run_id": str(run.id), "workflow": run.name, "failed_jobs": failed_jobs}


@router.post("/workflow-runs/{run_id}/analyze", response_model=CIAnalysisResponse)
async def analyze_workflow_run(
    run_id: UUID,
    conversation_id: UUID | None = Query(default=None),
    use_pr_snapshot: bool = Query(default=True),
    db: Session = Depends(get_db),
) -> CIAnalysisResponse:
    """POST /api/workflow-runs/{run_id}/analyze：同步版 CI 根因分析。

    入参：
        conversation_id: 可选，补充会话证据。
        use_pr_snapshot: 是否尝试准备关联 PR 的分支快照（默认开；失败会自动降级）。
    返回：CIAnalysisResponse(analysis_id, result)。
    异常：run 不存在 → 404；大模型失败向上抛（全局 500）。
    副作用：可能 clone/worktree + 建索引、调用 LLM、写 AnalysisResult。
    """
    run = db.get(WorkflowRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="未找到 workflow 运行")
    agent = CIDebugAgent()
    snapshot = _prepare_ci_pr_snapshot(db, run, enabled=use_pr_snapshot)
    _query, context, counts = _build_ci_analysis_context(db, run, conversation_id)
    _augment_ci_context_with_snapshot(db, run, _query, context, counts, snapshot)
    result = await agent.run(_workflow_payload(run), context)
    analysis = _persist_ci_analysis(db, run, result, agent.llm.model, conversation_id, counts)
    return CIAnalysisResponse(analysis_id=analysis.id, result=result)


@router.post("/workflow-runs/{run_id}/analyze/stream")
async def analyze_workflow_run_stream(
    run_id: UUID,
    conversation_id: UUID | None = Query(default=None),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """POST /api/workflow-runs/{run_id}/analyze/stream：流式 CI 根因分析（SSE）。

    入参：run_id、可选 conversation_id。
    返回：StreamingResponse，media_type=text/event-stream。
    异常：**不抛 4xx**——run 不存在/分析失败都是 SSE error 事件。
    副作用：同 /analyze；另外会一直占用一个数据库连接直到流结束。
    """
    return StreamingResponse(_ci_analysis_sse(run_id, conversation_id, db), media_type="text/event-stream")
