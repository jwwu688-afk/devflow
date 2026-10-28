"""Issue 相关接口，挂在 /api 前缀下。

路径分两类：
    GET  /api/repos/{repo_id}/issues        仓库下的 Issue 列表
    GET  /api/issues/{issue_id}             单个 Issue 详情
    POST /api/issues/{issue_id}/analyze     Issue 智能分析（同步返回 JSON）
    POST /api/issues/{issue_id}/analyze/stream  同上，但用 SSE 边分析边推送进度
    GET  /api/issues/{issue_id}/similar     找相似历史 Issue（查重用）

分析流程：抓证据（相似 Issue / 代码 / 项目文档 / 会话上下文 / 团队画像）→ 交给 IssueAgent
产出结论、优先级、复杂度、建议负责人 → 落 AnalysisResult 表。证据是「先检索再喂模型」，
不是让模型凭空猜，所以上面的 _build_* 函数都在做检索与裁剪。
"""

import asyncio
import json
from collections.abc import AsyncIterator
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.db.models import AnalysisResult, Conversation, Document, Issue, Repository
from app.db.session import get_db
from app.schemas.issues import IssueAnalysisResponse, IssueResponse
from app.services.agents.issue_agent import IssueAgent
from app.services.chat_memory import ContextAssembler
from app.services.code_search import search_repository_code
from app.services.rag.retrieval import search_similar_documents

router = APIRouter()

# 「项目文档」类的 source_type 集合：这些是人工沉淀的知识（知识库文件、记忆笔记、项目文档），
# 在 Issue 分析里作为背景资料；和代码、Issue 本身区分开，便于单独检索/兜底。
PROJECT_DOC_TYPES = {"knowledge_file", "memory_note", "project_doc"}
# 让模型输出「公开分析过程」的提示词。关键是那句「不输出隐藏链路思维」：
# 前端要把这段文字直接展示给用户看，所以只要可审计的摘要，不要内部推理链。
PUBLIC_REASONING_PROMPT = """
你是 DevFlow AI 的 Issue Agent。请用中文流式输出一段“公开分析过程”，帮助用户理解你如何分析当前 Issue。
要求：
- 只输出面向用户的可审计分析摘要，不输出隐藏链路思维或私有推理链。
- 简洁说明你检查了哪些证据、如何判断结论/优先级/复杂度/负责人。
- 不要编造输入中没有的文件、人员、Issue 或 PR。
- 不要输出最终 JSON。
"""


def _encode_sse(event: str, data: dict) -> str:
    """把一条消息编码成 SSE 帧。

    入参：event 为事件名（trace / thinking_delta / final / error），data 为要推送的负载。
    返回：符合 Server-Sent Events 规范的字符串。
    说明：
        格式必须是 "event: xxx\\ndata: {...}\\n\\n"——结尾的空行才是帧分隔符，
        少了它浏览器会一直等下一帧、消息永远不派发。
        ensure_ascii=False 保留中文原样，前端不必额外解码。
    """
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _document_evidence(items: list[dict], default_source_type: str) -> list[dict]:
    """把检索结果规整成统一的证据结构，并丢掉不合格的条目。

    入参：
        items: 检索服务返回的原始字典列表。
        default_source_type: 原始结果没带 source_type 时用的默认值。
    返回：统一字段（source_type/title/snippet/source_id/score/metadata）的证据列表。
    说明：
        缺标题或缺摘要的条目直接丢弃——它们在前端证据卡片里没有可展示的内容，
        留着只会占位置，还可能让模型在没依据的情况下去编。
    """
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
                # 不同检索服务的 ID 字段名不一样（source_id / id），逐个兜底。
                "source_id": str(item.get("source_id") or item.get("id") or ""),
                "score": item.get("score"),
                "metadata": item.get("metadata") or {},
            }
        )
    return evidence


def _fallback_project_documents(db: Session, repo_id: UUID) -> list[dict]:
    """语义检索没找到项目文档时的兜底：直接取最近的 4 篇。

    为什么需要兜底：向量检索对短 Issue 标题经常召回为空，但「仓库有哪些文档」本身是有用信息，
    给模型几篇最近的文档比什么都不给更能减少胡编。
    返回结构与 _document_evidence 的产出保持一致，便于上层无差别合并。
    """
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
            # 截 260 字符：只是给个概览，全文会撑爆 prompt。
            "snippet": doc.content[:260],
            "source_id": str(doc.source_id),
            # 兜底数据没有相关性分数，显式置 None，前端据此不显示「相似度」。
            "score": None,
            "metadata": doc.meta or {},
        }
        for doc in rows
    ]


def _conversation_evidence(db: Session, issue: Issue, conversation_id: UUID | None, query: str) -> list[dict]:
    """把「当前会话」的上下文也当作证据。

    入参：
        issue: 当前 Issue，用来校验会话是否属于同一仓库。
        conversation_id: 前端从聊天页跳转过来时带上的会话；没有就返回空列表。
        query: 组装上下文用的检索词（Issue 标题 + 正文）。
    返回：最多 5 条会话证据。
    说明：
        conversation.repo_id != issue.repo_id 时直接返回空：会话可能属于别的仓库，
        混进来会泄露无关内容，也会污染分析结论。
    """
    if not conversation_id:
        return []
    conversation = db.get(Conversation, conversation_id)
    if conversation is None or conversation.repo_id != issue.repo_id:
        return []
    repo = db.get(Repository, issue.repo_id)
    # limit=8 是检索上限，最终只取 5 条（见函数末尾切片），留出裁剪空间。
    assembled = ContextAssembler(db).assemble(repo, conversation, query, limit=8)
    evidence = _document_evidence(assembled.evidence, "conversation")
    if assembled.system_context.strip():
        # 除逐条证据外，再补一条「会话摘要」：有时候上下文的价值在于整体脉络而非某条片段。
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


def _issue_payload(issue: Issue) -> dict:
    """把 Issue 行转成给 Agent 的输入快照（只挑模型需要的字段，不带时间戳等无关列）。"""
    return {
        "id": str(issue.id),
        "title": issue.title,
        "body": issue.body,
        "labels": issue.labels,
        "number": issue.number,
        "state": issue.state,
        "assignees": issue.assignees,
        "author": issue.author,
    }


def _build_issue_analysis_context(db: Session, issue: Issue, conversation_id: UUID | None = None) -> tuple[str, dict, dict]:
    """为 Issue 分析收集全部证据。

    入参：
        issue: 目标 Issue。
        conversation_id: 可选，用于补充会话上下文。
    返回：
        (query, context, counts)：
            query   —— 拼好的检索词（标题 + 正文），也用于后续相似度搜索；
            context —— 喂给 Agent 的证据包；
            counts  —— 各类证据条数，用于 SSE 进度文案和落库快照。
    副作用：只读，但会触发多次向量检索（embedding + Milvus 查询），有一定耗时。
    """
    query = f"{issue.title}\n{issue.body or ''}"
    repo = db.get(Repository, issue.repo_id)
    similar_docs = search_similar_documents(db, str(issue.repo_id), query, "issue", 4)
    duplicates = [
        {
            "issue_id": item["source_id"],
            "title": item["title"],
            # 把相似度夹到 [0, 1]：不同后端算出的分数范围不一（余弦可能为负），
            # 前端按百分比展示，越界值会显示成 -12% 这种明显错误的数字。
            "score": max(0.0, min(1.0, float(item["score"]))),
        }
        for item in similar_docs
        # 排除自己：检索用标题当 query，自身几乎必然命中，留着会让「重复 Issue」判断失真。
        if item["source_id"] != str(issue.id)
    ]
    code_docs = _document_evidence(
        # repo 可能已被删但 Issue 还在，此时不做代码检索（否则 search_repository_code 会炸）。
        search_repository_code(repo, query, limit=4) if repo else [],
        "code",
    )
    project_docs = _document_evidence(
        search_similar_documents(
            db,
            str(issue.repo_id),
            query,
            limit=4,
            # 只在这几类文档里搜；sorted 是为了让调用参数稳定（便于缓存/日志对比）。
            source_types=sorted(PROJECT_DOC_TYPES),
        ),
        "document",
    )
    if not project_docs:
        project_docs = _fallback_project_documents(db, issue.repo_id)
    chat_evidence = _conversation_evidence(db, issue, conversation_id, query)
    # 团队画像来自 Document(source_type="team_member")：姓名/角色/擅长/技术栈，
    # 供 Agent 推荐负责人时只在真实成员里选，避免编出不存在的同事。
    team_members = [
        {
            # 优先用 meta 里的 name；老数据没有 meta 时，从标题「团队成员：张三」里剥出名字。
            "name": (doc.meta or {}).get("name") or doc.title.replace("团队成员：", "").replace("Team member: ", ""),
            "role": (doc.meta or {}).get("role") or "",
            "strengths": (doc.meta or {}).get("strengths") or "",
            "techStack": (doc.meta or {}).get("tech_stack") or "",
        }
        for doc in db.query(Document).filter(Document.repo_id == issue.repo_id, Document.source_type == "team_member").all()
    ]
    context = {
        "repo_id": str(issue.repo_id),
        "duplicate_candidates": duplicates,
        "code_references": code_docs,
        "project_documents": project_docs,
        "conversation_evidence": chat_evidence,
        "team_members": team_members,
    }
    counts = {
        "duplicates": len(duplicates),
        "code": len(code_docs),
        "documents": len(project_docs),
        "conversation": len(chat_evidence),
        "team_members": len(team_members),
    }
    return query, context, counts


def _persist_issue_analysis(
    db: Session,
    issue: Issue,
    result: dict,
    model_name: str,
    conversation_id: UUID | None,
    evidence_counts: dict,
) -> AnalysisResult:
    """把一次分析结果落库，便于历史回溯与效果评估。

    入参：
        result: Agent 产出的结构化结论。
        model_name: 实际使用的模型名（不同模型效果差异大，落库才能事后归因）。
        evidence_counts: 当时各类证据的条数，用于复现「结论建立在多少证据上」。
    返回：已持久化的 AnalysisResult。
    副作用：**内部直接 commit**——所以调用方不要把它包在需要回滚的大事务里。
    """
    analysis = AnalysisResult(
        target_type="issue",
        target_id=issue.id,
        analysis_type="issue_analysis",
        input_snapshot={
            "issue_id": str(issue.id),
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


def _trace_item(kind: str, title: str, content: str, status: str = "done") -> dict:
    """构造一条前端时间线（trace）条目。

    kind 取值 step/tool/thinking/result 决定前端图标；
    status 为 running/done，用来显示「进行中」的转圈状态。
    """
    return {"kind": kind, "title": title, "content": content, "status": status}


def _public_reasoning_fallback(issue: Issue, context: dict) -> list[str]:
    """模型一个字都没吐出来时的兜底「分析过程」。

    用途：LLM 不可用/超时也要让前端看到有内容的进度，而不是空白面板干等。
    文案里填的是真实统计数（证据条数），所以即使没有模型，这五句话也是准确的。
    """
    duplicates = context.get("duplicate_candidates") or []
    code_refs = context.get("code_references") or []
    docs = context.get("project_documents") or []
    members = context.get("team_members") or []
    return [
        f"- 我先读取 Issue #{issue.number} 的标题、正文、标签和当前状态，确认它描述的是「{issue.title}」。\n",
        f"- 然后检索相似历史 Issue，找到 {len(duplicates)} 个候选；如果相似度很高，会优先建议合并到已有 Issue。\n",
        f"- 接着查看代码和项目文档证据：代码线索 {len(code_refs)} 条，文档/记忆线索 {len(docs)} 条，用于判断影响范围和复杂度。\n",
        f"- 最后结合团队画像，当前可参考成员 {len(members)} 人；负责人只会从已配置成员或明确方向中选择。\n",
        "- 这些公开分析步骤会汇总成结构化结论：结论、优先级、复杂度、负责人、证据、checklist 和可选草稿。\n",
    ]


async def _stream_public_reasoning(agent: IssueAgent, issue: Issue, payload: dict, context: dict) -> AsyncIterator[str]:
    """流式产出「公开分析过程」文本，模型无输出时退化为模板文案。

    入参：
        agent: 已构造的 IssueAgent，复用它的 llm 客户端（省一次初始化）。
        payload: Issue 快照。
        context: 完整证据包。
    返回：异步字符串生成器，逐段吐出分析文字。
    副作用：调用大模型（流式）。
    说明：
        context_preview 每类只截前 3 条：公开分析过程是给人看的摘要，
        喂全量证据既费 token 又容易让模型把证据原文抄一遍。
    """
    # streamed 用来判断模型是否真的产出过内容；False 才走兜底。
    streamed = False
    user_payload = {
        "issue": payload,
        "evidence_counts": {
            "duplicates": len(context.get("duplicate_candidates") or []),
            "code_references": len(context.get("code_references") or []),
            "project_documents": len(context.get("project_documents") or []),
            "conversation_evidence": len(context.get("conversation_evidence") or []),
            "team_members": len(context.get("team_members") or []),
        },
        "context_preview": {
            "duplicates": context.get("duplicate_candidates", [])[:3],
            "code_references": context.get("code_references", [])[:3],
            "project_documents": context.get("project_documents", [])[:3],
            "team_members": context.get("team_members", [])[:3],
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
    for line in _public_reasoning_fallback(issue, context):
        # 逐行吐出时人为加 60ms 间隔：兜底文案是瞬间生成的，
        # 不加节流会在同一帧全推给前端，看起来像卡住后突然刷屏。
        await asyncio.sleep(0.06)
        yield line


async def _issue_analysis_sse(issue_id: UUID, conversation_id: UUID | None, db: Session) -> AsyncIterator[str]:
    """SSE 主流程：边分析边推进度事件，最后给结构化结果。

    事件序列：trace(读取 Issue) → trace(各类检索) → trace(思考中) → thinking_delta* →
        trace(结论) → final{analysis_id, result}；任何异常都转成 error 事件。
    异常处理说明：
        响应头在生成器开始执行前就已发出，此时**无法再改 HTTP 状态码**，
        所以「Issue 不存在」「分析失败」都只能作为 SSE 的 error 事件推给前端，
        前端需要监听 error 事件而不是只看 HTTP 状态。
    副作用：大模型调用 + 写 AnalysisResult。
    """
    issue = db.get(Issue, issue_id)
    if issue is None:
        yield _encode_sse("error", {"message": "未找到 Issue"})
        return
    agent = IssueAgent()
    payload = _issue_payload(issue)
    try:
        # 先推一条 running 的 trace，让前端立刻有反应（真正的检索随后才开始）。
        yield _encode_sse("trace", _trace_item("step", "读取 Issue", f"读取 Issue #{issue.number} 的标题、正文、标签、状态和已分配人员。", "running"))
        query, context, counts = _build_issue_analysis_context(db, issue, conversation_id)
        yield _encode_sse("trace", _trace_item("tool", "检索相似历史", f"找到 {counts['duplicates']} 个相似 Issue 候选。"))
        yield _encode_sse("trace", _trace_item("tool", "检索代码与文档", f"找到 {counts['code']} 条代码线索、{counts['documents']} 条项目文档/记忆线索。"))
        yield _encode_sse("trace", _trace_item("tool", "读取团队画像", f"读取 {counts['team_members']} 位团队成员，用于建议负责人。"))
        if counts["conversation"]:
            # 没有会话上下文时不推这条，避免出现「找到 0 条」的无意义噪音。
            yield _encode_sse("trace", _trace_item("tool", "读取会话上下文", f"找到 {counts['conversation']} 条与当前会话相关的证据。"))
        yield _encode_sse("trace", _trace_item("thinking", "LLM 公开分析过程", "开始综合 Issue、证据和团队画像。", "running"))
        async for chunk in _stream_public_reasoning(agent, issue, payload, context):
            yield _encode_sse("thinking_delta", {"content": chunk})
        result = await agent.run(payload, context)
        analysis = _persist_issue_analysis(db, issue, result, agent.llm.model, conversation_id, counts)
        yield _encode_sse("trace", _trace_item("result", "生成结构化结论", f"结论：{result.get('conclusion')}；优先级：{result.get('priority')}；复杂度：{result.get('complexity')}；负责人：{result.get('suggested_owner')}。"))
        yield _encode_sse("final", {"analysis_id": str(analysis.id), "result": result})
    except Exception as exc:
        # 中途失败要回滚：_build_* 阶段可能有未提交的 flush，留着会污染后续请求的 session。
        db.rollback()
        # 不 raise：一旦抛出，生成器结束，前端只会看到连接断开，拿不到失败原因。
        yield _encode_sse("error", {"message": f"分析失败：{exc}"})


@router.get("/repos/{repo_id}/issues", response_model=list[IssueResponse])
async def list_issues(repo_id: UUID, db: Session = Depends(get_db)) -> list[Issue]:
    """GET /api/repos/{repo_id}/issues：列出仓库全部 Issue。

    入参：repo_id（路径）；db 由 get_db 注入。
    返回：按 issue 编号倒序（新的在前）的完整列表，**未分页**。
    异常：无（仓库不存在时返回空列表）。
    副作用：只读。
    """
    return db.query(Issue).filter(Issue.repo_id == repo_id).order_by(Issue.number.desc()).all()


@router.get("/issues/{issue_id}", response_model=IssueResponse)
async def get_issue(issue_id: UUID, db: Session = Depends(get_db)) -> Issue:
    """GET /api/issues/{issue_id}：Issue 详情。

    入参：issue_id（UUID，格式非法由 FastAPI 直接 422）。
    返回：IssueResponse。
    异常：不存在 → 404。
    副作用：只读。
    """
    issue = db.get(Issue, issue_id)
    if issue is None:
        raise HTTPException(status_code=404, detail="未找到 Issue")
    return issue


@router.post("/issues/{issue_id}/analyze", response_model=IssueAnalysisResponse)
async def analyze_issue(
    issue_id: UUID,
    conversation_id: UUID | None = Query(default=None),
    db: Session = Depends(get_db),
) -> IssueAnalysisResponse:
    """POST /api/issues/{issue_id}/analyze：同步版 Issue 分析。

    入参：
        issue_id: 目标 Issue。
        conversation_id: 可选查询参数，带上则把该会话上下文作为证据。
    返回：IssueAnalysisResponse(analysis_id, result)。
    异常：Issue 不存在 → 404；大模型/检索失败 → 异常向上抛（走全局 500）。
    副作用：调用 LLM，并写入一条 AnalysisResult。

    和 /analyze/stream 的关系：逻辑完全一样，只是这个接口要等全部算完才返回，
    适合脚本/测试调用；前端交互请用 stream 版，避免长时间白屏。
    """
    issue = db.get(Issue, issue_id)
    if issue is None:
        raise HTTPException(status_code=404, detail="未找到 Issue")
    agent = IssueAgent()
    # _query 这里用不到（检索已在内部完成），下划线前缀表示有意忽略。
    _query, context, counts = _build_issue_analysis_context(db, issue, conversation_id)
    result = await agent.run(_issue_payload(issue), context)
    analysis = _persist_issue_analysis(db, issue, result, agent.llm.model, conversation_id, counts)
    return IssueAnalysisResponse(analysis_id=analysis.id, result=result)


@router.post("/issues/{issue_id}/analyze/stream")
async def analyze_issue_stream(
    issue_id: UUID,
    conversation_id: UUID | None = Query(default=None),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """POST /api/issues/{issue_id}/analyze/stream：流式版 Issue 分析（SSE）。

    入参：同 /analyze。
    返回：StreamingResponse，media_type=text/event-stream。
    异常：**这里不会抛 4xx**——Issue 不存在、分析失败都以 SSE error 事件下发，
        因为响应头在生成器启动前就已确定为 200。
    副作用：与 /analyze 相同（LLM + 落库），另外持有数据库连接直到流结束——
        客户端中途断开时生成器被取消，连接才释放，长连接较多时要注意连接池大小。
    """
    return StreamingResponse(_issue_analysis_sse(issue_id, conversation_id, db), media_type="text/event-stream")


@router.get("/issues/{issue_id}/similar")
async def similar_issues(issue_id: UUID, db: Session = Depends(get_db)) -> list[dict]:
    """GET /api/issues/{issue_id}/similar：找相似历史 Issue（查重用）。

    入参：issue_id。
    返回：最多 5 条相似文档（含 title / source_id / score 等，不排除自身）。
    异常：Issue 不存在 → 404。
    副作用：只读；会调用 embedding + 向量检索。

    注意与 _build_issue_analysis_context 的差异：那边排除自身并夹紧分数，
    这个接口是「给用户看的相似列表」，行为更原始；调用方需要自行处理自身命中的情况。
    """
    issue = db.get(Issue, issue_id)
    if issue is None:
        raise HTTPException(status_code=404, detail="未找到 Issue")
    return search_similar_documents(db, str(issue.repo_id), f"{issue.title}\n{issue.body or ''}", "issue", 5)
