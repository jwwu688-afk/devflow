"""Pull Request 相关接口，挂在 /api 前缀下。

核心是「PR 合并就绪度评审」：把 PR 的 diff、相关代码、项目文档、历史 Issue、CI 结果、
review 评论和团队画像一起喂给 PRReviewAgent，产出一份可执行的合入建议
（建议合入 / 修改后合入 / 暂缓 / 拒绝）+ 风险点 + 阻塞项 + 检查清单。

接口：
    GET  /api/repos/{repo_id}/pull-requests      PR 列表
    GET  /api/pull-requests/{pr_id}              详情
    POST /api/pull-requests/{pr_id}/analyze      同步分析
    POST /api/pull-requests/{pr_id}/analyze/stream   SSE 流式分析（前端主用）
    POST /api/pull-requests/{pr_id}/snapshot     准备该 PR 的分支快照（worktree + 代码索引）
    POST /api/pull-requests/{pr_id}/worktree     同上，历史别名
    POST /api/repos/{repo_id}/pr-snapshots/cleanup  清理过期快照
    POST /api/pull-requests/{pr_id}/review-checklist 只要检查清单

为什么要「PR 分支快照」：评审要看的是 PR 分支上的代码，而仓库默认 checkout 的是主干。
快照 = 在独立目录里 checkout 该 PR 的 head commit 并单独索引，评审时优先从快照取代码。
"""

import asyncio
import json
from collections.abc import AsyncIterator
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import or_
from sqlalchemy.orm import Session, selectinload

from app.db.models import AnalysisResult, CodeRelation, CodeSymbol, Conversation, Document, PullRequest, Repository, WorkflowRun
from app.db.session import get_db
from app.schemas.pull_requests import PRAnalysisResponse, PullRequestResponse
from app.services.agents.pr_review_agent import PRReviewAgent
from app.services.chat_memory import ContextAssembler
from app.services.code_search import search_repository_code
from app.services.rag.retrieval import search_similar_documents
from app.services.worktree_manager import cleanup_pr_snapshots, prepare_pr_worktree

router = APIRouter()

# 与 issues.py 保持一致的「项目文档」类型集合，保证两个 Agent 看到的背景资料口径相同。
PROJECT_DOC_TYPES = {"knowledge_file", "memory_note", "project_doc"}
# 让模型输出「公开执行过程」的提示词。两条关键约束：
#   1. 只给可审计摘要，不给私有推理链（这段文字会直接展示给用户）；
#   2. 结论必须落在那四个枚举值里——下游前端按固定值渲染颜色/图标，不能自由发挥。
PUBLIC_REASONING_PROMPT = """
你是 DevFlow AI 的 PR 合并就绪度 Agent。请用中文流式输出一段“公开执行过程”，帮助用户理解你如何按计划审查当前 PR。
要求：
- 只输出面向用户的可审计分析摘要，不输出隐藏链路思维或私有推理链。
- 按计划说明你检查了 PR 目标、diff 风险、相关代码/文档、CI、Review 评论，并如何形成合入建议。
- 最终建议只能围绕：建议合入 / 修改后合入 / 暂缓 / 拒绝。
- 不要编造输入中没有的文件、CI、Issue、人员或评论。
- 不要输出最终 JSON。
"""


def _encode_sse(event: str, data: dict) -> str:
    """编码一条 SSE 帧（event + JSON data + 空行结尾）。

    结尾的空行是必需的帧分隔符；少了它浏览器不会派发这条消息。
    ensure_ascii=False 保留中文，省一次前端解码。
    """
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _trace_item(kind: str, title: str, content: str, status: str = "done") -> dict:
    """构造前端时间线条目：kind 决定图标（step/tool/thinking/result），status 决定是否转圈。"""
    return {"kind": kind, "title": title, "content": content, "status": status}


def _document_evidence(items: list[dict], default_source_type: str) -> list[dict]:
    """把检索结果规整成证据结构；缺标题或摘要的条目直接丢弃（没有可展示内容，留着只会占位）。"""
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
                # 不同检索源的 ID 字段名不同，逐个兜底。
                "source_id": str(item.get("source_id") or item.get("id") or ""),
                "score": item.get("score"),
                "metadata": item.get("metadata") or {},
            }
        )
    return evidence


def _dedupe_evidence(items: list[dict], limit: int) -> list[dict]:
    """按 (来源类型, 来源ID, 标题) 去重并截断到 limit 条。

    为什么需要去重：PR 快照代码和主干代码可能检索到同一处，直接叠加会让同一段代码在
    证据里出现两次，既浪费 token 又让模型误以为有两处不同的问题。
    保持原有顺序（快照在前），所以是「先到先得」，不是随机保留。
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
    """语义检索没命中项目文档时，兜底取最近 4 篇，避免模型完全没有项目背景。"""
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
            # 截 260 字符只是概览，全文会撑爆 prompt。
            "snippet": doc.content[:260],
            "source_id": str(doc.source_id),
            # 兜底数据没有相关性分数，置 None 让前端不显示相似度。
            "score": None,
            "metadata": doc.meta or {},
        }
        for doc in rows
    ]


def _conversation_evidence(db: Session, pr: PullRequest, conversation_id: UUID | None, query: str) -> list[dict]:
    """把当前会话上下文作为额外证据（最多 5 条）。

    会话不属于本 PR 所在仓库时返回空——否则会把无关仓库的对话内容带进评审，
    既是噪音也是越权。
    """
    if not conversation_id:
        return []
    conversation = db.get(Conversation, conversation_id)
    if conversation is None or conversation.repo_id != pr.repo_id:
        return []
    repo = db.get(Repository, pr.repo_id)
    assembled = ContextAssembler(db).assemble(repo, conversation, query, limit=8)
    evidence = _document_evidence(assembled.evidence, "conversation")
    if assembled.system_context.strip():
        # 除了逐条证据，再补一条整体会话摘要（有时价值在于脉络而非单条片段）。
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


def _pr_payload(pr: PullRequest) -> dict:
    """把 PR 行转成给 Agent 的输入快照。

    包含变更文件（含 patch）和 review 评论——这两块是评审的核心输入。
    注意 pr.files / pr.review_comments 必须是已 selectinload 预加载的，
    否则这里会触发 N+1 查询（详情/列表接口都加了 options(selectinload(...))）。
    """
    return {
        "id": str(pr.id),
        "number": pr.number,
        "title": pr.title,
        "body": pr.body,
        "state": pr.state,
        "author": pr.author,
        "base_branch": pr.base_branch,
        "head_branch": pr.head_branch,
        # 时间统一转 ISO 字符串：后续要 json.dumps 进 prompt，datetime 无法序列化。
        "merged_at": pr.merged_at.isoformat() if pr.merged_at else None,
        "files": [
            {
                "filename": file.filename,
                "status": file.status,
                "additions": file.additions,
                "deletions": file.deletions,
                "patch": file.patch,
            }
            for file in pr.files
        ],
        "comments": [
            {"body": comment.body, "path": comment.path, "line": comment.line, "author": comment.author}
            for comment in pr.review_comments
        ],
    }


def _pr_review_plan() -> list[str]:
    """固定的六步评审计划。

    固定（而非让 LLM 自己规划）的好处：步骤可枚举，SSE 能把「执行 1/6」这类进度
    准确地推给前端，评审过程对用户可见、可对照。
    """
    return [
        "理解 PR 标题、描述、变更文件和关联上下文。",
        "按模块拆分 diff 风险。",
        "读取相关源代码、测试文件和项目文档。",
        "检查 CI 状态和失败信号。",
        "检索团队约定、历史对话和 review comments。",
        "形成合入建议和 review comments。",
    ]


def _build_pr_analysis_context(db: Session, pr: PullRequest, conversation_id: UUID | None = None) -> tuple[str, dict, dict]:
    """为 PR 评审收集全部证据。

    入参：
        pr: 目标 PR（需已预加载 files / review_comments）。
        conversation_id: 可选，用于补充会话证据。
    返回：(检索词, context, counts)。检索词是拼出来的复合查询，供快照阶段复用。
    副作用：只读，但会跑多次向量/代码检索。
    设计要点：
        这里用了两个不同的 query——
            code_query    带上 diff 正文，因为找相关代码靠的是「改了哪些代码」；
            semantic_query 不带 diff（只标题+描述+评论），因为找文档/Issue 靠的是「意图」，
            塞进 diff 会把检索带偏成「找相似代码」。
    """
    repo = db.get(Repository, pr.repo_id)
    file_names = [file.filename for file in pr.files]
    # 只取前 12 个文件的 patch：大 PR 的 diff 可能有几十万字符，全塞进去既超 token 也没必要。
    diff_text = "\n".join(f"{file.filename}\n{file.patch or ''}" for file in pr.files[:12])
    # filter(None, ...) 顺手过滤掉空标题/空正文，避免拼出一堆空行。
    code_query = "\n".join(filter(None, [pr.title, pr.body or "", " ".join(file_names), diff_text[:5000]]))
    review_text = "\n".join(comment.body or "" for comment in pr.review_comments)
    semantic_query = "\n".join(filter(None, [pr.title, pr.body or "", review_text[:3000]]))
    code_docs = _document_evidence(
        # repo 可能已被删，此时跳过代码检索而不是让 service 报错。
        search_repository_code(repo, code_query, limit=6) if repo else [],
        "code",
    )
    project_docs = _document_evidence(
        search_similar_documents(
            db,
            str(pr.repo_id),
            semantic_query,
            limit=4,
            source_types=sorted(PROJECT_DOC_TYPES),
        ),
        "document",
    )
    if not project_docs:
        project_docs = _fallback_project_documents(db, pr.repo_id)
    related_issues = _document_evidence(
        search_similar_documents(db, str(pr.repo_id), semantic_query, "issue", 4),
        "issue",
    )
    chat_evidence = _conversation_evidence(db, pr, conversation_id, semantic_query)
    # 只看最近 8 次 CI：更早的 run 与当前 PR 基本无关，还会让模型被历史噪音干扰。
    recent_runs = (
        db.query(WorkflowRun)
        .filter(WorkflowRun.repo_id == pr.repo_id)
        .order_by(WorkflowRun.created_at.desc())
        .limit(8)
        .all()
    )
    failed_runs = [run for run in recent_runs if run.conclusion == "failure"]
    # 团队画像：让模型推荐 reviewer 时只能在真实成员里选。
    team_members = [
        {
            "name": (doc.meta or {}).get("name") or doc.title.replace("团队成员：", "").replace("Team member: ", ""),
            "role": (doc.meta or {}).get("role") or "",
            "strengths": (doc.meta or {}).get("strengths") or "",
            "techStack": (doc.meta or {}).get("tech_stack") or "",
        }
        for doc in db.query(Document).filter(Document.repo_id == pr.repo_id, Document.source_type == "team_member").all()
    ]
    total_additions = sum(file.additions or 0 for file in pr.files)
    total_deletions = sum(file.deletions or 0 for file in pr.files)
    # 敏感文件识别：命中这些关键词的文件（鉴权/安全/迁移/表结构/支付/权限/配置/CI）
    # 会让评审额外谨慎。这是启发式规则，不追求精确，目的是提醒模型和用户重点看这些 diff。
    sensitive_files = [
        name
        for name in file_names
        if any(token in name.lower() for token in ["auth", "security", "migration", "schema", "payment", "permission", "config", "ci", "workflow"])
    ]
    context = {
        "repo_id": str(pr.repo_id),
        "plan": _pr_review_plan(),
        "diff_stats": {
            "files": len(pr.files),
            "additions": total_additions,
            "deletions": total_deletions,
            "sensitive_files": sensitive_files,
        },
        "code_references": code_docs,
        "project_documents": project_docs,
        "related_issues": related_issues,
        "conversation_evidence": chat_evidence,
        # 复用 _pr_payload 里已组装好的评论列表，避免再写一遍映射。
        "review_comments": _pr_payload(pr)["comments"],
        "ci_summary": {
            "recent_runs": len(recent_runs),
            "failed_runs": len(failed_runs),
            # 失败日志只截 260 字符 × 最多 4 条：日志动辄几十 KB，全给会直接爆上下文。
            "failed": [
                {"name": run.name, "conclusion": run.conclusion, "logs_text": (run.logs_text or "")[:260]}
                for run in failed_runs[:4]
            ],
        },
        "team_members": team_members,
    }
    counts = {
        "files": len(pr.files),
        "comments": len(pr.review_comments),
        "code": len(code_docs),
        "documents": len(project_docs),
        "issues": len(related_issues),
        "conversation": len(chat_evidence),
        "failed_ci": len(failed_runs),
        "team_members": len(team_members),
    }
    return code_query, context, counts


def _snapshot_summary(snapshot: dict) -> dict:
    """把快照准备结果裁成前端需要的字段（会直接出现在「公开执行过程」里）。

    name 兜底为「PR 分支快照」：某些失败分支不返回 snapshot_name。
    """
    return {
        "status": snapshot.get("status"),
        "message": snapshot.get("message"),
        "name": snapshot.get("snapshot_name") or "PR 分支快照",
        "branch": snapshot.get("branch"),
        "commit_sha": snapshot.get("commit_sha"),
        "pr_id": snapshot.get("pr_id"),
        "pr_number": snapshot.get("pr_number"),
        "indexed_symbols": snapshot.get("indexed_symbols"),
    }


def _prepare_pr_snapshot_context(db: Session, pr: PullRequest, enabled: bool) -> dict:
    """按需准备 PR 分支快照（checkout + 建索引），返回状态字典。

    入参：enabled=False 时直接跳过（调用方显式关掉快照）。
    返回：含 status 的字典；status="ready" 才表示快照可用，
        其它值（skipped / missing_repo / 各种失败）都只是「降级」，不抛异常。
    副作用：会真实 clone/worktree + 跑代码分析，比较重；
        所以只在 analyze 时按需触发，列表接口不会调用。
    """
    if not enabled:
        return {"status": "skipped", "message": "未请求 PR 分支快照。"}
    repo = db.get(Repository, pr.repo_id)
    if repo is None:
        return {"status": "missing_repo", "message": "未找到仓库。"}
    return prepare_pr_worktree(db, repo, pr)


def _pr_code_graph_impact(db: Session, pr: PullRequest) -> dict:
    """从代码图谱里算出「这个 PR 影响到哪些符号和关系」。

    入参：pr（需已预加载 files）。
    返回：{"changed_paths", "symbols", "relations"}，用于给模型一个「改动的影响面」视角，
        也让前端能画变更影响图。
    说明：
        - 只看该 PR 自己那份索引（pr_id 过滤），避免和主干符号混在一起；
        - dict.fromkeys 用来「去重且保序」——同一个文件可能出现在多次提交里；
        - symbols 取 30 个、relations 取 60 条，都是给模型看的规模上限，
          超出的部分对结论边际价值很低，反而占满上下文。
    """
    changed_paths = list(dict.fromkeys(file.filename for file in pr.files if file.filename))
    symbol_query = db.query(CodeSymbol).filter(CodeSymbol.repo_id == pr.repo_id, CodeSymbol.pr_id == pr.id)
    if changed_paths:
        symbol_query = symbol_query.filter(CodeSymbol.path.in_(changed_paths))
    symbols = symbol_query.order_by(CodeSymbol.path.asc(), CodeSymbol.start_line.asc()).limit(30).all()
    symbol_names = [symbol.name for symbol in symbols]
    relation_query = db.query(CodeRelation).filter(CodeRelation.repo_id == pr.repo_id, CodeRelation.pr_id == pr.id)
    relation_filters = []
    if changed_paths:
        relation_filters.append(CodeRelation.path.in_(changed_paths))
    if symbol_names:
        # 关系边可能落在未变更的文件里（比如别处调用了改动的函数），
        # 所以除了「路径在变更列表里」，还要按「源/目标符号名命中」再捞一批。
        relation_filters.extend([CodeRelation.source_name.in_(symbol_names), CodeRelation.target_name.in_(symbol_names)])
    if relation_filters:
        # or_(*[]) 在没有条件时会退化成空条件，所以先判空再拼，避免生成非法 SQL。
        relation_query = relation_query.filter(or_(*relation_filters))
    relations = relation_query.limit(60).all()
    return {
        "changed_paths": changed_paths[:30],
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
                # 行号存在 meta 里（不同语言解析器给的字段不完全一致），缺失就是 None。
                "line": (relation.meta or {}).get("line"),
            }
            for relation in relations
        ],
    }


def _augment_pr_context_with_snapshot(db: Session, pr: PullRequest, query: str, context: dict, counts: dict, snapshot: dict) -> None:
    """把「PR 分支快照」的证据并入已有 context（原地修改，无返回值）。

    入参：
        query: _build_pr_analysis_context 返回的复合检索词。
        context / counts: 就地补充 pr_snapshot、code_graph_impact 和若干计数。
        snapshot: _prepare_pr_snapshot_context 的结果。
    副作用：只读数据库与快照目录；但会改写传入的 context/counts 字典。
    说明：
        快照没准备好（status != "ready"）时不做代码检索，只塞一个空的 impact 结构，
        保证 context 的字段形状始终一致——下游模板和前端不用到处判 None。
        快照代码优先于主干代码（放在列表前面），因为评审要看的是 PR 里的版本。
    """
    context["pr_snapshot"] = _snapshot_summary(snapshot)
    # 先初始化计数为 0：即使下面提前 return，counts 的键也是齐的，SSE 文案不会 KeyError。
    counts["snapshot_code"] = 0
    counts["impacted_symbols"] = 0
    counts["impacted_relations"] = 0
    if snapshot.get("status") != "ready":
        context["code_graph_impact"] = {"changed_paths": [file.filename for file in pr.files], "symbols": [], "relations": []}
        return

    repo = db.get(Repository, pr.repo_id)
    snapshot_docs = _document_evidence(
        search_repository_code(
            repo,
            query,
            limit=8,
            # 指定 checkout_path：让代码检索指向 PR 快照目录而不是主干 checkout。
            checkout_path=snapshot.get("snapshot_path"),
            metadata={
                "checkout_kind": "pr_snapshot",
                "pr_id": str(pr.id),
                "pr_number": pr.number,
                "branch": snapshot.get("branch"),
                "commit_sha": snapshot.get("commit_sha"),
            },
        )
        if repo and snapshot.get("snapshot_path")
        else [],
        "code",
    )
    # 快照代码排前面 + 去重合并主干代码，最终最多 10 条。
    context["code_references"] = _dedupe_evidence([*snapshot_docs, *(context.get("code_references") or [])], 10)
    impact = _pr_code_graph_impact(db, pr)
    context["code_graph_impact"] = impact
    counts["snapshot_code"] = len(snapshot_docs)
    counts["impacted_symbols"] = len(impact["symbols"])
    counts["impacted_relations"] = len(impact["relations"])
    # 覆盖 counts["code"]：去重合并后真实条数可能变少，进度文案要用新值。
    counts["code"] = len(context["code_references"])


def _persist_pr_analysis(
    db: Session,
    pr: PullRequest,
    result: dict,
    model_name: str,
    conversation_id: UUID | None,
    evidence_counts: dict,
) -> AnalysisResult:
    """把评审结果落库为 AnalysisResult（target_type="pull_request"）。

    返回：已持久化的记录。
    副作用：**内部 commit**，调用方不要再把它包进需要回滚的事务里。
    保留 model_name / evidence_counts 是为了事后能回答「这个结论是基于哪个模型、
        看了多少证据得出的」。
    """
    analysis = AnalysisResult(
        target_type="pull_request",
        target_id=pr.id,
        analysis_type="pr_review",
        input_snapshot={
            "pr_id": str(pr.id),
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


def _public_reasoning_fallback(pr: PullRequest, context: dict, counts: dict) -> list[str]:
    """LLM 无输出时的兜底「执行过程」（数字全部来自真实统计，因此内容依然准确）。"""
    stats = context.get("diff_stats") or {}
    return [
        f"- 我先制定 Review 计划，再读取 PR #{pr.number} 的标题、描述、分支和 {counts['files']} 个变更文件。\n",
        f"- Diff 规模为 +{stats.get('additions', 0)} / -{stats.get('deletions', 0)}，其中敏感文件 {len(stats.get('sensitive_files') or [])} 个。\n",
        f"- 我检索了相关代码 {counts['code']} 条，其中 PR 快照代码 {counts.get('snapshot_code', 0)} 条；项目文档/记忆 {counts['documents']} 条、历史 Issue {counts['issues']} 条和会话证据 {counts['conversation']} 条。\n",
        f"- CI 侧读取到 {counts['failed_ci']} 个失败信号，并结合 {counts['comments']} 条 review comment 判断是否阻塞。\n",
        "- 最后将风险点、阻塞项、测试建议汇总为合入建议：建议合入、修改后合入、暂缓或拒绝。\n",
    ]


async def _stream_public_reasoning(agent: PRReviewAgent, pr: PullRequest, payload: dict, context: dict, counts: dict) -> AsyncIterator[str]:
    """流式产出「公开执行过程」，模型无输出时退化为模板文案。

    入参：agent 复用其 llm 客户端；context_preview 每类只截 3 条
        （公开过程是给人看的摘要，喂全量只会让模型把证据原文抄一遍）。
    返回：异步字符串生成器。
    副作用：调用大模型（流式）。
    """
    # streamed 标记模型是否真的产出过内容，False 才走兜底。
    streamed = False
    user_payload = {
        "pull_request": payload,
        "plan": context.get("plan") or [],
        "evidence_counts": counts,
        "context_preview": {
            "diff_stats": context.get("diff_stats"),
            "pr_snapshot": context.get("pr_snapshot"),
            "code_graph_impact": context.get("code_graph_impact"),
            "code_references": context.get("code_references", [])[:3],
            "project_documents": context.get("project_documents", [])[:3],
            "ci_summary": context.get("ci_summary"),
            "review_comments": context.get("review_comments", [])[:3],
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
    for line in _public_reasoning_fallback(pr, context, counts):
        # 逐行节流：兜底文案是瞬时的，不分段会被一次性刷屏。
        await asyncio.sleep(0.06)
        yield line


async def _pr_analysis_sse(pr_id: UUID, conversation_id: UUID | None, db: Session) -> AsyncIterator[str]:
    """SSE 主流程：按六步计划推进度，最后给出结构化评审结果。

    事件序列：trace(计划) → trace(各步骤) → trace(思考中) → thinking_delta* →
        trace(结论) → final{analysis_id, result}；异常一律转成 error 事件。
    异常处理说明：
        响应头在生成器启动前就已发出，无法再改 HTTP 状态码，
        所以「PR 不存在」「分析失败」都只能作为 SSE error 事件下发。
    副作用：准备 PR 快照（clone + 建索引）、调用大模型、写 AnalysisResult。
    注意：这里用 selectinload 预加载 files / review_comments，
        否则 _pr_payload 遍历时会退化成 N+1 查询。
    """
    pr = db.query(PullRequest).options(selectinload(PullRequest.files), selectinload(PullRequest.review_comments)).filter(PullRequest.id == pr_id).one_or_none()
    if pr is None:
        yield _encode_sse("error", {"message": "未找到 PR"})
        return
    agent = PRReviewAgent()
    payload = _pr_payload(pr)
    plan = _pr_review_plan()
    try:
        # 先把完整计划整条推给前端，让用户知道一共会做几步（后续文案里的 1/6 与之对应）。
        yield _encode_sse("trace", _trace_item("step", "制定 Review 计划", "\n".join(f"{index + 1}. {item}" for index, item in enumerate(plan)), "done"))
        yield _encode_sse("trace", _trace_item("step", "执行 1/6：理解 PR", f"读取 PR #{pr.number} 的标题、描述、分支、作者和变更文件列表。", "running"))
        snapshot = _prepare_pr_snapshot_context(db, pr, enabled=True)
        yield _encode_sse(
            "trace",
            _trace_item(
                "tool",
                "准备 PR 分支快照",
                # 快照可能失败/跳过，这里优先展示 service 给的 message，其次才是状态兜底。
                str(snapshot.get("message") or f"状态：{snapshot.get('status')}；代码图符号：{snapshot.get('indexed_symbols') or 0}。"),
            ),
        )
        _query, context, counts = _build_pr_analysis_context(db, pr, conversation_id)
        _augment_pr_context_with_snapshot(db, pr, _query, context, counts, snapshot)
        # 再写一次 plan：_build 里也放了一份，这里覆盖以保证 context 里就是当前这份计划。
        context["plan"] = plan
        yield _encode_sse("trace", _trace_item("tool", "执行 2/6：拆分 diff 风险", f"扫描 {counts['files']} 个文件，识别敏感文件 {len((context.get('diff_stats') or {}).get('sensitive_files') or [])} 个。"))
        yield _encode_sse("trace", _trace_item("tool", "执行 3/6：读取代码和测试", f"找到 {counts['code']} 条代码线索，其中 PR 快照代码 {counts.get('snapshot_code', 0)} 条；项目文档/记忆线索 {counts['documents']} 条。"))
        yield _encode_sse("trace", _trace_item("tool", "执行 4/6：检查 CI", f"读取最近 CI 信号，失败 run 数量：{counts['failed_ci']}。"))
        yield _encode_sse("trace", _trace_item("tool", "执行 5/6：检索上下文", f"找到 {counts['issues']} 条历史 Issue、{counts['conversation']} 条会话证据、{counts['comments']} 条 review comment。"))
        yield _encode_sse("trace", _trace_item("thinking", "LLM 公开执行过程", "开始综合 PR diff、代码、CI、文档和 review comment。", "running"))
        async for chunk in _stream_public_reasoning(agent, pr, payload, context, counts):
            yield _encode_sse("thinking_delta", {"content": chunk})
        result = await agent.run(payload, context)
        analysis = _persist_pr_analysis(db, pr, result, agent.llm.model, conversation_id, counts)
        yield _encode_sse("trace", _trace_item("result", "执行 6/6：形成合入建议", f"结论：{result.get('merge_recommendation')}；风险点：{len(result.get('risk_points') or [])}；阻塞项：{len(result.get('blocking_issues') or [])}。"))
        yield _encode_sse("final", {"analysis_id": str(analysis.id), "result": result})
    except Exception as exc:
        # 回滚可能残留的未提交写入；错误作为事件下发（抛异常会让前端只看到连接中断）。
        db.rollback()
        yield _encode_sse("error", {"message": f"分析失败：{exc}"})


@router.get("/repos/{repo_id}/pull-requests", response_model=list[PullRequestResponse])
async def list_pull_requests(repo_id: UUID, db: Session = Depends(get_db)) -> list[PullRequest]:
    """GET /api/repos/{repo_id}/pull-requests：PR 列表。

    入参：repo_id（路径）。
    返回：按 PR 编号倒序的完整列表，未分页。
    异常：无（仓库不存在时返回空列表）。
    副作用：只读。
    说明：用 selectinload 一次性预加载 files 和 review_comments——
        列表项要显示「改了哪些文件/有几条评论」，不预加载就会变成 N+1 查询。
    """
    return (
        db.query(PullRequest)
        .options(selectinload(PullRequest.files), selectinload(PullRequest.review_comments))
        .filter(PullRequest.repo_id == repo_id)
        .order_by(PullRequest.number.desc())
        .all()
    )


@router.get("/pull-requests/{pr_id}", response_model=PullRequestResponse)
async def get_pull_request(pr_id: UUID, db: Session = Depends(get_db)) -> PullRequest:
    """GET /api/pull-requests/{pr_id}：PR 详情（含变更文件与评论）。

    入参：pr_id。
    返回：PullRequestResponse。
    异常：不存在 → 404。
    副作用：只读。
    """
    pr = db.query(PullRequest).options(selectinload(PullRequest.files), selectinload(PullRequest.review_comments)).filter(PullRequest.id == pr_id).one_or_none()
    if pr is None:
        raise HTTPException(status_code=404, detail="未找到 PR")
    return pr


@router.post("/pull-requests/{pr_id}/analyze", response_model=PRAnalysisResponse)
async def analyze_pull_request(
    pr_id: UUID,
    conversation_id: UUID | None = Query(default=None),
    use_pr_snapshot: bool = Query(default=True),
    use_worktree: bool | None = Query(default=None),
    db: Session = Depends(get_db),
) -> PRAnalysisResponse:
    """POST /api/pull-requests/{pr_id}/analyze：同步版 PR 评审。

    入参：
        conversation_id: 可选，带上会话上下文作为证据。
        use_pr_snapshot: 是否准备 PR 分支快照（默认开）。
        use_worktree: 旧参数名，非 None 时**优先于** use_pr_snapshot
            （保留它是为了兼容老前端，新代码请用 use_pr_snapshot）。
    返回：PRAnalysisResponse(analysis_id, result)。
    异常：PR 不存在 → 404；大模型/快照失败向上抛（走全局 500）。
    副作用：可能 clone/worktree + 建索引、调用 LLM、写 AnalysisResult。
    与 stream 版逻辑一致，只是不逐段推送进度，适合脚本调用。
    """
    pr = db.query(PullRequest).options(selectinload(PullRequest.files), selectinload(PullRequest.review_comments)).filter(PullRequest.id == pr_id).one_or_none()
    if pr is None:
        raise HTTPException(status_code=404, detail="未找到 PR")
    agent = PRReviewAgent()
    # 兼容处理：老参数 use_worktree 一旦显式传入就以它为准。
    snapshot_enabled = use_pr_snapshot if use_worktree is None else use_worktree
    snapshot = _prepare_pr_snapshot_context(db, pr, snapshot_enabled)
    _query, context, counts = _build_pr_analysis_context(db, pr, conversation_id)
    _augment_pr_context_with_snapshot(db, pr, _query, context, counts, snapshot)
    result = await agent.run(_pr_payload(pr), context)
    analysis = _persist_pr_analysis(db, pr, result, agent.llm.model, conversation_id, counts)
    return PRAnalysisResponse(analysis_id=analysis.id, result=result)


@router.post("/pull-requests/{pr_id}/analyze/stream")
async def analyze_pull_request_stream(
    pr_id: UUID,
    conversation_id: UUID | None = Query(default=None),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """POST /api/pull-requests/{pr_id}/analyze/stream：流式 PR 评审（SSE）。

    入参：pr_id、可选 conversation_id。
    返回：StreamingResponse，media_type=text/event-stream。
    异常：**不抛 4xx**——PR 不存在/分析失败都是 SSE error 事件（响应头已发，改不了状态码）。
    副作用：同 /analyze；另外会一直占用一个数据库连接直到流结束。
    """
    return StreamingResponse(_pr_analysis_sse(pr_id, conversation_id, db), media_type="text/event-stream")


@router.post("/pull-requests/{pr_id}/snapshot")
async def prepare_pull_request_snapshot(pr_id: UUID, db: Session = Depends(get_db)) -> dict:
    """POST /api/pull-requests/{pr_id}/snapshot：单独准备 PR 分支快照。

    入参：pr_id。
    返回：快照状态字典（status / snapshot_path / indexed_symbols ...）。
    异常：PR 不存在 → 404（快照本身失败不报错，以 status 体现）。
    副作用：checkout PR 分支 + 建代码索引，比较重。前端「提前预热」时用，
        这样之后点评审就不用再等快照。
    """
    pr = db.query(PullRequest).options(selectinload(PullRequest.files)).filter(PullRequest.id == pr_id).one_or_none()
    if pr is None:
        raise HTTPException(status_code=404, detail="未找到 PR")
    return _prepare_pr_snapshot_context(db, pr, enabled=True)


@router.post("/pull-requests/{pr_id}/worktree")
async def prepare_pull_request_worktree(pr_id: UUID, db: Session = Depends(get_db)) -> dict:
    """POST /api/pull-requests/{pr_id}/worktree：同上，历史别名接口。

    保留原因：早期前端调用的是 /worktree，改路径会破坏旧客户端，所以两个路由并存、
        实现完全一致。新代码建议用 /snapshot。
    传入参数、返回、异常、副作用均与 /snapshot 相同。
    """
    pr = db.query(PullRequest).options(selectinload(PullRequest.files)).filter(PullRequest.id == pr_id).one_or_none()
    if pr is None:
        raise HTTPException(status_code=404, detail="未找到 PR")
    return _prepare_pr_snapshot_context(db, pr, enabled=True)


@router.post("/repos/{repo_id}/pr-snapshots/cleanup")
async def cleanup_pull_request_snapshots(
    repo_id: UUID,
    max_age_hours: int = Query(default=168, ge=1, le=24 * 90),
    db: Session = Depends(get_db),
) -> dict:
    """POST /api/repos/{repo_id}/pr-snapshots/cleanup：清理过期的 PR 快照。

    入参：
        max_age_hours: 超过多少小时算过期，默认 168（7 天），范围 1 小时 ~ 90 天。
    返回：清理结果 dict（由 cleanup_pr_snapshots 决定字段）。
    异常：仓库不存在 → 404。
    副作用：删除磁盘上的快照目录。快照是「可重建的缓存」，删掉只影响下次评审要重新准备。
    """
    repo = db.get(Repository, repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到仓库")
    return cleanup_pr_snapshots(db, repo, max_age_hours=max_age_hours)


@router.post("/pull-requests/{pr_id}/review-checklist")
async def review_checklist(pr_id: UUID, db: Session = Depends(get_db)) -> dict:
    """POST /api/pull-requests/{pr_id}/review-checklist：只要评审检查清单。

    入参：pr_id。
    返回：{"pr_id", "checklist"}，checklist 取自完整评审结果的 review_checklist 字段。
    异常：PR 不存在 → 404；分析失败向上抛。
    副作用：**这是一次完整评审**——内部直接调用 analyze_pull_request，
        同样会准备快照、调 LLM、落 AnalysisResult，成本与 /analyze 完全相同。
        如果只需要清单，请注意它并不是一个「廉价接口」。
    """
    response = await analyze_pull_request(pr_id, db=db)
    return {"pr_id": str(pr_id), "checklist": response.result.review_checklist}
