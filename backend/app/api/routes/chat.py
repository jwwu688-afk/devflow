"""对话（Chat）主入口，挂载前缀 /api/chat。

这是整个产品的核心：用户在聊天框里提问，系统决定要调哪些工具（查历史/看代码/分析
Issue/PR/CI/出周报/走写操作审批），再把结果流式吐回前端。本文件负责四件事：
    1. 组装上下文：拼「客户端上下文 + Skill 运行时 + 会话摘要 + 证据」四层；
    2. 暴露工具：_build_conversation_tools 里每个函数就是 ChatAgent 能调的一个工具；
    3. 管理会话：读写消息、压缩上下文、封存本轮、更新会话时间戳；
    4. 对外接口：/ 同步、/stream 流式、/plan 出计划、/plan/execute/stream 执行计划、
       /history 历史、/recommendations 推荐追问、/workflow-runs 工作流记录。

两条重要设计：
    - 「先计划后执行」：/plan 只让 PlannerAgent 产出 WorkflowSpec 给人确认，
      /plan/execute/stream 才真正跑，避免 AI 擅自执行多步操作。
    - 「写操作只出草稿」：涉及发评论/建 issue 的请求一律生成 ActionDraft 等人工确认，
      绝不在这里直接改 GitHub。

另外 workspace.* 这几个工具让 Agent 能读本机源码目录，但只允许白名单后缀、
禁止敏感文件（.env 等）、限制在工作区内——这条安全边界由 _workspace_path 保证。
"""

import uuid
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import re
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session, selectinload

from app.core.config import settings
from app.db.models import ActionDraft, AgentRun, AgentTaskRun, AgentWorkflowRun, ChatMessage, Conversation, Issue, PullRequest, Repository, WorkflowRun
from app.db.session import get_db
from app.schemas.chat import ChatHistoryMessage, ChatPlanExecuteRequest, ChatRequest, ChatResponse, PromptRecommendationRequest, PromptRecommendationResponse
from app.services.agents.ci_debug_agent import CIDebugAgent
from app.services.agents.chat_agent import ChatAgent, encode_sse
from app.services.agents.issue_agent import IssueAgent
from app.services.agents.planner_agent import PlannerAgent
from app.services.agents.pr_review_agent import PRReviewAgent
from app.services.agents.recommendation_agent import RecommendationAgent
from app.services.agents.report_agent import ReportAgent
from app.services.agents.safety_agent import SafetyAgent
from app.services.agents.workflow_orchestrator import WorkflowOrchestrator
from app.services.chat_memory import ContextAssembler, EvidenceStore, MessageStore, ProgressiveContextManager, SessionSealer
from app.services.code_search import query_terms, search_checkout_code
from app.services.conversations import ensure_conversation, touch_conversation
from app.services.memory_hub import record_recall_event
from app.services.mcp_client import StdioMcpClient
from app.services.rag.retrieval import search_similar_documents
from app.services.rag.vector_store import add_document
from app.schemas.workflow import WorkflowSpec, WorkflowTask, WorkflowTaskResult
from app.services.skills import SkillRegistry, get_skill_registry
from app.services.permissions import ensure_demo_user, write_audit_log

router = APIRouter()

# parents[4]：routes → api → app → backend → 项目根（DevFlow-AI 自身）。
# 没有选中任何仓库时，workspace.* 工具就以这个目录作为工作区——
# 相当于让 AI 能读「DevFlow AI 自己的源码」，用于自我问答/演示。
PROJECT_ROOT = Path(__file__).resolve().parents[4]
# parents[3] 是 backend 目录，用作相对路径（如 repo_checkout_dir）的解析基准。
BACKEND_ROOT = Path(__file__).resolve().parents[3]
# 遍历工作区时跳过的目录：都是依赖/构建产物/版本库内部目录，
# 既没有分析价值，又会把遍历拖到几十万文件级别。
WORKSPACE_SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".next",
    ".nuxt",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    "coverage",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}
# 黑名单：这些文件里是密钥，任何情况下都不允许 Agent 读到。
WORKSPACE_SECRET_FILENAMES = {".env", ".env.local", ".env.production", ".npmrc", ".pypirc"}
# 无扩展名但需要当文本读的常见工程文件（Dockerfile/Makefile 等）。
WORKSPACE_TEXT_FILENAMES = {"Dockerfile", "Makefile", "README", "LICENSE", "package.json", "requirements.txt", "pyproject.toml"}
# 允许读取的文本类扩展名白名单（默认拒绝，避免误读二进制/大文件）。
WORKSPACE_TEXT_EXTENSIONS = {
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",
    ".json",
    ".md",
    ".mdx",
    ".txt",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".sql",
    ".sh",
    ".ps1",
    ".bat",
    ".cmd",
    ".html",
    ".css",
    ".scss",
}


@router.get("/workflow-runs")
async def list_workflow_runs(
    repo_id: uuid.UUID | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    """GET /api/chat/workflow-runs：Agent 工作流执行记录列表。

    入参：repo_id 可选（不传则跨仓库）；limit 1~100，默认 20。
    返回：每条 workflow 含 spec/observation/final_answer/metrics 以及它的每个子任务。
    异常：无。
    副作用：只读。
    实现说明：
        子任务用「一次 IN 查询 + 内存分组」而不是逐条查（避免 N+1），
        workflow_ids 为空时直接给 []，避免生成 IN () 的空条件。
    """
    query = db.query(AgentWorkflowRun)
    if repo_id is not None:
        query = query.filter(AgentWorkflowRun.repo_id == repo_id)
    rows = query.order_by(AgentWorkflowRun.created_at.desc()).limit(limit).all()
    workflow_ids = [row.id for row in rows]
    tasks = (
        db.query(AgentTaskRun)
        .filter(AgentTaskRun.workflow_id.in_(workflow_ids))
        .order_by(AgentTaskRun.started_at.asc(), AgentTaskRun.task_id.asc())
        .all()
        if workflow_ids
        else []
    )
    tasks_by_workflow: dict[uuid.UUID, list[AgentTaskRun]] = {}
    for task in tasks:
        tasks_by_workflow.setdefault(task.workflow_id, []).append(task)
    return [
        {
            "id": str(row.id),
            "repo_id": str(row.repo_id) if row.repo_id else None,
            "conversation_id": str(row.conversation_id) if row.conversation_id else None,
            "goal": row.goal,
            "status": row.status,
            "spec": row.spec_json or {},
            "observation": row.observation_json or {},
            "final_answer": row.final_answer,
            "metrics": row.metrics or {},
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "completed_at": row.completed_at.isoformat() if row.completed_at else None,
            "tasks": [
                {
                    "id": str(task.id),
                    "task_id": task.task_id,
                    "agent_name": task.agent_name,
                    "task_type": task.task_type,
                    "status": task.status,
                    # summary 藏在 result_json 里，这里提出来给前端列表直接显示。
                    "summary": str((task.result_json or {}).get("summary") or ""),
                    "error": task.error,
                    "started_at": task.started_at.isoformat() if task.started_at else None,
                    "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                }
                for task in tasks_by_workflow.get(row.id, [])
            ],
        }
        for row in rows
    ]


@router.get("/history", response_model=list[ChatHistoryMessage])
async def chat_history(
    repo_id: uuid.UUID = Query(...),
    conversation_id: uuid.UUID | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    before_message_id: uuid.UUID | None = Query(default=None),
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    """GET /api/chat/history?repo_id=...：拉取会话历史消息。

    入参：
        repo_id: 必填。
        conversation_id: 可选；不传则落到该仓库的默认会话。
        limit: 1~500，默认 100。
        before_message_id: 游标分页——只看这条消息之前的记录（向上翻页用）。
    返回：消息列表，含工具调用、Agent 事件、压缩统计等元信息。
    异常：仓库不存在 → 404。
    副作用：**会写库**——ensure_conversation 可能创建默认会话，所以下面有 commit；
        另外 MessageStore.timeline 可能触发历史数据回填。
    说明：过滤掉 user/assistant 之外的角色（如 system/tool），
        它们属于内部过程，不该出现在聊天界面。
    """
    _repo_or_404(db, repo_id)
    conversation = ensure_conversation(db, repo_id, conversation_id)
    conversation_id = conversation.id
    db.commit()
    rows = [
        item
        for item in MessageStore(db).timeline(
            repo_id,
            conversation_id,
            limit=limit,
            before_message_id=before_message_id,
        )
        if item.role in {"user", "assistant"}
    ]
    return [
        {
            "id": str(item.id),
            "conversation_id": str(item.conversation_id) if item.conversation_id else None,
            "role": item.role,
            "content": item.content,
            "route": item.route,
            "tool_calls": item.tool_calls or [],
            # 下面这些都在 meta(JSON) 里，缺省给空值，前端免判空。
            "agent_events": (item.meta or {}).get("agent_events", []),
            "compression_stats": (item.meta or {}).get("compression_stats", {}),
            "progressive_compaction": (item.meta or {}).get("progressive_compaction", {}),
            "model_usage": (item.meta or {}).get("model_usage", []),
            "context_diagnostics": (item.meta or {}).get("context_diagnostics", {}),
            "created_at": item.created_at.isoformat() if item.created_at else None,
        }
        for item in rows
    ]


def _repo_or_404(db: Session, repo_id: uuid.UUID) -> Repository:
    """取仓库，不存在则 404（本文件几乎所有接口的第一道校验）。"""
    repo = db.get(Repository, repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到指定仓库。")
    return repo


def _client_context(value: dict[str, Any] | None) -> dict[str, Any]:
    """过滤前端传来的 context，只保留客户端有资格提供的键。

    白名单式排除（黑名单）这里用的是排除法：服务端自己算出来的字段
    （repo_id / 会话 ID / 工具表 / 证据 / 压缩统计 / Skill 注册表）一律不准客户端覆盖，
    否则前端可以伪造 evidence 让模型引用假证据，或塞入自定义 tools 执行任意逻辑。
    """
    server_owned = {
        "repo_id",
        "conversation_id",
        "tools",
        "server_context",
        "server_messages",
        "evidence",
        "compression_stats",
        "progressive_compaction",
        "registered_skills",
        "skill_description_by_tool",
    }
    return {key: item for key, item in (value or {}).items() if key not in server_owned}


def _item_time(item: Any) -> datetime:
    """取「最近活动时间」用于排序：优先 updated_at，其次 created_at。

    都没有时返回 datetime.min 兜底。注意这里可能返回 naive datetime，
    排序时只互相比较、不与带时区的时间比，所以不会触发 TypeError。
    """
    return item.updated_at or item.created_at or datetime.min


def _join_items(items: list[str], empty_text: str) -> str:
    """把列表拼成「；」分隔的一行；空列表时返回给定的兜底文案。"""
    return "; ".join(items) if items else empty_text


def _source_type_from_message(message: str) -> str | None:
    """从用户话里猜要检索哪类数据（issue / pull_request / workflow_run / knowledge_file）。

    为什么用关键词猜：检索接口的 source_type 是硬过滤，猜错就一条都搜不到；
    但多半不猜也能用（返回 None 表示不限制）。这里只做「明显意图」的粗判，
    顺序从具体到宽泛，命中即返回。
    注意 "pr" 是子串匹配，"spring"、"process" 之类也会命中，属于已知误判。
    """
    lowered = message.lower()
    if "issue" in lowered:
        return "issue"
    if "pr" in lowered or "pull request" in lowered or "diff" in lowered:
        return "pull_request"
    if "ci" in lowered or "workflow" in lowered:
        return "workflow_run"
    if any(token in lowered for token in ["knowledge", "知识库", "上传资料", "上传文档"]):
        return "knowledge_file"
    return None


def _risky_action(message: str) -> str | None:
    """判断用户是不是在要求「会写入外部系统」的动作，返回动作类型或 None。

    返回非 None 就意味着必须走安全草稿 + 人工确认，不能直接执行。
    这也是为什么判断偏保守：宁可多拦一次（多确认一步），也不要漏拦。
    """
    lowered = message.lower()
    if any(token in lowered for token in ["comment", "reply", "post"]):
        return "comment"
    if any(token in lowered for token in ["create issue", "new issue"]):
        return "create_issue"
    if any(token in lowered for token in ["send", "send_report"]):
        return "send_report"
    if any(token in lowered for token in ["close issue", "label", "write"]):
        return "write_back"
    return None


def _date_window_from_context(context: dict[str, Any]) -> tuple[datetime, datetime]:
    """从上下文里解析出统计的时间窗（左闭右闭）。

    入参：context 里可能有 time_range（如 "7d" / "30d"）、start_date、end_date。
    返回：(start_at, end_at)，均为 UTC aware。
    规则：
        - 默认看最近 7 天；time_range 只认「N d」这种格式，解析失败就按 7 天；
        - start_at 减的是 days-1，这样 "1d" 就是「今天一天」而不是「今天和昨天」；
        - 显式传了 start_date / end_date 时以它们为准（可以只覆盖其中一端）；
        - end_date 会被拉到当天 23:59:59，否则「到今天」会漏掉今天的记录。
    注意：naive 的 fromisoformat 结果被强行打上 UTC 时区，这假定前端传的就是 UTC 日期。
    """
    end_at = datetime.now(timezone.utc)
    time_range = str(context.get("time_range") or "7d").lower()
    days = int(time_range[:-1]) if time_range.endswith("d") and time_range[:-1].isdigit() else 7
    start_at = end_at - timedelta(days=max(1, days) - 1)
    if context.get("start_date"):
        start_at = datetime.fromisoformat(str(context["start_date"])).replace(tzinfo=timezone.utc)
    if context.get("end_date"):
        end_at = datetime.fromisoformat(str(context["end_date"])).replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)
    return start_at, end_at


def _within(value: datetime | None, start_at: datetime, end_at: datetime) -> bool:
    """闭区间判定，并把 naive 时间按 UTC 补齐（SQLite 读回来没有时区）。"""
    if not value:
        return False
    current = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return start_at <= current <= end_at


def _repo_payload(repo_id: uuid.UUID, db: Session) -> tuple[list[Issue], list[PullRequest], list[WorkflowRun]]:
    """一次性把仓库的 Issue / PR / CI 全量取出来。

    返回：(issues, prs, runs)。
    说明：
        多处工具都要这三份数据，直接全量拉回内存比反复条件查询简单；
        PR 用 selectinload 预加载 files/review_comments，否则后面遍历会 N+1。
        数据量很大时这里会成为瓶颈（已知的简化点）。
    """
    issues = db.query(Issue).filter(Issue.repo_id == repo_id).all()
    prs = (
        db.query(PullRequest)
        .options(selectinload(PullRequest.files), selectinload(PullRequest.review_comments))
        .filter(PullRequest.repo_id == repo_id)
        .all()
    )
    runs = db.query(WorkflowRun).filter(WorkflowRun.repo_id == repo_id).all()
    return issues, prs, runs


def _recommendation_payload(
    repo: Repository | None,
    issues: list[Issue],
    prs: list[PullRequest],
    runs: list[WorkflowRun],
    exclude: list[str],
    limit: int,
) -> dict[str, Any]:
    """组装「推荐追问」Agent 的输入。

    入参：
        exclude: 已经展示/已被用户点过的推荐文案，交给模型去重（避免反复推同一条）。
        limit: 期望条数。
    返回：含仓库、issue、pr、run 摘要的字典。
    说明：这里把 ORM 对象手工摊平成 dict —— 直接把模型对象塞给 Agent 会序列化失败，
        而且会把一堆内部字段（如 github_id）带进 prompt。
    """
    return {
        "repo": {"id": str(repo.id), "full_name": repo.full_name} if repo else None,
        "issues": [
            {
                "id": str(item.id),
                "number": item.number,
                "title": item.title,
                "state": item.state,
                "labels": item.labels or [],
                "assignees": item.assignees or [],
                "created_at": item.created_at,
                "updated_at": item.updated_at,
                "closed_at": item.closed_at,
            }
            for item in issues
        ],
        "pull_requests": [
            {
                "id": str(item.id),
                "number": item.number,
                "title": item.title,
                "state": item.state,
                "author": item.author,
                "created_at": item.created_at,
                "updated_at": item.updated_at,
                "merged_at": item.merged_at,
            }
            for item in prs
        ],
        "workflow_runs": [
            {
                "id": str(item.id),
                "name": item.name,
                "status": item.status,
                "conclusion": item.conclusion,
                "created_at": item.created_at,
                "updated_at": item.updated_at,
            }
            for item in runs
        ],
        "exclude": exclude,
        "limit": limit,
    }


def _repo_health_answer(repo: Repository, issues: list[Issue], prs: list[PullRequest], runs: list[WorkflowRun]) -> str:
    """用模板拼一段「仓库健康度」回答（不调大模型）。

    为什么不用 LLM：这类纯统计问题用模板更准、更快、免费，也不会编造数字。
    返回：Markdown 文本，含 issue/pr 概况、最近几条标题和 CI 失败数。
    """
    open_issues = [item for item in issues if item.state == "open"]
    open_prs = [item for item in prs if item.state == "open"]
    merged_prs = [item for item in prs if item.merged_at]
    failed_runs = [item for item in runs if item.conclusion == "failure"]
    # 各取最近 3 条标题：再多回答就太长，用户也不会在聊天框里读完。
    latest_issues = sorted(open_issues, key=_item_time, reverse=True)[:3]
    latest_prs = sorted(prs, key=_item_time, reverse=True)[:3]
    issue_titles = _join_items([f"#{item.number} {item.title}" for item in latest_issues], "没有打开的 Issue")
    pr_titles = _join_items([f"#{item.number} {item.title}" for item in latest_prs], "没有 PR 记录")
    ci_text = f"已同步 {len(failed_runs)} 次失败 CI 运行" if failed_runs else "还没有同步到失败 CI 运行"
    return (
        f"当前仓库：{repo.full_name}\n\n"
        f"- Issues：{len(open_issues)} 个打开。近期关注：{issue_titles}。\n"
        f"- PRs：{len(open_prs)} 个打开，{len(merged_prs)} 个已合并。近期 PR：{pr_titles}。\n"
        f"- CI：{ci_text}。\n\n"
        "建议下一步：先看最新打开的 Issue，或检查最新 PR 风险；如果同步后出现失败 CI，可以让我继续排障。"
    )


def _semantic_search(repo: Repository, message: str, context: dict[str, Any], db: Session) -> tuple[str, list[dict[str, str]], list[dict[str, Any]]]:
    """跑一次向量检索并把结果渲染成人读的文本。

    入参：
        message: 检索语句（通常就是用户原话，或工具改写后的 query）。
        context: 可能带 source_type / limit / 各类元数据过滤。
    返回：(回答文本, 引用列表, 原始结果列表)——
        引用给前端渲染来源卡片，原始结果用于记录 recall 事件和评测。
    说明：
        过滤条件同样用「假值即忽略」的方式收集，前端传空串等于不过滤。
    """
    filters = {
        "label": context.get("label"),
        "path": context.get("path"),
        "module": context.get("module"),
        "state": context.get("state"),
        "start_date": context.get("start_date"),
        "end_date": context.get("end_date"),
    }
    results = search_similar_documents(
        db,
        repo.id,
        message,
        # source_type 优先用调用方显式指定的，其次才从话里猜。
        source_type=context.get("source_type") or _source_type_from_message(message),
        limit=int(context.get("limit", 5)),
        metadata_filters={key: value for key, value in filters.items() if value},
    )
    if not results:
        return "没有找到足够相似的历史记录。可以尝试放宽标签、路径或时间过滤条件。", [], []
    lines = ["找到以下相关历史记录："]
    citations: list[dict[str, str]] = []
    for index, item in enumerate(results, start=1):
        meta = item.get("metadata") or {}
        number = meta.get("number")
        # 有编号就带 # 前缀（issue/PR 的展示习惯），没有就省略。
        prefix = f"#{number} " if number else ""
        lines.append(f"- [{index}] {item['source_type']} {prefix}{item['title']}：相似度 {item['score']}")
        snippet = str(item.get("snippet") or "").strip()
        if snippet:
            lines.append(f"  证据：{snippet}")
        citations.append({"type": item["source_type"], "id": item["source_id"], "title": item["title"]})
    return "\n".join(lines), citations, results


def _checkout_root() -> Path:
    """本系统管理 clone 的根目录（相对路径按 backend 目录解析，再 resolve 成绝对路径）。"""
    root = Path(settings.repo_checkout_dir)
    if not root.is_absolute():
        root = BACKEND_ROOT / root
    return root.resolve()


def _repo_checkout_path(repo: Repository) -> Path:
    """找出某个仓库在磁盘上的源码目录。

    查找顺序：
        1. repo.local_path（用户接入时指定的本地仓库）——存在就直接用；
        2. {checkout_root}/{repo.id}-{owner__name}——本系统 clone 的标准位置；
        3. 目录里任意「后缀匹配 owner__name」的旧 checkout（取 mtime 最新的）。
    返回：路径（可能不存在，由调用方判断）。
    第 3 步的原因：仓库记录删除后重建会拿到新 UUID，但磁盘上的 clone 还是旧的 UUID 前缀，
        不兜这一层的话「工作区工具」会突然读不到源码。
    """
    local_path = getattr(repo, "local_path", None)
    if local_path:
        path = Path(os.path.expandvars(os.path.expanduser(str(local_path))))
        if not path.is_absolute():
            path = BACKEND_ROOT / path
        resolved = path.resolve()
        if resolved.exists():
            return resolved

    # 目录名里的 / 换成 __，避免 owner/name 被当成两级目录。
    safe_name = repo.full_name.replace("/", "__").replace("\\", "__")
    checkout_root = _checkout_root()
    expected = (checkout_root / f"{repo.id}-{safe_name}").resolve()
    if expected.exists():
        return expected

    # 仓库记录可能被删除后重建，而本地 clone 仍保留旧 UUID。
    # 复用同名 checkout，避免工作区工具静默失去源码访问能力。
    candidates = [
        path
        for path in checkout_root.glob(f"*-{safe_name}")
        if path.is_dir() and path.resolve() != expected
    ]
    if candidates:
        # 同名的可能有多个，取最近修改的（通常就是当前在用的那个）。
        candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        return candidates[0].resolve()
    return expected


def _workspace_root(repo: Repository | None) -> Path:
    """确定「工作区」根目录：选了仓库就用仓库源码，没选就用本项目源码。

    异常：仓库已记录但代码还没 clone 到本地时抛 ValueError（文案直接给用户看）。
    说明：这个路径是所有 workspace.* 工具的边界基准，_workspace_path 会据此做越界检查。
    """
    if repo is None:
        return PROJECT_ROOT
    checkout_path = _repo_checkout_path(repo)
    if not checkout_path.exists():
        raise ValueError(
            f"{repo.full_name} 的仓库代码还没有克隆到本地。"
            "读取源码前，请先同步仓库或重新连接仓库。"
        )
    return checkout_path


def _workspace_relative(path: Path, workspace_root: Path) -> str:
    """转成相对工作区的 posix 路径（统一用 / 分隔，避免 Windows 反斜杠出现在引用里）。"""
    return path.relative_to(workspace_root).as_posix()


def _workspace_path(workspace_root: Path, raw_path: str | None = None) -> Path:
    """把用户/模型给的路径解析成工作区内的绝对路径，并做安全校验。

    入参：raw_path 可以是相对路径、"/foo"、甚至 "C:\\xxx"（绝对路径会被剥离盘符后再拼接）。
    返回：解析后的绝对 Path。
    异常：越界、命中跳过目录、指向敏感文件时抛 ValueError（文案直接反馈给用户）。
    校验三道关（顺序很重要，先 resolve 再判断）：
        1. resolve 后必须仍在 workspace_root 下——挡住 ../.. 逃逸和符号链接逃逸；
        2. 路径中任何一段命中 WORKSPACE_SKIP_DIRS 就拒绝（node_modules 等）；
        3. 文件名命中敏感名单直接拒绝（.env 之类）。
    """
    normalized_path = str(raw_path or ".").strip()
    # 各种「当前目录」写法统一成 "."，后面再拼到根目录上。
    if normalized_path in {"", ".", "./", "/", "\\"}:
        normalized_path = "."
    elif not re.match(r"^[a-zA-Z]:", normalized_path):
        # 非盘符开头的路径，去掉前导斜杠，强制当相对路径处理——
        # 否则 "/etc/passwd" 会被 Path 当成绝对路径，直接越过工作区。
        normalized_path = normalized_path.lstrip("/\\") or "."
    candidate = (workspace_root / normalized_path).resolve()
    try:
        candidate.relative_to(workspace_root)
    except ValueError as exc:
        raise ValueError("路径位于项目工作区之外。") from exc
    if any(part in WORKSPACE_SKIP_DIRS for part in candidate.relative_to(workspace_root).parts):
        raise ValueError("路径位于被跳过的工作区目录中。")
    if candidate.name in WORKSPACE_SECRET_FILENAMES:
        raise ValueError("不会向对话 Agent 暴露敏感配置文件。")
    return candidate


def _is_workspace_text(path: Path) -> bool:
    """判断文件是否允许当作文本读给 Agent。

    规则：敏感文件、日志/数据库文件一律拒绝；其余按「无扩展名白名单 + 扩展名白名单」放行。
    Dockerfile 额外用 endswith 兜住（有人的文件名是 dockerfile.dev 之类）。
    """
    if path.name in WORKSPACE_SECRET_FILENAMES or path.suffix.lower() in {".log", ".db", ".sqlite", ".sqlite3"}:
        return False
    return path.name in WORKSPACE_TEXT_FILENAMES or path.suffix.lower() in WORKSPACE_TEXT_EXTENSIONS or path.name.lower().endswith("dockerfile")


def _iter_workspace_files(root: Path, workspace_root: Path, max_files: int = 1000):
    """惰性遍历工作区里的可读文本文件。

    入参：root 为遍历起点（可以是单个文件）；max_files 上限（默认 1000）。
    返回：生成器，逐个产出 Path。
    说明：
        1. 传入的是文件就直接产出（方便「针对某个文件」的调用）；
        2. 用 rglob("*") 递归；跳过目录是靠「路径任一段命中 SKIP_DIRS」实现的，
           而不是 os.walk 剪枝，代价是会先遍历到再过滤（性能上不是最优，够用即可）；
        3. max_files 是硬上限：给 Agent 的输出必须有界，否则大仓库会把上下文撑爆。
    """
    if root.is_file():
        yield root
        return
    count = 0
    for path in root.rglob("*"):
        if count >= max_files:
            break
        if not path.is_file():
            continue
        relative_parts = path.relative_to(workspace_root).parts
        if any(part in WORKSPACE_SKIP_DIRS for part in relative_parts):
            continue
        if not _is_workspace_text(path):
            continue
        count += 1
        yield path


def _read_workspace_text(path: Path, max_bytes: int = 250_000) -> str:
    """读取工作区文本文件，带大小/二进制/编码三重保护。

    入参：max_bytes 上限（默认 250KB）。
    返回：解码后的文本。
    异常：文件不存在、不是可读文本、过大、疑似二进制、无法解码 → ValueError。
    说明：
        1. 先看前 4096 字节有没有 \\x00 判断二进制——比 is_text_file 之类的库更轻量；
        2. 编码按 utf-8-sig → utf-8 → gb18030 → latin-1 依次尝试：
           gb18030 覆盖中文 Windows 的老文件，latin-1 永不失败（兜底保证总能返回）。
    """
    if not path.exists() or not path.is_file():
        raise ValueError("未找到工作区文件。")
    if not _is_workspace_text(path):
        raise ValueError("只能读取非敏感的文本工作区文件。")
    data = path.read_bytes()
    if len(data) > max_bytes or b"\x00" in data[:4096]:
        raise ValueError("文件过大，或看起来是二进制文件。")
    for encoding in ["utf-8-sig", "utf-8", "gb18030", "latin-1"]:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("无法将工作区文件解码为文本。")


def _draft_for_action(action: str, message: str, repo: Repository, issues: list[Issue], prs: list[PullRequest], runs: list[WorkflowRun]) -> tuple[str, list[dict[str, str]]]:
    """按动作类型生成「写操作草稿」正文（Markdown）。

    入参：action 由 _risky_action 得出；其余为仓库当前数据，用于填数字。
    返回：(草稿正文, 引用列表)。引用里始终包含仓库本身，便于前端展示来源。
    说明：
        这些草稿只是给人看的模板，不执行任何写操作；真正的执行在 action_drafts 的确认接口里。
        目标对象用「最近更新的那个 PR/issue」兜底，因为聊天里没指明具体目标。
    """
    citations: list[dict[str, str]] = [{"type": "repository", "id": str(repo.id), "title": repo.full_name}]
    latest_issue = sorted(issues, key=_item_time, reverse=True)[0] if issues else None
    latest_pr = sorted(prs, key=_item_time, reverse=True)[0] if prs else None
    failed_runs = [run for run in runs if run.conclusion == "failure"]
    if action == "create_issue":
        # 标题取用户原话前 120 字：够表达意图，又不会被一整段长文撑破标题栏。
        return f"## Issue 草稿\n标题：{message.strip()[:120]}\n\n### 背景\n请补充影响范围、复现步骤和预期行为。\n\n### 验收标准\n- 预期行为清晰\n- 包含验证步骤\n", citations
    if action == "comment" and latest_pr:
        citations.append({"type": "pull_request", "id": str(latest_pr.id), "title": latest_pr.title})
        return f"## PR 评论草稿\n针对 PR #{latest_pr.number}：{latest_pr.title}\n\n合并前请确认测试覆盖、未解决的 Review 评论，以及是否存在高风险文件。\n", citations
    if action == "comment" and latest_issue:
        citations.append({"type": "issue", "id": str(latest_issue.id), "title": latest_issue.title})
        return f"## Issue 评论草稿\n针对 Issue #{latest_issue.number}：{latest_issue.title}\n\n请补充复现步骤、影响范围、预期行为，以及当前是否有临时绕过方案。\n", citations
    if action == "send_report":
        draft = (
            "## 报告发送草稿\n"
            f"仓库：{repo.full_name}\n\n"
            f"- 打开的 Issue：{len([issue for issue in issues if issue.state == 'open'])}\n"
            f"- 打开的 PR：{len([pr for pr in prs if pr.state == 'open'])}\n"
            f"- 失败 CI 运行：{len(failed_runs)}\n\n"
            "发送前请确认收件人、日期范围和敏感日志内容。\n"
        )
        return draft, citations
    return f"## 写操作草稿\n这可能会写入 GitHub 或外部渠道。MVP 模式只生成动作草稿。\n\n用户请求：{message.strip()}\n", citations


def _safe_confidence(value: Any, default: float) -> float:
    """把模型给的置信度夹到 [0,1]；类型不对就用默认值（不让脏数据污染排序/门禁）。"""
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _skipped_workflow_result(task: WorkflowTask, summary: str) -> WorkflowTaskResult:
    """构造「跳过」的任务结果：没有可用数据源时用，而不是报错。

    置信度固定 0：跳过意味着没有依据，前端据此不把它算进结论权重。
    """
    return WorkflowTaskResult(
        task_id=task.id,
        agent_name=task.agent_name,
        task_type=task.task_type,
        status="skipped",
        summary=summary,
        confidence=0.0,
    )


def _agent_context(context: dict[str, Any]) -> dict[str, Any]:
    """剔除内部字段后的上下文副本。

    tools 是可调用对象、_skill_registry 是注册表实例，
    直接传给 Agent 会导致序列化失败或把内部实现暴露进 prompt，必须剥掉。
    """
    return {key: value for key, value in context.items() if key not in {"tools", "_skill_registry"}}


class WorkflowTargetResolutionError(ValueError):
    """工作流目标解析失败：显式指定的 issue/PR/CI 编号无法安全定位时抛出。

    继承 ValueError 是为了让调用方既能按具体类型捕获（chat_plan 里转 422），
    也能被通用的 except ValueError 兜住。
    设计意图：**宁可报错也不回退到「最近一个对象」**——用户明确说了 #123，
        却静默分析了另一个 PR，比直接失败危险得多。
    """


# 从自然语言里抽「目标编号」的正则表：每个类型都支持中英文两种说法。
# 前后的 (?<![A-Za-z0-9_]) / (?![A-Za-z0-9_]) 是单词边界，防止 "printer12" 被当成 "pr 12"。
_WORKFLOW_TARGET_PATTERNS: dict[str, tuple[str, ...]] = {
    "issue": (
        r"(?<![A-Za-z0-9_])issue(?![A-Za-z0-9_])\s*(?:#|＃|no\.?|number|id)?\s*(\d+)",
        r"(?:工单|议题)\s*(?:#|＃|编号)?\s*(\d+)",
    ),
    "pull_request": (
        r"(?<![A-Za-z0-9_])(?:pr|pull[\s_-]*request)(?![A-Za-z0-9_])\s*(?:#|＃|no\.?|number|id)?\s*(\d+)",
        r"合并请求\s*(?:#|＃|编号)?\s*(\d+)",
    ),
    "workflow_run": (
        r"(?<![A-Za-z0-9_])(?:ci|workflow[\s_-]*run)(?![A-Za-z0-9_])\s*(?:#|＃|no\.?|number|id)?\s*(\d+)",
        r"(?<![A-Za-z0-9_])run(?![A-Za-z0-9_])\s*(?:#|＃|id)\s*(\d+)",
        r"(?:CI\s*运行|工作流运行|流水线)\s*(?:#|＃|编号|ID)?\s*(\d+)",
    ),
}

# 报错文案里用的中文名称。
_WORKFLOW_TARGET_LABELS = {
    "issue": "Issue",
    "pull_request": "PR",
    "workflow_run": "CI/Workflow Run",
}


def _requested_workflow_target_number(message: str, target_type: str) -> int | None:
    """从消息里抽出指定类型的编号；出现多个不同编号就报错。

    入参：target_type 必须是 _WORKFLOW_TARGET_PATTERNS 的键。
    返回：唯一编号，或 None（没提）。
    异常：提到多个不同编号 → WorkflowTargetResolutionError。
    说明：用 set 去重——「PR #12 ... 这个 PR #12」重复提到同一个编号是正常的，
        只有出现两个不同编号才算歧义（工作流一次只能为一个目标建立 claim）。
    """
    numbers = {
        int(match.group(1))
        for pattern in _WORKFLOW_TARGET_PATTERNS[target_type]
        for match in re.finditer(pattern, message, flags=re.IGNORECASE)
    }
    if len(numbers) > 1:
        references = "、".join(f"#{number}" for number in sorted(numbers))
        raise WorkflowTargetResolutionError(
            f"一次工作流只能为一个 {_WORKFLOW_TARGET_LABELS[target_type]} 建立明确 claim；检测到多个目标：{references}。"
        )
    return next(iter(numbers), None)


def _target_by_number(
    items: list[Any],
    number: int | None,
    *,
    target_type: str,
    number_attribute: str,
) -> Any | None:
    """按编号在列表里找目标对象。

    入参：
        number: 用户显式说的编号；None 表示没指定，直接返回 None。
        number_attribute: 从对象上取编号的属性名——issue/pr 用 "number"，
            workflow run 用 "github_run_id"（CI 的编号是 GitHub 的 run id，不是 PR 号）。
    返回：命中的对象或 None。
    异常：指定了编号但仓库里没有 → WorkflowTargetResolutionError（**不回退到最新对象**）。
    """
    if number is None:
        return None
    target = next((item for item in items if getattr(item, number_attribute, None) == number), None)
    if target is None:
        raise WorkflowTargetResolutionError(
            f"当前仓库没有同步 {_WORKFLOW_TARGET_LABELS[target_type]} #{number}，已停止生成计划；不会回退到最新对象。"
        )
    return target


def _build_engineering_workflow_runtime(
    repo: Repository,
    db: Session,
    message: str,
    context: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """构造「工程工作流」的运行环境：上下文变量 + 任务执行函数表。

    入参：
        repo: 目标仓库。
        message: 用户原话（从中解析显式指定的 issue/PR/CI 编号）。
        context: 上游组装好的对话上下文（含 Skill 注册表、证据等）。
    返回：(workflow_context, task_runners)——
        workflow_context 是给 Planner/Orchestrator 的变量表（含 has_issue / has_pr 等开关）；
        task_runners 是 {task_type: async runner}，编排器按 task_type 查表执行。
    异常：用户提到多个编号、或编号在仓库里找不到 → WorkflowTargetResolutionError（向上转 422）。
    副作用：只读数据库 + 可能激活 Skill（有内部缓存）。
    设计要点：
        1. 先把「用户显式指定的目标」解析出来（selected_*），没有才回退到「最近一个」；
        2. 「显式指定」和「回退」用 *_explicit 标记区分，Planner 据此决定计划的严格程度；
        3. 每个 runner 都通过 claimed_target 二次校验任务 claim 指向的对象，
           防止模型编一个不存在的目标 ID 去跑分析。
    """
    issues, prs, runs = _repo_payload(repo.id, db)
    failed_runs = [run for run in runs if run.conclusion == "failure"]
    # 三个编号各自独立解析：用户可能只说 PR 号，也可能三个都不说。
    requested_issue_number = _requested_workflow_target_number(message, "issue")
    requested_pr_number = _requested_workflow_target_number(message, "pull_request")
    requested_run_number = _requested_workflow_target_number(message, "workflow_run")
    # 显式指定却找不到会直接抛错（不回退），所以能走到下一步说明编号都有效。
    selected_issue = _target_by_number(
        issues,
        requested_issue_number,
        target_type="issue",
        number_attribute="number",
    )
    selected_pr = _target_by_number(
        prs,
        requested_pr_number,
        target_type="pull_request",
        number_attribute="number",
    )
    selected_run = _target_by_number(
        runs,
        requested_run_number,
        target_type="workflow_run",
        # CI 的「编号」是 GitHub 的 run id，不是自增序号。
        number_attribute="github_run_id",
    )
    # 没指定时的默认目标：issue/pr 取最近更新的；CI 优先取最近一次失败的（更有分析价值）。
    if selected_issue is None:
        selected_issue = sorted(issues, key=_item_time, reverse=True)[0] if issues else None
    if selected_pr is None:
        selected_pr = sorted(prs, key=_item_time, reverse=True)[0] if prs else None
    if selected_run is None:
        selected_run = sorted(failed_runs, key=_item_time, reverse=True)[0] if failed_runs else None

    def claimed_target(task: WorkflowTask, target_type: str, items: list[Any], fallback: Any | None) -> Any | None:
        """校验任务 claim 指向的目标，并返回对应的实体对象。

        入参：task 计划任务；items 候选集；fallback 为「计划没写 target_id」时的默认对象。
        返回：目标对象或 None（fallback 也为空时）。
        异常：claim 类型不匹配、target_id 不是 UUID、目标不在本仓库 → WorkflowTargetResolutionError。
        为什么这么严：claim 是模型写的，可能凭空编造 ID。这里逐条核对，
            宁可让整个工作流失败，也不去分析一个「看起来像」的对象。
        """
        if task.claim.target_type != target_type:
            raise WorkflowTargetResolutionError(
                f"任务 {task.id} 的 claim 类型为 {task.claim.target_type!r}，预期为 {target_type!r}。"
            )
        if not task.claim.target_id:
            # 计划里没写具体目标，用上游选定的默认对象（通常是「最近一个」）。
            return fallback
        try:
            target_id = uuid.UUID(str(task.claim.target_id))
        except (TypeError, ValueError) as exc:
            raise WorkflowTargetResolutionError(f"任务 {task.id} 的 claim.target_id 不是有效 UUID。") from exc
        target = next((item for item in items if item.id == target_id), None)
        if target is None:
            raise WorkflowTargetResolutionError(
                f"任务 {task.id} 的 claim 目标不存在或不属于仓库 {repo.full_name}；不会回退到最新对象。"
            )
        return target

    safe_context = _agent_context(context)
    skill_registry = context.get("_skill_registry")
    # 按 entrypoint 缓存 Skill 激活结果：同一个 Agent 在一个工作流里可能被调用多次，
    # 每次都重新激活会重复读文件/算 prompt。
    agent_skill_cache: dict[str, dict[str, Any]] = {}

    def agent_skill_context(entrypoint: str) -> dict[str, Any]:
        """取某个 Agent 入口对应的 Skill 运行时上下文（含提示词块与 trace 元数据）。

        返回空 dict 表示没有匹配的 Skill（或注册表不可用），调用方用 **展开即可退化成普通上下文。
        """
        if entrypoint in agent_skill_cache:
            return agent_skill_cache[entrypoint]
        if not isinstance(skill_registry, SkillRegistry):
            agent_skill_cache[entrypoint] = {}
            return {}
        activation = skill_registry.activate_entrypoint(entrypoint)
        if activation is None:
            agent_skill_cache[entrypoint] = {}
            return {}
        payload = activation.model_dump(mode="json")
        runtime = {
            "active_skills": [payload],
            "skill_instructions": activation.prompt_block(),
            "skill_runtime": {
                **activation.trace_metadata(),
                "entrypoint": activation.entrypoint,
                "output_contract": activation.output_contract,
                "safety_level": activation.safety_level,
                "instructions_loaded": True,
            },
        }
        agent_skill_cache[entrypoint] = runtime
        return runtime

    def with_skill_runtime(result: dict[str, Any], entrypoint: str) -> dict[str, Any]:
        """给 Agent 输出挂上 Skill 运行时信息，标记「该入口已走完」。

        validation="entrypoint_completed" 是给前端/评测判断「这个 Agent 的输出是否经过 Skill 校验」的标记。
        没有 Skill 时原样返回（不塞空对象，避免污染输出结构）。
        """
        runtime = agent_skill_context(entrypoint).get("skill_runtime")
        return {**result, "_skill_runtime": {**runtime, "validation": "entrypoint_completed"}} if runtime else result

    workflow_context = {
        **safe_context,
        "repo_id": str(repo.id),
        "repo_full_name": repo.full_name,
        "has_repo": True,
        # 下面一组 has_* 是给 Planner 的「可用性开关」：没有 Issue 时它就不该排 issue_analysis 任务。
        "has_issue": selected_issue is not None,
        "issue_id": str(selected_issue.id) if selected_issue else None,
        "issue_number": selected_issue.number if selected_issue else None,
        # *_explicit 表示「用户明确点名了」，Planner 可以据此强调「就是这一个，不要换」。
        "issue_explicit": requested_issue_number is not None,
        "has_pr": selected_pr is not None,
        "pr_id": str(selected_pr.id) if selected_pr else None,
        "pr_number": selected_pr.number if selected_pr else None,
        "pr_explicit": requested_pr_number is not None,
        "has_ci_run": selected_run is not None,
        # has_failed_ci 单独给一个布尔：只有「失败」的 CI 才值得排排障任务。
        "has_failed_ci": selected_run is not None and selected_run.conclusion == "failure",
        "workflow_run_id": str(selected_run.id) if selected_run else None,
        "workflow_run_number": selected_run.github_run_id if selected_run else None,
        "workflow_run_explicit": requested_run_number is not None,
        # 涉及写操作就置 True，Planner 会额外排一个 safety_review 任务。
        "requires_safety": bool(_risky_action(message)),
        "team_members": safe_context.get("team_members") or [],
    }

    async def run_repo_health(task: WorkflowTask, spec: WorkflowSpec, workflow_context: dict[str, Any]) -> WorkflowTaskResult:
        """任务：仓库健康度概览（纯模板，不调模型，所以快且免费）。

        返回：WorkflowTaskResult，output.answer 是给人看的短文，另附三个计数字段。
        说明：confidence 0.82 是经验值——模板统计准确，但对「该怎么办」没有判断力，
            所以不给更高分；evidence_count 用三类数据的条数总和表示依据量。
        """
        answer = _repo_health_answer(repo, issues, prs, runs)
        open_issue_count = len([item for item in issues if item.state == "open"])
        open_pr_count = len([item for item in prs if item.state == "open"])
        return WorkflowTaskResult(
            task_id=task.id,
            agent_name=task.agent_name,
            task_type=task.task_type,
            status="success",
            summary=f"{repo.full_name}: {open_issue_count} open issues, {open_pr_count} open PRs, {len(failed_runs)} failed CI runs.",
            output={
                "answer": answer,
                "open_issues": open_issue_count,
                "open_prs": open_pr_count,
                "failed_ci": len(failed_runs),
            },
            evidence_count=len(issues) + len(prs) + len(runs),
            confidence=0.82,
            citations=[{"type": "repository", "id": str(repo.id), "title": repo.full_name}],
        )

    async def run_issue_task(task: WorkflowTask, spec: WorkflowSpec, workflow_context: dict[str, Any]) -> WorkflowTaskResult:
        """任务：分析一个 Issue（交给 IssueAgent）。

        先按 claim 定位目标 Issue；没有可用 Issue 时返回 skipped 而不是报错——
            「仓库里还没有 issue」是正常状态，不该让整个工作流失败。
        副作用：调用大模型。
        """
        issue = claimed_target(task, "issue", issues, selected_issue)
        if issue is None:
            return _skipped_workflow_result(task, "No synced issue is available.")
        result = await IssueAgent().run(
            {
                "id": str(issue.id),
                "title": issue.title,
                "body": issue.body,
                "labels": issue.labels,
                "number": issue.number,
                "state": issue.state,
                "assignees": issue.assignees,
                "author": issue.author,
            },
            {**safe_context, **agent_skill_context("issue_agent.analyze_issue")},
        )
        return WorkflowTaskResult(
            task_id=task.id,
            agent_name=task.agent_name,
            task_type=task.task_type,
            status="success",
            summary=f"Issue #{issue.number}: {result.get('conclusion')} / {result.get('priority')} / owner {result.get('suggested_owner')}.",
            output=with_skill_runtime(result, "issue_agent.analyze_issue"),
            evidence_count=len(result.get("evidence") or []),
            confidence=_safe_confidence(result.get("confidence"), 0.62),
            citations=[{"type": "issue", "id": str(issue.id), "title": issue.title}],
        )

    async def run_pr_task(task: WorkflowTask, spec: WorkflowSpec, workflow_context: dict[str, Any]) -> WorkflowTaskResult:
        """任务：评审一个 PR（交给 PRReviewAgent）。

        与独立接口的区别：这里不准备 PR 分支快照（太重），只把 CI 概况塞进上下文，
            够模型判断「有没有失败 CI 会阻塞合入」。
        ci_task 的存在性会影响 CI 归属：如果计划里也有 ci_debug 任务，
            就复用它的 claim 来定位同一条 CI，避免两个任务看着不同的 run。
        副作用：调用大模型；evidence_count 用变更文件数 + 评论数近似表示依据量。
        """
        pull_request = claimed_target(task, "pull_request", prs, selected_pr)
        if pull_request is None:
            return _skipped_workflow_result(task, "No synced pull request is available.")
        ci_task = next((item for item in spec.tasks if item.task_type == "ci_debug"), None)
        claimed_ci_run = claimed_target(ci_task, "workflow_run", runs, selected_run) if ci_task else selected_run
        pr_context = {
            **safe_context,
            **agent_skill_context("pr_review_agent.analyze_pr"),
            "repo_id": str(repo.id),
            "ci_summary": {
                "failed_runs": len(failed_runs),
                "latest_failed_run": claimed_ci_run.name if claimed_ci_run and claimed_ci_run.conclusion == "failure" else None,
            },
        }
        result = await PRReviewAgent().run(
            {
                "number": pull_request.number,
                "title": pull_request.title,
                "body": pull_request.body,
                "state": pull_request.state,
                "author": pull_request.author,
                "merged_at": pull_request.merged_at,
                "files": [
                    {
                        "filename": file.filename,
                        "status": file.status,
                        "additions": file.additions,
                        "deletions": file.deletions,
                        "patch": file.patch,
                    }
                    for file in pull_request.files
                ],
                "comments": [
                    {
                        "body": comment.body,
                        "path": comment.path,
                        "line": comment.line,
                        "author": comment.author,
                    }
                    for comment in pull_request.review_comments
                ],
            },
            pr_context,
        )
        return WorkflowTaskResult(
            task_id=task.id,
            agent_name=task.agent_name,
            task_type=task.task_type,
            status="success",
            summary=f"PR #{pull_request.number}: {result.get('merge_recommendation')} with {len(result.get('risk_points') or [])} risk points.",
            output=with_skill_runtime(result, "pr_review_agent.analyze_pr"),
            evidence_count=len(pull_request.files) + len(pull_request.review_comments),
            confidence=_safe_confidence(result.get("confidence"), 0.62),
            citations=[{"type": "pull_request", "id": str(pull_request.id), "title": pull_request.title}],
        )

    async def run_ci_task(task: WorkflowTask, spec: WorkflowSpec, workflow_context: dict[str, Any]) -> WorkflowTaskResult:
        """任务：分析一次 CI 失败（交给 CIDebugAgent）。

        confidence 默认 0.58，是本文件里最低的一档：CI 日志噪音大、根因常常不在日志里，
            模型判断的不确定性明显高于其他任务。
        evidence_count 用「job 数 + 是否有日志」近似——真正的证据是日志内容，无法精确计数。
        副作用：调用大模型（日志可能很长，token 消耗较高）。
        """
        workflow_run = claimed_target(task, "workflow_run", runs, selected_run)
        if workflow_run is None:
            return _skipped_workflow_result(task, "No synced CI run is available.")
        result = await CIDebugAgent().run(
            {
                "name": workflow_run.name,
                "status": workflow_run.status,
                "conclusion": workflow_run.conclusion,
                "jobs": workflow_run.jobs or [],
                "logs_text": workflow_run.logs_text,
            },
            {**safe_context, **agent_skill_context("ci_debug_agent.analyze_workflow_run"), "repo_id": str(repo.id)},
        )
        return WorkflowTaskResult(
            task_id=task.id,
            agent_name=task.agent_name,
            task_type=task.task_type,
            status="success",
            summary=f"CI {workflow_run.name}: {result.get('failure_type')}, merge blocking={result.get('is_merge_blocking')}.",
            output=with_skill_runtime(result, "ci_debug_agent.analyze_workflow_run"),
            evidence_count=len(workflow_run.jobs or []) + (1 if workflow_run.logs_text else 0),
            confidence=_safe_confidence(result.get("confidence"), 0.58),
            citations=[{"type": "workflow_run", "id": str(workflow_run.id), "title": workflow_run.name}],
        )

    async def run_rag_task(task: WorkflowTask, spec: WorkflowSpec, workflow_context: dict[str, Any]) -> WorkflowTaskResult:
        """任务：检索历史记录/文档作为证据（向量检索，不调生成模型）。

        检索词优先用用户原话（message），没有就用计划目标（spec.goal）。
        副作用：会写一条 recall 事件（所以工作流执行会留下检索痕迹）。
        confidence 分两档：有命中 0.68、没命中 0.45——
            检索结果本身不算结论，所以即使命中也不给高分。
        """
        query = message or spec.goal
        answer, citations, results = _semantic_search(repo, query, safe_context, db)
        record_recall_event(
            db,
            repo_id=repo.id,
            conversation_id=uuid.UUID(str(context["conversation_id"])) if context.get("conversation_id") else None,
            tool_name="workflow.rag_search",
            query=query,
            results=results,
            citations=citations,
            scope="project",
            mode="hybrid",
            metadata={"route": "workflow_rag_task", "task_id": task.id},
        )
        return WorkflowTaskResult(
            task_id=task.id,
            agent_name=task.agent_name,
            task_type=task.task_type,
            status="success",
            summary=f"Retrieved {len(citations)} related historical records.",
            output={"answer": answer, "citations": citations},
            evidence_count=len(citations),
            confidence=0.68 if citations else 0.45,
            citations=citations,
        )

    async def run_safety_task(task: WorkflowTask, spec: WorkflowSpec, workflow_context: dict[str, Any]) -> WorkflowTaskResult:
        """任务：对「可能写外部系统」的请求做风险评估。

        流程：判定动作类型 → 生成草稿模板 → 交给 SafetyAgent 判风险等级。
        本任务**只产出评估结果**，不会创建 ActionDraft——真正落草稿是 check_safety 工具的事，
            这样计划执行阶段不会偷偷产生待办。
        confidence 固定 0.9：动作分类规则明确，风险判断本身是确定性的。
        """
        action = _risky_action(message) or "write_back"
        draft, citations = _draft_for_action(action, message or spec.goal, repo, issues, prs, runs)
        result = await SafetyAgent().run(
            {"action": action, "draft": draft},
            {**safe_context, **agent_skill_context("safety_agent.classify_action_risk")},
        )
        return WorkflowTaskResult(
            task_id=task.id,
            agent_name=task.agent_name,
            task_type=task.task_type,
            status="success",
            summary=f"Safety risk={result.get('risk_level')}, confirmation required={result.get('requires_confirmation')}.",
            output=with_skill_runtime(result, "safety_agent.classify_action_risk"),
            evidence_count=1,
            confidence=0.9,
            citations=citations,
        )

    # 返回的字典键就是 WorkflowSpec 里的 task_type：
    # PlannerAgent 生成计划时只会用这几个 task_type，编排器再按类型查表调用。
    # 新增任务类型时，这里和 Planner 的可用类型列表必须同步改。
    return workflow_context, {
        "repo_health": run_repo_health,
        "issue_analysis": run_issue_task,
        "pr_review": run_pr_task,
        "ci_debug": run_ci_task,
        "rag_search": run_rag_task,
        "safety_review": run_safety_task,
    }


def _persist_engineering_workflow(
    db: Session,
    repo: Repository | None,
    conversation: Conversation | None,
    report: dict[str, Any],
) -> uuid.UUID:
    """把一次工作流执行结果落库（一条 workflow + N 条 task）。

    入参：report 是编排器的输出，含 spec / observation / final_answer / metrics / task_results。
    返回：新建的 AgentWorkflowRun.id。
    副作用：写两张表（**内部只 flush 不 commit**，事务由调用方提交）。
    说明：
        task_results 里只有 task_id，任务的 objective/依赖/claim 要从 spec.tasks 里取，
        所以先用 task_by_id 建索引再逐条写；
        status 由 error_count 推导——只要有一个任务报错，整个工作流就记为 error。
    """
    spec = report.get("spec") or {}
    task_by_id = {
        str(task.get("id")): task
        for task in spec.get("tasks", [])
        if isinstance(task, dict) and task.get("id")
    }
    workflow = AgentWorkflowRun(
        repo_id=repo.id if repo else None,
        conversation_id=conversation.id if conversation else None,
        goal=str(spec.get("goal") or ""),
        spec_json=spec,
        observation_json=report.get("observation") or {},
        final_answer=str(report.get("final_answer") or ""),
        metrics=report.get("metrics") or {},
        status="error" if int((report.get("metrics") or {}).get("error_count") or 0) else "success",
        completed_at=datetime.now(timezone.utc),
    )
    db.add(workflow)
    db.flush()
    for result in report.get("task_results") or []:
        if not isinstance(result, dict):
            continue
        task_id = str(result.get("task_id") or "")
        task = task_by_id.get(task_id, {})
        db.add(
            AgentTaskRun(
                workflow_id=workflow.id,
                task_id=task_id,
                agent_name=str(result.get("agent_name") or ""),
                task_type=str(result.get("task_type") or ""),
                claim_json=task.get("claim") if isinstance(task.get("claim"), dict) else {},
                input_json={"objective": task.get("objective"), "dependencies": task.get("dependencies", [])},
                result_json=result,
                status=str(result.get("status") or "unknown"),
                error=result.get("error"),
            )
        )
    return workflow.id


def _build_conversation_tools(repo: Repository | None, conversation: Conversation | None, db: Session) -> dict[str, Any]:
    """构造 ChatAgent 可调用的工具表（工具名 → 异步函数）。

    入参：
        repo: 当前仓库，可能为 None（用户还没接入仓库，此时多数工具降级成仓库健康度提示）。
        conversation: 当前会话，用于把 repo_id/conversation_id 自动注入工具参数。
        db: 请求级 Session，闭包捕获后供各工具读写。
    返回：{工具名: 可调用对象}，直接塞进 context["tools"] 交给 ChatAgent。
    工具约定（很重要，所有工具都遵守）：
        入参是 (input_data, context)，返回 dict 且必须含 route / tool_calls / answer / citations；
        失败尽量抛 ValueError（被 Agent 捕获后转成工具错误回给模型），而不是抛 HTTPException。
    副作用：不同工具差异很大——有的只读，有的会写 recall 事件、建草稿（check_safety）。
    """
    # MCP 客户端在这里创建一次，所有 devflow_* 工具共用同一个 stdio 子进程配置。
    mcp_client = StdioMcpClient()

    def agent_context(context: dict[str, Any]) -> dict[str, Any]:
        """剥掉 tools / 注册表后的上下文副本（传给子 Agent 时用，避免递归暴露工具对象）。"""
        return {key: value for key, value in context.items() if key not in {"tools", "_skill_registry"}}

    async def call_memory_mcp(tool_name: str, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """统一的 MCP 工具调用封装：注入 repo/conversation，再把 MCP 响应转成本项目工具格式。

        入参：tool_name 为 MCP 侧工具名（devflow_ 前缀）；input_data 由模型给出。
        返回：{route, tool_calls, answer, citations}。
        副作用：真实拉起 MCP 子进程并通信（网络/进程开销）。
        说明：
            repo_id / conversation_id 由服务端强制注入（覆盖模型传的值）——
            模型不该有能力跨仓库查记忆。
            MCP 返回的是 content 列表，这里只取 type=="text" 的片段拼成 answer。
        """
        arguments = dict(input_data)
        if repo:
            arguments["repo_id"] = str(repo.id)
        if conversation:
            arguments["conversation_id"] = str(conversation.id)
        if tool_name == "devflow_search_evidence":
            arguments.setdefault("query", input_data.get("message") or "")
        result = await mcp_client.call_tool(tool_name, arguments)
        text_parts = [
            str(item.get("text") or "")
            for item in result.get("content", [])
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return {
            "route": tool_name.replace("devflow_", ""),
            "tool_calls": [{"tool_name": tool_name, "arguments": arguments, "status": "success"}],
            "answer": "\n".join(text_parts).strip(),
            "citations": [],
        }

    async def search_evidence_mcp(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """工具 devflow_search_evidence：通过 MCP 检索会话/项目证据。"""
        return await call_memory_mcp("devflow_search_evidence", input_data, context)

    async def get_thread_context_mcp(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """工具 devflow_get_thread_context：通过 MCP 取当前会话/线程的上下文摘要。"""
        return await call_memory_mcp("devflow_get_thread_context", input_data, context)

    async def read_session_events_mcp(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """工具 devflow_read_session_events：通过 MCP 读本会话的历史事件流（工具调用轨迹）。"""
        return await call_memory_mcp("devflow_read_session_events", input_data, context)

    async def list_workspace_files(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """工具 workspace.list_files：列工作区里的可读文本文件（按相对路径子串过滤）。

        入参：path（起点，默认 "."）、pattern（路径子串，大小写不敏感）、limit（1~200，默认 80）。
        异常：路径越界/命中跳过目录 → ValueError（由上层转成工具错误）。
        """
        workspace_root = _workspace_root(repo)
        root = _workspace_path(workspace_root, str(input_data.get("path") or "."))
        limit = min(max(int(input_data.get("limit") or 80), 1), 200)
        raw_pattern = str(input_data.get("pattern") or "").strip().lower()
        rows: list[str] = []
        for path in _iter_workspace_files(root, workspace_root):
            relative = _workspace_relative(path, workspace_root)
            if raw_pattern and raw_pattern not in relative.lower():
                continue
            rows.append(relative)
            if len(rows) >= limit:
                break
        answer = "\n".join(rows) if rows else "没有找到匹配的工作区文件。"
        return {
            "route": "workspace_list_files",
            "tool_calls": [{"tool_name": "workspace.list_files", "arguments": {"path": str(input_data.get("path") or "."), "pattern": raw_pattern, "limit": limit}, "status": "success"}],
            "answer": answer,
            "citations": [{"type": "workspace", "id": str(workspace_root), "title": repo.full_name if repo else "项目工作区"}],
        }

    async def read_workspace_file(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """工具 workspace.read_file：按行区间读一个文件，返回带行号的文本。

        入参：path（必填）、start_line（≥1）、line_count（1~400）。
        返回：answer 首行是 "路径:起-止"，后面是 "行号: 内容"——
            带行号是刻意的，模型引用代码时能直接说出第几行。
        异常：缺 path、越界、非文本、过大 → ValueError。
        说明：end_line 会被文件实际行数截断，所以请求超过末尾不会报错，只是拿到的更少。
        """
        raw_path = str(input_data.get("path") or "").strip()
        if not raw_path:
            raise ValueError("path 为必填项")
        workspace_root = _workspace_root(repo)
        path = _workspace_path(workspace_root, raw_path)
        text = _read_workspace_text(path)
        lines = text.splitlines()
        start_line = max(int(input_data.get("start_line") or 1), 1)
        line_count = min(max(int(input_data.get("line_count") or 160), 1), 400)
        end_line = min(start_line + line_count - 1, len(lines))
        numbered = [f"{line_number}: {lines[line_number - 1]}" for line_number in range(start_line, end_line + 1)]
        relative = _workspace_relative(path, workspace_root)
        return {
            "route": "workspace_read_file",
            "tool_calls": [{"tool_name": "workspace.read_file", "arguments": {"path": relative, "start_line": start_line, "line_count": line_count}, "status": "success"}],
            "answer": f"{relative}:{start_line}-{end_line}\n" + "\n".join(numbered),
            "citations": [{"type": "workspace_file", "id": relative, "title": relative}],
        }

    async def search_workspace_code(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """工具 workspace.search_code：在工作区里做关键字/正则搜索，返回命中片段。

        入参：query（必填，用 query_terms 解析成检索词）、path（限定子目录）、limit（1~50，默认 12）。
        异常：query 为空、路径越界 → ValueError。
        说明：
            先调 _workspace_path 校验目录合法（防止把搜索范围指到工作区外），
            再交给 search_checkout_code 做实际扫描。
        """
        query = str(input_data.get("query") or input_data.get("message") or "").strip()
        if not query_terms(query):
            raise ValueError("query 为必填项")
        workspace_root = _workspace_root(repo)
        relative_path = str(input_data.get("path") or ".")
        _workspace_path(workspace_root, relative_path)
        limit = min(max(int(input_data.get("limit") or 12), 1), 50)
        matches = search_checkout_code(workspace_root, query, limit=limit, relative_path=relative_path)
        rows = [item["snippet"] for item in matches]
        citations = [
            {"type": "workspace_file", "id": item["source_id"], "title": item["title"]}
            for item in matches
        ]
        return {
            "route": "workspace_search_code",
            "tool_calls": [{"tool_name": "workspace.search_code", "arguments": {"query": query, "path": relative_path, "limit": limit}, "status": "success"}],
            "answer": "\n".join(rows) if rows else "没有找到匹配的工作区代码。",
            "citations": citations,
        }

    async def repo_health(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """工具 repo_health.generate_summary：仓库健康度概览（也是「没仓库时」的统一降级出口）。"""
        if repo is None:
            return {"route": "unknown", "tool_calls": [], "answer": "当前还没有可用仓库，请先连接仓库。", "citations": []}
        issues, prs, runs = _repo_payload(repo.id, db)
        return {
            "route": "repo_health",
            "tool_calls": [{"tool_name": "repo_health.generate_summary", "arguments": {"repo_id": str(repo.id)}, "status": "success"}],
            "answer": _repo_health_answer(repo, issues, prs, runs),
            "citations": [{"type": "repository", "id": str(repo.id), "title": repo.full_name}],
        }

    async def semantic_search(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """工具 rag.search_similar_documents：语义检索历史 issue/PR/文档。

        入参：query（无则退回用户原话 message）、以及 context 里的 source_type/limit 等过滤项。
        异常：query 为空 → ValueError。
        返回：除 answer/citations 外还带 retrieved_contexts 与 retrieval_results——
            这两个字段是给评测（RAGAS 的 context 指标）用的原始检索现场。
        副作用：写一条 recall 事件；metadata 里记录「query 是否被改写」，
            便于分析「模型改写检索词到底有没有用」。
        注意：没有 repo 时直接降级到仓库健康度，而不是报错。
        """
        if repo is None:
            return await repo_health(input_data, context)
        original_message = str(input_data.get("message") or "").strip()
        query = str(input_data.get("query") or original_message).strip()
        if not query:
            raise ValueError("query 为必填项")
        answer, citations, results = _semantic_search(repo, query, context, db)
        record_recall_event(
            db,
            repo_id=repo.id,
            conversation_id=conversation.id if conversation else None,
            tool_name="rag.search_similar_documents",
            query=query,
            results=results,
            citations=citations,
            scope="conversation" if conversation else "project",
            mode="hybrid",
            metadata={
                "source_type": context.get("source_type"),
                "limit": context.get("limit", 5),
                "original_message": original_message,
                "query_rewritten": bool(original_message and query != original_message),
            },
        )
        return {
            "route": "semantic_search",
            "tool_calls": [
                {
                    "tool_name": "rag.search_similar_documents",
                    "arguments": {"repo_id": str(repo.id), "query": query},
                    "status": "success",
                }
            ],
            "answer": answer,
            "citations": citations,
            "retrieved_contexts": [
                str(item.get("snippet") or "")
                for item in results
                if str(item.get("snippet") or "").strip()
            ],
            "retrieval_results": [
                {
                    "id": str(item.get("id") or ""),
                    "source_id": str(item.get("source_id") or ""),
                    "source_type": str(item.get("source_type") or ""),
                    "title": str(item.get("title") or ""),
                    "score": item.get("score"),
                    "snippet": str(item.get("snippet") or ""),
                    "retrieval": item.get("retrieval") or {},
                }
                for item in results
            ],
        }

    async def analyze_issue(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """工具 issue_agent.analyze_issue：分析「最新同步到的那个 Issue」。

        注意：这里**不接受指定编号**，总是取 created_at 最新的 Issue。
            需要针对某个编号分析时，走 /api/issues/{id}/analyze 或工程工作流（支持 claim 指定）。
        副作用：调用 IssueAgent（大模型）。
        """
        if repo is None:
            return await repo_health(input_data, context)
        issue = db.query(Issue).filter(Issue.repo_id == repo.id).order_by(Issue.created_at.desc()).first()
        if not issue:
            return {"route": "issue_analysis", "tool_calls": [], "answer": f"仓库 {repo.full_name} 还没有同步到 Issue。", "citations": []}
        result = await IssueAgent().run(
            {
                "id": str(issue.id),
                "title": issue.title,
                "body": issue.body,
                "labels": issue.labels,
                "number": issue.number,
                "state": issue.state,
                "assignees": issue.assignees,
                "author": issue.author,
            },
            agent_context(context),
        )
        return {
            "route": "issue_analysis",
            "tool_calls": [{"tool_name": "issue_agent.analyze_issue", "arguments": {"issue_id": str(issue.id)}, "status": "success"}],
            "answer": (
                f"Issue #{issue.number} 的结论：{result['conclusion']}。\n\n"
                f"- 分类：{result['category']}\n"
                f"- 优先级：{result['priority']}\n"
                f"- 复杂度：{result['complexity']}\n"
                f"- 建议负责人：{result['suggested_owner']}，{result['owner_reason']}\n\n"
                f"理由：{result['conclusion_reason']}\n\n"
                f"下一步：{_join_items(result.get('checklist') or result.get('action_items') or [], '补充复现步骤、影响范围和验收标准。')}"
            ),
            "citations": [{"type": "issue", "id": str(issue.id), "title": issue.title}],
        }

    async def analyze_pr(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """工具 pr_review_agent.analyze_pr：评审「最新创建的 PR」。

        与 analyze_issue 一样不支持指定编号；同样用 selectinload 预加载 files/comments。
        副作用：调用 PRReviewAgent（大模型）。
        """
        if repo is None:
            return await repo_health(input_data, context)
        pr = (
            db.query(PullRequest)
            .options(selectinload(PullRequest.files), selectinload(PullRequest.review_comments))
            .filter(PullRequest.repo_id == repo.id)
            .order_by(PullRequest.created_at.desc())
            .first()
        )
        if not pr:
            return {"route": "pr_review", "tool_calls": [], "answer": f"仓库 {repo.full_name} 还没有同步到 PR。", "citations": []}
        result = await PRReviewAgent().run(
            {
                "number": pr.number,
                "title": pr.title,
                "body": pr.body,
                "files": [{"filename": file.filename, "status": file.status, "patch": file.patch} for file in pr.files],
                "comments": [{"body": comment.body, "path": comment.path, "line": comment.line, "author": comment.author} for comment in pr.review_comments],
            },
            agent_context(context) | {"repo_id": str(repo.id)},
        )
        return {
            "route": "pr_review",
            "tool_calls": [{"tool_name": "pr_review_agent.analyze_pr", "arguments": {"pr_id": str(pr.id)}, "status": "success"}],
            "answer": f"PR #{pr.number} 摘要：{result['summary']}\n\n主要风险：{_join_items(result['risk_points'], '没有明显风险。')}\n测试建议：{_join_items(result['test_suggestions'], '运行相关测试和核心流程检查。')}",
            "citations": [{"type": "pull_request", "id": str(pr.id), "title": pr.title}],
        }

    async def analyze_ci(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """工具 ci_debug_agent.analyze_workflow_run：排障「最近一次失败的 CI」。

        没有失败 CI 时不是报错，而是返回仓库健康度 + 一句「还没有失败 CI」的说明，
            这样用户的提问也能得到有意义的回应。
        副作用：调用 CIDebugAgent（大模型，日志长时 token 较多）。
        """
        if repo is None:
            return await repo_health(input_data, context)
        run = db.query(WorkflowRun).filter(WorkflowRun.repo_id == repo.id, WorkflowRun.conclusion == "failure").order_by(WorkflowRun.created_at.desc()).first()
        if not run:
            issues, prs, runs = _repo_payload(repo.id, db)
            return {
                "route": "repo_health",
                "tool_calls": [{"tool_name": "repo_health.generate_summary", "arguments": {"repo_id": str(repo.id), "reason": "no_failed_ci"}, "status": "success"}],
                "answer": _repo_health_answer(repo, issues, prs, runs) + "\n\n说明：这个仓库目前还没有同步到失败 CI 运行。",
                "citations": [{"type": "repository", "id": str(repo.id), "title": repo.full_name}],
            }
        result = await CIDebugAgent().run({"name": run.name, "status": run.status, "conclusion": run.conclusion, "jobs": run.jobs or [], "logs_text": run.logs_text}, agent_context(context) | {"repo_id": str(repo.id)})
        return {
            "route": "ci_debug",
            "tool_calls": [{"tool_name": "ci_debug_agent.analyze_workflow_run", "arguments": {"run_id": str(run.id)}, "status": "success"}],
            "answer": f"最新失败 CI：{run.name}。失败类型：{result['failure_type']}。\n\n{result['failure_summary']}\n\n排查步骤：{_join_items(result['debug_steps'], '没有明确可用的排查步骤。')}",
            "citations": [{"type": "workflow_run", "id": str(run.id), "title": run.name}],
        }

    async def generate_report(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """工具 report_agent.generate_weekly_report：按时间窗生成研发周报。

        时间窗来自 context（time_range / start_date / end_date），默认最近 7 天。
        副作用：调用 ReportAgent（大模型），并把周报写入 RAG 文档库；
            注意 add_document 这里传了 commit=False——由外层请求事务统一提交，
            避免工具调用中途就把半成品数据落库。
        """
        if repo is None:
            return await repo_health(input_data, context)
        issues, prs, runs = _repo_payload(repo.id, db)
        start_at, end_at = _date_window_from_context(context)
        scoped_issues = [item for item in issues if any(_within(value, start_at, end_at) for value in [item.created_at, item.updated_at, item.closed_at])]
        scoped_prs = [item for item in prs if any(_within(value, start_at, end_at) for value in [item.created_at, item.updated_at, item.merged_at])]
        scoped_runs = [item for item in runs if any(_within(value, start_at, end_at) for value in [item.created_at, item.updated_at])]
        result = await ReportAgent().run(
            {
                "issues": [{"number": item.number, "title": item.title, "state": item.state, "closed_at": item.closed_at} for item in scoped_issues],
                "pull_requests": [{"number": item.number, "title": item.title, "state": item.state, "merged_at": item.merged_at} for item in scoped_prs],
                "workflow_runs": [{"name": item.name, "conclusion": item.conclusion} for item in scoped_runs],
                "metrics": {
                    "issues": len(scoped_issues),
                    "pull_requests": len(scoped_prs),
                    "merged_prs": len([item for item in scoped_prs if item.merged_at]),
                    "open_prs": len([item for item in scoped_prs if item.state == "open"]),
                    "failed_ci": len([item for item in scoped_runs if item.conclusion == "failure"]),
                },
            },
            {"repo_id": str(repo.id), "repo_name": repo.full_name, "start_date": start_at.date().isoformat(), "end_date": end_at.date().isoformat()},
        )
        add_document(
            db,
            repo.id,
            "weekly_report",
            None,
            f"研发周报 {start_at.date().isoformat()} - {end_at.date().isoformat()}",
            result["report_markdown"],
            {
                "display_title": f"研发周报 {start_at.date().isoformat()} - {end_at.date().isoformat()}",
                "start_date": start_at.date().isoformat(),
                "end_date": end_at.date().isoformat(),
                "repo_name": repo.full_name,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "source": "chat",
            },
            commit=False,
        )
        return {
            "route": "weekly_report",
            "tool_calls": [{"tool_name": "report_agent.generate_weekly_report", "arguments": {"repo_id": str(repo.id)}, "status": "success"}],
            "answer": result["report_markdown"],
            "citations": [{"type": "repository", "id": str(repo.id), "title": repo.full_name}],
        }

    async def check_safety(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """工具 safety_agent.classify_action_risk：把写操作请求转成「待确认草案」。

        这是整个系统最重要的安全闸门，流程：
            判定动作类型 → 生成草稿模板 → SafetyAgent 判风险 → 落 ActionDraft → 写审计。
        入参：input_data["message"] 必须是用户原话（用来判动作类型）。
        返回：route="safety_draft"，answer 里带上草稿 ID，让前端直接跳去确认页。
        副作用：
            写 action_draft（仅 flush，事务由外层提交）+ 写审计日志；
            **不会**真的调用 GitHub —— 真正的执行在 POST /api/action-drafts/{id}/confirm。
        目标选择：评论优先挂最近的 PR，其余动作挂最近的 Issue；
            都没有具体目标时退化成 target_type="repository"。
        """
        if repo is None:
            return await repo_health(input_data, context)
        issues, prs, runs = _repo_payload(repo.id, db)
        action = _risky_action(input_data["message"]) or "write_back"
        draft, citations = _draft_for_action(action, input_data["message"], repo, issues, prs, runs)
        safety = await SafetyAgent().run({"action": action, "draft": draft}, agent_context(context))
        latest_issue = sorted(issues, key=_item_time, reverse=True)[0] if issues else None
        latest_pr = sorted(prs, key=_item_time, reverse=True)[0] if prs else None
        target_type = "repository"
        target_id = None
        metadata: dict[str, Any] = {"source": "chat", "message": input_data["message"]}
        if action == "comment" and latest_pr:
            target_type = "pull_request"
            target_id = latest_pr.id
            metadata["pr_number"] = latest_pr.number
        elif action in {"comment", "create_issue", "close_issue"} and latest_issue:
            target_type = "issue"
            target_id = latest_issue.id
            metadata["issue_number"] = latest_issue.number
        draft_type = {
            "comment": "pr_comment" if target_type == "pull_request" else "issue_comment",
            "create_issue": "create_issue",
            "send_report": "send_report",
            "write_back": "comment",
        }.get(action, action)
        user = ensure_demo_user(db)
        action_draft = ActionDraft(
            repo_id=repo.id,
            draft_type=draft_type,
            target_type=target_type,
            target_id=target_id,
            title=f"{draft_type} 草稿",
            content=str(safety.get("draft") or draft),
            risk_level=str(safety.get("risk_level") or "medium"),
            created_by=user.id,
            meta=metadata,
        )
        db.add(action_draft)
        db.flush()
        write_audit_log(
            db,
            user=user,
            repo_id=repo.id,
            action="action_draft.create_from_chat",
            target_type=target_type,
            target_id=str(target_id) if target_id else None,
            request_json={"action": action, "draft_type": draft_type},
            result_json={"draft_id": str(action_draft.id)},
        )
        return {
            "route": "safety_draft",
            "tool_calls": [{"tool_name": "safety_agent.classify_action_risk", "arguments": {"action": action, "draft_id": str(action_draft.id)}, "status": "success"}],
            "answer": f"这个请求可能写入 GitHub 或发送到外部渠道，已生成待确认安全草稿。\n\n草稿 ID：{action_draft.id}\n\n{str(safety.get('draft') or draft)}",
            "citations": citations,
        }

    async def run_engineering_workflow(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """工具 workflow.run_engineering_review：让编排器自己规划并执行多步任务。

        这是「ChatAgent 把活外包给工作流」的入口：ChatAgent 只负责决定要不要用，
            真正的拆解（Planner）与执行（Orchestrator）都在工作流内部。
        副作用：调用多个子 Agent（可能多次大模型调用）、写 workflow/task 记录。
        """
        if repo is None:
            return await repo_health(input_data, context)

        message = str(input_data.get("message") or input_data.get("query") or "").strip()
        workflow_context, task_runners = _build_engineering_workflow_runtime(repo, db, message, context)
        report = await WorkflowOrchestrator(task_runners).run({"message": message or "Run an engineering workflow."}, workflow_context)
        workflow_run_id = _persist_engineering_workflow(db, repo, conversation, report)
        report["metrics"]["workflow_run_id"] = str(workflow_run_id)
        return _engineering_workflow_response(repo, workflow_run_id, report)

    # 键名就是模型看到的工具名，改了会直接破坏模型侧的调用约定（prompt 里也是按这个名字教的）。
    # 前三个是 workspace.*（读本地源码），devflow_* 三个是 MCP 桥接的记忆工具。
    return {
        "workspace.list_files": list_workspace_files,
        "workspace.read_file": read_workspace_file,
        "workspace.search_code": search_workspace_code,
        "repo_health.generate_summary": repo_health,
        "rag.search_similar_documents": semantic_search,
        "issue_agent.analyze_issue": analyze_issue,
        "pr_review_agent.analyze_pr": analyze_pr,
        "ci_debug_agent.analyze_workflow_run": analyze_ci,
        "report_agent.generate_weekly_report": generate_report,
        "safety_agent.classify_action_risk": check_safety,
        "workflow.run_engineering_review": run_engineering_workflow,
        "devflow_search_evidence": search_evidence_mcp,
        "devflow_get_thread_context": get_thread_context_mcp,
        "devflow_read_session_events": read_session_events_mcp,
    }


def _skill_runtime_context() -> dict[str, Any]:
    """取 Skill 注册表及其派生的三份运行时信息，供 Agent 上下文使用。

    返回：
        _skill_registry          —— 注册表实例本身（内部字段，传给子 Agent 前会被剥掉）；
        registered_skills        —— 给模型看的 Skill 清单（prompt 片段）；
        skill_trace_by_tool      —— 工具名 → Skill 追踪元数据（落库用于追溯）；
        skill_description_by_tool—— 工具名 → 追加到工具描述后的说明（教模型何时用）。
    副作用：get_skill_registry() 首次调用会扫描目录并缓存，之后是内存读取。
    """
    registry = get_skill_registry()
    return {
        "_skill_registry": registry,
        "registered_skills": registry.prompt_context(),
        "skill_trace_by_tool": registry.trace_metadata_by_tool(),
        "skill_description_by_tool": registry.tool_description_suffixes(),
    }


def _engineering_workflow_response(repo: Repository, workflow_run_id: uuid.UUID, report: dict[str, Any]) -> dict[str, Any]:
    """把工作流输出包装成「工具调用结果」的标准格式。

    返回：{route, tool_calls, answer, citations, agent_events}。
    说明：
        tool_calls 的第一条固定是 workflow.run_engineering_review 本身，
        后面用 * 展开 report 里的子工具调用——这样前端时间线既能显示「跑了一个工作流」，
        也能展开看到每个子 Agent 各自的调用。
    """
    return {
        "route": "engineering_workflow",
        "tool_calls": [
            {
                "tool_name": "workflow.run_engineering_review",
                "arguments": {"repo_id": str(repo.id), "workflow_run_id": str(workflow_run_id)},
                "status": "success",
            },
            *report.get("tool_calls", []),
        ],
        "answer": report.get("final_answer") or "",
        "citations": report.get("citations") or [],
        "agent_events": report.get("agent_events") or [],
    }


def _workflow_plan_answer(spec: WorkflowSpec) -> str:
    """把 WorkflowSpec 渲染成给人读的「计划确认」文案。

    入参：spec 为 PlannerAgent 产出的计划。
    返回：多行文本，逐条列出任务（含依赖与「关键任务」标记）和验收标准。
    说明：这段文字是用户确认前的唯一依据，所以必须把依赖关系和关键任务标出来——
        用户需要知道「哪些步骤会真的改东西」，才谈得上确认。
    """
    lines = [
        "已创建执行计划，确认后我会按这个 WorkflowSpec 执行。",
        "",
        f"目标：{spec.goal}",
        "",
        "任务：",
    ]
    for index, task in enumerate(spec.tasks, start=1):
        deps = f"；依赖：{', '.join(task.dependencies)}" if task.dependencies else ""
        critical = "；关键任务" if task.critical else ""
        lines.append(f"{index}. {task.agent_name} / {task.task_type}: {task.objective}{deps}{critical}")
    if spec.acceptance_criteria:
        lines.extend(["", "验收标准："])
        lines.extend(f"- {item}" for item in spec.acceptance_criteria)
    return "\n".join(lines)


async def _conversation_plan(payload: ChatRequest, db: Session) -> dict[str, Any]:
    """只生成计划、不执行（POST /api/chat/plan 的实现）。

    入参：payload.message 用户需求；payload.repo_id 当前仓库。
    返回：{route, conversation_id, answer, plan, tool_calls, agent_events}——
        plan 是 WorkflowSpec 的 JSON，前端拿它展示可编辑的计划卡片。
    异常：仓库不存在 → 404；目标编号歧义/找不到 → WorkflowTargetResolutionError（由路由转 422）。
    副作用：ensure_conversation 可能创建默认会话（会写库并提交）；调用 PlannerAgent（大模型）。
    """
    repo = _repo_or_404(db, payload.repo_id)
    conversation = ensure_conversation(db, repo.id, payload.conversation_id)
    db.commit()
    context = {
        **_client_context(payload.context),
        **_skill_runtime_context(),
        "repo_id": str(repo.id),
        "conversation_id": str(conversation.id),
    }
    workflow_context, _task_runners = _build_engineering_workflow_runtime(repo, db, payload.message, context)
    spec_data = await PlannerAgent().run({"message": payload.message}, workflow_context)
    spec = WorkflowSpec.model_validate(spec_data)
    spec_json = spec.model_dump(mode="json")
    return {
        "route": "engineering_plan",
        "conversation_id": str(conversation.id),
        "answer": _workflow_plan_answer(spec),
        "plan": spec_json,
        "tool_calls": [
            {
                "tool_name": "planner_agent.create_workflow_spec",
                "arguments": {"repo_id": str(repo.id), "workflow_id": spec.workflow_id},
                "status": "success",
            }
        ],
        "agent_events": [
            {
                "type": "workflow_spec",
                "workflow_id": spec.workflow_id,
                "goal": spec.goal,
                "tasks": [
                    {
                        "id": task.id,
                        "agent_name": task.agent_name,
                        "task_type": task.task_type,
                        "dependencies": task.dependencies,
                        "claim": task.claim.model_dump(mode="json"),
                    }
                    for task in spec.tasks
                ],
            }
        ],
    }


def _workflow_execution_trace_events(report: dict[str, Any], workflow_tool_id: str) -> list[dict[str, Any]]:
    """把工作流的执行痕迹翻译成前端认识的 SSE tool_result 事件。

    入参：report 为编排器输出（含 agent_events）；workflow_tool_id 是外层「工具开始」事件的 ID。
    返回：事件列表，第一条对应整个工作流，其余分别对应子任务/重规划/观察者。
    说明：
        这里做的是「事件格式适配」：编排器的 agent_events 是内部格式，
        前端时间线只认 tool_result 这套结构，所以在这里做一次映射，
        并给每条事件编一个带前缀的 tool_use_id（如 {workflow}_{task}），保证前端去重不冲突。
    """
    events = [
        {
            "event": "tool_result",
            "data": {
                "tool_use_id": workflow_tool_id,
                "tool_name": "workflow.run_engineering_review",
                "status": "success",
                "route": "engineering_workflow",
                "content": {
                    "answer_preview": "Confirmed plan executed.",
                    "citations_count": len(report.get("citations") or []),
                },
            },
        }
    ]
    for item in report.get("agent_events") or []:
        if not isinstance(item, dict):
            continue
        event_type = item.get("type")
        if event_type == "workflow_task_result":
            events.append(
                {
                    "event": "tool_result",
                    "data": {
                        "tool_use_id": f"{item.get('workflow_id')}_{item.get('task_id')}",
                        "tool_name": f"workflow.{item.get('task_type')}",
                        "status": item.get("status") or "success",
                        "route": "engineering_workflow",
                        "content": {
                            "answer_preview": item.get("summary") or "",
                            "citations_count": item.get("evidence_count") or 0,
                        },
                    },
                }
            )
        elif event_type == "workflow_replan":
            events.append(
                {
                    "event": "tool_result",
                    "data": {
                        "tool_use_id": f"{item.get('workflow_id')}_replan_{item.get('iteration')}",
                        "tool_name": "planner_agent.replan_workflow_spec",
                        "status": "success",
                        "route": "engineering_workflow",
                        "content": {
                            "answer_preview": (
                                f"Replan #{item.get('iteration')}: {item.get('reason')}; "
                                f"added tasks={', '.join(str(task) for task in item.get('added_tasks') or []) or 'none'}"
                            ),
                            "citations_count": 0,
                        },
                    },
                }
            )
        elif event_type == "workflow_observer_result":
            events.append(
                {
                    "event": "tool_result",
                    "data": {
                        "tool_use_id": f"{item.get('workflow_id')}_observer",
                        "tool_name": "workflow.observer",
                        "status": "success",
                        "route": "engineering_workflow",
                        "content": {
                            "answer_preview": f"Observer confidence={item.get('overall_confidence')}, findings={len(item.get('findings') or [])}",
                            "citations_count": 0,
                        },
                    },
                }
            )
    return events


async def _conversation_plan_execute_sse(payload: ChatPlanExecuteRequest, db: Session):
    """执行用户已确认的计划，并以 SSE 流式返回进度（POST /api/chat/plan/execute/stream 的实现）。

    入参：payload.workflow_spec 是前端回传的（可能被用户编辑过的）计划；payload.message 是原始需求。
    返回：异步生成器，产出 encode_sse 编码的事件。
    事件序列：tool_use(开始) → 若干 tool_result(各子任务) → delta*(正文分片) → final(最终结果)。
    异常：这里**不向外抛**——仓库不存在等错误已被 _repo_or_404 提前抛出（在生成器启动前），
        而执行期异常会在下方被兜住，写一条 error 消息后以 final 事件返回。
    副作用：
        重建证据库、调用多个子 Agent、写消息/AgentRun/workflow 记录、封存会话。
        注意先把用户消息落库（append + commit），保证即使后续失败，用户那句话也不会丢。
    """
    repo = _repo_or_404(db, payload.repo_id)

    conversation = ensure_conversation(db, repo.id, payload.conversation_id)
    if conversation:
        db.commit()
    compaction_stats = await ProgressiveContextManager(db).ensure_headroom(repo, conversation, payload.message)
    if compaction_stats.get("attempted"):
        db.commit()
    message_store = MessageStore(db)
    user_message = message_store.append(repo_id=repo.id, conversation_id=conversation.id, role="user", content=payload.message)
    EvidenceStore(db).rebuild_repo(repo, conversation)
    db.commit()
    assembled = ContextAssembler(db).assemble(repo, conversation, payload.message)
    context = {
        **_client_context(payload.context),
        **_skill_runtime_context(),
        "repo_id": str(repo.id),
        "conversation_id": str(conversation.id),
        "server_context": assembled.system_context,
        "server_messages": assembled.history_messages,
        "evidence": assembled.evidence,
        "compression_stats": assembled.compression_stats,
        "progressive_compaction": compaction_stats,
    }
    # 工具调用 ID：优先用计划里的 workflow_id，缺了就随机取 8 位——
    # 前端靠它把「开始事件」和后续的 tool_result 关联起来。
    workflow_tool_id = f"confirmed_{payload.workflow_spec.get('workflow_id') or uuid.uuid4().hex[:8]}"
    try:
        spec = WorkflowSpec.model_validate(payload.workflow_spec)
        yield encode_sse(
            {
                "event": "tool_use",
                "data": {
                    "id": workflow_tool_id,
                    "tool_name": "workflow.run_engineering_review",
                    "arguments": {"repo_id": str(repo.id), "workflow_id": spec.workflow_id, "confirmed_plan": True},
                },
            }
        )
        workflow_context, task_runners = _build_engineering_workflow_runtime(repo, db, payload.message, context)
        report = await WorkflowOrchestrator(task_runners).run_spec(spec, workflow_context)
        workflow_run_id = _persist_engineering_workflow(db, repo, conversation, report)
        report["metrics"]["workflow_run_id"] = str(workflow_run_id)
        for event in _workflow_execution_trace_events(report, workflow_tool_id):
            yield encode_sse(event)

        result = _engineering_workflow_response(repo, workflow_run_id, report)
        run_id = uuid.uuid4()
        result["run_id"] = str(run_id)
        result["compression_stats"] = assembled.compression_stats
        result["progressive_compaction"] = compaction_stats
        # 按 48 字符切片推送正文：这些内容其实早已生成完毕，切片只是为了做出
        # 「逐字输出」的观感（含中文时 48 字符约等于一行的视觉量）。想改快慢就改这个步长。
        for index in range(0, len(result["answer"]), 48):
            yield encode_sse({"event": "delta", "data": {"content": result["answer"][index : index + 48]}})

        assistant_message = message_store.append(
            repo_id=repo.id,
            conversation_id=conversation.id,
            role="assistant",
            content=result["answer"],
            route=result["route"],
            tool_calls=result["tool_calls"],
            meta={
                "run_id": str(run_id),
                "agent_events": result.get("agent_events", []),
                "model_usage": result.get("model_usage", []),
                "context_diagnostics": result.get("context_diagnostics", {}),
                "compression_stats": assembled.compression_stats,
                "progressive_compaction": compaction_stats,
            },
        )
        result["message_id"] = str(assistant_message.id)
        db.add(AgentRun(id=run_id, repo_id=repo.id, conversation_id=conversation.id, user_message=payload.message, route=result["route"], tool_calls=result["tool_calls"], final_answer=result["answer"], status="success"))
        touch_conversation(db, conversation, user_message=payload.message, increment=2)
        await SessionSealer(db).seal(
            repo=repo,
            conversation=conversation,
            user_message=user_message,
            assistant_message=assistant_message,
            route=result["route"],
            tool_calls=result["tool_calls"],
            agent_events=result.get("agent_events", []),
            model_usage=result.get("model_usage", []),
            context_diagnostics=result.get("context_diagnostics", {}),
        )
        db.commit()
        yield encode_sse({"event": "final", "data": result})
    except Exception as exc:
        # 执行失败也要「完整收尾」：回滚 → 写一条 route="error" 的助手消息 → 记 AgentRun
        # → 封存会话 → 提交。目的是让这次失败同样进入聊天历史和记忆，
        # 而不是留下一条只有用户提问、没有回答的残缺会话。
        db.rollback()
        run_id = uuid.uuid4()
        answer = f"执行计划失败：{exc}"
        assistant_message = message_store.append(
            repo_id=repo.id,
            conversation_id=conversation.id,
            role="assistant",
            content=answer,
            route="error",
            tool_calls=[],
            meta={"run_id": str(run_id), "error": str(exc)},
        )
        db.add(AgentRun(id=run_id, repo_id=repo.id, conversation_id=conversation.id, user_message=payload.message, route="error", tool_calls=[], final_answer=answer, status="error"))
        touch_conversation(db, conversation, user_message=payload.message, increment=2)
        await SessionSealer(db).seal(
            repo=repo,
            conversation=conversation,
            user_message=user_message,
            assistant_message=assistant_message,
            route="error",
            tool_calls=[],
            agent_events=[],
        )
        db.commit()
        yield encode_sse(
            {
                "event": "final",
                "data": {
                    "run_id": str(run_id),
                    "message_id": str(assistant_message.id),
                    "route": "error",
                    "tool_calls": [],
                    "answer": answer,
                    "citations": [],
                    "agent_events": [],
                },
            }
        )


async def _conversation_response(payload: ChatRequest, db: Session) -> dict[str, Any]:
    """同步版对话（POST /api/chat 的实现）：跑完 ChatAgent 一次性返回。

    入参：payload.message / repo_id / conversation_id / context。
    返回：ChatAgent 的工具调用结果（含 answer / route / tool_calls / citations 等）。
    异常：仓库不存在 → 404；Agent 内部异常向上抛（全局 500，不会写错误消息）。
        这是它与 SSE 版的行为差异：非流式接口失败就是失败，不留半轮记录。
    副作用：可能压缩上下文、重建证据库、写用户/助手消息、AgentRun、更新会话、封存会话。
    流程要点（SSE 版完全一致）：
        1. 先确保会话存在；2. 上下文超限就先压缩腾地方；3. 落用户消息并重建证据库；
        4. 组装上下文；5. 挂上工具表；6. 执行；7. 落助手消息 + 记忆封存。
    """
    repo = _repo_or_404(db, payload.repo_id)
    conversation = ensure_conversation(db, repo.id, payload.conversation_id)
    db.commit()
    compaction_stats = await ProgressiveContextManager(db).ensure_headroom(repo, conversation, payload.message)
    if compaction_stats.get("attempted"):
        db.commit()
    message_store = MessageStore(db)
    user_message = message_store.append(repo_id=repo.id, conversation_id=conversation.id, role="user", content=payload.message)
    EvidenceStore(db).rebuild_repo(repo, conversation)
    db.commit()
    assembled = ContextAssembler(db).assemble(repo, conversation, payload.message)
    context = {
        **_client_context(payload.context),
        **_skill_runtime_context(),
        "repo_id": str(repo.id),
        "conversation_id": str(conversation.id),
        "server_context": assembled.system_context,
        "server_messages": assembled.history_messages,
        "evidence": assembled.evidence,
        "compression_stats": assembled.compression_stats,
        "progressive_compaction": compaction_stats,
    }
    context["tools"] = _build_conversation_tools(repo, conversation, db)
    result = await ChatAgent().run({"message": payload.message}, context)
    run_id = uuid.uuid4()
    result["run_id"] = str(run_id)
    result["compression_stats"] = assembled.compression_stats
    result["progressive_compaction"] = compaction_stats
    assistant_message: ChatMessage | None = None
    assistant_message = message_store.append(
        repo_id=repo.id,
        conversation_id=conversation.id,
        role="assistant",
        content=result["answer"],
        route=result["route"],
        tool_calls=result["tool_calls"],
        meta={
            "run_id": str(run_id),
            "agent_events": result.get("agent_events", []),
            "model_usage": result.get("model_usage", []),
            "context_diagnostics": result.get("context_diagnostics", {}),
            "compression_stats": assembled.compression_stats,
            "progressive_compaction": compaction_stats,
        },
    )
    result["message_id"] = str(assistant_message.id)
    db.add(AgentRun(id=run_id, repo_id=repo.id, conversation_id=conversation.id, user_message=payload.message, route=result["route"], tool_calls=result["tool_calls"], final_answer=result["answer"], status="success"))
    touch_conversation(db, conversation, user_message=payload.message, increment=2)
    await SessionSealer(db).seal(
        repo=repo,
        conversation=conversation,
        user_message=user_message,
        assistant_message=assistant_message,
        route=result["route"],
        tool_calls=result["tool_calls"],
        agent_events=result.get("agent_events", []),
        model_usage=result.get("model_usage", []),
        context_diagnostics=result.get("context_diagnostics", {}),
    )
    db.commit()
    return result


async def _conversation_sse(payload: ChatRequest, db: Session):
    """流式版对话（POST /api/chat/stream 的实现）。

    入参/副作用与 _conversation_response 相同，区别在于：
        - 用 ChatAgent().run_stream 边生成边推送；
        - 只有收到 event=="final" 时才落库（此时才有完整答案）；
        - 异常不抛出，而是回滚后补写一条 route="error" 的助手消息，再推一条 final 事件。
    为什么要处理 final 事件：流式协议里只有 final 帧才带完整 answer/tool_calls，
        中间帧都是增量 delta；把落库放在 final 保证「存下来的和用户看到的一致」。
    """
    repo = _repo_or_404(db, payload.repo_id)
    conversation = ensure_conversation(db, repo.id, payload.conversation_id)
    db.commit()
    compaction_stats = await ProgressiveContextManager(db).ensure_headroom(repo, conversation, payload.message)
    if compaction_stats.get("attempted"):
        db.commit()
    message_store = MessageStore(db)
    user_message = message_store.append(repo_id=repo.id, conversation_id=conversation.id, role="user", content=payload.message)
    EvidenceStore(db).rebuild_repo(repo, conversation)
    db.commit()
    assembled = ContextAssembler(db).assemble(repo, conversation, payload.message)
    context = {
        **_client_context(payload.context),
        **_skill_runtime_context(),
        "repo_id": str(repo.id),
        "conversation_id": str(conversation.id),
        "server_context": assembled.system_context,
        "server_messages": assembled.history_messages,
        "evidence": assembled.evidence,
        "compression_stats": assembled.compression_stats,
        "progressive_compaction": compaction_stats,
    }
    context["tools"] = _build_conversation_tools(repo, conversation, db)
    try:
        async for event in ChatAgent().run_stream({"message": payload.message}, context):
            if event["event"] == "final":
                run_id = uuid.uuid4()
                event["data"]["run_id"] = str(run_id)
                event["data"]["compression_stats"] = assembled.compression_stats
                event["data"]["progressive_compaction"] = compaction_stats
                assistant_message = message_store.append(
                    repo_id=repo.id,
                    conversation_id=conversation.id,
                    role="assistant",
                    content=event["data"]["answer"],
                    route=event["data"]["route"],
                    tool_calls=event["data"]["tool_calls"],
                    meta={
                        "run_id": str(run_id),
                        "agent_events": event["data"].get("agent_events", []),
                        "model_usage": event["data"].get("model_usage", []),
                        "context_diagnostics": event["data"].get("context_diagnostics", {}),
                        "compression_stats": assembled.compression_stats,
                        "progressive_compaction": compaction_stats,
                    },
                )
                event["data"]["message_id"] = str(assistant_message.id)
                db.add(AgentRun(id=run_id, repo_id=repo.id, conversation_id=conversation.id, user_message=payload.message, route=event["data"]["route"], tool_calls=event["data"]["tool_calls"], final_answer=event["data"]["answer"], status="success"))
                touch_conversation(db, conversation, user_message=payload.message, increment=2)
                await SessionSealer(db).seal(
                    repo=repo,
                    conversation=conversation,
                    user_message=user_message,
                    assistant_message=assistant_message,
                    route=event["data"]["route"],
                    tool_calls=event["data"]["tool_calls"],
                    agent_events=event["data"].get("agent_events", []),
                    model_usage=event["data"].get("model_usage", []),
                    context_diagnostics=event["data"].get("context_diagnostics", {}),
                )
                db.commit()
            yield encode_sse(event)
    except Exception as exc:
        db.rollback()
        run_id = uuid.uuid4()
        answer = f"请求失败：{exc}"
        failed_event = {
            "event": "final",
            "data": {
                "run_id": str(run_id),
                "route": "error",
                "tool_calls": [],
                "answer": answer,
                "citations": [],
            },
        }
        assistant_message = message_store.append(
            repo_id=repo.id,
            conversation_id=conversation.id,
            role="assistant",
            content=answer,
            route="error",
            tool_calls=[],
            meta={"run_id": str(run_id), "error": str(exc)},
        )
        failed_event["data"]["message_id"] = str(assistant_message.id)
        db.add(AgentRun(id=run_id, repo_id=repo.id, conversation_id=conversation.id, user_message=payload.message, route="error", tool_calls=[], final_answer=answer, status="error"))
        touch_conversation(db, conversation, user_message=payload.message, increment=2)
        await SessionSealer(db).seal(
            repo=repo,
            conversation=conversation,
            user_message=user_message,
            assistant_message=assistant_message,
            route="error",
            tool_calls=[],
            agent_events=[],
        )
        db.commit()
        yield encode_sse(failed_event)


@router.post("/plan")
async def chat_plan(payload: ChatRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    """POST /api/chat/plan：让 AI 先出执行计划，不真正执行。

    入参：ChatRequest（message / repo_id / conversation_id / context）。
    返回：{route:"engineering_plan", conversation_id, answer, plan, tool_calls, agent_events}。
    异常：
        WorkflowTargetResolutionError（用户提到的编号有歧义或不存在）→ 422，
            用 422 而不是 400：语义是「请求能理解，但引用的目标无法处理」；
        仓库不存在 → 404（由内部 _repo_or_404 抛）。
    副作用：可能创建默认会话、调用 PlannerAgent。
    这是「人在回路」的关键一环：计划回给前端给人看/改，确认后才走 /plan/execute/stream。
    """
    try:
        return await _conversation_plan(payload, db)
    except WorkflowTargetResolutionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/plan/execute/stream")
async def chat_plan_execute_stream(payload: ChatPlanExecuteRequest, db: Session = Depends(get_db)) -> StreamingResponse:
    """POST /api/chat/plan/execute/stream：执行已确认的计划（SSE 流式）。

    入参：ChatPlanExecuteRequest，比 ChatRequest 多一个 workflow_spec（前端回传的计划）。
    返回：StreamingResponse(text/event-stream)。
    异常：仓库不存在会在生成器启动前抛 404；执行期异常转成 error 消息 + final 事件。
    副作用：见 _conversation_plan_execute_sse。
    安全性：这里没有校验 spec 是否「未经用户确认」——信任前端只在用户点确认后才调用；
        但由于所有写操作都会落成待确认草稿，即使被绕过也不会直接改 GitHub。
    """
    return StreamingResponse(_conversation_plan_execute_sse(payload, db), media_type="text/event-stream")


@router.post("", response_model=ChatResponse)
async def chat(payload: ChatRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    """POST /api/chat：同步对话接口（一次性返回完整回答）。

    入参：ChatRequest。
    返回：ChatResponse。
    异常：仓库不存在 → 404；Agent 异常向上抛（500）。
    副作用：写消息/AgentRun、更新会话、封存记忆（详见 _conversation_response）。
    适用场景：脚本、测试、或前端不需要流式效果的场景；聊天界面请用 /stream。
    """
    return await _conversation_response(payload, db)


@router.post("/stream")
async def chat_stream(payload: ChatRequest, db: Session = Depends(get_db)) -> StreamingResponse:
    """POST /api/chat/stream：流式对话接口（前端聊天窗的主入口）。

    入参：ChatRequest。
    返回：StreamingResponse(text/event-stream)，事件由 ChatAgent.run_stream 产出。
    异常：仓库不存在 → 404；执行期异常转成 error 助手消息 + final 事件，不中断连接。
    副作用：同 /chat。
    """
    return StreamingResponse(_conversation_sse(payload, db), media_type="text/event-stream")


@router.post("/recommendations", response_model=PromptRecommendationResponse)
async def prompt_recommendations(payload: PromptRecommendationRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    """POST /api/chat/recommendations：生成「接下来可以问什么」的推荐问题。

    入参：repo_id、exclude（已展示过的推荐，用于去重）、limit（条数）。
    返回：{agent, suggestions}。
    异常：仓库不存在 → 404；大模型失败向上抛。
    副作用：调用 RecommendationAgent（大模型）；不写库。
    说明：这里只返回 agent 和 suggestions 两个字段，其余诊断信息（如上下文用量）不透出。
    """
    repo = _repo_or_404(db, payload.repo_id)
    issues, prs, runs = _repo_payload(repo.id, db)
    result = await RecommendationAgent().run(
        _recommendation_payload(repo, issues, prs, runs, payload.exclude, payload.limit),
        {"repo_id": str(repo.id)},
    )
    return {
        "agent": result["agent"],
        "suggestions": result["suggestions"],
    }
