"""工作区（Workspace）接口，挂载前缀 /api/workspaces。

「工作区」= 一组仓库的集合，用来做跨仓库视角的汇总，比如一次给 5 个仓库出周报。
三个接口：
    POST /                    新建工作区（需要 settings:manage，属于配置类操作）
    GET  /                    列出工作区（只需 repo:read）
    POST /multi-repo-report   把多个仓库的 issue/PR/CI 汇总成一份 Markdown 周报

和 reports.py 的区别：那边是单仓库 + 大模型生成，这边是多仓库 + 纯模板拼装（确定、快、免费）。
"""

from datetime import datetime, time, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.models import Issue, PullRequest, Repository, User, WorkflowRun, Workspace
from app.db.session import get_db
from app.schemas.workspaces import MultiRepoReportRequest, MultiRepoReportResponse, WorkspaceCreateRequest, WorkspaceResponse
from app.services.permissions import require_permission, write_audit_log
from app.services.rag.vector_store import add_document

router = APIRouter()


def _bounds(payload: MultiRepoReportRequest) -> tuple[datetime, datetime]:
    """把请求里的日期区间转成 UTC 时间戳闭区间（右端取当天最后一刻）。

    与 reports.py 的同名逻辑一致：右端必须用 time.max，否则 end_date 当天的数据会被漏掉。
    """
    return (
        datetime.combine(payload.start_date, time.min, tzinfo=timezone.utc),
        datetime.combine(payload.end_date, time.max, tzinfo=timezone.utc),
    )


def _normalize(value: datetime | None) -> datetime | None:
    """给 naive datetime 补 UTC 时区，避免与 _bounds 产出的 aware 时间比较时报错。"""
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _in_range(value: datetime | None, start_at: datetime, end_at: datetime) -> bool:
    """闭区间判定；None 视为不在区间内。"""
    normalized = _normalize(value)
    return normalized is not None and start_at <= normalized <= end_at


def _repo_ids_from_payload(db: Session, payload: MultiRepoReportRequest) -> list[UUID]:
    """合并「显式传入的 repo_ids」与「workspace 自带的 repo_ids」，并按出现顺序去重。

    入参：
        payload: 可能给了 repo_ids、可能给了 workspace_id，也可能两者都给。
    返回：去重后的仓库 ID 列表（保持先 repo_ids 后 workspace 的顺序）。
    异常：workspace_id 指向不存在的工作区 → 404。
    说明：
        两者取并集是为了支持「在工作区基础上临时加几个仓库」的用法；
        去重同时又不能用 set 直接返回，否则顺序随机，报告里的仓库顺序每次都不一样。
    """
    repo_ids = list(payload.repo_ids)
    if payload.workspace_id:
        workspace = db.get(Workspace, payload.workspace_id)
        if workspace is None:
            raise HTTPException(status_code=404, detail="未找到 workspace")
        # workspace.repo_ids 在库里是字符串数组（JSON 列），这里转回 UUID 以统一类型。
        repo_ids.extend(UUID(str(item)) for item in (workspace.repo_ids or []))
    seen: set[UUID] = set()
    output = []
    for repo_id in repo_ids:
        if repo_id not in seen:
            seen.add(repo_id)
            output.append(repo_id)
    return output


@router.post("", response_model=WorkspaceResponse)
async def create_workspace(
    payload: WorkspaceCreateRequest,
    db: Session = Depends(get_db),
    user: User = Depends(require_permission("settings:manage")),
) -> Workspace:
    """POST /api/workspaces：新建工作区。

    入参：
        payload: 名称、描述、仓库 ID 列表。
        user: 需要 settings:manage 权限，用于记录 created_by 和审计。
    返回：新建的 Workspace。
    异常：
        有仓库 ID 不存在时 → 404，detail 里会列出全部非法 ID（便于前端一次性提示，而不是一个个试）。
    副作用：写 workspaces + 一条 workspace.create 审计日志。
    """
    # 一次查出所有存在的仓库 ID，再和入参做差集，避免逐个 db.get 造成 N 次查询。
    known_repo_ids = {repo.id for repo in db.query(Repository).filter(Repository.id.in_(payload.repo_ids)).all()} if payload.repo_ids else set()
    missing = [str(repo_id) for repo_id in payload.repo_ids if repo_id not in known_repo_ids]
    if missing:
        raise HTTPException(status_code=404, detail=f"仓库不存在：{', '.join(missing)}")
    workspace = Workspace(
        name=payload.name,
        description=payload.description,
        # 落库时转成字符串列表：repo_ids 是 JSON 列，存 str 才能被 JSON 序列化。
        repo_ids=[str(repo_id) for repo_id in payload.repo_ids],
        created_by=user.id,
    )
    db.add(workspace)
    write_audit_log(db, user=user, action="workspace.create", request_json=payload.model_dump(mode="json"))
    db.commit()
    db.refresh(workspace)
    return workspace


@router.get("", response_model=list[WorkspaceResponse])
async def list_workspaces(
    db: Session = Depends(get_db),
    _user: User = Depends(require_permission("repo:read")),
) -> list[Workspace]:
    """GET /api/workspaces：列出全部工作区。

    入参：无（用户仅用于权限校验）。
    返回：按 updated_at 倒序的完整列表——**没有分页也没有条数上限**，
        因为工作区是「配置」级数据，数量很少；若将来变多需要自己加 limit。
    异常：无 repo:read → 403。
    副作用：只读。
    """
    return db.query(Workspace).order_by(Workspace.updated_at.desc()).all()


@router.post("/multi-repo-report", response_model=MultiRepoReportResponse)
async def multi_repo_report(
    payload: MultiRepoReportRequest,
    db: Session = Depends(get_db),
    user: User = Depends(require_permission("agent:run")),
) -> dict:
    """POST /api/workspaces/multi-repo-report：生成多仓库汇总周报。

    入参：
        payload: 仓库列表或 workspace_id 二选一（也可都给，取并集）+ 日期区间。
        user: 需要 agent:run 权限；同时用于审计。
    返回：{"report_markdown", "metrics", "repo_summaries"}。
    异常：
        仓库列表为空 → 400（这是调用方的参数错误，不是资源不存在）；
        workspace_id 不存在 → 404；无 agent:run → 403。
    副作用：
        对每个仓库各写入一份 weekly_report 文档到 RAG（同一份正文复制多份，
        这样在任一仓库内搜索都能命中这份汇总），整个循环共用一个事务，
        最后连同审计日志一起 commit。
    """
    repo_ids = _repo_ids_from_payload(db, payload)
    if not repo_ids:
        raise HTTPException(status_code=400, detail="至少选择一个仓库")
    start_at, end_at = _bounds(payload)
    repos = db.query(Repository).filter(Repository.id.in_(repo_ids)).all()
    summaries = []
    # totals 先放好 repos 计数，省得最后再补一次。
    totals = {"open_issues": 0, "open_prs": 0, "merged_prs": 0, "failed_ci": 0, "repos": len(repos)}
    for repo in repos:
        issues = db.query(Issue).filter(Issue.repo_id == repo.id).all()
        prs = db.query(PullRequest).filter(PullRequest.repo_id == repo.id).all()
        runs = db.query(WorkflowRun).filter(WorkflowRun.repo_id == repo.id).all()
        # 「本周期内有动静」的集合：任一关键时间点落在区间内即算（理由同 reports.py）。
        ranged_issues = [item for item in issues if any(_in_range(value, start_at, end_at) for value in [item.created_at, item.updated_at, item.closed_at])]
        ranged_prs = [item for item in prs if any(_in_range(value, start_at, end_at) for value in [item.created_at, item.updated_at, item.merged_at])]
        ranged_runs = [item for item in runs if any(_in_range(value, start_at, end_at) for value in [item.created_at, item.updated_at])]
        # 注意口径差异：open_* 看的是「当前快照」（不限日期），merged_prs / failed_ci 看的是「区间内」。
        # 因为「现在还有多少没关」是状态量，「这周合了几个」是区间量，两者不该用同一个时间窗。
        open_issues = len([item for item in issues if item.state == "open"])
        open_prs = len([item for item in prs if item.state == "open"])
        merged_prs = len([item for item in ranged_prs if item.merged_at])
        failed_ci = len([item for item in ranged_runs if item.conclusion == "failure"])
        totals["open_issues"] += open_issues
        totals["open_prs"] += open_prs
        totals["merged_prs"] += merged_prs
        totals["failed_ci"] += failed_ci
        summaries.append(
            {
                "repo_id": str(repo.id),
                "repo_name": repo.full_name,
                "range_issues": len(ranged_issues),
                "range_prs": len(ranged_prs),
                "open_issues": open_issues,
                "open_prs": open_prs,
                "merged_prs": merged_prs,
                "failed_ci": failed_ci,
                # 风险分级用固定阈值（5）而不是分位数：规则简单、可解释，
                # 优先看 CI 失败（有失败直接 high），其次看 PR 积压，最后看 issue 堆积。
                "risk_level": "high" if failed_ci or open_prs > 5 else "medium" if open_issues > 5 else "low",
            }
        )
    risk_repos = [item for item in summaries if item["risk_level"] == "high"]
    # 报告用字符串模板拼装而非调大模型：多仓库场景下要的是「数字准确、生成快」，
    # 不追求自然语言润色，也就不必付 LLM 的延迟和费用。
    lines = [
        f"# 多仓库研发周报 {payload.start_date} - {payload.end_date}",
        "",
        f"- 覆盖仓库：{len(repos)} 个",
        f"- Open Issues：{totals['open_issues']}",
        f"- Open PRs：{totals['open_prs']}",
        f"- 本周期 merged PRs：{totals['merged_prs']}",
        f"- 失败 CI：{totals['failed_ci']}",
        "",
        "## 仓库明细",
    ]
    for item in summaries:
        lines.append(
            f"- {item['repo_name']}：open issue {item['open_issues']}，open PR {item['open_prs']}，merged {item['merged_prs']}，失败 CI {item['failed_ci']}，风险 {item['risk_level']}"
        )
    # 用 extend 一次性追加「小节标题 + 各仓库条目」，避免在循环里反复判断是否要加标题。
    lines.extend(["", "## 重点关注", *(f"- {item['repo_name']} 需要优先处理失败 CI 或 PR backlog。" for item in risk_repos)])
    if not risk_repos:
        # 没有高风险也要给一句话结论，避免「重点关注」小节下面空着显得像出错。
        lines.append("- 暂无高风险仓库，建议继续关注 PR 停留时间和评审积压。")
    report_markdown = "\n".join(lines) + "\n"
    for repo in repos:
        # commit=False：循环里只 flush 不提交，最后统一 commit，
        # 避免写 N 个仓库时提交 N 次事务（慢且容易半途失败留下部分数据）。
        add_document(
            db,
            repo.id,
            "weekly_report",
            None,
            f"多仓库研发周报 {payload.start_date} - {payload.end_date}",
            report_markdown,
            {
                "display_title": f"多仓库研发周报 {payload.start_date} - {payload.end_date}",
                "start_date": str(payload.start_date),
                "end_date": str(payload.end_date),
                "workspace_id": str(payload.workspace_id) if payload.workspace_id else None,
                "repo_count": len(repos),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
            commit=False,
        )
    write_audit_log(db, user=user, action="workspace.multi_repo_report", request_json=payload.model_dump(mode="json"), result_json=totals)
    db.commit()
    return {"report_markdown": report_markdown, "metrics": totals, "repo_summaries": summaries}
