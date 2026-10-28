"""周报与仓库健康度接口，挂载前缀 /api/reports。

POST /weekly      让 ReportAgent 根据一段时间内的 issue/PR/CI 数据生成 Markdown 周报，
                  并把正文写进 RAG 文档库——这样同一份周报后续能被语义检索到。
POST /repo-health 轻量健康度快照，只做计数，不走大模型。

注意这里的两个 POST 都是「读数据 + 生成」，用 POST 只是因为要带结构化请求体，
并不是写操作（周报落库属于附带效果）。
"""

from datetime import date, datetime, time, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.models import Issue, PullRequest, Repository, WorkflowRun
from app.db.session import get_db
from app.schemas.reports import WeeklyReportRequest, WeeklyReportResponse
from app.services.agents.report_agent import ReportAgent
from app.services.rag.vector_store import add_document

router = APIRouter()


def _range_bounds(start_date: date, end_date: date) -> tuple[datetime, datetime]:
    """把「日期区间」扩成「时间戳闭区间」。

    入参：start_date / end_date 均为 date（不含时分秒）。
    返回：(起始日 00:00:00, 结束日 23:59:59.999999)，都带 UTC 时区。
    说明：
        统一用 UTC 是因为数据库里存的就是 UTC（见模型里的 datetime.now(timezone.utc)），
        而前端选的是「哪几天」，所以必须把右端点拉到当天最后一刻，
        否则选「今天」时会把今天产生的记录全部漏掉。
    """
    return (
        datetime.combine(start_date, time.min, tzinfo=timezone.utc),
        datetime.combine(end_date, time.max, tzinfo=timezone.utc),
    )


def _normalize_dt(value: datetime | None) -> datetime | None:
    """把可能「没有时区」的时间补成 UTC，避免和带时区的边界比较时报 TypeError。

    入参：value 为数据库取出的时间或 None。
    返回：补齐 tzinfo 的时间，或 None。
    说明：
        SQLite 不保存时区信息，经 SQLAlchemy 读回来是 naive datetime；
        而 _range_bounds 造的是 aware datetime。Python 不允许两者直接比较，
        所以统一按 UTC 解释——这与写入时用的 timezone.utc 一致。
    """
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _in_range(value: datetime | None, start_at: datetime, end_at: datetime) -> bool:
    """判断单个时间点是否落在闭区间内；None 一律视为不在区间（避免脏数据报错）。"""
    normalized = _normalize_dt(value)
    return normalized is not None and start_at <= normalized <= end_at


def _issue_in_range(issue: Issue, start_at: datetime, end_at: datetime) -> bool:
    """issue 在本区间内「有过动静」即算命中。

    为什么是 any 而不是只看 created_at：周报想汇总「这段时间团队干了什么」，
    一个上周创建、本周被关闭的 issue 也属于本周的工作量，只看创建时间会漏掉它。
    """
    return any(_in_range(value, start_at, end_at) for value in [issue.created_at, issue.updated_at, issue.closed_at])


def _pr_in_range(pr: PullRequest, start_at: datetime, end_at: datetime) -> bool:
    """PR 同理：创建、更新、合并任一落在区间内就算本周相关。"""
    return any(_in_range(value, start_at, end_at) for value in [pr.created_at, pr.updated_at, pr.merged_at])


@router.post("/weekly", response_model=WeeklyReportResponse)
async def weekly_report(payload: WeeklyReportRequest, db: Session = Depends(get_db)) -> dict:
    """POST /api/reports/weekly：生成研发周报。

    入参：
        payload: 含 repo_id、start_date、end_date（date 类型）。
        db: get_db 注入的 Session。
    返回：
        ReportAgent 的输出 dict，至少包含 report_markdown（响应模型会校验/补齐其它字段）。
    异常：
        仓库不存在 → 404；大模型调用失败会从 ReportAgent 内部抛出，走全局异常处理。
    副作用：
        会调用 LLM（耗时、耗配额），并把生成的周报作为 weekly_report 类型文档写入
        RAG 文档库（embedding + 向量库），因此重复调用会产生重复文档。
    """
    repo = db.get(Repository, payload.repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到仓库")
    start_at, end_at = _range_bounds(payload.start_date, payload.end_date)
    # 过滤放在 Python 层而不是 SQL：一是要跨多个时间列做 OR（created/updated/closed），
    # 二是 naive/aware 混用在 SQL 里比较不可靠。课程项目数据量小，全量拉回内存足够。
    issues = [
        item
        for item in db.query(Issue).filter(Issue.repo_id == payload.repo_id).all()
        if _issue_in_range(item, start_at, end_at)
    ]
    prs = [
        item
        for item in db.query(PullRequest).filter(PullRequest.repo_id == payload.repo_id).all()
        if _pr_in_range(item, start_at, end_at)
    ]
    runs = [
        item
        for item in db.query(WorkflowRun).filter(WorkflowRun.repo_id == payload.repo_id).all()
        if _in_range(item.created_at, start_at, end_at) or _in_range(item.updated_at, start_at, end_at)
    ]
    # 下面几组是喂给大模型的「量化指标」，避免模型自己在长列表里数数算错。
    merged_prs = [item for item in prs if item.merged_at]
    open_prs = [item for item in prs if item.state == "open"]
    failed_ci = [item for item in runs if item.conclusion == "failure"]
    result = await ReportAgent().run(
        # 第一个参数是「数据」：只挑周报需要的字段，减少 token 消耗，也避免把整行 ORM 对象序列化进去。
        {
            "issues": [{"number": item.number, "title": item.title, "state": item.state, "closed_at": item.closed_at} for item in issues],
            "pull_requests": [
                {"number": item.number, "title": item.title, "state": item.state, "merged_at": item.merged_at}
                for item in prs
            ],
            "workflow_runs": [{"name": item.name, "conclusion": item.conclusion} for item in runs],
            "metrics": {
                "issues": len(issues),
                "pull_requests": len(prs),
                "merged_prs": len(merged_prs),
                "open_prs": len(open_prs),
                "failed_ci": len(failed_ci),
            },
        },
        # 第二个参数是「上下文」：给模型交代仓库名和时间范围，让它写进标题/开头。
        {
            "repo_id": str(payload.repo_id),
            "repo_name": repo.full_name,
            "start_date": str(payload.start_date),
            "end_date": str(payload.end_date),
        },
    )
    # path 传 None：周报不属于任何源码文件，靠 source_type="weekly_report" 区分；
    # 入库后就能在搜索里被检索到，实现「写过的周报可以再问」。
    add_document(
        db,
        payload.repo_id,
        "weekly_report",
        None,
        f"研发周报 {payload.start_date} - {payload.end_date}",
        result["report_markdown"],
        {
            "display_title": f"研发周报 {payload.start_date} - {payload.end_date}",
            "start_date": str(payload.start_date),
            "end_date": str(payload.end_date),
            "repo_name": repo.full_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return result


@router.post("/repo-health")
async def repo_health(payload: WeeklyReportRequest, db: Session = Depends(get_db)) -> dict:
    """POST /api/reports/repo-health：仓库健康度快照（纯计数，不调大模型）。

    入参：payload 复用 WeeklyReportRequest，实际只用到 repo_id
        （请求模型借用是为了和 /weekly 保持同一套前端表单结构）。
    返回：dict，含 summary 文案、metrics 计数、risks 风险列表。
    异常：不校验仓库是否存在——即使 repo 不存在也返回全 0，接口足够轻，前端不会因此报错。
    副作用：只读，三次 count 查询。
    """
    open_issues = db.query(Issue).filter(Issue.repo_id == payload.repo_id, Issue.state == "open").count()
    open_prs = db.query(PullRequest).filter(PullRequest.repo_id == payload.repo_id, PullRequest.state == "open").count()
    failed_ci = db.query(WorkflowRun).filter(WorkflowRun.repo_id == payload.repo_id, WorkflowRun.conclusion == "failure").count()
    return {
        # 如实说明这是 demo 口径，避免使用者把这里的数字当正式度量指标。
        "summary": "仓库健康度以 demo 指标估算，建议结合近期失败 CI 和高风险 PR 判断。",
        # merged_prs_7d 写死 0：还没实现「近 7 天」的窗口统计，先占位保证前端字段不缺。
        "metrics": {"open_issues": open_issues, "open_prs": open_prs, "failed_ci_runs": failed_ci, "merged_prs_7d": 0},
        "risks": ["存在失败 CI，需要排查"] if failed_ci else [],
    }
