"""仓库数据同步：把 GitHub 上的 issue / PR / Actions 运行记录拉回本地库。

同步策略是「按远端 ID 做 upsert」而不是先清空再全量插入：
- 本地保留自增主键，已有引用（评审草稿、索引）不会因为一次同步而失效；
- 一次只拉 payload.limit 条，属于增量补齐，历史数据不动。

注意：整个过程只有一个事务，末尾才 commit。中间抛出的未捕获异常会整体回滚，
但也正因为「末尾统一 commit」，某些被吞掉的子请求异常（例如 PR 文件拉取失败）
留下的「已删除但未写回」状态会被一起提交——详见对应位置的行内注释。
"""

from datetime import datetime, timezone
from uuid import UUID

import httpx
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import decrypt_token
from app.db.models import Issue, PRFile, PRReviewComment, PullRequest, Repository, WorkflowRun
from app.schemas.repos import RepoSyncRequest, RepoSyncResponse
from app.services.github.actions import get_workflow_run_logs, list_workflow_run_jobs, list_workflow_runs
from app.services.github.issues import list_issues
from app.services.github.pull_requests import list_pull_request_files, list_pull_request_review_comments, list_pull_requests

# 更改返回时间格式：GitHub 返回 ISO8601 且以 "Z" 结尾，
# 而 fromisoformat 在部分 Python 版本上不认 "Z"，所以先替换成 +00:00 再解析。
# 结果统一是带时区的 datetime，写库时不会和本地时区混淆。
def parse_github_datetime(value: str | None) -> datetime | None:
    """把 GitHub 的时间字符串转成带时区的 datetime，空值返回 None。"""
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))

# 从返回中提取用户名：GitHub 的 user 字段可能是 null（账号已注销），
# 所以这里做空值保护，避免直接 .get 抛 AttributeError。
def login_of(user: dict | None) -> str | None:
    """取出 GitHub 用户对象里的 login，user 为空返回 None。"""
    return user.get("login") if user else None


async def sync_repository(db: Session, repo: Repository, payload: RepoSyncRequest) -> RepoSyncResponse:
    """按 payload 指定的范围同步一个仓库，返回各类实体的同步条数。

    三个 sync_* 开关互相独立，调用方可以只刷 PR 不刷 Actions，省 API 配额。
    """
    # 目前只实现了 GitHub 协议；其它 provider 直接 400，
    # 避免静默返回空数据让前端误以为「同步成功但确实没数据」。
    if repo.provider not in {"github", "github_compatible"}:
        raise HTTPException(status_code=400, detail=f"{repo.provider} sync is not implemented yet")

    synced = {"issues": 0, "pull_requests": 0, "workflow_runs": 0}
    # token 在库里是加密存的，这里临时解密成明文只用于本次请求，不落日志。
    token = decrypt_token(repo.github_token_encrypted, settings.token_encryption_key) if repo.github_token_encrypted else None
    # 支持 GitHub Enterprise 等自建实例：所有请求都带上仓库配置的 base_url。
    base_url = repo.api_base_url

    if payload.sync_issues:
        try:
            remote_issues = await list_issues(repo.owner, repo.name, token=token, limit=payload.limit, base_url=base_url)
        except httpx.HTTPStatusError as exc:
            # issue 是用户显式要求同步的主对象，失败就直接中断并透出状态码，
            # 不像 PR 子资源那样降级成空列表（否则会误报同步成功）。
            raise HTTPException(status_code=400, detail=f"GitHub issues sync failed: {exc.response.status_code} {exc.response.text}") from exc
        for item in remote_issues:
            # 以 (repo_id, github_issue_id) 作为业务唯一键做 upsert，避免重复插入。
            issue = db.query(Issue).filter(Issue.repo_id == repo.id, Issue.github_issue_id == item.get("id")).one_or_none()
            if issue is None:
                issue = Issue(repo_id=repo.id, github_issue_id=item.get("id"), number=item.get("number", 0), title=item.get("title") or "")
                db.add(issue)
            # 下面用 `新值 or 旧值`：远端字段缺失（None）时保留本地值，
            # 防止某次接口返回不完整把已同步好的数据覆盖成空。
            issue.number = item.get("number", issue.number)
            issue.title = item.get("title") or issue.title
            # body/state 则允许被覆盖成空或默认值：正文被清空、issue 被重开都是正常变更。
            issue.body = item.get("body")
            issue.state = item.get("state") or "open"
            # GitHub 的 labels 是对象数组，本地只存名字，顺便过滤掉没有 name 的脏数据。
            issue.labels = [label.get("name") for label in item.get("labels", []) if label.get("name")]
            issue.author = login_of(item.get("user"))
            issue.assignees = [login_of(user) for user in item.get("assignees", []) if login_of(user)]
            issue.created_at = parse_github_datetime(item.get("created_at"))
            issue.updated_at = parse_github_datetime(item.get("updated_at"))
            issue.closed_at = parse_github_datetime(item.get("closed_at"))
            synced["issues"] += 1

    if payload.sync_pull_requests:
        try:
            remote_prs = await list_pull_requests(repo.owner, repo.name, token=token, limit=payload.limit, base_url=base_url)
        except httpx.HTTPStatusError as exc:
            # 同 issue：PR 列表拉不到属于硬失败，直接报错而不是返回部分结果。
            raise HTTPException(status_code=400, detail=f"GitHub pull requests sync failed: {exc.response.status_code} {exc.response.text}") from exc
        for item in remote_prs:
            pr = db.query(PullRequest).filter(PullRequest.repo_id == repo.id, PullRequest.github_pr_id == item.get("id")).one_or_none()
            if pr is None:
                pr = PullRequest(repo_id=repo.id, github_pr_id=item.get("id"), number=item.get("number", 0), title=item.get("title") or "")
                db.add(pr)
                # 必须 flush 才能拿到自增主键：下面要用 pr.id 去删/插 PRFile 和评审评论。
                db.flush()
            pr.number = item.get("number", pr.number)
            pr.title = item.get("title") or pr.title
            pr.body = item.get("body")
            pr.state = item.get("state") or "open"
            pr.author = login_of(item.get("user"))
            # base/head 可能为 null（例如来源分支已删除），先兜成空字典再取 ref。
            pr.base_branch = (item.get("base") or {}).get("ref")
            pr.head_branch = (item.get("head") or {}).get("ref")
            pr.merged_at = parse_github_datetime(item.get("merged_at"))
            pr.created_at = parse_github_datetime(item.get("created_at"))
            pr.updated_at = parse_github_datetime(item.get("updated_at"))

            # ⚠️ 注意执行顺序：DELETE 在拉取之前执行，且在当前事务里立刻生效。
            # 这里的语义是「全量替换」——先清掉旧的变更文件，再用远端结果重建。
            # 风险：如果紧接着的 list_pull_request_files 失败（下面 except 把 files 置空），
            # 本地记录此时已经被删掉，而函数末尾依然会 commit，
            # 于是「拉取失败 → 本地记录被清空」，且新旧数据同时丢失。
            # 正确做法应是拉取成功后再删；按约定此处只说明风险，不改代码。
            db.query(PRFile).filter(PRFile.pr_id == pr.id).delete()
            try:
                files = await list_pull_request_files(repo.owner, repo.name, pr.number, token=token, base_url=base_url)
            except httpx.HTTPStatusError:
                # 降级成空列表：宁可这次不更新文件列表，也不让整个同步失败。
                # 副作用见上：配合前面的 delete，会把该 PR 已同步的文件记录清空。
                files = []
            for file_item in files:
                # 只存统计和 patch 文本，不存整份文件内容，以控制库体积；
                # patch 可能为 None（二进制文件、改动过大时 GitHub 直接省略）。
                db.add(
                    PRFile(
                        pr_id=pr.id,
                        filename=file_item.get("filename") or "",
                        status=file_item.get("status") or "modified",
                        additions=file_item.get("additions") or 0,
                        deletions=file_item.get("deletions") or 0,
                        patch=file_item.get("patch"),
                    )
                )
            # 评审评论同样是「先删后拉」，与上面的 PRFile 有完全相同的失败清空风险。
            db.query(PRReviewComment).filter(PRReviewComment.pr_id == pr.id).delete()
            try:
                review_comments = await list_pull_request_review_comments(repo.owner, repo.name, pr.number, token=token, base_url=base_url)
            except httpx.HTTPStatusError:
                review_comments = []
            for comment_item in review_comments:
                # line 是新版字段，老评论只有 original_line；两者都取不到就存 None（整文件级评论）。
                db.add(
                    PRReviewComment(
                        pr_id=pr.id,
                        github_comment_id=comment_item.get("id"),
                        body=comment_item.get("body"),
                        path=comment_item.get("path"),
                        line=comment_item.get("line") or comment_item.get("original_line"),
                        author=login_of(comment_item.get("user")),
                        created_at=parse_github_datetime(comment_item.get("created_at")),
                        updated_at=parse_github_datetime(comment_item.get("updated_at")),
                    )
                )
            synced["pull_requests"] += 1

    if payload.sync_workflow_runs:
        try:
            remote_runs = await list_workflow_runs(repo.owner, repo.name, token=token, limit=payload.limit, base_url=base_url)
        except httpx.HTTPStatusError:
            # Actions 是附加信息：拉不到就跳过，不影响 issue/PR 的同步结果。
            remote_runs = []
        for item in remote_runs:
            run = db.query(WorkflowRun).filter(WorkflowRun.repo_id == repo.id, WorkflowRun.github_run_id == item.get("id")).one_or_none()
            if run is None:
                run = WorkflowRun(repo_id=repo.id, github_run_id=item.get("id"), name=item.get("name") or "workflow")
                db.add(run)
            run.name = item.get("name") or run.name
            # 缺字段时给占位值，前端展示逻辑就不用到处判 None。
            run.status = item.get("status") or "unknown"
            run.conclusion = item.get("conclusion")
            run.html_url = item.get("html_url")
            run.created_at = parse_github_datetime(item.get("created_at"))
            run.updated_at = parse_github_datetime(item.get("updated_at"))
            if run.github_run_id:
                try:
                    jobs = await list_workflow_run_jobs(repo.owner, repo.name, run.github_run_id, token=token, base_url=base_url)
                    # 把接口的嵌套结构拍平成纯 dict 存进 JSON 列：
                    # 让前端和 Agent 不用再发一次请求就能看到每个 step 的成败。
                    run.jobs = [
                        {
                            "id": job.get("id"),
                            "name": job.get("name"),
                            "status": job.get("status"),
                            "conclusion": job.get("conclusion"),
                            "html_url": job.get("html_url"),
                            "started_at": job.get("started_at"),
                            "completed_at": job.get("completed_at"),
                            "steps": [
                                {
                                    "name": step.get("name"),
                                    "status": step.get("status"),
                                    "conclusion": step.get("conclusion"),
                                    "number": step.get("number"),
                                }
                                for step in job.get("steps", [])
                            ],
                        }
                        for job in jobs
                    ]
                except httpx.HTTPStatusError:
                    # 拉 jobs 失败保留上一次的值（首次同步则为空列表），不要覆盖成 None。
                    run.jobs = run.jobs or []
            # 只对失败的 run 下载日志：日志体积大且接口有速率限制，成功的不必拉。
            if run.conclusion == "failure" and run.github_run_id:
                try:
                    run.logs_text = await get_workflow_run_logs(repo.owner, repo.name, run.github_run_id, token=token, base_url=base_url)
                except httpx.HTTPStatusError:
                    # 日志可能还没生成完或已被清理，失败时原样保留旧值。
                    run.logs_text = run.logs_text
            synced["workflow_runs"] += 1

    # 统一落库：先 flush 把所有脏数据写进事务，再 commit（整个函数只有这一次提交）。
    # 用 UTC 记录同步时间，避免不同时区部署时判断「最近是否同步过」出现偏差。
    repo.last_sync_at = datetime.now(timezone.utc)
    db.flush()
    db.commit()
    return RepoSyncResponse(repo_id=repo.id, status="completed", synced=synced)


async def sync_repository_by_id(repo_id: UUID, payload: RepoSyncRequest) -> RepoSyncResponse | None:
    """按 ID 同步仓库，供后台定时任务等「没有请求上下文」的场景调用。

    自己建会话：调用方（如 scheduler）拿不到 FastAPI 的 get_db 依赖，
    在函数内 import SessionLocal 也能避免模块加载期就绑死数据库配置。
    仓库不存在返回 None（可能刚被删除），而不是抛异常打断批量循环。
    """
    from app.db.session import SessionLocal

    db = SessionLocal()
    try:
        repo = db.get(Repository, repo_id)
        if repo is None:
            return None
        return await sync_repository(db, repo, payload)
    finally:
        # 无论成功失败都要关会话，否则定时任务会一点点耗尽连接池。
        db.close()
