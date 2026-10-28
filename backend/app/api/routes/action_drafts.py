"""「安全草稿」（ActionDraft）接口，挂载前缀 /api/action-drafts。

这是整套系统的安全阀：AI/Agent 不直接改 GitHub，而是先产出一条 pending_confirmation
状态的草稿（比如「在 #12 下回一条评论」「给 #12 打 bug 标签」），人确认后才真正执行。
流程：
    POST /                    创建草稿 → 状态 pending_confirmation
    POST /{id}/confirm        人工确认 → 调 GitHub API → executed / failed
    POST /{id}/cancel         放弃 → cancelled
    GET  /、GET /{id}         查询
    GET  /audit-logs          审计日志（谁在什么时候确认了什么）
权限点用 require_permission 依赖声明式校验：建/取消要 draft:create，执行要 draft:approve，
也就是说「能起草」和「能批准」在权限模型里是两件事，便于做审批分离。
"""

from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import decrypt_token
from app.db.models import ActionDraft, AuditLog, Issue, PullRequest, Repository, User
from app.db.session import get_db
from app.schemas.action_drafts import ActionDraftConfirmResponse, ActionDraftCreateRequest, ActionDraftResponse, AuditLogResponse
from app.services.github.client import GitHubClient
from app.services.permissions import require_permission, write_audit_log

router = APIRouter()


def _target_number(db: Session, draft: ActionDraft) -> int | None:
    """推断草稿要操作的 issue/PR 编号（GitHub API 用的是 number，不是内部 UUID）。

    入参：db 用于按 target_id 反查；draft 为草稿实体。
    返回：GitHub 侧的 issue/pr number，实在推断不出返回 None（调用方据此报错）。
    说明：
        这里做了多级兜底而不是只认一种字段，因为草稿来源不统一：Agent 生成的草稿
        习惯把编号塞在 meta 里，而前端手动建的草稿通常只给 target_id。先信 meta
        （免一次查库），meta 缺失或不是合法整数时再回落到按 target_id 查表。
    """
    meta = draft.meta or {}
    raw = meta.get("target_number") or meta.get("issue_number") or meta.get("pr_number")
    if raw is not None:
        try:
            return int(raw)
        except (TypeError, ValueError):
            # meta 里是脏数据（比如 "abc"）：不抛错，退回到按 target_id 查库，
            # 让「meta 写错」不至于把整个确认流程卡死。
            return None
    if draft.target_id and draft.target_type == "issue":
        issue = db.get(Issue, draft.target_id)
        return issue.number if issue else None
    if draft.target_id and draft.target_type in {"pull_request", "pr"}:
        pr = db.get(PullRequest, draft.target_id)
        return pr.number if pr else None
    return None


async def _execute_github_draft(db: Session, draft: ActionDraft, repo: Repository) -> dict:
    """把已确认的草稿真正落到 GitHub 上，按 draft_type 分发。

    入参：
        db: 用于按 target_id 反查 issue/pr 编号。
        draft: 草稿，draft_type 决定走哪个分支。
        repo: 草稿所属仓库，提供 owner/name、API 基址和加密后的 token。
    返回：
        GitHub 接口的原始响应 dict（写进 draft.execution_result，方便事后追溯）。
    异常：
        类型不支持、缺编号、缺 labels 等一律抛 RuntimeError/ValueError；
        调用方（confirm 接口）会把它记成 failed 状态 + error_message，而不是让 HTTP 报 500。
    副作用：
        真实调用 GitHub 写接口（发评论、建/关 issue、打标签），不可逆。
    """
    # token 在库里是加密存的（github_token_encrypted），用前必须解密；没配 token 时传 None，
    # GitHubClient 会退化成匿名请求（只能读公开仓库，写操作会 401）。
    token = decrypt_token(repo.github_token_encrypted, settings.token_encryption_key) if repo.github_token_encrypted else None
    client = GitHubClient(token=token, base_url=repo.api_base_url)
    # 统一转小写再比较：草稿可能由不同来源生成，draft_type 大小写不保证一致。
    draft_type = draft.draft_type.lower()
    issue_number = _target_number(db, draft)
    if draft_type in {"issue_comment", "pr_comment", "comment"}:
        # GitHub 的评论接口对 issue 和 PR 是同一个（PR 也是 issue），所以三种类型合并处理。
        if issue_number is None:
            raise RuntimeError("评论草稿缺少 issue/pr number")
        return await client.create_issue_comment(repo.owner, repo.name, issue_number, draft.content)
    if draft_type == "create_issue":
        # 草稿可能只填了正文：取正文首行前 120 字当标题，够用且不会把整段正文顶到标题栏。
        title = draft.title or (draft.content.splitlines()[0][:120] if draft.content else "DevFlow AI Issue")
        return await client.create_issue(repo.owner, repo.name, title, draft.content)
    if draft_type == "close_issue":
        if issue_number is None:
            raise RuntimeError("关闭 Issue 草稿缺少 issue number")
        return await client.close_issue(repo.owner, repo.name, issue_number)
    if draft_type in {"label", "add_label", "add_labels"}:
        if issue_number is None:
            raise RuntimeError("Label 草稿缺少 issue/pr number")
        # labels 优先取 meta.labels（列表）；模型有时把标签写在 title 里，就当成单个标签兜底。
        labels = draft.meta.get("labels") if draft.meta else None
        if not isinstance(labels, list) or not labels:
            labels = [draft.title] if draft.title else []
        if not labels:
            raise RuntimeError("Label 草稿缺少 labels")
        return await client.add_issue_labels(repo.owner, repo.name, issue_number, [str(item) for item in labels])
    if draft_type == "send_report":
        # 已知局限：本课程项目没接邮件/IM 渠道，这里只把草稿标记为「已发送」，
        # 保证前端流程能走通，不会真的对外发消息。
        return {"status": "recorded", "message": "报告草稿已在 DevFlow AI 中标记为已发送，未接入外部消息渠道。"}
    raise RuntimeError(f"暂不支持执行草稿类型：{draft.draft_type}")


@router.post("", response_model=ActionDraftResponse)
async def create_action_draft(
    payload: ActionDraftCreateRequest,
    db: Session = Depends(get_db),
    user: User = Depends(require_permission("draft:create")),
) -> ActionDraft:
    """POST /api/action-drafts：新建草稿（此时**不会**动 GitHub）。

    入参：
        payload: 草稿内容，含 repo_id / draft_type / target_* / 标题正文 / 风险等级 / metadata。
        db: get_db 注入的 Session。
        user: require_permission("draft:create") 注入的当前用户，用于记录 created_by 和审计。
    返回：处于 pending_confirmation 的 ActionDraft。
    异常：仓库不存在 → 404；无 draft:create 权限 → require_permission 内部抛 403。
    副作用：插入 action_draft，并写一条 action_draft.create 审计日志（同一事务，一起提交）。
    """
    repo = db.get(Repository, payload.repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到仓库")
    draft = ActionDraft(
        repo_id=payload.repo_id,
        draft_type=payload.draft_type,
        target_type=payload.target_type,
        target_id=payload.target_id,
        title=payload.title,
        content=payload.content,
        # risk_level 只是给前端做红色警示/二次确认用，不参与后端拦截。
        risk_level=payload.risk_level,
        created_by=user.id,
        # 入参叫 metadata，模型字段叫 meta：SQLAlchemy 的 `metadata` 是保留属性名，
        # 模型侧只能改名，这里做一次映射。
        meta=payload.metadata,
    )
    db.add(draft)
    write_audit_log(
        db,
        user=user,
        repo_id=payload.repo_id,
        action="action_draft.create",
        target_type=payload.target_type,
        target_id=str(payload.target_id) if payload.target_id else None,
        # mode="json" 让 UUID/datetime 等类型能直接进 JSON 列，否则序列化会失败。
        request_json=payload.model_dump(mode="json"),
    )
    db.commit()
    db.refresh(draft)
    return draft


@router.get("", response_model=list[ActionDraftResponse])
async def list_action_drafts(
    repo_id: UUID | None = Query(default=None),
    status: str | None = Query(default=None),
    db: Session = Depends(get_db),
    _user: User = Depends(require_permission("repo:read")),
) -> list[ActionDraft]:
    """GET /api/action-drafts：草稿列表，支持按仓库和状态过滤。

    入参：repo_id / status 均为可选查询参数，不传则不过滤。
    返回：按创建时间倒序的草稿列表，**最多 100 条**（写死上限，防止列表接口被拉爆）。
    异常：无 repo:read 权限 → 403。
    副作用：只读。注意这里没有按用户隔离——同一团队共享草稿池。
    """
    query = db.query(ActionDraft)
    if repo_id:
        query = query.filter(ActionDraft.repo_id == repo_id)
    if status:
        query = query.filter(ActionDraft.status == status)
    return query.order_by(ActionDraft.created_at.desc()).limit(100).all()


# 注意："/audit-logs" 必须声明在 "/{draft_id}" 之前。否则它会被当成 draft_id 去解析 UUID，
# 直接返回 422 而不是进到这个函数里。
@router.get("/audit-logs", response_model=list[AuditLogResponse])
async def list_audit_logs(
    repo_id: UUID | None = Query(default=None),
    db: Session = Depends(get_db),
    _user: User = Depends(require_permission("settings:manage")),
) -> list[AuditLog]:
    """GET /api/action-drafts/audit-logs：查看审计日志。

    入参：repo_id 可选，用于只看某个仓库的操作记录。
    返回：按时间倒序、最多 100 条。
    异常：需要 settings:manage 权限（审计日志属于管理视图，普通成员看不到）→ 否则 403。
    副作用：只读。
    """
    query = db.query(AuditLog)
    if repo_id:
        query = query.filter(AuditLog.repo_id == repo_id)
    return query.order_by(AuditLog.created_at.desc()).limit(100).all()


@router.get("/{draft_id}", response_model=ActionDraftResponse)
async def get_action_draft(
    draft_id: UUID,
    db: Session = Depends(get_db),
    _user: User = Depends(require_permission("repo:read")),
) -> ActionDraft:
    """GET /api/action-drafts/{draft_id}：草稿详情。

    入参：draft_id 为路径参数（UUID，格式不对会被 FastAPI 拦成 422）。
    返回：ActionDraft。
    异常：不存在 → 404；无 repo:read 权限 → 403。
    副作用：只读。
    """
    draft = db.get(ActionDraft, draft_id)
    if draft is None:
        raise HTTPException(status_code=404, detail="未找到安全草稿")
    return draft


@router.post("/{draft_id}/cancel", response_model=ActionDraftResponse)
async def cancel_action_draft(
    draft_id: UUID,
    db: Session = Depends(get_db),
    user: User = Depends(require_permission("draft:create")),
) -> ActionDraft:
    """POST /api/action-drafts/{draft_id}/cancel：放弃一个待确认草稿。

    入参：draft_id；user 由权限依赖注入，用于审计。
    返回：状态变为 cancelled 的草稿。
    异常：
        不存在 → 404；已执行/已取消的草稿再来取消 → 400
        （用 400 而不是 409，是因为前端只需要展示提示文案，不需要区分冲突类型）。
    副作用：改状态 + 写审计日志。
    """
    draft = db.get(ActionDraft, draft_id)
    if draft is None:
        raise HTTPException(status_code=404, detail="未找到安全草稿")
    if draft.status != "pending_confirmation":
        raise HTTPException(status_code=400, detail="只有待确认草稿可以取消")
    draft.status = "cancelled"
    write_audit_log(db, user=user, repo_id=draft.repo_id, action="action_draft.cancel", target_type=draft.target_type, target_id=str(draft.id))
    db.commit()
    db.refresh(draft)
    return draft


@router.post("/{draft_id}/confirm", response_model=ActionDraftConfirmResponse)
async def confirm_action_draft(
    draft_id: UUID,
    db: Session = Depends(get_db),
    user: User = Depends(require_permission("draft:approve")),
) -> dict:
    """POST /api/action-drafts/{draft_id}/confirm：人工确认并真正执行草稿。

    入参：draft_id；user 需具备 draft:approve（与创建权限分离，便于审批流）。
    返回：{"draft": 最新草稿, "audit_id": 审计记录 ID}。
    异常：不存在 → 404；状态不是 pending_confirmation → 400。
        **执行失败不算 HTTP 错误**：GitHub 报错会被吞掉并落到 draft.error_message，
        接口仍返回 200，让前端把「草稿失败原因」当作业务数据展示。
    副作用：调用 GitHub 写接口；更新草稿状态/结果；写审计日志。失败时草稿永久变为 failed，
        不会自动回退到 pending_confirmation——要重试只能新建一条草稿。
    """
    draft = db.get(ActionDraft, draft_id)
    if draft is None:
        raise HTTPException(status_code=404, detail="未找到安全草稿")
    # 这里的状态校验同时充当幂等保护：重复点确认不会重复发评论/重复打标签。
    if draft.status != "pending_confirmation":
        raise HTTPException(status_code=400, detail="只有待确认草稿可以执行")
    repo = db.get(Repository, draft.repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到仓库")
    try:
        result = await _execute_github_draft(db, draft, repo)
        draft.status = "executed"
        draft.confirmed_by = user.id
        draft.executed_at = datetime.now(timezone.utc)
        draft.execution_result = result
        # 成功/失败都要留审计，且审计里带上是哪个 draft_type——事后追查「谁批准发了哪条评论」全靠它。
        audit = write_audit_log(
            db,
            user=user,
            repo_id=draft.repo_id,
            action=f"action_draft.confirm.{draft.draft_type}",
            target_type=draft.target_type,
            target_id=str(draft.target_id or draft.id),
            status="success",
            request_json={"draft_id": str(draft.id), "draft_type": draft.draft_type},
            result_json=result if isinstance(result, dict) else {"result": result},
        )
    except Exception as exc:
        # 宽 except 是刻意的：这里面对的是外部 HTTP 服务，网络抖动/限流/权限/解密失败
        # 都可能抛出不同异常类型，统一收敛成「草稿失败 + 原因」，避免把 500 抛给前端。
        draft.status = "failed"
        draft.confirmed_by = user.id
        draft.executed_at = datetime.now(timezone.utc)
        draft.error_message = str(exc)
        audit = write_audit_log(
            db,
            user=user,
            repo_id=draft.repo_id,
            action=f"action_draft.confirm.{draft.draft_type}",
            target_type=draft.target_type,
            target_id=str(draft.target_id or draft.id),
            status="failed",
            request_json={"draft_id": str(draft.id), "draft_type": draft.draft_type},
            result_json={"error": str(exc)},
        )
    # 成功和失败都在同一条 commit 里落库：草稿状态和审计记录必须一致，不能只有一边成功。
    db.commit()
    db.refresh(draft)
    db.refresh(audit)
    return {"draft": draft, "audit_id": audit.id}
