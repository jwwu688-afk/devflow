"""GitHub Webhook 接收端（挂载前缀 /api/webhooks，入口 POST /api/webhooks/github）。

职责：校验签名 → 从 payload 里认出是哪个已接入的仓库 → 触发一次增量同步。
设计要点：
    1. 同步是重活（拉 issue/PR/CI），必须丢进 BackgroundTasks，接口本身要秒回，
       否则 GitHub 会因 10 秒超时判失败并反复重投。
    2. 认不出的仓库/事件不报错，一律 200 + accepted 字段，详见下方注释。
    3. 本模块不注入 get_db：Webhook 请求和前端登录态无关，用独立的 SessionLocal。
"""

import hashlib
import hmac

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Request
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import Repository
from app.db.session import SessionLocal
from app.schemas.repos import RepoSyncRequest
from app.services.repos_sync import sync_repository_by_id

router = APIRouter()


def verify_signature(body: bytes, signature: str | None) -> None:
    """用共享密钥校验 GitHub 的 HMAC 签名，防止伪造推送。

    入参：
        body: 原始请求体字节。必须用「未解析」的原文，否则 JSON 重新序列化后
            空白/键序变化会导致签名对不上。
        signature: X-Hub-Signature-256 头，形如 "sha256=<hex>"。
    返回：无（通过即静默返回）。
    异常：缺头或签名不匹配时抛 401。
    说明：
        若 settings.github_webhook_secret 未配置则直接放行——本地开发不想为了收
        webhook 去配密钥；但这也意味着生产环境必须配好该值，否则接口等于裸奔。
    """
    if not settings.github_webhook_secret:
        return
    if not signature or not signature.startswith("sha256="):
        raise HTTPException(status_code=401, detail="缺少 GitHub webhook 签名")
    expected = hmac.new(settings.github_webhook_secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    actual = signature.removeprefix("sha256=")
    # 用 compare_digest 而不是 ==，避免按字节比较带来的时序侧信道（攻击者可逐位试探）。
    if not hmac.compare_digest(expected, actual):
        raise HTTPException(status_code=401, detail="GitHub webhook 签名无效")


def find_repo_by_full_name(full_name: str) -> Repository | None:
    """按 owner/repo 全名查仓库，查不到返回 None。

    入参：full_name，来自 payload.repository.full_name。
    返回：Repository 实例或 None。
    副作用：自建并关闭 Session——因为这里不在请求依赖链上，没有 get_db 帮忙收尾；
        手动 try/finally 关闭可避免连接池被慢慢耗尽。
    """
    db: Session = SessionLocal()
    try:
        return db.query(Repository).filter(Repository.full_name == full_name).one_or_none()
    finally:
        db.close()


@router.post("/github")
async def github_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_github_event: str | None = Header(default=None),
    x_hub_signature_256: str | None = Header(default=None),
) -> dict:
    """POST /api/webhooks/github：接收 GitHub 事件并调度仓库同步。

    入参：
        request: 用来读取原始 body（先验签，再解析 JSON，顺序不能反）。
        background_tasks: FastAPI 注入，用于把同步任务挪到响应之后执行。
        x_github_event: GitHub 事件类型头（push / issues / pull_request ...）。
        x_hub_signature_256: 签名头，见 verify_signature。
    返回：
        dict，形如 {"accepted": bool, "event": ..., "repo_id": ..., "action": ...}。
        注意：未接入的仓库、缺字段、无关事件都返回 200 + accepted=false，
        这是刻意为之——返回 4xx/5xx 会让 GitHub 认为投递失败并持续重试。
    异常：仅签名校验失败时 401。
    副作用：命中白名单事件时，后台异步执行 sync_repository_by_id（会写库、调 GitHub API）。
    """
    body = await request.body()
    verify_signature(body, x_hub_signature_256)
    payload = await request.json()
    repository = payload.get("repository") or {}
    full_name = repository.get("full_name")
    if not full_name:
        return {"accepted": False, "reason": "payload.repository.full_name is missing"}

    repo = find_repo_by_full_name(full_name)
    if repo is None:
        return {"accepted": False, "reason": f"{full_name} is not connected"}

    # 只有会影响同步数据的事件才触发同步，其余（如 star、fork）记 action=ignored 直接丢弃，
    # 避免无意义地打满 GitHub API 配额。
    if x_github_event in {"issues", "pull_request", "workflow_run", "push", "check_suite", "check_run"}:
        background_tasks.add_task(sync_repository_by_id, repo.id, RepoSyncRequest(limit=settings.auto_sync_limit))
        return {"accepted": True, "event": x_github_event, "repo_id": str(repo.id), "action": "sync_scheduled"}

    return {"accepted": True, "event": x_github_event, "repo_id": str(repo.id), "action": "ignored"}
