"""轻量级权限与审计。

当前是「单用户 demo」阶段：没有登录体系，所有请求都归到最后那个 demo 用户，
因此这里的角色判断只是给将来的多租户留好接入点，不构成真正的安全边界。
"""

from collections.abc import Callable

from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.models import AuditLog, User
from app.db.session import get_db

# 角色 → 权限点集合。用「一个权限一个字符串」而不是布尔开关，
# 好处是新增接口时只要声明 require_permission("xxx")，不必改动角色表结构。
# 注意这是累加式的白名单：子角色不会自动继承父角色，缺哪个就显式写哪个。
ROLE_PERMISSIONS: dict[str, set[str]] = {
    "owner": {"repo:read", "repo:sync", "agent:run", "draft:create", "draft:approve", "eval:read", "settings:manage"},
    "maintainer": {"repo:read", "repo:sync", "agent:run", "draft:create", "draft:approve", "eval:read"},
    "developer": {"repo:read", "agent:run", "draft:create", "eval:read"},
    "viewer": {"repo:read", "eval:read"},
}


def ensure_demo_user(db: Session) -> User:
    """取（或创建）demo 阶段唯一的内置用户。

    取创建时间最早的那条：这样数据库里即使混进了其他用户，
    也不会因为排序抖动导致「当前用户」在请求之间来回跳。
    """
    user = db.query(User).order_by(User.created_at.asc()).first()
    if user is None:
        # 库里一个人都没有时补一个 owner，保证任何接口第一次被调用都不会 403。
        user = User(name="demo-owner", role="owner")
        db.add(user)
        db.commit()
        db.refresh(user)
    if not getattr(user, "role", None):
        # 兼容历史数据：老表里 role 可能是 NULL，这里就地补成 owner。
        # getattr 而非直接 user.role，是为了容错 role 字段尚未迁移的情况。
        user.role = "owner"
        db.commit()
        db.refresh(user)
    return user


def has_permission(user: User, permission: str) -> bool:
    """判断用户是否拥有某个权限点。

    role 为空或拼错时按 viewer 处理（最小权限兜底），
    宁可少给权限也不要因为脏数据放行。
    """
    return permission in ROLE_PERMISSIONS.get((user.role or "viewer").lower(), set())


def require_permission(permission: str) -> Callable:
    """生成一个 FastAPI 依赖，用于在路由上一次性完成「取用户 + 校验权限」。

    以「工厂函数」形式提供，是因为权限点是编译期常量、用户是请求期变量；
    把后者包成 dependency，路由只要写 Depends(require_permission("repo:sync")) 即可。
    """

    def dependency(db: Session = Depends(get_db)) -> User:
        user = ensure_demo_user(db)
        if not has_permission(user, permission):
            raise HTTPException(status_code=403, detail=f"需要权限：{permission}")
        return user

    return dependency


def write_audit_log(
    db: Session,
    *,
    user: User | None,
    repo_id=None,
    action: str,
    target_type: str | None = None,
    target_id: str | None = None,
    status: str = "success",
    request_json: dict | None = None,
    result_json: dict | None = None,
) -> AuditLog:
    """往审计表插一条记录并 flush（不 commit）。

    刻意只 flush 不 commit：审计要和业务写入落在同一个事务里，
    由调用方统一提交，避免出现「业务回滚了但审计已落库」的假记录。
    关键字参数用 * 强制，防止调用方按位置传参把 status 传串位。
    """
    row = AuditLog(
        user_id=user.id if user else None,
        repo_id=repo_id,
        action=action,
        target_type=target_type,
        target_id=target_id,
        status=status,
        # JSON 列不接受 None，统一落成空字典，读的时候就不用判空。
        request_json=request_json or {},
        result_json=result_json or {},
    )
    db.add(row)
    db.flush()
    return row
