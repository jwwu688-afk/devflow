"""后台定时任务：周期性同步仓库。

服务启动后由 main 里挂一个 asyncio task 长期跑；它不依赖请求上下文，
所以数据库会话必须自己开关，不能复用 FastAPI 的 get_db 依赖。
"""

import asyncio
import logging

from app.core.config import settings
from app.db.models import Repository
from app.db.session import SessionLocal
from app.schemas.repos import RepoSyncRequest
from app.services.repos_sync import sync_repository_by_id

logger = logging.getLogger(__name__)


async def periodic_repo_sync_loop() -> None:
    """无限循环地按固定间隔拉取所有 GitHub 类仓库的变更。

    单个仓库同步失败只记 warning 不中断，避免一个坏仓库（token 失效、
    仓库被删）把整条定时链路打死；下一轮仍会重试它。
    """
    # 启动后先静默 5 秒：等服务完成初始化（DB 建表、配置加载）再查库，
    # 否则冷启动瞬间容易拿到「表不存在」的报错。
    await asyncio.sleep(5)
    # 每轮都重新判断开关，运维改配置后无需重启即可停掉后台同步。
    while settings.auto_sync_enabled:
        # 只在本轮查询期间持有连接，查完立刻归还连接池，
        # 因为后面逐个仓库同步可能耗时很久，不能一直占着连接。
        db = SessionLocal()
        try:
            repos = db.query(Repository).filter(Repository.provider.in_(["github", "github_compatible"])).all()
            repo_ids = [repo.id for repo in repos]
        finally:
            db.close()

        # 串行同步：并发拉取会放大 GitHub 的速率限制（rate limit）风险。
        for repo_id in repo_ids:
            try:
                await sync_repository_by_id(repo_id, RepoSyncRequest(limit=settings.auto_sync_limit))
            except Exception as exc:
                logger.warning("Periodic repo sync failed for %s: %s", repo_id, exc)

        # 下限 30 秒：防止配置被误填成 0 或负数导致死循环式狂刷接口。
        await asyncio.sleep(max(settings.auto_sync_interval_seconds, 30))
