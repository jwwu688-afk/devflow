"""数据库会话管理：提供 SQLAlchemy 引擎、声明式基类与会话工厂。

本文件是「数据层入口」，models.py 里的 ORM 类都继承此处的 Base，
所有依赖注入（Depends(get_db)）最终都落到这里的 SessionLocal 上。
"""

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import settings


class Base(DeclarativeBase):
    """所有 ORM 模型的公共基类（SQLAlchemy 2.0 风格）。

    继承它之后，模型类才具备表名解析、字段映射与 metadata 收集能力，
    建表脚本 / 迁移工具也是通过 Base.metadata 拿到全部表定义的。
    """
    pass


# pool_pre_ping=True：从连接池取连接前先探活，避免 MySQL/PG 侧 idle 超时后拿到失效连接报错，
# 代价是每次取连接多一次往返，对本项目的并发量可以接受。
engine = create_engine(settings.database_url, pool_pre_ping=True)
# 创建会话工厂
# autoflush=False：查询前不自动 flush 未提交的改动，避免隐式写库导致难以排查的行为；
# autocommit=False：显式 commit，让事务边界由业务代码掌握。
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

# 提供数据库会话的依赖函数
def get_db() -> Generator[Session, None, None]:
    """FastAPI 依赖：每个请求一个独立 Session，请求结束后必定关闭。

    用 yield 而不是 return，是为了让 FastAPI 在响应完成后回到 finally 执行 close，
    从而把连接归还连接池——否则高并发下很快会耗尽连接。
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        # 放在 finally 保证即使路由抛异常，连接也不会泄漏
        db.close()
