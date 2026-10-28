"""仓库「项目索引」（memory-index）的三个接口，挂在 /api/repos 前缀下。

项目索引是给 AI 用的仓库级项目记忆：扫描一遍目录结构/关键文件，生成概览，
后续对话把这份概览当上下文。这里只做三件事：
    1. 查状态（GET .../state）——前端用来决定要不要弹「检测到新仓库，是否建立索引？」；
    2. 手动触发扫描（POST .../scan）——重活丢后台，立刻回 202；
    3. 稍后提醒（POST .../snooze）——用户点「以后再说」，写入 snooze 截止时间。
具体扫描与状态机在 app.services.project_indexing。
"""

import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.models import Repository
from app.db.session import get_db
from app.schemas.project_index import (
    ProjectIndexScanResponse,
    ProjectIndexSnoozeRequest,
    ProjectIndexSnoozeResponse,
    ProjectIndexStateResponse,
)
from app.services.project_indexing import (
    get_project_index_state,
    mark_project_index_building,
    scan_project_index_by_id,
    snooze_project_index,
)

router = APIRouter()


def _repo_or_404(db: Session, repo_id: uuid.UUID) -> Repository:
    """取仓库，不存在则抛 404。

    抽成小函数是因为三个接口都要这一步；顺带保证「先确认仓库存在，再动索引状态」，
    避免给不存在的 repo_id 写出一条孤儿状态记录。
    """
    repo = db.get(Repository, repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到仓库")
    return repo


@router.get("/{repo_id}/memory-index/state", response_model=ProjectIndexStateResponse)
async def get_memory_index_state(repo_id: uuid.UUID, db: Session = Depends(get_db)) -> dict:
    """GET /api/repos/{repo_id}/memory-index/state：读取索引状态。

    入参：repo_id 为路径参数；db 由 get_db 注入。
    返回：ProjectIndexStateResponse 对应的 dict（状态、上次扫描时间、snooze 截止时间等）。
    异常：仓库不存在 → 404。
    副作用：只读。
    """
    repo = _repo_or_404(db, repo_id)
    return get_project_index_state(db, repo)


@router.post("/{repo_id}/memory-index/scan", response_model=ProjectIndexScanResponse, status_code=202)
async def start_memory_index_scan(
    repo_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> ProjectIndexScanResponse:
    """POST /api/repos/{repo_id}/memory-index/scan：触发一次重建索引。

    入参：repo_id；background_tasks 由 FastAPI 注入。
    返回：ProjectIndexScanResponse(started=True, status="building")。
    异常：仓库不存在 → 404。
    副作用：先把状态改成 building 再排后台任务——这样前端在任务真正开跑前
        轮询 state 就能立刻看到 building，不会出现「点了没反应」的空窗；
        后台任务会读仓库文件并写 documents/索引表。

    用 202 而不是 200：语义上表示「已受理但尚未完成」，前端据此走轮询而不是等待响应体。
    """
    repo = _repo_or_404(db, repo_id)
    mark_project_index_building(db, repo)
    background_tasks.add_task(scan_project_index_by_id, repo.id)
    return ProjectIndexScanResponse(repo_id=repo.id, started=True, status="building")


@router.post("/{repo_id}/memory-index/snooze", response_model=ProjectIndexSnoozeResponse)
async def snooze_memory_index(
    repo_id: uuid.UUID,
    payload: ProjectIndexSnoozeRequest | None = None,
    db: Session = Depends(get_db),
) -> ProjectIndexSnoozeResponse:
    """POST /api/repos/{repo_id}/memory-index/snooze：把建索引提醒推迟若干天。

    入参：
        repo_id: 仓库 ID。
        payload: 可选请求体；不传（前端只点「稍后」）时按 7 天处理，见下方常量说明。
    返回：ProjectIndexSnoozeResponse，含 snoozed_until 截止时间。
    异常：仓库不存在 → 404。
    副作用：更新索引状态行的 snoozed_until 字段。
    """
    repo = _repo_or_404(db, repo_id)
    # 7 是「以后再说」的默认静默期：不带 body 直接 POST 也能用，
    # 兼顾最简单的按钮调用和需要自定义天数的调用方。
    row = snooze_project_index(db, repo, days=(payload.days if payload else 7))
    return ProjectIndexSnoozeResponse(repo_id=repo.id, snoozed_until=row.snoozed_until)
