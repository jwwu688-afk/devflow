"""仓库下的「会话」（Conversation）管理接口，挂在 /api/repos 前缀下。

一个仓库可以有多个会话，聊天记录、证据、Agent 运行记录都按 conversation_id 隔离。
本模块只做 HTTP 层：校验仓库是否存在 → 调 app.services.conversations → 提交事务。
注意所有写操作都在这里显式 commit，service 层只用 flush（只发 SQL、不提交），
这样「一次请求一个事务」的边界清晰，出错时整体回滚。
"""

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.models import Repository
from app.db.session import get_db
from app.schemas.conversations import ConversationCreateRequest, ConversationResponse
from app.services.conversations import create_conversation, delete_conversation, list_conversations

router = APIRouter()


def _repo_or_404(db: Session, repo_id: uuid.UUID) -> Repository:
    """取仓库，不存在则 404。

    三个接口都要先确认仓库存在：否则 list_conversations 会自动给一个不存在的
    repo_id 建出「默认会话」，脏数据就这么来的。
    """
    repo = db.get(Repository, repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到仓库")
    return repo


@router.get("/{repo_id}/conversations", response_model=list[ConversationResponse])
async def get_conversations(repo_id: uuid.UUID, db: Session = Depends(get_db)) -> list[Any]:
    """GET /api/repos/{repo_id}/conversations：列出该仓库的活跃会话。

    入参：repo_id（路径）；db 由 get_db 注入。
    返回：按 updated_at 倒序的会话列表（最近用过的排前面）。
    异常：仓库不存在 → 404。
    副作用：**这是带写入的 GET**——list_conversations 内部会兜底创建「默认会话」，
        并把历史遗留数据回填到当前会话下，所以这里必须 commit，不能只读。
    """
    _repo_or_404(db, repo_id)
    conversations = list_conversations(db, repo_id)
    db.commit()
    return conversations


@router.post("/{repo_id}/conversations", response_model=ConversationResponse)
async def add_conversation(
    repo_id: uuid.UUID,
    payload: ConversationCreateRequest | None = None,
    db: Session = Depends(get_db),
) -> Any:
    """POST /api/repos/{repo_id}/conversations：新建会话。

    入参：
        repo_id: 仓库 ID。
        payload: 可选，只带 title；不传或标题为空时由 service 生成「默认会话 / 新会话 N」。
    返回：新建的 Conversation（含数据库生成的 id / 时间戳）。
    异常：仓库不存在 → 404。
    副作用：插入一行 conversation 并提交。
    """
    _repo_or_404(db, repo_id)
    conversation = create_conversation(db, repo_id, payload.title if payload else None)
    db.commit()
    # commit 之后对象会过期（expire_on_commit），必须 refresh 才能安全读取服务端默认值。
    db.refresh(conversation)
    return conversation


@router.delete("/{repo_id}/conversations/{conversation_id}")
async def remove_conversation(
    repo_id: uuid.UUID,
    conversation_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """DELETE /api/repos/{repo_id}/conversations/{conversation_id}：删除会话。

    入参：repo_id、conversation_id（均为路径参数）。
    返回：手拼的 dict，而不是 response_model —— 因为除了被删 ID，还要顺手回传
        「剩下的会话列表 + 应该切换到哪个会话」，前端删完立刻就能渲染，省一次请求。
    异常：仓库不存在 → 404；会话不存在或被重复删除时**不报 404**，
        service 会退化成「确保有一个默认会话」，保证前端永远有会话可用。
    副作用：软删除（status 置 deleted）并级联清理消息、证据、Agent 运行记录等关联表；
        若删的是最后一个会话，会自动补一个新的默认会话。
    """
    _repo_or_404(db, repo_id)
    replacement = delete_conversation(db, repo_id, conversation_id)
    conversations = list_conversations(db, repo_id)
    db.commit()
    # 这里手动序列化而非依赖 response_model：字段名与 ConversationResponse 保持一致，
    # 但多包了一层 active_conversation_id，供前端直接切到替代会话。
    return {
        "deleted": str(conversation_id),
        "active_conversation_id": str(replacement.id),
        "conversations": [
            {
                "id": str(item.id),
                "repo_id": str(item.repo_id),
                "title": item.title,
                "status": item.status,
                "message_count": item.message_count,
                "created_at": item.created_at.isoformat() if item.created_at else None,
                "updated_at": item.updated_at.isoformat() if item.updated_at else None,
            }
            for item in conversations
        ],
    }
