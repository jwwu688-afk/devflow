"""语义检索 HTTP 入口（挂载前缀 /api/search，最终路径 POST /api/search）。

前端「全局搜索 / 相关片段跳转」用它按自然语言在某个仓库的已索引文档里找相似片段。
真正的向量检索与元数据过滤在 app.services.rag.retrieval.search_similar_documents，
本模块只负责：校验仓库是否存在、拼装过滤条件、把结果包成响应模型。
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.models import Repository
from app.db.session import get_db
from app.schemas.search import SearchRequest, SearchResponse
from app.services.rag.retrieval import search_similar_documents

router = APIRouter()


@router.post("", response_model=SearchResponse)
async def semantic_search(payload: SearchRequest, db: Session = Depends(get_db)) -> SearchResponse:
    """在指定仓库内做语义检索。

    入参：
        payload: 查询体，含 repo_id / query / source_type / limit 以及若干元数据过滤项。
        db: 由 get_db 依赖注入的同步 Session，随请求结束由 FastAPI 关闭。
    返回：
        SearchResponse，results 是命中的文档片段列表。
    异常：
        仓库不存在时返回 404——不返回空列表，避免前端把「仓库不存在」误显示成「没搜到内容」。
    副作用：
        本接口不写库；但底层会调用 embedding 模型，并访问 Milvus（不可用时由
        main.py 注册的 MilvusUnavailableError 处理器统一转成 503 降级响应）。
    """
    repo = db.get(Repository, payload.repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到仓库")

    # 先收齐所有候选过滤维度，便于对照 Schema 增删字段；至于「哪些真正生效」交给
    # 下面的推导式判断：None / "" / 0（假值）一律被丢弃，语义等于「该维度不限制」。
    # 所以前端清空某个筛选框时直接传空串即可，不需要额外传 null。
    filters = {
        "label": payload.label,
        "path": payload.path,
        "module": payload.module,
        "state": payload.state,
        "start_date": payload.start_date,
        "end_date": payload.end_date,
    }
    results = search_similar_documents(
        db,
        payload.repo_id,
        payload.query,
        source_type=payload.source_type,
        limit=payload.limit,
        metadata_filters={key: value for key, value in filters.items() if value},
    )
    return SearchResponse(results=results)
