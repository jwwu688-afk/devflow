"""代码图谱（符号 + 关系）查询接口，挂在 /api 前缀下，路径都以 /repos/{repo_id}/ 开头。

三件事：
    POST /repos/{id}/code-graph/rebuild   重新解析仓库源码，落 code_symbols / code_relations
    GET  /repos/{id}/code-symbols         按路径/PR 列符号
    GET  /repos/{id}/code-graph/search    按关键字搜符号，并顺手带出关系边和代码片段

code_graph 支撑前端的「代码地图 / 影响面分析」视图：符号是从源码里抽出来的函数/类，
关系边是调用、导入这类连线。所有接口都要 repo:read 权限。
"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.db.models import CodeRelation, CodeSymbol, PullRequest, Repository, User
from app.db.session import get_db
from app.schemas.code_graph import CodeGraphSearchResponse, CodeSymbolResponse
from app.services.code_analysis import sync_repository_code_analysis
from app.services.code_search import read_code_excerpt, repository_checkout_path
from app.services.permissions import require_permission
from app.services.worktree_manager import pr_worktree_path

router = APIRouter()


@router.post("/repos/{repo_id}/code-graph/rebuild")
async def rebuild_code_graph(
    repo_id: UUID,
    db: Session = Depends(get_db),
    _user: User = Depends(require_permission("repo:read")),
) -> dict:
    """POST /api/repos/{repo_id}/code-graph/rebuild：重建代码图谱。

    入参：repo_id（路径）；db / 当前用户由依赖注入（用户只用于权限校验，故命名 _user）。
    返回：{"repo_id": ..., "indexed_symbols": 符号数}。
    异常：仓库不存在 → 404；无 repo:read → 403。
    副作用：**同步**执行整套解析（遍历文件、抽取符号、写两张表），请求会一直挂到大文件
        解析完才返回；仓库很大时前端需要自己加 loading 或超时提示。
    """
    repo = db.get(Repository, repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到仓库")
    symbol_count = await sync_repository_code_analysis(db, repo)
    return {"repo_id": str(repo.id), "indexed_symbols": symbol_count}


@router.get("/repos/{repo_id}/code-symbols", response_model=list[CodeSymbolResponse])
async def list_code_symbols(
    repo_id: UUID,
    path: str | None = None,
    pr_id: UUID | None = None,
    pr_number: int | None = None,
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db),
    _user: User = Depends(require_permission("repo:read")),
) -> list[CodeSymbol]:
    """GET /api/repos/{repo_id}/code-symbols：列出代码符号。

    入参：
        path: 文件路径的**子串**（不是前缀匹配），方便按目录/文件名粗筛。
        pr_id / pr_number: 只看某个 PR 的符号快照（PR 分析场景用），二者可任选其一。
        limit: 1~500，默认 100；上限写死 500 防止一次拉爆整个索引。
    返回：按 (path, start_line) 升序的符号列表，保证前端渲染顺序稳定。
    异常：无 repo:read → 403（不单独校验仓库是否存在：查不到符号就是空列表）。
    副作用：只读。
    """
    query = db.query(CodeSymbol).filter(CodeSymbol.repo_id == repo_id)
    if path:
        query = query.filter(CodeSymbol.path.contains(path))
    if pr_id:
        query = query.filter(CodeSymbol.pr_id == pr_id)
    if pr_number:
        query = query.filter(CodeSymbol.pr_number == pr_number)
    return query.order_by(CodeSymbol.path.asc(), CodeSymbol.start_line.asc()).limit(limit).all()


@router.get("/repos/{repo_id}/code-graph/search", response_model=CodeGraphSearchResponse)
async def search_code_graph(
    repo_id: UUID,
    q: str = Query(default=""),
    pr_id: UUID | None = None,
    pr_number: int | None = None,
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
    _user: User = Depends(require_permission("repo:read")),
) -> dict:
    """GET /api/repos/{repo_id}/code-graph/search：代码图谱搜索（前端代码地图的主查询）。

    入参：
        q: 关键字，同时匹配符号名和文件路径；留空则不过滤（等于「列前 N 个符号」）。
        pr_id / pr_number: 指定 PR 时只看该 PR 的快照，代码片段也从 PR 的 worktree 里读。
        limit: 符号条数，1~100，默认 20。
    返回：
        {"query", "symbols", "relations", "documents"}：
            symbols   —— 命中的符号；
            relations —— 与这些符号相连的边（limit*4 条上限）；
            documents —— 符号所在位置的代码片段，供前端直接预览，不用再请求文件内容。
    异常：仓库不存在 → 404；无 repo:read → 403。
    副作用：只读库表；但会读磁盘上的代码文件（仓库 checkout 或 PR worktree）。
    """
    repo = db.get(Repository, repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到仓库")
    text = q.strip()
    query = db.query(CodeSymbol).filter(CodeSymbol.repo_id == repo_id)
    if pr_id:
        query = query.filter(CodeSymbol.pr_id == pr_id)
    if pr_number:
        query = query.filter(CodeSymbol.pr_number == pr_number)
    if text:
        # 名字或路径命中任一即可：用户可能记得函数名，也可能只记得文件名。
        query = query.filter(or_(CodeSymbol.name.contains(text), CodeSymbol.path.contains(text)))
    symbols = query.order_by(CodeSymbol.path.asc(), CodeSymbol.start_line.asc()).limit(limit).all()
    symbol_ids = [item.id for item in symbols]
    relations = []
    if symbol_ids:
        # 关系边多取一些（limit*4）：一个符号可能既被调用又调用别人，边数天然多于点数。
        rows = (
            db.query(CodeRelation)
            .filter(
                CodeRelation.repo_id == repo_id,
                # 下面两个三元表达式是 SQLAlchemy 里写「可选过滤」的惯用法：
                # 条件不成立时塞 True，渲染成 SQL 的 true，等价于不过滤；
                # 这样能在一个 filter 里保持条件数量固定，不必分支拼 query。
                CodeRelation.pr_id == pr_id if pr_id else True,
                CodeRelation.pr_number == pr_number if pr_number else True,
                # 三种关联方式都要：作为调用方、作为被调用方（有 id），
                # 以及只按名字匹配的「目标名字对得上」（跨文件/未解析到 id 的调用边）。
                or_(CodeRelation.source_symbol_id.in_(symbol_ids), CodeRelation.target_symbol_id.in_(symbol_ids), CodeRelation.target_name.in_([item.name for item in symbols])),
            )
            .limit(limit * 4)
            .all()
        )
        relations = [
            {
                "id": str(row.id),
                "source_symbol_id": str(row.source_symbol_id) if row.source_symbol_id else None,
                "target_symbol_id": str(row.target_symbol_id) if row.target_symbol_id else None,
                "source_name": row.source_name,
                "target_name": row.target_name,
                "relation_type": row.relation_type,
                "path": row.path,
                "branch": row.branch,
                "pr_id": str(row.pr_id) if row.pr_id else None,
                "pr_number": row.pr_number,
                # meta 可能为 NULL，兜成空字典，前端少一次判空。
                "metadata": row.meta or {},
            }
            for row in rows
        ]
    documents = []
    checkout_path = repository_checkout_path(repo)
    selected_pr = None
    if pr_id:
        selected_pr = db.get(PullRequest, pr_id)
    elif pr_number:
        selected_pr = (
            db.query(PullRequest)
            .filter(PullRequest.repo_id == repo_id, PullRequest.number == pr_number)
            .one_or_none()
        )
    # 有 PR 且该 PR 的 worktree 还在磁盘上时，优先读 PR 快照：
    # 这样展示的是「这个 PR 里的代码」，而不是主分支的当前内容。
    if selected_pr and selected_pr.repo_id == repo_id:
        snapshot_path = pr_worktree_path(repo, selected_pr)
        if snapshot_path.exists():
            checkout_path = snapshot_path
    if checkout_path.exists():
        # 只为命中的符号读片段；文件不存在（仓库没 clone 下来 / 已被清理）时整段跳过，
        # 接口仍然能返回 symbols 和 relations，前端只是少了预览。
        for symbol in symbols[:limit]:
            snippet = read_code_excerpt(checkout_path, symbol.path, symbol.start_line, line_count=7)
            if not snippet:
                continue
            documents.append(
                {
                    # id 用「路径:起始行」拼：在一次响应内唯一，前端可当 React key 用。
                    "id": f"workspace:{symbol.path}:{symbol.start_line}",
                    "title": f"{symbol.path}:{symbol.start_line}",
                    "source_id": symbol.path,
                    "path": symbol.path,
                    "symbol": symbol.name,
                    "start_line": symbol.start_line,
                    # 7 行片段 → 末行 = 起始行 + 6（含首行）。
                    "end_line": symbol.start_line + 6,
                    "pr_id": str(selected_pr.id) if selected_pr else None,
                    "pr_number": selected_pr.number if selected_pr else None,
                    # 截断到 1200 字符：个别文件单行极长（压缩产物），防止响应体被一条撑爆。
                    "snippet": snippet[:1200],
                }
            )
    return {"query": text, "symbols": symbols, "relations": relations, "documents": documents}
