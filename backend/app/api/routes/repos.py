"""仓库接入与管理接口，挂载前缀 /api/repos。

接口：
    POST   /connect             接入仓库（三种方式：本地路径 / owner+repo / 演示模式）
    GET    ""                   仓库列表（按最近使用排序）
    POST   /{repo_id}/sync      手动同步 Issue/PR/CI，并后台重建代码图谱
    POST   /{repo_id}/select    标记「当前选中」，即刷新最近使用时间
    GET    /{repo_id}/sync-status  同步状态
    DELETE /{repo_id}           彻底删除仓库及其全部衍生数据

为什么 connect 这么长：接入是本项目最复杂的入口，要同时处理
「本地已有 clone」和「远端仓库」两种情况、推断 provider 与 API 基址、
在无 token 时优雅降级成演示模式，还要决定哪些同步是同步做、哪些丢后台。
"""

import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.db.models import ActionDraft, AgentRun, AgentTaskRun, AgentWorkflowRun, AnalysisResult, AuditLog, ChatFeedback, ChatMessage, ChatSession, CodeRelation, CodeSymbol, Conversation, ConversationMemory, Document, EvidenceItem, Issue, KnowledgeBaseConfig, KnowledgeGraphEdge, KnowledgeSourceDocument, MemoryCandidate, PRFile, PRReviewComment, ProjectIndexState, PullRequest, RecallEvent, Repository, RetrievalTestRun, ThreadMemory, WorkflowRun, Workspace
from app.db.session import get_db
from app.schemas.repos import RepoConnectRequest, RepoConnectResponse, RepoResponse, RepoSyncRequest, RepoSyncResponse
from app.core.config import settings
from app.core.security import encrypt_token
from app.services.github.client import GitHubClient
from app.services.code_analysis import sync_repository_code_analysis_by_id
from app.services.rag.vector_store import delete_milvus_documents
from app.services.repos_sync import sync_repository

router = APIRouter()

# 目前真正支持「联网接入」的 provider。其它值（如 gitlab）会在 connect 里被明确拒绝，
# 而不是假装成功——避免用户以为接上了、实际一条数据都没同步。
SUPPORTED_CONNECT_PROVIDERS = {"github", "github_compatible"}
# 视为「原生 GitHub」的主机名；不在这里的（如自建 Gitea/GitLab）会归为 github_compatible。
GITHUB_HOSTS = {"github.com", "www.github.com"}


@dataclass(frozen=True)
class GitRemoteInfo:
    """从本地仓库 git remote 解析出来的元信息。

    frozen=True：它只是解析结果的载体，不该在中途被修改，避免多处引用时互相污染。
    """

    owner: str
    repo: str
    clone_url: str
    host: str | None = None
    default_branch: str | None = None


def _clean_text(value: str | None) -> str | None:
    """去空白并把空串归一成 None。

    统一成 None 的好处：下游判断「有没有传」只需 `if x`，不用再区分 "" 和 None。
    """
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _resolve_user_path(raw_path: str, field_name: str) -> Path:
    """把用户填的路径规整成绝对路径。

    入参：
        raw_path: 用户输入，可能带引号、~、%USERPROFILE% 之类的环境变量。
        field_name: 出错时回显的字段名，便于前端指出是哪个输入框有问题。
    返回：resolve 后的绝对 Path。
    异常：清洗后为空 → 400。
    说明：
        用户从资源管理器复制路径时常常带双引号，Windows 上还会用 %VAR%，
        所以先 strip 引号再 expandvars/expanduser；相对路径按进程 cwd 解析。
    """
    cleaned = raw_path.strip().strip('"').strip("'")
    if not cleaned:
        raise HTTPException(status_code=400, detail=f"{field_name} cannot be empty.")
    expanded = os.path.expandvars(os.path.expanduser(cleaned))
    path = Path(expanded)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _git_output(args: list[str], cwd: Path) -> str:
    """执行一条 git 命令并返回 stdout（去掉首尾空白）。

    入参：args 为 git 子命令参数；cwd 为执行目录。
    返回：命令输出的字符串。
    异常：
        git 未安装 → 400；
        命令失败（非 git 仓库、没有 origin 等）→ 400，detail 带上 git 的报错原文。
    说明：
        这里把错误映射成 400 而不是 500——原因都是「用户给的路径不对」，
        属于调用方输入问题，前端应提示用户改路径而不是报「服务器错误」。
        timeout=30 防止在网络盘/超大仓库上卡死请求线程。
    """
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail="未找到 git 可执行文件，无法检查本地仓库。") from exc
    except subprocess.CalledProcessError as exc:
        # 优先用 stderr（git 的错误信息都在这里），没有就退回 stdout / 异常字符串。
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise HTTPException(status_code=400, detail=f"Failed to inspect local repository: {detail}") from exc
    return result.stdout.strip()


def _parse_git_remote_url(remote_url: str) -> GitRemoteInfo:
    """从 git remote URL 解析出 owner/repo。

    入参：形如 git@github.com:foo/bar.git、https://github.com/foo/bar.git、ssh://... 的地址。
    返回：GitRemoteInfo（host 统一小写）。
    异常：解析不出 owner/repo → 400。
    说明：
        scp 风格（user@host:path）不是标准 URL，urlparse 解不了，所以先用正则单独识别；
        正则里要求不含 "://" 才走 scp 分支，否则 https 地址会被误判。
        路径取最后两段作为 owner/repo——这让 GitLab 那种多级 group（a/b/c/repo）
        也能取到 c/repo，但会丢掉更上层的分组信息，属于已知取舍。
    """
    value = remote_url.strip()
    host: str | None = None
    path = ""
    scp_like = re.match(r"^(?:[^@]+@)?(?P<host>[^:]+):(?P<path>.+)$", value)

    if scp_like and "://" not in value:
        host = scp_like.group("host")
        path = scp_like.group("path")
    else:
        parsed = urlparse(value)
        host = parsed.hostname or parsed.netloc or None
        path = parsed.path

    # 反斜杠统一成正斜杠，兼容 Windows 上手工输入的本地风格路径。
    parts = [item.strip() for item in path.replace("\\", "/").strip("/").split("/") if item.strip()]
    if len(parts) < 2:
        raise HTTPException(status_code=400, detail=f"Cannot parse owner/repository from git remote URL: {remote_url}")

    owner = parts[-2]
    # 去掉 .git 后缀，忽略大小写（有人的 remote 写的是 .GIT）。
    repo = re.sub(r"\.git$", "", parts[-1], flags=re.IGNORECASE)
    if not owner or not repo:
        raise HTTPException(status_code=400, detail=f"Cannot parse owner/repository from git remote URL: {remote_url}")
    return GitRemoteInfo(owner=owner, repo=repo, clone_url=value, host=host.lower() if host else None)


def _inspect_local_repository(raw_path: str) -> tuple[Path, GitRemoteInfo]:
    """检查用户给的本地目录是不是一个可用的 git 仓库。

    入参：raw_path 用户输入路径（可以是仓库里的任意子目录）。
    返回：(仓库根目录, remote 解析结果)。
    异常：路径不存在/不是目录 → 400；不在 git 工作区内 → 400；没有 origin → 400（由 _git_output 抛出）。
    说明：
        用户在输入框里可能填的是仓库里的某个子目录，所以用 rev-parse --show-toplevel
        上溯到真正的仓库根，之后的路径/分支都以根目录为准。
        当前分支被当作 default_branch 记下来，便于后续按分支取代码。
    """
    path = _resolve_user_path(raw_path, "local_path")
    if not path.exists() or not path.is_dir():
        raise HTTPException(status_code=400, detail=f"Local repository path does not exist or is not a directory: {path}")

    inside_work_tree = _git_output(["rev-parse", "--is-inside-work-tree"], path).lower()
    if inside_work_tree != "true":
        raise HTTPException(status_code=400, detail=f"Local path is not inside a git repository: {path}")

    repo_root = Path(_git_output(["rev-parse", "--show-toplevel"], path)).resolve()
    remote_url = _git_output(["remote", "get-url", "origin"], repo_root)
    remote_info = _parse_git_remote_url(remote_url)
    # 空仓库（刚 init 没有任何提交）取不到分支名，返回空串，这里归一成 None。
    current_branch = _git_output(["branch", "--show-current"], repo_root) or None
    return repo_root, GitRemoteInfo(
        owner=remote_info.owner,
        repo=remote_info.repo,
        clone_url=remote_info.clone_url,
        host=remote_info.host,
        default_branch=current_branch,
    )


def _infer_provider(requested_provider: str | None, remote_host: str | None) -> str:
    """推断 provider：用户没指定就当 github；域名不是 GitHub 时自动降级为 github_compatible。

    为什么要这一步：用户填了自建 Gitea 的本地仓库时，如果还按 github 处理，
    请求会被打到 api.github.com，必然 404。按 host 自动切换能省掉一次手工配置。
    """
    provider = requested_provider or "github"
    if remote_host and remote_host not in GITHUB_HOSTS and provider == "github":
        return "github_compatible"
    return provider


def _effective_api_base_url(provider: str, raw_base_url: str | None, remote_host: str | None) -> str | None:
    """决定调 GitHub API 时用的 base_url。

    优先级：
        github  → 固定在 api.github.com（不信任用户填的值，避免把 token 发到奇怪域名）；
        github_compatible → 用户显式填的优先；否则按 remote host 推 https://{host}/api/v3
        （Gitea/GitLab 的惯例路径）；都推不出来就返回 None，由 client 用默认值。
    """
    if provider == "github":
        return "https://api.github.com"
    base_url = _clean_text(raw_base_url)
    if base_url:
        # rstrip("/") 防止后续拼出 //repos 这种双斜杠路径。
        return base_url.rstrip("/")
    if remote_host and remote_host not in GITHUB_HOSTS:
        return f"https://{remote_host}/api/v3"
    return None


def _clone_target_from_parent(raw_parent: str, repo_name: str) -> Path:
    """算出「要 clone 到哪个目录」，并提前校验目标目录可用。

    入参：raw_parent 用户选的父目录；repo_name 仓库名（作为子目录名）。
    返回：目标目录 Path。
    异常：
        父路径不是目录 → 400；
        目标已存在且是文件 → 400；
        目标已存在、不是 git 仓库且非空 → 400。
    说明：
        这个预检是为了避免 git clone 跑到一半才报「目录非空」，
        同时也是防误删：如果用户指到了一个已有内容的目录，宁可拒绝也不覆盖。
    """
    parent = _resolve_user_path(raw_parent, "clone_parent_dir")
    if parent.exists() and not parent.is_dir():
        raise HTTPException(status_code=400, detail=f"Download parent path is not a directory: {parent}")

    target = (parent / repo_name).resolve()
    if target.exists() and not target.is_dir():
        raise HTTPException(status_code=400, detail=f"Download target exists and is not a directory: {target}")
    # 已经是 git 仓库（有 .git）就允许复用；空目录也允许；其余一律拒绝。
    if target.exists() and not (target / ".git").exists() and any(target.iterdir()):
        raise HTTPException(
            status_code=400,
            detail=f"Download target already exists and is not an empty git repository: {target}",
        )
    return target


@router.post("/connect", response_model=RepoConnectResponse)
async def connect_repo(payload: RepoConnectRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)) -> RepoConnectResponse:
    """POST /api/repos/connect：接入一个仓库。

    入参：
        payload.local_path: 本地仓库路径（填了就以它为准）。
        payload.owner / repo: 远端仓库标识（没填 local_path 时必填）。
        payload.clone_parent_dir: 想把仓库 clone 到哪（可选）。
        payload.provider / api_base_url / token(github_token): 接入参数。
        payload.demo_mode: 演示模式，跳过 GitHub 校验直接建本地记录。
        payload.sync_* / initial_sync_limit: 首同步的范围与条数。
        background_tasks: 用于把耗时的代码分析挪到响应之后。
    返回：RepoConnectResponse，含最终采用的 provider / base_url / checkout 方式 / 首同步结果。
    异常：
        既没 local_path 又没 owner/repo → 400；
        本地路径非法/不是 git 仓库 → 400；
        provider 不支持 → 400；
        GitHub 校验失败且有 token（或全局未开 demo）→ 400。
    副作用：
        写/更新 repositories 行（token 加密存储）；
        非 demo 模式下**同步**执行一次 Issue/PR/CI 同步（可能较慢）；
        随后后台执行代码分析（clone + 建符号索引）。
    """
    local_path = _clean_text(payload.local_path)
    clone_parent_dir = _clean_text(payload.clone_parent_dir)
    owner = _clean_text(payload.owner)
    repo_name = _clean_text(payload.repo)
    requested_provider = payload.provider or "github"
    # token / github_token 是历史字段名并存，取到哪个用哪个。
    token = payload.token or payload.github_token
    checkout_path: Path | None = None
    # managed = 由本系统管理 clone 目录；local = 直接用用户已有的本地仓库。
    checkout_mode = "managed"
    remote_info: GitRemoteInfo | None = None

    if local_path:
        # 本地模式的 owner/repo 一律以 git remote 为准，覆盖用户填写的值：
        # 用户手填的可能是笔误，remote 里的才是真实来源。
        checkout_path, remote_info = _inspect_local_repository(local_path)
        owner = remote_info.owner
        repo_name = remote_info.repo
        checkout_mode = "local"
    elif not owner or not repo_name:
        raise HTTPException(status_code=400, detail="Provide either a local repository path or owner/repository.")

    provider = _infer_provider(requested_provider, remote_info.host if remote_info else None)
    base_url = _effective_api_base_url(provider, payload.api_base_url, remote_info.host if remote_info else None)
    if provider not in SUPPORTED_CONNECT_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"{provider} real connection is not implemented yet. Use demo mode or github_compatible with a GitHub-compatible REST API.",
        )

    if not local_path and clone_parent_dir:
        # 只有远端模式才需要决定 clone 目标；本地模式直接用用户给的目录。
        checkout_path = _clone_target_from_parent(clone_parent_dir, repo_name or "repository")

    full_name = f"{owner}/{repo_name}"
    effective_demo_mode = payload.demo_mode
    fallback_message: str | None = None
    # 先备一份「拿不到远端信息时」的兜底数据，避免后面到处判空。
    repo_data = {
        "id": None,
        "description": "Demo repository for DevFlow AI",
        "default_branch": remote_info.default_branch if remote_info and remote_info.default_branch else "main",
        "clone_url": remote_info.clone_url if remote_info else f"https://github.com/{owner}/{repo_name}.git",
    }
    if not payload.demo_mode:
        try:
            repo_data = await GitHubClient(token, base_url).get_repo(owner or "", repo_name or "")
        except Exception as exc:
            # 有 token 还失败 = 真接不通（token 失效/仓库不存在），必须报错让用户修；
            # 没 token 且全局开了 demo_mode 才降级，否则等于静默假装成功。
            if token or not settings.demo_mode:
                raise HTTPException(status_code=400, detail=f"Repository connection failed: {exc}") from exc
            effective_demo_mode = True
            fallback_message = (
                f"GitHub verification failed without a token: {exc}. "
                "Added the repository in local cache mode; add a token or retry later for full GitHub sync."
            )

    # 按 full_name 做幂等：重复 connect 同一个仓库是「更新配置」而不是插入第二条。
    repo = db.query(Repository).filter(Repository.full_name == full_name).one_or_none()
    if repo is None:
        repo = Repository(owner=owner or "", name=repo_name or "", full_name=full_name)
        db.add(repo)
    repo.owner = owner or repo.owner
    repo.name = repo_name or repo.name
    repo.provider = provider
    repo.api_base_url = base_url
    if token:
        # 明文 token 绝不入库，统一加密后再存；读取时用 decrypt_token 还原。
        repo.github_token_encrypted = encrypt_token(token, settings.token_encryption_key)
    repo.description = repo_data.get("description")
    repo.default_branch = repo_data.get("default_branch")
    # clone_url 三级兜底：远端返回值 → 本地 remote → 按 github 拼一个。
    repo.clone_url = repo_data.get("clone_url") or (remote_info.clone_url if remote_info else f"https://github.com/{owner}/{repo_name}.git")
    repo.local_path = str(checkout_path) if checkout_path else None
    repo.checkout_mode = checkout_mode
    repo.github_id = repo_data.get("id")
    db.commit()
    db.refresh(repo)
    sync_result: RepoSyncResponse | None = None
    if not effective_demo_mode:
        # 首同步是同步执行的：用户点「接入」后立刻看到 issue/PR 数量才有反馈感。
        # 代价是请求时长取决于仓库规模（受 initial_sync_limit 限制）。
        sync_result = await sync_repository(
            db,
            repo,
            RepoSyncRequest(
                sync_issues=payload.sync_issues,
                sync_pull_requests=payload.sync_pull_requests,
                sync_workflow_runs=payload.sync_workflow_runs,
                limit=payload.initial_sync_limit,
            ),
        )
        background_tasks.add_task(sync_repository_code_analysis_by_id, repo.id)
    elif fallback_message:
        # 降级模式下没有远端数据可同步，但如果本地有 clone 目录，代码分析仍然有意义，照样排后台。
        background_tasks.add_task(sync_repository_code_analysis_by_id, repo.id)
    return RepoConnectResponse(
        repo_id=repo.id,
        full_name=repo.full_name,
        provider=repo.provider,
        api_base_url=repo.api_base_url,
        clone_url=repo.clone_url,
        local_path=repo.local_path,
        checkout_mode=repo.checkout_mode,
        default_branch=repo.default_branch,
        connected=True,
        demo_mode=effective_demo_mode,
        # message 三段式：优先回显降级原因，其次区分演示模式/正常接入，方便前端直接展示。
        message=(
            fallback_message
            or (
                "Demo mode only created a local repository record."
                if effective_demo_mode
                else "Repository connection verified. Initial Issue/PR/CI sync completed; code checkout and graph analysis started."
            )
        ),
        # sync_status 三态：真的同步过 → 用同步结果的状态；降级 → fallback；纯演示 → skipped。
        sync_status=sync_result.status if sync_result else ("fallback" if fallback_message else "skipped"),
        synced=sync_result.synced if sync_result else {"issues": 0, "pull_requests": 0, "workflow_runs": 0},
    )


@router.get("", response_model=list[RepoResponse])
async def list_repos(db: Session = Depends(get_db)) -> list[Repository]:
    """GET /api/repos：仓库列表。

    入参：无。
    返回：按 updated_at 倒序（最近使用/同步过的在前）的仓库列表。
    异常：无。
    副作用：只读。
    说明：过滤掉 provider="knowledge_base" 的记录——那是「知识库」形态的伪仓库，
        只在知识库页面出现，不该混进代码仓库列表。
    """
    return (
        db.query(Repository)
        .filter(Repository.provider != "knowledge_base")
        .order_by(Repository.updated_at.desc())
        .all()
    )


@router.post("/{repo_id}/sync", response_model=RepoSyncResponse)
async def sync_repo(
    repo_id: UUID,
    payload: RepoSyncRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> RepoSyncResponse:
    """POST /api/repos/{repo_id}/sync：手动触发一次同步。

    入参：payload 指定同步哪几类（issue / pr / ci）以及条数上限。
    返回：RepoSyncResponse（状态 + 各类同步条数）。
    异常：仓库不存在 → 404；GitHub 调用失败由 service 抛出。
    副作用：
        同步执行远端拉取（会写库，可能较慢）；无论成败都在后台排一次代码分析，
        保证「数据变了，代码索引也跟着更新」。
    """
    repo = db.get(Repository, repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到仓库")
    result = await sync_repository(db, repo, payload)
    background_tasks.add_task(sync_repository_code_analysis_by_id, repo.id)
    return result


@router.post("/{repo_id}/select")
async def select_repo(repo_id: UUID, db: Session = Depends(get_db)) -> dict:
    """POST /api/repos/{repo_id}/select：把某个仓库标为「最近使用」。

    入参：repo_id。
    返回：{"repo_id", "full_name", "selected": true}。
    异常：仓库不存在 → 404。
    副作用：把 updated_at 改成当前时间并提交。
    说明：
        没有单独的「当前选中仓库」字段——选中状态就是「updated_at 最新」，
        所以列表按 updated_at 排序时，用户刚点过的仓库自然排第一。
        这样既省一个字段，也避免多端同时使用时的状态冲突。
    """
    repo = db.get(Repository, repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到仓库")
    repo.updated_at = datetime.now(timezone.utc)
    db.commit()
    return {"repo_id": str(repo.id), "full_name": repo.full_name, "selected": True}


@router.delete("/{repo_id}")
async def delete_repo(repo_id: UUID, db: Session = Depends(get_db)) -> dict:
    """DELETE /api/repos/{repo_id}：删除仓库，并清理它的全部衍生数据。

    入参：repo_id（路径）。
    返回：{"repo_id", "deleted", "cleanup": {"milvus_deleted", "removed_files", "file_errors"}}。
    异常：仓库不存在 → 404。清理阶段的失败不会让接口报错，而是记录在 cleanup 里。
    副作用（不可逆）：
        1. 手工逐表删除：issue/PR/CI、代码符号与关系、文档与图谱、会话与消息、
           反馈、审计、草稿、候选记忆等；
        2. 把该仓库从所有 workspace 的 repo_ids 里摘掉；
        3. 删除 Milvus 里的向量（strict=False，向量库不可用也不阻塞删除）；
        4. 删除磁盘上的知识库原文件。
    为什么手写这么多 delete 而不是配 cascade：
        模型层没有统一的级联配置，靠数据库外键级联在 SQLite 上行为不一致；
        显式按依赖顺序删，顺序可控，也方便出问题时一眼看出漏了哪张表。
        注意删除顺序是「子表 → 父表」，最后才删 repositories 行。
    """
    repo = db.get(Repository, repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到仓库")

    # 先把所有关联 ID 收集出来，后面很多表要按这些 ID 做二次过滤（属于「间接关联」的数据）。
    issue_ids = [item.id for item in db.query(Issue.id).filter(Issue.repo_id == repo.id).all()]
    pr_ids = [item.id for item in db.query(PullRequest.id).filter(PullRequest.repo_id == repo.id).all()]
    run_ids = [item.id for item in db.query(WorkflowRun.id).filter(WorkflowRun.repo_id == repo.id).all()]
    conversation_ids = [item.id for item in db.query(Conversation.id).filter(Conversation.repo_id == repo.id).all()]
    # 注意 "xxx.in_(ids) if ids else False" 这个写法：
    # 列表为空时传 False，SQLAlchemy 会渲染成恒假条件，
    # 保证 or_ 的另一个分支（repo_id 匹配）仍然生效，同时又不会误把 in_() 空集合解释成「匹配全部」。
    session_ids = [
        item.id
        for item in db.query(ChatSession.id)
        .filter(
            or_(
                ChatSession.repo_id == repo.id,
                ChatSession.conversation_id.in_(conversation_ids) if conversation_ids else False,
            )
        )
        .all()
    ]
    workflow_ids = [
        item.id
        for item in db.query(AgentWorkflowRun.id)
        .filter(
            or_(
                AgentWorkflowRun.repo_id == repo.id,
                AgentWorkflowRun.conversation_id.in_(conversation_ids) if conversation_ids else False,
            )
        )
        .all()
    ]

    # AnalysisResult 是「多态」表（target_type + target_id），没有 repo_id，
    # 只能按 (类型, ID) 组合删除，所以要先有上面的 issue_ids/pr_ids/run_ids。
    for target_type, ids in [
        ("issue", issue_ids),
        ("pull_request", pr_ids),
        ("workflow_run", run_ids),
    ]:
        if ids:
            db.query(AnalysisResult).filter(
                AnalysisResult.target_type == target_type,
                AnalysisResult.target_id.in_(ids),
            ).delete(synchronize_session=False)

    # 这里先记住原文件的磁盘路径，等数据库提交后再删文件：
    # 如果先删文件、后回滚，就会留下「库里还有记录但文件没了」的坏数据。
    knowledge_sources = (
        db.query(KnowledgeSourceDocument)
        .filter(KnowledgeSourceDocument.repo_id == repo.id)
        .all()
    )
    stored_paths = [Path(source.stored_path) for source in knowledge_sources]

    if pr_ids:
        db.query(PRFile).filter(PRFile.pr_id.in_(pr_ids)).delete(synchronize_session=False)
        db.query(PRReviewComment).filter(PRReviewComment.pr_id.in_(pr_ids)).delete(synchronize_session=False)
    db.query(Document).filter(Document.repo_id == repo.id).delete(synchronize_session=False)
    db.query(RetrievalTestRun).filter(RetrievalTestRun.repo_id == repo.id).delete(synchronize_session=False)
    db.query(KnowledgeSourceDocument).filter(KnowledgeSourceDocument.repo_id == repo.id).delete(synchronize_session=False)
    db.query(KnowledgeBaseConfig).filter(KnowledgeBaseConfig.repo_id == repo.id).delete(synchronize_session=False)
    db.query(KnowledgeGraphEdge).filter(KnowledgeGraphEdge.repo_id == repo.id).delete(synchronize_session=False)
    db.query(ProjectIndexState).filter(ProjectIndexState.repo_id == repo.id).delete(synchronize_session=False)
    db.query(CodeRelation).filter(CodeRelation.repo_id == repo.id).delete(synchronize_session=False)
    db.query(CodeSymbol).filter(CodeSymbol.repo_id == repo.id).delete(synchronize_session=False)
    db.query(ActionDraft).filter(ActionDraft.repo_id == repo.id).delete(synchronize_session=False)
    db.query(AuditLog).filter(AuditLog.repo_id == repo.id).delete(synchronize_session=False)
    db.query(ChatFeedback).filter(ChatFeedback.repo_id == repo.id).delete(synchronize_session=False)
    db.query(MemoryCandidate).filter(
        or_(
            MemoryCandidate.repo_id == repo.id,
            MemoryCandidate.conversation_id.in_(conversation_ids) if conversation_ids else False,
            MemoryCandidate.session_id.in_(session_ids) if session_ids else False,
        )
    ).delete(synchronize_session=False)
    db.query(RecallEvent).filter(
        or_(
            RecallEvent.repo_id == repo.id,
            RecallEvent.conversation_id.in_(conversation_ids) if conversation_ids else False,
        )
    ).delete(synchronize_session=False)
    if workflow_ids:
        db.query(AgentTaskRun).filter(AgentTaskRun.workflow_id.in_(workflow_ids)).delete(synchronize_session=False)
    db.query(AgentWorkflowRun).filter(
        or_(
            AgentWorkflowRun.repo_id == repo.id,
            AgentWorkflowRun.conversation_id.in_(conversation_ids) if conversation_ids else False,
        )
    ).delete(synchronize_session=False)
    db.query(ChatMessage).filter(
        or_(
            ChatMessage.repo_id == repo.id,
            ChatMessage.conversation_id.in_(conversation_ids) if conversation_ids else False,
        )
    ).delete(synchronize_session=False)
    db.query(ChatSession).filter(
        or_(
            ChatSession.repo_id == repo.id,
            ChatSession.conversation_id.in_(conversation_ids) if conversation_ids else False,
        )
    ).delete(synchronize_session=False)
    db.query(ConversationMemory).filter(
        or_(
            ConversationMemory.repo_id == repo.id,
            ConversationMemory.conversation_id.in_(conversation_ids) if conversation_ids else False,
        )
    ).delete(synchronize_session=False)
    db.query(ThreadMemory).filter(ThreadMemory.repo_id == repo.id).delete(synchronize_session=False)
    db.query(EvidenceItem).filter(
        or_(
            EvidenceItem.repo_id == repo.id,
            EvidenceItem.conversation_id.in_(conversation_ids) if conversation_ids else False,
        )
    ).delete(synchronize_session=False)
    db.query(AgentRun).filter(
        or_(
            AgentRun.repo_id == repo.id,
            AgentRun.conversation_id.in_(conversation_ids) if conversation_ids else False,
        )
    ).delete(synchronize_session=False)
    db.query(Conversation).filter(Conversation.repo_id == repo.id).delete(synchronize_session=False)
    db.query(Issue).filter(Issue.repo_id == repo.id).delete(synchronize_session=False)
    db.query(PullRequest).filter(PullRequest.repo_id == repo.id).delete(synchronize_session=False)
    db.query(WorkflowRun).filter(WorkflowRun.repo_id == repo.id).delete(synchronize_session=False)
    # 工作区里存的是 JSON 数组（字符串形式），只能应用层遍历剔除；
    # 比较时两边都转 str，避免 UUID 对象与字符串不相等导致「删了仓库但工作区里还留着」。
    for workspace in db.query(Workspace).all():
        next_repo_ids = [item for item in (workspace.repo_ids or []) if str(item) != str(repo.id)]
        if next_repo_ids != (workspace.repo_ids or []):
            workspace.repo_ids = next_repo_ids
    db.delete(repo)
    db.commit()

    # 数据库提交完成后才动外部资源：先删向量库（strict=False，Milvus 挂了也不影响主流程），
    # 再逐个删原文件，单个文件删不掉只记录不中断——尽量把能清的都清掉。
    milvus_deleted = delete_milvus_documents(repo_id=repo.id, strict=False)
    removed_files = 0
    file_cleanup_errors: list[str] = []
    for path in stored_paths:
        try:
            if path.exists() and path.is_file():
                path.unlink()
                removed_files += 1
        except OSError as exc:
            file_cleanup_errors.append(f"{path}: {exc}")

    return {
        "repo_id": str(repo_id),
        "deleted": True,
        "cleanup": {
            "milvus_deleted": milvus_deleted,
            "removed_files": removed_files,
            "file_errors": file_cleanup_errors,
        },
    }


@router.get("/{repo_id}/sync-status")
async def sync_status(repo_id: UUID, db: Session = Depends(get_db)) -> dict:
    """GET /api/repos/{repo_id}/sync-status：查询同步状态。

    入参：repo_id。
    返回：{"repo_id", "status", "last_sync_at", "message"}；
        status 只有 completed / not_started 两种——因为同步是同步执行、没有后台队列，
        不存在「进行中」这个中间态。
    异常：仓库不存在 → 404。
    副作用：只读。
    """
    repo = db.get(Repository, repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到仓库")
    return {
        "repo_id": str(repo.id),
        # last_sync_at 是最直接的「同步过没有」判据，不必再额外维护状态字段。
        "status": "completed" if repo.last_sync_at else "not_started",
        "last_sync_at": repo.last_sync_at,
        "message": "最近一次同步已成功完成。" if repo.last_sync_at else "仓库还没有同步过。",
    }
