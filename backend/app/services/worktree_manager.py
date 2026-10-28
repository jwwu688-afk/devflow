"""Git worktree lifecycle for live PR code analysis.

为「在 PR 分支上做在线代码分析」管理 git worktree 快照的生命周期：
准备快照（prepare_pr_worktree）→ 建索引 → 过期清理（cleanup_pr_snapshots）。

为什么用 worktree 而不是 clone：同一个 base 仓库的对象库可以共享，
新建一个 PR 快照只多占工作区文件，不用重复下载整个仓库历史。
代价是 worktree 与主仓库耦合，删除时必须先让 git 自己 remove，
否则主仓库里会残留 worktree 登记信息（详见清理函数里的注释）。
"""

import json
import shutil
import subprocess
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import decrypt_token
from app.db.models import CodeRelation, CodeSymbol, PullRequest, Repository
from app.services.code_analysis import _git_text, _sync_checkout, _sync_code_graph

# 每个 (repo, pr) 一把锁：同一 PR 的并发准备/覆盖会同时写同一个目录，
# 必须串行化，否则 git 会报 index.lock 冲突、索引也会互相覆盖。
_SNAPSHOT_LOCKS: dict[str, threading.Lock] = {}
# 保护上面那个字典本身的锁：进程内多线程第一次创建锁时会有竞态。
_SNAPSHOT_LOCKS_GUARD = threading.Lock()


def _checkout_root() -> Path:
    """返回（并确保存在）所有仓库 checkout 的根目录。

    配置里可能是相对路径，相对路径按当前工作目录解析成绝对路径：
    因为后续 git 命令会在别的 cwd 下执行，留着相对路径会指向错误位置。
    """
    root = Path(settings.repo_checkout_dir)
    if not root.is_absolute():
        root = Path.cwd() / root
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _snapshot_lock(key: str) -> threading.Lock:
    """按 key 取（必要时创建）一把进程内互斥锁。"""
    with _SNAPSHOT_LOCKS_GUARD:
        lock = _SNAPSHOT_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _SNAPSHOT_LOCKS[key] = lock
        return lock


def _run_git(args: list[str], cwd: Path) -> str:
    """执行一条 git 命令并返回去掉首尾空白的 stdout。

    check=True：git 非零退出直接抛异常，由上层统一转成 fallback 响应；
    timeout=180：网络卡住（fetch 大仓库）时必须能放弃，否则请求会一直挂住。
    """
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=180,
    )
    return result.stdout.strip()


def pr_worktree_path(repo: Repository, pr: PullRequest) -> Path:
    """算出某个 PR 快照目录的路径（不创建）。

    路径里拼了 repo.id 做一级隔离、safe_name 便于人肉辨认；
    把 full_name 里的 / 和 \\ 替换掉，是为了避免 owner/repo 这种带斜杠的名字
    被当成多级目录。
    """
    safe_name = repo.full_name.replace("/", "__").replace("\\", "__")
    return _checkout_root() / "worktrees" / str(repo.id) / f"pr-{pr.number}-{safe_name}"


def _snapshot_meta_path(path: Path) -> Path:
    """快照元信息文件名（放在快照目录根部，随目录一起被删除）。"""
    return path / ".devflow-pr-snapshot.json"


def _write_snapshot_meta(path: Path, meta: dict[str, Any]) -> None:
    """写入快照元信息，并补上 updated_at 供过期清理判断。

    以点开头命名，避免被当作用户代码索引；indent=2 + 中文不转义，便于排查问题时直接看。
    """
    payload = {**meta, "updated_at": datetime.now(timezone.utc).isoformat()}
    _snapshot_meta_path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _sync_pr_snapshot_analysis(
    db: Session,
    repo: Repository,
    pr: PullRequest,
    snapshot_path: Path,
    branch: str,
    commit_sha: str | None,
) -> int:
    """对快照目录跑一遍代码图谱同步，返回索引到的符号数。

    这里复用 code_analysis 的 _sync_code_graph，并额外带上 pr_id/pr_number，
    让符号和关系能区分「默认分支的」和「某个 PR 快照的」两套数据。
    """
    return _sync_code_graph(
        db,
        repo,
        snapshot_path,
        branch=branch,
        commit_sha=commit_sha,
        pr_id=pr.id,
        pr_number=pr.number,
    )


def _snapshot_response(
    *,
    status: str,
    message: str | None = None,
    branch: str | None = None,
    commit_sha: str | None = None,
    pr: PullRequest,
    snapshot_path: Path | None = None,
    indexed_symbols: int | None = None,
    fallback: str | None = None,
) -> dict[str, Any]:
    """统一构造快照准备结果。

    所有分支（成功/不支持/失败）都返回同一套字段，前端只按 status 分支即可，
    不用为每种情况写不同的取数逻辑。
    checkout_kind 直接由 status 推导：只有 ready 才算真的用上了 PR 快照，
    其余情况前端展示的其实是默认分支的代码。
    """
    return {
        "status": status,
        "message": message,
        "snapshot_name": "PR 分支快照",
        "branch": branch,
        "commit_sha": commit_sha,
        "pr_id": str(pr.id),
        "pr_number": pr.number,
        "snapshot_path": str(snapshot_path) if snapshot_path else None,
        "indexed_symbols": indexed_symbols,
        "fallback": fallback,
        "checkout_kind": "pr_snapshot" if status == "ready" else "default_checkout",
    }


def prepare_pr_worktree(db: Session, repo: Repository, pr: PullRequest) -> dict[str, Any]:
    """Prepare a PR branch snapshot for live search and refresh its code graph.

    准备（或更新）某个 PR 的分支快照，并刷新它的代码图谱，返回统一结构的结果。
    设计原则是**尽力而为**：任何一步失败都退化成 fallback（用默认分支代码继续），
    而不是让用户的分析请求直接报错——查不到 PR 分支代码总比什么都查不了好。
    """
    lock_key = f"{repo.id}:{pr.id}"
    lock = _snapshot_lock(lock_key)
    with lock:
        # 本地仓库模式：代码就在用户机器上，devflow 不碰它的 git 状态，
        # 只回落到 local_path，避免误改用户正在用的工作区。
        if (repo.checkout_mode or "managed") == "local":
            return _snapshot_response(status="unsupported", message="本地仓库模式不自动创建或删除 PR 分支快照。", pr=pr, fallback=repo.local_path)
        # 没有 head branch 就无法定位 PR 代码（例如数据还没同步完整）。
        if not pr.head_branch:
            return _snapshot_response(status="missing_head", message="PR 缺少 head branch，继续使用默认分支代码。", pr=pr)
        # 环境里没有 git 时同样降级，而不是抛异常。
        if not shutil.which("git"):
            return _snapshot_response(status="failed", message="未找到 git，可继续使用默认分支代码。", pr=pr)
        token = decrypt_token(repo.github_token_encrypted, settings.token_encryption_key) if repo.github_token_encrypted else None
        target = pr_worktree_path(repo, pr)
        try:
            # 先确保 base 仓库存在（首次会 clone），worktree 必须挂在它下面。
            base_checkout = _sync_checkout(repo, token)
            target.parent.mkdir(parents=True, exist_ok=True)
            ref = pr.head_branch
            # 本地分支名固定按 PR 号命名：同一个 PR 反复准备时会强制覆盖，
            # 不会因为分支名变化而堆积一堆 worktree。
            local_branch = f"devflow/pr-{pr.number}"
            _run_git(["fetch", "origin", ref], cwd=base_checkout)
            if target.exists():
                # 已存在就地更新：再 fetch 一次，然后 -B 强制重置到最新 FETCH_HEAD。
                # 用 -B（大写）而不是 -b，因为分支可能已存在。
                _run_git(["fetch", "origin", ref], cwd=target)
                _run_git(["checkout", "-B", local_branch, "FETCH_HEAD"], cwd=target)
            else:
                # --force：允许路径已被占用/登记过时仍然创建，避免历史残留卡死流程。
                _run_git(["worktree", "add", "--force", "-B", local_branch, str(target), "FETCH_HEAD"], cwd=base_checkout)
            # 记录实际检出的 commit，之后可以判断快照是否过期（PR 又推了新提交）。
            commit_sha = _git_text(["rev-parse", "HEAD"], target)
            indexed_symbols = _sync_pr_snapshot_analysis(db, repo, pr, target, ref, commit_sha)
            # 元信息写在快照目录内：清理时只看目录就能判断它属于哪个 PR、多久没更新。
            _write_snapshot_meta(
                target,
                {
                    "repo_id": str(repo.id),
                    "pr_id": str(pr.id),
                    "pr_number": pr.number,
                    "branch": ref,
                    "commit_sha": commit_sha,
                    "indexed_symbols": indexed_symbols,
                    "snapshot_kind": "pr_snapshot",
                },
            )
            # 索引是本次新建的数据，这里立即提交，避免后续清理/concurrent 请求看到半成品。
            db.commit()
            return _snapshot_response(
                status="ready",
                branch=ref,
                commit_sha=commit_sha,
                pr=pr,
                snapshot_path=target,
                indexed_symbols=indexed_symbols,
            )
        except Exception as exc:
            # 回滚本次写入的符号/关系，磁盘上的半成品目录留着不动
            # （下次准备时会走 target.exists() 分支就地修复，比删掉重建更省）。
            db.rollback()
            return _snapshot_response(
                status="fallback",
                message=f"PR 分支快照准备失败，继续使用默认分支代码：{exc}",
                branch=repo.default_branch or "main",
                pr=pr,
            )


def cleanup_pr_snapshots(db: Session, repo: Repository, *, max_age_hours: int = 168) -> dict[str, Any]:
    """删除超过 max_age_hours 未更新的 PR 快照，并清掉它们贡献的代码图谱数据。

    默认 168 小时（7 天）：PR 讨论通常会持续几天，快照留着能省一次重新拉取；
    但长期不用的分支快照会占满磁盘，所以要定期回收。
    删除依据优先用元信息里的 updated_at，缺失/损坏时退回目录 mtime，
    保证即使元信息丢了也不会把快照永久留在磁盘上。
    """
    root = _checkout_root() / "worktrees" / str(repo.id)
    if not root.exists():
        return {"removed": 0, "paths": []}
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    removed_paths: list[str] = []
    # 用 set 去重：一次清理可能删多个目录，但真正要清的 pr_id 不会重复。
    removed_pr_ids: set[uuid.UUID] = set()
    for path in root.iterdir():
        if not path.is_dir():
            continue
        meta_path = _snapshot_meta_path(path)
        updated_at = None
        snapshot_pr_id: uuid.UUID | None = None
        if meta_path.exists():
            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                updated_raw = metadata.get("updated_at")
                updated_at = datetime.fromisoformat(updated_raw) if updated_raw else None
                if metadata.get("pr_id"):
                    snapshot_pr_id = uuid.UUID(str(metadata["pr_id"]))
            except (OSError, ValueError, json.JSONDecodeError):
                # 元信息损坏/手写错误时不让清理整体失败，后面会退回 mtime 兜底。
                updated_at = None
        if updated_at is None:
            # 兜底：用目录本身的最后修改时间。注意可能带上时区信息缺失的情况，
            # 所以显式指定 tz=utc，避免和 cutoff 比较时抛「naive vs aware」异常。
            updated_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        if updated_at >= cutoff:
            continue
        try:
            # Remove from Git first so the main checkout does not keep stale worktree records.
            # 必须先让 git 自己 remove：直接 rmtree 会留下 .git/worktrees 里的登记，
            # 导致下次同名 worktree add 报「already registered」。
            # 主 checkout 目录名形如 "<repo_id>-<hash>"，取第一个匹配项即可。
            main_checkout = next((item for item in _checkout_root().glob(f"{repo.id}-*") if item.is_dir()), None)
            if main_checkout:
                _run_git(["worktree", "remove", "--force", str(path)], cwd=main_checkout)
            elif path.exists():
                # 找不到主仓库（可能已被删除）时只能硬删目录。
                shutil.rmtree(path)
        except Exception:
            # git remove 失败（比如目录已损坏）仍要兜底删除，否则永远清不掉；
            # ignore_errors 保证清理流程本身不抛异常。
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
        removed_paths.append(str(path))
        if snapshot_pr_id:
            removed_pr_ids.add(snapshot_pr_id)
    if removed_pr_ids:
        # 目录删了，属于这些 PR 的符号/关系就没有对应的源码可指向了，一并清掉，
        # 否则代码检索会命中「文件已不存在」的幽灵符号。
        db.query(CodeRelation).filter(
            CodeRelation.repo_id == repo.id,
            CodeRelation.pr_id.in_(removed_pr_ids),
        ).delete(synchronize_session=False)
        db.query(CodeSymbol).filter(
            CodeSymbol.repo_id == repo.id,
            CodeSymbol.pr_id.in_(removed_pr_ids),
        ).delete(synchronize_session=False)
    db.commit()
    return {"removed": len(removed_paths), "paths": removed_paths}
