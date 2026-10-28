"""代码索引前置步骤——「同步 checkout 目录」的测试。

索引任何仓库前都要先把源码落到本地：要么用已缓存的 checkout，要么用用户配置的本地路径，
要么现场 clone。这里验证三种分支的取舍，尤其是「出问题时宁可降级也不要中断」的边界：
git 拉取失败要退回缓存副本、local 模式绝不能偷偷 pull 用户的工作区、
配置的下载目标非空时不能覆盖已有数据。
"""

import subprocess
import uuid
from types import SimpleNamespace

from app.core.config import settings
from app.services import code_analysis


def test_sync_checkout_uses_cached_checkout_when_update_fails(tmp_path, monkeypatch) -> None:
    """场景：本地已有一份缓存 checkout，但 git 更新时网络失败（exit 128）。

    期望：不抛异常，直接返回那份缓存目录继续索引。
    为什么重要：索引服务在弱网/临时故障时仍要可用；
    拿旧代码建索引最多是索引略旧，而抛错会让整个仓库直接不可用。
    """
    monkeypatch.setattr(settings, "repo_checkout_dir", str(tmp_path))
    repo = SimpleNamespace(
        id=uuid.uuid4(),
        full_name="owner/repo",
        clone_url="https://github.com/owner/repo.git",
        default_branch="main",
    )
    checkout = code_analysis._repo_checkout_path(repo)
    # 造出 .git 目录，模拟"这里确实是一份已存在的 checkout"
    (checkout / ".git").mkdir(parents=True)

    def fail_run(args, cwd=None):
        # 模拟所有 git 子命令都因网络失败返回非 0
        raise subprocess.CalledProcessError(128, ["git", *args], stderr="network failed")

    monkeypatch.setattr(code_analysis, "_run_git", fail_run)

    assert code_analysis._sync_checkout(repo, None) == checkout


def test_sync_checkout_uses_configured_local_repo_without_updating(tmp_path, monkeypatch) -> None:
    """场景：仓库配置为 local 模式，指向用户自己机器上的目录。

    期望：直接返回该目录的绝对路径，且期间一次 git 都没有执行（fetch/checkout/pull 全禁止）。
    为什么重要：local 模式常用于用户正在开发的私有工作区，
    偷偷执行 pull / checkout 会改乱别人的工作区、甚至冲掉未提交的改动。
    """
    local_repo = tmp_path / "local-repo"
    local_repo.mkdir()
    repo = SimpleNamespace(
        id=uuid.uuid4(),
        full_name="owner/repo",
        clone_url="https://github.com/owner/repo.git",
        default_branch="main",
        local_path=str(local_repo),
        checkout_mode="local",
    )

    # 让路径校验通过：只把这个目录当成合法的 git 工作树
    monkeypatch.setattr(code_analysis, "_is_git_work_tree", lambda path: path == local_repo.resolve())

    def fail_run(args, cwd=None):
        # 一旦执行 git 就失败，用来证明 local 模式完全不做任何远程操作
        raise AssertionError("local repository indexing should not fetch, checkout, or pull")

    monkeypatch.setattr(code_analysis, "_run_git", fail_run)

    assert code_analysis._sync_checkout(repo, None) == local_repo.resolve()


def test_sync_checkout_rejects_non_empty_configured_download_target(tmp_path, monkeypatch) -> None:
    """场景：managed 模式下用户指定的下载目标目录已经存在且非空（里面有 README.md）。

    期望：抛 RuntimeError，且错误信息含 "already exists"，拒绝覆盖。
    为什么重要：目标目录非空可能装的是别的东西（甚至另一个仓库），
    盲目 clone 进去会污染或覆盖用户数据，必须让用户自己先清理。
    """
    target = tmp_path / "repo"
    target.mkdir()
    # 放一个文件让目录变成"非空"，触发拒绝覆盖的保护分支
    (target / "README.md").write_text("# occupied\n", encoding="utf-8")
    repo = SimpleNamespace(
        id=uuid.uuid4(),
        full_name="owner/repo",
        clone_url="https://github.com/owner/repo.git",
        default_branch="main",
        local_path=str(target),
        checkout_mode="managed",
    )

    # 让环境检查通过（git 可用、目标目录不是 git 工作树），使流程能走到"目录非空"这一关
    monkeypatch.setattr(code_analysis.shutil, "which", lambda name: "git")
    monkeypatch.setattr(code_analysis, "_is_git_work_tree", lambda path: False)

    try:
        code_analysis._sync_checkout(repo, None)
    except RuntimeError as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("expected configured non-empty download target to be rejected")
