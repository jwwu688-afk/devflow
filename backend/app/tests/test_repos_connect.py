"""仓库接入（connect）流程中几个纯函数的测试。

验证行为：用户粘贴一个 Git 远端地址并接入仓库时，
1) 能从 HTTPS / SCP 风格 / ssh:// 三种写法里正确解析出 owner、repo、host；
2) 能根据 host 判断这是官方 GitHub 还是自建 GitHub 兼容服务；
3) 本地克隆目标目录已存在且非空时，必须拒绝覆盖；
4) 接入请求的默认初始同步量要小而可控，避免一接就拉爆 API 配额。

这些函数虽小，但直接决定「用户第一次点接入能不能成功」，出错时的表现是
把代码克隆到错误目录或覆盖已有工作区，因此单独做了用例。
"""

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.api.routes.repos import _clone_target_from_parent, _infer_provider, _parse_git_remote_url
from app.schemas.repos import RepoConnectRequest


@pytest.mark.parametrize(
    ("remote_url", "owner", "repo", "host"),
    [
        # 最常见的 HTTPS 写法，带 .git 后缀
        ("https://github.com/openai/codex.git", "openai", "codex", "github.com"),
        # SCP 风格（git@host:owner/repo），Git 默认的 SSH 简写
        ("git@github.com:openai/codex.git", "openai", "codex", "github.com"),
        # 完整的 ssh:// URL 写法，且是自建域名，用来验证 host 不是硬编码 github.com
        ("ssh://git@github.example.com/platform/devflow.git", "platform", "devflow", "github.example.com"),
    ],
)
def test_parse_git_remote_url(remote_url: str, owner: str, repo: str, host: str) -> None:
    """测的是远端地址解析：三种主流写法都要能拆出 owner/repo/host。

    场景：用户复制粘贴的地址格式不统一（HTTPS、SCP、ssh:// 都有）。
    期望：三种写法都解析出相同的三段信息，且 .git 后缀被剥掉。
    重要性：解析错会导致后续 API 调用打到不存在的仓库，报错信息还很难懂。
    """
    parsed = _parse_git_remote_url(remote_url)

    assert parsed.owner == owner
    assert parsed.repo == repo
    assert parsed.host == host


def test_infer_provider_for_custom_remote_host() -> None:
    """测的是 provider 推断：官方域名才算 github，其他域名一律当作兼容实现。

    场景：企业内网常自建 GitHub（如 github.example.com），API 路径与限流规则
    都可能不同，不能当成官方 GitHub 处理。
    期望：host 为 github.com 时返回 "github"；非官方域名返回 "github_compatible"。
    重要性：认错 provider 会让后面所有 GitHub API 调用走错客户端/错误限流策略。
    """
    assert _infer_provider("github", "github.example.com") == "github_compatible"
    assert _infer_provider("github", "github.com") == "github"


def test_clone_target_from_parent_rejects_occupied_directory(tmp_path) -> None:
    """测的是克隆目标目录保护：目录已存在且有内容时必须拒绝。

    场景：用 tmp_path 造一个「已存在的 codex 目录」，里面放一个 README 模拟
    用户已有的工作副本。
    期望：_clone_target_from_parent 抛 HTTPException，而不是返回路径让上层覆盖。
    重要性：一旦允许覆盖，等于静默删除用户本地未提交的代码，是不可逆的数据事故。
    """
    target = tmp_path / "codex"
    target.mkdir()
    # 放个文件让目录「非空」，与「空目录可直接复用」的情况区分开
    (target / "README.md").write_text("# occupied\n", encoding="utf-8")

    with pytest.raises(HTTPException):
        _clone_target_from_parent(str(tmp_path), "codex")


def test_repo_connect_request_uses_a_small_bounded_initial_sync() -> None:
    """测的是接入请求的默认值与下界约束。

    场景：不传 initial_sync_limit 时使用 schema 默认值。
    期望：默认只同步 5 条，且 issue / PR / 流水线三类默认都开；
    同时 initial_sync_limit=0 必须被 pydantic 校验拒绝。
    重要性：默认值过大在接入大型仓库时会瞬间打满 GitHub API 限流，
    下界约束则防止前端传 0 导致「同步成功但一条数据都没有」的迷惑现象。
    """
    payload = RepoConnectRequest(owner="openai", repo="openai-quickstart-python")

    assert payload.initial_sync_limit == 5
    assert payload.sync_issues is True
    assert payload.sync_pull_requests is True
    assert payload.sync_workflow_runs is True

    # 0 超出允许区间，必须抛 ValidationError（而不是被静默接受）
    with pytest.raises(ValidationError):
        RepoConnectRequest(owner="openai", repo="openai-quickstart-python", initial_sync_limit=0)
