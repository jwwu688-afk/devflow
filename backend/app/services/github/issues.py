"""Issue 列表查询。

只有一个函数，但这里藏着一个容易出错的点：GitHub 的 /issues 接口会把
Pull Request 也当成 issue 返回（PR 与 issue 共用编号空间），所以必须显式过滤。
"""

from app.services.github.client import GitHubClient


async def list_issues(
        owner: str,
        repo: str,
        token: str | None = None,
        limit: int = 30,
        base_url: str | None = None
    ) -> list[dict]:
    """拉取仓库的 issue 列表（已关闭 + 未关闭），并把 PR 剔除掉。

    limit 只作为 per_page 下发，是单页上限而非总量上限——底层 get() 没有翻页。
    """
    client = GitHubClient(token, base_url)
    # state=all 表示不区分 open/closed，一起要；只传 per_page 拿单页
    items = await client.get(
        f"/repos/{owner}/{repo}/issues",
        {"state": "all", "per_page": limit})  # 把所有的处理完的和没处理的都拉取过来

    # PR 在 issues 接口里也会出现，其特征是附带 pull_request 字段；这里据此排除，
    # 否则下游统计 issue 数量 / 做 issue 分析时会被 PR 污染
    return [item for item in items if "pull_request" not in item]
