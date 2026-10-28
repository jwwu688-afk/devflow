"""Pull Request 相关查询：列表、变更文件、行内审查评论。

PR 分析链路一般按这个顺序调用：先 list_pull_requests 拿到候选 PR，再用
list_pull_request_files 看改了哪些文件（判断影响面），最后用
list_pull_request_review_comments 读已有的评审意见（避免 AI 重复指出别人已提的问题）。

三个函数都受底层单页请求的限制：per_page 是单页容量，没有翻页循环。
"""

from app.services.github.client import GitHubClient

# 获取这个仓库所有的 PR
async def list_pull_requests(owner: str, repo: str, token: str | None = None, limit: int = 30, base_url: str | None = None) -> list[dict]:
    """列出 PR（open + closed + merged 全都算，state=all）。

    limit 是单页上限；PR 数量多的活跃仓库需要调用方自己翻页补齐。
    """
    client = GitHubClient(token, base_url)
    return await client.get(f"/repos/{owner}/{repo}/pulls", {"state": "all", "per_page": limit})

# 给定一个 PR 编号（number），看这个 PR 到底改了哪些文件
async def list_pull_request_files(owner: str, repo: str, number: int, token: str | None = None, base_url: str | None = None) -> list[dict]:
    """获取 PR 的文件变更清单（含 patch 片段）。

    固定 per_page=100（单页上限）：GitHub 不返回 diff 全文，只给每个文件
    最多若干行的 patch 上下文，所以超长文件的重构在这里是看不全的。
    """
    client = GitHubClient(token, base_url)
    return await client.get(f"/repos/{owner}/{repo}/pulls/{number}/files", {"per_page": 100})

# 给定一个 PR 编号，拉取在这个 PR 上留下的代码行内审查评论
async def list_pull_request_review_comments(owner: str, repo: str, number: int, token: str | None = None, base_url: str | None = None) -> list[dict]:
    """获取行内（review comment）评论列表。

    注意这里只取「行内评论」，PR 会话区的普通讨论要另走 issues 评论接口。
    同样固定 100 条单页，超过部分不会自动补齐。
    """
    client = GitHubClient(token, base_url)
    return await client.get(f"/repos/{owner}/{repo}/pulls/{number}/comments", {"per_page": 100})
