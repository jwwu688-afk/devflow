"""GitHub REST API 的极简异步客户端。

设计取舍：这里没有引入 PyGithub 之类的 SDK，而是直接用 httpx 打 REST 接口。
原因是本项目只需要 issue / PR / 评论这几类操作，SDK 带来的额外抽象反而会
掩盖「一次请求 = 一页数据」这个限制（见 get() 的说明）。

注意：每个方法都**新建一个 AsyncClient**，用完即关。这样做省去了连接池的
生命周期管理（在 Celery/异步任务里不容易泄漏），代价是失去了 TCP 连接复用，
高频调用时会慢一些。当前调用量很小，这个取舍是划算的。
"""

from typing import Any

import httpx

from app.core.config import settings


class GitHubClient:
    """GitHub REST API 客户端，所有方法都是 async。

    鉴权走 Personal Access Token（PAT），而不是 OAuth App：本工具是内部研发
    助手，PAT 配置一次即可，不需要多租户授权流程。
    """

    def __init__(self, token: str | None = None, base_url: str | None = None) -> None:
        """初始化客户端，未显式传参时回落到全局配置。

        token 允许为空：匿名请求可以访问公开仓库，但速率限额从 5000/小时
        降到 60/小时，仅够做只读探测，写操作一定会 401。
        """
        self.token = token or settings.github_token
        # rstrip("/") 是为了后续统一用 f"{base_url}{path}" 拼接；如果配置里
        # 带了尾斜杠，会拼出 //repos/... 这种双斜杠路径
        self.base_url = (base_url or settings.github_api_base_url).rstrip("/")

    def _headers(self) -> dict[str, str]:
        """构造请求头。

        X-GitHub-Api-Version 必须显式指定：不指定时 GitHub 会回落到某个
        历史版本的行为，不同时间部署可能拿到不一样的字段结构。
        """
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        # 只有拿到 token 才加 Authorization 头，空串会导致 GitHub 返回 401
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    async def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """发起单页 GET 请求并返回解析后的 JSON。

        重要：本方法**只取一页**，不读取 Link 响应头、也不做翻页循环。
        GitHub 列表类接口默认每页 30 条，调用方如果需要全量数据，必须自己
        在 params 里传 per_page 并通过 page 递增多次调用；否则会静默丢数据。
        这是当前实现里最容易踩的坑，改动时请勿默认它返回了全集。
        """
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            response = await client.get(f"{self.base_url}{path}", headers=self._headers(), params=params)
            # raise_for_status 会在 4xx/5xx 时抛 HTTPStatusError，
            # 让调用方用统一的异常路径处理限流(403/429)、仓库不存在(404) 等情况
            response.raise_for_status()
            return response.json()

    async def get_bytes(self, path: str, params: dict[str, Any] | None = None) -> bytes:
        """下载原始字节（用于归档包、文件内容等二进制响应）。

        超时给到 60s，比普通 JSON 请求更宽松：二进制响应体积大、传输慢。
        """
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
            response = await client.get(f"{self.base_url}{path}", headers=self._headers(), params=params)
            response.raise_for_status()
            return response.content

    async def post(self, path: str, payload: dict[str, Any]) -> Any:
        """发起 POST 请求（创建 issue、评论、打标签等写操作）。"""
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            response = await client.post(f"{self.base_url}{path}", headers=self._headers(), json=payload)
            response.raise_for_status()
            return response.json()

    async def patch(self, path: str, payload: dict[str, Any]) -> Any:
        """发起 PATCH 请求（局部更新，例如关闭 issue）。"""
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            response = await client.patch(f"{self.base_url}{path}", headers=self._headers(), json=payload)
            response.raise_for_status()
            return response.json()

    async def get_repo(self, owner: str, repo: str) -> dict[str, Any]:
        """获取仓库元信息（默认分支、语言、是否私有等）。"""
        return await self.get(f"/repos/{owner}/{repo}")

    async def create_issue_comment(self, owner: str, repo: str, issue_number: int, body: str) -> dict[str, Any]:
        """在指定 issue（PR 也复用 issue 编号空间）下追加一条评论。"""
        return await self.post(f"/repos/{owner}/{repo}/issues/{issue_number}/comments", {"body": body})

    async def create_issue(self, owner: str, repo: str, title: str, body: str) -> dict[str, Any]:
        """新建 issue。注意 GitHub 的 issue 与 PR 共用同一套编号序列。"""
        return await self.post(f"/repos/{owner}/{repo}/issues", {"title": title, "body": body})

    async def close_issue(self, owner: str, repo: str, issue_number: int) -> dict[str, Any]:
        """关闭 issue。GitHub 只能通过 PATCH state=closed 关闭，没有独立的 close 接口。"""
        return await self.patch(f"/repos/{owner}/{repo}/issues/{issue_number}", {"state": "closed"})

    async def add_issue_labels(self, owner: str, repo: str, issue_number: int, labels: list[str]) -> dict[str, Any]:
        """给 issue 追加标签。该接口是**增量**语义，不会覆盖已有标签。"""
        return await self.post(f"/repos/{owner}/{repo}/issues/{issue_number}/labels", {"labels": labels})
