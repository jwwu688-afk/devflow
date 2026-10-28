"""GitHub Actions（CI 流水线）相关的查询封装。

三个函数对应「列出运行记录 → 列出某次运行里的每个作业 → 取某次运行的完整日志」，
是排障链路上的递进关系：先看哪次跑挂了，再看是哪个 job 挂的，最后才拉日志原文。

共同的限制：底层 GitHubClient.get/get_bytes 都是**单页请求**，这里只是把
per_page 传进去，并没有实现翻页。所以 limit 超过 100（GitHub 单页上限）时
拿到的仍是一页数据，不会被自动补齐。日志接口本身也不分页，返回的是一个 zip。
"""

from io import BytesIO
from zipfile import BadZipFile, ZipFile

from app.services.github.client import GitHubClient


async def list_workflow_runs(owner: str, repo: str, token: str | None = None, limit: int = 30, base_url: str | None = None) -> list[dict]:
    """列出仓库最近的 workflow 运行记录。

    limit 直接作为 per_page 下发，只影响单页容量，不会触发表格翻页；
    取不到 workflow_runs 字段时回退成空列表，避免调用方收到 None 再去解包报错。
    """
    client = GitHubClient(token, base_url)
    payload = await client.get(f"/repos/{owner}/{repo}/actions/runs", {"per_page": limit})
    return payload.get("workflow_runs", [])


async def list_workflow_run_jobs(owner: str, repo: str, run_id: int, token: str | None = None, base_url: str | None = None) -> list[dict]:
    """列出某次 run 下的所有 job 及其步骤状态。

    per_page 固定写 100（GitHub 允许的最大值）而不是做成参数：一个 run 里的
    job 数量通常远小于这个量级，固定值可以少一次翻页判断。
    """
    client = GitHubClient(token, base_url)
    payload = await client.get(f"/repos/{owner}/{repo}/actions/runs/{run_id}/jobs", {"per_page": 100})
    return payload.get("jobs", [])


async def get_workflow_run_logs(owner: str, repo: str, run_id: int, token: str | None = None, base_url: str | None = None, max_chars: int = 80000) -> str:
    """拉取一次运行的全部日志并拼成纯文本，超出上限就截断。

    GitHub 的日志接口返回的是一个 zip（里面每个 job 一个 .log 文件），所以这里
    要先解压；但 token 权限不足或 run 过旧时接口可能直接吐 HTML/纯文本，
    解压会抛 BadZipFile，这种情况按原文降级返回。

    max_chars 的存在是为了控制下游 LLM 的上下文成本——完整日志动辄几十 MB，
    直接塞进 prompt 会超上下文窗口或烧掉大量 token。截断策略是按文件名排序后
    依次累加，攒够预算就停，因此被截掉的一定是字典序靠后的 job 日志。
    """
    client = GitHubClient(token, base_url)
    payload = await client.get_bytes(f"/repos/{owner}/{repo}/actions/runs/{run_id}/logs")
    try:
        with ZipFile(BytesIO(payload)) as archive:
            parts: list[str] = []
            # 按名称排序是为了让输出稳定：zip 内部条目顺序不保证一致，
            # 不排序的话同样的 run 两次拉取可能得到不同的截断结果
            for name in sorted(archive.namelist()):
                # 目录条目没有内容，跳过
                if name.endswith("/"):
                    continue
                raw = archive.read(name)
                # errors="replace" 而非严格解码：日志里可能混入 ANSI 转义或
                # 非 UTF-8 字节，严格模式会直接抛异常导致整个排障流程失败
                text = raw.decode("utf-8", errors="replace")
                parts.append(f"### {name}\n{text}")
                # 每次追加后重新求总和，超出预算就停止读后续文件
                if sum(len(part) for part in parts) >= max_chars:
                    break
            # 最终再切一次：最后一个 part 可能本身就把预算撑爆了
            return "\n\n".join(parts)[:max_chars]
    except BadZipFile:
        return payload.decode("utf-8", errors="replace")[:max_chars]
