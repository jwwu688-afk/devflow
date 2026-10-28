"""极简 MCP（Model Context Protocol）客户端：用 stdio 子进程方式调用 MCP server。

为什么自己实现而不是用官方 SDK：这里只需要 initialize / tools/list / tools/call
三个方法，且必须和 Web 主进程解耦（子进程崩了不能带走 FastAPI）。
协议细节：一条消息 = 一行 JSON（JSON-RPC 2.0），用换行分帧，所以读写都是按行处理。
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from app.core.config import settings


class McpClientError(RuntimeError):
    """MCP 调用失败（进程起不来、超时、返回非法 JSON、协议报错等）。

    继承 RuntimeError 而非 Exception，是为了让上层「已知的外部依赖故障」
    和真正的编程错误区分开，方便统一按 502/503 处理。
    """


class StdioMcpClient:
    """以子进程 + stdio 方式运行一个 MCP server，并按需发起单次调用。

    每次调用都会重新拉起进程（call_tool 内部 start/stop），不做长连接复用：
    实现简单、天然隔离状态；代价是每次有进程启动开销。默认指向仓库内的
    app/mcp/memory_server.py（记忆服务的 MCP 封装）。
    """

    def __init__(self, command: str | None = None, args: list[str] | None = None, cwd: str | None = None) -> None:
        # parents[2]：services/mcp_client.py → services → app → backend，
        # 用文件位置反推 backend 根目录，避免依赖启动时的工作目录。
        backend_root = Path(__file__).resolve().parents[2]
        # 默认用当前解释器启动子进程，保证子进程和主进程的依赖环境一致。
        self.command = command or sys.executable
        self.args = args or [str(backend_root / "app" / "mcp" / "memory_server.py")]
        self.cwd = cwd or str(backend_root)
        # JSON-RPC 的 id 从 1 递增；因为是「一发一收」的同步模式，不需要做 id 匹配表。
        self._next_id = 1

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """调用一个 MCP 工具并返回 result。

        必须完成 initialize → notifications/initialized 之后才能 tools/call，
        否则服务端会按协议拒绝；finally 里一定 stop，防止僵尸子进程堆积。
        """
        process = await self._start()
        try:
            await self._initialize(process)
            await self._notify(process, "notifications/initialized", {})
            return await self._request(process, "tools/call", {"name": name, "arguments": arguments})
        finally:
            await self._stop(process)

    async def inspect_server(self) -> dict[str, Any]:
        """Perform a real MCP handshake and return the server's advertised tools.

        做一次真实的握手 + tools/list，用于健康检查/调试页展示「服务端声明了哪些工具」。
        返回值里同时带上 initialize 结果，便于排查协议版本不匹配的问题。
        """
        process = await self._start()
        try:
            initialize = await self._initialize(process)
            await self._notify(process, "notifications/initialized", {})
            tools = await self._request(process, "tools/list", {})
            # tools 缺失时统一成空列表，前端就不用判 None。
            return {"initialize": initialize, "tools": tools.get("tools") or []}
        finally:
            await self._stop(process)

    async def _initialize(self, process: asyncio.subprocess.Process) -> dict[str, Any]:
        """发 initialize 请求，声明协议版本与客户端身份。

        protocolVersion 写死 2024-11-05 是为了和服务端实现对齐；
        服务端若不支持会在 result 里返回它自己的版本，由调用方决定是否继续。
        """
        return await self._request(
            process,
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "devflow-web-agent", "version": "0.1.0"},
            },
        )

    async def _start(self) -> asyncio.subprocess.Process:
        """拉起子进程并接好 stdin/stdout/stderr 三根管道。"""
        env = os.environ.copy()
        # 强制子进程用 UTF-8 收发：Windows 默认是 GBK，中文内容会乱码/解码失败。
        env["PYTHONIOENCODING"] = "utf-8"
        # 把主进程用的数据库地址透传给子进程，保证记忆读写的是同一个库。
        env["DATABASE_URL"] = settings.database_url
        # 子进程不该再起一个定时同步任务（会变成多份重复拉取），
        # 用 setdefault 是为了允许外部显式覆盖。
        env.setdefault("AUTO_SYNC_ENABLED", "false")
        process = await asyncio.create_subprocess_exec(
            self.command,
            *self.args,
            cwd=self.cwd,
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        return process

    async def _notify(self, process: asyncio.subprocess.Process, method: str, params: dict[str, Any]) -> None:
        """发送 JSON-RPC 通知（无 id、不需要响应）。"""
        await self._write(process, {"jsonrpc": "2.0", "method": method, "params": params})

    async def _request(self, process: asyncio.subprocess.Process, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """发送带 id 的请求并同步等待一行响应。

        因为是「写一条、读一条」的严格串行，收到的那行必定是本请求的响应，
        所以不做 id 比对；若将来改成并发调用，这里必须补上按 id 分发的逻辑。
        """
        request_id = self._next_id
        self._next_id += 1
        await self._write(process, {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        if process.stdout is None:
            raise McpClientError("MCP process has no stdout")
        try:
            # 30 秒兜底：子进程卡死（例如等待标准输入）时必须放弃，否则请求会永久挂起。
            line = await asyncio.wait_for(process.stdout.readline(), timeout=30)
        except TimeoutError as exc:
            raise McpClientError(f"MCP request timed out: {method}") from exc
        if not line:
            # 读到 EOF 说明子进程已经退出，把 stderr 一起带出来，否则报错信息毫无线索。
            stderr = ""
            if process.stderr is not None:
                stderr = (await process.stderr.read()).decode("utf-8", errors="replace")
            raise McpClientError(f"MCP server closed stdout. {stderr}")
        try:
            response = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            # 常见原因：子进程把普通 print 输出混进了 stdout，破坏了按行分帧的协议。
            raise McpClientError(f"MCP server returned an invalid JSON response for {method}") from exc
        if "error" in response:
            error = response["error"]
            raise McpClientError(str(error.get("message") or error))
        # result 可能为 null，统一成空字典让调用方可以安全 .get。
        return response.get("result") or {}

    async def _write(self, process: asyncio.subprocess.Process, payload: dict[str, Any]) -> None:
        """把一条 JSON-RPC 消息序列化成「一行 JSON + 换行」写入子进程。"""
        if process.stdin is None:
            raise McpClientError("MCP process has no stdin")
        # ensure_ascii=False：中文原样输出，配合子进程的 UTF-8 编码；
        # 结尾必须补 \n，否则服务端按行读取会一直等不到完整消息。
        process.stdin.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
        await process.stdin.drain()

    async def _stop(self, process: asyncio.subprocess.Process) -> None:
        """结束子进程：还活着就直接 kill，然后 wait 回收，避免僵尸进程。"""
        if process.returncode is None:
            process.kill()
        await process.wait()
