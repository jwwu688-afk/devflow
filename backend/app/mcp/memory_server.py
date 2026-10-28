#!/usr/bin/env python
"""Memory MCP Server：以 stdio 方式对外暴露「记忆检索」能力的独立进程。

架构定位：
- 它不是一个 HTTP 服务，而是被 MCP 客户端（Claude Desktop / IDE / Agent 宿主）当子进程拉起，
  通过 stdin/stdout 收发逐行 JSON-RPC 2.0 消息，所以日志一律走 stderr，
  否则会污染 stdout 上的协议数据流导致解析失败。
- 之所以独立成进程而不是塞进 FastAPI：MCP 客户端只认标准输入输出，
  独立进程也便于按需启动、崩溃不影响主服务。
- 本文件是「协议适配层」：只做参数校验与结果包装，真正的检索逻辑在
  app.services.chat_memory.MemoryTools，避免两处实现漂移。
"""

import asyncio
import json
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any


# 作为脚本直接运行时（python app/mcp/memory_server.py），sys.path[0] 是脚本所在目录，
# 无法 import 到 app 包；这里手动把 backend 根目录加入搜索路径。
# 必须在 import app.* 之前执行，因此下面的 import 才会带 noqa: E402（模块级非顶部导入告警）。
BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from sqlalchemy.orm import Session  # noqa: E402

from app.db.models import Conversation, Repository  # noqa: E402
from app.db.session import Base, SessionLocal, engine  # noqa: E402
from app.services.chat_memory import MemoryTools  # noqa: E402
from app.services.conversations import ensure_conversation  # noqa: E402


SERVER_NAME = "devflow-memory-mcp"  # 握手时返回给客户端的服务标识
SERVER_VERSION = "0.1.0"  # 协议实现版本，变更工具契约时应同步升级

# Windows 控制台默认不是 UTF-8，中文内容会直接抛 UnicodeEncodeError 把进程搞挂；
# errors="replace" 是兜底，宁可输出乱码字符也不要让整个 server 退出。
for stream in (sys.stdin, sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


def _json_schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    """拼装 JSON Schema 片段，避免每个工具都重复写 type/required 样板。

    additionalProperties=False 是刻意收紧：模型若臆造未声明的参数会直接被 schema 拦下，
    比在业务代码里静默忽略更早暴露「模型理解错了工具」。
    """
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


# 工具声明清单：MCP 客户端的 tools/list 会原样返回它，模型据此决定何时调用哪个工具。
# description 写「何时用 / 何时不用」而不是「这个工具做什么」——
# 路由判断靠的是使用场景，写清边界能显著减少误调用（例如拿它查实时代码）。
TOOLS: list[dict[str, Any]] = [
    # 工具 1：语义检索仓库知识与持久记忆（覆盖 issue/PR/CI/文档/会话摘要/决策）
    {
        "name": "devflow_search_evidence",
        "description": (
            "搜索 DevFlow 仓库知识和持久记忆。"
            "当需要回忆历史 Issue/PR、失败 CI、上传知识、项目文档、会话摘要、Agent 运行记录或已批准决策时，优先使用这个工具。"
            "当前代码、团队成员和实时状态应使用工作区或结构化工具。"
        ),
        "inputSchema": _json_schema(
            {
                "query": {"type": "string", "description": "搜索查询"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20, "description": "最大结果数"},
                "repo_id": {"type": "string", "description": "必填仓库 UUID。"},
                "conversation_id": {"type": "string", "description": "可选 DevFlow 会话 UUID。"},
            },
            ["query", "repo_id"],
        ),
    },
    # 工具 2：逐字读取原始聊天转录（可跨压缩边界，支持按消息 ID / 前后范围 / 字符偏移分页）
    # 与工具 1 的分工：1 用于「找线索」，本工具用于「原文核实」，因此上限给到 100 条 / 16000 字符
    {
        "name": "devflow_get_thread_context",
        "description": (
            "读取 DevFlow 仓库会话的完整原始聊天转录，可跨越 Compact Boundary。"
            "支持关键词、消息 UUID、前后范围和字符偏移分页；当用户询问之前说过什么或需要逐字核实时使用。"
        ),
        "inputSchema": _json_schema(
            {
                "keyword": {"type": "string", "description": "可选关键词过滤"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100, "description": "最大消息数"},
                "message_id": {"type": "string", "description": "可选消息 UUID；用于分页读取一条精确原文。"},
                "before_message_id": {"type": "string", "description": "可选消息 UUID；只读取它之前的记录。"},
                "after_message_id": {"type": "string", "description": "可选消息 UUID；只读取它之后的记录。"},
                "offset": {"type": "integer", "minimum": 0, "description": "读取单条消息时的字符偏移量。"},
                "max_chars": {"type": "integer", "minimum": 200, "maximum": 16000, "description": "本次最多返回的正文字符数。"},
                "repo_id": {"type": "string", "description": "必填仓库 UUID。"},
                "conversation_id": {"type": "string", "description": "可选 DevFlow 会话 UUID。"},
            },
            ["repo_id"],
        ),
    },
    # 工具 3：读取已封存的会话片段（ChatSession 表），含摘要、原始消息与工具事件
    {
        "name": "devflow_read_session_events",
        "description": (
            "读取已封存的 DevFlow 聊天会话、摘要、原始消息和工具事件。"
            "在 search_evidence 之后，或需要会话级回忆时使用。"
        ),
        "inputSchema": _json_schema(
            {
                "session_id": {"type": "string", "description": "可选会话片段 UUID，默认使用最近会话。"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20, "description": "最大会话数"},
                "include_transcript": {"type": "boolean", "description": "是否返回封存的原始消息和工具事件，默认 true。"},
                "max_chars": {"type": "integer", "minimum": 400, "maximum": 20000, "description": "本次最多返回的正文字符数。"},
                "repo_id": {"type": "string", "description": "必填仓库 UUID。"},
                "conversation_id": {"type": "string", "description": "可选 DevFlow 会话 UUID。"},
            },
            ["repo_id"],
        ),
    },
]


class McpError(Exception):
    """协议级错误（区别于程序 bug）：会被转成 JSON-RPC 的 error 对象回给客户端。

    携带 code 是为了让客户端能区分「参数错」还是「服务内部错」：
    -32602 invalid params / -32601 method not found / -32603 internal error / -32700 parse error
    都是 JSON-RPC 2.0 规范里的固定码，复用它可以让通用 MCP 客户端正确展示提示。
    """

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _content(text: str) -> dict[str, Any]:
    """把纯文本包装成 MCP tools/call 约定的 content 数组结构。

    即使只有一个文本块也必须用数组：协议允许多模态内容（图片/资源）混排，
    统一成数组形态客户端才好解析。
    """
    return {"content": [{"type": "text", "text": text}]}


def _resolve_repo(db: Session, args: dict[str, Any]) -> Repository:
    """从工具参数解析并校验仓库，任何异常都转成 McpError 而不是 500。

    这里显式做「必填 → 格式 → 存在性」三级校验，
    是因为错误信息会直接进入模型上下文，写得越具体模型越容易自我纠正。
    """
    repo_id = args.get("repo_id")
    if not repo_id:
        raise McpError(-32602, "repo_id 为必填项")
    try:
        repo = db.get(Repository, uuid.UUID(str(repo_id)))
    except ValueError as exc:
        # uuid.UUID 抛 ValueError 表示格式非法（例如传了仓库名而不是 UUID）
        raise McpError(-32602, f"repo_id 无效：{repo_id}") from exc
    if repo is None:
        raise McpError(-32602, f"未找到仓库：{repo_id}")
    return repo


def _resolve_conversation(db: Session, repo: Repository, args: dict[str, Any]) -> Conversation:
    """解析会话；未指定或不合法格式时按仓库取/建默认会话。

    用 ensure_conversation 而不是直接查库：MCP 客户端常常只给 repo_id，
    需要一个「兜底会话」承载记忆写入，否则检索结果无处归属。
    """
    conversation_id = args.get("conversation_id")
    if conversation_id:
        try:
            return ensure_conversation(db, repo.id, uuid.UUID(str(conversation_id)))
        except ValueError as exc:
            raise McpError(-32602, f"conversation_id 无效：{conversation_id}") from exc
    return ensure_conversation(db, repo.id)


def _repo_label(repo: Repository) -> str:
    """返回给模型看的仓库标识，用 full_name（owner/name）比裸 UUID 更可读。"""
    return repo.full_name


async def _call_memory_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """按工具名分发到具体的记忆检索实现，并统一处理会话、事务与返回格式。

    每次调用独立建 Session：MCP 是长驻进程，若复用同一个 Session，
    一次异常就会让后续请求都处于脏会话状态（需手动 rollback），隔离成本更低。
    """
    # create_all 是兜底：MCP 进程可能先于后端启动（或被单独拉起），此时表还没建。
    # 它只在表不存在时才建，已有表不会再动，因此可以和正式迁移共存。
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        repo = _resolve_repo(db, args)
        conversation = _resolve_conversation(db, repo, args)
        tools = MemoryTools(db, repo, conversation)

        if name == "devflow_search_evidence":
            query = str(args.get("query") or "").strip()
            if not query:
                raise McpError(-32602, "query 为必填项")
            # 用 `or 5` 而不是默认参数：模型可能显式传 null，此时也要落到默认值
            result = await tools.search_evidence({"query": query, "limit": int(args.get("limit") or 5)}, {})
            # 检索过程可能顺带写入召回事件等记录，故提交后再返回
            db.commit()
            return _content(f"仓库：{_repo_label(repo)}\n会话：{conversation.title if conversation else '全局'}\n\n{result.get('answer') or ''}")

        if name == "devflow_get_thread_context":
            result = await tools.get_thread_context(
                {
                    "keyword": str(args.get("keyword") or ""),
                    "limit": int(args.get("limit") or 30),  # 默认 30 条，兼顾上下文完整与 token 成本
                    "message_id": args.get("message_id"),
                    "before_message_id": args.get("before_message_id"),
                    "after_message_id": args.get("after_message_id"),
                    "offset": int(args.get("offset") or 0),
                    "max_chars": int(args.get("max_chars") or 6000),
                },
                {},
            )
            db.commit()
            return _content(f"仓库：{_repo_label(repo)}\n会话：{conversation.title if conversation else '全局'}\n\n{result.get('answer') or ''}")

        if name == "devflow_read_session_events":
            result = await tools.read_session_events(
                {
                    "session_id": args.get("session_id"),
                    "limit": int(args.get("limit") or 5),
                    # 用 bool(args.get(...)) 而不是 args.get(...)：模型有时会传字符串 "false"，
                    # 这里保持原有语义（非空即为真），属于已知的行为特性而非 bug。
                    "include_transcript": bool(args.get("include_transcript", True)),
                    "max_chars": int(args.get("max_chars") or 8000),
                },
                {},
            )
            db.commit()
            return _content(f"仓库：{_repo_label(repo)}\n会话：{conversation.title if conversation else '全局'}\n\n{result.get('answer') or ''}")

        # 走到这里说明客户端调用了一个未在 TOOLS 中声明的名字（通常是对接版本不一致）
        raise McpError(-32601, f"未知工具：{name}")
    finally:
        # 无论成功失败都要关闭：MCP 进程长期运行，连接泄漏会逐渐耗尽数据库连接池
        db.close()


def _initialize() -> dict[str, Any]:
    """应答 initialize 握手：声明协议版本、服务身份与能力。

    protocolVersion 固定为 2024-11-05，是客户端与服务端协商兼容性的依据；
    listChanged=False 表示工具清单在运行期不会变化，客户端无需监听变更通知。
    """
    return {
        "protocolVersion": "2024-11-05",
        "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        "capabilities": {"tools": {"listChanged": False}},
    }


async def handle_request(request: dict[str, Any]) -> dict[str, Any] | None:
    """把一条 JSON-RPC 请求分派到对应处理逻辑，并保证返回结构符合协议。

    返回 None 表示「这是通知（notification），不需要回复」——
    JSON-RPC 规定没有 id 的消息一律不得响应，否则客户端会收到多余数据。
    """
    request_id = request.get("id")
    method = request.get("method")
    params = request.get("params") or {}

    # 客户端在握手后发来的「已初始化」通知，按协议无需任何处理
    if method == "notifications/initialized":
        return None

    try:
        if method == "initialize":
            result = _initialize()
        elif method == "tools/list":
            result = {"tools": TOOLS}
        elif method == "tools/call":
            name = str(params.get("name") or "")
            arguments = params.get("arguments") or {}
            # 先校验类型再进业务：非 dict 的 arguments 会让 arg.get 直接 AttributeError
            if not isinstance(arguments, dict):
                raise McpError(-32602, "arguments 必须是对象")
            result = await _call_memory_tool(name, arguments)
        else:
            # 未知方法按协议返回 method not found，而不是静默忽略
            raise McpError(-32601, f"未找到方法：{method}")
        return {"jsonrpc": "2.0", "id": request_id, "result": result}
    except McpError as exc:
        # 业务/参数错误：错误信息可安全回传，便于模型自我修正
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": exc.code, "message": exc.message}}
    except Exception as exc:
        # 未预期异常：完整堆栈只写 stderr 供人排查，回给客户端的是精简信息，避免泄漏内部实现
        print(traceback.format_exc(), file=sys.stderr)
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32603, "message": str(exc)}}


async def main() -> None:
    """主循环：逐行读取 stdin 上的 JSON-RPC 请求，处理后写回 stdout。

    MCP 的 stdio 传输协议是「一行一条消息」，因此这里严格按行解析；
    启动横幅走 stderr，不能走 stdout（否则客户端会把横幅当成非法 JSON）。
    """
    print(f"[{SERVER_NAME}] 正在 stdio 上运行", file=sys.stderr)
    while True:
        # readline 是阻塞调用，直接 await 会堵住事件循环；
        # to_thread 把它丢到线程池，保证单个慢请求不会冻结整个进程。
        line = await asyncio.to_thread(sys.stdin.readline)
        if not line:
            break  # 读到 EOF 说明客户端已关闭管道，正常退出
        line = line.strip()
        if not line:
            continue  # 跳过空行，避免无意义的解析错误
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            # 解析失败无法得知 id，按规范回 id=None 的 -32700，便于客户端定位是传输层问题
            response = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(exc)}}
        else:
            response = await handle_request(request)
        if response is not None:
            # flush=True：stdout 被缓冲时若不刷新，客户端会一直等不到响应
            print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
