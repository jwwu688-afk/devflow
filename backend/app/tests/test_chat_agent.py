"""ChatAgent（对话主编排 Agent）的核心行为测试。

这是整个助手最关键的编排层：它要把「用户消息 + 预取证据 + 会话历史」组装成提示词，
驱动 LLM 做多轮工具调用（工作区读文件、RAG 检索、会话记忆），并处理上下文超限、
工具超时重试、模型死循环等异常路径。本文件覆盖：
- 用量统计与提示词 token 拆解；
- 上下文溢出后的「就地压缩重试」；
- 工具结果投影（压掉旧观测但不污染审计状态）；
- 记忆召回策略（近期原话 / 需查工具 / 跨会话）的选择与理由上报；
- 流式事件顺序、项目概览预取、RAG 混合检索、评测模式 trace；
- 无 LLM 的降级回答、团队人员为空时的提示词约束；
- 工具重试与重复调用熔断。
"""

import asyncio
import uuid
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from app.api.routes.chat import _build_conversation_tools, _repo_checkout_path
from app.core.config import settings
from app.services.agents.chat_agent import ChatAgent


class ScriptedToolModel:
    """按调用次数「写死剧本」的假模型：第 1 次要求列目录，第 2 次要求读 README，第 3 次给出最终答案。

    用它替代真实 LLM，就能确定性地驱动出「list_files → read_file → 回答」这条多轮工具链，
    不需要网络、也不会有模型随机性。
    """

    def __init__(self) -> None:
        self.calls = 0

    def bind_tools(self, tools: list[Any]) -> "ScriptedToolModel":
        # 保存绑定进来的工具列表，便于断言工具是否暴露给了模型
        self.tools = tools
        return self

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {"name": "workspace__list_files", "args": {"path": ".", "limit": 5}, "id": "call_list"},
                ],
            )
        if self.calls == 2:
            return AIMessage(
                content="",
                tool_calls=[
                    {"name": "workspace__read_file", "args": {"path": "README.md"}, "id": "call_read"},
                ],
            )
        return AIMessage(content="README.md says this is the project overview.")


class SummaryModel:
    """永远直接给一段总结的假模型：用来测「预取证据 → 直接总结」路径，不走工具调用。"""

    def bind_tools(self, tools: list[Any]) -> "SummaryModel":
        return self

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        return AIMessage(content="The README says this project coordinates AI agents.")


class RecentRecallModel:
    """假模型：直接根据「注入的历史消息」回答"你刚刚说的是：hi"。

    额外统计 bind_tools_calls，用来验证「近期原话召回」场景下
    Agent 根本不该把工具 schema 交给模型（省 token、避免模型乱调工具）。
    """

    def __init__(self) -> None:
        self.calls = 0
        self.bind_tools_calls = 0

    def bind_tools(self, tools: list[Any]) -> "RecentRecallModel":
        self.bind_tools_calls += 1
        return self

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        self.calls += 1
        return AIMessage(content="你刚刚说的是：hi")


class CapturingModel:
    """假模型：把收到的 prompt 消息全量抓下来，供测试断言提示词内容。

    用于验证「团队为空时」提示词里确实写了约束，而不是只验证最终答案。
    """

    def __init__(self) -> None:
        self.messages: list[Any] = []

    def bind_tools(self, tools: list[Any]) -> "CapturingModel":
        return self

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        self.messages = messages
        return AIMessage(content="No team members are configured.")


class SingleToolModel:
    """假模型：只发起一次 read_file 工具调用，随后立刻给最终答案（不再调用工具）。"""

    def __init__(self) -> None:
        self.calls = 0

    def bind_tools(self, tools: list[Any]) -> "SingleToolModel":
        self.tools = tools
        return self

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {"name": "workspace__read_file", "args": {"path": "README.md"}, "id": "call_read"},
                ],
            )
        return AIMessage(content="The README has been read.")


class RepeatingToolModel:
    """假模型：每次都发起「同一个 read_file 调用」，永不收敛。

    用来验证 Agent 的重复调用熔断（否则真实模型陷入死循环会一直烧 token）。
    """

    def __init__(self) -> None:
        self.calls = 0

    def bind_tools(self, tools: list[Any]) -> "RepeatingToolModel":
        self.tools = tools
        return self

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        self.calls += 1
        return AIMessage(
            content="",
            tool_calls=[
                {"name": "workspace__read_file", "args": {"path": "README.md"}, "id": f"call_read_{self.calls}"},
            ],
        )


class RewrittenRagModel:
    """假模型：把一个复合问题拆成两次「改写后的」RAG 检索，最后一次给结论。

    模拟真实场景：用户一句「刷新为什么失败 + 当初定的重试策略是什么」，
    模型应改写成两个更精准的子查询分别检索，而不是原句直接搜。
    """

    def __init__(self) -> None:
        self.calls = 0

    def bind_tools(self, tools: list[Any]) -> "RewrittenRagModel":
        self.tools = tools
        return self

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "rag__search_similar_documents",
                        "args": {"query": "token refresh timeout history"},
                        "id": "call_rag_history",
                    }
                ],
            )
        if self.calls == 2:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "rag__search_similar_documents",
                        "args": {"query": "idempotent retry decision"},
                        "id": "call_rag_decision",
                    }
                ],
            )
        return AIMessage(content="The history and retry decision agree: retry once with an idempotency key.")


class UsageModel:
    """假模型：返回固定答案，并附带 langchain 标准的 usage_metadata（token 用量）。"""

    def bind_tools(self, tools: list[Any]) -> "UsageModel":
        return self

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        return AIMessage(
            content="usage captured",
            usage_metadata={"input_tokens": 123, "output_tokens": 7, "total_tokens": 130},
        )


class OverflowThenSuccessModel:
    """假模型：第一次抛「超长上下文」错误，第二次正常返回。

    伪造这种"先失败后成功"的序列，是为了验证 Agent 只对出错的那一次调用
    做压缩重试，而不是整轮对话重来。
    """

    def __init__(self) -> None:
        self.calls = 0

    def bind_tools(self, tools: list[Any]) -> "OverflowThenSuccessModel":
        return self

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("maximum context length exceeded")
        return AIMessage(content="recovered after compact projection")


def _history_context(*, complete_and_exact: bool, exact_suffix_messages: int = 3) -> dict[str, Any]:
    """伪造「服务端注入的会话历史 + 历史覆盖度统计」。

    complete_and_exact 决定这份历史是否完整精确：
    - True：完整（omitted=0，未触达来源上限），Agent 应直接用历史回答；
    - False：历史被截断（omitted=1，source_limit_reached=True），Agent 应改去查会话记录工具。
    通过这一个开关，就能分别覆盖「近期原话够用」和「近期原话不够用」两条策略分支。
    """
    return {
        "server_messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "你好！有什么可以帮你？"},
            {"role": "user", "content": "我刚刚说了什么？"},
        ],
        "compression_stats": {
            "history_coverage": {
                "complete_and_exact": complete_and_exact,
                "exact_suffix_messages": exact_suffix_messages,
                "omitted_messages": 0 if complete_and_exact else 1,
                "truncated_messages": 0,
                "source_limit_reached": not complete_and_exact,
                "has_compaction_boundary": False,
                "microcompacted_messages": 0,
            }
        },
    }


@pytest.mark.asyncio
async def test_chat_agent_records_provider_usage_and_prompt_breakdown() -> None:
    """场景：模型返回 usage_metadata，且本次没有绑定任何工具。

    期望：final 里记录 input_tokens=123 且 source 标记为 "provider"（用量来自模型厂商而非本地估算）；
    诊断信息里最后一次 prompt 调用的估算 token 数 > 0，工具 schema token 数为 0。
    为什么重要：token 计费与成本归因依赖这里的分账；把「厂商真实值」和「本地估算值」
    区分开，才能在厂商不返回用量时仍给出可信度不同的数据。
    """
    final = await ChatAgent(model=UsageModel()).run(  # type: ignore[arg-type]
        {"message": "Give me a concise answer."},
        {"tools": {}},
    )

    assert final["answer"] == "usage captured"
    assert final["model_usage"][-1]["input_tokens"] == 123
    assert final["model_usage"][-1]["source"] == "provider"
    prompt_call = final["context_diagnostics"]["prompt_calls"][-1]
    assert prompt_call["estimated_prompt_tokens"] > 0
    assert prompt_call["tool_schema_tokens"] == 0


@pytest.mark.asyncio
async def test_context_overflow_retries_only_the_failed_model_call() -> None:
    """场景：第一次调用模型直接抛 "maximum context length exceeded"，第二次正常返回。

    期望：最终拿到答案、模型恰好被调用 2 次（只重试失败那一次）、
    reactive_overflow_retry 标记为 True，并留下 context_overflow_recovery 事件。
    为什么重要：超限是长会话最常见的线上错误，整轮重跑既慢又重复烧 token；
    记录事件则让排障时能看到"这次回答是压缩后才成功的"。
    """
    model = OverflowThenSuccessModel()
    final = await ChatAgent(model=model).run(  # type: ignore[arg-type]
        {"message": "Continue the task."},
        {"tools": {}},
    )

    assert final["answer"] == "recovered after compact projection"
    assert model.calls == 2
    assert final["context_diagnostics"]["reactive_overflow_retry"] is True
    assert any(event["type"] == "context_overflow_recovery" for event in final["agent_events"])


def test_runtime_tool_projection_compacts_old_observations_without_mutating_audit_state() -> None:
    """场景：4 条各 5000 次重复的结果（模拟巨大的工具返回），未设置 model（纯离线逻辑）。

    期望：投影后的提示词观测总 token 不超过预算、确实省下了 token，
    但原始 state 里第一条观测的 answer_preview 一字未变。
    为什么重要：工具结果既要在提示词里压缩，又要在审计/回放里保持原样；
    如果在原地裁剪，事后复盘就再也看不到模型当时真正看到了什么。
    """
    agent = ChatAgent(model=None)
    state = {
        "observations": [
            {
                "tool_name": f"tool-{index}",
                "arguments": {"query": "auth"},
                "observation": {"answer_preview": (f"result-{index} " * 5000)},
            }
            for index in range(4)
        ],
        "agent_events": [],
    }
    original = state["observations"][0]["observation"]["answer_preview"]

    # 投影只产出「给提示词看的版本」，返回值里还带事件与统计信息
    prompt_observations, _events, stats = agent._runtime_tool_context(state)  # type: ignore[arg-type]

    assert stats["estimated_tokens_saved"] > 0
    assert stats["prompt_observation_tokens"] <= stats["observation_budget_tokens"]
    # 关键断言：原始审计数据未被就地改写
    assert state["observations"][0]["observation"]["answer_preview"] == original
    assert len(prompt_observations) >= 1


@pytest.mark.asyncio
async def test_recent_recall_uses_injected_history_without_memory_tool() -> None:
    """场景：用户问「我刚刚说了什么？」，而服务端已经注入了完整精确的历史。

    期望：直接答出"你刚刚说的是：hi"，route=direct_answer、tool_calls 为空，
    记忆查询工具一次都没被调用，模型只被调用 1 次且从未绑定工具，
    上下文策略事件记下 mode=direct_recent_history / reason=exact_recent_messages_available。
    为什么重要：历史就在提示词里还去调工具，既多花一次 LLM 往返又容易引入噪声；
    这个短路是最省时省钱的路径，必须保证它真的被触发。
    """
    model = RecentRecallModel()
    memory_tool_calls = 0

    async def get_thread_context(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        # 计数器用于证明「没有被调用」——被调用了断言就会失败
        nonlocal memory_tool_calls
        memory_tool_calls += 1
        return {"route": "thread_context", "tool_calls": [], "answer": "user: hi", "citations": []}

    final = await ChatAgent(model=model).run(  # type: ignore[arg-type]
        {"message": "我刚刚说了什么？"},
        {
            **_history_context(complete_and_exact=True),
            "tools": {"devflow_get_thread_context": get_thread_context},
        },
    )

    assert final["answer"] == "你刚刚说的是：hi"
    assert final["route"] == "direct_answer"
    assert final["tool_calls"] == []
    assert memory_tool_calls == 0
    assert model.calls == 1
    assert model.bind_tools_calls == 0
    policy = next(event for event in final["agent_events"] if event["type"] == "context_policy")
    assert policy["mode"] == "direct_recent_history"
    assert policy["reason"] == "exact_recent_messages_available"


@pytest.mark.asyncio
async def test_recent_recall_fetches_thread_context_when_prompt_history_is_incomplete() -> None:
    """场景：同样问「我刚刚说了什么？」，但注入的历史被截断（complete_and_exact=False）。

    期望：此时必须调用 devflow_get_thread_context 补齐，且不要跑去找证据（search_evidence），
    tool_calls 里只出现 thread_context 一个工具，策略原因为 recent_history_incomplete。
    为什么重要：历史不全却硬答，模型就会凭空编造"你刚才说过"；而乱调证据检索
    只会拿到与对话无关的文档。这里要验证降级路径选对了工具。
    """
    model = RecentRecallModel()
    memory_tool_calls = 0

    async def get_thread_context(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        nonlocal memory_tool_calls
        memory_tool_calls += 1
        return {
            "route": "thread_context",
            "tool_calls": [
                {"tool_name": "devflow_get_thread_context", "arguments": input_data, "status": "success"}
            ],
            "answer": "user: hi",
            "citations": [],
        }

    final = await ChatAgent(model=model).run(  # type: ignore[arg-type]
        {"message": "我刚刚说了什么？"},
        {
            **_history_context(complete_and_exact=False, exact_suffix_messages=1),
            "tools": {"devflow_get_thread_context": get_thread_context},
        },
    )

    assert memory_tool_calls == 1
    assert [call["tool_name"] for call in final["tool_calls"]] == ["devflow_get_thread_context"]
    # 负向断言：不能顺带去搜证据文档，否则说明策略路由串了
    assert not any(call["tool_name"] == "devflow_search_evidence" for call in final["tool_calls"])
    policy = next(event for event in final["agent_events"] if event["type"] == "context_policy")
    assert policy["reason"] == "recent_history_incomplete"


@pytest.mark.asyncio
async def test_exact_recall_request_verifies_with_thread_context() -> None:
    """场景：历史虽然完整，但用户明确要求「逐字核实我刚刚说的原话」。

    期望：即使历史完整也要调一次 thread_context 去核对原话，
    策略原因记为 exact_verification_requested。
    为什么重要：用户主动要求精确引用时，"大概记得"是不可接受的；
    宁可多一次工具调用，也不能给出不准确的原话。
    """
    invocations = 0

    async def get_thread_context(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        nonlocal invocations
        invocations += 1
        return {
            "route": "thread_context",
            "tool_calls": [
                {"tool_name": "devflow_get_thread_context", "arguments": input_data, "status": "success"}
            ],
            "answer": "user: hi",
            "citations": [],
        }

    context = _history_context(complete_and_exact=True)
    # 把最后一条用户消息改成「逐字核实」，模拟真实请求里出现精确性要求
    context["server_messages"][-1]["content"] = "请逐字核实我刚刚说的原话"
    final = await ChatAgent(model=RecentRecallModel()).run(  # type: ignore[arg-type]
        {"message": "请逐字核实我刚刚说的原话"},
        {**context, "tools": {"devflow_get_thread_context": get_thread_context}},
    )

    assert invocations == 1
    assert [call["tool_name"] for call in final["tool_calls"]] == ["devflow_get_thread_context"]
    policy = next(event for event in final["agent_events"] if event["type"] == "context_policy")
    assert policy["reason"] == "exact_verification_requested"


@pytest.mark.asyncio
async def test_cross_session_recall_uses_session_events() -> None:
    """场景：用户问「上次会话我们决定了什么？」——历史里没有跨会话信息。

    期望：调用 devflow_read_session_events 去读历史会话事件，只调这一个工具，
    策略原因为 cross_session_request。
    为什么重要：跨会话记忆是研发助手的核心卖点，选错工具（比如去搜 RAG 文档）
    就只能答"我在文档里没找到"，用户体验直接崩塌。
    """
    invocations = 0

    async def read_session_events(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        nonlocal invocations
        invocations += 1
        return {
            "route": "session_events",
            "tool_calls": [
                {"tool_name": "devflow_read_session_events", "arguments": input_data, "status": "success"}
            ],
            "answer": "上次会话决定使用短期令牌。",
            "citations": [],
        }

    context = _history_context(complete_and_exact=True)
    context["server_messages"][-1]["content"] = "上次会话我们决定了什么？"
    final = await ChatAgent(model=RecentRecallModel()).run(  # type: ignore[arg-type]
        {"message": "上次会话我们决定了什么？"},
        {**context, "tools": {"devflow_read_session_events": read_session_events}},
    )

    assert invocations == 1
    assert [call["tool_name"] for call in final["tool_calls"]] == ["devflow_read_session_events"]
    policy = next(event for event in final["agent_events"] if event["type"] == "context_policy")
    assert policy["reason"] == "cross_session_request"


@pytest.mark.asyncio
async def test_chat_agent_can_continue_from_list_files_to_read_file() -> None:
    """场景：模型先列目录、再读 README、最后作答（连续两轮工具调用）。

    期望：模型被调用 3 次；两个工具按 list → read 的顺序各执行一次；
    最终 route 取自最后一次工具（workspace_read_file）；
    agent_events 里 tool_use / tool_result 严格成对交替出现 4 条。
    为什么重要：多轮工具编排是 Agent 与普通问答的根本区别，
    事件顺序错乱会让前端进度条和审计日志都显示错误。
    """
    model = ScriptedToolModel()
    tool_invocations: list[tuple[str, dict[str, Any]]] = []

    async def list_files(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        # 记录调用顺序，用来断言模型确实按剧本调了工具
        tool_invocations.append(("list", input_data))
        return {
            "route": "workspace_list_files",
            "tool_calls": [{"tool_name": "workspace.list_files", "arguments": input_data, "status": "success"}],
            "answer": "README.md",
            "citations": [],
        }

    async def read_file(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        tool_invocations.append(("read", input_data))
        # 返回值照搬真实工具契约：answer 带 file:line 头，citations 供前端跳转
        return {
            "route": "workspace_read_file",
            "tool_calls": [{"tool_name": "workspace.read_file", "arguments": input_data, "status": "success"}],
            "answer": "README.md:1-1\n1: # Project",
            "citations": [{"type": "workspace_file", "id": "README.md", "title": "README.md"}],
        }

    agent = ChatAgent(model=model)  # type: ignore[arg-type]
    events = [
        event
        async for event in agent.run_stream(
            {"message": "Find the overview files, then read README.md."},
            {"tools": {"workspace.list_files": list_files, "workspace.read_file": read_file}},
        )
    ]

    final = [event for event in events if event["event"] == "final"][-1]["data"]
    # 两轮工具 + 一次收尾作答 = 3 次模型调用
    assert model.calls == 3
    assert [name for name, _input_data in tool_invocations] == ["list", "read"]
    assert final["route"] == "workspace_read_file"
    assert [call["tool_name"] for call in final["tool_calls"]] == ["workspace.list_files", "workspace.read_file"]
    assert [event["type"] for event in final["agent_events"] if event["type"] in {"tool_use", "tool_result"}] == [
        "tool_use",
        "tool_result",
        "tool_use",
        "tool_result",
    ]


@pytest.mark.asyncio
async def test_chat_agent_streams_tool_use_before_tool_finishes() -> None:
    """场景：工具内部用 Event 卡住不返回，测试在流式迭代中观察事件到达时机。

    期望：第一个事件就是 tool_use，且此时工具函数其实还没开始执行（started 未 set）；
    放开工具后 tool_result 到达，最后以 final 收尾；期间流不能卡住。
    为什么重要：前端要立刻显示「正在读取 README.md」的进度，
    如果实现是"等工具跑完再一起吐事件"，用户会对着空白等好几秒。
    """
    started = asyncio.Event()
    release = asyncio.Event()

    async def read_file(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        # 进入工具即置位 started，用于证明「事件先于执行」的时序
        started.set()
        # 挂起直到测试主动 release，模拟慢工具
        await release.wait()
        return {
            "route": "workspace_read_file",
            "tool_calls": [{"tool_name": "workspace.read_file", "arguments": input_data, "status": "success"}],
            "answer": "README.md:1-1\n1: # Project",
            "citations": [{"type": "workspace_file", "id": "README.md", "title": "README.md"}],
        }

    agent = ChatAgent(model=SingleToolModel())  # type: ignore[arg-type]
    stream = agent.run_stream(
        {"message": "Read README.md."},
        {"tools": {"workspace.read_file": read_file}},
    )

    first_event = await stream.__anext__()
    assert first_event["event"] == "tool_use"
    assert first_event["data"]["tool_name"] == "workspace.read_file"
    # 关键时序断言：还没真正执行工具，事件已经先发出去了
    assert not started.is_set()

    # 手动推进流：此时工具开始执行并阻塞在 release 上，所以下一个事件不该就绪
    next_event = asyncio.create_task(stream.__anext__())
    await asyncio.wait_for(started.wait(), timeout=1)
    assert not next_event.done()
    release.set()
    tool_result = await asyncio.wait_for(next_event, timeout=1)
    assert tool_result["event"] == "tool_result"
    remaining_events = [event async for event in stream]
    assert remaining_events[-1]["event"] == "final"


@pytest.mark.asyncio
async def test_project_overview_prefetches_readme_without_waiting_for_model_tool_choice() -> None:
    """场景：用户问「这个项目是什么？」，只提供了 list_files / read_file 两个工具。

    期望：即使用假模型根本不会选择工具（SummaryModel 直接给答案），
    Agent 也会主动预取：先 list 一次，再依次 read 了 3 个候选文件（README.md、
    README.zh-CN.md、package.json），共 4 次工具调用、4 对 tool_use/tool_result。
    为什么重要：问「项目是干什么的」这种高频问题，让模型凭工具选择去摸索要多好几轮往返；
    预取把首字延迟压到最低，同时保证答案基于真实仓库内容而不是模型臆想。
    """
    tool_invocations: list[tuple[str, dict[str, Any]]] = []

    async def list_files(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        tool_invocations.append(("list", input_data))
        # 返回三个候选概览文件，验证预取会按优先级逐个读取
        return {
            "route": "workspace_list_files",
            "tool_calls": [{"tool_name": "workspace.list_files", "arguments": input_data, "status": "success"}],
            "answer": "README.md\nREADME.zh-CN.md\npackage.json",
            "citations": [],
        }

    async def read_file(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        tool_invocations.append(("read", input_data))
        # 用传入路径动态生成答案，便于断言"每个候选文件都真的被读了"
        return {
            "route": "workspace_read_file",
            "tool_calls": [{"tool_name": "workspace.read_file", "arguments": input_data, "status": "success"}],
            "answer": f"{input_data['path']}:1-1\n1: # Clowder AI",
            "citations": [{"type": "workspace_file", "id": input_data["path"], "title": input_data["path"]}],
        }

    agent = ChatAgent(model=SummaryModel())  # type: ignore[arg-type]
    events = [
        event
        async for event in agent.run_stream(
            {"message": "What is this project?"},
            {"tools": {"workspace.list_files": list_files, "workspace.read_file": read_file}},
        )
    ]

    final = [event for event in events if event["event"] == "final"][-1]["data"]
    # 1 次 list + 3 次 read = 预取行为完全由代码决定，与模型无关
    assert [name for name, _input_data in tool_invocations] == ["list", "read", "read", "read"]
    assert final["route"] == "workspace_read_file"
    assert "coordinates AI agents" in final["answer"]
    assert len([event for event in final["agent_events"] if event["type"] == "tool_use"]) == 4
    assert len([event for event in final["agent_events"] if event["type"] == "tool_result"]) == 4


@pytest.mark.asyncio
async def test_chat_agent_combines_prefetched_evidence_with_rewritten_rag_searches() -> None:
    """场景：用户一个复合问题（刷新为什么失败 + 当初定的重试策略），
    同时上游已预取到一条 evidence，并注入了 server_context 文本。

    期望：模型改写出两个精准子查询并各检索一次（顺序与内容完全一致）；
    rag_strategy 事件标记为 hybrid_rag，pre_retrieval 记录了预取证据数、原始问题与来源，
    reasoning_retrieval 允许查询改写；最终 tool_call_count 为 2。
    为什么重要：混合检索（预取 + 推理改写）是 RAG 质量的关键，
    这里同时验证了"改写生效"和"预取证据没有被吞掉"，缺一就会出现答案只覆盖半个问题。
    """
    model = RewrittenRagModel()
    queries: list[str] = []

    async def search_rag(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        # 记录模型实际发出的每个 query，用于断言是"改写后"的而不是原句
        query = str(input_data.get("query") or "")
        queries.append(query)
        return {
            "route": "semantic_search",
            "tool_calls": [
                {
                    "tool_name": "rag.search_similar_documents",
                    "arguments": {"query": query},
                    "status": "success",
                }
            ],
            "answer": f"Evidence for {query}",
            "citations": [{"type": "project_doc", "id": query, "title": query}],
        }

    events = [
        event
        async for event in ChatAgent(model=model).run_stream(  # type: ignore[arg-type]
            {"message": "Why does token refresh fail, and what retry policy did we choose?"},
            {
                "tools": {"rag.search_similar_documents": search_rag},
                "evidence": [{"title": "prefetched auth overview"}],
                "server_context": "[Related Evidence]\nauth overview\n[/Related Evidence]",
            },
        )
    ]

    final = [event for event in events if event["event"] == "final"][-1]["data"]
    # 两个子查询必须按模型说出的顺序执行，不能去重也不能合并
    assert queries == ["token refresh timeout history", "idempotent retry decision"]
    assert [call["arguments"]["query"] for call in final["tool_calls"]] == queries
    strategy = next(event for event in final["agent_events"] if event["type"] == "rag_strategy")
    assert strategy["mode"] == "hybrid_rag"
    assert strategy["pre_retrieval"]["evidence_count"] == 1
    assert strategy["pre_retrieval"]["query"] == "Why does token refresh fail, and what retry policy did we choose?"
    assert strategy["pre_retrieval"]["sources"] == [
        {
            "source_id": "",
            "source_type": "evidence",
            "title": "prefetched auth overview",
            "score": None,
        }
    ]
    assert strategy["reasoning_retrieval"]["supports_query_rewrite"] is True
    assert final["completion"]["tool_call_count"] == 2


@pytest.mark.asyncio
async def test_chat_agent_evaluation_mode_preserves_prefetch_and_tool_retrieval_contexts() -> None:
    """场景：评测模式（evaluation_mode=True）下跑一次问答，既有预取 evidence 又有两次 RAG 工具检索。

    期望：evaluation_trace.target 为 ChatAgent；retrievals 的阶段序列是
    ["prefetch", "agent_tool", "agent_tool"]；预取片段以 "[project_doc] 标题: 内容" 形式
    进入 retrieved_contexts，工具检索的片段也在其中；tool_observations 有 2 条。
    非评测模式下则不应出现 evaluation_trace。
    为什么重要：RAG 评测要拿"模型实际看到的全部上下文"打分，
    如果预取证据没进 trace，评测分数会虚高（看起来模型很准，其实它压根没看到那段材料）。
    """
    model = RewrittenRagModel()

    async def search_rag(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        query = str(input_data.get("query") or "")
        # source_id 由 query 长度派生：同一 query 必然得到同一 id，便于断言去重与命中
        source_id = f"source-{len(query)}"
        snippet = f"Evidence returned for {query}."
        return {
            "route": "semantic_search",
            "tool_calls": [{"tool_name": "rag.search_similar_documents", "arguments": {"query": query}, "status": "success"}],
            "answer": snippet,
            "citations": [{"type": "project_doc", "id": source_id, "title": query}],
            "retrieved_contexts": [snippet],
            "retrieval_results": [
                {
                    "id": source_id,
                    "source_id": source_id,
                    "source_type": "project_doc",
                    "title": query,
                    "snippet": snippet,
                    "score": 0.9,
                }
            ],
        }

    final = await ChatAgent(model=model).run(  # type: ignore[arg-type]
        {"message": "How should auth retries work?"},
        {
            "tools": {"rag.search_similar_documents": search_rag},
            "evaluation_mode": True,
            "evidence": [
                {
                    "source_id": "prefetch-1",
                    "source_type": "project_doc",
                    "title": "Authentication policy",
                    "snippet": "Authentication retries must be idempotent.",
                    "score": 1.0,
                }
            ],
        },
    )

    trace = final["evaluation_trace"]
    assert trace["target"] == "ChatAgent"
    # 阶段顺序说明：先预取，再是模型主动发起的两次工具检索
    assert [item["stage"] for item in trace["retrievals"]] == ["prefetch", "agent_tool", "agent_tool"]
    assert (
        "[project_doc] Authentication policy: Authentication retries must be idempotent."
        in trace["retrieved_contexts"]
    )
    assert any(
        item.endswith("Evidence returned for token refresh timeout history.")
        for item in trace["retrieved_contexts"]
    )
    assert len(trace["tool_observations"]) == 2

    # 对照组：普通模式下不应产出评测 trace，避免把评测字段泄漏到线上响应
    normal = await ChatAgent(model=SummaryModel()).run(  # type: ignore[arg-type]
        {"message": "Summarize the evidence."},
        {"tools": {}, "evidence": []},
    )
    assert "evaluation_trace" not in normal


@pytest.mark.asyncio
async def test_project_overview_without_llm_uses_workspace_evidence(monkeypatch) -> None:
    """场景：没配 LLM（llm_api_key=None），用户仍问「这个项目做什么」。

    期望：不依赖大模型，靠预取的文件内容拼出答案（含 "Core capability: repository analysis"），
    完成状态为 completed、原因 project_context_loaded，工具错误数 0。
    为什么重要：没配 key / key 失效是自部署用户最常见的状态；
    此时仍要给出基于真实文件的可信回答，而不是报错或胡编。
    """
    # 把 API key 抹掉：模拟"完全不可用大模型"的部署环境
    monkeypatch.setattr(settings, "llm_api_key", None)

    async def list_files(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        return {
            "route": "workspace_list_files",
            "tool_calls": [{"tool_name": "workspace.list_files", "arguments": input_data, "status": "success"}],
            "answer": "README.md frontend/package.json",
            "citations": [],
        }

    async def read_file(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        return {
            "route": "workspace_read_file",
            "tool_calls": [{"tool_name": "workspace.read_file", "arguments": input_data, "status": "success"}],
            "answer": "README.md:1-2\n1: # DevFlow AI\n2: Core capability: repository analysis.",
            "citations": [],
        }

    events = [
        event
        async for event in ChatAgent(model=None).run_stream(
            {"message": "What does this project do?"},
            {"tools": {"workspace.list_files": list_files, "workspace.read_file": read_file}},
        )
    ]

    final = [event for event in events if event["event"] == "final"][-1]["data"]
    assert "Core capability: repository analysis" in final["answer"]
    assert final["completion"]["tool_call_count"] == 2
    assert final["completion"]["tool_error_count"] == 0
    assert final["completion"]["status"] == "completed"
    assert final["completion"]["reason"] == "project_context_loaded"


@pytest.mark.asyncio
async def test_open_issue_question_without_llm_uses_prefetched_issue_evidence(monkeypatch) -> None:
    """场景：没有 LLM，用户要求列出「未关闭的 Issue」并说明数据来源；
    上游预取了两条 issue evidence，其中 #18 是 open、#11 是 closed。

    期望：答案里包含 #18、排除 #11（按 state 过滤），并明确写出
    "本轮没有调用工作区源码工具"；tool_calls 为空、route=prefetched_evidence。
    工作区工具被替换成会抛异常的函数——一旦被调用测试就会失败，
    以此证明这条路径完全没有走源码工具。
    为什么重要：无 LLM 时最容易犯的错就是"把预取材料一股脑全倒出来"，
    导致把已关闭的 Issue 当成待办；同时"说明了用了哪些数据"是可信度要求。
    """
    monkeypatch.setattr(settings, "llm_api_key", None)

    async def unexpected_workspace_call(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        # 埋雷：如果路由错误地走到工作区工具，这里立刻炸出来
        raise AssertionError("Issue query must not be routed to workspace tools")

    events = [
        event
        async for event in ChatAgent(model=None).run_stream(
            {"message": "请列出当前仓库未关闭的 Issue，并说明你使用了哪些数据。"},
            {
                "tools": {"workspace.list_files": unexpected_workspace_call},
                # 两条 issue 证据：一条 open、一条 closed，用于验证 state 过滤逻辑
                "evidence": [
                    {
                        "source_type": "issue",
                        "source_id": "issue-18",
                        "title": "Issue #18: Token refresh race",
                        "snippet": "Token refresh can race under concurrent requests. state: open",
                        "metadata": {"number": 18, "state": "open"},
                    },
                    {
                        "source_type": "issue",
                        "source_id": "issue-11",
                        "title": "Issue #11: Retired task",
                        "snippet": "This task is complete. state: closed",
                        "metadata": {"number": 11, "state": "closed"},
                    },
                ],
            },
        )
    ]

    final = [event for event in events if event["event"] == "final"][-1]["data"]
    assert "Issue #18: Token refresh race" in final["answer"]
    # 已关闭的 Issue 必须被过滤掉，否则用户会去处理一个已经完成的任务
    assert "Issue #11: Retired task" not in final["answer"]
    assert "本轮没有调用工作区源码工具" in final["answer"]
    assert final["tool_calls"] == []
    assert final["route"] == "prefetched_evidence"
    assert final["completion"]["status"] == "completed"
    assert final["completion"]["reason"] == "prefetched_evidence_available"


@pytest.mark.asyncio
async def test_conversation_rag_tool_prefers_agent_rewritten_query(monkeypatch) -> None:
    """场景：对话工具 rag.search_similar_documents 同时收到原始 message 和模型改写后的 query。

    期望：检索与召回埋点都用改写后的 query（"refresh token idempotency race condition"），
    metadata.query_rewritten 为 True，返回的 tool_calls 参数里也是改写后的 query。
    为什么重要：用户口语化提问直接拿去向量检索命中率很低；
    改写生效却没有真正传给检索层，是这类 RAG 最常见的"假优化"。
    """
    # 用 SimpleNamespace 造最轻量的 repo / conversation，避免为这一个测试建整库
    repo = SimpleNamespace(id=uuid.uuid4(), full_name="acme/auth-service")
    conversation = SimpleNamespace(id=uuid.uuid4())
    captured: dict[str, Any] = {}

    def fake_semantic_search(repo_arg, query, context, db):
        # 抓取真正传给检索层的 query
        captured["search_query"] = query
        # 返回三元组：(答案文本, citations, retrieval_results)，与真实实现契约一致
        return (
            "rewritten evidence",
            [{"type": "project_doc", "id": "doc-1", "title": "Retry policy"}],
            [{"source_type": "project_doc", "source_id": "doc-1", "title": "Retry policy"}],
        )

    def fake_record_recall_event(*args, **kwargs):
        # 召回埋点也要用改写后的 query，否则分析用户真实意图时数据是错的
        captured["recall_query"] = kwargs["query"]
        captured["metadata"] = kwargs["metadata"]

    # 在模块级打桩：被测函数内部是按模块属性名调用这两个依赖的
    monkeypatch.setattr("app.api.routes.chat._semantic_search", fake_semantic_search)
    monkeypatch.setattr("app.api.routes.chat.record_recall_event", fake_record_recall_event)
    tools = _build_conversation_tools(repo, conversation, object())

    result = await tools["rag.search_similar_documents"](
        {
            "message": "Why does authentication fail?",
            "query": "refresh token idempotency race condition",
        },
        {},
    )

    assert captured["search_query"] == "refresh token idempotency race condition"
    assert captured["recall_query"] == "refresh token idempotency race condition"
    assert captured["metadata"]["query_rewritten"] is True
    assert result["tool_calls"][0]["arguments"]["query"] == "refresh token idempotency race condition"


def test_repo_checkout_path_reuses_existing_same_full_name_checkout(tmp_path, monkeypatch) -> None:
    """场景：checkout 目录里已存在一个旧路径 "old-id-296569015__clowder-ai"，
    而当前仓库 id 是新的 uuid、full_name 仍为 296569015/clowder-ai。

    期望：直接复用旧目录，而不是按新 id 新建一份。
    为什么重要：仓库重建记录（或 id 变更）后若按 id 建目录，
    会把整个仓库重新 clone 一遍，既慢又占磁盘，历史 checkout 也变成孤儿目录。
    """
    monkeypatch.setattr(settings, "repo_checkout_dir", str(tmp_path))
    # 用 tmp_path 造出"已存在的旧 checkout"，目录名故意与当前 repo.id 不匹配
    stale_checkout = tmp_path / "old-id-296569015__clowder-ai"
    stale_checkout.mkdir()
    repo = SimpleNamespace(id=uuid.uuid4(), full_name="296569015/clowder-ai")

    assert _repo_checkout_path(repo) == stale_checkout.resolve()


def test_assignment_prompt_treats_empty_team_as_no_candidates() -> None:
    """场景：问「这个 Issue 应该分给谁」，但仓库里没有任何 DevFlow 团队成员；
    server_context 里恰好提到 README 中的角色/比喻名（布偶猫、opus）。

    期望：提示词里带 team_members: []，明确写出"当前仓库还没有配置 DevFlow 团队成员"，
    并禁止把仓库内容里的人名/代号当候选人。
    为什么重要：README 里的宠物名/项目代号很容易被模型误当成人名，
    给出"建议分给布偶猫"这种荒谬结论；必须靠提示词硬约束兜住。
    """
    agent = ChatAgent(model=SummaryModel())  # type: ignore[arg-type]
    # 直接调提示词构造函数，绕开 LLM，纯测提示词内容
    prompt = agent._tool_calling_user_prompt(
        {
            "message": "这个桌面化 Issue 应该分给谁？",
            "context": {"team_members": [], "server_context": "README mentions 布偶猫 / opus as project personas."},
            "observations": [],
            "agent_events": [],
            "tools": {},
        }
    )

    assert "team_members: []" in prompt
    assert "当前仓库还没有配置 DevFlow 团队成员" in prompt
    assert "不要把仓库内容" in prompt


@pytest.mark.asyncio
async def test_final_prompt_keeps_assignment_candidates_scoped_to_team_members() -> None:
    """场景：同上（团队为空 + 仓库里出现昵称），但走的是「根据工具观测生成最终答案」这条提示词。

    期望：CapturingModel 抓到的最终提示词里仍有 team_members: []、
    "团队比喻都不能当成人"、"在团队面板中添加开发者" 三条约束。
    为什么重要：tool-calling 提示词和 final 提示词是两套模板，
    只在一处加约束很常见；漏掉后者，模型仍会在最终答案里编出不存在的人。
    """
    model = CapturingModel()
    agent = ChatAgent(model=model)  # type: ignore[arg-type]
    # 观测里故意提供一段含"布偶猫"的工具结果，作为诱导性上下文
    await agent._final_from_observations(
        {
            "message": "这个桌面化 Issue 应该分给谁？",
            "context": {"team_members": [], "server_context": "README mentions 布偶猫 / opus as project personas."},
            "observations": [{"answer": "Issue text mentions 布偶猫 and desktop work."}],
            "agent_events": [],
        }
    )
    # 把抓到的全部 prompt 消息拼成一整段再断言，避免关心消息被拆成几条
    prompt = "\n".join(str(message.content) for message in model.messages)

    assert "team_members: []" in prompt
    assert "团队比喻都不能当成人" in prompt
    assert "在团队面板中添加开发者" in prompt


@pytest.mark.asyncio
async def test_chat_agent_retries_transient_tool_error_once() -> None:
    """场景：读文件工具第一次抛 TimeoutError，第二次成功。

    期望：工具被调用 2 次，最终 tool_calls[0].status 为 success，
    agent_events 里留下 tool_retry + error_kind=timeout，完成状态 completed。
    为什么重要：网络抖动导致的超时非常常见，自动重试一次能救回大量偶发失败；
    但必须记录重试事件，否则排障时无法区分"一次成功"和"重试后成功"。
    """
    attempts = 0

    async def flaky_read_file(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        nonlocal attempts
        attempts += 1
        # 只有第一次失败，第二次走正常返回，模拟瞬时故障
        if attempts == 1:
            raise TimeoutError("temporary timeout")
        return {
            "route": "workspace_read_file",
            "tool_calls": [{"tool_name": "workspace.read_file", "arguments": input_data, "status": "success"}],
            "answer": "README.md:1-1\n1: # Project",
            "citations": [],
        }

    agent = ChatAgent(model=SingleToolModel())  # type: ignore[arg-type]
    events = [
        event
        async for event in agent.run_stream(
            {"message": "Read README.md."},
            {"tools": {"workspace.read_file": flaky_read_file}},
        )
    ]

    final = [event for event in events if event["event"] == "final"][-1]["data"]
    # 恰好重试一次：既证明重试生效，也证明没有无脑重试多次
    assert attempts == 2
    assert final["tool_calls"][0]["status"] == "success"
    assert any(event["type"] == "tool_retry" and event["error_kind"] == "timeout" for event in final["agent_events"])
    assert final["completion"]["status"] == "completed"


@pytest.mark.asyncio
async def test_chat_agent_stops_repeated_identical_tool_calls() -> None:
    """场景：模型每次都请求"完全相同的 read_file 调用"，永远不收敛。

    期望：工具实际只执行了 max_repeated_tool_calls 次就熔断，
    route 变为 tool_loop_guard，完成原因为 repeated_tool_call，并留下 agent_loop_stop 事件。
    为什么重要：真实模型偶尔会陷入自我循环，没有熔断就会无上限地烧 token 且永不返回，
    这是最容易造成线上费用事故的一类 bug。
    """
    invocations = 0

    async def read_file(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        nonlocal invocations
        invocations += 1
        return {
            "route": "workspace_read_file",
            "tool_calls": [{"tool_name": "workspace.read_file", "arguments": input_data, "status": "success"}],
            "answer": "README.md:1-1\n1: # Project",
            "citations": [],
        }

    agent = ChatAgent(model=RepeatingToolModel())  # type: ignore[arg-type]
    events = [
        event
        async for event in agent.run_stream(
            {"message": "Read README.md."},
            {"tools": {"workspace.read_file": read_file}},
        )
    ]

    final = [event for event in events if event["event"] == "final"][-1]["data"]
    # 熔断阈值取自实例属性，断言执行次数恰好等于阈值（而不是无限次）
    assert invocations == agent.max_repeated_tool_calls
    assert final["route"] == "tool_loop_guard"
    assert final["completion"]["reason"] == "repeated_tool_call"
    assert any(event["type"] == "agent_loop_stop" and event["reason"] == "repeated_tool_call" for event in final["agent_events"])
