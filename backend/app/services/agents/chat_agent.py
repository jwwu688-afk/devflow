"""对话 Agent：本项目的「前台」Agent。

与其它 Agent 的关系：它**不属于** Planner → 专用 Agent → Observer → Synthesis 那条批处理链，
而是用户直接对话时走的主路径。区别在于：
- 那条链是「一次性规划 + 多 Agent 并行 + 综合」，产出工程决策备忘录；
- 这个 Agent 是「流式多轮工具调用」，边想边调工具、边把结果推给前端，最终给一段回答。

运行形态：LangGraph 的 StateGraph 循环（model_decision → tool_action → 回到 model_decision），
由 max_steps 限制轮数。每次循环都是一次 LLM 调用，所以 max_steps 同时是「成本上限」和
「防死循环」的保险。

**花钱的地方（按调用顺序）**：
1. 每轮 model_decision 一次 chat 调用（带完整工具 schema + 历史 + 证据，是主要成本）；
2. 上下文溢出时会压缩历史后重试一次（_compress_history_for_retry）；
3. 工具本身可能触发下游 Agent（如 workflow.run_engineering_review），那些 Agent 又会各自调模型。

**省钱/省时的旁路**（在进入模型循环之前就短路，见 run_stream）：
- 技能激活失败 → 直接报错；
- 记忆策略判定「最近历史就能回答」→ 不调模型，直接由历史生成回答；
- 命中「必须先读某个记忆工具」→ 只跑那一个工具，再总结；
- 问项目概览 → 直接预取文件上下文，不跑完整循环；
- 没有配置模型 → 用预取证据拼一个本地回答。

**trace 记录**：全程往 state["agent_events"] 里追加事件
（agent_loop_start / skill_activated / rag_strategy / context_policy / tool_call / ...），
最终随 final 事件一次性吐给前端，前端时间线就是这么画出来的。
"""

import json
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Literal, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from app.core.config import settings
from app.services.agents.base import BaseAgent
from app.services.context_compression import (
    calculate_context_pressure,
    clip_to_token_budget,
    default_context_budget,
    estimate_tokens,
    is_context_overflow_error,
    prompt_cache_model_kwargs,
)
from app.services.skills import SkillRegistry, SkillRegistryError


# 工具的统一签名：(动作参数, 运行时上下文) → 结果 dict。
# 由上层（API 层）组装后通过 context["tools"] 注入，本 Agent 不直接依赖具体工具实现。
ConversationTool = Callable[[dict[str, Any], dict[str, Any]], Awaitable[dict[str, Any]]]
# 工具错误分类。分类的目的是决定「要不要重试」：timeout / rate_limited /
# transient_network 可重试，其余（参数错、没权限、数据不存在）重试也是同样结果。
ToolErrorKind = Literal[
    "unknown_tool",
    "timeout",
    "rate_limited",
    "transient_network",
    "permission_denied",
    "data_not_found",
    "model_output_invalid",
    "tool_runtime_error",
]


class GenericToolArgs(BaseModel):
    """工具参数的「万能 schema」。

    为什么用一个超集模型而不是给每个工具单独定义参数类：
    工具是运行时注入的，编译期不知道有哪些工具；用 extra="allow" 的宽模型
    让模型可以传任意字段，再在 _normalize_action_input 里按工具名做字段整形。
    代价是 schema 很长（占 prompt token，见 _tool_schema_estimated_tokens）。
    """

    # extra="allow" 是关键：允许模型传入下面没声明的字段，否则会被 pydantic 丢弃
    model_config = ConfigDict(extra="allow")

    message: str | None = Field(default=None, description="原始用户消息")
    query: str | None = Field(default=None, description="搜索查询")
    keyword: str | None = Field(default=None, description="关键词过滤")
    limit: int | None = Field(default=None, description="结果数量上限")
    repo_id: str | None = Field(default=None, description="仓库 UUID")
    repo_full_name: str | None = Field(default=None, description="仓库完整名称")
    conversation_id: str | None = Field(default=None, description="会话 UUID")
    session_id: str | None = Field(default=None, description="会话片段 UUID")
    message_id: str | None = Field(default=None, description="需要精确读取的消息 UUID")
    before_message_id: str | None = Field(default=None, description="只读取该消息之前的记录")
    after_message_id: str | None = Field(default=None, description="只读取该消息之后的记录")
    offset: int | None = Field(default=None, description="从消息正文的字符偏移量开始读取")
    max_chars: int | None = Field(default=None, description="本次最多返回的正文字符数")
    include_transcript: bool | None = Field(default=None, description="读取封存会话时是否包含原始 transcript")
    route: str | None = Field(default=None, description="仓库路由或意图")
    path: str | None = Field(default=None, description="相对工作区的文件或目录路径")
    pattern: str | None = Field(default=None, description="文本或 glob 模式")
    start_line: int | None = Field(default=None, description="读取文件时从第几行开始，行号从 1 开始")
    line_count: int | None = Field(default=None, description="最多读取多少行")


class ConversationState(TypedDict, total=False):
    """LangGraph 的图中状态。

    total=False 表示所有键都可选：图可以在任意字段缺失的情况下运行，
    也方便用 state.get(...) 而不是硬下标（避免 KeyError 打断流式输出）。
    这个 state 既是图的输入输出，也是最终的 trace 载体。
    """

    message: str  # 本轮的原始用户消息
    context: dict[str, Any]  # 运行时上下文（工具、仓库信息、历史、技能等）
    tools: dict[str, StructuredTool]  # 给模型看的 LangChain 工具
    tool_name_map: dict[str, str]  # 安全工具名 → 对外展示名（前端不认内部名）
    observations: list[dict[str, Any]]  # 每次工具调用的结果
    agent_events: list[dict[str, Any]]  # trace 事件流，前端时间线的数据源
    tool_calls: list[dict[str, Any]]  # 工具调用记录（含状态、错误、耗时）
    citations: list[dict[str, Any]]  # 引用来源，从工具结果里汇总
    pending_tool_calls: list[dict[str, Any]]  # 当前这轮待执行的工具调用
    stream_events: list[dict[str, Any]]  # 待推送给前端的增量事件
    route: str  # 意图路由，决定最终回答走哪条路
    step: int  # 已执行的循环轮数，与 max_steps 比较
    final_answer: str  # 最终回答
    stop_reason: str  # 停止原因（用于解释「为什么没继续跑」）
    completion_status: str  # running / completed / blocked
    tool_call_counts: dict[str, int]  # 每个工具被调几次，用于检测重复调用
    evaluation_trace: dict[str, Any]  # 评测模式下的额外记录（线上不影响）
    model_usage: list[dict[str, Any]]  # 每次模型调用的 token 用量
    prompt_diagnostics: list[dict[str, Any]]  # prompt 体积诊断
    runtime_context_stats: dict[str, Any]  # 上下文裁剪统计
    context_overflow_recovery_attempted: bool  # 是否已尝试过溢出恢复（只重试一次）


class ChatAgent(BaseAgent):
    """原生工具调用的对话 Agent。"""

    name = "chat_agent"
    description = "使用原生工具调用的对话 Agent。LLM 选择 LangChain 工具，记忆工具通过 MCP 执行。"
    # 这里列的是「可能被用到」的工具名（含子 Agent 与工作流入口）。
    # 实际可用工具由 context["tools"] 决定，这份清单主要用于文档与能力声明。
    available_tools = [
        "devflow_search_evidence",
        "devflow_get_thread_context",
        "devflow_read_session_events",
        "workspace.list_files",
        "workspace.read_file",
        "workspace.search_code",
        "repo_health.generate_summary",
        "rag.search_similar_documents",
        "issue_agent.analyze_issue",
        "pr_review_agent.analyze_pr",
        "ci_debug_agent.analyze_workflow_run",
        "report_agent.generate_weekly_report",
        "safety_agent.classify_action_risk",
        "workflow.run_engineering_review",
    ]

    # 三个「刹车」参数：
    # max_steps=12 限制循环轮数，直接封住单次对话的模型调用成本；
    # max_repeated_tool_calls=2 允许同一工具重试一次（应对瞬时失败），第三次就判死循环；
    # tool_retry_attempts=1 表示可重试错误只额外试一次，避免在限流时越试越糟。
    max_steps = 12
    max_repeated_tool_calls = 2
    tool_retry_attempts = 1

    def __init__(self, model: ChatOpenAI | None = None) -> None:
        """构造时可注入模型（测试用假模型），否则按配置创建。

        没有 api_key 时 model 保持 None —— 这不是错误状态，run_stream 里有
        「无模型也能给出基于证据的回答」的分支，用于未配置 LLM 的部署环境。
        """
        self.model = model
        if self.model is None and settings.llm_api_key:
            self.model = ChatOpenAI(
                model=settings.llm_model,
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url,
                # temperature=0：工程助手场景要的是稳定可复现，不需要创造性
                temperature=0,
                timeout=60,
                # 开启 prompt cache：同一段系统提示在多轮/多次调用间可复用，显著降本
                model_kwargs=prompt_cache_model_kwargs(),
            )

    async def run(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """非流式入口：把流式输出「抽干」，只留最后一个 final 事件。

        这样 run 和 run_stream 共享同一套逻辑，不必写两遍；
        代价是非流式调用方拿不到中间过程（中间事件被丢弃，但 trace 仍在 final 里）。
        """
        final: dict[str, Any] | None = None
        async for event in self.run_stream(input_data, context):
            if event["event"] == "final":
                final = event["data"]
        # 兜底：理论上流一定会给 final，但生成器被提前关闭时可能没有
        return final or {
            "run_id": "",
            "route": "unknown",
            "tool_calls": [],
            "answer": "没有生成最终回答。",
            "citations": [],
        }

    async def run_stream(self, input_data: dict[str, Any], context: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """流式主入口：一连串「旁路判断」→ 要么短路返回，要么进入 LangGraph 工具循环。

        事件协议：yield {"event": <类型>, "data": ...}，最后一帧固定是 "final"。
        调用方（API 层）把它转成 SSE 推给前端。

        这段代码的整体形状是「一连串 if ... return」，每个 return 之前都会
        yield 完整回答 + final 事件。看起来重复，但每条的短路条件与路由名不同，
        合并成配置表反而更难读、更容易漏掉某个 final 字段。
        """
        message = str(input_data.get("message") or "")
        # 复制一份 context：下面会往里写 active_skills，不能污染调用方持有的 dict
        runtime_context = dict(context)
        # 技能激活可能失败（注册表损坏/配置错），失败时直接结束，不进入模型循环
        try:
            active_skills = self._activate_skills(message, runtime_context)
        except SkillRegistryError as exc:
            answer = f"Skill 激活失败：{exc}"
            async for chunk in self._stream_answer(answer):
                yield chunk
            yield self._event(
                "final",
                {
                    "run_id": "",
                    "route": "skill_error",
                    "tool_calls": [],
                    "agent_events": [{"type": "skill_activation_error", "error": str(exc)}],
                    "answer": answer,
                    "citations": [],
                    # completion.status=blocked：告诉前端这是「被阻断」而不是「正常完成」
                    "completion": {"status": "blocked", "reason": "skill_activation_error", "steps": 0, "tool_call_count": 0, "tool_error_count": 0},
                },
            )
            return
        runtime_context["active_skills"] = active_skills
        # 工具由外部注入；这里做一次「dict 签名 → LangChain 工具」的适配
        raw_tools: dict[str, ConversationTool] = runtime_context["tools"]
        tools = self._build_langchain_tools(raw_tools, runtime_context)
        # 初始化图状态：所有字段都显式给初值，避免图里 state.get() 拿到 None
        state: ConversationState = {
            "message": message,
            "context": runtime_context,
            "tools": tools,
            "tool_name_map": self._tool_name_map(raw_tools),
            "observations": [],
            "agent_events": [],
            "tool_calls": [],
            "citations": [],
            "pending_tool_calls": [],
            "stream_events": [],
            "route": "direct_answer",
            "step": 0,
            "final_answer": "",
            "stop_reason": "",
            "completion_status": "running",
            "tool_call_counts": {},
            "model_usage": [],
            "prompt_diagnostics": [],
            "runtime_context_stats": {},
            "context_overflow_recovery_attempted": False,
        }
        # evaluation_mode 只在评测/回放场景开启，生产默认关闭：它会额外记录每一步的判定依据
        if context.get("evaluation_mode"):
            state["evaluation_trace"] = self._initial_evaluation_trace(message, context)
        # 第一帧 trace：记录这次循环用的是什么模式、上限多少、有哪些工具可选
        self._append_agent_event(
            state,
            "agent_loop_start",
            {
                "mode": "langgraph_tool_calling_loop",
                "max_steps": self.max_steps,
                "available_tools": list(self._tool_name_map(raw_tools).values()),
            },
        )
        # 每个被激活的技能都记一条事件：前端据此展示「本次用了哪些技能」，
        # 也是排查「技能为什么没生效」的第一手数据
        for skill in active_skills:
            activation_event = {
                key: value
                for key, value in skill.items()
                if key
                in {
                    "skill_name",
                    "skill_title",
                    "skill_version",
                    "entrypoint",
                    "tools",
                    "workflow_steps",
                    "output_contract",
                    "safety_level",
                    "activation_mode",
                    "activation_reason",
                    "instruction_digest",
                }
            }
            # 技能内的资源只记元信息，剔除 content：技能资源可能很大（整份文档），
            # 全塞进 trace 会让事件体积失控
            activation_event["resources"] = [
                {key: value for key, value in resource.items() if key != "content"}
                for resource in skill.get("resources") or []
                if isinstance(resource, dict)
            ]
            self._append_agent_event(
                state,
                "skill_activated",
                activation_event,
            )
        # 「预检索」：上游可能已经把检索结果放进 context["evidence"]，这里不再重复查。
        # "evidence" 是否存在于 context 是开关（哪怕空列表也算开启），这是与调用方的隐式契约
        prefetched_evidence = context.get("evidence") if isinstance(context.get("evidence"), list) else []
        pre_retrieval_enabled = "evidence" in context
        # 「推理中检索」：模型可以在循环里主动调 rag 工具补查，能力更强但更贵（多一次工具+回复）
        reasoning_retrieval_enabled = "rag.search_similar_documents" in raw_tools
        # 只取前 8 条来源写进 trace：这是给人看的摘要，不是给模型用的全部数据
        pre_retrieval_sources = [
            {
                "source_id": str(item.get("source_id") or item.get("id") or ""),
                "source_type": str(item.get("source_type") or "evidence"),
                "title": str(item.get("title") or "未命名证据"),
                "score": item.get("score") if isinstance(item.get("score"), (int, float)) else None,
            }
            for item in prefetched_evidence[:8]
            if isinstance(item, dict)
        ]
        # 记录本次 RAG 策略：hybrid 表示「预检索 + 模型可自行再查」，tool_calling 表示只有后者。
        # 这个字段是排查「回答为什么不带来源」的关键——先看这里有没有预检索的证据
        self._append_agent_event(
            state,
            "rag_strategy",
            {
                "mode": "hybrid_rag" if pre_retrieval_enabled and reasoning_retrieval_enabled else "tool_calling",
                "pre_retrieval": {
                    "enabled": pre_retrieval_enabled,
                    "evidence_count": len(prefetched_evidence),
                    "query": message,
                    "sources": pre_retrieval_sources,
                },
                "reasoning_retrieval": {
                    "enabled": reasoning_retrieval_enabled,
                    "supports_query_rewrite": reasoning_retrieval_enabled,
                    "supports_multiple_searches": reasoning_retrieval_enabled,
                },
            },
        )

        # 记忆上下文策略：判断「这个问题是否必须先去翻会话历史」。
        # 返回 None 表示没有特殊要求，走正常的工具循环
        memory_policy = self._memory_context_policy(message, context)
        if memory_policy:
            self._append_agent_event(state, "context_policy", memory_policy)
        # 旁路 1：判定「最近的历史消息就足以回答」——比如「刚才那个 PR 叫什么」。
        # 这样能省掉一整轮模型循环（最便宜的路径），代价是判断本身依赖关键词启发式
        if memory_policy and memory_policy.get("mode") == "direct_recent_history":
            state["final_answer"] = await self._direct_answer_from_recent_history(state)
            answer = str(state.get("final_answer") or "").strip()
            # 模型没给出可用答案时，用本地模板从历史消息拼一句——保证这条旁路永远有输出
            if not answer:
                answer = self._local_recent_history_answer(state)
            self._stop_loop(state, "recent_history_available", "completed")
            self._record_skill_validation(state)
            async for chunk in self._stream_answer(answer):
                yield chunk
            yield self._event(
                "final",
                {
                    "run_id": "",
                    "route": "direct_answer",
                    "tool_calls": [],
                    "agent_events": state.get("agent_events") or [],
                    "answer": answer,
                    "citations": [],
                    "completion": self._completion_summary(state),
                    **self._runtime_metadata(state),
                    **self._evaluation_result(state),
                },
            )
            return

        required_memory_tool = str((memory_policy or {}).get("tool") or "") or None
        # 旁路 2：策略指定了「必须先读这个记忆工具」（如读取封存会话）。
        # 这里只跑这一个工具，然后直接总结，不进模型循环——因为继续让模型自由选工具
        # 反而可能跳过必要的那次读取，导致答非所问
        if required_memory_tool and required_memory_tool in raw_tools:
            async for event in self._prefetch_memory_context(state, required_memory_tool):
                yield event
            state["final_answer"] = await self._final_from_observations(state)
            answer = str(state.get("final_answer") or "").strip()
            if not answer:
                answer = await self._fallback_answer(state)
            self._stop_loop(state, "required_memory_context_loaded", self._tool_completion_status(state))
            self._record_skill_validation(state)
            async for chunk in self._stream_answer(answer):
                yield chunk
            yield self._event(
                "final",
                {
                    "run_id": "",
                    "route": state.get("route") or required_memory_tool,
                    "tool_calls": state.get("tool_calls") or [],
                    "agent_events": state.get("agent_events") or [],
                    "answer": answer,
                    "citations": state.get("citations") or [],
                    "completion": self._completion_summary(state),
                    **self._runtime_metadata(state),
                    **self._evaluation_result(state),
                },
            )
            return

        # 旁路 3：「讲讲这个项目的结构」这类概览问题，直接预取文件上下文再总结。
        # 不走循环的原因：概览需要的是广度（多读几个文件），而不是模型反复试探要读哪个
        if self._asks_project_overview(message):
            async for event in self._prefetch_project_context(state):
                yield event
            state["final_answer"] = await self._final_from_observations(state)
            answer = str(state.get("final_answer") or "").strip()
            if not answer:
                answer = await self._fallback_answer(state)
            self._stop_loop(state, "project_context_loaded", self._tool_completion_status(state))
            self._record_skill_validation(state)
            async for chunk in self._stream_answer(answer):
                yield chunk
            yield self._event(
                "final",
                {
                    "run_id": "",
                    "route": state.get("route") or "workspace_read_file",
                    "tool_calls": state.get("tool_calls") or [],
                    "agent_events": state.get("agent_events") or [],
                    "answer": answer,
                    "citations": state.get("citations") or [],
                    "completion": self._completion_summary(state),
                    **self._runtime_metadata(state),
                    **self._evaluation_result(state),
                },
            )
            return

        # 旁路 4：未配置模型（无 api_key）。此时不报错，而是用预取证据拼一个本地回答，
        # 让「没接大模型」的部署也能用上检索功能；状态标 blocked 以区分「没配模型」与「跑完了」
        if self.model is None:
            answer = self._local_prefetched_evidence_answer(state) or (
                "当前没有取得足够证据。请先确认仓库已连接并完成同步；开放式对话还需要配置 LLM。"
            )
            evidence = (state.get("context") or {}).get("evidence")
            has_evidence = isinstance(evidence, list) and any(isinstance(item, dict) for item in evidence)
            self._stop_loop(
                state,
                "prefetched_evidence_available" if has_evidence else "llm_not_configured",
                "completed" if has_evidence else "blocked",
            )
            self._record_skill_validation(state)
            async for chunk in self._stream_answer(answer):
                yield chunk
            yield self._event(
                "final",
                {
                    "run_id": "",
                    "route": "prefetched_evidence" if has_evidence else "direct_answer",
                    "tool_calls": [],
                    "agent_events": state.get("agent_events") or [],
                    "answer": answer,
                    "citations": self._prefetched_evidence_citations(state),
                    "completion": self._completion_summary(state),
                    **self._runtime_metadata(state),
                    **self._evaluation_result(state),
                },
            )
            return

        # 主路径：构建 LangGraph 图并流式跑完整个「模型决策 ↔ 工具执行」循环。
        # _stream_agent_graph 内部会边跑边 yield 增量事件（文本块、工具调用、观察结果）
        graph = self._build_agent_graph(self._tool_model(tools))
        async for event in self._stream_agent_graph(graph, state):
            yield event

        # 图跑完后统一收尾：下面这段是「所有路径共用」的，保证 final 帧结构一致
        answer = str(state.get("final_answer") or "").strip()
        if not answer:
            # 图跑完却没有答案（比如达到步数上限、或模型一直只调工具不说话）→ 兜底回答
            answer = await self._fallback_answer(state)
        if state.get("completion_status") == "running":
            self._stop_loop(state, "fallback_answer_ready", self._tool_completion_status(state))
        self._record_skill_validation(state)
        async for chunk in self._stream_answer(answer):
            yield chunk
        yield self._event(
            "final",
            {
                "run_id": "",
                "route": state.get("route") or "direct_answer",
                "tool_calls": state.get("tool_calls") or [],
                "agent_events": state.get("agent_events") or [],
                "answer": answer,
                "citations": state.get("citations") or [],
                "completion": self._completion_summary(state),
                **self._runtime_metadata(state),
                **self._evaluation_result(state),
            },
        )

    def _build_agent_graph(self, tool_model: Any):
        """构建 LangGraph 状态图。

        拓扑（三个节点 + 条件边）：
            START → model_decision
            model_decision --有工具调用--> tool_action --还有待执行--> tool_action（自环）
                                             |--否则且未超步数--> model_decision（回到模型）
                                             |--超步数且无答案--> max_steps_stop → END
            model_decision --无工具调用--> END

        注意 tool_action 的自环：一次模型回复可能带多个工具调用，state 里用
        pending_tool_calls 队列逐个消费，避免并发写同一份 state。
        """
        async def model_decision_node(state: ConversationState) -> dict[str, Any]:
            """图节点包装：LangGraph 节点只能接收 state，所以用闭包补上 tool_model。"""
            # 用闭包把 tool_model 绑进节点：LangGraph 节点签名只接收 state，不能直接传参
            return await self._graph_model_decision(state, tool_model)

        graph = StateGraph(ConversationState)
        graph.add_node("model_decision", model_decision_node)
        graph.add_node("tool_action", self._graph_tool_action)
        graph.add_node("max_steps_stop", self._graph_max_steps_stop)
        graph.add_edge(START, "model_decision")
        graph.add_conditional_edges(
            "model_decision",
            self._route_after_model_decision,
            {"tools": "tool_action", "end": END},
        )
        graph.add_conditional_edges(
            "tool_action",
            self._route_after_tool_action,
            {"tools": "tool_action", "model": "model_decision", "limit": "max_steps_stop"},
        )
        graph.add_edge("max_steps_stop", END)
        return graph.compile(name="chat_agent_tool_calling_graph")

    async def _stream_agent_graph(self, graph: Any, state: ConversationState) -> AsyncIterator[dict[str, Any]]:
        """把图执行过程转成前端事件流。

        stream_mode="debug" 给的是节点级事件，这里只关心两类：
        - type=="task" 且节点是 tool_action：工具即将执行，先推一个「正在用工具」预览，
          让用户不用等工具跑完就能看到进度（体验关键）；
        - type=="task_result"：节点跑完后回填 state，并把它产生的 stream_events 推出去。

        节点返回值只有 stream_events 不回写 state（它是纯增量），其余键覆盖当前状态。
        """
        current_state: ConversationState = dict(state)
        # recursion_limit 给的是「图内部步数」上限，比业务 max_steps 宽（×8+20），
        # 因为一次业务步可能对应多个节点跳转；真正的业务刹车在 max_steps 上
        config = {"recursion_limit": self.max_steps * 8 + 20}
        async for graph_event in graph.astream(current_state, config=config, stream_mode="debug"):
            if not isinstance(graph_event, dict):
                continue
            payload = graph_event.get("payload") if isinstance(graph_event.get("payload"), dict) else {}
            node_name = str(payload.get("name") or "")
            if graph_event.get("type") == "task" and node_name == "tool_action":
                # 预览事件用 payload["input"]（节点入参 = 执行前的 state），
                # 因为此时工具还没跑，state 里是待执行的调用信息
                preview = self._preview_tool_use_event(payload.get("input"))
                if preview:
                    yield self._event("tool_use", preview)
                continue
            if graph_event.get("type") != "task_result":
                continue
            result = payload.get("result")
            if not isinstance(result, dict):
                continue
            for key, value in result.items():
                if key != "stream_events":
                    # 双写：current_state 供后续 astream 使用，state 供调用方在图跑完后读取
                    current_state[key] = value
                    state[key] = value
            for event in result.get("stream_events") or []:
                if isinstance(event, dict):
                    yield event

    async def _graph_model_decision(self, state: ConversationState, tool_model: Any) -> dict[str, Any]:
        """「模型决策」节点：一次 LLM 调用，决定「继续调工具」还是「给出最终答案」。

        这是整个循环里唯一稳定花钱的地方（每步一次）。三步走：
        1. 组 prompt（系统提示 + 用户提示，用户提示里塞了历史、证据、工具结果摘要）；
        2. 调模型；若报上下文溢出，压缩后**只重试一次**；
        3. 按模型输出分流：有 tool_calls → 交给 tool_action；没有 → 这段文本就是最终答案。
        """
        next_state: ConversationState = dict(state)
        # step 在这里自增：模型决策次数即「业务步数」，是 max_steps 的计量口径
        step = int(next_state.get("step") or 0) + 1
        next_state["step"] = step
        self._append_agent_event(next_state, "agent_step", {"step": step, "phase": "model_decision"})
        system_prompt = self._tool_calling_system_prompt()
        user_prompt = self._tool_calling_user_prompt(next_state)
        messages: list[BaseMessage] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        # 先估算 prompt token 并记录诊断，再真调模型：事后可对比「估算 vs 实际用量」，
        # 是排查成本异常的第一手数据
        estimated_prompt_tokens = self._record_prompt_diagnostics(
            next_state,
            phase="model_decision",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        try:
            response = await tool_model.ainvoke(messages)
        except Exception as exc:
            # 只处理「上下文溢出」这一种异常，其他异常照原样抛给上层（上层会转成错误回答）。
            # 二次溢出就直接抛：说明压缩也没救回来，重试只会再烧一次钱
            if not is_context_overflow_error(exc) or next_state.get("context_overflow_recovery_attempted"):
                raise
            next_state["context_overflow_recovery_attempted"] = True
            next_state["runtime_context_stats"] = {
                **(next_state.get("runtime_context_stats") or {}),
                "reactive_compaction": "aggressive_retry",
                "overflow_error": str(exc),
            }
            self._append_agent_event(
                next_state,
                "context_overflow_recovery",
                {"phase": "model_decision", "strategy": "aggressive_runtime_projection"},
            )
            # aggressive=True：用更激进的预算重新裁剪上下文（丢更多历史/证据）
            user_prompt = self._tool_calling_user_prompt(next_state, aggressive=True)
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            estimated_prompt_tokens = self._record_prompt_diagnostics(
                next_state,
                phase="model_decision_reactive_retry",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            response = await tool_model.ainvoke(messages)
        # 记录 token 用量（对外可见的成本明细）
        self._record_model_usage(
            next_state,
            response,
            phase="model_decision",
            estimated_prompt_tokens=estimated_prompt_tokens,
        )
        response_text = self._message_content(response).strip()
        # tool_calls 取不到时用 [] 兜底：有些模型/网关会返回 None
        raw_tool_calls = getattr(response, "tool_calls", None) or []
        tool_calls = [dict(call) for call in raw_tool_calls if isinstance(call, dict)]
        stream_events: list[dict[str, Any]] = []
        # 「既有文本又要调工具」：这段文本是模型的推理说明，作为 thought 事件推给前端，
        # 让用户看到它打算干什么（也便于以后排查模型的决策路径）
        if response_text and tool_calls:
            self._append_agent_event(next_state, "thought", {"content": response_text})
            stream_events.append(self._event("thought", {"content": response_text}))
        if not tool_calls:
            # 没有工具调用 = 循环结束，这段文本就是最终答案
            next_state["final_answer"] = response_text
            next_state["pending_tool_calls"] = []
            self._stop_loop(next_state, "model_final_answer", "completed")
        else:
            # 防死循环守卫：模型反复请求「完全相同的工具+参数」时强制中止
            repeated = self._repeated_tool_call(tool_calls, next_state)
            if repeated:
                next_state["pending_tool_calls"] = []
                next_state["route"] = "tool_loop_guard"
                # 这里给的不是「答案」而是「解释为什么停下」，措辞要能让用户知道下一步该怎么做
                next_state["final_answer"] = (
                    "我停止了这轮 Agent Loop，因为模型连续请求相同工具和参数，"
                    "继续执行只会重复消耗预算。请根据已有工具结果调整问题，或让我换一种检索/分析路径。"
                )
                # 把这次「被拦下的调用」也记成一条 observation，前端能看到它试图重复调用什么
                next_state["observations"] = [
                    *(next_state.get("observations") or []),
                    {
                        "tool_name": repeated["tool_name"],
                        "arguments": repeated["arguments"],
                        "observation": {
                            "route": "tool_loop_guard",
                            "answer_preview": "Detected repeated identical tool call and stopped the loop.",
                            "repeat_count": repeated["repeat_count"],
                            "fingerprint": repeated["fingerprint"],
                        },
                    },
                ]
                # completion_status="blocked"：这是「被迫中止」，不是正常完成
                self._stop_loop(
                    next_state,
                    "repeated_tool_call",
                    "blocked",
                    {
                        "tool_name": repeated["tool_name"],
                        "arguments": repeated["arguments"],
                        "repeat_count": repeated["repeat_count"],
                    },
                )
                return self._graph_state_update(next_state, stream_events)
            next_state["pending_tool_calls"] = tool_calls
        return self._graph_state_update(next_state, stream_events)

    async def _graph_tool_action(self, state: ConversationState) -> dict[str, Any]:
        """「工具执行」节点：每次只消费 pending_tool_calls 里的**第一个**。

        一次只跑一个（而不是并发跑全部）：
        - 同一份 state 是共享的，并发写会互相覆盖；
        - 工具之间可能有隐含顺序（先读文件再搜内容），串行更符合预期。
        多个调用靠图的自环逐个消化。
        """
        next_state: ConversationState = dict(state)
        pending = list(next_state.get("pending_tool_calls") or [])
        if not pending:
            return self._graph_state_update(next_state, [])
        tool_call = pending[0]
        # 先出队再执行：即使本次执行抛错，也不会在重入时又拿到同一条
        next_state["pending_tool_calls"] = pending[1:]
        public_action, action_input, observation, _tool_message, _tool_use_event, tool_result_event = await self._execute_tool_call(tool_call, next_state)
        # 三种事件按「结果 → 动作 → 观察」的顺序推：工具已经执行完，
        # 所以 tool_result 先到，前端可以立即更新调用状态
        stream_events = [
            self._event("tool_result", tool_result_event),
            self._event("action", {"tool_name": public_action, "arguments": action_input}),
            self._event("observation", {"tool_name": public_action, "content": observation}),
        ]
        return self._graph_state_update(next_state, stream_events)

    async def _graph_max_steps_stop(self, state: ConversationState) -> dict[str, Any]:
        """步数耗尽的收尾节点：给一个「基于已有证据」的说明性回答。

        注意 completion_status="budget_exhausted" 与 "blocked" 不同：
        前者是「钱/步数用完了」，后者是「被判定为异常行为」。
        """
        next_state: ConversationState = dict(state)
        next_state["route"] = "tool_loop_budget_exhausted"
        # 这段话刻意强调「只能基于已完成的调用」：避免用户误以为结论是完整分析
        next_state["final_answer"] = (
            "这轮 Agent Loop 已达到步骤预算上限。下面的结论只能基于已经完成的工具调用和观察结果，"
            "如果还需要继续，我会从当前证据继续缩小范围。"
        )
        self._stop_loop(next_state, "max_steps_exhausted", "budget_exhausted", {"max_steps": self.max_steps})
        return self._graph_state_update(next_state, [])

    def _route_after_model_decision(self, state: ConversationState) -> str:
        """模型决策后的分支：有待执行工具就去跑工具，否则结束。"""
        return "tools" if state.get("pending_tool_calls") else "end"

    def _route_after_tool_action(self, state: ConversationState) -> str:
        """工具执行后的分支，三选一，顺序重要。

        1. 队列里还有待执行工具 → 继续跑（保证一次回复里的多个调用都被执行）；
        2. 已达步数上限且还没有答案 → 走 limit 节点收尾；
        3. 否则回模型，让它基于新观察继续思考或作答。
        """
        if state.get("pending_tool_calls"):
            return "tools"
        # 第二个条件带 "还没有答案"：如果模型最后一步既调工具又给了答案，
        # 此时答案已就绪，没必要再进 limit 节点覆盖掉它
        if int(state.get("step") or 0) >= self.max_steps and not str(state.get("final_answer") or "").strip():
            return "limit"
        return "model"

    def _graph_state_update(self, state: ConversationState, stream_events: list[dict[str, Any]]) -> dict[str, Any]:
        """把节点内部状态投影成「要回写给图的那部分」。

        白名单而不是 `return dict(state)`：context / tools 这些大对象不需要在节点间来回传，
        只传会变化的那批字段，既省内存也让图的状态变更更清晰。
        """
        keys = [
            "observations",
            "agent_events",
            "tool_calls",
            "citations",
            "pending_tool_calls",
            "route",
            "step",
            "final_answer",
            "stop_reason",
            "completion_status",
            "tool_call_counts",
            "evaluation_trace",
            "model_usage",
            "prompt_diagnostics",
            "runtime_context_stats",
            "context_overflow_recovery_attempted",
        ]
        update = {key: state[key] for key in keys if key in state}
        # stream_events 是「本节点新产生的事件」，不是累积的，所以直接覆盖
        update["stream_events"] = stream_events
        return update

    def _stop_loop(
        self,
        state: ConversationState,
        reason: str,
        completion_status: str,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """标记循环结束并留一条 trace。

        所有退出路径都必须经过这里（而不是直接 return），这样前端时间线的最后
        一定有一条 agent_loop_stop 说明「为什么停」——这是排查「回答怎么突然结束」的入口。
        """
        state["stop_reason"] = reason
        state["completion_status"] = completion_status
        return self._append_agent_event(
            state,
            "agent_loop_stop",
            {"reason": reason, "completion_status": completion_status, **(extra or {})},
        )

    def _repeated_tool_call(self, tool_calls: list[dict[str, Any]], state: ConversationState) -> dict[str, Any] | None:
        """检测重复工具调用，超限则返回拦截信息。

        指纹是「工具名 + 排序后的参数 JSON」，只认完全相同的调用；
        参数稍有不同就算新调用（模型换了检索范围是合理行为）。
        计数在 state 里累积，跨步生效——所以是「同一请求累计超过 2 次」才拦。
        """
        counts = dict(state.get("tool_call_counts") or {})
        for tool_call in tool_calls:
            safe_action, public_action, _tool_use_id, action_input = self._tool_call_parts(tool_call, state)
            fingerprint = self._tool_call_fingerprint(safe_action, action_input)
            repeat_count = counts.get(fingerprint, 0) + 1
            counts[fingerprint] = repeat_count
            if repeat_count > self.max_repeated_tool_calls:
                # 先写回计数再返回，保证拦截这件事本身也被记录
                state["tool_call_counts"] = counts
                return {
                    "tool_name": public_action,
                    "arguments": action_input,
                    "fingerprint": fingerprint,
                    "repeat_count": repeat_count,
                }
        state["tool_call_counts"] = counts
        return None

    def _tool_call_fingerprint(self, safe_action: str, action_input: dict[str, Any]) -> str:
        """生成调用指纹。

        sort_keys=True 让参数顺序不影响判等（模型每次给的键顺序可能不同）；
        default=str 兜住不可序列化的值（如 datetime），避免这里因类型报错中断循环。
        """
        return f"{safe_action}:{json.dumps(action_input, ensure_ascii=False, sort_keys=True, default=str)}"

    def _completion_summary(self, state: ConversationState) -> dict[str, Any]:
        """生成给前端/日志用的「本次运行摘要」。

        有一个隐含升级：状态本来是 completed，但有工具报错时改成
        completed_with_tool_errors——让调用方既能一眼看出「跑完了」，
        又不会误认为「一切顺利」。
        """
        tool_calls = [call for call in state.get("tool_calls") or [] if isinstance(call, dict)]
        error_calls = [call for call in tool_calls if call.get("status") == "error"]
        status = state.get("completion_status") or "completed"
        if status == "completed" and error_calls:
            status = "completed_with_tool_errors"
        return {
            "status": status,
            # reason 兜底成 model_final_answer：绝大多数正常结束都是模型直接作答
            "reason": state.get("stop_reason") or "model_final_answer",
            "steps": int(state.get("step") or 0),
            "tool_call_count": len(tool_calls),
            "tool_error_count": len(error_calls),
        }

    def _tool_completion_status(self, state: ConversationState) -> str:
        """按「有没有成功的工具调用」判定完成状态。

        有调用但全失败 = blocked（用户拿不到任何新信息）；
        没有调用 = completed（可能是纯对话，不需要工具）。
        """
        calls = [call for call in state.get("tool_calls") or [] if isinstance(call, dict)]
        if calls and not any(call.get("status") == "success" for call in calls):
            return "blocked"
        return "completed"

    def _preview_tool_use_event(self, raw_state: Any) -> dict[str, Any] | None:
        """从「即将执行的 state」里提取一个 tool_use 预览事件。

        用于工具真正开跑之前先告诉前端「我要用哪个工具、带什么参数」，
        这样长耗时的工具（如全库检索）不会让界面干等。
        取不到待执行调用时返回 None（调用方会跳过，不推空事件）。
        """
        if not isinstance(raw_state, dict):
            return None
        pending = raw_state.get("pending_tool_calls")
        if not isinstance(pending, list) or not pending or not isinstance(pending[0], dict):
            return None
        safe_action, public_action, tool_use_id, action_input = self._tool_call_parts(pending[0], raw_state)
        return {
            "type": "tool_use",
            "step": raw_state.get("step", 0),
            "id": tool_use_id,
            "tool_name": public_action,
            "arguments": action_input,
            # 技能相关元信息：前端据此标注「这次调用属于哪个技能」
            **self._skill_event_metadata(public_action, raw_state),
        }

    async def _execute_tool_call_events(
        self,
        tool_call: dict[str, Any],
        state: ConversationState,
    ) -> AsyncIterator[dict[str, Any]]:
        """执行一次工具调用，并把过程拆成四个事件依次产出。

        事件顺序固定：tool_use（开始）→ tool_result（结果）→ action（参数回放）
        → observation（给用户看的观察摘要）。
        与 _graph_tool_action 的区别只在于「推事件」还是「返回结果」：
        这条路径用于旁路（预取），那条用于图节点。
        """
        safe_action, public_action, tool_use_id, action_input, tool_use_event = self._prepare_tool_call(tool_call, state)
        yield self._event("tool_use", tool_use_event)
        observation, _tool_message, tool_result_event = await self._complete_tool_call(
            safe_action,
            public_action,
            tool_use_id,
            action_input,
            state,
        )
        yield self._event("tool_result", tool_result_event)
        yield self._event("action", {"tool_name": public_action, "arguments": action_input})
        yield self._event("observation", {"tool_name": public_action, "content": observation})

    async def _execute_tool_call(
        self,
        tool_call: dict[str, Any],
        state: ConversationState,
    ) -> tuple[str, dict[str, Any], dict[str, Any], ToolMessage, dict[str, Any], dict[str, Any]]:
        """执行一次工具调用并返回全部产物（供图节点使用，不推事件）。

        六个返回值依次是：对外工具名、实际入参、观察结果、给模型的 ToolMessage、
        tool_use 事件、tool_result 事件——调用方按需取用，避免再查一遍 state。
        """
        safe_action, public_action, tool_use_id, action_input, tool_use_event = self._prepare_tool_call(tool_call, state)
        observation, tool_message, tool_result_event = await self._complete_tool_call(
            safe_action,
            public_action,
            tool_use_id,
            action_input,
            state,
        )
        return public_action, action_input, observation, tool_message, tool_use_event, tool_result_event

    def _prepare_tool_call(
        self,
        tool_call: dict[str, Any],
        state: ConversationState,
    ) -> tuple[str, str, str, dict[str, Any], dict[str, Any]]:
        """解析调用 + 立刻记一条 tool_use 事件。

        注意这里就落 trace 了（不是等执行完）：如果工具执行过程中进程崩溃，
        trace 里至少能看到「它开始执行了」，这是排查卡死的关键信息。
        """
        safe_action, public_action, tool_use_id, action_input = self._tool_call_parts(tool_call, state)
        tool_use_event = self._append_agent_event(
            state,
            "tool_use",
            {"id": tool_use_id, "tool_name": public_action, "arguments": action_input, **self._skill_event_metadata(public_action, state)},
        )
        return safe_action, public_action, tool_use_id, action_input, tool_use_event

    def _tool_call_parts(
        self,
        tool_call: dict[str, Any],
        state: ConversationState | dict[str, Any],
    ) -> tuple[str, str, str, dict[str, Any]]:
        """从模型给出的 tool_call 里拆出四个要素。

        容错点：
        - id 缺失时用事件数生成一个（保证 tool_use_id 唯一，前端的调用卡片才能对齐）；
        - args 可能是 JSON 字符串（部分模型/网关的行为），解析失败就给空 dict，
          宁可让工具收到空参数（它会报参数缺失）也不要在这里崩。
        """
        safe_action = str(tool_call.get("name") or "").strip()
        public_action = self._public_tool_name(safe_action, state)  # type: ignore[arg-type]
        tool_use_id = str(tool_call.get("id") or f"tool_use_{len(state.get('agent_events') or []) + 1}")
        raw_args = tool_call.get("args") if isinstance(tool_call, dict) else {}
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args)
            except json.JSONDecodeError:
                raw_args = {}
        # 最后统一走 _normalize_action_input 补齐/纠正参数
        action_input = self._normalize_action_input(public_action, raw_args if isinstance(raw_args, dict) else {}, state)  # type: ignore[arg-type]
        return safe_action, public_action, tool_use_id, action_input

    async def _complete_tool_call(
        self,
        safe_action: str,
        public_action: str,
        tool_use_id: str,
        action_input: dict[str, Any],
        state: ConversationState,
    ) -> tuple[dict[str, Any], ToolMessage, dict[str, Any]]:
        """真正执行工具并回写状态，返回 (观察摘要, 给模型的 ToolMessage, 结果事件)。

        流程：前置校验（技能约束/未知工具）→ 带恢复的执行 → 补齐 tool_calls 结构
        → 写 trace 与累积状态 → 构造 ToolMessage。
        这是整条链中「唯一与外部世界交互」的地方，所有错误都在这里被转成结构化结果。
        """
        # 执行前有两道前置校验（都在真正调工具之前拦下，避免白跑一次外部调用）：
        # 1) 技能约束：某些技能要求「先做 A 才能做 B」，顺序错了直接拒；
        # 2) 未知工具：模型幻觉出一个不存在的工具名。
        tools = state.get("tools") or {}
        skill_violation = self._explicit_skill_tool_violation(public_action, state)
        if skill_violation:
            error = skill_violation
            error_kind: ToolErrorKind = "permission_denied"
            observation = {
                "route": "tool_error",
                "answer": error,
                "citations": [],
                "tool_calls": [
                    {
                        "tool_name": public_action,
                        "arguments": action_input,
                        "status": "error",
                        "error": error,
                        "error_kind": error_kind,
                        "retryable": False,
                        "attempts": 1,
                        "tool_use_id": tool_use_id,
                    }
                ],
                "error_kind": error_kind,
                "retryable": False,
            }
        elif safe_action not in tools:
            # 未知工具时把可用的工具名回给模型，它下一轮通常能自我纠正
            error = f"未知工具：{public_action}。可用工具：{', '.join(state.get('tool_name_map', {}).values())}"
            error_kind: ToolErrorKind = "unknown_tool"
            observation = {
                "route": "tool_error",
                "answer": error,
                "citations": [],
                "tool_calls": [
                    {
                        "tool_name": public_action,
                        "arguments": action_input,
                        "status": "error",
                        "error": error,
                        "error_kind": error_kind,
                        "retryable": False,
                        "attempts": 1,
                        "tool_use_id": tool_use_id,
                    }
                ],
                "error_kind": error_kind,
                "retryable": False,
            }
        else:
            observation = await self._invoke_tool_with_recovery(
                tools[safe_action],
                public_action,
                tool_use_id,
                action_input,
                state,
            )

        # 工具可能返回一个「子调用列表」（例如工作流内部又跑了多个 Agent）。
        # 若没返回，就按「一次成功的单调用」补一条，保证 tool_calls 结构始终完整可渲染
        if not observation.get("tool_calls"):
            observation["tool_calls"] = [
                {
                    "tool_name": public_action,
                    "arguments": action_input,
                    "status": "success",
                    "tool_use_id": tool_use_id,
                    **self._skill_event_metadata(public_action, state),
                }
            ]
        for call in observation.get("tool_calls", []):
            if isinstance(call, dict):
                # setdefault 而不是赋值：子调用自己填过的值优先，这里只补齐缺失字段
                call.setdefault("tool_use_id", tool_use_id)
                if observation.get("attempts") is not None:
                    call.setdefault("attempts", observation.get("attempts"))
                if observation.get("error_kind") is not None:
                    call.setdefault("error_kind", observation.get("error_kind"))
                if observation.get("retryable") is not None:
                    call.setdefault("retryable", observation.get("retryable"))
                # 技能元信息必须覆盖（update 而非 setdefault）：它由当前激活技能决定，
                # 而不是子调用自己带的值
                call.update(self._skill_event_metadata(str(call.get("tool_name") or public_action), state))
        # 评测模式下把这次观察记进评测 trace（生产模式是空操作）
        self._capture_evaluation_observation(
            state,
            tool_name=public_action,
            arguments=action_input,
            observation=observation,
        )
        summary = self._summarize_observation(observation)
        # 只要有一条子调用报错，整体就标 error：不能让「部分失败」看起来像成功
        status = "error" if any(call.get("status") == "error" for call in observation.get("tool_calls", []) if isinstance(call, dict)) else "success"
        tool_result_event = self._append_agent_event(
            state,
            "tool_result",
            {
                "tool_use_id": tool_use_id,
                "tool_name": public_action,
                "status": status,
                "route": observation.get("route"),
                "error_kind": observation.get("error_kind"),
                "retryable": observation.get("retryable"),
                "content": summary,
                **self._skill_event_metadata(public_action, state),
            },
        )
        # 把本次结果并入累积状态：observations（摘要）、tool_calls（明细）、citations（来源）
        state["observations"] = [
            *(state.get("observations") or []),
            {"tool_name": public_action, "arguments": action_input, "observation": summary},
        ]
        state["tool_calls"] = [*(state.get("tool_calls") or []), *observation.get("tool_calls", [])]
        state["citations"] = [*(state.get("citations") or []), *observation.get("citations", [])]
        # 工具自己产生的事件（如工作流内部的 trace）也并进 agent_events，前端才能看到嵌套过程
        extra_events = observation.get("agent_events")
        if isinstance(extra_events, list):
            state["agent_events"] = [*(state.get("agent_events") or []), *[event for event in extra_events if isinstance(event, dict)]]
        # route 更新规则：工具给了路由就采用，但有一个例外——如果这是 tool_error
        # 且之前已经有成功的工具调用，就保留原来的 route。
        # 理由是「已经有了部分收获」，不该因为最后一次失败把整体路由改写成错误态
        observation_route = str(observation.get("route") or "")
        has_successful_tool = any(call.get("status") == "success" for call in state.get("tool_calls") or [])
        if observation_route and (observation_route != "tool_error" or not has_successful_tool):
            state["route"] = observation_route

        # 回给模型一条 ToolMessage：tool_call_id 必须与请求时的 id 一致，
        # 否则模型的工具调用协议会断掉（下一轮会报「找不到对应调用」）
        tool_message = ToolMessage(
            content=json.dumps(summary, ensure_ascii=False),
            tool_call_id=tool_use_id,
        )
        return summary, tool_message, tool_result_event

    async def _invoke_tool_with_recovery(
        self,
        tool: StructuredTool,
        public_action: str,
        tool_use_id: str,
        action_input: dict[str, Any],
        state: ConversationState,
    ) -> dict[str, Any]:
        """带重试的工具执行，保证「失败也返回结果」而不是抛异常。

        只对瞬时错误重试（见 _is_retriable_tool_error），且最多 1 次。
        最终失败时返回结构化的错误观察，让模型能看到错误并改变策略——
        这比抛异常中断整轮对话有用得多。
        """
        # 总尝试次数 = 1 次初始 + tool_retry_attempts 次重试（默认 1，即最多 2 次）
        max_attempts = 1 + self.tool_retry_attempts
        last_error: Exception | None = None
        # 初始值给「运行时错误 / 不可重试」：即使循环体一次都没进（理论上不会），
        # 也不会把未定义变量带进最终结果
        last_error_kind: ToolErrorKind = "tool_runtime_error"
        retryable = False
        for attempt in range(1, max_attempts + 1):
            try:
                observation = await tool.ainvoke(action_input)
                if isinstance(observation, dict):
                    # attempts 记录「这是第几次才算成功」，排查瞬时故障时很有用
                    observation.setdefault("attempts", attempt)
                    return observation
                # 工具返回非 dict（比如直接返回字符串）时，包装成标准结构，
                # 保证下游 _complete_tool_call 拿到的一直是 dict
                return {
                    "route": public_action,
                    "answer": str(observation),
                    "citations": [],
                    "tool_calls": [],
                    "attempts": attempt,
                }
            except Exception as exc:
                last_error = exc
                last_error_kind = self._classify_tool_error(exc)
                retryable = self._is_retriable_tool_error(last_error_kind, public_action)
                # 不可重试，或已经是最后一次 → 直接跳出，不再浪费一次调用
                if not retryable or attempt >= max_attempts:
                    break
                # 只有真要重试时才记事件：这解释了「为什么同一个工具出现了两次」
                self._append_agent_event(
                    state,
                    "tool_retry",
                    {
                        "tool_use_id": tool_use_id,
                        "tool_name": public_action,
                        "attempt": attempt,
                        "next_attempt": attempt + 1,
                        "error_kind": last_error_kind,
                        "error": str(exc),
                    },
                )
        # 走到这里说明最终失败，返回结构化错误而不是抛异常：
        # 让模型能「看到」失败原因并改换策略，而不是整轮对话崩掉
        error_text = str(last_error or "unknown tool failure")
        return {
            "route": "tool_error",
            "answer": f"工具 {public_action} 执行失败：{error_text}",
            "citations": [],
            "tool_calls": [
                {
                    "tool_name": public_action,
                    "arguments": action_input,
                    "status": "error",
                    "error": error_text,
                    "error_kind": last_error_kind,
                    "retryable": retryable,
                    "attempts": max_attempts if retryable else 1,
                    "tool_use_id": tool_use_id,
                }
            ],
            "error_kind": last_error_kind,
            "retryable": retryable,
            "attempts": max_attempts if retryable else 1,
        }

    def _classify_tool_error(self, exc: Exception) -> ToolErrorKind:
        """把异常文本归类成固定的错误类型。

        这里只能靠字符串匹配（工具/httpx 的异常类型不统一，没法用 isinstance 全覆盖）。
        规则顺序从「最确定」到「最模糊」：先看异常类型，再按关键词；都不匹配归为
        运行时错误——反正它不可重试，归类不精确也不会导致错误重试。
        """
        if isinstance(exc, TimeoutError):
            return "timeout"
        text = str(exc).lower()
        if any(token in text for token in ["timeout", "timed out", "deadline"]):
            return "timeout"
        if any(token in text for token in ["rate limit", "429", "too many requests"]):
            return "rate_limited"
        if any(token in text for token in ["connection", "temporarily", "unavailable", "reset by peer", "503", "502", "504"]):
            return "transient_network"
        if any(token in text for token in ["permission", "forbidden", "unauthorized", "401", "403"]):
            return "permission_denied"
        if any(token in text for token in ["not found", "missing", "404"]):
            return "data_not_found"
        if any(token in text for token in ["validation", "schema", "json"]):
            return "model_output_invalid"
        return "tool_runtime_error"

    def _is_retriable_tool_error(self, error_kind: ToolErrorKind, public_action: str) -> bool:
        """判断某个错误是否值得重试。

        两个条件同时满足才重试：
        1) 错误类型是瞬时的（超时/限流/网络抖动）——参数错、没权限重试也没用；
        2) 工具**没有副作用**。安全分类、周报生成、工作流审查这三个会被排除：
           它们内部可能触发写操作或昂贵的多 Agent 调用，盲目重试可能造成重复提交/重复花钱。
        """
        if error_kind not in {"timeout", "rate_limited", "transient_network"}:
            return False
        # 这份排除名单是「有副作用或高成本」的工具白名单（反向），新增此类工具时要同步加进来
        if public_action in {"safety_agent.classify_action_risk", "report_agent.generate_weekly_report", "workflow.run_engineering_review"}:
            return False
        return True

    def _tool_model(self, tools: dict[str, StructuredTool]):
        """把工具绑定到模型上（原生 function calling）。

        bind_tools 让模型在回复里直接产出 tool_calls 结构，比「让模型输出 JSON 再解析」可靠得多。
        没有模型时抛错是合理的：这条路径本来就只在「有模型」时才会走到
        （run_stream 里 model is None 的分支已经提前返回）。
        """
        if self.model is None:
            raise RuntimeError("ChatAgent requires an LLM for open-ended tool selection")
        return self.model.bind_tools(list(tools.values()))

    def _tool_calling_system_prompt(self) -> str:
        """系统提示词：稳定前缀，会被 prompt cache 复用。

        编写原则（改的时候请保持）：
        - 这里只放「永远成立」的规则（别编造工具结果、结论要讲 WHY、分配只能从
          team_members 里选、写操作先过安全工具），不放具体的仓库/上下文数据；
        - 具体数据全部走 user prompt，这样这段前缀在多轮、多次请求间完全一致，
          缓存命中率才高。一旦在这里拼时间戳或仓库名，缓存就废了。
        """
        return (
            "你是 DevFlow AI 的对话控制器。这个稳定前缀会被缓存。"
            "请像长任务软件 Agent 一样工作：先规划，调用工具，读取 tool_result 观察结果，修正计划，"
            "持续推进直到用户请求真正完成，或步骤预算耗尽。"
            "当需要仓库、记忆、工作区代码或写操作安全上下文时，使用原生工具调用。"
            "绝不要编造工具结果。"
            "工程结论必须说明 WHY，并区分证据、推断和不确定项；证据不足时直接说缺什么，不要硬猜。"
            "当输出审查结论时，不要用 looks good、没什么问题这类空话代替检查；必须给出风险、级别和下一步。"
            "当回答交接、总结或工作流结论时，尽量保留 What、Why、Tradeoff、Open Questions、Next Action。"
            "当用户询问负责人、已分配人员、责任人或任务分配时，只能从明确的 DevFlow team_members "
            "或工具返回的现有 GitHub assignees 中推荐人员。仓库文档、Issue 文本、代码、示例、模型名、"
            "角色设定、团队比喻和记忆片段都不能当成人，除非它们同时出现在 DevFlow team_members 中。"
            "如果没有配置团队成员，请说明这一点，并要求用户先添加开发者后再分配任务。"
            "当用户请求 GitHub 写入、关闭、打标签、评论、发布、回复或发送时，必须先调用安全工具；"
            "除非存在单独的已确认写入工具，否则只返回草稿。"
            "对于合并就绪度、当前阻塞、优先级规划，或需要 Issue + PR + CI + RAG 综合判断的跨领域工程决策，"
            "优先使用 workflow.run_engineering_review 工具。"
        )

    def _tool_calling_user_prompt(self, state: ConversationState, *, aggressive: bool = False) -> str:
        """用户提示词（可变部分）：把「本次运行的所有数据」拼进去。

        aggressive=True 只在上下文溢出后的重试里用：把服务端上下文和历史
        再压到预算的 1/3、1/2，牺牲信息量换「这次能塞进窗口」。

        这段 prompt 的体积直接决定每次模型调用的成本，所以历史与观察结果
        都经过 _runtime_tool_context 的预算投影，而不是原样拼接。
        """
        prompt_observations, prompt_events, runtime_stats = self._runtime_tool_context(
            state,
            aggressive=aggressive,
        )
        # 投影统计写回 state：最终会进 trace，用来解释「为什么模型看到的信息变少了」
        state["runtime_context_stats"] = {
            **(state.get("runtime_context_stats") or {}),
            **runtime_stats,
        }
        context = state.get("context") or {}
        server_context = self._server_context(context)
        history = self._history_messages(context)
        if aggressive:
            # 两个 max(512, ...) 是下限保护：预算算出来再小也要留够最低信息量，
            # 否则压到几十 token 等于什么都没给，模型只会胡猜
            budget = default_context_budget()
            server_context = clip_to_token_budget(server_context, max(512, budget.available_prompt_tokens // 3))
            history = self._compress_history_for_retry(history, max(512, budget.recent_tokens // 2))
        return (
            "上下文使用规则：如果服务端提供的最近消息已经包含回答所需的完整原文，直接基于这些消息回答，"
            "不要为了展示工具调用而重复读取记忆。只有目标消息不在最近窗口、内容被截断或压缩、用户明确要求"
            "核实原话，或者问题跨越会话时，才调用一个最匹配的 MCP 记忆工具。当前会话原始消息使用 "
            "devflow_get_thread_context；封存会话和跨会话摘要使用 devflow_read_session_events；项目长期记忆、"
            "历史决策和仓库证据使用 devflow_search_evidence。询问当前会话原话时，不要额外调用 "
            "devflow_search_evidence。\n\n"
            "硬性规则：如果用户询问这个项目/仓库做什么、技术栈、架构、启动方式、依赖或源码行为，"
            "回答前必须检查工作区文件。优先读取 README、package 或 manifest 文件；如果工作区文件不可用，要明确说明。\n\n"
            "涉及源码的问题，优先使用 workspace.search_code、workspace.read_file 或 workspace.list_files。"
            "涉及已索引的仓库历史和上传知识，使用 rag.search_similar_documents 或 devflow_search_evidence。\n\n"
            "混合 RAG 约定：服务端上下文中的 [Related Evidence] 是根据原始问题完成的调用前检索。"
            "如果这些证据不足、互相冲突，或问题包含多个子问题，请改写为更具体的 query 调用 "
            "rag.search_similar_documents；可以用不同 query 连续补查，并在最终回答前比较多次 tool_result。\n\n"
            "工作流硬性规则：如果用户请求多 Agent 审查、合并就绪度、当前阻塞、工程状态、跨 Agent 综合，"
            "或依赖 Issue、PR、CI 与仓库证据共同判断的决策，必须先调用 workflow.run_engineering_review，"
            "再给出最终建议。\n\n"
            "分配硬性规则：如果用户询问谁应该负责、处理、接手或被分配某个 Issue/PR/任务，"
            "可选人员只能来自下方列出的 DevFlow team_members，以及工具明确返回的现有 GitHub assignees。"
            "不要把仓库内容、项目角色设定、模型名、示例、作者或虚构标签当成团队成员。"
            "如果 team_members 为空且没有现有 assignee，请回答当前还没有添加项目成员，"
            "并要求用户在团队面板中添加开发者。可以说明可能需要的能力方向，但不要点名具体人员。\n\n"
            "循环约定：每次工具调用都会先记录为 tool_use，再记录为 tool_result。决定下一步前必须阅读之前的 tool_result 内容。"
            "如果目录列表里出现值得继续查看的 README、manifest、代码文件、Issue、PR 或 CI 记录，不要停在目录列表。"
            "除非新证据使重复调用有必要，不要用相同参数重复调用同一个工具。\n\n"
            f"当前 Agent 配置的运行模型：{settings.llm_model}\n\n"
            f"可用原生工具：\n{self._tool_help(state)}\n\n"
            f"已注册的 DevFlow 技能：\n{self._registered_skills_context(state.get('context') or {})}\n\n"
            f"本轮按需加载的 Skill 完整指令：\n{self._active_skill_prompt(state.get('context') or {})}\n\n"
            f"DevFlow 团队上下文：\n{self._team_assignment_context(state.get('context') or {})}\n\n"
            # 下面这一段是「数据区」，顺序固定：先规则（工具/技能），
            # 再静态上下文（服务端/团队），最后是动态的（历史消息、观察结果、工具索引、用户消息）。
            # 用户消息放最末，是为了让模型在长上下文里也能一眼看到当前的直接指令
            f"服务端上下文：\n{server_context}\n\n"
            f"上下文覆盖度：\n{json.dumps(self._history_coverage(context), ensure_ascii=False)}\n\n"
            f"最近消息：\n{json.dumps(history, ensure_ascii=False)}\n\n"
            f"已有观察结果（已按总预算投影，完整结果保存在会话转录）：\n{json.dumps(prompt_observations, ensure_ascii=False)}\n\n"
            f"工具调用/结果索引：\n{json.dumps(prompt_events, ensure_ascii=False)}\n\n"
            f"用户消息：{state.get('message') or ''}"
        )

    def _normalize_action_input(self, action: str, action_input: dict[str, Any], state: ConversationState) -> dict[str, Any]:
        """按工具名补齐/覆盖参数，是「模型参数 → 工具真实入参」的适配层。

        为什么要这层：模型经常漏传关键参数（比如只给 query 不给 message），
        或者传了不该由它决定的参数（repo_id 必须用服务端上下文里的真实值，
        不能让模型随便填一个——那会查到别的仓库去）。

        三条规则：
        1. 大部分工具都要 message（原始用户话语作为兜底输入）；
        2. 检索类工具需要 query，缺了就用用户原话；
        3. devflow_* 系列一律强制使用服务端注入的 repo_id / conversation_id，
           并删掉模型可能瞎填的 repo_full_name。
        """
        normalized = dict(action_input)
        message = str(state.get("message") or "")
        context = state.get("context") or {}

        # 这组工具的共同点：都接受 message 作为「用户诉求」的原始表述。
        # 用 setdefault：模型自己给了就不覆盖
        if action in {
            "repo_health.generate_summary",
            "rag.search_similar_documents",
            "issue_agent.analyze_issue",
            "pr_review_agent.analyze_pr",
            "ci_debug_agent.analyze_workflow_run",
            "report_agent.generate_weekly_report",
            "safety_agent.classify_action_risk",
            "workflow.run_engineering_review",
            "workspace.search_code",
        }:
            normalized.setdefault("message", message)
        if action == "workspace.search_code":
            normalized.setdefault("query", message)
        if action == "rag.search_similar_documents":
            # 优先用 message（可能已被上面补过），没有才退回用户原话
            normalized.setdefault("query", str(normalized.get("message") or message))
        # 「项目是做什么的」而模型又没指定路径时，默认读 README——这是最快的概览来源
        if action == "workspace.read_file" and not normalized.get("path") and self._asks_project_overview(message):
            normalized["path"] = "README.md"
        if action == "devflow_search_evidence":
            normalized.setdefault("query", message)
        # 记忆类工具的默认条数：当前会话读 20 条（够上下文）、封存会话摘要读 5 条（摘要本身就短）
        if action == "devflow_get_thread_context":
            normalized.setdefault("limit", 20)
        if action == "devflow_read_session_events":
            normalized.setdefault("limit", 5)
        # 仓库/会话作用域强制覆盖（不是 setdefault）：这是权限与数据正确性的边界，
        # 模型无权决定访问哪个仓库
        if action.startswith("devflow_") and context.get("repo_id"):
            normalized["repo_id"] = context["repo_id"]
            normalized.pop("repo_full_name", None)
        if action.startswith("devflow_") and context.get("conversation_id"):
            normalized["conversation_id"] = context["conversation_id"]
        return normalized

    def _asks_project_overview(self, message: str) -> bool:
        """判断是不是在问「这个项目是做什么的/技术栈/架构」。

        先排除：只要提到 issue/pr/ci 等具体对象，就不算概览问题——
        那种问题需要看具体数据，走预读 README 的旁路会答偏。
        词表里同时有中文和转义形式（\\u505a\\u4ec0\\u4e48 即「做什么」），
        说明这份词表在某个时点被工具自动处理过，改的时候注意别把转义写坏。
        """
        lowered = message.lower()
        if any(token in lowered for token in ["issue", "pull request", "pr #", "ci", "未关闭", "待处理"]):
            return False
        return any(
            token in lowered
            for token in [
                "what is this project",
                "what does this project",
                "project about",
                "repository about",
                "tech stack",
                "architecture",
                "setup",
                "项目介绍",
                "项目概览",
                "这个项目",
                "该项目",
                "这个仓库",
                "该仓库",
                "\u505a\u4ec0\u4e48",
                "\u67b6\u6784",
                "\u6280\u672f\u6808",
            ]
        )

    def _required_memory_tool(self, message: str, context: dict[str, Any] | None = None) -> str | None:
        """便捷方法：只取记忆策略里的「必须调用的工具名」，没有就返回 None。"""
        policy = self._memory_context_policy(message, context or {})
        return str((policy or {}).get("tool") or "") or None

    def _memory_context_policy(self, message: str, context: dict[str, Any]) -> dict[str, Any] | None:
        """决定「这个问题要不要翻记忆、翻到哪一层」。

        返回 None 表示没有记忆需求（走正常工具循环）；否则返回
        {mode, tool, reason}，mode 有两种：
        - required_tool：必须先调某个记忆工具（回原始对话、跨会话、要原话核实）；
        - direct_recent_history：最近的历史消息已足够，直接回答，省掉整轮模型循环。

        判定是有序的，从「最明确」到「最模糊」：
        用户直接点名工具 → 提到历史/记忆 → 提到跨会话 → 问刚说过的话 → 问之前说过什么。
        """
        lowered = message.lower()
        # 第一优先：用户在问题里直接写了工具名（调试/高级用法），照办
        for tool_name in [
            "devflow_search_evidence",
            "devflow_get_thread_context",
            "devflow_read_session_events",
        ]:
            if tool_name in lowered:
                return {"mode": "required_tool", "tool": tool_name, "reason": "explicit_tool_request"}
        # 第二优先：明确在问历史决策/项目记忆
        if any(token in lowered for token in ["历史证据", "历史决策", "项目记忆", "historical evidence", "past decision"]):
            return {"mode": "required_tool", "tool": "devflow_search_evidence", "reason": "project_history_request"}
        # 第三优先：跨会话（当前会话的上下文里没有别的会话内容，必须专门去读）
        if any(
            token in lowered
            for token in [
                "封存会话",
                "会话摘要",
                "上次会话",
                "上个会话",
                "跨会话",
                "session digest",
                "session events",
                "previous session",
                "previous conversation",
            ]
        ):
            return {"mode": "required_tool", "tool": "devflow_read_session_events", "reason": "cross_session_request"}

        # 「刚刚说过」与「之前说过」分开标记：前者更可能被最近窗口完整覆盖，可以走短路
        immediate_recall = any(
            token in lowered
            for token in [
                "刚刚说",
                "刚才说",
                "上一条消息",
                "上一句话",
                "上一句",
                "what did i just say",
                "my last message",
            ]
        )
        broad_recall = immediate_recall or any(
            token in lowered
            for token in [
                "之前说过",
                "上次讨论",
                "聊天历史",
                "会话历史",
                "你记得",
                "前面说过",
                "what did we say",
                "earlier in this chat",
            ]
        )
        if not broad_recall:
            return None

        # 要「原话/逐字/审计」级别时，绝不能走短路：最近窗口里的消息可能已被压缩过，
        # 必须回原始 transcript 取，否则会出现「引用不准」这种最难发现的问题
        requires_exact_verification = any(
            token in lowered
            for token in ["原话", "逐字", "准确核实", "精确核实", "审计", "verbatim", "exact quote", "verify the record"]
        )
        if requires_exact_verification:
            return {"mode": "required_tool", "tool": "devflow_get_thread_context", "reason": "exact_verification_requested"}

        # 信息足够就短路（省一次模型循环）；不够才回原始会话读
        if self._recent_history_can_answer(message, context, immediate=immediate_recall):
            return {
                "mode": "direct_recent_history",
                "tool": None,
                "reason": "exact_recent_messages_available",
            }
        return {"mode": "required_tool", "tool": "devflow_get_thread_context", "reason": "recent_history_incomplete"}

    def _recent_history_can_answer(self, message: str, context: dict[str, Any], *, immediate: bool) -> bool:
        """判断「最近窗口里的原文」是否足以回答这个问题。

        关键在于「完整性」而不是「有多少条消息」：
        - 没有压缩统计（coverage 为空）→ 说明历史是原始未压缩的，可以相信；
        - immediate（问「刚刚说了什么」）→ 要求「末尾连续未压缩的消息数」覆盖到
          上一条用户消息的位置，否则中间被压缩过，答出来就是错的；
        - 其他回忆类问题 → 要求整段历史都完整未压缩（complete_and_exact）。
        这个判断偏保守：宁可多调一次记忆工具，也不要基于被压缩的残缺上下文编答案。
        """
        history = self._history_messages(context)
        prior_user_index = self._last_prior_user_index(history, message)
        # 找不到「上一条用户消息」就没有可回忆的对象，直接否掉
        if prior_user_index is None:
            return False
        coverage = self._history_coverage(context)
        if not coverage:
            return True
        if immediate:
            exact_suffix = int(coverage.get("exact_suffix_messages") or 0)
            # 距离末尾越远，需要的未压缩消息越多；exact_suffix 不够就说明中间断了
            distance_from_end = len(history) - prior_user_index
            return exact_suffix >= distance_from_end
        return bool(coverage.get("complete_and_exact"))

    def _last_prior_user_index(self, history: list[dict[str, str]], message: str) -> int | None:
        """找出「上一条用户消息」在历史里的下标。

        注意要排除本轮自己的提问：调用方通常已经把当前消息追加进历史，
        它会是最后一个 user 条目，不排除就会「回忆到自己」。
        """
        normalized_current = self._normalize_history_text(message)
        indexes = [index for index, item in enumerate(history) if item.get("role") == "user"]
        # 只有当最后一条 user 消息正好等于当前问题时才剔除，避免误删用户真的重复提问过的那条
        if indexes and self._normalize_history_text(history[indexes[-1]].get("content") or "") == normalized_current:
            indexes.pop()
        return indexes[-1] if indexes else None

    def _normalize_history_text(self, value: str) -> str:
        """历史文本归一化：仅用于「是不是同一条消息」的比较，不用于展示。"""
        return re.sub(r"\s+", " ", str(value or "").strip().lower())

    async def _direct_answer_from_recent_history(self, state: ConversationState) -> str:
        """短路路径：直接用最近历史回答（一次模型调用，不走工具循环）。

        这是最便宜的「回答回忆类问题」的方式：把历史原文和问题一起给模型，
        明确要求「不要声称调用了工具」，避免它编造工具调用记录。
        """
        if self.model is None:
            return self._local_recent_history_answer(state)
        history = self._history_messages(state.get("context") or {})
        try:
            # 提示里特别强调「最后一条与当前问题相同的 user 消息是本轮问题」：
            # 否则模型很容易把当前提问本身当成「用户想回忆的上一条内容」
            system_prompt = (
                "请直接根据提供的近期原始消息回答，不要声称调用了任何工具。"
                "最后一条与当前问题相同的 user 消息是本轮问题，不要把它当成用户要回忆的上一条内容。"
                "只回答上下文能证明的内容；如果用户问上一条消息，优先简洁复述上一条 user 消息。"
            )
            user_prompt = (
                f"近期原始消息：{json.dumps(history, ensure_ascii=False)}\n"
                f"当前用户问题：{state.get('message') or ''}"
            )
            estimated = self._record_prompt_diagnostics(
                state,
                phase="direct_recent_history",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            response = await self.model.ainvoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            )
            self._record_model_usage(
                state,
                response,
                phase="direct_recent_history",
                estimated_prompt_tokens=estimated,
            )
            # 模型返回空时退回本地模板：这条路径承诺「一定给出回答」，不能返回空串
            return self._message_content(response).strip() or self._local_recent_history_answer(state)
        except Exception:
            # 模型调用失败（网络/限流）也不该让这个旁路失败，直接用本地文本兜底
            return self._local_recent_history_answer(state)

    def _local_recent_history_answer(self, state: ConversationState) -> str:
        """零成本兜底：直接从历史里取出上一条用户消息原样复述。

        只在「问上一条说了什么」这类问题上正确；其他回忆类问题不适用，
        但在模型不可用/报错时，这是唯一还能给出确定答案的方式。
        """
        history = self._history_messages(state.get("context") or {})
        prior_user_index = self._last_prior_user_index(history, str(state.get("message") or ""))
        if prior_user_index is None:
            return "当前上下文里没有可确认的上一条用户消息。"
        content = history[prior_user_index].get("content") or ""
        return f"你刚刚说的是：{content}"

    def _history_coverage(self, context: dict[str, Any]) -> dict[str, Any]:
        """取上下文压缩器给出的「历史覆盖度」统计。

        多层 isinstance 判空是必需的：这些字段来自上游 JSON，结构不保证，
        而这里只是个查询辅助函数，不该因为脏数据抛异常。
        """
        compression_stats = context.get("compression_stats") or {}
        if not isinstance(compression_stats, dict):
            return {}
        coverage = compression_stats.get("history_coverage") or {}
        return coverage if isinstance(coverage, dict) else {}

    async def _prefetch_memory_context(self, state: ConversationState, public_name: str) -> AsyncIterator[dict[str, Any]]:
        """旁路 2 的落地实现：只调指定的那一个记忆工具。

        需要把「对外名」翻回「安全名」才能取到工具（前端/模型看到的是对外名）。
        参数按工具类型手工给定，不走模型——这正是这条旁路「确定、省钱」的原因。
        """
        tools = state.get("tools") or {}
        tool_names = state.get("tool_name_map") or {}
        # 反查：tool_name_map 是 safe→public，这里要 public→safe
        safe_name = next((safe for safe, public in tool_names.items() if public == public_name), None)
        if not safe_name or safe_name not in tools:
            return
        message = str(state.get("message") or "")
        args: dict[str, Any]
        if public_name == "devflow_search_evidence":
            # 用用户原话直接搜，取 5 条：旁路追求快，不追求全
            args = {"query": message, "limit": 5}
        elif public_name == "devflow_get_thread_context":
            # keyword 传空串 = 不过滤，取最近 20 条，对应「回看当前会话」
            args = {"keyword": "", "limit": 20}
        else:
            # 封存会话摘要默认 5 条（摘要本身信息密度高）
            args = {"limit": 5}
        # 伪造一个 tool_call 复用正常的执行链路，这样 trace、重试、错误处理全部一致
        async for event in self._execute_tool_call_events(
            {"name": safe_name, "args": args, "id": f"required_{public_name}_{len(state.get('agent_events') or [])}"},
            state,
        ):
            yield event

    async def _prefetch_project_context(self, state: ConversationState) -> AsyncIterator[dict[str, Any]]:
        """旁路 3 的落地实现：列目录 → 按存在的文件读 README/manifest。

        读哪些文件不是写死盲读，而是先看 list_files 的输出里到底有哪些，
        避免对不存在的 package.json 发起 README 式的无效读取（每次无效读取都是浪费）。
        """
        tools = state.get("tools") or {}
        tool_names = state.get("tool_name_map") or {}
        safe_by_public = {public: safe for safe, public in tool_names.items()}

        async def call(public_name: str, args: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
            """内部小工具：按对外名调一次工具，不存在就静默跳过。"""
            safe_name = safe_by_public.get(public_name)
            if not safe_name or safe_name not in tools:
                return
            async for event in self._execute_tool_call_events(
                {"name": safe_name, "args": args, "id": f"prefetch_{public_name}_{len(state.get('agent_events') or [])}"},
                state,
            ):
                yield event

        # 120 是经验上限：一般项目根目录不会超过这个数量，全列出来会浪费上下文
        async for event in call("workspace.list_files", {"path": ".", "limit": 120}):
            yield event
        # 从刚才 list_files 的结果文本里「捞」出实际存在的路径。
        # 用字符串切分而不是解析结构化结果，是因为不同工具的返回格式不统一，
        # 这里只需要知道「文件名是否出现过」，粗糙匹配够用
        observed_paths: set[str] = set()
        for item in state.get("observations") or []:
            if not isinstance(item, dict) or not isinstance(item.get("observation"), dict):
                continue
            preview = str(item["observation"].get("answer_preview") or "")
            observed_paths.update(token.strip("'\"`[](),").lower() for token in preview.split())
        # 顺序即优先级：README 优先，其次依赖清单与启动说明；220 行足够覆盖概览所需内容
        for path in ["README.md", "README.zh-CN.md", "package.json", "SETUP.md"]:
            if path.lower() in observed_paths:
                async for event in call("workspace.read_file", {"path": path, "line_count": 220}):
                    yield event

    def _build_langchain_tools(self, raw_tools: dict[str, ConversationTool], context: dict[str, Any]) -> dict[str, StructuredTool]:
        """把内部工具函数包装成 LangChain StructuredTool。

        两个关键点：
        - 工具名要「安全化」（_safe_tool_name）：OpenAI function name 只允许
          [a-zA-Z0-9_-]，工具名里的点（如 workspace.read_file）会被替换成双下划线；
        - kwargs 里丢掉 None 值：模型常把没填的可选参数传成 null，
          原样透传会让工具把 None 当成「显式指定为空」。
        """
        wrapped: dict[str, StructuredTool] = {}
        for name, func in raw_tools.items():
            async def call_tool(_func: ConversationTool = func, **kwargs: Any) -> dict[str, Any]:
                """LangChain 工具入口：丢掉 None 参数后调用原始工具函数。"""
                # 用默认参数绑住 func：否则闭包会捕获循环变量，所有工具都指向最后一个
                input_data = {key: value for key, value in kwargs.items() if value is not None}
                return await _func(input_data, context)

            safe_name = self._safe_tool_name(name)
            # __name__ 要改成安全名，LangChain 会用它作为工具名
            call_tool.__name__ = safe_name
            wrapped[safe_name] = StructuredTool.from_function(
                coroutine=call_tool,
                name=safe_name,
                description=self._tool_description(name, context),
                # 统一用万能 schema；infer_schema=False 表示不要从函数签名推断，
                # 因为签名是 **kwargs，推断出来会是空 schema
                args_schema=GenericToolArgs,
                infer_schema=False,
            )
        return wrapped

    def _tool_name_map(self, raw_tools: dict[str, ConversationTool]) -> dict[str, str]:
        """安全名 → 对外名的映射，前端展示与 trace 都用对外名。"""
        return {self._safe_tool_name(name): name for name in raw_tools}

    def _safe_tool_name(self, name: str) -> str:
        """把工具名转成符合 function-calling 命名规范的形式（非法字符变双下划线）。"""
        return re.sub(r"[^a-zA-Z0-9_-]", "__", name)

    def _public_tool_name(self, safe_name: str, state: ConversationState) -> str:
        """安全名反查对外名；查不到就用原名（防御未知工具）。"""
        return (state.get("tool_name_map") or {}).get(safe_name, safe_name)

    def _tool_description(self, name: str, context: dict[str, Any] | None = None) -> str:
        """生成工具描述。

        这些描述会进 prompt（工具 schema 的一部分），所以措辞会直接影响模型选哪个工具：
        尤其是 rag 与 workspace 的区别（历史/知识 vs 当前源码）、
        workflow 的适用场景（跨领域决策），写得含糊就会导致模型乱选工具、多花一次调用。
        """
        descriptions = {
            "devflow_search_evidence": "MCP 记忆工具。搜索持久记忆、会话摘要、仓库证据、Issue、PR、CI 和 Agent 运行记录。",
            "devflow_get_thread_context": "MCP 记忆工具。跨压缩边界读取当前仓库会话的完整原始聊天转录，支持消息 UUID 和字符偏移分页。",
            "devflow_read_session_events": "MCP 记忆工具。读取已封存的会话摘要、原始消息和工具事件。",
            "workspace.list_files": "列出所选仓库本地 checkout 下的文件。只读，用于探索源码树。",
            "workspace.read_file": "从所选仓库 checkout 中读取文本文件，可指定 start_line 和 line_count。只读。",
            "workspace.search_code": "在所选仓库 checkout 中搜索代码或文档文本。只读。",
            "repo_health.generate_summary": "总结当前仓库健康状态。",
            "rag.search_similar_documents": "搜索项目历史与非结构化知识，包括 Issue 描述、PR 说明与 Review 讨论、失败 CI 日志、项目文档、上传知识和已批准的长期记忆。可传入改写后的 query，并用不同 query 连续调用以补查和交叉验证。源码优先使用 workspace 工具，团队成员和实时状态使用对应结构化工具。",
            "issue_agent.analyze_issue": "分析最新同步的 Issue。",
            "pr_review_agent.analyze_pr": "分析最新同步的 PR。",
            "ci_debug_agent.analyze_workflow_run": "分析最新失败的 CI 运行。",
            "report_agent.generate_weekly_report": "生成仓库报告。",
            "safety_agent.classify_action_risk": "安全起草有风险的写操作或外部发送动作。",
            "workflow.run_engineering_review": "运行 Planner -> 专用 Agent -> Observer -> Synthesis 工作流，用于 PR 就绪度、阻塞项、优先级和多 Agent 仓库审查等跨领域工程决策。",
        }
        suffixes = {}
        if context:
            # 技能可以给某个工具追加描述（例如「本技能要求先读 A」），
            # 拼在末尾而不覆盖原描述：基础语义永远保留
            raw_suffixes = context.get("skill_description_by_tool") or {}
            if isinstance(raw_suffixes, dict):
                suffixes = raw_suffixes
        return f"公开工具：{name}。{descriptions.get(name, 'DevFlow 工具。')}{suffixes.get(name, '')}"

    def _tool_help(self, state: ConversationState) -> str:
        """把工具清单渲染成 prompt 里的列表段（- 名字: 描述）。"""
        tools = state.get("tools") or {}
        return "\n".join(f"- {name}: {tool.description}" for name, tool in tools.items())

    async def _final_from_observations(self, state: ConversationState) -> str:
        """基于「已完成工具调用的观察结果」生成最终回答。

        用于旁路 2/3：已经拿好了数据，只需要一次模型调用把结果转成自然语言，
        不再让模型自己决定调什么工具（那样又多几轮调用）。
        """
        if self.model is None:
            return self._local_observation_answer(state)
        try:
            # 这段系统提示强调三件事：只用已观察到的内容、判断要讲 WHY、
            # 人名只能来自 team_members / assignees（防止编造负责人）
            system_prompt = (
                "请使用用户的语言回答，并且只能基于观察结果、tool_result 记录和上下文。"
                "关键判断要说明 WHY；把证据、推断、不确定项和下一步区分开。"
                "如果是审查或合并建议，不能用 looks good、没什么问题一笔带过，必须保留风险和阻塞项。"
                "对于负责人、已分配人员、责任人或任务分配问题，只能从明确的 DevFlow team_members "
                "或工具返回的现有 GitHub assignees 中推荐人员。仓库文档、Issue 文本、代码、示例、"
                "角色设定、模型名和团队比喻都不能当成人。如果没有配置团队成员，也没有观察到现有 assignee，"
                "请说明当前还没有添加项目成员，并要求用户在团队面板中添加开发者。"
            )
            prompt_observations, prompt_events, runtime_stats = self._runtime_tool_context(state)
            state["runtime_context_stats"] = {**(state.get("runtime_context_stats") or {}), **runtime_stats}
            user_prompt = (
                f"用户消息：{state.get('message') or ''}\n"
                f"当前 Agent 配置的运行模型：{settings.llm_model}\n"
                f"DevFlow 团队上下文：{self._team_assignment_context(state.get('context') or {})}\n"
                f"上下文：{self._server_context(state.get('context') or {})}\n"
                f"观察结果：{json.dumps(prompt_observations, ensure_ascii=False)}\n"
                f"工具调用/结果索引：{json.dumps(prompt_events, ensure_ascii=False)}"
            )
            estimated = self._record_prompt_diagnostics(
                state,
                phase="final_from_observations",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            response = await self.model.ainvoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            )
            self._record_model_usage(
                state,
                response,
                phase="final_from_observations",
                estimated_prompt_tokens=estimated,
            )
            return self._message_content(response)
        except Exception:
            # 模型不可用时退回本地拼接：旁路 2/3 的价值在于「已经拿到数据」，
            # 不能因为最后一步总结失败就把数据也丢掉
            return self._local_observation_answer(state)

    def _local_observation_answer(self, state: ConversationState) -> str:
        """零成本兜底：把观察结果原文摘成条目输出。

        文本质量不如模型总结，但保证「有数据就一定能给出东西」。
        四级兜底顺序：观察摘录 → 预检索证据 → 服务端上下文 → 明确的「没有证据」提示。
        """
        observations = state.get("observations") or []
        excerpts: list[str] = []
        for item in observations:
            if not isinstance(item, dict):
                continue
            # 不同工具返回的字段名不同（answer/content/result/message），逐个试
            value = item.get("answer") or item.get("content") or item.get("result") or item.get("message")
            if not value and isinstance(item.get("observation"), dict):
                nested = item["observation"]
                # 报错的观察不摘：把错误信息当「证据」展示出去会误导用户
                if nested.get("route") == "tool_error":
                    continue
                value = nested.get("answer_preview") or nested.get("answer") or nested.get("content")
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            # 合并所有空白为单空格：工具输出常带大量换行缩进，直接展示会很乱
            text = " ".join(str(value or "").split())
            if text:
                # 每条截 600 字符，控制最终回答长度
                excerpts.append(text[:600])
        if not excerpts:
            # 四级兜底（见 docstring），这里从第二级开始
            evidence_answer = self._local_prefetched_evidence_answer(state)
            if evidence_answer:
                return evidence_answer
            context_text = " ".join(self._server_context(state.get("context") or {}).split())
            if context_text:
                return "已读取当前仓库上下文。\n\n" + context_text[:1200]
            return "当前没有取得足够证据。请先确认仓库已连接并完成同步；开放式对话还需要配置 LLM。"
        # 最多展示 4 条：这是「摘要」，不是完整报告
        return "已读取当前仓库证据。\n\n" + "\n\n".join(
            f"{index + 1}. {excerpt}" for index, excerpt in enumerate(excerpts[:4])
        )

    def _local_prefetched_evidence_answer(self, state: ConversationState) -> str:
        """用预检索证据（context["evidence"]）直接拼一个回答。

        针对「未关闭的 Issue」这类高频问题做了特化：能直接从 metadata.state 过滤出
        open 的条目，比让模型从一堆证据里挑更准，也不需要额外调用。
        其余情况按通用格式逐条罗列。
        """
        context = state.get("context") or {}
        raw_evidence = context.get("evidence")
        if not isinstance(raw_evidence, list):
            return ""
        evidence = [item for item in raw_evidence if isinstance(item, dict)]
        if not evidence:
            return ""

        # 两个条件同时满足才走特化：既提到 issue，又表达了「未关闭/待处理」的筛选意图
        message = str(state.get("message") or "").lower()
        asks_open_issues = "issue" in message and any(
            token in message for token in ["未关闭", "待处理", "open", "开放"]
        )
        if asks_open_issues:
            # 状态在 metadata.state 里，不在顶层：检索证据的结构是 {source_type, metadata: {...}}
            open_issues = [
                item
                for item in evidence
                if str(item.get("source_type") or "") == "issue"
                and str((item.get("metadata") or {}).get("state") or "").lower() == "open"
            ]
            if open_issues:
                lines = ["根据本轮预检索证据，当前仓库有这些未关闭的 Issue："]
                for item in open_issues[:8]:
                    title = str(item.get("title") or "未命名 Issue").strip()
                    snippet = " ".join(str(item.get("snippet") or "").split())
                    lines.append(f"- {title}" + (f"：{snippet[:240]}" if snippet else ""))
                # 明确说明「本轮没有调用工作区源码工具」：让用户知道结论的边界，
                # 避免他以为已经看过代码
                lines.append("\n数据来自当前仓库已同步的 Issue 证据；本轮没有调用工作区源码工具。")
                return "\n".join(lines)

        rendered = [self._render_evaluation_context(item) for item in evidence[:8]]
        rendered = [item for item in rendered if item]
        if not rendered:
            return ""
        return "根据当前仓库已同步的证据：\n\n" + "\n\n".join(
            f"{index + 1}. {item}" for index, item in enumerate(rendered)
        )

    def _prefetched_evidence_citations(self, state: ConversationState) -> list[dict[str, str]]:
        """把预检索证据转成引用列表（最多 8 条），与工具调用产生的 citations 结构一致。"""
        context = state.get("context") or {}
        evidence = context.get("evidence")
        if not isinstance(evidence, list):
            return []
        return [
            {
                "type": str(item.get("source_type") or "evidence"),
                "id": str(item.get("source_id") or item.get("id") or ""),
                "title": str(item.get("title") or "未命名证据"),
            }
            for item in evidence[:8]
            if isinstance(item, dict)
        ]

    def _append_agent_event(self, state: ConversationState, event_type: str, data: dict[str, Any]) -> dict[str, Any]:
        """追加一条 trace 事件并返回它。

        所有事件都自动带上当前 step，前端据此把事件按轮次分组展示；
        返回事件对象是为了让调用方（如 tool_use）能继续往里塞字段后复用。
        """
        event = {
            "type": event_type,
            "step": state.get("step", 0),
            **data,
        }
        state["agent_events"] = [*(state.get("agent_events") or []), event]
        return event

    def _tool_event_transcript(self, state: ConversationState) -> list[dict[str, Any]]:
        """筛出「与工具调用相关」的事件，用作 prompt 里的工具调用索引。

        只保留这五类：tool_use / tool_result / tool_retry / agent_step / agent_loop_stop。
        排除 thought、skill_activated 等：它们对「让模型记住自己做过什么」没帮助，
        放进去只会白占 token。
        """
        return [
            event
            for event in state.get("agent_events") or []
            if event.get("type") in {"tool_use", "tool_result", "tool_retry", "agent_step", "agent_loop_stop"}
        ]

    def _runtime_tool_context(
        self,
        state: ConversationState,
        *,
        aggressive: bool = False,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        """Project complete tool state into a bounded prompt-only view.

        The full observations and events remain in ``state`` and are sealed in
        ChatSession.  Only this derived view is shortened for the next model
        call, matching Claude Code's separation between durable transcript and
        active context.

        中文补充（这段是成本控制的核心）：
        完整观察结果留在 state / 会话转录里，**只有**投影后的精简视图进 prompt。
        原因是工具结果往往几千 token，而模型下一步真正需要的只是「做了什么、看到什么要点」。
        投影从**最新**往前取（reversed），因为最近的工具结果最相关，旧的先被丢弃。

        返回三样东西：精简后的观察列表、压缩后的工具事件索引、以及一份统计
        （进 trace，用来解释「这次为什么只给模型看了这么点」）。
        """
        # 观察结果预算，aggressive 时砍到 1/4：这是溢出重试时的最后手段
        raw_observations = [item for item in state.get("observations") or [] if isinstance(item, dict)]
        budget = default_context_budget().tool_observation_tokens
        if aggressive:
            budget = max(512, budget // 4)
        # 每条的上限按「条目数」均分预算；夹在 [160, 1400]：
        # 160 保证单条不会短到没信息，1400 防止只有一条时把整个预算吃光
        per_item_budget = max(160, min(1400, budget // max(len(raw_observations), 1)))
        remaining = budget
        kept_reversed: list[dict[str, Any]] = []
        omitted = 0
        for item in reversed(raw_observations):
            next_item = dict(item)
            observation = next_item.get("observation")
            if isinstance(observation, dict):
                next_observation = dict(observation)
                # 三个字段是同一内容的不同命名，取第一个有值的作为 answer_preview，
                # 然后删掉另外两个，避免同一段文本在一份投影里出现三次
                preview = str(
                    next_observation.get("answer_preview")
                    or next_observation.get("answer")
                    or next_observation.get("content")
                    or ""
                )
                if preview:
                    next_observation["answer_preview"] = clip_to_token_budget(preview, per_item_budget)
                    next_observation.pop("answer", None)
                    next_observation.pop("content", None)
                next_item["observation"] = next_observation
            else:
                # 非 dict 的观察（比如直接是字符串）也要按预算截断
                next_item["observation"] = clip_to_token_budget(str(observation or ""), per_item_budget)

            # 用序列化后的 token 数作为真实成本估算——比按字符数估算更贴近实际计费
            serialized = json.dumps(next_item, ensure_ascii=False, default=str)
            cost = estimate_tokens(serialized)
            # 已经放进了至少一条且预算不够 → 丢弃这条（并计数），保证「至少留一条最新的」
            if cost > remaining and kept_reversed:
                omitted += 1
                continue
            # 一条都还没放进去就超预算 → 强行压缩成一个最小占位，至少让模型知道
            # 「曾经有一次调用，结果被压缩了」，而不是完全消失
            if cost > remaining:
                next_item = {
                    "tool_name": str(item.get("tool_name") or "tool"),
                    "observation": clip_to_token_budget(serialized, max(120, remaining)),
                    "compacted": True,
                }
                cost = estimate_tokens(json.dumps(next_item, ensure_ascii=False))
            kept_reversed.append(next_item)
            remaining = max(0, remaining - cost)
            if remaining <= 0:
                # 预算耗尽：剩下更早的直接全算作省略，不再逐条处理
                omitted += max(0, len(raw_observations) - len(kept_reversed))
                break

        # 反转为时间正序：模型读「先做了什么、后做了什么」更自然
        prompt_observations = list(reversed(kept_reversed))
        if omitted:
            # 在最前面插一条说明，告诉模型「有更早的结果被省略，需要时可去 transcript 取」，
            # 否则模型会以为自己只调用过这么几次工具
            prompt_observations.insert(
                0,
                {
                    "compacted": True,
                    "omitted_old_observations": omitted,
                    "recovery": "完整工具结果保存在本轮 ChatSession transcript，可按 session 读取。",
                },
            )

        # 工具事件索引：只保留「结构性字段」，让模型能回顾调用顺序与状态，
        # 但不重复携带结果正文（正文已在 observations 里）
        prompt_events: list[dict[str, Any]] = []
        for event in self._tool_event_transcript(state):
            summary = {
                key: value
                for key, value in event.items()
                if key
                in {
                    "type",
                    "step",
                    "id",
                    "tool_use_id",
                    "tool_name",
                    "arguments",
                    "status",
                    "route",
                    "error_kind",
                    "retryable",
                    "reason",
                }
            }
            if "arguments" in summary:
                # 参数本身可能很长（比如超长 query），超过 240 token 就截断
                args_text = json.dumps(summary["arguments"], ensure_ascii=False, default=str)
                if estimate_tokens(args_text) > 240:
                    summary["arguments"] = clip_to_token_budget(args_text, 240)
            prompt_events.append(summary)

        # 统计「省了多少 token」：这是判断投影策略是否过激/过松的关键指标
        original_tokens = estimate_tokens(json.dumps(raw_observations, ensure_ascii=False, default=str))
        prompt_tokens = estimate_tokens(json.dumps(prompt_observations, ensure_ascii=False, default=str))
        return prompt_observations, prompt_events, {
            "mode": "runtime_tool_projection",
            "aggressive": aggressive,
            "observation_budget_tokens": budget,
            "original_observation_tokens": original_tokens,
            "prompt_observation_tokens": prompt_tokens,
            "estimated_tokens_saved": max(0, original_tokens - prompt_tokens),
            "source_observations": len(raw_observations),
            "prompt_observations": len(prompt_observations),
            "omitted_observations": omitted,
            "tool_event_index_entries": len(prompt_events),
        }

    def _compress_history_for_retry(
        self,
        history: list[dict[str, str]],
        max_tokens: int,
    ) -> list[dict[str, str]]:
        """从最新往前取历史，直到用完预算。

        关键顺序：reversed 后从最新开始，超预算立刻 break —— 保证「最近几轮」一定在，
        丢掉的是最早的对话。这与人类回忆一致，也是多轮对话最需要的语义。
        每条内容也单独截到 min(700, 预算/3)：防止某一条超长消息独占整个预算。
        """
        kept: list[dict[str, str]] = []
        used = 0
        for item in reversed(history):
            content = clip_to_token_budget(item.get("content") or "", max(120, min(700, max_tokens // 3)))
            cost = estimate_tokens(item.get("role")) + estimate_tokens(content)
            # kept 非空才允许 break：至少保留一条，否则可能返回空历史
            if kept and used + cost > max_tokens:
                break
            kept.append({"role": item.get("role") or "user", "content": content})
            used += cost
        return list(reversed(kept))

    def _record_prompt_diagnostics(
        self,
        state: ConversationState,
        *,
        phase: str,
        system_prompt: str,
        user_prompt: str,
    ) -> int:
        """记录一次模型调用前的 prompt 体积诊断，并返回估算的总 token。

        三段分开统计（system / user / 工具 schema）的原因：出问题时能立刻看出
        是「提示词太长」还是「工具太多」。工具 schema 往往是被忽视的大头
        （十几个工具的 JSON schema 加起来可能上万 token）。
        """
        system_tokens = estimate_tokens(system_prompt)
        user_tokens = estimate_tokens(user_prompt)
        tool_schema_tokens = self._tool_schema_estimated_tokens(state)
        total = system_tokens + user_tokens + tool_schema_tokens
        # pressure 是相对上下文窗口的占用比例，用于提前预警（而不是等真溢出）
        pressure = calculate_context_pressure(total)
        diagnostic = {
            "phase": phase,
            "estimated_prompt_tokens": total,
            "system_tokens": system_tokens,
            "user_context_tokens": user_tokens,
            "tool_schema_tokens": tool_schema_tokens,
            "pressure": pressure.as_dict(),
        }
        state["prompt_diagnostics"] = [*(state.get("prompt_diagnostics") or []), diagnostic]
        return total

    def _tool_schema_estimated_tokens(self, state: ConversationState) -> int:
        """估算工具 schema 占用的 token。

        工具 schema 是「隐形的大头」：每个工具的 name + description + args JSON schema
        都会进 prompt。这里把它算进总预算，才能准确判断上下文压力——
        否则会低估，导致溢出重试比预期更频繁。
        """
        total = 0
        for tool in (state.get("tools") or {}).values():
            total += estimate_tokens(str(getattr(tool, "name", "")))
            total += estimate_tokens(str(getattr(tool, "description", "")))
            args_schema = getattr(tool, "args_schema", None)
            try:
                schema = args_schema.model_json_schema() if args_schema else {}
            except (AttributeError, TypeError, ValueError):
                # schema 生成失败不该影响主流程，按 0 计（只会略微低估）
                schema = {}
            total += estimate_tokens(json.dumps(schema, ensure_ascii=False, default=str))
        return total

    def _record_model_usage(
        self,
        state: ConversationState,
        response: Any,
        *,
        phase: str,
        estimated_prompt_tokens: int,
    ) -> None:
        """记录一次模型调用的 token 用量。

        source 字段很重要：provider 表示「真的拿到了账单数据」，estimate 表示
        「模型网关没返用量，用的是本地估算」。事后核算成本时必须区分这两类，
        否则汇总出来的数字会真假混杂。
        """
        usage = self._normalize_model_usage(response)
        usage.update(
            {
                "phase": phase,
                "estimated_prompt_tokens": estimated_prompt_tokens,
                "source": "provider" if usage.get("input_tokens") is not None else "estimate",
            }
        )
        # 拿不到真实用量就用估算值补齐，保证每条记录都有可计算的输入 token
        if usage.get("input_tokens") is None:
            usage["input_tokens"] = estimated_prompt_tokens
        if usage.get("total_tokens") is None:
            usage["total_tokens"] = int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0)
        state["model_usage"] = [*(state.get("model_usage") or []), usage]

    def _normalize_model_usage(self, response: Any) -> dict[str, Any]:
        """把不同 provider/网关返回的用量结构统一成一套字段。

        要兼容三种来源：LangChain 的 usage_metadata（可能是对象或 dict）、
        OpenAI 风格的 response_metadata.token_usage（prompt_tokens/completion_tokens）、
        以及部分网关的 response_metadata.usage。都取不到时保留 None，
        交给 _record_model_usage 标记为 estimate。
        """
        raw = getattr(response, "usage_metadata", None)
        if raw is not None and not isinstance(raw, dict) and hasattr(raw, "model_dump"):
            raw = raw.model_dump()
        raw = raw if isinstance(raw, dict) else {}
        response_metadata = getattr(response, "response_metadata", None)
        response_metadata = response_metadata if isinstance(response_metadata, dict) else {}
        token_usage = response_metadata.get("token_usage") or response_metadata.get("usage") or {}
        token_usage = token_usage if isinstance(token_usage, dict) else {}
        # cache_read / cache_creation 是成本分析的关键：缓存命中的 token 单价低得多
        input_details = raw.get("input_token_details") or {}
        input_details = input_details if isinstance(input_details, dict) else {}
        return {
            "input_tokens": raw.get("input_tokens", token_usage.get("prompt_tokens")),
            "output_tokens": raw.get("output_tokens", token_usage.get("completion_tokens")),
            "total_tokens": raw.get("total_tokens", token_usage.get("total_tokens")),
            "cache_read_input_tokens": input_details.get("cache_read", token_usage.get("cache_read_input_tokens", 0)),
            "cache_creation_input_tokens": input_details.get("cache_creation", token_usage.get("cache_creation_input_tokens", 0)),
            "model": response_metadata.get("model_name") or settings.llm_model,
        }

    def _runtime_metadata(self, state: ConversationState) -> dict[str, Any]:
        """打包「运行时诊断信息」，会被展开进 final 事件。

        三块内容回答三个问题：花了多少 token、每次 prompt 多大、是否发生过溢出重试。
        这些字段不影响回答本身，但决定了线上能不能定位成本与稳定性问题。
        """
        return {
            "model_usage": state.get("model_usage") or [],
            "context_diagnostics": {
                "prompt_calls": state.get("prompt_diagnostics") or [],
                "runtime_tool_context": state.get("runtime_context_stats") or {},
                "reactive_overflow_retry": bool(state.get("context_overflow_recovery_attempted")),
            },
        }

    async def _fallback_answer(self, state: ConversationState) -> str:
        """兜底回答：复用「基于观察结果总结」这条路径。

        单独留一个方法名是为了语义清晰——它表示「主流程没能给出答案」，
        而不是一次正常的总结调用。
        """
        return await self._final_from_observations(state)

    def _history_messages(self, context: dict[str, Any]) -> list[dict[str, str]]:
        """从上下文里取出可用的对话历史。

        只保留 user / assistant 且内容非空的纯文本消息：
        - tool 角色、系统消息不进历史（会干扰模型对「对话」的理解）；
        - 非字符串 content（多模态数组等）暂时不支持，直接跳过。
        server_messages 优先于 messages：前者是服务端整理过的完整视图。
        """
        raw_messages = context.get("server_messages") or context.get("messages") or []
        if not isinstance(raw_messages, list):
            return []
        messages: list[dict[str, str]] = []
        for item in raw_messages:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = item.get("content")
            if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
                messages.append({"role": role, "content": content})
        return messages

    def _server_context(self, context: dict[str, Any]) -> str:
        """服务端预组装好的上下文文本（仓库信息、相关证据摘要等），原样透传。"""
        value = context.get("server_context")
        return str(value) if value else ""

    def _registered_skills_context(self, context: dict[str, Any]) -> str:
        """技能目录摘要。没有技能时给一句明确的「没有」，而不是空串——
        空串会让模型以为「这里有内容但被吞了」，从而反复试探。"""
        value = context.get("registered_skills")
        return str(value) if value else "当前没有注册 DevFlow 技能。"

    def _activate_skills(self, message: str, context: dict[str, Any]) -> list[dict[str, Any]]:
        """按消息匹配并激活技能。

        注册表由 context["_skill_registry"] 注入（下划线开头表示内部依赖，不算公开上下文）；
        不是 SkillRegistry 实例时直接返回空列表——技能是可选能力，缺失不应阻断对话。
        激活结果可能是多个：一个请求可以同时命中多个技能。
        """
        registry = context.get("_skill_registry")
        if not isinstance(registry, SkillRegistry):
            return []
        # requested_skill / skill_name 是显式指定（前端点击技能入口时传入），
        # 有它时注册表会优先按名字激活，而不是靠关键词匹配
        requested_name = context.get("requested_skill") or context.get("skill_name")
        activations = registry.activate(
            message,
            requested_name=str(requested_name).strip() if requested_name else None,
        )
        return [activation.model_dump(mode="json") for activation in activations]

    def _active_skills(self, context: dict[str, Any]) -> list[dict[str, Any]]:
        """取已激活的技能列表，过滤掉非 dict 的脏数据。"""
        skills = context.get("active_skills") or []
        return [item for item in skills if isinstance(item, dict)] if isinstance(skills, list) else []

    def _active_skill_prompt(self, context: dict[str, Any]) -> str:
        """把激活技能的完整指令拼成 prompt 段。

        这里是「按需加载」的落点：技能目录一直在 prompt 里（轻量），但完整指令正文
        只有被激活时才拼进来——技能正文通常很长，全量常驻会让每次调用都变贵。
        最后用 6000 token 上限截断，避免一个超大技能把预算吃光。
        """
        skills = self._active_skills(context)
        if not skills:
            return "本轮没有匹配到需要加载完整正文的 Skill；仅保留轻量目录摘要。"
        blocks = []
        for skill in skills:
            # 资源（附带的参考文件）逐个渲染；没有 content 的资源跳过（可能只是占位声明）
            resources = skill.get("resources") if isinstance(skill.get("resources"), list) else []
            resource_sections = []
            for resource in resources:
                if not isinstance(resource, dict) or not str(resource.get("content") or "").strip():
                    continue
                resource_sections.append(f"### 资源：{resource.get('path')}\n{resource.get('content')}")
            resource_text = "\n\n按需读取的资源：\n" + "\n\n".join(resource_sections) if resource_sections else ""
            # 头部把「激活方式/原因/允许工具/输出契约/安全级别」都写出来：
            # 让模型知道自己此刻受哪些约束，尤其是「允许工具」——越界会被拦下
            blocks.append(
                f"## {skill.get('skill_title')} ({skill.get('skill_name')} v{skill.get('skill_version')})\n"
                f"激活方式：{skill.get('activation_mode')}；原因：{skill.get('activation_reason')}\n"
                f"允许工具：{', '.join(str(item) for item in skill.get('tools') or []) or '无'}\n"
                f"输出契约：{skill.get('output_contract')}；安全级别：{skill.get('safety_level')}\n\n"
                f"{str(skill.get('instructions') or '').strip()}"
                f"{resource_text}"
            )
        # suffix 会在截断处明确标注，避免模型读到半截指令还以为完整
        return clip_to_token_budget("\n\n".join(blocks), 6000, suffix="\n[Skill 指令因上下文预算被截断]")

    def _explicit_skill_tool_violation(self, tool_name: str, state: ConversationState) -> str | None:
        """检查工具调用是否越出「显式激活技能」的允许范围。

        只对 activation_mode == "explicit" 的技能生效：用户明确点了某技能，
        就应当严格限制在该技能声明的工具内（这是用户预期，也是权限边界）。
        自动匹配的技能不限制，否则会把它误伤成「什么都不能调」。
        """
        context = state.get("context") or {}
        explicit = [
            skill
            for skill in self._active_skills(context)
            if skill.get("activation_mode") == "explicit"
        ]
        if not explicit:
            return None
        # 多个显式技能时取并集：叫了同时调用两个技能也算合法
        allowed = {
            str(tool)
            for skill in explicit
            for tool in skill.get("tools") or []
        }
        if tool_name in allowed:
            return None
        skill_names = "、".join(str(skill.get("skill_name")) for skill in explicit)
        return f"工具 {tool_name} 不在显式 Skill（{skill_names}）声明的允许工具列表中，已拒绝执行。"

    def _record_skill_validation(self, state: ConversationState) -> None:
        """收尾时给每个激活的技能出一份「执行合规报告」。

        回答的问题是：这个技能声明的步骤/入口，本轮到底有没有被执行、有没有越界。
        判定依据全部来自 agent_events（tool_use / tool_result / workflow_task_result），
        所以必须在流程结束后调用，且不能用它来阻断执行——它只是审计。
        """
        context = state.get("context") or {}
        active = self._active_skills(context)
        if not active:
            return
        observed_tools = [
            str(event.get("tool_name") or "")
            for event in state.get("agent_events") or []
            if isinstance(event, dict) and event.get("type") == "tool_use" and event.get("tool_name")
        ]
        successful_tools = [
            str(event.get("tool_name") or "")
            for event in state.get("agent_events") or []
            if isinstance(event, dict)
            and event.get("type") == "tool_result"
            and event.get("status") == "success"
            and event.get("tool_name")
        ]
        explicit_allowed = {
            str(tool)
            for skill in active
            if skill.get("activation_mode") == "explicit"
            for tool in skill.get("tools") or []
        }
        explicit_violations = [tool for tool in observed_tools if explicit_allowed and tool not in explicit_allowed]
        for skill in active:
            allowed = {str(tool) for tool in skill.get("tools") or []}
            used = [tool for tool in observed_tools if tool in allowed]
            entrypoint_called = str(skill.get("entrypoint") or "") in used
            successful_used = [tool for tool in successful_tools if tool in allowed]
            entrypoint_succeeded = str(skill.get("entrypoint") or "") in successful_used
            delegated_tasks = [
                event
                for event in state.get("agent_events") or []
                if isinstance(event, dict)
                and event.get("type") == "workflow_task_result"
                and event.get("skill_name") == skill.get("skill_name")
                and event.get("status") == "success"
            ]
            # 状态判定是有序的，先判「违规」再判「完成」：
            # violated 优先级最高，因为一旦调用了越界工具，即使主流程跑通了也不算合规
            if explicit_violations and skill.get("activation_mode") == "explicit":
                status = "violated"
            elif entrypoint_succeeded or successful_used or delegated_tasks:
                status = "completed"
            elif used:
                # 调用过但没有任何成功 → failed（而不是 not_executed）
                status = "failed"
            else:
                status = "not_executed"
            self._append_agent_event(
                state,
                "skill_validation",
                {
                    "skill_name": skill.get("skill_name"),
                    "skill_title": skill.get("skill_title"),
                    "skill_version": skill.get("skill_version"),
                    "status": status,
                    "instructions_loaded": bool(str(skill.get("instructions") or "").strip()),
                    "instruction_digest": skill.get("instruction_digest"),
                    "declared_steps": skill.get("workflow_steps") or [],
                    "observed_tools": used,
                    "entrypoint_called": entrypoint_called,
                    "entrypoint_succeeded": entrypoint_succeeded,
                    "delegated_task_ids": [str(event.get("task_id") or "") for event in delegated_tasks],
                    "tool_contract_violations": explicit_violations if skill.get("activation_mode") == "explicit" else [],
                    "step_verification": (
                        "entrypoint_completed"
                        if entrypoint_succeeded
                        else "delegated_entrypoint_completed"
                        if delegated_tasks
                        else "supporting_tools_observed"
                        if successful_used
                        else "skill_tool_failed"
                        if used
                        else "no_skill_tool_observed"
                    ),
                },
            )

    def _skill_event_metadata(self, tool_name: str, state: ConversationState) -> dict[str, Any]:
        """给工具事件附加「这次调用属于哪个技能」的元信息。

        三级匹配，从精确到宽泛：
        1. 工具正好是该技能的入口（entrypoint）→ 最准；
        2. 工具在技能声明的工具列表里；
        3. 兜底取任一「显式激活」的技能（用户点了技能，那这次调用大概率属于它）。
        都不命中时再查 context 里的静态映射（skill_trace_by_tool），仍无则返回空 dict。
        """
        context = state.get("context") or {}
        active = self._active_skills(context)
        selected = next((skill for skill in active if skill.get("entrypoint") == tool_name), None)
        if selected is None:
            selected = next((skill for skill in active if tool_name in (skill.get("tools") or [])), None)
        if selected is None:
            selected = next((skill for skill in active if skill.get("activation_mode") == "explicit"), None)
        if selected is not None:
            return {
                "skill_name": selected.get("skill_name"),
                "skill_title": selected.get("skill_title"),
                "skill_version": selected.get("skill_version"),
                "skill_steps": selected.get("workflow_steps") or [],
                "skill_activation_mode": selected.get("activation_mode"),
                "skill_activation_reason": selected.get("activation_reason"),
                "skill_instruction_digest": selected.get("instruction_digest"),
            }
        # 静态映射兜底：允许上层直接指定「某工具属于某技能」，不依赖激活状态
        mapping = context.get("skill_trace_by_tool") or {}
        if not isinstance(mapping, dict):
            return {}
        metadata = mapping.get(tool_name)
        if not isinstance(metadata, dict):
            return {}
        return {
            key: value
            for key, value in metadata.items()
            if key in {"skill_name", "skill_title", "skill_version", "skill_steps"}
        }

    def _team_assignment_context(self, context: dict[str, Any]) -> str:
        """渲染团队成员上下文，供 prompt 里的「分配硬性规则」引用。

        空列表时给的是**明确的一句话**而不是 "[]"：模型看到空数组容易自己补人，
        看到「还没有配置」才会按规则要求用户去添加成员。
        """
        members = self._team_members(context)
        if not members:
            return "team_members: []（当前仓库还没有配置 DevFlow 团队成员。）"
        return "team_members: " + json.dumps(members, ensure_ascii=False)

    def _team_members(self, context: dict[str, Any]) -> list[dict[str, str]]:
        """规整团队成员数据，最多 20 人。

        统一字段名（techStack / tech_stack 合并成一个）并补齐空串，
        这样 prompt 里的 JSON 结构稳定，模型更容易稳定引用。
        20 人上限是为了控制 prompt 体积——团队通常不会超过这个规模。
        """
        raw_members = context.get("team_members") or []
        if not isinstance(raw_members, list):
            return []
        members: list[dict[str, str]] = []
        for member in raw_members:
            if not isinstance(member, dict):
                continue
            name = str(member.get("name") or "").strip()
            if not name:
                continue
            members.append(
                {
                    "name": name,
                    "role": str(member.get("role") or "").strip(),
                    "strengths": str(member.get("strengths") or "").strip(),
                    "techStack": str(member.get("techStack") or member.get("tech_stack") or "").strip(),
                }
            )
        return members[:20]

    def _summarize_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """把工具原始结果压成「观察摘要」，这是进 prompt 和 trace 的版本。

        预算按工具类型区分：workspace_* 工具给全额预算（源码片段需要完整上下文，
        截断会让模型基于半截代码下判断），其他工具只给 1/3（摘要够用）。
        同时记录压缩前后的 token 数，便于评估压缩是否过狠。
        """
        answer = str(observation.get("answer") or "")
        budget = default_context_budget()
        # startswith("workspace_") 判断依据是路由名，不是工具名——两者在工具实现里保持一致
        limit = budget.tool_observation_tokens if str(observation.get("route") or "").startswith("workspace_") else max(300, budget.tool_observation_tokens // 3)
        compressed = clip_to_token_budget(answer, limit)
        return {
            "route": observation.get("route"),
            "answer_preview": compressed,
            "compression": {
                "mode": "tool_observation_budget",
                "original_estimated_tokens": estimate_tokens(answer),
                "compressed_estimated_tokens": estimate_tokens(compressed),
                "budget_tokens": limit,
            },
            "citations_count": len(observation.get("citations", [])),
        }

    def _initial_evaluation_trace(self, message: str, context: dict[str, Any]) -> dict[str, Any]:
        """初始化评测 trace，把「预检索」阶段先记进去。

        retrievals 只有一个 stage=prefetch 条目：那是调用方在进来之前就做好的检索。
        若没任何预检索证据，retrievals 保持空列表——不要塞一条空记录，
        否则评测时会把「没检索」误算成「检索了但没结果」。
        """
        evidence = context.get("evidence") if isinstance(context.get("evidence"), list) else []
        contexts: list[str] = []
        sources: list[dict[str, Any]] = []
        for item in evidence:
            if not isinstance(item, dict):
                continue
            rendered_context = self._render_evaluation_context(item)
            if rendered_context:
                contexts.append(rendered_context)
            sources.append(
                {
                    "id": str(item.get("id") or ""),
                    "source_id": str(item.get("source_id") or ""),
                    "source_type": str(item.get("source_type") or ""),
                    "title": str(item.get("title") or ""),
                    "score": item.get("score"),
                }
            )
        retrievals = []
        if contexts or sources:
            retrievals.append(
                {
                    "stage": "prefetch",
                    "query": message,
                    "contexts": contexts,
                    "sources": sources,
                }
            )
        return {
            "target": "ChatAgent",
            "retrievals": retrievals,
            "tool_observations": [],
        }

    def _capture_evaluation_observation(
        self,
        state: ConversationState,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        observation: dict[str, Any],
    ) -> None:
        """评测模式：把一次工具调用记为「观测」，供离线评估检索质量。

        非评测模式下直接 return（这是第一行的作用），所以生产调用完全没有额外开销。
        评测关心的两类信息：
        - tool_observations：调了什么、成功与否、原始回答；
        - retrievals：检索类工具返回了哪些来源与上下文。
        """
        if not (state.get("context") or {}).get("evaluation_mode"):
            return
        # dict(...) 复制后再改，最后整体写回 state：避免中途异常留下半更新的 trace
        trace = dict(state.get("evaluation_trace") or {})
        tool_observations = list(trace.get("tool_observations") or [])
        tool_observations.append(
            {
                "tool_name": tool_name,
                "arguments": arguments,
                # 与 _complete_tool_call 的口径一致：任一子调用失败即判 error
                "status": "error"
                if any(
                    call.get("status") == "error"
                    for call in observation.get("tool_calls", [])
                    if isinstance(call, dict)
                )
                else "success",
                "content": str(observation.get("answer") or ""),
            }
        )
        trace["tool_observations"] = tool_observations

        # 两种检索结果结构都要兼容：retrieval_results（结构化来源）与
        # retrieved_contexts（纯文本上下文）。优先用前者渲染，没有才用后者
        raw_sources = observation.get("retrieval_results")
        sources = [dict(item) for item in raw_sources if isinstance(item, dict)] if isinstance(raw_sources, list) else []
        contexts = [self._render_evaluation_context(item) for item in sources]
        contexts = [item for item in contexts if item]
        if not contexts:
            contexts = [
                str(item).strip()
                for item in observation.get("retrieved_contexts", [])
                if str(item).strip()
            ]
        # 记录条件：是 rag 工具，或确实带了检索产物。普通读文件工具即使成功也不记 retrieval
        if tool_name == "rag.search_similar_documents" or contexts or sources:
            retrievals = list(trace.get("retrievals") or [])
            retrievals.append(
                {
                    "stage": "agent_tool",
                    "query": str(arguments.get("query") or arguments.get("message") or ""),
                    "contexts": contexts,
                    "sources": sources,
                }
            )
            trace["retrievals"] = retrievals
        state["evaluation_trace"] = trace

    @staticmethod
    def _render_evaluation_context(item: dict[str, Any]) -> str:
        """Render the same title/source/snippet evidence shape shown to ChatAgent.

        中文补充：把一条证据渲染成「[来源类型] 标题: 片段」。
        必须是静态方法且与展示逻辑保持一致——评测要评估的就是「模型实际看到的样子」，
        渲染格式一变，历史评测结果就不可比了。
        """

        source_type = str(item.get("source_type") or "").strip()
        title = str(item.get("title") or "").strip()
        snippet = str(item.get("snippet") or item.get("content") or "").strip()
        # 用空格连接非空部分：缺来源类型或缺标题时自动省略，不产生多余空格或空方括号
        heading = " ".join(part for part in ([f"[{source_type}]" if source_type else "", title]) if part)
        if heading and snippet:
            return f"{heading}: {snippet}"
        return heading or snippet

    def _evaluation_result(self, state: ConversationState) -> dict[str, Any]:
        """收尾时汇总评测 trace：去重所有上下文，并附上工具事件转录。

        非评测模式返回空 dict（展开进 final 事件时等于不加字段）。
        上下文去重是必须的：同一条证据可能在预检索和工具检索里各出现一次，
        重复会让评测的「召回数」虚高。
        """
        if not (state.get("context") or {}).get("evaluation_mode"):
            return {}
        trace = dict(state.get("evaluation_trace") or {})
        contexts: list[str] = []
        seen: set[str] = set()
        for retrieval in trace.get("retrievals") or []:
            if not isinstance(retrieval, dict):
                continue
            for item in retrieval.get("contexts") or []:
                text = str(item).strip()
                if text and text not in seen:
                    seen.add(text)
                    contexts.append(text)
        trace["retrieved_contexts"] = contexts
        trace["agent_events"] = self._tool_event_transcript(state)
        return {"evaluation_trace": trace}

    def _message_content(self, message: BaseMessage) -> str:
        """从 LangChain 消息里取纯文本。

        三种形态都要处理：纯字符串、内容块列表（多模态/部分 provider 的返回）、其他。
        列表形态只在有 text/content 字段时取值，其余块（图片等）忽略——
        本 Agent 只消费文本。
        """
        content = message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.append(str(item.get("text") or item.get("content") or ""))
            return "".join(parts)
        return str(content or "")

    async def _stream_answer(self, answer: str) -> AsyncIterator[dict[str, Any]]:
        """把完整回答切成 delta 事件推给前端，模拟「打字机」效果。

        注意这不是真的流式模型输出（答案已经生成好了），只是让前端体验一致。
        真正的流式需要模型层支持 stream，代价是这层逻辑要重写。
        """
        for part in self._chunk_text(answer):
            yield self._event("delta", {"content": part})

    def _chunk_text(self, text: str, size: int = 36) -> list[str]:
        """按固定字符数切块。

        36 是前端渲染节奏的经验值：太小会让事件数量暴增（每个事件都有 SSE 开销），
        太大又看不出「逐字」的感觉。
        """
        if not text:
            return []
        return [text[index : index + size] for index in range(0, len(text), size)]

    def _event(self, event: str, data: dict[str, Any]) -> dict[str, Any]:
        """统一的流事件信封：{"event": 类型, "data": 负载}。"""
        return {"event": event, "data": data}


def encode_sse(event: dict[str, Any]) -> str:
    """把事件编码成 SSE 帧。

    格式必须是「event: ...\\ndata: ...\\n\\n」：
    结尾的空行是 SSE 协议的分帧标志，少了它前端会一直等下一个事件。
    ensure_ascii=False 保证中文直接输出而不是 \\uXXXX 转义（体积更小，也便于抓包排查）。
    """
    return f"event: {event['event']}\ndata: {json.dumps(event['data'], ensure_ascii=False)}\n\n"
