"""RAG 质量评估的执行器：跑真实 ChatAgent，再用 Ragas + 确定性规则打分。

为什么要「真实跑一遍 Agent」而不是拿历史回答来评分：
只有走完整条链路（检索 → 组装上下文 → 工具调用 → 生成），
才能测出「检索没召回」「上下文被压缩掉」「工具没被调用」这类端到端问题。

三类分数并存，各自解决不同问题：
1. 确定性硬规则（_hard_rules）：来源命中、必要工具、必含/禁含短语、正则格式。
   完全可解释、零成本、不依赖模型，是发布门禁的底线。
2. 检索指标（_retrieval_metrics）：recall@k / precision@k / MRR / nDCG，只看检索层。
3. Ragas 语义指标：context precision/recall、faithfulness、answer relevancy、
   以及 agent 目标达成与工具调用准确度。这一层需要 judge 模型，可能失败，
   所以失败时记为「不完整」而不是「不通过」——避免把基础设施问题算成质量问题。
"""

import asyncio
import copy
import importlib.metadata
import inspect
import math
import re
import sys
import time
import types
import uuid
import warnings
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import Conversation, Document, Issue, PullRequest, Repository, WorkflowRun
from app.services.agents.chat_agent import ChatAgent
from app.services.chat_memory import ContextAssembler, EvidenceStore
from app.services.rag.embeddings import embed_text
from app.services.rag.source_policy import RAG_SOURCE_TYPES


# Ragas 的四个经典 RAG 指标：上下文精确率/召回率、忠实度、答案相关性。
# 这四个缺任何一个都无法给出「完整」的评测结论，所以单列出来。
CORE_RAGAS_METRICS = (
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
)
# 门禁必需项 = 四个核心 + agent 目标达成（后者衡量「有没有按要求完成任务」）。
REQUIRED_RAGAS_METRICS = (*CORE_RAGAS_METRICS, "agent_goal_accuracy")
# agent 专项指标：只有在用例声明了 expected_tool_calls 时才会真正计算。
AGENT_RAGAS_METRICS = (
    "agent_goal_accuracy",
    "tool_call_accuracy",
    "tool_call_f1",
)

# faithfulness 打分时单块的最大字符数（见 _split_faithfulness_response 的说明）。
_FAITHFULNESS_CHUNK_CHARS = 700


def _split_faithfulness_response(text: str, max_chars: int = _FAITHFULNESS_CHUNK_CHARS) -> list[str]:
    """Split a long answer without dropping content before native Ragas scoring.

    Faithfulness first asks the judge to extract all claims. Some OpenAI-compatible
    reasoning models can exhaust their structured-output budget even for a moderate
    answer. Scoring smaller semantic chunks keeps that native metric reliable; the
    final score is weighted by the amount of answer text in each chunk.

    译文：在交给原生 Ragas 打分之前，把长答案切块，但**不丢任何内容**。
    faithfulness 的第一步是让 judge 抽出答案里的全部断言；部分 OpenAI 兼容的推理模型
    即使面对中等长度的答案也会把结构化输出预算耗尽。切成语义小块能让该指标稳定下来，
    最终分数按每块答案的文本长度加权（长块权重大，避免短块主导结果）。

    切分策略：先按句子/空行切，过长的单句再硬切成 max_chars。
    保证「拼接所有块 == 原文（只差换行）」是这里最关键的约束。
    """

    normalized = text.strip()
    if not normalized:
        # 返回一个空串块而不是空列表：调用方可以照常走打分流程（得到 0 分或异常）。
        return [""]

    # 先切分句（句末标点或换行），再按长度装箱；这样块边界落在语义边界上。
    units = [unit.strip() for unit in re.split(r"(?<=[。！？.!?])\s+|\n+", normalized) if unit.strip()]
    chunks: list[str] = []
    current = ""
    for unit in units:
        # 单个"句子"就超过上限（比如一长串没有标点的日志），只能硬切。
        while len(unit) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(unit[:max_chars])
            unit = unit[max_chars:]
        candidate = f"{current}\n{unit}".strip() if current else unit
        if current and len(candidate) > max_chars:
            # 装不下就先结算当前块，把这句话作为新块的开头。
            chunks.append(current)
            current = unit
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks or [normalized]


class RagasUnavailableError(RuntimeError):
    """Ragas, its judge model, or its evaluation embedding is unavailable.

    译文：Ragas、judge 模型或评测用 embedding 不可用（未安装、未配置、初始化失败）。
    单独用这个异常类型，是为了让上层把「基础设施不可用」和「评测不达标」区分开：
    前者应报 error/不完整，后者才是 failed。
    """


@dataclass
class RagEvalCase:
    """一条评测用例。

    字段分三类：必需（问题/参考答案/期望来源）、语义约束（必含/禁含/正则）、
    以及可选行为约束（必须调用哪些工具、期望的工具调用序列）。
    大量字段用 field(default_factory=list) 而不是 []，避免可变默认值在实例间共享。
    """

    question: str
    reference: str
    expected_source_ids: list[str]
    source_type: str | None = None
    required_tools: list[str] = field(default_factory=list)
    expected_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    # kind 用来标记正例/负例（如 negative 表示应当拒答），默认 positive。
    kind: str = "positive"
    must_include: list[str] = field(default_factory=list)
    must_not_include: list[str] = field(default_factory=list)
    answer_regex: str | None = None
    source_file: str | None = None
    # id 由调用方决定（评测集里显式写了就用它），空串表示由运行器按序号生成。
    id: str = ""


class SemanticScorer(Protocol):
    """语义打分器的协议：只要实现 score 就能替换掉真实的 Ragas（测试里常用假实现）。

    用 Protocol 而不是抽象基类，是为了保持结构化子类型（鸭子类型），
    真实实现和测试替身都不需要显式继承。
    """

    version: str
    judge_model: str

    async def score(
        self,
        *,
        user_input: str,
        response: str,
        retrieved_contexts: list[str],
        reference: str,
        tool_calls: list[dict[str, Any]],
        tool_observations: list[dict[str, Any]],
        expected_tool_calls: list[dict[str, Any]],
        metric_names: set[str] | None = None,
    ) -> tuple[dict[str, float], dict[str, str]]: ...


# 进度回调：同步或异步都支持（见 _emit_progress 的 isawaitable 判断）。
ProgressCallback = Callable[[dict[str, Any]], Awaitable[None] | None]


async def _emit_progress(callback: ProgressCallback | None, **payload: Any) -> None:
    """安全地发一次进度事件。

    callback 为空直接返回（评测可以无人值守跑）；
    返回值可能是 awaitable 也可能不是，用 inspect.isawaitable 统一处理，
    这样调用方传同步函数（打印日志）或异步函数（写数据库）都能工作。
    """
    if callback is None:
        return
    pending = callback(payload)
    if inspect.isawaitable(pending):
        await pending


def _ensure_ragas_compatibility() -> None:
    """Bridge a Ragas 0.4.3 import removed by langchain-community 0.4.

    Ragas imports the legacy ChatVertexAI module only for an isinstance check. The
    project does not use VertexAI, so a scoped placeholder keeps the supported
    Collections API usable without downgrading LangChain/LangGraph.

    译文：Ragas 0.4.3 引用了 langchain-community 0.4 已删除的 ChatVertexAI 模块，
    这里做一层兼容桥接。Ragas 只是在做 isinstance 判断时用到它，本项目并不使用 VertexAI，
    所以塞一个空占位类就能让 Ragas Collections API 正常工作，而不必降级 LangChain/LangGraph。

    实现要点：只有在**这个模块**确实不存在时才注入占位（ModuleNotFoundError.name 校验），
    别的模块缺失要照常抛出——否则会掩盖真正的依赖问题。
    """

    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return
    try:
        with warnings.catch_warnings():
            # 屏蔽 langchain-community 的「即将停止维护」弃用警告，避免刷屏；
            # 只针对这一条消息，其它警告照常输出。
            warnings.filterwarnings(
                "ignore",
                message="`langchain-community` is being sunset.*",
                category=DeprecationWarning,
            )
            __import__(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        shim = types.ModuleType(module_name)
        shim.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules[module_name] = shim


class RagasCollectionsScorer:
    """Adapter from DevFlow's model clients to the Ragas 0.4 Collections API.

    译文：把 DevFlow 自己的模型客户端适配到 Ragas 0.4 的 Collections API。

    设计取舍：只装配 Ragas 的 metric 对象，不把项目的 prompt/Agent 逻辑塞进 Ragas，
    这样评测跑的是**生产链路**，Ragas 只当裁判。
    """

    def __init__(self) -> None:
        """初始化 judge 模型、embedding 和七个指标对象。

        任何一步失败都包装成 RagasUnavailableError（而不是裸异常）：
        上层据此把整轮评测标记为「judge 不可用」而不是让接口 500。
        """
        _ensure_ragas_compatibility()
        try:
            from openai import AsyncOpenAI
            from ragas.embeddings.base import BaseRagasEmbedding
            from ragas.llms import llm_factory
            from ragas.metrics.collections import (
                AgentGoalAccuracyWithReference,
                AnswerRelevancy,
                ContextPrecision,
                ContextRecall,
                Faithfulness,
                ToolCallAccuracy,
                ToolCallF1,
            )
        except Exception as exc:  # pragma: no cover - depends on optional packages
            # 延迟导入：ragas 是可选依赖，没装也不该影响服务启动，只有要跑评测时才报错。
            raise RagasUnavailableError(f"Unable to import Ragas 0.4 Collections API: {exc}") from exc

        class ProjectEmbedding(BaseRagasEmbedding):
            """把项目的 embed_text 包成 Ragas 需要的 embedding 接口。

            同步版本直接调用，异步版本用 to_thread 包装：
            embed_text 是阻塞的（可能走网络），直接在事件循环里跑会卡住其它请求。
            answer_relevancy 需要 embedding 来判断「答案与问题是否同义」。
            """

            def embed_text(self, text: str, **_: Any) -> list[float]:
                return embed_text(text)

            async def aembed_text(self, text: str, **_: Any) -> list[float]:
                return await asyncio.to_thread(embed_text, text)

        # judge 优先用专用配置，没配才回落到主模型：这样可以用更强的模型做裁判，
        # 同时不影响线上回答用的模型。
        api_key = settings.ragas_judge_api_key or settings.llm_api_key
        base_url = (settings.ragas_judge_base_url or settings.llm_base_url).rstrip("/")
        model = settings.ragas_judge_model or settings.llm_model
        if not api_key:
            raise RagasUnavailableError("RAGAS judge API key is not configured")
        if not base_url or not model:
            raise RagasUnavailableError("RAGAS judge base URL or model is not configured")

        try:
            client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=settings.ragas_timeout_seconds)
            llm = llm_factory(
                model,
                client=client,
                # temperature=0：评分要不稳定就没法做回归对比，必须尽量确定。
                temperature=0,
                max_tokens=settings.ragas_judge_max_tokens,
            )
            embedding = ProjectEmbedding()
            self.metrics = {
                "context_precision": ContextPrecision(llm=llm),
                "context_recall": ContextRecall(llm=llm),
                "faithfulness": Faithfulness(llm=llm),
                "answer_relevancy": AnswerRelevancy(
                    llm=llm,
                    embeddings=embedding,
                    # strictness 是生成候选问题的次数，至少 1 次；越大越稳但越慢越贵。
                    strictness=max(1, settings.ragas_answer_relevancy_strictness),
                ),
                "agent_goal_accuracy": AgentGoalAccuracyWithReference(llm=llm),
                # strict_order=False：工具调用顺序不作为扣分项，只看「调没调对」。
                "tool_call_accuracy": ToolCallAccuracy(strict_order=False),
                "tool_call_f1": ToolCallF1(),
            }
            self.message_types = self._message_types()
            # 记录 ragas 版本：不同版本的指标实现不同，历史结果必须能对上版本才有可比性。
            self.version = importlib.metadata.version("ragas")
            self.judge_model = model
        except Exception as exc:
            raise RagasUnavailableError(f"Unable to initialize Ragas judge: {exc}") from exc

    @staticmethod
    def _message_types() -> tuple[Any, Any, Any, Any]:
        """取 Ragas 的消息类型（延迟导入，避免顶层就依赖 ragas）。"""
        from ragas.messages import AIMessage, HumanMessage, ToolCall, ToolMessage

        return HumanMessage, AIMessage, ToolMessage, ToolCall

    def _messages(
        self,
        user_input: str,
        response: str,
        tool_calls: list[dict[str, Any]],
        tool_observations: list[dict[str, Any]],
    ) -> list[Any]:
        """把一次 Agent 运行还原成 Ragas 的对话消息序列。

        顺序必须是「用户提问 → AI 发起工具调用 → 工具返回 → AI 最终回答」，
        因为 agent_goal_accuracy / tool_call_* 都按消息序列判断轨迹是否合理。
        AI 的工具调用消息 content 留空：内容都在 tool_calls 里，写点废话反而干扰裁判。
        """
        HumanMessage, AIMessage, ToolMessage, ToolCall = self.message_types
        messages: list[Any] = [HumanMessage(content=user_input)]
        ragas_calls = [
            ToolCall(
                # 工具名缺失时给 "unknown"：宁可少判对一项，也不因缺字段直接报错。
                name=str(item.get("tool_name") or item.get("name") or "unknown"),
                args=dict(item.get("arguments") or item.get("args") or {}),
            )
            for item in tool_calls
            if isinstance(item, dict)
        ]
        if ragas_calls:
            messages.append(AIMessage(content="", tool_calls=ragas_calls))
        for item in tool_observations:
            if isinstance(item, dict):
                messages.append(ToolMessage(content=str(item.get("content") or "")))
        messages.append(AIMessage(content=response))
        return messages

    async def _score_faithfulness(
        self,
        *,
        user_input: str,
        response: str,
        retrieved_contexts: list[str],
    ) -> float:
        """分块计算 faithfulness，并按块长度加权平均。

        并发度限制为 2（Semaphore(2)）：judge 模型通常有速率限制，
        全量并发容易触发 429，反而整体更慢。
        每块都套 wait_for(timeout)：单块卡住不能让整轮评测挂起。
        任一块失败就整体失败（而不是按 0 分算）：用 0 分会把「judge 故障」
        误读成「模型胡编」，这会误导后续优化方向。
        """
        chunks = _split_faithfulness_response(response)
        semaphore = asyncio.Semaphore(2)

        async def score_chunk(chunk: str) -> Any:
            async with semaphore:
                return await asyncio.wait_for(
                    self.metrics["faithfulness"].ascore(
                        user_input=user_input,
                        response=chunk,
                        retrieved_contexts=retrieved_contexts,
                    ),
                    timeout=settings.ragas_timeout_seconds,
                )

        results = await asyncio.gather(*(score_chunk(chunk) for chunk in chunks), return_exceptions=True)
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            first = failures[0]
            raise RagasUnavailableError(
                f"Faithfulness chunk scoring failed ({len(failures)}/{len(chunks)}): "
                f"{type(first).__name__}: {first}"
            ) from first

        # 权重至少为 1，避免空块把分母拉成 0；按字符数加权 = 长块更可信。
        weights = [max(1, len(chunk)) for chunk in chunks]
        return sum(float(result.value) * weight for result, weight in zip(results, weights, strict=True)) / sum(weights)

    async def score(
        self,
        *,
        user_input: str,
        response: str,
        retrieved_contexts: list[str],
        reference: str,
        tool_calls: list[dict[str, Any]],
        tool_observations: list[dict[str, Any]],
        expected_tool_calls: list[dict[str, Any]],
        metric_names: set[str] | None = None,
    ) -> tuple[dict[str, float], dict[str, str]]:
        """并发跑所选指标，返回 (分数, 错误)。

        返回两个字典而不是抛异常：单个指标失败不该毁掉其余指标的结果，
        调用方会把错误逐个记在该用例上（重试时只重跑失败的那几个）。
        metric_names=None 表示「全都要」；传集合则只算其中的指标（重试场景用）。
        """
        messages = self._messages(user_input, response, tool_calls, tool_observations)
        # 闭包形式的「是否要算这个指标」判断，避免每处都写 metric_names is None or ...
        wants = lambda name: metric_names is None or name in metric_names
        calls: dict[str, Any] = {}
        if wants("context_precision"):
            calls["context_precision"] = self.metrics["context_precision"].ascore(
                user_input=user_input,
                reference=reference,
                retrieved_contexts=retrieved_contexts,
            )
        if wants("context_recall"):
            calls["context_recall"] = self.metrics["context_recall"].ascore(
                user_input=user_input,
                reference=reference,
                retrieved_contexts=retrieved_contexts,
            )
        if wants("faithfulness"):
            calls["faithfulness"] = self._score_faithfulness(
                user_input=user_input,
                response=response,
                retrieved_contexts=retrieved_contexts,
            )
        if wants("answer_relevancy"):
            # 注意这里不需要 contexts：它只比较「答案 ↔ 问题」的相关性。
            calls["answer_relevancy"] = self.metrics["answer_relevancy"].ascore(
                user_input=user_input,
                response=response,
            )
        if wants("agent_goal_accuracy"):
            # 这个指标吃的是完整消息序列，而不是纯文本，才能判断「任务是否被完成」。
            calls["agent_goal_accuracy"] = self.metrics["agent_goal_accuracy"].ascore(
                user_input=messages,
                reference=reference,
            )
        if expected_tool_calls and (wants("tool_call_accuracy") or wants("tool_call_f1")):
            # 只有用例声明了期望工具调用才计算这两项：
            # 没有参考答案时算出来的分数没有意义，会污染均值。
            _, _, _, ToolCall = self.message_types
            reference_calls = [
                ToolCall(
                    name=str(item.get("tool_name") or item.get("name") or "unknown"),
                    args=dict(item.get("arguments") or item.get("args") or {}),
                )
                for item in expected_tool_calls
            ]
            if wants("tool_call_accuracy"):
                calls["tool_call_accuracy"] = self.metrics["tool_call_accuracy"].ascore(
                    user_input=messages,
                    reference_tool_calls=reference_calls,
                )
            if wants("tool_call_f1"):
                calls["tool_call_f1"] = self.metrics["tool_call_f1"].ascore(
                    user_input=messages,
                    reference_tool_calls=reference_calls,
                )

        names = list(calls)
        # 并发执行 + return_exceptions=True：把异常当结果收回来，逐个记录而不是整体失败。
        results = await asyncio.gather(
            *(
                asyncio.wait_for(calls[name], timeout=settings.ragas_timeout_seconds)
                for name in names
            ),
            return_exceptions=True,
        )
        scores: dict[str, float] = {}
        errors: dict[str, str] = {}
        for name, result in zip(names, results, strict=True):
            if isinstance(result, BaseException):
                errors[name] = f"{type(result).__name__}: {result}"
                continue
            # 不同指标可能返回裸 float，也可能返回带 .value 的结果对象，两种都兼容。
            value = result if isinstance(result, (int, float)) else result.value
            scores[name] = round(float(value), 4)
        return scores, errors


def _default_cases(db: Session, repo_id: str, limit: int | None = None) -> list[RagEvalCase]:
    """没有指定评测集时，自动从仓库数据里凑一套用例（冒烟用）。

    优先级：文档索引 → issue → PR → 失败的 CI run。
    为什么是这个顺序：文档是「知识型问答」的最佳素材（有正文可当参考答案）；
    结构化数据（issue/PR/CI）是退路，用于刚接入、文档还没索引的仓库。
    顺带的好处是它天然覆盖了三类检索来源，能验证链路是否都通。
    """
    repo_uuid = uuid.UUID(str(repo_id))
    case_limit = limit or settings.ragas_max_cases
    # 只取 RAG 允许检索的来源类型，否则会造出「检索本来就查不到」的用例，白扣分。
    docs = (
        db.query(Document)
        .filter(Document.repo_id == repo_uuid, Document.source_type.in_(RAG_SOURCE_TYPES))
        .order_by(Document.created_at.desc())
        .limit(case_limit)
        .all()
    )
    cases: list[RagEvalCase] = []
    for index, doc in enumerate(docs, start=1):
        # 问句按来源类型定制，避免所有用例都用同一句式（那会让指标失去区分度）。
        if doc.source_type == "issue":
            question = f"What context exists for issue {doc.title}?"
        elif doc.source_type == "pull_request":
            question = f"What changed in pull request {doc.title}?"
        elif doc.source_type == "workflow_run":
            question = f"What does the CI run {doc.title} show?"
        else:
            question = f"What project knowledge is available about {doc.title}?"
        cases.append(
            RagEvalCase(
                id=f"document-{index}",
                question=question,
                # 参考答案截断到 2000 字符：judge 只需要能判断「答案是否覆盖要点」，
                # 整篇文档塞进去既贵又容易超出 judge 的输入上限。
                reference=doc.content[:2000],
                # 同时登记 document.id 和 source_id：检索层返回哪种 id 都算命中。
                expected_source_ids=[str(doc.id), str(doc.source_id)],
                source_type=doc.source_type,
            )
        )
    if cases:
        return cases

    # A freshly connected repository can have structured Issue/PR/CI data before
    # the reusable Document index is populated. Production ChatAgent prefetches
    # those records through EvidenceStore, so the default suite must cover them.
    # 译文：刚接入的仓库可能已经有 Issue/PR/CI 结构化数据，但可复用的文档索引还没建好。
    # 生产环境的 ChatAgent 会通过 EvidenceStore 预取这些记录，所以默认用例集也必须覆盖它们，
    # 否则「刚接入、还没索引文档」的仓库会得到 0 条用例（等于没评测）。
    issues = (
        db.query(Issue)
        .filter(Issue.repo_id == repo_uuid)
        .order_by(Issue.updated_at.desc(), Issue.created_at.desc())
        .limit(case_limit)
        .all()
    )
    for issue in issues:
        reference = "\n".join(
            value
            for value in [
                issue.title,
                issue.body or "",
                f"state: {issue.state}",
                f"labels: {', '.join(issue.labels or [])}",
            ]
            if value
        )
        cases.append(
            RagEvalCase(
                id=f"issue-{issue.number}",
                question=f"What project context is recorded for issue #{issue.number}: {issue.title}?",
                reference=reference[:2000],
                expected_source_ids=[str(issue.id)],
                source_type="issue",
            )
        )
        if len(cases) >= case_limit:
            return cases

    pull_requests = (
        db.query(PullRequest)
        .filter(PullRequest.repo_id == repo_uuid)
        .order_by(PullRequest.updated_at.desc(), PullRequest.created_at.desc())
        .limit(case_limit - len(cases))
        .all()
    )
    for pull_request in pull_requests:
        reference = "\n".join(
            value
            for value in [
                pull_request.title,
                pull_request.body or "",
                f"state: {pull_request.state}",
                f"base: {pull_request.base_branch}",
                f"head: {pull_request.head_branch}",
            ]
            if value
        )
        cases.append(
            RagEvalCase(
                id=f"pull-request-{pull_request.number}",
                question=f"What changed in pull request #{pull_request.number}: {pull_request.title}?",
                reference=reference[:2000],
                expected_source_ids=[str(pull_request.id)],
                source_type="pull_request",
            )
        )
        if len(cases) >= case_limit:
            return cases

    workflow_runs = (
        db.query(WorkflowRun)
        .filter(WorkflowRun.repo_id == repo_uuid, WorkflowRun.conclusion == "failure")
        .order_by(WorkflowRun.created_at.desc())
        .limit(case_limit - len(cases))
        .all()
    )
    for workflow_run in workflow_runs:
        reference = "\n".join(
            value
            for value in [
                workflow_run.name,
                workflow_run.status,
                workflow_run.conclusion or "",
                (workflow_run.logs_text or "")[:1600],
            ]
            if value
        )
        cases.append(
            RagEvalCase(
                id=f"workflow-run-{workflow_run.github_run_id}",
                question=f"What does the failed CI run {workflow_run.name} show?",
                reference=reference[:2000],
                expected_source_ids=[str(workflow_run.id)],
                source_type="workflow_run",
            )
        )
    return cases


def _source_hit(result: dict[str, Any], expected_source_ids: set[str]) -> bool:
    """判断一条检索结果是否命中期望来源。

    同时比对 id（文档主键）和 source_id（业务来源标识）：
    不同检索通道返回的 id 口径不同，只看一个会漏判。
    """
    return str(result.get("id")) in expected_source_ids or str(result.get("source_id")) in expected_source_ids


def _dcg(hits: list[int]) -> float:
    """折损累积增益（Discounted Cumulative Gain）。

    位置越靠后权重越低：分母 log2(index + 2)，即第 1 位权重 1、第 2 位约 0.63…
    用 +2 而不是 +1，是为了让第 1 位的分母为 log2(2)=1（标准定义）。
    """
    return sum(hit / math.log2(index + 2) for index, hit in enumerate(hits))


def _retrieval_metrics(cases: list[RagEvalCase], rows: list[dict[str, Any]], k: int) -> dict[str, float]:
    """计算检索层的四个经典指标（只看前 k 条结果）。

    - recall@k：期望来源里只要命中一个就算 1（二值化）。多期望来源时这是「宽松召回」，
      不要求全部命中，因为很多用例的 expected_source_ids 只是「相关来源之一」。
    - precision@k：命中数 / 实际返回数。用 max(len(results),1) 防除零。
    - MRR：第一个命中的排名倒数；没命中且没有期望来源时按 1.0 处理（无期望 = 无需召回）。
    - nDCG：理想排序是「命中在第 1 位」，用实际 DCG 除以理想 DCG 归一化。
    所有指标都是「用例级平均」，再四舍五入到 4 位，方便和历史结果比对。
    """
    if not cases:
        return {"recall_at_k": 0.0, "precision_at_k": 0.0, "mrr": 0.0, "ndcg": 0.0}
    recall_values: list[float] = []
    precision_values: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcg_values: list[float] = []
    # strict=True：cases 与 rows 必须一一对应，长度不一致说明上游漏跑/多跑了用例，应当立刻报错。
    for case, row in zip(cases, rows, strict=True):
        expected = set(case.expected_source_ids)
        results = row["results"][:k]
        hits = [1 if _source_hit(result, expected) else 0 for result in results]
        recall_values.append(1.0 if not expected or any(hits) else 0.0)
        precision_values.append(sum(hits) / max(len(results), 1))
        # next(...) 找第一个命中位置；+1 是因为名次从 1 开始。
        first_hit = next((index + 1 for index, hit in enumerate(hits) if hit), None)
        reciprocal_ranks.append(1.0 / first_hit if first_hit else (1.0 if not expected else 0.0))
        # 理想 DCG：只有第 1 位命中，其余都是 0。
        ideal = [1] + [0] * max(len(hits) - 1, 0)
        ndcg_values.append(_dcg(hits) / (_dcg(ideal) or 1.0) if expected else 1.0)
    return {
        "recall_at_k": round(sum(recall_values) / len(recall_values), 4),
        "precision_at_k": round(sum(precision_values) / len(precision_values), 4),
        "mrr": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4),
        "ndcg": round(sum(ndcg_values) / len(ndcg_values), 4),
    }


def _trace_rows(trace: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    """从 evaluation_trace 里抽出「检索到的上下文文本」和「来源元信息」。

    两者都要去重，但去重键不同：
    - contexts 按文本去重（同一段内容被多次检索到只保留一次）；
    - sources 按 (id, source_id, title) 三元组去重（同一来源可能有不同表述的标题）。
    去重很重要：重复上下文会让 faithfulness/context_precision 虚高。
    """
    contexts: list[str] = []
    sources: list[dict[str, Any]] = []
    seen_contexts: set[str] = set()
    seen_sources: set[tuple[str, str, str]] = set()
    for retrieval in trace.get("retrievals") or []:
        if not isinstance(retrieval, dict):
            continue
        for value in retrieval.get("contexts") or []:
            text = str(value).strip()
            if text and text not in seen_contexts:
                seen_contexts.add(text)
                contexts.append(text)
        for raw_source in retrieval.get("sources") or []:
            if not isinstance(raw_source, dict):
                continue
            # 复制一份再放进结果：避免后续修改污染原始 trace。
            source = dict(raw_source)
            key = (
                str(source.get("id") or ""),
                str(source.get("source_id") or ""),
                str(source.get("title") or ""),
            )
            if key not in seen_sources:
                seen_sources.add(key)
                sources.append(source)
    return contexts, sources


def _hard_rules(case: RagEvalCase, result: dict[str, Any], sources: list[dict[str, Any]], contexts: list[str]) -> dict[str, Any]:
    """执行确定性硬规则检查（不依赖 judge 模型）。

    全部条件同时满足才 passed：
    来源命中、必要工具都调过、工具没报错、有回答、有上下文、
    必含短语都在、禁含短语都不在、正则格式匹配。
    之所以要有这套规则：语义指标有波动且有成本，而「有没有调工具」「有没有引用」
    这类事实完全可以用代码判定，作为发布门禁更可靠。
    """
    expected = set(case.expected_source_ids)
    # 没有期望来源时视为天然命中（该用例不检查这一项）。
    source_hit = not expected or any(_source_hit(item, expected) for item in sources)
    actual_tools = [
        str(item.get("tool_name") or "")
        for item in result.get("tool_calls") or []
        if isinstance(item, dict)
    ]
    missing_tools = [name for name in case.required_tools if name not in actual_tools]
    tool_errors = [
        {
            "tool_name": item.get("tool_name"),
            "error": item.get("error"),
        }
        for item in result.get("tool_calls") or []
        if isinstance(item, dict) and item.get("status") == "error"
    ]
    answer = str(result.get("answer") or "").strip()
    answer_present = bool(answer)
    contexts_present = bool(contexts)
    # casefold 而不是 lower：对德文 ß 之类更正确，且对中英文结果一致。
    missing_phrases = [value for value in case.must_include if value.casefold() not in answer.casefold()]
    forbidden_phrases = [value for value in case.must_not_include if value.casefold() in answer.casefold()]
    # answer_regex 为 None 表示不校验格式；有值则要求**整段**匹配（fullmatch），
    # 这样「必须恰好三句话」这类约束才能表达。
    answer_pattern_matched = case.answer_regex is None or re.fullmatch(case.answer_regex, answer) is not None
    passed = (
        source_hit
        and not missing_tools
        and not tool_errors
        and answer_present
        and contexts_present
        and not missing_phrases
        and not forbidden_phrases
        and answer_pattern_matched
    )
    return {
        "passed": passed,
        "source_hit": source_hit,
        "answer_present": answer_present,
        "contexts_present": contexts_present,
        "missing_tools": missing_tools,
        "tool_errors": tool_errors,
        "missing_phrases": missing_phrases,
        "forbidden_phrases": forbidden_phrases,
        "answer_pattern_matched": answer_pattern_matched,
        "citation_count": len(result.get("citations") or []),
    }


def _evaluation_project_context(db: Session, repo_id: uuid.UUID) -> dict[str, Any]:
    """Load the same project-owned context that the frontend sends to ChatAgent.

    译文：加载与前端发给 ChatAgent 完全一致的项目上下文（团队成员 + 项目记忆）。

    为什么要在这里重放一遍：如果评测时不带这些上下文，Agent 的表现会和线上不一致，
    测出来的分数就没有参考价值。
    """

    documents = (
        db.query(Document)
        .filter(
            Document.repo_id == repo_id,
            Document.source_type.in_(("team_member", "memory_note")),
        )
        .order_by(Document.created_at.asc())
        .all()
    )
    team_members: list[dict[str, str]] = []
    project_memory: list[dict[str, str]] = []
    for document in documents:
        metadata = document.meta or {}
        if document.source_type == "team_member":
            team_members.append(
                {
                    # 老数据的标题形如「团队成员：张三」，去掉前缀当兜底名字。
                    "name": str(metadata.get("name") or document.title.replace("团队成员：", "")),
                    "role": str(metadata.get("role") or ""),
                    "strengths": str(metadata.get("strengths") or ""),
                    # 同时兼容 snake_case 和 camelCase 两种历史写法。
                    "techStack": str(metadata.get("tech_stack") or metadata.get("techStack") or ""),
                }
            )
        elif document.source_type == "memory_note":
            project_memory.append({"title": document.title, "content": document.content})
    return {"team_members": team_members, "project_memory": project_memory}


async def _run_chatagent_case(
    db: Session,
    repo: Repository,
    case: RagEvalCase,
    *,
    limit: int,
    agent_factory: Callable[[], ChatAgent],
) -> dict[str, Any]:
    """在**隔离的**会话里跑一条用例，返回 Agent 的原始输出。

    隔离手段是 SAVEPOINT（db.begin_nested）：Agent 运行过程会写消息、记忆、证据等，
    评测结束后统一 rollback 掉，既不污染真实数据，也不会留下垃圾会话。
    用 savepoint 而不是独立事务/独立库：能复用当前连接，且能精确回滚这一条用例的写入。

    agent_factory 以便于注入假 Agent（测试或对比实验）。
    """
    # Each trial gets a clean conversation and rolls back every runtime write.
    # 译文：每次试验都用全新会话，并回滚运行期的所有写入。
    savepoint = db.begin_nested()
    try:
        conversation = Conversation(
            repo_id=repo.id,
            title=f"[RAGAS Eval] {case.id or case.question[:40]}",
            # status=evaluation 让其它查询（会话列表等）默认看不到这条评测会话。
            status="evaluation",
        )
        db.add(conversation)
        # 先 flush 拿到 id：下面重建证据、组装上下文都要用到它。
        db.flush()
        EvidenceStore(db).rebuild_repo(repo, conversation)
        assembled = ContextAssembler(db).assemble(repo, conversation, case.question)
        # 延迟导入 API 路由里的工具构造函数：routes → services 已经存在依赖，
        # 顶层导入会形成循环。
        from app.api.routes.chat import _build_conversation_tools, _skill_runtime_context

        context: dict[str, Any] = {
            **_evaluation_project_context(db, repo.id),
            **_skill_runtime_context(),
            "repo_id": str(repo.id),
            "conversation_id": str(conversation.id),
            "server_context": assembled.system_context,
            "server_messages": assembled.history_messages,
            "evidence": assembled.evidence,
            "compression_stats": assembled.compression_stats,
            "source_type": case.source_type,
            "limit": limit,
            # evaluation_mode=True 会关闭一些「只该在生产环境做」的副作用（如真实推送通知）。
            "evaluation_mode": True,
        }
        context["tools"] = _build_conversation_tools(repo, conversation, db)
        return await agent_factory().run({"message": case.question}, context)
    finally:
        # 只有 savepoint 还活着才 rollback；被外部先提交/回滚过就跳过，避免二次回滚报错。
        if savepoint.is_active:
            savepoint.rollback()


def _thresholds() -> dict[str, float]:
    """从配置读各指标的通过门槛（集中一处，改门槛只改配置）。"""
    return {
        "context_precision": settings.ragas_context_precision_min,
        "context_recall": settings.ragas_context_recall_min,
        "faithfulness": settings.ragas_faithfulness_min,
        "answer_relevancy": settings.ragas_answer_relevancy_min,
        "agent_goal_accuracy": settings.ragas_agent_goal_accuracy_min,
        "tool_call_accuracy": settings.ragas_tool_call_accuracy_min,
        "tool_call_f1": settings.ragas_tool_call_f1_min,
    }


def _mean(values: list[float]) -> float | None:
    """求均值，先剔除 NaN/Inf。

    judge 偶尔会返回 nan，如果不过滤，一个 nan 就能把整个均值变成 nan，
    整轮评测就废了。全部无效时返回 None（表示「没测出来」而不是 0 分）。
    """
    finite = [value for value in values if math.isfinite(value)]
    return round(sum(finite) / len(finite), 4) if finite else None


def _metric_details(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """按指标聚合所有用例的分数，并判定是否达标。

    scored_cases 很关键：它告诉你这个分数是基于几条用例算出来的。
    例如 tool_call_* 只在部分用例上计算，scored_cases=0 意味着「这个指标根本没测」，
    不能当成 0 分来判失败（_finalize_result 里就是这么用的）。
    """
    thresholds = _thresholds()
    values: dict[str, list[float]] = {
        name: [] for name in (*CORE_RAGAS_METRICS, *AGENT_RAGAS_METRICS)
    }
    for row in rows:
        for name, value in (row.get("ragas") or {}).items():
            # 只收已知指标名且为有限数值的分数；未知指标（将来新增）不参与聚合。
            if name in values and isinstance(value, (int, float)) and math.isfinite(float(value)):
                values[name].append(float(value))
    details: dict[str, dict[str, Any]] = {}
    for name, metric_values in values.items():
        score = _mean(metric_values)
        details[name] = {
            "score": score,
            "threshold": thresholds[name],
            # 没分数时 passed 为 None（三态：通过/不通过/无法判定）。
            "passed": None if score is None else score >= thresholds[name],
            "scored_cases": len(metric_values),
        }
    return details


def _case_diagnoses(row: dict[str, Any]) -> list[dict[str, str]]:
    """把一条用例的各项得分翻译成「可执行的诊断结论」。

    每条诊断统一成 {category, severity, title, summary, action} 四要素，
    其中 action 是这份报告的核心价值：告诉使用者「分数低该去改哪一层」，
    而不是只甩一个 0.42 出来。
    category 也用于跨用例聚合，看是普遍性问题还是个别case的问题。
    """
    scores = row.get("ragas") or {}
    errors = row.get("ragas_errors") or {}
    hard = row.get("hard_rules") or {}
    thresholds = _thresholds()
    diagnoses: list[dict[str, str]] = []

    # judge 异常优先诊断：此时其他指标可能也是错的，先提示「重试评分，别急着调 RAG」。
    if errors:
        names = "、".join(errors)
        diagnoses.append(
            {
                "category": "judge_error",
                "severity": "error",
                "title": "Judge 指标计算异常",
                "summary": f"{names} 未得到可信分数；先仅重试评分，不要据此调整 RAG。",
                "action": "重试异常指标；若仍失败，再检查超时、模型结构化输出和答案长度。",
            }
        )
    if not hard.get("source_hit", True) or not hard.get("contexts_present", True):
        diagnoses.append(
            {
                "category": "retrieval",
                "severity": "failed",
                "title": "目标证据没有进入回答上下文",
                "summary": "预期来源未命中，或生产检索链没有返回可评分上下文。",
                "action": "检查切片、Query 改写、来源过滤、Top-K 和 Rerank。",
            }
        )
    rule_failures = (
        list(hard.get("missing_phrases") or [])
        + list(hard.get("forbidden_phrases") or [])
    )
    if rule_failures or hard.get("answer_pattern_matched") is False:
        diagnoses.append(
            {
                "category": "generation",
                "severity": "failed",
                "title": "回答违反确定性验收规则",
                "summary": "回答遗漏关键事实、包含禁用事实，或没有遵守要求的输出格式。",
                "action": "收紧回答指令，并审查 must_include、must_not_include 与格式规则。",
            }
        )

    precision = scores.get("context_precision")
    recall = scores.get("context_recall")
    faithfulness = scores.get("faithfulness")
    relevancy = scores.get("answer_relevancy")
    goal = scores.get("agent_goal_accuracy")
    # 区分「召回不全」和「召回全了但有噪声」：
    # 前者要改检索（切片/改写/Top-K），后者要改排序（rerank/来源过滤），处理方式完全不同。
    precision_low = isinstance(precision, (int, float)) and precision < thresholds["context_precision"]
    recall_low = isinstance(recall, (int, float)) and recall < thresholds["context_recall"]
    if precision_low and not recall_low:
        diagnoses.append(
            {
                "category": "retrieval_noise",
                "severity": "failed",
                "title": "证据找全了，但上下文噪声过多",
                "summary": "Context Recall 达标而 Context Precision 未达标。",
                "action": "比较 Top-K、Rerank 和来源状态特征，减少旧资料或同主题噪声。",
            }
        )
    elif recall_low:
        diagnoses.append(
            {
                "category": "retrieval_recall",
                "severity": "failed",
                "title": "关键证据召回不完整",
                "summary": "Context Recall 未达到当前评测门槛。",
                "action": "检查切片、Query 改写、多文档召回和来源过滤。",
            }
        )
    if (
        isinstance(faithfulness, (int, float))
        and faithfulness < thresholds["faithfulness"]
    ):
        diagnoses.append(
            {
                "category": "generation_faithfulness",
                "severity": "failed",
                "title": "回答没有忠于检索证据",
                "summary": "回答包含证据无法支持的扩写、推断或错误选择。",
                "action": "缩短答案，要求逐项依据证据回答，并加强冲突来源处理。",
            }
        )
    if isinstance(relevancy, (int, float)) and relevancy < thresholds["answer_relevancy"]:
        diagnoses.append(
            {
                "category": "generation_relevancy",
                "severity": "failed",
                "title": "回答没有直接回应问题",
                "summary": "答案相关性低于当前门槛，可能存在过度展开或遗漏用户追问。",
                "action": "改进问题拆解和回答结构，删除无关背景与下一步建议。",
            }
        )
    if isinstance(goal, (int, float)) and goal < thresholds["agent_goal_accuracy"]:
        diagnoses.append(
            {
                "category": "agent_goal",
                "severity": "failed",
                "title": "ChatAgent 没有完成题目要求",
                "summary": "回答可能事实相关，但没有遵守句数、格式或任务约束。",
                "action": "让生成链优先执行用户明确约束，再补充必要解释。",
            }
        )
    if not diagnoses:
        # 一条问题都没有时也要给一条「通过」诊断：
        # 前端按「有没有诊断项」渲染，空列表会显示成「未分析」而不是「没问题」。
        diagnoses.append(
            {
                "category": "passed",
                "severity": "passed",
                "title": "该用例未发现门禁问题",
                "summary": "硬规则和已得到的语义指标均符合当前门槛。",
                "action": "保留为回归基线。",
            }
        )
    return diagnoses


def _diagnostic_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总所有用例的诊断：会**顺带把 diagnoses 写回每行**（副作用）。

    failed_cases / error_cases 分别统计「有实质问题」和「judge 出错」的用例数：
    两者要分开看——error_cases 多说明评测环境有问题，failed_cases 多才是产品问题。
    category_counts 用于找出最主要的问题类型（哪种诊断出现最多，就先修哪个）。
    """
    counts: dict[str, int] = {}
    failed_cases = 0
    error_cases = 0
    for row in rows:
        # 这里有副作用：把诊断结果挂回该行，后面 _finalize_result 输出时直接可用。
        row["diagnoses"] = _case_diagnoses(row)
        # 排除 passed，只统计真正的问题类别。
        categories = {item["category"] for item in row["diagnoses"] if item["category"] != "passed"}
        if categories:
            failed_cases += 1
        if row.get("ragas_errors"):
            error_cases += 1
        for category in categories:
            counts[category] = counts.get(category, 0) + 1
    return {
        "failed_cases": failed_cases,
        "error_cases": error_cases,
        "category_counts": counts,
    }


def _finalize_result(
    *,
    rows: list[dict[str, Any]],
    retrieval: dict[str, float],
    performance: dict[str, Any],
    judge_requested: bool,
    scorer: SemanticScorer | None,
    judge_error: str | None,
    repo_id: str,
    limit: int,
    evaluation_set: dict[str, Any] | None,
    rescore_of: str | None = None,
) -> dict[str, Any]:
    """汇总逐用例结果，判定整体状态，产出最终评测报告。

    状态机（status / quality_status 双轨）：
    - 没有用例 → no_cases / not_evaluated（没跑，不等于通过）；
    - 硬规则不过 或 语义指标未达标 → failed；若此时 judge 还没跑完整，则报 error；
    - 未要求跑 judge → partial（只有硬规则和检索指标，结论不完整）；
    - 要求了但指标没算全 → error（是评测基础设施的问题，不是质量问题）；
    - 其余 → passed。
    刻意区分 error 和 failed：error 表示「这次评测不可信」，failed 表示「模型/检索确实不行」。
    混为一谈会导致把 judge 超时误判成质量退化。
    """
    metric_rows = _metric_details(rows)
    hard_passed_cases = sum(1 for row in rows if (row.get("hard_rules") or {}).get("passed"))
    hard_pass_rate = round(hard_passed_cases / len(rows), 4) if rows else 0.0
    case_metric_errors = sum(bool(row.get("ragas_errors")) for row in rows)
    # 必需指标必须全部有分数，才算「judge 真的跑起来了」。
    required_scored = all(metric_rows[name]["score"] is not None for name in REQUIRED_RAGAS_METRICS)
    ragas_available = bool(judge_requested and scorer is not None and required_scored)
    # complete 比 available 更严：不允许任何用例有指标错误。
    evaluation_complete = bool(ragas_available and case_metric_errors == 0)
    gated_names = list(REQUIRED_RAGAS_METRICS)
    # tool_call_* 只在真的算过（scored_cases > 0）时才纳入门禁：
    # 如果所有用例都没声明期望工具调用，这个指标是「不适用」，不该判失败。
    gated_names.extend(
        name
        for name in ("tool_call_accuracy", "tool_call_f1")
        if metric_rows[name]["scored_cases"] > 0
    )
    semantic_failed = any(metric_rows[name]["passed"] is False for name in gated_names)
    ragas_passed = evaluation_complete and not semantic_failed
    # 硬门禁要求**每一条**用例都过（不是平均过），因为安全/引用类规则不能有漏网。
    hard_passed = bool(rows) and hard_passed_cases == len(rows)
    if not rows:
        quality_status = "not_evaluated"
        status = "no_cases"
    elif not hard_passed or semantic_failed:
        quality_status = "failed"
        # 语义不过但 judge 本身没跑完时，报 error：此时无法确认是质量问题。
        status = "error" if judge_requested and not evaluation_complete else "failed"
    elif not judge_requested:
        quality_status = "partial"
        status = "partial"
    elif not evaluation_complete:
        quality_status = "incomplete"
        status = "error"
    else:
        quality_status = "passed"
        status = "passed"

    summary_metrics = {
        name: value["score"]
        for name, value in metric_rows.items()
        if value["score"] is not None
    }
    summary_metrics.update(
        {
            "hard_gate_pass_rate": hard_pass_rate,
            "retrieval_recall_at_k": retrieval.get("recall_at_k", 0.0),
            "retrieval_precision_at_k": retrieval.get("precision_at_k", 0.0),
        }
    )
    diagnostics = _diagnostic_summary(rows)
    # 顶层同时平铺 summary_metrics 和嵌套结构：前端老代码直接取顶层字段，
    # 新代码用嵌套结构，两套都保留以免破坏兼容。
    return {
        "kind": "chatagent_ragas",
        "status": status,
        # execution_status 只表示「有没有跑完」，与质量判定（quality_status）分开，
        # 避免「执行成功但质量不达标」被显示成执行失败。
        "execution_status": "completed",
        "quality_status": quality_status,
        "evaluation_complete": evaluation_complete,
        "passed": status == "passed",
        # rescore 模式表示「复用上次的 Agent 现场，只重算失败指标」。
        "run_mode": "rescore" if rescore_of else "full",
        "rescore_of": rescore_of,
        "target": {
            "name": "ChatAgent",
            "repo_id": str(repo_id),
            "execution": "stored_evidence_rescore" if rescore_of else "production_agent_with_isolated_conversation",
            "evaluation_mode": True,
        },
        # 没指定评测集时回落到「自动冒烟」，保证报告里总有可展示的来源信息。
        "evaluation_set": evaluation_set or {
            "id": "auto-document-smoke",
            "name": "auto-document-smoke",
            "label": "自动文档冒烟测试",
            "source": "auto_smoke",
            "sha256": None,
            "case_count": len(rows),
        },
        "pipeline": {"top_k": limit},
        **summary_metrics,
        "summary_metrics": summary_metrics,
        "retrieval": retrieval,
        "performance": performance,
        "hard_gate": {
            "passed": hard_passed,
            "pass_rate": hard_pass_rate,
            "passed_cases": hard_passed_cases,
            "total_cases": len(rows),
        },
        "ragas": {
            "requested": judge_requested,
            "available": ragas_available,
            "complete": evaluation_complete,
            # 评测不完整时 passed 给 None（三态），避免把「没测出来」显示成「没通过」。
            "passed": ragas_passed if evaluation_complete else None,
            "version": scorer.version if scorer is not None else None,
            "judge_model": scorer.judge_model if scorer is not None else None,
            "error": judge_error or (
                f"{case_metric_errors} evaluation case(s) contain Ragas metric errors"
                if case_metric_errors
                else None
            ),
            "case_errors": case_metric_errors,
            "metrics": metric_rows,
        },
        "diagnostics": diagnostics,
        "cases": rows,
    }


async def run_rag_quality_eval(
    db: Session,
    repo_id: str,
    *,
    cases: list[RagEvalCase] | None = None,
    limit: int = 5,
    run_judge: bool | None = None,
    scorer: SemanticScorer | None = None,
    agent_factory: Callable[[], ChatAgent] | None = None,
    progress_callback: ProgressCallback | None = None,
    evaluation_set: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate the production ChatAgent path with Ragas and deterministic gates.

    译文：用 Ragas + 确定性门禁来评估**生产 ChatAgent 链路**。

    参数说明：
    - cases 为空时自动从仓库数据生成用例（_default_cases），适合快速冒烟；
    - run_judge 为 None 时看配置开关，显式传值可临时关掉昂贵的语义评分；
    - scorer / agent_factory 可注入替身，便于测试和对比实验；
    - progress_callback 用于向前端推送进度（长任务，用户需要看到进展）。

    即使 judge 初始化失败也不抛异常：会带着 judge_error 继续跑硬规则和检索指标，
    这比整个接口 500 有用得多（至少能拿到「回答有没有引用」这类确定性结论）。
    """

    repo = db.get(Repository, uuid.UUID(str(repo_id)))
    if repo is None:
        raise ValueError(f"Repository {repo_id} was not found")
    # cases is not None 而不是 truthy：显式传空列表表示「确实没有用例」，应按空跑而不是回退到默认集。
    evaluation_cases = cases if cases is not None else _default_cases(db, repo_id)
    judge_requested = settings.ragas_enabled if run_judge is None else run_judge
    total_cases = len(evaluation_cases)
    await _emit_progress(
        progress_callback,
        stage="preparing",
        message=f"已准备 {total_cases} 条评测用例，正在初始化 Judge。",
        completed_cases=0,
        total_cases=total_cases,
    )
    judge_error: str | None = None
    if judge_requested and scorer is None:
        try:
            scorer = RagasCollectionsScorer()
        except RagasUnavailableError as exc:
            # 只记录不抛：降级成「无 judge」模式继续跑，最终状态会标成 incomplete/error。
            judge_error = str(exc)

    rows: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, case in enumerate(evaluation_cases, start=1):
        await _emit_progress(
            progress_callback,
            stage="chatagent",
            message=f"正在运行第 {index}/{total_cases} 条真实 ChatAgent 用例。",
            current_case=index,
            case_id=case.id or f"case-{index}",
            completed_cases=index - 1,
            total_cases=total_cases,
        )
        item_started = time.perf_counter()
        result = await _run_chatagent_case(
            db,
            repo,
            case,
            limit=limit,
            agent_factory=agent_factory or ChatAgent,
        )
        trace = result.get("evaluation_trace") if isinstance(result.get("evaluation_trace"), dict) else {}
        # 检索轨迹里既有给 judge 看的上下文文本，也有给指标算的来源 id，一次遍历全取出来。
        contexts, sources = _trace_rows(trace)
        hard_rules = _hard_rules(case, result, sources, contexts)
        semantic_scores: dict[str, float] | None = None
        semantic_errors: dict[str, str] = {}
        skipped_reason: str | None = None
        if not judge_requested:
            # 记录跳过原因而不是简单留空：报告里要能解释「为什么没有语义分数」。
            skipped_reason = "judge_disabled"
        elif scorer is None:
            skipped_reason = "judge_unavailable"
        elif not contexts:
            # 没有任何上下文时算 RAG 指标没有意义（分母为空），直接跳过并标注原因。
            skipped_reason = "no_retrieved_contexts"
        else:
            await _emit_progress(
                progress_callback,
                stage="ragas",
                message=f"ChatAgent 已完成第 {index}/{total_cases} 条，正在计算 RAGAS 指标。",
                current_case=index,
                case_id=case.id or f"case-{index}",
                completed_cases=index - 1,
                total_cases=total_cases,
            )
            semantic_scores, semantic_errors = await scorer.score(
                user_input=case.question,
                response=str(result.get("answer") or ""),
                retrieved_contexts=contexts,
                reference=case.reference,
                tool_calls=[dict(item) for item in result.get("tool_calls") or [] if isinstance(item, dict)],
                tool_observations=[
                    dict(item)
                    for item in trace.get("tool_observations") or []
                    if isinstance(item, dict)
                ],
                expected_tool_calls=case.expected_tool_calls,
            )

        latency_ms = round((time.perf_counter() - item_started) * 1000, 2)
        retrieval_rows.append({"results": sources})
        rows.append(
            {
                "id": case.id or f"case-{index}",
                "question": case.question,
                "reference": case.reference,
                "expected_source_ids": case.expected_source_ids,
                "source_file": case.source_file,
                "kind": case.kind,
                "must_include": case.must_include,
                "must_not_include": case.must_not_include,
                "answer_regex": case.answer_regex,
                "required_tools": case.required_tools,
                "expected_tool_calls": case.expected_tool_calls,
                "response": result.get("answer"),
                "route": result.get("route"),
                "citations": result.get("citations") or [],
                "tool_calls": result.get("tool_calls") or [],
                "tool_observations": trace.get("tool_observations") or [],
                "completion": result.get("completion") or {},
                "retrieved_contexts": contexts,
                "retrieved_sources": sources,
                "retrieval_trace": trace.get("retrievals") or [],
                "hard_rules": hard_rules,
                "ragas": semantic_scores,
                "ragas_errors": semantic_errors,
                "ragas_skipped_reason": skipped_reason,
                "latency_ms": latency_ms,
            }
        )
        await _emit_progress(
            progress_callback,
            stage="case_complete",
            message=f"第 {index}/{total_cases} 条评测已完成。",
            current_case=index,
            case_id=case.id or f"case-{index}",
            completed_cases=index,
            total_cases=total_cases,
        )

    await _emit_progress(
        progress_callback,
        stage="finalizing",
        message="正在汇总指标并生成最终判定。",
        completed_cases=total_cases,
        total_cases=total_cases,
    )
    # 检索指标单独算：它只依赖 sources 的顺序，不需要 judge，所以即使 judge 挂了也有值。
    retrieval = _retrieval_metrics(evaluation_cases, retrieval_rows, limit)
    latencies = [float(row["latency_ms"]) for row in rows]
    # 性能指标和准确率一起记录：只看分数不看延迟，容易选出「准确但慢到没法用」的配置。
    performance = {
        "cases": len(rows),
        "total_latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        "max_latency_ms": max(latencies) if latencies else 0.0,
    }
    return _finalize_result(
        rows=rows,
        retrieval=retrieval,
        performance=performance,
        judge_requested=judge_requested,
        scorer=scorer,
        judge_error=judge_error,
        repo_id=str(repo.id),
        limit=limit,
        evaluation_set=evaluation_set,
    )


async def rescore_rag_result(
    result: dict[str, Any],
    *,
    eval_id: str,
    scorer: SemanticScorer | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Retry missing or failed Judge metrics without running ChatAgent again.

    译文：只重试缺失或失败的 judge 指标，**不重新跑 ChatAgent**。

    这是省钱又省时的关键设计：Agent 的输出、检索上下文、工具调用都已经存在旧结果里，
    judge 超时/限流失败时完全没必要重跑整条链路（那会把同样的钱再花一遍，
    还可能因为模型随机性得到不一样的回答，导致前后对比失真）。
    重评分只合并新分数、清掉对应的错误项，已有分数原样保留。
    """

    # 深拷贝：不能修改调用方传入的历史结果（它可能已经落库/展示过）。
    rows = copy.deepcopy(result.get("cases") or [])
    if not rows:
        raise ValueError("原评测没有可用于重评分的逐题现场")
    retry_by_case: list[set[str]] = []
    for row in rows:
        scores = row.get("ragas") or {}
        errors = row.get("ragas_errors") or {}
        # 两类需要重试：明确报错的指标，以及必需但压根没有分数的指标（可能被跳过）。
        retry_names = {name for name in errors if name in (*CORE_RAGAS_METRICS, *AGENT_RAGAS_METRICS)}
        retry_names.update(name for name in REQUIRED_RAGAS_METRICS if name not in scores)
        expected_calls = row.get("expected_tool_calls") or []
        if expected_calls:
            # 只有该用例声明了期望工具调用，才补算工具类指标。
            retry_names.update(
                name for name in ("tool_call_accuracy", "tool_call_f1") if name not in scores
            )
        retry_by_case.append(retry_names)
    retry_count = sum(bool(names) for names in retry_by_case)
    if retry_count == 0:
        # 没有任何待重试项：明确报错，避免产出一次「什么都没做」的假重评分记录。
        raise ValueError("原评测没有失败或缺失的 Judge 指标")

    judge_error: str | None = None
    if scorer is None:
        try:
            scorer = RagasCollectionsScorer()
        except RagasUnavailableError as exc:
            # 同样降级处理：judge 起不来也要把「无法重试」写回到每条待重试项上。
            judge_error = str(exc)
    started = time.perf_counter()
    completed = 0
    for index, (row, retry_names) in enumerate(zip(rows, retry_by_case, strict=True), start=1):
        if not retry_names:
            # 这条用例本来就没有待重试项，跳过（不占用 retry_count 计数）。
            continue
        await _emit_progress(
            progress_callback,
            stage="rescore",
            message=f"正在重试第 {completed + 1}/{retry_count} 条异常评分。",
            current_case=index,
            case_id=str(row.get("id") or f"case-{index}"),
            completed_cases=completed,
            total_cases=retry_count,
        )
        existing_errors = dict(row.get("ragas_errors") or {})
        if scorer is None:
            # 逐个写错误而不是直接 return：前端能看到每条用例都失败了及其原因。
            for name in retry_names:
                existing_errors[name] = f"RagasUnavailableError: {judge_error or 'Judge unavailable'}"
            row["ragas_errors"] = existing_errors
            completed += 1
            continue
        contexts = [str(value) for value in row.get("retrieved_contexts") or [] if str(value).strip()]
        if not contexts:
            for name in retry_names:
                existing_errors[name] = "ValueError: no retrieved contexts"
            row["ragas_errors"] = existing_errors
            completed += 1
            continue
        scores, errors = await scorer.score(
            user_input=str(row.get("question") or ""),
            response=str(row.get("response") or ""),
            retrieved_contexts=contexts,
            reference=str(row.get("reference") or ""),
            tool_calls=[dict(item) for item in row.get("tool_calls") or [] if isinstance(item, dict)],
            tool_observations=[
                dict(item) for item in row.get("tool_observations") or [] if isinstance(item, dict)
            ],
            expected_tool_calls=[
                dict(item) for item in row.get("expected_tool_calls") or [] if isinstance(item, dict)
            ],
            metric_names=retry_names,
        )
        merged_scores = dict(row.get("ragas") or {})
        # 新分数覆盖同名旧分数，未重试的指标保持不变。
        merged_scores.update(scores)
        for name in retry_names:
            # 先清掉旧的错误项：这次算出来的结果（成功或新的失败）以最新一次为准。
            existing_errors.pop(name, None)
        existing_errors.update(errors)
        row["ragas"] = merged_scores
        row["ragas_errors"] = existing_errors
        # 重试成功后这条用例就不再是「被跳过」的状态了。
        row["ragas_skipped_reason"] = None
        completed += 1
        await _emit_progress(
            progress_callback,
            stage="case_complete",
            message=f"第 {completed}/{retry_count} 条异常评分已重试。",
            current_case=index,
            case_id=str(row.get("id") or f"case-{index}"),
            completed_cases=completed,
            total_cases=retry_count,
        )

    performance = dict(result.get("performance") or {})
    # 标记 agent_reused=True 并单独记录重评分耗时：
    # 让报告使用者知道这次的总耗时里不含 Agent 执行时间，别拿它和全量运行比性能。
    performance.update(
        {
            "agent_reused": True,
            "rescore_cases": retry_count,
            "rescore_latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }
    )
    # top_k 沿用原评测的配置（缺省 5），保证重评分后的检索指标口径与原来一致。
    top_k = int((result.get("pipeline") or {}).get("top_k") or 5)
    return _finalize_result(
        rows=rows,
        retrieval=dict(result.get("retrieval") or {}),
        performance=performance,
        judge_requested=True,
        scorer=scorer,
        judge_error=judge_error,
        repo_id=str((result.get("target") or {}).get("repo_id") or ""),
        limit=top_k,
        evaluation_set=dict(result.get("evaluation_set") or {}),
        rescore_of=eval_id,
    )
