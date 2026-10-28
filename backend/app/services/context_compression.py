"""上下文压缩：在有限上下文窗口里尽量保住「能继续干活」的信息。

整体分四层，代价从低到高、逐级兜底：
1. estimate_tokens / ContextPressure：不压缩，只**估算并预警**当前用量；
2. microcompact：把「可重新获取」的工具输出（读文件、搜索、抓网页）换成摘要+提示，
   这是性价比最高的一层——信息随时能重跑工具拿回来；
3. compress_messages / compress_retrieval_results：按预算裁剪，只留最相关的句子；
4. LLM 生成 compact summary（本模块只负责 prompt、兜底和保真校验）。
关键约束：压缩绝不能破坏「工具调用 → 工具结果」的配对，否则上游 API 会直接报错，
所以所有切片边界都要经过 adjust_keep_start_to_preserve_api_invariants 修正。
"""

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from app.core.config import settings


# 估算 token 的分词器，三条分支对应三类字符：
# - 单个汉字算 1 token（中文大致 1 字 1 token）；
# - 连续的字母数字下划线算一个「词」，按 4 字符≈1 token 折算；
# - 其余符号（标点、emoji 等）各算 1 token。
# 只是估算，不做精确计费，重点是「变化趋势正确」和「本地零依赖」。
TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]|[a-zA-Z0-9_\-]+|[^\s]")
# 强调「词」而非「符号」：用于相关性打分（句子命中几个查询词）。
WORD_PATTERN = re.compile(r"[a-zA-Z0-9_\-\u4e00-\u9fff]+")
# 句子切分：在句末标点/换行之后切开，用后顾断言保留标点本身。
# 同时覆盖中英文标点（。！？），因为项目里中英混排很常见。
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?\n\u3002\uff01\uff1f])\s+")

# 各模型的上下文窗口（输入+输出总和）。硬编码而不是从 API 查：
# 这个信息要在「还没来得及调用模型」时就知道（比如决定要不要压缩）。
MODEL_CONTEXT_WINDOWS = {
    "deepseek-v4-pro": 1_000_000,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4.1": 1_000_000,
    "gpt-4.1-mini": 1_000_000,
    "gpt-4.1-nano": 1_000_000,
    "gpt-5": 400_000,
    "gpt-5-mini": 400_000,
    "gpt-5-nano": 400_000,
    "o1": 200_000,
    "o1-mini": 128_000,
    "o3": 200_000,
    "o3-mini": 200_000,
    "o4-mini": 200_000,
    "claude-3-5-sonnet": 200_000,
    "claude-3-7-sonnet": 200_000,
    "claude-sonnet-4": 200_000,
    "claude-opus-4": 200_000,
}

# 各模型单次最多能输出多少 token。
# 用途：算「给回答预留多少空间」。预留不够会被截断，预留过多又会浪费可用输入。
MODEL_OUTPUT_WINDOWS = {
    "deepseek-v4-pro": 384_000,
    "gpt-4o": 16_384,
    "gpt-4o-mini": 16_384,
    "gpt-4.1": 32_768,
    "gpt-4.1-mini": 32_768,
    "gpt-4.1-nano": 32_768,
    "gpt-5": 128_000,
    "gpt-5-mini": 128_000,
    "gpt-5-nano": 128_000,
    "o1": 100_000,
    "o1-mini": 65_536,
    "o3": 100_000,
    "o3-mini": 100_000,
    "o4-mini": 100_000,
    "claude-3-5-sonnet": 8_192,
    "claude-3-7-sonnet": 64_000,
    "claude-sonnet-4": 64_000,
    "claude-opus-4": 64_000,
}

# 这些工具的输出「可以重跑拿到」，所以是最安全的压缩对象。
# 用子串匹配（见 is_regenerable_context_message）：工具名可能带命名空间前缀，
# 精确等值匹配很容易漏，子串匹配宁滥勿缺——反正压错了还能重新调用。
# 注意 read/grep/shell 这类裸名字是为了兼容 Claude Code 风格的工具名。
COMPACTABLE_TOOL_HINTS = (
    "workspace.read_file",
    "workspace.search_code",
    "workspace.list_files",
    "rag.search_similar_documents",
    "devflow_search_evidence",
    "repo_health.generate_summary",
    "ci_debug_agent.analyze_workflow_run",
    "read",
    "grep",
    "search",
    "shell",
    "web_fetch",
    "webfetch",
)

# 压缩分界标记：在消息流里插一条 system 消息作为「分界线」，
# 之后只读取分界线之后的消息（旧内容已经被摘要替代）。
COMPACT_BOUNDARY_SUBTYPE = "compact_boundary"
# microcompact 时替换掉的工具结果会留下这个标记，
# 一是让模型知道「这里原本有内容但被清理了」，二是便于统计/排查。
MICROCOMPACT_MARKER = "[Old regenerable tool/context result cleared by microcompact]"


# 压力等级：normal < warning < auto_compact < aggressive < manual_path（越往后越紧张）。
PressureStage = Literal["normal", "warning", "auto_compact", "aggressive", "manual_path"]


def estimate_tokens(text: str | None) -> int:
    """粗略估算文本的 token 数（中文按字、英文按 4 字符≈1 token）。

    只用于预算判断，不追求与真实分词器一致；
    但同一套算法在「估算用量」和「裁剪预算」两边都用，所以两者的相对关系是自洽的。
    """
    if not text:
        return 0
    total = 0
    for token in TOKEN_PATTERN.findall(text):
        if re.fullmatch(r"[\u4e00-\u9fff]", token):
            total += 1
        elif re.fullmatch(r"[a-zA-Z0-9_\-]+", token):
            # (len + 3) // 4 等价于向上取整的 len/4，且至少算 1；
            # 至少 1 是为了避免超短词（"a"）被算成 0 导致预算失控。
            total += max(1, (len(token) + 3) // 4)
        else:
            total += 1
    return total


def is_context_overflow_error(exc: Exception) -> bool:
    """Recognize provider-specific prompt/context overflow failures.

    OpenAI-compatible gateways do not expose one stable exception type, so we
    deliberately use a narrow set of message markers and only allow callers to
    retry once.

    译文：识别各家「提示词/上下文超长」的错误。兼容 OpenAI 协议的各家网关
    并没有统一的异常类型，所以这里刻意只用一小组**消息特征串**判断，
    并且（调用方约定）只允许重试一次，避免在真正的逻辑错误上无限重试。
    标记刻意取得比较窄：宁可漏判（走正常失败流程），也不要误判成超长而反复重试。
    """
    message = str(exc or "").lower()
    markers = (
        "context_length_exceeded",
        "maximum context length",
        "prompt is too long",
        "prompt too long",
        "context window",
        "too many tokens",
        "request too large",
    )
    return any(marker in message for marker in markers)


def _normalize_model_name(model: str | None) -> str:
    """把模型名归一成表里的键。

    做三件事：没传就用配置里的默认模型；转小写；
    去掉 "provider/" 前缀和结尾的日期版本号（如 gpt-4o-2024-08-06 → gpt-4o），
    否则同一个模型的日期版本会被当成未知模型。
    """
    value = str(model or settings.llm_model or "").strip().lower()
    if not value:
        return "unknown"
    value = value.split("/")[-1]
    value = re.sub(r"-(20\d{2}[-_]\d{2}[-_]\d{2}|\d{4}[-_]\d{2}[-_]\d{2})$", "", value)
    return value


def _model_lookup(model: str | None, table: dict[str, int]) -> int | None:
    """在表里查模型对应的窗口大小：先精确匹配，再按前缀匹配。

    前缀匹配是为了覆盖 "-turbo"、"-0613" 之类的变体；
    用 startswith 而不是 contains，避免 "gpt-4" 误匹配到 "gpt-4o-mini" 之类。
    """
    normalized = _normalize_model_name(model)
    if normalized in table:
        return table[normalized]
    for key, value in table.items():
        if normalized.startswith(key):
            return value
    return None


def model_context_window_tokens(model: str | None = None) -> int:
    """取模型的上下文窗口大小（token）。

    优先级：配置显式指定 > 内置表 > 按「最大输入 + 预留输出」估算。
    所有返回值都不低于 4096：窗口配得再小也要能装下一轮对话，否则压缩逻辑没有腾挪空间。
    """
    if settings.context_model_window_tokens:
        return max(4096, settings.context_model_window_tokens)
    known = _model_lookup(model, MODEL_CONTEXT_WINDOWS)
    if known:
        return known
    return max(4096, settings.context_max_input_tokens + settings.context_reserved_response_tokens)


def model_output_window_tokens(model: str | None = None) -> int:
    """取模型单次最大输出 token 数；未知模型退化为配置里的预留值（至少 512）。"""
    known = _model_lookup(model, MODEL_OUTPUT_WINDOWS)
    if known:
        return known
    return max(512, settings.context_reserved_response_tokens)


def effective_context_window_tokens(model: str | None = None, *, for_compaction: bool = False) -> int:
    """计算「实际可用于输入」的窗口大小（窗口 - 给输出预留的空间）。

    两个容易搞错的点：
    - 预留值不能超过窗口的一半（model_window // 2），否则小窗口模型会被预留吃光；
    - for_compaction=True 时用另一套预留（compact_reserved_output_tokens）：
      压缩阶段只需要模型输出一份摘要，不需要按完整回答预留那么多。
    context_dynamic_model_window_enabled 打开时取「配置上限与动态窗口的较大者」，
    即允许大模型用满自己的窗口；关闭时取较小者，即配置上限是硬约束。
    """
    model_window = model_context_window_tokens(model)
    if for_compaction:
        reserved = min(model_output_window_tokens(model), max(512, settings.context_compact_reserved_output_tokens))
    else:
        reserved = max(512, min(settings.context_reserved_response_tokens, model_window // 2))
    dynamic_window = max(4096, model_window - reserved)
    if settings.context_dynamic_model_window_enabled:
        return max(settings.context_max_input_tokens, dynamic_window)
    return max(4096, min(settings.context_max_input_tokens, dynamic_window))


@dataclass(frozen=True)
class ContextPressure:
    """某一时刻的上下文压力快照，供调度层决定「现在要不要压缩」。

    同时带上了各档阈值，前端可以直接画进度条刻度，不用重复计算。
    """

    stage: PressureStage
    token_usage: int
    effective_window_tokens: int
    remaining_tokens: int
    warning_threshold: int
    auto_compact_threshold: int
    aggressive_threshold: int
    manual_threshold: int
    utilization: float

    @property
    def should_warn(self) -> bool:
        """是否该给用户提示（任何非 normal 状态都提示）。"""
        return self.stage != "normal"

    @property
    def should_auto_compact(self) -> bool:
        """是否该自动压缩：从 auto_compact 起，含更紧张的两档。"""
        return self.stage in {"auto_compact", "aggressive", "manual_path"}

    def as_dict(self) -> dict[str, Any]:
        """序列化成接口输出（字段名保持 snake_case，与其它评估接口一致）。"""
        return {
            "stage": self.stage,
            "token_usage": self.token_usage,
            "effective_window_tokens": self.effective_window_tokens,
            "remaining_tokens": self.remaining_tokens,
            "warning_threshold": self.warning_threshold,
            "auto_compact_threshold": self.auto_compact_threshold,
            "aggressive_threshold": self.aggressive_threshold,
            "manual_threshold": self.manual_threshold,
            "utilization": self.utilization,
        }


def _thresholds_for_window(window: int) -> tuple[int, int, int, int]:
    """按窗口大小算出四个阈值（warning / auto / aggressive / manual）。

    每档的「缓冲区」都是 min(配置值, 窗口的某个比例)：
    配置绝对值适合大窗口，但小窗口模型（如 8k）会被这些绝对值吃光，
    所以再用比例兜一层；每档比例依次收紧（1/4 → 1/5 → 1/8 → 1/16），
    表示越接近上限、留给我们的操作空间越小。
    各阈值之间用 ±1 强制严格递增，保证计算出的 stage 是唯一且有序的
    （否则小窗口下算出来的阈值可能重叠，导致阶段跳变）。
    """
    warning_buffer = min(settings.context_warning_buffer_tokens, max(512, window // 4))
    auto_buffer = min(settings.context_auto_compact_buffer_tokens, max(512, window // 5))
    aggressive_buffer = min(settings.context_aggressive_buffer_tokens, max(512, window // 8))
    manual_buffer = min(settings.context_manual_buffer_tokens, max(256, window // 16))
    auto_threshold = max(1024, window - auto_buffer)
    warning_threshold = max(512, min(auto_threshold - 1, auto_threshold - warning_buffer))
    aggressive_threshold = max(auto_threshold + 1, window - aggressive_buffer)
    manual_threshold = max(aggressive_threshold + 1, window - manual_buffer)
    return warning_threshold, auto_threshold, aggressive_threshold, manual_threshold


def calculate_context_pressure(token_usage: int, model: str | None = None) -> ContextPressure:
    """根据当前 token 用量判定压力等级。

    用 for_compaction=True 的有效窗口：这里问的是「还能不能继续往上下文里塞」，
    而不是「模型最多能输出多少」，所以按压缩口径预留更合理。
    判断从最高档往下比（>= manual 最先命中），保证返回的是最严重的那个等级。
    """
    effective_window = effective_context_window_tokens(model, for_compaction=True)
    warning_threshold, auto_threshold, aggressive_threshold, manual_threshold = _thresholds_for_window(effective_window)
    if token_usage >= manual_threshold:
        stage: PressureStage = "manual_path"
    elif token_usage >= aggressive_threshold:
        stage = "aggressive"
    elif token_usage >= auto_threshold:
        stage = "auto_compact"
    elif token_usage >= warning_threshold:
        stage = "warning"
    else:
        stage = "normal"
    return ContextPressure(
        stage=stage,
        token_usage=token_usage,
        effective_window_tokens=effective_window,
        # 用 max(0, ...) 兜底：超限时剩余量显示 0 而不是负数。
        remaining_tokens=max(0, effective_window - token_usage),
        warning_threshold=warning_threshold,
        auto_compact_threshold=auto_threshold,
        aggressive_threshold=aggressive_threshold,
        manual_threshold=manual_threshold,
        utilization=round(token_usage / effective_window, 4) if effective_window else 0.0,
    )


def is_compact_boundary_message(message: dict[str, Any]) -> bool:
    """判断一条消息是不是「压缩分界线」。

    约定：role=system 且 meta.subtype == "compact_boundary"。
    用 subtype 而不是靠内容特征识别，是为了让标记显式、可靠、可被前端过滤掉。
    """
    meta = message.get("meta") or message.get("metadata") or {}
    return str(message.get("role") or "") == "system" and meta.get("subtype") == COMPACT_BOUNDARY_SUBTYPE


def find_last_compact_boundary_index(messages: list[dict[str, Any]]) -> int:
    """从后往前找最后一条分界线，返回下标；没有返回 -1。

    必须找「最后一条」：一次会话可能压缩过多次，只有最新的分界线之后的才是当前有效上下文。
    """
    for index in range(len(messages) - 1, -1, -1):
        if is_compact_boundary_message(messages[index]):
            return index
    return -1


def messages_after_compact_boundary(messages: list[dict[str, Any]], *, include_boundary: bool = True) -> list[dict[str, Any]]:
    """只保留最后一条分界线之后的消息。

    没有分界线时原样返回（未压缩过的会话）。
    include_boundary=True 时保留分界线本身：它往往带着摘要正文，是后续对话的起点。
    """
    boundary_index = find_last_compact_boundary_index(messages)
    if boundary_index < 0:
        return messages
    start = boundary_index if include_boundary else boundary_index + 1
    return messages[start:]


def _message_meta(message: dict[str, Any]) -> dict[str, Any]:
    """安全取消息的 meta（兼容 meta/metadata 两种键，非 dict 一律当空）。

    历史数据和不同构造方写法不统一，这里统一归一，避免上层到处写 isinstance 判断。
    """
    meta = message.get("meta") or message.get("metadata") or {}
    return meta if isinstance(meta, dict) else {}


def _message_provider_id(message: dict[str, Any]) -> str | None:
    """取该消息在模型供应商侧的 id（用于识别「同一条 assistant 消息被拆成多条」）。

    依次尝试顶层和 meta 里的两种命名，取到第一个非空值。
    """
    meta = _message_meta(message)
    value = (
        message.get("message_id")
        or message.get("provider_message_id")
        or meta.get("message_id")
        or meta.get("provider_message_id")
    )
    return str(value) if value else None


def _content_blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    """取出多模态/工具调用格式的 content 块列表（content 不是数组时返回空）。

    过滤掉非 dict 的块：有些实现会在数组里混入纯字符串，直接 .get 会抛异常。
    """
    content = message.get("content")
    if isinstance(content, list):
        return [block for block in content if isinstance(block, dict)]
    return []


def _tool_use_ids(message: dict[str, Any]) -> set[str]:
    """收集一条消息里「发起工具调用」的 id。

    覆盖三种格式：OpenAI 风格的 tool_calls、Anthropic 风格的 tool_use content 块、
    以及放在 meta 里的 id（仅在 assistant/tool_use 角色下才信 meta 里的值，
    否则普通消息里的同名字段会被误当成工具调用）。
    """
    ids: set[str] = set()
    for call in message.get("tool_calls") or []:
        if isinstance(call, dict):
            value = (
                call.get("id")
                or call.get("tool_call_id")
                or call.get("tool_use_id")
                or call.get("call_id")
            )
            if value:
                ids.add(str(value))
    for block in _content_blocks(message):
        if block.get("type") == "tool_use" and block.get("id"):
            ids.add(str(block["id"]))
    meta = _message_meta(message)
    for key in ("tool_call_id", "tool_use_id"):
        if meta.get(key) and str(message.get("role") or "") in {"assistant", "tool_use"}:
            ids.add(str(meta[key]))
    return ids


def _tool_result_ids(message: dict[str, Any]) -> set[str]:
    """收集一条消息里「工具返回结果」对应的调用 id。

    与 _tool_use_ids 对称，但角色集合不同（tool/tool_result/user：
    Anthropic 把工具结果放在 user 消息的 tool_result 块里）。
    """
    ids: set[str] = set()
    role = str(message.get("role") or "")
    for key in ("tool_call_id", "tool_use_id"):
        value = message.get(key)
        if value:
            ids.add(str(value))
    for block in _content_blocks(message):
        if block.get("type") == "tool_result" and block.get("tool_use_id"):
            ids.add(str(block["tool_use_id"]))
    meta = _message_meta(message)
    for key in ("tool_call_id", "tool_use_id"):
        if meta.get(key) and role in {"tool", "tool_result", "user"}:
            ids.add(str(meta[key]))
    return ids


def adjust_keep_start_to_preserve_api_invariants(messages: list[dict[str, Any]], start_index: int) -> int:
    """Move a suffix boundary left when slicing there would orphan tool results.

    译文：当「从这里往后切」会导致工具结果失去对应的工具调用时，把切分点往前挪。

    这是整个压缩流程里最关键的一处修正。两种会被 API 拒绝的坏情况：
    1. 保留了 tool_result 却把对应的 tool_use 切掉了 → 往前找到那条发起调用的消息；
    2. 同一条 assistant 消息被拆到两段（provider 会在多条消息里复用同一个 id 表示
       「这是一次输出被拆开」）→ 继续往前合并，直到不再连续为止。
    返回新的起点；无法修正（比如起点本身就是 0）时按原样返回或收敛到合法范围。
    """
    if start_index <= 0 or start_index >= len(messages):
        return max(0, min(start_index, len(messages)))

    adjusted = start_index
    kept_results: set[str] = set()
    kept_uses: set[str] = set()
    # 先统计「切点之后」实际保留了什么，才知道缺哪些 use。
    for message in messages[adjusted:]:
        kept_results.update(_tool_result_ids(message))
        kept_uses.update(_tool_use_ids(message))

    # 差集 = 有结果但没调用的 id，必须逐个往前找回它们的 tool_use。
    missing_uses = kept_results - kept_uses
    if missing_uses:
        for index in range(adjusted - 1, -1, -1):
            uses = _tool_use_ids(messages[index])
            if uses & missing_uses:
                adjusted = index
                # 用 -= 而不是 remove：一条消息可能同时提供多个缺失的 use。
                missing_uses -= uses
                if not missing_uses:
                    break

    # 收集切点之后所有 assistant 消息的 provider id，用于判断「上一条是否属于同一次输出」。
    kept_provider_ids = {
        provider_id
        for message in messages[adjusted:]
        if str(message.get("role") or "") == "assistant"
        for provider_id in [_message_provider_id(message)]
        if provider_id
    }
    while adjusted > 0:
        previous = messages[adjusted - 1]
        if str(previous.get("role") or "") != "assistant":
            break
        previous_provider_id = _message_provider_id(previous)
        # 只有「id 相同」才继续前移；没有 id 就无法确认关联，宁可停在这里。
        if not previous_provider_id or previous_provider_id not in kept_provider_ids:
            break
        adjusted -= 1

    return adjusted


def _words(text: str) -> list[str]:
    """切成小写词序列（用于相关性打分，保留中文连续段）。"""
    return WORD_PATTERN.findall((text or "").lower())


def _unique_preserve_order(items: list[str], limit: int = 12) -> list[str]:
    """去重（大小写不敏感）并保持原顺序，最多返回 limit 条。

    用「小写后的值」做去重键但保留原始大小写输出：
    "README" 和 "readme" 视为重复，但展示时保留第一次出现的写法。
    顺便压缩内部空白，避免多行/多空格串进最终上下文。
    """
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        clean = " ".join(str(item or "").strip().split())
        if not clean:
            continue
        key = clean.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(clean)
        if len(output) >= limit:
            break
    return output


def clip_to_token_budget(text: str, max_tokens: int, suffix: str = "...") -> str:
    """按 token 预算截断文本，保证「截断后 + 后缀」仍不超预算。

    实现是「先按 4 字符≈1 token 粗切，再逐次回退 24 字符直到符合预算」：
    直接按 token 精确切需要重新分词，成本高；这种逼近法最多循环十几次，足够快。
    预算 <= 0 或空文本返回空串（表示「一点空间都没有」）。
    """
    clean = str(text or "").strip()
    if max_tokens <= 0 or not clean:
        return ""
    if estimate_tokens(clean) <= max_tokens:
        return clean
    # 至少切 32 字符：太小会导致几乎每次都要多轮回退，反而更慢。
    approx_chars = max(32, max_tokens * 4)
    clipped = clean[:approx_chars].rstrip()
    while clipped and estimate_tokens(clipped + suffix) > max_tokens:
        clipped = clipped[: max(0, len(clipped) - 24)].rstrip()
    # 全部退光就返回空串，而不是只返回一个后缀。
    return f"{clipped}{suffix}" if clipped else ""


@dataclass(frozen=True)
class ContextBudget:
    """各部分（系统提示/记忆/证据/近期对话/工具观察）的 token 配额。

    分项预算的作用是防止某一部分（通常是工具输出）把额度吃光，
    导致系统提示或用户最新问题被挤掉——那是最影响回答质量的。
    """

    max_input_tokens: int
    reserved_response_tokens: int
    system_tokens: int
    memory_tokens: int
    evidence_tokens: int
    recent_tokens: int
    tool_observation_tokens: int

    @property
    def available_prompt_tokens(self) -> int:
        """可用于提示词的总量（扣掉预留的输出空间），至少 512。"""
        return max(512, self.max_input_tokens - self.reserved_response_tokens)


def default_context_budget(model: str | None = None) -> ContextBudget:
    """按模型和配置算出各部分的默认预算。

    分配方式是「可用量 × 各比例」，每项再夹一个 512 下限：
    比例配得再小也要保证每部分有点空间，否则该部分内容会被整体丢弃。
    """
    if settings.context_dynamic_model_window_enabled:
        max_input = max(settings.context_max_input_tokens, model_context_window_tokens(model))
    else:
        max_input = max(4096, settings.context_max_input_tokens)
    reserved = max(512, min(settings.context_reserved_response_tokens, max_input // 2))
    available = max_input - reserved
    return ContextBudget(
        max_input_tokens=max_input,
        reserved_response_tokens=reserved,
        system_tokens=max(512, int(available * settings.context_system_ratio)),
        memory_tokens=max(512, int(available * settings.context_memory_ratio)),
        evidence_tokens=max(512, int(available * settings.context_evidence_ratio)),
        recent_tokens=max(512, int(available * settings.context_recent_ratio)),
        tool_observation_tokens=max(512, int(available * settings.context_tool_ratio)),
    )


def compress_messages(messages: list[dict[str, Any]], max_tokens: int) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """保留最近若干条对话消息，把它们塞进 max_tokens 预算内。

    策略要点：
    - **从最新往旧保留**（reversed）：越近的消息越重要，旧消息先被丢掉；
    - 只保留 user/assistant 且非空的消息，工具消息/系统消息在这里不参与；
    - 单条消息也设了上限（max_tokens//3，夹在 96..900），防止一条超长回答独占预算；
    - 第一条消息即使超预算也会保留（`kept_reversed and ...`）：至少要有内容，不能返回空。
    返回值第二项是统计信息，用于评估压缩效果（truncated/exact_suffix 等）。
    """
    kept_reversed: list[dict[str, str]] = []
    used = 0
    truncated_messages = 0
    exact_suffix_messages = 0
    exact_suffix_open = True
    for message in reversed(messages):
        role = str(message.get("role") or "")
        content = str(message.get("content") or "")
        if role not in {"user", "assistant"} or not content.strip():
            continue
        # 单条预算随总预算缩放，但夹在 96..900：太小会切碎语义，太大会挤掉其它消息。
        per_message_budget = max(96, min(900, max_tokens // 3))
        clipped = clip_to_token_budget(content, per_message_budget)
        cost = estimate_tokens(role) + estimate_tokens(clipped)
        if kept_reversed and used + cost > max_tokens:
            break
        kept_reversed.append({"role": role, "content": clipped})
        if clipped == content.strip():
            # exact_suffix 统计「尾部连续完整保留了几条」：
            # 一旦遇到需要截断的消息就关闭，用来判断尾部上下文是否完整未损。
            if exact_suffix_open:
                exact_suffix_messages += 1
        else:
            truncated_messages += 1
            exact_suffix_open = False
        used += cost
        if used >= max_tokens:
            break
    kept = list(reversed(kept_reversed))
    total_eligible_messages = sum(
        1
        for message in messages
        if str(message.get("role") or "") in {"user", "assistant"}
        and str(message.get("content") or "").strip()
    )
    return kept, {
        "input_messages": len(messages),
        "eligible_messages": total_eligible_messages,
        "kept_messages": len(kept),
        # omitted 用「合格消息数 - 保留数」算，而不是直接数被 break 掉的，
        # 这样包含「因角色/空内容被跳过」的消息，口径更贴近用户直觉。
        "omitted_messages": max(0, total_eligible_messages - len(kept)),
        "truncated_messages": truncated_messages,
        "exact_suffix_messages": exact_suffix_messages,
        "estimated_tokens": used,
        "budget_tokens": max_tokens,
    }


def _tool_names(message: dict[str, Any]) -> list[str]:
    """取一条消息涉及的工具名：先加路由名，再加每个 tool_call 的工具名。

    路由名也放进去，是因为有些工具结果消息只带 route 不带 tool_calls，
    靠 route 才能判断它是否属于「可重跑」的那类工具。
    """
    names: list[str] = []
    route = str(message.get("route") or "").strip()
    if route:
        names.append(route)
    for call in message.get("tool_calls") or []:
        if isinstance(call, dict):
            name = str(call.get("tool_name") or "").strip()
            if name:
                names.append(name)
    return names


def is_regenerable_context_message(message: dict[str, Any]) -> bool:
    """判断这条消息是否是「可以清掉、需要时重跑工具就能拿回来」的工具输出。

    三重条件：
    - 角色是工具结果（或 meta 标了 tool_result）；
    - 没有显式标记 regenerable=False（有些工具结果本身是结论，不能丢）；
    - 工具名命中 COMPACTABLE_TOOL_HINTS。
    任一不满足就保守地保留内容——压缩错了的代价比多占点 token 大得多。
    """
    role = str(message.get("role") or "")
    meta = _message_meta(message)
    if role not in {"tool", "tool_result"} and meta.get("subtype") != "tool_result":
        return False
    if meta.get("regenerable") is False:
        return False
    haystack = " ".join(_tool_names(message)).lower()
    if not haystack:
        return False
    return any(hint in haystack for hint in COMPACTABLE_TOOL_HINTS)


def microcompact_messages(
    messages: list[dict[str, Any]],
    query: str,
    *,
    keep_recent: int | None = None,
    max_excerpt_tokens: int = 90,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """把「旧的可重跑工具输出」就地替换成「标记 + 摘要 + 恢复提示」。

    只处理最近 keep_recent 条之前的部分（更近的输出可能还要用），
    并且要求原文足够长（> max_excerpt_tokens * 2），否则压缩收益还不如占位标记本身。
    与 compress_messages 不同，这个函数**不删消息**，只改 content：
    消息条数和工具调用配对关系完全不变，所以不会触发 API 的配对校验问题。
    """
    keep = keep_recent if keep_recent is not None else settings.context_microcompact_keep_recent_messages
    # 先算出「从哪条开始算近期」，再用 API 不变量修正，避免切在工具调用中间。
    keep_start = adjust_keep_start_to_preserve_api_invariants(
        messages,
        max(0, len(messages) - max(0, keep)),
    )
    output: list[dict[str, Any]] = []
    compacted = 0
    tokens_before = 0
    tokens_after = 0
    compacted_tools: list[str] = []

    for index, message in enumerate(messages):
        content = str(message.get("content") or "")
        before = estimate_tokens(content)
        tokens_before += before
        # 复制一份再改：不能原地修改调用方传入的消息（它还要用于审计/落库）。
        next_message = dict(message)
        if (
            index < keep_start
            and is_regenerable_context_message(message)
            and before > max_excerpt_tokens * 2
        ):
            names = _tool_names(message)
            excerpt = extract_relevant_snippet(query, content, max_tokens=max_excerpt_tokens)
            tool_label = ", ".join(_unique_preserve_order(names, 4)) or "tool output"
            # 摘要把三件事告诉模型：这里被清理过、原本是什么工具、需要时怎么恢复。
            # 「Recoverable」这句是关键，否则模型会以为信息本身就不存在了。
            compacted_content = f"{MICROCOMPACT_MARKER}\nSource: {tool_label}\nRecoverable: rerun the corresponding workspace, RAG, or GitHub tool if exact details are needed."
            if excerpt:
                compacted_content += f"\nKey excerpt: {excerpt}"
            next_message["content"] = compacted_content
            compacted += 1
            compacted_tools.extend(names)
        tokens_after += estimate_tokens(str(next_message.get("content") or ""))
        output.append(next_message)

    return output, {
        "mode": "microcompact_regenerable_context",
        "input_messages": len(messages),
        "compacted_messages": compacted,
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
        "estimated_tokens_saved": max(0, tokens_before - tokens_after),
        # 实际保留的近期条数可能小于 keep（消息总数不足时），所以取 min。
        "kept_recent_messages": max(0, min(keep, len(messages))),
        "compacted_tools": _unique_preserve_order(compacted_tools, 12),
    }


def extract_relevant_snippet(query: str, text: str, max_tokens: int = 220) -> str:
    """从长文本里挑出与 query 最相关的几个句子，拼成摘要。

    打分方式：命中不同词的数量 × 3 + 各词出现次数；query 整串原样出现再 +4（强信号）。
    没有任何句子得分 > 0 时，退化成「直接截取开头的预算内容」——
    宁可给一段可能无关的开头，也不要返回空（空摘要会让模型丢掉这块信息的线索）。
    选中的句子按原顺序拼接（而不是按得分排），保证读起来仍然连贯。
    """
    # 把各种空白（含 \r）统一成单空格：先归一化才能可靠地按句子切分。
    clean = " ".join(str(text or "").replace("\r", "\n").split())
    if not clean:
        return ""
    terms = set(_words(query))
    if not terms:
        # query 没有任何可用词（纯符号等），无法判断相关性，直接按预算截断。
        return clip_to_token_budget(clean, max_tokens)

    sentences = [part.strip() for part in SENTENCE_SPLIT_PATTERN.split(clean) if part.strip()]
    if not sentences:
        sentences = [clean]

    scored: list[tuple[float, int, str]] = []
    for index, sentence in enumerate(sentences):
        lower = sentence.lower()
        matched = [term for term in terms if term and term in lower]
        score = len(set(matched)) * 3.0 + sum(lower.count(term) for term in matched)
        if query.lower().strip() and query.lower().strip() in lower:
            score += 4.0
        scored.append((score, index, sentence))

    positive = [item for item in scored if item[0] > 0]
    if not positive:
        return clip_to_token_budget(clean, max_tokens)

    # 先按得分取前 4 句（同分按原文顺序），再按原文顺序重排，保证语义连贯。
    positive.sort(key=lambda item: (-item[0], item[1]))
    selected = sorted(positive[:4], key=lambda item: item[1])
    snippet = " ".join(sentence for _score, _index, sentence in selected)
    return clip_to_token_budget(snippet, max_tokens)


def compress_retrieval_results(
    query: str,
    items: list[dict[str, Any]],
    max_tokens_per_item: int = 260,
    max_total_tokens: int | None = None,
) -> list[dict[str, Any]]:
    """压缩检索结果列表：每条只保留与 query 相关的片段，并遵守单条 + 全局双预算。

    两个预算同时生效：
    - max_tokens_per_item 保证单条不至于挤占别人；
    - max_total_tokens 保证整体可控。
    全局预算剩余不足「单条预算」的 1/4（最多 24 token）时就提前停止，
    因为再塞进去也只能得到被截得七零八落的句子，不如干脆不放。
    原始全文只用于计算压缩率，之后会被剔除，避免又把大段内容带进上下文。
    """
    compressed: list[dict[str, Any]] = []
    # 显式区分 None（不限制）和 0（不允许任何内容），所以用 is not None 判断。
    total_budget = max(0, int(max_total_tokens)) if max_total_tokens is not None else None
    used_tokens = 0
    for item in items:
        if total_budget is not None and used_tokens >= total_budget:
            break
        if (
            total_budget is not None
            and compressed
            and total_budget - used_tokens < min(24, max(1, int(max_tokens_per_item)))
        ):
            break
        item_budget = max(1, int(max_tokens_per_item))
        if total_budget is not None:
            # 本条预算不能超过全局剩余额度。
            item_budget = min(item_budget, total_budget - used_tokens)
        # 依次回退取原文：_raw_content 是完整内容，content/snippet 是已有摘要。
        raw_text = str(item.get("_raw_content") or item.get("content") or item.get("snippet") or "")
        snippet = extract_relevant_snippet(query, raw_text, max_tokens=item_budget)
        original_tokens = estimate_tokens(raw_text)
        compressed_tokens = estimate_tokens(snippet)
        # 剔除 _raw_content/content：只保留 snippet，否则压缩就白做了。
        next_item = {key: value for key, value in item.items() if key not in {"_raw_content", "content"}}
        next_item["snippet"] = snippet
        next_item["compression"] = {
            "mode": "query_aware_extractive",
            "original_estimated_tokens": original_tokens,
            "compressed_estimated_tokens": compressed_tokens,
            # ratio < 1 表示压缩有效；原文为空时给 1.0（没有压缩也没有膨胀）。
            "ratio": round(compressed_tokens / original_tokens, 4) if original_tokens else 1.0,
            "item_budget_tokens": item_budget,
            "global_budget_tokens": total_budget,
            "global_tokens_before_item": used_tokens,
        }
        retrieval = dict(next_item.get("retrieval") or {})
        # 把压缩信息同时挂到 retrieval 里：审计/排障时只看 retrieval 就能判断是否被压缩过。
        retrieval["context_compression"] = next_item["compression"]
        retrieval["selected_for_context"] = True
        next_item["retrieval"] = retrieval
        compressed.append(next_item)
        used_tokens += compressed_tokens
    return compressed


# 让模型写「可继续工作的摘要」而不是「简短总结」。
# 提示词里强调「保留原文」（路径、命令、错误信息）和「不要编造」：
# 前者是 Agent 后续动作的关键锚点，后者是压缩阶段最常见的幻觉来源。
COMPACT_SUMMARY_SYSTEM_PROMPT = """请在上下文压缩后，为 DevFlow AI 创建一份可延续工作的摘要。
目标不是单纯减少 token，而是为软件工程 Agent 保留工作连续性。

请用纯文本返回以下部分：
1. 用户意图和明确要求。
2. 当前工作面：现在正在处理什么；如相关，请包含仓库、文件、函数、命令、测试和 UI 状态。
3. 持久决策和约束。
4. 重要事实、代码引用、错误和修复。
5. 待办任务和下一步，并以最近一次用户请求为依据。

当文件路径、标识符、日期、命令、错误消息和用户偏好很重要时，请保留原文。
不要编造细节。如果某个细节只存在于可以重新生成的工具输出中，请说明它可以通过相关工具恢复。"""

# 识别「像文件路径」的片段（可带 Windows 盘符、可带 :行号）。
# 压缩后要校验这些路径有没有丢——丢了路径，Agent 就找不到文件了。
PATH_LIKE_PATTERN = re.compile(
    r"(?:[A-Za-z]:[\\/])?(?:[\w.@()+\- ]+[\\/])+[\w.@()+\- ]+\.[A-Za-z0-9_]+(?::\d+)?"
)
# 命令类关键词：带这些词的整行值得原样保留（执行命令是复现问题的关键）。
COMMAND_HINT_PATTERN = re.compile(
    r"\b(pytest|npm|pnpm|yarn|git|docker|uvicorn|alembic|python|pip|ruff|mypy|pytest)\b",
    flags=re.IGNORECASE,
)
# 「必须保留」的语义关键词：错误、决策、待办、偏好。
# 中英各一份，因为对话本身就是中英混合的。
PRESERVE_LINE_HINTS = (
    "error",
    "failed",
    "failure",
    "exception",
    "traceback",
    "decision",
    "decide",
    "agreed",
    "todo",
    "next step",
    "prefer",
    "preference",
    "错误",
    "失败",
    "异常",
    "决定",
    "确定",
    "待办",
    "下一步",
    "偏好",
)


def _must_preserve_candidates(messages: list[dict[str, Any]], limit: int = 18) -> list[str]:
    """从消息里挑出「压缩后必须还能找到」的片段（路径、命令、关键结论行）。

    这是压缩质量的度量基准：先记下这些必须存在的片段，再检查摘要里有没有它们。
    只取 18 条（默认）是为了让校验本身足够轻量，也避免把整段对话都算成「必须保留」。
    """
    candidates: list[str] = []
    for message in messages:
        content = str(message.get("content") or "")
        if not content:
            continue
        candidates.extend(match.group(0).strip() for match in PATH_LIKE_PATTERN.finditer(content))
        for raw_line in re.split(r"[\r\n]+", content):
            # 去掉行首的行内代码/列表符号和多余空白，避免同一句话因格式不同被当成两条。
            line = " ".join(raw_line.strip(" -\t").split())
            if not line:
                continue
            lower = line.lower()
            if COMMAND_HINT_PATTERN.search(line) or any(hint in lower for hint in PRESERVE_LINE_HINTS):
                candidates.append(clip_to_token_budget(line, 120))
    return _unique_preserve_order(candidates, limit)


def compaction_preservation_report(messages: list[dict[str, Any]], summary: str) -> dict[str, Any]:
    """校验摘要是否覆盖了必须保留的片段，返回缺失清单。

    判断方式是「摘要里是否包含该片段（大小写不敏感）」——
    虽然粗糙（改写了措辞就算缺失），但对「路径/命令/错误串」这类要求原样保留的片段恰好合适。
    """
    required = _must_preserve_candidates(messages)
    summary_lower = str(summary or "").lower()
    missing = [item for item in required if item.lower() not in summary_lower]
    return {
        "required_count": len(required),
        "missing_count": len(missing),
        "required": required,
        "missing": missing,
    }


def ensure_compaction_summary_preserves_key_facts(
    messages: list[dict[str, Any]],
    summary: str,
    max_tokens: int,
) -> tuple[str, dict[str, Any]]:
    """对模型生成的摘要做「保真兜底」：缺什么就在末尾补一个补充清单。

    为什么不重新生成摘要：多一次模型调用既慢又可能引入新的遗漏。
    直接在末尾追加缺失项更可靠——这些片段本来就是原文摘录，补进去不会失真。
    预算分配：先扣掉补充清单的 token，剩下的才给摘要正文，保证补充清单一定放得下
    （它是「必须保留」的内容，优先级高于摘要的措辞完整性）。
    """
    report = compaction_preservation_report(messages, summary)
    missing = [str(item) for item in report.get("missing") or []]
    if not missing:
        # 没有缺失也要按预算裁剪一次：模型可能超长输出。
        return clip_to_token_budget(summary, max_tokens), report

    appendix = "保真补充：\n" + "\n".join(
        # 最多补 12 条、每条 96 token：再多就不是「补充」而是把原文搬回来了。
        f"- {clip_to_token_budget(item, 96)}" for item in missing[:12]
    )
    # 预留 12 token 给拼接用的空行和换行符。
    body_budget = max(160, max_tokens - estimate_tokens(appendix) - 12)
    combined = "\n\n".join([clip_to_token_budget(summary, body_budget), appendix]).strip()
    # 最后再整体裁一次：极端情况下（appendix 本身就很长）仍可能超预算。
    combined = clip_to_token_budget(combined, max_tokens)
    final_report = compaction_preservation_report(messages, combined)
    # 记录「这次是靠追加补回来的」，便于统计兜底触发频率。
    final_report["appended_missing_count"] = len(missing)
    return combined, final_report


def build_compaction_summary_payload(messages: list[dict[str, Any]], max_tokens: int = 6000) -> str:
    """把待压缩的对话拼成「喂给模型写摘要」的纯文本。

    每条消息按「角色[ tools=...]: 内容」格式，条目之间空行分隔；
    单条预算是总预算均分（至少 160），避免某条超长消息占满整个输入。
    分界线消息单独标成 compact_boundary，让模型知道这里已经压缩过一次。
    """
    lines: list[str] = []
    for message in messages:
        role = str(message.get("role") or "unknown")
        if role == "system" and is_compact_boundary_message(message):
            role = "compact_boundary"
        content = clip_to_token_budget(str(message.get("content") or ""), max(160, max_tokens // max(len(messages), 1)))
        if not content:
            continue
        tool_names = _tool_names(message)
        # 把工具名带上：模型判断「这条是不是工具输出」时会用到。
        suffix = f" tools={tool_names}" if tool_names else ""
        lines.append(f"{role}{suffix}: {content}")
    return clip_to_token_budget("\n\n".join(lines), max_tokens)


def format_compaction_summary(summary: str) -> str:
    """清洗模型返回的摘要文本。

    有些模型会按提示词模板输出 <analysis>…</analysis> 和 <summary>…</summary>：
    分析过程是给模型自己看的，不该进入上下文，所以整段删掉；
    summary 标签内的内容则提取出来并加上「摘要：」前缀。
    最后把三个以上连续换行压成两个，避免空行过多浪费 token。
    """
    formatted = str(summary or "").strip()
    formatted = re.sub(r"<analysis>[\s\S]*?</analysis>", "", formatted, flags=re.IGNORECASE).strip()
    match = re.search(r"<summary>([\s\S]*?)</summary>", formatted, flags=re.IGNORECASE)
    if match:
        formatted = "摘要：\n" + match.group(1).strip()
    formatted = re.sub(r"\n{3,}", "\n\n", formatted)
    return formatted.strip()


def fallback_compaction_summary(messages: list[dict[str, Any]], max_tokens: int | None = None) -> str:
    """不调用 LLM 的规则式压缩摘要：按关键词把行分成决策/任务/事实三类。

    什么时候用：模型不可用、额度耗尽、或调用失败时，保证压缩流程仍能完成
    （否则上下文一旦爆掉，整个会话就彻底卡死）。
    质量明显低于 LLM 摘要，但保留了「结构 + 最近几条原话」这个骨架，
    至少让模型知道之前在讨论什么。
    """
    user_messages = [str(item.get("content") or "") for item in messages if item.get("role") == "user"]
    assistant_messages = [str(item.get("content") or "") for item in messages if item.get("role") == "assistant"]
    decisions: list[str] = []
    tasks: list[str] = []
    facts: list[str] = []
    for content in [*user_messages, *assistant_messages]:
        for line in [part.strip(" -\t") for part in content.splitlines() if part.strip()]:
            lower = line.lower()
            # 中文关键词用转义形式写（\u51b3\u5b9a=决定、\u786e\u5b9a=确定），
            # 效果等同直接写汉字，这里保持原样不动。
            if any(token in lower for token in ["decision", "decide", "agreed", "\u51b3\u5b9a", "\u786e\u5b9a"]):
                decisions.append(line)
            if any(token in lower for token in ["todo", "next step", "implement", "fix", "\u5f85\u529e", "\u4e0b\u4e00\u6b65"]):
                tasks.append(line)
            # 「事实」没有关键词可依，退化成「足够长的行」，且最多 10 条防止膨胀。
            if len(line) > 20 and len(facts) < 10:
                facts.append(line)
    summary = "\n\n".join(
        [
            "This conversation was compacted. Continue from the preserved recent messages and this summary.",
            # 用户消息取最近 8 条、助手消息取最近 5 条：越近越重要，且用户意图比回答更需要保真。
            "User intent:\n" + "\n".join(f"- {clip_to_token_budget(item, 120)}" for item in user_messages[-8:]),
            "Assistant context:\n" + "\n".join(f"- {clip_to_token_budget(item, 120)}" for item in assistant_messages[-5:]),
            "Durable decisions:\n" + "\n".join(f"- {item}" for item in _unique_preserve_order(decisions, 12)),
            "Pending tasks:\n" + "\n".join(f"- {item}" for item in _unique_preserve_order(tasks, 12)),
            "Important facts:\n" + "\n".join(f"- {item}" for item in _unique_preserve_order(facts, 12)),
        ]
    )
    return clip_to_token_budget(summary, max_tokens or settings.context_compact_summary_tokens)


def structured_memory_from_model(parsed: dict[str, Any]) -> dict[str, Any]:
    """把模型（或 fallback）返回的松散 JSON 规范成固定结构的记忆对象。

    做三件事：全部字段强行转成字符串列表、去重并限制条数、
    citations 只保留 dict 类型。目的是让下游（渲染/入库/建图）拿到的结构永远一致，
    不用再对每个字段做类型防御。
    不同字段的条数上限不同：事实/决策/任务/仓库上下文给 16，
    未决问题/偏好给 12（这两类更容易重复且价值密度低）。
    """
    return {
        "summary": str(parsed.get("summary") or "").strip(),
        "facts": _unique_preserve_order([str(item) for item in parsed.get("facts") or []], 16),
        "decisions": _unique_preserve_order([str(item) for item in parsed.get("decisions") or []], 16),
        "open_questions": _unique_preserve_order([str(item) for item in parsed.get("open_questions") or []], 12),
        "tasks": _unique_preserve_order([str(item) for item in parsed.get("tasks") or []], 16),
        "user_preferences": _unique_preserve_order([str(item) for item in parsed.get("user_preferences") or []], 12),
        "repo_context": _unique_preserve_order([str(item) for item in parsed.get("repo_context") or []], 16),
        # 只保留 dict 且最多 16 条：citations 下游会按字段取值，混入字符串会直接崩。
        "citations": [item for item in parsed.get("citations") or [] if isinstance(item, dict)][:16],
    }


def fallback_memory_update(user_message: str, assistant_message: str, route: str, tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
    """规则式的「记忆增量抽取」：从一轮问答里提取决策/问题/任务/偏好。

    一轮对话就调用一次 LLM 抽记忆太贵，所以先用关键词做粗筛；
    抽出来的条目都带原文句子，偏保守（宁可多记几条，后面还有人工审核环节）。
    问题类用「含问号或以 why/how/what 开头」判断，是跨中英文都还算可靠的启发式。
    """
    joined = "\n".join([user_message or "", assistant_message or ""])
    lines = [line.strip(" -\t") for line in joined.splitlines() if line.strip()]
    decisions: list[str] = []
    open_questions: list[str] = []
    tasks: list[str] = []
    facts: list[str] = []
    preferences: list[str] = []

    for line in lines:
        lower = line.lower()
        # 五个 if 不是 elif：同一行可以既是待办又是决策，允许同时进入多个桶。
        if "?" in line or "\uff1f" in line or lower.startswith(("why ", "how ", "what ")):
            open_questions.append(line)
        if any(token in lower for token in ["decide", "decision", "agreed", "chosen", "\u786e\u5b9a", "\u51b3\u5b9a"]):
            decisions.append(line)
        if any(token in lower for token in ["todo", "next step", "action", "fix", "implement", "\u5f85\u529e", "\u4e0b\u4e00\u6b65"]):
            tasks.append(line)
        if any(token in lower for token in ["prefer", "preference", "\u559c\u6b22", "\u504f\u597d"]):
            preferences.append(line)
        # 事实没有关键词，用「长度 > 20」当代理指标，最多 6 条。
        if len(line) > 20 and len(facts) < 6:
            facts.append(line)

    tools = ", ".join(str(call.get("tool_name") or "未知") for call in tool_calls) or "无"
    # 摘要里保留路由和工具名：这是排查「这轮回答是怎么来的」的最小线索集。
    summary = (
        f"路由：{route}\n"
        f"用户请求：{clip_to_token_budget(user_message, 90)}\n"
        f"助手回答：{clip_to_token_budget(assistant_message, 140)}\n"
        f"工具：{tools}"
    )
    return structured_memory_from_model(
        {
            "summary": summary,
            "facts": facts,
            "decisions": decisions,
            "open_questions": open_questions,
            "tasks": tasks,
            "user_preferences": preferences,
            "repo_context": [f"使用路由：{route}"] if route else [],
            "citations": [],
        }
    )


def merge_memory(existing: dict[str, Any], update: dict[str, Any], digest: str, max_tokens: int) -> dict[str, Any]:
    """把新的记忆增量合并进已有记忆，返回合并后的新对象（不修改入参）。

    合并规则：
    - 摘要「追加」而不是替换（existing + update），再用 digest 兜底，保证历史信息不丢；
      摘要会按 max_tokens 截断，所以无限追加也不会失控。
    - 列表字段拼接后去重（新的排在后面），受各字段的条数上限约束。
    - citations 取最后 16 条而不是前 16 条：引用越新越可能对应最近的结论。
    """
    existing_summary = str(existing.get("summary") or "").strip()
    # update 没有摘要时用 digest（通常是本轮对话的简短摘要）兜底，避免摘要断档。
    update_summary = str(update.get("summary") or digest or "").strip()
    summary_parts = [part for part in [existing_summary, update_summary] if part]
    summary = "\n\n".join(summary_parts)
    summary = clip_to_token_budget(summary, max_tokens)
    return {
        "summary": summary,
        "facts": _unique_preserve_order([*(existing.get("facts") or []), *(update.get("facts") or [])], 16),
        "decisions": _unique_preserve_order([*(existing.get("decisions") or []), *(update.get("decisions") or [])], 16),
        "open_questions": _unique_preserve_order([*(existing.get("open_questions") or []), *(update.get("open_questions") or [])], 12),
        "tasks": _unique_preserve_order([*(existing.get("tasks") or []), *(update.get("tasks") or [])], 16),
        "user_preferences": _unique_preserve_order([*(existing.get("user_preferences") or []), *(update.get("user_preferences") or [])], 12),
        "repo_context": _unique_preserve_order([*(existing.get("repo_context") or []), *(update.get("repo_context") or [])], 16),
        "citations": [*(existing.get("citations") or []), *(update.get("citations") or [])][-16:],
    }


def render_structured_memory(memory: Any, max_tokens: int | None = None) -> str:
    """把结构化记忆渲染成给模型读的中文文本块。

    参数类型是 Any 且用 getattr 取属性：调用方可能传 ORM 对象、dataclass，
    也可能传用 type() 临时构造的对象（见 evals/runner），只要是属性访问就都能工作。
    每类最多渲染 12 条，避免记忆越攒越多把上下文挤满。
    """
    sections: list[str] = []
    summary = str(getattr(memory, "summary", "") or "").strip()
    if summary:
        sections.append("摘要：\n" + summary)
    for label, attr in [
        ("事实", "facts"),
        ("决策", "decisions"),
        ("未决问题", "open_questions"),
        ("任务", "tasks"),
        ("用户偏好", "user_preferences"),
        ("仓库上下文", "repo_context"),
    ]:
        values = getattr(memory, attr, None) or []
        if values:
            sections.append(label + ":\n" + "\n".join(f"- {item}" for item in values[:12]))
    text = "\n\n".join(sections)
    # max_tokens 为 None 表示「不限制」，用于把记忆整篇写进知识库文档的场景。
    return clip_to_token_budget(text, max_tokens) if max_tokens else text


def memory_document_text(memory: Any) -> str:
    """生成「记忆笔记」文档正文（不截断，完整落库）。"""
    return render_structured_memory(memory, max_tokens=None)


def prompt_cache_model_kwargs() -> dict[str, Any]:
    """组装提示词缓存相关的模型调用参数。

    缓存能显著降低重复系统提示的成本，但不是所有供应商都支持，
    所以开关关闭时返回空 dict（调用方 **展开即可，不影响其它参数）。
    两个可选参数都只在配置里写了才透传，避免给不支持的服务端发空值。
    """
    if not settings.prompt_cache_enabled:
        return {}
    kwargs: dict[str, Any] = {}
    if settings.prompt_cache_key:
        kwargs["prompt_cache_key"] = settings.prompt_cache_key
    if settings.prompt_cache_retention:
        kwargs["prompt_cache_retention"] = settings.prompt_cache_retention
    return kwargs


def compression_eval_summary(before_text: str, after_text: str, retrieved: int, kept: int) -> dict[str, Any]:
    """算压缩效果的汇总指标（用于评估面板和自检）。

    compression_ratio = 压缩后 / 压缩前，越小压缩越狠；
    原文为空时给 1.0（视为「没有可压缩的内容」，而不是 0 或无穷大）。
    token_savings 用 max(0, ...)：压缩后反而变长时记为 0 节省，不显示负数。
    """
    before = estimate_tokens(before_text)
    after = estimate_tokens(after_text)
    return {
        "estimated_tokens_before": before,
        "estimated_tokens_after": after,
        "compression_ratio": round(after / before, 4) if before else 1.0,
        "retrieved_items": retrieved,
        "kept_items": kept,
        "token_savings": max(0, before - after),
    }


def dumps_compact(value: Any) -> str:
    """紧凑 JSON 序列化：不转义中文、无多余空格，datetime 等不可序列化对象转成字符串。

    用于写进 prompt/日志这类「越短越好」的场景；
    与标准 json.dumps 的区别就是去掉了空格并加了 default=str 兜底。
    """
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
