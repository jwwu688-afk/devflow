"""答案门禁（answer gate）：在把检索结果交给 LLM 生成之前，先判断「该不该答」。

为什么需要这一层：向量检索永远会返回 top-k，哪怕库里根本没有相关内容，
也会给出几条「看起来最像」的片段。如果直接喂给模型，模型几乎必然会
顺着这几条无关片段编出一个自信的答案——这是 RAG 系统最主要的幻觉来源。
所以必须在生成之前加一道独立的、不依赖模型的判定。

门禁按「越便宜越先做」的顺序排列，一旦命中就短路返回：

    1. 没有证据                    → insufficient_evidence
    2. top1 分数低于门槛           → insufficient_evidence
    3. 查询里的强标识（文件名/错误名/编号）在所有证据中都没出现
                                   → insufficient_evidence（检索跑偏了）
    4. 高排名证据互相矛盾           → conflict（宁可说不知道，也不要选一个）
    5. 问题本身指代不明             → ask_clarification（先反问，别瞎猜）
    6. 以上都过                     → answer

输出的 decision 字段是下游唯一需要消费的东西；其余字段（score、signals、
conflict 明细）用于前端展示「为什么不能答」以及排查检索质量问题。
"""

import re
from typing import Any


# 「指代不明」的信号词：出现这些词说明问题依赖上下文才能定位对象，
# 例如「这个怎么办」——「这个」指什么，光看问题本身无法确定
AMBIGUOUS_MARKERS = ("这个", "那个", "它", "这段", "这里", "怎么办", "怎么处理")

# 冲突检测的「极性词表」：(冲突类型, 正向词, 反向词)。
# 只覆盖三类最容易造成实质误解的对立：权限、结果、必选/可选。
# 之所以用词表而不是让模型判断，是因为这一步必须在生成之前完成、
# 且要足够便宜和可解释——冲突判定错了，人工一眼就能看出来是哪个词导致的。
CONFLICT_POLARITIES = (
    ("permission", ("允许", "可以", "enabled", "enable"), ("禁止", "不允许", "不能", "disabled", "disable")),
    ("outcome", ("成功", "通过", "success", "passed"), ("失败", "未通过", "failure", "failed")),
    ("requirement", ("必须", "需要", "required"), ("可选", "无需", "optional")),
)


def _normalise(text: str) -> str:
    """统一小写并把连续空白压成单空格。

    目的是让后面的子串匹配不受大小写和换行影响（检出的日志/文档常常
    带大量换行与缩进）。空值兜底成 ""，避免 None 传进来直接崩。
    """
    return " ".join(str(text or "").lower().split())


def _evidence_text(item: dict[str, Any]) -> str:
    """把一条证据里「可用于核对」的文本拼成一个字符串。

    只取有语义的字段：标题、片段正文、文件路径、符号名、分支名。
    刻意不取 id / score / 时间戳——它们不参与关键词核对，混进来只会
    增加误匹配（比如 id 里恰好含数字）。
    snippet 缺失时退回 _raw_content，因为部分来源只保留了原文。
    """
    metadata = item.get("metadata") or {}
    return _normalise(
        "\n".join(
            [
                str(item.get("title") or ""),
                str(item.get("snippet") or item.get("_raw_content") or ""),
                str(metadata.get("path") or ""),
                str(metadata.get("symbol") or ""),
                str(metadata.get("branch") or metadata.get("base_branch") or ""),
            ]
        )
    )


def _strong_query_signals(query: str) -> list[str]:
    """从问题中抽出「强标识」——那种必须原样出现在证据里才算命中的 token。

    抽取规则：英文/数字开头的连续串、#123 形式的编号、3 位以上的纯数字。
    然后只保留满足以下任一条件的：
      - 含 _ . / - 之一（文件名、模块路径、配置键的特征）
      - 含数字（错误码、issue 号、端口号的普遍特征）
      - 以 Error / Exception 结尾（异常类名）

    为什么只要这三类：普通的英文单词（如 "config"、"user"）太泛，几乎必然
    在任何证据里都能找到，用它们做核对等于没核对；只有这些「高熵标识」
    缺失时才足以证明检索跑偏了。去重同时保持顺序。
    """
    candidates = re.findall(r"[A-Za-z][A-Za-z0-9_./\-]*|#\d+|\b\d{3,}\b", query)
    signals: list[str] = []
    for candidate in candidates:
        if (
            any(marker in candidate for marker in ("_", ".", "/", "-"))
            or any(char.isdigit() for char in candidate)
            or candidate.endswith(("Error", "Exception"))
        ):
            signals.append(candidate.lower())
    return list(dict.fromkeys(signals))


def _detect_conflict(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    """检测高排名证据之间是否存在「同一个命题的正反两种说法」。

    两个前置条件：
    1. 至少 2 条分数 >= 0.6 的证据。0.6 是经验阈值——低于它的片段本身就
       不可靠，拿它们判冲突会制造大量误报（噪音之间当然互相矛盾）。
    2. 对每一类极性，必须同时找到「只说正向」和「只说反向」的两组证据，
       且两组不是同一批（集合不相等）。

    关键细节：正向判定里加了 `not any(negative)` 排除，即一条同时包含
    「允许」和「禁止」的证据（比如文档在讲「什么情况不允许」）不算正向证据。
    否则描述规则本身的文档会被误判成冲突方。

    返回 None 表示没有冲突；有冲突时返回冲突类型与双方证据 id，
    便于前端把两派证据都摆给用户看，而不是让模型替用户选一个。
    """
    high_confidence = [item for item in items if float(item.get("score") or 0.0) >= 0.6]
    if len(high_confidence) < 2:
        return None
    texts = [(str(item.get("id") or item.get("source_id") or ""), _evidence_text(item)) for item in high_confidence]
    for conflict_type, positive_markers, negative_markers in CONFLICT_POLARITIES:
        negative_ids = [item_id for item_id, text in texts if any(marker in text for marker in negative_markers)]
        positive_ids = [
            item_id
            for item_id, text in texts
            if any(marker in text for marker in positive_markers)
            and not any(marker in text for marker in negative_markers)
        ]
        # 两组都非空、且不是同一批证据 → 认为存在实质冲突
        if positive_ids and negative_ids and set(positive_ids) != set(negative_ids):
            return {
                "type": conflict_type,
                "positive_evidence_ids": positive_ids,
                "negative_evidence_ids": negative_ids,
            }
    return None


def _is_ambiguous_query(query: str, items: list[dict[str, Any]]) -> bool:
    """判断问题是否「指代不明、需要反问」。

    两个条件必须同时成立：
    1. 问题里出现指代词（这个/它/怎么办…）；
    2. 并且满足其一：证据来自多个不同 source（说明检索没锁定到单一对象），
       或者问题本身极短（<= 8 字符，信息量不足以定位）。

    第 2 条是防误伤的关键：如果问题说「这个 PR 的 CI 为什么失败」但检索
    只命中了同一个 PR 下的多条证据，那「这个」其实是被唯一确定的，
    不该再反问用户——那只会让人烦。
    """
    compact = re.sub(r"\s+", "", query)
    if not any(marker in compact for marker in AMBIGUOUS_MARKERS):
        return False
    source_ids = {str(item.get("source_id") or "") for item in items if item.get("source_id")}
    return len(source_ids) > 1 or len(compact) <= 8


def evaluate_answer_gate(
    query: str,
    items: list[dict[str, Any]],
    *,
    score_threshold: float = 0.0,
    threshold_enabled: bool = True,
) -> dict[str, Any]:
    """门禁主入口：输入问题与候选证据，输出一个 decision 字典。

    items 应当是**重排之后**的候选（顺序即相关性排序），因为 top_score
    直接取的是当前列表里的最大值，且冲突检测只看高分项。

    score_threshold / threshold_enabled 做成可关：某些调用场景（如「帮我在
    全部文档里找找看」）用户明确接受了低相关结果，此时把门槛置 0 或关掉，
    否则会被一律判为证据不足。

    返回结构统一含 evidence_count / top_score / score_threshold 三个诊断字段，
    无论哪种 decision 都带上，前端可以直接展示「最高分 0.42，门槛 0.5」。
    """
    top_score = max((float(item.get("score") or 0.0) for item in items), default=0.0)
    # base 是各分支共享的诊断信息，用 **base 展开，避免每个 return 重复写三行
    base = {
        "evidence_count": len(items),
        "top_score": round(top_score, 4),
        # 门槛被关闭时显式返回 None，而不是返回 0.0——0.0 会被误读成
        # 「门槛是 0 但启用了」，None 才准确表达「本次未启用门槛」
        "score_threshold": round(float(score_threshold), 4) if threshold_enabled else None,
    }
    # 分支 1：完全没有证据，最确定的一种「不能答」
    if not items:
        return {**base, "decision": "insufficient_evidence", "reason": "没有检索到满足条件的证据"}
    # 分支 2：有证据但最高分不过关。注意是「最高分」而非平均分：
    # 只要有一条强相关就足够支撑回答，被大量弱相关拉低平均分不该成为拒绝理由
    if threshold_enabled and top_score < score_threshold:
        return {**base, "decision": "insufficient_evidence", "reason": "最高证据分数未达到回答门槛"}

    # 分支 3：关键标识核对。只有「全部强标识都找不到」才拒绝；
    # 部分命中是允许的（问题里可能夹带了若干个库里本就没有的词）
    signals = _strong_query_signals(query)
    evidence_text = "\n".join(_evidence_text(item) for item in items)
    missing_signals = [signal for signal in signals if signal not in evidence_text]
    if signals and len(missing_signals) == len(signals):
        return {
            **base,
            "decision": "insufficient_evidence",
            "reason": "查询中的关键名称没有出现在候选证据中",
            "required_signals": signals,
            "missing_signals": missing_signals,
        }

    # 分支 4：证据冲突。排在分支 3 之后，是因为「检索跑偏」比「证据矛盾」
    # 更根本——先把明显跑偏的挡掉，再讨论剩下的证据是否可信
    conflict = _detect_conflict(items)
    if conflict:
        return {**base, "decision": "conflict", "reason": "高排名证据给出了相反结论", "conflict": conflict}

    # 分支 5：问题本身不明确。放在最后，因为它的前提是「证据看着是相关的」——
    # 证据都不相关的话，反问用户也没意义，应该直接说资料不足
    if _is_ambiguous_query(query, items):
        return {
            **base,
            "decision": "ask_clarification",
            "reason": "问题缺少可定位的对象或范围",
            # 这三项是给用户的具体引导，直接告诉他要补什么信息
            "suggested_fields": ["文件或模块", "错误名称", "分支或版本"],
        }

    # 全部通过：matched_signals 让前端能显示「命中了哪些关键标识」，增强可信度
    return {
        **base,
        "decision": "answer",
        "reason": "证据达到分数门槛，关键名称可以在候选中核对",
        "matched_signals": [signal for signal in signals if signal in evidence_text],
    }


def gate_response_message(gate: dict[str, Any]) -> str:
    """把门禁结果转成给终端用户看的一句中文提示。

    三种非 answer 的决策各有一套话术：
    - ask_clarification：告诉用户补什么（可操作）
    - conflict：解释为什么不能给结论（不是系统坏了）
    - 其他（含 insufficient_evidence 与未知值）：统一按「资料不足」处理

    默认兜底成 insufficient_evidence，是为了防止上游新增了 decision 取值
    而这里忘了同步时，界面出现空白提示。
    """
    decision = str(gate.get("decision") or "insufficient_evidence")
    if decision == "ask_clarification":
        return "当前问题的对象或范围不够明确。请补充文件、模块、错误名称、分支或版本后再试。"
    if decision == "conflict":
        return "当前知识库中的高排名证据存在冲突，暂时无法给出确定结论。请先核对对应来源或版本。"
    return "当前知识库资料不足，无法确认。请补充相关资料、检查检索范围，或换一种更具体的问法。"
