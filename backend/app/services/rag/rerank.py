"""重排（rerank）：在向量召回之后、生成之前，重新给候选排序。

位置：切块 → 向量化 → 检索 → **重排** → 生成 → 答案门禁。
输入是检索阶段拿到的候选（可能混合了向量召回与关键词召回），
输出是重排后、准备交给 LLM 的候选列表（数量被 limit 截断）。

为什么需要重排：向量召回本质是「语义相近」，它擅长把可能相关的都捞出来（高召回），
但不擅长把最相关的排到第一（低精度）。常见的失败是「一份泛泛而谈的文档因为
主题词重合度高，把精确命中的代码片段挤到第 5 位」。重排用更细的特征把顺序纠正过来。

**本实现是启发式的，不是模型重排**。没有调用 cross-encoder 之类的小模型，
而是用「词覆盖率 + 标题命中 + 路径命中 + 多路召回 + 新鲜度」加权求和。
取舍：零成本、零延迟、完全可解释（每条都带 rank_reason），
代价是精度不如真正的语义重排。函数名与 retrieval 里的 rerank_mode="heuristic"
都明确标注了这一点，避免误以为是模型打分。

权重是人工调的固定值，都写死在 _heuristic_score 里，改权重会直接改变
答案引用哪几条证据——调参时请配合评测集一起看。
"""

import math
import re
from datetime import datetime, timezone
from typing import Any

from app.core.config import settings


def _tokens(text: str) -> list[str]:
    """切词：英文按符号切，中文用「单字 + 二字组」。

    与 embeddings 里的 _deterministic_tokens 故意保持一致的分词策略——
    重排的打分要和向量化看到同样的粒度，否则「向量说像，重排说没命中」会互相打架。

    二字组（bigram）是中文不分词时的常用折中：单字召回太宽（"的"会在任何
    文本里命中），二字组已经能表达大部分词级信息。
    """
    lowered = text.lower()
    tokens = re.findall(r"[a-z0-9_\-]+", lowered)
    for segment in re.findall(r"[\u4e00-\u9fff]+", lowered):
        tokens.extend(segment)
        tokens.extend(segment[index : index + 2] for index in range(max(0, len(segment) - 1)))
    return tokens


def _normalise(value: float) -> float:
    """把分数夹到 [0, 1]，并把 NaN / inf 归零。

    必须处理 NaN/inf：候选分数可能来自外部（Milvus、SQL 计算），
    一旦出现 NaN，后面的加权求和会把整个分数污染成 NaN，
    最终排序结果完全随机且不报错——这是最难查的一类 bug。
    """
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return max(0.0, min(1.0, value))


def _candidate_text(candidate: dict[str, Any]) -> str:
    """抽出参与匹配的文本，全部转小写。

    只取标题、路径、片段三部分，**不取全文**：全文太长会让覆盖率天然接近 1
    （任何词都能在长文里找到），权重就失去区分度了。
    """
    metadata = candidate.get("metadata") or {}
    path = str(metadata.get("path") or "")
    return "\n".join(
        [
            str(candidate.get("title") or ""),
            path,
            str(candidate.get("snippet") or ""),
        ]
    ).lower()


def _freshness_bonus(metadata: dict[str, Any]) -> float:
    """按时间新旧给一个很小的加分（0 / 0.02 / 0.04）。

    存在意义：知识库里同一件事往往有多个版本（旧 issue、已关闭的 PR），
    同样的语义匹配下应该优先呈现较新的证据。但**加分刻意很小**，
    不能让它压过语义相关性——旧但精确的文档依然应该排前面。

    时间字段按 updated_at → created_at → merged_at → closed_at 的顺序找第一个
    有值的，因为不同类型的证据可用的时间字段不同。

    时区处理：ISO 字符串里的 Z 要转成 +00:00（Python 3.11 之前的
    fromisoformat 不认 Z）；没有时区的按 UTC 处理，避免与 now 相减报错。
    解析失败就跳过这个字段，继续尝试下一个。
    """
    for key in ("updated_at", "created_at", "merged_at", "closed_at"):
        value = metadata.get(key)
        if not value:
            continue
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        # max(0, ...)：未来时间（时钟偏差）当作 0 天，不给额外奖励
        age_days = max(0, (datetime.now(timezone.utc) - parsed).days)
        # 三档而不是连续衰减：连续函数会让不同候选的分数差变得难以解释，
        # 分档更容易在 rank_reason 里说明白
        if age_days <= 30:
            return 0.04
        if age_days <= 180:
            return 0.02
        return 0.0
    return 0.0


def _heuristic_score(query: str, candidate: dict[str, Any]) -> tuple[float, str]:
    """给单个候选打启发式分，并返回可读的命中原因。

    分项含义（括号内是最大贡献）：
    - base 0.31：候选自身携带的分数（通常是多路融合后的总分）
    - vector 0.20：纯向量相似度
    - keyword 0.16：纯关键词相似度
    - coverage 0.17：查询词在候选文本中的**去重覆盖率**
    - title_hits 0.04：命中在标题里（说明主题对得上）
    - path_hits 0.02：命中在文件路径里
    - phrase_bonus 0.12：查询整串作为短语出现（最强的精确信号）
    - source_bonus 0.08：至少命中 2 个不同查询词
    - dual_recall_bonus 0.05：同时被向量和关键词两路召回（两路都认为相关，可信度更高）
    - freshness_bonus 0.04：证据较新
    - source_type_bonus 0.03：查询词命中了来源类型名（如问"issue"）

    权重之和刻意 > 1.0，所以最终结果要再 _normalise 一次夹到 [0,1]。
    这样设计的好处是「多个正向信号叠加」时能突破单个信号的 1.0 上限，
    让明显更好的候选拉开差距。

    reasons 用英文短句拼接，最终会写进 rank_reason 字段供前端展示与调试；
    一个都没命中时给中文兜底 "弱关键词匹配"，避免出现空字符串。
    """
    terms = _tokens(query)
    # 查询切不出任何词（全是标点/空白）：无法做词匹配，只能退回原始分数
    if not terms:
        return _normalise(float(candidate.get("score") or 0.0)), "base score only"
    text = _candidate_text(candidate)
    title = str(candidate.get("title") or "").lower()
    metadata = candidate.get("metadata") or {}
    path = str(metadata.get("path") or "").lower()
    matched = [term for term in terms if term in text]
    # 分母用去重后的词数：否则重复词（如"登录 登录 失败"）会把覆盖率算低
    coverage = len(set(matched)) / max(len(set(terms)), 1)
    title_hits = sum(1 for term in terms if term in title)
    path_hits = sum(1 for term in terms if term in path)
    phrase_bonus = 0.12 if query.lower().strip() and query.lower().strip() in text else 0.0
    source_bonus = 0.08 if len({term for term in terms if term in text}) >= 2 else 0.0
    base = _normalise(float(candidate.get("score") or 0.0))
    keyword = _normalise(float(candidate.get("keyword_score") or 0.0))
    vector = _normalise(float(candidate.get("vector_score") or 0.0))
    # 双路召回判定：候选的 retrieval.sources 里同时含 vector 与 keyword
    sources = set((candidate.get("retrieval") or {}).get("sources") or [])
    dual_recall_bonus = 0.05 if {"vector", "keyword"}.issubset(sources) else 0.0
    freshness_bonus = _freshness_bonus(metadata)
    source_type = str(candidate.get("source_type") or "").lower()
    source_type_bonus = 0.03 if source_type and any(term in source_type for term in terms) else 0.0
    score = (
        0.31 * base
        + 0.20 * vector
        + 0.16 * keyword
        + 0.17 * coverage
        # min(...,3)/3：命中 3 个词就封顶，避免长查询天然占优
        + 0.04 * min(title_hits, 3) / 3
        + 0.02 * min(path_hits, 3) / 3
        + phrase_bonus
        + source_bonus
        + dual_recall_bonus
        + freshness_bonus
        + source_type_bonus
    )
    reasons = []
    if coverage:
        reasons.append(f"matched {len(set(matched))}/{len(set(terms))} query terms")
    if title_hits:
        reasons.append("title match")
    if path_hits:
        reasons.append("path match")
    if vector:
        reasons.append("vector candidate")
    if keyword:
        reasons.append("keyword candidate")
    if dual_recall_bonus:
        reasons.append("dual recall")
    if freshness_bonus:
        reasons.append("fresh evidence")
    if source_type_bonus:
        reasons.append("source type match")
    return _normalise(score), ", ".join(reasons) or "弱关键词匹配"


def rerank_documents(query: str, candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """重排主入口：只重排前 top_n 个，其余原样垫在后面，最后取前 limit 条。

    为什么只重排头部：启发式打分只对「本来就有希望」的候选有意义，
    对排名 50 开外的候选做精算纯属浪费；而且 top_n 之外的部分本来就会被
    limit 截掉。top_n = min(max(rag_rerank_top_n, limit), 候选总数)，
    保证「至少覆盖要返回的条数」，也不会超过候选总数。

    尾部 tail 不参与重排、直接追加：它们保持原有的召回顺序，
    只有在头部重排后仍然不足 limit 时才会被用到。

    排序键是二元组 (rerank_score, combined_score) 降序：
    rerank 分相同时，用上游的融合分作为次级依据，保证结果稳定可复现。

    原地修改的注意点：这里给每个候选生成了**新字典**（{**candidate, ...}），
    不会污染调用方传进来的列表。同时把 rerank 信息写进 retrieval 子字典，
    这样下游（answer_gate、前端）既能读到顶层 score，也能看到原始召回信息。
    """
    if not candidates:
        return []
    top_n = min(max(settings.rag_rerank_top_n, limit), len(candidates))
    head = candidates[:top_n]
    tail = candidates[top_n:]
    ranked: list[dict[str, Any]] = []
    for candidate in head:
        heuristic_score, heuristic_reason = _heuristic_score(query, candidate)
        ranked.append(
            {
                **candidate,
                # 顶层 score 被**覆盖**为新分数：下游（尤其是 answer_gate 的
                # 分数门槛与冲突检测）看到的就是重排后的分数，语义统一
                "score": round(heuristic_score, 4),
                "rerank_score": round(heuristic_score, 4),
                "rank_reason": heuristic_reason,
                "retrieval": {
                    **(candidate.get("retrieval") or {}),
                    # 标记为 heuristic，提醒读代码的人这里没有用模型重排
                    "rerank_mode": "heuristic",
                    "rerank_score": round(heuristic_score, 4),
                    "rank_reason": heuristic_reason,
                },
            }
        )
    ranked.sort(key=lambda item: (float(item.get("rerank_score") or item.get("score") or 0.0), float(item.get("combined_score") or 0.0)), reverse=True)
    return [*ranked, *tail][:limit]
