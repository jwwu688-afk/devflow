"""知识库编排层：配置管理、模型可用性探测、以及**检索侧的总流水线**。

本文件的 retrieve_knowledge 是整条读链路的唯一编排点，把各环节串起来：

    1. 解析参数（配置默认值 + 调用方覆盖）
    2. 查询向量化（只有 vector / hybrid 且来源里含可向量化类型时才做）
    3. 召回（search_similar_documents，内部含向量 + 关键词双路）
    4. 重排（Qwen API 优先，失败/未配置则降级到本地启发式）
    5. 取前 top_k
    6. 父块扩展（expand_parent_context，把小块的上下文补全成大块）
    7. 分数门槛过滤
    8. 上下文压缩（按 token 预算裁剪）
    9. 答案门禁（evaluate_answer_gate → answer / insufficient_evidence / conflict / ask_clarification）

**顺序不能随意调换**，几个关键约束：
- 重排必须在召回之后、截断之前：先召回到足够多的候选（candidate_limit 最多 100 条），
  再从中挑最相关的 top_k，而不是只召回 top_k 条再重排；
- 父块扩展在重排之后：它改变的是证据的**内容长度**而不是相关性，
  不应该影响排序；
- 门槛过滤在父块扩展之后（但用的是重排分数）：先补全上下文再判断是否够格，
  避免「内容够但分数略低」的好证据被误杀；
- 门禁放最后：它需要看到最终真正会进提示词的那批证据，否则判断的是「假想上下文」。

每一步都会往 pipeline_trace 里写一条 stage 记录（输入数 / 输出数 / 参数），
前端可以把它渲染成一条可视化流水线，排查「为什么这次没答案」时极其有用。
"""

import importlib.util
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import KnowledgeBaseConfig, RetrievalTestRun
from app.services.context_compression import compress_retrieval_results, estimate_tokens
from app.services.rag.answer_gate import evaluate_answer_gate
from app.services.rag.embeddings import (
    embed_knowledge_query,
    embedding_contract,
    normalize_embedding_provider,
    validate_embedding_contract,
)
from app.services.rag.rerank import rerank_documents
from app.services.rag.retrieval import expand_parent_context, search_similar_documents
from app.services.rag.source_policy import normalize_source_types, should_vectorize


def as_uuid(value: str | uuid.UUID) -> uuid.UUID:
    """字符串/UUID 归一化。已是 UUID 就原样返回，避免无谓的对象重建。"""
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


def get_or_create_config(db: Session, repo_id: str | uuid.UUID, *, commit: bool = True) -> KnowledgeBaseConfig:
    """取仓库的知识库配置，没有就按当前全局向量契约创建一份。

    创建时用的是 embedding_contract()（当前环境实际生效的 provider/model/维度），
    而不是硬编码默认值，这样配置落库的那一刻就与环境一致。

    无论新建还是复用，最后都会 validate_embedding_contract 校验一遍。
    这一步是**全局向量空间一致性**的第二道闸：即使数据库里存着旧配置
    （比如模型换过一次），只要与当前环境不符就会立刻报错，
    而不是悄悄用两套向量空间混着检索。

    commit=False 用于调用方需要把「建配置」与后续操作放进同一事务的场景。
    """
    repo_uuid = as_uuid(repo_id)
    config = db.query(KnowledgeBaseConfig).filter(KnowledgeBaseConfig.repo_id == repo_uuid).one_or_none()
    if config is None:
        contract = embedding_contract()
        config = KnowledgeBaseConfig(
            repo_id=repo_uuid,
            embedding_provider=contract["provider"],
            embedding_model=contract["model"],
            embedding_dimensions=contract["dimensions"],
        )
        db.add(config)
        # flush 拿到主键，后面可能要用；是否 commit 交给调用方
        db.flush()
        if commit:
            db.commit()
            db.refresh(config)
    # 已存在的配置也要校验：这是发现「历史数据与新模型不兼容」的唯一时机
    validate_embedding_contract(
        config.embedding_provider,
        config.embedding_model,
        config.embedding_dimensions,
    )
    return config


def config_dict(config: KnowledgeBaseConfig) -> dict[str, Any]:
    """把 ORM 配置摊平成字典，用于 API 响应与 trace 快照。

    显式逐字段列出而不是 __dict__：只暴露白名单字段，
    避免把 SQLAlchemy 的内部状态（_sa_instance_state）泄漏出去。
    """
    return {
        "embedding_provider": config.embedding_provider,
        "embedding_model": config.embedding_model,
        "embedding_dimensions": config.embedding_dimensions,
        "retrieval_method": config.retrieval_method,
        "rerank_enabled": config.rerank_enabled,
        "rerank_provider": config.rerank_provider,
        "rerank_model": config.rerank_model,
        "top_k": config.top_k,
        "score_threshold_enabled": config.score_threshold_enabled,
        "score_threshold": config.score_threshold,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
    }


def model_catalog() -> dict[str, Any]:
    """探测当前环境下各模型是否真的可用，供前端展示与降级决策。

    为什么要在运行时探测而不是读配置：
    「配了 key」不等于「能用」。这里分三种情况判断 embedding 可用性——
    - 本地模型：检查 sentence_transformers 是否装了（find_spec 只查不导入，很轻）；
    - dashscope：只要有 api_key；
    - openai_compatible：embedding 专用 key 或 LLM key 任一存在即可（共用网关）；
    - deterministic：永远可用（纯本地计算）。

    embedding_models 只返回**一项**（当前生效的那个），并标注
    「固定向量空间」——这是刻意的：不允许用户在同一知识库里换 embedding 模型，
    因为换了之后历史向量全部作废。UI 上只有一个选项就不存在误操作。

    rerank_models 则给两项：Qwen API（推荐，质量好）与本地启发式（永远可用的兜底），
    并明确写出 fallback 关系，让用户知道 API 挂了会发生什么。
    """
    sentence_transformers_ready = importlib.util.find_spec("sentence_transformers") is not None
    # rerank 的 key 可以单独配，也可以复用 dashscope 的
    qwen_api_ready = bool((settings.rerank_api_key or settings.dashscope_api_key) and settings.rerank_base_url)
    contract = embedding_contract()
    provider = normalize_embedding_provider(str(contract["provider"]))
    if provider in {"qwen_local", "local"}:
        embedding_available = sentence_transformers_ready
    elif provider == "dashscope":
        embedding_available = bool(settings.dashscope_api_key)
    elif provider == "openai_compatible":
        embedding_available = bool(settings.embedding_api_key or settings.llm_api_key)
    else:
        # 只剩 deterministic：不需要任何外部依赖，因此恒为 True
        embedding_available = provider == "deterministic"
    return {
        "embedding_models": [
            {
                "provider": provider,
                "model": contract["model"],
                "label": f"{contract['model']}（固定向量空间）",
                "dimensions": contract["dimensions"],
                "available": embedding_available,
                "recommended": True,
                # embedding 没有兜底选项：换模型意味着重建全部向量
                "fallback": None,
            },
        ],
        "rerank_models": [
            {
                "provider": "qwen_api",
                "model": "qwen3-rerank",
                "label": "Qwen3-Rerank（百炼 API）",
                "available": qwen_api_ready,
                "recommended": True,
                # 明确声明降级目标，前端可以据此提示「API 不可用时会自动降级」
                "fallback": "heuristic",
            },
            {
                "provider": "heuristic",
                "model": "heuristic-reranker",
                "label": "本地启发式重排",
                "available": True,
                "recommended": False,
                "fallback": None,
            },
        ],
        "deepseek": {
            "embedding_available": False,
            "rerank_available": False,
            "note": "DeepSeek 官方 API 当前未提供 Embedding 或 Rerank 接口。",
        },
    }


def runtime_status(config: KnowledgeBaseConfig) -> dict[str, Any]:
    """把配置与 model_catalog 对照，得出「当前配置实际能不能用」。

    用 next(..., None) 而不是直接索引：配置里可能存着一个已经下线的模型名，
    这时 catalog 里找不到，应该报 available=False 而不是抛异常。

    fallback 用 (item or {}).get(...) 兜住找不到的情况，
    保证返回结构始终稳定（前端不用做空判断）。
    """
    catalog = model_catalog()
    embedding = next(
        (
            item
            for item in catalog["embedding_models"]
            if item["provider"] == config.embedding_provider and item["model"] == config.embedding_model
        ),
        None,
    )
    rerank = next(
        (
            item
            for item in catalog["rerank_models"]
            if item["provider"] == config.rerank_provider and item["model"] == config.rerank_model
        ),
        None,
    )
    return {
        "embedding": {
            "provider": config.embedding_provider,
            "model": config.embedding_model,
            "dimensions": config.embedding_dimensions,
            "available": bool(embedding and embedding["available"]),
            "fallback": (embedding or {}).get("fallback"),
        },
        "rerank": {
            "enabled": config.rerank_enabled,
            "provider": config.rerank_provider,
            "model": config.rerank_model,
            "available": bool(rerank and rerank["available"]),
            "fallback": (rerank or {}).get("fallback"),
        },
    }


async def _qwen_rerank(
    query: str,
    items: list[dict[str, Any]],
    config: KnowledgeBaseConfig,
    limit: int,
) -> list[dict[str, Any]]:
    """调用 Qwen3-Rerank（百炼 API）做语义重排。

    与本地启发式重排的关键差别：模型能理解「查询与文档的语义是否真的对应」，
    而不只是词面重合。代价是一次额外的网络请求（几十到几百毫秒）。

    endpoint 的拼法做了兼容：用户可能把 base_url 配成带 /reranks 的完整地址，
    也可能只配到域名。用 endswith 判断，两种都能用。

    documents 的构造与向量化时保持一致（标题 + 路径 + 正文），并且截到 4000 字符——
    重排模型的输入也有上下文窗口限制，超长输入会被服务端截断或直接报错，
    在客户端先切能保证行为可预期。

    **返回的 index 必须校验范围**：模型有时会返回越界下标（尤其是请求里
    documents 被服务端过滤过时），直接 items[index] 会 IndexError。

    任一环节失败都会向上抛异常，由 retrieve_knowledge 捕获后降级到启发式重排——
    这里不做内部兜底，是为了让降级决策集中在一个地方，便于统计 fallback 率。
    """
    api_key = settings.rerank_api_key or settings.dashscope_api_key
    base_url = (settings.rerank_base_url or "").rstrip("/")
    if not api_key or not base_url:
        raise RuntimeError("Qwen rerank API is not configured")
    # 兼容「base_url 已含 /reranks」与「只给到域名」两种配置
    endpoint = base_url if base_url.endswith("/reranks") else f"{base_url}/reranks"
    documents = [
        "\n".join(
            part
            for part in (
                str(item.get("title") or ""),
                str((item.get("metadata") or {}).get("path") or ""),
                # 优先用完整正文，没有才退到 snippet
                str(item.get("_raw_content") or item.get("snippet") or ""),
            )
            if part
        )[:4000]
        for item in items
    ]
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            endpoint,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": config.rerank_model,
                "query": query,
                "documents": documents,
                # top_n 夹在 [1, 文档数] 之间，避免请求非法值
                "top_n": min(max(1, limit), len(documents)),
                # instruct 是任务指令，能显著影响重排效果（告诉模型「这是在找代码证据」）
                "instruct": settings.qwen_retrieval_instruction,
            },
        )
        response.raise_for_status()
        payload = response.json()
    # 两种响应结构都兼容：顶层 results 或 output.results（不同版本/网关的差异）
    rows = payload.get("results") or payload.get("output", {}).get("results") or []
    ranked: list[dict[str, Any]] = []
    for row in rows:
        index = int(row.get("index", -1))
        # 越界即丢弃：宁可少一条，也不能取错证据（取错比没有更危险）
        if index < 0 or index >= len(items):
            continue
        score = float(row.get("relevance_score") or row.get("score") or 0.0)
        item = items[index]
        ranked.append(
            {
                **item,
                # 与启发式重排保持同样的字段语义，下游无需区分用的哪种方式
                "score": round(score, 4),
                "rerank_score": round(score, 4),
                "rank_reason": "qwen3-rerank",
                "retrieval": {
                    **(item.get("retrieval") or {}),
                    "rerank_mode": "qwen_api",
                    "rerank_model": config.rerank_model,
                    "rerank_score": round(score, 4),
                },
            }
        )
    return ranked


async def retrieve_knowledge(
    db: Session,
    repo_id: str | uuid.UUID,
    query: str,
    *,
    top_k: int | None = None,
    retrieval_method: str | None = None,
    rerank_enabled: bool | None = None,
    score_threshold_enabled: bool | None = None,
    score_threshold: float | None = None,
    source_types: list[str] | None = None,
    metadata_filters: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], int]:
    """检索总入口：返回 (候选列表, trace 字典, 耗时毫秒)。

    所有关键字参数都是「可选覆盖」语义：传 None 表示使用知识库配置里的值。
    这个设计让同一个接口既能给普通用户用（走默认配置），
    也能给调试面板用（临时改参数做对比实验）。

    详见模块 docstring 里的九步流程。下面只强调几个不直观的地方：

    - resolved_top_k 上限 20：不是性能限制，而是**上下文预算**限制。
      每条证据平均占用几百 token，20 条已经接近模型上下文的可用空间；
    - candidate_limit 的倍数（重排开 8 倍 / 关 2 倍，再夹到 [20, 100]）：
      重排要挑出 top_k，召回必须远多于 top_k 才有挑选余地；不开重排时
      只需要少量冗余，2 倍足够；
    - 只有 method 是 vector/hybrid 且来源里确实有可向量化的类型时才向量化查询，
      否则会白白花一次 embedding 调用（比如用户只要查 team_member）；
    - search_similar_documents 调用时显式关闭了 rerank 与 compression
      （apply_rerank=False / apply_compression=False），因为编排层要自己控制
      这两个阶段，避免在下层里被做两次。
    """
    started = time.perf_counter()
    config = get_or_create_config(db, repo_id)
    # top_k 夹到 [1, 20]，防止调用方传 0 或 1000
    resolved_top_k = min(max(top_k or config.top_k, 1), 20)
    method = retrieval_method or config.retrieval_method
    # 三个「配置值 + 可被参数覆盖」的三元组，用 is None 判断而不是 or，
    # 因为 False / 0.0 都是合法的显式覆盖值
    use_rerank = config.rerank_enabled if rerank_enabled is None else rerank_enabled
    threshold_on = config.score_threshold_enabled if score_threshold_enabled is None else score_threshold_enabled
    threshold = config.score_threshold if score_threshold is None else score_threshold

    requested_source_types = normalize_source_types(source_types)
    # 空列表归一成 None：语义是「不限来源」，而非「什么都不查」
    effective_source_types = requested_source_types or None
    # 不限来源时视为「可能包含向量来源」；指定了来源则逐个检查
    has_vector_sources = not effective_source_types or any(
        should_vectorize(source_type) for source_type in effective_source_types
    )
    query_embedding = None
    if method in {"vector", "hybrid"} and has_vector_sources:
        # 用 embed_knowledge_query（is_query=True）而不是 embed_text：
        # 非对称模型（Qwen3）在查询侧需要不同的 prompt 前缀
        query_embedding = embed_knowledge_query(
            query,
            provider=config.embedding_provider,
            model=config.embedding_model,
            dimensions=config.embedding_dimensions,
        )
    candidate_limit = min(max(resolved_top_k * (8 if use_rerank else 2), 20), 100)
    pipeline_trace: dict[str, Any] = {}
    items = search_similar_documents(
        db,
        repo_id,
        query,
        limit=candidate_limit,
        query_embedding=query_embedding,
        retrieval_method=method,
        # 这两个开关关掉，交给本函数后面的阶段处理
        apply_rerank=False,
        apply_compression=False,
        source_types=effective_source_types,
        metadata_filters=metadata_filters,
        pipeline_trace=pipeline_trace,
    )
    # —— 重排阶段 ——
    rerank_stage: dict[str, Any] = {
        "name": "rerank",
        "input_count": len(items),
        # requested 记录「本来想用哪种」，与实际 mode 分开，
        # 这样 fallback 时能看出「想用 qwen 但降级成了 heuristic」
        "requested": "qwen_api" if use_rerank and config.rerank_provider == "qwen_api" else "heuristic",
    }
    if use_rerank and items:
        if config.rerank_provider == "qwen_api":
            try:
                items = await _qwen_rerank(query, items, config, resolved_top_k)
                # 返回空列表等同于失败：静默返回空会让整个问答变成「无证据」
                if not items:
                    raise RuntimeError("Qwen rerank returned no valid results")
                rerank_stage.update({"mode": "qwen_api", "fallback": False})
            except Exception as exc:
                # 任何异常（网络、鉴权、响应格式）都降级到启发式，保证可用性优先。
                # 只记录异常类型名而不是完整消息：trace 要保持精简，
                # 详细信息在日志里有
                items = rerank_documents(query, items, resolved_top_k)
                rerank_stage.update(
                    {
                        "mode": "heuristic",
                        "fallback": True,
                        "fallback_reason": type(exc).__name__,
                    }
                )
        else:
            items = rerank_documents(query, items, resolved_top_k)
            rerank_stage.update({"mode": "heuristic", "fallback": False})
    elif use_rerank:
        # 开了重排但没有候选：不是错误，单独标出来便于区分「空召回」
        rerank_stage.update({"mode": "not_run", "fallback": False, "reason": "no candidates"})
    else:
        rerank_stage.update({"mode": "disabled", "fallback": False})
    items = items[:resolved_top_k]
    # setdefault 而不是直接 append：pipeline_trace 可能已由下游初始化过
    pipeline_trace.setdefault("stages", []).append({**rerank_stage, "output_count": len(items)})

    # —— 父块扩展 ——
    # neighbor_window=1 表示把相邻的子块也带上，补全被切断的上下文
    items, parent_stage = expand_parent_context(db, repo_id, items, neighbor_window=1)
    pipeline_trace["stages"].append(parent_stage)

    # —— 分数门槛 ——
    before_threshold = len(items)
    if threshold_on:
        items = [item for item in items if float(item.get("score") or 0.0) >= threshold]
    pipeline_trace["stages"].append(
        {
            "name": "score_threshold",
            "enabled": threshold_on,
            "threshold": threshold if threshold_on else None,
            "input_count": before_threshold,
            "output_count": len(items),
        }
    )

    # —— 上下文压缩 ——
    # 总预算按 top_k 线性增长，夹在 [520, 1800] token：
    # 太少装不下证据，太多会挤占模型留给回答的空间。
    # 每条最多 260 token，保证单条证据不会独吞整个预算
    context_budget_tokens = min(1800, max(520, resolved_top_k * 260))
    compression_input_count = len(items)
    items = compress_retrieval_results(
        query,
        items,
        max_tokens_per_item=260,
        max_total_tokens=context_budget_tokens,
    )
    compressed_tokens = sum(estimate_tokens(str(item.get("snippet") or "")) for item in items)
    pipeline_trace["stages"].append(
        {
            "name": "context_compression",
            "input_count": compression_input_count,
            "output_count": len(items),
            "budget_tokens": context_budget_tokens,
            "selected_tokens": compressed_tokens,
        }
    )

    # —— 答案门禁 ——
    # 放在最后：它看到的就是即将进入提示词的这批证据（压缩后），
    # 因此判定的是「真实的生成条件」而不是理想条件下的条件
    gate = evaluate_answer_gate(
        query,
        items,
        score_threshold=threshold,
        threshold_enabled=threshold_on,
    )
    # 把门禁结果**回写到每条证据**上：前端可以直接按 selected_for_context
    # 把「会进提示词的」和「仅供参考的」区分展示
    for item in items:
        retrieval = dict(item.get("retrieval") or {})
        retrieval["answer_gate"] = gate["decision"]
        retrieval["selected_for_context"] = gate["decision"] == "answer"
        item["retrieval"] = retrieval
    pipeline_trace["stages"].append({"name": "answer_gate", **gate})
    duration_ms = round((time.perf_counter() - started) * 1000)
    # trace 顶层同时给出配置快照与运行时状态，便于「同一次请求的可复现性」：
    # 只看这条 trace 就能知道当时用的什么模型、什么参数、卡在哪一步
    trace = {
        **config_dict(config),
        "top_k": resolved_top_k,
        "source_types": requested_source_types,
        "filters": {
            "repo_id": str(repo_id),
            "source_types": requested_source_types,
            "metadata": metadata_filters or {},
        },
        "runtime": runtime_status(config),
        # hybrid 的实际融合方式是加权求和，这里点明以免误解
        "fusion_method": "weighted" if method == "hybrid" else method,
        "context_budget_tokens": context_budget_tokens,
        "answer_gate": gate,
        "pipeline": pipeline_trace,
    }
    return items, trace, duration_ms


def record_retrieval_test(
    db: Session,
    repo_id: str | uuid.UUID,
    query: str,
    retrieval_config: dict[str, Any],
    results: list[dict[str, Any]],
    duration_ms: int,
) -> RetrievalTestRun:
    """把一次检索实验落库，用于后续对比调参效果。

    存储的是完整快照（配置 + 结果 + 耗时）而不是只存 query：
    调试面板需要能回放「当时这个参数组合返回了哪些证据」，
    否则调参就只能凭记忆。

    这里直接 commit（不走 commit 参数）：检索测试是一次独立动作，
    没有需要与外部事务合并的场景。
    """
    run = RetrievalTestRun(
        repo_id=as_uuid(repo_id),
        query=query,
        retrieval_config=retrieval_config,
        results_json=results,
        duration_ms=duration_ms,
        created_at=datetime.now(timezone.utc),
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run
