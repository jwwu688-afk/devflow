"""向量化环节：把文本转成定长浮点向量（RAG 链路的第 2 步）。

这一步在链路上的位置是「切块 → **向量化** → 检索 → 重排 → 生成 → 门禁」。
输入是若干段文本，输出是与文本一一对应的向量列表，顺序必须严格保持。

本模块支持五种 provider，可归为三类：

1. 远程 API（openai_compatible / dashscope）：走 HTTP /embeddings。
   dashscope 只是换了 api_key 与 base_url 的 openai_compatible，所以复用同一实现。
2. 本地模型（local / qwen_local）：sentence-transformers 在本机推理，
   模型只加载一次（lru_cache），适合离线或数据不能出内网的场景。
3. deterministic：纯 hash 造的伪向量，**只用于本地演示与评测**，
   没有任何语义能力，绝不能在生产检索里使用。

贯穿全文件的一条硬规则：**维度契约不可协商**。
所有向量必须恰好等于配置的 dimensions，既不截断也不补零——
原因见 _validate_vector_dimensions 的说明。
"""

import hashlib
import math
import os
import re
from functools import lru_cache
from typing import Any

import httpx

from app.core.config import settings

def _deterministic_tokens(text: str) -> list[str]:
    """把文本切成 token 序列，供 deterministic_embedding 使用。

    两种切法：
    - 英文/数字：按 [a-z0-9_-]+ 直接切词；
    - 中文：正则先取出连续汉字串，然后把**整串**和**所有相邻二字组合**都加入。
      为什么要 unigram + bigram：中文没有空格，逐字切会丢掉词序信息；
      而这里又没有分词器，用二字组近似「词」，是最轻的折中。

    注意 `segment[index : index + 2]` 依赖 Python 对 str 的按字符切片，
    而不是按字节——所以一个汉字算一个字符，二字组正好是两个汉字。
    """
    lowered = (text or "").lower()
    tokens = re.findall(r"[a-z0-9_\-]+", lowered)
    for segment in re.findall(r"[\u4e00-\u9fff]+", lowered):
        tokens.extend(segment)
        tokens.extend(segment[index : index + 2] for index in range(max(0, len(segment) - 1)))
    return tokens


def deterministic_embedding(text: str, dimensions: int = 128) -> list[float]:
    # 本地演示和 Eval 的兜底实现；生产环境请替换为 LLM embedding API。
    # 原理：每个 token 做 sha256，取前 2 字节当桶下标，在该维度上 +1（词袋哈希），
    # 最后做 L2 归一化。相同文本必然得到相同向量（可复现），但**没有语义**——
    # 「登录失败」和「登陆失败」会被映射到几乎无关的方向上。
    # 默认 128 维很短，是为了演示与单测跑得快。
    vector = [0.0] * dimensions
    for token in _deterministic_tokens(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:2], "big") % dimensions
        vector[index] += 1.0
    # `or 1.0` 处理全零向量（空文本）：避免下面除零
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def _validate_vector_dimensions(vector: list[float], dimensions: int, provider: str) -> list[float]:
    """校验单个向量维度，不符直接抛错（**不做截断/补零**）。

    「拒绝截断或补零」是本文件最重要的约束：同一个 Milvus collection 里
    所有向量必须来自同一个向量空间。如果这里偷偷把 1024 维截成 768 维，
    数据能写进去、检索也不会报错，但相似度会变成毫无意义的数字——
    这种「静默错误」比启动失败难排查一万倍。

    所以策略是 fail fast：维度不一致立刻炸，把问题暴露在写入/查询的第一时间。
    """
    if len(vector) != dimensions:
        raise ValueError(
            f"Embedding provider '{provider}' returned {len(vector)} dimensions; expected {dimensions}. "
            "Refusing to truncate or pad vectors."
        )
    return vector


def normalize_embedding_provider(provider: str) -> str:
    """把各种别名统一成五个内部规范名。

    为什么要这一层：配置项是人工填的，历史上出现过 api / openai / remote 等
    多种写法，也出现过 "sentence-transformer"（连字符）和 "sentence_transformer"
    （下划线）混用。这里先把 - 统一成 _ 再查别名表，一次解决拼写差异。

    未知取值直接抛 ValueError 而不是回落到默认值——provider 配错时如果
    悄悄用 deterministic 顶上，线上会得到「能检索但全是垃圾结果」的效果。
    """
    provider = (provider or "").strip().lower().replace("-", "_")
    aliases = {
        "api": "openai_compatible",
        "openai": "openai_compatible",
        "remote": "openai_compatible",
        "openai_compatible": "openai_compatible",
        "sentence_transformer": "local",
        "sentence_transformers": "local",
        "local_sentence_transformers": "local",
        "local": "local",
        "qwen_local": "qwen_local",
        "dashscope": "dashscope",
        "hash": "deterministic",
        "deterministic": "deterministic",
    }
    resolved = aliases.get(provider)
    if resolved is None:
        # 错误信息里用 <empty> 标出空串，否则日志里会显示成 "provider: " 看不出问题
        raise ValueError(f"Unsupported embedding provider: {provider or '<empty>'}")
    return resolved


def validate_embedding_contract(provider: str, model: str, dimensions: int) -> None:
    """校验调用方请求的 (provider, model, dimensions) 与全局配置完全一致。

    这是「向量空间唯一性」的守门人。三元组任意一项不同，产出的向量就属于
    不同空间，彼此算余弦相似度等于随机数。而它们会被写进同一个 collection，
    所以必须在入口拦住。

    报错信息里同时打印 requested 与 configured 两组值，是为了让人一眼看出
    是「代码里写死的参数过时了」还是「环境变量没配对」——这是实际排查中
    最常见的两种情况。

    注意这里会再调一次 normalize，所以 provider 的别名差异不会造成误判
    （"openai" 与 "openai_compatible" 视为相同）。
    """
    configured = (
        normalize_embedding_provider(settings.embedding_provider),
        settings.embedding_model.strip(),
        settings.embedding_dimensions,
    )
    # model 做 strip 比较：配置文件里多一个尾空格不该被判为不同模型
    requested = (normalize_embedding_provider(provider), model.strip(), dimensions)
    if requested != configured:
        raise ValueError(
            "Embedding contract mismatch: "
            f"requested provider={requested[0]}, model={requested[1]}, dimensions={requested[2]}; "
            f"configured provider={configured[0]}, model={configured[1]}, dimensions={configured[2]}. "
            "All repositories sharing the Milvus collection must use one vector space."
        )


def embedding_contract() -> dict[str, Any]:
    """导出当前生效的向量契约，通常用于落库元数据与前端展示。

    返回的是规范化之后的 provider，便于跨环境比较。
    """
    return {
        "provider": normalize_embedding_provider(settings.embedding_provider),
        "model": settings.embedding_model,
        "dimensions": settings.embedding_dimensions,
    }


def embedding_provider_name() -> str:
    """只取规范化后的 provider 名（用于日志、健康检查等轻量场景）。"""
    return normalize_embedding_provider(settings.embedding_provider)


def _embedding_payload(texts: list[str], dimensions: int, model: str) -> dict[str, Any]:
    """构造 /embeddings 请求体。

    dimensions 只在其为真值时才写入：部分服务不支持该参数（老版本 OpenAI、
    某些自建网关），无条件传会被判为非法字段直接 400。
    """
    payload: dict[str, Any] = {
        "model": model,
        "input": texts,
    }
    if dimensions:
        payload["dimensions"] = dimensions
    return payload


def _embed_openai_compatible(
    texts: list[str],
    dimensions: int,
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> list[list[float]]:
    """按 OpenAI /embeddings 协议批量取向量。

    api_key 的回落链：显式入参 → embedding 专用 key → 环境变量 OPENAI_API_KEY
    → 复用 LLM 的 key。最后一步是为了「同一个网关同时提供 chat 与 embedding」
    这种常见部署，省去重复配置。

    base_url 同理，embedding 专用地址优先，没有就用 LLM 的网关地址。

    返回前做了两件事：
    1. 按 index 排序。协议要求响应带 index 且不保证顺序，某些实现会乱序返回；
       不排序的话向量会和文本错位，写库后表现为「A 的向量挂在 B 的内容上」，
       而且完全不会报错。
    2. 逐条校验维度，长度不符直接抛错（不允许截断/补零，理由见上）。
    """
    api_key = api_key or settings.embedding_api_key or os.getenv("OPENAI_API_KEY") or settings.llm_api_key
    base_url = (base_url or settings.embedding_base_url or settings.llm_base_url).rstrip("/")
    model = model or settings.embedding_model
    if not api_key:
        raise RuntimeError("embedding API key is not configured")

    # 这里用同步 httpx.Client（不是 AsyncClient），因为本函数会被同步的
    # 索引/评测脚本直接调用；异步入口另走 embed_texts（见文件末尾说明）
    with httpx.Client(timeout=60) as client:
        response = client.post(
            f"{base_url}/embeddings",
            headers={"Authorization": f"Bearer {api_key}"},
            json=_embedding_payload(texts, dimensions, model),
        )
        response.raise_for_status()
        rows = sorted(response.json().get("data", []), key=lambda item: int(item.get("index", 0)))
        vectors = [item.get("embedding") for item in rows]
        # 条数核对：少一条或多一条都说明响应不可信，必须以条数对齐为准
        if len(vectors) != len(texts):
            raise ValueError("embedding response length mismatch")
        return [
            _validate_vector_dimensions([float(value) for value in vector], dimensions, "openai_compatible")
            for vector in vectors
        ]


@lru_cache(maxsize=4)
def _local_sentence_transformer(model_name: str, device: str | None) -> Any:
    """加载（并缓存）本地 SentenceTransformer 模型。

    lru_cache(maxsize=4) 的意义：模型加载要几百 MB 内存和数秒时间，
    绝不能每次 embed 都重建。缓存键是 (model_name, device)，所以同一进程内
    切换不同本地模型最多保留 4 个，避免无上限涨内存。

    依赖是可选的：没装 sentence-transformers 时抛带指引的 RuntimeError，
    提示去装 requirements-local-embedding.txt，而不是一个裸 ImportError。
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - optional local dependency
        raise RuntimeError(
            "sentence-transformers is not installed; install backend/requirements-local-embedding.txt"
        ) from exc

    # device 为空时不传该参数，交给库自己选（通常自动用 GPU）
    kwargs = {"device": device} if device else {}
    return SentenceTransformer(model_name, **kwargs)


def _embed_local(
    texts: list[str],
    dimensions: int,
    *,
    model_name: str | None = None,
    is_query: bool = False,
) -> list[list[float]]:
    """用本地模型编码。

    两个关键参数：
    - normalize_embeddings=True：输出即 L2 归一化，之后可直接用内积算余弦相似度，
      与 Milvus 的 IP/COSINE 度量保持一致；
    - batch_size=32：经验值，兼顾显存/内存占用与吞吐。

    is_query 的存在是因为 Qwen3-Embedding 是**非对称**模型：查询与文档要用
    不同的 prompt 前缀，效果才最好。所以只有 qwen3-embedding 系列且明确是
    查询时才设置 prompt_name="query"；其他模型传这个参数反而会报错，
    因此用条件判断而不是无条件传。
    """
    resolved_model = model_name or settings.local_embedding_model
    model = _local_sentence_transformer(resolved_model, settings.local_embedding_device)
    encode_kwargs: dict[str, Any] = {}
    if is_query and "qwen3-embedding" in resolved_model.lower():
        encode_kwargs["prompt_name"] = "query"
    vectors = model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
        **encode_kwargs,
    )
    output: list[list[float]] = []
    for vector in vectors:
        # 兼容 numpy 数组（有 tolist）与普通序列，最终统一成 Python float 列表
        values = vector.tolist() if hasattr(vector, "tolist") else list(vector)
        output.append(_validate_vector_dimensions([float(value) for value in values], dimensions, "local"))
    return output


def embed_texts(texts: list[str], dimensions: int | None = None) -> list[list[float]]:
    """批量向量化（用全局配置的 provider/model），不区分查询与文档。

    这是给「索引写入」用的默认入口——写库时不需要 query prompt 语义。
    dimensions 不传时取配置值。
    """
    dimensions = dimensions or settings.embedding_dimensions
    return embed_knowledge_texts(
        texts,
        provider=settings.embedding_provider,
        model=settings.embedding_model,
        dimensions=dimensions,
    )


def embed_text(text: str, dimensions: int | None = None) -> list[float]:
    """单条文本向量化，取批量结果的第 0 项。"""
    return embed_texts([text], dimensions=dimensions)[0]


def embed_knowledge_texts(
    texts: list[str],
    *,
    provider: str,
    model: str,
    dimensions: int,
    is_query: bool = False,
) -> list[list[float]]:
    """向量化总调度：先校验契约，再按 provider 分发。

    执行顺序是刻意的：
    1. 空输入直接返回 []（连契约校验都跳过，避免无谓的配置检查）；
    2. validate_embedding_contract 先行——在真正花网络/GPU 计算之前就把
       「参数与配置不符」的错误拦下来；
    3. 再 normalize 一次拿到规范名做分发。

    `text or ""` 把 None 兜成空串：部分来源（如空文件）会传来 None，
    而各个 provider 对 None 的处理不一致，统一成空串更可控。
    """
    clean_texts = [text or "" for text in texts]
    if not clean_texts:
        return []
    validate_embedding_contract(provider, model, dimensions)
    resolved = normalize_embedding_provider(provider)
    # qwen_local 与 local 走同一套本地实现，区别仅在模型名与是否用 query prompt
    if resolved in {"qwen_local", "local"}:
        return _embed_local(clean_texts, dimensions, model_name=model, is_query=is_query)
    # dashscope 本质是「换 key + 换 base_url 的 openai 兼容接口」
    if resolved == "dashscope":
        return _embed_openai_compatible(
            clean_texts,
            dimensions,
            model=model,
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_embedding_base_url,
        )
    if resolved == "openai_compatible":
        return _embed_openai_compatible(clean_texts, dimensions, model=model)
    if resolved == "deterministic":
        return [deterministic_embedding(text, dimensions) for text in clean_texts]
    # 理论上 normalize 已经挡住了未知值，这里只是防御性兜底（新增分支时忘了加实现）
    raise ValueError(f"Unsupported embedding provider: {resolved}")


def embed_knowledge_query(text: str, *, provider: str, model: str, dimensions: int) -> list[float]:
    """向量化「查询串」（检索时的入口），与文档向量必须使用同一契约。

    与 embed_texts 的区别只有 is_query=True：让非对称模型（Qwen3）走 query
    侧的前缀。如果漏掉这个参数，检索精度会明显下降但不会报错，
    所以查询侧必须显式调用本函数，不要图省事用 embed_text。
    """
    return embed_knowledge_texts(
        [text],
        provider=provider,
        model=model,
        dimensions=dimensions,
        is_query=True,
    )[0]
