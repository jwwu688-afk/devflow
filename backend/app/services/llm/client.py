"""OpenAI 兼容协议的大模型客户端（非流式 JSON / 非流式文本 / SSE 流式文本）。

为什么三种模式而不是一个通用方法：调用方对失败的容忍度不同。

- chat_json：结构化抽取（分类、打分、生成待办等），失败返回 {}，
  上层用「空字典 = 模型没说话」来降级，所以**绝不能抛异常**。
- chat_text：一次性长文本，失败返回 ""。
- chat_text_stream：面向用户打字机效果，逐 token yield。

统一走 /chat/completions + Bearer 鉴权，因此可以对接任何 OpenAI 兼容网关
（官方、vLLM、OneAPI 等），换供应商只改配置里 base_url 与 model。
"""

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.core.config import settings
from app.services.context_compression import prompt_cache_model_kwargs


class LLMClient:
    """无状态客户端：每次调用自建 httpx 连接，不做重试。

    不做重试是刻意的——上层多数场景（RAG 问答）本身有降级路径，
    在这里叠加重试会让一次请求的最坏耗时变成 3 倍，用户体感更差。
    """

    def __init__(self) -> None:
        """从全局配置读网关地址、密钥、模型名。"""
        # rstrip("/")：配置里若带尾斜杠，拼 /chat/completions 会出现 // 路径
        self.base_url = settings.llm_base_url.rstrip("/")
        self.api_key = settings.llm_api_key
        self.model = settings.llm_model

    async def chat_json(self, system_prompt: str, user_payload: dict[str, Any]) -> dict[str, Any]:
        """要求模型输出 JSON 对象，解析后返回 dict。

        输入：system_prompt 定角色与输出契约；user_payload 是结构化上下文，
        序列化时用 ensure_ascii=False 保留中文原文（否则中文变 \\uXXXX 会
        显著增加 token 数，且模型理解略差）。

        输出：解析成功的 dict；任何失败一律返回 {}，调用方需自行判空。

        response_format={"type": "json_object"} 让网关强制输出合法 JSON，
        但仍有模型吐出被截断 JSON 的可能，所以 json.loads 也在兜底范围内。
        """
        # 没配密钥时直接短路：本地开发/单测环境常常不接真实模型
        if not self.api_key:
            return {}

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            "response_format": {"type": "json_object"},
        }
        # 注入 prompt cache 相关的可选参数（例如 cache_control 标记）。
        # 这里用 update 而不是写死在 payload 里，是为了在不支持缓存的网关上
        # 返回空字典时保持 payload 完全不变
        payload.update(prompt_cache_model_kwargs())
        async with httpx.AsyncClient(timeout=60) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                return json.loads(content)
            except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
                # 五类异常覆盖：网络/状态码错误、响应结构缺字段、choices 为空、
                # 字段类型不符、JSON 解析失败。全部吞掉返回 {}，让业务层走降级
                return {}

    async def chat_text(self, system_prompt: str, user_message: str) -> str:
        """单轮文本补全，返回纯文本；HTTP 失败时返回空串。"""
        if not self.api_key:
            return ""

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        }
        payload.update(prompt_cache_model_kwargs())
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
            )
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError:
                return ""
            # 与 chat_json 不同，这里不捕获 KeyError/IndexError：
            # 200 却结构异常属于网关故障，应当显式暴露而不是静默变成空回复。
            # content 可能是 None（模型只回了 tool_call 或命中过滤），用 or "" 兜住
            return str(response.json()["choices"][0]["message"].get("content") or "")

    async def chat_text_stream(self, system_prompt: str, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        """流式文本补全，逐段 yield 增量文本（SSE 协议解析）。

        输入 messages 是完整的多轮历史，system_prompt 会被拼到最前面，
        保证角色设定在任何一轮都不会被挤掉。

        输出：一个异步生成器。注意它在无 api_key 或 HTTP 失败时**直接 return**，
        不 yield 任何内容——调用方看到的是「流正常结束但没吐字」，
        这与「真的生成空回复」在表现上无法区分，是已知的取舍。
        """
        if not self.api_key:
            return

        payload_messages = [{"role": "system", "content": system_prompt}, *messages]
        payload: dict[str, Any] = {"model": self.model, "messages": payload_messages, "stream": True}
        payload.update(prompt_cache_model_kwargs())
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
            ) as response:
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError:
                    # 状态码在拿到流之后就已知，这里提前退出比抛出更好：
                    # 前端已经建立了 SSE 连接，抛异常只会留下一条断开的流
                    return
                async for line in response.aiter_lines():
                    # SSE 每行形如 "data: {...}"，其他行（空行、event:、: 心跳注释）忽略
                    if not line.startswith("data: "):
                        continue
                    data = line.removeprefix("data: ").strip()
                    # 流结束标记，收到就停，避免把后续心跳也算进回答
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        # 单个分片坏掉不该中断整个回答，跳过这行继续读
                        continue
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content")
                    # 首个 chunk 的 delta 通常只有 role 没有 content，
                    # 所以必须判空，否则会 yield 出一堆空串
                    if content:
                        yield str(content)
