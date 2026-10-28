"""上下文压缩相关工具的测试（长对话不爆上下文的核心手段）。

对话越长，历史消息 + 工具结果越容易撑爆模型窗口。这里的函数负责分层次地瘦身：
1. compress_messages：整段历史按 token 预算裁剪，但必须保住最近的问句；
2. compress_retrieval_results：检索片段按「与查询的相关度」抽句，而不是从头截断；
3. microcompact_messages：把旧的可再生成的工具结果替换成摘要，但绝不动最终答案；
4. ensure_compaction_summary_preserves_key_facts：压缩摘要里必须保留路径/报错/待办；
5. 压力评估与 API 不变式（tool_use 与 tool_result 必须成对，不能只留一半）。
"""

import uuid
from types import SimpleNamespace

from app.services.chat_memory import ProgressiveContextManager
from app.services.context_compression import (
    MICROCOMPACT_MARKER,
    adjust_keep_start_to_preserve_api_invariants,
    calculate_context_pressure,
    compress_messages,
    compress_retrieval_results,
    ensure_compaction_summary_preserves_key_facts,
    estimate_tokens,
    fallback_memory_update,
    messages_after_compact_boundary,
    merge_memory,
    microcompact_messages,
)


def test_compress_messages_keeps_recent_turns_under_budget() -> None:
    """场景：两轮超长旧消息 + 最新一个短问句，预算只给 120 token。

    期望：最新问句一字不改地留在末尾；有消息被丢弃（omitted>0）、
    精确保留的尾部消息至少 1 条，压缩后总 token 不超过 140（略超预算的容差）。
    为什么重要：压缩如果连"用户当前问的是什么"都裁掉，回答必然跑题；
    统计字段（kept/omitted/exact_suffix）是上层判断"历史是否完整"的依据。
    """
    # 前两条各 200 次重复，属于"该被丢掉"的旧上下文
    messages = [
        {"role": "user", "content": "old setup note " * 200},
        {"role": "assistant", "content": "old answer " * 200},
        {"role": "user", "content": "What did we decide about auth middleware?"},
    ]

    compressed, stats = compress_messages(messages, max_tokens=120)

    assert compressed[-1]["content"] == "What did we decide about auth middleware?"
    assert stats["kept_messages"] < stats["input_messages"]
    assert stats["omitted_messages"] > 0
    assert stats["exact_suffix_messages"] >= 1
    assert estimate_tokens("\n".join(item["content"] for item in compressed)) <= 140


def test_retrieval_compression_prefers_query_relevant_sentence() -> None:
    """场景：一条检索结果里 100 句是无关的 billing 内容，只有最后一句讲 auth 中间件刷新 token。

    期望：snippet 保留与查询相关的那句；压缩后 token 数小于原始值；
    内部的 _raw_content 字段被清除（不把原始全文带进上下文）。
    为什么重要：检索片段常常前面大段噪声、关键句在末尾，按开头硬截断会把答案截掉；
    不清理 _raw_content 则等于"压缩了个寂寞"，token 一点没省。
    """
    results = compress_retrieval_results(
        "auth token refresh middleware",
        [
            {
                "id": "auth",
                "title": "auth docs",
                "source_type": "workspace_file",
                "source_id": "auth",
                "score": 1.0,
                "_raw_content": "billing copy. " * 100 + "The auth middleware refreshes expired tokens before validating the session.",
            }
        ],
        max_tokens_per_item=40,
    )

    assert "auth middleware refreshes expired tokens" in results[0]["snippet"]
    # 留档压缩前后 token 数，方便上层观测压缩比
    assert results[0]["compression"]["compressed_estimated_tokens"] < results[0]["compression"]["original_estimated_tokens"]
    assert "_raw_content" not in results[0]


def test_retrieval_compression_respects_global_token_budget() -> None:
    """场景：5 条同样长的检索结果，单条限 40 token，全局只给 75 token。

    期望：被保留的结果总 token <= 75；留下items 都标记 selected_for_context=True；
    条数少于 5（必然要丢弃一部分）。
    为什么重要：把 5 条都按单条上限压完仍可能超窗口；
    全局预算能保证"要么完整保留、要么整条丢掉"，避免留下半条没意义的碎片。
    """
    results = compress_retrieval_results(
        "auth token",
        [
            # 分数递减，模拟检索排序；超预算时应优先牺牲低分项
            {
                "id": f"doc-{index}",
                "score": 1.0 - index * 0.1,
                "_raw_content": ("auth token refresh evidence. " * 80),
            }
            for index in range(5)
        ],
        max_tokens_per_item=40,
        max_total_tokens=75,
    )

    assert sum(item["compression"]["compressed_estimated_tokens"] for item in results) <= 75
    # 保留在结果里的每一条都必须被标记为"已选入上下文"，否则上层可能重复添加
    assert all(item["retrieval"]["selected_for_context"] is True for item in results)
    assert len(results) < 5


def test_memory_update_merges_structured_sections() -> None:
    """场景：用户说"我喜欢简洁回答"，助手回复里带了决策和下一步；
    用规则版（非 LLM）记忆更新去做合并。

    期望：合并结果里 decisions / tasks / user_preferences 三个小节都非空。
    为什么重要：结构化记忆是跨轮次一致性的来源；
    如果只留下 summary 文本，后续就再也没法回答"我们当初怎么定的"。
    """
    update = fallback_memory_update(
        "I prefer concise answers. Can we keep token refresh in middleware?",
        "Decision: keep token refresh in middleware. Next step: add tests.",
        "direct_answer",
        [],
    )
    merged = merge_memory(
        # 用一份"空骨架"作为已有记忆，验证从零合并也能正确分区
        {"summary": "", "facts": [], "decisions": [], "open_questions": [], "tasks": [], "user_preferences": [], "repo_context": [], "citations": []},
        update,
        update["summary"],
        max_tokens=180,
    )

    assert merged["decisions"]
    assert merged["tasks"]
    assert merged["user_preferences"]


def test_pressure_uses_effective_window_stages() -> None:
    """场景：已用掉 116k token，模型是窗口较小的 gpt-4o-mini。

    期望：压力分级落在 auto_compact / aggressive / manual_path 之一（即已进入压缩区间），
    有效窗口大于 100k（不是名义窗口，而是扣掉预留后的可用值），剩余 token 不为负。
    为什么重要：分级决定"自动压缩还是让用户手动开新会话"；
    有效窗口算错会导致误判——明明快满了却以为还有很大空间。
    """
    pressure = calculate_context_pressure(116_000, model="gpt-4o-mini")

    assert pressure.stage in {"auto_compact", "aggressive", "manual_path"}
    assert pressure.effective_window_tokens > 100_000
    assert pressure.remaining_tokens >= 0


def test_microcompact_clears_old_regenerable_tool_context() -> None:
    """场景：一条又长又啰嗦的旧工具结果（前面 300 次无关内容，关键句在末尾），
    后面跟着两轮正常对话。

    期望：旧工具结果被替换、带上 MICROCOMPACT_MARKER 标记，但关键句仍保留；
    最近的助手回答原样不动；只有 1 条消息被压缩，且省下了 token。
    为什么重要：工具结果是可以重新执行得到的"可再生成"内容，压它有收益；
    而关键信息必须留下，否则模型会"忘掉"刚刚读到的代码。
    """
    messages = [
        # 第一条是可再生成的旧工具结果：内容超长，但末尾有关键结论
        {
            "role": "tool_result",
            "content": "irrelevant setup. " * 300 + "The auth middleware refreshes expired tokens before validation.",
            "route": "workspace.read_file",
            "tool_calls": [{"tool_name": "workspace.read_file", "arguments": {}, "status": "success"}],
        },
        {"role": "user", "content": "What did we decide about auth middleware?"},
        {"role": "assistant", "content": "Keep token refresh in middleware."},
    ]

    compacted, stats = microcompact_messages(messages, "auth middleware", keep_recent=2, max_excerpt_tokens=50)

    assert MICROCOMPACT_MARKER in compacted[0]["content"]
    assert "auth middleware refreshes expired tokens" in compacted[0]["content"]
    assert compacted[-1]["content"] == "Keep token refresh in middleware."
    assert stats["compacted_messages"] == 1
    assert stats["estimated_tokens_saved"] > 0


def test_compaction_summary_preserves_paths_errors_and_todos() -> None:
    """场景：一段包含"决策 + 下一步命令"的对话，和一条带文件路径与异常类型的报错。

    期望：模型写的摘要（只有一句"我们讨论了 auth 中间件"）被补全：
    路径 backend/app/auth.py、命令 pytest backend/app/tests/test_auth.py、异常名 TimeoutError
    都出现在最终摘要里，补写条数 >= 3，缺失数归零。
    为什么重要：LLM 摘要极易丢掉路径/报错/待办这类"硬事实"，
    而它们恰恰是下一轮继续干活时最需要的信息，丢了就得让用户重述一遍。
    """
    messages = [
        {
            "role": "user",
            "content": "Decision: keep token refresh in middleware.\nNext step: run pytest backend/app/tests/test_auth.py.",
        },
        {
            "role": "assistant",
            "content": "backend/app/auth.py raised TimeoutError: token refresh failed.",
        },
    ]

    summary, report = ensure_compaction_summary_preserves_key_facts(
        messages,
        # 故意给一个信息量极低的摘要，模拟模型压得太狠的情况
        "We discussed auth middleware.",
        max_tokens=180,
    )

    assert "backend/app/auth.py" in summary
    assert "pytest backend/app/tests/test_auth.py" in summary
    assert "TimeoutError" in summary
    assert report["appended_missing_count"] >= 3
    assert report["missing_count"] == 0


def test_messages_after_compact_boundary_uses_latest_boundary() -> None:
    """场景：历史里有两个压缩边界标记（first / latest），边界之间还有消息。

    期望：只从「最新」边界开始切，结果是 ["latest", "current"]。
    为什么重要：多次压缩会留下多个边界，若从最早的边界切，
    早该丢掉的旧消息会全部回到上下文，压缩等于白做。
    """
    messages = [
        {"role": "user", "content": "old"},
        {"role": "system", "content": "first", "meta": {"subtype": "compact_boundary"}},
        {"role": "assistant", "content": "middle"},
        # 第二个（最新的）边界，之后的内容才是有效上下文
        {"role": "system", "content": "latest", "meta": {"subtype": "compact_boundary"}},
        {"role": "user", "content": "current"},
    ]

    sliced = messages_after_compact_boundary(messages)

    assert [item["content"] for item in sliced] == ["latest", "current"]


def test_keep_start_moves_back_to_preserve_tool_use_result_pair() -> None:
    """场景：打算只保留最后 1 条消息（keep_start=1），但那条之前紧邻着
    assistant 的 tool_calls 和对应的 tool_result。

    期望：保留起点被回退到 0，即把整对 tool_use + tool_result 一起留下。
    为什么重要：OpenAI/Claude 的接口有硬约束——tool_result 必须紧跟在
    对应的 tool_call 之后，否则直接 400 报错，整个会话就废了。
    """
    messages = [
        {
            "role": "assistant",
            "content": "calling read",
            "tool_calls": [{"id": "call_read", "tool_name": "workspace.read_file"}],
        },
        {"role": "tool_result", "content": "file contents", "tool_call_id": "call_read"},
        {"role": "user", "content": "continue"},
    ]

    # 返回 0 表示必须从第一条开始保留，不能从中间截断这对调用/结果
    assert adjust_keep_start_to_preserve_api_invariants(messages, 1) == 0


def test_formal_compaction_keep_rows_preserves_tool_pair() -> None:
    """场景：正式压缩（写入 DB 的那条路径）时只要保留 2 条 DB 行，
    但最后 2 条里包含 tool_result，它的 tool_call 在第 1 条。

    期望：实际保留全部 3 条，即整对工具调用与结果都要留下。
    为什么重要：内存版（上一个用例）和落库版是两套代码，
    只在其中一处做不变式保护，走另一条路径照样会 400。
    """
    # 只测 _messages_to_keep 的纯逻辑，不需要真实 db，传 None 即可
    manager = ProgressiveContextManager(db=None)  # type: ignore[arg-type]
    rows = [
        SimpleNamespace(
            id=uuid.uuid4(),
            role="assistant",
            content="calling read",
            route=None,
            tool_calls=[{"id": "call_read"}],
            meta={},
        ),
        # tool_result 通过 meta.tool_call_id 与上一条的 call_read 关联
        SimpleNamespace(
            id=uuid.uuid4(),
            role="tool_result",
            content="file contents",
            route=None,
            tool_calls=[],
            meta={"tool_call_id": "call_read"},
        ),
        SimpleNamespace(id=uuid.uuid4(), role="user", content="continue", route=None, tool_calls=[], meta={}),
    ]

    kept = manager._messages_to_keep(rows, keep_count=2)  # type: ignore[arg-type]

    # 期望退化成"三条全留"，证明回退逻辑在落库路径上同样生效
    assert kept == rows


def test_microcompact_never_rewrites_assistant_final_answer() -> None:
    """场景：一条带工具调用记录的超长助手回答（最终答案本身很长）。

    期望：内容一字未改，compacted_messages 为 0。
    为什么重要：助手的历史回答一旦被改写，用户回头看到的"当时说了什么"
    就跟真实情况不一致了（也无法作为反馈/评测的证据）；
    只有可再生成的工具结果才允许压。
    """
    messages = [
        {
            "role": "assistant",
            "content": "Final answer based on workspace.read_file. " * 300,
            "route": "workspace.read_file",
            "tool_calls": [{"tool_name": "workspace.read_file", "status": "success"}],
        },
        {"role": "user", "content": "Continue."},
    ]

    compacted, stats = microcompact_messages(messages, "continue", keep_recent=1)

    # 逐字相同才算"没被改写"，不能只比长度
    assert compacted[0]["content"] == messages[0]["content"]
    assert stats["compacted_messages"] == 0
