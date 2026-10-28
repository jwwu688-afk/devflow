"""Skill 注册表（SkillRegistry）与 ChatAgent 技能运行时的测试。

验证行为：本项目把「技能」做成带 frontmatter 的 SKILL.md（含触发器、工具白名单、
工作流步骤），运行时按需加载。本文件锁住以下关键行为：
1) 内置技能注册完整，入口工具 → 技能的映射正确；
2) 自然语言能自动命中技能并加载完整正文与按需资源；
3) 资源路径做了目录逃逸防护（../../ 之类必须被拒绝）；
4) 工具调用被打上技能溯源（技能名/版本/步骤），三处展示保持一致；
5) 技能正文很贵，所以"没匹配到就不加载"；
6) 显式指定技能时，越权调用契约外的工具必须被拒绝且不执行；
7) 显式点名不存在的技能时快速失败，不消耗模型调用。
"""

from typing import Any

import pytest
from langchain_core.messages import AIMessage

from app.services.agents.chat_agent import ChatAgent
from app.services.mcp_client import StdioMcpClient
from app.services.skills import SkillRegistry, SkillRegistryError, get_skill_registry


class IssueSkillModel:
    """假模型：第一轮发起 issue 工具调用，第二轮给出最终答复。

    用 self.calls 计数区分轮次，借此真实驱动 ChatAgent 的多轮工具循环，
    而不是只测"单次调用能不能跑通"。
    """

    def __init__(self) -> None:
        self.calls = 0
        self.tool_descriptions: list[str] = []

    def bind_tools(self, tools: list[Any]) -> "IssueSkillModel":
        # 留存工具描述，用例据此断言技能说明被拼进了工具描述文本
        self.tool_descriptions = [str(tool.description) for tool in tools]
        return self

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        self.calls += 1
        if self.calls == 1:
            # 工具名用双下划线形式（agent 名 + 工具名），是内部注册时的实际调用名
            return AIMessage(
                content="",
                tool_calls=[
                    {"name": "issue_agent__analyze_issue", "args": {"message": "triage the latest issue"}, "id": "call_issue"},
                ],
            )
        return AIMessage(content="Issue triage completed.")


class MemoryPrefetchModel:
    """假模型：从不发起工具调用，只统计调用次数。

    用来验证"显式要求先查记忆"时，工具由编排层预先执行，
    模型只需消费结果作答——所以 calls 应恰好为 1。
    """

    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        self.calls += 1
        return AIMessage(content="根据 MCP 返回的历史证据，Access Token 的有效期为 15 分钟。")


class SkillRuntimeModel:
    """通用假模型：第一轮可选发起一次工具调用，之后返回完成语。

    prompts 保留每轮收到的全部消息内容，便于断言"提示词里到底注入了哪些
    技能正文与资源"。tool_name 为 None 时全程不调工具（模拟普通闲聊）。
    """

    def __init__(self, tool_name: str | None = None) -> None:
        self.calls = 0
        self.tool_name = tool_name
        self.prompts: list[str] = []

    def bind_tools(self, tools: list[Any]) -> "SkillRuntimeModel":
        return self

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        self.calls += 1
        # 用 getattr 取 content，兼容 system/human/ai 等不同类型的消息对象
        self.prompts.append("\n".join(str(getattr(message, "content", "")) for message in messages))
        if self.calls == 1 and self.tool_name:
            return AIMessage(content="", tool_calls=[{"name": self.tool_name, "args": {}, "id": "skill_call"}])
        return AIMessage(content="Skill runtime completed.")


def test_skill_registry_loads_builtin_skills() -> None:
    """测的是内置技能清单与入口映射的完整性。

    场景：直接读取全局注册表。
    期望：6 个内置技能名都在（子集断言，允许以后新增而不改本用例）；
    pr_review_agent.analyze_pr → pr-review、devflow_search_evidence →
    project-investigation；ci-debug 的首个工作流步骤为
    collect_failed_jobs_and_logs；prompt_context 里能查到 ci-debug。
    重要性：入口映射错了技能就永远不会被激活（功能静默失效）；
    prompt_context 是给模型看的"技能目录"，缺项等于模型不知道它存在。
    """
    registry = get_skill_registry()

    names = {skill.manifest.name for skill in registry.list_skills()}
    # 用 >= 子集比较：新增技能不破坏本用例，但删除或改名会让它失败
    assert {
        "issue-triage",
        "pr-review",
        "ci-debug",
        "weekly-report",
        "safety-draft",
        "project-investigation",
    } <= names
    assert registry.skill_for_entrypoint("pr_review_agent.analyze_pr").manifest.name == "pr-review"
    assert registry.skill_for_entrypoint("devflow_search_evidence").manifest.name == "project-investigation"
    # 首个步骤是固定契约，前端按顺序渲染排障流程
    assert registry.get_skill("ci-debug").manifest.workflow_steps[0] == "collect_failed_jobs_and_logs"
    assert "ci-debug" in registry.prompt_context()


def test_skill_registry_selects_and_loads_full_instructions() -> None:
    """测的是自动激活：一句自然语言即可命中技能并加载完整正文与资源。

    场景：输入「请审查 PR #86 有没有 P1/P2 风险」（未点名任何技能）。
    期望：恰好激活 pr-review、模式为 automatic；正文含该技能最具代表性的
    约束「不能用「looks good」」；带 instruction_digest（用于缓存与审计）；
    首个资源为 references/risk-checklist.md 且内容已读出。
    重要性：这是技能体系的核心价值——用户不必知道技能名。digest 与资源内容
    的存在，才证明加载的是"真正文"而非一句描述。
    """
    registry = get_skill_registry()

    activations = registry.activate("请审查 PR #86 有没有 P1/P2 风险")

    assert [item.skill_name for item in activations] == ["pr-review"]
    assert activations[0].activation_mode == "automatic"
    # 用该技能的原话验证正文全文加载，而不是只加载了摘要
    assert "不能用「looks good」" in activations[0].instructions
    assert activations[0].instruction_digest
    assert activations[0].resources[0]["path"] == "references/risk-checklist.md"
    assert "接口与数据" in activations[0].resources[0]["content"]


def test_skill_registry_explicit_selection_and_unknown_skill() -> None:
    """测的是显式指定技能：支持中文别名，未知技能要明确报错。

    场景：requested_name 传中文「CI 排障」；随后传不存在的 "not-installed"。
    期望：前者映射到 ci-debug 且模式标为 explicit；后者抛 SkillRegistryError，
    且错误信息里回显请求的名字。
    重要性：中文别名让用户能直接说"用 CI 排障"；错误里带原名字，
    用户才知道是自己拼错了还是技能真没装，而不是面对一句无用的报错。
    """
    registry = get_skill_registry()

    activation = registry.activate("请处理这个任务", requested_name="CI 排障")
    assert [item.skill_name for item in activation] == ["ci-debug"]
    # 与自动激活区分开，便于统计与排查"用户是主动选的还是系统猜的"
    assert activation[0].activation_mode == "explicit"

    with pytest.raises(SkillRegistryError, match="not-installed"):
        registry.activate("请处理这个任务", requested_name="not-installed")


def test_skill_registry_loads_only_referenced_safe_resources(tmp_path) -> None:
    """测的是按需加载资源：只有正文引用到的资源才被读进上下文。

    场景：临时目录造一个技能，正文写「需要时读取 references/checklist.md」。
    期望：activation.resources 恰好包含该文件，且内容已读出（含 Verify evidence）。
    重要性：技能资源可达数十个文件，全量加载会瞬间撑爆上下文并浪费 token；
    "只加载被引用的"是这套按需设计能成立的前提。
    """
    skill_dir = tmp_path / "resource-skill"
    references = skill_dir / "references"
    references.mkdir(parents=True)
    (references / "checklist.md").write_text("# Runtime checklist\nVerify evidence.", encoding="utf-8")
    # frontmatter 必备字段齐全（name/entrypoint/tools/triggers），否则注册会失败
    (skill_dir / "SKILL.md").write_text(
        """---
name: resource-skill
title: 资源技能
version: 0.1.0
description: 测试按需资源。
entrypoint: resource.tool
tools:
  - resource.tool
triggers:
  - 资源测试
workflow_steps:
  - verify
---

# 资源技能

需要时读取 `references/checklist.md`。
""",
        encoding="utf-8",
    )
    registry = SkillRegistry(tmp_path)

    # "资源测试" 命中 triggers；只取第一个激活结果
    activation = registry.activate("资源测试")[0]

    assert activation.resources[0]["path"] == "references/checklist.md"
    assert "Verify evidence" in activation.resources[0]["content"]


def test_skill_registry_rejects_resource_outside_skill_directory(tmp_path) -> None:
    """测的是目录逃逸防护（路径穿越）。

    场景：技能正文引用 `references/../../secret.md`，而 secret.md 位于
    技能目录之外的父目录。
    期望：抛 SkillRegistryError 且信息含「目录外资源」，内容绝不返回。
    重要性：技能包可能由第三方提供。若不校验路径，一份精心构造的 SKILL.md
    就能读取服务器任意文件（如 .env、SSH 私钥）并直接送进模型上下文，
    属于典型的任意文件读取漏洞。
    """
    skill_dir = tmp_path / "unsafe-skill"
    skill_dir.mkdir()
    # 陷阱文件：位于技能目录之外，正常加载路径不该碰到它
    (tmp_path / "secret.md").write_text("secret", encoding="utf-8")
    (skill_dir / "SKILL.md").write_text(
        """---
name: unsafe-skill
title: 越界资源技能
version: 0.1.0
description: 测试目录约束。
entrypoint: unsafe.tool
tools:
  - unsafe.tool
triggers:
  - 越界测试
---

读取 `references/../../secret.md`。
""",
        encoding="utf-8",
    )
    registry = SkillRegistry(tmp_path)

    with pytest.raises(SkillRegistryError, match="目录外资源"):
        registry.activate("越界测试")


def test_skill_registry_annotates_entrypoint_tool_calls() -> None:
    """测的是工具调用的技能溯源注解。

    场景：给一条 safety_agent.classify_action_risk 的调用记录做注解。
    期望：skill_name 为 safety-draft、版本 0.1.0、步骤含 require_confirmation。
    重要性：前端要展示"这次调用属于哪个技能的哪一步"，事后审计也依赖它；
    而 require_confirmation 直接关联"是否需要用户二次确认"的交互逻辑。
    """
    registry = get_skill_registry()

    annotated = registry.annotate_tool_call(
        {"tool_name": "safety_agent.classify_action_risk", "arguments": {"action": "comment"}, "status": "success"}
    )

    assert annotated["skill_name"] == "safety-draft"
    assert annotated["skill_version"] == "0.1.0"
    assert "require_confirmation" in annotated["skill_steps"]


@pytest.mark.asyncio
async def test_stdio_mcp_client_inspects_real_server_and_tools() -> None:
    """测的是真实 MCP 服务端的握手与工具清单（不做任何 mock）。

    场景：用 StdioMcpClient 拉起 devflow-memory-mcp 子进程做 inspect。
    期望：serverInfo 为 devflow-memory-mcp/0.1.0、协议版本 2024-11-05、
    工具集合恰为 search_evidence / get_thread_context / read_session_events。
    重要性：这是唯一的进程级集成测试，能发现"代码没问题但子进程起不来、
    协议版本不匹配、工具改名后忘了同步"这类只在真实运行时才暴露的问题。
    """
    inspection = await StdioMcpClient().inspect_server()

    assert inspection["initialize"]["serverInfo"] == {"name": "devflow-memory-mcp", "version": "0.1.0"}
    # 协议版本是客户端/服务端的硬约定，不一致会直接握手失败
    assert inspection["initialize"]["protocolVersion"] == "2024-11-05"
    assert {tool["name"] for tool in inspection["tools"]} == {
        "devflow_search_evidence",
        "devflow_get_thread_context",
        "devflow_read_session_events",
    }


@pytest.mark.asyncio
async def test_chat_agent_executes_explicit_memory_tool_before_model_answer() -> None:
    """测的是"先查记忆再作答"的预取时序。

    场景：用户明确要求先调 devflow_search_evidence；该工具返回一条历史决策
    （Token 15 分钟）和一条引用；模型自身不会发起工具调用。
    期望：模型只被调用 1 次；事件流里有 tool_use 与 tool_result；
    tool_result 标注为 project-investigation 技能；
    final.tool_calls[0].skill_version 为 0.1.0。
    重要性：记忆检索必须发生在生成之前，否则模型会先凭猜测作答、
    再"补"证据形成事后合理化。calls == 1 正是这条时序的量化证据。
    """
    registry = get_skill_registry()
    model = MemoryPrefetchModel()

    # 假工具：模拟 MCP 命中的历史决策，带一条 memory 类型的引用
    async def evidence_tool(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        return {
            "route": "memory_search",
            "tool_calls": [],
            "answer": "历史决策：Access Token 固定为 15 分钟，减少泄露后的暴露窗口。",
            "citations": [{"type": "memory", "id": "AUTH-ADR-007", "title": "令牌生命周期评审"}],
        }

    # 事件流是异步生成器，先收齐再断言（不能只看最后一条）
    events = [
        event
        async for event in ChatAgent(model=model).run_stream(  # type: ignore[arg-type]
            {"message": "请先调用 devflow_search_evidence 搜索历史证据"},
            {
                "tools": {"devflow_search_evidence": evidence_tool},
                "skill_trace_by_tool": registry.trace_metadata_by_tool(),
                "skill_description_by_tool": registry.tool_description_suffixes(),
            },
        )
    ]

    tool_use = next(event for event in events if event["event"] == "tool_use")
    tool_result = next(event for event in events if event["event"] == "tool_result")
    final = events[-1]["data"]
    # 只调度一次模型：工具结果是喂给同一次模型调用消费的
    assert model.calls == 1
    assert tool_use["data"]["tool_name"] == "devflow_search_evidence"
    # 溯源信息要落在 tool_result 上，前端据此展示"由哪个技能执行"
    assert tool_result["data"]["skill_name"] == "project-investigation"
    assert final["tool_calls"][0]["skill_version"] == "0.1.0"


@pytest.mark.asyncio
async def test_chat_agent_adds_skill_trace_to_tool_events_and_calls() -> None:
    """测的是技能溯源贯穿三处：工具描述、事件流、最终 tool_calls。

    场景：模型第一轮调 issue_agent__analyze_issue，第二轮收尾。
    期望：绑定给模型的工具描述含「注册技能：Issue 分诊」；
    tool_use / tool_result 均标 issue-triage；最终 tool_calls[0] 带 skill_name
    且 skill_steps 含 produce_action_items。
    重要性：三处任一缺失，前端"技能标识"就会在某个环节消失
    （例如卡片上有、详情弹窗里没有），这是本项目出现过的 UI 不一致问题。
    """
    registry = get_skill_registry()
    model = IssueSkillModel()

    async def issue_tool(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        return {
            "route": "issue_analysis",
            "tool_calls": [{"tool_name": "issue_agent.analyze_issue", "arguments": input_data, "status": "success"}],
            "answer": "Issue category: Bug",
            "citations": [],
        }

    agent = ChatAgent(model=model)  # type: ignore[arg-type]
    events = [
        event
        async for event in agent.run_stream(
            {"message": "triage the latest issue"},
            {
                "tools": {"issue_agent.analyze_issue": issue_tool},
                "registered_skills": registry.prompt_context(),
                "skill_trace_by_tool": registry.trace_metadata_by_tool(),
                "skill_description_by_tool": registry.tool_description_suffixes(),
            },
        )
    ]

    tool_use = next(event for event in events if event["event"] == "tool_use")
    tool_result = next(event for event in events if event["event"] == "tool_result")
    # 取最后一个 final：工具循环过程中可能产生多个 final 候选事件
    final = [event for event in events if event["event"] == "final"][-1]["data"]

    # ① 工具描述里注入了技能说明
    assert "注册技能：Issue 分诊" in model.tool_descriptions[0]
    # ② 事件流两处都带技能名
    assert tool_use["data"]["skill_name"] == "issue-triage"
    assert tool_result["data"]["skill_name"] == "issue-triage"
    # ③ 最终调用记录里既有技能名也有工作流步骤
    assert final["tool_calls"][0]["skill_name"] == "issue-triage"
    assert "produce_action_items" in final["tool_calls"][0]["skill_steps"]


@pytest.mark.asyncio
async def test_chat_agent_auto_loads_full_skill_and_records_validation() -> None:
    """测的是自动激活后"整包注入"：正文与资源都进提示词，且记录校验结果。

    场景：用户问「请审查 PR #86 有没有风险」，模型调用 PR 审查工具；
    工具内部断言 context.active_skills 里已有 pr-review 的完整正文。
    期望：提示词含该技能的硬约束与资源标题「PR 风险检查清单」，
    且不含未命中的 ci-debug 正文；
    agent_events 里有 skill_activated（automatic）与 status=completed 的
    skill_validation（entrypoint_called 为 True）。
    重要性：注入范围必须"只给命中的"，否则无关技能会挤占上下文、
    抬高成本并让模型被别的技能指令带偏；validation 则是"技能契约是否被履行"的证据。
    """
    registry = get_skill_registry()
    # 指定工具名，驱动模型第一轮发起 PR 审查调用
    model = SkillRuntimeModel(tool_name="pr_review_agent__analyze_pr")

    async def pr_tool(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        # 工具侧断言：执行到这里时技能正文必须已经注入 context
        active = context.get("active_skills") or []
        assert active[0]["skill_name"] == "pr-review"
        assert "不能用「looks good」" in active[0]["instructions"]
        return {
            "route": "pr_review",
            "tool_calls": [{"tool_name": "pr_review_agent.analyze_pr", "arguments": {}, "status": "success"}],
            "answer": "PR reviewed.",
            "citations": [],
        }

    final = await ChatAgent(model=model).run(
        {"message": "请审查 PR #86 有没有风险"},
        {
            "tools": {"pr_review_agent.analyze_pr": pr_tool},
            # _skill_registry 为内部注入点，允许传入自定义注册表（测试/多租户）
            "_skill_registry": registry,
            "registered_skills": registry.prompt_context(),
            "skill_trace_by_tool": registry.trace_metadata_by_tool(),
            "skill_description_by_tool": registry.tool_description_suffixes(),
        },
    )

    # 技能正文进了提示词
    assert any("不能用「looks good」" in prompt for prompt in model.prompts)
    # 被引用到的资源内容也进了提示词
    assert any("PR 风险检查清单" in prompt for prompt in model.prompts)
    # 未命中的 ci-debug 正文（其标志性句子）不得出现
    assert all("优先提取首个关键错误块" not in prompt for prompt in model.prompts)
    activation = next(event for event in final["agent_events"] if event["type"] == "skill_activated")
    validation = next(event for event in final["agent_events"] if event["type"] == "skill_validation")
    assert activation["skill_name"] == "pr-review"
    assert activation["activation_mode"] == "automatic"
    assert validation["status"] == "completed"
    # 入口工具确实被调用，才算技能被履行
    assert validation["entrypoint_called"] is True
    assert final["tool_calls"][0]["skill_activation_mode"] == "automatic"


@pytest.mark.asyncio
async def test_chat_agent_does_not_load_unmatched_skill_body() -> None:
    """测的是"没命中就不加载正文"，避免无谓的上下文污染。

    场景：用户只说「你好」（无工具、无触发词）。
    期望：route 为 direct_answer；提示词里明确声明"本轮没有匹配到需要加载
    完整正文的 Skill"；不含任何技能正文；agent_events 无 skill_activated。
    重要性：技能正文通常很长，闲聊时也注入会浪费大量 token，还会让模型
    误以为自己处在"审查模式"；那句显式声明则让模型知道这是有意省略。
    """
    registry = get_skill_registry()
    model = SkillRuntimeModel()

    final = await ChatAgent(model=model).run(
        {"message": "你好"},
        {
            "tools": {},
            "_skill_registry": registry,
            "registered_skills": registry.prompt_context(),
            "skill_trace_by_tool": registry.trace_metadata_by_tool(),
            "skill_description_by_tool": registry.tool_description_suffixes(),
        },
    )

    assert final["route"] == "direct_answer"
    # 显式声明"本轮无技能"，而不是静默省略
    assert any("本轮没有匹配到需要加载完整正文的 Skill" in prompt for prompt in model.prompts)
    # 任何技能正文都不该出现
    assert all("不能用「looks good」" not in prompt for prompt in model.prompts)
    assert not any(event["type"] == "skill_activated" for event in final["agent_events"])


@pytest.mark.asyncio
async def test_explicit_skill_rejects_tools_outside_its_contract() -> None:
    """测的是技能契约的强制力（最小权限原则）。

    场景：用户说「使用 pr-review Skill 审查这个变更」，但模型却发起
    issue_agent__analyze_issue 调用（不在 pr-review 的 tools 白名单内）。
    期望：issue_tool 完全没有执行（issue_called 仍为 False）；该调用被记为
    error / permission_denied；skill_validation 为 violated 并列出违规工具名。
    重要性：若放任模型在 pr-review 语境下调任意工具，技能声明的 tools 清单
    就毫无约束力，"用只读技能做写操作"这类越权行为将无法阻止。
    """
    registry = get_skill_registry()
    model = SkillRuntimeModel(tool_name="issue_agent__analyze_issue")
    issue_called = False

    async def issue_tool(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        nonlocal issue_called
        # 只要被执行就置位，最后用它验证"确实没跑"
        issue_called = True
        return {"route": "issue_analysis", "tool_calls": [], "answer": "unexpected", "citations": []}

    async def pr_tool(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        return {"route": "pr_review", "tool_calls": [], "answer": "unused", "citations": []}

    final = await ChatAgent(model=model).run(
        {"message": "使用 pr-review Skill 审查这个变更"},
        {
            "tools": {
                "issue_agent.analyze_issue": issue_tool,
                "pr_review_agent.analyze_pr": pr_tool,
            },
            "_skill_registry": registry,
            "registered_skills": registry.prompt_context(),
            "skill_trace_by_tool": registry.trace_metadata_by_tool(),
            "skill_description_by_tool": registry.tool_description_suffixes(),
        },
    )

    # 最关键的一条：越权工具的函数体一行都没跑
    assert issue_called is False
    assert final["tool_calls"][0]["status"] == "error"
    assert final["tool_calls"][0]["error_kind"] == "permission_denied"
    validation = next(event for event in final["agent_events"] if event["type"] == "skill_validation")
    assert validation["status"] == "violated"
    # 违规清单精确到工具名，便于区分"模型选错"还是"工具名配错"
    assert validation["tool_contract_violations"] == ["issue_agent.analyze_issue"]


@pytest.mark.asyncio
async def test_chat_agent_reports_unknown_explicit_skill_without_calling_model() -> None:
    """测的是显式点名未知技能时的快速失败。

    场景：用户说「使用 not-installed Skill 处理这个任务」。
    期望：route 为 skill_error；答案里回显 "not-installed"；模型调用次数为 0。
    重要性：calls == 0 是重点——先花一次模型调用再报"技能不存在"既浪费成本，
    也可能让模型编一个答案把配置错误掩盖过去。
    """
    registry = get_skill_registry()
    model = SkillRuntimeModel()

    final = await ChatAgent(model=model).run(
        {"message": "使用 not-installed Skill 处理这个任务"},
        # 只传最小上下文，验证错误路径不依赖 tools 等配置
        {"tools": {}, "_skill_registry": registry},
    )

    assert final["route"] == "skill_error"
    # 回显用户写的技能名，方便其核对拼写
    assert "not-installed" in final["answer"]
    assert model.calls == 0
