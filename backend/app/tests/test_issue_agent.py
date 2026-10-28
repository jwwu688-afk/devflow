"""Issue 分诊 Agent 的测试。

IssueAgent 要做四件事：判断类型（Bug/Feature）、判优先级与结论（是否需要澄清/可合并）、
给出处理清单、以及推荐负责人。这里重点验证：
- LLM 不可用（返回空 JSON）时的规则兜底仍然可用；
- 高相似度重复 Issue 建议合并，且相同证据去重；
- 负责人只能从团队名单里选，"桌面化"这类需求能匹配到擅长 electron 的人；
- 团队为空、或 LLM 凭空编造人名（如仓库文档里的"布偶猫"）时必须拒绝分派。
"""

import pytest

from app.services.agents.issue_agent import IssueAgent


class NoopLLM:
    """假 LLM：返回空 dict，模拟"大模型不可用/没有输出"，逼 Agent 走纯规则兜底路径。"""

    async def chat_json(self, system_prompt, user_payload):
        return {}


class InventedOwnerLLM:
    """假 LLM：故意编造一个不在团队名单里的负责人（布偶猫），
    用来验证服务端会不会对模型输出做人名校验。"""

    async def chat_json(self, system_prompt, user_payload):
        return {"suggested_owner": "布偶猫", "owner_reason": "仓库文档里提到了布偶猫。"}


@pytest.mark.asyncio
async def test_issue_agent_fallback_classifies_bug() -> None:
    """场景：LLM 返回空，Issue 带 bug 标签、标题写着 API 500。

    期望：仍能判定 category=Bug、priority=P1、结论为"等待澄清"，并给出非空 checklist。
    为什么重要：线上模型超时/限流是常态，分诊能力不能因此整体失效；
    "等待澄清"说明信息不足时不硬下结论，checklist 则告诉提交者该补什么。
    """
    result = await IssueAgent(llm=NoopLLM()).run({"title": "API returns 500", "body": "Login fails", "labels": ["bug"]}, {})
    assert result["category"] == "Bug"
    assert result["conclusion"] == "等待澄清"
    assert result["priority"] == "P1"
    assert result["checklist"]


@pytest.mark.asyncio
async def test_issue_agent_recommends_merge_for_strong_duplicate() -> None:
    """场景：提交一个 JWT 过期返回 500 的 Bug，上游已给出一个相似度 0.91 的候选 Issue。

    期望：结论为"合并到已有 Issue"，候选分数透传，并把相似 Issue 记为一条证据
    （source_type=similar_issue）。
    为什么重要：重复 Issue 会分散讨论、让同一问题被修两遍；
    阈值足够高（0.91）时应该果断合并，且合并建议必须有据可查。
    """
    result = await IssueAgent(llm=NoopLLM()).run(
        # 正文按"步骤 / 期望 / 实际 / 影响"写全，是触发高质量判定的典型输入
        {
            "title": "Login API returns 500 when token expires",
            "body": "Steps: expire JWT, call /login, expected 401 but actual 500. Impact: users cannot renew sessions.",
            "labels": ["bug"],
        },
        # 相似候选由向量检索给出，score 0.91 远高于合并阈值
        {"duplicate_candidates": [{"issue_id": "issue-1", "title": "JWT expiry returns 500", "score": 0.91}]},
    )

    assert result["conclusion"] == "合并到已有 Issue"
    assert result["duplicate_candidates"][0]["score"] == 0.91
    assert any(item["source_type"] == "similar_issue" for item in result["evidence"])


@pytest.mark.asyncio
async def test_issue_agent_dedupes_repeated_evidence() -> None:
    """场景：同一条代码证据同时出现在 code_references 和 conversation_evidence 里。

    期望：最终 evidence 里同 source_type + reference 的条目只剩 1 条。
    为什么重要：两条来源会天然重叠（代码检索和对话检索可能命中同一段代码），
    不去重会让提示词里同一段代码重复出现，浪费 token 还放大它的权重。
    """
    # 同一份证据对象被两个上游各塞了一次，正是需要去重的典型形态
    evidence = {
        "source_type": "workspace_file",
        "title": "backend/app/auth/middleware.py#refresh",
        "snippet": "Refresh expired tokens before validating the session.",
        "source_id": "code-1",
        "score": 0.8,
    }
    result = await IssueAgent(llm=NoopLLM()).run(
        {
            "title": "Expired token refresh fails",
            "body": "Steps: expire JWT, call API, expected refresh but actual 401. Impact: users lose sessions.",
            "labels": ["bug"],
        },
        {"code_references": [evidence], "conversation_evidence": [evidence]},
    )

    matching = [
        item
        for item in result["evidence"]
        if item["source_type"] == "workspace_file" and item["reference"] == "code-1"
    ]
    # 只允许一条：多余的就是同一份证据的重复副本
    assert len(matching) == 1


@pytest.mark.asyncio
async def test_issue_agent_matches_desktop_issue_to_electron_owner() -> None:
    """场景：Issue 请求"增加桌面化能力"，团队里只有一位前端工程师，其技能写着"擅长 electron、js"。

    期望：分类为 Feature；推荐负责人为该工程师；推荐理由里明确提到"桌面化/electron"的匹配依据。
    为什么重要：负责人推荐必须可解释（有理由才能让人信服地接受指派），
    而且要能把"桌面化/窗口管理"这类业务描述映射到"electron"这类技术关键词上。
    """
    result = await IssueAgent(llm=NoopLLM()).run(
        # 标题+正文同时出现"桌面化"，与团队成员的 electron 技能形成匹配信号
        {
            "title": "Feature: 增加桌面化能力",
            "body": "### What problem does this solve?\n可以把这个项目桌面化\n\n### Proposed solution\n解决需要打开网页才能使用的问题",
            "labels": [],
        },
        {
            "team_members": [
                {
                    "name": "李勇宏",
                    "role": "前端工程师",
                    "strengths": "擅长electron、js",
                    "techStack": "js, python",
                }
            ]
        },
    )

    assert result["category"] == "Feature"
    assert result["suggested_owner"] == "李勇宏"
    assert "桌面化/electron" in result["owner_reason"]


@pytest.mark.asyncio
async def test_issue_agent_does_not_assign_person_when_team_is_empty() -> None:
    """场景：团队一个人都没有，但项目文档里出现了"布偶猫 / opus"这类协作角色名。

    期望：负责人给"待分配"，理由说明"还没有添加团队成员"。
    为什么重要：文档里的角色/代号被误认成人，是这类推荐功能最典型的幻觉；
    宁可空着让用户手动指派，也不能给出一个不存在的人。
    """
    result = await IssueAgent(llm=NoopLLM()).run(
        # 故意在 project_documents 里埋入诱饵文本，检验是否会被当成候选人
        {
            "title": "Feature: 增加桌面化能力",
            "body": "希望把项目桌面化，支持 Electron 和窗口管理。",
            "labels": [],
        },
        {"project_documents": [{"title": "README", "snippet": "项目文档里提到了布偶猫 / opus 这样的协作角色。"}]},
    )

    assert result["category"] == "Feature"
    assert result["suggested_owner"] == "待分配"
    assert "还没有添加团队成员" in result["owner_reason"]


@pytest.mark.asyncio
async def test_issue_agent_rejects_llm_owner_not_in_team_members() -> None:
    """场景：LLM 硬是返回"布偶猫"作为负责人，但团队名单为空。

    期望：服务端把结果纠正为"待分配"，理由里也不能出现"布偶猫"。
    为什么重要：只靠提示词约束模型是不可靠的，必须在代码层做白名单校验
    （否则模型换个说法仍能把不存在的人塞进结果，用户照着去指派人就出事了）。
    """
    result = await IssueAgent(llm=InventedOwnerLLM()).run(
        # 团队为空 + 会编造人名的 LLM，构成"提示词失效"的最坏组合
        {
            "title": "Feature: 增加桌面化能力",
            "body": "希望把项目桌面化，支持 Electron 和窗口管理。",
            "labels": [],
        },
        {"team_members": []},
    )

    assert result["suggested_owner"] == "待分配"
    assert "布偶猫" not in result["owner_reason"]
