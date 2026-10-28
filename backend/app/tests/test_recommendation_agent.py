"""推荐 Agent（RecommendationAgent）的行为测试。

验证行为：首页给用户展示的「推荐提问」卡片，必须做到三件事：
1) 与仓库真实状态对齐——有失败 CI 才推「CI 为什么失败」，有 PR 才推「看 PR #x」，
   仓库空的时候不能凭空捏造不存在的对象；
2) 通过 exclude 参数避开用户已经看过的推荐，避免每次刷新都是同样几条；
3) 候选被穷尽后能优雅循环回退，不能返回空列表把前端搞崩。
"""

import pytest

from app.services.agents.recommendation_agent import RecommendationAgent


@pytest.mark.asyncio
async def test_recommendation_agent_avoids_excluded_prompts() -> None:
    """测的是 exclude 去重：第二次调用必须完全避开第一次给出的推荐。

    场景：仓库里有 1 个 issue、1 个 PR、1 条失败的 CI 流水线，够生成 4~5 条推荐。
    期望：两次结果的集合互不相交（isdisjoint），且各自都还凑得满 4~5 条。
    重要性：用户点「换一批」时如果重复推同一条，等于功能失效。
    """
    agent = RecommendationAgent()
    # 伪造一份「有内容但不多」的仓库快照：刚好够撑起若干条不同模板的推荐，
    # 又不足以无限刷，从而把 exclude 去重逻辑逼到边界上。
    payload = {
        "issues": [{"number": 12, "title": "Login returns 500", "state": "open"}],
        "pull_requests": [{"number": 8, "title": "Refactor auth middleware", "state": "open"}],
        "workflow_runs": [{"name": "backend-tests", "status": "completed", "conclusion": "failure"}],
        "limit": 5,
    }

    # 第一次正常生成；第二次把第一次的推荐整体塞进 exclude，模拟前端「换一批」
    first = await agent.run(payload, {})
    second = await agent.run({**payload, "exclude": first["suggestions"]}, {})

    # 两次都要凑够数量（4~5 条），说明只是换了一批而不是被去重清空
    assert 4 <= len(first["suggestions"]) <= 5
    assert 4 <= len(second["suggestions"]) <= 5
    # 两批推荐不能有任何重叠
    assert set(first["suggestions"]).isdisjoint(second["suggestions"])


@pytest.mark.asyncio
async def test_recommendation_agent_returns_contextual_prompts_for_empty_repo() -> None:
    """测的是空仓库兜底：没有任何 issue / PR / 流水线时仍返回通用型提问。

    场景：全新接入、还没同步到数据的仓库。
    期望：仍能给出 4~5 条推荐，且这些推荐里不含「失败」「PR #」「Issue #」
    这类指向具体对象的措辞——因为此时根本没有对象可指。
    重要性：空仓库是新用户的第一印象，既不能空白，也不能谎称存在数据。
    """
    result = await RecommendationAgent().run({"issues": [], "pull_requests": [], "workflow_runs": [], "limit": 5}, {})

    assert 4 <= len(result["suggestions"]) <= 5
    # 关键断言：无失败流水线、无 PR、无 issue 时，措辞里不能出现对应引用
    assert all("失败" not in prompt and "PR #" not in prompt and "Issue #" not in prompt for prompt in result["suggestions"])


@pytest.mark.asyncio
async def test_recommendation_agent_cycles_when_candidates_are_exhausted() -> None:
    """测的是候选池被穷尽后的循环回退：连续 exclude 到底也不能返回空。

    场景：空仓库下候选模板数量有限，连续三次调用并把前两次结果全部排除，
    此时候选池已被穷尽。
    期望：三次调用都仍返回非空推荐（内部按环形顺序回绕复用模板）。
    重要性：前端「换一批」是高频操作，返回空数组会让 UI 直接塌掉。
    """
    agent = RecommendationAgent()
    payload = {"issues": [], "pull_requests": [], "workflow_runs": [], "limit": 5}
    # 逐轮累积排除项，第三次时已把前面所有推荐都排掉
    first = await agent.run(payload, {})
    second = await agent.run({**payload, "exclude": first["suggestions"]}, {})
    third = await agent.run({**payload, "exclude": [*first["suggestions"], *second["suggestions"]]}, {})

    assert first["suggestions"]
    assert second["suggestions"]
    assert third["suggestions"]


@pytest.mark.asyncio
async def test_recommendation_agent_does_not_suggest_missing_pr_or_failed_ci() -> None:
    """测的是「无 PR、无流水线」时不得出现相关推荐。

    场景：仓库只有 issue，pull_requests 和 workflow_runs 均为空。
    期望：推荐文案里既不能出现 "PR"，也不能出现「CI 为什么失败」。
    重要性：这是与上一个用例互补的负向断言——上一个看「该出现什么」，
    这个看「不该出现什么」，防止模板条件判断写反导致误报。
    """
    result = await RecommendationAgent().run(
        {
            "issues": [{"number": 3, "title": "Feature request", "state": "open"}],
            "pull_requests": [],
            "workflow_runs": [],
            "limit": 5,
        },
        {},
    )

    assert all("PR" not in prompt and "CI 为什么失败" not in prompt for prompt in result["suggestions"])
