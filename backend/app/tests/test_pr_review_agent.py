"""PR 代码评审 Agent 的测试。

PRReviewAgent 要输出三样东西：findings（问题清单，带严重级别与是否阻塞）、
blocking_issues（必须处理完才能合入的项）、merge_recommendation（合入建议）。
本文件验证两条关键行为：
1. LLM 不可用时，纯规则也能发现问题（CI 失败、大改动、缺少测试证据）并给出"暂缓"；
2. LLM 说"建议合入"但它自己列了 blocking=True 的问题时，服务端必须把结论降级为"修改后合入"。
   这条最重要——模型的乐观结论不能被直接当成放行依据。
"""

import pytest

from app.services.agents.pr_review_agent import PRReviewAgent


class NoopLLM:
    """假 LLM：返回空 dict，用于验证规则兜底产出的 finding 质量。"""

    async def chat_json(self, system_prompt, user_payload):
        return {}


class OverconfidentLLM:
    """假 LLM：结论是"建议合入"，但 findings 里却标了 blocking=True 的问题。

    这种自相矛盾的输出在真实模型上并不罕见，用它来验证服务端的"结论降级"逻辑。
    """

    async def chat_json(self, system_prompt, user_payload):
        return {
            "merge_recommendation": "建议合入",
            "review_findings": [
                {
                    "severity": "P2",
                    "title": "权限边界缺少验证",
                    "evidence": "diff 修改了权限判断，但没有对应测试证据。",
                    "required_action": "补充权限边界测试或人工验证记录。",
                    "blocking": True,
                }
            ],
            "blocking_issues": [],
        }


@pytest.mark.asyncio
async def test_pr_review_fallback_emits_blocking_findings() -> None:
    """场景：LLM 不可用；这个 PR 改动了鉴权中间件、+320/-12 行、CI 有 1 次失败、还有一条评论
    提醒"确认 token 不会被打印到日志"。

    期望：至少产出 P1 和 P2 两个级别的问题，merge_recommendation 为"暂缓"，blocking_issues 非空。
    为什么重要：鉴权代码 + 大改动 + CI 失败三者叠加时，无模型也必须能拦下；
    如果兜底只会给"P3 / 可以合入"，那没有模型时评审就形同虚设（等于放行高危改动）。
    """
    result = await PRReviewAgent(llm=NoopLLM()).run(
        # 字段刻意凑齐触发规则：鉴权路径文件名、超大 diff、CI 失败、含敏感词评论
        {
            "number": 12,
            "state": "open",
            "author": "alice",
            "title": "Update auth middleware",
            "body": "Refactor permission checks.",
            "files": [
                {
                    "filename": "backend/app/auth/middleware.py",
                    "additions": 320,
                    "deletions": 12,
                    "patch": "permission change",
                }
            ],
            "comments": [{"body": "Please verify token handling is not logged."}],
        },
        # CI 有失败运行 → 触发"不能直接合入"的判断
        {"ci_summary": {"failed_runs": 1}},
    )

    severities = {item["severity"] for item in result["review_findings"]}
    # P1（鉴权改动/CI 失败）与 P2（缺少测试证据等）都应命中
    assert {"P1", "P2"} <= severities
    assert result["merge_recommendation"] == "暂缓"
    assert result["blocking_issues"]


@pytest.mark.asyncio
async def test_pr_review_downgrades_merge_when_llm_returns_blocking_finding() -> None:
    """场景：LLM 明确说"建议合入"，但同时给出一条 blocking=True 的 P2 问题（权限边界缺验证）；
    PR 本身改动很小、CI 也没有失败信息。

    期望：最终结论被降级为"修改后合入"，blocking_issues 里包含该问题的标题，
    finding 的严重级别保持 P2 不被篡改。
    为什么重要：合入建议直接决定代码是否上线；
    只要存在阻塞项，就不能因为模型一句"建议合入"而放行——这是最后一道安全阀。
    """
    result = await PRReviewAgent(llm=OverconfidentLLM()).run(
        # PR 数据本身很"干净"，排除其它规则干扰，确保结论变化只来自 blocking finding
        {
            "number": 13,
            "state": "open",
            "author": "alice",
            "title": "Update permission UI",
            "body": "Ready to merge.",
            "files": [
                {
                    "filename": "frontend/app/page.tsx",
                    "additions": 20,
                    "deletions": 3,
                    "patch": "permission UI change",
                }
            ],
            "comments": [],
        },
        {},
    )

    assert result["merge_recommendation"] == "修改后合入"
    # 阻塞项标题要透传到 blocking_issues，前端据此展示"必须处理"清单
    assert any("权限边界缺少验证" in item for item in result["blocking_issues"])
    # 原始严重级别保持不变（降级的是结论，不是 finding 本身）
    assert result["review_findings"][0]["severity"] == "P2"
