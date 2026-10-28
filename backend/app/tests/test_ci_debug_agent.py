"""CI 失败诊断 Agent 的单元测试。

验证 CIDebugAgent 拿到一份「GitHub Actions 工作流运行结果」后，
能正确识别失败类型（failure_type）并生成包含关键信息的失败摘要。
"""

import pytest

from app.services.agents.ci_debug_agent import CIDebugAgent


@pytest.mark.asyncio
async def test_ci_debug_agent_includes_failed_job_context() -> None:
    """场景：一次 backend-test 工作流失败，其中 pytest 步骤挂了，日志里带 AssertionError。

    期望：failure_type 被判定为 "test"（断言/测试失败，而不是构建或依赖问题），
    且 failure_summary 里保留了 "pytest" 这个关键线索。
    为什么重要：CI 失败原因分类决定后续给用户的修复建议（改测试 vs 改构建配置），
    如果分类错了，Agent 会给出完全无关的排查方向。
    """
    # 构造最小化的 workflow run 结构：整体 failure，jobs 里只有 pytest 一个失败步骤
    result = await CIDebugAgent().run(
        {
            "name": "backend-test",
            "status": "completed",
            "conclusion": "failure",
            "jobs": [{"name": "pytest", "conclusion": "failure", "steps": [{"name": "Run tests", "conclusion": "failure"}]}],
            # logs_text 伪造一段 pytest 失败日志，用于触发「测试失败」分类规则
            "logs_text": "pytest tests/test_api.py FAILED AssertionError",
        },
        # 第二个参数是上下文（是否已尝试过的修复等），这里不需要，传空字典
        {},
    )
    assert result["failure_type"] == "test"
    assert "pytest" in result["failure_summary"]
