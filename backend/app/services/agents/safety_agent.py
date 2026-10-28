"""操作安全闸门 Agent。

职责边界：判断「某个动作是否需要人工确认」。它是写操作前的最后一道卡口，
位于 Agent 产出结果之后、真正对外发送（评论 / 建 issue / 发报告）之前。

关键设计：判定**用关键字规则而不是 LLM**。
- 安全判定必须确定、可复现、零延迟、零成本，模型的不确定性在这里是缺陷不是特性；
- 代价是只能识别词表内的动作，词表外的写操作会被判成 "low" 直接放行，
  所以新增写能力时必须同步扩充下面的 token 列表。
"""

from typing import Any

from app.schemas.analysis import SafetyOutput
from app.services.agents.base import BaseAgent


class SafetyAgent(BaseAgent):
    """基于关键字匹配的风险分级器（MVP 版）。"""

    name = "safety_agent"
    description = "检查某个操作是否需要人工确认。"
    available_tools = ["classify_action_risk"]

    async def run(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """返回 SafetyOutput 的 dict 形式：requires_confirmation / risk_level / reason / draft。

        输入：input_data["action"] 是动作标识（中英文都可能出现，因此词表也准备了中英两套）；
              input_data["draft"] 是待确认的草稿内容，原样透传，不在这里生成。
        输出：模型序列化后的 dict，供编排层决定是暂停等待用户确认还是继续执行。
        """
        # lower() 只对英文有效，中文不受影响；统一大小写是为了让 "Post" / "POST" 都能命中
        action = input_data.get("action", "").lower()
        # 子串匹配（不是精确相等）：动作名常带上下文前缀，如 "pr_comment"、"create_issue_draft"，
        # 用子串能覆盖这些变体；副作用是可能误报（例如 "postpone" 含 "post"），
        # 但安全场景下误报只是多重一次确认，漏报才是事故，所以宁可宽进
        risky = any(
            token in action
            for token in [
                # 英文动作词表
                "comment",
                "create_issue",
                "label",
                "close_issue",
                "send_report",
                "write",
                "post",
                "send",
                # 中文动作词表：前端/AI 有时直接传中文动作名，两套词表缺一不可
                "评论",
                "创建",
                "新建",
                "发给",
                "发送",
                "改标签",
                "关闭",
                "写回",
            ]
        )
        # 风险等级只区分 medium / low：MVP 阶段所有写操作都只产出草稿、不直接落库，
        # 真正的不可逆风险不存在，因此没有 high 档；等开放「自动执行」时才需要补上
        return SafetyOutput(
            requires_confirmation=risky,
            risk_level="medium" if risky else "low",
            # reason 会被直接展示给用户解释「为什么要你确认」，所以写成完整句子
            reason="MVP 阶段所有外部写操作只生成草稿。" if risky else "只读分析操作不需要人工确认。",
            # 草稿原样带上，让确认弹窗能直接预览要发出去的内容
            draft=input_data.get("draft"),
        ).model_dump()
