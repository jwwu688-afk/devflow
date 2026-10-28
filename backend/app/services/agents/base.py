"""Agent 基类模块。

Planner → 专用 Agent → Observer → Synthesis 这条编排链上，所有 Agent 共享的契约都在这里：
- name / description / available_tools 三个类属性，是编排层做「能力发现」时读取的元数据；
- run() 是唯一执行入口，签名统一为 (input_data, context) → dict。

统一签名是这条链能跑起来的前提：编排层不必按 Agent 类型写分支，一套代码即可调度任意 Agent。
这里刻意不提供任何 LLM / 工具调用的实现——各 Agent 的模型、工具、输出结构差异极大，
把公共实现抽上来反而会逼子类绕开基类。
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """所有专用 Agent 的抽象基类。

    基类只强制实现 run()，另外两个方法给的是「默认不做任何事」的兜底语义，
    免得每个 Agent 都重复写一遍空实现。
    """

    # 编排层依赖这三个属性做路由与展示，子类必须在类体中覆盖成具体值
    name: str  # 唯一标识，会被写进 trace 事件，用于定位「哪一步出的问题」
    description: str  # 自然语言职责描述，供 Planner/编排层判断该不该选它
    available_tools: list[str]  # 声明式工具清单；仅作声明，实际调用仍由子类自己完成

    @abstractmethod
    async def run(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """执行 Agent 主逻辑，返回结构化结果。

        必须是 async：编排层用 asyncio 并发调度多个 Agent，同步实现会把整条链堵死。
        input_data 是上游（Planner 或前序 Agent）产出的、本 Agent 专属的输入；
        context 是贯穿整个工作流的共享上下文（repo_id、用户信息、trace 等）。
        """
        raise NotImplementedError

    def validate_output(self, output: dict[str, Any]) -> dict[str, Any]:
        """输出校验钩子，默认原样放行。

        默认不校验是为了不让简单 Agent 被迫实现一遍；需要约束结构的子类可覆盖。
        注意：覆盖后若在此抛异常，会落到 handle_error，进而触发降级而不是中断整条链。
        """
        return output

    def handle_error(self, error: Exception, context: dict[str, Any]) -> dict[str, Any]:
        """把异常转成结构化错误字典，让编排层能继续往下跑而不是整体崩溃。

        只保留 repo_id 一个上下文字段，是因为错误结果会进日志和前端：
        塞完整 diff / prompt 之类的大对象会把日志打爆，也可能泄露用户代码。
        """
        return {
            "error": str(error),
            "agent": self.name,
            # 先判空再 str()，避免 None 被转成字符串 "None" 写进结果、误导排查方向
            "context": {"repo_id": str(context.get("repo_id")) if context.get("repo_id") else None},
        }
