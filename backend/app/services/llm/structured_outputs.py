"""结构化输出的 re-export 门面。

真正的 Pydantic 模型定义在 app/schemas/analysis.py，这里只做一次转发。
为什么要多这一层：LLM 相关代码统一从 services.llm 引入输出模型，
schema 层以后怎么挪位置都不会影响业务侧 import。

这些模型与 prompts.py 里的各 PROMPT 一一对应，是「模型返回 JSON → 校验
→ 下游消费」这条链上的校验点。__all__ 显式列出，避免误导出未审阅的符号。
"""

from app.schemas.analysis import CIDebugOutput, IssueAnalysisOutput, PRReviewFinding, PRReviewOutput, SafetyOutput

__all__ = [
    "CIDebugOutput",
    "IssueAnalysisOutput",
    "PRReviewFinding",
    "PRReviewOutput",
    "SafetyOutput",
]
