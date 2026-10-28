"""评测相关模型（业务对象：离线评测集里的一条用例，以及一次评测任务的配置）。

为什么评测用例要写这么细（must_include / answer_regex / expected_tool_calls 等）：
端到端评测不能只看「答案像不像」，而要分别考核检索命中、工具选择、答案约束。
多维度断言才能定位「是检索错了还是生成错了」。
"""

from typing import Any

from pydantic import BaseModel, Field
from uuid import UUID


class RagEvalCaseRequest(BaseModel):
    """单条评测用例（业务对象：一个标准问题 + 期望答案 + 若干硬性断言）。"""
    id: str = ""  # 用例编号；为空时运行期再生成，便于临时内联用例
    question: str  # 输入问题
    reference: str  # 参考答案（用于 RAGAS 的 context recall / faithfulness 计算）
    expected_source_ids: list[str] = Field(default_factory=list)  # 期望被召回的文档 ID，用于算检索命中率
    source_type: str | None = None  # 期望的证据来源类型，为空表示不限
    required_tools: list[str] = Field(default_factory=list)  # 必须调用的工具集合，缺一个即失败
    expected_tool_calls: list[dict[str, Any]] = Field(default_factory=list)  # 期望的调用序列（含参数），用于算 tool call F1
    kind: str = "positive"  # 用例类型：positive=应当回答；negative=应当拒答/承认不知道
    must_include: list[str] = Field(default_factory=list)  # 答案中必须出现的字符串（关键词断言）
    must_not_include: list[str] = Field(default_factory=list)  # 答案中不得出现的字符串（防踩坑/防泄漏）
    answer_regex: str | None = None  # 正则断言，用于校验格式（如行号、路径）
    source_file: str | None = None  # 用例来源文件，便于回溯这套断言从哪来


class EvalRunRequest(BaseModel):
    """发起一次评测的配置（业务对象：一次评测任务的参数单）。"""
    name: str = "basic_eval"  # 批次名，报告中用于区分不同实验
    repo_id: UUID | None = None  # 限定仓库；为空表示用内置数据集
    # 默认同时跑四类评测：Issue 分类、RAG、PR 摘要、工具调用。
    # 用 lambda 提供默认值而不是直接写列表字面量，避免所有实例共享同一个 list 对象
    eval_types: list[str] = Field(default_factory=lambda: ["issue_classification", "rag", "pr_summary", "tool_calls"])
    run_judge: bool = True  # 是否调用 LLM 裁判打分；关掉可只跑规则断言（更快更省）
    top_k: int = Field(default=5, ge=1, le=20)  # 检索条数；上下限防止误传导致评测跑飞
    suite_id: str | None = None  # 使用哪个已保存的评测集
    cases: list[RagEvalCaseRequest] | None = None  # 内联用例；与 suite_id 二选一，便于快速试跑


class EvalIssueDraftRequest(BaseModel):
    """Issue 草稿评测请求（业务对象：针对单条标题生成草稿并评测其质量）。"""
    title: str | None = None  # 为空时使用内置样例，便于冒烟测试


class EvalRunResponse(BaseModel):
    """评测结果响应（业务对象：一次评测任务的总体结论）。

    result 保持 dict：不同 eval_type 产出的指标维度不同，
    强类型化会让新增评测类型都必须改这个模型，得不偿失。
    """
    eval_id: str  # 本次评测 ID（对应 eval_runs 表）
    status: str  # success / failed / running
    result: dict  # 各维度指标与门禁结论
