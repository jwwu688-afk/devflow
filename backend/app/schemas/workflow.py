"""多智能体工作流的领域模型（业务对象：一次「规划 → 并行执行 → 交叉复核 → 汇总」的结构化记录）。

三个层次，读代码时按这个顺序理解：
1. WorkflowSpec —— 计划书：目标 + 任务图（谁是节点、谁依赖谁）；
2. WorkflowTaskResult / WorkflowObservation —— 执行与复核结果；
3. WorkflowReport —— 打包好的最终产物，直接落库并返回给前端。

为什么全部用具名模型而不是裸 dict：任务图是「模型生成的产物」，用严格模型解析可以让
格式错误在边界处就暴露（而不是在调度器里莫名其妙 KeyError），这是本项目控制幻觉的主要手段。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


# 任务状态机：pending（待执行）→ running → success / error；skipped 用于依赖失败后的连带跳过
WorkflowTaskStatus = Literal["pending", "running", "success", "skipped", "error"]
# 复核意见的严重度：info 仅提示，warning 建议处理，blocker 阻塞最终交付
WorkflowFindingSeverity = Literal["info", "warning", "blocker"]


def _workflow_id() -> str:
    """生成工作流 ID：固定前缀 + 12 位随机十六进制。

    不用完整 UUID 是为了让日志里一眼能看出「这是 workflow 而不是别的实体」，
    12 位（48 bit）在单项目量级下碰撞概率可忽略，且短 ID 便于人工复制排查。
    """
    return f"workflow_{uuid.uuid4().hex[:12]}"


class WorkflowClaim(BaseModel):
    """任务的「声明」（业务对象：该任务在执行前对自身边界的自述）。

    先声明再执行，是这一层的核心设计：
    - allowed_sources 限定可读的数据源，防止子任务越权乱翻；
    - write_intent 提前暴露「我要写操作」，让调度器/人工在执行前就能拦截。
    """
    target_type: str  # 目标对象类型（issue / pr / ci_run ...）
    target_id: str | None = None  # 目标对象 ID；全局任务可为空
    allowed_sources: list[str] = Field(default_factory=list)  # 允许读取的数据源白名单
    write_intent: bool = False  # 是否打算执行写操作（默认 False，最小权限）
    reason: str = ""  # 声明的理由，便于人工判断是否合理


class WorkflowTask(BaseModel):
    """任务节点（业务对象：分配给某个 Agent 的一个子任务）。"""
    id: str  # 任务 ID，与 dependencies 中的引用对应
    agent_name: str  # 执行该任务的 Agent 角色
    task_type: str  # 任务类型，决定 prompt 与工具集
    objective: str  # 该任务要达成的具体目标（自然语言，直接进 prompt）
    dependencies: list[str] = Field(default_factory=list)  # 前置任务 ID；调度器据此决定拓扑顺序
    claim: WorkflowClaim  # 执行前的边界声明
    input_hint: dict[str, Any] = Field(default_factory=dict)  # 建议的输入（如要看的文件名），仅为提示不强制
    critical: bool = False  # 是否关键任务；关键任务失败则整个工作流判定失败


class WorkflowSpec(BaseModel):
    """工作流计划书（业务对象：一次多 Agent 协作的完整蓝图）。

    它由规划器生成、可被用户修改后再执行（对应 ChatPlanExecuteRequest.workflow_spec），
    因此这里的默认值同时承担「让模型少填字段」和「让前端表单有合理初值」两个职责。
    """
    workflow_id: str = Field(default_factory=_workflow_id)  # 用 factory 而非固定值，保证每个实例唯一
    goal: str  # 用户总目标
    trigger_message: str  # 触发本次工作流的原始提问，保留用于复盘
    acceptance_criteria: list[str] = Field(default_factory=list)  # 验收标准，汇总阶段据此自检
    tasks: list[WorkflowTask] = Field(default_factory=list)  # 任务节点列表（构成 DAG）
    # 并发上限 1~8：太小浪费并行度，太大容易触发模型侧限流并让日志难以阅读
    max_parallel_tasks: int = Field(default=3, ge=1, le=8)
    requires_human_confirmation: bool = False  # 是否需人工确认后再执行（含写操作时应置 True）
    # 用带时区的 UTC 时间：工作流可能跨机器/跨时区调度，naive datetime 会导致排序错乱
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class WorkflowTaskResult(BaseModel):
    """单个任务的执行结果（业务对象：某个子任务跑完后的产出与自评）。"""
    task_id: str  # 对应 WorkflowTask.id
    agent_name: str  # 实际执行的 Agent
    task_type: str  # 任务类型（冗余保存，便于单独查询结果时无需回表计划书）
    status: WorkflowTaskStatus
    summary: str = ""  # 结果摘要，汇总阶段主要读这个字段
    output: dict[str, Any] = Field(default_factory=dict)  # 结构化产出，结构随任务类型变化
    evidence_count: int = 0  # 引用证据条数；为 0 往往意味着答案缺乏依据
    confidence: float | None = Field(default=None, ge=0, le=1)  # 自评置信度；None 表示未评估
    citations: list[dict[str, Any]] = Field(default_factory=list)  # 引用来源
    error: str | None = None  # 失败原因
    started_at: datetime | None = None
    completed_at: datetime | None = None  # 与 started_at 之差用于定位慢任务


class WorkflowFinding(BaseModel):
    """复核意见（业务对象：观察员 Agent 对某个任务/整体结果提出的一条质疑）。"""
    finding_type: str  # 意见类型（缺证据 / 结论矛盾 / 越权等）
    severity: WorkflowFindingSeverity  # 严重度，决定是否影响交付
    message: str  # 具体描述
    task_ids: list[str] = Field(default_factory=list)  # 涉及的任务；空表示针对整体
    recommendation: str = ""  # 建议的处理方式


class WorkflowObservation(BaseModel):
    """交叉复核结论（业务对象：观察员 Agent 对整轮执行的体检报告）。

    这是「自检环节」：让另一个 Agent 专门挑毛病，比让执行者自我评价更能暴露问题。
    """
    observer_agent: str = "observer_agent"  # 复核者角色名
    findings: list[WorkflowFinding] = Field(default_factory=list)  # 发现的问题清单
    overall_confidence: float = Field(default=0.0, ge=0, le=1)  # 整体置信度；默认 0 表示「未经评估不要轻信」
    human_review_required: bool = False  # 是否建议转人工复核
    summary: str = ""  # 复核总结


class WorkflowReport(BaseModel):
    """工作流最终报告（业务对象：一次完整协作的交付物，直接落库/返回前端）。

    把 spec、各任务结果、复核意见、工具轨迹全部打包在一起：
    报告本身即「可复盘的证据链」，不必再去关联查询多张表。
    """
    spec: WorkflowSpec  # 原始计划书
    task_results: list[WorkflowTaskResult]  # 各任务执行结果
    observation: WorkflowObservation  # 复核结论
    final_answer: str  # 汇总后的回答
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)  # 全链路工具调用（跨任务合并）
    citations: list[dict[str, Any]] = Field(default_factory=list)  # 全链路引用（去重后）
    metrics: dict[str, Any] = Field(default_factory=dict)  # 耗时、token、并发度等指标
    agent_events: list[dict[str, Any]] = Field(default_factory=list)  # 过程事件流，供前端回放
