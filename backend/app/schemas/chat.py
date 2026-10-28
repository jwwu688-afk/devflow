"""聊天接口的请求/响应模型（业务对象：一次问答的输入，以及返回给前端的完整结果）。

设计要点：
- 响应体刻意做得「宽」：除了 answer 还带回工具调用轨迹、引用、压缩统计、token 用量。
  因为这些是前端「可解释面板」的数据来源，若拆成多个接口会造成一次提问多次往返。
- 大量字段用 dict / list[dict] 承载：结构随 Agent 能力演进，强类型化会带来频繁的 schema 迁移。
"""

from uuid import UUID

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """发起问答的请求体（业务对象：用户在一个仓库会话里提出的一句话）。"""
    repo_id: UUID  # 必填：检索范围、记忆隔离都以仓库为界
    conversation_id: UUID | None = None  # 为空表示新建会话，服务端会创建并回填
    message: str  # 用户提问原文
    context: dict = Field(default_factory=dict)  # 前端附加的上下文（当前打开的文件、diff 等），可选


class ChatPlanExecuteRequest(ChatRequest):
    """「按计划执行」请求体（业务对象：用户确认工作流计划后，要求按该计划跑一遍）。

    继承 ChatRequest 复用字段，额外带上 workflow_spec：
    这样同一条提问既能走自动路由，也能被用户钉死成某个既定计划，便于复现与调参。
    """
    workflow_spec: dict  # 规划阶段产出的任务图/步骤定义


class ToolCallTrace(BaseModel):
    """一次工具调用的执行轨迹（业务对象：Agent 调用某个工具的过程与结果）。

    这张「轨迹」是排查问题的第一手材料：
    模型为什么这么答，往往取决于它调了什么工具、拿到什么、失败了重试几次。
    """
    tool_name: str  # 被调用的工具名
    arguments: dict  # 实际入参（已脱敏）
    status: str  # success / error / skipped
    error: str | None = None  # 错误信息（给人看）
    error_kind: str | None = None  # 错误分类（给程序判断：timeout / validation / upstream ...）
    retryable: bool | None = None  # 是否可重试；None 表示未判定
    attempts: int | None = None  # 实际尝试次数，>1 说明发生过重试
    tool_use_id: str | None = None  # 模型侧的工具调用 ID，用于把结果与请求配对
    # 以下 skill_* 字段描述「技能（skill）」的激活情况：
    # 技能是工具的上一层封装（带步骤与指令），记录这些便于分析激活是否合理
    skill_name: str | None = None
    skill_title: str | None = None
    skill_version: str | None = None
    skill_steps: list[str] = Field(default_factory=list)
    skill_activation_mode: str | None = None  # 激活方式：auto / manual
    skill_activation_reason: str | None = None  # 为什么激活该技能，用于评估路由准确率
    skill_instruction_digest: str | None = None  # 指令摘要（通常是指令文本的哈希），用于追溯当时的 prompt 版本


class ChatResponse(BaseModel):
    """问答响应体（业务对象：一条完整的助手回答及其全部运行元数据）。

    字段较杂但都是前端一次性渲染所需：正文、引用、工具轨迹、上下文压缩情况、token 用量。
    """
    # 用 UUID | str 而不是纯 UUID：历史数据或降级路径可能给字符串 ID，
    # 联合类型避免为了兼容旧记录而让整个响应 500
    run_id: UUID | str  # 本次 Agent 运行 ID，对应 agent_runs 表
    message_id: UUID | str  # 落库后的助手消息 ID，前端据此提交反馈
    route: str  # 命中的处理链路：rag / agent / chat ...，可解释「为什么这样答」
    tool_calls: list[ToolCallTrace]  # 工具调用轨迹
    agent_events: list[dict] = []  # Agent 过程事件流（思考、规划、子任务等），结构随实现演进
    answer: str  # 最终回答正文
    citations: list[dict] = []  # 引用列表，前端渲染角标与来源卡片
    completion: dict = {}  # 模型补全统计（结束原因、耗时等）
    compression_stats: dict = {}  # 上下文压缩统计（压缩前后 token 数）
    progressive_compaction: dict = {}  # 渐进式压缩的中间状态，用于解释「历史被压掉了多少」
    model_usage: list[dict] = []  # 各模型的 token 用量明细（可能一次请求调用多个模型）
    context_diagnostics: dict = {}  # 上下文预算诊断：各分区实际占用 vs 配额，调 prompt 时主要看它


class ChatHistoryMessage(BaseModel):
    """历史消息条目（业务对象：会话回放时的一条消息）。

    与 ChatResponse 的区别：多 role / created_at，少 answer/tool_calls 的强结构，
    因为历史里 user 消息没有工具轨迹，assistant 消息才可能有。
    """
    id: UUID | str
    conversation_id: UUID | str | None = None  # 允许为空，兼容早期未归档的消息
    role: str  # user / assistant / system / tool
    content: str
    route: str | None = None  # 仅助手消息有值
    tool_calls: list[ToolCallTrace] = []
    agent_events: list[dict] = []
    compression_stats: dict = {}
    progressive_compaction: dict = {}
    model_usage: list[dict] = []
    context_diagnostics: dict = {}
    # 用字符串而不是 datetime：历史接口可能直接透传数据库里的 ISO 字符串，避免二次解析失败
    created_at: str | None = None


class PromptRecommendationRequest(BaseModel):
    """提示词推荐请求（业务对象：为首页生成几条「你可以这样问」的示例问题）。"""
    repo_id: UUID  # 依据哪个仓库的上下文来推荐（不同仓库的推荐问题应不同）
    exclude: list[str] = Field(default_factory=list)  # 已展示过的问题，避免重复推荐
    # 限制 4~5 条：少于 4 条首页会显得空，多于 5 条会挤占输入框注意力，属于刻意的产品约束
    limit: int = Field(default=5, ge=4, le=5)


class PromptRecommendationResponse(BaseModel):
    """提示词推荐结果（业务对象：推荐问题列表 + 生成它的 Agent 名称）。"""
    agent: str  # 哪个 Agent 生成的推荐，便于 A/B 对比
    suggestions: list[str]  # 推荐问题文案
