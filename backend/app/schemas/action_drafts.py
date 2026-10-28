"""动作草稿相关出入参模型（业务对象：AI 拟好、等待人工确认后才会执行的高危操作）。

本模块只描述「接口边界」的数据形状，不承载业务规则；
真正的风险判定在服务层，schema 负责让非法请求在进门时就被 422 拦下。
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class ActionDraftCreateRequest(BaseModel):
    """创建动作草稿的请求体（业务对象：一次「让 AI 帮我提 issue/评论/建 PR」的委托）。

    注意这里没有 status 字段：草稿状态由服务端强制初始化为待确认（pending_confirmation），
    不接受客户端指定，避免有人直接传 executed 绕过人工确认环节。
    """
    repo_id: UUID  # 目标仓库；必填，因为草稿的所有权限与索引都挂在仓库下
    draft_type: str = Field(default="issue_comment")  # 动作类型，默认「在 issue 下评论」这一最低风险动作
    target_type: str = Field(default="issue")  # 作用对象类型，默认 issue
    target_id: UUID | None = None  # 作用对象 ID；创建类动作（如新建 issue）可以没有目标
    title: str = ""  # 草稿标题，允许为空（评论类动作没有标题）
    content: str  # 草稿正文，必填——没有正文的草稿没有意义
    risk_level: str = "medium"  # 客户端可初判风险，服务端仍会复核；默认 medium 偏保守
    metadata: dict = Field(default_factory=dict)  # 扩展信息（模型名、来源 trace 等）；用 factory 避免可变默认值共享


class ActionDraftResponse(BaseModel):
    """动作草稿的返回体（业务对象：前端列表/详情页展示的一条草稿）。

    从 ORM 对象直接转换（from_attributes），因此字段名需与 ActionDraft 模型保持一致。
    """
    id: UUID
    repo_id: UUID
    draft_type: str
    target_type: str
    target_id: UUID | None = None
    title: str
    content: str
    risk_level: str
    status: str  # pending_confirmation / confirmed / executed / failed，由服务端推进
    execution_result: dict | None = None  # 执行结果（如新 PR 链接）；未执行时为 None
    error_message: str | None = None  # 执行失败原因
    # 列名叫 meta（SQLAlchemy 里 metadata 是保留字），对外 JSON 暴露成 metadata 更符合直觉；
    # populate_by_name=True 让两种键名都能被接受，便于前端与内部代码各用各的
    metadata: dict = Field(default_factory=dict, alias="meta")
    created_at: datetime
    updated_at: datetime | None = None

    model_config = {"from_attributes": True, "populate_by_name": True}


class ActionDraftConfirmResponse(BaseModel):
    """确认执行草稿的返回体（业务对象：执行后的草稿 + 对应的审计记录）。

    把 audit_id 一并返回，是为了让前端能立刻跳转/展示「这次操作留下了哪条审计日志」，
    形成「谁批准、执行了什么」的闭环。
    """
    draft: ActionDraftResponse  # 执行后的草稿最新状态
    audit_id: UUID  # 本次执行产生的审计日志 ID


class AuditLogResponse(BaseModel):
    """审计日志返回体（业务对象：一条不可篡改的操作留痕）。

    字段与 AuditLog 表一一对应；request_json / result_json 保持 dict 原样，
    因为审计内容结构随动作类型变化，不适合在这里做强类型约束。
    """
    id: UUID
    user_id: UUID | None = None  # 操作者；系统自动任务为空
    repo_id: UUID | None = None  # 相关仓库
    action: str  # 动作名，如 draft.execute
    target_type: str | None = None
    target_id: str | None = None  # 字符串而非 UUID：审计对象可能是外部实体
    status: str  # success / failed / denied
    request_json: dict  # 请求快照（已脱敏）
    result_json: dict  # 结果摘要
    created_at: datetime

    model_config = {"from_attributes": True}
