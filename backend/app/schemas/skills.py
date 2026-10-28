"""技能（Skill）与 MCP 运行时模型（业务对象：可被 Agent 加载的能力包及其运行状态）。

技能 = 一段自然语言指令 + 若干工具 + 使用约束，本质是「把复杂操作封装成一次调用」。
manifest 是技能的元数据（可被检索与路由），instructions 才是正文，
分开的原因是：路由阶段只读 manifest（省 token），命中后才加载 instructions。
"""

from typing import Any

from pydantic import BaseModel, Field


class SkillManifestResponse(BaseModel):
    """技能清单项（业务对象：一个技能的自描述元数据，用于列表展示与自动路由）。

    triggers 与 category 是路由的关键输入：前者做关键词/语义匹配，
    后者用于同类技能之间的兜底选择。
    """
    name: str  # 技能唯一名（英文，程序用）
    title: str  # 展示标题（中文，给人看）
    version: str  # 版本号；升级指令内容时应递增，便于回溯「当时用的是哪版」
    description: str  # 一句话说明「这个技能能做什么」
    category: str  # 分类，用于分组展示与同类兜底
    entrypoint: str  # 入口标识（对应实现里的处理函数/路由键）
    tools: list[str] = Field(default_factory=list)  # 该技能内部会调用的工具名
    input_modes: list[str] = Field(default_factory=list)  # 支持的输入形态（文本/文件/URL 等）
    triggers: list[str] = Field(default_factory=list)  # 触发词，用于自动激活
    workflow_steps: list[str] = Field(default_factory=list)  # 固定步骤，保证执行可复现
    output_contract: str  # 产出契约：说明结果长什么样，便于下游解析
    safety_level: str  # 安全级别；高于阈值时需人工确认后才执行写操作


class SkillResponse(BaseModel):
    """技能详情（业务对象：清单 + 本地路径 + 指令正文）。

    instructions 允许为空：列表接口不返回正文以节省带宽，详情接口才带上；
    path 暴露技能目录位置，便于开发者直接打开文件调整指令。
    """
    manifest: SkillManifestResponse
    path: str  # 技能在磁盘上的目录
    instructions: str | None = None  # 指令正文；列表场景为 None


class SkillListResponse(BaseModel):
    """技能列表响应（业务对象：当前可用的全部技能）。"""
    skills: list[SkillResponse]


class McpToolResponse(BaseModel):
    """MCP 工具描述（业务对象：某个 MCP Server 暴露的一个可调用工具）。"""
    name: str
    description: str
    # 字段名沿用 MCP 协议的 snake_case（协议原始是 inputSchema），
    # 这里统一成 Python 风格；值为 JSON Schema，结构不固定故用 dict
    input_schema: dict[str, Any] = Field(default_factory=dict)


class McpRuntimeResponse(BaseModel):
    """MCP 运行时状态（业务对象：MCP Server 的连通性与能力清单）。

    前端「设置 → MCP 状态」页展示它：status 为 error 时说明子进程没起来或握手失败，
    此时 tools 通常为空，可据此提示排查方向。
    """
    status: str  # ok / error / unknown
    server_name: str  # 握手返回的 serverInfo.name
    server_version: str  # 服务版本
    protocol_version: str  # 协商出的 MCP 协议版本
    transport: str  # 传输方式，本项目为 stdio
    tools: list[McpToolResponse] = Field(default_factory=list)  # 工具清单
