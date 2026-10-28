"""Skill 相关的只读接口（挂载前缀 /api/skills）。

项目里的 Skill 就是 skills/ 目录下的一份份 SKILL.md 清单，由
app.services.skills.get_skill_registry() 扫描并缓存成注册表。本模块对外暴露三件事：
列表、单个 Skill 详情（含完整提示词正文）、以及内置 MCP Server 的运行状态探测。
"""

from fastapi import APIRouter, HTTPException

from app.schemas.skills import McpRuntimeResponse, McpToolResponse, SkillListResponse, SkillManifestResponse, SkillResponse
from app.services.mcp_client import McpClientError, StdioMcpClient
from app.services.skills import get_skill_registry

router = APIRouter()


def _skill_response(skill, include_instructions: bool = False) -> SkillResponse:
    """把注册表里的 Skill 对象转成响应模型。

    入参：
        skill: 注册表条目，至少有 manifest / path / instructions 三个属性。
        include_instructions: 是否带上 SKILL.md 正文。
    返回：
        SkillResponse。
    说明：
        列表接口默认不带正文（include_instructions=False，字段为 None），
        因为正文动辄几十 KB，全部塞进列表会让首屏响应体暴涨；
        只在单个详情接口里按需返回。
    """
    # model_dump() 再 model_validate() 是刻意的：manifest 可能是另一种 pydantic 模型
    # 或 dataclass，走一遍字典能统一字段名/类型校验，避免直接传对象时的类型不兼容。
    return SkillResponse(
        manifest=SkillManifestResponse.model_validate(skill.manifest.model_dump()),
        path=skill.path,
        instructions=skill.instructions if include_instructions else None,
    )


@router.get("", response_model=SkillListResponse)
async def list_skills() -> SkillListResponse:
    """GET /api/skills：列出全部可用 Skill（不含正文）。

    入参：无。
    返回：SkillListResponse，skills 为精简版清单。
    异常：注册表加载失败会直接在 get_skill_registry() 内抛出，交给全局处理器。
    副作用：首次调用可能触发目录扫描/缓存预热。
    """
    registry = get_skill_registry()
    return SkillListResponse(skills=[_skill_response(skill) for skill in registry.list_skills()])


# 注意：这个静态路径必须排在下面的 "/{skill_name}" 之前声明。
# FastAPI 按注册顺序匹配，若顺序颠倒，"mcp-status" 会被当成 skill_name 命中详情接口。
@router.get("/mcp-status", response_model=McpRuntimeResponse)
async def get_mcp_status() -> McpRuntimeResponse:
    """GET /api/skills/mcp-status：探测内置 MCP Server 是否可用。

    入参：无（MCP 的启动命令来自 settings）。
    返回：McpRuntimeResponse，status 固定为 "online"，附带 server 名/版本/协议版本和工具清单。
    异常：拉起子进程失败（McpClientError）或进程不可执行（OSError）时返回 503，
        前端据此把 MCP 面板显示成「离线」，而不是让整个页面报错。
    副作用：每次调用都会真实启动一个 stdio 子进程做握手，然后退出——比较重，
        所以前端只在打开设置页时调用，不要放进轮询。
    """
    try:
        inspection = await StdioMcpClient().inspect_server()
    except (McpClientError, OSError) as exc:
        raise HTTPException(status_code=503, detail=f"MCP Server 验证失败：{exc}") from exc

    # 下面全是防御式取值：MCP 握手返回的字段在不同 SDK 版本里可能缺失，
    # 缺了就退化成空对象/unknown，保证接口永远能返回一个结构完整的响应。
    initialize = inspection.get("initialize") or {}
    server_info = initialize.get("serverInfo") or {}
    tools = [
        McpToolResponse(
            name=str(item.get("name") or ""),
            description=str(item.get("description") or ""),
            input_schema=item.get("inputSchema") or {},
        )
        for item in inspection.get("tools") or []
    ]
    return McpRuntimeResponse(
        status="online",
        server_name=str(server_info.get("name") or "unknown"),
        server_version=str(server_info.get("version") or "unknown"),
        protocol_version=str(initialize.get("protocolVersion") or "unknown"),
        transport="stdio",
        tools=tools,
    )


@router.get("/{skill_name}", response_model=SkillResponse)
async def get_skill(skill_name: str) -> SkillResponse:
    """GET /api/skills/{skill_name}：取单个 Skill 详情，含 SKILL.md 正文。

    入参：
        skill_name: 路径参数，即注册表里的 Skill 标识（一般是目录名）。
    返回：
        SkillResponse，instructions 里是完整正文，供前端渲染 Skill 说明页。
    异常：
        未注册的 Skill 返回 404。
    副作用：无。
    """
    registry = get_skill_registry()
    skill = registry.get_skill(skill_name)
    if skill is None:
        raise HTTPException(status_code=404, detail="未找到 Skill")
    return _skill_response(skill, include_instructions=True)
