"""仓库接入与同步模型（业务对象：把 GitHub 仓库挂进 DevFlow，以及后续的数据同步）。

安全注意：这些模型里出现的 token 字段只用于「传入」，任何响应模型都不得包含它；
入库时统一加密为 *_encrypted 列，返回给前端时也不回显，避免凭证泄漏到浏览器缓存与日志。
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class RepoConnectRequest(BaseModel):
    """接入仓库的请求（业务对象：用户填写的一次接入表单）。

    所有字段几乎都可空，是因为支持多种接入方式：
    远端接入（owner+repo+token）、本地目录接入（local_path）、演示模式（demo_mode）。
    这样同一个接口能覆盖三类场景，代价是校验逻辑必须放到服务层（见接入服务的分支判断）。
    """
    owner: str | None = None  # 仓库所有者（用户/组织）
    repo: str | None = None  # 仓库名
    # token 与 github_token 同时存在是历史兼容：早期前端传 token，后来统一成 github_token，
    # 保留两者可避免旧客户端接入失败，服务层按「github_token 优先」取值
    token: str | None = None
    github_token: str | None = None
    provider: str = "github"  # 托管方，为将来支持 GitLab 等预留
    api_base_url: str | None = None  # 私有部署 API 地址；为空用官方
    local_path: str | None = None  # 直接指向本地已有目录，跳过克隆（内网/离线场景）
    clone_parent_dir: str | None = None  # 克隆到哪个父目录；为空用 settings.repo_checkout_dir
    demo_mode: bool = False  # 演示模式：不真的访问 GitHub，用内置假数据
    # 首次同步条数上限 1~50：太多会让首次接入等待过久并触发 GitHub 限流
    initial_sync_limit: int = Field(default=5, ge=1, le=50)
    sync_issues: bool = True  # 是否同步 issue
    sync_pull_requests: bool = True  # 是否同步 PR
    sync_workflow_runs: bool = True  # 是否同步 CI 运行记录


class RepoResponse(BaseModel):
    """仓库详情（业务对象：仓库列表/详情页展示的信息）。

    刻意不包含 github_token_encrypted：密文也不外泄，前端永远拿不到凭证相关字段。
    """
    id: UUID
    owner: str
    name: str
    full_name: str  # owner/name，唯一标识
    provider: str = "github"
    api_base_url: str | None = None  # 已脱敏的地址，不含 token
    clone_url: str | None = None  # 可能含用户名但不含 token
    local_path: str | None = None  # 本地检出目录，便于用户确认索引的是哪份代码
    checkout_mode: str = "managed"  # managed=系统克隆 / attached=外部目录
    description: str | None = None
    default_branch: str | None = None
    last_sync_at: datetime | None = None  # 上次成功同步时间，用于展示数据新鲜度
    updated_at: datetime | None = None

    model_config = {"from_attributes": True}


class RepoConnectResponse(BaseModel):
    """接入结果（业务对象：一次接入操作的回执，含首次同步情况）。

    connected 与 sync_status 分开：接入成功不代表同步成功，
    前端可以先提示「已接入」再单独提示「同步进行中/失败」，避免一个失败把整个流程判死。
    """
    repo_id: UUID
    full_name: str
    provider: str
    api_base_url: str | None = None
    clone_url: str | None = None
    local_path: str | None = None
    checkout_mode: str = "managed"
    default_branch: str | None  # 注意此处无默认值：必须由服务端探测后回填
    connected: bool  # 接入动作本身是否成功
    demo_mode: bool  # 是否走了演示模式，前端据此提示「数据为示例」
    message: str  # 供直接展示的提示文案
    sync_status: str = "not_started"  # 首次同步状态
    synced: dict[str, int] | None = None  # 各类数据同步条数，如 {"issues": 12, "pull_requests": 3}


class RepoSyncRequest(BaseModel):
    """手动同步请求（业务对象：用户点「立即同步」时选择的同步范围）。"""
    sync_issues: bool = True
    sync_pull_requests: bool = True
    sync_workflow_runs: bool = True
    # 这里默认 30 且没有上限约束（与接入时的 1~50 不同）：
    # 手动同步允许拉更多历史数据，风险由用户自己承担
    limit: int = 30


class RepoSyncResponse(BaseModel):
    """同步结果（业务对象：本次同步各类型的入库条数）。"""
    repo_id: UUID
    status: str  # success / partial / failed
    synced: dict[str, int]  # 各类型同步条数
