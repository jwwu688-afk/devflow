/**
 * 全局共享的前端类型定义。
 *
 * 这里只放「跨页面/跨组件复用」的类型，页面私有的局部类型仍写在各自文件里。
 * 字段与后端 Pydantic / SQLAlchemy 模型（backend/app/models 与 schemas）保持同名，
 * 方便直接把接口返回的 JSON 当作本类型使用，不做驼峰转换。
 */
export type Repository = {
  // 数据库主键，前端所有 /repos/{id} 路由都靠它定位仓库
  id: string;
  // 仓库归属方（GitHub 用户名或组织名）
  owner: string;
  // 仓库名，不含 owner
  name: string;
  // "owner/name" 拼好的全名，列表展示与去重都用这个字段
  full_name: string;
  // 代码托管平台标识（如 github / gitlab），用于区分不同来源
  provider?: string;
  // 自建 Git 服务时覆盖默认 API 根地址
  api_base_url?: string;
  // 远端克隆地址；私有仓库可能为空，因此允许 null
  clone_url?: string | null;
  // 服务端把代码拉到本地后的目录，仅在「本地检出」模式下有值
  local_path?: string | null;
  // 检出策略（如 remote / local），决定分析时读远端 API 还是本地文件
  checkout_mode?: string;
  // 仓库描述，仅作展示
  description?: string;
  // 默认分支，创建索引/比对 PR 时作为基准
  default_branch?: string;
  // 最近一次同步时间（ISO 字符串），为空表示从未同步
  last_sync_at?: string | null;
};
