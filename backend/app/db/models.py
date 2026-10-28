"""全库 ORM 模型定义：DevFlow AI 的所有持久化表都集中在这里。

设计约定（通读时先记住这三条，后面每张表都遵守）：
1. 主键统一用 UUID 而不是自增整数：仓库/会话/消息的 ID 会出现在 URL、日志、前端缓存里，
   UUID 不泄露业务量、可跨库合并、也能在入库前由应用侧先生成（便于批量写入与幂等重试）。
2. 时间列统一 DateTime(timezone=True) + server_default=func.now()：
   存带时区的时间戳，由数据库生成默认值，避免「应用服务器时钟不准」导致排序错乱；
   跨时区展示时前端再转本地时区即可。
3. 半结构化数据用 JSON 列而不是拆表：工具调用记录、评测结果这类字段结构随版本演进，
   用 JSON 可以先行落地、后置规范化，代价是无法对它做高效的范围查询与索引。

表关系总览（外键方向都是「子表 → 父表」）：
    users ─┬─< workspaces / action_drafts / audit_logs / chat_feedback
    repositories ─┬─< issues / pull_requests(─< pr_files, pr_review_comments) / workflow_runs
                  ├─< conversations(─< conversation_memories) / chat_messages / agent_runs
                  ├─< code_symbols(─< code_relations) / documents / evidence_items
                  └─ project_index_states / knowledge_base_configs / 各类 memory 与 event 表
    agent_workflow_runs ─< agent_task_runs（一次工作流拆成多个子任务）
删除策略：属于「仓库下属内容」的表一律由 ORM cascade 连带删除；
chat_feedback 则在数据库层面用 ondelete=CASCADE/SET NULL 兜底（见该类注释）。
"""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, BigInteger, DateTime, Float, ForeignKey, Integer, String, Text, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


class User(Base):
    """用户表（业务对象：DevFlow 的使用者，当前版本按单用户/演示用户简化）。

    没有登录体系，name 默认 "demo-user"，role 用于区分 owner/member 等权限语义。
    """
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(200), default="demo-user")  # 展示名；默认值让演示模式免注册即可用
    role: Mapped[str] = mapped_column(String(80), default="owner", index=True)  # 角色，加索引便于按角色筛人
    # GitHub PAT 的密文（Fernet 加密后是 base64 字符串，长度不定，故用 Text 而非定长 String）；
    # 绝不存明文，解密只在调用 GitHub API 的那一刻发生
    github_token_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())  # 注册时间（UTC）


class Repository(Base):
    """仓库表（业务对象：一个被接入分析的代码仓库，是所有业务数据的根节点）。

    绝大多数表都通过 repo_id 挂到这张表下，因此它承担「租户/项目」的角色：
    同一份数据在不同仓库之间天然隔离，查询时几乎总要带 repo_id 过滤。
    """
    __tablename__ = "repositories"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner: Mapped[str] = mapped_column(String(200), index=True)  # 仓库所有者（用户或组织名）
    name: Mapped[str] = mapped_column(String(200), index=True)  # 仓库名，不含 owner
    # owner/name 组合展示用；加 unique 是为了 upsert 时能以它做冲突判定，
    # 避免同一仓库被重复接入两次（重复接入会导致索引与统计数据翻倍）
    full_name: Mapped[str] = mapped_column(String(400), unique=True, index=True)
    provider: Mapped[str] = mapped_column(String(80), default="github")  # 代码托管方，预留 gitlab/gitee 扩展
    api_base_url: Mapped[str | None] = mapped_column(Text, nullable=True)  # 私有部署的 API 根地址，为空用官方地址
    clone_url: Mapped[str | None] = mapped_column(Text, nullable=True)  # 远端克隆地址（优先 https，便于带 token 拉取）
    local_path: Mapped[str | None] = mapped_column(Text, nullable=True)  # 本地检出目录，复用可避免反复全量克隆
    checkout_mode: Mapped[str] = mapped_column(String(80), default="managed")  # managed=由系统克隆；可扩展 attached=指向已有目录
    # 仓库级 token 覆盖用户级 token：私有仓库常需细粒度授权，因此允许单独配置
    github_token_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)  # 仓库简介，用于列表页展示与检索
    default_branch: Mapped[str | None] = mapped_column(String(200), nullable=True)  # 默认分支，代码图谱按分支区分
    # GitHub 侧的数字 ID；用 BigInteger 因为 GitHub 的 ID 已超过 32 位整型上限
    github_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    # 上次成功同步时间：自动同步任务据此判断是否需要拉取，也用于前端展示「新鲜度」
    last_sync_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())  # 任意字段变更自动刷新

    # 一对多关系；cascade="all, delete-orphan" 表示删除仓库时连带清掉子表数据，
    # 否则会留下大量孤儿行，且这些行还会被统计口径计入
    issues: Mapped[list["Issue"]] = relationship(back_populates="repo", cascade="all, delete-orphan")
    pull_requests: Mapped[list["PullRequest"]] = relationship(back_populates="repo", cascade="all, delete-orphan")
    workflow_runs: Mapped[list["WorkflowRun"]] = relationship(back_populates="repo", cascade="all, delete-orphan")


class Workspace(Base):
    """工作区表（业务对象：把多个仓库聚合成一个「项目视角」）。

    Agent 回答跨仓库问题时（例如前端仓 + 后端仓的接口对齐），需要同时检索多个 repo，
    工作区就是这种「多仓组合」的持久化配置。
    """
    __tablename__ = "workspaces"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(200), index=True)  # 工作区名
    description: Mapped[str | None] = mapped_column(Text, nullable=True)  # 用途说明，帮助用户区分多个工作区
    # 刻意用 JSON 数组存仓库 ID 字符串，而不建 workspace_repos 关联表：
    # 工作区只是「检索范围」的便捷集合，不需要按仓库反查工作区，省一张表与一次 join；
    # 代价是数据库无法保证数组里的 ID 仍然存在（仓库被删后需在读取时过滤）
    repo_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    # 创建人；nullable 是为了兼容演示模式下的匿名数据
    created_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)  # 索引支撑「最近创建」排序
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class ProjectIndexState(Base):
    """项目索引状态表（业务对象：某个仓库代码索引任务的当前进度快照）。

    与 repositories 是 1:1 关系（repo_id 上有 unique 约束），
    这样前端只需一次查询就知道「要不要重新索引、上次失败原因是什么」。
    """
    __tablename__ = "project_index_states"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # unique=True 是 1:1 关系的实现方式：一个仓库只能有一条索引状态
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id"), unique=True, index=True)
    # 状态机：missing(未索引) → indexing(进行中) → ready(就绪)，失败回到 missing/stale
    status: Mapped[str] = mapped_column(String(40), default="missing", index=True)
    # 仓库内容指纹（通常是 commit sha 或文件列表哈希）：与上次不同才触发重扫，避免无谓的全量索引
    fingerprint: Mapped[str] = mapped_column(String(200), default="")
    last_scan_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)  # 上次扫描时间
    # 用户「稍后提醒」的时间点：在此之前不再弹索引进度提示，避免打扰
    snoozed_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    docs_indexed: Mapped[int] = mapped_column(Integer, default=0)  # 已索引文档数（进度条的分子）
    docs_total: Mapped[int] = mapped_column(Integer, default=0)  # 待索引文档总数（分母）
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)  # 最近一次失败原因，直接展示给用户
    # 索引摘要（文件数/语言分布/依赖概览）；JSON 便于随时加字段而不用迁移表结构
    summary_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # 单向关系（无 back_populates）：只在这里按需取仓库信息，Repository 侧不需要反向集合
    repo: Mapped[Repository] = relationship()


class Conversation(Base):
    """会话表（业务对象：一个仓库下的一段连续对话，类似 ChatGPT 的 thread）。

    它是消息、运行记录、会话级记忆的聚合根，前端左侧列表展示的就是它。
    """
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id"), index=True)  # 会话必须归属某个仓库
    title: Mapped[str] = mapped_column(String(300), default="默认会话")  # 首条消息后可自动改写为新标题
    status: Mapped[str] = mapped_column(String(80), default="active", index=True)  # active / archived 等
    # 冗余的消息条数：列表页要显示条数，若每条会话都 count(*) 会产生 N+1 查询
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    # updated_at 也建索引：会话列表按「最近活跃」倒序分页，是最热的查询路径
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), index=True)


class Issue(Base):
    """Issue 表（业务对象：从 GitHub 同步过来的 issue，供 AI 做分类/摘要/定位相关代码）。

    数据来源是 GitHub API，因此除本地主键外还保留 github_issue_id 用于回写与去重。
    """
    __tablename__ = "issues"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id"), index=True)  # 所属仓库
    github_issue_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)  # GitHub 全局 ID，用于 upsert 去重
    number: Mapped[int] = mapped_column(Integer, index=True)  # 仓库内的展示编号（#123）
    title: Mapped[str] = mapped_column(String(500))  # 标题；500 足够覆盖 GitHub 上限
    body: Mapped[str | None] = mapped_column(Text, nullable=True)  # 正文 Markdown，长度不可控故用 Text
    state: Mapped[str] = mapped_column(String(50), default="open")  # open / closed
    labels: Mapped[list[str]] = mapped_column(JSON, default=list)  # 标签数组；标签本身是字符串列表，无需建标签表
    author: Mapped[str | None] = mapped_column(String(200), nullable=True)  # 作者登录名
    assignees: Mapped[list[str]] = mapped_column(JSON, default=list)  # 指派人列表（可多人）
    # 注意：这里的时间来自 GitHub，可能为空（历史数据/接口裁剪），因此全部 nullable
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)  # 关闭时间，用于统计解决时长

    repo: Mapped[Repository] = relationship(back_populates="issues")  # 反向引用 Repository.issues


class PullRequest(Base):
    """PR 表（业务对象：GitHub Pull Request，是 AI 代码评审功能的核心输入）。

    一条 PR 下挂两类子数据：变更文件清单（PRFile）与评审评论（PRReviewComment），
    评审时先看文件 patch，再看已有评论避免重复提意见。
    """
    __tablename__ = "pull_requests"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id"), index=True)  # 所属仓库
    github_pr_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)  # GitHub 数字 ID，BigInteger 防溢出
    number: Mapped[int] = mapped_column(Integer, index=True)  # 仓库内 PR 编号（#456）
    title: Mapped[str] = mapped_column(String(500))
    body: Mapped[str | None] = mapped_column(Text, nullable=True)  # PR 描述，评审 prompt 的重要上下文
    state: Mapped[str] = mapped_column(String(50), default="open")  # open / closed（是否合并看 merged_at）
    author: Mapped[str | None] = mapped_column(String(200), nullable=True)
    base_branch: Mapped[str | None] = mapped_column(String(200), nullable=True)  # 目标分支，diff 的「旧侧」
    head_branch: Mapped[str | None] = mapped_column(String(200), nullable=True)  # 源分支，diff 的「新侧」
    # merged_at 非空即代表已合并，因此不再单独维护 merged 布尔字段，避免两处状态不一致
    merged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    repo: Mapped[Repository] = relationship(back_populates="pull_requests")
    # PR 被删除（或仓库重建索引）时，文件与评论必须一起消失，否则会残留脏数据
    files: Mapped[list["PRFile"]] = relationship(back_populates="pr", cascade="all, delete-orphan")
    review_comments: Mapped[list["PRReviewComment"]] = relationship(back_populates="pr", cascade="all, delete-orphan")


class PRFile(Base):
    """PR 变更文件表（业务对象：一个 PR 中被改动的单个文件及其 diff）。

    单独建表而不是塞进 PR 的 JSON：文件数可能上百，且要按 filename 检索「这个文件被哪些 PR 改过」。
    """
    __tablename__ = "pr_files"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pr_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("pull_requests.id"), index=True)  # 外键指向 PR
    filename: Mapped[str] = mapped_column(String(500))  # 文件路径（仓库内相对路径）
    status: Mapped[str] = mapped_column(String(50))  # added / modified / removed / renamed
    additions: Mapped[int] = mapped_column(Integer, default=0)  # 新增行数，用于统计改动规模
    deletions: Mapped[int] = mapped_column(Integer, default=0)  # 删除行数
    # 统一 diff 文本；大文件可能被 GitHub 截断或为空（二进制文件），因此 nullable
    patch: Mapped[str | None] = mapped_column(Text, nullable=True)

    pr: Mapped[PullRequest] = relationship(back_populates="files")


class PRReviewComment(Base):
    """PR 评审评论表（业务对象：PR 上的一条行内/整体评论，含 AI 生成的评审意见）。

    保留 github_comment_id，是为了区分「AI 已发过哪些评论」以避免重复刷屏，
    同时支持把 AI 评论与人工评论一起展示。
    """
    __tablename__ = "pr_review_comments"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pr_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("pull_requests.id"), index=True)  # 所属 PR
    github_comment_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)  # 回写 GitHub 后拿到的 ID
    body: Mapped[str | None] = mapped_column(Text, nullable=True)  # 评论正文 Markdown
    path: Mapped[str | None] = mapped_column(String(500), nullable=True)  # 行内评论所属文件；整体评论为空
    line: Mapped[int | None] = mapped_column(Integer, nullable=True)  # 行号；整体评论为空
    author: Mapped[str | None] = mapped_column(String(200), nullable=True)  # 评论者（AI 评论会标记为机器人）
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    pr: Mapped[PullRequest] = relationship(back_populates="review_comments")


class WorkflowRun(Base):
    """CI 运行记录表（业务对象：GitHub Actions 的一次 workflow run）。

    用于「CI 失败后让 AI 读日志定位原因」这条链路：
    jobs 存结构化的各 job 状态，logs_text 存拉取下来的原始日志文本。
    """
    __tablename__ = "workflow_runs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id"), index=True)  # 所属仓库
    github_run_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)  # GitHub 侧运行 ID
    name: Mapped[str] = mapped_column(String(300))  # workflow 名称（如 "CI"）
    status: Mapped[str] = mapped_column(String(80))  # queued / in_progress / completed
    conclusion: Mapped[str | None] = mapped_column(String(80), nullable=True)  # success / failure / cancelled；运行中为空
    html_url: Mapped[str | None] = mapped_column(Text, nullable=True)  # 跳转 GitHub 的链接，方便用户核对
    # 各 job 的明细（名字/状态/耗时），结构随 GitHub API 变化，用 JSON 免去频繁改表
    jobs: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    logs_text: Mapped[str | None] = mapped_column(Text, nullable=True)  # 原始日志，可能很大，用 Text 且允许为空
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    repo: Mapped[Repository] = relationship(back_populates="workflow_runs")


class AnalysisResult(Base):
    """通用分析结果表（业务对象：AI 对任意对象产出的一次分析结论）。

    用 target_type + target_id 做「多态外键」：目标是 issue / pr / ci_run 等不同类型，
    无法用单一外键表达，所以只存 ID 不加外键约束。
    后果：删除目标对象时不会级联清理本表，统计时需按 target_type 过滤并容忍悬空引用。
    """
    __tablename__ = "analysis_results"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    target_type: Mapped[str] = mapped_column(String(80), index=True)  # 目标类型：issue / pull_request / ci_run ...
    target_id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), index=True)  # 目标对象 ID（多态，故意不加外键）
    analysis_type: Mapped[str] = mapped_column(String(100), index=True)  # 分析种类：summary / triage / review ...
    # 输入快照：把当时喂给模型的原始内容存下来，便于事后复现与追责（模型输出不可复现，只能靠输入回溯）
    input_snapshot: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    result_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # 结构化分析结果
    model_name: Mapped[str | None] = mapped_column(String(200), nullable=True)  # 产出该结果的模型，用于对比不同模型效果
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Document(Base):
    """知识库文档块表（业务对象：RAG 检索的最小单元，通常是切分后的一段文本）。

    真正的向量存在 Milvus 里，本表存原文与元数据：
    向量库只负责「找相似」，命中后要回这里取完整文本与出处，两者以 id 对应。
    """
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id"), index=True)  # 所属仓库，检索时必带此过滤
    source_type: Mapped[str] = mapped_column(String(80), index=True)  # 来源类型：code / issue / pr / ci / upload ...
    source_id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), index=True)  # 来源对象 ID（多态，不加外键）
    title: Mapped[str] = mapped_column(String(500))  # 展示用标题（如 "app/main.py:42"）
    content: Mapped[str] = mapped_column(Text)  # 文本块正文，长度不可控用 Text
    # 属性名用 meta 而列名仍是 "metadata"：metadata 是 SQLAlchemy Declarative 的保留属性
    # （Base.metadata），同名会直接覆盖掉它，因此只能用别名映射。
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class KnowledgeBaseConfig(Base):
    """知识库配置表（业务对象：某个仓库的检索/切分参数，决定 RAG 的行为）。

    与 repositories 是 1:1（repo_id unique）：每个仓库可以调不同的切分粒度与召回策略，
    例如大型后端仓切小一点、文档仓切大一点。
    """
    __tablename__ = "knowledge_base_configs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id"), unique=True, index=True)  # 一仓一配置
    # 以下五项必须与向量库里实际的模型/维度一致，否则查询向量与库中向量不可比
    embedding_provider: Mapped[str] = mapped_column(String(80), default="qwen_local")  # 向量化方式
    embedding_model: Mapped[str] = mapped_column(String(240), default="Qwen/Qwen3-Embedding-0.6B")  # 模型名
    embedding_dimensions: Mapped[int] = mapped_column(Integer, default=1024)  # 维度，与 Milvus 集合绑定
    retrieval_method: Mapped[str] = mapped_column(String(40), default="hybrid")  # hybrid=向量+关键词；可切 vector / keyword
    rerank_enabled: Mapped[bool] = mapped_column(default=True)  # 是否启用精排（更准但更慢、更贵）
    rerank_provider: Mapped[str] = mapped_column(String(80), default="qwen_api")  # 精排服务来源
    rerank_model: Mapped[str] = mapped_column(String(240), default="qwen3-rerank")  # 精排模型名
    top_k: Mapped[int] = mapped_column(Integer, default=3)  # 最终返回条数；越小越省 token，越大召回越全
    score_threshold_enabled: Mapped[bool] = mapped_column(default=True)  # 是否启用相似度阈值过滤
    # 阈值默认 0.8 偏高：宁可少召回几条，也不要塞入不相关内容污染答案（配合 RAGAS 的精确率门禁）
    score_threshold: Mapped[float] = mapped_column(Float, default=0.8)
    chunk_size: Mapped[int] = mapped_column(Integer, default=800)  # 切块字符数；800 约等于一段函数的长度
    chunk_overlap: Mapped[int] = mapped_column(Integer, default=100)  # 相邻块重叠，防止关键句被切断导致语义丢失
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class KnowledgeSourceDocument(Base):
    """知识源文档表（业务对象：用户上传的一个原始文件及其入库处理进度）。

    与 Document 的区别：Document 是切分后的「块」，本表是切分前的「文件」，
    通过它做去重（checksum）、失败重试与进度展示。
    """
    __tablename__ = "knowledge_source_documents"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id"), index=True)  # 归属仓库
    name: Mapped[str] = mapped_column(String(500))  # 原始文件名（含扩展名）
    source_type: Mapped[str] = mapped_column(String(80), default="upload_file")  # 来源：upload_file / url / repo_path
    status: Mapped[str] = mapped_column(String(40), default="queued", index=True)  # queued → parsing → indexed / failed
    stored_path: Mapped[str] = mapped_column(Text)  # 落盘路径（在 settings.upload_dir 下），便于重跑解析
    content_type: Mapped[str | None] = mapped_column(String(200), nullable=True)  # MIME 类型，决定用哪个解析器
    # checksum 加索引是为了按内容哈希判重：同一文件重复上传时直接复用，不必重新解析和向量化
    checksum: Mapped[str] = mapped_column(String(64), index=True)  # SHA-256 十六进制正好 64 字符
    file_size: Mapped[int] = mapped_column(Integer, default=0)  # 字节数
    character_count: Mapped[int] = mapped_column(Integer, default=0)  # 解析出的字符数，用于估算 chunk 数量
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)  # 实际生成的文本块数
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)  # 解析失败原因
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)  # 处理完成时间，用于算耗时


class RetrievalTestRun(Base):
    """检索调试记录表（业务对象：知识库页面里「试一次检索」的每次实验）。

    存下 query、参数与结果，好处是调参时能对比「同一问题在不同参数下的召回差异」，
    也便于把一次表现好的参数直接固化进 KnowledgeBaseConfig。
    """
    __tablename__ = "retrieval_test_runs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id"), index=True)  # 在哪个仓库的库上试验
    query: Mapped[str] = mapped_column(Text)  # 用户输入的检索语句
    retrieval_config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # 本次快照的检索参数（top_k/阈值/精排开关）
    results_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)  # 返回的候选及分数
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)  # 耗时毫秒，用于评估性能开销
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)


class CodeSymbol(Base):
    """代码符号表（业务对象：从源码里解析出来的函数/类/方法等定义）。

    这是「代码图谱」的节点表，配合 CodeRelation 构成有向图，
    支撑「这个函数被谁调用」「改动影响面有多大」这类问题。
    """
    __tablename__ = "code_symbols"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id"), index=True)  # 所属仓库
    path: Mapped[str] = mapped_column(String(500), index=True)  # 文件相对路径，定位源码位置
    name: Mapped[str] = mapped_column(String(300), index=True)  # 符号名（函数名/类名），按名检索
    kind: Mapped[str] = mapped_column(String(80), default="symbol", index=True)  # function / class / method / variable
    language: Mapped[str] = mapped_column(String(80), default="text", index=True)  # 语言，决定用哪个解析器
    # 起止行号允许为空：某些语言或退化解析只能拿到符号名，拿不到精确位置
    start_line: Mapped[int | None] = mapped_column(Integer, nullable=True)
    end_line: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # 分支/SHA/PR 三个维度区分「同名符号的不同版本」，避免 main 与 PR 分支的符号互相覆盖
    branch: Mapped[str | None] = mapped_column(String(200), nullable=True, index=True)
    commit_sha: Mapped[str | None] = mapped_column(String(120), nullable=True, index=True)
    pr_id: Mapped[uuid.UUID | None] = mapped_column(Uuid(as_uuid=True), nullable=True, index=True)  # PR 场景下的临时符号
    pr_number: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)  # 冗余 PR 编号，前端直接展示
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)  # 扩展信息（签名、docstring 等）；列名避开保留字
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)


class CodeRelation(Base):
    """代码关系表（业务对象：代码图谱的一条有向边，如「A 调用 B」）。

    source/target 同时保存 id 与 name 两份：
    跨文件解析时可能只认得出名字（目标符号未入库），所以 name 是必填、id 可空，
    这样「弱关系」也能先落库，等后续解析到目标再补 id。
    """
    __tablename__ = "code_relations"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id"), index=True)  # 所属仓库
    # 两端都指向 code_symbols，构成自引用外键；nullable 是因为解析可能只拿到名字
    source_symbol_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("code_symbols.id"), nullable=True, index=True)
    target_symbol_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("code_symbols.id"), nullable=True, index=True)
    source_name: Mapped[str | None] = mapped_column(String(300), nullable=True, index=True)  # 起点符号名（模块级调用可能为空）
    target_name: Mapped[str] = mapped_column(String(300), index=True)  # 终点符号名，必填，保证边总有语义
    relation_type: Mapped[str] = mapped_column(String(80), default="calls", index=True)  # calls / imports / inherits / references
    path: Mapped[str | None] = mapped_column(String(500), nullable=True, index=True)  # 关系发生的文件
    branch: Mapped[str | None] = mapped_column(String(200), nullable=True, index=True)  # 分支维度，与 CodeSymbol 对齐
    pr_id: Mapped[uuid.UUID | None] = mapped_column(Uuid(as_uuid=True), nullable=True, index=True)
    pr_number: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)  # 扩展信息（行号、置信度等）
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)


class ChatMessage(Base):
    """聊天消息表（业务对象：一条用户提问或助手回答）。

    这是最热的表：按 conversation_id 拉历史、按 repo_id 做统计。
    repo_id 与 conversation_id 都允许为空，以兼容「单轮问答、不落会话」的临时调用。
    """
    __tablename__ = "chat_messages"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("repositories.id"), nullable=True, index=True)  # 所属仓库
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("conversations.id"), nullable=True, index=True)  # 所属会话
    role: Mapped[str] = mapped_column(String(40), index=True)  # user / assistant / system / tool
    content: Mapped[str] = mapped_column(Text)  # 消息正文；即使工具调用也保留可读文本，便于回放
    route: Mapped[str | None] = mapped_column(String(100), nullable=True)  # 该消息被路由到的处理链路（rag/agent/chat）
    tool_calls: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)  # 工具调用明细，结构随工具演进，用 JSON
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)  # 引用、token 用量等附加信息
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)  # 按时间排序取历史


class ChatSession(Base):
    """会话封存表（业务对象：一次「问答回合」结束后的存档，用于生成长期记忆）。

    与 conversations 的区别：conversation 是跨多轮的长线程，ChatSession 是其中一个被封存的片段。
    封存（seal）时机是「助手回答完成」，此后 transcript 不再变动，可作为记忆抽取的稳定输入。
    """
    __tablename__ = "chat_sessions"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("repositories.id"), nullable=True, index=True)
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("conversations.id"), nullable=True, index=True)
    # 下面两个只存消息 ID 不加外键：消息可能因清理策略被删除，而封存记录需要长期保留
    user_message_id: Mapped[uuid.UUID | None] = mapped_column(Uuid(as_uuid=True), nullable=True, index=True)
    assistant_message_id: Mapped[uuid.UUID | None] = mapped_column(Uuid(as_uuid=True), nullable=True, index=True)
    route: Mapped[str] = mapped_column(String(100), default="unknown")  # 该回合走的链路；unknown 用于兼容旧数据
    transcript_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # 完整对话快照，保证日后可复现
    digest: Mapped[str] = mapped_column(Text, default="")  # 本回合的摘要文本，供记忆抽取与人工审阅
    status: Mapped[str] = mapped_column(String(80), default="sealed", index=True)  # sealed=已封存；可扩展 pending 等
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    sealed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)  # 封存时刻，用于统计延迟


class ThreadMemory(Base):
    """仓库级长期记忆表（业务对象：跨会话沉淀下来的「项目常识」）。

    与 ConversationMemory 的分工：本表回答「这个项目整体是什么样」，
    例如技术栈约定、架构决策、长期偏好；一个仓库只有一条（repo_id unique）。
    """
    __tablename__ = "thread_memories"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # nullable 但 unique：允许存在一条「不属于任何仓库」的全局记忆，同时保证每仓最多一条
    repo_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("repositories.id"), nullable=True, unique=True, index=True)
    summary: Mapped[str] = mapped_column(Text, default="")  # 记忆总述，注入 system prompt 的精炼版本
    # 以下五类用独立的 JSON 数组分桶而不是一段自由文本：
    # 便于合并去重、按类型检索，也方便前端分类展示与人工删改
    facts: Mapped[list[str]] = mapped_column(JSON, default=list)  # 客观事实（技术栈、目录约定）
    decisions: Mapped[list[str]] = mapped_column(JSON, default=list)  # 已定架构决策（为什么这么设计）
    open_questions: Mapped[list[str]] = mapped_column(JSON, default=list)  # 悬而未决的问题，下次对话可主动追问
    tasks: Mapped[list[str]] = mapped_column(JSON, default=list)  # 进行中的任务，支持断点续做
    user_preferences: Mapped[list[str]] = mapped_column(JSON, default=list)  # 用户偏好（语言、风格、详略）
    repo_context: Mapped[list[str]] = mapped_column(JSON, default=list)  # 仓库上下文摘要
    citations: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)  # 每条记忆的出处，便于溯源核对
    # 已并入的记忆条数：记忆是增量合并的，靠它判断是否需要触发一次整体压缩
    sessions_incorporated: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class ConversationMemory(Base):
    """会话级记忆表（业务对象：单个对话线程内的上下文摘要）。

    与 ThreadMemory 结构相同但粒度更细：会话内追问依赖它（「刚才说的那个函数」），
    一个会话一条（conversation_id unique），也便于在会话删除时一起清理。
    """
    __tablename__ = "conversation_memories"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id"), index=True)  # 冗余仓库 ID，便于按仓库批量统计
    conversation_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("conversations.id"), unique=True, index=True)  # 一会话一记忆
    summary: Mapped[str] = mapped_column(Text, default="")  # 会话摘要
    facts: Mapped[list[str]] = mapped_column(JSON, default=list)  # 本会话确认的事实
    decisions: Mapped[list[str]] = mapped_column(JSON, default=list)  # 本会话做出的决定
    open_questions: Mapped[list[str]] = mapped_column(JSON, default=list)  # 待澄清问题
    tasks: Mapped[list[str]] = mapped_column(JSON, default=list)  # 待办
    user_preferences: Mapped[list[str]] = mapped_column(JSON, default=list)  # 本会话观察到的偏好
    repo_context: Mapped[list[str]] = mapped_column(JSON, default=list)  # 相关仓库上下文
    citations: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)  # 引用来源
    sessions_incorporated: Mapped[int] = mapped_column(Integer, default=0)  # 已合并的封存片段数
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class EvidenceItem(Base):
    """证据表（业务对象：一次回答所引用的具体材料快照）。

    为什么不直接引用 documents 表：证据需要「冻结」——即使原文后续被重新索引或删除，
    历史回答的引用仍要能展示当时看到的内容，所以这里存的是副本。
    """
    __tablename__ = "evidence_items"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("repositories.id"), nullable=True, index=True)
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("conversations.id"), nullable=True, index=True)
    source_type: Mapped[str] = mapped_column(String(80), index=True)  # 证据来源类型：document / code / issue / pr ...
    # 这里用 String 而不是 UUID：来源可能是 Milvus 主键、文件路径、issue 编号等异构标识
    source_id: Mapped[str] = mapped_column(String(120), index=True)
    title: Mapped[str] = mapped_column(String(500))  # 引用标题（文件名/符号名），前端角标展示
    content: Mapped[str] = mapped_column(Text)  # 引用正文快照
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)  # 分数、行号等元信息
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)


class KnowledgeGraphEdge(Base):
    """知识图谱边表（业务对象：业务概念之间的关系，如「模块 A 依赖 服务 B」）。

    与 code_relations 的区别：那张表是「代码符号级」的精确调用图，
    本表是「知识级」的弱关系，端点可以是文档、issue、决策等异构对象，故用字符串类型/ID 表达。
    """
    __tablename__ = "knowledge_graph_edges"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id"), index=True)  # 图谱按仓库隔离
    # 起止端点都用 type + id 字符串：端点类型不固定，无法建外键，只能靠应用层保证一致性
    from_type: Mapped[str] = mapped_column(String(80), index=True)  # document / symbol / issue / decision ...
    from_id: Mapped[str] = mapped_column(String(180), index=True)  # 起点对象 ID（字符串以兼容多种 ID 形式）
    to_type: Mapped[str] = mapped_column(String(80), index=True)  # 终点类型
    to_id: Mapped[str] = mapped_column(String(180), index=True)  # 终点 ID
    relation: Mapped[str] = mapped_column(String(80), index=True)  # 关系语义：depends_on / implements / references ...
    # 置信度：LLM 抽取的边不如 AST 精确，用 0~1 分数让上层可设置阈值或做加权排序
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    source: Mapped[str] = mapped_column(String(120), default="explicit")  # 边来自显式声明还是模型推断，决定可信程度
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)  # 证据片段等附加信息
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)


class RecallEvent(Base):
    """召回事件表（业务对象：Agent 每次调用检索工具的一条可观测记录）。

    存在的意义是「可解释 + 可调优」：回答不准时先看是没召回（result_count 为 0）
    还是召回了但没用上，从而区分检索问题与生成问题。
    """
    __tablename__ = "recall_events"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("repositories.id"), nullable=True, index=True)
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("conversations.id"), nullable=True, index=True)
    tool_name: Mapped[str] = mapped_column(String(120), index=True)  # 触发的工具名（search_code / search_docs ...）
    query: Mapped[str] = mapped_column(Text, default="")  # 实际发给检索的语句（可能被改写过）
    scope: Mapped[str] = mapped_column(String(120), default="project")  # 检索范围：project / workspace / global
    mode: Mapped[str] = mapped_column(String(80), default="hybrid")  # 检索模式，与 KnowledgeBaseConfig.retrieval_method 对应
    status: Mapped[str] = mapped_column(String(80), default="success", index=True)  # success / failed / timeout
    result_count: Mapped[int] = mapped_column(Integer, default=0)  # 命中条数，0 往往意味着索引缺失或 query 表达偏差
    results_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)  # 命中结果与分数快照
    citations_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)  # 最终进入回答的引用
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)  # 耗时、模型等运行信息
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)


class MemoryCandidate(Base):
    """记忆候选表（业务对象：系统自动抽取、等待人工确认的「可能值得长期记住」的条目）。

    为什么要人工确认：自动写记忆一旦写错会长期污染后续所有回答，
    因此先落成 pending 候选，用户确认（accepted）后才合并进 ThreadMemory。
    """
    __tablename__ = "memory_candidates"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("repositories.id"), nullable=True, index=True)
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("conversations.id"), nullable=True, index=True)
    session_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("chat_sessions.id"), nullable=True, index=True)  # 来自哪次封存
    kind: Mapped[str] = mapped_column(String(80), index=True)  # 与记忆分桶对应：fact / decision / question / task / preference
    title: Mapped[str] = mapped_column(String(500))  # 单行摘要，列表页展示
    content: Mapped[str] = mapped_column(Text)  # 完整内容，确认后写入记忆
    status: Mapped[str] = mapped_column(String(80), default="pending", index=True)  # pending / accepted / rejected
    source: Mapped[str] = mapped_column(String(120), default="session_sealer")  # 产生方式：session_sealer / manual / import
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)  # 置信度、抽取模型等
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)  # 审核时间；pending 时为空


class AgentRun(Base):
    """Agent 运行记录表（业务对象：一次用户提问触发的完整 Agent 执行过程）。

    与 chat_messages 的区别：消息是「展示层」的问答对，本表是「执行层」的完整轨迹
    （路由决策、工具调用序列、最终答案），用于排查「为什么这样回答」以及做效果分析。
    """
    __tablename__ = "agent_runs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("repositories.id"), nullable=True)  # 允许无仓库的通用问答
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("conversations.id"), nullable=True, index=True)
    user_message: Mapped[str] = mapped_column(Text)  # 原始提问，作为轨迹的起点
    route: Mapped[str] = mapped_column(String(100))  # 路由结果（rag / agent / chat），决定走了哪条链路
    tool_calls: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)  # 工具调用序列，含参数与返回摘要
    final_answer: Mapped[str] = mapped_column(Text)  # 最终返回给用户的答案
    status: Mapped[str] = mapped_column(String(80), default="success")  # success / failed / partial
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ChatFeedback(Base):
    """回答反馈表（业务对象：用户对某条助手回答的点赞/点踩及原因）。

    这是效果闭环的入口：rating + reason 会被汇总成评测样本，也会通过飞书/邮件通知到负责人。
    删除策略在这里显式写在数据库层（ondelete），而不是只靠 ORM cascade：
    - 仓库/会话/消息被删时，反馈失去意义，用 CASCADE 一起清理；
    - 运行记录/用户被删时，反馈本身仍有统计价值，用 SET NULL 保留下来。
    """
    __tablename__ = "chat_feedback"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id", ondelete="CASCADE"), index=True)  # 所属仓库
    conversation_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"), index=True)  # 所属会话
    # unique：一条消息只允许一条反馈，天然实现「重复提交即覆盖」的幂等语义，也防止刷分
    assistant_message_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("chat_messages.id", ondelete="CASCADE"), unique=True, index=True
    )
    # 关联到执行轨迹，便于按「坏答案」回溯当时的工具调用；轨迹被清理时置空即可
    run_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("agent_runs.id", ondelete="SET NULL"), nullable=True, index=True)
    user_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    rating: Mapped[str] = mapped_column(String(40), index=True)  # up / down（用字符串而非布尔，预留「中性」等扩展）
    reason: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)  # 预设原因码，便于聚合统计
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)  # 用户自由补充说明
    review_status: Mapped[str] = mapped_column(String(40), default="open", index=True)  # 人工处理状态：open / triaged / closed
    review_note: Mapped[str | None] = mapped_column(Text, nullable=True)  # 处理结论
    # 通知状态与错误分开存：先落库再异步推送，推送失败不能影响反馈本身入库
    notification_status: Mapped[str] = mapped_column(String(40), default="pending", index=True)  # pending / sent / failed
    notification_error: Mapped[str | None] = mapped_column(Text, nullable=True)  # 推送失败原因，便于重试
    notified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)  # 推送成功时间
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)  # 人工处理时间
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), index=True
    )


class AgentWorkflowRun(Base):
    """多智能体工作流记录表（业务对象：一次「规划 → 分派 → 执行 → 汇总」的完整协作过程）。

    一个 workflow 会拆成多个 AgentTaskRun 子任务，因此本表是子任务的父表（聚合根）。
    """
    __tablename__ = "agent_workflow_runs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("repositories.id"), nullable=True, index=True)
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("conversations.id"), nullable=True, index=True)
    goal: Mapped[str] = mapped_column(Text)  # 用户设定的总目标，规划器据此拆任务
    spec_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # 规划产物：任务清单与依赖关系
    observation_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # 执行过程中的观察与中间结论
    final_answer: Mapped[str] = mapped_column(Text, default="")  # 汇总后的最终答复；未完成时为空串
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # 耗时/调用次数/token 等运行指标
    status: Mapped[str] = mapped_column(String(80), default="success", index=True)  # running / success / failed
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)  # 结束时间，用于算总耗时


class AgentTaskRun(Base):
    """工作流子任务表（业务对象：工作流中交给单个 Agent 的一个具体任务）。

    每条记录既是执行日志也是「声明（claim）→ 结果（result）」的对照，
    便于发现「Agent 说了要做但没做」或「做了但跑偏」的情况。
    """
    __tablename__ = "agent_task_runs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workflow_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("agent_workflow_runs.id"), index=True)  # 父工作流
    # task_id 是规划阶段生成的业务编号（如 "t1"），与自增/UUID 主键分开，方便人工对照计划书
    task_id: Mapped[str] = mapped_column(String(120), index=True)
    agent_name: Mapped[str] = mapped_column(String(120), index=True)  # 执行该任务的 Agent 角色名
    task_type: Mapped[str] = mapped_column(String(120), index=True)  # 任务类别，决定用哪套 prompt 与工具
    claim_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # Agent 自述的执行计划（声明）
    input_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # 实际输入（上游产出 + 上下文）
    result_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # 实际产出
    status: Mapped[str] = mapped_column(String(80), default="success", index=True)  # success / failed / skipped
    error: Mapped[str | None] = mapped_column(Text, nullable=True)  # 失败堆栈或原因
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)  # 与 started_at 之差即单任务耗时


class EvalRun(Base):
    """评测运行表（业务对象：一次离线评测的结果，如 RAGAS 指标或 Agent 命中率回归）。

    只存 name + 整体 JSON：指标项会随评测集调整而增减，用 JSON 免去频繁改表；
    代价是无法直接用 SQL 对单个指标做趋势聚合，需要应用层解析。
    """
    __tablename__ = "eval_runs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(200))  # 评测名称/批次标识，便于横向对比
    result_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # 各指标得分与明细
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ActionDraft(Base):
    """动作草稿表（业务对象：AI 拟好但尚未执行的高风险操作，如「提 PR」「评论 issue」）。

    核心设计是「人在环」：AI 只负责生成草稿，必须由用户确认（confirmed_by）后才执行，
    执行结果另行落库；这样即使模型幻觉，也不会直接改动真实仓库。
    """
    __tablename__ = "action_drafts"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id"), index=True)  # 作用于哪个仓库
    draft_type: Mapped[str] = mapped_column(String(100), index=True)  # 动作类型：create_pr / comment / create_issue ...
    target_type: Mapped[str] = mapped_column(String(100), index=True)  # 作用对象类型
    target_id: Mapped[uuid.UUID | None] = mapped_column(Uuid(as_uuid=True), nullable=True, index=True)  # 作用对象 ID；新建类动作可为空
    title: Mapped[str] = mapped_column(String(300), default="")  # 草稿标题（PR 标题等）
    content: Mapped[str] = mapped_column(Text, default="")  # 草稿正文（Markdown）
    # 风险等级由生成阶段评估，用于前端决定是否强提示（high 需要二次确认）
    risk_level: Mapped[str] = mapped_column(String(80), default="medium", index=True)
    # 状态机：pending_confirmation → confirmed → executed / failed；默认待确认是最安全的落点
    status: Mapped[str] = mapped_column(String(80), default="pending_confirmation", index=True)
    created_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)  # 发起人
    confirmed_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)  # 确认人（与发起人分离以支持审批）
    executed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)  # 实际执行时间
    execution_result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)  # 执行返回（如新 PR 的 URL/编号）；未执行时为空
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)  # 执行失败原因
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)  # 模型、prompt 版本等溯源信息
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class AuditLog(Base):
    """审计日志表（业务对象：一次对外部系统产生影响的动作留痕）。

    与 RecallEvent 等「内部观测」表不同，本表面向合规与追责：
    谁在哪个仓库做了什么、结果如何，全部只增不改。
    """
    __tablename__ = "audit_logs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)  # 操作者；系统任务为空
    repo_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("repositories.id"), nullable=True, index=True)  # 相关仓库
    action: Mapped[str] = mapped_column(String(160), index=True)  # 动作名，如 repo.connect / draft.execute
    target_type: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)  # 目标类型
    # 目标 ID 用 String 而非 UUID：审计对象可能是任意实体（含外部 ID），不做强类型约束
    target_id: Mapped[str | None] = mapped_column(String(120), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(80), default="success", index=True)  # success / failed / denied
    request_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # 请求快照（已脱敏，不能记录 token 明文）
    result_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # 结果摘要
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
