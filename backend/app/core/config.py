"""全局配置中心：把环境变量与 .env 收敛成唯一的 Settings 实例，供全项目 import。

为什么用 pydantic-settings 而不是到处 os.getenv：
- 字段声明就是「配置契约」，类型写错会在进程启动时立刻报错，而不是跑到某个接口才 500。
- 环境变量名与字段名大小写无关（github_token 可由 GITHUB_TOKEN 注入），部署时无需改代码。
- 默认值写在这里，使本地开发可以「零配置启动」，这也是 demo_mode 默认 True 的原因。
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用全部可调参数的集合（业务对象：一份运行期配置快照）。

    下面的字段按用途分组：应用基础 → 外部集成 → 模型与 RAG → 上下文预算 → 运行时开关。
    未在 .env 中出现的字段一律取此处默认值；带 | None 的字段表示「可选能力」，
    为 None 时对应功能会被上层显式降级，而不是直接抛异常。
    """

    # ── 应用基础信息 ────────────────────────────────────────────────
    app_env: str = "development"  # 环境标识：development / staging / production，用于区分调试行为
    app_name: str = "DevFlow AI"  # 服务名，出现在 OpenAPI 标题、日志与告警里
    api_host: str = "0.0.0.0"  # 监听所有网卡，便于容器内被外部访问；生产由网关再收口
    api_port: int = 8000  # 后端端口，需与 docker-compose 的端口映射一致
    # 用逗号分隔字符串而非 list：环境变量只能传字符串，避免 pydantic 解析 JSON 失败
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001,http://127.0.0.1:3001"

    # 数据库连接串；驱动用 psycopg(v3)，同步引擎，配合 pool_pre_ping 应对连接被中间件掐断
    database_url: str = "postgresql+psycopg://devflow:devflow@localhost:5432/devflow"

    # ── GitHub 集成 ────────────────────────────────────────────────
    github_token: str | None = None  # PAT；为空时只能访问公开仓库，且会受未认证限流约束
    github_api_base_url: str = "https://api.github.com"  # 抽成配置是为了能指向 GitHub Enterprise

    # ── 对话大模型（OpenAI 兼容协议）──────────────────────────────
    llm_api_key: str | None = None  # 为空时聊天/分析链路会走降级分支，而不是直接崩
    llm_base_url: str = "https://api.deepseek.com"  # OpenAI 兼容端点，换厂商只需改这里
    llm_model: str = "deepseek-v4-pro"  # 主模型名，决定能力与计费，切换前需回归评测
    rag_llm_enabled: bool = True  # 总开关：关掉后 RAG 只做检索不做生成，便于隔离排查
    # ── 向量化（embedding）────────────────────────────────────────
    embedding_provider: str = "qwen_local"  # qwen_local = 本地推理；可切远程 API
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"  # 语义向量模型；换模型必须重建整个集合
    embedding_dimensions: int = 1024  # 必须与模型输出维度一致，也与 Milvus 集合 schema 绑定
    embedding_api_key: str | None = None  # 走远程 embedding 服务时才需要
    embedding_base_url: str | None = None  # 为空表示用厂商默认地址
    local_embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"  # 本地推理的权重名（HF 仓库 ID）
    local_embedding_device: str | None = None  # None 表示自动选 cuda/cpu，避免写死机器差异
    qwen_embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"  # DashScope 侧模型名，与本地名分开便于独立切换
    dashscope_api_key: str | None = None  # 阿里云百炼 Key，用于远程 embedding / rerank
    dashscope_embedding_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 兼容 OpenAI 协议的路径
    rerank_api_key: str | None = None  # 精排服务 Key；为空则跳过 rerank，仅用向量召回
    rerank_base_url: str | None = None  # 精排服务地址
    qwen_rerank_model: str = "qwen3-rerank"  # 精排模型名
    # 检索指令（instruction）：Qwen3-Embedding 系列会按指令调整向量语义，
    # 固化成常量可保证「建库」与「查询」用的是同一套语义，避免召回漂移。
    qwen_retrieval_instruction: str = "Given a user question, retrieve passages that contain the facts needed to answer it."

    # ── 向量库（Milvus）──────────────────────────────────────────
    milvus_enabled: bool = True  # 关掉后检索走内存/pg 兜底，方便没有 Milvus 的机器跑单测
    milvus_uri: str = "http://localhost:19530"  # Milvus 服务地址，也可写成本地 lite 路径
    milvus_token: str | None = None  # 云上 Milvus 才需要 token
    # 集合名里带 1024，是为了在维度变更时「换库」而不是「改库」，避免旧向量被误用
    milvus_collection: str = "devflow_rag_qwen1024"
    rag_rerank_top_n: int = 24  # 精排候选数：先多召回再精排，兼顾召回率与延迟
    # ── RAG 效果评测（RAGAS）──────────────────────────────────────
    ragas_enabled: bool = True  # 是否在回归测试中跑 RAGAS 评测
    ragas_judge_api_key: str | None = None  # 裁判模型 Key；与业务模型分开，避免自评偏差
    ragas_judge_base_url: str | None = None
    ragas_judge_model: str | None = None  # 裁判模型名
    ragas_timeout_seconds: float = 120.0  # 单次评测超时，评测慢但不必让 CI 卡死
    ragas_judge_max_tokens: int = 8192  # 裁判模型输出上限，需容纳长上下文评分理由
    ragas_answer_relevancy_strictness: int = 1  # RAGAS 内部计算严格度参数
    ragas_max_cases: int = 5  # 每次评测抽样条数，控制成本
    # 以下为质量门禁阈值（0~1）：指标低于阈值即判定回归失败，宁可误报也不放过劣化
    ragas_context_precision_min: float = 0.75  # 上下文精确率：召回内容里有多少是真有用
    ragas_context_recall_min: float = 0.80  # 上下文召回率：标准答案的证据有多少被找回
    ragas_faithfulness_min: float = 0.90  # 忠实度：回答是否只依据上下文，防幻觉
    ragas_answer_relevancy_min: float = 0.80  # 答案相关性：是否正面回答了问题
    ragas_agent_goal_accuracy_min: float = 0.80  # Agent 目标达成率
    ragas_tool_call_accuracy_min: float = 0.90  # 工具调用正确率（比目标达成更严）
    ragas_tool_call_f1_min: float = 0.90  # 工具调用的 F1，惩罚漏调与多调
    # ── 上下文窗口预算 ───────────────────────────────────────────
    context_max_input_tokens: int = 24000  # 单次请求允许塞给模型的输入预算软上限
    context_reserved_response_tokens: int = 3000  # 预留给输出的配额，必须从输入预算里扣掉
    context_dynamic_model_window_enabled: bool = True  # 是否按模型真实窗口动态调整预算
    context_model_window_tokens: int | None = None  # 手动指定模型窗口；None 表示用内置映射表
    context_compact_reserved_output_tokens: int = 12000  # 压缩摘要时预留给摘要输出的 token 数
    context_auto_compact_enabled: bool = True  # 是否在接近上限时自动压缩历史
    # 四档水位线：剩余空间低于哪一档，就触发对应强度的上下文治理策略
    context_warning_buffer_tokens: int = 20000  # 最宽松：仅告警，不裁剪
    context_auto_compact_buffer_tokens: int = 13000  # 触发自动摘要压缩
    context_aggressive_buffer_tokens: int = 8000  # 激进裁剪，优先保最近消息与证据
    context_manual_buffer_tokens: int = 3000  # 极限：只保留必要内容，丢弃一切可选上下文
    context_max_compaction_failures: int = 3  # 连续压缩失败次数上限，防止死循环烧 token
    context_microcompact_keep_recent_messages: int = 8  # 轻量压缩保留的最近消息条数
    context_compact_keep_recent_messages: int = 8  # 完整压缩保留的最近消息条数
    context_compact_summary_tokens: int = 1800  # 历史摘要的目标长度，太长等于没压缩
    # 各分区预算占比，合计 1.0：按「重要度」切蛋糕，保证证据与最近对话优先拿到额度
    context_system_ratio: float = 0.12  # system prompt / 规则
    context_memory_ratio: float = 0.22  # 长期记忆
    context_evidence_ratio: float = 0.34  # 检索证据（占比最高，因为最影响答案正确性）
    context_recent_ratio: float = 0.22  # 最近对话
    context_tool_ratio: float = 0.10  # 工具调用结果
    # ── 工作流与提示词缓存 ───────────────────────────────────────
    workflow_task_timeout_seconds: float = 90.0  # 单个工作流节点的硬超时，避免长尾任务拖垮队列
    prompt_cache_enabled: bool = True  # 命中缓存可显著降低重复前缀的 token 成本
    prompt_cache_key: str | None = "devflow-agent-v1"  # 缓存分区键；改 prompt 结构时应同步升版
    prompt_cache_retention: str | None = None  # 缓存保留策略，None 表示用服务端默认
    # ── 本地文件落盘位置 ────────────────────────────────────────
    repo_checkout_dir: str = ".data/repos"  # 克隆仓库的本地缓存目录，便于重复索引时增量复用
    upload_dir: str = ".data/uploads"  # 用户上传文件的临时落盘目录

    # ── 安全与演示开关 ──────────────────────────────────────────
    token_encryption_key: str = "change-me-in-local-dev"  # Fernet 派生种子；生产必须由环境变量覆盖
    demo_mode: bool = True  # 演示模式：放宽部分校验并使用内置假数据，生产应关闭
    github_webhook_secret: str | None = None  # Webhook 签名密钥；为空则跳过签名校验
    feishu_feedback_webhook_url: str | None = None  # 反馈同步到飞书群机器人
    feedback_trace_base_url: str | None = None  # 反馈里拼链路追踪链接的前缀
    # ── 仓库自动同步 ────────────────────────────────────────────
    auto_sync_enabled: bool = True  # 后台定时拉取仓库变更，保持索引新鲜
    auto_sync_interval_seconds: int = 300  # 轮询间隔；太短会撞 GitHub 限流
    auto_sync_limit: int = 50  # 每轮最多处理的仓库数，控制单轮耗时

    # env_file 按顺序尝试（先找 backend/../.env 即项目根目录，再找当前目录），
    # extra="ignore" 让 .env 里的多余变量（如给前端用的）不会导致启动失败。
    model_config = SettingsConfigDict(env_file=("../.env", ".env"), env_file_encoding="utf-8", extra="ignore")


# 模块级单例：配置在 import 时即完成解析与校验，启动期就能暴露配置错误
settings = Settings()
