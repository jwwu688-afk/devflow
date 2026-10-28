/**
 * RAG / 知识库模块的前端共享类型与格式化工具。
 *
 * 类型全部按后端 backend/app/api/routes/rag.py（挂载前缀 /api/rag）与
 * backend/app/schemas 里的响应模型逐字段映射，请求路径示例：
 * - GET    /api/rag/repositories                      → KnowledgeBase[]
 * - GET    /api/rag/repositories/{repoId}             → KnowledgeBase
 * - PATCH  /api/rag/repositories/{repoId}/config      → KnowledgeBase（改检索配置）
 * - GET    /api/rag/repositories/{repoId}/documents   → KnowledgeDocument[]
 * - POST   /api/rag/repositories/{repoId}/documents   → KnowledgeDocument（上传，202 异步处理）
 * - GET    /api/rag/repositories/{repoId}/documents/{docId}/chunks → KnowledgeChunk[]
 * - POST   /api/rag/repositories/{repoId}/retrieval-tests → RetrievalResponse（检索测试）
 * - GET    /api/rag/models                            → ModelCatalog（可选模型清单）
 *
 * 之所以把类型集中在此：知识库列表页、详情页、检索设置组件都要用同一套字段，
 * 分散定义很容易和后端字段名漂移。
 */
export type RetrievalMethod = "vector" | "fulltext" | "hybrid";

/** 单个知识库的检索与切分配置，对应后端 KnowledgeConfig schema */
export type KnowledgeConfig = {
  // 向量化使用的服务商（如 openai / deepseek / 本地模型）
  embedding_provider: string;
  // 向量化模型名
  embedding_model: string;
  // 向量维度，换了模型必须与库中已存向量维度一致，否则检索会报错
  embedding_dimensions: number;
  // 检索方式：纯向量 / 纯全文 / 混合
  retrieval_method: RetrievalMethod;
  // 是否开启重排（成本更高但精度更好）
  rerank_enabled: boolean;
  rerank_provider: string;
  rerank_model: string;
  // 召回条数
  top_k: number;
  // 是否启用相似度阈值过滤
  score_threshold_enabled: boolean;
  score_threshold: number;
  // 文本切块长度与重叠，重叠用于避免语义被切断
  chunk_size: number;
  chunk_overlap: number;
};

/** 运行时模型状态：available 表示服务端确实拿到了 key/依赖，前端据此禁用不可用选项 */
export type RuntimeModel = {
  provider: string;
  model: string;
  available: boolean;
  // 主模型不可用时会退化到该模型
  fallback?: string | null;
  enabled?: boolean;
  dimensions?: number;
};

/** 知识库实体，列表页与详情页共用；runtime 描述当前 embedding/rerank 是否真的可用 */
export type KnowledgeBase = {
  id: string;
  name: string;
  full_name: string;
  description: string;
  provider: string;
  // 统计信息用于卡片展示规模
  document_count: number;
  chunk_count: number;
  character_count: number;
  config: KnowledgeConfig;
  runtime: { embedding: RuntimeModel; rerank: RuntimeModel };
  created_at?: string;
  updated_at?: string;
};

/** 知识库中的一篇文档；status 是异步流水线的阶段，前端用来做轮询与进度展示 */
export type KnowledgeDocument = {
  id: string;
  // 所属仓库，删除/重试接口都要带上
  repo_id: string;
  name: string;
  source_type: string;
  status: "queued" | "parsing" | "splitting" | "embedding" | "completed" | "failed";
  content_type?: string;
  file_size: number;
  character_count: number;
  chunk_count: number;
  // 仅 failed 状态有值，直接展示给用户排查
  error_message?: string;
  created_at?: string;
  completed_at?: string;
};

/** 文档切块后的片段；enabled=false 的块不参与检索，可在详情页手动开关 */
export type KnowledgeChunk = {
  id: string;
  // 在原文中的顺序，前端按此排序展示
  position: number;
  title: string;
  content: string;
  character_count: number;
  enabled: boolean;
};

/** 模型目录项，available 由后端探测 key 是否存在得出，recommended 用于默认选中 */
export type CatalogModel = {
  provider: string;
  model: string;
  label: string;
  available: boolean;
  recommended: boolean;
  fallback?: string | null;
  dimensions?: number;
};

/** GET /api/rag/models 的响应：可选的向量化/重排模型列表 */
export type ModelCatalog = {
  embedding_models: CatalogModel[];
  rerank_models: CatalogModel[];
  // DeepSeek 目前不具备 embedding/rerank 能力，后端用 note 说明原因，前端据此提示
  deepseek: { embedding_available: boolean; rerank_available: boolean; note: string };
};

/** 单条检索命中结果，citation 是答案里 [n] 角标的编号 */
export type RetrievalResult = {
  citation: number;
  document_id: string;
  source_id: string;
  source_type: string;
  title: string;
  snippet: string;
  score: number;
  metadata: Record<string, unknown>;
  // 记录该条命中来自哪一路召回、是否被重排提权，便于调试检索效果
  retrieval: Record<string, unknown>;
};

/**
 * POST /api/rag/repositories/{repoId}/retrieval-tests 的响应。
 * retrieval_config 在知识库配置基础上追加了运行时信息：
 * answer_gate 是后端判断「证据是否足够回答」的结论，pipeline/timeline 用于展示检索链路。
 */
export type RetrievalResponse = {
  id: string;
  query: string;
  results: RetrievalResult[];
  result_count: number;
  took_ms: number;
  retrieval_config: KnowledgeConfig & {
    runtime?: unknown;
    fusion_method?: string;
    answer_gate?: {
      decision?: "answer" | "ask_clarification" | "insufficient_evidence" | "conflict";
      reason?: string;
      evidence_count?: number;
      top_score?: number;
    };
    pipeline?: {
      retrieval_method?: string;
      fusion_method?: string;
      stages?: Array<Record<string, unknown> & { name?: string }>;
    };
  };
  created_at?: string;
};

/** 检索历史条目（GET /api/rag/repositories/{repoId}/retrieval-tests），只保留列表所需字段 */
export type RetrievalHistory = {
  id: string;
  query: string;
  result_count: number;
  took_ms: number;
  created_at?: string;
};

/** 文档状态 → 中文文案，页面直接查表，避免各处散落 if/else */
export const statusLabel: Record<KnowledgeDocument["status"], string> = {
  queued: "排队中",
  parsing: "解析中",
  splitting: "分段中",
  embedding: "向量化",
  completed: "可用",
  failed: "失败"
};

/** 数字千分位格式化（1,234），用于文档数/字符数等统计展示 */
export function formatNumber(value: number) {
  return new Intl.NumberFormat("zh-CN").format(value);
}

/** 字节数转可读体积；只做到 MB，是因为知识库上传限制在百 MB 级别 */
export function formatBytes(value: number) {
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / 1024 / 1024).toFixed(1)} MB`;
}

/** 时间格式化：只显示「月-日 时:分」，列表空间有限；空值统一显示 "-" */
export function formatDate(value?: string) {
  if (!value) return "-";
  return new Intl.DateTimeFormat("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" }).format(
    new Date(value)
  );
}
