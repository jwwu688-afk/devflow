"use client";
/*
 * 工作台首页（路由 /）——DevFlow AI 的主控台，也是整个前端最重的一个文件。
 *
 * 页面结构（三段式）：
 * 1) 左侧栏：仓库列表（可展开看会话）、新建会话、折叠/展开、接入新仓库入口；
 * 2) 中间：对话区。输入框 + 推荐提问 + 消息列表（支持 Markdown、工具调用轨迹、上下文诊断、反馈按钮），
 *    并提供「普通提问」与「计划模式（先出计划再执行）」两种交互；
 * 3) 右侧：工作台面板，按 workspaceTab 切到 Issue / PR / CI / 团队 / 记忆&知识库 / 技能&MCP / 图谱。
 *
 * 数据来源（后端按 main.py 挂载的前缀）：
 * - /api/repos、/api/repos/{id}/conversations、/api/repos/{id}/issues|pull-requests|workflow-runs
 * - /api/chat/stream（SSE 对话）、/api/chat/recommendations（推荐提问）、/api/chat/workflow-runs（计划执行结果）
 * - /api/issues|pull-requests|workflow-runs/{id}/analyze/stream（分析流）
 * - /api/skills、/api/skills/mcp-status（技能与 MCP 运行时）
 * - /api/repos/{id}/knowledge|memory|team-members|project-index|knowledge/graph（项目知识相关）
 *
 * 阅读建议：先看 DashboardPage 里的状态分组与 6 个 useEffect（首屏加载、切仓库、切会话、滚底、推荐提问宽度测量），
 * 再看 askAgent（SSE 主流程）与 executeWorkflowPlan（计划执行），最后按菜单看下方各 Workspace 组件。
 * 注意：本文件刻意不做组件拆分，状态集中在 DashboardPage，子组件全部通过 props 受控。
 */

import Link from "next/link";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  AlertTriangle,
  BookOpen,
  Cable,
  CheckCircle2,
  ChevronDown,
  Clock3,
  Database,
  FileText,
  Folder,
  Gauge,
  GitPullRequest,
  ListTodo,
  Network,
  PanelLeftClose,
  PanelLeftOpen,
  Plus,
  RefreshCw,
  Search,
  Send,
  ShieldCheck,
  Square,
  Trash2,
  Upload,
  X,
  XCircle,
  type LucideIcon
} from "lucide-react";
import { useEffect, useMemo, useRef, useState, type MouseEvent } from "react";

import { apiDelete, apiGet, apiPost, apiPostStream, apiUpload } from "@/lib/api";
import type { Repository } from "@/lib/types";
import { FeedbackControls, type ChatFeedbackRecord } from "@/components/feedback-controls";

/* ============ 类型定义区 ============
 * 这些类型与后端返回结构逐字段对齐（字段命名保持 snake_case），
 * 之所以都写在本文件：它们只服务首页，放到 lib 反而增加跨页耦合。
 */
type Issue = {
  id: string;
  number: number;
  title: string;
  state: string;
  labels?: string[];
  body?: string | null;
  author?: string | null;
  assignees?: string[];
  created_at?: string | null;
  updated_at?: string | null;
};

type PRFile = {
  id: string;
  filename: string;
  status: string;
  additions: number;
  deletions: number;
  patch?: string | null;
};

type PullRequest = {
  id: string;
  number: number;
  title: string;
  state: string;
  body?: string | null;
  author?: string | null;
  base_branch?: string | null;
  head_branch?: string | null;
  merged_at?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
  files?: PRFile[];
};

type WorkflowRun = {
  id: string;
  name: string;
  status: string;
  conclusion?: string | null;
  html_url?: string | null;
  logs_text?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
};

type WorkflowPlanClaim = {
  target_type: string;
  target_id?: string | null;
  allowed_sources?: string[];
  write_intent?: boolean;
  reason?: string;
};

type WorkflowPlanTask = {
  id: string;
  agent_name: string;
  task_type: string;
  objective: string;
  dependencies?: string[];
  claim: WorkflowPlanClaim;
  critical?: boolean;
};

type WorkflowPlan = {
  workflow_id: string;
  goal: string;
  trigger_message: string;
  acceptance_criteria?: string[];
  tasks: WorkflowPlanTask[];
  max_parallel_tasks?: number;
  requires_human_confirmation?: boolean;
  created_at?: string;
};

type ChatResponse = {
  route: string;
  tool_calls: ToolCallTrace[];
  agent_events?: Array<Record<string, unknown>>;
  answer: string;
  citations: Array<Record<string, string>>;
  compression_stats?: Record<string, unknown>;
  progressive_compaction?: Record<string, unknown>;
  model_usage?: Array<Record<string, unknown>>;
  context_diagnostics?: Record<string, unknown>;
};

type ChatPlanResponse = {
  route: string;
  conversation_id: string;
  answer: string;
  plan: WorkflowPlan;
  tool_calls: ToolCallTrace[];
  agent_events?: Array<Record<string, unknown>>;
};

// ---- SSE 事件负载类型：对应 /api/chat/stream 推送的 thought / delta / tool_use / tool_result / final ----
// final 比普通响应多出 run_id 与 message_id，用于把本地消息与后端记录关联起来
type ChatStreamFinal = ChatResponse & {
  run_id: string;
  message_id: string;
};

type ChatStreamDelta = {
  content?: string;
};

type ChatStreamToolUse = {
  id?: string;
  tool_name?: string;
  arguments?: Record<string, unknown>;
  skill_name?: string;
  skill_title?: string;
  skill_version?: string;
  skill_steps?: string[];
  skill_activation_mode?: string;
  skill_activation_reason?: string;
  skill_instruction_digest?: string;
};

type ChatStreamToolResult = {
  tool_use_id?: string;
  tool_name?: string;
  status?: string;
  route?: string;
  content?: unknown;
  error?: string;
  skill_name?: string;
  skill_title?: string;
  skill_version?: string;
  skill_steps?: string[];
  skill_activation_mode?: string;
  skill_activation_reason?: string;
  skill_instruction_digest?: string;
};

type PromptRecommendationsResponse = {
  agent: string;
  suggestions: string[];
};

type DuplicateCandidate = {
  issue_id: string;
  title: string;
  score: number;
};

type IssueEvidence = {
  source_type: string;
  title: string;
  snippet: string;
  reference?: string | null;
  confidence?: number | null;
};

type IssueDrafts = {
  clarification_comment?: string | null;
  task_breakdown?: string | null;
};

type IssueAnalysisResult = {
  summary: string;
  conclusion: "进入开发" | "等待澄清" | "关闭" | "拆分" | "合并到已有 Issue" | string;
  conclusion_reason: string;
  category: string;
  priority: "P0" | "P1" | "P2" | "P3" | string;
  complexity: "S" | "M" | "L" | "XL" | string;
  suggested_owner: string;
  owner_reason: string;
  duplicate_candidates?: DuplicateCandidate[];
  evidence?: IssueEvidence[];
  checklist?: string[];
  action_items?: string[];
  drafts?: IssueDrafts;
  confidence: number;
};

type IssueAnalysisResponse = {
  analysis_id: string;
  result: IssueAnalysisResult;
};

type PRAnalysisResult = {
  summary: string;
  plan?: string[];
  executed_steps?: string[];
  merge_recommendation: "建议合入" | "修改后合入" | "暂缓" | "拒绝" | string;
  recommendation_reason?: string;
  key_changes?: string[];
  risk_points?: string[];
  blocking_issues?: string[];
  review_checklist?: string[];
  test_suggestions?: string[];
  files_need_attention?: string[];
  review_comments?: string[];
  confidence: number;
};

type PRAnalysisResponse = {
  analysis_id: string;
  result: PRAnalysisResult;
};

type CIAnalysisResult = {
  failure_summary: string;
  failure_type: "test" | "build" | "lint" | "dependency" | "permission" | "environment" | "unknown" | string;
  plan?: string[];
  executed_steps?: string[];
  first_error?: string | null;
  root_cause?: string;
  possible_causes?: string[];
  fix_steps?: string[];
  debug_steps?: string[];
  related_files?: string[];
  is_merge_blocking: boolean;
  blocking_reason?: string;
  confidence: number;
};

type CIAnalysisResponse = {
  analysis_id: string;
  result: CIAnalysisResult;
};

type IssueAnalysisTraceItem = {
  id: string;
  kind: "step" | "tool" | "thinking" | "result" | "error" | string;
  title: string;
  content: string;
  status?: "running" | "done" | "error" | string;
};

type ChatHistoryMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  route?: string | null;
  tool_calls?: ToolCallTrace[];
  agent_events?: Array<Record<string, unknown>>;
  compression_stats?: Record<string, unknown>;
  progressive_compaction?: Record<string, unknown>;
  model_usage?: Array<Record<string, unknown>>;
  context_diagnostics?: Record<string, unknown>;
  created_at?: string | null;
};

type RepoConnectResult = {
  repo_id: string;
  full_name: string;
  provider: string;
  api_base_url?: string | null;
  clone_url?: string | null;
  local_path?: string | null;
  checkout_mode?: string;
  default_branch?: string | null;
  connected: boolean;
  demo_mode: boolean;
  message: string;
  sync_status?: string;
  synced?: Record<string, number>;
};

type ProjectConversation = {
  id: string;
  repo_id: string;
  title: string;
  status: string;
  message_count: number;
  created_at?: string | null;
  updated_at?: string | null;
};

type ConversationDeleteResult = {
  deleted: string;
  active_conversation_id: string;
  conversations: ProjectConversation[];
};

/**
 * 界面上的单条消息模型（与后端消息结构不同）：
 * 除了正文，还携带本地 UI 状态（轨迹折叠、pending、计划状态等），
 * 这些状态不需要持久化，刷新后会从后端历史重建。
 */
type ChatMessage = {
  id: string;
  messageId?: string;
  role: "user" | "assistant";
  content: string;
  route?: string;
  toolCalls?: ToolCallTrace[];
  traceItems?: ChatTraceItem[];
  traceCollapsed?: boolean;
  responsePending?: boolean;
  compressionStats?: Record<string, unknown>;
  progressiveCompaction?: Record<string, unknown>;
  modelUsage?: Array<Record<string, unknown>>;
  contextDiagnostics?: Record<string, unknown>;
  contextCollapsed?: boolean;
  workflowPlan?: WorkflowPlan;
  planSourceMessage?: string;
  planStatus?: "ready" | "executing" | "done" | "error";
  feedback?: ChatFeedbackRecord;
};

type ToolCallTrace = {
  tool_name: string;
  status: string;
  error?: string;
  tool_use_id?: string;
  skill_name?: string;
  skill_title?: string;
  skill_version?: string;
  skill_steps?: string[];
  skill_activation_mode?: string;
  skill_activation_reason?: string;
  skill_instruction_digest?: string;
};

type ChatTraceItem = {
  id: string;
  kind: "thinking" | "tool" | "error" | string;
  title: string;
  content: string;
  status?: "running" | "done" | "error" | string;
  toolUseId?: string;
  skillName?: string;
  skillTitle?: string;
  skillVersion?: string;
  skillSteps?: string[];
  skillActivationMode?: string;
  skillActivationReason?: string;
  skillInstructionDigest?: string;
  evidenceCount?: number;
};

type SkillManifest = {
  name: string;
  title: string;
  version: string;
  description: string;
  category: string;
  entrypoint: string;
  tools: string[];
  input_modes: string[];
  triggers: string[];
  workflow_steps: string[];
  output_contract: string;
  safety_level: string;
};

type SkillResponse = {
  manifest: SkillManifest;
  path: string;
  instructions?: string | null;
};

type McpRuntime = {
  status: string;
  server_name: string;
  server_version: string;
  protocol_version: string;
  transport: string;
  tools: Array<{
    name: string;
    description: string;
    input_schema: {
      required?: string[];
      properties?: Record<string, unknown>;
    };
  }>;
};

// 右栏当前选中的条目（三种类型共用一个联合类型，父组件据此决定渲染哪块详情）
type WorkspaceSelection =
  | { type: "issue"; id: string }
  | { type: "pr"; id: string }
  | { type: "ci"; id: string };

type WorkspaceItem = {
  id: string;
  title: string;
  meta: string;
  detail?: string | null;
  badges?: string[];
  selected: boolean;
  onClick: () => void;
};

type TeamMember = {
  id: string;
  name: string;
  role: string;
  strengths: string;
  techStack: string;
};

type TeamMemberDraft = Omit<TeamMember, "id">;

type MemoryNote = {
  id: string;
  title: string;
  content: string;
};

type MemoryNoteDraft = Omit<MemoryNote, "id">;

type KnowledgeItem = {
  id: string;
  document_ids: string[];
  source_type: "memory_note" | "knowledge_file" | "team_member" | "weekly_report" | string;
  title: string;
  content_preview: string;
  metadata?: Record<string, unknown>;
  chunk_count: number;
  created_at?: string | null;
};

// 项目索引状态机：missing（从未扫）→ building（扫描中）→ ready；内容变化会变 stale，失败为 failed
type ProjectIndexStatus = "missing" | "stale" | "building" | "ready" | "failed";

type ProjectIndexSummary = {
  projectName?: string;
  rootPath?: string;
  techStack?: string[];
  dirStructure?: string[];
  docsList?: Array<{ path: string; tier: string; source_type?: string }>;
  tierCoverage?: Record<string, number>;
  sourceTypeCoverage?: Record<string, number>;
  docsIndexed?: number;
  indexPolicy?: {
    mode?: "auto" | "allowlist";
    include?: string[];
    exclude?: string[];
    includeManifests?: boolean;
    configPath?: string | null;
  };
  excludedFiles?: number;
  excludedPaths?: string[];
  discoveryTruncated?: boolean;
};

type ProjectIndexState = {
  repo_id: string;
  status: ProjectIndexStatus;
  fingerprint: string;
  docs_indexed: number;
  docs_total: number;
  error_message?: string | null;
  summary_json?: ProjectIndexSummary | null;
  snoozed_until?: string | null;
  last_scan_at?: string | null;
};

type RecallEvent = {
  id: string;
  tool_name: string;
  query: string;
  scope: string;
  mode: string;
  status: string;
  result_count: number;
  results: Array<Record<string, unknown>>;
  citations: Array<Record<string, unknown>>;
  metadata?: Record<string, unknown>;
  created_at?: string | null;
};

type MemoryStatus = {
  repo_id: string;
  index: Record<string, unknown>;
  documents: Record<string, unknown>;
  evidence: Record<string, unknown>;
  recall: Record<string, unknown>;
  candidates: Record<string, unknown>;
  vector: Record<string, unknown>;
};

type MemoryCandidate = {
  id: string;
  kind: string;
  title: string;
  content: string;
  status: string;
  source: string;
  metadata?: Record<string, unknown>;
  created_at?: string | null;
  reviewed_at?: string | null;
};

type MemorySearchResult = {
  title: string;
  source_type: string;
  source_id: string;
  snippet: string;
  score?: number;
  metadata?: Record<string, unknown>;
  retrieval?: Record<string, unknown>;
};

type KnowledgeGraphNode = {
  id: string;
  type: string;
  title: string;
  subtitle?: string;
  metadata?: Record<string, unknown>;
  highlighted?: boolean;
};

type KnowledgeGraphEdge = {
  id: string;
  from_type: string;
  from_id: string;
  to_type: string;
  to_id: string;
  relation: string;
  confidence: number;
  source: string;
  metadata?: Record<string, unknown>;
};

type KnowledgeGraphResponse = {
  repo_id: string;
  center_type?: string | null;
  center_id?: string | null;
  depth: number;
  nodes: KnowledgeGraphNode[];
  edges: KnowledgeGraphEdge[];
  stats: Record<string, unknown>;
};

type AddProjectStatus = {
  type: "success" | "error";
  title: string;
  detail?: string;
};

type AddProjectSource = "remote" | "local";

type PromptSuggestionState = {
  suggestions: string[];
  exclude: string[];
};
// 右侧工作台的可切换页签；用 as const 让 WorkspaceTab 自动收窄为联合类型，新增页签只需改这一处
const workspaceTabs = ["Issue", "PR", "CI", "团队", "记忆&知识库", "技能&MCP", "图谱"] as const;
type WorkspaceTab = (typeof workspaceTabs)[number];

// 草稿会话用前缀标记：这类会话还没落库（首条消息发送时才创建），因此没有真实 id
const DRAFT_CONVERSATION_PREFIX = "draft:";

/** 生成一个「本地草稿会话」id，形如 draft:{repoId}:{uuid} */
function makeDraftConversationId(repoId: string) {
  return `${DRAFT_CONVERSATION_PREFIX}${repoId}:${crypto.randomUUID()}`;
}

/** 判断当前选中的会话是不是尚未落库的草稿会话 */
function isDraftConversationId(value: string) {
  return value.startsWith(DRAFT_CONVERSATION_PREFIX);
}

/** 基于 setTimeout 的等待，用于轮询/重试之间的延迟 */
function wait(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

/** 取首句并截断到 24 字，作为会话标题（避免侧栏被长问题撑开） */
function summarizeConversationTitle(text: string) {
  const normalized = text.replace(/\s+/g, " ").trim();
  const firstSentence = normalized.split(/[。！？!?.]/)[0]?.trim() || normalized;
  const title = firstSentence || normalized || "新会话";
  return title.length > 24 ? `${title.slice(0, 24)}...` : title;
}

/** 把任意值压成可读 JSON 字符串并截断，用于轨迹/调试信息的展示 */
function compactJson(value: unknown, maxLength = 900) {
  if (value === undefined || value === null) return "";
  const text = typeof value === "string" ? value : JSON.stringify(value, null, 2);
  if (!text) return "";
  return text.length > maxLength ? `${text.slice(0, maxLength)}...` : text;
}

/** 从工具返回里挑出最适合展示的一段文本：优先 answer_preview，其次其它字符串/序列化结果 */
function traceContentFromResult(content: unknown) {
  if (content && typeof content === "object" && !Array.isArray(content)) {
    const record = content as Record<string, unknown>;
    if (typeof record.answer_preview === "string" && record.answer_preview.trim()) {
      return record.answer_preview;
    }
  }
  return compactJson(content);
}

/** 把 SSE 的 tool_use 事件转成界面上的轨迹条目（工具卡片） */
function makeToolUseTrace(data: ChatStreamToolUse, fallbackId: string): ChatTraceItem {
  const toolName = data.tool_name || "tool";
  const args = compactJson(data.arguments, 700);
  return {
    id: data.id || fallbackId,
    kind: "tool",
    title: `调用工具 ${toolName}`,
    content: args ? `参数:\n${args}` : "正在执行...",
    status: "running",
    toolUseId: data.id,
    skillName: data.skill_name,
    skillTitle: data.skill_title,
    skillVersion: data.skill_version,
    skillSteps: data.skill_steps,
    skillActivationMode: data.skill_activation_mode,
    skillActivationReason: data.skill_activation_reason,
    skillInstructionDigest: data.skill_instruction_digest
  };
}

/** 把 SSE 的 tool_result 事件转成轨迹条目；失败时通过 status 让 UI 标红 */
function makeToolResultTrace(data: ChatStreamToolResult, fallbackId: string): ChatTraceItem {
  const toolName = data.tool_name || "tool";
  const failed = data.status === "error";
  const content = data.error || traceContentFromResult(data.content) || data.route || "";
  return {
    id: data.tool_use_id || fallbackId,
    kind: failed ? "error" : "tool",
    title: `${failed ? "工具失败" : "工具完成"} ${toolName}`,
    content,
    status: failed ? "error" : "done",
    toolUseId: data.tool_use_id,
    skillName: data.skill_name,
    skillTitle: data.skill_title,
    skillVersion: data.skill_version,
    skillSteps: data.skill_steps,
    skillActivationMode: data.skill_activation_mode,
    skillActivationReason: data.skill_activation_reason,
    skillInstructionDigest: data.skill_instruction_digest
  };
}

/**
 * 把后端持久化的 agent 事件数组（历史消息回放时拿到）转换成轨迹条目。
 * 历史消息在库里只存了事件流，没有界面状态，因此需要在这里「重放」成与实时流一致的形状。
 */
function traceItemsFromAgentEvents(events: Array<Record<string, unknown>> = []): ChatTraceItem[] {
  const items: ChatTraceItem[] = [];
  events.forEach((event, index) => {
    const type = String(event.type || "");
    if (type === "skill_activated") {
      const resources = Array.isArray(event.resources) ? event.resources.map((value) => contextRecord(value)) : [];
      items.push({
        id: `history-skill-activated-${index}`,
        kind: "tool",
        title: "Skill 已按需激活",
        content: [
          `方式：${contextString(event.activation_mode) || "unknown"}`,
          contextString(event.activation_reason) ? `原因：${contextString(event.activation_reason)}` : "",
          contextString(event.entrypoint) ? `入口：${contextString(event.entrypoint)}` : "",
          contextString(event.instruction_digest) ? `指令摘要：${contextString(event.instruction_digest)}` : "",
          resources.length ? `按需资源：${resources.map((resource) => contextString(resource.path)).filter(Boolean).join(", ")}` : ""
        ].filter(Boolean).join("\n"),
        status: "done",
        skillName: contextString(event.skill_name),
        skillTitle: contextString(event.skill_title),
        skillVersion: contextString(event.skill_version),
        skillSteps: Array.isArray(event.workflow_steps) ? event.workflow_steps.map(String) : [],
        skillActivationMode: contextString(event.activation_mode),
        skillActivationReason: contextString(event.activation_reason),
        skillInstructionDigest: contextString(event.instruction_digest)
      });
      return;
    }
    if (type === "skill_validation") {
      const status = contextString(event.status) || "not_executed";
      const failed = status === "violated" || status === "failed";
      const violations = Array.isArray(event.tool_contract_violations) ? event.tool_contract_violations.map(String) : [];
      items.push({
        id: `history-skill-validation-${index}`,
        kind: failed ? "error" : "tool",
        title: status === "violated" ? "Skill 契约校验失败" : status === "failed" ? "Skill 执行失败" : "Skill 运行校验",
        content: [
          `状态：${status}`,
          `完整指令：${event.instructions_loaded === true ? "已加载" : "未加载"}`,
          `步骤验证：${contextString(event.step_verification) || "unknown"}`,
          Array.isArray(event.observed_tools) && event.observed_tools.length ? `观察到的工具：${event.observed_tools.map(String).join(", ")}` : "",
          violations.length ? `越界工具：${violations.join(", ")}` : ""
        ].filter(Boolean).join("\n"),
        status: failed ? "error" : "done",
        skillName: contextString(event.skill_name),
        skillTitle: contextString(event.skill_title),
        skillVersion: contextString(event.skill_version),
        skillSteps: Array.isArray(event.declared_steps) ? event.declared_steps.map(String) : []
      });
      return;
    }
    if (type === "rag_strategy") {
      const preRetrieval = contextRecord(event.pre_retrieval);
      const reasoningRetrieval = contextRecord(event.reasoning_retrieval);
      const enabled = preRetrieval.enabled === true;
      const evidenceCount = contextNumber(preRetrieval.evidence_count);
      const query = contextString(preRetrieval.query);
      const sourceLines = Array.isArray(preRetrieval.sources)
        ? preRetrieval.sources
            .slice(0, 8)
            .map((value) => contextRecord(value))
            .map((source) => {
              const title = contextString(source.title) || "未命名证据";
              const sourceType = contextString(source.source_type) || "evidence";
              return `- [${sourceType}] ${title}`;
            })
        : [];
      items.push({
        id: `history-rag-strategy-${index}`,
        kind: "rag_prefetch",
        title: "RAG 预检索",
        content: [
          `模式：${contextString(event.mode) === "hybrid_rag" ? "混合 RAG" : "工具调用"}`,
          query ? `查询：${query}` : "",
          `调用前检索：${enabled ? `已启用 · 命中 ${evidenceCount} 条证据` : "未启用"}`,
          sourceLines.length ? `证据来源：\n${sourceLines.join("\n")}` : "",
          `Agent 二次检索：${reasoningRetrieval.enabled === true ? "可用（支持查询改写与多次补查）" : "未启用"}`
        ]
          .filter(Boolean)
          .join("\n"),
        status: "done",
        evidenceCount
      });
      return;
    }
    if (type === "thought" && typeof event.content === "string" && event.content.trim()) {
      items.push({
        id: `history-thought-${index}`,
        kind: "thinking",
        title: "公开思路",
        content: event.content,
        status: "done"
      });
      return;
    }
    if (type === "tool_use") {
      items.push(
        makeToolUseTrace(
          {
            id: typeof event.id === "string" ? event.id : `history-tool-${index}`,
            tool_name: typeof event.tool_name === "string" ? event.tool_name : undefined,
            arguments: event.arguments && typeof event.arguments === "object" ? (event.arguments as Record<string, unknown>) : undefined,
            skill_name: typeof event.skill_name === "string" ? event.skill_name : undefined,
            skill_title: typeof event.skill_title === "string" ? event.skill_title : undefined,
            skill_version: typeof event.skill_version === "string" ? event.skill_version : undefined,
            skill_steps: Array.isArray(event.skill_steps) ? event.skill_steps.map(String) : undefined,
            skill_activation_mode: typeof event.skill_activation_mode === "string" ? event.skill_activation_mode : undefined,
            skill_activation_reason: typeof event.skill_activation_reason === "string" ? event.skill_activation_reason : undefined,
            skill_instruction_digest: typeof event.skill_instruction_digest === "string" ? event.skill_instruction_digest : undefined
          },
          `history-tool-${index}`
        )
      );
      return;
    }
    if (type === "tool_result") {
      const result = makeToolResultTrace(
        {
          tool_use_id: typeof event.tool_use_id === "string" ? event.tool_use_id : undefined,
          tool_name: typeof event.tool_name === "string" ? event.tool_name : undefined,
          status: typeof event.status === "string" ? event.status : undefined,
          route: typeof event.route === "string" ? event.route : undefined,
          content: event.content,
          skill_name: typeof event.skill_name === "string" ? event.skill_name : undefined,
          skill_title: typeof event.skill_title === "string" ? event.skill_title : undefined,
          skill_version: typeof event.skill_version === "string" ? event.skill_version : undefined,
          skill_steps: Array.isArray(event.skill_steps) ? event.skill_steps.map(String) : undefined,
          skill_activation_mode: typeof event.skill_activation_mode === "string" ? event.skill_activation_mode : undefined,
          skill_activation_reason: typeof event.skill_activation_reason === "string" ? event.skill_activation_reason : undefined,
          skill_instruction_digest: typeof event.skill_instruction_digest === "string" ? event.skill_instruction_digest : undefined
        },
        `history-result-${index}`
      );
      const existingIndex = result.toolUseId ? items.findIndex((item) => item.toolUseId === result.toolUseId) : -1;
      if (existingIndex >= 0) {
        items[existingIndex] = { ...items[existingIndex], ...result, id: items[existingIndex].id };
      } else {
        items.push(result);
      }
      return;
    }
    if (type === "workflow_replan") {
      const addedTasks = Array.isArray(event.added_tasks) ? event.added_tasks.map(String).filter(Boolean) : [];
      items.push({
        id: `history-workflow-replan-${index}`,
        kind: "tool",
        title: "Planner 重新规划",
        content: [
          typeof event.reason === "string" ? event.reason : "",
          addedTasks.length ? `新增任务: ${addedTasks.join(", ")}` : ""
        ]
          .filter(Boolean)
          .join("\n"),
        status: "done"
      });
      return;
    }
    if (type === "workflow_task_result") {
      const status = typeof event.status === "string" ? event.status : "done";
      const failed = status === "error";
      items.push({
        id: `history-workflow-task-${index}`,
        kind: failed ? "error" : "tool",
        title: `${failed ? "任务失败" : "任务完成"} ${typeof event.task_type === "string" ? event.task_type : "workflow"}`,
        content: typeof event.summary === "string" ? event.summary : "",
        status: failed ? "error" : "done",
        skillName: contextString(event.skill_name),
        skillTitle: contextString(event.skill_title),
        skillVersion: contextString(event.skill_version),
        skillSteps: Array.isArray(event.skill_steps) ? event.skill_steps.map(String) : [],
        skillActivationMode: contextString(event.skill_activation_mode),
        skillActivationReason: contextString(event.skill_activation_reason),
        skillInstructionDigest: contextString(event.skill_instruction_digest)
      });
      return;
    }
    if (type === "workflow_observer_result") {
      const findings = Array.isArray(event.findings) ? event.findings.length : 0;
      items.push({
        id: `history-workflow-observer-${index}`,
        kind: "thinking",
        title: "Observer 检查",
        content: `confidence=${event.overall_confidence ?? "-"}, findings=${findings}`,
        status: "done"
      });
    }
  });
  return items;
}

/**
 * 优先用后端持久化的事件重放轨迹；没有事件（如旧数据）时回退到前端内存里的 fallback，
 * 保证历史消息打开时轨迹区不会突然变空。
 */
function finalizedTraceItems(
  events: Array<Record<string, unknown>> | undefined,
  fallback: ChatTraceItem[] | undefined
) {
  const finalized = traceItemsFromAgentEvents(events ?? []);
  return finalized.length > 0 ? finalized : fallback ?? [];
}

/** 相对时间（刚刚 / N 分钟 / N 小时 / N 天），用于会话列表；非法时间返回空串 */
function formatRelativeTime(value?: string | null) {
  if (!value) return "";
  const timestamp = new Date(value).getTime();
  if (Number.isNaN(timestamp)) return "";
  const diffMs = Date.now() - timestamp;
  const minute = 60 * 1000;
  const hour = 60 * minute;
  const day = 24 * hour;
  if (diffMs < minute) return "刚刚";
  if (diffMs < hour) return `${Math.floor(diffMs / minute)} 分钟`;
  if (diffMs < day) return `${Math.floor(diffMs / hour)} 小时`;
  return `${Math.floor(diffMs / day)} 天`;
}

const issueCategoryMeta = [
  { key: "unanswered", label: "未回复", color: "text-[#f47b5b]" },
  { key: "discussing", label: "讨论中", color: "text-[#38a7f4]" },
  { key: "decision", label: "待决策", color: "text-[#f08a00]" },
  { key: "accepted", label: "已处理", color: "text-[#20a35b]" },
  { key: "rejected", label: "已拒绝", color: "text-[#6b7280]" },
  { key: "closed", label: "已关闭", color: "text-[#7c8794]" }
] as const;

const prCategoryMeta = [
  { key: "reviewing", label: "审核中", color: "text-[#38a7f4]" },
  { key: "needsOwner", label: "需主审", color: "text-[#f08a00]" },
  { key: "conflict", label: "有冲突", color: "text-[#f47b5b]" },
  { key: "done", label: "已完成", color: "text-[#20a35b]" },
  { key: "rejected", label: "已拒绝", color: "text-[#6b7280]" }
] as const;

/**
 * 首页主组件：把「仓库 / 会话 / 对话 / 右侧工作台」四块状态集中在这一个组件里管理。
 *
 * 为什么状态这么集中：左侧栏、对话区、右侧面板互相联动（比如在 Issue 面板点「分析」后要
 * 打开弹窗并流式写入轨迹，同时会话上下文要带给后端），拆开反而要用大量 context/props 传递。
 * @sideEffect 挂载后加载仓库与会话；切换仓库/会话会触发多组数据加载；
 *            对话与分析都通过 SSE 流式更新本地状态；组件卸载时由 streamAbortRef 中断流。
 */
export default function DashboardPage() {
  // ---- 仓库与会话 ----
  const [repos, setRepos] = useState<Repository[]>([]);
  const [selectedRepoId, setSelectedRepoId] = useState("");
  // 以 repoId 为 key 缓存各仓库的会话列表，避免来回切换时重复请求
  const [conversationMap, setConversationMap] = useState<Record<string, ProjectConversation[]>>({});
  const [selectedConversationId, setSelectedConversationId] = useState("");
  const [expandedRepoIds, setExpandedRepoIds] = useState<Record<string, boolean>>({});
  // ---- 右侧工作台数据：Issue / PR / CI ----
  const [issues, setIssues] = useState<Issue[]>([]);
  const [prs, setPrs] = useState<PullRequest[]>([]);
  const [runs, setRuns] = useState<WorkflowRun[]>([]);
  // ---- 对话区 ----
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  // ---- 工作台面板与筛选 ----
  const [workspaceTab, setWorkspaceTab] = useState<WorkspaceTab>("Issue");
  // 右侧面板宽度，可拖拽调整
  const [workspaceWidth, setWorkspaceWidth] = useState(420);
  const [workspaceQuery, setWorkspaceQuery] = useState("");
  const [workspaceStatus, setWorkspaceStatus] = useState("all");
  const [workspaceOwner, setWorkspaceOwner] = useState("all");
  const [workspaceTime, setWorkspaceTime] = useState("all");
  const [selectedWorkspaceItem, setSelectedWorkspaceItem] = useState<WorkspaceSelection | null>(null);
  const [collapsedGroups, setCollapsedGroups] = useState<Record<string, boolean>>({});
  // ---- 团队 / 记忆 / 知识库（右侧「团队」「记忆&知识库」页签用） ----
  const [teamMembers, setTeamMembers] = useState<TeamMember[]>([]);
  const [memberDraft, setMemberDraft] = useState<TeamMemberDraft>({ name: "", role: "", strengths: "", techStack: "" });
  const [memoryNotes, setMemoryNotes] = useState<MemoryNote[]>([]);
  const [memoryDraft, setMemoryDraft] = useState<MemoryNoteDraft>({ title: "", content: "" });
  const [knowledgeItems, setKnowledgeItems] = useState<KnowledgeItem[]>([]);
  const [uploadingKnowledge, setUploadingKnowledge] = useState(false);
  // ---- 技能 & MCP 运行时（懒加载：只有切到该页签才请求，见下方 useEffect） ----
  const [mcpRuntime, setMcpRuntime] = useState<McpRuntime | null>(null);
  const [skills, setSkills] = useState<SkillResponse[]>([]);
  const [selectedSkillName, setSelectedSkillName] = useState("project-investigation");
  const [skillRuntimeLoading, setSkillRuntimeLoading] = useState(false);
  const [skillRuntimeError, setSkillRuntimeError] = useState<string | null>(null);
  // ---- 项目索引与知识图谱：按仓库缓存，切换仓库不必重新拉 ----
  const [projectIndexByRepo, setProjectIndexByRepo] = useState<Record<string, ProjectIndexState>>({});
  const [indexScanningRepoId, setIndexScanningRepoId] = useState<string | null>(null);
  const [indexActionError, setIndexActionError] = useState<string | null>(null);
  const [knowledgeGraphByRepo, setKnowledgeGraphByRepo] = useState<Record<string, KnowledgeGraphResponse | null>>({});
  const [graphLoadingRepoId, setGraphLoadingRepoId] = useState<string | null>(null);
  const [graphError, setGraphError] = useState<string | null>(null);
  // 图谱的本地筛选条件（前端过滤，不重新请求后端）
  const [graphQuery, setGraphQuery] = useState("");
  const [selectedGraphNodeKey, setSelectedGraphNodeKey] = useState<string | null>(null);
  const [hiddenGraphNodeTypes, setHiddenGraphNodeTypes] = useState<Record<string, boolean>>({});
  const [hiddenGraphRelations, setHiddenGraphRelations] = useState<Record<string, boolean>>({});
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  // ---- 接入新项目弹窗（表单字段与提交状态） ----
  const [isAddProjectOpen, setIsAddProjectOpen] = useState(false);
  const [projectSource, setProjectSource] = useState<AddProjectSource>("remote");
  const [projectUrl, setProjectUrl] = useState("");
  const [projectLocalPath, setProjectLocalPath] = useState("");
  const [projectCloneParentDir, setProjectCloneParentDir] = useState("");
  const [projectProvider, setProjectProvider] = useState("github");
  const [projectApiBaseUrl, setProjectApiBaseUrl] = useState("https://api.github.com");
  const [projectToken, setProjectToken] = useState("");
  const [addProjectStatus, setAddProjectStatus] = useState<AddProjectStatus | null>(null);
  const [connectingProject, setConnectingProject] = useState(false);
  // loading 表示「Agent 正在回答」：既用来禁用发送按钮，也用来决定滚底动画
  const [loading, setLoading] = useState(false);
  // planMode 打开时先让后端产出计划，用户确认后才执行工具链
  const [planMode, setPlanMode] = useState(false);
  // 推荐提问按仓库缓存：suggestions 是展示项，exclude 是已展示过的历史（避免重复推荐）
  const [promptStateByRepo, setPromptStateByRepo] = useState<Record<string, PromptSuggestionState>>({});
  const [loadingPromptRepoId, setLoadingPromptRepoId] = useState<string | null>(null);
  // 加载动画的点数（. → .. → …），纯装饰
  const [promptLoadingDots, setPromptLoadingDots] = useState(".");
  // ---- 三类分析（Issue / PR / CI）的「结果 + 流式轨迹 + 弹窗打开状态」按 id 分组存放 ----
  // 用 Record 而不是单个 state：允许同时保留多次分析结果，切换条目时不必重跑
  const [issueAnalysisById, setIssueAnalysisById] = useState<Record<string, IssueAnalysisResponse>>({});
  const [issueAnalysisTraceById, setIssueAnalysisTraceById] = useState<Record<string, IssueAnalysisTraceItem[]>>({});
  const [analysisModalIssueId, setAnalysisModalIssueId] = useState<string | null>(null);
  const [analyzingIssueId, setAnalyzingIssueId] = useState<string | null>(null);
  const [issueAnalysisError, setIssueAnalysisError] = useState<string | null>(null);
  const [prAnalysisById, setPrAnalysisById] = useState<Record<string, PRAnalysisResponse>>({});
  const [prAnalysisTraceById, setPrAnalysisTraceById] = useState<Record<string, IssueAnalysisTraceItem[]>>({});
  const [analysisModalPrId, setAnalysisModalPrId] = useState<string | null>(null);
  const [analyzingPrId, setAnalyzingPrId] = useState<string | null>(null);
  const [prAnalysisError, setPrAnalysisError] = useState<string | null>(null);
  const [ciAnalysisById, setCiAnalysisById] = useState<Record<string, CIAnalysisResponse>>({});
  const [ciAnalysisTraceById, setCiAnalysisTraceById] = useState<Record<string, IssueAnalysisTraceItem[]>>({});
  const [analysisModalRunId, setAnalysisModalRunId] = useState<string | null>(null);
  const [analyzingRunId, setAnalyzingRunId] = useState<string | null>(null);
  const [ciAnalysisError, setCiAnalysisError] = useState<string | null>(null);
  const [syncing, setSyncing] = useState(false);
  // ---- 若干「标志位」型 ref：只用于跨渲染传信号，不需要触发重渲染，所以用 ref 而不是 state ----
  // skipHistoryLoadRef：刚发出第一条消息时，避免紧接着的历史加载把本地乐观消息覆盖掉
  const skipHistoryLoadRef = useRef(false);
  // preserveWorkspaceSelectionRef：刷新仓库/切换页签时保留右栏选中项
  const preserveWorkspaceSelectionRef = useRef(false);
  // streamAbortRef：保存当前 SSE 的 AbortController，用于「停止生成」和卸载清理
  const streamAbortRef = useRef<AbortController | null>(null);
  const chatScrollRef = useRef<HTMLDivElement>(null);
  // 推荐提问区：row 是可视容器，measure 是隐藏的测量容器（用于算能放几个 chip）
  const promptRowRef = useRef<HTMLDivElement>(null);
  const promptMeasureRef = useRef<HTMLDivElement>(null);
  const [visiblePromptCount, setVisiblePromptCount] = useState(5);

  // ---- 派生数据：全部由已有 state 计算，不额外存一份，避免不一致 ----
  const selectedRepo = useMemo(() => repos.find((repo) => repo.id === selectedRepoId) ?? repos[0], [repos, selectedRepoId]);
  const selectedRepoConversations = selectedRepo ? conversationMap[selectedRepo.id] ?? [] : [];
  const isDraftConversation = isDraftConversationId(selectedConversationId);
  // 草稿会话没有真实 id，因此 selectedConversation 为 undefined；此时会话相关请求都要跳过
  const selectedConversation = isDraftConversation
    ? undefined
    : selectedRepoConversations.find((item) => item.id === selectedConversationId) ?? selectedRepoConversations[0];
  const selectedPromptState = selectedRepo ? promptStateByRepo[selectedRepo.id] : undefined;
  const selectedProjectIndexState = selectedRepo ? projectIndexByRepo[selectedRepo.id] : undefined;
  const selectedKnowledgeGraph = selectedRepo ? knowledgeGraphByRepo[selectedRepo.id] : null;
  const promptSuggestions = selectedPromptState?.suggestions ?? [];
  const promptExclude = selectedPromptState?.exclude ?? [];
  const loadingPrompts = Boolean(selectedRepo?.id && loadingPromptRepoId === selectedRepo.id);

  /**
   * 加载仓库列表及每个仓库的会话。
   * 会话请求用 catch 降级为空数组：某个仓库会话拉不到不该让整页失败。
   * 首次加载时默认展开第一个仓库、选中第一个仓库与它的第一个会话。
   */
  async function loadRepos() {
    const items = await apiGet<Repository[]>("/api/repos");
    setRepos(items);
    const pairs = await Promise.all(
      items.map(async (repo) => [repo.id, await apiGet<ProjectConversation[]>(`/api/repos/${repo.id}/conversations`).catch(() => [])] as const)
    );
    const nextMap = Object.fromEntries(pairs);
    setConversationMap(nextMap);
    setExpandedRepoIds((current) => {
      const firstRepoId = items[0]?.id;
      return firstRepoId && Object.keys(current).length === 0 ? { [firstRepoId]: true } : current;
    });
    setSelectedRepoId((current) => current || items[0]?.id || "");
    setSelectedConversationId((current) => current || (items[0] ? nextMap[items[0].id]?.[0]?.id ?? "" : ""));
  }

  /** 只刷新单个仓库的会话列表并合并进 conversationMap（用于新建/删除会话后） */
  async function loadRepoConversations(repoId: string) {
    const conversations = await apiGet<ProjectConversation[]>(`/api/repos/${repoId}/conversations`).catch(() => []);
    setConversationMap((items) => ({ ...items, [repoId]: conversations }));
    return conversations;
  }

  /**
   * 加载右栏三块列表数据（Issue / PR / CI）。
   * 三个接口互不依赖，用 Promise.all 并行；任一失败降级为空数组，保证右栏至少能渲染其余两块。
   */
  async function loadRepoData(repoId: string) {
    const [issueItems, prItems, runItems] = await Promise.all([
      apiGet<Issue[]>(`/api/repos/${repoId}/issues`).catch(() => []),
      apiGet<PullRequest[]>(`/api/repos/${repoId}/pull-requests`).catch(() => []),
      apiGet<WorkflowRun[]>(`/api/repos/${repoId}/workflow-runs`).catch(() => [])
    ]);
    setIssues(issueItems);
    setPrs(prItems);
    setRuns(runItems);
  }

  /**
   * 加载技能清单与 MCP 运行时状态（GET /api/skills 与 GET /api/skills/mcp-status）。
   * 若当前选中的技能已不在列表中（被卸载），自动回落到第一个，避免右侧面板空白。
   */
  async function loadSkillRuntime() {
    setSkillRuntimeLoading(true);
    setSkillRuntimeError(null);
    try {
      const [skillList, runtime] = await Promise.all([
        apiGet<{ skills: SkillResponse[] }>("/api/skills"),
        apiGet<McpRuntime>("/api/skills/mcp-status")
      ]);
      setSkills(skillList.skills);
      setMcpRuntime(runtime);
      setSelectedSkillName((current) =>
        skillList.skills.some((item) => item.manifest.name === current)
          ? current
          : skillList.skills[0]?.manifest.name ?? ""
      );
    } catch (error) {
      setSkillRuntimeError(getFriendlyError(error));
    } finally {
      setSkillRuntimeLoading(false);
    }
  }

  /**
   * 向 /api/chat/recommendations 请求 5 条推荐提问。
   * exclude 传入已出现过的推荐，避免「换一批」时重复；
   * 请求失败时只更新 exclude（不改 suggestions），这样界面上原有的推荐不会被清空。
   */
  async function loadPromptSuggestions(repoId: string, exclude: string[] = []) {
    setLoadingPromptRepoId(repoId);
    try {
      const result = await apiPost<PromptRecommendationsResponse>("/api/chat/recommendations", {
        repo_id: repoId,
        exclude,
        limit: 5
      });
      const nextSuggestions = result.suggestions;
      setPromptStateByRepo((items) => {
        const current = items[repoId] ?? { suggestions: [], exclude: [] };
        return {
          ...items,
          [repoId]: {
            ...current,
            suggestions: nextSuggestions.length ? nextSuggestions : current.suggestions,
            exclude: nextSuggestions.length ? [...exclude, ...nextSuggestions] : []
          }
        };
      });
    } catch {
      setPromptStateByRepo((items) => {
        const current = items[repoId] ?? { suggestions: [], exclude: [] };
        return {
          ...items,
          [repoId]: {
            ...current,
            exclude
          }
        };
      });
    } finally {
      setLoadingPromptRepoId((current) => (current === repoId ? null : current));
    }
  }

  /**
   * 分析单个 Issue：POST /api/issues/{id}/analyze/stream（SSE）。
   * 流程：先清空该 Issue 的旧结果、塞一条「准备分析」轨迹，再逐条消费服务端事件：
   * - trace：结构化步骤，直接追加；
   * - thinking_delta：模型思考增量，合并到最后一条 thinking 轨迹里（形成流式打字效果）；
   * - 最终结果事件由 finalizedTraceItems 在收尾时统一整理。
   * @sideEffect 打开分析弹窗、持续写 issueAnalysisById / issueAnalysisTraceById。
   */
  async function analyzeIssue(issue: Issue) {
    setAnalysisModalIssueId(issue.id);
    setIssueAnalysisError(null);
    setIssueAnalysisById((items) => {
      const next = { ...items };
      delete next[issue.id];
      return next;
    });
    setIssueAnalysisTraceById((items) => ({
      ...items,
      [issue.id]: [
        {
          id: `${Date.now()}-start`,
          kind: "step",
          title: "准备分析",
          content: `准备分析 Issue #${issue.number}，将流式展示可见的分析过程。`,
          status: "running"
        }
      ]
    }));
    setAnalyzingIssueId(issue.id);
    try {
      const conversationQuery = selectedConversation?.id && !isDraftConversation ? `?conversation_id=${selectedConversation.id}` : "";
      await apiPostStream(
        `/api/issues/${issue.id}/analyze/stream${conversationQuery}`,
        {},
        (event, data) => {
          if (event === "trace") {
            const payload = data as Omit<IssueAnalysisTraceItem, "id">;
            const item: IssueAnalysisTraceItem = { id: `${Date.now()}-${Math.random()}`, ...payload };
            setIssueAnalysisTraceById((items) => ({
              ...items,
              [issue.id]: [...(items[issue.id] ?? []), item]
            }));
            return;
          }
          if (event === "thinking_delta") {
            const payload = data as { content?: string };
            const content = payload.content ?? "";
            if (!content) return;
            setIssueAnalysisTraceById((items) => {
              const current = items[issue.id] ?? [];
              const last = current[current.length - 1];
              if (last?.kind === "thinking" && last.status === "running") {
                return {
                  ...items,
                  [issue.id]: [...current.slice(0, -1), { ...last, content: `${last.content}${content}` }]
                };
              }
              return {
                ...items,
                [issue.id]: [
                  ...current,
                  {
                    id: `${Date.now()}-${Math.random()}`,
                    kind: "thinking",
                    title: "LLM 公开分析过程",
                    content,
                    status: "running"
                  }
                ]
              };
            });
            return;
          }
          if (event === "final") {
            const result = data as IssueAnalysisResponse;
            setIssueAnalysisById((items) => ({ ...items, [issue.id]: result }));
            setIssueAnalysisTraceById((items) => {
              const current = (items[issue.id] ?? []).map((item) => (item.kind === "thinking" ? { ...item, status: "done" } : item));
              return {
                ...items,
                [issue.id]: [
                  ...current,
                  {
                    id: `${Date.now()}-final`,
                    kind: "result",
                    title: "分析完成",
                    content: `已生成结构化结论：${result.result.conclusion}，优先级 ${result.result.priority}，复杂度 ${result.result.complexity}。`,
                    status: "done"
                  }
                ]
              };
            });
            return;
          }
          if (event === "error") {
            const payload = data as { message?: string };
            const message = payload.message || "分析失败";
            setIssueAnalysisError(message);
            setIssueAnalysisTraceById((items) => ({
              ...items,
              [issue.id]: [
                ...(items[issue.id] ?? []),
                { id: `${Date.now()}-error`, kind: "error", title: "分析失败", content: message, status: "error" }
              ]
            }));
          }
        }
      );
    } catch (error) {
      setIssueAnalysisError(getFriendlyError(error));
      setIssueAnalysisTraceById((items) => ({
        ...items,
        [issue.id]: [
          ...(items[issue.id] ?? []),
          { id: `${Date.now()}-error`, kind: "error", title: "请求失败", content: getFriendlyError(error), status: "error" }
        ]
      }));
    } finally {
      setAnalyzingIssueId((current) => (current === issue.id ? null : current));
    }
  }

  /** 分析单个 PR：与 analyzeIssue 同一套 SSE 协议，结果存 prAnalysisById / prAnalysisTraceById */
  async function analyzePr(pr: PullRequest) {
    setAnalysisModalPrId(pr.id);
    setPrAnalysisError(null);
    setPrAnalysisById((items) => {
      const next = { ...items };
      delete next[pr.id];
      return next;
    });
    setPrAnalysisTraceById((items) => ({
      ...items,
      [pr.id]: [
        {
          id: `${Date.now()}-start`,
          kind: "step",
          title: "准备分析",
          content: `准备分析 PR #${pr.number}，将按 plan-and-execute 方式展示 Review 计划、执行过程和合入建议。`,
          status: "running"
        }
      ]
    }));
    setAnalyzingPrId(pr.id);
    try {
      const conversationQuery = selectedConversation?.id && !isDraftConversation ? `?conversation_id=${selectedConversation.id}` : "";
      await apiPostStream(
        `/api/pull-requests/${pr.id}/analyze/stream${conversationQuery}`,
        {},
        (event, data) => {
          if (event === "trace") {
            const payload = data as Omit<IssueAnalysisTraceItem, "id">;
            const item: IssueAnalysisTraceItem = { id: `${Date.now()}-${Math.random()}`, ...payload };
            setPrAnalysisTraceById((items) => ({
              ...items,
              [pr.id]: [...(items[pr.id] ?? []), item]
            }));
            return;
          }
          if (event === "thinking_delta") {
            const payload = data as { content?: string };
            const content = payload.content ?? "";
            if (!content) return;
            setPrAnalysisTraceById((items) => {
              const current = items[pr.id] ?? [];
              const last = current[current.length - 1];
              if (last?.kind === "thinking" && last.status === "running") {
                return {
                  ...items,
                  [pr.id]: [...current.slice(0, -1), { ...last, content: `${last.content}${content}` }]
                };
              }
              return {
                ...items,
                [pr.id]: [
                  ...current,
                  {
                    id: `${Date.now()}-${Math.random()}`,
                    kind: "thinking",
                    title: "LLM 公开执行过程",
                    content,
                    status: "running"
                  }
                ]
              };
            });
            return;
          }
          if (event === "final") {
            const result = data as PRAnalysisResponse;
            setPrAnalysisById((items) => ({ ...items, [pr.id]: result }));
            setPrAnalysisTraceById((items) => {
              const current = (items[pr.id] ?? []).map((item) => (item.kind === "thinking" ? { ...item, status: "done" } : item));
              return {
                ...items,
                [pr.id]: [
                  ...current,
                  {
                    id: `${Date.now()}-final`,
                    kind: "result",
                    title: "分析完成",
                    content: `已生成 PR 合入建议：${result.result.merge_recommendation}，风险点 ${(result.result.risk_points ?? []).length} 个，阻塞项 ${(result.result.blocking_issues ?? []).length} 个。`,
                    status: "done"
                  }
                ]
              };
            });
            return;
          }
          if (event === "error") {
            const payload = data as { message?: string };
            const message = payload.message || "分析失败";
            setPrAnalysisError(message);
            setPrAnalysisTraceById((items) => ({
              ...items,
              [pr.id]: [
                ...(items[pr.id] ?? []),
                { id: `${Date.now()}-error`, kind: "error", title: "分析失败", content: message, status: "error" }
              ]
            }));
          }
        }
      );
    } catch (error) {
      setPrAnalysisError(getFriendlyError(error));
      setPrAnalysisTraceById((items) => ({
        ...items,
        [pr.id]: [
          ...(items[pr.id] ?? []),
          { id: `${Date.now()}-error`, kind: "error", title: "请求失败", content: getFriendlyError(error), status: "error" }
        ]
      }));
    } finally {
      setAnalyzingPrId((current) => (current === pr.id ? null : current));
    }
  }

  /** 分析一次 CI 失败：同样走 SSE，把失败归因与排查建议流式展示到弹窗 */
  async function analyzeCi(run: WorkflowRun) {
    setAnalysisModalRunId(run.id);
    setCiAnalysisError(null);
    setCiAnalysisById((items) => {
      const next = { ...items };
      delete next[run.id];
      return next;
    });
    setCiAnalysisTraceById((items) => ({
      ...items,
      [run.id]: [
        {
          id: `${Date.now()}-start`,
          kind: "step",
          title: "准备排查",
          content: `准备排查 CI「${run.name}」，将按 plan-and-execute 方式展示排查计划、执行过程、根因和阻塞判断。`,
          status: "running"
        }
      ]
    }));
    setAnalyzingRunId(run.id);
    try {
      const conversationQuery = selectedConversation?.id && !isDraftConversation ? `?conversation_id=${selectedConversation.id}` : "";
      await apiPostStream(
        `/api/workflow-runs/${run.id}/analyze/stream${conversationQuery}`,
        {},
        (event, data) => {
          if (event === "trace") {
            const payload = data as Omit<IssueAnalysisTraceItem, "id">;
            const item: IssueAnalysisTraceItem = { id: `${Date.now()}-${Math.random()}`, ...payload };
            setCiAnalysisTraceById((items) => ({
              ...items,
              [run.id]: [...(items[run.id] ?? []), item]
            }));
            return;
          }
          if (event === "thinking_delta") {
            const payload = data as { content?: string };
            const content = payload.content ?? "";
            if (!content) return;
            setCiAnalysisTraceById((items) => {
              const current = items[run.id] ?? [];
              const last = current[current.length - 1];
              if (last?.kind === "thinking" && last.status === "running") {
                return {
                  ...items,
                  [run.id]: [...current.slice(0, -1), { ...last, content: `${last.content}${content}` }]
                };
              }
              return {
                ...items,
                [run.id]: [
                  ...current,
                  {
                    id: `${Date.now()}-${Math.random()}`,
                    kind: "thinking",
                    title: "LLM 公开排查过程",
                    content,
                    status: "running"
                  }
                ]
              };
            });
            return;
          }
          if (event === "final") {
            const result = data as CIAnalysisResponse;
            setCiAnalysisById((items) => ({ ...items, [run.id]: result }));
            setCiAnalysisTraceById((items) => {
              const current = (items[run.id] ?? []).map((item) => (item.kind === "thinking" ? { ...item, status: "done" } : item));
              return {
                ...items,
                [run.id]: [
                  ...current,
                  {
                    id: `${Date.now()}-final`,
                    kind: "result",
                    title: "排查完成",
                    content: `已生成 CI 根因判断：${result.result.failure_type}；${result.result.is_merge_blocking ? "阻塞合并" : "不阻塞合并"}。`,
                    status: "done"
                  }
                ]
              };
            });
            return;
          }
          if (event === "error") {
            const payload = data as { message?: string };
            const message = payload.message || "分析失败";
            setCiAnalysisError(message);
            setCiAnalysisTraceById((items) => ({
              ...items,
              [run.id]: [
                ...(items[run.id] ?? []),
                { id: `${Date.now()}-error`, kind: "error", title: "排查失败", content: message, status: "error" }
              ]
            }));
          }
        }
      );
    } catch (error) {
      setCiAnalysisError(getFriendlyError(error));
      setCiAnalysisTraceById((items) => ({
        ...items,
        [run.id]: [
          ...(items[run.id] ?? []),
          { id: `${Date.now()}-error`, kind: "error", title: "请求失败", content: getFriendlyError(error), status: "error" }
        ]
      }));
    } finally {
      setAnalyzingRunId((current) => (current === run.id ? null : current));
    }
  }

  /** 加载项目知识库条目（团队/记忆页签与知识上传共用） */
  async function loadProjectKnowledge(repoId: string) {
    const [members, knowledge] = await Promise.all([
      apiGet<TeamMember[]>(`/api/repos/${repoId}/team-members`).catch(() => []),
      apiGet<KnowledgeItem[]>(`/api/repos/${repoId}/knowledge`).catch(() => [])
    ]);
    setTeamMembers(members);
    setKnowledgeItems(knowledge);
    setMemoryNotes(
      knowledge
        .filter((item) => item.source_type === "memory_note")
        .map((item) => ({
          id: item.id,
          title: item.title,
          content: item.content_preview
      }))
    );
  }

  /** 加载项目索引状态（missing/stale/building/ready/failed），决定是否提示「建立索引」 */
  async function loadProjectIndexState(repoId: string) {
    const state = await apiGet<ProjectIndexState>(`/api/repos/${repoId}/memory-index/state`).catch(() => null);
    if (state) {
      setProjectIndexByRepo((items) => ({ ...items, [repoId]: state }));
    }
    return state;
  }

  /**
   * 加载知识图谱。
   * @param center 可选的中心节点，展开某节点时会传，用来让后端按该节点视角返回关系
   */
  async function loadKnowledgeGraph(repoId: string, center?: { type: string; id: string }) {
    setGraphLoadingRepoId(repoId);
    setGraphError(null);
    try {
      const params = new URLSearchParams({ depth: "1" });
      if (center?.type && center.id) {
        params.set("center_type", center.type);
        params.set("center_id", center.id);
      }
      const graph = await apiGet<KnowledgeGraphResponse>(`/api/repos/${repoId}/knowledge/graph?${params.toString()}`);
      setKnowledgeGraphByRepo((items) => ({ ...items, [repoId]: graph }));
      const preferred = center ? graph.nodes.find((node) => node.type === center.type && node.id === center.id) : graph.nodes[0];
      setSelectedGraphNodeKey(preferred ? knowledgeGraphNodeKey(preferred) : null);
      return graph;
    } catch (error) {
      setGraphError(getFriendlyError(error));
      setKnowledgeGraphByRepo((items) => ({ ...items, [repoId]: null }));
      return null;
    } finally {
      setGraphLoadingRepoId((current) => (current === repoId ? null : current));
    }
  }

  /** 重建当前仓库的知识图谱（后端重新抽取实体与关系），完成后重新拉取并清空选中节点 */
  async function rebuildSelectedKnowledgeGraph() {
    if (!selectedRepo) return;
    setGraphLoadingRepoId(selectedRepo.id);
    setGraphError(null);
    try {
      await apiPost(`/api/repos/${selectedRepo.id}/knowledge/graph/rebuild`, {});
      await loadKnowledgeGraph(selectedRepo.id);
    } catch (error) {
      setGraphError(getFriendlyError(error));
    } finally {
      setGraphLoadingRepoId((current) => (current === selectedRepo.id ? null : current));
    }
  }

  /**
   * 触发项目记忆索引扫描（POST /api/repos/{id}/memory-index/scan），并轮询到结束。
   * 为什么用轮询而不是等待接口返回：扫描是后台长任务，接口只是把它排进队列。
   * 轮询上限 45 次 × 1.2 秒 ≈ 54 秒，超时后不再等待（避免无限占用）；ready 时才刷新知识与图谱。
   */
  async function startProjectIndexScan() {
    if (!selectedRepo) return;
    const repoId = selectedRepo.id;
    setIndexActionError(null);
    setIndexScanningRepoId(repoId);
    setProjectIndexByRepo((items) => ({
      ...items,
      [repoId]: {
        repo_id: repoId,
        status: "building",
        fingerprint: items[repoId]?.fingerprint ?? "",
        docs_indexed: items[repoId]?.docs_indexed ?? 0,
        docs_total: items[repoId]?.docs_total ?? 0,
        error_message: null,
        summary_json: items[repoId]?.summary_json ?? null,
        snoozed_until: null,
        last_scan_at: items[repoId]?.last_scan_at ?? null
      }
    }));
    try {
      await apiPost(`/api/repos/${repoId}/memory-index/scan`, {});
      let latest: ProjectIndexState | null = null;
      for (let attempt = 0; attempt < 45; attempt += 1) {
        await wait(1200);
        latest = await loadProjectIndexState(repoId);
        if (!latest || latest.status !== "building") break;
      }
      if (latest?.status === "ready") {
        await loadProjectKnowledge(repoId);
        await loadKnowledgeGraph(repoId);
      }
      if (latest?.status === "failed") {
        setIndexActionError(latest.error_message || "记忆索引扫描失败");
      }
    } catch (error) {
      setIndexActionError(getFriendlyError(error));
      await loadProjectIndexState(repoId);
    } finally {
      setIndexScanningRepoId((current) => (current === repoId ? null : current));
    }
  }

  /** 把「建立索引」提示延后 7 天（用户暂时不想建索引时用），随后刷新状态让提示消失 */
  async function snoozeProjectIndexPrompt() {
    if (!selectedRepo) return;
    const repoId = selectedRepo.id;
    try {
      await apiPost(`/api/repos/${repoId}/memory-index/snooze`, { days: 7 });
      await loadProjectIndexState(repoId);
    } catch (error) {
      setIndexActionError(getFriendlyError(error));
    }
  }

  /** 空会话时的欢迎消息（本地生成，不落库）；id 带 repo/conversation 前缀保证唯一 */
  function welcomeMessage(repo: Repository, conversation?: ProjectConversation): ChatMessage {
    return {
      id: `welcome-${repo.id}-${conversation?.id ?? "default"}`,
      role: "assistant",
      content: `已切换到仓库 ${repo.full_name}${conversation ? ` / ${conversation.title}` : ""}。你可以直接问我 Issue、PR、CI 或周报相关问题。`
    };
  }

  /**
   * 加载会话的完整历史消息 + 每条消息的反馈记录。
   * 历史接口是游标分页（before_message_id 往前翻），这里循环拉到不足一页为止，
   * 再用 unshift 把后拉到的更早消息插到前面，保证最终数组是时间正序。
   * 同时并行拉反馈，回填到消息上，让点赞/点踩状态在刷新后仍然保留。
   */
  async function loadChatHistory(repo: Repository, conversation: ProjectConversation) {
    async function loadCompleteHistory() {
      const pageSize = 500;
      const history: ChatHistoryMessage[] = [];
      let beforeMessageId: string | undefined;
      while (true) {
        const cursor = beforeMessageId ? `&before_message_id=${beforeMessageId}` : "";
        const page = await apiGet<ChatHistoryMessage[]>(
          `/api/chat/history?repo_id=${repo.id}&conversation_id=${conversation.id}&limit=${pageSize}${cursor}`
        );
        history.unshift(...page);
        if (page.length < pageSize) break;
        const nextCursor = page[0]?.id;
        if (!nextCursor || nextCursor === beforeMessageId) break;
        beforeMessageId = nextCursor;
      }
      return history;
    }

    const [history, feedbackRows] = await Promise.all([
      loadCompleteHistory(),
      apiGet<ChatFeedbackRecord[]>(`/api/chat/feedback?repo_id=${repo.id}&conversation_id=${conversation.id}&limit=200`)
    ]);
    if (!history.length) {
      setMessages([welcomeMessage(repo, conversation)]);
      return;
    }
    const feedbackByMessage = new Map(feedbackRows.map((item) => [item.assistant_message_id, item]));
    setMessages(
      history
        .filter((item) => item.role === "user" || item.role === "assistant")
        .map((item) => ({
          id: item.id,
          messageId: item.id,
          role: item.role,
          content: item.content,
          route: item.route ?? undefined,
          toolCalls: item.tool_calls ?? [],
          traceItems: traceItemsFromAgentEvents(item.agent_events ?? []),
          traceCollapsed: true,
          compressionStats: item.compression_stats,
          progressiveCompaction: item.progressive_compaction,
          modelUsage: item.model_usage,
          contextDiagnostics: item.context_diagnostics,
          contextCollapsed: true,
          responsePending: false,
          feedback: feedbackByMessage.get(item.id)
        }))
    );
  }

  // 首屏：加载仓库与会话。失败时退化为空仓库列表，界面仍可点「接入项目」。
  useEffect(() => {
    loadRepos().catch(() => setRepos([]));
  }, []);

  // 技能与 MCP 运行时按需加载：只有切到「技能&MCP」页签且尚未加载过才请求，
  // 避免首页一进来就发一个多数用户用不到的请求。
  useEffect(() => {
    if (workspaceTab === "技能&MCP" && !mcpRuntime && !skillRuntimeLoading) {
      void loadSkillRuntime();
    }
  }, [workspaceTab]);

  // 切换仓库：并行拉取该仓库的全部数据（Issue/PR/CI、知识、索引状态、图谱），
  // 并在没有推荐提问缓存时生成一次。没有选中仓库时清空右栏相关状态，避免展示上一个仓库的残留。
  useEffect(() => {
    if (selectedRepo?.id) {
      loadRepoData(selectedRepo.id);
      loadProjectKnowledge(selectedRepo.id);
      loadProjectIndexState(selectedRepo.id);
      loadKnowledgeGraph(selectedRepo.id);
      if (!promptStateByRepo[selectedRepo.id]) {
        loadPromptSuggestions(selectedRepo.id).catch(() => undefined);
      }
    } else {
      setTeamMembers([]);
      setMemoryNotes([]);
      setKnowledgeItems([]);
      setKnowledgeGraphByRepo({});
      setSelectedGraphNodeKey(null);
      setMessages([]);
    }
  }, [selectedRepo?.id]);

  // 推荐提问加载时的省略号动画（. → .. → ...），纯视觉反馈；不加载时立即复位
  useEffect(() => {
    if (!loadingPrompts) {
      setPromptLoadingDots(".");
      return;
    }
    const timer = window.setInterval(() => {
      setPromptLoadingDots((current) => (current.length >= 3 ? "." : `${current}.`));
    }, 360);
    return () => window.clearInterval(timer);
  }, [loadingPrompts]);

  // 保证「选中的会话一定属于当前仓库」：仓库还没有会话时拉一次并选中第一条；
  // 已有会话但当前选中的 id 不在其中（如刚切换仓库）则回落到第一条。草稿会话跳过此逻辑。
  useEffect(() => {
    if (!selectedRepo?.id) return;
    if (isDraftConversation) return;
    const conversations = conversationMap[selectedRepo.id] ?? [];
    if (conversations.length === 0) {
      loadRepoConversations(selectedRepo.id).then((items) => {
        setSelectedConversationId((current) => (items.some((item) => item.id === current) ? current : items[0]?.id || ""));
      });
      return;
    }
    if (!conversations.some((item) => item.id === selectedConversationId)) {
      setSelectedConversationId(conversations[0].id);
    }
  }, [selectedRepo?.id, conversationMap, selectedConversationId, isDraftConversation]);

  // 切换会话时加载历史消息。skipHistoryLoadRef 用于跳过「刚创建会话并已插入本地消息」的那一次触发，
  // 否则会被后端历史覆盖掉刚发出的消息；加载失败时退回一条欢迎语，保证界面不空白。
  useEffect(() => {
    if (isDraftConversation) return;
    if (skipHistoryLoadRef.current) {
      skipHistoryLoadRef.current = false;
      return;
    }
    if (selectedRepo?.id && selectedConversation?.id) {
      loadChatHistory(selectedRepo, selectedConversation).catch(() => setMessages([welcomeMessage(selectedRepo, selectedConversation)]));
    }
  }, [selectedRepo?.id, selectedConversation?.id, isDraftConversation]);

  // 切换仓库或右栏页签时重置所有筛选与弹窗状态：
  // 筛选条件与当前数据强相关（比如 CI 状态筛选不该带到 Issue 列表），
  // 弹窗也要关掉，避免展示上一个仓库的分析结果。
  useEffect(() => {
    setWorkspaceQuery("");
    setWorkspaceStatus("all");
    setWorkspaceOwner("all");
    setWorkspaceTime("all");
    if (preserveWorkspaceSelectionRef.current) {
      preserveWorkspaceSelectionRef.current = false;
    } else {
      setSelectedWorkspaceItem(null);
    }
    setGraphQuery("");
    setHiddenGraphNodeTypes({});
    setHiddenGraphRelations({});
    setAnalysisModalIssueId(null);
    setIssueAnalysisError(null);
    setAnalysisModalPrId(null);
    setPrAnalysisError(null);
    setAnalysisModalRunId(null);
    setCiAnalysisError(null);
    setCollapsedGroups({});
  }, [selectedRepo?.id, workspaceTab]);

  // 新消息或进入加载态时自动滚到对话底部：
  // 加载中用 smooth 让读者跟得上流式输出，非加载态（如切换会话回放历史）直接跳转，避免长距离滚动动画。
  useEffect(() => {
    const node = chatScrollRef.current;
    if (!node) return;
    node.scrollTo({ top: node.scrollHeight, behavior: loading ? "smooth" : "auto" });
  }, [messages, loading]);

  // 计算推荐提问「一行能放几个」：用隐藏的测量容器逐个累加 chip 宽度，
  // 放不下就截断。配合 ResizeObserver 在窗口/面板宽度变化时重算，避免出现半截被裁掉的 chip。
  useEffect(() => {
    if (promptSuggestions.length === 0) {
      setVisiblePromptCount(5);
      return;
    }

    function updateVisiblePromptCount() {
      const row = promptRowRef.current;
      const measure = promptMeasureRef.current;
      if (!row || !measure) return;
      const rowWidth = row.clientWidth;
      const chips = Array.from(measure.querySelectorAll<HTMLButtonElement>("[data-prompt-chip-measure]"));
      if (!rowWidth || chips.length === 0) return;

      const gap = 8;
      let usedWidth = 0;
      let nextCount = 0;
      for (const chip of chips) {
        const chipWidth = Math.ceil(chip.getBoundingClientRect().width);
        const widthWithGap = chipWidth + (nextCount > 0 ? gap : 0);
        if (nextCount === 0 || usedWidth + widthWithGap <= rowWidth) {
          usedWidth += widthWithGap;
          nextCount += 1;
          continue;
        }
        break;
      }
      setVisiblePromptCount(Math.min(promptSuggestions.length, Math.max(1, nextCount)));
    }

    const frame = window.requestAnimationFrame(updateVisiblePromptCount);
    let observer: ResizeObserver | null = null;
    if (typeof ResizeObserver !== "undefined" && promptRowRef.current) {
      observer = new ResizeObserver(updateVisiblePromptCount);
      observer.observe(promptRowRef.current);
    }
    window.addEventListener("resize", updateVisiblePromptCount);
    return () => {
      window.cancelAnimationFrame(frame);
      observer?.disconnect();
      window.removeEventListener("resize", updateVisiblePromptCount);
    };
  }, [promptSuggestions, loadingPrompts]);

  /**
   * 开始拖拽调整右栏宽度。
   * 用 window 级别的 mousemove/mouseup 而不是元素级：鼠标移出元素后仍能继续拖动，
   * 松手时移除监听，避免监听器泄漏。宽度限制在 360~640 之间，防止布局被拖坏。
   */
  function startResize(event: MouseEvent<HTMLDivElement>) {
    event.preventDefault();
    const startX = event.clientX;
    const startWidth = workspaceWidth;

    function onMove(moveEvent: globalThis.MouseEvent) {
      const delta = startX - moveEvent.clientX;
      setWorkspaceWidth(Math.min(640, Math.max(360, startWidth + delta)));
    }

    function onUp() {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    }

    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  }

  /** 折叠/展开右栏的分组（按 key 记录，未记录视为展开） */
  function toggleCategory(key: string) {
    setCollapsedGroups((items) => ({ ...items, [key]: !items[key] }));
  }

  /** 在侧栏点击仓库：切换选中仓库、切换其展开状态，并保证选中会话属于该仓库 */
  async function selectProject(repo: Repository) {
    setSelectedRepoId(repo.id);
    setExpandedRepoIds((items) => ({ ...items, [repo.id]: !items[repo.id] }));
    const conversations = conversationMap[repo.id]?.length ? conversationMap[repo.id] : await loadRepoConversations(repo.id);
    setSelectedConversationId((current) => (conversations.some((item) => item.id === current) ? current : conversations[0]?.id || ""));
  }

  /**
   * 切换会话。除了本地状态，还要 POST /api/repos/{id}/select 让后端记住「当前活跃仓库」，
   * 这样其它页面（issues / PR 等）默认就指向同一个上下文。
   */
  async function selectConversation(repo: Repository, conversation: ProjectConversation) {
    setSelectedRepoId(repo.id);
    setExpandedRepoIds((items) => ({ ...items, [repo.id]: true }));
    setSelectedConversationId(conversation.id);
    await apiPost(`/api/repos/${repo.id}/select`, {});
  }

  /**
   * 新建会话：这里只创建「本地草稿」（draft id），不请求后端。
   * 真正的会话记录在用户发出第一条消息时由 ensureActiveConversation 创建，
   * 这样避免产生一堆空会话。
   */
  async function addConversation(repo: Repository) {
    setExpandedRepoIds((items) => ({ ...items, [repo.id]: true }));
    setSelectedRepoId(repo.id);
    setSelectedConversationId(makeDraftConversationId(repo.id));
    setMessages([]);
    setMessage("");
  }

  /**
   * 删除会话（只删聊天上下文，不动知识库数据，确认文案里已说明）。
   * 后端返回删除后的会话列表与新的 active_conversation_id，直接采用它们，
   * 省掉一次列表请求，也保证删的正是当前选中的会话时能立刻切到有效会话。
   */
  async function removeConversation(repo: Repository, conversation: ProjectConversation) {
    const confirmed = window.confirm(`确定删除会话“${conversation.title}”吗？\n\n只会删除这个会话的聊天上下文，不会删除项目知识库。`);
    if (!confirmed) return;
    const result = await apiDelete<ConversationDeleteResult>(`/api/repos/${repo.id}/conversations/${conversation.id}`);
    setConversationMap((items) => ({ ...items, [repo.id]: result.conversations }));
    if (selectedConversationId === conversation.id) {
      setSelectedRepoId(repo.id);
      setSelectedConversationId(result.active_conversation_id);
    }
  }

  /**
   * 新增团队成员。成员信息会进入知识图谱与记忆检索（作为「谁负责什么」的来源），
   * 所以创建后要顺带刷新知识与图谱两块数据。
   */
  async function addTeamMember() {
    if (!selectedRepo || !memberDraft.name.trim()) return;
    const created = await apiPost<TeamMember>(`/api/repos/${selectedRepo.id}/team-members`, {
      name: memberDraft.name.trim(),
      role: memberDraft.role.trim(),
      strengths: memberDraft.strengths.trim(),
      techStack: memberDraft.techStack.trim()
    });
    setTeamMembers((items) => [...items.filter((item) => item.id !== created.id), created]);
    setMemberDraft({ name: "", role: "", strengths: "", techStack: "" });
    await loadProjectKnowledge(selectedRepo.id);
    await loadKnowledgeGraph(selectedRepo.id);
  }

  /** 删除团队成员；同理需要刷新知识库与图谱，避免残留已删除成员的关系 */
  async function removeTeamMember(memberId: string) {
    if (!selectedRepo) return;
    await apiDelete(`/api/repos/${selectedRepo.id}/team-members/${memberId}`);
    setTeamMembers((items) => items.filter((item) => item.id !== memberId));
    await loadProjectKnowledge(selectedRepo.id);
    await loadKnowledgeGraph(selectedRepo.id);
  }

  /** 新增记忆笔记（POST /api/repos/{id}/knowledge/notes）：标题与正文至少填一项，避免写入空记录 */
  async function addMemoryNote() {
    if (!selectedRepo || (!memoryDraft.title.trim() && !memoryDraft.content.trim())) return;
    await apiPost<KnowledgeItem>(`/api/repos/${selectedRepo.id}/knowledge/notes`, {
      title: memoryDraft.title.trim(),
      content: memoryDraft.content.trim()
    });
    setMemoryDraft({ title: "", content: "" });
    await loadProjectKnowledge(selectedRepo.id);
    await loadKnowledgeGraph(selectedRepo.id);
  }

  /** 删除记忆笔记：删除接口按 source_type 区分数据来源，笔记固定传 memory_note */
  async function removeMemoryNote(noteId: string) {
    if (!selectedRepo) return;
    await apiDelete(`/api/repos/${selectedRepo.id}/knowledge/${noteId}?source_type=memory_note`);
    setMemoryNotes((items) => items.filter((item) => item.id !== noteId));
    await loadProjectKnowledge(selectedRepo.id);
    await loadKnowledgeGraph(selectedRepo.id);
  }

  /**
   * 上传知识文件（multipart → POST /api/repos/{id}/knowledge/upload）。
   * 后端解析/切分/向量化是异步的，这里只负责上传与刷新列表；
   * 失败不做提示（由 uploading 状态回弹表现），保持首页交互轻量。
   */
  async function uploadKnowledgeFile(file: File) {
    if (!selectedRepo) return;
    setUploadingKnowledge(true);
    try {
      const formData = new FormData();
      formData.append("file", file);
      await apiUpload(`/api/repos/${selectedRepo.id}/knowledge/upload`, formData);
      await loadProjectKnowledge(selectedRepo.id);
      await loadKnowledgeGraph(selectedRepo.id);
    } finally {
      setUploadingKnowledge(false);
    }
  }

  /** 删除知识库条目：source_type 作为查询参数，因为同一 id 可能来自不同类型的数据源 */
  async function removeKnowledgeItem(item: KnowledgeItem) {
    if (!selectedRepo) return;
    await apiDelete(`/api/repos/${selectedRepo.id}/knowledge/${item.id}?source_type=${encodeURIComponent(item.source_type)}`);
    await loadProjectKnowledge(selectedRepo.id);
    await loadKnowledgeGraph(selectedRepo.id);
  }

  /**
   * 「刷新项目」：先让后端同步 GitHub 数据（issues / PR / CI，limit=50），
   * 再并行刷新页面上所有依赖该仓库的数据；同时清掉该仓库的推荐提问缓存并重新生成，
   * 因为旧推荐是基于旧数据生成的，同步后可能已经过时。
   */
  async function refreshSelectedRepo() {
    if (!selectedRepo) return;
    setSyncing(true);
    try {
      await apiPost(`/api/repos/${selectedRepo.id}/sync`, {
        sync_issues: true,
        sync_pull_requests: true,
        sync_workflow_runs: true,
        limit: 50
      });
      await Promise.all([
        loadRepos(),
        loadRepoData(selectedRepo.id),
        loadProjectKnowledge(selectedRepo.id),
        loadProjectIndexState(selectedRepo.id),
        loadKnowledgeGraph(selectedRepo.id),
        loadRepoConversations(selectedRepo.id)
      ]);
      setPromptStateByRepo((items) => {
        const next = { ...items };
        delete next[selectedRepo.id];
        return next;
      });
      await loadPromptSuggestions(selectedRepo.id);
    } finally {
      setSyncing(false);
    }
  }

  /** 打开「接入项目」弹窗，并清空上一次的提交结果提示 */
  function openAddProject() {
    setIsAddProjectOpen(true);
    setAddProjectStatus(null);
  }

  /** 关闭弹窗；连接进行中时忽略关闭操作，避免请求还在跑却把界面状态丢掉 */
  function closeAddProject() {
    if (connectingProject) return;
    setIsAddProjectOpen(false);
  }

  /**
   * 接入新项目：前端先解析输入（支持完整 URL 或 owner/repo），再 POST /api/repos/connect。
   * 成功后并行预加载该仓库的 Issue/PR/CI、知识、索引状态，并切到它的第一个会话，
   * 让用户点完「连接」立刻能开始提问，而不是面对空页面。
   * 提示文案按 sync_status 区分：completed（已同步具体数量）/ fallback（本地缓存模式）/ 其他。
   */
  async function connectProject() {
    const isLocalSource = projectSource === "local";
    const parsed = isLocalSource ? null : parseRepositoryInput(projectUrl);
    if (!isLocalSource && !parsed) {
      setAddProjectStatus({
        type: "error",
        title: "连接失败",
        detail: "请输入 GitHub 仓库地址，或 owner/repo 格式，例如 https://github.com/openai/codex。"
      });
      return;
    }
    if (isLocalSource && !projectLocalPath.trim()) {
      setAddProjectStatus({
        type: "error",
        title: "连接失败",
        detail: "请输入本地 Git 仓库路径。"
      });
      return;
    }

    setConnectingProject(true);
    setAddProjectStatus(null);
    try {
      const connected = await apiPost<RepoConnectResult>("/api/repos/connect", {
        owner: parsed?.owner,
        repo: parsed?.repo,
        local_path: isLocalSource ? projectLocalPath.trim() : undefined,
        clone_parent_dir: !isLocalSource && projectCloneParentDir.trim() ? projectCloneParentDir.trim() : undefined,
        provider: projectProvider,
        api_base_url: projectProvider === "github" ? undefined : projectApiBaseUrl.trim(),
        token: projectToken.trim() || undefined,
        demo_mode: false
      });
      await Promise.all([
        loadRepos(),
        loadRepoData(connected.repo_id),
        loadProjectKnowledge(connected.repo_id),
        loadProjectIndexState(connected.repo_id)
      ]);
      const conversations = await loadRepoConversations(connected.repo_id);
      setSelectedRepoId(connected.repo_id);
      setSelectedConversationId(conversations[0]?.id || "");
      setExpandedRepoIds((items) => ({ ...items, [connected.repo_id]: true }));
      setProjectUrl("");
      setProjectLocalPath("");
      setProjectCloneParentDir("");
      setProjectToken("");
      const synced = connected.synced ?? {};
      const syncDetail =
        connected.sync_status === "completed"
          ? `已同步 ${synced.issues ?? 0} 个 Issue、${synced.pull_requests ?? 0} 个 PR、${synced.workflow_runs ?? 0} 条 CI 记录`
          : connected.sync_status === "fallback"
            ? `已使用本地缓存模式连接。${connected.message}`
            : "已连接仓库";
      const codePathDetail = connected.local_path ? `代码路径：${connected.local_path}` : "代码同步和代码图分析已在后台启动";
      setAddProjectStatus({
        type: "success",
        title: "连接成功",
        detail: `${connected.full_name} 已添加为当前项目，${syncDetail}。${codePathDetail}。`
      });
    } catch (error) {
      setAddProjectStatus({
        type: "error",
        title: "连接失败",
        detail: getFriendlyError(error)
      });
    } finally {
      setConnectingProject(false);
    }
  }

  /**
   * 移除项目：删除后端数据后，还要清理所有以该仓库为 key 的前端缓存
   * （会话、展开状态、推荐提问），否则侧栏会残留已删除项目的痕迹。
   * 若删的是当前选中项，则自动切到剩余仓库的第一个并清空右侧列表与对话。
   */
  async function removeRepo(repo: Repository) {
    const confirmed = window.confirm(`确定移除项目「${repo.full_name}」吗？\n\n这会删除本地保存的项目、Issue、PR、CI、分析结果和对话记录。`);
    if (!confirmed) return;

    try {
      await apiDelete(`/api/repos/${repo.id}`);
    } catch (error) {
      window.alert(`删除项目失败：${getFriendlyError(error)}`);
      return;
    }

    const remainingRepos = repos.filter((item) => item.id !== repo.id);
    setRepos(remainingRepos);
    setConversationMap((items) => {
      const next = { ...items };
      delete next[repo.id];
      return next;
    });
    setExpandedRepoIds((items) => {
      const next = { ...items };
      delete next[repo.id];
      return next;
    });
    setPromptStateByRepo((items) => {
      const next = { ...items };
      delete next[repo.id];
      return next;
    });
    if (selectedRepoId === repo.id) {
      setSelectedRepoId(remainingRepos[0]?.id || "");
      setSelectedConversationId(remainingRepos[0] ? conversationMap[remainingRepos[0].id]?.[0]?.id || "" : "");
      setIssues([]);
      setPrs([]);
      setRuns([]);
      setMessages([]);
    }
  }

  /** 「换一批」推荐提问：把已出现过的项作为 exclude 传回去，避免重复 */
  async function refreshPromptSuggestions() {
    if (!selectedRepo || loadingPrompts) return;
    const exclude = promptExclude;
    await loadPromptSuggestions(selectedRepo.id, exclude);
  }

  // 停止生成：直接 abort 正在进行的 SSE 请求，后续由 isAbortError 分支收尾
  function stopAgentResponse() {
    streamAbortRef.current?.abort();
  }

  /** 用户中途停止后给这条 assistant 消息打上「已停止响应」标记，并收起轨迹、取消 pending 态 */
  function markAssistantStopped(assistantId: string) {
    setMessages((items) =>
      items.map((item) => {
        if (item.id !== assistantId) return item;
        const content = item.content.trim() ? `${item.content.trim()}\n\n已停止响应。` : "已停止响应。";
        return { ...item, content, responsePending: false, traceCollapsed: true };
      })
    );
  }

  /** 区分「用户主动中止」和真实错误：前者不该弹错误提示，只需静默收尾 */
  function isAbortError(error: unknown) {
    return error instanceof DOMException && error.name === "AbortError";
  }

  /** 追加一条轨迹并展开轨迹区；responsePending=true 让 UI 知道「还在跑」 */
  function appendAssistantTrace(assistantId: string, traceItem: ChatTraceItem) {
    setMessages((items) =>
      items.map((item) =>
        item.id === assistantId
          ? { ...item, traceItems: [...(item.traceItems ?? []), traceItem], traceCollapsed: false, responsePending: true }
          : item
      )
    );
  }

  /**
   * 用 tool_result 回填对应的 tool_use 轨迹条目。
   * 匹配依据是 toolUseId：同一工具可能在一次回答里出现多次，
   * 不能按名字匹配。找不到对应条目时退化为追加一条新轨迹。
   */
  function updateAssistantToolTrace(assistantId: string, result: ChatTraceItem) {
    setMessages((items) =>
      items.map((item) => {
        if (item.id !== assistantId) return item;
        const traceItems = item.traceItems ?? [];
        const existingIndex = result.toolUseId ? traceItems.findIndex((trace) => trace.toolUseId === result.toolUseId) : -1;
        if (existingIndex < 0) {
          return { ...item, traceItems: [...traceItems, result], traceCollapsed: false, responsePending: true };
        }
        return {
          ...item,
          traceCollapsed: false,
          responsePending: true,
          traceItems: traceItems.map((trace, index) =>
            index === existingIndex ? { ...trace, ...result, id: trace.id } : trace
          )
        };
      })
    );
  }

  /** 折叠/展开某条消息的「执行轨迹」 */
  function toggleAssistantTrace(messageId: string) {
    setMessages((items) =>
      items.map((item) =>
        item.id === messageId ? { ...item, traceCollapsed: !item.traceCollapsed } : item
      )
    );
  }

  /** 折叠/展开某条消息的「上下文诊断」（压缩比、token 用量等） */
  function toggleContextDiagnostics(messageId: string) {
    setMessages((items) =>
      items.map((item) =>
        item.id === messageId ? { ...item, contextCollapsed: !item.contextCollapsed } : item
      )
    );
  }

  /**
   * 取最近 8 条有效消息作为对话上下文。
   * 过滤 welcome- 开头的本地欢迎语与空消息：它们不该占用上下文 token。
   * 8 条是长度与成本的折中，后端还会做压缩，见 contextDiagnostics。
   */
  function recentChatHistory() {
    return messages
      .filter((item) => !item.id.startsWith("welcome-") && item.content.trim())
      .slice(-8)
      .map((item) => ({ role: item.role, content: item.content }));
  }

  /**
   * 组装 /api/chat 请求的 context 字段：
   * time_range 限定检索时间窗，messages 是近期对话，team_members 与 project_memory
   * 是首页维护的「团队与项目记忆」，让 Agent 在不查询工具的情况下也知道谁负责什么。
   */
  function chatRequestContext(chatHistory: Array<{ role: string; content: string }>) {
    return {
      time_range: "7d",
      messages: chatHistory,
      team_members: teamMembers.map(({ name, role, strengths, techStack }) => ({ name, role, strengths, techStack })),
      project_memory: memoryNotes.map(({ title, content }) => ({ title, content }))
    };
  }

  /**
   * 确保有可用的真实会话：
   * 草稿会话（或还没选会话）时，用首条消息的摘要作为标题创建会话，并把本地 draft 换成真实 id。
   * skipHistoryLoadRef 置 true，避免随后 selectedConversationId 变化触发历史加载，
   * 把刚发出的消息冲掉（详情见 loadChatHistory 的调用处）。
   * @returns 真实会话对象；没有选中仓库时返回 null
   */
  async function ensureActiveConversation(initialMessage: string) {
    if (!selectedRepo) return null;
    let activeConversation = selectedConversation;
    if (!activeConversation) {
      activeConversation = await apiPost<ProjectConversation>(`/api/repos/${selectedRepo.id}/conversations`, {
        title: summarizeConversationTitle(initialMessage)
      });
      skipHistoryLoadRef.current = true;
      setConversationMap((items) => {
        const existing = items[selectedRepo.id] ?? [];
        return {
          ...items,
          [selectedRepo.id]: [activeConversation!, ...existing.filter((item) => item.id !== activeConversation!.id)]
        };
      });
      setExpandedRepoIds((items) => ({ ...items, [selectedRepo.id]: true }));
      setSelectedConversationId(activeConversation.id);
    }
    return activeConversation;
  }

  /**
   * 主对话流程：POST /api/chat/stream（SSE）。
   *
   * 步骤：
   * 1) 先确保有真实会话（草稿会话在这里落库）；
   * 2) 乐观插入用户消息与一个空的 assistant 占位消息，清空输入框；
   * 3) 逐条消费事件：thought（公开思路）/ delta（正文增量）/ tool_use / tool_result / final；
   * 4) 结束或出错时收尾：重置 pending、关闭 loading、abort 的控制器置空。
   * @sideEffect 创建会话、持续写 messages、控制 loading；abortController 存入 streamAbortRef 供「停止」使用。
   */
  async function askAgent(text = message) {
    const trimmedText = text.trim();
    if (!selectedRepo || !trimmedText) return;
    const activeConversation = await ensureActiveConversation(trimmedText);
    if (!activeConversation) return;

    const userMessage: ChatMessage = { id: crypto.randomUUID(), role: "user", content: trimmedText };
    const assistantId = crypto.randomUUID();
    const chatHistory = recentChatHistory();
    setMessages((items) => [...items, userMessage]);
    setMessage("");
    setLoading(true);
    const abortController = new AbortController();
    streamAbortRef.current = abortController;
    try {
      setMessages((items) => [
        ...items,
        {
          id: assistantId,
          role: "assistant",
          content: "",
          traceCollapsed: false,
          responsePending: true
        }
      ]);
      await apiPostStream("/api/chat/stream", {
        repo_id: selectedRepo.id,
        conversation_id: activeConversation.id,
        message: trimmedText,
        context: chatRequestContext(chatHistory)
      }, (event, rawData) => {
        // thought：Agent 公开的推理摘要，作为独立轨迹展示（不混进正文）
        if (event === "thought") {
          const data = rawData as { content?: string };
          if (data.content?.trim()) {
            appendAssistantTrace(assistantId, {
              id: crypto.randomUUID(),
              kind: "thinking",
              title: "公开思路",
              content: data.content,
              status: "done"
            });
          }
          return;
        }
        // tool_use：记录一次工具调用（尚未有结果）；tool_result 会按 toolUseId 回填到同一条轨迹
        if (event === "tool_use") {
          appendAssistantTrace(assistantId, makeToolUseTrace(rawData as ChatStreamToolUse, crypto.randomUUID()));
          return;
        }
        if (event === "tool_result") {
          updateAssistantToolTrace(assistantId, makeToolResultTrace(rawData as ChatStreamToolResult, crypto.randomUUID()));
          return;
        }
        // delta：正文增量，直接拼接形成打字机效果（final 到达时才用权威答案覆盖）
        if (event === "delta") {
          const data = rawData as ChatStreamDelta;
          if (!data.content) return;
          setMessages((items) =>
            items.map((item) =>
              item.id === assistantId ? { ...item, content: item.content + data.content, responsePending: true } : item
              )
          );
          return;
        }
        // final：本轮结束，用后端权威结果覆盖正文，并补齐轨迹、压缩统计、模型用量、上下文诊断
        if (event === "final") {
          const data = rawData as ChatStreamFinal;
          setMessages((items) =>
            items.map((item) =>
              item.id === assistantId
                ? {
                    ...item,
                    content: data.answer,
                    route: data.route,
                    toolCalls: data.tool_calls,
                    traceItems: finalizedTraceItems(data.agent_events, item.traceItems),
                    compressionStats: data.compression_stats,
                    progressiveCompaction: data.progressive_compaction,
                    modelUsage: data.model_usage,
                    contextDiagnostics: data.context_diagnostics,
                    messageId: data.message_id,
                    contextCollapsed: true,
                    traceCollapsed: true,
                    responsePending: false
                  }
                : item
            )
          );
          if (data.route === "weekly_report") {
            void loadProjectKnowledge(selectedRepo.id);
          }
          void loadRepoConversations(selectedRepo.id);
        }
      }, { signal: abortController.signal });
    } catch (error) {
      if (isAbortError(error)) {
        markAssistantStopped(assistantId);
        return;
      }
      setMessages((items) =>
        items.map((item) =>
          item.id === assistantId ? { ...item, content: `请求失败：${getFriendlyError(error)}`, responsePending: false, traceCollapsed: true } : item
        )
      );
    } finally {
      setMessages((items) =>
        items.map((item) =>
          item.id === assistantId ? { ...item, traceCollapsed: true, responsePending: false } : item
        )
      );
      if (streamAbortRef.current === abortController) {
        streamAbortRef.current = null;
      }
      setLoading(false);
    }
  }

  /**
   * 计划模式：POST /api/chat/plan（非流式）让后端先产出可审阅的执行计划。
   * 计划以一条 assistant 消息呈现（带 WorkflowPlan），用户点确认后才调用 executeWorkflowPlan，
   * 这样涉及写操作的步骤不会在用户不知情时执行。
   */
  async function createWorkflowPlan(text = message) {
    const trimmedText = text.trim();
    if (!selectedRepo || !trimmedText) return;
    const activeConversation = await ensureActiveConversation(trimmedText);
    if (!activeConversation) return;

    const userMessage: ChatMessage = { id: crypto.randomUUID(), role: "user", content: trimmedText };
    const assistantId = crypto.randomUUID();
    const chatHistory = recentChatHistory();
    setMessages((items) => [
      ...items,
      userMessage,
      {
        id: assistantId,
        role: "assistant",
        content: "正在创建执行计划...",
        route: "engineering_plan",
        traceCollapsed: false,
        responsePending: true
      }
    ]);
    setMessage("");
    setLoading(true);
    try {
      const data = await apiPost<ChatPlanResponse>("/api/chat/plan", {
        repo_id: selectedRepo.id,
        conversation_id: activeConversation.id,
        message: trimmedText,
        context: chatRequestContext(chatHistory)
      });
      setMessages((items) =>
        items.map((item) =>
          item.id === assistantId
            ? {
                ...item,
                content: data.answer,
                route: data.route,
                toolCalls: data.tool_calls,
                traceItems: traceItemsFromAgentEvents(data.agent_events ?? []),
                workflowPlan: data.plan,
                planSourceMessage: trimmedText,
                planStatus: "ready",
                traceCollapsed: true,
                responsePending: false
              }
            : item
        )
      );
      void loadRepoConversations(selectedRepo.id);
    } catch (error) {
      setMessages((items) =>
        items.map((item) =>
          item.id === assistantId
            ? {
                ...item,
                content: `创建计划失败：${getFriendlyError(error)}`,
                planStatus: "error",
                responsePending: false,
                traceCollapsed: true
              }
            : item
        )
      );
    } finally {
      setLoading(false);
    }
  }

  /**
   * 执行已确认的计划：POST /api/chat/workflow-runs（流式，返回 agent 事件）。
   * 执行前把原计划消息清空正文并标记为运行中（planStatus），
   * 再把流式事件写进这条消息的轨迹；结束后刷新会话列表（标题/更新时间可能变化）。
   * 依赖 loading 做互斥，防止同一条计划被重复执行。
   */
  async function executeWorkflowPlan(messageId: string) {
    if (!selectedRepo || loading) return;
    const planMessage = messages.find((item) => item.id === messageId);
    const workflowPlan = planMessage?.workflowPlan;
    const sourceMessage = planMessage?.planSourceMessage;
    if (!workflowPlan || !sourceMessage) return;
    const activeConversation = await ensureActiveConversation(sourceMessage);
    if (!activeConversation) return;

    const chatHistory = recentChatHistory();
    setLoading(true);
    const abortController = new AbortController();
    streamAbortRef.current = abortController;
    setMessages((items) =>
      items.map((item) =>
        item.id === messageId
          ? {
              ...item,
              content: "",
              route: "engineering_workflow",
              traceItems: [],
              traceCollapsed: false,
              responsePending: true,
              planStatus: "executing"
            }
          : item
      )
    );
    try {
      await apiPostStream("/api/chat/plan/execute/stream", {
        repo_id: selectedRepo.id,
        conversation_id: activeConversation.id,
        message: sourceMessage,
        workflow_spec: workflowPlan,
        context: chatRequestContext(chatHistory)
      }, (event, rawData) => {
        if (event === "tool_use") {
          appendAssistantTrace(messageId, makeToolUseTrace(rawData as ChatStreamToolUse, crypto.randomUUID()));
          return;
        }
        if (event === "tool_result") {
          updateAssistantToolTrace(messageId, makeToolResultTrace(rawData as ChatStreamToolResult, crypto.randomUUID()));
          return;
        }
        if (event === "delta") {
          const data = rawData as ChatStreamDelta;
          if (!data.content) return;
          setMessages((items) =>
            items.map((item) =>
              item.id === messageId ? { ...item, content: item.content + data.content, responsePending: true } : item
            )
          );
          return;
        }
        if (event === "final") {
          const data = rawData as ChatStreamFinal;
          setMessages((items) =>
            items.map((item) =>
              item.id === messageId
                ? {
                    ...item,
                    content: data.answer,
                    route: data.route,
                    toolCalls: data.tool_calls,
                    traceItems: finalizedTraceItems(data.agent_events, item.traceItems),
                    messageId: data.message_id,
                    traceCollapsed: true,
                    responsePending: false,
                    planStatus: data.route === "error" ? "error" : "done"
                  }
                : item
            )
          );
          void loadRepoConversations(selectedRepo.id);
        }
      }, { signal: abortController.signal });
    } catch (error) {
      if (isAbortError(error)) {
        markAssistantStopped(messageId);
        return;
      }
      setMessages((items) =>
        items.map((item) =>
          item.id === messageId
            ? {
                ...item,
                content: `执行计划失败：${getFriendlyError(error)}`,
                planStatus: "error",
                responsePending: false,
                traceCollapsed: true
              }
            : item
        )
      );
    } finally {
      setMessages((items) =>
        items.map((item) =>
          item.id === messageId ? { ...item, traceCollapsed: true, responsePending: false } : item
        )
      );
      if (streamAbortRef.current === abortController) {
        streamAbortRef.current = null;
      }
      setLoading(false);
    }
  }

  // 负责人筛选项：从 Issue 的 assignees/author 与 PR 的 author 汇总去重，
  // 保证下拉里的选项都真实存在于当前数据中，不会出现选了却筛不出结果的人。
  const ownerOptions = useMemo(() => {
    const people = new Set<string>();
    issues.forEach((issue) => {
      issue.assignees?.forEach((assignee) => people.add(assignee));
      if (issue.author) people.add(issue.author);
    });
    prs.forEach((pr) => {
      if (pr.author) people.add(pr.author);
    });
    return Array.from(people).sort((a, b) => a.localeCompare(b));
  }, [issues, prs]);

  // 三个列表的筛选都是「关键词 + 状态 + 负责人 + 时间」四条件的与运算，纯前端过滤：
  // 数据量在百级、且已在内存中，重新请求后端反而更慢。
  const filteredIssues = useMemo(
    () =>
      issues.filter((issue) => {
        const people = [...(issue.assignees ?? []), issue.author ?? ""];
        return (
          matchesQuery(workspaceQuery, [issue.title, issue.body, issue.state, ...(issue.labels ?? [])]) &&
          matchesStatus(workspaceStatus, issue.state) &&
          matchesOwner(workspaceOwner, people) &&
          matchesTime(workspaceTime, issue.updated_at ?? issue.created_at)
        );
      }),
    [issues, workspaceOwner, workspaceQuery, workspaceStatus, workspaceTime]
  );

  // PR 的「已合并」不是 state 字段，而是由 merged_at 推导，所以状态筛选要先算 derivedStatus
  const filteredPrs = useMemo(
    () =>
      prs.filter((pr) => {
        const derivedStatus = pr.merged_at ? "merged" : pr.state;
        const fileNames = pr.files?.map((file) => file.filename) ?? [];
        return (
          matchesQuery(workspaceQuery, [pr.title, pr.body, derivedStatus, pr.author, pr.base_branch, pr.head_branch, ...fileNames]) &&
          matchesStatus(workspaceStatus, derivedStatus) &&
          matchesOwner(workspaceOwner, [pr.author ?? ""]) &&
          matchesTime(workspaceTime, pr.updated_at ?? pr.created_at ?? pr.merged_at)
        );
      }),
    [prs, workspaceOwner, workspaceQuery, workspaceStatus, workspaceTime]
  );

  // CI 没有负责人维度，因此只按关键词、状态（conclusion 优先）与时间过滤
  const filteredRuns = useMemo(
    () =>
      runs.filter((run) => {
        const derivedStatus = run.conclusion ?? run.status;
        return (
          matchesQuery(workspaceQuery, [run.name, run.status, run.conclusion, run.logs_text]) &&
          matchesStatus(workspaceStatus, derivedStatus) &&
          matchesTime(workspaceTime, run.updated_at ?? run.created_at)
        );
      }),
    [runs, workspaceQuery, workspaceStatus, workspaceTime]
  );

  const openIssues = issues.filter((item) => item.state === "open");
  const openPrs = prs.filter((item) => item.state === "open");
  const failedRuns = runs.filter((item) => item.conclusion === "failure");
  const mergedPrs = prs.filter((item) => item.merged_at);
  const rejectedIssues = issues.filter((item) => classifyIssue(item) === "rejected");
  const handledIssues = issues.filter((item) => ["accepted", "closed"].includes(classifyIssue(item)));
  const selectedIssue = selectedWorkspaceItem?.type === "issue" ? issues.find((item) => item.id === selectedWorkspaceItem.id) : undefined;
  const selectedPr = selectedWorkspaceItem?.type === "pr" ? prs.find((item) => item.id === selectedWorkspaceItem.id) : undefined;
  const selectedRun = selectedWorkspaceItem?.type === "ci" ? runs.find((item) => item.id === selectedWorkspaceItem.id) : undefined;
  const hasWorkspaceDetail = Boolean(selectedIssue || selectedPr || selectedRun);
  const analysisModalIssue = analysisModalIssueId ? issues.find((item) => item.id === analysisModalIssueId) : undefined;
  const analysisModalPr = analysisModalPrId ? prs.find((item) => item.id === analysisModalPrId) : undefined;
  const analysisModalRun = analysisModalRunId ? runs.find((item) => item.id === analysisModalRunId) : undefined;
  // isEmptyConversation：只有欢迎语也算空会话，此时展示推荐提问而不是消息列表
  const isEmptyConversation = messages.length === 0 || (messages.length === 1 && messages[0]?.id.startsWith("welcome-"));
  // indexSnoozed：用户点过「稍后再说」且冷却期未过，此时不再弹索引提示
  const indexSnoozed = Boolean(
    selectedProjectIndexState?.snoozed_until && new Date(selectedProjectIndexState.snoozed_until).getTime() > Date.now()
  );

  // 布局：两列 Grid（左栏 + 主区），主区内部再按 workspaceWidth 分成「对话」与「工作台」两栏；
  // 左栏折叠时宽度从 300px 变为 56px，用 grid-template-columns 过渡实现动画。
  return (
    <div
      className="relative grid h-[calc(100vh-24px)] min-h-0 overflow-hidden rounded-lg border border-line bg-white/80 shadow-sm"
      style={{ gridTemplateColumns: `${isSidebarCollapsed ? "56px" : "300px"} minmax(0,1fr)`, transition: "grid-template-columns 260ms ease" }}
    >
      <aside className="flex min-h-0 min-w-0 flex-col overflow-hidden border-r border-line bg-[#fbf7f2] transition-all duration-300">
        <div className={`border-b border-line ${isSidebarCollapsed ? "p-2" : "p-4"}`}>
          <div className={`flex items-center ${isSidebarCollapsed ? "flex-col gap-3" : "gap-3"}`}>
            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-accent text-sm font-bold text-white">DF</div>
            {!isSidebarCollapsed && (
              <div className="min-w-0 flex-1">
                <div className="text-sm font-semibold text-ink">DevFlow AI</div>
                <div className="truncate text-xs text-slate-500">研发团队 PR / Issue 智能协作 Agent</div>
              </div>
            )}
            <button
              onClick={() => setIsSidebarCollapsed((value) => !value)}
              className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-md border border-line bg-white text-slate-600 transition hover:bg-teal-50"
              title={isSidebarCollapsed ? "展开项目侧边栏" : "隐藏项目侧边栏"}
            >
              {isSidebarCollapsed ? <PanelLeftOpen className="h-4 w-4" /> : <PanelLeftClose className="h-4 w-4" />}
            </button>
          </div>

          {isSidebarCollapsed ? (
            <div className="mt-4 flex flex-col items-center gap-3">
              <button onClick={openAddProject} className="inline-flex h-8 w-8 items-center justify-center rounded-md text-slate-700 transition hover:bg-white" title="添加项目">
                <Plus className="h-4 w-4" />
              </button>
              <button
                onClick={refreshSelectedRepo}
                disabled={!selectedRepo || syncing}
                className="inline-flex h-8 w-8 items-center justify-center rounded-md text-slate-700 transition hover:bg-white disabled:opacity-50"
                title="刷新当前项目"
              >
                <RefreshCw className={`h-4 w-4 ${syncing ? "animate-spin" : ""}`} />
              </button>
            </div>
          ) : (
            <div className="mt-5 flex items-center justify-between">
              <div>
                <div className="text-xs font-medium uppercase tracking-[0.18em] text-accent">Projects</div>
                <h2 className="mt-1 text-lg font-semibold text-ink">项目</h2>
              </div>
              <button onClick={openAddProject} className="inline-flex items-center gap-1 rounded-md bg-accent px-3 py-1.5 text-xs font-medium text-white">
                <Plus className="h-3.5 w-3.5" />
                添加项目
              </button>
            </div>
          )}
        </div>

        <div className={`min-h-0 flex-1 overflow-y-auto ${isSidebarCollapsed ? "p-2" : "px-3 py-2"}`}>
          {repos.length === 0 && !isSidebarCollapsed && <div className="rounded-md bg-white p-3 text-sm leading-6 text-slate-600">还没有项目。请先添加一个项目。</div>}
          {!isSidebarCollapsed && repos.length > 0 && <div className="mb-2 px-1 text-xs text-slate-400">项目</div>}
          {repos.map((repo) => {
            const conversations = conversationMap[repo.id] ?? [];
            const isExpanded = expandedRepoIds[repo.id] ?? selectedRepo?.id === repo.id;
            const isSelectedRepo = selectedRepo?.id === repo.id;
            return isSidebarCollapsed ? (
              <button
                key={repo.id}
                onClick={() => selectProject(repo).catch(() => undefined)}
                className={`mb-2 inline-flex h-9 w-9 items-center justify-center rounded-md transition ${
                  isSelectedRepo ? "bg-white text-ink shadow-sm" : "text-slate-600 hover:bg-white"
                }`}
                title={repo.full_name}
              >
                <Folder className="h-4 w-4" />
              </button>
            ) : (
              <div key={repo.id} className="mb-1">
                <div className={`group flex h-9 items-center gap-1 rounded-md px-1 transition ${isSelectedRepo ? "bg-[#ecebea]" : "hover:bg-[#eeeeec]"}`}>
                  <button onClick={() => selectProject(repo).catch(() => undefined)} className="inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-md text-slate-500 transition hover:bg-white">
                    <ChevronDown className={`h-3.5 w-3.5 transition ${isExpanded ? "" : "-rotate-90"}`} />
                  </button>
                  <button onClick={() => selectProject(repo).catch(() => undefined)} className="flex min-w-0 flex-1 items-center gap-2 text-left">
                    <Folder className="h-4 w-4 shrink-0 text-slate-500" />
                    <span className="truncate text-sm text-slate-700" title={repo.full_name}>
                      {repo.name || repo.full_name}
                    </span>
                  </button>
                  <button
                    onClick={(event) => {
                      event.stopPropagation();
                      addConversation(repo).catch(() => undefined);
                    }}
                    className="inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-md text-slate-400 opacity-0 transition hover:bg-white hover:text-ink group-hover:opacity-100"
                    title="添加会话"
                  >
                    <Plus className="h-3.5 w-3.5" />
                  </button>
                  <button
                    onClick={(event) => {
                      event.stopPropagation();
                      removeRepo(repo).catch(() => undefined);
                    }}
                    className="inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-md text-slate-400 opacity-0 transition hover:bg-rose-50 hover:text-rose-600 group-hover:opacity-100"
                    title="移除项目"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </div>
                {isExpanded && (
                  <div className="ml-7 mt-1 space-y-0.5">
                    {conversations.length === 0 && <div className="px-2 py-1.5 text-xs text-slate-400">暂无会话</div>}
                    {conversations.map((conversation) => (
                      <div key={conversation.id} className="group/conversation flex items-center gap-1">
                        <button
                          onClick={() => selectConversation(repo, conversation).catch(() => undefined)}
                          className={`flex h-8 min-w-0 flex-1 items-center justify-between gap-2 rounded-md px-2 text-left text-sm transition ${
                            selectedConversationId === conversation.id ? "bg-white text-ink shadow-sm" : "text-slate-600 hover:bg-white"
                          }`}
                          title={conversation.title}
                        >
                          <span className="truncate">{conversation.title}</span>
                          <span className="shrink-0 text-xs text-slate-400">{formatRelativeTime(conversation.updated_at)}</span>
                        </button>
                        <button
                          onClick={() => removeConversation(repo, conversation).catch(() => undefined)}
                          className="inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-md text-slate-400 opacity-0 transition hover:bg-rose-50 hover:text-rose-600 group-hover/conversation:opacity-100"
                          title="删除会话"
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {!isSidebarCollapsed && (
          <div className="border-t border-line p-3">
            <button
              onClick={refreshSelectedRepo}
              disabled={!selectedRepo || syncing}
              className="inline-flex w-full items-center justify-center gap-2 rounded-md border border-line bg-white px-3 py-2 text-sm font-medium text-slate-700 transition hover:bg-teal-50 disabled:opacity-60"
            >
              <RefreshCw className={`h-4 w-4 ${syncing ? "animate-spin" : ""}`} />
              {syncing ? "刷新中..." : "刷新当前项目"}
            </button>
          </div>
        )}
      </aside>

      <section
        className="grid min-h-0 min-w-0 bg-[#fffaf6]"
        style={{ gridTemplateColumns: `minmax(460px,1fr) 8px ${workspaceWidth}px`, gridTemplateRows: "auto minmax(0,1fr)" }}
      >
        <section className="col-start-1 row-start-1 shrink-0 border-b border-line bg-white/90 px-4 py-2">
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0">
              <div className="flex min-w-0 flex-wrap items-baseline gap-x-3 gap-y-1">
                <h2 className="font-semibold text-ink">总览</h2>
                <span className="text-xs text-slate-500">当前代码仓</span>
                <span className="truncate text-sm font-semibold text-ink">{selectedRepo?.full_name || "未选择仓库"}</span>
                {selectedConversation && <span className="rounded-full bg-panel px-2 py-0.5 text-xs text-slate-600">会话：{selectedConversation.title}</span>}
              </div>
            </div>
            <div className="flex shrink-0 items-center gap-3">
              <div className="rounded-full border border-teal-200 bg-teal-50 px-3 py-1 text-xs font-medium text-teal-800">Production Workspace</div>
              <button onClick={openAddProject} className="text-xs text-accent hover:underline">
                管理项目
              </button>
            </div>
          </div>
          <div className="mt-2 grid grid-cols-2 gap-2 md:grid-cols-3 xl:grid-cols-6">
            <MetricCard icon={ListTodo} label="待处理 Issue" value={openIssues.length} />
            <MetricCard icon={GitPullRequest} label="待 Review PR" value={openPrs.length} />
            <MetricCard icon={CheckCircle2} label="已处理 Issue" value={handledIssues.length} />
            <MetricCard icon={XCircle} label="已拒绝 Issue" value={rejectedIssues.length} />
            <MetricCard icon={AlertTriangle} label="失败 CI" value={failedRuns.length} />
            <MetricCard icon={CheckCircle2} label="已合并 PR" value={mergedPrs.length} />
          </div>
        </section>

        <div className="col-start-2 row-span-2 row-start-1 cursor-col-resize border-x border-line bg-slate-100 transition hover:bg-teal-100" onMouseDown={startResize} title="拖拽调整 Workspace 宽度" />

        <aside className="col-start-3 row-span-2 row-start-1 flex min-h-0 min-w-0 flex-col bg-[#fffaf6]">
          <div className="border-b border-line bg-white/80 px-4 py-3">
            <div className="flex items-center justify-between gap-3">
              <h2 className="font-semibold text-ink">Workspace</h2>
              <div className="flex shrink-0 items-center gap-2">
                <div className="text-[11px] text-slate-500">
                  Issues: {filteredIssues.length} · PRs: {filteredPrs.length}
                </div>
                <button
                  onClick={refreshSelectedRepo}
                  disabled={!selectedRepo || syncing}
                  className="inline-flex h-8 items-center gap-1.5 rounded-md border border-line bg-white px-2 text-xs font-medium text-slate-700 transition hover:bg-teal-50 disabled:opacity-60"
                  title="刷新 Issue/PR/CI"
                  aria-label="刷新 Issue/PR/CI"
                >
                  <RefreshCw className={`h-3.5 w-3.5 ${syncing ? "animate-spin" : ""}`} />
                  {syncing ? "刷新中" : "刷新"}
                </button>
              </div>
            </div>
            <div className="mt-3 flex gap-1 overflow-x-auto">
              {workspaceTabs.map((tab) => (
                <button
                  key={tab}
                  onClick={() => {
                    if (tab === "图谱" && selectedRepo && selectedWorkspaceItem) {
                      const center = workspaceSelectionToGraphCenter(selectedWorkspaceItem);
                      if (center) void loadKnowledgeGraph(selectedRepo.id, center);
                    }
                    setWorkspaceTab(tab);
                  }}
                  className={`shrink-0 rounded-md px-3 py-1.5 text-xs font-medium transition ${
                    workspaceTab === tab ? "bg-accent text-white" : "bg-panel text-slate-600 hover:bg-teal-50"
                  }`}
                >
                  {tab}
                </button>
              ))}
            </div>
          </div>

          <div className="min-h-0 flex-1 overflow-y-auto p-4">
          {!hasWorkspaceDetail && (workspaceTab === "Issue" || workspaceTab === "PR" || workspaceTab === "CI") && (
            <WorkspaceControls
              query={workspaceQuery}
              status={workspaceStatus}
              owner={workspaceOwner}
              time={workspaceTime}
              owners={ownerOptions}
              activeTab={workspaceTab}
              onQueryChange={(value) => {
                setWorkspaceQuery(value);
                setSelectedWorkspaceItem(null);
              }}
              onStatusChange={(value) => {
                setWorkspaceStatus(value);
                setSelectedWorkspaceItem(null);
              }}
              onOwnerChange={(value) => {
                setWorkspaceOwner(value);
                setSelectedWorkspaceItem(null);
              }}
              onTimeChange={(value) => {
                setWorkspaceTime(value);
                setSelectedWorkspaceItem(null);
              }}
            />
          )}

          {hasWorkspaceDetail && (
            <WorkspaceDetail
              issue={selectedIssue}
              pr={selectedPr}
              run={selectedRun}
              analyzingIssue={Boolean(selectedIssue && analyzingIssueId === selectedIssue.id)}
              analyzingPr={Boolean(selectedPr && analyzingPrId === selectedPr.id)}
              analyzingRun={Boolean(selectedRun && analyzingRunId === selectedRun.id)}
              onAnalyzeIssue={selectedIssue ? () => analyzeIssue(selectedIssue) : undefined}
              onAnalyzePr={selectedPr ? () => analyzePr(selectedPr) : undefined}
              onAnalyzeRun={selectedRun ? () => analyzeCi(selectedRun) : undefined}
              onClose={() => setSelectedWorkspaceItem(null)}
            />
          )}

          {!hasWorkspaceDetail && workspaceTab === "Issue" && (
            <div className="mt-4 space-y-3">
              {issueCategoryMeta.map((category) => {
                const items = filteredIssues.filter((item) => classifyIssue(item) === category.key);
                const groupKey = `issue-${category.key}`;
                return (
                  <CategoryGroup
                    key={category.key}
                    label={category.label}
                    color={category.color}
                    count={items.length}
                    isCollapsed={collapsedGroups[groupKey] ?? false}
                    onToggle={() => toggleCategory(groupKey)}
                    items={items.map((item) => ({
                      id: item.id,
                      title: `#${item.number} ${item.title}`,
                      meta: item.state,
                      detail: item.body,
                      badges: item.labels,
                      selected: selectedWorkspaceItem?.type === "issue" && selectedWorkspaceItem.id === item.id,
                      onClick: () => setSelectedWorkspaceItem({ type: "issue", id: item.id })
                    }))}
                  />
                );
              })}
            </div>
          )}

          {!hasWorkspaceDetail && workspaceTab === "PR" && (
            <div className="mt-4 space-y-3">
              {prCategoryMeta.map((category) => {
                const items = filteredPrs.filter((item) => classifyPr(item) === category.key);
                const groupKey = `pr-${category.key}`;
                return (
                  <CategoryGroup
                    key={category.key}
                    label={category.label}
                    color={category.color}
                    count={items.length}
                    isCollapsed={collapsedGroups[groupKey] ?? false}
                    onToggle={() => toggleCategory(groupKey)}
                    items={items.map((item) => ({
                      id: item.id,
                      title: `#${item.number} ${item.title}`,
                      meta: item.merged_at ? "merged" : item.state,
                      detail: item.body,
                      badges: [item.author, item.head_branch].filter(Boolean) as string[],
                      selected: selectedWorkspaceItem?.type === "pr" && selectedWorkspaceItem.id === item.id,
                      onClick: () => setSelectedWorkspaceItem({ type: "pr", id: item.id })
                    }))}
                  />
                );
              })}
            </div>
          )}

          {!hasWorkspaceDetail && workspaceTab === "CI" && (
            <div className="mt-4 space-y-2">
              {filteredRuns.length === 0 && <EmptyWorkspaceState label="暂无 Workflow Run" />}
              {filteredRuns.map((item) => (
                <WorkspaceItemButton
                  key={item.id}
                  item={{
                    id: item.id,
                    title: item.name,
                    meta: item.conclusion || item.status,
                    detail: item.logs_text,
                    selected: selectedWorkspaceItem?.type === "ci" && selectedWorkspaceItem.id === item.id,
                    onClick: () => setSelectedWorkspaceItem({ type: "ci", id: item.id })
                  }}
                />
              ))}
            </div>
          )}

          {!hasWorkspaceDetail && workspaceTab === "团队" && (
            <TeamWorkspace
              members={teamMembers}
              draft={memberDraft}
              onDraftChange={setMemberDraft}
              onAddMember={addTeamMember}
              onRemoveMember={removeTeamMember}
            />
          )}

          {!hasWorkspaceDetail && workspaceTab === "记忆&知识库" && (
            <MemoryHubWorkspace
              repoId={selectedRepo?.id}
              notes={memoryNotes}
              draft={memoryDraft}
              knowledgeItems={knowledgeItems}
              uploading={uploadingKnowledge}
              onDraftChange={setMemoryDraft}
              onAddNote={addMemoryNote}
              onRemoveNote={removeMemoryNote}
              onUploadFile={uploadKnowledgeFile}
              onRemoveKnowledgeItem={removeKnowledgeItem}
              onRefreshKnowledge={() => selectedRepo && loadProjectKnowledge(selectedRepo.id)}
            />
          )}

          {!hasWorkspaceDetail && workspaceTab === "技能&MCP" && (
            <SkillMcpWorkspace
              runtime={mcpRuntime}
              skills={skills}
              selectedSkillName={selectedSkillName}
              loading={skillRuntimeLoading}
              error={skillRuntimeError}
              onSelectSkill={setSelectedSkillName}
              onRefresh={loadSkillRuntime}
            />
          )}

          {!hasWorkspaceDetail && workspaceTab === "图谱" && (
            <KnowledgeGraphWorkspace
              graph={selectedKnowledgeGraph ?? null}
              loading={graphLoadingRepoId === selectedRepo?.id}
              error={graphError}
              query={graphQuery}
              selectedNodeKey={selectedGraphNodeKey}
              hiddenNodeTypes={hiddenGraphNodeTypes}
              hiddenRelations={hiddenGraphRelations}
              onQueryChange={setGraphQuery}
              onSelectNode={setSelectedGraphNodeKey}
              onToggleNodeType={(type) => setHiddenGraphNodeTypes((items) => ({ ...items, [type]: !items[type] }))}
              onToggleRelation={(relation) => setHiddenGraphRelations((items) => ({ ...items, [relation]: !items[relation] }))}
              onRefresh={() => selectedRepo && loadKnowledgeGraph(selectedRepo.id)}
              onRebuild={rebuildSelectedKnowledgeGraph}
              onFocusNode={(node) => selectedRepo && loadKnowledgeGraph(selectedRepo.id, { type: node.type, id: node.id })}
              onOpenNode={(node) => openGraphNodeInWorkspace(node, setWorkspaceTab, setSelectedWorkspaceItem, preserveWorkspaceSelectionRef)}
            />
          )}
          </div>
        </aside>

          <section className="flex min-h-0 min-w-0 flex-col bg-[#fffaf6]">
            <div ref={chatScrollRef} className="min-h-0 flex-1 overflow-y-auto px-6 py-5">
              <div className="flex w-full flex-col gap-4">
                {messages.map((item) => {
                  const isWelcome = item.id.startsWith("welcome-");
                  const traceItems = item.traceItems ?? [];
                  const failedTools = (item.toolCalls ?? []).filter((call) => call.status === "error");
                  const failedToolTitle = failedTools.map((call) => `${call.tool_name}: ${call.error ?? "failed"}`).join("\n");
                  return (
                    <div key={item.id} className={`flex ${item.role === "user" ? "justify-end" : "justify-start"}`}>
                      <div
                        className={`${isWelcome ? "w-full max-w-none rounded-md" : "max-w-[80%] rounded-2xl"} px-4 py-3 text-sm leading-7 shadow-sm ${
                          item.role === "user" ? "bg-[#ffd9ca] text-slate-800" : "border border-line bg-white text-slate-700"
                        }`}
                      >
                        {item.role === "assistant" ? (
                          <>
                            {item.workflowPlan && (
                              <WorkflowPlanCard
                                plan={item.workflowPlan}
                                status={item.planStatus ?? "ready"}
                                disabled={loading || Boolean(item.responsePending) || item.planStatus === "done"}
                                onExecute={() => executeWorkflowPlan(item.id)}
                              />
                            )}
                            {item.content.trim() ? (
                              <div className={item.workflowPlan ? "mt-3" : ""}>
                                <MarkdownMessage content={item.content} />
                              </div>
                            ) : traceItems.length === 0 ? (
                              <div className="text-slate-500">Agent 正在分析...</div>
                            ) : null}
                          </>
                        ) : (
                          <div className="whitespace-pre-wrap">{item.content}</div>
                        )}
                        {item.route && (
                          <div className="mt-3 flex flex-wrap gap-2 border-t border-line pt-2 text-xs text-slate-500">
                            <span>route: {item.route}</span>
                            <span>tools: {item.toolCalls?.length ?? 0}</span>
                            {failedTools.length > 0 && (
                              <span className="text-rose-600" title={failedToolTitle}>
                                failed: {failedTools.length}
                              </span>
                            )}
                          </div>
                        )}
                        {item.role === "assistant" && item.compressionStats && Object.keys(item.compressionStats).length > 0 && (
                          <ContextDiagnostics
                            compressionStats={item.compressionStats}
                            progressiveCompaction={item.progressiveCompaction ?? {}}
                            modelUsage={item.modelUsage ?? []}
                            contextDiagnostics={item.contextDiagnostics ?? {}}
                            collapsed={item.contextCollapsed !== false}
                            onToggle={() => toggleContextDiagnostics(item.id)}
                          />
                        )}
                        {item.role === "assistant" && traceItems.length > 0 && (
                          <div className="mt-3">
                            <ChatTraceInline
                              traceItems={traceItems}
                              collapsed={item.responsePending ? false : Boolean(item.traceCollapsed)}
                              responsePending={Boolean(item.responsePending)}
                              onToggle={() => toggleAssistantTrace(item.id)}
                            />
                          </div>
                        )}
                        {item.role === "assistant" && item.responsePending && (traceItems.length > 0 || item.content.trim()) && (
                          <div className="mt-3 flex items-center gap-2 rounded-md border border-teal-100 bg-teal-50 px-3 py-2 text-xs text-slate-600">
                            <span className="h-2 w-2 shrink-0 animate-pulse rounded-full bg-accent" />
                            <span>Agent 正在回复，工具结果已收集，正在整理最终回答...</span>
                          </div>
                        )}
                        {item.role === "assistant" && item.messageId && !item.responsePending && selectedRepo && selectedConversation && (
                          <FeedbackControls
                            repoId={selectedRepo.id}
                            conversationId={selectedConversation.id}
                            messageId={item.messageId}
                            initialFeedback={item.feedback}
                            onSaved={(feedback) =>
                              setMessages((current) =>
                                current.map((message) => (message.id === item.id ? { ...message, feedback } : message))
                              )
                            }
                          />
                        )}
                      </div>
                    </div>
                  );
                })}
                {selectedRepo && isEmptyConversation && selectedProjectIndexState && !indexSnoozed && selectedProjectIndexState.status !== "ready" && (
                  <MemoryIndexPromptCard
                    repo={selectedRepo}
                    state={selectedProjectIndexState}
                    scanning={indexScanningRepoId === selectedRepo.id || selectedProjectIndexState.status === "building"}
                    error={indexActionError}
                    onStartScan={startProjectIndexScan}
                    onSnooze={snoozeProjectIndexPrompt}
                  />
                )}
              </div>
            </div>

            <div className="border-t border-line bg-white/90 p-4">
              <div className="w-full">
                {(promptSuggestions.length > 0 || loadingPrompts) && (
                  <div className="mb-3 flex items-center gap-2">
                    <div ref={promptRowRef} className="relative min-w-0 flex-1 overflow-hidden">
                      <div className="flex min-w-0 items-center gap-2 overflow-hidden whitespace-nowrap">
                      {loadingPrompts && promptSuggestions.length === 0 && (
                        <span className="shrink-0 rounded-full border border-line bg-panel px-3 py-1.5 text-xs text-slate-500">
                          推荐生成中{promptLoadingDots}
                        </span>
                      )}
                      {promptSuggestions.slice(0, visiblePromptCount).map((prompt) => (
                        <button
                          key={prompt}
                          onClick={() => askAgent(prompt)}
                          disabled={!selectedRepo || loading || loadingPrompts}
                          className="max-w-full shrink-0 truncate rounded-full border border-line bg-panel px-3 py-1.5 text-xs text-slate-600 transition hover:border-teal-200 hover:bg-teal-50 disabled:opacity-60"
                          title={prompt}
                        >
                          {prompt}
                        </button>
                      ))}
                      </div>
                      <div
                        ref={promptMeasureRef}
                        className="pointer-events-none invisible absolute -left-[9999px] top-0 flex items-center gap-2 whitespace-nowrap"
                        aria-hidden="true"
                      >
                        {promptSuggestions.map((prompt) => (
                          <button
                            key={prompt}
                            type="button"
                            data-prompt-chip-measure
                            className="shrink-0 rounded-full border border-line bg-panel px-3 py-1.5 text-xs text-slate-600"
                            tabIndex={-1}
                          >
                            {prompt}
                          </button>
                        ))}
                      </div>
                    </div>
                    {promptSuggestions.length > 0 && (
                      <button
                        onClick={refreshPromptSuggestions}
                        disabled={!selectedRepo || loading || loadingPrompts}
                        className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-md border border-line bg-white text-slate-500 transition hover:border-teal-200 hover:bg-teal-50 disabled:cursor-not-allowed disabled:opacity-50"
                        title="刷新推荐提示语"
                        aria-label="刷新推荐提示语"
                      >
                        <RefreshCw className={`h-4 w-4 ${loadingPrompts ? "animate-spin" : ""}`} />
                      </button>
                    )}
                  </div>
                )}
                <div className="flex items-stretch gap-3">
                  <textarea
                    className={`max-h-40 min-h-[5.5rem] flex-1 resize-none rounded-md border bg-white px-3 py-3 text-sm leading-6 outline-none transition focus:border-teal-500 focus:ring-2 focus:ring-teal-100 ${
                      planMode ? "border-teal-200" : "border-line"
                    }`}
                    value={message}
                    onChange={(event) => setMessage(event.target.value)}
                    placeholder="输入消息，询问当前代码仓..."
                    onKeyDown={(event) => {
                      if (event.key === "Enter" && !event.shiftKey) {
                        event.preventDefault();
                        if (!loading) {
                          planMode ? createWorkflowPlan() : askAgent();
                        }
                      }
                    }}
                  />
                  <div className="flex w-12 shrink-0 flex-col gap-2">
                    <button
                      type="button"
                      onClick={() => setPlanMode((value) => !value)}
                      disabled={loading}
                      aria-pressed={planMode}
                      className={`inline-flex h-8 w-12 items-center justify-center rounded-md border transition disabled:opacity-60 ${
                        planMode
                          ? "border-teal-600 bg-teal-600 text-white shadow-sm"
                          : "border-line bg-white text-slate-500 hover:border-teal-200 hover:bg-teal-50"
                      }`}
                      title={planMode ? "切换到对话模式" : "切换到计划模式"}
                    >
                      <ListTodo className="h-4 w-4" />
                    </button>
                    <button
                      onClick={() => (loading ? stopAgentResponse() : planMode ? createWorkflowPlan() : askAgent())}
                      disabled={!selectedRepo || (!loading && !message.trim())}
                      className="inline-flex min-h-12 flex-1 items-center justify-center rounded-md bg-accent text-white transition hover:bg-teal-700 disabled:opacity-60"
                      title={loading ? "停止响应" : "发送"}
                    >
                      {loading ? <Square className="h-4 w-4 fill-current" /> : <Send className="h-5 w-5" />}
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </section>

      </section>
      {isAddProjectOpen && (
        <AddProjectOverlay
          source={projectSource}
          repoUrl={projectUrl}
          localPath={projectLocalPath}
          cloneParentDir={projectCloneParentDir}
          provider={projectProvider}
          apiBaseUrl={projectApiBaseUrl}
          token={projectToken}
          loading={connectingProject}
          status={addProjectStatus}
          onSourceChange={setProjectSource}
          onRepoUrlChange={setProjectUrl}
          onLocalPathChange={setProjectLocalPath}
          onCloneParentDirChange={setProjectCloneParentDir}
          onProviderChange={(value) => {
            setProjectProvider(value);
            if (value === "github") setProjectApiBaseUrl("https://api.github.com");
            if (value === "github_compatible") setProjectApiBaseUrl("https://api.example.com");
          }}
          onApiBaseUrlChange={setProjectApiBaseUrl}
          onTokenChange={setProjectToken}
          onConnect={connectProject}
          onClose={closeAddProject}
        />
      )}
      {analysisModalIssue && (
        <IssueAnalysisOverlay
          issue={analysisModalIssue}
          analysis={issueAnalysisById[analysisModalIssue.id]?.result}
          traceItems={issueAnalysisTraceById[analysisModalIssue.id] ?? []}
          loading={analyzingIssueId === analysisModalIssue.id}
          error={issueAnalysisError}
          onRetry={() => analyzeIssue(analysisModalIssue)}
          onClose={() => {
            setAnalysisModalIssueId(null);
            setIssueAnalysisError(null);
          }}
        />
      )}
      {analysisModalPr && (
        <PRAnalysisOverlay
          pr={analysisModalPr}
          analysis={prAnalysisById[analysisModalPr.id]?.result}
          traceItems={prAnalysisTraceById[analysisModalPr.id] ?? []}
          loading={analyzingPrId === analysisModalPr.id}
          error={prAnalysisError}
          onRetry={() => analyzePr(analysisModalPr)}
          onClose={() => {
            setAnalysisModalPrId(null);
            setPrAnalysisError(null);
          }}
        />
      )}
      {analysisModalRun && (
        <CIAnalysisOverlay
          run={analysisModalRun}
          analysis={ciAnalysisById[analysisModalRun.id]?.result}
          traceItems={ciAnalysisTraceById[analysisModalRun.id] ?? []}
          loading={analyzingRunId === analysisModalRun.id}
          error={ciAnalysisError}
          onRetry={() => analyzeCi(analysisModalRun)}
          onClose={() => {
            setAnalysisModalRunId(null);
            setCiAnalysisError(null);
          }}
        />
      )}
    </div>
  );
}

/**
 * 计划卡片：展示后端 Planner 生成的工作流（任务及其依赖），并提供「执行」入口。
 * 纯展示 + 回调：不自己发请求，执行动作交给父组件的 onExecute。
 */
function WorkflowPlanCard({
  plan,
  status,
  disabled,
  onExecute
}: {
  plan: WorkflowPlan;
  status: "ready" | "executing" | "done" | "error";
  disabled: boolean;
  onExecute: () => void;
}) {
  const statusLabel = status === "executing" ? "执行中" : status === "done" ? "已执行" : status === "error" ? "执行失败" : "等待确认";
  return (
    <div className="rounded-md border border-teal-100 bg-teal-50/60 p-3 text-xs leading-5 text-slate-700">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2 font-semibold text-ink">
            <ListTodo className="h-4 w-4 text-teal-700" />
            <span>计划模式</span>
            <span className="rounded-md bg-white px-2 py-0.5 text-[11px] font-medium text-slate-500">{statusLabel}</span>
          </div>
          <div className="mt-1 line-clamp-2 text-slate-600">{plan.goal}</div>
        </div>
        <button
          type="button"
          onClick={onExecute}
          disabled={disabled || status === "executing" || status === "done"}
          className="inline-flex h-8 shrink-0 items-center gap-1.5 rounded-md bg-accent px-3 text-xs font-medium text-white transition hover:bg-teal-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {status === "executing" ? <RefreshCw className="h-3.5 w-3.5 animate-spin" /> : <CheckCircle2 className="h-3.5 w-3.5" />}
          <span>{status === "done" ? "已执行" : status === "executing" ? "执行中" : "执行计划"}</span>
        </button>
      </div>
      <div className="mt-3 space-y-2">
        {plan.tasks.map((task, index) => (
          <div key={task.id} className="rounded-md border border-teal-100 bg-white px-3 py-2">
            <div className="flex items-center justify-between gap-2">
              <span className="font-semibold text-ink">
                {index + 1}. {task.agent_name}
              </span>
              <span className="shrink-0 rounded-md bg-slate-100 px-2 py-0.5 text-[11px] text-slate-500">{task.task_type}</span>
            </div>
            <div className="mt-1 text-slate-600">{task.objective}</div>
            <div className="mt-1 flex flex-wrap gap-2 text-[11px] text-slate-500">
              {task.dependencies?.length ? <span>依赖：{task.dependencies.join(", ")}</span> : <span>无依赖</span>}
              <span>范围：{task.claim.target_type}</span>
              {task.critical && <span className="text-rose-600">关键任务</span>}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/**
 * 上下文诊断面板：展示证据压缩统计、渐进式压缩过程、模型 token 用量与诊断明细。
 * 用于回答「为什么这次回答质量下降/成本变高」，默认折叠，展开才看细节。
 */
function ContextDiagnostics({
  compressionStats,
  progressiveCompaction,
  modelUsage,
  contextDiagnostics,
  collapsed,
  onToggle
}: {
  compressionStats: Record<string, unknown>;
  progressiveCompaction: Record<string, unknown>;
  modelUsage: Array<Record<string, unknown>>;
  contextDiagnostics: Record<string, unknown>;
  collapsed: boolean;
  onToggle: () => void;
}) {
  const pressure = contextRecord(compressionStats.pressure);
  const rawPressure = contextRecord(compressionStats.raw_history_pressure);
  const microcompact = contextRecord(compressionStats.microcompact);
  const boundary = contextRecord(compressionStats.compact_boundary);
  const promptBreakdown = contextRecord(compressionStats.prompt_breakdown);
  const historyCoverage = contextRecord(compressionStats.history_coverage);
  const progressivePressure = contextRecord(progressiveCompaction.pressure);
  const runtimeToolContext = contextRecord(contextDiagnostics.runtime_tool_context);
  const promptCalls = Array.isArray(contextDiagnostics.prompt_calls) ? contextDiagnostics.prompt_calls : [];
  const latestPromptCall = promptCalls.length > 0 ? contextRecord(promptCalls[promptCalls.length - 1]) : {};
  const latestUsage = modelUsage.length > 0 ? contextRecord(modelUsage[modelUsage.length - 1]) : {};
  const stage = contextString(progressivePressure.stage) || contextString(pressure.stage) || "normal";
  const tokenUsage = contextNumber(progressivePressure.token_usage) || contextNumber(pressure.token_usage);
  const windowTokens = contextNumber(progressivePressure.effective_window_tokens) || contextNumber(pressure.effective_window_tokens);
  const remainingTokens = contextNumber(progressivePressure.remaining_tokens) || contextNumber(pressure.remaining_tokens);
  const utilization = windowTokens > 0 ? Math.min(100, Math.round((tokenUsage / windowTokens) * 100)) : 0;
  const attempted = progressiveCompaction.attempted === true;
  const compacted = progressiveCompaction.compacted === true;
  const reason = contextString(progressiveCompaction.reason) || (compacted ? "compacted" : "未触发正式压缩");
  const microMessages = contextNumber(microcompact.compacted_messages);
  const microSaved = contextNumber(microcompact.estimated_tokens_saved);
  const rawTokens = contextNumber(progressivePressure.raw_segment_estimated_tokens) || contextNumber(rawPressure.token_usage);
  const assembledTokens = contextNumber(progressivePressure.assembled_context_estimated_tokens) || contextNumber(compressionStats.assembled_prompt_estimated_tokens) || contextNumber(compressionStats.system_context_estimated_tokens);
  const actualInputTokens = contextNumber(latestUsage.input_tokens);
  const providerUsage = contextString(latestUsage.source) === "provider";
  const toolSchemaTokens = contextNumber(latestPromptCall.tool_schema_tokens) || contextNumber(promptBreakdown.tool_schema_tokens);
  const runtimeSaved = contextNumber(runtimeToolContext.estimated_tokens_saved);
  const exactHistory = historyCoverage.complete_and_exact === true;
  const hasBoundary = historyCoverage.has_compaction_boundary === true;
  const stageStyle =
    stage === "manual_path"
      ? "bg-rose-100 text-rose-700"
      : stage === "aggressive"
        ? "bg-orange-100 text-orange-700"
        : stage === "auto_compact"
          ? "bg-amber-100 text-amber-700"
          : stage === "warning"
            ? "bg-yellow-100 text-yellow-700"
            : "bg-emerald-100 text-emerald-700";

  return (
    <div className="mt-3 border-t border-line pt-3">
      <button
        type="button"
        onClick={onToggle}
        className="flex w-full items-center justify-between gap-3 text-left text-xs text-slate-600"
      >
        <span className="flex min-w-0 items-center gap-2">
          <Gauge className="h-4 w-4 shrink-0 text-accent" />
          <span className="font-semibold text-ink">上下文诊断</span>
          <span className={`shrink-0 rounded-md px-2 py-0.5 text-[11px] font-medium ${stageStyle}`}>{contextStageLabel(stage)}</span>
          {compacted && <span className="shrink-0 text-accent">已插入压缩边界</span>}
        </span>
        <ChevronDown className={`h-3.5 w-3.5 shrink-0 transition ${collapsed ? "-rotate-90" : ""}`} />
      </button>

      {!collapsed && (
        <div className="mt-3 space-y-3 rounded-md bg-slate-50 p-3">
          <div>
            <div className="flex items-center justify-between text-[11px] text-slate-500">
              <span>上下文窗口占用</span>
              <span>{tokenUsage.toLocaleString()} / {windowTokens.toLocaleString()} tokens</span>
            </div>
            <div className="mt-2 h-2 overflow-hidden rounded-full bg-slate-200">
              <div
                className={`h-full rounded-full ${stage === "normal" ? "bg-emerald-500" : stage === "warning" ? "bg-yellow-500" : "bg-orange-500"}`}
                style={{ width: `${Math.max(2, utilization)}%` }}
              />
            </div>
            <div className="mt-1 flex justify-between text-[10px] text-slate-400">
              <span>normal</span>
              <span>warning</span>
              <span>auto compact</span>
              <span>manual</span>
            </div>
          </div>

          <div className="grid grid-cols-3 gap-px overflow-hidden rounded-md bg-line text-xs">
            <div className="bg-white px-3 py-2">
              <div className="text-slate-400">原始片段</div>
              <div className="mt-1 font-semibold text-ink">{rawTokens.toLocaleString()} tokens</div>
            </div>
            <div className="bg-white px-3 py-2">
              <div className="text-slate-400">组装后上下文</div>
              <div className="mt-1 font-semibold text-ink">{assembledTokens.toLocaleString()} tokens</div>
            </div>
            <div className="bg-white px-3 py-2">
              <div className="text-slate-400">剩余空间</div>
              <div className="mt-1 font-semibold text-ink">{remainingTokens.toLocaleString()} tokens</div>
            </div>
          </div>

          <div className="rounded-md border border-slate-200 bg-white px-3 py-2 text-xs">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <span className="font-semibold text-ink">完整 Prompt 计量</span>
              <span className={providerUsage ? "text-emerald-700" : "text-slate-500"}>
                {providerUsage ? "供应商实际 usage" : "请求前估算"}
              </span>
            </div>
            <div className="mt-1 grid grid-cols-2 gap-x-4 gap-y-1 text-slate-500 sm:grid-cols-4">
              <span>实际输入：{actualInputTokens.toLocaleString()}</span>
              <span>组装估算：{assembledTokens.toLocaleString()}</span>
              <span>近期历史：{contextNumber(promptBreakdown.recent_history).toLocaleString()}</span>
              <span>工具 Schema：{toolSchemaTokens.toLocaleString()}</span>
            </div>
            {runtimeSaved > 0 && (
              <div className="mt-1 text-teal-700">轮内工具投影节省约 {runtimeSaved.toLocaleString()} tokens，完整结果仍保存在封存转录。</div>
            )}
          </div>

          <div className="flex flex-wrap items-center gap-2 text-[11px] text-slate-500">
            <span className={exactHistory ? "text-emerald-700" : "text-amber-700"}>
              近期历史：{exactHistory ? "完整原文" : "已裁剪或省略"}
            </span>
            {hasBoundary && <span>更早原文可通过 transcript 工具跨 Boundary 精确回查</span>}
            {contextDiagnostics.reactive_overflow_retry === true && <span className="text-orange-700">本轮执行过一次溢出恢复</span>}
          </div>

          {(microMessages > 0 || microSaved > 0) && (
            <div className="flex items-start justify-between gap-4 border-l-2 border-teal-300 pl-3 text-xs leading-5">
              <div>
                <div className="font-semibold text-ink">Microcompact 轻量压缩</div>
                <div className="text-slate-500">清理 {microMessages} 条可重新获取的旧工具结果</div>
              </div>
              <div className="shrink-0 font-semibold text-teal-700">节省约 {microSaved.toLocaleString()} tokens</div>
            </div>
          )}

          <div className="border-l-2 border-amber-300 pl-3 text-xs leading-5">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <span className="font-semibold text-ink">正式压缩</span>
              <span className={attempted ? "text-amber-700" : "text-slate-500"}>{attempted ? (compacted ? "已完成" : "已尝试") : "未执行"}</span>
            </div>
            <div className="text-slate-500">原因：{contextReasonLabel(reason)}</div>
            {compacted && (
              <div className="mt-1 text-slate-600">
                摘要 {contextNumber(progressiveCompaction.messages_summarized)} 条，保留最近 {contextNumber(progressiveCompaction.messages_preserved)} 条，压缩后约 {contextNumber(progressiveCompaction.post_compact_estimated_tokens).toLocaleString()} tokens
              </div>
            )}
            {(contextString(progressiveCompaction.boundary_id) || contextString(boundary.id)) && (
              <div className="mt-1 truncate font-mono text-[11px] text-slate-400">
                boundary: {contextString(progressiveCompaction.boundary_id) || contextString(boundary.id)}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ---- 上下文诊断用的宽松取值助手：后端字段是 unknown，这里统一兜底避免 NaN/undefined 渗透到 UI ----
/** 安全转对象：非对象/数组一律返回空对象 */
function contextRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
}

/** 安全转有限数字：其它情况返回 0（用于计数器/耗时等） */
function contextNumber(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

/** 安全转字符串：非字符串返回空串 */
function contextString(value: unknown): string {
  return typeof value === "string" ? value : "";
}

/** 压缩阶段标识 → 中文（正常 / 接近阈值 / 自动压缩 / 强压缩 / 正式压缩） */
function contextStageLabel(stage: string): string {
  if (stage === "warning") return "接近阈值";
  if (stage === "auto_compact") return "自动压缩";
  if (stage === "aggressive") return "强压缩";
  if (stage === "manual_path") return "正式压缩";
  return "正常";
}

/** 压缩原因码 → 人话说明，让用户知道「为什么这次没有触发压缩」 */
function contextReasonLabel(reason: string): string {
  if (reason === "microcompact_sufficient") return "轻量压缩后已经恢复安全余量";
  if (reason === "normal") return "当前上下文仍在安全区";
  if (reason === "warning") return "已经接近阈值，但暂时不需要正式压缩";
  if (reason === "not_enough_messages") return "历史消息不足，无法生成压缩摘要";
  if (reason === "nothing_to_summarize") return "没有需要继续摘要的旧消息";
  if (reason === "compacted") return "原始片段超过阈值，已生成摘要边界";
  return reason;
}

/**
 * 「技能 & MCP」面板：左侧技能清单，右侧展示选中技能的 manifest 与 MCP 工具运行时状态。
 * 全部受控：选中、刷新都通过 props 回调交给父组件，以免与首页其它状态分叉。
 */
function SkillMcpWorkspace({
  runtime,
  skills,
  selectedSkillName,
  loading,
  error,
  onSelectSkill,
  onRefresh
}: {
  runtime: McpRuntime | null;
  skills: SkillResponse[];
  selectedSkillName: string;
  loading: boolean;
  error: string | null;
  onSelectSkill: (name: string) => void;
  onRefresh: () => void;
}) {
  const selectedSkill = skills.find((item) => item.manifest.name === selectedSkillName) ?? skills[0];

  return (
    <div className="space-y-5">
      <section aria-label="MCP Server 验证">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="flex items-center gap-2 text-sm font-semibold text-ink">
              <Cable className="h-4 w-4 text-accent" />
              MCP Server 验证
            </div>
            <p className="mt-1 text-xs leading-5 text-slate-500">页面会真实执行 initialize、initialized 与 tools/list。</p>
          </div>
          <button
            type="button"
            onClick={onRefresh}
            disabled={loading}
            className="inline-flex h-8 shrink-0 items-center gap-1.5 rounded-md border border-line bg-white px-2.5 text-xs font-medium text-slate-700 transition hover:bg-teal-50 disabled:opacity-60"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
            重新验证
          </button>
        </div>

        {error && <div className="mt-3 rounded-md border border-rose-200 bg-rose-50 p-3 text-xs text-rose-700">{error}</div>}
        {!runtime && !error && (
          <div className="mt-3 rounded-md border border-dashed border-line bg-white p-4 text-center text-xs text-slate-500">
            {loading ? "正在启动 stdio 子进程并执行 MCP 握手..." : "等待验证 MCP Server"}
          </div>
        )}
        {runtime && (
          <>
            <div className="mt-3 border-y border-line bg-white py-3">
              <div className="flex items-center justify-between gap-3">
                <div className="min-w-0">
                  <div className="truncate text-sm font-semibold text-ink">{runtime.server_name}</div>
                  <div className="mt-1 text-xs text-slate-500">Server v{runtime.server_version}</div>
                </div>
                <span className="inline-flex shrink-0 items-center gap-1.5 rounded-full bg-emerald-50 px-2.5 py-1 text-xs font-semibold text-emerald-700">
                  <span className="h-2 w-2 rounded-full bg-emerald-500" />
                  握手成功
                </span>
              </div>
              <dl className="mt-3 grid grid-cols-2 gap-x-3 gap-y-2 text-xs">
                <div><dt className="text-slate-400">Protocol</dt><dd className="mt-0.5 font-medium text-slate-700">{runtime.protocol_version}</dd></div>
                <div><dt className="text-slate-400">Transport</dt><dd className="mt-0.5 font-medium text-slate-700">{runtime.transport}</dd></div>
              </dl>
            </div>
            <div className="mt-3">
              <div className="flex items-center justify-between text-xs">
                <span className="font-semibold text-ink">tools/list 返回结果</span>
                <span className="text-slate-500">{runtime.tools.length} 个只读工具</span>
              </div>
              <div className="mt-2 divide-y divide-line border-y border-line bg-white">
                {runtime.tools.map((tool) => (
                  <div key={tool.name} className="py-3">
                    <div className="break-all font-mono text-xs font-semibold text-accent">{tool.name}</div>
                    <p className="mt-1 text-xs leading-5 text-slate-600">{tool.description}</p>
                    <div className="mt-1 text-[11px] text-slate-400">
                      必填参数：{tool.input_schema.required?.join("、") || "无"}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}
      </section>

      <section aria-label="Skill Registry 验证" className="border-t border-line pt-5">
        <div className="flex items-center gap-2 text-sm font-semibold text-ink">
          <BookOpen className="h-4 w-4 text-accent" />
          Skill Registry 验证
        </div>
        <label className="mt-3 block text-xs font-medium text-slate-600" htmlFor="skill-selector">选择已加载 Skill</label>
        <select
          id="skill-selector"
          value={selectedSkill?.manifest.name ?? ""}
          onChange={(event) => onSelectSkill(event.target.value)}
          className="mt-1 h-9 w-full rounded-md border border-line bg-white px-2 text-xs text-slate-700 outline-none focus:border-teal-400"
        >
          {skills.map((item) => <option key={item.manifest.name} value={item.manifest.name}>{item.manifest.title}</option>)}
        </select>

        {selectedSkill && (
          <div className="mt-3 border-y border-line bg-white py-3 text-xs">
            <div className="flex items-start justify-between gap-3">
              <div>
                <div className="text-sm font-semibold text-ink">{selectedSkill.manifest.title}</div>
                <div className="mt-1 font-mono text-[11px] text-slate-500">{selectedSkill.manifest.name}</div>
              </div>
              <span className="rounded-full bg-teal-50 px-2 py-1 font-semibold text-teal-700">v{selectedSkill.manifest.version}</span>
            </div>
            <p className="mt-3 leading-5 text-slate-600">{selectedSkill.manifest.description}</p>
            <dl className="mt-3 grid grid-cols-2 gap-3">
              <div><dt className="text-slate-400">Entrypoint</dt><dd className="mt-0.5 break-all font-mono text-[11px] font-medium text-slate-700">{selectedSkill.manifest.entrypoint}</dd></div>
              <div><dt className="text-slate-400">Safety</dt><dd className="mt-0.5 inline-flex items-center gap-1 font-medium text-emerald-700"><ShieldCheck className="h-3.5 w-3.5" />{selectedSkill.manifest.safety_level}</dd></div>
            </dl>
            <div className="mt-4 font-semibold text-ink">自动触发条件</div>
            <div className="mt-2 flex flex-wrap gap-1.5">
              {selectedSkill.manifest.triggers.map((trigger) => <span key={trigger} className="rounded bg-teal-50 px-2 py-1 text-[10px] text-teal-700">{trigger}</span>)}
            </div>
            <div className="mt-4 font-semibold text-ink">工作流步骤</div>
            <ol className="mt-2 space-y-2">
              {selectedSkill.manifest.workflow_steps.map((step, index) => (
                <li key={step} className="flex items-start gap-2 text-slate-600">
                  <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-slate-100 text-[10px] font-semibold text-slate-600">{index + 1}</span>
                  <span className="break-all font-mono text-[11px] leading-5">{step}</span>
                </li>
              ))}
            </ol>
            <div className="mt-4 font-semibold text-ink">绑定工具</div>
            <div className="mt-2 flex flex-wrap gap-1.5">
              {selectedSkill.manifest.tools.map((tool) => <span key={tool} className="rounded bg-slate-100 px-2 py-1 font-mono text-[10px] text-slate-600">{tool}</span>)}
            </div>
          </div>
        )}
      </section>
    </div>
  );
}

/**
 * 消息内嵌的执行轨迹：按顺序展示思考、工具调用与工具结果。
 * @param collapsed 是否折叠；@param responsePending 为 true 时显示「进行中」动画
 */
function ChatTraceInline({
  traceItems,
  collapsed,
  responsePending,
  onToggle
}: {
  traceItems: ChatTraceItem[];
  collapsed: boolean;
  responsePending: boolean;
  onToggle: () => void;
}) {
  if (!traceItems.length) return null;
  const runningCount = traceItems.filter((item) => item.status === "running").length;
  const errorCount = traceItems.filter((item) => item.status === "error").length;
  const toolCount = traceItems.filter((item) => item.kind === "tool" || item.kind === "error").length;
  const ragPrefetch = traceItems.find((item) => item.kind === "rag_prefetch");
  const statusLabel = responsePending ? "回复中" : runningCount > 0 ? "执行中" : errorCount > 0 ? `${errorCount} 个失败` : "已完成";
  return (
    <div className="space-y-2 border-l-2 border-teal-100 pl-3">
      <button
        type="button"
        onClick={onToggle}
        className="flex w-full items-center justify-between gap-3 rounded-md border border-line bg-slate-50 px-3 py-2 text-left text-xs text-slate-600 transition hover:border-teal-200 hover:bg-teal-50"
      >
        <span className="flex min-w-0 items-center gap-2">
          <ChevronDown className={`h-3.5 w-3.5 shrink-0 transition ${collapsed ? "-rotate-90" : ""}`} />
          <span className="truncate font-semibold text-ink">工具调用过程</span>
          <span className="shrink-0 text-slate-500">
            {toolCount} 个工具 · {ragPrefetch ? `RAG 预检索 ${ragPrefetch.evidenceCount ?? 0} 条 · ` : ""}{traceItems.length} 条事件
          </span>
        </span>
        <span className={errorCount > 0 ? "shrink-0 text-rose-600" : responsePending || runningCount > 0 ? "shrink-0 text-accent" : "shrink-0 text-slate-500"}>
          {statusLabel}
        </span>
      </button>
      {!collapsed &&
        traceItems.map((item) => (
          <div
            key={item.id}
            className={`rounded-md px-3 py-2 text-xs leading-5 ${
              item.status === "error"
                ? "bg-rose-50 text-rose-700"
                : item.status === "running"
                  ? "bg-teal-50 text-slate-700"
                  : item.kind === "rag_prefetch"
                    ? "border border-cyan-100 bg-cyan-50 text-slate-700"
                  : "bg-slate-50 text-slate-600"
            }`}
          >
            <div className="mb-1 flex items-center justify-between gap-2">
              <span className="font-semibold text-ink">{item.title}</span>
              {item.status === "running" && <span className="h-2 w-2 shrink-0 animate-pulse rounded-full bg-accent" />}
            </div>
            {item.skillTitle && (
              <div className="mb-2 flex flex-wrap items-center gap-1.5 border-y border-teal-100 py-1.5 text-[11px] text-teal-700">
                <BookOpen className="h-3.5 w-3.5" />
                <span className="font-semibold">Skill：{item.skillTitle}</span>
                {item.skillVersion && <span>v{item.skillVersion}</span>}
                {item.skillName && <span className="font-mono text-teal-600">{item.skillName}</span>}
                {item.skillActivationMode && <span>· {item.skillActivationMode}</span>}
              </div>
            )}
            {item.kind === "thinking" ? (
              <MarkdownMessage content={item.content} />
            ) : (
              <div className="max-h-32 overflow-auto whitespace-pre-wrap">{item.content}</div>
            )}
            {item.skillSteps && item.skillSteps.length > 0 && (
              <div className="mt-2 border-t border-slate-200 pt-2">
                <div className="mb-1 font-semibold text-slate-600">Skill 工作流</div>
                <div className="flex flex-wrap gap-1">
                  {item.skillSteps.map((step, index) => (
                    <span key={step} className="rounded bg-white px-1.5 py-0.5 font-mono text-[10px] text-slate-500">{index + 1}. {step}</span>
                  ))}
                </div>
              </div>
            )}
          </div>
        ))}
    </div>
  );
}

/** 用 react-markdown 渲染回答，开启 GFM 以支持表格/删除线/任务列表等写法 */
function MarkdownMessage({ content }: { content: string }) {
  return (
    <div className="markdown-body">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          table: ({ children, ...props }) => (
            <div className="markdown-table-wrap">
              <table {...props}>{children}</table>
            </div>
          )
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}

/**
 * Issue 分析弹窗：左侧渲染结构化分析结果（分类、优先级、负责人建议等），
 * 右侧用 AgentTraceAside 同步展示流式分析过程。loading 时右侧滚动，error 时提供重试。
 */
function IssueAnalysisOverlay({
  issue,
  analysis,
  traceItems,
  loading,
  error,
  onRetry,
  onClose
}: {
  issue: Issue;
  analysis?: IssueAnalysisResult;
  traceItems: IssueAnalysisTraceItem[];
  loading: boolean;
  error: string | null;
  onRetry: () => void;
  onClose: () => void;
}) {
  const checklist = analysis?.checklist?.length ? analysis.checklist : analysis?.action_items ?? [];
  const drafts = analysis?.drafts;
  const hasDrafts = Boolean(drafts?.clarification_comment || drafts?.task_breakdown);
  const traceScrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const node = traceScrollRef.current;
    if (!node) return;
    node.scrollTo({ top: node.scrollHeight, behavior: "smooth" });
  }, [traceItems]);

  return (
    <div className="absolute inset-0 z-50 grid grid-cols-1 items-center gap-3 bg-slate-950/35 p-4 backdrop-blur-sm xl:grid-cols-[280px_minmax(0,1fr)] xl:p-6">
      <aside className="order-2 flex max-h-[28vh] min-h-0 flex-col overflow-hidden rounded-lg border border-teal-100 bg-white/95 shadow-xl xl:order-1 xl:h-[86vh] xl:max-h-[86vh]">
        <div className="border-b border-line px-4 py-3">
          <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-accent">Thinking Trace</div>
          <h3 className="mt-1 text-sm font-semibold text-ink">公开分析过程</h3>
          <p className="mt-1 text-xs leading-5 text-slate-500">展示工具调用、证据收集和面向用户的分析摘要。</p>
        </div>
        <div ref={traceScrollRef} className="min-h-0 flex-1 space-y-2 overflow-y-auto p-3">
          {traceItems.length === 0 && (
            <div className="rounded-md border border-dashed border-line bg-panel p-3 text-xs leading-5 text-slate-500">
              点击“重新分析”后，这里会流式显示分析步骤。
            </div>
          )}
          {traceItems.map((item) => (
            <div
              key={item.id}
              className={`rounded-md border p-3 text-xs leading-5 shadow-sm ${
                item.kind === "error"
                  ? "border-rose-200 bg-rose-50 text-rose-700"
                  : item.kind === "thinking"
                    ? "border-teal-200 bg-teal-50/80 text-slate-700"
                    : "border-line bg-white text-slate-600"
              }`}
            >
              <div className="mb-1 flex items-center justify-between gap-2">
                <span className="font-semibold text-ink">{item.title}</span>
                {item.status === "running" && <span className="h-2 w-2 shrink-0 animate-pulse rounded-full bg-accent" />}
              </div>
              {item.kind === "thinking" ? <MarkdownMessage content={item.content} /> : <div className="whitespace-pre-wrap">{item.content}</div>}
            </div>
          ))}
        </div>
      </aside>

      <div role="dialog" aria-modal="true" className="order-1 flex max-h-[86vh] min-h-0 w-full flex-col overflow-hidden rounded-lg border border-line bg-white shadow-2xl xl:order-2">
        <div className="flex items-start justify-between gap-4 border-b border-line px-5 py-4">
          <div className="min-w-0">
            <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-accent">Issue Agent</div>
            <h3 className="mt-1 line-clamp-2 text-base font-semibold text-ink">
              #{issue.number} {issue.title}
            </h3>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            <button
              onClick={onRetry}
              disabled={loading}
              className="inline-flex h-8 items-center gap-1 rounded-md border border-line bg-white px-2.5 text-xs text-slate-600 transition hover:bg-teal-50 disabled:opacity-50"
              title="重新分析"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
              重新分析
            </button>
            <button onClick={onClose} className="inline-flex h-8 w-8 items-center justify-center rounded-md border border-line bg-white text-slate-500 transition hover:bg-panel" title="关闭">
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto px-5 py-4">
          {loading && !analysis && (
            <div className="rounded-md border border-line bg-panel p-4 text-sm text-slate-600">Issue Agent 正在检索 Issue、相似历史、代码、文档和会话上下文...</div>
          )}

          {error && (
            <div className="rounded-md border border-rose-200 bg-rose-50 p-4 text-sm leading-6 text-rose-700">
              分析失败：{error}
            </div>
          )}

          {analysis && (
            <div className="space-y-4">
              <section className="rounded-md border border-teal-200 bg-teal-50/60 p-4">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="rounded-full bg-accent px-3 py-1 text-xs font-semibold text-white">{analysis.conclusion}</span>
                  <span className="rounded-full bg-white px-2.5 py-1 text-xs text-slate-600">优先级 {analysis.priority}</span>
                  <span className="rounded-full bg-white px-2.5 py-1 text-xs text-slate-600">复杂度 {analysis.complexity}</span>
                  <span className="rounded-full bg-white px-2.5 py-1 text-xs text-slate-600">分类 {analysis.category}</span>
                  <span className="rounded-full bg-white px-2.5 py-1 text-xs text-slate-600">置信度 {Math.round((analysis.confidence ?? 0) * 100)}%</span>
                </div>
                <p className="mt-3 text-sm leading-6 text-slate-700">{analysis.conclusion_reason || analysis.summary}</p>
              </section>

              <section className="rounded-md border border-line bg-white p-4">
                <h4 className="text-sm font-semibold text-ink">建议负责人</h4>
                <p className="mt-2 text-sm leading-6 text-slate-700">
                  <span className="font-semibold text-ink">{analysis.suggested_owner}</span>
                  <span className="ml-2">{analysis.owner_reason}</span>
                </p>
              </section>

              <section className="rounded-md border border-line bg-white p-4">
                <h4 className="text-sm font-semibold text-ink">证据</h4>
                <div className="mt-3 space-y-2">
                  {(analysis.evidence ?? []).length === 0 && <div className="text-sm text-slate-500">暂无可用证据。</div>}
                  {(analysis.evidence ?? []).map((item, index) => (
                    <div key={`${item.source_type}-${item.reference ?? "ref"}-${index}`} className="rounded-md bg-[#fbf7f2] p-3 text-sm leading-6 text-slate-700">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="rounded-full bg-white px-2 py-0.5 text-[11px] uppercase tracking-[0.08em] text-slate-500">{item.source_type}</span>
                        <span className="font-medium text-ink">{item.title}</span>
                        {typeof item.confidence === "number" && <span className="text-xs text-slate-400">{Math.round(item.confidence * 100)}%</span>}
                      </div>
                      <p className="mt-1">{item.snippet}</p>
                    </div>
                  ))}
                </div>
              </section>

              <section className="rounded-md border border-line bg-white p-4">
                <h4 className="text-sm font-semibold text-ink">下一步 checklist</h4>
                <ul className="mt-3 space-y-2 text-sm leading-6 text-slate-700">
                  {checklist.map((item) => (
                    <li key={item} className="flex gap-2">
                      <CheckCircle2 className="mt-1 h-4 w-4 shrink-0 text-accent" />
                      <span>{item}</span>
                    </li>
                  ))}
                </ul>
              </section>

              {hasDrafts && (
                <section className="rounded-md border border-line bg-white p-4">
                  <h4 className="text-sm font-semibold text-ink">可选草稿</h4>
                  {drafts?.clarification_comment && (
                    <div className="mt-3 rounded-md bg-panel p-3 text-sm leading-6 text-slate-700">
                      <div className="mb-2 text-xs font-semibold text-slate-500">澄清评论</div>
                      <MarkdownMessage content={drafts.clarification_comment} />
                    </div>
                  )}
                  {drafts?.task_breakdown && (
                    <div className="mt-3 rounded-md bg-panel p-3 text-sm leading-6 text-slate-700">
                      <div className="mb-2 text-xs font-semibold text-slate-500">任务拆分说明</div>
                      <MarkdownMessage content={drafts.task_breakdown} />
                    </div>
                  )}
                </section>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/**
 * 通用的「Agent 过程轨迹」侧栏（Issue / PR / CI 三种分析弹窗复用）。
 * 只负责渲染轨迹列表与自动滚动到底部，具体数据仍由调用方传入。
 */
function AgentTraceAside({
  eyebrow,
  title,
  description,
  emptyLabel,
  traceItems
}: {
  eyebrow: string;
  title: string;
  description: string;
  emptyLabel: string;
  traceItems: IssueAnalysisTraceItem[];
}) {
  const traceScrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const node = traceScrollRef.current;
    if (!node) return;
    node.scrollTo({ top: node.scrollHeight, behavior: "smooth" });
  }, [traceItems]);

  return (
    <aside className="order-2 flex max-h-[28vh] min-h-0 flex-col overflow-hidden rounded-lg border border-teal-100 bg-white/95 shadow-xl xl:order-1 xl:h-[86vh] xl:max-h-[86vh]">
      <div className="border-b border-line px-4 py-3">
        <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-accent">{eyebrow}</div>
        <h3 className="mt-1 text-sm font-semibold text-ink">{title}</h3>
        <p className="mt-1 text-xs leading-5 text-slate-500">{description}</p>
      </div>
      <div ref={traceScrollRef} className="min-h-0 flex-1 space-y-2 overflow-y-auto p-3">
        {traceItems.length === 0 && (
          <div className="rounded-md border border-dashed border-line bg-panel p-3 text-xs leading-5 text-slate-500">{emptyLabel}</div>
        )}
        {traceItems.map((item) => (
          <div
            key={item.id}
            className={`rounded-md border p-3 text-xs leading-5 shadow-sm ${
              item.kind === "error"
                ? "border-rose-200 bg-rose-50 text-rose-700"
                : item.kind === "thinking"
                  ? "border-teal-200 bg-teal-50/80 text-slate-700"
                  : "border-line bg-white text-slate-600"
            }`}
          >
            <div className="mb-1 flex items-center justify-between gap-2">
              <span className="font-semibold text-ink">{item.title}</span>
              {item.status === "running" && <span className="h-2 w-2 shrink-0 animate-pulse rounded-full bg-accent" />}
            </div>
            {item.kind === "thinking" ? <MarkdownMessage content={item.content} /> : <div className="whitespace-pre-wrap">{item.content}</div>}
          </div>
        ))}
      </div>
    </aside>
  );
}

/** PR 分析弹窗：展示摘要、风险点、Review Checklist 与测试建议，右侧同步流式轨迹 */
function PRAnalysisOverlay({
  pr,
  analysis,
  traceItems,
  loading,
  error,
  onRetry,
  onClose
}: {
  pr: PullRequest;
  analysis?: PRAnalysisResult;
  traceItems: IssueAnalysisTraceItem[];
  loading: boolean;
  error: string | null;
  onRetry: () => void;
  onClose: () => void;
}) {
  return (
    <div className="absolute inset-0 z-50 grid grid-cols-1 items-center gap-3 bg-slate-950/35 p-4 backdrop-blur-sm xl:grid-cols-[280px_minmax(0,1fr)] xl:p-6">
      <AgentTraceAside
        eyebrow="Plan & Execute"
        title="PR Review 过程"
        description="展示 Review 计划、执行步骤、证据收集和合入判断。"
        emptyLabel="点击“重新分析”后，这里会流式显示 PR Review 计划和执行过程。"
        traceItems={traceItems}
      />

      <div role="dialog" aria-modal="true" className="order-1 flex max-h-[86vh] min-h-0 w-full flex-col overflow-hidden rounded-lg border border-line bg-white shadow-2xl xl:order-2">
        <div className="flex items-start justify-between gap-4 border-b border-line px-5 py-4">
          <div className="min-w-0">
            <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-accent">PR Merge Readiness Agent</div>
            <h3 className="mt-1 line-clamp-2 text-base font-semibold text-ink">
              #{pr.number} {pr.title}
            </h3>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            <button
              onClick={onRetry}
              disabled={loading}
              className="inline-flex h-8 items-center gap-1 rounded-md border border-line bg-white px-2.5 text-xs text-slate-600 transition hover:bg-teal-50 disabled:opacity-50"
              title="重新分析"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
              重新分析
            </button>
            <button onClick={onClose} className="inline-flex h-8 w-8 items-center justify-center rounded-md border border-line bg-white text-slate-500 transition hover:bg-panel" title="关闭">
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto px-5 py-4">
          {loading && !analysis && (
            <div className="rounded-md border border-line bg-panel p-4 text-sm text-slate-600">PR Agent 正在制定 Review 计划，并检查 diff、代码、CI、文档和 review comments...</div>
          )}

          {error && (
            <div className="rounded-md border border-rose-200 bg-rose-50 p-4 text-sm leading-6 text-rose-700">
              分析失败：{error}
            </div>
          )}

          {analysis && (
            <div className="space-y-4">
              <section className="rounded-md border border-teal-200 bg-teal-50/60 p-4">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="rounded-full bg-accent px-3 py-1 text-xs font-semibold text-white">{analysis.merge_recommendation}</span>
                  <span className="rounded-full bg-white px-2.5 py-1 text-xs text-slate-600">置信度 {Math.round((analysis.confidence ?? 0) * 100)}%</span>
                  <span className="rounded-full bg-white px-2.5 py-1 text-xs text-slate-600">风险点 {(analysis.risk_points ?? []).length}</span>
                  <span className="rounded-full bg-white px-2.5 py-1 text-xs text-slate-600">阻塞项 {(analysis.blocking_issues ?? []).length}</span>
                </div>
                <p className="mt-3 text-sm leading-6 text-slate-700">{analysis.recommendation_reason || analysis.summary}</p>
              </section>

              <section className="rounded-md border border-line bg-white p-4">
                <h4 className="text-sm font-semibold text-ink">Review 计划与执行</h4>
                <div className="mt-3 grid gap-3 lg:grid-cols-2">
                  <AnalysisBulletList title="计划" items={analysis.plan ?? []} emptyLabel="暂无计划。" />
                  <AnalysisBulletList title="已执行" items={analysis.executed_steps ?? []} emptyLabel="暂无执行记录。" />
                </div>
              </section>

              <section className="rounded-md border border-line bg-white p-4">
                <h4 className="text-sm font-semibold text-ink">变更与风险</h4>
                <div className="mt-3 grid gap-3 lg:grid-cols-2">
                  <AnalysisBulletList title="关键变更" items={analysis.key_changes ?? []} emptyLabel="暂无关键变更摘要。" />
                  <AnalysisBulletList title="风险点" items={analysis.risk_points ?? []} emptyLabel="暂无明显风险点。" tone="warn" />
                  <AnalysisBulletList title="阻塞项" items={analysis.blocking_issues ?? []} emptyLabel="暂无阻塞项。" tone="danger" />
                  <AnalysisBulletList title="重点文件" items={analysis.files_need_attention ?? []} emptyLabel="暂无重点文件。" />
                </div>
              </section>

              <section className="rounded-md border border-line bg-white p-4">
                <h4 className="text-sm font-semibold text-ink">合入前动作</h4>
                <div className="mt-3 grid gap-3 lg:grid-cols-2">
                  <AnalysisBulletList title="Review checklist" items={analysis.review_checklist ?? []} emptyLabel="暂无 checklist。" />
                  <AnalysisBulletList title="测试建议" items={analysis.test_suggestions ?? []} emptyLabel="暂无测试建议。" />
                </div>
              </section>

              {(analysis.review_comments ?? []).length > 0 && (
                <section className="rounded-md border border-line bg-white p-4">
                  <h4 className="text-sm font-semibold text-ink">Review comments 摘要</h4>
                  <AnalysisBulletList items={analysis.review_comments ?? []} emptyLabel="暂无 review comments。" />
                </section>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/** CI 分析弹窗：展示失败归因、排查步骤与风险提示，右侧同步流式轨迹 */
function CIAnalysisOverlay({
  run,
  analysis,
  traceItems,
  loading,
  error,
  onRetry,
  onClose
}: {
  run: WorkflowRun;
  analysis?: CIAnalysisResult;
  traceItems: IssueAnalysisTraceItem[];
  loading: boolean;
  error: string | null;
  onRetry: () => void;
  onClose: () => void;
}) {
  return (
    <div className="absolute inset-0 z-50 grid grid-cols-1 items-center gap-3 bg-slate-950/35 p-4 backdrop-blur-sm xl:grid-cols-[280px_minmax(0,1fr)] xl:p-6">
      <AgentTraceAside
        eyebrow="Plan & Execute"
        title="CI 排查过程"
        description="展示排查计划、失败定位、日志证据和阻塞判断。"
        emptyLabel="点击“重新分析”后，这里会流式显示 CI 排查计划和执行过程。"
        traceItems={traceItems}
      />

      <div role="dialog" aria-modal="true" className="order-1 flex max-h-[86vh] min-h-0 w-full flex-col overflow-hidden rounded-lg border border-line bg-white shadow-2xl xl:order-2">
        <div className="flex items-start justify-between gap-4 border-b border-line px-5 py-4">
          <div className="min-w-0">
            <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-accent">CI Root Cause Agent</div>
            <h3 className="mt-1 line-clamp-2 text-base font-semibold text-ink">{run.name}</h3>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            <button
              onClick={onRetry}
              disabled={loading}
              className="inline-flex h-8 items-center gap-1 rounded-md border border-line bg-white px-2.5 text-xs text-slate-600 transition hover:bg-teal-50 disabled:opacity-50"
              title="重新分析"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
              重新分析
            </button>
            <button onClick={onClose} className="inline-flex h-8 w-8 items-center justify-center rounded-md border border-line bg-white text-slate-500 transition hover:bg-panel" title="关闭">
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto px-5 py-4">
          {loading && !analysis && (
            <div className="rounded-md border border-line bg-panel p-4 text-sm text-slate-600">CI Agent 正在制定排查计划，并检查失败 job、错误日志、相关文件和最近 PR...</div>
          )}

          {error && (
            <div className="rounded-md border border-rose-200 bg-rose-50 p-4 text-sm leading-6 text-rose-700">
              分析失败：{error}
            </div>
          )}

          {analysis && (
            <div className="space-y-4">
              <section className="rounded-md border border-teal-200 bg-teal-50/60 p-4">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="rounded-full bg-accent px-3 py-1 text-xs font-semibold text-white">{analysis.failure_type}</span>
                  <span className={`rounded-full px-2.5 py-1 text-xs ${analysis.is_merge_blocking ? "bg-rose-50 text-rose-700" : "bg-emerald-50 text-emerald-700"}`}>
                    {analysis.is_merge_blocking ? "阻塞合并" : "不阻塞合并"}
                  </span>
                  <span className="rounded-full bg-white px-2.5 py-1 text-xs text-slate-600">置信度 {Math.round((analysis.confidence ?? 0) * 100)}%</span>
                </div>
                <p className="mt-3 text-sm leading-6 text-slate-700">{analysis.root_cause || analysis.failure_summary}</p>
                {analysis.blocking_reason && <p className="mt-2 text-xs leading-5 text-slate-500">{analysis.blocking_reason}</p>}
              </section>

              <section className="rounded-md border border-line bg-white p-4">
                <h4 className="text-sm font-semibold text-ink">排查计划与执行</h4>
                <div className="mt-3 grid gap-3 lg:grid-cols-2">
                  <AnalysisBulletList title="计划" items={analysis.plan ?? []} emptyLabel="暂无计划。" />
                  <AnalysisBulletList title="已执行" items={analysis.executed_steps ?? []} emptyLabel="暂无执行记录。" />
                </div>
              </section>

              {analysis.first_error && (
                <section className="rounded-md border border-line bg-white p-4">
                  <h4 className="text-sm font-semibold text-ink">首个关键错误块</h4>
                  <pre className="mt-3 max-h-56 overflow-auto whitespace-pre-wrap rounded-md border border-rose-200 bg-rose-50 p-3 text-xs leading-5 text-rose-900">{analysis.first_error}</pre>
                </section>
              )}

              <section className="rounded-md border border-line bg-white p-4">
                <h4 className="text-sm font-semibold text-ink">根因与修复</h4>
                <div className="mt-3 grid gap-3 lg:grid-cols-2">
                  <AnalysisBulletList title="可能原因" items={analysis.possible_causes ?? []} emptyLabel="暂无可能原因。" tone="warn" />
                  <AnalysisBulletList title="修复步骤" items={analysis.fix_steps ?? []} emptyLabel="暂无修复步骤。" />
                  <AnalysisBulletList title="排查步骤" items={analysis.debug_steps ?? []} emptyLabel="暂无排查步骤。" />
                  <AnalysisBulletList title="相关文件" items={analysis.related_files ?? []} emptyLabel="暂无相关文件。" />
                </div>
              </section>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/**
 * 分析结果里的列表区块（标题 + 条目），tone 控制配色：
 * default 常规、warn 风险、good 建议等，避免在三个弹窗里重复写同样的结构。
 */
function AnalysisBulletList({
  title,
  items,
  emptyLabel,
  tone = "default"
}: {
  title?: string;
  items: string[];
  emptyLabel: string;
  tone?: "default" | "warn" | "danger";
}) {
  const iconClass = tone === "danger" ? "text-rose-500" : tone === "warn" ? "text-amber-500" : "text-accent";
  const Icon = tone === "default" ? CheckCircle2 : AlertTriangle;
  return (
    <div className="rounded-md bg-[#fbf7f2] p-3">
      {title && <div className="text-xs font-semibold text-slate-500">{title}</div>}
      {items.length === 0 ? (
        <div className={`${title ? "mt-2" : ""} text-sm text-slate-500`}>{emptyLabel}</div>
      ) : (
        <ul className={`${title ? "mt-2" : ""} space-y-2 text-sm leading-6 text-slate-700`}>
          {items.map((item) => (
            <li key={item} className="flex gap-2">
              <Icon className={`mt-1 h-4 w-4 shrink-0 ${iconClass}`} />
              <span>{item}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

/**
 * 「接入项目」全屏弹窗：远端 GitHub 地址与本地路径两种方式，
 * 表单全部受控（每个字段一对 value/onChange），提交与关闭由父组件处理。
 * 保持完全受控的原因：连接成功后父组件要清空表单并刷新仓库列表。
 */
function AddProjectOverlay({
  source,
  repoUrl,
  localPath,
  cloneParentDir,
  provider,
  apiBaseUrl,
  token,
  loading,
  status,
  onSourceChange,
  onRepoUrlChange,
  onLocalPathChange,
  onCloneParentDirChange,
  onProviderChange,
  onApiBaseUrlChange,
  onTokenChange,
  onConnect,
  onClose
}: {
  source: AddProjectSource;
  repoUrl: string;
  localPath: string;
  cloneParentDir: string;
  provider: string;
  apiBaseUrl: string;
  token: string;
  loading: boolean;
  status: AddProjectStatus | null;
  onSourceChange: (value: AddProjectSource) => void;
  onRepoUrlChange: (value: string) => void;
  onLocalPathChange: (value: string) => void;
  onCloneParentDirChange: (value: string) => void;
  onProviderChange: (value: string) => void;
  onApiBaseUrlChange: (value: string) => void;
  onTokenChange: (value: string) => void;
  onConnect: () => void;
  onClose: () => void;
}) {
  const parsed = source === "remote" ? parseRepositoryInput(repoUrl) : null;
  const canConnect = source === "local" ? Boolean(localPath.trim()) : Boolean(repoUrl.trim());

  return (
    <div className="absolute inset-0 z-50 flex items-center justify-center bg-slate-950/35 p-6 backdrop-blur-sm">
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="add-project-title"
        className="flex max-h-[calc(100%-24px)] w-full max-w-5xl flex-col overflow-hidden rounded-lg border border-line bg-[#eef9fb] shadow-2xl"
      >
        <div className="flex items-start justify-between gap-4 border-b border-teal-100 bg-[#eef9fb] px-5 py-4">
          <div className="min-w-0">
            <div className="text-xs font-semibold uppercase tracking-[0.18em] text-accent">Add Project</div>
            <h2 id="add-project-title" className="mt-1 text-xl font-semibold text-ink">
              添加项目
            </h2>
          </div>
          <button
            onClick={onClose}
            disabled={loading}
            className="inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-md border border-line bg-white text-slate-600 transition hover:bg-teal-50 disabled:opacity-60"
            title="关闭"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto p-5">
          <div className="grid gap-4 xl:grid-cols-[minmax(0,1.1fr)_minmax(280px,0.9fr)]">
          <section className="rounded-md border border-line bg-white p-4 shadow-sm">
            <div className="grid gap-4">
              <div className="grid grid-cols-2 overflow-hidden rounded-md border border-line bg-slate-50 p-1">
                <button
                  type="button"
                  onClick={() => onSourceChange("remote")}
                  className={`h-9 rounded px-3 text-sm font-medium transition ${
                    source === "remote" ? "bg-white text-ink shadow-sm" : "text-slate-600 hover:text-ink"
                  }`}
                >
                  GitHub 地址
                </button>
                <button
                  type="button"
                  onClick={() => onSourceChange("local")}
                  className={`h-9 rounded px-3 text-sm font-medium transition ${
                    source === "local" ? "bg-white text-ink shadow-sm" : "text-slate-600 hover:text-ink"
                  }`}
                >
                  本地仓库
                </button>
              </div>

              {source === "remote" ? (
                <div className="grid gap-4">
                  <label className="text-sm font-medium text-slate-700">
                    GitHub 地址
                    <input
                      value={repoUrl}
                      onChange={(event) => onRepoUrlChange(event.target.value)}
                      onKeyDown={(event) => {
                        if (event.key === "Enter") {
                          event.preventDefault();
                          onConnect();
                        }
                      }}
                      placeholder="https://github.com/owner/repository 或 owner/repository"
                      className="mt-2 h-11 w-full rounded-md border border-line bg-white px-3 text-sm outline-none transition focus:border-teal-500 focus:ring-2 focus:ring-teal-100"
                    />
                  </label>

                  <label className="text-sm font-medium text-slate-700">
                    下载父目录
                    <input
                      value={cloneParentDir}
                      onChange={(event) => onCloneParentDirChange(event.target.value)}
                      placeholder="例如 D:\\code\\repos，留空则使用默认缓存目录"
                      className="mt-2 h-10 w-full rounded-md border border-line bg-white px-3 text-sm outline-none transition focus:border-teal-500 focus:ring-2 focus:ring-teal-100"
                    />
                  </label>
                </div>
              ) : (
                <label className="text-sm font-medium text-slate-700">
                  本地仓库路径
                  <div className="mt-2 flex gap-2">
                    <input
                      value={localPath}
                      onChange={(event) => onLocalPathChange(event.target.value)}
                      onKeyDown={(event) => {
                        if (event.key === "Enter") {
                          event.preventDefault();
                          onConnect();
                        }
                      }}
                      placeholder="例如 D:\\code\\my-repo"
                      className="h-11 min-w-0 flex-1 rounded-md border border-line bg-white px-3 text-sm outline-none transition focus:border-teal-500 focus:ring-2 focus:ring-teal-100"
                    />
                    <span className="inline-flex h-11 w-11 shrink-0 items-center justify-center rounded-md border border-line bg-white text-slate-500">
                      <Folder className="h-4 w-4" />
                    </span>
                  </div>
                </label>
              )}

              <div className="grid gap-3 md:grid-cols-2">
                <label className="text-sm font-medium text-slate-700">
                  托管平台
                  <select
                    value={provider}
                    onChange={(event) => onProviderChange(event.target.value)}
                    className="mt-2 h-10 w-full rounded-md border border-line bg-white px-3 text-sm outline-none transition focus:border-teal-500 focus:ring-2 focus:ring-teal-100"
                  >
                    <option value="github">GitHub</option>
                    <option value="github_compatible">GitHub-compatible API</option>
                  </select>
                </label>

                <label className="text-sm font-medium text-slate-700">
                  API Base URL
                  <input
                    value={apiBaseUrl}
                    disabled={provider === "github"}
                    onChange={(event) => onApiBaseUrlChange(event.target.value)}
                    className="mt-2 h-10 w-full rounded-md border border-line bg-white px-3 text-sm outline-none transition focus:border-teal-500 focus:ring-2 focus:ring-teal-100 disabled:bg-slate-100 disabled:text-slate-500"
                  />
                </label>
              </div>

              <label className="text-sm font-medium text-slate-700">
                Access Token
                <input
                  type="password"
                  value={token}
                  onChange={(event) => onTokenChange(event.target.value)}
                  placeholder="public repo 可留空，私有仓库需要 token"
                  className="mt-2 h-10 w-full rounded-md border border-line bg-white px-3 text-sm outline-none transition focus:border-teal-500 focus:ring-2 focus:ring-teal-100"
                />
              </label>

              {status && (
                <div
                  className={`rounded-md border px-3 py-2 text-sm leading-6 ${
                    status.type === "success" ? "border-emerald-200 bg-emerald-50 text-emerald-800" : "border-rose-200 bg-rose-50 text-rose-800"
                  }`}
                >
                  <div className="font-semibold">{status.title}</div>
                  {status.detail && <div className="mt-1 break-words">{status.detail}</div>}
                </div>
              )}

              <div className="flex flex-wrap items-center gap-3">
                <button
                  onClick={onConnect}
                  disabled={loading || !canConnect}
                  className="inline-flex h-10 items-center justify-center rounded-md bg-accent px-4 text-sm font-medium text-white transition hover:bg-teal-700 disabled:opacity-60"
                >
                  {loading ? "连接并同步中..." : "连接并同步"}
                </button>
                <button
                  onClick={onClose}
                  disabled={loading}
                  className="inline-flex h-10 items-center justify-center rounded-md border border-line bg-white px-4 text-sm font-medium text-slate-700 transition hover:bg-panel disabled:opacity-60"
                >
                  关闭
                </button>
              </div>
            </div>
          </section>

          <section className="rounded-md border border-line bg-white p-4 shadow-sm">
            <h3 className="text-sm font-semibold text-ink">识别结果</h3>
            <div className="mt-3 rounded-md bg-[#fbf7f2] p-3 text-sm leading-7 text-slate-600">
              {source === "remote" ? (
                <>
                  <div>
                    Owner: <span className="font-medium text-ink">{parsed?.owner || "待识别"}</span>
                  </div>
                  <div>
                    Repository: <span className="font-medium text-ink">{parsed?.repo || "待识别"}</span>
                  </div>
                  <div>
                    下载到: <span className="break-all font-medium text-ink">{cloneParentDir.trim() || "默认缓存目录"}</span>
                  </div>
                </>
              ) : (
                <>
                  <div>
                    本地路径: <span className="break-all font-medium text-ink">{localPath.trim() || "待选择"}</span>
                  </div>
                  <div>
                    远程仓库: <span className="font-medium text-ink">提交后从 origin 识别</span>
                  </div>
                </>
              )}
            </div>
            <div className="mt-4 space-y-2 text-xs leading-6 text-slate-500">
              {source === "remote" ? (
                <div>支持 `https://github.com/owner/repo`、`git@github.com:owner/repo.git`、`owner/repo`。</div>
              ) : (
                <div>本地仓库需要已配置 `origin` 远程地址。</div>
              )}
              <div>连接成功后会同步 GitHub 数据，并在后台索引代码用于代码 RAG 和风险分析。</div>
              <div>如果连接失败，会直接显示后端返回的原因，不再展开原始 JSON。</div>
            </div>
          </section>
          </div>
        </div>
      </div>
    </div>
  );
}

/** 右上角指标卡：图标 + 标签 + 数值，用于 Issue/PR/CI 数量概览 */
function MetricCard({ icon: Icon, label, value }: { icon: LucideIcon; label: string; value: number }) {
  return (
    <div className="rounded-md border border-line bg-white px-3 py-2">
      <div className="flex items-center justify-between">
        <div className="text-xs text-slate-500">{label}</div>
        <Icon className="h-3.5 w-3.5 text-accent" />
      </div>
      <div className="mt-1 text-lg font-semibold text-ink">{value}</div>
    </div>
  );
}

/**
 * 右栏顶部筛选区：关键词搜索 + 状态 / 负责人 / 时间三个下拉。
 * owners 由父组件从当前数据汇总而来；activeTab 决定某些筛选项是否可用（如 CI 没有负责人）。
 */
function WorkspaceControls({
  query,
  status,
  owner,
  time,
  owners,
  activeTab,
  onQueryChange,
  onStatusChange,
  onOwnerChange,
  onTimeChange
}: {
  query: string;
  status: string;
  owner: string;
  time: string;
  owners: string[];
  activeTab: WorkspaceTab;
  onQueryChange: (value: string) => void;
  onStatusChange: (value: string) => void;
  onOwnerChange: (value: string) => void;
  onTimeChange: (value: string) => void;
}) {
  return (
    <div className="space-y-2">
      <div className="relative">
        <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
        <input
          value={query}
          onChange={(event) => onQueryChange(event.target.value)}
          placeholder="关键词搜索标题、正文、标签"
          className="h-10 w-full rounded-md border border-line bg-white pl-9 pr-3 text-sm outline-none transition focus:border-teal-500 focus:ring-2 focus:ring-teal-100"
        />
      </div>
      <div className="grid grid-cols-3 gap-2">
        <SelectFilter value={status} onChange={onStatusChange} label="状态">
          <option value="all">全部状态</option>
          <option value="open">open</option>
          <option value="closed">closed</option>
          {activeTab === "PR" && <option value="merged">merged</option>}
          {activeTab === "CI" && <option value="failure">failure</option>}
          {activeTab === "CI" && <option value="success">success</option>}
        </SelectFilter>
        <SelectFilter value={owner} onChange={onOwnerChange} label="负责人" disabled={activeTab === "CI"}>
          <option value="all">全部负责人</option>
          <option value="unassigned">未分配</option>
          {owners.map((item) => (
            <option key={item} value={item}>
              {item}
            </option>
          ))}
        </SelectFilter>
        <SelectFilter value={time} onChange={onTimeChange} label="时间">
          <option value="all">全部时间</option>
          <option value="7d">近 7 天</option>
          <option value="30d">近 30 天</option>
        </SelectFilter>
      </div>
    </div>
  );
}

/** 统一的筛选下拉：label 作为无障碍标题，children 为选项；disabled 用于「当前页签不适用」的场景 */
function SelectFilter({
  value,
  label,
  disabled,
  children,
  onChange
}: {
  value: string;
  label: string;
  disabled?: boolean;
  children: React.ReactNode;
  onChange: (value: string) => void;
}) {
  return (
    <label className="relative block">
      <span className="sr-only">{label}</span>
      <select
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(event.target.value)}
        className="h-9 w-full appearance-none rounded-md border border-line bg-white px-2 pr-7 text-xs text-slate-700 outline-none transition focus:border-teal-500 focus:ring-2 focus:ring-teal-100 disabled:bg-slate-100 disabled:text-slate-400"
      >
        {children}
      </select>
      <ChevronDown className="pointer-events-none absolute right-2 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-slate-400" />
    </label>
  );
}

/**
 * 可折叠的分组（按状态/时间等维度聚合条目）。
 * 折叠状态由父组件传入 isCollapsed，保证刷新后仍保持用户的折叠选择。
 */
function CategoryGroup({
  label,
  color,
  count,
  items,
  isCollapsed,
  onToggle
}: {
  label: string;
  color: string;
  count: number;
  items: WorkspaceItem[];
  isCollapsed: boolean;
  onToggle: () => void;
}) {
  return (
    <section>
      <button type="button" onClick={onToggle} className="flex h-7 w-full items-center justify-between">
        <div className={`flex items-center gap-2 text-sm font-semibold ${color}`}>
          <span>{label}</span>
          <span className="rounded-full bg-[#f2eee8] px-2 py-0.5 text-xs font-medium text-slate-500">{count}</span>
        </div>
        <ChevronDown className={`h-3.5 w-3.5 text-slate-400 transition ${isCollapsed ? "-rotate-90" : ""}`} />
      </button>
      {!isCollapsed && items.length > 0 && (
        <div className="mt-2 space-y-2">
          {items.map((item) => (
            <WorkspaceItemButton key={item.id} item={item} />
          ))}
        </div>
      )}
    </section>
  );
}

/** 右栏列表项按钮：点击回传给父组件以打开详情（item.onClick 由生成 WorkspaceItem 时绑定） */
function WorkspaceItemButton({ item }: { item: WorkspaceItem }) {
  return (
    <button
      onClick={item.onClick}
      className={`w-full rounded-md border bg-white p-3 text-left transition ${
        item.selected ? "border-teal-400 ring-2 ring-teal-100" : "border-line hover:border-teal-200 hover:bg-teal-50"
      }`}
    >
      <div className="line-clamp-2 text-sm font-medium text-ink">{item.title}</div>
      <div className="mt-1 flex flex-wrap items-center gap-1.5 text-xs text-slate-500">
        <span>{item.meta}</span>
        {item.badges?.slice(0, 3).map((badge) => (
          <span key={badge} className="rounded-full bg-panel px-2 py-0.5">
            {badge}
          </span>
        ))}
      </div>
      {item.detail && <div className="mt-2 line-clamp-2 text-xs leading-5 text-slate-500">{item.detail}</div>}
    </button>
  );
}

/**
 * 右栏详情面板：显示选中条目（Issue / PR / CI 三选一存在）的字段，
 * 并提供「分析」按钮（analyzingXxx 控制按钮加载态）。
 */
function WorkspaceDetail({
  issue,
  pr,
  run,
  analyzingIssue,
  analyzingPr,
  analyzingRun,
  onAnalyzeIssue,
  onAnalyzePr,
  onAnalyzeRun,
  onClose
}: {
  issue?: Issue;
  pr?: PullRequest;
  run?: WorkflowRun;
  analyzingIssue?: boolean;
  analyzingPr?: boolean;
  analyzingRun?: boolean;
  onAnalyzeIssue?: () => void;
  onAnalyzePr?: () => void;
  onAnalyzeRun?: () => void;
  onClose: () => void;
}) {
  const title = issue ? `#${issue.number} ${issue.title}` : pr ? `#${pr.number} ${pr.title}` : run?.name ?? "";
  const body = issue?.body ?? pr?.body ?? run?.logs_text ?? "暂无详情内容";
  const status = issue?.state ?? (pr?.merged_at ? "merged" : pr?.state) ?? run?.conclusion ?? run?.status ?? "";
  const owner = issue?.assignees?.join(", ") || issue?.author || pr?.author || "未分配";
  const updatedAt = issue?.updated_at ?? issue?.created_at ?? pr?.updated_at ?? pr?.created_at ?? run?.updated_at ?? run?.created_at;

  return (
    <div className="mb-4 rounded-md border border-teal-200 bg-white p-4 shadow-sm">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-accent">详情</div>
          <h3 className="mt-1 text-sm font-semibold leading-6 text-ink">{title}</h3>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          {issue && (
            <button
              onClick={onAnalyzeIssue}
              disabled={!onAnalyzeIssue || analyzingIssue}
              className="inline-flex items-center gap-1 rounded-md bg-accent px-2.5 py-1 text-xs font-medium text-white transition hover:bg-teal-700 disabled:opacity-60"
            >
              <ListTodo className="h-3.5 w-3.5" />
              {analyzingIssue ? "分析中..." : "分析当前 Issue"}
            </button>
          )}
          {pr && (
            <button
              onClick={onAnalyzePr}
              disabled={!onAnalyzePr || analyzingPr}
              className="inline-flex items-center gap-1 rounded-md bg-accent px-2.5 py-1 text-xs font-medium text-white transition hover:bg-teal-700 disabled:opacity-60"
            >
              <GitPullRequest className="h-3.5 w-3.5" />
              {analyzingPr ? "分析中..." : "分析当前 PR"}
            </button>
          )}
          {run && (
            <button
              onClick={onAnalyzeRun}
              disabled={!onAnalyzeRun || analyzingRun}
              className="inline-flex items-center gap-1 rounded-md bg-accent px-2.5 py-1 text-xs font-medium text-white transition hover:bg-teal-700 disabled:opacity-60"
            >
              <AlertTriangle className="h-3.5 w-3.5" />
              {analyzingRun ? "排查中..." : "分析当前 CI"}
            </button>
          )}
          <button onClick={onClose} className="rounded-md border border-line px-2 py-1 text-xs text-slate-500 transition hover:bg-panel">
            关闭
          </button>
        </div>
      </div>
      <div className="mt-3 flex flex-wrap gap-2">
        <MetaPill label="状态" value={status} />
        <MetaPill label="负责人" value={owner} />
        <MetaPill label="更新时间" value={formatDate(updatedAt)} />
      </div>
      {(issue?.labels?.length ?? 0) > 0 && (
        <div className="mt-3 flex flex-wrap gap-1.5">
          {issue?.labels?.map((label) => (
            <span key={label} className="rounded-full bg-teal-50 px-2 py-0.5 text-xs text-teal-700">
              {label}
            </span>
          ))}
        </div>
      )}
      <div className="mt-3 whitespace-pre-wrap rounded-md bg-[#fbf7f2] p-3 text-xs leading-6 text-slate-700">{body}</div>
      {pr?.files && pr.files.length > 0 && (
        <div className="mt-3 space-y-1.5">
          {pr.files.map((file) => (
            <div key={file.id} className="rounded-md border border-line bg-white px-2 py-1.5 text-xs text-slate-600">
              <span className="font-medium text-ink">{file.filename}</span>
              <span className="ml-2 text-emerald-600">+{file.additions}</span>
              <span className="ml-1 text-rose-500">-{file.deletions}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/** 详情里的键值小标签（如「作者 / 分支」），统一视觉 */
function MetaPill({ label, value }: { label: string; value: string }) {
  return (
    <span className="rounded-full bg-panel px-2.5 py-1 text-xs text-slate-600">
      {label}: {value}
    </span>
  );
}

/** 右栏空态占位（未选中条目 / 数据为空时） */
function EmptyWorkspaceState({ label }: { label: string }) {
  return <div className="rounded-md border border-line bg-white p-4 text-sm text-slate-500">{label}</div>;
}

/**
 * 「团队」页签：维护成员名单（姓名/角色/优势/技术栈）。
 * 成员会被写入项目记忆，供 Agent 判断「某个模块该找谁」，所以这里是知识的一部分而不只是通讯录。
 */
function TeamWorkspace({
  members,
  draft,
  onDraftChange,
  onAddMember,
  onRemoveMember
}: {
  members: TeamMember[];
  draft: TeamMemberDraft;
  onDraftChange: (value: TeamMemberDraft) => void;
  onAddMember: () => void;
  onRemoveMember: (memberId: string) => void;
}) {
  return (
    <div className="space-y-4">
      <section className="rounded-md border border-line bg-white p-4">
        <div className="flex items-start justify-between gap-3">
          <div>
            <h3 className="text-sm font-semibold text-ink">人员管理</h3>
            <p className="mt-1 text-xs leading-5 text-slate-500">配置项目工作者，Agent 会在分配 Issue 时参考职位、擅长方向和技术栈。</p>
          </div>
          <span className="rounded-full bg-panel px-2 py-0.5 text-xs text-slate-500">{members.length} 人</span>
        </div>

        <div className="mt-4 grid gap-2">
          <input
            value={draft.name}
            onChange={(event) => onDraftChange({ ...draft, name: event.target.value })}
            placeholder="姓名，例如 Alice"
            className="h-9 rounded-md border border-line bg-white px-3 text-sm outline-none focus:border-teal-500 focus:ring-2 focus:ring-teal-100"
          />
          <input
            value={draft.role}
            onChange={(event) => onDraftChange({ ...draft, role: event.target.value })}
            placeholder="职位，例如 后端工程师 / 前端工程师 / Tech Lead"
            className="h-9 rounded-md border border-line bg-white px-3 text-sm outline-none focus:border-teal-500 focus:ring-2 focus:ring-teal-100"
          />
          <textarea
            value={draft.strengths}
            onChange={(event) => onDraftChange({ ...draft, strengths: event.target.value })}
            placeholder="擅长，例如 鉴权、API 设计、CI 排查"
            className="min-h-16 resize-none rounded-md border border-line bg-white px-3 py-2 text-sm outline-none focus:border-teal-500 focus:ring-2 focus:ring-teal-100"
          />
          <input
            value={draft.techStack}
            onChange={(event) => onDraftChange({ ...draft, techStack: event.target.value })}
            placeholder="技术栈，例如 Python, FastAPI, PostgreSQL"
            className="h-9 rounded-md border border-line bg-white px-3 text-sm outline-none focus:border-teal-500 focus:ring-2 focus:ring-teal-100"
          />
          <button
            onClick={onAddMember}
            disabled={!draft.name.trim()}
            className="inline-flex h-9 items-center justify-center rounded-md bg-accent px-3 text-sm font-medium text-white transition hover:bg-teal-700 disabled:opacity-60"
          >
            添加成员
          </button>
        </div>
      </section>

      <section className="rounded-md border border-line bg-white p-4">
        <h3 className="text-sm font-semibold text-ink">团队成员</h3>
        <div className="mt-3 space-y-2">
          {members.length === 0 && <div className="rounded-md bg-panel p-3 text-sm text-slate-500">还没有成员。先添加一个项目工作者。</div>}
          {members.map((member) => (
            <div key={member.id} className="rounded-md border border-line bg-white p-3">
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="font-medium text-ink">{member.name}</div>
                  <div className="mt-1 text-xs text-slate-500">{member.role || "未配置职位"}</div>
                </div>
                <button onClick={() => onRemoveMember(member.id)} className="rounded-md border border-line px-2 py-1 text-xs text-slate-500 hover:bg-panel">
                  移除
                </button>
              </div>
              <div className="mt-3 space-y-1 text-xs leading-5 text-slate-600">
                <div>擅长：{member.strengths || "未配置"}</div>
                <div>技术栈：{member.techStack || "未配置"}</div>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="rounded-md border border-line bg-white p-4 text-sm leading-7 text-slate-600">
        <div className="font-medium text-ink">报告管理</div>
        <p className="mt-2">可在中间对话框输入“生成本周研发周报”，或进入周报页面查看 Markdown 预览。</p>
        <Link href="/reports" className="mt-3 inline-flex rounded-md bg-accent px-3 py-1.5 text-xs font-medium text-white">
          打开周报页
        </Link>
      </section>
    </div>
  );
}

/**
 * 「建议建立索引」提示卡：项目还没有索引或索引已过期时出现，
 * 提供「开始扫描」与「7 天后再说」两个动作；扫描中的进度与错误也在这里展示。
 */
function MemoryIndexPromptCard({
  repo,
  state,
  scanning,
  error,
  onStartScan,
  onSnooze
}: {
  repo: Repository;
  state: ProjectIndexState;
  scanning: boolean;
  error: string | null;
  onStartScan: () => void;
  onSnooze: () => void;
}) {
  const isFailed = state.status === "failed";
  const isStale = state.status === "stale";
  const title = scanning ? "正在建立项目记忆索引..." : isFailed ? "记忆索引构建失败" : isStale ? "记忆索引已过期" : "这个项目还没有记忆索引";
  const description = scanning
    ? "正在扫描项目文档、README 和配置清单，完成后 Agent 就能在当前仓库里检索这些上下文。"
    : isFailed
      ? `项目 ${repo.full_name} 上次扫描没有完成：${state.error_message || error || "未知错误"}`
      : isStale
        ? `项目 ${repo.full_name} 的本地内容已变化，建议更新索引，让回答保持新鲜。`
        : "建立索引后，DevFlow AI 可以检索项目说明、架构文档、ADR、计划和经验记录，不必每次从零开始。";
  const actionLabel = scanning ? "扫描中..." : isFailed ? "重试扫描" : isStale ? "更新索引" : "开始扫描";

  return (
    <div className="flex justify-start">
      <div className="w-full rounded-md border border-teal-100 bg-white px-4 py-4 text-sm shadow-sm">
        <div className="flex items-start gap-3">
          <div className="mt-0.5 flex h-10 w-10 shrink-0 items-center justify-center rounded-md bg-teal-50 text-accent">
            {isFailed ? <AlertTriangle className="h-5 w-5" /> : scanning ? <RefreshCw className="h-5 w-5 animate-spin" /> : <Database className="h-5 w-5" />}
          </div>
          <div className="min-w-0 flex-1">
            <div className="font-semibold text-ink">{title}</div>
            <p className="mt-1 text-xs leading-5 text-slate-500">{description}</p>
            {!isFailed && (
              <div className="mt-3 grid gap-2 text-xs text-slate-500 sm:grid-cols-3">
                <div className="inline-flex items-center gap-1.5">
                  <Folder className="h-3.5 w-3.5 text-accent" />
                  README / docs / manifest
                </div>
                <div className="inline-flex items-center gap-1.5">
                  <Clock3 className="h-3.5 w-3.5 text-accent" />
                  后台扫描，可继续对话
                </div>
                <div className="inline-flex items-center gap-1.5">
                  <ShieldCheck className="h-3.5 w-3.5 text-accent" />
                  仅写入本地知识库
                </div>
              </div>
            )}
            {error && !isFailed && <div className="mt-3 rounded-md border border-rose-100 bg-rose-50 px-3 py-2 text-xs text-rose-700">{error}</div>}
            <div className="mt-4 flex flex-wrap items-center gap-2">
              <button
                onClick={onSnooze}
                disabled={scanning}
                className="rounded-md border border-line bg-white px-3 py-2 text-xs font-medium text-slate-500 transition hover:bg-panel disabled:opacity-60"
              >
                稍后再说
              </button>
              <button
                onClick={onStartScan}
                disabled={scanning}
                className="inline-flex items-center gap-1.5 rounded-md bg-accent px-3 py-2 text-xs font-medium text-white transition hover:bg-teal-700 disabled:opacity-60"
              >
                <RefreshCw className={`h-3.5 w-3.5 ${scanning ? "animate-spin" : ""}`} />
                {actionLabel}
              </button>
              <span className="text-xs text-slate-400">稍后提醒会冷却 7 天，也可以在知识库页手动管理索引内容。</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * 「记忆&知识库」顶层容器：把索引提示、记忆检索入口、笔记与知识文件几块拼在一起。
 * 它只做编排，具体数据（notes/knowledgeItems）与操作（新增/删除/上传）都由父组件下发。
 */
function MemoryHubWorkspace({
  repoId,
  notes,
  draft,
  knowledgeItems,
  uploading,
  onDraftChange,
  onAddNote,
  onRemoveNote,
  onUploadFile,
  onRemoveKnowledgeItem,
  onRefreshKnowledge
}: {
  repoId?: string;
  notes: MemoryNote[];
  draft: MemoryNoteDraft;
  knowledgeItems: KnowledgeItem[];
  uploading: boolean;
  onDraftChange: (value: MemoryNoteDraft) => void;
  onAddNote: () => void;
  onRemoveNote: (noteId: string) => void;
  onUploadFile: (file: File) => void;
  onRemoveKnowledgeItem: (item: KnowledgeItem) => void;
  onRefreshKnowledge: () => void;
}) {
  const [activeTab, setActiveTab] = useState<"recall" | "search" | "status" | "review">("recall");
  const [recallEvents, setRecallEvents] = useState<RecallEvent[]>([]);
  const [memoryStatus, setMemoryStatus] = useState<MemoryStatus | null>(null);
  const [candidates, setCandidates] = useState<MemoryCandidate[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<MemorySearchResult[]>([]);
  const [loadingPanel, setLoadingPanel] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const visibleKnowledgeItems = knowledgeItems.filter(
    (item) => item.source_type !== "memory_note" && item.source_type !== "team_member"
  );

  async function loadRecallEvents() {
    if (!repoId) return;
    setLoadingPanel("recall");
    setError(null);
    try {
      setRecallEvents(await apiGet<RecallEvent[]>(`/api/repos/${repoId}/memory/recall-events?limit=30`));
    } catch (err) {
      setError(getFriendlyError(err));
    } finally {
      setLoadingPanel((current) => (current === "recall" ? null : current));
    }
  }

  async function loadMemoryStatus() {
    if (!repoId) return;
    setLoadingPanel("status");
    setError(null);
    try {
      setMemoryStatus(await apiGet<MemoryStatus>(`/api/repos/${repoId}/memory/status`));
    } catch (err) {
      setError(getFriendlyError(err));
    } finally {
      setLoadingPanel((current) => (current === "status" ? null : current));
    }
  }

  async function loadCandidates(status = "pending") {
    if (!repoId) return;
    setLoadingPanel("review");
    setError(null);
    try {
      setCandidates(await apiGet<MemoryCandidate[]>(`/api/repos/${repoId}/memory/candidates?status=${encodeURIComponent(status)}&limit=50`));
    } catch (err) {
      setError(getFriendlyError(err));
    } finally {
      setLoadingPanel((current) => (current === "review" ? null : current));
    }
  }

  async function runMemorySearch() {
    if (!repoId || !searchQuery.trim()) return;
    setLoadingPanel("search");
    setError(null);
    try {
      const response = await apiGet<{ results: MemorySearchResult[] }>(
        `/api/repos/${repoId}/memory/search?q=${encodeURIComponent(searchQuery.trim())}&limit=10`
      );
      setSearchResults(response.results);
      await loadRecallEvents();
    } catch (err) {
      setError(getFriendlyError(err));
    } finally {
      setLoadingPanel((current) => (current === "search" ? null : current));
    }
  }

  async function approveCandidate(candidateId: string) {
    if (!repoId) return;
    setLoadingPanel(candidateId);
    setError(null);
    try {
      await apiPost(`/api/repos/${repoId}/memory/candidates/${candidateId}/approve`, {});
      await Promise.all([loadCandidates(), loadMemoryStatus()]);
      onRefreshKnowledge();
    } catch (err) {
      setError(getFriendlyError(err));
    } finally {
      setLoadingPanel((current) => (current === candidateId ? null : current));
    }
  }

  async function rejectCandidate(candidateId: string) {
    if (!repoId) return;
    setLoadingPanel(candidateId);
    setError(null);
    try {
      await apiPost(`/api/repos/${repoId}/memory/candidates/${candidateId}/reject`, {});
      await Promise.all([loadCandidates(), loadMemoryStatus()]);
    } catch (err) {
      setError(getFriendlyError(err));
    } finally {
      setLoadingPanel((current) => (current === candidateId ? null : current));
    }
  }

  useEffect(() => {
    if (!repoId) return;
    loadRecallEvents();
    loadMemoryStatus();
    loadCandidates();
  }, [repoId]);

  const tabs = [
    { id: "recall", label: "召回记录" },
    { id: "search", label: "搜索" },
    { id: "status", label: "索引状态" },
    { id: "review", label: "待确认知识" }
  ] as const;
  const vectorRuntime = recordFromUnknown(memoryStatus?.vector);
  const vectorEmbedding = recordFromUnknown(vectorRuntime.embedding);
  const vectorBackend = memoryStatus
    ? `${String(vectorRuntime.backend ?? "未配置")} · ${String(vectorRuntime.status ?? "unknown")}`
    : "未加载";

  return (
    <div className="space-y-4">
      <div className="flex gap-1 overflow-x-auto rounded-md border border-line bg-white p-1">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            type="button"
            onClick={() => setActiveTab(tab.id)}
            className={`shrink-0 rounded-md px-3 py-1.5 text-xs font-medium transition ${
              activeTab === tab.id ? "bg-accent text-white" : "text-slate-600 hover:bg-panel"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {error && <div className="rounded-md border border-red-100 bg-red-50 p-3 text-sm text-red-700">{error}</div>}

      {activeTab === "recall" && (
        <section className="space-y-2">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-ink">召回记录</h3>
            <button type="button" onClick={loadRecallEvents} className="rounded-md border border-line px-2 py-1 text-xs text-slate-600 hover:bg-panel">
              刷新
            </button>
          </div>
          {loadingPanel === "recall" && <div className="rounded-md border border-line bg-white p-4 text-sm text-slate-500">正在读取召回记录...</div>}
          {!loadingPanel && recallEvents.length === 0 && <div className="rounded-md border border-line bg-white p-4 text-sm text-slate-500">还没有召回记录。智能体调用 search_evidence 或 RAG 搜索后会显示在这里。</div>}
          {recallEvents.map((event) => (
            <div key={event.id} className="rounded-md border border-line bg-white p-3">
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="text-xs font-medium text-accent">{event.tool_name}</div>
                  <div className="mt-1 line-clamp-2 text-sm font-semibold text-ink">{event.query || "空查询"}</div>
                  <div className="mt-1 text-xs text-slate-500">{event.result_count} 个结果 · {event.mode} · {formatDateTime(event.created_at)}</div>
                </div>
                <span className="rounded-md bg-panel px-2 py-1 text-[11px] text-slate-600">{event.scope}</span>
              </div>
              <div className="mt-2 space-y-1">
                {(event.results ?? []).slice(0, 3).map((item, index) => (
                  <div key={`${event.id}-${index}`} className="rounded-md bg-panel px-3 py-2 text-xs text-slate-600">
                    <span className="font-medium text-ink">{String(item.source_type || "source")}</span>
                    <span className="mx-1">·</span>
                    {String(item.title || "")}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </section>
      )}

      {activeTab === "search" && (
        <section className="space-y-4">
          <div className="rounded-md border border-line bg-white p-4">
            <h3 className="text-sm font-semibold text-ink">统一记忆搜索</h3>
            <div className="mt-3 flex gap-2">
              <input
                value={searchQuery}
                onChange={(event) => setSearchQuery(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") void runMemorySearch();
                }}
                placeholder="搜索 Issue、PR、CI、文档、会话记忆..."
                className="h-9 min-w-0 flex-1 rounded-md border border-line px-3 text-sm outline-none focus:border-teal-500 focus:ring-2 focus:ring-teal-100"
              />
              <button
                type="button"
                onClick={runMemorySearch}
                disabled={!searchQuery.trim() || loadingPanel === "search"}
                className="inline-flex h-9 items-center justify-center rounded-md bg-accent px-3 text-sm font-medium text-white transition hover:bg-teal-700 disabled:opacity-60"
              >
                搜索
              </button>
            </div>
            <div className="mt-3 space-y-2">
              {searchResults.map((item) => (
                <div key={`${item.source_type}-${item.source_id}-${item.title}`} className="rounded-md bg-panel p-3">
                  <div className="text-xs font-medium text-accent">{knowledgeTypeLabel(item.source_type)}</div>
                  <div className="mt-1 text-sm font-semibold text-ink">{item.title}</div>
                  <div className="mt-1 line-clamp-3 text-xs leading-5 text-slate-600">{item.snippet}</div>
                </div>
              ))}
            </div>
          </div>

          <section className="rounded-md border border-line bg-white p-4">
            <h3 className="text-sm font-semibold text-ink">项目记忆</h3>
            <p className="mt-1 text-xs leading-5 text-slate-500">记录项目约定、架构背景和长期偏好，会写入当前项目的 RAG 文档。</p>
            <div className="mt-4 grid gap-2">
              <input
                value={draft.title}
                onChange={(event) => onDraftChange({ ...draft, title: event.target.value })}
                placeholder="标题，例如 认证模块约定"
                className="h-9 rounded-md border border-line bg-white px-3 text-sm outline-none focus:border-teal-500 focus:ring-2 focus:ring-teal-100"
              />
              <textarea
                value={draft.content}
                onChange={(event) => onDraftChange({ ...draft, content: event.target.value })}
                placeholder="记忆内容，例如 登录相关 Issue 优先分给熟悉 JWT 和 FastAPI 的成员"
                className="min-h-24 resize-none rounded-md border border-line bg-white px-3 py-2 text-sm outline-none focus:border-teal-500 focus:ring-2 focus:ring-teal-100"
              />
              <button
                onClick={onAddNote}
                disabled={!draft.title.trim() && !draft.content.trim()}
                className="inline-flex h-9 items-center justify-center rounded-md bg-accent px-3 text-sm font-medium text-white transition hover:bg-teal-700 disabled:opacity-60"
              >
                保存记忆
              </button>
            </div>
          </section>

          <section className="rounded-md border border-line bg-white p-4">
            <div className="flex items-start justify-between gap-3">
              <div>
                <h3 className="text-sm font-semibold text-ink">知识库文件</h3>
                <p className="mt-1 text-xs leading-5 text-slate-500">上传文档后会切片写入向量索引，并参与统一记忆搜索。</p>
              </div>
              <FileText className="h-4 w-4 text-accent" />
            </div>
            <label className="mt-4 flex min-h-24 cursor-pointer flex-col items-center justify-center rounded-md border border-dashed border-teal-200 bg-teal-50/50 px-3 py-4 text-center transition hover:bg-teal-50">
              <Upload className="h-5 w-5 text-accent" />
              <span className="mt-2 text-sm font-medium text-ink">{uploading ? "上传并索引中..." : "上传文件到知识库"}</span>
              <span className="mt-1 text-xs text-slate-500">支持文本、Markdown、JSON、CSV、PDF、DOCX 等可抽取文本的文件</span>
              <input
                type="file"
                className="hidden"
                disabled={uploading}
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) onUploadFile(file);
                  event.currentTarget.value = "";
                }}
              />
            </label>
            <div className="mt-3 grid grid-cols-2 gap-2 text-xs text-slate-600">
              <div className="rounded-md bg-panel px-3 py-2">知识条目：{visibleKnowledgeItems.length}</div>
            </div>
          </section>

          <MemoryItemLists notes={notes} knowledgeItems={visibleKnowledgeItems} onRemoveNote={onRemoveNote} onRemoveKnowledgeItem={onRemoveKnowledgeItem} />
        </section>
      )}

      {activeTab === "status" && (
        <section className="space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-ink">索引与健康状态</h3>
            <button type="button" onClick={loadMemoryStatus} className="rounded-md border border-line px-2 py-1 text-xs text-slate-600 hover:bg-panel">
              刷新
            </button>
          </div>
          <div className="grid grid-cols-2 gap-2 text-xs text-slate-600">
            <MemoryStatusTile label="索引状态" value={String(memoryStatus?.index?.status ?? "missing")} />
            <MemoryStatusTile label="项目文档" value={`${numberFromUnknown(memoryStatus?.documents?.total)} 条`} />
            <MemoryStatusTile label="Evidence" value={`${numberFromUnknown(memoryStatus?.evidence?.total)} 条`} />
            <MemoryStatusTile label="召回记录" value={`${numberFromUnknown(memoryStatus?.recall?.total)} 次`} />
            <MemoryStatusTile label="待确认" value={`${numberFromUnknown(memoryStatus?.candidates?.pending)} 条`} />
            <MemoryStatusTile label="向量后端" value={vectorBackend} />
          </div>
          <div className="rounded-md border border-line bg-white p-3 text-xs leading-5 text-slate-600">
            <div>最后扫描：{String(memoryStatus?.index?.last_scan_at ?? "暂无")}</div>
            <div>Embedding Provider：{String(vectorEmbedding.provider ?? "未配置")}</div>
            <div>Embedding：{String(vectorEmbedding.model ?? "未配置")}</div>
            <div>维度：{String(vectorEmbedding.dimensions ?? "未配置")}</div>
            <div>Collection：{String(vectorRuntime.collection ?? "未配置")}</div>
            <div>状态说明：{String(vectorRuntime.error ?? vectorRuntime.status ?? "暂无")}</div>
          </div>
        </section>
      )}

      {activeTab === "review" && (
        <section className="space-y-2">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-ink">待确认知识</h3>
            <button type="button" onClick={() => loadCandidates()} className="rounded-md border border-line px-2 py-1 text-xs text-slate-600 hover:bg-panel">
              刷新
            </button>
          </div>
          {candidates.length === 0 && <div className="rounded-md border border-line bg-white p-4 text-sm text-slate-500">还没有待确认知识。新的会话总结会在这里沉淀为候选项。</div>}
          {candidates.map((candidate) => (
            <div key={candidate.id} className="rounded-md border border-line bg-white p-3">
              <div className="text-xs font-medium text-accent">{memoryCandidateKindLabel(candidate.kind)}</div>
              <div className="mt-1 text-sm font-semibold text-ink">{candidate.title}</div>
              <div className="mt-2 whitespace-pre-wrap text-xs leading-5 text-slate-600">{candidate.content}</div>
              <div className="mt-3 flex gap-2">
                <button
                  type="button"
                  disabled={loadingPanel === candidate.id}
                  onClick={() => approveCandidate(candidate.id)}
                  className="rounded-md bg-accent px-3 py-1.5 text-xs font-medium text-white hover:bg-teal-700 disabled:opacity-60"
                >
                  确认入库
                </button>
                <button
                  type="button"
                  disabled={loadingPanel === candidate.id}
                  onClick={() => rejectCandidate(candidate.id)}
                  className="rounded-md border border-line px-3 py-1.5 text-xs text-slate-600 hover:bg-panel disabled:opacity-60"
                >
                  拒绝
                </button>
              </div>
            </div>
          ))}
        </section>
      )}
    </div>
  );
}

/** 记忆面板的状态小格（如「候选数 / 已批准」这类统计） */
function MemoryStatusTile({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded-md border border-line bg-white px-3 py-2">
      <div className="text-[11px] text-slate-500">{label}</div>
      <div className="mt-1 font-semibold text-ink">{value}</div>
    </div>
  );
}

/** 记忆笔记与知识文件两个列表（带删除操作），供记忆类页面复用 */
function MemoryItemLists({
  notes,
  knowledgeItems,
  onRemoveNote,
  onRemoveKnowledgeItem
}: {
  notes: MemoryNote[];
  knowledgeItems: KnowledgeItem[];
  onRemoveNote: (noteId: string) => void;
  onRemoveKnowledgeItem: (item: KnowledgeItem) => void;
}) {
  return (
    <>
      <section className="space-y-2">
        {notes.length === 0 && <div className="rounded-md border border-line bg-white p-4 text-sm text-slate-500">还没有项目记忆。</div>}
        {notes.map((note) => (
          <div key={note.id} className="rounded-md border border-line bg-white p-3">
            <div className="flex items-start justify-between gap-3">
              <div className="font-medium text-ink">{note.title}</div>
              <button onClick={() => onRemoveNote(note.id)} className="rounded-md border border-line px-2 py-1 text-xs text-slate-500 hover:bg-panel">
                移除
              </button>
            </div>
            <div className="mt-2 whitespace-pre-wrap text-xs leading-6 text-slate-600">{note.content || "无内容"}</div>
          </div>
        ))}
      </section>

      <section className="space-y-2">
        {knowledgeItems.length === 0 && <div className="rounded-md border border-line bg-white p-4 text-sm text-slate-500">还没有上传文件或生成周报。</div>}
        {knowledgeItems.map((item) => (
          <div key={`${item.source_type}-${item.id}`} className="rounded-md border border-line bg-white p-3">
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <div className="line-clamp-1 font-medium text-ink">{item.title}</div>
                <div className="mt-1 text-xs text-slate-500">
                  {knowledgeTypeLabel(item.source_type)} · {item.chunk_count} 个切片
                </div>
              </div>
              <button onClick={() => onRemoveKnowledgeItem(item)} className="rounded-md border border-line px-2 py-1 text-xs text-slate-500 hover:bg-panel">
                移除
              </button>
            </div>
            <div className="mt-2 line-clamp-3 text-xs leading-6 text-slate-600">{item.content_preview || "无内容预览"}</div>
          </div>
        ))}
      </section>
    </>
  );
}

function numberFromUnknown(value: unknown) {
  return typeof value === "number" ? value : Number(value || 0) || 0;
}

function recordFromUnknown(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
}

/** 记忆面板的时间格式（精确到分钟，缺值时给 "-"） */
function formatDateTime(value?: string | null) {
  if (!value) return "暂无时间";
  const time = new Date(value);
  if (Number.isNaN(time.getTime())) return value;
  return time.toLocaleString();
}

function memoryCandidateKindLabel(kind: string) {
  const labels: Record<string, string> = {
    decision: "决策",
    fact: "事实",
    task: "任务",
    preference: "偏好",
    repo_context: "项目上下文"
  };
  return labels[kind] ?? kind;
}

/** 简版的记忆工作区（笔记 + 知识文件，不含索引提示与检索），比 MemoryHubWorkspace 更轻 */
function MemoryWorkspace({
  notes,
  draft,
  knowledgeItems,
  uploading,
  onDraftChange,
  onAddNote,
  onRemoveNote,
  onUploadFile,
  onRemoveKnowledgeItem
}: {
  notes: MemoryNote[];
  draft: MemoryNoteDraft;
  knowledgeItems: KnowledgeItem[];
  uploading: boolean;
  onDraftChange: (value: MemoryNoteDraft) => void;
  onAddNote: () => void;
  onRemoveNote: (noteId: string) => void;
  onUploadFile: (file: File) => void;
  onRemoveKnowledgeItem: (item: KnowledgeItem) => void;
}) {
  const visibleKnowledgeItems = knowledgeItems.filter(
    (item) => item.source_type !== "memory_note" && item.source_type !== "team_member"
  );

  return (
    <div className="space-y-4">
      <section className="rounded-md border border-line bg-white p-4">
        <h3 className="text-sm font-semibold text-ink">项目记忆</h3>
        <p className="mt-1 text-xs leading-5 text-slate-500">记录项目约定、架构背景和长期偏好，会写入当前项目独立的 RAG 文档。</p>
        <div className="mt-4 grid gap-2">
          <input
            value={draft.title}
            onChange={(event) => onDraftChange({ ...draft, title: event.target.value })}
            placeholder="标题，例如 认证模块约定"
            className="h-9 rounded-md border border-line bg-white px-3 text-sm outline-none focus:border-teal-500 focus:ring-2 focus:ring-teal-100"
          />
          <textarea
            value={draft.content}
            onChange={(event) => onDraftChange({ ...draft, content: event.target.value })}
            placeholder="记忆内容，例如 登录相关 Issue 优先分给熟悉 JWT 和 FastAPI 的成员"
            className="min-h-24 resize-none rounded-md border border-line bg-white px-3 py-2 text-sm outline-none focus:border-teal-500 focus:ring-2 focus:ring-teal-100"
          />
          <button
            onClick={onAddNote}
            disabled={!draft.title.trim() && !draft.content.trim()}
            className="inline-flex h-9 items-center justify-center rounded-md bg-accent px-3 text-sm font-medium text-white transition hover:bg-teal-700 disabled:opacity-60"
          >
            保存记忆
          </button>
        </div>
      </section>

      <section className="rounded-md border border-line bg-white p-4">
        <div className="flex items-start justify-between gap-3">
          <div>
            <h3 className="text-sm font-semibold text-ink">知识库文件</h3>
            <p className="mt-1 text-xs leading-5 text-slate-500">上传文档后会切片写入 Milvus，只在当前项目内检索。</p>
          </div>
          <FileText className="h-4 w-4 text-accent" />
        </div>
        <label className="mt-4 flex min-h-24 cursor-pointer flex-col items-center justify-center rounded-md border border-dashed border-teal-200 bg-teal-50/50 px-3 py-4 text-center transition hover:bg-teal-50">
          <Upload className="h-5 w-5 text-accent" />
          <span className="mt-2 text-sm font-medium text-ink">{uploading ? "上传并索引中..." : "上传文件到知识库"}</span>
          <span className="mt-1 text-xs text-slate-500">支持文本、Markdown、JSON、CSV、PDF、DOCX 等可抽取文本的文件</span>
          <input
            type="file"
            className="hidden"
            disabled={uploading}
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) onUploadFile(file);
              event.currentTarget.value = "";
            }}
          />
        </label>
        <div className="mt-3 grid grid-cols-2 gap-2 text-xs text-slate-600">
          <div className="rounded-md bg-panel px-3 py-2">知识条目：{visibleKnowledgeItems.length}</div>
        </div>
      </section>

      <section className="space-y-2">
        {notes.length === 0 && <div className="rounded-md border border-line bg-white p-4 text-sm text-slate-500">还没有项目记忆。</div>}
        {notes.map((note) => (
          <div key={note.id} className="rounded-md border border-line bg-white p-3">
            <div className="flex items-start justify-between gap-3">
              <div className="font-medium text-ink">{note.title}</div>
              <button onClick={() => onRemoveNote(note.id)} className="rounded-md border border-line px-2 py-1 text-xs text-slate-500 hover:bg-panel">
                移除
              </button>
            </div>
            <div className="mt-2 whitespace-pre-wrap text-xs leading-6 text-slate-600">{note.content || "无内容"}</div>
          </div>
        ))}
      </section>

      <section className="space-y-2">
        {visibleKnowledgeItems.length === 0 && <div className="rounded-md border border-line bg-white p-4 text-sm text-slate-500">还没有上传文件或生成周报。</div>}
        {visibleKnowledgeItems.map((item) => (
          <div key={`${item.source_type}-${item.id}`} className="rounded-md border border-line bg-white p-3">
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <div className="line-clamp-1 font-medium text-ink">{item.title}</div>
                <div className="mt-1 text-xs text-slate-500">
                  {knowledgeTypeLabel(item.source_type)} · {item.chunk_count} 个切片
                </div>
              </div>
              <button onClick={() => onRemoveKnowledgeItem(item)} className="rounded-md border border-line px-2 py-1 text-xs text-slate-500 hover:bg-panel">
                移除
              </button>
            </div>
            <div className="mt-2 line-clamp-3 text-xs leading-6 text-slate-600">{item.content_preview || "无内容预览"}</div>
          </div>
        ))}
      </section>
    </div>
  );
}

/**
 * 「图谱」页签：用环形布局把知识节点画成可交互的图。
 * 数据来自 GET /api/repos/{id}/knowledge/graph；筛选、显隐全部在本地完成（见文件底部的
 * filterKnowledgeGraph / layoutKnowledgeGraph），点节点可下钻或跳转到对应工作台条目。
 */
function KnowledgeGraphWorkspace({
  graph,
  loading,
  error,
  query,
  selectedNodeKey,
  hiddenNodeTypes,
  hiddenRelations,
  onQueryChange,
  onSelectNode,
  onToggleNodeType,
  onToggleRelation,
  onRefresh,
  onRebuild,
  onFocusNode,
  onOpenNode
}: {
  graph: KnowledgeGraphResponse | null;
  loading: boolean;
  error: string | null;
  query: string;
  selectedNodeKey: string | null;
  hiddenNodeTypes: Record<string, boolean>;
  hiddenRelations: Record<string, boolean>;
  onQueryChange: (value: string) => void;
  onSelectNode: (value: string) => void;
  onToggleNodeType: (type: string) => void;
  onToggleRelation: (relation: string) => void;
  onRefresh: () => void;
  onRebuild: () => void;
  onFocusNode: (node: KnowledgeGraphNode) => void;
  onOpenNode: (node: KnowledgeGraphNode) => void;
}) {
  const allNodes = graph?.nodes ?? [];
  const allEdges = graph?.edges ?? [];
  const nodeByKey = useMemo(() => new Map(allNodes.map((node) => [knowledgeGraphNodeKey(node), node])), [allNodes]);
  const nodeTypes = useMemo(() => [...new Set(allNodes.map((node) => node.type))], [allNodes]);
  const relations = useMemo(() => [...new Set(allEdges.map((edge) => edge.relation))], [allEdges]);
  const filtered = useMemo(
    () => filterKnowledgeGraph(allNodes, allEdges, query, hiddenNodeTypes, hiddenRelations),
    [allNodes, allEdges, query, hiddenNodeTypes, hiddenRelations]
  );
  const centerKey = graph?.center_type && graph.center_id ? `${graph.center_type}:${graph.center_id}` : undefined;
  const positions = useMemo(() => layoutKnowledgeGraph(filtered.nodes, filtered.edges, centerKey), [filtered.nodes, filtered.edges, centerKey]);
  const selectedNode =
    (selectedNodeKey && filtered.nodes.find((node) => knowledgeGraphNodeKey(node) === selectedNodeKey)) ?? filtered.nodes[0] ?? null;
  const selectedEdges = selectedNode
    ? filtered.edges.filter(
        (edge) =>
          (edge.from_type === selectedNode.type && edge.from_id === selectedNode.id) ||
          (edge.to_type === selectedNode.type && edge.to_id === selectedNode.id)
      )
    : [];
  const showEdgeLabels = filtered.edges.length <= 8;
  const visibleNodeCount = graphStatNumber(graph, "visible_nodes", allNodes.length);
  const visibleEdgeCount = graphStatNumber(graph, "visible_edges", allEdges.length);
  const totalEdgeCount = graphStatNumber(graph, "total_edges", allEdges.length);

  return (
    <div className="space-y-3">
      <section className="rounded-md border border-line bg-white p-4">
        <div className="flex items-start justify-between gap-3">
          <div>
            <div className="flex items-center gap-2">
              <Network className="h-4 w-4 text-accent" />
              <h3 className="text-sm font-semibold text-ink">项目知识图谱</h3>
            </div>
            <p className="mt-1 text-xs leading-5 text-slate-500">把 Issue、PR、CI、成员、文档、记忆和代码文件按可追溯关系串起来。</p>
          </div>
          <div className="flex shrink-0 gap-1">
            <button
              type="button"
              onClick={onRefresh}
              disabled={loading}
              title="刷新图谱"
              className="inline-flex h-8 w-8 items-center justify-center rounded-md border border-line text-slate-500 transition hover:bg-panel disabled:opacity-50"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
            </button>
            <button
              type="button"
              onClick={onRebuild}
              disabled={loading}
              title="重建关系边"
              className="inline-flex h-8 w-8 items-center justify-center rounded-md border border-line text-slate-500 transition hover:bg-panel disabled:opacity-50"
            >
              <Database className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>

        <div className="mt-3 grid grid-cols-3 gap-2 text-xs text-slate-600">
          <div className="rounded-md bg-panel px-3 py-2">节点 {visibleNodeCount}</div>
          <div className="rounded-md bg-panel px-3 py-2">关系 {visibleEdgeCount}</div>
          <div className="rounded-md bg-panel px-3 py-2">总边 {totalEdgeCount}</div>
        </div>

        <div className="mt-3 flex items-center gap-2 rounded-md border border-line bg-white px-3">
          <Search className="h-4 w-4 shrink-0 text-slate-400" />
          <input
            value={query}
            onChange={(event) => onQueryChange(event.target.value)}
            placeholder="搜索节点或关系..."
            className="h-9 min-w-0 flex-1 bg-transparent text-sm outline-none"
          />
          {query && (
            <button type="button" onClick={() => onQueryChange("")} className="text-slate-400 hover:text-slate-600" title="清空搜索">
              <X className="h-3.5 w-3.5" />
            </button>
          )}
        </div>

        {Boolean(nodeTypes.length) && (
          <div className="mt-3 flex flex-wrap gap-1.5">
            {nodeTypes.map((type) => (
              <button
                key={type}
                type="button"
                onClick={() => onToggleNodeType(type)}
                className={`rounded-md border px-2 py-1 text-[11px] font-medium transition ${
                  hiddenNodeTypes[type] ? "border-line bg-white text-slate-400" : "border-transparent bg-panel text-slate-700"
                }`}
              >
                {graphNodeTypeLabel(type)}
              </button>
            ))}
          </div>
        )}

        {Boolean(relations.length) && (
          <div className="mt-2 flex flex-wrap gap-1.5">
            {relations.map((relation) => (
              <button
                key={relation}
                type="button"
                onClick={() => onToggleRelation(relation)}
                className={`rounded-md border px-2 py-1 text-[11px] font-medium transition ${
                  hiddenRelations[relation] ? "border-line bg-white text-slate-400" : "border-transparent bg-panel text-slate-700"
                }`}
              >
                {graphRelationLabel(relation)}
              </button>
            ))}
          </div>
        )}
      </section>

      {error && <div className="rounded-md border border-red-100 bg-red-50 p-3 text-sm text-red-700">{error}</div>}

      {!error && !loading && (!graph || graph.nodes.length === 0) && (
        <div className="rounded-md border border-line bg-white p-4 text-sm text-slate-500">还没有可展示的项目关系。同步 Issue/PR/CI，或添加项目记忆后再重建图谱。</div>
      )}

      {loading && !graph && <div className="rounded-md border border-line bg-white p-4 text-sm text-slate-500">正在加载项目知识图谱...</div>}

      {graph && filtered.nodes.length === 0 && (
        <div className="rounded-md border border-line bg-white p-4 text-sm text-slate-500">当前搜索或过滤条件下没有匹配节点。</div>
      )}

      {graph && filtered.nodes.length > 0 && (
        <section className="rounded-md border border-line bg-white p-3">
          <div className="overflow-x-auto rounded-md bg-panel">
            <svg viewBox="0 0 720 420" className="h-[360px] min-w-[680px] w-full" role="img" aria-label="项目知识图谱">
              {filtered.edges.map((edge) => {
                const from = positions.get(`${edge.from_type}:${edge.from_id}`);
                const to = positions.get(`${edge.to_type}:${edge.to_id}`);
                if (!from || !to) return null;
                const color = graphRelationColor(edge.relation);
                return (
                  <g key={edge.id}>
                    <line x1={from.x} y1={from.y} x2={to.x} y2={to.y} stroke={color} strokeWidth={edge.relation === "used_as_evidence" ? 2.8 : 1.6} strokeOpacity="0.58" />
                    {showEdgeLabels && (
                      <text x={(from.x + to.x) / 2} y={(from.y + to.y) / 2 - 5} textAnchor="middle" fontSize="10" fill={color} fontWeight="700">
                        {graphRelationLabel(edge.relation)}
                      </text>
                    )}
                  </g>
                );
              })}
              {filtered.nodes.map((node) => {
                const pos = positions.get(knowledgeGraphNodeKey(node));
                if (!pos) return null;
                const selected = selectedNode ? knowledgeGraphNodeKey(selectedNode) === knowledgeGraphNodeKey(node) : false;
                const style = graphNodeStyle(node.type);
                return (
                  <g key={knowledgeGraphNodeKey(node)} role="button" tabIndex={0} onClick={() => onSelectNode(knowledgeGraphNodeKey(node))} className="cursor-pointer">
                    <circle
                      cx={pos.x}
                      cy={pos.y}
                      r={selected ? 19 : 15}
                      fill={node.highlighted ? "#fff7ed" : style.fill}
                      stroke={node.highlighted ? "#f97316" : selected ? "#111827" : style.stroke}
                      strokeWidth={node.highlighted || selected ? 2.6 : 1.6}
                    />
                    <text x={pos.x} y={pos.y + 34} textAnchor="middle" fontSize="11" fontWeight={selected ? "700" : "600"} fill="#334155">
                      {clipGraphLabel(node.title, 18)}
                    </text>
                    <text x={pos.x} y={pos.y + 48} textAnchor="middle" fontSize="9" fill="#64748b">
                      {graphNodeTypeLabel(node.type)}
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>

          {selectedNode && (
            <div className="mt-3 rounded-md border border-line bg-white p-3">
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="text-xs font-medium text-accent">{graphNodeTypeLabel(selectedNode.type)}</div>
                  <div className="mt-1 line-clamp-2 text-sm font-semibold text-ink">{selectedNode.title}</div>
                  {selectedNode.subtitle && <div className="mt-1 text-xs text-slate-500">{selectedNode.subtitle}</div>}
                </div>
                <div className="flex shrink-0 gap-1">
                  <button
                    type="button"
                    onClick={() => onFocusNode(selectedNode)}
                    className="rounded-md border border-line px-2 py-1 text-xs text-slate-600 transition hover:bg-panel"
                  >
                    聚焦
                  </button>
                  <button
                    type="button"
                    onClick={() => onOpenNode(selectedNode)}
                    className="rounded-md bg-accent px-2 py-1 text-xs font-medium text-white transition hover:bg-teal-700"
                  >
                    打开
                  </button>
                </div>
              </div>

              <div className="mt-3 space-y-1.5">
                {selectedEdges.length === 0 && <div className="text-xs text-slate-500">这个节点暂时没有可见关系。</div>}
                {selectedEdges.slice(0, 8).map((edge) => {
                  const isOutgoing = edge.from_type === selectedNode.type && edge.from_id === selectedNode.id;
                  const other = nodeByKey.get(isOutgoing ? `${edge.to_type}:${edge.to_id}` : `${edge.from_type}:${edge.from_id}`);
                  return (
                    <div key={edge.id} className="rounded-md bg-panel px-3 py-2 text-xs text-slate-600">
                      <span className="font-medium text-ink">{isOutgoing ? "→" : "←"} {graphRelationLabel(edge.relation)}</span>
                      <span className="ml-2">{other?.title ?? edge.to_id}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </section>
      )}
    </div>
  );
}

/* ============ 知识图谱工具函数 ============ */

/** 知识来源类型 → 中文标签；未知类型原样返回，避免丢失信息 */
function knowledgeTypeLabel(sourceType: string) {
  if (sourceType === "knowledge_file") return "上传文件";
  if (sourceType === "weekly_report") return "周报";
  if (sourceType === "project_overview") return "项目概况";
  if (sourceType === "project_doc") return "项目文档";
  if (sourceType === "project_manifest") return "项目清单";
  if (sourceType === "memory_note") return "记忆";
  if (sourceType === "team_member") return "团队成员";
  return sourceType;
}

/** 节点唯一键：type + id，因为不同实体类型的 id 可能重复 */
function knowledgeGraphNodeKey(node: KnowledgeGraphNode) {
  return `${node.type}:${node.id}`;
}

/**
 * 右栏选中的条目 → 图谱中心节点。
 * 两边的类型命名不同（pr ↔ pull_request、ci ↔ ci_run），这里做一次映射；
 * 其它类型没有对应节点，返回 null 表示不做中心聚焦。
 */
function workspaceSelectionToGraphCenter(selection: WorkspaceSelection): { type: string; id: string } | null {
  if (selection.type === "issue") return { type: "issue", id: selection.id };
  if (selection.type === "pr") return { type: "pull_request", id: selection.id };
  if (selection.type === "ci") return { type: "ci_run", id: selection.id };
  return null;
}

/**
 * 点击图谱节点后跳转到对应工作台：
 * Issue / PR / CI 会切页签并选中该条目；成员与知识类节点只能切到对应页签（没有单条详情）。
 * preserveSelectionRef 置 true 让切页签时的状态重置逻辑保留这次选中项。
 */
function openGraphNodeInWorkspace(
  node: KnowledgeGraphNode,
  setWorkspaceTab: (tab: WorkspaceTab) => void,
  setSelectedWorkspaceItem: (item: WorkspaceSelection | null) => void,
  preserveSelectionRef: { current: boolean }
) {
  if (node.type === "issue") {
    preserveSelectionRef.current = true;
    setWorkspaceTab("Issue");
    setSelectedWorkspaceItem({ type: "issue", id: node.id });
    return;
  }
  if (node.type === "pull_request") {
    preserveSelectionRef.current = true;
    setWorkspaceTab("PR");
    setSelectedWorkspaceItem({ type: "pr", id: node.id });
    return;
  }
  if (node.type === "ci_run") {
    preserveSelectionRef.current = true;
    setWorkspaceTab("CI");
    setSelectedWorkspaceItem({ type: "ci", id: node.id });
    return;
  }
  setWorkspaceTab(node.type === "team_member" ? "团队" : "记忆&知识库");
  setSelectedWorkspaceItem(null);
}

/**
 * 图谱本地过滤：先按「隐藏的节点类型」过滤，再按关键词筛节点。
 * 有关键词时做一次邻居扩散：命中节点的邻居、以及关系名命中的边两端也会保留，
 * 这样搜到的实体不会孤零零地悬在图上，仍能看到它的上下文。
 * 最后只保留两端都可见的边，避免出现指向隐藏节点的悬空连线。
 */
function filterKnowledgeGraph(
  nodes: KnowledgeGraphNode[],
  edges: KnowledgeGraphEdge[],
  query: string,
  hiddenNodeTypes: Record<string, boolean>,
  hiddenRelations: Record<string, boolean>
) {
  const normalizedQuery = query.trim().toLowerCase();
  const baseAllowed = new Set(
    nodes.filter((node) => !hiddenNodeTypes[node.type]).map((node) => knowledgeGraphNodeKey(node))
  );
  const matches = new Set(
    nodes
      .filter((node) => baseAllowed.has(knowledgeGraphNodeKey(node)))
      .filter((node) => !normalizedQuery || graphNodeMatches(node, normalizedQuery))
      .map((node) => knowledgeGraphNodeKey(node))
  );
  if (normalizedQuery) {
    for (const edge of edges) {
      if (hiddenRelations[edge.relation]) continue;
      const fromKey = `${edge.from_type}:${edge.from_id}`;
      const toKey = `${edge.to_type}:${edge.to_id}`;
      const relationMatches = graphRelationLabel(edge.relation).toLowerCase().includes(normalizedQuery) || edge.relation.toLowerCase().includes(normalizedQuery);
      if (relationMatches || matches.has(fromKey) || matches.has(toKey)) {
        if (baseAllowed.has(fromKey)) matches.add(fromKey);
        if (baseAllowed.has(toKey)) matches.add(toKey);
      }
    }
  }
  const allowed = normalizedQuery ? matches : baseAllowed;
  const visibleEdges = edges.filter((edge) => {
    if (hiddenRelations[edge.relation]) return false;
    return allowed.has(`${edge.from_type}:${edge.from_id}`) && allowed.has(`${edge.to_type}:${edge.to_id}`);
  });
  return {
    nodes: nodes.filter((node) => allowed.has(knowledgeGraphNodeKey(node))),
    edges: visibleEdges
  };
}

/** 单个节点是否命中关键词：标题 + 副标题 + 类型 + 前 6 个 metadata 值一起匹配 */
function graphNodeMatches(node: KnowledgeGraphNode, query: string) {
  const metadataText = Object.values(node.metadata ?? {})
    .slice(0, 6)
    .join(" ")
    .toLowerCase();
  return `${node.title} ${node.subtitle ?? ""} ${node.type} ${metadataText}`.toLowerCase().includes(query);
}

/**
 * 计算图谱节点坐标（纯前端布局，不引第三方图库）：
 * 中心节点放在画布中央，它的一级邻居排内环，其余节点排外环。
 * 没指定中心时取「度数最高」的节点作中心——它通常是最有信息量的实体。
 */
function layoutKnowledgeGraph(nodes: KnowledgeGraphNode[], edges: KnowledgeGraphEdge[], centerKey?: string) {
  const positions = new Map<string, { x: number; y: number }>();
  if (nodes.length === 0) return positions;
  const degrees = new Map<string, number>();
  for (const node of nodes) degrees.set(knowledgeGraphNodeKey(node), 0);
  for (const edge of edges) {
    const fromKey = `${edge.from_type}:${edge.from_id}`;
    const toKey = `${edge.to_type}:${edge.to_id}`;
    degrees.set(fromKey, (degrees.get(fromKey) ?? 0) + 1);
    degrees.set(toKey, (degrees.get(toKey) ?? 0) + 1);
  }
  const requestedCenter = centerKey ? nodes.find((node) => knowledgeGraphNodeKey(node) === centerKey) : undefined;
  const resolvedCenter =
    requestedCenter ??
    [...nodes].sort((a, b) => (degrees.get(knowledgeGraphNodeKey(b)) ?? 0) - (degrees.get(knowledgeGraphNodeKey(a)) ?? 0))[0];
  const centerNodeKey = knowledgeGraphNodeKey(resolvedCenter);
  positions.set(centerNodeKey, { x: 360, y: 190 });
  const neighborKeys = new Set<string>();
  for (const edge of edges) {
    const fromKey = `${edge.from_type}:${edge.from_id}`;
    const toKey = `${edge.to_type}:${edge.to_id}`;
    if (fromKey === centerNodeKey) neighborKeys.add(toKey);
    if (toKey === centerNodeKey) neighborKeys.add(fromKey);
  }
  const neighbors = nodes.filter((node) => neighborKeys.has(knowledgeGraphNodeKey(node)));
  const remaining = nodes.filter((node) => knowledgeGraphNodeKey(node) !== centerNodeKey && !neighborKeys.has(knowledgeGraphNodeKey(node)));
  placeGraphRing(positions, neighbors, 360, 190, neighbors.length > 10 ? 145 : 122, -Math.PI / 2);
  placeGraphRing(positions, remaining, 360, 190, 184, Math.PI / 9);
  return positions;
}

/**
 * 把一组节点均匀摆到一个圆环上（等角度分布）。
 * @param start 起始角度，用于错开内外环的重叠位置
 */
function placeGraphRing(
  positions: Map<string, { x: number; y: number }>,
  nodes: KnowledgeGraphNode[],
  cx: number,
  cy: number,
  radius: number,
  start: number
) {
  if (nodes.length === 0) return;
  nodes.forEach((node, index) => {
    const angle = start + (Math.PI * 2 * index) / nodes.length;
    positions.set(knowledgeGraphNodeKey(node), {
      x: cx + Math.cos(angle) * radius,
      y: cy + Math.sin(angle) * radius
    });
  });
}

/** 节点类型 → 中文标签（用于图例与筛选） */
function graphNodeTypeLabel(type: string) {
  const labels: Record<string, string> = {
    issue: "Issue",
    pull_request: "PR",
    ci_run: "CI",
    team_member: "成员",
    memory_note: "记忆",
    document: "文档",
    code_file: "代码",
    session_decision: "决策"
  };
  return labels[type] ?? type;
}

/** 关系类型 → 中文；未知关系把下划线换成空格，至少保持可读 */
function graphRelationLabel(relation: string) {
  const labels: Record<string, string> = {
    resolves: "修复",
    mentions: "提及",
    assigned_to: "分配",
    blocked_by: "阻塞",
    failed_in_ci: "CI失败",
    touches_file: "改动",
    documents: "说明",
    cites: "引用",
    produced_from_session: "沉淀",
    used_as_evidence: "召回"
  };
  return labels[relation] ?? relation.replace(/_/g, " ");
}

/** 节点配色：按类型区分填充/描边，保证同一张图里类型一眼可辨；未知类型用中性灰 */
function graphNodeStyle(type: string) {
  const styles: Record<string, { fill: string; stroke: string }> = {
    issue: { fill: "#fef3c7", stroke: "#d97706" },
    pull_request: { fill: "#ccfbf1", stroke: "#0f766e" },
    ci_run: { fill: "#fee2e2", stroke: "#dc2626" },
    team_member: { fill: "#e0e7ff", stroke: "#4f46e5" },
    memory_note: { fill: "#fce7f3", stroke: "#be185d" },
    document: { fill: "#dbeafe", stroke: "#2563eb" },
    code_file: { fill: "#dcfce7", stroke: "#16a34a" },
    session_decision: { fill: "#f3e8ff", stroke: "#9333ea" }
  };
  return styles[type] ?? { fill: "#f1f5f9", stroke: "#64748b" };
}

/** 连线颜色：与关系语义对应（如 blocked_by 用红色、resolves 用绿色） */
function graphRelationColor(relation: string) {
  const colors: Record<string, string> = {
    resolves: "#16a34a",
    mentions: "#64748b",
    assigned_to: "#4f46e5",
    blocked_by: "#dc2626",
    failed_in_ci: "#ef4444",
    touches_file: "#0f766e",
    documents: "#2563eb",
    cites: "#be185d",
    produced_from_session: "#9333ea",
    used_as_evidence: "#f97316"
  };
  return colors[relation] ?? "#64748b";
}

/** 读取图谱统计数值，缺失或类型不符时用 fallback 兜底 */
function graphStatNumber(graph: KnowledgeGraphResponse | null, key: string, fallback: number) {
  const value = graph?.stats?.[key];
  return typeof value === "number" ? value : fallback;
}

/** 截断图上的长标签并加省略号（否则节点文字会溢出） */
function clipGraphLabel(text: string, limit: number) {
  return text.length <= limit ? text : `${text.slice(0, limit - 1).trim()}…`;
}

/**
 * Issue 归类（决定右栏分组展示）：按标签 > 状态 > 信息的优先级判断。
 * 优先级这样排是因为标签最能表达人的意图，而「没人负责且没正文」几乎等同于待认领。
 */
function classifyIssue(issue: Issue) {
  const labels = (issue.labels ?? []).map((label) => label.toLowerCase());
  const text = `${issue.title} ${issue.body ?? ""}`.toLowerCase();
  if (labels.some((label) => ["rejected", "wontfix", "invalid", "duplicate"].includes(label)) || text.includes("reject")) return "rejected";
  if (labels.some((label) => ["accepted", "resolved", "done", "fixed"].includes(label))) return "accepted";
  if (issue.state === "closed") return "closed";
  if (labels.some((label) => ["decision", "needs-decision", "rfc", "proposal"].includes(label))) return "decision";
  if ((issue.assignees?.length ?? 0) === 0 && !issue.body) return "unanswered";
  if (labels.length === 0 && (issue.assignees?.length ?? 0) === 0) return "unanswered";
  return "discussing";
}

/**
 * PR 归类：关闭但未合并视为被拒绝；diff 里出现 conflict 字样视为有冲突；
 * 已合并为完成；没有作者则提示需要负责人，其余归为评审中。
 */
function classifyPr(pr: PullRequest) {
  const text = `${pr.title} ${pr.body ?? ""} ${(pr.files ?? []).map((file) => file.patch ?? file.filename).join(" ")}`.toLowerCase();
  if (pr.state === "closed" && !pr.merged_at) return "rejected";
  if (text.includes("conflict")) return "conflict";
  if (pr.merged_at) return "done";
  if (!pr.author) return "needsOwner";
  return "reviewing";
}

/** 关键词匹配：任意字段包含即命中；空查询视为全部命中 */
function matchesQuery(query: string, values: Array<string | null | undefined>) {
  const normalizedQuery = query.trim().toLowerCase();
  if (!normalizedQuery) return true;
  return values.some((value) => value?.toLowerCase().includes(normalizedQuery));
}

/** 状态筛选：all 表示不过滤，否则精确相等 */
function matchesStatus(filter: string, value?: string | null) {
  if (filter === "all") return true;
  return value === filter;
}

/** 负责人筛选：all 不过滤；unassigned 只看无人负责的；其余按名字精确匹配 */
function matchesOwner(filter: string, people: string[]) {
  const normalizedPeople = people.filter(Boolean);
  if (filter === "all") return true;
  if (filter === "unassigned") return normalizedPeople.length === 0;
  return normalizedPeople.includes(filter);
}

/**
 * 时间筛选：7d / 30d 内更新过。
 * 缺失或非法时间一律视为命中——宁可多显示，也不要因为脏数据把条目藏起来。
 */
function matchesTime(filter: string, value?: string | null) {
  if (filter === "all" || !value) return true;
  const timestamp = new Date(value).getTime();
  if (Number.isNaN(timestamp)) return true;
  const days = filter === "7d" ? 7 : 30;
  return Date.now() - timestamp <= days * 24 * 60 * 60 * 1000;
}

/** 右栏详情用的日期格式（月-日 时:分）；无值时显示「无」而不是空白 */
function formatDate(value?: string | null) {
  if (!value) return "无";
  return new Date(value).toLocaleDateString("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit"
  });
}

/**
 * 解析用户输入的仓库地址，支持三种写法：
 * 1) 完整 URL（https://github.com/owner/repo）——取 pathname；
 * 2) SSH 形式（git@github.com:owner/repo.git）——取冒号后的部分；
 * 3) 简写 owner/repo——直接按斜杠拆分。
 * 解析不出 owner/repo 时返回 null，由调用方给出错误提示。
 */
function parseRepositoryInput(value: string): { owner: string; repo: string } | null {
  const input = value.trim();
  if (!input) return null;

  let path = input;
  const sshMatch = input.match(/^git@[^:]+:(.+)$/);
  if (sshMatch) {
    path = sshMatch[1];
  } else {
    try {
      const url = new URL(input);
      path = url.pathname;
    } catch {
      path = input;
    }
  }

  const parts = path
    .replace(/^\/+/, "")
    .split("/")
    .map((item) => item.trim())
    .filter(Boolean);
  if (parts.length < 2) return null;

  const owner = parts[0];
  const repo = parts[1].replace(/\.git$/i, "");
  if (!owner || !repo) return null;
  return { owner, repo };
}

/**
 * 把技术错误翻译成用户能懂的话：
 * - 「Failed to fetch」几乎总是后端没起或 CORS 未放行，直接给出可操作提示；
 * - 后端把错误包在 JSON 字符串里时，抽出 detail / message；
 * - 其余原样返回，保证不吞掉信息。
 */
function getFriendlyError(error: unknown) {
  const raw = error instanceof Error ? error.message : String(error);
  if (raw === "Failed to fetch") {
    return "无法访问后端 API。请确认后端服务已启动，并且 CORS 允许当前前端端口。";
  }
  try {
    const parsed = JSON.parse(raw) as { detail?: unknown; message?: unknown };
    if (typeof parsed.detail === "string") return parsed.detail;
    if (typeof parsed.message === "string") return parsed.message;
  } catch {
    // 下面回退到原始错误消息。
  }
  return raw || "未知错误";
}
