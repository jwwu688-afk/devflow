"use client";
/*
 * RAG 工作室（路由 /rag）——面向单个项目的「资料入库 + 检索问答」工作台。
 *
 * 三栏布局：
 * - 左栏：知识库条目列表（上传资料、重建向量索引、删除资料）；
 * - 中栏：问答会话（提问后由后端检索证据并生成回答，回答支持 Markdown 渲染）；
 * - 右栏：检索证据（引用编号、来源、相似度分数、片段）。
 * 顶部指标条展示资料数、切片数、embedding 配置、向量库状态与生成模式。
 * 数据来源：
 * - GET    /api/rag/repositories                          项目（知识库）列表
 * - GET    /api/repos/{repoId}/knowledge                  知识库条目
 * - GET    /api/rag/{repoId}/status                       运行状态（embedding/向量库/生成器）
 * - POST   /api/rag/repositories/{repoId}/documents        上传资料（multipart，异步解析入库）
 * - DELETE /api/rag/repositories/{repoId}/documents/{id}   删除条目及其切片
 * - POST   /api/rag/{repoId}/reindex                      重建向量索引
 * - POST   /api/rag/{repoId}/ask                          提问（检索 + 生成）
 * - POST   /api/rag/{repoId}/search                       只检索证据、不生成
 * 交互状态：repoId、turns（对话轮次）、sources（证据）、busy（当前进行中的操作标识）、error/notice。
 */

import {
  BookOpen,
  Database,
  FileText,
  LoaderCircle,
  RefreshCw,
  Search,
  Send,
  Sparkles,
  Trash2,
  Upload,
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";

import { apiDelete, apiGet, apiPost, apiUpload } from "@/lib/api";
import type { Repository } from "@/lib/types";

// 知识库条目（后端 /api/repos/{id}/knowledge 的返回项）
type KnowledgeItem = {
  id: string;
  source_type: string;
  title: string;
  content_preview: string;
  metadata: Record<string, unknown>;
  chunk_count: number;
  created_at?: string | null;
};

// 一条检索证据：citation 是回答里 [n] 的编号，score 是相似度
type RAGSource = {
  citation: number;
  document_id: string;
  source_id: string;
  source_type: string;
  title: string;
  snippet: string;
  score: number;
  metadata: Record<string, unknown>;
  retrieval: Record<string, unknown>;
};

// /ask 的响应：generation_mode 说明是 LLM 生成、纯证据抽取还是证据不足
type RAGAnswer = {
  question: string;
  answer: string;
  sources: RAGSource[];
  retrieval_count: number;
  generation_mode: "llm" | "extractive" | "no_evidence";
  model?: string | null;
  took_ms: number;
};

// /search 的响应：只返回证据，不含生成的回答
type RAGSearch = {
  query: string;
  results: RAGSource[];
  result_count: number;
  took_ms: number;
};

// /status 的响应：用于顶部指标条判断「向量库是否降级」「有没有配置 LLM」
type RAGStatus = {
  repo_id: string;
  document_count: number;
  source_count: number;
  source_types: Record<string, number>;
  embedding: { provider?: string; model?: string; dimensions?: number };
  vector_store: { backend?: string; status?: string; available?: boolean; error?: string | null; fallback?: string };
  generation: { llm_configured?: boolean; model?: string | null; fallback?: string };
  supported_files: string[];
};

// 一轮对话：只在前端内存里保存，切换仓库或清空会话即消失
type ChatTurn = {
  id: string;
  role: "user" | "assistant";
  content: string;
  mode?: RAGAnswer["generation_mode"];
  tookMs?: number;
  sourceCount?: number;
};

// 证据来源类型 → 中文标签，用于列表与筛选下拉
const sourceTypeLabels: Record<string, string> = {
  knowledge_file: "文档",
  memory_note: "笔记",
  project_doc: "项目文档",
  project_overview: "项目概览",
  issue: "Issue",
  pull_request: "PR",
  workflow_run: "CI",
  team_member: "成员",
  weekly_report: "周报",
};

/** 未知来源类型降级为把下划线换成空格，保证界面上不出现裸的英文标识 */
function sourceLabel(value: string) {
  return sourceTypeLabels[value] ?? value.replaceAll("_", " ");
}

/** 时间展示：只到分钟；非法日期返回空串而不是 "Invalid Date" */
function formatDate(value?: string | null) {
  if (!value) return "";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "" : new Intl.DateTimeFormat("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" }).format(date);
}

/** 统一把 unknown 异常转成可展示的字符串 */
function errorMessage(error: unknown) {
  return error instanceof Error ? error.message : String(error);
}

/**
 * RAG 工作室页面组件。
 * @sideEffect 挂载加载项目列表；切换仓库重新加载知识库与状态；对话结束后自动滚动到底部。
 */
export default function RAGPage() {
  const [repos, setRepos] = useState<Repository[]>([]);
  const [repoId, setRepoId] = useState("");
  const [knowledge, setKnowledge] = useState<KnowledgeItem[]>([]);
  const [status, setStatus] = useState<RAGStatus | null>(null);
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [sources, setSources] = useState<RAGSource[]>([]);
  const [question, setQuestion] = useState("");
  const [topK, setTopK] = useState(5);
  // 空字符串表示不做来源类型过滤
  const [sourceType, setSourceType] = useState("");
  // busy 用具体操作名（upload / ask / search / reindex / delete:id）而不是布尔，
  // 这样能精确地只禁用正在执行的那个按钮，其余按钮保持可用。
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  // 对话容器引用：新消息或加载中时自动滚到底部
  const qaScrollRef = useRef<HTMLDivElement>(null);

  const currentRepo = useMemo(() => repos.find((repo) => repo.id === repoId) ?? null, [repos, repoId]);
  // 只有状态里出现过的来源类型才可作为筛选项，避免下拉出现空数据选项
  const availableSourceTypes = useMemo(() => Object.keys(status?.source_types ?? {}).sort(), [status]);

  /**
   * 加载项目列表并决定选中项，优先级：显式指定 → 当前选中（仍存在）→ 第一个。
   * 这样刷新列表不会把用户正在看的项目切走。
   */
  async function loadRepos(preferredId?: string) {
    const items = await apiGet<Repository[]>("/api/rag/repositories");
    setRepos(items);
    const nextId = preferredId && items.some((item) => item.id === preferredId) ? preferredId : repoId && items.some((item) => item.id === repoId) ? repoId : items[0]?.id ?? "";
    setRepoId(nextId);
  }

  /**
   * 加载某仓库的知识库条目与运行状态。
   * 两个接口并行：它们互不依赖，串行会拉长首屏时间。
   */
  async function loadKnowledge(id: string) {
    const [nextKnowledge, nextStatus] = await Promise.all([
      apiGet<KnowledgeItem[]>(`/api/repos/${id}/knowledge`),
      apiGet<RAGStatus>(`/api/rag/${id}/status`),
    ]);
    setKnowledge(nextKnowledge);
    setStatus(nextStatus);
  }

  // 首屏：项目列表；失败直接展示错误条
  useEffect(() => {
    loadRepos().catch((loadError) => setError(errorMessage(loadError)));
  }, []);

  // repoId 变化（含首次赋值）后加载知识库与状态；没有选中仓库时清空，避免展示上一个项目的数据。
  useEffect(() => {
    if (!repoId) {
      setKnowledge([]);
      setStatus(null);
      return;
    }
    setError(null);
    loadKnowledge(repoId).catch((loadError) => setError(errorMessage(loadError)));
  }, [repoId]);

  // 会话有更新或进入加载态时把对话框滚到底部，保证用户能看到最新一条消息
  useEffect(() => {
    const node = qaScrollRef.current;
    if (!node) return;
    node.scrollTo({ top: node.scrollHeight, behavior: "smooth" });
  }, [turns, busy]);

  /**
   * 上传单个资料文件。
   * 后端是异步流水线（解析 → 分段 → 向量化），接口会很快返回，
   * 所以这里等 900ms 再刷新一次，尽量让用户看到「排队/解析中」的条目，而不是空列表。
   */
  async function uploadFile(file?: File) {
    if (!file || !repoId) return;
    setBusy("upload");
    setError(null);
    setNotice(null);
    try {
      const body = new FormData();
      body.append("file", file);
      await apiUpload(`/api/rag/repositories/${repoId}/documents`, body);
      setNotice("文件已提交，正在解析、分段并建立索引");
      await new Promise((resolve) => window.setTimeout(resolve, 900));
      await loadKnowledge(repoId);
    } catch (uploadError) {
      setError(errorMessage(uploadError));
    } finally {
      setBusy(null);
      // 清空 input，允许重新选择同一个文件
      if (fileInput.current) fileInput.current.value = "";
    }
  }

  /**
   * 删除知识库条目（连同其切片），删除前二次确认。
   * busy 精确到条目 id，避免一条删除时锁住其他条目的删除按钮。
   */
  async function removeKnowledge(item: KnowledgeItem) {
    if (!repoId || !window.confirm(`删除“${item.title}”及其全部切片？`)) return;
    setBusy(`delete:${item.id}`);
    setError(null);
    try {
      await apiDelete(`/api/rag/repositories/${repoId}/documents/${item.id}`);
      await loadKnowledge(repoId);
      setNotice("资料已从索引中删除");
    } catch (deleteError) {
      setError(errorMessage(deleteError));
    } finally {
      setBusy(null);
    }
  }

  /**
   * 重建整个仓库的向量索引（换了 embedding 模型或索引损坏时使用）。
   * 返回重建的文档数用于提示。
   */
  async function reindex() {
    if (!repoId) return;
    setBusy("reindex");
    setError(null);
    try {
      const result = await apiPost<{ indexed_documents: number }>(`/api/rag/${repoId}/reindex`, {});
      await loadKnowledge(repoId);
      setNotice(`已重建 ${result.indexed_documents} 个文档向量`);
    } catch (reindexError) {
      setError(errorMessage(reindexError));
    } finally {
      setBusy(null);
    }
  }

  // 提问与只检索共用同一份检索参数，抽成函数避免两处字段不一致
  function queryBody() {
    return { question: question.trim(), top_k: topK, source_types: sourceType ? [sourceType] : [] };
  }

  /**
   * 提问：检索证据并由后端生成回答。
   * 流程上先插入用户消息并清空输入框（乐观更新，交互更跟手），
   * 成功后再追加 assistant 消息；失败也追加一条提示消息，让对话保持完整。
   * @sideEffect 写入 turns / sources。
   */
  async function ask() {
    const prompt = question.trim();
    if (!repoId || !prompt || busy) return;
    setTurns((current) => [...current, { id: crypto.randomUUID(), role: "user", content: prompt }]);
    setQuestion("");
    setBusy("ask");
    setError(null);
    try {
      const result = await apiPost<RAGAnswer>(`/api/rag/${repoId}/ask`, { ...queryBody(), question: prompt });
      setSources(result.sources);
      setTurns((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: result.answer,
          mode: result.generation_mode,
          tookMs: result.took_ms,
          sourceCount: result.retrieval_count,
        },
      ]);
    } catch (askError) {
      setError(errorMessage(askError));
      // 失败也要落一条 assistant 消息，否则界面上只剩用户提问、看起来像没响应
      setTurns((current) => [...current, { id: crypto.randomUUID(), role: "assistant", content: "请求失败，请检查后端配置后重试。", mode: "no_evidence" }]);
    } finally {
      setBusy(null);
    }
  }

  /**
   * 只检索证据、不生成回答：用于先看召回质量再决定是否让模型作答。
   * @sideEffect 覆盖右栏 sources。
   */
  async function searchOnly() {
    if (!repoId || !question.trim() || busy) return;
    setBusy("search");
    setError(null);
    try {
      const result = await apiPost<RAGSearch>(`/api/rag/${repoId}/search`, queryBody());
      setSources(result.results);
      setNotice(`召回 ${result.result_count} 条证据，耗时 ${result.took_ms} ms`);
    } catch (searchError) {
      setError(errorMessage(searchError));
    } finally {
      setBusy(null);
    }
  }

  return (
    <div className="-m-3 flex min-h-[calc(100vh-53px)] flex-col bg-[#f4f6f7] lg:h-[calc(100vh-53px)] lg:overflow-hidden">
      <header className="flex flex-wrap items-center justify-between gap-3 border-b border-line bg-white px-4 py-3">
        <div className="flex items-center gap-3">
          <span className="flex h-9 w-9 items-center justify-center rounded-md bg-[#143642] text-white">
            <BookOpen className="h-4 w-4" />
          </span>
          <div>
            <h1 className="text-base font-semibold text-ink">RAG 工作室</h1>
            <p className="text-xs text-slate-500">{currentRepo?.full_name ?? "未选择项目"}</p>
          </div>
        </div>
        <div className="flex min-w-0 items-center gap-2">
          <select
            value={repoId}
            onChange={(event) => {
              setRepoId(event.target.value);
              setTurns([]);
              setSources([]);
            }}
            className="h-9 min-w-0 max-w-64 rounded-md border border-line bg-white px-3 text-sm outline-none focus:border-teal-600"
            aria-label="选择项目"
          >
            {repos.length === 0 && <option value="">暂无项目</option>}
            {repos.map((repo) => (
              <option key={repo.id} value={repo.id}>{repo.full_name}</option>
            ))}
          </select>
          {repoId ? <Link href={`/knowledge/${repoId}`} className="hidden h-9 items-center rounded-md border border-line bg-white px-3 text-sm text-slate-600 hover:bg-slate-50 sm:inline-flex">管理项目知识</Link> : null}
          <button type="button" onClick={() => repoId && loadKnowledge(repoId)} className="flex h-9 w-9 items-center justify-center rounded-md border border-line bg-white text-slate-600 hover:bg-slate-50" title="刷新" aria-label="刷新">
            <RefreshCw className="h-4 w-4" />
          </button>
        </div>
      </header>

      <div className="grid border-b border-line bg-white sm:grid-cols-2 lg:grid-cols-5">
        <StatusItem label="资料" value={`${status?.source_count ?? 0} 份`} />
        <StatusItem label="切片" value={`${status?.document_count ?? 0} 条`} />
        <StatusItem label="Embedding" value={status ? `${status.embedding.provider} · ${status.embedding.dimensions}d` : "-"} />
        <StatusItem
          label="向量索引"
          value={status?.vector_store
            ? `${status.vector_store.backend ?? "vector"} · ${status.vector_store.available === false ? "degraded" : status.vector_store.status ?? "ready"}`
            : "-"}
        />
        <StatusItem label="生成" value={status?.generation.llm_configured ? status.generation.model ?? "LLM" : "本地证据模式"} />
      </div>

      {(error || notice) && (
        <div className={`border-b px-4 py-2 text-sm ${error ? "border-red-200 bg-red-50 text-red-700" : "border-emerald-200 bg-emerald-50 text-emerald-800"}`}>
          {error ?? notice}
        </div>
      )}

      <div className="grid min-h-0 flex-1 lg:grid-cols-[300px_minmax(420px,1fr)_340px] lg:overflow-hidden">
        <aside className="flex min-h-[360px] flex-col border-r border-line bg-[#f8faf9]">
          <div className="flex items-center justify-between border-b border-line px-4 py-3">
            <div className="flex items-center gap-2 text-sm font-semibold text-ink"><Database className="h-4 w-4 text-teal-700" />知识库</div>
            <button type="button" onClick={reindex} disabled={!repoId || busy === "reindex"} title="重建向量索引" aria-label="重建向量索引" className="flex h-8 w-8 items-center justify-center rounded-md text-slate-500 hover:bg-white disabled:opacity-50">
              <RefreshCw className={`h-4 w-4 ${busy === "reindex" ? "animate-spin" : ""}`} />
            </button>
          </div>
          <div className="border-b border-line p-3">
            <input ref={fileInput} type="file" accept={(status?.supported_files ?? []).join(",")} className="hidden" onChange={(event) => uploadFile(event.target.files?.[0])} />
            <button type="button" disabled={!repoId || busy === "upload"} onClick={() => fileInput.current?.click()} className="flex h-11 w-full items-center justify-center gap-2 rounded-md border border-dashed border-teal-400 bg-white text-sm font-medium text-teal-800 hover:bg-teal-50 disabled:opacity-50">
              {busy === "upload" ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <Upload className="h-4 w-4" />}
              {busy === "upload" ? "解析与入库中" : "上传资料"}
            </button>
          </div>
          <div className="min-h-0 flex-1 space-y-2 overflow-y-auto p-3">
            {knowledge.length === 0 && <EmptyPanel icon={FileText} label="知识库为空" />}
            {knowledge.map((item) => (
              <div key={`${item.source_type}:${item.id}`} className="group rounded-md border border-line bg-white p-3">
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0">
                    <div className="truncate text-sm font-medium text-ink" title={item.title}>{item.title}</div>
                    <div className="mt-1 text-xs text-slate-500">{sourceLabel(item.source_type)} · {item.chunk_count} 切片 {formatDate(item.created_at)}</div>
                  </div>
                  <button type="button" onClick={() => removeKnowledge(item)} disabled={busy === `delete:${item.id}`} title="删除资料" aria-label="删除资料" className="flex h-7 w-7 shrink-0 items-center justify-center rounded text-slate-400 opacity-0 hover:bg-red-50 hover:text-red-700 group-hover:opacity-100 focus:opacity-100">
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </div>
                <p className="mt-2 line-clamp-3 text-xs leading-5 text-slate-600">{item.content_preview}</p>
              </div>
            ))}
          </div>
        </aside>

        <main className="flex min-h-[520px] min-w-0 flex-col bg-white">
          <div className="flex items-center justify-between border-b border-line px-4 py-3">
            <div className="flex items-center gap-2 text-sm font-semibold text-ink"><Sparkles className="h-4 w-4 text-[#9a5b13]" />问答</div>
            {turns.length > 0 && <button type="button" onClick={() => { setTurns([]); setSources([]); }} className="text-xs text-slate-500 hover:text-ink">清空会话</button>}
          </div>
          <div ref={qaScrollRef} className="min-h-0 flex-1 space-y-4 overflow-y-auto px-4 py-5">
            {turns.length === 0 && <EmptyPanel icon={BookOpen} label={status?.document_count ? "输入问题开始检索" : "上传资料后开始提问"} />}
            {turns.map((turn) => (
              <div key={turn.id} className={turn.role === "user" ? "ml-auto max-w-[82%]" : "max-w-[92%]"}>
                <div className={`rounded-md px-4 py-3 text-sm leading-7 ${turn.role === "user" ? "bg-[#143642] text-white" : "border border-line bg-[#f7f9f8] text-ink"}`}>
                  {turn.role === "assistant" ? <div className="markdown-body"><ReactMarkdown remarkPlugins={[remarkGfm]}>{turn.content}</ReactMarkdown></div> : turn.content}
                </div>
                {turn.role === "assistant" && (
                  <div className="mt-1.5 flex flex-wrap gap-2 text-[11px] text-slate-500">
                    <span>{turn.mode === "llm" ? "LLM 生成" : turn.mode === "extractive" ? "证据提取" : "无证据"}</span>
                    <span>{turn.sourceCount ?? 0} 个引用</span>
                    {typeof turn.tookMs === "number" && <span>{turn.tookMs} ms</span>}
                  </div>
                )}
              </div>
            ))}
            {busy === "ask" && <div className="flex items-center gap-2 text-sm text-slate-500"><LoaderCircle className="h-4 w-4 animate-spin" />正在检索并生成回答</div>}
          </div>
          <div className="border-t border-line bg-[#f8faf9] p-3">
            <textarea
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  ask();
                }
              }}
              placeholder="向当前知识库提问"
              rows={3}
              className="w-full resize-none rounded-md border border-line bg-white px-3 py-2 text-sm leading-6 text-ink outline-none focus:border-teal-600 focus:ring-2 focus:ring-teal-100"
            />
            <div className="mt-2 flex flex-wrap items-center gap-2">
              <select value={sourceType} onChange={(event) => setSourceType(event.target.value)} className="h-9 rounded-md border border-line bg-white px-2 text-xs text-slate-600" aria-label="资料类型">
                <option value="">全部资料</option>
                {availableSourceTypes.map((type) => <option key={type} value={type}>{sourceLabel(type)}</option>)}
              </select>
              <label className="flex h-9 items-center gap-2 rounded-md border border-line bg-white px-2 text-xs text-slate-600">Top K
                <select value={topK} onChange={(event) => setTopK(Number(event.target.value))} className="bg-transparent font-medium text-ink outline-none">
                  {[3, 5, 8, 10].map((value) => <option key={value} value={value}>{value}</option>)}
                </select>
              </label>
              <div className="flex-1" />
              <button type="button" title="只检索证据" onClick={searchOnly} disabled={!repoId || !question.trim() || Boolean(busy)} className="flex h-9 w-9 items-center justify-center rounded-md border border-line bg-white text-slate-600 hover:bg-slate-50 disabled:opacity-40"><Search className="h-4 w-4" /></button>
              <button type="button" onClick={ask} disabled={!repoId || !question.trim() || Boolean(busy)} className="inline-flex h-9 items-center gap-2 rounded-md bg-teal-700 px-4 text-sm font-medium text-white hover:bg-teal-800 disabled:opacity-40"><Send className="h-4 w-4" />提问</button>
            </div>
          </div>
        </main>

        <aside className="flex min-h-[360px] flex-col border-l border-line bg-[#f8faf9]">
          <div className="flex items-center justify-between border-b border-line px-4 py-3">
            <div className="flex items-center gap-2 text-sm font-semibold text-ink"><Search className="h-4 w-4 text-teal-700" />检索证据</div>
            <span className="text-xs text-slate-500">{sources.length} 条</span>
          </div>
          <div className="min-h-0 flex-1 space-y-3 overflow-y-auto p-3">
            {sources.length === 0 && <EmptyPanel icon={Search} label="暂无检索结果" />}
            {sources.map((source) => (
              <article key={`${source.document_id}:${source.citation}`} className="rounded-md border border-line bg-white p-3">
                <div className="flex items-start gap-2">
                  <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded bg-[#143642] text-xs font-semibold text-white">{source.citation}</span>
                  <div className="min-w-0 flex-1">
                    <h2 className="break-words text-sm font-medium leading-5 text-ink">{source.title}</h2>
                    <div className="mt-1 text-xs text-slate-500">{sourceLabel(source.source_type)} · {(source.score * 100).toFixed(1)}%</div>
                  </div>
                </div>
                <div className="mt-2 h-1 overflow-hidden rounded bg-slate-100"><div className="h-full bg-teal-600" style={{ width: `${Math.max(3, Math.min(100, source.score * 100))}%` }} /></div>
                <p className="mt-3 whitespace-pre-wrap text-xs leading-5 text-slate-600">{source.snippet}</p>
                {Boolean(source.metadata.filename || source.metadata.path) && <div className="mt-2 break-all border-t border-line pt-2 text-[11px] text-slate-500">{String(source.metadata.filename ?? source.metadata.path)}</div>}
              </article>
            ))}
          </div>
        </aside>
      </div>
    </div>
  );
}

/** 顶部状态栏的单格：小写标签 + 单行截断的值（title 里放完整值，避免截断后看不全） */
function StatusItem({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0 border-r border-line px-4 py-2.5 last:border-r-0">
      <div className="text-[11px] font-medium uppercase text-slate-500">{label}</div>
      <div className="mt-0.5 truncate text-sm font-semibold text-ink" title={value}>{value}</div>
    </div>
  );
}

/** 空态占位：虚线框 + 图标 + 一句提示，用于知识库为空 / 暂无证据 / 尚未提问 */
function EmptyPanel({ icon: Icon, label }: { icon: typeof Search; label: string }) {
  return (
    <div className="flex min-h-36 flex-col items-center justify-center gap-2 rounded-md border border-dashed border-line bg-white/60 p-4 text-center text-sm text-slate-500">
      <Icon className="h-5 w-5 text-slate-400" />
      <span>{label}</span>
    </div>
  );
}
