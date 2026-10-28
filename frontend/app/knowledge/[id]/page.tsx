"use client";
/*
 * 单个知识库详情页（路由 /knowledge/[id]）。
 *
 * 展示内容：顶部是仓库名与规模统计，左侧是三个 Tab，右侧随 Tab 切换：
 * - documents：文档列表（解析/向量化状态），点开可用文档看真实分段，失败文档可重试；
 * - retrieval：召回测试，输入问题后展示每一段召回内容、分数、检索链路（Retrieval Trace）与答案判断；
 * - settings：分段/模型/检索参数配置。
 * 数据来源（全部以后端 rag 路由为准）：
 * - GET   /api/rag/repositories/{id}                          知识库详情（含 config 与 runtime）
 * - GET   /api/rag/repositories/{id}/documents                文档列表
 * - GET   /api/rag/models                                     可选模型目录
 * - POST  /api/rag/repositories/{id}/documents                上传文档（multipart，202 异步处理）
 * - GET   /api/rag/repositories/{id}/documents/{docId}/chunks 文档分段
 * - POST  /api/rag/repositories/{id}/documents/{docId}/retry  重试失败文档
 * - POST  /api/rag/repositories/{id}/retrieval-tests          执行召回测试
 * - GET   /api/rag/repositories/{id}/retrieval-tests          测试历史
 * - PATCH /api/rag/repositories/{id}/config                   保存配置
 * - POST  /api/rag/{id}/reindex                               换 embedding 模型后重建向量
 * 状态要点：文档在异步流水线中会用 1 秒轮询刷新；Tab 通过 URL 的 ?tab= 参数持久化，刷新不丢。
 */

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import {
  ArrowLeft,
  BookOpen,
  Check,
  ChevronRight,
  FileSearch,
  FileText,
  History,
  LoaderCircle,
  Play,
  Plus,
  RotateCcw,
  Save,
  Settings,
  TestTube2,
  X
} from "lucide-react";

import { RetrievalSettings } from "@/components/rag/retrieval-settings";
import { apiGet, apiPatch, apiPost, apiUpload } from "@/lib/api";
import {
  formatBytes,
  formatDate,
  formatNumber,
  statusLabel,
  type KnowledgeBase,
  type KnowledgeChunk,
  type KnowledgeConfig,
  type KnowledgeDocument,
  type ModelCatalog,
  type RetrievalHistory,
  type RetrievalResponse
} from "@/lib/rag";

// 三个 Tab：文档管理 / 召回测试 / 参数设置
type Tab = "documents" | "retrieval" | "settings";

const tabs: { value: Tab; label: string; icon: typeof FileText }[] = [
  { value: "documents", label: "文档", icon: FileText },
  { value: "retrieval", label: "召回测试", icon: TestTube2 },
  { value: "settings", label: "设置", icon: Settings }
];

// 检索链路阶段名 → 中文：后端返回的是英文阶段标识，直接展示可读性差
const pipelineStageLabel: Record<string, string> = {
  vector_recall: "向量召回",
  keyword_recall: "BM25 召回",
  weighted_fusion: "加权融合",
  single_retriever_selection: "单路候选",
  deduplication_and_parent_aggregation: "去重聚合",
  rerank: "重排",
  parent_expansion: "父级扩展",
  score_threshold: "分数门槛",
  context_compression: "证据压缩",
  answer_gate: "回答判断"
};

// 后端对「证据是否够回答问题」的判定结论 → 中文
const answerGateLabel: Record<string, string> = {
  answer: "证据充足",
  ask_clarification: "需要补充范围",
  insufficient_evidence: "证据不足",
  conflict: "证据冲突"
};

/**
 * 知识库详情页组件。
 * @sideEffect 挂载时加载详情/文档/模型目录并读取测试历史；文档处于解析中时启动 1 秒轮询；
 *            上传、重试、保存配置都会重新拉取数据。
 */
export default function KnowledgeBaseDetailPage() {
  // 路由参数 id 即知识库（仓库）id，后续所有接口路径都基于它
  const id = String(useParams<{ id: string }>().id);
  // 隐藏的 file input：由「添加文件」按钮触发，保证上传入口样式统一
  const fileInput = useRef<HTMLInputElement>(null);
  const [tab, setTab] = useState<Tab>("documents");
  const [knowledge, setKnowledge] = useState<KnowledgeBase | null>(null);
  const [documents, setDocuments] = useState<KnowledgeDocument[]>([]);
  const [catalog, setCatalog] = useState<ModelCatalog | null>(null);
  // config 是「表单草稿」：与 knowledge.config（服务端已保存值）分离，便于保存时判断是否有变更
  const [config, setConfig] = useState<KnowledgeConfig | null>(null);
  // 非空时右侧抽屉展示该文档的分段内容
  const [selectedDocument, setSelectedDocument] = useState<KnowledgeDocument | null>(null);
  const [chunks, setChunks] = useState<KnowledgeChunk[]>([]);
  const [query, setQuery] = useState("");
  const [retrieval, setRetrieval] = useState<RetrievalResponse | null>(null);
  const [history, setHistory] = useState<RetrievalHistory[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [testing, setTesting] = useState(false);
  const [saving, setSaving] = useState(false);
  // message 是成功提示（绿色），error 是失败提示（红色），两者互斥使用
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  /**
   * 并行加载详情、文档、模型目录三个接口。
   * 用 useCallback 固定引用并依赖 id：既能在 effect 里安全调用，也能作为轮询回调复用。
   */
  const load = useCallback(async () => {
    // 三者互不依赖，用 Promise.all 并行，减少首屏等待
    const [base, docs, models] = await Promise.all([
      apiGet<KnowledgeBase>(`/api/rag/repositories/${id}`),
      apiGet<KnowledgeDocument[]>(`/api/rag/repositories/${id}/documents`),
      apiGet<ModelCatalog>("/api/rag/models")
    ]);
    setKnowledge(base);
    setDocuments(docs);
    setCatalog(models);
    // 每次刷新都把草稿重置为服务端值，避免旧编辑覆盖新数据
    setConfig(base.config);
  }, [id]);

  // 单独拉测试历史：历史列表失败不影响主内容，所以静默降级为 null
  const loadHistory = useCallback(() => {
    apiGet<RetrievalHistory[]>(`/api/rag/repositories/${id}/retrieval-tests`).then(setHistory).catch(() => null);
  }, [id]);

  // 首屏：优先用 URL 的 ?tab= 恢复上次所在 Tab（支持分享链接直接定位到召回测试）
  useEffect(() => {
    const initial = new URLSearchParams(window.location.search).get("tab");
    if (initial === "retrieval" || initial === "settings") setTab(initial);
    load()
      .catch((reason: Error) => setError(reason.message))
      .finally(() => setLoading(false));
    loadHistory();
  }, [load, loadHistory]);

  // 文档解析是后端异步任务（queued → parsing → splitting → embedding → completed），
  // 只要还有未完成的文档就每秒刷新一次；全部结束后 effect 重跑并清掉定时器，避免空转。
  useEffect(() => {
    if (!documents.some((item) => !["completed", "failed"].includes(item.status))) return;
    const timer = window.setInterval(() => load().catch(() => null), 1000);
    return () => window.clearInterval(timer);
  }, [documents, load]);

  /**
   * 切换 Tab 并同步写回 URL。
   * 用 replaceState 而非 pushState：切换 Tab 不该在浏览器历史里堆一堆记录，否则「后退」会很难用。
   */
  function changeTab(next: Tab) {
    setTab(next);
    window.history.replaceState(null, "", `${window.location.pathname}?tab=${next}`);
  }

  /**
   * 逐个上传选中的文件。
   * 串行 for-await：后端解析/向量化是重任务，并发上传容易把队列打满；
   * 单个失败会中断后续文件并提示错误。
   * @sideEffect 上传完成后刷新文档列表，并清空 file input 以便重复选择同一文件。
   */
  async function uploadFiles(list: FileList | null) {
    if (!list?.length) return;
    setUploading(true);
    setError("");
    try {
      for (const file of Array.from(list)) {
        const form = new FormData();
        form.append("file", file);
        await apiUpload<KnowledgeDocument>(`/api/rag/repositories/${id}/documents`, form);
      }
      await load();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "上传失败");
    } finally {
      setUploading(false);
      // 重置 value：否则再次选择同一个文件不会触发 change 事件
      if (fileInput.current) fileInput.current.value = "";
    }
  }

  /**
   * 打开某篇文档的分段抽屉。
   * 先清空 chunks，避免短暂显示上一篇文档的残留内容。
   */
  async function openChunks(document: KnowledgeDocument) {
    setSelectedDocument(document);
    setChunks([]);
    try {
      setChunks(await apiGet<KnowledgeChunk[]>(`/api/rag/repositories/${id}/documents/${document.id}/chunks`));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "读取分段失败");
    }
  }

  /**
   * 重试失败的文档：后端会重新走一遍解析+向量化流水线。
   * 刷新后状态变为排队中，轮询 effect 会自动接管进度展示。
   */
  async function retry(document: KnowledgeDocument) {
    await apiPost(`/api/rag/repositories/${id}/documents/${document.id}/retry`, {});
    await load();
  }

  /**
   * 执行一次召回测试。
   * 请求体显式带上 top_k / 检索方式 / 重排 / 阈值：测试用的是「当前表单值」而非服务端已存配置，
   * 这样用户可以在保存之前反复试参数。
   * @sideEffect 写入 retrieval 结果并刷新测试历史。
   */
  async function runRetrieval() {
    if (!query.trim() || !config) return;
    setTesting(true);
    setError("");
    try {
      const result = await apiPost<RetrievalResponse>(`/api/rag/repositories/${id}/retrieval-tests`, {
        query: query.trim(),
        top_k: config.top_k,
        retrieval_method: config.retrieval_method,
        rerank_enabled: config.rerank_enabled,
        score_threshold_enabled: config.score_threshold_enabled,
        score_threshold: config.score_threshold
      });
      setRetrieval(result);
      loadHistory();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "召回测试失败");
    } finally {
      setTesting(false);
    }
  }

  /**
   * 保存知识库配置，并在必要时触发重建索引。
   * 为什么要重建：换了 embedding 模型/服务商后，历史向量与新模型的向量空间不兼容，
   * 必须把已有分段重新向量化才能继续检索；仅改分段长度则只影响后续新文档，无需重建。
   * @sideEffect PATCH 配置；可能再调用 reindex；更新 knowledge/config 并给出成功提示。
   */
  async function saveSettings() {
    if (!config || !knowledge) return;
    setSaving(true);
    setMessage("");
    setError("");
    // 与已保存值比较，判断是否发生了「模型维度」层面的变更
    const reindexNeeded = config.embedding_model !== knowledge.config.embedding_model || config.embedding_provider !== knowledge.config.embedding_provider;
    try {
      const updated = await apiPatch<KnowledgeBase>(`/api/rag/repositories/${id}/config`, config);
      setKnowledge(updated);
      setConfig(updated.config);
      // 只有已有分段（chunk_count > 0）时重建才有意义
      if (reindexNeeded && updated.chunk_count) {
        await apiPost(`/api/rag/${id}/reindex`, {});
        setMessage("设置已保存，现有分段已重新向量化。平时修改分段长度只会影响新上传的文档。");
      } else {
        setMessage("设置已保存。");
      }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "保存失败");
    } finally {
      setSaving(false);
    }
  }

  if (loading) return <div className="grid min-h-[70vh] place-items-center text-sm text-slate-500">正在加载项目知识...</div>;
  if (!knowledge || !config) return <div className="m-6 border border-red-200 bg-red-50 p-4 text-sm text-red-700">{error || "未找到仓库"}</div>;

  return (
    <div className="min-h-[calc(100vh-74px)] bg-[#f7f8fa]">
      <header className="border-b border-line bg-white px-4 py-3">
        <div className="flex min-w-0 items-center gap-3">
          <Link href="/knowledge" title="返回项目知识" className="grid h-9 w-9 shrink-0 place-items-center rounded-md border border-line text-slate-600 hover:text-ink"><ArrowLeft className="h-4 w-4" /></Link>
          <span className="grid h-9 w-9 shrink-0 place-items-center rounded-md bg-teal-50 text-teal-700"><BookOpen className="h-4 w-4" /></span>
          <div className="min-w-0">
            <h1 className="truncate text-sm font-semibold text-ink">{knowledge.name}</h1>
            <p className="truncate text-xs text-slate-400">{knowledge.full_name}</p>
          </div>
          <div className="ml-auto hidden items-center gap-5 text-xs text-slate-500 sm:flex">
            <span>{formatNumber(knowledge.document_count)} 文档</span><span>{formatNumber(knowledge.chunk_count)} 分段</span><span>{formatNumber(knowledge.character_count)} 字符</span>
          </div>
        </div>
      </header>

      <div className="grid min-h-[calc(100vh-135px)] md:grid-cols-[190px_minmax(0,1fr)]">
        <aside className="border-b border-line bg-white p-3 md:border-b-0 md:border-r">
          <nav className="flex gap-1 overflow-x-auto md:block md:space-y-1">
            {tabs.map((item) => {
              const Icon = item.icon;
              return (
                <button key={item.value} type="button" onClick={() => changeTab(item.value)} className={`flex h-10 shrink-0 items-center gap-2 rounded-md px-3 text-sm transition md:w-full ${tab === item.value ? "bg-teal-50 font-medium text-teal-800" : "text-slate-600 hover:bg-slate-50"}`}>
                  <Icon className="h-4 w-4" /> {item.label}
                </button>
              );
            })}
          </nav>
          <div className="mt-5 hidden border-t border-line pt-4 text-xs leading-5 text-slate-400 md:block">
            <div className="font-medium text-slate-600">运行状态</div>
            <div className="mt-2">Embedding</div>
            <div className={knowledge.runtime.embedding.available ? "text-emerald-700" : "text-amber-700"}>
              {knowledge.runtime.embedding.available
                ? `${knowledge.runtime.embedding.provider} 就绪`
                : `${knowledge.runtime.embedding.provider} 不可用`}
            </div>
            <div className="mt-2">Rerank</div>
            <div className={knowledge.runtime.rerank.available ? "text-emerald-700" : "text-amber-700"}>
              {knowledge.runtime.rerank.available
                ? `${knowledge.runtime.rerank.provider} 就绪`
                : knowledge.runtime.rerank.fallback === "heuristic"
                  ? "启发式重排"
                  : `${knowledge.runtime.rerank.provider} 不可用`}
            </div>
          </div>
        </aside>

        <main className="min-w-0 p-4 sm:p-6">
          {error ? <div className="mb-4 flex items-center justify-between bg-red-50 px-4 py-3 text-sm text-red-700"><span>{error}</span><button onClick={() => setError("")} title="关闭"><X className="h-4 w-4" /></button></div> : null}

          {tab === "documents" ? (
            <div className="mx-auto max-w-6xl">
              <div className="flex items-end justify-between gap-4">
                <div><h2 className="text-lg font-semibold text-ink">文档</h2><p className="mt-1 text-sm text-slate-500">查看解析状态和真实分段内容。</p></div>
                <input ref={fileInput} type="file" multiple accept=".txt,.md,.markdown,.pdf,.docx,.json,.csv,.log" className="hidden" onChange={(event) => uploadFiles(event.target.files)} />
                <button type="button" onClick={() => fileInput.current?.click()} disabled={uploading} className="inline-flex h-10 items-center gap-2 rounded-md bg-teal-700 px-4 text-sm font-medium text-white disabled:opacity-50">
                  {uploading ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <Plus className="h-4 w-4" />} 添加文件
                </button>
              </div>

              <div className="mt-5 overflow-hidden border border-line bg-white shadow-sm">
                <div className="hidden grid-cols-[minmax(220px,1fr)_120px_100px_100px_140px_34px] gap-3 border-b border-line bg-slate-50 px-4 py-2.5 text-xs font-medium text-slate-500 md:grid">
                  <span>名称</span><span>状态</span><span>分段</span><span>字符数</span><span>上传时间</span><span />
                </div>
                {documents.map((document) => (
                  <button key={document.id} type="button" onClick={() => document.status === "completed" && openChunks(document)} className="grid w-full gap-2 border-b border-slate-100 px-4 py-3 text-left last:border-b-0 hover:bg-slate-50 md:grid-cols-[minmax(220px,1fr)_120px_100px_100px_140px_34px] md:items-center md:gap-3">
                    <span className="flex min-w-0 items-center gap-2"><FileText className="h-4 w-4 shrink-0 text-teal-700" /><span className="truncate text-sm font-medium text-ink">{document.name}</span></span>
                    <span className={`w-fit rounded px-2 py-1 text-xs ${document.status === "completed" ? "bg-emerald-50 text-emerald-700" : document.status === "failed" ? "bg-red-50 text-red-700" : "bg-amber-50 text-amber-700"}`}>
                      {statusLabel[document.status]}
                    </span>
                    <span className="text-xs text-slate-500"><span className="md:hidden">分段 </span>{formatNumber(document.chunk_count)}</span>
                    <span className="text-xs text-slate-500"><span className="md:hidden">字符 </span>{formatNumber(document.character_count)}</span>
                    <span className="text-xs text-slate-400">{formatDate(document.created_at)}</span>
                    {document.status === "failed" ? <span onClick={(event) => { event.stopPropagation(); retry(document); }} title="重试" className="grid h-7 w-7 place-items-center text-red-600"><RotateCcw className="h-4 w-4" /></span> : <ChevronRight className="hidden h-4 w-4 text-slate-300 md:block" />}
                  </button>
                ))}
                {!documents.length ? <div className="py-20 text-center text-sm text-slate-500">还没有文档，添加一份资料开始构建。</div> : null}
              </div>
            </div>
          ) : null}

          {tab === "retrieval" ? (
            <div className="mx-auto max-w-7xl">
              <div><h2 className="text-lg font-semibold text-ink">召回测试</h2><p className="mt-1 text-sm text-slate-500">输入真实问题，观察被召回的分段、分数和排序。</p></div>
              <div className="mt-5 grid gap-5 xl:grid-cols-[minmax(360px,.72fr)_minmax(520px,1.28fr)]">
                <div className="space-y-5">
                  <section className="border border-line bg-white p-4 shadow-sm">
                    <label className="text-xs font-medium text-slate-600">查询文本</label>
                    <textarea value={query} onChange={(event) => setQuery(event.target.value)} onKeyDown={(event) => { if ((event.ctrlKey || event.metaKey) && event.key === "Enter") runRetrieval(); }} maxLength={250} placeholder="例如：新员工可以请几天年假？" className="mt-2 min-h-36 w-full resize-y rounded-md border border-line p-3 text-sm leading-6 outline-none focus:border-teal-600" />
                    <div className="mt-3 flex items-center justify-between"><span className="text-xs text-slate-400">{query.length} / 250</span><button type="button" disabled={!query.trim() || testing} onClick={runRetrieval} className="inline-flex h-9 items-center gap-2 rounded-md bg-teal-700 px-4 text-sm font-medium text-white disabled:opacity-40">{testing ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />} 测试</button></div>
                  </section>
                  <section className="border border-line bg-white p-4 shadow-sm">
                    <div className="flex items-center justify-between"><h3 className="flex items-center gap-2 text-sm font-semibold text-ink"><History className="h-4 w-4" /> 测试记录</h3><span className="text-xs text-slate-400">最近 {history.length} 条</span></div>
                    <div className="mt-3 divide-y divide-slate-100">
                      {history.map((item) => <div key={item.id} className="py-3"><div className="truncate text-sm text-slate-700">{item.query}</div><div className="mt-1 flex gap-3 text-xs text-slate-400"><span>{item.result_count} 条结果</span><span>{item.took_ms} ms</span><span>{formatDate(item.created_at)}</span></div></div>)}
                      {!history.length ? <div className="py-8 text-center text-xs text-slate-400">暂无测试记录</div> : null}
                    </div>
                  </section>
                </div>

                <section className="min-w-0">
                  <div className="flex items-center justify-between"><h3 className="text-sm font-semibold text-ink">召回结果</h3>{retrieval ? <span className="text-xs text-slate-500">{retrieval.result_count} 条 · {retrieval.took_ms} ms</span> : null}</div>
                  {retrieval?.retrieval_config.pipeline?.stages?.length ? (
                    <div className="mt-3 border border-line bg-white p-4 shadow-sm">
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <div className="text-xs font-medium text-slate-600">Retrieval Trace</div>
                        <span className={`rounded px-2 py-1 text-xs font-medium ${retrieval.retrieval_config.answer_gate?.decision === "answer" ? "bg-emerald-50 text-emerald-700" : "bg-amber-50 text-amber-700"}`}>
                          {answerGateLabel[String(retrieval.retrieval_config.answer_gate?.decision ?? "")] ?? "等待判断"}
                        </span>
                      </div>
                      <div className="mt-3 flex flex-wrap items-center gap-2">
                        {retrieval.retrieval_config.pipeline.stages.map((stage, index) => {
                          const name = String(stage.name ?? "unknown");
                          const outputCount = stage.output_count ?? stage.candidate_count;
                          return (
                            <div key={`${name}-${index}`} className="flex items-center gap-2">
                              {index ? <span className="text-slate-300">→</span> : null}
                              <span className="rounded border border-slate-200 bg-slate-50 px-2 py-1 text-xs text-slate-600">
                                {pipelineStageLabel[name] ?? name}{typeof outputCount === "number" ? ` · ${outputCount}` : ""}
                              </span>
                            </div>
                          );
                        })}
                      </div>
                      {retrieval.retrieval_config.answer_gate?.reason ? <p className="mt-3 text-xs leading-5 text-slate-500">{retrieval.retrieval_config.answer_gate.reason}</p> : null}
                    </div>
                  ) : null}
                  <div className="mt-3 space-y-3">
                    {retrieval?.results.map((result, index) => (
                      <article key={`${result.document_id}-${index}`} className="border border-line bg-white p-4 shadow-sm">
                        <div className="flex items-center gap-3 border-b border-slate-100 pb-3">
                          <span className="grid h-7 w-7 shrink-0 place-items-center rounded bg-teal-50 text-xs font-semibold text-teal-700">{index + 1}</span>
                          <div className="min-w-0 flex-1"><h4 className="truncate text-sm font-medium text-ink">{result.title}</h4><p className="mt-0.5 truncate text-xs text-slate-400">{String(result.metadata.filename ?? result.source_type)}</p></div>
                          <span className="rounded bg-emerald-50 px-2 py-1 text-xs font-semibold text-emerald-700">{result.score.toFixed(3)}</span>
                        </div>
                        <p className="mt-3 whitespace-pre-wrap text-sm leading-6 text-slate-700">{result.snippet}</p>
                        <div className="mt-3 flex flex-wrap gap-3 text-xs text-slate-400"><span>分段 #{String(result.metadata.chunk_index ?? index + 1)}</span><span>{result.snippet.length} 字符</span><span>{String(result.retrieval.keyword_backend ?? result.retrieval.vector_backend ?? "retrieval")}</span><span>{String(result.retrieval.fusion_method ?? retrieval.retrieval_config.fusion_method ?? config.retrieval_method)}</span><span>{String(result.retrieval.rerank_mode ?? result.retrieval.mode ?? "retrieval")}</span></div>
                      </article>
                    ))}
                    {!retrieval ? <div className="grid min-h-80 place-items-center border border-dashed border-slate-300 bg-white text-center"><div><FileSearch className="mx-auto h-8 w-8 text-slate-300" /><p className="mt-3 text-sm text-slate-500">输入问题后查看真实召回分段</p></div></div> : null}
                    {retrieval && !retrieval.results.length ? <div className="grid min-h-64 place-items-center border border-line bg-white text-sm text-slate-500">没有分段通过当前检索条件。</div> : null}
                  </div>
                </section>
              </div>
            </div>
          ) : null}

          {tab === "settings" ? (
            <div className="mx-auto max-w-4xl">
              <div><h2 className="text-lg font-semibold text-ink">知识库设置</h2><p className="mt-1 text-sm text-slate-500">控制后续入库和召回测试使用的模型与参数。</p></div>
              <div className="mt-5 border border-line bg-white p-5 shadow-sm sm:p-7">
                <RetrievalSettings value={config} catalog={catalog} onChange={setConfig} />
                {message ? <div className="mt-5 flex items-center gap-2 bg-emerald-50 px-3 py-2.5 text-sm text-emerald-700"><Check className="h-4 w-4" /> {message}</div> : null}
                <div className="mt-6 flex justify-end border-t border-line pt-5"><button type="button" onClick={saveSettings} disabled={saving || config.chunk_overlap >= config.chunk_size} className="inline-flex h-10 items-center gap-2 rounded-md bg-teal-700 px-5 text-sm font-medium text-white disabled:opacity-40">{saving ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <Save className="h-4 w-4" />} 保存设置</button></div>
              </div>
            </div>
          ) : null}
        </main>
      </div>

      {selectedDocument ? (
        <div className="fixed inset-0 z-50 flex justify-end bg-black/30" onMouseDown={() => setSelectedDocument(null)}>
          <div className="h-full w-full max-w-2xl overflow-y-auto border-l border-line bg-[#f7f8fa] shadow-2xl" onMouseDown={(event) => event.stopPropagation()}>
            <header className="sticky top-0 z-10 flex items-center gap-3 border-b border-line bg-white px-5 py-4"><FileText className="h-5 w-5 text-teal-700" /><div className="min-w-0 flex-1"><h2 className="truncate text-sm font-semibold text-ink">{selectedDocument.name}</h2><p className="text-xs text-slate-400">{selectedDocument.chunk_count} 个分段 · {formatBytes(selectedDocument.file_size)}</p></div><button title="关闭" onClick={() => setSelectedDocument(null)} className="grid h-8 w-8 place-items-center text-slate-500"><X className="h-4 w-4" /></button></header>
            <div className="space-y-3 p-4 sm:p-5">
              {chunks.map((chunk) => <article key={chunk.id} className="border border-line bg-white p-4 shadow-sm"><div className="mb-3 flex items-center justify-between"><span className="text-xs font-semibold text-teal-700">分段 #{chunk.position}</span><span className="text-xs text-slate-400">{chunk.character_count} 字符</span></div><p className="whitespace-pre-wrap text-sm leading-6 text-slate-700">{chunk.content}</p></article>)}
              {!chunks.length ? <div className="py-20 text-center text-sm text-slate-500">正在读取分段...</div> : null}
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
