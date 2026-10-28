"use client";
/*
 * 知识库「分段 + 索引 + 检索」配置表单（受控组件）。
 *
 * 用途：知识库详情页与检索测试页共用同一套表单。组件本身不请求后端，
 * 只把用户改动通过 onChange 回抛给父组件，由父组件决定是本地暂存还是 PATCH 提交
 * （对应接口 PATCH /api/rag/repositories/{repoId}/config）。
 * props：
 * - value 当前配置（KnowledgeConfig，定义见 lib/rag.ts）
 * - catalog 可选模型目录（GET /api/rag/models），为 null 时下拉退化为只显示当前值
 * - onChange 任意字段变更的回调，返回一份新的完整配置对象
 * - showChunking 是否展示分段设置：检索测试页不需要重新切块，可关闭
 * 说明：模型下拉的 value 采用 "provider::model" 拼接，避免同名模型跨厂商冲突。
 */

import { Check, FileSearch, Layers3, Search } from "lucide-react";

import type { KnowledgeConfig, ModelCatalog, RetrievalMethod } from "@/lib/rag";

/**
 * 组件 props。
 * @param value 受控的当前配置
 * @param catalog 模型目录，null 表示还没加载到（或接口失败）
 * @param onChange 字段变更回调
 * @param showChunking 是否显示分段长度/重叠设置，默认显示
 */
type Props = {
  value: KnowledgeConfig;
  catalog: ModelCatalog | null;
  onChange: (value: KnowledgeConfig) => void;
  showChunking?: boolean;
};

// 三种检索方式：note 用一句话说明适用场景，帮助用户选型
const methods: { value: RetrievalMethod; label: string; note: string; icon: typeof Search }[] = [
  { value: "vector", label: "向量检索", note: "擅长理解语义相近的表达", icon: Layers3 },
  { value: "fulltext", label: "全文检索", note: "适合专有名词与精确关键词", icon: FileSearch },
  { value: "hybrid", label: "混合检索", note: "融合语义与关键词结果", icon: Search }
];

/**
 * 检索配置表单。
 * 纯受控组件：无内部状态、无副作用，所有变更都通过 onChange 上抛。
 */
export function RetrievalSettings({ value, catalog, onChange, showChunking = true }: Props) {
  // 统一更新入口：用展开生成新对象，保证父组件拿到的是新引用（React 才能感知变化）
  const update = <K extends keyof KnowledgeConfig,>(key: K, next: KnowledgeConfig[K]) => onChange({ ...value, [key]: next });

  return (
    <div className="space-y-7">
      {showChunking ? (
        <section>
          <h3 className="text-sm font-semibold text-ink">分段设置</h3>
          <p className="mt-1 text-xs text-slate-500">按自然段落切分，过长段落会继续按长度拆开。</p>
          <div className="mt-3 grid gap-3 sm:grid-cols-2">
            <label className="text-xs font-medium text-slate-600">
              最大分段长度
              <input
                type="number"
                min={200}
                max={4000}
                value={value.chunk_size}
                onChange={(event) => update("chunk_size", Number(event.target.value))}
                className="mt-1.5 h-10 w-full rounded-md border border-line bg-white px-3 text-sm text-ink outline-none focus:border-teal-600"
              />
            </label>
            <label className="text-xs font-medium text-slate-600">
              分段重叠长度
              <input
                type="number"
                min={0}
                max={1000}
                value={value.chunk_overlap}
                onChange={(event) => update("chunk_overlap", Number(event.target.value))}
                className="mt-1.5 h-10 w-full rounded-md border border-line bg-white px-3 text-sm text-ink outline-none focus:border-teal-600"
              />
            </label>
          </div>
        </section>
      ) : null}

      <section>
        <h3 className="text-sm font-semibold text-ink">索引方式</h3>
        <p className="mt-1 text-xs text-slate-500">DeepSeek 官方 API 暂无 Embedding，默认使用 Qwen。</p>
        <label className="mt-3 block text-xs font-medium text-slate-600">
          Embedding 模型
          <select
            value={`${value.embedding_provider}::${value.embedding_model}`}
            onChange={(event) => {
              // 换 embedding 模型必须同步维度：向量库中的列宽由模型决定，维度不一致会检索失败
              const [provider, model] = event.target.value.split("::");
              const selected = catalog?.embedding_models.find((item) => item.provider === provider && item.model === model);
              onChange({ ...value, embedding_provider: provider, embedding_model: model, embedding_dimensions: selected?.dimensions ?? value.embedding_dimensions });
            }}
            className="mt-1.5 h-10 w-full rounded-md border border-line bg-white px-3 text-sm text-ink outline-none focus:border-teal-600"
          >
            {(catalog?.embedding_models ?? []).map((model) => (
              <option key={`${model.provider}-${model.model}`} value={`${model.provider}::${model.model}`}>
                {model.label}{model.available ? "" : " · 未配置，自动降级"}
              </option>
            ))}
            {!catalog ? <option value={`${value.embedding_provider}::${value.embedding_model}`}>{value.embedding_model}</option> : null}
          </select>
        </label>
      </section>

      <section>
        <h3 className="text-sm font-semibold text-ink">检索设置</h3>
        <div className="mt-3 grid gap-2 sm:grid-cols-3">
          {methods.map((method) => {
            // 从配置项里取出图标组件与选中态，供下方卡片渲染
            const Icon = method.icon;
            const selected = value.retrieval_method === method.value;
            return (
              <button
                key={method.value}
                type="button"
                onClick={() => update("retrieval_method", method.value)}
                className={`relative min-h-28 rounded-md border p-3 text-left transition ${selected ? "border-teal-600 bg-teal-50/60" : "border-line bg-white hover:border-slate-400"}`}
              >
                {selected ? <Check className="absolute right-2 top-2 h-4 w-4 text-teal-700" /> : null}
                <Icon className={`h-5 w-5 ${selected ? "text-teal-700" : "text-slate-500"}`} />
                <span className="mt-3 block text-sm font-medium text-ink">{method.label}</span>
                <span className="mt-1 block text-xs leading-5 text-slate-500">{method.note}</span>
              </button>
            );
          })}
        </div>
      </section>

      <section className="border-t border-line pt-5">
        <div className="flex items-center justify-between gap-4">
          <div>
            <h3 className="text-sm font-semibold text-ink">Rerank 模型</h3>
            <p className="mt-1 text-xs text-slate-500">对召回候选重新排序，API 不可用时自动使用本地重排。</p>
          </div>
          <button
            type="button"
            role="switch"
            aria-checked={value.rerank_enabled}
            onClick={() => update("rerank_enabled", !value.rerank_enabled)}
            className={`relative h-6 w-11 shrink-0 rounded-full transition ${value.rerank_enabled ? "bg-teal-700" : "bg-slate-300"}`}
          >
            <span className={`absolute top-1 h-4 w-4 rounded-full bg-white transition ${value.rerank_enabled ? "left-6" : "left-1"}`} />
          </button>
        </div>
        {value.rerank_enabled ? (
          <select
            value={`${value.rerank_provider}::${value.rerank_model}`}
            onChange={(event) => {
              const [provider, model] = event.target.value.split("::");
              onChange({ ...value, rerank_provider: provider, rerank_model: model });
            }}
            className="mt-3 h-10 w-full rounded-md border border-line bg-white px-3 text-sm text-ink outline-none focus:border-teal-600"
          >
            {(catalog?.rerank_models ?? []).map((model) => (
              <option key={`${model.provider}-${model.model}`} value={`${model.provider}::${model.model}`}>
                {model.label}{model.available ? "" : " · 未配置，自动降级"}
              </option>
            ))}
            {!catalog ? <option value={`${value.rerank_provider}::${value.rerank_model}`}>{value.rerank_model}</option> : null}
          </select>
        ) : null}
      </section>

      <section className="grid gap-4 border-t border-line pt-5 sm:grid-cols-2">
        <label className="text-xs font-medium text-slate-600">
          Top K
          <div className="mt-2 flex items-center gap-3">
            <input type="range" min={1} max={10} value={value.top_k} onChange={(event) => update("top_k", Number(event.target.value))} className="min-w-0 flex-1 accent-teal-700" />
            <input type="number" min={1} max={20} value={value.top_k} onChange={(event) => update("top_k", Number(event.target.value))} className="h-9 w-16 rounded-md border border-line text-center text-sm outline-none focus:border-teal-600" />
          </div>
        </label>
        <div>
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-600">Score 阈值</span>
            <input type="checkbox" checked={value.score_threshold_enabled} onChange={(event) => update("score_threshold_enabled", event.target.checked)} className="h-4 w-4 accent-teal-700" />
          </div>
          <div className="mt-2 flex items-center gap-3">
            <input type="range" min={0} max={1} step={0.01} disabled={!value.score_threshold_enabled} value={value.score_threshold} onChange={(event) => update("score_threshold", Number(event.target.value))} className="min-w-0 flex-1 accent-teal-700 disabled:opacity-40" />
            <span className="w-12 text-right text-sm text-slate-600">{value.score_threshold.toFixed(2)}</span>
          </div>
        </div>
      </section>
    </div>
  );
}
