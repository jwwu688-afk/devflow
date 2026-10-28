"use client";
/*
 * 工作流运行记录页（路由 /workflow-runs）。
 *
 * 展示内容：顶部一句话指标（运行数 / 完成任务数 / Observer 发现数）与仓库切换；
 * 左侧是运行列表，右侧展示 Planner 任务图、每个 Agent 的执行摘要、
 * Synthesis 最终结论与 Observer 交叉检查的发现（含置信度）。
 * 数据来源：
 * - GET /api/repos                              仓库下拉
 * - GET /api/chat/workflow-runs?repo_id=...     某仓库的工作流运行记录（含 tasks 明细）
 * 交互状态：repoId（切换仓库会重新拉列表）、selectedId（当前查看的运行）、loading。
 */

import { useEffect, useMemo, useState } from "react";
import { AlertTriangle, CheckCircle2, Clock3, Database, GitBranch, Workflow } from "lucide-react";

import { apiGet } from "@/lib/api";
import type { Repository } from "@/lib/types";
import { PageTitle } from "@/components/ui/card";

// 单个 Agent 任务的执行结果（Planner 拆分出来的最小执行单元）
type WorkflowTask = {
  id: string;
  task_id: string;
  agent_name: string;
  task_type: string;
  status: string;
  summary: string;
  error?: string | null;
};

// 一次完整的工作流运行：spec 是计划，observation 是 Observer 的复核，final_answer 是综合结论
type WorkflowRun = {
  id: string;
  repo_id?: string | null;
  goal: string;
  status: string;
  spec: { workflow_id?: string; tasks?: Array<{ id: string; dependencies?: string[] }> };
  observation: { status?: string; confidence?: number; findings?: Array<{ severity?: string; message?: string }> };
  final_answer: string;
  metrics: Record<string, number | string>;
  created_at?: string | null;
  completed_at?: string | null;
  tasks: WorkflowTask[];
};

// 状态 → 中文；success 与 completed 在后端是两种写法，这里统一显示为「运行成功」
const statusLabel: Record<string, string> = {
  success: "运行成功",
  completed: "运行成功",
  failed: "运行失败",
  running: "运行中"
};

/**
 * 工作流运行页面组件。
 * @sideEffect 先加载仓库列表，再由 repoId 触发第二次请求拉取该仓库的运行记录。
 */
export default function WorkflowRunsPage() {
  const [repos, setRepos] = useState<Repository[]>([]);
  const [repoId, setRepoId] = useState("");
  const [runs, setRuns] = useState<WorkflowRun[]>([]);
  // 用 id 而不是对象保存选中项：列表刷新后对象引用会变，id 更稳定
  const [selectedId, setSelectedId] = useState("");
  const [loading, setLoading] = useState(true);

  // 第一次请求：仓库列表；失败就退化成空下拉，不阻塞页面
  useEffect(() => {
    apiGet<Repository[]>("/api/repos")
      .then((items) => {
        setRepos(items);
        setRepoId(items[0]?.id || "");
      })
      .catch(() => setRepos([]));
  }, []);

  // 第二次请求：repoId 变化时重新拉运行记录（含切换仓库的场景）。
  // 若当前选中的 id 仍在新列表里就保留，否则回落到第一条，避免右栏空白。
  useEffect(() => {
    if (!repoId) return;
    setLoading(true);
    apiGet<WorkflowRun[]>(`/api/chat/workflow-runs?repo_id=${repoId}`)
      .then((items) => {
        setRuns(items);
        setSelectedId((current) => (items.some((item) => item.id === current) ? current : items[0]?.id || ""));
      })
      .finally(() => setLoading(false));
  }, [repoId]);

  // 派生数据用 useMemo 计算，避免每次渲染都重复 find/filter
  const selected = useMemo(() => runs.find((item) => item.id === selectedId) ?? runs[0], [runs, selectedId]);
  // 兼容后端两种成功状态写法：success / completed
  const completedTasks = selected?.tasks.filter((task) => task.status === "success" || task.status === "completed").length ?? 0;
  const findings = selected?.observation.findings ?? [];

  return (
    <>
      <PageTitle title="工作流运行记录" description="按一次工程目标查看 Planner 任务图、各 Agent 执行结果、Observer 发现和最终综合结论。" />

      <section className="grid gap-3 border-y border-line bg-white px-4 py-4 md:grid-cols-[minmax(280px,1fr)_repeat(3,minmax(150px,220px))]">
        <label className="text-sm font-medium text-slate-700">
          仓库
          <select className="mt-2 w-full rounded-md border border-line bg-white px-3 py-2 outline-none focus:border-teal-500" value={repoId} onChange={(event) => setRepoId(event.target.value)}>
            {repos.map((repo) => <option key={repo.id} value={repo.id}>{repo.full_name}</option>)}
          </select>
        </label>
        <Metric icon={Workflow} label="工作流" value={runs.length} />
        <Metric icon={CheckCircle2} label="完成任务" value={completedTasks} />
        <Metric icon={AlertTriangle} label="Observer 发现" value={findings.length} />
      </section>

      <div className="mt-4 grid min-h-[560px] gap-4 lg:grid-cols-[320px_minmax(0,1fr)]">
        <aside className="border-r border-line bg-white pr-4">
          <div className="mb-3 text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">运行列表</div>
          <div className="space-y-2">
            {runs.map((run) => (
              <button key={run.id} onClick={() => setSelectedId(run.id)} className={`w-full rounded-md border px-3 py-3 text-left transition ${selected?.id === run.id ? "border-teal-300 bg-teal-50" : "border-line bg-white hover:bg-panel"}`}>
                <div className="line-clamp-2 text-sm font-semibold text-ink">{run.goal}</div>
                <div className="mt-2 flex items-center justify-between text-xs text-slate-500">
                  <span>{statusLabel[run.status] || run.status}</span>
                  <span>{run.created_at ? new Date(run.created_at).toLocaleString("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" }) : ""}</span>
                </div>
              </button>
            ))}
            {!loading && runs.length === 0 && <div className="rounded-md border border-dashed border-line p-4 text-sm text-slate-500">暂无工作流运行记录。</div>}
          </div>
        </aside>

        <main className="min-w-0">
          {selected ? (
            <div className="space-y-5">
              <section className="border-b border-line pb-4">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="rounded-full bg-teal-600 px-3 py-1 text-xs font-semibold text-white">{statusLabel[selected.status] || selected.status}</span>
                  <span className="inline-flex items-center gap-1 text-xs text-slate-500"><GitBranch className="h-3.5 w-3.5" />{selected.spec.workflow_id || selected.id.slice(0, 8)}</span>
                  <span className="inline-flex items-center gap-1 text-xs text-slate-500"><Database className="h-3.5 w-3.5" />已持久化</span>
                </div>
                <h2 className="mt-3 text-xl font-semibold text-ink">{selected.goal}</h2>
              </section>

              <section>
                <div className="mb-3 flex items-center justify-between">
                  <h3 className="font-semibold text-ink">Agent 执行链</h3>
                  <span className="text-xs text-slate-500">{completedTasks}/{selected.tasks.length} 完成</span>
                </div>
                <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-4">
                  {selected.tasks.map((task, index) => (
                    <div key={task.id} className="relative rounded-md border border-line bg-white p-3">
                      <div className="flex items-center justify-between gap-2">
                        <span className="flex h-7 w-7 items-center justify-center rounded-full bg-teal-50 text-xs font-semibold text-teal-700">{index + 1}</span>
                        <CheckCircle2 className="h-4 w-4 text-emerald-600" />
                      </div>
                      <div className="mt-3 text-sm font-semibold text-ink">{task.agent_name}</div>
                      <div className="mt-1 text-xs text-slate-500">{task.task_type}</div>
                      <p className="mt-3 text-xs leading-5 text-slate-600">{task.summary || "任务已完成"}</p>
                    </div>
                  ))}
                </div>
              </section>

              <section className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(300px,0.7fr)]">
                <div className="rounded-md border border-teal-200 bg-teal-50/70 p-4">
                  <div className="flex items-center gap-2 text-sm font-semibold text-teal-900"><Workflow className="h-4 w-4" />Synthesis 最终结论</div>
                  <p className="mt-3 text-sm leading-7 text-slate-700">{selected.final_answer}</p>
                </div>
                <div className="rounded-md border border-amber-200 bg-amber-50/70 p-4">
                  <div className="flex items-center justify-between gap-2 text-sm font-semibold text-amber-900">
                    <span className="inline-flex items-center gap-2"><AlertTriangle className="h-4 w-4" />Observer 交叉检查</span>
                    {typeof selected.observation.confidence === "number" && <span className="text-xs">置信度 {Math.round(selected.observation.confidence * 100)}%</span>}
                  </div>
                  <div className="mt-3 space-y-2">
                    {findings.map((finding, index) => <div key={index} className="text-sm leading-6 text-slate-700">{finding.message}</div>)}
                  </div>
                </div>
              </section>

              <div className="flex items-center gap-2 border-t border-line pt-3 text-xs text-slate-500">
                <Clock3 className="h-3.5 w-3.5" />
                创建于 {selected.created_at ? new Date(selected.created_at).toLocaleString("zh-CN") : "未知"}，完成后写入 workflow 与 task 两级记录。
              </div>
            </div>
          ) : <div className="text-sm text-slate-500">{loading ? "正在读取工作流记录..." : "请选择一条运行记录。"}</div>}
        </main>
      </div>
    </>
  );
}

/**
 * 顶部指标卡：图标 + 标签 + 数值。
 * @param icon 传入 lucide 图标组件本身（需大写开头才能作为组件渲染）
 */
function Metric({ icon: Icon, label, value }: { icon: typeof Workflow; label: string; value: number }) {
  return (
    <div className="flex items-center gap-3 border-l border-line pl-4">
      <Icon className="h-5 w-5 text-teal-700" />
      <div><div className="text-xs text-slate-500">{label}</div><div className="mt-1 text-xl font-semibold text-ink">{value}</div></div>
    </div>
  );
}
