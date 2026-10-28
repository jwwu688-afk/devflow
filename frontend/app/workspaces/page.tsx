"use client";
/*
 * 团队工作区页（路由 /workspaces）。
 *
 * 展示内容：左侧可创建 workspace（多选仓库）并选择报表范围，右侧显示已有 workspace 列表
 * 与跨仓库研发周报（Markdown）。导航栏里这一项显示为 "Reports"。
 * 数据来源：
 * - GET  /api/repos                            仓库候选
 * - GET  /api/workspaces                       已有 workspace
 * - POST /api/workspaces                       创建 workspace（{name, repo_ids}）
 * - POST /api/workspaces/multi-repo-report     生成多仓库周报
 * 交互状态：勾选的仓库、当前 workspace、日期范围、报表结果、loading。
 */

import { useEffect, useMemo, useState } from "react";

import { apiGet, apiPost } from "@/lib/api";
import type { Repository } from "@/lib/types";
import { Card, PageTitle, PrimaryButton } from "@/components/ui/card";

// 工作区：把多个仓库聚合成一个报表口径
type Workspace = {
  id: string;
  name: string;
  description?: string | null;
  repo_ids: string[];
};

// 多仓库周报响应：正文用于展示，metrics / repo_summaries 预留给后续做图表与分仓库明细
type MultiRepoReport = {
  report_markdown: string;
  metrics: Record<string, number>;
  repo_summaries: Array<Record<string, unknown>>;
};

/** Date → yyyy-MM-dd，供 <input type="date"> 使用 */
function toDateInput(date: Date) {
  return date.toISOString().slice(0, 10);
}

/**
 * 工作区页面组件。
 * @sideEffect 挂载时并行加载仓库与 workspace；创建/生成周报后会用最新数据刷新列表。
 */
export default function WorkspacesPage() {
  // 默认范围固定为「最近 7 天」，用 useMemo 冻结初始值，避免重渲染重置用户输入
  const today = useMemo(() => new Date(), []);
  const start = useMemo(() => {
    const value = new Date(today);
    value.setDate(value.getDate() - 6);
    return value;
  }, [today]);
  const [repos, setRepos] = useState<Repository[]>([]);
  const [workspaces, setWorkspaces] = useState<Workspace[]>([]);
  const [name, setName] = useState("研发团队");
  const [selectedRepos, setSelectedRepos] = useState<string[]>([]);
  // 空字符串表示「不按 workspace 汇总，而是用上方勾选的仓库」
  const [workspaceId, setWorkspaceId] = useState("");
  const [startDate, setStartDate] = useState(toDateInput(start));
  const [endDate, setEndDate] = useState(toDateInput(today));
  const [report, setReport] = useState<MultiRepoReport | null>(null);
  const [loading, setLoading] = useState(false);

  /**
   * 拉取仓库与 workspace 并重置选择状态。
   * 创建 workspace 后也会复用它刷新，所以抽成独立函数而不是直接写在 useEffect 里。
   */
  async function load() {
    // Promise.all 并行请求：两个接口互不依赖，串行会白白多一个往返
    const [repoItems, workspaceItems] = await Promise.all([apiGet<Repository[]>("/api/repos"), apiGet<Workspace[]>("/api/workspaces")]);
    setRepos(repoItems);
    setWorkspaces(workspaceItems);
    // 默认勾选前两个仓库，属于「一键生成」的体验优化
    setSelectedRepos(repoItems.slice(0, 2).map((repo) => repo.id));
    setWorkspaceId(workspaceItems[0]?.id || "");
  }

  // 首屏加载；失败时静默（catch 吞掉），页面仍可手动操作
  useEffect(() => {
    load().catch(() => undefined);
  }, []);

  /**
   * 用当前名称与勾选仓库创建 workspace，然后刷新列表让新项立刻出现。
   * @sideEffect 写入后端并重新 load()。
   */
  async function createWorkspace() {
    if (!name.trim()) return;
    setLoading(true);
    try {
      await apiPost<Workspace>("/api/workspaces", { name, repo_ids: selectedRepos });
      await load();
    } finally {
      setLoading(false);
    }
  }

  /**
   * 生成多仓库周报。
   * 二选一策略：选了 workspace 就只传 workspace_id（repo_ids 置空），
   * 否则按勾选的 repo_ids 直接聚合，避免后端出现「两份来源冲突」的歧义。
   */
  async function generateReport() {
    setLoading(true);
    try {
      const data = await apiPost<MultiRepoReport>("/api/workspaces/multi-repo-report", {
        workspace_id: workspaceId || undefined,
        repo_ids: workspaceId ? [] : selectedRepos,
        start_date: startDate,
        end_date: endDate
      });
      setReport(data);
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <PageTitle title="团队工作区" description="把多个仓库聚合成 workspace，生成跨仓库研发周报和风险概览。" />
      <div className="grid gap-4 lg:grid-cols-[380px_1fr]">
        <Card>
          <label className="text-sm font-medium text-slate-700">
            Workspace 名称
            <input className="mt-2 w-full rounded-md border border-line px-3 py-2 outline-none" value={name} onChange={(event) => setName(event.target.value)} />
          </label>
          <div className="mt-4 text-sm font-medium text-slate-700">仓库</div>
          <div className="mt-2 grid gap-2">
            {repos.map((repo) => (
              <label key={repo.id} className="flex items-center gap-2 rounded-md border border-line px-3 py-2 text-sm">
                <input
                  type="checkbox"
                  checked={selectedRepos.includes(repo.id)}
                  onChange={(event) => {
                    setSelectedRepos((current) => (event.target.checked ? [...current, repo.id] : current.filter((id) => id !== repo.id)));
                  }}
                />
                {repo.full_name}
              </label>
            ))}
          </div>
          <PrimaryButton className="mt-4" disabled={loading || selectedRepos.length === 0} onClick={createWorkspace}>
            创建 Workspace
          </PrimaryButton>

          <div className="mt-6 grid gap-3 border-t border-line pt-4">
            <label className="text-sm font-medium text-slate-700">
              报表范围
              <select className="mt-2 w-full rounded-md border border-line px-3 py-2" value={workspaceId} onChange={(event) => setWorkspaceId(event.target.value)}>
                <option value="">使用上方勾选仓库</option>
                {workspaces.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.name}
                  </option>
                ))}
              </select>
            </label>
            <input className="rounded-md border border-line px-3 py-2" type="date" value={startDate} onChange={(event) => setStartDate(event.target.value)} />
            <input className="rounded-md border border-line px-3 py-2" type="date" value={endDate} onChange={(event) => setEndDate(event.target.value)} />
            <PrimaryButton disabled={loading || (!workspaceId && selectedRepos.length === 0)} onClick={generateReport}>
              生成多仓库周报
            </PrimaryButton>
          </div>
        </Card>

        <div className="grid gap-4">
          <Card>
            <h2 className="font-semibold text-ink">Workspace 列表</h2>
            <div className="mt-3 grid gap-2">
              {workspaces.map((item) => (
                <div key={item.id} className="rounded-md border border-line bg-panel px-3 py-2 text-sm text-slate-700">
                  {item.name} · {item.repo_ids.length} 个仓库
                </div>
              ))}
              {workspaces.length === 0 && <div className="text-sm text-slate-500">暂无 workspace。</div>}
            </div>
          </Card>
          <Card>
            <h2 className="font-semibold text-ink">多仓库周报</h2>
            {report ? (
              <pre className="mt-4 max-h-[620px] overflow-auto rounded-md border border-slate-800 bg-slate-950 p-4 text-sm leading-7 text-slate-100">{report.report_markdown}</pre>
            ) : (
              <div className="mt-4 rounded-md border border-line bg-panel p-4 text-sm text-slate-600">创建 workspace 或勾选仓库后生成周报。</div>
            )}
          </Card>
        </div>
      </div>
    </>
  );
}
