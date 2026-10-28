"use client";
/*
 * PR Review 助手页（路由 /pull-requests）。
 *
 * 展示内容：左栏是仓库的 Pull Request 列表；右栏展示选中 PR 的变更文件（增删行数）
 * 以及点击「生成 PR 分析」后后端返回的 Review 报告。
 * 数据来源：
 * - GET  /api/repos                              取仓库列表（当前只用第一个仓库）
 * - GET  /api/repos/{repoId}/pull-requests       取 PR 列表（含 changed files）
 * - POST /api/pull-requests/{prId}/analyze       触发 AI Review，返回报告 JSON
 * 交互状态：选中的 PR（selected）与分析结果文本（analysis）。
 * 分析结果结构随模型输出变化，这里统一 JSON.stringify 后展示，避免前端猜测字段。
 */

import { useEffect, useState } from "react";

import { Card, GhostButton, PageTitle, PrimaryButton } from "@/components/ui/card";
import { apiGet, apiPost } from "@/lib/api";
import type { Repository } from "@/lib/types";

// PR 精简模型：files 用于在右栏展示改动规模，也是后端生成 Review 的输入
type PullRequest = {
  id: string;
  number: number;
  title: string;
  state: string;
  files: { filename: string; additions: number; deletions: number }[];
};

/**
 * PR Review 页面组件。
 * @sideEffect 挂载时先拉仓库再拉 PR；点击按钮时请求分析接口并写入 analysis。
 */
export default function PullRequestsPage() {
  const [prs, setPrs] = useState<PullRequest[]>([]);
  // 当前选中的 PR，null 时右栏为空且禁用分析按钮
  const [selected, setSelected] = useState<PullRequest | null>(null);
  const [analysis, setAnalysis] = useState("");

  // 首屏加载：串行两次请求（先仓库后 PR），因为 PR 接口需要 repoId。
  // 依赖数组为空，只在挂载时执行一次。
  useEffect(() => {
    apiGet<Repository[]>("/api/repos").then((repos) => {
      if (repos[0]) {
        apiGet<PullRequest[]>(`/api/repos/${repos[0].id}/pull-requests`).then((items) => {
          setPrs(items);
          // 默认选中第一个 PR，让用户一进页面就能看到内容
          setSelected(items[0] ?? null);
        });
      }
    });
  }, []);

  /**
   * 请求后端对当前 PR 生成 Review 报告。
   * 请求体为空对象：diff 与文件信息由后端按 prId 读取，前端不重复传输。
   */
  async function analyze() {
    if (!selected) return;
    const data = await apiPost(`/api/pull-requests/${selected.id}/analyze`, {});
    setAnalysis(JSON.stringify(data, null, 2));
  }

  return (
    <>
      <PageTitle title="PR Review 助手" description="读取 changed files 和 diff，生成 PR 摘要、风险点、Review Checklist 与测试建议。" />
      <div className="grid gap-4 lg:grid-cols-[380px_1fr]">
        <Card>
          <div className="flex items-center justify-between">
            <h2 className="font-semibold text-ink">Pull Requests</h2>
            <span className="rounded-full bg-panel px-2.5 py-1 text-xs text-slate-500">{prs.length} 个</span>
          </div>
          <div className="mt-4 space-y-2">
            {prs.map((pr) => (
              <GhostButton key={pr.id} onClick={() => setSelected(pr)} className={selected?.id === pr.id ? "border-teal-300 bg-teal-50 text-teal-900" : ""}>
                #{pr.number} {pr.title}
              </GhostButton>
            ))}
          </div>
        </Card>

        <Card>
          <h2 className="font-semibold text-ink">Review 报告</h2>
          {selected && (
            <div className="mt-3 rounded-md bg-panel p-3">
              <div className="text-sm font-medium text-ink">#{selected.number} {selected.title}</div>
              <div className="mt-3 grid gap-2">
                {selected.files.map((file) => (
                  <div key={file.filename} className="flex items-center justify-between rounded-md bg-white px-3 py-2 text-sm text-slate-600">
                    <span className="truncate">{file.filename}</span>
                    <span className="ml-3 whitespace-nowrap text-xs text-slate-500">+{file.additions} / -{file.deletions}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
          <PrimaryButton onClick={analyze} disabled={!selected} className="mt-4">生成 PR 分析</PrimaryButton>
          {analysis && <pre className="mt-4 max-h-[520px] overflow-auto rounded-md border border-slate-800 bg-slate-950 p-4 text-xs text-slate-100">{analysis}</pre>}
        </Card>
      </div>
    </>
  );
}
