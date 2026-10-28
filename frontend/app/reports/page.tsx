"use client";
/*
 * 研发周报生成页（路由 /reports）。
 *
 * 展示内容：左侧是生成表单（仓库 + 起止日期），右侧展示后端返回的 Markdown 周报。
 * 数据来源：
 * - GET  /api/repos               取仓库下拉选项
 * - POST /api/reports/weekly      入参 {repo_id, start_date, end_date}，返回 {report_markdown}
 * 交互状态：repoId / startDate / endDate 三个表单值，report 为结果，loading 控制按钮。
 * 默认时间范围是「最近 7 天」：周报通常按周看，用户进来即可直接点生成。
 */

import { useEffect, useMemo, useState } from "react";

import { Card, PageTitle, PrimaryButton } from "@/components/ui/card";
import { apiGet, apiPost } from "@/lib/api";
import type { Repository } from "@/lib/types";

// 周报接口的响应：后端直接吐 Markdown 文本，前端不做二次渲染，原样展示便于复制
type WeeklyReportResponse = {
  report_markdown: string;
};

/** Date → yyyy-MM-dd，配合 <input type="date"> 的取值格式 */
function toDateInput(date: Date) {
  return date.toISOString().slice(0, 10);
}

/**
 * 周报生成页面组件。
 * @sideEffect 挂载时拉取仓库列表；点击生成时 POST 周报接口。
 */
export default function ReportsPage() {
  // today / weekStart 只在首次渲染时算一次（useMemo 空依赖），
  // 避免每次重渲染都生成新的 Date 导致依赖变化、把用户改过的日期重置掉。
  const today = useMemo(() => new Date(), []);
  const weekStart = useMemo(() => {
    const date = new Date(today);
    date.setDate(date.getDate() - 6);
    return date;
  }, [today]);

  const [repos, setRepos] = useState<Repository[]>([]);
  const [repoId, setRepoId] = useState("");
  const [startDate, setStartDate] = useState(toDateInput(weekStart));
  const [endDate, setEndDate] = useState(toDateInput(today));
  const [report, setReport] = useState("");
  const [loading, setLoading] = useState(false);

  // 仓库加载失败时置空数组：页面仍可用（下拉为空，生成按钮禁用），不打断整体布局。
  useEffect(() => {
    apiGet<Repository[]>("/api/repos")
      .then((items) => {
        setRepos(items);
        setRepoId(items[0]?.id || "");
      })
      .catch(() => setRepos([]));
  }, []);

  /**
   * 按当前选中的仓库与日期范围生成周报。
   * 日期不做本地校验，交由后端判断范围是否合法并返回错误。
   * @sideEffect 写入 report 并切换 loading。
   */
  async function generateReport() {
    if (!repoId) return;
    setLoading(true);
    try {
      const data = await apiPost<WeeklyReportResponse>("/api/reports/weekly", {
        repo_id: repoId,
        start_date: startDate,
        end_date: endDate
      });
      setReport(data.report_markdown);
    } finally {
      // 失败也要解除禁用，否则用户会卡在「生成中...」
      setLoading(false);
    }
  }

  return (
    <>
      <PageTitle title="研发周报生成" description="按时间范围统计 Issues、PRs、Merged PRs、Open PRs 和失败 CI，生成适合发给团队负责人的 Markdown 周报。" />

      <div className="grid gap-4 lg:grid-cols-[360px_1fr]">
        <Card>
          <div className="grid gap-4">
            <label className="text-sm font-medium text-slate-700">
              仓库
              <select
                className="mt-2 w-full rounded-md border border-line bg-white px-3 py-2 outline-none transition focus:border-teal-500 focus:ring-2 focus:ring-teal-100"
                value={repoId}
                onChange={(event) => setRepoId(event.target.value)}
              >
                {repos.map((repo) => (
                  <option key={repo.id} value={repo.id}>
                    {repo.full_name}
                  </option>
                ))}
              </select>
            </label>

            <label className="text-sm font-medium text-slate-700">
              开始日期
              <input
                type="date"
                className="mt-2 w-full rounded-md border border-line bg-white px-3 py-2 outline-none transition focus:border-teal-500 focus:ring-2 focus:ring-teal-100"
                value={startDate}
                onChange={(event) => setStartDate(event.target.value)}
              />
            </label>

            <label className="text-sm font-medium text-slate-700">
              结束日期
              <input
                type="date"
                className="mt-2 w-full rounded-md border border-line bg-white px-3 py-2 outline-none transition focus:border-teal-500 focus:ring-2 focus:ring-teal-100"
                value={endDate}
                onChange={(event) => setEndDate(event.target.value)}
              />
            </label>

            <PrimaryButton onClick={generateReport} disabled={!repoId || loading}>
              {loading ? "生成中..." : "生成周报"}
            </PrimaryButton>
          </div>
        </Card>

        <Card>
          <h2 className="font-semibold text-ink">Markdown 周报</h2>
          {report ? (
            <pre className="mt-4 max-h-[640px] overflow-auto rounded-md border border-slate-800 bg-slate-950 p-4 text-sm leading-7 text-slate-100">{report}</pre>
          ) : (
            <div className="mt-4 rounded-md border border-line bg-panel p-4 text-sm text-slate-600">选择仓库和时间范围后生成周报。</div>
          )}
        </Card>
      </div>
    </>
  );
}
