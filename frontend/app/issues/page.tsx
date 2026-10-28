"use client";
/*
 * Issue 智能分析页（路由 /issues）。
 *
 * 展示内容：左侧是某个仓库的 Issue 列表，右侧展示选中 Issue 的正文与 AI 分析结果。
 * 数据来源：
 * - GET  /api/repos                     取仓库列表（当前只取第一个仓库）
 * - GET  /api/repos/{repoId}/issues     取该仓库的 Issue
 * - POST /api/issues/{issueId}/analyze  触发分析，返回分析 JSON
 * 交互状态：选中的 Issue（selected）与纯文本形式的分析结果（analysis）。
 * 分析结果后端结构较自由，这里直接序列化成 JSON 展示，便于调试与后续改结构化渲染。
 * 注："use client" 必须保持文件首行，故模块注释放在其下方。
 */

import { useEffect, useState } from "react";

import { Card, GhostButton, PageTitle, PrimaryButton } from "@/components/ui/card";
import { apiGet, apiPost } from "@/lib/api";
import type { Repository } from "@/lib/types";

// 后端 Issue 模型的精简子集：列表只需要这些字段，正文 body 用于右侧展示
type Issue = { id: string; number: number; title: string; state: string; labels: string[]; body?: string };

/**
 * Issue 分析页面组件。
 * @sideEffect 挂载时串行请求仓库与 Issue；点击按钮时发起分析请求并写入 analysis。
 */
export default function IssuesPage() {
  const [issues, setIssues] = useState<Issue[]>([]);
  // 当前选中的 Issue，null 表示尚未选中（此时禁用分析按钮）
  const [selected, setSelected] = useState<Issue | null>(null);
  // 分析结果以格式化 JSON 字符串保存，直接丢给 <pre> 渲染
  const [analysis, setAnalysis] = useState("");

  // 首屏加载：先取仓库列表，再取第一个仓库的 Issue。
  // 目前产品假定「单活跃仓库」，所以不渲染仓库选择器；依赖数组为空表示只跑一次。
  useEffect(() => {
    apiGet<Repository[]>("/api/repos").then((repos) => {
      if (repos[0]) {
        apiGet<Issue[]>(`/api/repos/${repos[0].id}/issues`).then((items) => {
          setIssues(items);
          // 默认选中第一条，避免用户进来看到空白右栏
          setSelected(items[0] ?? null);
        });
      }
    });
  }, []);

  /**
   * 对当前选中的 Issue 调用后端分析接口。
   * 请求体传空对象 {}，因为分析所需信息后端已经按 issueId 查库获得。
   */
  async function analyze() {
    if (!selected) return;
    const data = await apiPost(`/api/issues/${selected.id}/analyze`, {});
    setAnalysis(JSON.stringify(data, null, 2));
  }

  return (
    <>
      <PageTitle title="Issue 智能分析" description="自动生成 Issue 分类、优先级、复杂度、负责人建议、相似问题和下一步行动项。" />
      <div className="grid gap-4 lg:grid-cols-[380px_1fr]">
        <Card>
          <div className="flex items-center justify-between">
            <h2 className="font-semibold text-ink">Issue 列表</h2>
            <span className="rounded-full bg-panel px-2.5 py-1 text-xs text-slate-500">{issues.length} 条</span>
          </div>
          <div className="mt-4 space-y-2">
            {issues.map((issue) => (
              <GhostButton key={issue.id} onClick={() => setSelected(issue)} className={selected?.id === issue.id ? "border-teal-300 bg-teal-50 text-teal-900" : ""}>
                <div className="flex items-center justify-between gap-3">
                  <span>#{issue.number} {issue.title}</span>
                  <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs">{issue.state}</span>
                </div>
              </GhostButton>
            ))}
          </div>
        </Card>

        <Card>
          <h2 className="font-semibold text-ink">分析结果</h2>
          {selected && (
            <div className="mt-3 rounded-md bg-panel p-3 text-sm leading-6 text-slate-600">
              <div className="font-medium text-ink">#{selected.number} {selected.title}</div>
              <div className="mt-1">{selected.body || "暂无正文"}</div>
            </div>
          )}
          <PrimaryButton onClick={analyze} disabled={!selected} className="mt-4">分析当前 Issue</PrimaryButton>
          {analysis && <pre className="mt-4 max-h-[520px] overflow-auto rounded-md border border-slate-800 bg-slate-950 p-4 text-xs text-slate-100">{analysis}</pre>}
        </Card>
      </div>
    </>
  );
}
