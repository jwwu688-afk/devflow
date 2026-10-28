"use client";
/*
 * Agent 对话页（路由 /chat）。
 *
 * 展示内容：仓库选择器 + 问题输入框 + 示例问题快捷按钮 + 流式输出区域。
 * 数据来源：
 * - GET  /api/repos                取仓库下拉选项
 * - POST /api/chat/stream          SSE 流式对话（后端 chat 路由，支持工具调用）
 * 交互状态：repoId（选中的仓库）、message（输入内容）、result（累积的输出）、loading（请求中）。
 * 流式约定：后端逐条推送 "delta" 事件（增量文本）与 "final" 事件（完整结果），
 * 这里遇到 final 时用完整 JSON 覆盖，保证最终展示的是权威结果。
 */

import { useEffect, useState } from "react";

import { Card, PageTitle, PrimaryButton } from "@/components/ui/card";
import { apiGet, apiPostStream } from "@/lib/api";
import type { Repository } from "@/lib/types";

// 示例问题：覆盖 CI 排查、PR 风险、Issue 归属、周报、代码阅读几个典型场景
const samples = [
  "Why did the latest CI fail?",
  "Summarize the risk in the latest PR",
  "Who should own the latest Issue?",
  "Generate this week's engineering report",
  "Read backend/app/services/agents/chat_agent.py and explain tool calling",
];

/**
 * 对话页面组件。
 * @sideEffect 挂载时拉取仓库列表并默认选中第一个；发送时建立 SSE 流并持续 setState。
 */
export default function ChatPage() {
  const [repos, setRepos] = useState<Repository[]>([]);
  const [repoId, setRepoId] = useState("");
  const [message, setMessage] = useState("Why did the latest CI fail?");
  // result 累积流式文本；收到 final 事件后会被完整结果覆盖
  const [result, setResult] = useState("");
  // loading 期间禁用按钮，避免重复发起同一轮对话
  const [loading, setLoading] = useState(false);

  // 首屏加载仓库列表。用函数式 setState 只在「还没有选中值」时补默认值，
  // 避免竞态覆盖用户已经手动选择的仓库。
  useEffect(() => {
    apiGet<Repository[]>("/api/repos").then((items) => {
      setRepos(items);
      setRepoId((current) => current || items[0]?.id || "");
    });
  }, []);

  /**
   * 发送问题，消费 SSE 流。
   * 请求体约定（后端 /api/chat/stream）：repo_id 指定上下文仓库，
   * context.time_range 限制检索时间窗，messages 为历史对话（此页面暂不带历史）。
   * @sideEffect 会不断追加 result，结束后关闭 loading。
   */
  async function send() {
    setLoading(true);
    setResult("");
    try {
      await apiPostStream("/api/chat/stream", { repo_id: repoId, message, context: { time_range: "7d", messages: [] } }, (event, rawData) => {
        // delta：模型增量输出，直接追加形成打字机效果
        if (event === "delta") {
          const data = rawData as { content?: string };
          if (data.content) setResult((current) => current + data.content);
        }
        // final：本轮结束，用完整结果替换累积文本，防止 delta 丢包导致内容缺失
        if (event === "final") {
          setResult(JSON.stringify(rawData, null, 2));
        }
      });
    } finally {
      // 无论成功还是抛错都要复位，否则按钮会一直处于禁用态
      setLoading(false);
    }
  }

  return (
    <>
      <PageTitle
        title="Agent Chat"
        description="Use native tool calling to inspect code, search memory, run specialized analysis, and draft safe actions."
      />
      <Card>
        <select
          className="mb-3 w-full rounded-md border border-line bg-white px-3 py-2 text-sm"
          value={repoId}
          onChange={(event) => setRepoId(event.target.value)}
        >
          {repos.map((repo) => (
            <option key={repo.id} value={repo.id}>{repo.full_name}</option>
          ))}
        </select>
        <textarea
          className="h-32 w-full rounded-md border border-line bg-white px-3 py-3 text-sm leading-6 outline-none transition focus:border-teal-500 focus:ring-2 focus:ring-teal-100"
          value={message}
          onChange={(event) => setMessage(event.target.value)}
        />

        <div className="mt-3 flex flex-wrap gap-2 text-xs text-slate-500">
          {samples.map((sample) => (
            <button
              key={sample}
              onClick={() => setMessage(sample)}
              className="rounded-full border border-line bg-panel px-3 py-1.5 transition hover:border-teal-200 hover:bg-teal-50"
            >
              {sample}
            </button>
          ))}
        </div>

        <PrimaryButton onClick={send} disabled={loading || !repoId || !message.trim()} className="mt-4">
          {loading ? "Analyzing..." : "Ask Agent"}
        </PrimaryButton>

        {result && <pre className="mt-4 max-h-[560px] overflow-auto rounded-md border border-slate-800 bg-slate-950 p-4 text-xs text-slate-100">{result}</pre>}
      </Card>
    </>
  );
}
