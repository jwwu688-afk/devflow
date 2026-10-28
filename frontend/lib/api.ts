/**
 * 后端接口（FastAPI，默认 http://localhost:8000）的统一访问层。
 *
 * 设计意图：
 * - 所有页面不直接写 fetch，而是调用这里的 apiGet / apiPost / apiPatch / apiDelete / apiUpload，
 *   路径统一传后端路由的相对地址（如 "/api/repos"、"/api/chat/stream"），
 *   路由与前端页面的对应关系在 backend/app/main.py 的 include_router 里定义。
 * - 错误统一在 throwApiError 里转换成 Error，页面侧只需要 try/catch 或让 React Query 接管。
 * - 泛型 T 由调用方从 lib/types.ts 或 lib/rag.ts 里取，例如 apiGet<Repository[]>("/api/repos")。
 */
export const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

/**
 * GET 请求。
 * @param path 后端相对路径，例如 "/api/repos/{repoId}/issues"
 * @returns 解析后的 JSON，类型由调用方用泛型指定
 * @sideEffect 关闭 HTTP 缓存（no-store），保证列表页每次拿到最新数据
 */
export async function apiGet<T>(path: string): Promise<T> {
  // no-store：这类数据随任务状态实时变化，浏览器缓存会导致状态看起来「卡住」
  const response = await fetch(`${API_BASE_URL}${path}`, { cache: "no-store" });
  if (!response.ok) await throwApiError(response);
  return response.json();
}

/**
 * POST 请求（JSON body）。
 * 用于创建资源或触发后端动作，例如创建仓库、发起 PR 审查、提交 RAG 检索。
 */
export async function apiPost<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  if (!response.ok) await throwApiError(response);
  return response.json();
}

/**
 * PATCH 请求（JSON body），用于局部更新，例如修改知识库配置、启用/禁用某个 chunk。
 */
export async function apiPatch<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  if (!response.ok) await throwApiError(response);
  return response.json();
}

/**
 * 文件上传（multipart/form-data），例如知识库页面上传文档。
 * 注意：这里刻意不设置 Content-Type，交给浏览器自动带 boundary。
 */
export async function apiUpload<T>(path: string, body: FormData): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    body
  });
  if (!response.ok) await throwApiError(response);
  return response.json();
}

/**
 * 以 SSE（Server-Sent Events）方式消费后端的流式 POST 响应，典型场景是 /api/chat/stream 的对话流。
 *
 * @param path 流式接口路径
 * @param body 请求体，会被 JSON 序列化
 * @param onEvent 每解析出一条 SSE 事件回调一次，event 为事件名（缺省 "message"），data 为已 JSON.parse 的对象
 * @param options.signal 组件卸载时用来中断请求，避免内存泄漏和「卸载后 setState」警告
 * @sideEffect 会持续读取 ReadableStream，直到后端结束或 signal 被 abort
 */
export async function apiPostStream(
  path: string,
  body: unknown,
  onEvent: (event: string, data: unknown) => void,
  options: { signal?: AbortSignal } = {}
): Promise<void> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: options.signal
  });
  if (!response.ok) await throwApiError(response);
  // 流式接口必须返回 body，否则说明后端/网关把响应缓冲成了普通 JSON
  if (!response.body) throw new Error("Streaming response body is not available.");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  // buffer 保存「还没收到完整事件」的残片，网络分片可能把一个事件劈成两半
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    // stream: true 让多字节中文在跨分片时不被截断
    buffer += decoder.decode(value, { stream: true });
    // SSE 以空行分隔事件，最后一段可能不完整，pop 出去留到下一轮
    const events = buffer.split("\n\n");
    buffer = events.pop() ?? "";
    for (const rawEvent of events) {
      const lines = rawEvent.split("\n");
      // "event: xxx" 行决定事件类型，没有则按 SSE 默认的 message 处理
      const event = lines.find((line) => line.startsWith("event: "))?.slice(7) ?? "message";
      // 一个事件可能有多行 data，按 SSE 规范用换行拼接
      const dataText = lines
        .filter((line) => line.startsWith("data: "))
        .map((line) => line.slice(6))
        .join("\n");
      if (dataText) onEvent(event, JSON.parse(dataText));
    }
  }
}

/**
 * DELETE 请求，用于删除仓库、会话、草稿等资源。
 */
export async function apiDelete<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, { method: "DELETE" });
  if (!response.ok) await throwApiError(response);
  return response.json();
}

/**
 * 把非 2xx 响应转成 Error 抛出，尽量还原后端给出的可读信息。
 * 后端（FastAPI）错误体通常是 {"detail": "..."} 或 {"detail": [{msg: ...}]}（校验错误），
 * 所以这里按 content-type 分流：JSON 走 readApiErrorMessage，其他按纯文本处理。
 */
async function throwApiError(response: Response): Promise<never> {
  const fallback = `${response.status} ${response.statusText}`.trim();
  const contentType = response.headers.get("content-type") ?? "";
  if (contentType.includes("application/json")) {
    let message: string | null = null;
    try {
      const payload = await response.json();
      message = readApiErrorMessage(payload);
    } catch {
      // body 不是合法 JSON（例如被网关改写），退回状态码文案
      message = null;
    }
    throw new Error(message || fallback);
  }
  const text = (await response.text()).trim();
  throw new Error(text || fallback);
}

/**
 * 从各种形状的错误体里抽取一条人类可读的 message。
 * 兼容三种来源：纯字符串、FastAPI 的 detail（字符串或校验错误数组）、以及 {message} 包装。
 */
function readApiErrorMessage(payload: unknown): string | null {
  if (typeof payload === "string") return payload;
  if (!payload || typeof payload !== "object") return null;
  const detail = (payload as { detail?: unknown }).detail;
  if (typeof detail === "string") return detail;
  // FastAPI 422 的 detail 是数组，元素形如 {loc, msg, type}
  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) => {
        if (item && typeof item === "object" && "msg" in item) return String((item as { msg: unknown }).msg);
        return typeof item === "string" ? item : null;
      })
      .filter(Boolean);
    return messages.length ? messages.join("; ") : null;
  }
  const message = (payload as { message?: unknown }).message;
  return typeof message === "string" ? message : null;
}
