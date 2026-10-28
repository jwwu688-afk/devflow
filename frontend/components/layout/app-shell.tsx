/**
 * 全局外壳组件：顶部横向导航 + 内容容器。
 *
 * 职责：
 * - 维护导航菜单（navItems），新增页面时只需在这里加一项；
 * - 为所有页面提供统一的外层布局与内边距。
 * 它是纯展示组件，不持有状态、不发请求；导航用 next/link 做客户端跳转，避免整页刷新。
 */
import Link from "next/link";

// 导航项配置：href 必须与 app/ 下的目录路由一致，label 是顶栏显示文案
const navItems = [
  { href: "/", label: "Workspace" },
  { href: "/knowledge", label: "项目知识" },
  { href: "/rag", label: "RAG 工作室" },
  { href: "/repos", label: "Repos" },
  { href: "/action-drafts", label: "Drafts" },
  { href: "/workflow-runs", label: "Runs" },
  { href: "/workspaces", label: "Reports" },
  { href: "/code-graph", label: "Code Graph" },
  { href: "/evals", label: "Evals" }
];

/**
 * 页面外壳。
 * @param children 路由页面本身的内容
 */
export function AppShell({ children }: { children: React.ReactNode }) {
  // 用 href 作为 Link 的 key：路由唯一且稳定，label 改动不会导致列表整体重建
  return (
    <main className="min-h-screen">
      <div className="border-b border-line bg-white/85 px-4 py-3">
        <div className="flex flex-wrap items-center gap-2">
          <div className="mr-2 text-sm font-semibold text-ink">DevFlow AI</div>
          {navItems.map((item) => (
            <Link key={item.href} href={item.href} className="rounded-md px-3 py-1.5 text-sm text-slate-600 transition hover:bg-teal-50 hover:text-teal-800">
              {item.label}
            </Link>
          ))}
        </div>
      </div>
      <div className="w-full p-3">{children}</div>
    </main>
  );
}
