/**
 * Next.js App Router 的根布局。
 *
 * 职责：注入全局样式与页面 metadata，并用 AppShell 给所有路由套上统一的
 * 侧边栏 / 顶栏骨架，所以各个 page.tsx 里不需要重复写导航结构。
 * 该组件是服务端组件，不持有状态、不产生副作用。
 *
 * 注意：JSX 内部不使用 {/* *\/} 形式注释，统一把注释写在 JSX 之外，
 * 避免注释节点变成额外的 JSX 表达式容器。
 */
import type { Metadata } from "next";
import "./globals.css";
import { AppShell } from "@/components/layout/app-shell";

// SEO / 浏览器标签页信息，Next 会据此生成 <head> 中的 title 与 description
export const metadata: Metadata = {
  title: "DevFlow AI",
  description: "PR / Issue intelligent collaboration agent for engineering teams"
};

/**
 * 根布局组件。
 * @param children 当前路由匹配到的页面内容
 */
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  // lang="zh-CN" 让浏览器与读屏软件按中文处理排版和断词
  return (
    <html lang="zh-CN">
      <body>
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
