/**
 * 基础 UI 原子组件集合：卡片容器、页面标题、主按钮、次按钮。
 *
 * 设计意图：把 Tailwind 类名收敛到一处，页面只关心结构；
 * 所有组件都接受 className 并通过 clsx 合并，允许调用方做局部样式覆盖（后写的类会覆盖前写的）。
 * 这些组件没有任何状态与副作用，均为纯展示组件。
 */
import { clsx } from "clsx";

/**
 * 通用卡片容器，页面里的每个信息区块都套一层，保证圆角/描边/阴影一致。
 * @param className 追加的自定义类名，用于调整栅格或内边距
 */
export function Card({ className, children }: { className?: string; children: React.ReactNode }) {
  return (
    <section className={clsx("rounded-lg border border-line/80 bg-white/90 p-5 shadow-sm shadow-slate-200/60 backdrop-blur", className)}>
      {children}
    </section>
  );
}

/**
 * 页面头部标题区（品牌标识 + 主标题 + 说明文案）。
 * description 用于向使用者解释这个页面的数据来源与操作含义。
 */
export function PageTitle({ title, description }: { title: string; description: string }) {
  return (
    <div className="mb-6 flex flex-col gap-2 border-b border-line/80 pb-5">
      <div className="text-xs font-medium uppercase tracking-[0.18em] text-accent">DevFlow AI</div>
      <h1 className="text-2xl font-semibold tracking-normal text-ink md:text-3xl">{title}</h1>
      <p className="max-w-3xl text-sm leading-6 text-slate-600">{description}</p>
    </div>
  );
}

/**
 * 主操作按钮（强调色实心）。
 * className 放在后面，是为了让调用方可以覆盖默认背景色；
 * ...props 透传 onClick / disabled / type 等原生属性。
 */
export function PrimaryButton({ children, className, ...props }: React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      className={clsx(
        "inline-flex items-center justify-center rounded-md bg-accent px-4 py-2 text-sm font-medium text-white shadow-sm shadow-teal-900/10 transition hover:bg-teal-700 disabled:cursor-not-allowed disabled:opacity-60",
        className
      )}
      {...props}
    >
      {children}
    </button>
  );
}

/**
 * 次要操作按钮（描边 + 左对齐 + 占满宽度），常用于侧栏列表项或「取消/重置」类动作。
 */
export function GhostButton({ children, className, ...props }: React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      className={clsx(
        "w-full rounded-md border border-line bg-white px-3 py-2 text-left text-sm text-slate-700 transition hover:border-teal-200 hover:bg-teal-50/70 hover:text-teal-900",
        className
      )}
      {...props}
    >
      {children}
    </button>
  );
}
