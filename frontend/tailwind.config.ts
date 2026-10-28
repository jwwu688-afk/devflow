/**
 * Tailwind CSS 配置。
 *
 * 说明：这里只做增量扩展（theme.extend），不覆盖默认调色板，
 * 因此组件里仍然可以使用 bg-blue-500 这类内置颜色。
 * 自定义的语义色（ink/panel/line/accent/warn）用于保证全站配色一致。
 */
import type { Config } from "tailwindcss";

const config: Config = {
  // 扫描范围：只扫 app/components/lib 三个目录的 ts/tsx，
  // 生产构建时 Tailwind 据此做 tree-shaking，未使用的类名不会进产物。
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./lib/**/*.{ts,tsx}"],
  theme: {
    extend: {
      // 语义化色板：分别是主文字色、面板底色、分割线、强调色（品牌绿）、警告色
      colors: {
        ink: "#17212b",
        panel: "#f7f9fb",
        line: "#d8e0e8",
        accent: "#0f766e",
        warn: "#b45309"
      }
    }
  },
  // 暂未引入任何 Tailwind 插件（如 typography / forms）
  plugins: []
};

export default config;
