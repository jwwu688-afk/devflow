"""推荐提问 Agent。

职责边界：根据仓库当前状态（Issue / PR / CI）生成「用户接下来可能会问」的聊天建议。
它不分析问题时也不回答问题，只生产**提示语字符串**，属于交互层的辅助功能。

设计要点（都为了一个目标：前端永远能拿到可直接点击的建议）：
- 完全不调用 LLM，纯模板 + 数据驱动，零成本、零延迟，可以在页面加载时随时调用；
- 具体对象（失败的 CI、Open PR、Open Issue）优先，泛化建议兜底，保证建议「有上下文」；
- exclude 里是用户已经看过/已展示过的建议，要尽量避开；但如果避开后一条不剩，
  会放弃 exclude 重新给一遍（见 run() 里的降级分支），宁可重复也不给空列表。
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from app.services.agents.base import BaseAgent


class RecommendationAgent(BaseAgent):
    """基于仓库快照生成上下文相关的推荐提问。"""

    name = "recommendation_agent"
    description = "根据当前仓库的 Issue、PR 和 CI 状态推荐上下文相关的聊天提问。"
    available_tools = ["repository_snapshot"]

    async def run(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """挑选最终返回的建议列表。

        输入：issues / pull_requests / workflow_runs 为原始快照（可为空）；
              exclude 为需要避开的建议（前端已展示过的）；
              limit 为期望条数。
        输出：{agent, suggestions[], source_counts{}}。
              source_counts 一并返回，让前端/排查时能看出「这次建议是基于多少数据生成的」，
              空仓库与数据缺失导致的建议质量差异可以据此区分。
        """
        # 夹在 [4, 5]：UI 的推荐位固定是 4~5 个槽，多了放不下、少了页面显空。
        # 用 or 5 是为了让 limit=0/None 也走默认值而不是被夹成 4
        limit = min(max(int(input_data.get("limit") or 5), 4), 5)
        # 统一归一化后比较，避免「大小写/多余空格不同」被当成新建议重复推给用户
        excluded = {self._normalize(item) for item in input_data.get("exclude") or []}
        candidates = self._build_candidates(input_data)

        selected: list[str] = []
        seen = set(excluded)
        for prompt in candidates:
            normalized = self._normalize(prompt)
            # 空字符串直接跳过：模板里标题为空时会拼出 "总结 PR #12 的风险" 这类看似正常、
            # 但归一化后可能与前一条重合的结果，统一在入口挡掉
            if not normalized or normalized in seen:
                continue
            selected.append(prompt)
            seen.add(normalized)
            if len(selected) >= limit:
                break

        # 降级路径：候选全被 exclude 吃掉时，放弃去重再来一遍。
        # 这是刻意的取舍——推荐位空着比重复展示更糟（用户会以为功能坏了）
        if not selected and excluded:
            for prompt in candidates:
                normalized = self._normalize(prompt)
                if not normalized:
                    continue
                selected.append(prompt)
                if len(selected) >= limit:
                    break

        return {
            "agent": self.name,
            "suggestions": selected,
            "source_counts": {
                "issues": len(input_data.get("issues") or []),
                "pull_requests": len(input_data.get("pull_requests") or []),
                "workflow_runs": len(input_data.get("workflow_runs") or []),
            },
        }

    def _build_candidates(self, input_data: dict[str, Any]) -> list[str]:
        """按「具体 → 泛化」的顺序生成候选建议池。

        顺序很重要：run() 是从头取前 limit 条，所以越靠前越会被展示。
        因此失败的 CI 排最前（最紧急），其次是 Open PR / Open Issue，最后才是通用建议。
        """
        # 统一按时间倒序：用户最关心的是最近发生的事，旧数据排后面自然被 limit 截掉
        issues = sorted(input_data.get("issues") or [], key=self._sort_key, reverse=True)
        prs = sorted(input_data.get("pull_requests") or [], key=self._sort_key, reverse=True)
        runs = sorted(input_data.get("workflow_runs") or [], key=self._sort_key, reverse=True)
        # 冷启动：仓库还没同步到任何数据时，只能给「如何开始使用」这类引导性建议，
        # 否则会生成一堆指向不存在对象的空建议
        if not issues and not prs and not runs:
            return [
                "概览这个新仓库的当前状态",
                "查看这个项目的代码结构",
                "我该如何同步这个仓库的 Issue、PR 和 CI",
                "帮我建立项目记忆和团队分工",
                "生成初始化项目检查清单",
                "这个仓库接下来应该先配置什么",
                "帮我规划第一次项目同步和分析",
            ]

        # 只围绕「还活着的东西」给建议：已关闭的 Issue / 已合并的 PR 再问风险没有意义
        open_issues = [item for item in issues if str(item.get("state") or "").lower() == "open"]
        open_prs = [item for item in prs if str(item.get("state") or "").lower() == "open"]
        failed_runs = [item for item in runs if str(item.get("conclusion") or "").lower() == "failure"]

        candidates: list[str] = []

        # 每个失败 workflow 生成 4 条不同角度的排查建议（定位/修复步骤/与 PR 关联/checklist），
        # 覆盖用户「从哪下手」的几种常见意图；最多取 5 个 workflow，防止建议池被单一维度淹没
        for run in failed_runs[:5]:
            name = self._short(run.get("name") or "最近 CI")
            candidates.extend(
                [
                    f"定位 {name} 失败原因",
                    f"把 {name} 的失败日志整理成修复步骤",
                    f"分析 {name} 失败是否和最近 PR 有关",
                    f"给 {name} 生成排查 checklist",
                ]
            )

        # 没有失败但有运行记录：转为「健康度巡检」口径，避免用户以为 CI 出问题了
        if not failed_runs and runs:
            latest = self._short(runs[0].get("name") or "最近 CI")
            candidates.extend(
                [
                    f"检查 {latest} 的 CI 状态",
                    "最近 CI 有没有潜在风险",
                    "总结最近 workflow 的健康状况",
                    "如果 CI 失败，下一步应该先看哪里",
                ]
            )

        # PR 建议用「编号 + 标题」定位对象：只给标题时同名 PR 容易混淆
        for pr in open_prs[:5]:
            number = pr.get("number")
            # number 缺失（数据不全）时退化成 "最新 PR"，保证句子仍然通顺可读
            label = f"PR #{number}" if number else "最新 PR"
            title = self._short(pr.get("title") or "")
            suffix = f"：{title}" if title else ""
            candidates.extend(
                [
                    f"总结 {label}{suffix} 的风险",
                    f"检查 {label} 需要补哪些测试",
                    f"给 {label} 生成合并前 checklist",
                    f"{label} 是否会影响 CI 或发布",
                ]
            )

        for issue in open_issues[:5]:
            number = issue.get("number")
            label = f"Issue #{number}" if number else "最新 Issue"
            title = self._short(issue.get("title") or "")
            suffix = f"：{title}" if title else ""
            candidates.extend(
                [
                    f"{label}{suffix} 应该分给谁",
                    f"分析 {label} 的优先级和复杂度",
                    f"给 {label} 写一条澄清评论草稿",
                    f"把 {label} 拆成可执行任务",
                ]
            )

        # 以下是「不绑定具体对象」的泛化建议，作为兜底：
        # 具体建议可能因 limit 截断而全被前面的条目占满，这些保证覆盖面不至于太窄
        if failed_runs:
            candidates.extend(
                [
                    "最近 CI 为什么失败",
                    # 有 PR 数据时才把 CI 和 PR 串起来，否则给一条纯排查建议
                    "把失败 CI 和相关 PR 串起来分析" if prs else "把失败 CI 整理成排查步骤",
                ]
            )
        if prs:
            candidates.extend(
                [
                    "总结这个 PR 的风险",
                    "按风险排序当前打开的 PR",
                    "哪些 PR 适合今天合并",
                    "生成一份合并前风险检查清单",
                ]
            )
        if issues:
            candidates.extend(
                [
                    "这个 Issue 应该分给谁",
                    "按优先级整理当前打开的 Issue",
                    "哪些 Issue 需要补充复现信息",
                    "根据团队成员能力推荐 Issue 负责人",
                    "分析打开时间最久的问题",
                ]
            )
        # 只有同时有 Issue 和 PR 时，「站会摘要」这类跨对象的建议才成立
        if issues and prs:
            candidates.extend(["把当前 Issue 和 PR 做成站会摘要", "找出最可能拖慢交付的事项"])

        # 通用建议池：任何仓库状态都适用，保证候选数量足够撑满 limit
        candidates.extend(
            [
                "生成本周研发周报",
                "概览当前仓库状态",
                "找出最值得优先处理的 3 件事",
                "生成一份今日工程状态摘要",
                "梳理当前项目的主要阻塞点",
                "看一下这个项目的代码结构",
                "搜索当前代码里的测试配置",
                "检查最近变更可能带来的风险",
                "帮我制定下一步排查计划",
                "哪些任务需要先找负责人确认",
                "生成一份发给团队的进展说明",
                "检查是否有需要安全确认的写操作",
                "当前仓库有哪些知识库内容可用",
                "本周研发周报应该包含哪些重点",
                "总结当前项目的质量风险",
                "列出今天可以快速推进的事项",
            ]
        )
        return self._dedupe(candidates)

    def _dedupe(self, values: list[str]) -> list[str]:
        """按归一化结果去重，保留首次出现的原始写法（可读性更好的那个）。"""
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = self._normalize(value)
            if not normalized or normalized in seen:
                continue
            result.append(value)
            seen.add(normalized)
        return result

    def _normalize(self, value: Any) -> str:
        """归一化：去首尾空白 + 合并内部连续空白 + 转小写。

        只用于比较，不用于展示——展示仍用原串，否则中文没问题但英文缩写会被压成小写。
        """
        return re.sub(r"\s+", " ", str(value or "").strip().lower())

    def _short(self, value: Any, max_length: int = 22) -> str:
        """截断过长标题。

        22 字符是经验值：超过这个长度，建议句在推荐位里会被 CSS 折行或省略，
        与其让前端截断（会切在半个词上），不如这里主动加省略号。
        """
        text = re.sub(r"\s+", " ", str(value or "").strip())
        if len(text) <= max_length:
            return text
        return text[:max_length].rstrip() + "..."  # rstrip 避免截断处留下空格再跟省略号

    def _sort_key(self, item: dict[str, Any]) -> float:
        """把多种时间字段统一成可比较的 float 时间戳。

        按 updated_at → created_at → merged_at → closed_at 的顺序取第一个可用值：
        不同资源（issue/pr/run）的时间字段并不一致，用「最近被碰过的时间」排序最贴近用户关注度。
        全都解析不出来时返回 0.0，让这类脏数据沉到列表末尾而不是抛异常中断整个建议生成。
        """
        for key in ["updated_at", "created_at", "merged_at", "closed_at"]:
            value = item.get(key)
            # 上游可能已把字符串解析成 datetime，直接取 timestamp 最省事
            if isinstance(value, datetime):
                return value.timestamp()
            if isinstance(value, str) and value:
                try:
                    # GitHub 返回的时间是 "...Z" 结尾，Python 3.11 前 fromisoformat 不认 Z，
                    # 所以手动替换成 +00:00
                    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
                except ValueError:
                    # 单个字段格式异常就试下一个字段，不因为一条脏数据影响排序
                    continue
        return 0.0
