"""研发周报生成 Agent。

职责边界：把「已经取好的」原始数据（issue 列表、PR 列表、CI run、指标）聚合成一份
Markdown 周报。它自己**不去 GitHub 拉数据**（那是 github_retrieval_agent 的事），
也**不调用 LLM**——整份报告是纯模板拼接。

为什么不用模型：周报的骨架（概览 / 完成 / 进行中 / 风险 / 下周建议）是固定的，
模板化输出可复现、零成本、不会编造数字；代价是语言不够灵活，只能做统计口径的叙述。
这个取舍意味着：任何「措辞更像人写的」需求，都不该在这里塞 prompt，而应由
synthesis_agent 在上层润色。
"""

from typing import Any

from app.services.agents.base import BaseAgent


class ReportAgent(BaseAgent):
    """按固定模板把仓库活动数据渲染成 Markdown 周报。"""

    name = "report_agent"
    description = "根据仓库活动生成工程周报。"
    # 注意：这里列出的工具是「数据本应由这些工具取得」的声明，
    # run() 并不真正调用它们——数据由编排层预先抓好后通过 input_data 传入。
    available_tools = ["list_issues_by_time_range", "list_prs_by_time_range", "list_failed_workflow_runs"]

    async def run(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """渲染周报。

        输入：input_data 中的 issues / pull_requests / workflow_runs 为原始列表，
              metrics 为上游已算好的指标（可能为空，缺项时回退成现场统计）。
              context 提供 repo_name、start_date、end_date、repo_id。
        输出：{"report_markdown": str}，唯一的 key，便于前端直接渲染。
        """
        # 全部用 get(..., []) 兜底：某类数据缺失时仍要出一份完整周报，
        # 而不是因为 KeyError 让整次报告失败
        issues = input_data.get("issues", [])
        prs = input_data.get("pull_requests", [])
        ci_runs = input_data.get("workflow_runs", [])
        metrics = input_data.get("metrics", {})
        # 以下四个派生列表是周报所有数字的来源，统一在这里算一次，避免模板里重复遍历
        failed_ci = [run for run in ci_runs if run.get("conclusion") == "failure"]
        # 用 merged_at 判断「已合并」而非 state：GitHub 合并后的 PR state 仍是 closed
        merged_prs = [pr for pr in prs if pr.get("merged_at")]
        open_prs = [pr for pr in prs if pr.get("state") == "open"]
        closed_issues = [issue for issue in issues if issue.get("state") == "closed"]
        # repo_name 缺失时退化到 "Repo <id>"，保证标题永远不为空
        repo_name = context.get("repo_name") or f"Repo {context.get('repo_id', '')}".strip()
        # 起止日期缺失时渲染成 " 至 "，虽然难看但不会抛错
        date_range = f"{context.get('start_date', '')} 至 {context.get('end_date', '')}".strip()
        # 只取前 8 个编号：周报是摘要，列全量会让「本周完成」一段失控
        merged_pr_numbers = ", ".join(f"#{pr.get('number')}" for pr in merged_prs[:8]) or "暂无"
        closed_issue_numbers = ", ".join(f"#{issue.get('number')}" for issue in closed_issues[:8]) or "暂无"
        # 整个 markdown 用一次表达式拼完（而不是多次 +=），保证不会有中间态字符串泄漏出去
        markdown = (
            f"# {repo_name} 研发周报\n\n"
            f"> 时间范围：{date_range}\n\n"
            "## 数据概览\n"
            # 每个数字都是「优先用上游 metrics，缺项才本地 len()」：批量统计场景下上游口径更准，
            # 单仓库场景下本地统计足够，两条路都留着才不会出现空值
            f"- Issues：{metrics.get('issues', len(issues))} 个\n"
            f"- PRs：{metrics.get('pull_requests', len(prs))} 个，其中 Merged PR {metrics.get('merged_prs', len(merged_prs))} 个，Open PR {metrics.get('open_prs', len(open_prs))} 个\n"
            f"- 失败 CI：{metrics.get('failed_ci', len(failed_ci))} 次\n\n"
            "## 本周完成\n"
            f"- 合并 PR {len(merged_prs)} 个：{merged_pr_numbers}。\n"
            f"- 关闭 Issue {len(closed_issues)} 个：{closed_issue_numbers}。\n\n"
            "## 进行中\n"
            f"- 当前仍有 {len(open_prs)} 个 Open PR 需要跟进。\n"
            # 最多列 5 个，超出部分只体现在数量里；open_prs 为空时补一行「暂无」，避免留白
            + "\n".join(f"- #{pr.get('number')} {pr.get('title')}" for pr in open_prs[:5])
            + ("\n" if open_prs else "- 暂无 Open PR。\n")
            + "\n"
            "## 风险与阻塞\n"
            f"- 失败 CI {len(failed_ci)} 次，建议优先排查重复失败的 workflow。\n"
            # 兜底说明：没有失败 CI 时明确写出来，避免读者误以为数据没同步
            + ("- 本期没有同步到失败 CI。\n" if not failed_ci else "")
            + "\n"
            "## 需要关注的 PR\n"
            + "\n".join(f"- #{pr.get('number')} {pr.get('title')}" for pr in prs[:5])
            + "\n\n## 失败 CI\n"
            # 只列前 5 条失败：完整清单对周报读者价值不大，真要排查应该去 CI 页面看
            + "\n".join(f"- {run.get('name')}: {run.get('conclusion')}" for run in failed_ci[:5])
            + "\n\n## 下周建议\n"
            "- 补充关键模块测试覆盖。\n- 对高风险 PR 提前拆分 Review。\n"
        )
        # 只返回一个 key：报告是终态产物，交给前端/导出直接落盘，不再参与后续 Agent 计算
        return {"report_markdown": markdown}
