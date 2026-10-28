"""GitHub 数据检索 Agent。

职责边界：**只负责把 GitHub 上的原始数据取回来，不做任何分析、总结、判断**。
它是整条编排链里唯一的「外部数据入口」——Issue / PR / diff / Review 评论 /
workflow run 与失败日志全部经它进入系统，后续 Agent（issue_agent、pr_review_agent、
ci_debug_agent 等）都只是消费它的输出。

设计后果：
- 这里**不调用 LLM**，没有 token 成本，也因此它的输出结构可以保持「原样透传」，
  不做归一化——归一化交给下游 Agent，避免在这里丢掉后续可能需要的字段；
- 全部走 if 分支按 task 分发，而不是为每种资源建一个方法：调用方（编排层）
  只需要一个 run() 入口 + 一个 task 字符串，路由逻辑集中在编排层。
"""

from typing import Any

from app.services.agents.base import BaseAgent
from app.services.github.actions import get_workflow_run_logs, list_workflow_run_jobs, list_workflow_runs
from app.services.github.issues import list_issues
from app.services.github.pull_requests import list_pull_request_files, list_pull_request_review_comments, list_pull_requests


class GitHubRetrievalAgent(BaseAgent):
    """GitHub 只读检索 Agent，按 task 返回对应资源的原始字典。"""

    name = "github_retrieval_agent"
    description = "检索 GitHub Issue、PR、diff、Review 评论、workflow run 和失败日志。"
    # 与下面的 task 分支一一对应，Planner 据此判断该 Agent 能否满足某个子任务
    available_tools = [
        "list_issues",
        "list_pull_requests",
        "list_pull_request_files",
        "list_pull_request_review_comments",
        "list_workflow_runs",
        "list_workflow_run_jobs",
        "get_workflow_run_logs",
    ]

    async def run(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """按 input_data["task"] 抓取指定类型的 GitHub 数据。

        输入：owner / repo 必填；token、base_url 用于私有仓库与 GitHub Enterprise；
              limit 控制列表条数；task 决定走哪个分支。
        输出：统一包一层以资源名命名的 key（如 {"issues": [...]}），
              让编排层能直接判断「拿到了什么」，而不必按 task 反查。
        """
        owner = input_data["owner"]
        repo = input_data["repo"]
        # token/base_url 可选：公开仓库不需要 token，也兼容自建 GitHub 域名
        token = input_data.get("token")
        base_url = input_data.get("base_url")
        # 默认 30：一次性拉太多会让下游 prompt 变长、成本上升，30 是够用的概览量级
        limit = int(input_data.get("limit", 30))
        # 缺省 task 走仓库快照，保证「只给 owner/repo」也能跑通
        task = input_data.get("task", "repo_snapshot")

        if task == "issues":
            return {"issues": await list_issues(owner, repo, token=token, limit=limit, base_url=base_url)}
        if task == "pull_requests":
            return {"pull_requests": await list_pull_requests(owner, repo, token=token, limit=limit, base_url=base_url)}
        if task == "pull_request_detail":
            # PR 详情要「文件改动 + 行级评论」两份数据：评审结论必须同时看代码和已有讨论
            number = int(input_data["number"])
            files = await list_pull_request_files(owner, repo, number, token=token, base_url=base_url)
            comments = await list_pull_request_review_comments(owner, repo, number, token=token, base_url=base_url)
            return {"files": files, "review_comments": comments}
        if task == "workflow_runs":
            return {"workflow_runs": await list_workflow_runs(owner, repo, token=token, limit=limit, base_url=base_url)}
        if task == "workflow_run_failure":
            # 失败诊断需要「job 级状态 + 原始日志文本」：job 定位哪个步骤挂了，日志才是根因证据
            run_id = int(input_data["run_id"])
            jobs = await list_workflow_run_jobs(owner, repo, run_id, token=token, base_url=base_url)
            logs_text = await get_workflow_run_logs(owner, repo, run_id, token=token, base_url=base_url)
            return {"jobs": jobs, "logs_text": logs_text}

        # 兜底 = repo_snapshot：未知 task 也返回仓库全景，宁可多给数据也不让上游拿到空结果
        issues = await list_issues(owner, repo, token=token, limit=limit, base_url=base_url)
        pull_requests = await list_pull_requests(owner, repo, token=token, limit=limit, base_url=base_url)
        workflow_runs = await list_workflow_runs(owner, repo, token=token, limit=limit, base_url=base_url)
        return {"issues": issues, "pull_requests": pull_requests, "workflow_runs": workflow_runs}
