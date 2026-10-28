from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

DEFAULT_GOAL = "判断 PR #18 是否可以合并，并说明主要风险。"

PR_DATA = {
    "number": 18,
    "title": "Fix expired JWT handling",
    "changed_files": [
        "auth/middleware.py",
        "auth/exceptions.py",
        "tests/test_auth.py",
    ],
    "summary": "把 ExpiredSignatureError 映射成 401，并补充 token 过期测试。",
    "linked_issue": 12,
}

ISSUE_DATA = {
    "number": 12,
    "title": "登录 token 过期时返回 500",
    "expected": "token 过期、缺失、签名无效时都应该返回 401。",
    "severity": "medium",
}

CI_DATA = {
    "run_id": 56,
    "status": "failed",
    "failed_test": "test_expired_jwt_returns_401",
    "hint": "认证中间件和全局异常处理器的映射顺序需要继续检查。",
}

REVIEW_COMMENTS = [
    {
        "author": "reviewer-a",
        "file": "auth/middleware.py",
        "comment": "请确认 token 过期和签名无效会走同一条 401 分支。",
        "resolved": False,
    }
]

RELEASE_POLICY = {
    "auth": "认证、鉴权、权限、支付等高风险模块，只要 CI 未通过或 Review 未解决，就不能直接合并。",
    "write_action": "合并 PR、关闭 Issue、发布评论等写操作必须等待人工确认。",
}

def read_pr(number: int) -> dict[str, Any]:
    if number != PR_DATA["number"]:
        return {"error": f"没有找到 PR #{number}"}
    return PR_DATA


def read_issue(number: int) -> dict[str, Any]:
    if number != ISSUE_DATA["number"]:
        return {"error": f"没有找到 Issue #{number}"}
    return ISSUE_DATA


def read_ci(pr_number: int) -> dict[str, Any]:
    if pr_number != PR_DATA["number"]:
        return {"error": f"没有找到 PR #{pr_number} 的 CI 记录"}
    return CI_DATA


def read_review_comments(pr_number: int) -> dict[str, Any]:
    if pr_number != PR_DATA["number"]:
        return {"error": f"没有找到 PR #{pr_number} 的 Review 评论"}
    return {"comments": REVIEW_COMMENTS}


def read_release_policy(topic: str) -> dict[str, Any]:
    policy = RELEASE_POLICY.get(topic, "没有找到明确规则，建议人工确认。")
    return {"topic": topic, "policy": policy}


Tool = Callable[..., dict[str, Any]]

TOOLS: dict[str, Tool] = {
    "read_pr": read_pr,
    "read_issue": read_issue,
    "read_ci": read_ci,
    "read_review_comments": read_review_comments,
    "read_release_policy": read_release_policy,
}

TOOL_GUIDE = {
    "read_pr": "读取 PR 详情，参数：number",
    "read_issue": "读取关联 Issue，参数：number",
    "read_ci": "读取 PR 的 CI 状态，参数：pr_number",
    "read_review_comments": "读取 PR Review 评论，参数：pr_number",
    "read_release_policy": "读取发布/合并规则，参数：topic",
}

@dataclass
class TraceEvent:
    step: int
    actor: str
    action: str
    arguments: dict[str, Any] = field(default_factory=dict)
    observation: dict[str, Any] = field(default_factory=dict)
    thought: str = ""


@dataclass
class EvidenceState:
    pr: dict[str, Any] | None = None
    issue: dict[str, Any] | None = None
    ci: dict[str, Any] | None = None
    review_comments: list[dict[str, Any]] | None = None
    release_policy: dict[str, Any] | None = None
    trace: list[TraceEvent] = field(default_factory=list)  # TraceEvent 列表，保存完整操作历史

    def remember(
        self,
        actor: str,
        action: str,
        arguments: dict[str, Any] | None = None,
        observation: dict[str, Any] | None = None,
        thought: str = "",
    ) -> None:
        self.trace.append(
            TraceEvent(
                step=len(self.trace) + 1,
                actor=actor,
                action=action,
                arguments=arguments or {},
                observation=observation or {},
                thought=thought,
            )
        )

    def model_view(self) -> dict[str, Any]:
        return {
            "pr": self.pr,
            "issue": self.issue,
            "ci": self.ci,
            "review_comments": self.review_comments,
            "release_policy": self.release_policy,
            "missing_evidence": missing_evidence(self),
            "recent_trace": [asdict(item) for item in self.trace[-8:]],
        }


def apply_observation(
    state: EvidenceState,
    action: str,
    observation: dict[str, Any],
) -> None:
    if "error" in observation:
        return
    if action == "read_pr":
        state.pr = observation
    elif action == "read_issue":
        state.issue = observation
    elif action == "read_ci":
        state.ci = observation
    elif action == "read_review_comments":
        state.review_comments = observation.get("comments") or []
    elif action == "read_release_policy":
        state.release_policy = observation


def has_auth_change(state: EvidenceState) -> bool:
    files = (state.pr or {}).get("changed_files") or []
    return any(str(path).startswith("auth/") for path in files)


def missing_evidence(state: EvidenceState) -> list[str]:
    missing = []
    if state.pr is None:
        missing.append("PR 详情")
    if state.issue is None:
        missing.append("关联 Issue")
    if state.ci is None:
        missing.append("CI 状态")
    if state.review_comments is None:
        missing.append("Review 评论")
    if has_auth_change(state) and state.release_policy is None:
        missing.append("认证模块发布规则")
    return missing

decision = model.chat_json(
    system_prompt=REACT_PROMPT,
    user_payload={"goal": goal, "state": state.model_view()},
    purpose="ReAct 决策",
)


def normalize_arguments(
    action: str,
    arguments: Any,
    state: EvidenceState,
) -> dict[str, Any]:
    raw = arguments if isinstance(arguments, dict) else {}

    if action == "read_pr":
        return {"number": int(raw.get("number") or PR_DATA["number"])}

    if action == "read_issue":
        linked_issue = (state.pr or {}).get("linked_issue")
        default = linked_issue or ISSUE_DATA["number"]
        number = default if linked_issue else raw.get("number") or default
        return {"number": int(number)}

    if action in {"read_ci", "read_review_comments"}:
        default = (state.pr or {}).get("number") or PR_DATA["number"]
        return {"pr_number": int(raw.get("pr_number") or default)}

    if action == "read_release_policy":
        return {"topic": str(raw.get("topic") or "auth")}

    return raw


def run_tool(
    state: EvidenceState,
    actor: str,
    action: str,
    arguments: Any,
    thought: str = "",
) -> dict[str, Any]:
    if action not in TOOLS:
        observation = {"error": f"工具 {action} 不存在, 请从工具白名单里选择。"}
        state.remember(
            actor,
            "tool_error",
            {"action": action},
            observation,
            thought,
        )
        return observation

    normalized = normalize_arguments(action, arguments, state)
    observation = TOOLS[action](**normalized)
    apply_observation(state, action, observation)
    state.remember(actor, action, normalized, observation, thought)
    return observation


REACT_PROMPT = """
你是一个 ReAct 风格的 Agent 决策器。
你每次只能决定一个下一步动作，程序会负责真正执行工具。
请只输出合法 JSON 对象，不要输出 Markdown。
可用 action: read_pr、read_issue、read_ci、read_review_comments、read_release_policy、final。
输出字段:
{
    "thought": "为什么现在要做这一步",
    "action": "工具名或 final",
    "arguments": {},
    "final_answer": "只有 action=final 时填写"
}
规则:
1. 缺少 PR 详情时先 read_pr。
2. 缺少 Issue、CI、Review 时继续补证据。
3. 如果 PR 修改 auth/，还要读取 read_release_policy。
4. 证据没齐时不要 action=final。
"""


def run_react(model: ChatModel, goal: str, max_steps: int) -> dict[str, Any]:
    state = EvidenceState()

    for _ in range(max_steps):
        decision = model.chat_json(
            REACT_PROMPT,
            {
                "goal": goal,
                "state": state.model_view(),
                "available_tools": TOOL_GUIDE,
            },
            "ReAct 决策",
        )
        action = str(decision.get("action") or "").strip()
        thought = str(decision.get("thought") or "")
        arguments = decision.get("arguments") or {}

        if action == "final":
            missing = missing_evidence(state)
            if missing:
                state.remember(
                    "LoopGuard",
                    "final_rejected",
                    arguments={},
                    observation={"missing_evidence": missing},
                    thought=thought,
                )
                continue
            break

        run_tool(state, "ReActAgent", action, arguments, thought)

        if not missing_evidence(state):
            break
    else:
        state.remember(
            "LoopGuard",
            "max_steps_stop",
            arguments={},
            observation={"max_steps": max_steps},
            thought="",
        )

    final = synthesize_with_model(model, goal, state, "ReActAgent")
    return {
        "mode": "react",
        "trace": [asdict(item) for item in state.trace],
        "final": final,
        "state": state.model_view(),
    }
