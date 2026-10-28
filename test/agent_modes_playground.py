"""最基本的 Agent Loop 实验：模型自己决定读哪些证据，程序执行工具，最后判断 PR 能不能合并。

循环只有四步：想 → 做 → 记 → 判，实现在 run_react()。

运行：
    python test/agent_modes_playground.py               # 默认最多 8 步
    python test/agent_modes_playground.py --max-steps 12

模型配置读项目根目录的 .env：LLM_BASE_URL / LLM_API_KEY / LLM_MODEL。
一次运行通常 5~8 次模型调用，会产生 API 费用。
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

# =============================================================================
# 场景数据（假的 GitHub）
# =============================================================================
# 不用真接口：真接口要 token、要网络、数据还会变，没法稳定地测循环。
# 这个场景刻意设计成「不该合并」——改了 auth/ + CI 失败 + Review 未解决，
# 所以五类证据读全后，正确结论必然是「不可合并」。如果跑出「可以合并」，
# 说明循环漏读了证据。

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

# =============================================================================
# 工具层
# =============================================================================
# 三个约定：只读、纯函数、出错返回 {"error": ...} 而不是抛异常。
# 不抛异常是为了让「工具失败」变成一条普通 observation 继续流转，
# 模型才能看到失败原因并换个动作；抛异常会直接打断整个 loop。

def read_pr(number: int) -> dict[str, Any]:
    """读 PR 详情，落到 state.pr。"""
    if number != PR_DATA["number"]:
        return {"error": f"没有找到 PR #{number}"}
    return PR_DATA


def read_issue(number: int) -> dict[str, Any]:
    """读关联 Issue，落到 state.issue。"""
    if number != ISSUE_DATA["number"]:
        return {"error": f"没有找到 Issue #{number}"}
    return ISSUE_DATA


def read_ci(pr_number: int) -> dict[str, Any]:
    """读 CI 状态，落到 state.ci。"""
    if pr_number != PR_DATA["number"]:
        return {"error": f"没有找到 PR #{pr_number} 的 CI 记录"}
    return CI_DATA


def read_review_comments(pr_number: int) -> dict[str, Any]:
    """读 Review 评论，落到 state.review_comments。

    出参套了一层 {"comments": [...]}，模仿真实 GitHub 的分页响应，
    所以 apply_observation 里要取 .comments。
    """
    if pr_number != PR_DATA["number"]:
        return {"error": f"没有找到 PR #{pr_number} 的 Review 评论"}
    return {"comments": REVIEW_COMMENTS}


def read_release_policy(topic: str) -> dict[str, Any]:
    """读发布/合并规则。查不到也返回一句话，不返回 error。

    「查不到规则」不等于「工具坏了」：报 error 会让循环白白重试。
    """
    policy = RELEASE_POLICY.get(topic, "没有找到明确规则，建议人工确认。")
    return {"topic": topic, "policy": policy}


Tool = Callable[..., dict[str, Any]]

# 白名单：只有在这里的名字才能被执行。这是安全边界，不是普通注册表——
# 模型输出的是字符串，不做校验就直接 globals()[...] 派发，一句提示词注入就能调到 os.system。
TOOLS: dict[str, Tool] = {
    "read_pr": read_pr,
    "read_issue": read_issue,
    "read_ci": read_ci,
    "read_review_comments": read_review_comments,
    "read_release_policy": read_release_policy,
}

# 给模型看的工具说明，随每轮 prompt 一起发过去
TOOL_GUIDE = {
    "read_pr": "读取 PR 详情，参数：number",
    "read_issue": "读取关联 Issue，参数：number",
    "read_ci": "读取 PR 的 CI 状态，参数：pr_number",
    "read_review_comments": "读取 PR Review 评论，参数：pr_number",
    "read_release_policy": "读取发布/合并规则，参数：topic",
}

# =============================================================================
# 状态与 trace
# =============================================================================

@dataclass
class TraceEvent:
    """循环里发生的一件事。工具调用、被拒的 final，都记成同一种结构。"""

    step: int                                   # 第几步，由 remember 自动编号
    actor: str                                  # 谁做的：ReActAgent / LoopGuard
    action: str                                 # 工具名，或 final_rejected 这类守卫事件
    arguments: dict[str, Any] = field(default_factory=dict)     # 归一化后的真实入参
    observation: dict[str, Any] = field(default_factory=dict)   # 工具返回了什么
    thought: str = ""                            # 模型的原始理由，排查「它为什么这么干」最有用


@dataclass
class EvidenceState:
    """Agent 的工作台：一个字段 = 一类证据 = 一个工具的出参。

    这样 missing_evidence() 能机械地判断「还缺什么」，
    不用去问模型「你读过没有」——模型自述是最不可靠的信息来源。
    """

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
        """追加一条 trace，step 自动递增。"""
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
        """投影出「发给模型的那一份状态」。

        只给最近 8 条 trace：全量回灌会撑爆 prompt，模型真正需要的只是「我刚才干了什么」。
        主动把 missing_evidence 告诉它：与其让它自己推理缺什么，不如直接给答案，省一半步数。
        """
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
    """把工具返回值写进状态。

    「工具名 → 状态字段」的唯一映射点。出错就直接 return，不改状态：
    状态里只放确认过的证据，脏数据混进去会让后面所有结论跟着错。
    """
    if "error" in observation:
        return
    if action == "read_pr":
        state.pr = observation
    elif action == "read_issue":
        state.issue = observation
    elif action == "read_ci":
        state.ci = observation
    elif action == "read_review_comments":
        # 取 .comments：工具出参是分页信封，状态里只存纯列表
        state.review_comments = observation.get("comments") or []
    elif action == "read_release_policy":
        state.release_policy = observation


def has_auth_change(state: EvidenceState) -> bool:
    """PR 有没有动 auth/ 这类高风险目录。

    决定 release_policy 是不是必需证据：写死成「五类全要」更简单，
    但会让每个无关 PR 都多花一次工具调用。
    """
    files = (state.pr or {}).get("changed_files") or []
    return any(str(path).startswith("auth/") for path in files)


def missing_evidence(state: EvidenceState) -> list[str]:
    """还缺哪些证据。返回空列表 = 有资格下结论。

    这是整个循环的终止条件，也是驳回「模型提前收工」的唯一依据。
    只看状态、不听模型。
    """
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


# =============================================================================
# 参数归一化与工具执行
# =============================================================================

def normalize_arguments(
    action: str,
    arguments: Any,
    state: EvidenceState,
) -> dict[str, Any]:
    """把模型给的参数修正成工具能直接 **kwargs 调用的形状。

    模型会给出各种花样：{"number": "18"}、该传 pr_number 却传了 number、
    少参数、多参数。这里统一兜底，保证工具拿到的入参永远正确。
    """
    raw = arguments if isinstance(arguments, dict) else {}

    if action == "read_pr":
        return {"number": int(raw.get("number") or PR_DATA["number"])}

    if action == "read_issue":
        linked_issue = (state.pr or {}).get("linked_issue")
        default = linked_issue or ISSUE_DATA["number"]
        # 已知 PR 关联的 Issue 时以关联关系为准（纠偏）：关联关系是事实，模型给的号是猜测
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
    """执行一个动作并记 trace。这是循环里唯一产生副作用的地方。

    两道闸都集中在这里：白名单校验、参数归一化。不管谁调、想调什么，
    都得过同一道闸，不会漏。
    """
    # 闸①：白名单。不抛异常，而是返回 error observation，让模型看到后自己换个动作
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

    # 闸②：参数归一化 → 执行 → 落状态 → 记 trace
    normalized = normalize_arguments(action, arguments, state)
    observation = TOOLS[action](**normalized)
    apply_observation(state, action, observation)
    state.remember(actor, action, normalized, observation, thought)
    return observation


# =============================================================================
# 模型层：一个接口 + 一个真实实现
# =============================================================================
# 循环只认 ChatModel 这个接口，不认具体模型。想换 SDK 只改这一层。

class ChatModel(Protocol):
    """决策器接口：给 prompt + 状态，回一个 decision dict。

    run_react 里是按位置传三个参数的：chat_json(REACT_PROMPT, {...}, "ReAct 决策")。
    """

    def chat_json(
        self,
        system_prompt: str,
        user_payload: dict[str, Any],
        purpose: str = "",
    ) -> dict[str, Any]:
        ...


def _read_env_file(path: Path) -> dict[str, str]:
    """极简 .env 解析：只认 KEY=VALUE，忽略注释和空行。读不到就返回空字典。"""
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip().strip("'\"")
    return values


def _parse_json(text: str) -> dict[str, Any]:
    """从模型回复里取 JSON。真实模型常包一层 ```json 围栏，或前后加一句废话。"""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rstrip()
        if text.endswith("```"):
            text = text[:-3].rstrip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # 兜底：截取第一个 { 到最后一个 }
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return {}
        parsed = json.loads(text[start : end + 1])
    return parsed if isinstance(parsed, dict) else {}


class EnvChatModel:
    """真实模型：读 .env 里的 LLM_* 配置，调 OpenAI 兼容的 /chat/completions。

    DeepSeek、OpenAI、vLLM 都是这套协议，换个 base_url 就能用。
    用标准库 urllib，不额外依赖 httpx / openai。
    """

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 60.0) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "EnvChatModel":
        """读配置。真实环境变量优先于 .env，方便在 CI 里直接 export。"""
        values = _read_env_file(Path(__file__).resolve().parents[1] / ".env")
        for key in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
            if os.environ.get(key):
                values[key] = os.environ[key]

        base_url = values.get("LLM_BASE_URL", "").strip() or "https://api.deepseek.com"
        api_key = values.get("LLM_API_KEY", "").strip()
        model = values.get("LLM_MODEL", "").strip() or "deepseek-chat"
        if not api_key:
            raise SystemExit("没找到 LLM_API_KEY，请在项目根目录 .env 里填上：LLM_API_KEY=sk-...")
        return cls(base_url=base_url, api_key=api_key, model=model)

    def chat_json(
        self,
        system_prompt: str,
        user_payload: dict[str, Any],
        purpose: str = "",
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "temperature": 0,                              # 决策要可复现，不要创意
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            # 让服务端保证返回 JSON，省掉一半解析失败。个别服务不支持就删掉这一行
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        decision = _parse_json(str(body["choices"][0]["message"]["content"] or ""))

        # 模型常把 action 写成 "FINAL"，这里统一转小写，
        # 否则 run_react 会把它当成一个叫 FINAL 的工具，白跑一轮。
        decision["action"] = str(decision.get("action") or "").lower()
        return decision


# =============================================================================
# Prompt
# =============================================================================
# 四要点：一次只决定一个动作；把 missing_evidence 直接塞进 payload；
# 明说「证据没齐不许 final」（程序侧还会再校验一次）；只输出 JSON。

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


# =============================================================================
# 主循环
# =============================================================================

def run_react(model: ChatModel, goal: str, max_steps: int) -> dict[str, Any]:
    """ReAct 循环：每轮一次模型决策 + 至多一次工具执行。

    骨架四步：想 → 做 → 记 → 判。

    三个停止出口，缺一不可：
        ① 模型说 final 且证据齐全 → 正常收尾；
        ② 模型说 final 但证据不全 → 驳回继续（不能信模型自述）；
        ③ 步数用尽               → 记 max_steps_stop，带着「证据不全」的结论收尾。
    少了 ②，模型啥也没读就能给你一个看起来很专业的结论；
    少了 ③，一个乱调的模型能让循环永远转下去。
    """
    state = EvidenceState()

    # for-else：循环自然跑完（没 break）说明步数耗尽，进 else 记一笔
    for _ in range(max_steps):
        # ---- 想：把目标、当前状态、工具清单交给模型，让它决定下一步 ----
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

        # ---- 分支 A：模型想收工 ----
        if action == "final":
            missing = missing_evidence(state)
            if missing:
                # 出口②：硬驳回。记一条 trace，排障时能看出「它想收尾但被挡回来了」
                state.remember(
                    "LoopGuard",
                    "final_rejected",
                    arguments={},
                    observation={"missing_evidence": missing},
                    thought=thought,
                )
                continue
            # 出口①：证据齐了，接受收工
            break

        # ---- 做 + 记 ----
        run_tool(state, "ReActAgent", action, arguments, thought)

        # ---- 判：证据一读齐就跳出 ----
        # 注意这个 break 比「等模型自己说 final」更早生效，所以正常路径走不到上面的
        # final 分支。想让模型自己收尾就删掉这三行（有 max_steps 兜底，不会死循环），
        # 代价是多一次模型调用。
        if not missing_evidence(state):
            break
    else:
        # 出口③：步数耗尽。后面 synthesize 会把「证据不全」如实写进结论，不假装成功
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
        "trace": [asdict(item) for item in state.trace],   # asdict 后整个 result 可直接 json.dump
        "final": final,
        "state": state.model_view(),
    }


def synthesize_with_model(
    model: ChatModel,
    goal: str,
    state: EvidenceState,
    actor: str,
) -> dict[str, Any]:
    """把 state 里的证据汇总成结论。

    这里**不调模型**：结论固定才方便验证循环对不对（跑出「可以合并」就说明漏读证据），
    也省一次调用。model 参数保留只是让签名一致，将来想改成模型生成时不用动调用方。
    """
    blockers: list[str] = []   # 硬阻塞：有一条就不能合
    risks: list[str] = []      # 风险提示：不阻塞，但要人知道
    evidence: list[str] = []   # 已确认的证据

    pr = state.pr or {}
    issue = state.issue or {}
    ci = state.ci or {}

    if state.pr is None:
        blockers.append("缺少 PR 详情，无法判断改动范围")
    else:
        files = pr.get("changed_files") or []
        evidence.append(f"PR #{pr.get('number')} {pr.get('title')}，改动 {len(files)} 个文件")

    if state.issue is not None:
        evidence.append(f"Issue #{issue.get('number')} 期望：{issue.get('expected')}")

    if state.ci is None:
        blockers.append("缺少 CI 状态，无法确认测试是否通过")
    elif ci.get("status") != "success":
        # 用 != success 而不是 == failed：pending / cancelled 同样不能合
        blockers.append(f"CI {ci.get('status')}，失败用例 {ci.get('failed_test')}")
        if ci.get("hint"):
            risks.append(f"CI 提示：{ci.get('hint')}")

    if state.review_comments is None:
        blockers.append("缺少 Review 评论，无法确认是否有未解决的反对意见")
    else:
        for item in state.review_comments:
            if not item.get("resolved"):
                blockers.append(f"Review 未解决：{item.get('author')} @ {item.get('file')}")

    if has_auth_change(state):
        risks.append("改动落在 auth/ 高风险模块，需要额外人工复核")
        if state.release_policy is None:
            blockers.append("认证模块改动缺少发布规则确认")
        else:
            evidence.append(f"发布规则：{state.release_policy.get('policy')}")

    return {
        "goal": goal,
        "verdict": "不可合并" if blockers else "可以合并",
        "blockers": blockers,
        "risks": risks,
        "evidence": evidence,
        "missing_evidence": missing_evidence(state),
        "synthesized_by": actor,
    }


# =============================================================================
# 入口
# =============================================================================

def main() -> None:
    """跑一遍循环，把 trace 和结论打出来。"""
    parser = argparse.ArgumentParser(description="跑一遍最基础的 ReAct agent loop")
    parser.add_argument("--max-steps", type=int, default=8, help="循环步数上限（默认 8）")
    args = parser.parse_args()

    model = EnvChatModel.from_env()
    print(f"模型：{model.model}    步数上限：{args.max_steps}\n")

    result = run_react(model, DEFAULT_GOAL, max_steps=args.max_steps)

    # trace 是最该盯着看的部分：每步入参、出参、理由都在这里
    print("trace：")
    for event in result["trace"]:
        arguments = json.dumps(event["arguments"], ensure_ascii=False)
        observation = json.dumps(event["observation"], ensure_ascii=False)
        print(f"  {event['step']}. [{event['actor']}] {event['action']} {arguments}")
        print(f"       -> {observation[:200]}")
        if event["thought"]:
            print(f"       thought: {event['thought']}")

    final = result["final"]
    print(f"\n结论：{final['verdict']}")
    for label, key in (("阻塞项", "blockers"), ("风险", "risks"), ("已确认证据", "evidence")):
        if final[key]:
            print(f"{label}：")
            for item in final[key]:
                print(f"  - {item}")
    if final["missing_evidence"]:
        print(f"缺失证据：{'、'.join(final['missing_evidence'])}")


if __name__ == "__main__":
    main()
