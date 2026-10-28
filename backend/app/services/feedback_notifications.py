"""把用户对回答的「负反馈」推送到飞书群，方便产品/研发及时跟进 badcase。

设计要点：通知是**尽力而为**的旁路动作——失败只记日志并回写状态，
绝不能因为 webhook 挂了就让用户提交反馈的请求整体失败。
"""

import logging
from typing import Any

import httpx

from app.core.config import settings
from app.db.models import AgentRun, ChatFeedback, ChatMessage, Repository, User

logger = logging.getLogger(__name__)

# 前端提交的是稳定的英文枚举 key，这里才映射成中文文案。
# 集中维护而不是分散在拼接字符串处，新增原因时只改这一处。
REASON_LABELS = {
    "inaccurate": "内容不准确",
    "not_relevant": "没有解决问题",
    "missing_context": "缺少关键上下文",
    "unreliable_citation": "引用或证据不可靠",
    "tool_error": "工具执行异常",
    "other": "其他",
}


def _clip(value: str | None, limit: int) -> str:
    """压平空白并截断到 limit 字符，超出部分用 ... 表示。

    飞书文本消息对长度和换行都敏感，先 split/join 把多行、连续空格
    归一成单行，避免一条消息被撑成几十行把群消息刷屏。
    """
    text = " ".join((value or "").split())
    return text if len(text) <= limit else f"{text[:limit]}..."


def _trace_url(run_id: Any) -> str:
    """拼出该次 Agent 运行的前端可点击追踪链接。

    base_url 为空时退化成相对路径：本地开发没有公网域名，
    相对路径在群里点不开，但至少不会拼出 "None/api/..." 这种废链接。
    """
    path = f"/api/chat/feedback/trace/{run_id}"
    base_url = (settings.feedback_trace_base_url or "").strip().rstrip("/")
    return f"{base_url}{path}" if base_url else path


async def send_negative_feedback_notification(
    *,
    feedback: ChatFeedback,
    repo: Repository,
    message: ChatMessage,
    run: AgentRun | None,
    reporter: User | None,
) -> tuple[str, str | None]:
    """发送负反馈通知，返回 (状态, 错误详情)。

    状态取值：not_configured（没配 webhook）/ sent / failed，
    调用方据此在 feedback 记录上留痕，便于事后统计通知成功率。
    run 可能为 None——历史回答没有关联 run_id，此时退化成只发消息本身的信息。
    """
    webhook_url = (settings.feishu_feedback_webhook_url or "").strip()
    if not webhook_url:
        # 未配置就当成功跳过（返回 not_configured 而不是 failed），
        # 避免本地/私有化部署环境反复报错刷日志。
        return "not_configured", None

    run_id = feedback.run_id or "未关联"
    # 逐行拼装固定字段，顺序即为群里看到的顺序：先定位（项目/人/ID），再内容，最后链接。
    lines = [
        "DevFlow AI 收到一条负反馈",
        f"项目：{repo.full_name}",
        f"反馈人：{reporter.name if reporter else '内部用户'}",
        f"trace_id：{run_id}",
        f"message_id：{feedback.assistant_message_id}",
        f"路由：{run.route if run else message.route or 'unknown'}",
        f"原因：{REASON_LABELS.get(feedback.reason or '', '未填写')}",
    ]
    if feedback.comment:
        lines.append(f"补充说明：{_clip(feedback.comment, 300)}")
    if run:
        # 问答原文截断长度不同：问题通常短，取 300；回答容易长，取 500。
        lines.extend(
            [
                f"用户问题：{_clip(run.user_message, 300)}",
                f"Agent 回答：{_clip(run.final_answer, 500)}",
            ]
        )
    lines.append(f"执行轨迹：{_trace_url(run_id)}" if feedback.run_id else "执行轨迹：该历史回答未关联 run_id")

    try:
        # 5 秒超时：通知不该拖慢用户请求，超时就放弃。
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                webhook_url,
                json={"msg_type": "text", "content": {"text": "\n".join(lines)}},
            )
            response.raise_for_status()
            payload = response.json()
            # 飞书自定义机器人 HTTP 200 也可能业务失败，必须再看 body 里的 code：
            # 新版返回 StatusCode，旧版/兼容实现返回 code，都没有则视为 0（成功）。
            status_code = payload.get("StatusCode", payload.get("code", 0)) if isinstance(payload, dict) else 0
            if status_code not in {0, "0", None}:
                raise RuntimeError(str(payload.get("StatusMessage") or payload.get("msg") or payload))
    except Exception as exc:  # 通知失败不能影响已落库的反馈记录，因此这里吞掉异常只留痕。
        logger.warning("发送飞书负反馈通知失败：%s", exc)
        return "failed", _clip(str(exc), 1000)
    return "sent", None
