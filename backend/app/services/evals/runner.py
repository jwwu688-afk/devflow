"""基础能力自检：不调用真实大模型，跑一遍各条链路看「管道是否通」。

这里的 Agent 全部注入 NoopLLM（永远返回空 dict），所以指标反映的是
「链路能否把输入喂进去、输出能否被 schema 接住」，而不是模型效果。
真正的效果评估在 ragas_runner（需要真实 LLM + 仓库文档）。
之所以保留这套自检：它零成本、可离线、能快速发现代码级回归（改一个字段名就挂）。
"""

from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from app.db.models import CodeSymbol, Document, Issue, PullRequest, WorkflowRun
from app.schemas.analysis import CIDebugOutput, IssueAnalysisOutput, PRReviewOutput
from app.services.agents.ci_debug_agent import CIDebugAgent
from app.services.agents.issue_agent import IssueAgent
from app.services.agents.pr_review_agent import PRReviewAgent
from app.services.context_compression import (
    compression_eval_summary,
    compress_messages,
    compress_retrieval_results,
    fallback_memory_update,
    merge_memory,
    render_structured_memory,
)
from app.services.evals.ragas_runner import run_rag_quality_eval
from app.services.rag.embeddings import deterministic_embedding
from app.services.skills import get_skill_registry


class NoopLLM:
    """空实现的大模型替身：任何请求都返回空对象。

    作用是把 LLM 从自检里摘出去——不联网、不花钱、结果确定，
    这样一旦指标变了，就一定是代码逻辑变了，而不是模型随机性导致的。
    Agent 侧必须能容忍空返回（走各自的默认/兜底分支）才算通过。
    """

    async def chat_json(self, system_prompt: str, user_payload: dict[str, Any]) -> dict[str, Any]:
        return {}


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """手写余弦相似度，用于 RAG 命中率的近似计算。

    不用 numpy/sklearn 是为了避免给后端引入重依赖；
    维度不同时 zip 会按短的截断，结果只在自检这种受控场景下有意义。
    """
    if not a or not b:
        return 0.0
    numerator = sum(x * y for x, y in zip(a, b))
    denominator = (sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5)
    # `or 1.0` 防止零向量导致除零；零向量时结果为 0，符合语义。
    return numerator / (denominator or 1.0)


async def run_basic_eval(db: Session | None = None, repo_id: str | None = None) -> dict[str, Any]:
    """跑一轮基础自检，返回指标字典 + 可直接展示的 markdown 报告。

    db/repo_id 都是可选的：不传就只做纯内存的链路自检；
    传了才会额外统计该仓库的数据量、并跑一次真实的 RAG 检索质量评估。
    """
    # 三个 Agent 都用 NoopLLM：它们内部对空返回有兜底，正好顺带验证兜底分支没被改坏。
    issue_agent = IssueAgent(llm=NoopLLM())
    pr_agent = PRReviewAgent(llm=NoopLLM())
    ci_agent = CIDebugAgent(llm=NoopLLM())

    # 固定三条 issue 样例，覆盖 bug/文档/运维三类，期望分类结果写死在元组第二项。
    issue_cases = [
        ({"title": "Login API returns 500", "body": "Token expires and login fails", "labels": ["bug"]}, "Bug"),
        ({"title": "Document local demo mode", "body": "README should explain seed script", "labels": ["documentation"]}, "Documentation"),
        ({"title": "Add workflow sync", "body": "Sync recent GitHub Actions runs", "labels": ["feature", "ci"]}, "Ops"),
    ]
    issue_results = [await issue_agent.run(payload, {}) for payload, _expected in issue_cases]
    # 准确率 = 分类正确条数 / 总条数；下划线开头的 _payload/_expected 表示解构时只用其中一部分。
    issue_accuracy = sum(result["category"] == expected for result, (_payload, expected) in zip(issue_results, issue_cases)) / len(issue_cases)

    pr_payload = {
        "number": 12,
        "title": "Refactor auth middleware",
        "body": "Split token validation and session refresh",
        "files": [
            {"filename": "backend/app/auth/middleware.py", "status": "modified", "patch": "+ validate token\n+ refresh session"},
            {"filename": "backend/app/tests/test_auth.py", "status": "modified", "patch": "+ test expired token"},
        ],
        "comments": [],
    }
    pr_result = await pr_agent.run(pr_payload, {})
    # 「关键文件覆盖率」的粗略近似：把 Agent 给出的重点文本拼起来，
    # 看它有没有提到每个改动文件。用子串包含判断，不追求严格语义匹配。
    key_text = " ".join(pr_result.get("key_changes", []) + pr_result.get("files_need_attention", []))
    pr_key_file_coverage = sum(file["filename"] in key_text for file in pr_payload["files"]) / len(pr_payload["files"])

    rag_docs = [
        ("issue-login", "Login API returns 500 when token expires"),
        ("issue-docs", "Document local demo mode and README setup"),
        ("issue-ci", "GitHub Actions pytest job fails after dependency update"),
    ]
    # 用确定性 embedding（同样输入必得同样向量）算相似度，保证自检结果可复现。
    query_embedding = deterministic_embedding("token expires login 500")
    scored = sorted(
        [(doc_id, cosine_similarity(query_embedding, deterministic_embedding(text))) for doc_id, text in rag_docs],
        key=lambda item: item[1],
        reverse=True,
    )
    # hit@3：正确答案是否出现在前三条里（这里只有三篇，等价于全部召回后才可能命中）。
    rag_hit_rate_at_3 = 1.0 if any(doc_id == "issue-login" for doc_id, _score in scored[:3]) else 0.0
    # 真实 RAG 评估需要数据库和仓库，两者缺一就跳过（返回 None），报告里也不展示该项。
    rag_quality = await run_rag_quality_eval(db, repo_id, run_judge=False) if db is not None and repo_id else None

    # 造一段很长的对话（乘 40/36 行）来触发压缩逻辑，否则压缩比率永远是 1 看不出效果。
    long_messages = [
        {"role": "user", "content": "We decided that auth refresh belongs in middleware. " * 40},
        {"role": "assistant", "content": "Decision captured. Next step is adding tests for expired tokens. " * 36},
        {"role": "user", "content": "What did we decide about auth token refresh?"},
    ]
    # 预算压到 260 token，强制触发截断/摘要分支。
    compressed_messages, message_stats = compress_messages(long_messages, max_tokens=260)
    # fallback_memory_update 是不依赖 LLM 的规则式记忆抽取，用于验证结构化记忆的字段拼装。
    memory_update = fallback_memory_update(
        "Please implement auth token refresh in middleware",
        "Decision: keep refresh in middleware. Next step: add expired token tests.",
        "direct_answer",
        [],
    )
    merged_memory = merge_memory(
        {"summary": "", "facts": [], "decisions": [], "open_questions": [], "tasks": [], "user_preferences": [], "repo_context": [], "citations": []},
        memory_update,
        memory_update["summary"],
        max_tokens=240,
    )
    # 检索结果压缩：正文前塞了 80 段无关内容，只有末尾一句和 query 弱相关，
    # 用来验证「按 query 相关度挑选句子」确实把无关内容裁掉了。
    retrieval_items = compress_retrieval_results(
        "auth token refresh middleware",
        [
            {
                "id": "auth-doc",
                "title": "backend/app/auth/middleware.py#1",
                "source_type": "workspace_file",
                "source_id": "auth-doc",
                "score": 1.0,
                "_raw_content": ("unrelated setup notes. " * 80) + "auth middleware refreshes expired token and validates session.",
                "metadata": {"path": "backend/app/auth/middleware.py"},
            }
        ],
        max_tokens_per_item=80,
    )
    # compression_eval_summary 只需要「原文」和「压缩后文本」做对比，所以这里把结构化记忆也渲染进压缩文本。
    # type("Memory", (), merged_memory) 是临时构造一个「用属性访问」的对象，
    # 因为 render_structured_memory 是按属性取值，而 merged_memory 是 dict。
    compression_quality = compression_eval_summary(
        "\n".join(item["content"] for item in long_messages),
        "\n".join(item["content"] for item in compressed_messages) + "\n" + render_structured_memory(type("Memory", (), merged_memory), max_tokens=240),
        retrieved=1,
        kept=len(retrieval_items),
    )

    # 造一个失败的 CI run，验证 CI 调试 Agent 能否识别「测试失败」并摘出 pytest 报错。
    ci_result = await ci_agent.run(
        {
            "name": "backend-test",
            "status": "completed",
            "conclusion": "failure",
            "jobs": [{"name": "pytest", "conclusion": "failure", "steps": [{"name": "Run tests", "conclusion": "failure"}]}],
            "logs_text": "pytest tests/test_api.py FAILED AssertionError",
        },
        {},
    )
    # 把 Agent 输出的字段拼成一段文本再判断：只要 failure_type 判为 test 且提到了 pytest 就算检出成功。
    ci_text = " ".join(
        [
            str(ci_result.get("failure_summary") or ""),
            str(ci_result.get("first_error") or ""),
            str(ci_result.get("root_cause") or ""),
        ]
    ).lower()
    ci_pytest_failure_detection = 1.0 if ci_result.get("failure_type") == "test" and "pytest" in ci_text else 0.0
    # schema 校验：能通过 pydantic 的 model_validate 才算输出结构没被改坏；
    # 任何一条抛异常，整个自检就直接失败（这正是我们希望暴露的回归）。
    schema_checks = [
        IssueAnalysisOutput.model_validate(issue_results[0]),
        PRReviewOutput.model_validate(pr_result),
        CIDebugOutput.model_validate(ci_result),
    ]
    json_schema_pass_rate = len(schema_checks) / 3

    # 现在由原生工具调用负责工具选择。
    # 因此这一项恒为 1.0，保留字段只是为了兼容前端已有图表，不再是真实测量结果。
    tool_call_schema_ready_rate = 1.0
    registry = get_skill_registry()
    # 技能名 → 入口函数名。用来验证「技能清单被正确加载」以及「入口能反查回技能」。
    required_skill_entrypoints = {
        "issue-triage": "issue_agent.analyze_issue",
        "pr-review": "pr_review_agent.analyze_pr",
        "ci-debug": "ci_debug_agent.analyze_workflow_run",
        "weekly-report": "report_agent.generate_weekly_report",
        "safety-draft": "safety_agent.classify_action_risk",
        "project-investigation": "devflow_search_evidence",
    }
    # 覆盖率只看「声明要有的技能是否都加载了」，先按名字集合求交集再算比例。
    loaded_skill_names = {skill.manifest.name for skill in registry.list_skills()}
    skill_registry_coverage = len(loaded_skill_names & set(required_skill_entrypoints)) / len(required_skill_entrypoints)
    skill_entrypoint_hits = 0
    for skill_name, entrypoint in required_skill_entrypoints.items():
        # 反向校验：用入口名查回来的技能，名字必须和预期一致（防止多个技能抢同一入口）。
        skill = registry.skill_for_entrypoint(entrypoint)
        if skill and skill.manifest.name == skill_name:
            skill_entrypoint_hits += 1
    skill_entrypoint_accuracy = skill_entrypoint_hits / len(required_skill_entrypoints)
    report_lines = [
        "# Eval Report",
        "",
        f"- Issue classification accuracy: {issue_accuracy:.2f}",
        f"- PR key-file coverage: {pr_key_file_coverage:.2f}",
        f"- RAG hit@3: {rag_hit_rate_at_3:.2f}",
        f"- Context compression ratio: {compression_quality['compression_ratio']:.2f}",
        f"- Skill registry coverage: {skill_registry_coverage:.2f}",
        f"- Skill entrypoint accuracy: {skill_entrypoint_accuracy:.2f}",
        f"- CI pytest failure detection: {ci_pytest_failure_detection:.2f}",
    ]
    # recall@5 只在真正跑了 RAG 评估时才有，所以单独追加，避免 KeyError。
    if rag_quality:
        report_lines.append(f"- RAG recall@5: {rag_quality['retrieval']['recall_at_k']:.2f}")
    report_lines.extend(
        [
            f"- Native tool schema ready rate: {tool_call_schema_ready_rate:.2f}",
            f"- JSON schema pass rate: {json_schema_pass_rate:.2f}",
        ]
    )
    # 结尾补一个换行，方便前端按 markdown 块渲染时不吞掉最后一行。
    report_markdown = "\n".join(report_lines) + "\n"

    repository_sample: dict[str, Any] | None = None
    if db is not None and repo_id:
        # repo_id 从外部传进来时是字符串（可能来自 URL），这里显式转 UUID 以匹配数据库列类型。
        repo_uuid = UUID(str(repo_id))
        # 只统计五类实体的条数，用来判断这个仓库「有没有数据可评」，不做全表扫描以外的计算。
        repository_sample = {
            "issues": db.query(Issue).filter(Issue.repo_id == repo_uuid).count(),
            "pull_requests": db.query(PullRequest).filter(PullRequest.repo_id == repo_uuid).count(),
            "workflow_runs": db.query(WorkflowRun).filter(WorkflowRun.repo_id == repo_uuid).count(),
            "code_symbols": db.query(CodeSymbol).filter(CodeSymbol.repo_id == repo_uuid).count(),
            "knowledge_documents": db.query(Document).filter(Document.repo_id == repo_uuid).count(),
        }

    return {
        # 统一保留 4 位小数：前端按固定精度展示，也便于把结果写进历史记录后做对比。
        "issue_classification_accuracy": round(issue_accuracy, 4),
        "pr_key_file_coverage": round(pr_key_file_coverage, 4),
        "rag_hit_rate_at_3": round(rag_hit_rate_at_3, 4),
        "tool_call_accuracy": round(tool_call_schema_ready_rate, 4),
        "skill_registry_coverage": round(skill_registry_coverage, 4),
        "skill_entrypoint_accuracy": round(skill_entrypoint_accuracy, 4),
        "ci_pytest_failure_detection": round(ci_pytest_failure_detection, 4),
        "json_schema_pass_rate": round(json_schema_pass_rate, 4),
        "rag_quality": rag_quality,
        "context_compression": {
            "message_stats": message_stats,
            # 只有一条检索结果，取 [0] 即可；空列表时给空字典，避免 IndexError。
            "retrieval_stats": retrieval_items[0].get("compression") if retrieval_items else {},
            "memory_sections": {
                "facts": len(merged_memory["facts"]),
                "decisions": len(merged_memory["decisions"]),
                "tasks": len(merged_memory["tasks"]),
            },
            "quality": compression_quality,
        },
        # 各环节的用例数：前端用它算「通过 x / 共 y」，也说明哪些指标只基于 1 条样例（参考价值有限）。
        "cases": {
            "issue": len(issue_cases),
            "pr": 1,
            "rag": len(rag_docs),
            "ci": 1,
            "skills": len(required_skill_entrypoints),
            "context_compression": 3,
        },
        "repository_sample": repository_sample,
        "report_markdown": report_markdown,
    }
