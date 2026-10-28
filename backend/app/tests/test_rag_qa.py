"""RAG 问答链路（qa）的测试。

验证行为：从「证据」到「答案」这最后一公里必须满足：
1) 中文分词缺位时（如"年假"没有空格）deterministic embedding 仍能区分相关度；
2) 返回给前端的 sources 不能泄露服务器本地文件路径；
3) 有 LLM 时走 LLM 生成，且严格「只依据资料」并用 [n] 标注引用；
4) 没有 LLM 时退化为抽取式答案，不能直接空白；
5) 证据闸门判定为冲突/不足时，绝不调用 LLM，直接返回 no_evidence。
"""

import pytest

from app.core.config import settings
from app.services.rag.embeddings import deterministic_embedding
from app.services.rag.qa import answer_question, build_sources
from app.services.rag.retrieval import cosine_similarity


def test_deterministic_embedding_handles_chinese_without_spaces() -> None:
    """测的是中文无空格文本的伪向量仍保留语义相关性。

    场景：用确定性的本地 embedding 分别编码「问题」「相关资料」「无关资料」。
    期望：问题与相关资料（都含"年假"）的余弦相似度，高于与无关资料
    （数据库监控）的相似度。
    重要性：中文不像英文有天然空格分词，若 embedding 只按整段哈希，
    则会退化成"字面完全相同才算相关"，离线检索直接不可用。
    """
    question = deterministic_embedding("员工怎么申请年假？", 128)
    relevant = deterministic_embedding("员工年假审批流程和请假规定", 128)
    unrelated = deterministic_embedding("数据库连接池监控告警", 128)

    assert cosine_similarity(question, relevant) > cosine_similarity(question, unrelated)


def test_build_sources_hides_server_file_path() -> None:
    """测的是引用来源的字段白名单：本地存储路径必须被剔除。

    场景：检索结果的 metadata 里同时带 filename 和 stored_path
    （形如 C:/private/upload/handbook.md 的服务器绝对路径）。
    期望：citation 从 1 开始编号；filename 保留；stored_path 不出现在输出中。
    重要性：stored_path 属于服务端内部信息，泄漏出去既是信息泄露，
    也会让用户看到无意义的路径而不是文件名。
    """
    sources = build_sources(
        [
            {
                "id": "doc-1",
                "source_id": "source-1",
                "source_type": "knowledge_file",
                "title": "handbook.md#1",
                "snippet": "年假需要先在系统中提交审批。",
                "score": 0.91,
                "metadata": {"filename": "handbook.md", "stored_path": "C:/private/upload/handbook.md"},
            }
        ]
    )

    # 编号从 1 开始，与提示词里的 [资料 1] / 答案里的 [1] 对齐
    assert sources[0]["citation"] == 1
    assert sources[0]["metadata"]["filename"] == "handbook.md"
    # 核心断言：内部路径被过滤掉
    assert "stored_path" not in sources[0]["metadata"]


@pytest.mark.asyncio
async def test_answer_question_uses_llm_and_returns_numbered_sources(monkeypatch) -> None:
    """测的是正常路径：有 LLM 时走生成模式，并带编号引用。

    场景：把检索函数换成固定返回 1 条资料，LLM 换成 FakeLLM。
    FakeLLM 内部断言 system_prompt 含「只能依据」、user_message 含「[资料 1]」，
    以此校验提示词拼装正确（而不是只管返回值对不对）。
    期望：generation_mode 为 llm、retrieval_count 为 1、sources[0].citation 为 1、
    答案里保留 [1] 标注。
    重要性：引用编号是产品可信度的基础——用户点 [1] 要能跳回原文。
    """
    monkeypatch.setattr(settings, "llm_api_key", "test-key")
    # 伪造检索结果：一条与问题高度相关的知识文件片段
    monkeypatch.setattr(
        "app.services.rag.qa.search_similar_documents",
        lambda *args, **kwargs: [
            {
                "id": "doc-1",
                "source_id": "source-1",
                "source_type": "knowledge_file",
                "title": "auth.md#1",
                "snippet": "访问令牌过期后，客户端使用刷新令牌重试一次。",
                "score": 0.95,
                "metadata": {"filename": "auth.md"},
            }
        ],
    )

    class FakeLLM:
        """假 LLM：顺带校验传进去的提示词结构是否符合约定。"""

        async def chat_text(self, system_prompt: str, user_message: str) -> str:
            # 系统提示必须包含「只能依据」这一硬约束，防止模型自由发挥
            assert "只能依据" in system_prompt
            # 用户消息里资料必须按 [资料 1] 编号拼接
            assert "[资料 1]" in user_message
            return "访问令牌过期后应使用刷新令牌重试一次。[1]"

    # 第一个参数是 db 会话，这里的路径已被 mock 掉，用 object() 占位即可
    result = await answer_question(object(), "00000000-0000-0000-0000-000000000001", "登录过期后怎么办？", llm=FakeLLM())

    assert result["generation_mode"] == "llm"
    assert result["retrieval_count"] == 1
    assert result["sources"][0]["citation"] == 1
    assert "[1]" in result["answer"]


@pytest.mark.asyncio
async def test_answer_question_has_offline_extractive_fallback(monkeypatch) -> None:
    """测的是无 LLM 时的抽取式兜底。

    场景：llm_api_key 置为 None（未配置大模型），检索仍返回一条年假审批片段。
    期望：generation_mode 为 extractive；答案直接摘自原文（含「主管审批」）；
    仍带 [1] 引用标注。
    重要性：内网/离线部署常没有 LLM，此时必须还能用，不能返回空白页。
    """
    monkeypatch.setattr(settings, "llm_api_key", None)
    monkeypatch.setattr(
        "app.services.rag.qa.search_similar_documents",
        lambda *args, **kwargs: [
            {
                "id": "doc-1",
                "source_id": "source-1",
                "source_type": "knowledge_file",
                "title": "leave.md#1",
                "snippet": "员工申请年假时，需要先提交直属主管审批。审批通过后同步到考勤系统。",
                "score": 0.88,
                "metadata": {},
            }
        ],
    )

    result = await answer_question(object(), "00000000-0000-0000-0000-000000000001", "年假怎么审批？")

    assert result["generation_mode"] == "extractive"
    assert "主管审批" in result["answer"]
    assert "[1]" in result["answer"]


@pytest.mark.asyncio
async def test_answer_question_does_not_call_llm_when_gate_rejects_evidence(monkeypatch) -> None:
    """测的是证据闸门：判定证据冲突时必须短路，不进入 LLM。

    场景：retrieve_knowledge 被替换成返回「answer_gate 决策为 conflict」
    （高排名证据给出相反结论）的假实现；LLM 换成「一被调用就抛错」的哨兵。
    期望：generation_mode 为 no_evidence；answer_gate.decision 为 conflict；
    答案文案含「存在冲突」；且 LLM 从未被调用。
    重要性：这是防幻觉的关键闸门——证据自相矛盾时让模型硬答，
    比不答更糟糕；同时「不调用 LLM」也省下不必要的 token 成本。
    """
    monkeypatch.setattr(settings, "llm_api_key", "test-key")

    # 直接伪造 retrieve_knowledge 的三元组返回值：(文档列表, 闸门信息, 计数)
    async def fake_retrieve(*_args, **_kwargs):
        return (
            [
                {
                    "id": "doc-1",
                    "source_id": "source-1",
                    "source_type": "knowledge_file",
                    "title": "old.md",
                    "snippet": "旧版本允许部署。",
                    "score": 0.91,
                    "metadata": {"branch": "legacy"},
                    "retrieval": {"answer_gate": "conflict"},
                }
            ],
            {"answer_gate": {"decision": "conflict", "reason": "高排名证据给出了相反结论"}},
            1,
        )

    monkeypatch.setattr("app.services.rag.qa.retrieve_knowledge", fake_retrieve)

    class FakeDB:
        """占位 db：若检索没被 mock 掉，调用 query 会立刻报错暴露问题。"""

        def query(self):
            raise AssertionError("retrieve_knowledge is mocked")

    class FailingLLM:
        """哨兵 LLM：闸门拒绝证据后，任何调用都说明逻辑写错了。"""

        async def chat_text(self, _system_prompt: str, _user_message: str) -> str:
            raise AssertionError("LLM must not run when the answer gate rejects evidence")

    result = await answer_question(
        FakeDB(),
        "00000000-0000-0000-0000-000000000001",
        "现在允许部署吗？",
        llm=FailingLLM(),
    )

    assert result["generation_mode"] == "no_evidence"
    assert result["answer_gate"]["decision"] == "conflict"
    assert result["retrieval_count"] == 1
    # 文案要明确告知用户「证据冲突」，而不是给一个看似正常的答案
    assert "存在冲突" in result["answer"]
