"""重排序（re-rank）服务的单元测试。

验证行为：召回阶段给出的候选文档，经 rerank_documents 重排后，
「路径/标题精确命中查询词」的文档要能压过「向量分虚高但内容不相关」的文档。
这是 RAG 质量的关键一环——向量检索只看语义相似度，容易把泛泛而谈的
README 排在真正含答案的源码文件前面，重排负责纠偏。
"""

from app.services.rag.rerank import rerank_documents


def test_rerank_prefers_exact_path_and_query_terms() -> None:
    """测的是「关键词/路径精确命中优先于纯向量分」这一排序策略。

    场景：构造两个候选。general 向量分更高（0.8 对 0.5）但内容和
    "login token auth middleware" 无关；auth 的标题正是
    backend/app/auth/middleware.py 且关键词分满分。
    期望：auth 排第一；rerank_score 单调不增；每条结果都带 rank_reason。
    重要性：重排若失效，用户问鉴权细节会拿到一篇 README，答案质量直接退化。
    """
    candidates = [
        # 干扰项：向量分高（0.8），但标题/正文都是通用本地开发说明，不含查询里的
        # 任何关键实体，模拟「语义相近但没用」的召回噪声。
        {
            "id": "general",
            "title": "README setup",
            "snippet": "General local development notes",
            "score": 0.8,
            "vector_score": 0.8,
            "keyword_score": 0.1,
            "metadata": {},
        },
        # 目标项：标题即文件路径（metadata.path 也一致），snippet 覆盖
        # token/login/session，keyword_score 给满分，模拟「路径 + 关键词」双命中。
        {
            "id": "auth",
            "title": "backend/app/auth/middleware.py#1",
            "snippet": "validate token and refresh session when login expires",
            "score": 0.5,
            "vector_score": 0.5,
            "keyword_score": 1.0,
            "metadata": {"path": "backend/app/auth/middleware.py"},
        },
    ]

    # limit=2 表示两条都保留，只校验相对顺序，不触及截断逻辑
    results = rerank_documents("login token auth middleware", candidates, limit=2)

    # 精确命中的源码文件必须反超向量分更高的 README
    assert results[0]["id"] == "auth"
    # 重排分数必须保持降序，不允许后一条高于前一条
    assert results[0]["rerank_score"] >= results[1]["rerank_score"]
    # 必须附带排序理由字段，便于调试与线上可观测
    assert "rank_reason" in results[0]
