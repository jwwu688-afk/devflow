"""向量库（Milvus）与 embedding 契约的测试。

验证行为：RAG 的向量层有三条不能破的红线：
1) 写进 Milvus 的文本按 UTF-8 字节截断时不能把多字节字符截成乱码；
2) 向量维度以 Milvus collection 真实 schema 为准，且绝不做「截断/补零」凑维度；
3) embedding 失败时必须显式报错，不能静默退化成哈希向量——否则检索会
   在毫无提示的情况下变成随机结果，比直接报错危险得多。
"""

import uuid

import pytest

from app.core.config import settings
from app.db.models import Document
from app.services.rag.embeddings import embed_knowledge_texts, validate_embedding_contract
from app.services.rag.vector_store import _clip_utf8, _collection_dimensions, upsert_milvus_documents


def test_clip_utf8_limits_encoded_length_without_breaking_characters() -> None:
    """测的是 UTF-8 安全截断：按字节限长但不能把一个汉字切成两半。

    场景：用 3000 个「猫」字（每个 3 字节，共 9000 字节）去撞 8192 字节上限。
    期望：结果字节数不超限、非空、且每个字符仍是完整的「猫」。
    重要性：Milvus 字段有字节上限，若按字符数或直接切字节处理，
    会产生非法 UTF-8，导致写入失败或检索时出现乱码片段。
    """
    value = "\u732b" * 3000
    clipped = _clip_utf8(value, 8192)

    assert len(clipped.encode("utf-8")) <= 8192
    assert clipped
    # 关键：整串只能由完整汉字组成，不允许出现半个字符产生的替换符
    assert set(clipped) == {"\u732b"}


def test_collection_dimension_is_read_from_milvus_schema() -> None:
    """测的是维度来源：从 collection schema 的 embedding 字段读取，且能处理字符串。

    场景：Milvus describe_collection 返回的 dim 常是字符串 "1024" 而非整数。
    期望：解析成 int 1024。
    重要性：维度必须与线上 collection 实际一致，写死配置在换模型/重建索引后
    会静默错位，写入直接失败。
    """
    assert _collection_dimensions(
        {"fields": [{"name": "embedding", "params": {"dim": "1024"}}]}
    ) == 1024


def test_embedding_contract_rejects_per_repository_vector_spaces() -> None:
    """测的是 embedding 契约校验：不同模型产出的向量不能混在同一个空间里比较。

    场景：当前配置是 deterministic 模型，却传入 other-model 与 384 维。
    期望：抛 ValueError，错误信息含 "Embedding contract mismatch"。
    重要性：每个仓库若用各自的模型/维度，向量空间不一致，余弦相似度完全失去
    意义；这条校验把问题挡在写入前，而不是让检索结果悄悄变差。
    """
    with pytest.raises(ValueError, match="Embedding contract mismatch"):
        validate_embedding_contract("deterministic", "other-model", 384)


def test_explicit_deterministic_provider_is_allowed_without_becoming_a_fallback(monkeypatch) -> None:
    """测的是 deterministic provider 的「显式使用」路径。

    场景：配置本身就是 deterministic（测试/离线环境常用），并显式传入同样的
    provider/model/dimensions。
    期望：正常返回 1 条 8 维向量。
    重要性：deterministic 是合法的可选项，但只在显式选择时可用——
    它不能作为线上 provider 失败后的兜底（见下一个用例），这里先确认显式路径没被误拦。
    """
    monkeypatch.setattr(settings, "embedding_provider", "deterministic")
    monkeypatch.setattr(settings, "embedding_model", "deterministic-local")
    monkeypatch.setattr(settings, "embedding_dimensions", 8)

    vectors = embed_knowledge_texts(
        ["token failure"],
        provider="deterministic",
        model="deterministic-local",
        dimensions=8,
    )

    assert len(vectors) == 1
    assert len(vectors[0]) == 8


def test_embedding_provider_failure_is_not_replaced_with_hash_vectors(monkeypatch) -> None:
    """测的是「失败要冒泡」：真实 provider 挂掉时不得退化成哈希向量。

    场景：配置 openai_compatible，并把底层 _embed_openai_compatible 替换成
    必定抛 RuntimeError("provider offline") 的假实现。
    期望：异常原样抛出（不被 try/except 吞掉换成假向量）。
    重要性：这是最容易被「为了不报错」而写坏的逻辑——一旦静默降级，
    线上检索会返回看似正常实则无关的结果，排查成本极高。
    """
    monkeypatch.setattr(settings, "embedding_provider", "openai_compatible")
    monkeypatch.setattr(settings, "embedding_model", "configured-model")
    monkeypatch.setattr(settings, "embedding_dimensions", 8)
    # 用生成器表达式 + throw 构造「一调用就抛异常」的 lambda 替身
    monkeypatch.setattr(
        "app.services.rag.embeddings._embed_openai_compatible",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("provider offline")),
    )

    with pytest.raises(RuntimeError, match="provider offline"):
        embed_knowledge_texts(
            ["token failure"],
            provider="openai_compatible",
            model="configured-model",
            dimensions=8,
        )


def test_milvus_upsert_rejects_dimension_fitting_before_connecting(monkeypatch) -> None:
    """测的是维度不匹配要在「连 Milvus 之前」就拒绝。

    场景：配置维度 4，却只给了 1 条 2 维向量（[1.0, 0.0]）。
    期望：抛 ValueError，信息含 "Refusing to truncate or pad"。
    重要性：补零/截断能让写入「成功」，但向量语义已被破坏，检索质量无声崩坏；
    且提前失败可以避免在没有 Milvus 的环境里触发网络连接。
    """
    monkeypatch.setattr(settings, "embedding_dimensions", 4)
    # Document 用随机 uuid 即可，本用例只关心向量维度校验
    document = Document(
        id=uuid.uuid4(),
        repo_id=uuid.uuid4(),
        source_type="issue",
        source_id=uuid.uuid4(),
        title="Token failure",
        content="Refresh failed.",
        meta={},
    )

    with pytest.raises(ValueError, match="Refusing to truncate or pad"):
        upsert_milvus_documents([document], [[1.0, 0.0]])
