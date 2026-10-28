"""工作区代码检索（不走向量库、直接扫本地 checkout）的测试。

与 RAG 语义检索互补：这里验证的是「用户提问后需要看真实代码」时用到的关键词检索能力——
查询词怎么切、如何跳过敏感文件、如何只搜某个子目录，以及最关键的一点：
因为直接读盘，所以文件改了立刻能搜到新内容，不需要重建索引。
"""

import uuid
from types import SimpleNamespace

from app.services.code_search import query_terms, search_checkout_code, search_repository_code


def test_query_terms_prioritizes_code_identifiers() -> None:
    """场景：一句中英混排的口语化提问，里面夹着驼峰函数名和文件路径。

    期望：能从问句里抽出 "refreshaccesstoken"（小写化后的标识符）和 "auth/service.py"。
    为什么重要：用户不会用英文关键词提问，提取不到标识符就只能全文模糊匹配，
    检索精度会断崖式下跌。
    """
    terms = query_terms("为什么 refreshAccessToken 在 auth/service.py 里失败？")

    # 标识符统一小写化，便于与源码做大小写无关匹配
    assert "refreshaccesstoken" in terms
    assert "auth/service.py" in terms


def test_search_checkout_code_reads_current_files_without_database_index(tmp_path) -> None:
    """场景：磁盘上只有一个 .py 文件和一个含密钥的 .env 文件，没有任何数据库索引。

    期望：能命中 src/auth.py，source_type=workspace_file、retrieval_mode=workspace，
    且结果里绝不会出现 .env。
    为什么重要：.env 里是密钥，被检索进模型上下文就等于泄露；
    "不依赖索引就能搜"则保证刚 clone 的仓库立即可用。
    """
    (tmp_path / "src").mkdir()
    source = tmp_path / "src" / "auth.py"
    source.write_text(
        "def refresh_access_token():\n"
        "    return rotate_credentials()\n",
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text("REFRESH_ACCESS_TOKEN=secret\n", encoding="utf-8")

    results = search_checkout_code(tmp_path, "refresh_access_token", limit=5)

    assert results
    assert results[0]["source_type"] == "workspace_file"
    # source_id 用相对路径，前端据此跳转到源码位置
    assert results[0]["source_id"] == "src/auth.py"
    assert results[0]["metadata"]["retrieval_mode"] == "workspace"
    assert all(item["source_id"] != ".env" for item in results)


def test_search_checkout_code_respects_worktree_subdirectory(tmp_path) -> None:
    """场景：同一个关键词 "backend-tests" 同时出现在 CI 配置和 .py 文件里，
    但调用时用 relative_path 限定只搜 .github/workflows 子目录。

    期望：结果只来自 .github/workflows/test.yml，src/test.py 被排除。
    为什么重要：大型仓库里同名词遍地都是；用户问「CI 为什么失败」时
    如果返回一堆 Python 文件，答案就会跑偏。
    """
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    (tmp_path / "src").mkdir()
    (tmp_path / ".github" / "workflows" / "test.yml").write_text("name: backend-tests\n", encoding="utf-8")
    (tmp_path / "src" / "test.py").write_text("backend_tests = True\n", encoding="utf-8")

    results = search_checkout_code(
        tmp_path,
        "backend-tests",
        relative_path=".github/workflows",
    )

    assert results and {item["source_id"] for item in results} == {".github/workflows/test.yml"}


def test_search_repository_code_uses_configured_local_checkout(tmp_path) -> None:
    """场景：仓库对象只配置了 local_path（没走过 clone 流程）。

    期望：仍能搜到内容，且结果 metadata 带上 repo_id，便于多仓库场景下区分来源。
    为什么重要：local 模式是私有代码接入的主要方式，
    这条路径若不支持，那些不能上云的团队就完全用不了代码检索。
    """
    (tmp_path / "service.py").write_text("def handle_payment_webhook():\n    pass\n", encoding="utf-8")
    # 用 SimpleNamespace 造最小 repo 对象，只提供被读取的字段
    repo = SimpleNamespace(id=uuid.uuid4(), local_path=str(tmp_path), full_name="local/demo")

    results = search_repository_code(repo, "handle_payment_webhook")

    assert results and results[0]["metadata"]["repo_id"] == str(repo.id)


def test_search_checkout_code_observes_file_changes_without_reindex(tmp_path) -> None:
    """场景：先搜一个旧函数名，然后原地改写文件，再搜新旧两个名字。

    期望：改写后能搜到新函数名，且旧函数名再也搜不到（返回空列表）。
    为什么重要：开发过程中代码时刻在变，"必须重建索引才能搜到最新代码"
    会让助手拿过时内容回答，直接给出错误结论。
    """
    source = tmp_path / "service.py"
    source.write_text("def old_handler():\n    pass\n", encoding="utf-8")
    # 第一轮：确认旧内容可被检索到
    assert search_checkout_code(tmp_path, "old_handler")

    # 覆盖写入新内容，不做任何索引刷新
    source.write_text("def current_handler():\n    pass\n", encoding="utf-8")

    assert search_checkout_code(tmp_path, "current_handler")
    # 旧内容必须彻底消失，证明读的是磁盘实时状态
    assert search_checkout_code(tmp_path, "old_handler") == []
