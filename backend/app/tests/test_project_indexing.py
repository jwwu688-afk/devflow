"""项目文档索引（Project Index）的测试。

"项目文档"指 README、docs/ 下的 ADR/架构文档、package.json 等清单文件。
索引它们的作用是：回答"这个项目是做什么的"时不必读源码，直接用高质量文档。
本文件验证：
1. 扫描只做"预检/落库"，不依赖 embedding 与 Milvus（离线也能跑）；
2. 文件变化后状态会变 stale，但被 allowlist 排除的文件变化不算 stale；
3. allowlist 生效后，排除目录（课程材料、演示代码）必须从 Document、EvidenceItem、
   全文检索与知识图谱里全部消失——这是内容隔离的关键；
4. 无配置时退回旧的自动发现模式（向下兼容）；
5. snooze 能把索引置为 missing 并设冷却期。
"""

import os

# 必须先于 app.* 导入设置，db 模块导入时会读取
os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import Base, Document, EvidenceItem, Repository
from app.services.knowledge_graph import _collect_nodes
from app.services.project_indexing import discover_project_docs, get_project_index_state, scan_project_index, snooze_project_index
from app.services.rag.retrieval import search_similar_documents


def _db_session():
    """独立内存库，用例之间互不影响。"""
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _write_allowlist_config(project) -> None:
    """在项目里写一份 .devflow/index.yml，声明"只索引白名单内的文档"。

    故意同时配置了 include（README.md、docs/architecture/**）和 exclude
    （docs/course/**、"docs/demo code/**"，后者带空格用于验证引号路径能正确解析）。
    """
    config_dir = project / ".devflow"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "index.yml").write_text(
        """version: 1
project_docs:
  mode: allowlist
  include:
    - README.md
    - docs/architecture/**
  exclude:
    - docs/course/**
    - "docs/demo code/**"
include_manifests: true
""",
        encoding="utf-8",
    )


def test_scan_project_index_writes_state_and_project_documents(tmp_path, monkeypatch) -> None:
    """场景：项目里有 README.md、docs/ADR-001.md、package.json，无任何索引配置；扫描索引。

    期望：状态 ready；docs_total=3、docs_indexed>=4（文档会按块拆分）；
    tierCoverage.authoritative=2（README + ADR 属于权威来源）；
    Document 表按类型各写入 project_doc / project_manifest / project_overview，
    并且这些行上不含 embedding 字段（索引与向量化是两件事）。
    为什么重要：预检必须能独立于 embedding 运行（向量服务不可用时也要能告诉用户"有多少文档待索引"）；
    来源分层（authoritative/manifest/overview）则决定检索时的排序权重。
    """
    def fail_if_vector_runtime_is_used(*_args, **_kwargs):
        # 埋雷：一旦扫描过程调用了向量化/写 Milvus，测试立即失败
        raise AssertionError("project preflight must not depend on embeddings or Milvus")

    monkeypatch.setattr("app.services.rag.vector_store.embed_texts", fail_if_vector_runtime_is_used)
    monkeypatch.setattr("app.services.rag.vector_store.delete_milvus_documents", fail_if_vector_runtime_is_used)
    monkeypatch.setattr("app.services.rag.vector_store.upsert_milvus_documents", fail_if_vector_runtime_is_used)
    project = tmp_path / "demo"
    (project / "docs").mkdir(parents=True)
    (project / "README.md").write_text("# Demo\n\nProject overview.", encoding="utf-8")
    (project / "docs" / "ADR-001.md").write_text("# ADR 001\n\nUse FastAPI.", encoding="utf-8")
    (project / "package.json").write_text('{"name":"demo","version":"1.0.0"}', encoding="utf-8")

    db = _db_session()
    repo = Repository(owner="local", name="demo", full_name="local/demo", local_path=str(project), checkout_mode="local")
    db.add(repo)
    db.flush()

    state = scan_project_index(db, repo)

    assert state["status"] == "ready"
    # 3 个源文件，但检查结果按"文档块"计数，所以 docs_indexed 可能更多
    assert state["docs_total"] == 3
    assert state["docs_indexed"] >= 4
    assert state["summary_json"]["tierCoverage"]["authoritative"] == 2
    assert db.query(Document).filter(Document.repo_id == repo.id, Document.source_type == "project_doc").count() >= 2
    assert db.query(Document).filter(Document.repo_id == repo.id, Document.source_type == "project_manifest").count() == 1
    project_doc = db.query(Document).filter(Document.repo_id == repo.id, Document.source_type == "project_doc").first()
    # manifest（package.json）走"直接检索"，不需要向量化
    manifest = db.query(Document).filter(Document.repo_id == repo.id, Document.source_type == "project_manifest").one()
    overview = db.query(Document).filter(Document.repo_id == repo.id, Document.source_type == "project_overview").one()
    # 扫描阶段不得写入任何 embedding 相关字段，向量化由后续独立任务负责
    assert project_doc is not None and "embedding_dimensions" not in project_doc.meta
    assert not hasattr(project_doc, "embedding")
    assert manifest.meta["retrieval_mode"] == "direct"
    # overview 同样不该带向量字段
    assert not hasattr(overview, "embedding")


def test_project_index_state_reports_stale_when_files_change(tmp_path) -> None:
    """场景：先扫描一次（状态 ready），然后往项目里新增一个 CHANGELOG.md。

    期望：不重新扫描，只查状态就返回 stale。
    为什么重要：状态查询要极其轻量（前端每次进页面都会调），
    用它判断"要不要提示用户重建索引"，比跑一次完整扫描便宜得多。
    """
    project = tmp_path / "demo"
    project.mkdir()
    (project / "README.md").write_text("# Demo\n\nProject overview.", encoding="utf-8")

    db = _db_session()
    repo = Repository(owner="local", name="demo", full_name="local/demo", local_path=str(project), checkout_mode="local")
    db.add(repo)
    db.flush()

    assert scan_project_index(db, repo)["status"] == "ready"
    (project / "CHANGELOG.md").write_text("# Changelog\n\nInitial release.", encoding="utf-8")

    assert get_project_index_state(db, repo)["status"] == "stale"


def test_allowlist_rescan_removes_course_docs_from_rag_evidence_and_graph(tmp_path) -> None:
    """场景：第一次（无配置）把 docs/course、docs/demo code 里的教学内容也索引了；
    随后加了 allowlist 配置重新扫描。

    期望：状态 ready、docs_total=3、策略为 allowlist、excludedFiles=2；
    只留下 README.md 与 docs/architecture/system.md；
    .data/ 这类运行时产物目录不进索引；
    两个标记串（COURSE_ONLY_MARKER / DEMO_ONLY_MARKER）从 EvidenceItem、
    全文检索结果、知识图谱节点里全部消失，而架构文档仍可被检索到。
    为什么重要：教学内容/示例代码混进项目检索，会让模型把教程里的说法当成
    "这个项目的实际架构"；这是最容易造成答非所问、且用户极难察觉的污染。
    """
    project = tmp_path / "demo"
    (project / "docs" / "architecture").mkdir(parents=True)
    (project / "docs" / "course").mkdir(parents=True)
    (project / "docs" / "demo code").mkdir(parents=True)
    (project / ".data" / "runtime-checkout").mkdir(parents=True)
    (project / "README.md").write_text("# Demo\n\nProject overview.", encoding="utf-8")
    (project / "docs" / "architecture" / "system.md").write_text(
        "# System architecture\n\nFastAPI service boundary.", encoding="utf-8"
    )
    (project / "docs" / "course" / "lesson.md").write_text(
        "# Course lesson\n\nCOURSE_ONLY_MARKER should never reach project retrieval.", encoding="utf-8"
    )
    (project / "docs" / "demo code" / "README.md").write_text(
        "# Course demo\n\nDEMO_ONLY_MARKER is teaching material.", encoding="utf-8"
    )
    (project / "package.json").write_text('{"name":"demo","version":"1.0.0"}', encoding="utf-8")
    (project / ".data" / "runtime-checkout" / "package.json").write_text(
        '{"name":"runtime-copy","version":"1.0.0"}', encoding="utf-8"
    )

    db = _db_session()
    repo = Repository(owner="local", name="demo", full_name="local/demo", local_path=str(project), checkout_mode="local")
    db.add(repo)
    db.flush()

    assert scan_project_index(db, repo)["status"] == "ready"
    # 加配置前，课程文档确实被索引了（为下面"重新扫描后消失"提供对照）
    assert any(
        str((doc.meta or {}).get("path") or "").startswith("docs/course/")
        for doc in db.query(Document).filter(Document.repo_id == repo.id, Document.source_type == "project_doc").all()
    )

    _write_allowlist_config(project)
    state = scan_project_index(db, repo)

    # 重新扫描后库里实际留下的路径集合
    indexed_paths = {
        str((doc.meta or {}).get("path") or "")
        for doc in db.query(Document).filter(Document.repo_id == repo.id, Document.source_type == "project_doc").all()
    }
    assert state["status"] == "ready"
    assert state["docs_total"] == 3
    assert state["summary_json"]["indexPolicy"]["mode"] == "allowlist"
    # 两个被排除的目录各 1 个文件
    assert state["summary_json"]["excludedFiles"] == 2
    assert "README.md" in indexed_paths
    assert "docs/architecture/system.md" in indexed_paths
    assert not any(path.startswith("docs/course/") or path.startswith("docs/demo code/") for path in indexed_paths)
    assert not any(path.startswith(".data/") for path in indexed_paths)
    assert not any(
        "COURSE_ONLY_MARKER" in item.content or "DEMO_ONLY_MARKER" in item.content
        for item in db.query(EvidenceItem).filter(EvidenceItem.repo_id == repo.id).all()
    )
    # 全文检索也搜不到被排除的内容：说明不只是文档行被删，检索链路也干净了
    assert not search_similar_documents(
        db,
        repo.id,
        "COURSE_ONLY_MARKER",
        retrieval_method="fulltext",
        apply_rerank=False,
    )
    assert any(
        item["source_type"] == "project_doc"
        for item in search_similar_documents(
            db,
            repo.id,
            "FastAPI service boundary",
            retrieval_method="fulltext",
            apply_rerank=False,
        )
    )
    graph_nodes = _collect_nodes(db, repo.id).values()
    # 图谱节点里也不能残留被排除的路径
    assert not any(
        str((node.get("metadata") or {}).get("path") or "").startswith(("docs/course/", "docs/demo code/"))
        for node in graph_nodes
    )


def test_excluded_files_do_not_make_project_index_stale(tmp_path) -> None:
    """场景：索引完成后，先改被排除的课程文档，再改被包含的架构文档。

    期望：改课程文档后状态仍是 ready（不算过期）；改架构文档后变 stale。
    为什么重要：把被排除文件也算进指纹，会导致"每次改教程都提示重建索引"，
    用户会觉得提示不可信、最后干脆无视所有索引提醒。
    """
    project = tmp_path / "demo"
    (project / "docs" / "architecture").mkdir(parents=True)
    (project / "docs" / "course").mkdir(parents=True)
    (project / "README.md").write_text("# Demo\n\nProject overview.", encoding="utf-8")
    architecture = project / "docs" / "architecture" / "system.md"
    architecture.write_text("# Architecture\n\nVersion one.", encoding="utf-8")
    course = project / "docs" / "course" / "lesson.md"
    course.write_text("# Lesson\n\nVersion one.", encoding="utf-8")
    _write_allowlist_config(project)

    db = _db_session()
    repo = Repository(owner="local", name="demo", full_name="local/demo", local_path=str(project), checkout_mode="local")
    db.add(repo)
    db.flush()

    assert scan_project_index(db, repo)["status"] == "ready"
    # 改被排除的文件：不应影响索引有效性
    course.write_text("# Lesson\n\nA much longer excluded course revision.", encoding="utf-8")
    assert get_project_index_state(db, repo)["status"] == "ready"

    # 改被包含的文件：必须立刻反映为 stale，提示用户重建
    architecture.write_text("# Architecture\n\nVersion two changes project behavior.", encoding="utf-8")
    assert get_project_index_state(db, repo)["status"] == "stale"


def test_discover_project_docs_defaults_to_automatic_mode_without_config(tmp_path) -> None:
    """场景：项目里没有任何 .devflow/index.yml，只有一份 docs/course/lesson.md。

    期望：仍然按旧的自动模式把该文件发现出来（列表为 ["docs/course/lesson.md"]）。
    为什么重要：老仓库不会立刻补配置；升级后如果突然什么都不索引，
    等于所有存量用户的文档检索一夜之间失效，属于破坏性变更。
    """
    project = tmp_path / "demo"
    (project / "docs" / "course").mkdir(parents=True)
    lesson = project / "docs" / "course" / "lesson.md"
    lesson.write_text("# Lesson\n\nLegacy automatic discovery remains compatible.", encoding="utf-8")

    assert [item.relative_path for item in discover_project_docs(project)] == ["docs/course/lesson.md"]


def test_snooze_project_index_sets_missing_state_with_cooldown(tmp_path) -> None:
    """场景：用户选择"暂不建立项目索引"，冷却 3 天。

    期望：索引状态被置为 missing，且 snoozed_until 有值（冷却截止时间）。
    为什么重要：否则每次打开页面都会再弹一次"建议建立索引"，
    用户明确拒绝后必须尊重，等到冷却结束再提醒。
    """
    db = _db_session()
    repo = Repository(owner="local", name="demo", full_name="local/demo", local_path=str(tmp_path), checkout_mode="local")
    db.add(repo)
    db.flush()

    row = snooze_project_index(db, repo, days=3)

    assert row.status == "missing"
    assert row.snoozed_until is not None
