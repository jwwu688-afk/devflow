"""评测集（suite）的发现与加载。

「评测集」就是一份 JSON 文件，放在 docs/datas/evals 下，人工维护、进版本库，
每条 case 描述一个问题 + 参考答案 + 期望命中的来源文件。
本模块只负责「把它读成结构化的 RagEvalCase 列表」，不负责执行（执行在 ragas_runner）。

两个关键概念：
- ready：评测集里引用的来源文件是否已经在当前仓库的文档索引里出现。
  依赖的文档没索引就直接跑，会导致检索必然失败，指标看起来像模型很差，
  所以未 ready 的评测集在加载阶段就报错，避免产出误导性结论。
- auto_smoke：不落盘的虚拟评测集，用当前仓库文档临时出题，只验证链路通不通。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.db.models import Document, Repository
from app.services.evals.ragas_runner import RagEvalCase


# parents[4]：evals → services → app → backend → 项目根，再拼 docs/datas/evals，
# 用文件位置反推而不是读配置，保证测试与非测试环境指向同一份评测集。
DEFAULT_SUITE_DIR = Path(__file__).resolve().parents[4] / "docs" / "datas" / "evals"
# 虚拟评测集的固定 id，前端靠它区分「真实文件」与「临时冒烟」。
AUTO_SMOKE_SUITE_ID = "auto-document-smoke"


@dataclass(frozen=True)
class LoadedEvalSuite:
    """加载好的评测集：元信息（用于展示/入库去重）+ 可执行的用例列表。

    frozen 是为了让调用方没法在运行过程中偷偷改 case，保证一次运行的输入稳定。
    """

    metadata: dict[str, Any]
    cases: list[RagEvalCase]


def _read_suite(path: Path) -> tuple[dict[str, Any], str]:
    """读取并做最基础的结构校验，返回 (payload, 文件内容的 sha256)。

    sha256 基于原始字节计算：评测集内容一变指纹就变，
    上层可以用它判断「这次结果对应的是哪一版评测集」，避免改了题还拿旧分数对比。
    """
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"评测集 {path.name} 必须是 JSON 对象")
    name = str(payload.get("name") or "").strip()
    cases = payload.get("cases")
    # 三件事必须成立：有名字（用于 id）、cases 是数组、且非空。
    # 空评测集会让平均值算出 NaN，所以这里直接拒绝。
    if not name or not isinstance(cases, list) or not cases:
        raise ValueError(f"评测集 {path.name} 缺少 name 或 cases")
    return payload, hashlib.sha256(raw).hexdigest()


def _matching_documents(db: Session, repo_id: Any, source_file: str) -> list[Document]:
    """在仓库已索引的文档里，找出与 source_file 对应的那些。

    匹配是「文件名级」的模糊匹配而不是路径全等：同一份文档可能因为上传路径不同
    而产生多条记录，甚至一个文件被切分成多个 Document；全等匹配会漏。
    用 casefold 做大小写无关比较（Windows/macOS 上文件名大小写不敏感）。
    一次把仓库文档全查出来在内存里比，是因为评测集规模小、文档量也不大，
    比在 SQL 里对 JSON 字段做三段 LIKE 更简单也更可控。
    """
    expected = Path(source_file).name.casefold()
    documents = (
        db.query(Document)
        .filter(Document.repo_id == repo_id)
        .order_by(Document.created_at.asc())
        .all()
    )
    matches: list[Document] = []
    for document in documents:
        metadata = document.meta or {}
        # 三个候选字段都参与比较：不同入库路径写入的字段不同，谁也覆盖不了谁。
        candidates = {
            Path(str(metadata.get("filename") or "")).name.casefold(),
            Path(str(metadata.get("display_title") or "")).name.casefold(),
            Path(str(metadata.get("source_file") or "")).name.casefold(),
        }
        if expected in candidates:
            matches.append(document)
    return matches


def _suite_applies_to_repo(payload: dict[str, Any], repo: Repository) -> bool:
    """判断评测集是否适用于该仓库。

    未声明 repository_names 视为「通用评测集」，所有仓库都能用；
    声明了就必须命中 name 或 full_name（大小写不敏感），
    这样可以为特定项目维护专属题库，而不会出现在别的项目的列表里。
    """
    names = payload.get("repository_names")
    if not isinstance(names, list) or not names:
        return True
    accepted = {str(item).casefold() for item in names if str(item).strip()}
    return repo.name.casefold() in accepted or repo.full_name.casefold() in accepted


def _suite_metadata(
    db: Session,
    repo: Repository,
    path: Path,
    payload: dict[str, Any],
    sha256: str,
) -> dict[str, Any]:
    """汇总评测集的展示信息，并顺带算出 ready 状态。

    ready 的判定方式：逐条 case 检查它声明的 source_file 是否能在索引里找到；
    有一条找不到就算未就绪，并把缺失的文件名回传给前端提示用户先索引。
    """
    missing_sources: list[str] = []
    for item in payload["cases"]:
        source_file = str(item.get("source_file") or "").strip()
        # 没声明 source_file 的 case（例如只校验拒答）不参与依赖检查。
        if source_file and not _matching_documents(db, repo.id, source_file):
            missing_sources.append(source_file)
    # dict.fromkeys 去重且保持原顺序：多个 case 引用同一文件时只提示一次。
    missing_sources = list(dict.fromkeys(missing_sources))
    return {
        # id 直接用 name：评测集文件名可能被重命名，但 name 是内容里的稳定标识。
        "id": str(payload["name"]),
        "name": str(payload["name"]),
        "label": str(payload.get("label") or payload["name"]),
        "description": str(payload.get("description") or "人工维护的固定回归评测集"),
        # schema_version 预留给将来结构升级时做兼容分支，当前都按 1 处理。
        "schema_version": int(payload.get("schema_version") or 1),
        "case_count": len(payload["cases"]),
        "sha256": sha256,
        "source": "fixed",
        "filename": path.name,
        "ready": not missing_sources,
        "missing_sources": missing_sources,
        "recommended": bool(payload.get("recommended", False)),
    }


def list_eval_suites(
    db: Session,
    repo_id: Any,
    *,
    suite_dir: Path = DEFAULT_SUITE_DIR,
) -> list[dict[str, Any]]:
    """列出该仓库可用的全部评测集，最后追加一条「自动冒烟」虚拟评测集。

    排序规则：推荐的在前、就绪的在前、最后按 label 字典序，保证同一份数据
    每次返回的顺序一致（前端不做二次排序也能稳定展示）。
    suite_dir 可注入，方便测试时指向临时目录。
    """
    repo = db.get(Repository, repo_id)
    if repo is None:
        raise ValueError(f"Repository {repo_id} was not found")
    suites: list[dict[str, Any]] = []
    # 目录不存在时静默跳过（只保留自动冒烟项），不让缺目录导致整个功能 500。
    if suite_dir.is_dir():
        # sorted 保证文件遍历顺序固定，便于复现和 diff。
        for path in sorted(suite_dir.glob("*.json")):
            payload, sha256 = _read_suite(path)
            if _suite_applies_to_repo(payload, repo):
                suites.append(_suite_metadata(db, repo, path, payload, sha256))
    suites.sort(key=lambda item: (not item["recommended"], not item["ready"], item["label"]))
    # 自动冒烟永远排在最后：它是「临时出题」，不该和人工维护的基线混在一起优先展示。
    suites.append(
        {
            "id": AUTO_SMOKE_SUITE_ID,
            "name": AUTO_SMOKE_SUITE_ID,
            "label": "自动文档冒烟测试",
            "description": "从当前仓库文档临时出题，只验证链路能运行，不能作为发布基线。",
            "schema_version": 1,
            "case_count": None,
            "sha256": None,
            "source": "auto_smoke",
            "filename": None,
            "ready": True,
            "missing_sources": [],
            "recommended": False,
        }
    )
    return suites


def load_eval_suite(
    db: Session,
    repo_id: Any,
    suite_id: str,
    *,
    suite_dir: Path = DEFAULT_SUITE_DIR,
) -> LoadedEvalSuite:
    """按 id 加载一个评测集，并把 JSON 里的每条 case 转成 RagEvalCase。

    自动冒烟集返回空 cases：它的题目是运行时才从仓库文档里生成的，
    这里只给出元信息，真正的出题逻辑在 runner/ragas_runner 里。
    """
    if suite_id == AUTO_SMOKE_SUITE_ID:
        return LoadedEvalSuite(
            metadata={
                "id": AUTO_SMOKE_SUITE_ID,
                "name": AUTO_SMOKE_SUITE_ID,
                "label": "自动文档冒烟测试",
                "description": "从当前仓库文档临时出题，只验证链路能运行，不能作为发布基线。",
                "source": "auto_smoke",
                "sha256": None,
                "case_count": None,
            },
            cases=[],
        )

    repo = db.get(Repository, repo_id)
    if repo is None:
        raise ValueError(f"Repository {repo_id} was not found")
    # 条件表达式放在 for 的位置：目录不存在时遍历空列表，避免再写一层 if 缩进。
    for path in sorted(suite_dir.glob("*.json")) if suite_dir.is_dir() else []:
        payload, sha256 = _read_suite(path)
        # 同名但声明了别的仓库的评测集要跳过，否则会把 A 项目的题跑到 B 项目上。
        if str(payload["name"]) != suite_id or not _suite_applies_to_repo(payload, repo):
            continue
        metadata = _suite_metadata(db, repo, path, payload, sha256)
        if not metadata["ready"]:
            # 依赖文档没索引就报错：否则检索必然为空，得出的低分会被误读成模型退化。
            missing = "、".join(metadata["missing_sources"])
            raise ValueError(f"评测集 {suite_id} 缺少已索引来源：{missing}")
        cases: list[RagEvalCase] = []
        # start=1：编号从 1 开始，生成默认 id 时更符合人的阅读习惯。
        for index, item in enumerate(payload["cases"], start=1):
            case_id = str(item.get("id") or f"case-{index}").strip()
            question = str(item.get("question") or "").strip()
            reference = str(item.get("reference") or "").strip()
            # 问题和参考答案是评分的基础，缺任何一个都没法判分，直接失败而不是跳过。
            if not question or not reference:
                raise ValueError(f"评测集 {suite_id} 的 {case_id} 缺少 question 或 reference")
            source_file = str(item.get("source_file") or "").strip()
            documents = _matching_documents(db, repo.id, source_file) if source_file else []
            # 期望命中的 ID 同时收录 document.id 和 document.source_id：
            # 检索层可能返回任一种 id 形式，两种都算命中，避免因为 id 口径不同误判为未召回。
            expected_source_ids = list(
                dict.fromkeys(
                    str(value)
                    for document in documents
                    for value in (document.id, document.source_id)
                    if value
                )
            )
            cases.append(
                RagEvalCase(
                    id=case_id,
                    question=question,
                    reference=reference,
                    expected_source_ids=expected_source_ids,
                    # 默认按「知识文件」处理；代码类问题可在 JSON 里显式写 source_type。
                    source_type=str(item.get("source_type") or "knowledge_file"),
                    required_tools=[str(value) for value in item.get("required_tools") or []],
                    # 深拷贝成 dict：RagEvalCase 是可变对象，不该和原始 payload 共享引用。
                    expected_tool_calls=[dict(value) for value in item.get("expected_tool_calls") or []],
                    # kind 默认 positive，负例（如应当拒答）需显式声明。
                    kind=str(item.get("kind") or "positive"),
                    must_include=[str(value) for value in item.get("must_include") or []],
                    must_not_include=[str(value) for value in item.get("must_not_include") or []],
                    # 空字符串归一成 None，得分逻辑里就能用「is None」判断是否启用正则校验。
                    answer_regex=str(item.get("answer_regex") or "").strip() or None,
                    source_file=source_file or None,
                )
            )
        # 用实际加载到的条数覆盖文件里的声明值，两者不一致时以真实情况为准。
        metadata["case_count"] = len(cases)
        return LoadedEvalSuite(metadata=metadata, cases=cases)
    raise ValueError(f"未找到评测集：{suite_id}")
