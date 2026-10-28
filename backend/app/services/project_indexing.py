"""项目文档索引：自动发现仓库里的「说明性文档」并写进知识库。

和 code_analysis 的区别：那边索引代码符号，这边索引人写的文档
（README、docs/*.md、依赖清单），供 RAG 回答「这个项目是干什么的」这类问题。

三个设计要点：
1. **只挑文档**：靠文件名/目录/扩展名规则分类（_classify_file），不读全仓库，
   避免把日志、源码、生成物也塞进知识库。
2. **可信度分层**（tier）：authoritative（README/docs 正式文档）> derived（依赖清单推导）
   > soft_clue（CHANGELOG 等参考价值较低），检索时可以按 tier 加权或过滤。
3. **指纹 + 状态机**：用 (路径, mtime, 大小) 的哈希作为指纹，指纹变了就把 ready 判成 stale，
   前端据此提示「文档更新了，是否重新索引」，而不用每次都全量重建。
"""

import fnmatch
import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy.orm import Session

from app.db.models import ProjectIndexState, Repository
from app.db.session import SessionLocal
from app.services.knowledge_graph import rebuild_knowledge_graph
from app.services.rag.chunking import chunk_document
from app.services.code_analysis import _repo_checkout_path, _stored_checkout_path
from app.services.rag.vector_store import DocumentPayload, replace_documents, stable_source_id

# 本模块负责（也是唯一负责）的三种 source_type，重建时只替换这三类，
# 不会误删 memory_note 等其它来源的文档。
PROJECT_INDEX_SOURCE_TYPES = ["project_overview", "project_doc", "project_manifest"]
# 索引状态机的合法取值；库里出现别的值时一律按 missing 处理（见 _state_payload）。
PROJECT_INDEX_STATUSES = {"missing", "stale", "building", "ready", "failed"}
# 项目内可选的配置文件路径（放在被索引仓库里，随代码一起走版本控制）。
PROJECT_INDEX_CONFIG_PATH = Path(".devflow/index.yml")
# auto = 默认规则自动发现；allowlist = 只索引 include 里显式列出的文件。
PROJECT_INDEX_MODES = {"auto", "allowlist"}

# 顶层这些文件被视为「权威文档」：项目定位、架构、贡献指南通常写在这里。
AUTHORITATIVE_TOP_LEVEL = {"README", "ARCHITECTURE", "CONTRIBUTING"}
# 依赖清单：用来推导技术栈，内容结构稳定，归为 derived（由机器生成/推导）。
MANIFEST_FILES = {
    "package.json",
    "requirements.txt",
    "pyproject.toml",
    "poetry.lock",
    "pnpm-workspace.yaml",
    "Cargo.toml",
    "go.mod",
    "pom.xml",
    "build.gradle",
    "Gemfile",
    "composer.json",
}
# 变更日志类：有信息但是「历史」而非「现状」，只作为弱线索（soft_clue）。
SOFT_CLUE_TOP_LEVEL = {"CHANGELOG", "HISTORY", "RELEASES"}
# 依赖/构建/缓存目录，与其它扫描模块保持一致的跳过集合。
SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".next",
    ".nuxt",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    "coverage",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}
# 默认跳过所有隐藏目录（.git/.venv…），但 .github 里确实有文档（ISSUE_TEMPLATE），单独放行。
INDEXABLE_HIDDEN_DIRS = {".github"}
# 敏感文件：名字精确匹配 + 后缀黑名单 + credentials 前缀，三重判断见 _is_secret。
SECRET_NAMES = {".env", ".env.local", ".env.production", ".npmrc", ".pypirc", "credentials.json"}
SECRET_SUFFIXES = {".key", ".pem", ".p12", ".pfx"}


@dataclass(frozen=True)
class ProjectDocCandidate:
    """一个待索引的文档候选（发现阶段产出，尚未读取内容）。"""

    path: Path
    relative_path: str
    source_type: str
    tier: str


@dataclass(frozen=True)
class ProjectIndexPolicy:
    """索引策略，来自仓库内的 .devflow/index.yml；没有配置文件时用这里的默认值。

    frozen 保证同一次扫描里策略不会被中途改掉（策略会写进指纹，中途变会导致指纹失真）。
    include/exclude 用 tuple 而不是 list：tuple 不可变，可以安全地放进 frozen dataclass。
    """

    mode: str = "auto"
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    include_manifests: bool = True
    config_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """转成前端/指纹用的 dict（键名用驼峰，和前端约定一致）。"""
        return {
            "mode": self.mode,
            "include": list(self.include),
            "exclude": list(self.exclude),
            "includeManifests": self.include_manifests,
            "configPath": self.config_path,
        }


@dataclass(frozen=True)
class ProjectDocDiscovery:
    """一次文档发现的结果：候选、被策略排除的路径、所用策略、是否被数量上限截断。"""

    candidates: list[ProjectDocCandidate]
    excluded_paths: list[str]
    policy: ProjectIndexPolicy
    truncated: bool = False


def _now() -> datetime:
    """统一取 UTC 当前时间：库里所有时间都按 UTC 存，避免跨时区比较出错。"""
    return datetime.now(timezone.utc)


def _aware(value: datetime | None) -> datetime | None:
    """把可能缺失时区的时间补成 UTC。

    SQLite 读回来的是 naive datetime，直接和 aware 时间比较会抛 TypeError，
    所以所有从库/接口拿到的旧时间都要先过一遍这里。
    """
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _repo_root(repo: Repository) -> Path:
    """仓库代码根目录：优先本地路径（local 模式），否则用托管 checkout 目录。"""
    stored = _stored_checkout_path(repo)
    if stored:
        return stored
    return _repo_checkout_path(repo)


def _safe_project_root(repo: Repository) -> Path | None:
    """取一个「确实存在且可解析」的根目录，取不到返回 None。

    单独封装是因为 scan 流程里要先判断根目录是否可用再决定报什么错，
    而 resolve() 在路径过长/权限异常时会抛 OSError，这里统一吞掉当「不可用」。
    """
    root = _repo_root(repo)
    if not root.exists() or not root.is_dir():
        return None
    try:
        return root.resolve()
    except OSError:
        return None


def _is_secret(path: Path) -> bool:
    """判断是否敏感文件：精确名 / 私钥类后缀 / credentials 前缀三选一。

    前缀匹配兜住 credentials.json、credentials.yaml 这类变体。
    """
    name = path.name
    return name in SECRET_NAMES or path.suffix.lower() in SECRET_SUFFIXES or name.lower().startswith("credentials")


def _read_text(path: Path, max_bytes: int = 400_000) -> str | None:
    """读文本文件；超限/二进制/不可读都返回 None。

    上限比代码索引（180KB）宽松到 400KB：文档通常比单文件代码长，
    但也仍要设限，否则一份自动生成的巨型 md 会拖慢整次扫描。
    """
    try:
        with path.open("rb") as handle:
            data = handle.read(max_bytes + 1)
    except OSError:
        return None
    if not data or len(data) > max_bytes or b"\x00" in data[:4096]:
        return None
    for encoding in ["utf-8-sig", "utf-8", "gb18030", "latin-1"]:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


def _stem_upper(path: Path) -> str:
    """取大写文件名主干：README.md / readme.MD 都能归一成 README 再比对。"""
    return path.stem.upper()


def _classify_file(path: Path, rel: str) -> tuple[str, str] | None:
    """判断文件该不该索引、索引成什么类型、可信度属于哪一档。

    返回 (source_type, tier)，不认识的文件返回 None（即不索引）。
    规则刻意保守（只看 .md 和依赖清单）：宁可漏索引，也不要把源码当文档塞进知识库，
    那会让 RAG 在回答产品问题时召回一堆代码片段。
    """
    if _is_secret(path):
        return None
    name = path.name
    stem = _stem_upper(path)
    # 统一成 posix 分隔符，避免 Windows 上 "docs\\x.md" 匹配不到 "docs/" 前缀。
    rel_posix = rel.replace("\\", "/")
    if path.suffix.lower() == ".md":
        # docs/ 目录下的所有 md、顶层 README/ARCHITECTURE/CONTRIBUTING、以及 ADR 架构决策记录。
        if rel_posix.startswith("docs/") or stem in AUTHORITATIVE_TOP_LEVEL or stem.startswith("ADR"):
            return "project_doc", "authoritative"
        # 变更日志和 issue 模板只算弱线索：前者讲历史、后者是表单，都不是项目现状说明。
        if stem in SOFT_CLUE_TOP_LEVEL or rel_posix.startswith(".github/ISSUE_TEMPLATE/"):
            return "project_doc", "soft_clue"
    if name in MANIFEST_FILES:
        # 依赖清单不区分是否 md，按文件名精确匹配（大小写敏感，因为生态里就这些固定写法）。
        return "project_manifest", "derived"
    return None


def _is_inside(root: Path, candidate: Path) -> bool:
    """判断 candidate 是否在 root 之内。

    解析符号链接后再比：防止仓库里放一个指向 /etc 的软链把外部文件读进来。
    OSError（坏链接）和 ValueError（不在其下）都当「不在里面」。
    """
    try:
        candidate.resolve().relative_to(root)
        return True
    except (OSError, ValueError):
        return False


def _normalize_patterns(value: Any, field_name: str) -> tuple[str, ...]:
    """规范化 include/exclude 里的 glob 模式，并做安全/格式校验。

    处理内容：
    - 反斜杠转成正斜杠、去掉开头的 ./ 和 /，让用户怎么写都能匹配；
    - 拒绝 ".." 开头的模式（防止把索引范围指到仓库外）；
    - 以 / 结尾的目录模式自动补 "**"，否则 fnmatch 只会匹配目录本身而不是里面所有文件；
    - 最后去重并保持原顺序（dict.fromkeys）。
    任何格式问题都抛 ValueError，error message 带上字段名便于用户定位。
    """
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list of repository-relative glob patterns")
    patterns: list[str] = []
    for raw_pattern in value:
        if not isinstance(raw_pattern, str) or not raw_pattern.strip():
            raise ValueError(f"{field_name} entries must be non-empty strings")
        pattern = raw_pattern.strip().replace("\\", "/")
        # 用 while 而不是 if：用户可能写出 "./././docs"。
        while pattern.startswith("./"):
            pattern = pattern[2:]
        pattern = pattern.lstrip("/")
        if pattern == ".." or pattern.startswith("../"):
            raise ValueError(f"{field_name} patterns must stay inside the repository")
        if pattern.endswith("/"):
            pattern += "**"
        patterns.append(pattern)
    return tuple(dict.fromkeys(patterns))


def load_project_index_policy(root: Path) -> ProjectIndexPolicy:
    """读取并校验 .devflow/index.yml；文件不存在就返回默认策略（auto 模式）。

    所有校验失败都抛 ValueError（带明确原因）：配置写错时宁可报错让用户改，
    也不要静默用默认值索引了一堆不该索引的文件。
    """
    root = root.resolve()
    config_path = root / PROJECT_INDEX_CONFIG_PATH
    if not config_path.is_file():
        # 没有配置文件是最常见的情况，属于正常路径，不算错误。
        return ProjectIndexPolicy()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        # safe_load 而不是 load：仓库里的配置来自外部，不能允许它构造任意 Python 对象。
        raise ValueError(f"Could not read {PROJECT_INDEX_CONFIG_PATH.as_posix()}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{PROJECT_INDEX_CONFIG_PATH.as_posix()} must contain a YAML object")
    # 版本号目前只支持 1，留着是为了将来改结构时能做兼容分支而不是直接崩。
    version = raw.get("version", 1)
    if version != 1:
        raise ValueError(f"Unsupported project index config version: {version}")
    project_docs = raw.get("project_docs") or {}
    if not isinstance(project_docs, dict):
        raise ValueError("project_docs must be a YAML object")
    mode = str(project_docs.get("mode") or "auto").strip().lower()
    if mode not in PROJECT_INDEX_MODES:
        raise ValueError(f"project_docs.mode must be one of: {', '.join(sorted(PROJECT_INDEX_MODES))}")
    include = _normalize_patterns(project_docs.get("include"), "project_docs.include")
    exclude = _normalize_patterns(project_docs.get("exclude"), "project_docs.exclude")
    # allowlist 模式没有 include 就等于「什么都不索引」，这几乎肯定是配置漏写，直接报错。
    if mode == "allowlist" and not include:
        raise ValueError("project_docs.include cannot be empty when mode is allowlist")
    # include_manifests 严格要求 bool：填 "true"（字符串）这种是常见笔误，立即暴露。
    include_manifests = raw.get("include_manifests", True)
    if not isinstance(include_manifests, bool):
        raise ValueError("include_manifests must be true or false")
    return ProjectIndexPolicy(
        mode=mode,
        include=include,
        exclude=exclude,
        include_manifests=include_manifests,
        # 记录配置文件相对路径，回显给前端「你用的是哪个策略文件」。
        config_path=PROJECT_INDEX_CONFIG_PATH.as_posix(),
    )


def _matches_any(relative_path: str, patterns: tuple[str, ...]) -> bool:
    """任一 glob 命中即为真。

    用 fnmatchcase 而不是 fnmatch：Windows 上 fnmatch 会把 * 和 ? 按大小写不敏感处理，
    而仓库路径（尤其 Linux 上）是大小写敏感的，必须统一成区分大小写的行为。
    """
    return any(fnmatch.fnmatchcase(relative_path, pattern) for pattern in patterns)


def _policy_allows(candidate: ProjectDocCandidate, policy: ProjectIndexPolicy) -> bool:
    """按策略判断某个候选是否允许被索引。

    判断顺序很重要：exclude 优先于一切（显式排除永远生效），
    其次处理依赖清单（由 include_manifests 单独开关），最后才看 allowlist 的 include。
    """
    if _matches_any(candidate.relative_path, policy.exclude):
        return False
    if candidate.source_type == "project_manifest":
        return policy.include_manifests
    if policy.mode == "allowlist":
        return _matches_any(candidate.relative_path, policy.include)
    # auto 模式下，通过分类规则的全部放行。
    return True


def _discover_project_docs(root: Path, max_files: int = 500) -> ProjectDocDiscovery:
    """遍历仓库目录，按分类规则和策略挑出待索引文档。

    用 os.walk 而不是 rglob，是因为需要在遍历时就地剪枝（dir_names[:] = ...）：
    直接把 node_modules/.git 从待遍历列表里删掉，比「先遍历再过滤」快一个量级。
    目录和文件名都排序，保证同一份代码每次发现的顺序一致 —— 指纹计算依赖这一点。
    """
    root = root.resolve()
    policy = load_project_index_policy(root)
    results: list[ProjectDocCandidate] = []
    excluded_paths: list[str] = []
    truncated = False
    for current_root, dir_names, file_names in os.walk(root):
        # 就地改写 dir_names（切片赋值）才会影响 walk 的后续遍历，这就是剪枝。
        # 隐藏目录默认跳过，只放行 INDEXABLE_HIDDEN_DIRS 里的白名单。
        dir_names[:] = sorted(
            name
            for name in dir_names
            if name not in SKIP_DIRS and (not name.startswith(".") or name in INDEXABLE_HIDDEN_DIRS)
        )
        for filename in sorted(file_names):
            path = Path(current_root) / filename
            # 再校验一次是否在 root 内：防符号链接把扫描范围引出仓库。
            if not _is_inside(root, path):
                continue
            rel = path.relative_to(root).as_posix()
            classification = _classify_file(path, rel)
            if not classification:
                continue
            source_type, tier = classification
            candidate = ProjectDocCandidate(path=path, relative_path=rel, source_type=source_type, tier=tier)
            if not _policy_allows(candidate, policy):
                # 被策略排除的也记下来：前端要展示「哪些文件被规则挡掉了」，
                # 否则用户会疑惑「我明明有这个文档为什么没索引」。
                excluded_paths.append(rel)
                continue
            if len(results) >= max_files:
                # 达到上限就标记截断并终止整个遍历（break 外层也 break）。
                truncated = True
                break
            results.append(candidate)
        if truncated:
            break
    return ProjectDocDiscovery(results, excluded_paths, policy, truncated)


def discover_project_docs(root: Path, max_files: int = 500) -> list[ProjectDocCandidate]:
    """只要候选列表的便捷版本（丢弃策略/截断等元信息）。"""
    return _discover_project_docs(root, max_files=max_files).candidates


def project_fingerprint(repo: Repository) -> str:
    """计算项目文档的指纹，用于判断「索引是否过期」。

    组成：策略内容 + 每个候选的 (相对路径, 修改时间纳秒, 文件大小)。
    - 把策略也算进去：用户改了 include/exclude，即使文件没变，索引结果也应视为过期；
    - 用 mtime_ns 而不是秒级时间：同一秒内连续修改两次也能区分；
    - 用 st_size 兜住「内容变了但 mtime 没变」（某些同步工具会保留时间戳）的常见情况。
    前缀 "files:" 标明指纹算法版本，将来换算法时旧值不会误判为相同。
    """
    root = _safe_project_root(repo)
    if not root:
        # 仓库还没拉到本地时无法计算指纹，返回空串（调用方会当成「无从比较」）。
        return ""
    hasher = hashlib.sha256()
    # 指纹用 1000 的上限（比实际索引的 500 更宽）：宁可多算一些文件，也别漏掉变化。
    discovery = _discover_project_docs(root, max_files=1000)
    hasher.update(json.dumps(discovery.policy.as_dict(), ensure_ascii=False, sort_keys=True).encode("utf-8"))
    for candidate in discovery.candidates:
        try:
            stat = candidate.path.stat()
        except OSError:
            # 文件在遍历后被删了，跳过即可，不影响其它文件的指纹。
            continue
        hasher.update(candidate.relative_path.encode("utf-8"))
        hasher.update(str(stat.st_mtime_ns).encode("ascii"))
        hasher.update(str(stat.st_size).encode("ascii"))
    digest = hasher.hexdigest()
    return f"files:{digest}"


def _get_state_row(db: Session, repo_id: uuid.UUID) -> ProjectIndexState | None:
    """取索引状态行（每仓库至多一条），没有返回 None。"""
    return db.query(ProjectIndexState).filter(ProjectIndexState.repo_id == repo_id).one_or_none()


def _get_or_create_state(db: Session, repo_id: uuid.UUID) -> ProjectIndexState:
    """取索引状态行，没有就建一条 status=missing 的。"""
    row = _get_state_row(db, repo_id)
    if row:
        return row
    row = ProjectIndexState(repo_id=repo_id, status="missing")
    db.add(row)
    # flush 拿到 id（虽然这里暂时不用），并让同事务内的后续查询能看到这行。
    db.flush()
    return row


def _state_payload(repo: Repository, row: ProjectIndexState | None, fingerprint: str | None = None) -> dict[str, Any]:
    """把状态行转成前端结构，并顺手把 ready 修正为 stale。

    关键逻辑：status=ready 但当前指纹和记录的不一致 → 说明文档变了但没重新索引，
    对外报 stale。这个「读时判定」避免了后台定时扫描，代价是每次读状态都要算一次指纹。
    """
    # 允许调用方传入已算好的指纹，避免一次请求里重复扫描目录。
    current_fingerprint = fingerprint if fingerprint is not None else project_fingerprint(repo)
    if row is None:
        return {
            "repo_id": repo.id,
            "status": "missing",
            "fingerprint": current_fingerprint,
            "docs_indexed": 0,
            "docs_total": 0,
            "error_message": None,
            "summary_json": None,
            "snoozed_until": None,
            "last_scan_at": None,
        }
    # 库里可能出现非法状态（手工改库/版本回滚），统一降级成 missing，前端就不会遇到未知状态。
    status = row.status if row.status in PROJECT_INDEX_STATUSES else "missing"
    if status == "ready" and current_fingerprint and row.fingerprint != current_fingerprint:
        status = "stale"
    return {
        "repo_id": row.repo_id,
        "status": status,
        # 优先回显记录里的指纹（代表「当前索引是基于哪一版文件建的」）；
        # 只有它为空时才用实时指纹填充，避免前端拿到空值。
        "fingerprint": row.fingerprint or current_fingerprint,
        "docs_indexed": row.docs_indexed or 0,
        "docs_total": row.docs_total or 0,
        "error_message": row.error_message,
        "summary_json": row.summary_json,
        "snoozed_until": row.snoozed_until,
        "last_scan_at": row.last_scan_at,
    }


def get_project_index_state(db: Session, repo: Repository) -> dict[str, Any]:
    """查询索引状态（只读，不触发扫描）。"""
    return _state_payload(repo, _get_state_row(db, repo.id))


def mark_project_index_building(db: Session, repo: Repository) -> ProjectIndexState:
    """把状态置为 building 并立即提交。

    立即提交是刻意的：真正的扫描可能耗时几十秒，先落一个 building 状态，
    前端轮询时就能显示进度，而不是傻等到结束才知道有没有在跑。
    同时清掉 error_message 和 snoozed_until（用户重新触发 = 不再忽略提示）。
    """
    row = _get_or_create_state(db, repo.id)
    row.status = "building"
    row.fingerprint = project_fingerprint(repo)
    row.error_message = None
    row.snoozed_until = None
    row.updated_at = _now()
    db.commit()
    return row


def snooze_project_index(db: Session, repo: Repository, days: int = 7) -> ProjectIndexState:
    """「稍后提醒」：在 days 天内不再提示用户去建索引。

    默认 7 天：足够让用户处理一轮迭代，又不会拖到彻底忘记。
    如果当前既没就绪也没在建，把状态压回 missing —— 否则一个 stale/failed 状态
    被 snooze 后前端可能显示成「一切正常」，反而误导。
    """
    row = _get_or_create_state(db, repo.id)
    row.snoozed_until = _now() + timedelta(days=days)
    row.updated_at = _now()
    if row.status not in {"ready", "building"}:
        row.status = "missing"
    db.commit()
    # refresh 拿到数据库端默认值和触发器生成的值，保证返回给前端的行是最新的。
    db.refresh(row)
    return row


def _extract_title(text: str, fallback: str) -> str:
    """取 markdown 的一级标题（"# xxx"）作为文档标题，取不到就用路径兜底。

    只认行首 "# "（带空格）而不认 "## "：一级标题才代表整篇文档的主题。
    截到 500 字是因为标题会写进 metadata，过长没有意义。
    """
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()[:500] or fallback
    return fallback


def _summary_from_text(text: str, limit: int = 360) -> str:
    """从正文里挑出「第一段像正文的段落」当摘要。

    跳过标题（#）、表格（|）、代码块（```）：这些都不适合当摘要。
    如果整篇没有合格段落（例如纯表格的 README），退化成取前 4 行。
    这个函数目前只在 summary 里备用，属于「不用 LLM 也能给个摘要」的兜底策略。
    """
    cleaned = text.strip()
    if not cleaned:
        return ""
    paragraphs = []
    for block in cleaned.split("\n\n"):
        # 段落内部的换行合并成空格，避免摘要里出现硬换行。
        value = " ".join(line.strip() for line in block.splitlines() if line.strip())
        if value and not value.startswith("#") and not value.startswith("|") and not value.startswith("```"):
            paragraphs.append(value)
    summary = paragraphs[0] if paragraphs else " ".join(cleaned.splitlines()[:4])
    # 保留 limit-3 个字符再加 "..."，保证总长不超过 limit。
    return summary[: limit - 3].rstrip() + "..." if len(summary) > limit else summary


def _detect_tech_stack(root: Path) -> list[str]:
    """按根目录下有哪些标志文件来推断技术栈。

    只看根目录且只看文件是否存在，不解析内容：这里要的是「大致是什么项目」，
    几十毫秒就能出结果，不需要精确版本信息。
    同一个技术栈可能被多个文件命中（如 python 命中 pyproject 和 requirements），
    所以调用处会 sorted(set(...)) 去重。
    """
    detectors = {
        "package.json": "node",
        "tsconfig.json": "typescript",
        "pyproject.toml": "python",
        "requirements.txt": "python",
        "go.mod": "go",
        "Cargo.toml": "rust",
        "pom.xml": "java",
        "build.gradle": "java",
    }
    return [tech for filename, tech in detectors.items() if (root / filename).exists()]


def _safe_dirs(root: Path, limit: int = 12) -> list[str]:
    """列出根目录下的顶层子目录名（跳隐藏目录和依赖目录），最多 limit 个。

    用来在「项目概况」里给模型一个目录结构印象；限制 12 个是防止 monorepo 刷出上百行。
    """
    dirs: list[str] = []
    try:
        # 按小写名排序：让前端展示顺序稳定，且不受大小写影响。
        for child in sorted(root.iterdir(), key=lambda item: item.name.lower()):
            if len(dirs) >= limit:
                break
            if child.name.startswith(".") or child.name in SKIP_DIRS:
                continue
            if child.is_dir():
                dirs.append(child.name)
    except OSError:
        # 目录读不了（权限/已删除）就返回空列表，摘要里少一项而已，不必报错。
        return []
    return dirs


def _build_summary(
    repo: Repository,
    root: Path,
    candidates: list[ProjectDocCandidate],
    docs_indexed: int,
    discovery: ProjectDocDiscovery,
) -> dict[str, Any]:
    """生成「索引摘要」，既有用户视角的概览，也有排障视角的统计。

    统计 tier/source_type 的分布，是为了让用户一眼看出文档质量：
    如果全是 soft_clue（只有 CHANGELOG），说明这个项目缺少正式文档。
    """
    tier_coverage: dict[str, int] = {}
    source_type_coverage: dict[str, int] = {}
    docs_list: list[dict[str, str]] = []
    for item in candidates:
        # 手写计数而不是 Counter：这里要的是普通 dict，直接能 JSON 序列化。
        tier_coverage[item.tier] = tier_coverage.get(item.tier, 0) + 1
        source_type_coverage[item.source_type] = source_type_coverage.get(item.source_type, 0) + 1
        docs_list.append({"path": item.relative_path, "tier": item.tier, "source_type": item.source_type})
    return {
        "projectName": repo.full_name,
        "rootPath": str(root),
        "techStack": sorted(set(_detect_tech_stack(root))),
        "dirStructure": _safe_dirs(root),
        # 列表都截断到 80 条：这是给前端展示/给模型看的摘要，不是完整清单；
        # 真实数量看 tierCoverage/docsIndexed 等计数字段。
        "docsList": docs_list[:80],
        "tierCoverage": tier_coverage,
        "sourceTypeCoverage": source_type_coverage,
        "docsIndexed": docs_indexed,
        "indexPolicy": discovery.policy.as_dict(),
        "excludedFiles": len(discovery.excluded_paths),
        "excludedPaths": discovery.excluded_paths[:80],
        "discoveryTruncated": discovery.truncated,
    }


def _payloads_for_candidates(repo: Repository, root: Path, candidates: list[ProjectDocCandidate], fingerprint: str) -> list[DocumentPayload]:
    """把候选文档读出来、切块、组装成待写入知识库的 payload 列表。

    切块参数 1800 字符 / 180 重叠 是经验值：块太大检索不精准，太小会丢上下文；
    重叠是为了避免关键句正好被切断。
    """
    payloads: list[DocumentPayload] = []
    # 所有块共用同一个扫描时间戳，便于事后按批次排查「这批文档是什么时候进来的」。
    scanned_at = _now().isoformat()
    for candidate in candidates:
        text = _read_text(candidate.path)
        if not text or not text.strip():
            continue
        # source_id 用 stable_source_id 生成（由 repo + 类型 + 相对路径决定）：
        # 同一份文档重扫时 id 不变，才能实现「替换」而不是「不断新增」。
        source_id = stable_source_id(repo.id, candidate.source_type, candidate.relative_path)
        title = _extract_title(text, candidate.relative_path)
        chunks = chunk_document(text, candidate.relative_path, max_chars=1800, overlap=180)
        for index, chunk in enumerate(chunks):
            content = str(chunk["content"])
            section_title = chunk.get("section_title")
            payloads.append(
                DocumentPayload(
                    source_type=candidate.source_type,
                    source_id=source_id,
                    # 标题三种形态：有章节名用「标题 · 章节」，多块无章节用「标题#序号」，单块直接用标题。
                    title=f"{title} · {section_title}" if section_title else (f"{title}#{index + 1}" if len(chunks) > 1 else title),
                    content=content,
                    metadata={
                        # 把 chunk 自带的其它字段（章节名等）平铺进 metadata，但排除 content 本身和 None 值。
                        **{key: value for key, value in chunk.items() if key != "content" and value is not None},
                        "display_title": title,
                        "path": candidate.relative_path,
                        "root_path": str(root),
                        "provenance_tier": candidate.tier,
                        "chunk_index": index,
                        "chunk_chars": len(content),
                        "total_chunks": len(chunks),
                        # 记录切块策略版本：将来改算法后，可以据此识别出「旧算法切的块」并重建。
                        "chunk_strategy": "structure_aware_v1",
                        "fingerprint": fingerprint,
                        "scanned_at": scanned_at,
                    },
                )
            )
    return payloads


def scan_project_index(db: Session, repo: Repository) -> dict[str, Any]:
    """执行一次完整的项目文档索引，返回最终状态。

    流程：置 building → 发现文档 → 切块 → 写入知识库 → 更新状态 → 重建证据与图谱。
    失败不抛异常而是把状态落成 failed 并把错误信息回传给前端：
    这是用户主动触发的操作，他需要看到失败原因，而不是一个 500。

    注意 replace_documents 传了 generate_embeddings=False / sync_vector_store=False，
    即这里只写库不生成向量，向量化由后续流程异步补，避免一次请求里做太重的事。
    """
    row = _get_or_create_state(db, repo.id)
    fp = project_fingerprint(repo)
    row.status = "building"
    row.fingerprint = fp
    row.error_message = None
    row.snoozed_until = None
    row.updated_at = _now()
    # 先提交 building 状态，让前端能立刻看到「正在建索引」。
    db.commit()

    try:
        root = _safe_project_root(repo)
        if not root:
            # 中文提示直接给用户看，告诉他下一步该做什么（连仓库或先同步代码）。
            raise RuntimeError("项目本地路径不存在，请先连接本地仓库或完成代码同步")

        discovery = _discover_project_docs(root)
        candidates = discovery.candidates
        payloads = _payloads_for_candidates(repo, root, candidates, fp)
        # 除了文档本身，再插一条「项目概况」：它把仓库名/描述/默认分支/本地路径/文件清单
        # 打包成一条结构化文档，用来回答「这个项目是做什么的」这种全局问题。
        overview = {
            "repo": repo.full_name,
            "description": repo.description or "",
            "default_branch": repo.default_branch or "",
            "local_path": str(root),
            "indexed_files": [candidate.relative_path for candidate in candidates[:80]],
        }
        payloads.insert(
            0,
            DocumentPayload(
                source_type="project_overview",
                source_id=stable_source_id(repo.id, "project_overview", "overview"),
                title=f"项目概况：{repo.full_name}",
                content=json.dumps(overview, ensure_ascii=False, indent=2),
                metadata={"display_title": "项目概况", "fingerprint": fp, "root_path": str(root), "chunk_index": 0},
            ),
        )

        # 只替换这三类 source_type：memory_note 等其它来源的文档不受影响。
        docs = replace_documents(
            db,
            repo.id,
            PROJECT_INDEX_SOURCE_TYPES,
            payloads,
            commit=False,
            generate_embeddings=False,
            sync_vector_store=False,
        )
        summary = _build_summary(repo, root, candidates, len(docs), discovery)
        # 重新取一次状态行：上面的 replace_documents 可能已经触发过 flush/事务变化，
        # 用新引用能避免拿到过期对象。
        row = _get_or_create_state(db, repo.id)
        row.status = "ready"
        row.fingerprint = fp
        row.docs_indexed = len(docs)
        # docs_total 记候选数而不是实际入块数：用来体现「发现了几份文档」。
        row.docs_total = len(candidates)
        row.summary_json = summary
        row.last_scan_at = _now()
        row.error_message = None
        row.updated_at = _now()
        # EvidenceItems are a denormalized search surface used by ChatAgent's
        # memory tools. Refresh them in the same transaction so excluded
        # project documents cannot survive a successful rescan.
        # 译文：EvidenceItem 是 ChatAgent 记忆工具使用的「反规范化检索面」。
        # 必须在**同一个事务**里重建，否则被这次索引排除掉的旧文档仍会残留在证据表里，
        # 导致「文档已经不在索引中，但 Agent 还能检索到它」。
        from app.services.chat_memory import EvidenceStore

        EvidenceStore(db).rebuild_repo(repo)
        db.commit()
        db.refresh(row)
        # 建图放在最后且独立提交（rebuild_knowledge_graph 内部自己 commit）：
        # 图谱只是展示层，它的失败不该让已经成功的索引变成 failed。
        rebuild_knowledge_graph(db, repo)
        return _state_payload(repo, row, fingerprint=fp)
    except Exception as exc:
        # 回滚半成品，再把状态单独写成 failed 并提交（否则回滚会把 failed 也一起滚掉）。
        db.rollback()
        row = _get_or_create_state(db, repo.id)
        row.status = "failed"
        row.error_message = str(exc)
        row.updated_at = _now()
        db.commit()
        db.refresh(row)
        return _state_payload(repo, row, fingerprint=fp)


async def scan_project_index_by_id(repo_id: uuid.UUID) -> None:
    """按 ID 触发索引（后台任务用），无返回值、不向调用方抛异常。

    自己开会话是因为后台任务没有请求上下文；scan_project_index 内部已把异常
    转成 failed 状态，所以这里正常不会抛。
    """
    db = SessionLocal()
    try:
        repo = db.get(Repository, repo_id)
        if repo is None:
            return
        scan_project_index(db, repo)
    finally:
        db.close()


def is_snoozed(state: dict[str, Any]) -> bool:
    """判断该仓库是否处于「稍后提醒」的冷却期内。"""
    until = _aware(state.get("snoozed_until"))
    return bool(until and until > _now())
