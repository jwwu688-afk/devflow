"""在工作区（已 checkout 的仓库目录）里做基于关键词的代码检索。

这是「没有向量库也要能查代码」的兜底通道：
- 有 ripgrep 就优先用它（快、跳过二进制、自带多线程）；
- 没有或执行失败则退回纯 Python 逐文件扫描（慢但零依赖）。
两套实现返回同一结构，上层无需感知用的是哪一个（metadata.search_tool 里会标明）。

安全边界由 resolve_search_root 统一把守：不允许查出仓库目录之外，
也不允许读敏感配置文件——检索结果会进模型上下文，泄密风险必须在这里拦住。
"""

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from app.db.models import Repository


# 这些目录里全是构建产物/依赖/缓存，搜代码时既没价值又会拖慢扫描，直接跳过。
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

# 明确禁止读取的文件名：即使落在允许的目录里，也一律不搜、不返回内容。
# 白名单式的扩展名过滤拦不住这些「无扩展名」的密钥文件，所以要单独列一份黑名单。
SECRET_FILENAMES = {".env", ".env.local", ".env.production", ".npmrc", ".pypirc"}

# 允许当作文本读取的扩展名白名单。用白名单而不是黑名单：
# 未知类型（二进制、图片、压缩包）默认不读，避免把乱码塞进上下文。
TEXT_EXTENSIONS = {
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",
    ".java",
    ".go",
    ".rs",
    ".cs",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
    ".php",
    ".rb",
    ".swift",
    ".kt",
    ".kts",
    ".scala",
    ".sql",
    ".sh",
    ".ps1",
    ".bat",
    ".cmd",
    ".html",
    ".css",
    ".scss",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".md",
    ".mdx",
    ".txt",
}

# 没有扩展名但值得检索的常见工程文件名（构建脚本、依赖清单等）。
TEXT_FILENAMES = {
    "Dockerfile",
    "Makefile",
    "README",
    "LICENSE",
    "package.json",
    "requirements.txt",
    "pyproject.toml",
}

# 把用户问题切成「搜索词」的正则，三个分支覆盖典型输入：
# 1. A-Za-z_$ 开头：标识符/函数名/URL 片段，允许带 . $ : / \ - 等代码常见分隔符；
# 2. 连续 3 位以上数字：错误码、端口、issue 号；
# 3. 连续 2 个以上汉字：中文提问里没有空格，必须整段取，再靠停用词过滤噪音。
_QUERY_TOKEN_RE = re.compile(r"[A-Za-z_$][A-Za-z0-9_.$:/\\-]*|[0-9]{3,}|[\u4e00-\u9fff]{2,}")
# 高频但无区分度的词：几乎每个文件都出现，带上它们只会让评分变成噪音。
# 中英混排是因为用户的提问本身常常中英夹杂。
_STOP_TERMS = {
    "about",
    "after",
    "before",
    "code",
    "error",
    "failed",
    "failure",
    "file",
    "from",
    "into",
    "issue",
    "project",
    "pull",
    "request",
    "test",
    "tests",
    "that",
    "the",
    "this",
    "with",
    "代码",
    "问题",
    "失败",
    "项目",
    "相关",
    "这个",
}


def repository_checkout_path(repo: Repository) -> Path:
    """返回该仓库代码在本地的目录。

    优先用用户配置的 local_path（本地仓库模式：直接搜用户自己的目录，不用 clone）；
    没配才回落到托管模式的 checkout 目录。
    expanduser 支持 "~/xxx" 这种写法；相对路径按当前工作目录补全成绝对路径。
    """
    raw_path = str(getattr(repo, "local_path", None) or "").strip()
    if raw_path:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        return path.resolve()

    # 延迟导入：code_analysis 会 import 本模块，放顶层会形成循环导入。
    from app.services.code_analysis import _repo_checkout_path

    return _repo_checkout_path(repo).resolve()


def resolve_search_root(checkout_path: str | Path, relative_path: str | None = None) -> tuple[Path, Path]:
    """把「仓库根 + 相对子路径」解析成 (checkout 根, 实际搜索起点)，并做安全校验。

    所有校验失败都抛 ValueError（而不是返回 None）：调用方据此统一转成 400 或空结果。
    校验顺序有意从「路径逃逸」开始——先用 relative_to 确认没跑出仓库，
    再判断是否命中跳过目录/敏感文件，最后才看是否存在。
    """
    checkout = Path(checkout_path).expanduser().resolve()
    if not checkout.exists() or not checkout.is_dir():
        raise ValueError("仓库代码还没有同步到本地。")

    requested = str(relative_path or ".").strip()
    # 各种「根目录」写法统一归成 "."，再 lstrip 去掉开头的斜杠，
    # 否则 "/xxx" 会被 Path 当成绝对路径，直接跳出仓库。
    requested = "." if requested in {"", ".", "./", "/", "\\"} else requested.lstrip("/\\")
    root = (checkout / requested).resolve()
    try:
        # 解析（含软链接）之后再判断包含关系，才能防住 ".." 和符号链接两种逃逸。
        root.relative_to(checkout)
    except ValueError as exc:
        raise ValueError("搜索路径位于仓库工作区之外。") from exc
    if any(part in SKIP_DIRS for part in root.relative_to(checkout).parts):
        raise ValueError("搜索路径位于被跳过的目录中。")
    if root.name in SECRET_FILENAMES:
        raise ValueError("不会搜索敏感配置文件。")
    if not root.exists():
        raise ValueError("搜索路径不存在。")
    return checkout, root


def query_terms(query: str, limit: int = 16) -> list[str]:
    """把自然语言 query 拆成若干检索词，按「重要性」排序后截断。

    排序规则（元组依次比较）：
    1) 是否像代码符号（含 _ . $ : / - 数字，或大小写混排如 getUserName）——优先，
       因为用户问代码时最有效的词就是这些；
    2) 词更长更具体，优先；
    3) 字典序兜底，保证结果可复现。
    全部命中停用词时退回未过滤列表：宁可搜些泛词，也好过返回空结果。
    """
    candidates: list[tuple[int, int, str]] = []
    fallback_candidates: list[tuple[int, int, str]] = []
    seen: set[str] = set()
    # position 只用于遍历计数，实际排序不依赖它（顺序由后面 sort 决定）。
    for position, match in enumerate(_QUERY_TOKEN_RE.finditer(query or "")):
        value = match.group(0).strip()
        normalized = value.lower()
        # 单字符词（如变量 a、汉字「的」）区分度太低，直接丢；重复词只保留一次。
        if len(normalized) < 2 or normalized in seen:
            continue
        seen.add(normalized)
        # 「像代码」的判定：含代码常见分隔符，或者是驼峰命名（同时有小写和大写）。
        code_like = bool(
            re.search(r"[_.$:/\\\-0-9]", value)
            or (any(char.islower() for char in value) and any(char.isupper() for char in value))
        )
        candidate = (1 if code_like else 0, len(value), normalized)
        fallback_candidates.append(candidate)
        if normalized not in _STOP_TERMS:
            candidates.append(candidate)
    candidates = candidates or fallback_candidates
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
    # max(1, limit)：limit 传 0 时至少留一个词，避免上层拿到空列表直接返回无结果。
    return [item[2] for item in candidates[: max(1, limit)]]


def _is_text_file(path: Path) -> bool:
    """判断该文件是否值得按文本读取。

    三重过滤：敏感文件名、日志/数据库等明确排除的类型、以及白名单命中。
    name.lower().endswith("dockerfile") 兜住 Dockerfile.dev 这类带后缀的变体。
    """
    if path.name in SECRET_FILENAMES or path.suffix.lower() in {".log", ".db", ".sqlite", ".sqlite3"}:
        return False
    return path.name in TEXT_FILENAMES or path.suffix.lower() in TEXT_EXTENSIONS or path.name.lower().endswith("dockerfile")


def _read_text(path: Path, max_bytes: int = 180_000) -> str | None:
    """尽力把文件读成字符串，读不了就返回 None（调用方跳过即可）。

    三种情况直接放弃：
    - 空文件；
    - 超过 max_bytes（180KB 约等于几千行，再大就不适合塞进检索/上下文）；
    - 前 4096 字节里出现 \\x00 —— 这是二进制文件的可靠特征（UTF-16 除外，但代码库基本不用）。
    编码按 utf-8-sig → utf-8 → gb18030 → latin-1 依次尝试：
    latin-1 永不失败，作为最后的兜底，保证至少能拿到内容（可能有个别乱码字符）。
    """
    try:
        data = path.read_bytes()
    except OSError:
        # 权限不足、文件被占用等，属于正常情况，不该让整次检索失败。
        return None
    if not data or len(data) > max_bytes or b"\x00" in data[:4096]:
        return None
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


def read_code_excerpt(checkout_path: str | Path, relative_path: str, start_line: int, line_count: int = 7) -> str:
    """读取指定文件的若干行，返回带行号的代码片段。

    输出第一行是 "相对路径:起-止" 头，随后每行形如 "12: code"，
    这样模型引用时能直接说出准确行号，不需要自己数。
    默认 7 行是为了「看得到上下文又不至于太长」，一次多取几段也不会撑爆上下文。
    传入的行号越界时统一收敛到文件范围内，完全越界则返回空串。
    """
    checkout, path = resolve_search_root(checkout_path, relative_path)
    if not path.is_file() or not _is_text_file(path):
        return ""
    text = _read_text(path)
    if text is None:
        return ""
    lines = text.splitlines()
    # 起点至少是 1（行号从 1 开始），终点不超过总行数，避免切片越界。
    start = max(int(start_line), 1)
    end = min(start + max(int(line_count), 1) - 1, len(lines))
    if start > end:
        return ""
    # 统一用 posix 风格的相对路径，跨平台返回一致。
    relative = path.relative_to(checkout).as_posix()
    return f"{relative}:{start}-{end}\n" + "\n".join(
        f"{line_number}: {lines[line_number - 1]}" for line_number in range(start, end + 1)
    )


def _score_match(relative_path: str, line: str, terms: list[str]) -> float:
    """给「某一行 + 它所在的路径」打一个粗糙的相关性分数。

    三项加权（系数是经验值，目的是让排序合理而非精确）：
    - matched：命中了几种不同的词，权重最高（4）——覆盖更多查询词通常更相关；
    - occurrences：词在行内出现的次数，每个词最多计 4 次，防止某个词刷分；
    - path_hits：词出现在文件路径上的次数，权重 2——路径命中往往意味着整个文件都相关
      （例如搜 middleware 时 auth/middleware.py 应该排前面）。
    分数是相对值，最终会除以最高分归一化，所以绝对值大小不重要。
    """
    path_text = relative_path.lower()
    line_text = line.lower()
    matched = sum(1 for term in terms if term in line_text or term in path_text)
    occurrences = sum(min(line_text.count(term), 4) for term in terms)
    path_hits = sum(path_text.count(term) for term in terms)
    return float(matched * 4 + occurrences + path_hits * 2)


def _ripgrep_matches(checkout: Path, root: Path, terms: list[str], max_matches: int) -> list[tuple[float, str, int, str]]:
    """用 ripgrep 搜索，返回每个文件「得分最高的那一行」。

    为什么按文件只留一条：检索结果要喂给模型，同一文件二十行命中会把上下文挤满，
    而给每文件一个最佳命中既能定位又省 token（drill-down 交给 read_code_excerpt）。
    """
    # 多个词用 | 拼成正则或，一次扫完，比逐个词跑 rg 快很多。
    # re.escape 防止查询里带正则元字符（如 "list[int]"）导致语法错误。
    pattern = "|".join(re.escape(term) for term in terms)
    command = [
        "rg",
        "--json",
        "--ignore-case",
        "--line-number",
        # 超长行（压缩后的 js、大 json）会拖慢解析，截断到 500 列即可。
        "--max-columns",
        "500",
        # 再叠一层 glob 黑名单：rg 自己也要遵守 SECRET_FILENAMES 的约定，
        # 因为这些文件常常没有扩展名，扩展名白名单管不住。
        "--glob",
        "!**/.env*",
        "--glob",
        "!**/.npmrc",
        "--glob",
        "!**/.pypirc",
        "--glob",
        "!**/*.log",
        "--glob",
        "!**/*.db",
        "--glob",
        "!**/*.sqlite*",
        pattern,
        str(root),
    ]
    result = subprocess.run(
        command,
        cwd=str(checkout),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        # errors="replace"：仓库里可能有非法 UTF-8 文件名/内容，不让解码错误中断整次搜索。
        errors="replace",
        # 15 秒上限：超大仓库或磁盘慢时不至于把请求拖死，超时会由上层退回 Python 扫描。
        timeout=15,
    )
    # rg 的退出码：0=有匹配，1=无匹配（都算正常），其它才算真失败。
    if result.returncode not in {0, 1}:
        raise RuntimeError(result.stderr.strip() or "ripgrep 搜索失败")

    best_by_file: dict[str, tuple[float, str, int, str]] = {}
    # rg --json 是「一行一个 JSON 事件」，需要逐行解析；begin/end/summary 事件直接忽略。
    for raw_line in result.stdout.splitlines():
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "match":
            continue
        data = event.get("data") or {}
        raw_path = str((data.get("path") or {}).get("text") or "")
        path = Path(raw_path)
        if not path.is_absolute():
            path = checkout / path
        try:
            # 统一转成相对仓库根的 posix 路径；转不了说明命中了仓库外的文件，丢弃。
            relative = path.resolve().relative_to(checkout).as_posix()
        except ValueError:
            continue
        # rg 不认 .gitignore 之外的这些规则（也不看我们的 SKIP_DIRS），这里再兜一次。
        if any(part in SKIP_DIRS for part in Path(relative).parts) or Path(relative).name in SECRET_FILENAMES:
            continue
        line_number = int(data.get("line_number") or 1)
        # 单行内容截到 320 字符：够看清语义，又不至于让一条结果占满上下文。
        line = str((data.get("lines") or {}).get("text") or "").strip()[:320]
        score = _score_match(relative, line, terms)
        candidate = (score, relative, line_number, line)
        current = best_by_file.get(relative)
        if current is None or candidate[0] > current[0]:
            best_by_file[relative] = candidate
        # 达到上限就提前退出：rg 的输出是按文件顺序的，早停可以少读很多行。
        if len(best_by_file) >= max_matches:
            break
    return list(best_by_file.values())


def _python_matches(checkout: Path, root: Path, terms: list[str], max_files: int = 1000) -> list[tuple[float, str, int, str]]:
    """纯 Python 的兜底扫描：逐文件读文本、逐行打分，取每个文件的最佳行。

    max_files=1000 是硬上限：没有 rg 时全量扫描大仓库可能要几分钟，
    截断虽然会漏结果，但至少能保证请求在可接受时间内返回（且结果仍是高分优先的近似）。
    """
    # root 可能是单个文件（用户指定了具体路径），此时不递归。
    paths = [root] if root.is_file() else root.rglob("*")
    matches: list[tuple[float, str, int, str]] = []
    files_seen = 0
    for path in paths:
        if not path.is_file():
            continue
        relative = path.relative_to(checkout)
        if any(part in SKIP_DIRS for part in relative.parts) or not _is_text_file(path):
            continue
        files_seen += 1
        # 计数放在读过类型判断之后：只统计「真正要读的文本文件」，上限才有意义。
        if files_seen > max_files:
            break
        text = _read_text(path)
        if text is None:
            continue
        relative_text = relative.as_posix()
        best: tuple[float, str, int, str] | None = None
        # start=1：文件行号从 1 开始，和编辑器/read_code_excerpt 的口径保持一致。
        for line_number, line in enumerate(text.splitlines(), start=1):
            score = _score_match(relative_text, line, terms)
            if score <= 0:
                continue
            candidate = (score, relative_text, line_number, line.strip()[:320])
            if best is None or candidate[0] > best[0]:
                best = candidate
        if best:
            matches.append(best)
    return matches


def search_checkout_code(
    checkout_path: str | Path,
    query: str,
    *,
    limit: int = 12,
    relative_path: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """在工作区目录里检索代码，返回统一格式的结果列表（可直接当检索结果用）。

    流程：拆词 → 定位搜索根 → 优先 rg、失败退回 Python 扫描 → 排序截断 → 归一化打分。
    返回的 score 被缩放到 (0, 1]，与向量检索的相似度处于同一量级，
    上层融合排序时不必区分来源。
    """
    terms = query_terms(query)
    if not terms:
        # query 全是停用词/单字符时无法构造有意义的搜索，直接返回空而不是全库乱扫。
        return []
    checkout, root = resolve_search_root(checkout_path, relative_path)
    # 上限夹到 1..50：太少不够参考，太多会把上下文撑爆。
    capped_limit = min(max(int(limit), 1), 50)
    search_tool = "python_scan"
    if shutil.which("rg"):
        try:
            # 多取 8 倍候选（至少 80）：rg 的截断是按文件顺序的，
            # 多取一些才能在排序后仍然凑满 limit 条真正相关的。
            matches = _ripgrep_matches(checkout, root, terms, max_matches=max(capped_limit * 8, 80))
            search_tool = "ripgrep"
        except (OSError, RuntimeError, subprocess.TimeoutExpired):
            # rg 不可用/超时/报错都降级到 Python 扫描，保证功能可用（只是慢一些）。
            matches = _python_matches(checkout, root, terms)
    else:
        matches = _python_matches(checkout, root, terms)
    # 排序键：分数降序，同分按路径和行号升序，保证结果稳定可复现。
    matches.sort(key=lambda item: (-item[0], item[1], item[2]))
    if not matches:
        return []
    # 归一化用最高分做分母；`or 1.0` 防全零分导致除零。
    max_score = max(item[0] for item in matches) or 1.0
    output = []
    for score, path, line_number, line in matches[:capped_limit]:
        item_metadata = {
            "path": path,
            "line": line_number,
            # start/end 都等于命中行：给后续「按行取上下文」留的接口，
            # 将来若改成返回代码块，只需改这两个值。
            "start_line": line_number,
            "end_line": line_number,
            "checkout_path": str(checkout),
            # retrieval_mode/workspace 标明来源，方便审计区分「向量召回」和「工作区检索」。
            "retrieval_mode": "workspace",
            "search_tool": search_tool,
            # 外部 metadata 放最后，允许调用方覆盖默认字段（例如补 repo_id/pr_id）。
            **(metadata or {}),
        }
        output.append(
            {
                # id 三元组保证唯一：同一文件的不同行是不同结果。
                "id": f"workspace:{path}:{line_number}",
                "title": f"{path}:{line_number}",
                "source_type": "workspace_file",
                "source_id": path,
                "snippet": f"{path}:{line_number}: {line}",
                "score": score / max_score,
                "metadata": item_metadata,
            }
        )
    return output


def search_repository_code(
    repo: Repository,
    query: str,
    *,
    limit: int = 12,
    relative_path: str | None = None,
    checkout_path: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """按仓库检索代码（对外入口）。

    checkout_path 可显式指定，用于「PR 快照」这类非默认工作区；
    不指定就走 repository_checkout_path（local_path 或托管 checkout）。
    这里把 ValueError 统一吞成空列表：路径非法/不存在对调用方而言
    等价于「没搜到」，不该让整轮对话失败。
    """
    root = Path(checkout_path).resolve() if checkout_path else repository_checkout_path(repo)
    if not root.exists():
        return []
    try:
        return search_checkout_code(
            root,
            query,
            limit=limit,
            relative_path=relative_path,
            # repo_id 放前面，调用方传 metadata 时仍可覆盖（虽然不该这么做）。
            metadata={"repo_id": str(repo.id), **(metadata or {})},
        )
    except ValueError:
        return []
