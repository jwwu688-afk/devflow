"""Repository checkout synchronization and code-graph analysis.

两件事：把仓库代码拉到本地（_sync_checkout），再从代码里抽取符号与调用/导入关系（_sync_code_graph）。

抽取方式是**正则级别的轻量静态分析**，不是真正的 AST/语言服务器：
- 换来的是零依赖、任意语言都能跑、速度可控；
- 代价是只认「行首 def/class/function 声明」这种主流写法，
  装饰器、动态定义、多行签名都可能漏，属于有意识的取舍（这份图谱用于辅助检索和定位，不用于精确重构）。

图谱按 (repo, branch, pr_id, pr_number) 四元组隔离：
同一份代码在默认分支和某个 PR 快照下会各存一份，互不覆盖。
"""

import asyncio
import logging
import os
import shutil
import subprocess
import uuid
import re
from pathlib import Path
from urllib.parse import quote, urlparse, urlunparse

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import decrypt_token
from app.db.models import CodeRelation, CodeSymbol, Document, Repository
from app.db.session import SessionLocal

logger = logging.getLogger(__name__)

# 跳过依赖/构建/缓存目录，理由同 code_search：无价值且拖慢扫描。
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

# 参与符号抽取的扩展名。比 code_search 多一个 ".dockerfile"，
# 因为索引阶段希望把 Dockerfile 也算进来（这里不做敏感文件过滤，见 _is_text_candidate）。
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
    ".dockerfile",
}

# 无扩展名但可能有符号定义的文件名（这类文件基本只用来做 imports 关系，符号很少）。
TEXT_FILENAMES = {
    "Dockerfile",
    "Makefile",
    "README",
    "LICENSE",
    "package.json",
    "requirements.txt",
    "pyproject.toml",
}

# 四个抽取用的正则，都是「行首锚定 + 单行匹配」的简化写法：
# 只认缩进后的 def/class（含 async）和常见 TS 声明形式，不做语法解析，
# 因此宁可漏（嵌套定义、装饰器后换行）也不误报太多噪音。
PY_SYMBOL_RE = re.compile(r"^\s*(def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)
TS_SYMBOL_RE = re.compile(
    r"^\s*(?:export\s+)?(?:async\s+)?(?:function|class|interface|type|const|let)\s+([A-Za-z_$][A-Za-z0-9_$]*)",
    re.MULTILINE,
)
# 任何「标识符(」都当作一次调用；后面在 _sync_code_graph 里用关键字黑名单过滤 if/for 这类伪调用。
CALL_RE = re.compile(r"\b([A-Za-z_][$A-Za-z0-9_]*)\s*\(")
# import 三种写法（from x import / import x / import {a} from 'x'）用三个分组并取第一个非空组。
IMPORT_RE = re.compile(r"^\s*(?:from\s+([\w.\/-]+)\s+import|import\s+([\w.\/-]+)|import\s+.*?\s+from\s+['\"]([^'\"]+)['\"])", re.MULTILINE)


def _checkout_root() -> Path:
    """托管模式的 checkout 根目录（不存在则创建）。

    相对路径按进程 cwd 解析成绝对路径，避免子进程切目录后指向别处。
    """
    root = Path(settings.repo_checkout_dir)
    if not root.is_absolute():
        root = Path.cwd() / root
    root.mkdir(parents=True, exist_ok=True)
    return root


def _repo_checkout_path(repo: Repository) -> Path:
    """托管模式下某个仓库的固定目录名："<repo_id>-<owner__repo>"。

    前面拼 repo.id 保证唯一（不同 owner 可能有同名仓库），
    后面拼名字只是方便人肉排查时认出来。
    """
    safe_name = repo.full_name.replace("/", "__").replace("\\", "__")
    return _checkout_root() / f"{repo.id}-{safe_name}"


def _stored_checkout_path(repo: Repository) -> Path | None:
    """读取用户配置的 local_path（支持 ~ 和 $ENV 展开），未配置返回 None。

    注意这个字段在「local 模式」下是用户自己的仓库目录，
    在「managed 模式」下也可以作为自定义下载目标，所以两种模式都会用到它。
    """
    raw_path = getattr(repo, "local_path", None)
    if not raw_path:
        return None
    path = Path(os.path.expandvars(os.path.expanduser(str(raw_path))))
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _clone_url(repo: Repository) -> str:
    """取克隆地址，没配置就按 GitHub 惯例拼一个。"""
    if repo.clone_url:
        return repo.clone_url
    return f"https://github.com/{repo.owner}/{repo.name}.git"


def _clone_url_with_token(url: str, token: str | None) -> str:
    """把 token 嵌进 URL 里用于匿名克隆私有仓库。

    只在 http/https 且 netloc 里还没有身份信息时注入，避免：
    - ssh 地址被破坏（ssh:// 不能这样带 token）；
    - 用户已在 URL 里写了凭据时被覆盖。
    token 经过 quote 处理，防止里面含 @ : / 等字符破坏 URL 结构。
    """
    if not token:
        return url
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or "@" in parsed.netloc:
        return url
    safe_token = quote(token, safe="")
    return urlunparse(parsed._replace(netloc=f"x-access-token:{safe_token}@{parsed.netloc}"))


def _run_git(args: list[str], cwd: Path | None = None) -> None:
    """执行 git 命令，失败即抛 CalledProcessError（由调用方决定是否降级）。

    stdout 直接丢弃：这些命令的输出没人用，留着只会占内存；
    stderr 保留：报错时要靠它拼出可读的失败原因。
    """
    subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=180,
    )


def _is_git_work_tree(path: Path) -> bool:
    """判断目录是不是 git 工作树（而不是「碰巧存在的一个目录」）。

    用 rev-parse 而不是看有没有 .git 目录：worktree 子目录里没有 .git，
    但依然是合法工作树；反过来裸仓库的 .git 也不能当工作区用。
    """
    if not path.exists() or not path.is_dir() or not shutil.which("git"):
        return False
    result = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=str(path),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        timeout=30,
    )
    return result.returncode == 0 and result.stdout.strip().lower() == "true"


def _sync_checkout(repo: Repository, token: str | None) -> Path:
    """确保仓库代码在本地可用，返回代码目录（可能触发 clone/fetch）。

    三条路径：
    1. local 模式：用户自己的目录，只校验是不是 git 工作树，**绝不改动它**；
    2. 目录已是 git 工作树：拉取更新，失败只告警并继续用旧代码（离线也能用）；
    3. 全新目录：浅克隆（--depth 1）默认分支，省带宽和时间。
    """
    stored_path = _stored_checkout_path(repo)
    checkout_mode = getattr(repo, "checkout_mode", None) or "managed"

    if stored_path and checkout_mode == "local":
        # 本地模式不做任何写操作，避免污染用户正在开发的工作区。
        if not _is_git_work_tree(stored_path):
            raise RuntimeError(f"已配置的本地仓库路径不是 git 工作树：{stored_path}")
        return stored_path

    target = stored_path or _repo_checkout_path(repo)
    clone_url = _clone_url_with_token(_clone_url(repo), token)
    branch = repo.default_branch or "main"

    if not shutil.which("git"):
        raise RuntimeError("未找到 git 可执行文件，无法克隆仓库代码")

    if (target / ".git").exists() or _is_git_work_tree(target):
        try:
            # --depth 1：只取最新一次提交；checkout 是幂等的，已在目标分支时无副作用；
            # --ff-only 禁止产生 merge commit，冲突时宁可失败也不改动工作区。
            _run_git(["fetch", "--depth", "1", "origin", branch], cwd=target)
            _run_git(["checkout", branch], cwd=target)
            _run_git(["pull", "--ff-only", "origin", branch], cwd=target)
        except subprocess.CalledProcessError as exc:
            # 降级：网络/权限问题不该让整个分析失败，用已有缓存代码继续。
            logger.warning("更新现有 checkout %s 失败，将使用缓存文件：%s", target, (exc.stderr or exc).strip())
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        # 目录已存在但不是 git 仓库：如果里面有内容就直接报错（可能是用户的重要目录，
        # 不能被我们删掉）；只有空目录才允许清理后重克隆。
        if stored_path and any(target.iterdir()):
            raise RuntimeError(f"Configured download target already exists and is not an empty git work tree: {target}")
        shutil.rmtree(target)
    clone_args = ["clone", "--depth", "1"]
    if branch:
        clone_args.extend(["--branch", branch])
    clone_args.extend([clone_url, str(target)])
    _run_git(clone_args)
    return target


def _is_text_candidate(path: Path) -> bool:
    """判断文件是否参与索引（只看类型，不做敏感文件过滤）。"""
    if path.name in TEXT_FILENAMES:
        return True
    suffix = path.suffix.lower()
    return suffix in TEXT_EXTENSIONS or path.name.lower().endswith("dockerfile")


def _read_text(path: Path, max_bytes: int = 300_000) -> str | None:
    """读文本文件，读不了返回 None。

    读 max_bytes+1 字节是为了能判断「是否超限」：读到恰好 max_bytes 时无法区分
    「刚好这么大」和「被截断了」，多读 1 字节就能确定。
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


def _language_for(path: Path) -> str:
    """从扩展名推语言名，没有扩展名就用文件名（如 Dockerfile 本身）。"""
    suffix = path.suffix.lower().lstrip(".")
    return suffix or path.name


def _git_text(args: list[str], cwd: Path) -> str | None:
    """执行 git 命令取 stdout，任何失败都返回 None（调用方按「未知」处理）。

    strip 后为空也返回 None：像 rev-parse 输出空串基本等于失败，
    归一成 None 让调用方少写一次判空。
    """
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=30,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _line_for_offset(text: str, offset: int) -> int:
    """字符偏移 → 行号（1 起）。

    数 offset 之前的换行个数再加 1 即可；正则 match.start() 给的是偏移，需要这样换算。
    """
    return text.count("\n", 0, offset) + 1


def _symbol_matches(path: Path, text: str) -> list[dict[str, object]]:
    """按语言抽取符号定义，返回 [{name, kind, language, start_line}]。

    只处理 Python 和 TS/JS 家族；其它语言返回空列表（仍会参与 imports/calls 关系抽取）。
    TS 的 kind 是从声明文本里反推的（正则没单独分组），顺序上 class 优先判断，
    因为 "class X" 里也包含不了别的关键字，而 "type"/"interface" 需要各自判断。
    """
    language = _language_for(path)
    matches = []
    if path.suffix.lower() == ".py":
        for match in PY_SYMBOL_RE.finditer(text):
            # Python 正则有两个分组：(def|class, 名字)，kind 直接取第 1 组。
            matches.append({"name": match.group(2), "kind": match.group(1), "language": language, "start_line": _line_for_offset(text, match.start())})
    elif path.suffix.lower() in {".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"}:
        for match in TS_SYMBOL_RE.finditer(text):
            declaration = match.group(0)
            kind = "class" if "class " in declaration else "interface" if "interface " in declaration else "type" if "type " in declaration else "function"
            matches.append({"name": match.group(1), "kind": kind, "language": language, "start_line": _line_for_offset(text, match.start())})
    return matches


def _sync_code_graph(
    db: Session,
    repo: Repository,
    checkout_path: Path,
    *,
    branch: str | None,
    commit_sha: str | None,
    pr_id: uuid.UUID | None = None,
    pr_number: int | None = None,
    max_files: int = 500,
) -> int:
    """重建某个 (repo, branch, pr) 的代码图谱，返回抽到的符号数。

    两步扫描：
    1. 先抽符号，建立 name → 符号行 的索引；
    2. 再扫一遍抽 import 和 calls，用第 1 步的索引把调用关联到具体符号。
    必须先有符号索引，所以没法只扫一遍；文件量受 max_files 限制，两遍的代价可接受。

    `pr_id == pr_id` 这种写法看着怪，其实是 SQLAlchemy 在生成 "IS NULL"：
    默认分支对应的 pr_id/pr_number 是 NULL，用 == None 才能正确匹配到那一批行。
    """
    # 先清空本维度的旧数据再做全量重建：本函数随后一定会重新扫描，
    # 且调用方（sync_repository_code_analysis）在最后统一 commit，
    # 中途抛异常会整体回滚，不会出现「删了但没写回」。
    db.query(CodeRelation).filter(CodeRelation.repo_id == repo.id, CodeRelation.branch == branch, CodeRelation.pr_id == pr_id, CodeRelation.pr_number == pr_number).delete(synchronize_session=False)
    db.query(CodeSymbol).filter(CodeSymbol.repo_id == repo.id, CodeSymbol.branch == branch, CodeSymbol.pr_id == pr_id, CodeSymbol.pr_number == pr_number).delete(synchronize_session=False)
    symbol_rows: list[CodeSymbol] = []
    files_seen = 0
    for path in checkout_path.rglob("*"):
        if files_seen >= max_files:
            break
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(checkout_path).parts):
            continue
        if not _is_text_candidate(path):
            continue
        # 索引阶段用 180KB：比 _read_text 默认的 300KB 更保守，避免超大生成文件拖慢整体。
        text = _read_text(path, max_bytes=180_000)
        if not text:
            continue
        symbols = _symbol_matches(path, text)
        if not symbols:
            continue
        relative_path = path.relative_to(checkout_path).as_posix()
        # 注意 files_seen 只在「有符号的文件」上才 +1：
        # 因此 max_files 限制的是「产出符号的文件数」，而不是扫描文件数。
        # 后果是：符号稀疏的仓库会扫远多于 max_files 个文件（扫描成本高于预期）。
        files_seen += 1
        for item in symbols:
            row = CodeSymbol(
                repo_id=repo.id,
                path=relative_path,
                name=str(item["name"]),
                kind=str(item["kind"]),
                language=str(item["language"]),
                start_line=int(item["start_line"]),
                # end_line 留空：正则看不出函数体结束位置，需要 AST 才能准确计算。
                end_line=None,
                branch=branch,
                commit_sha=commit_sha,
                pr_id=pr_id,
                pr_number=pr_number,
                meta={},
            )
            db.add(row)
            symbol_rows.append(row)
    # flush 拿到所有符号的自增 id，后面创建 calls 关系时要用 target_symbol_id。
    db.flush()
    # 同名符号只保留最后一个（后者覆盖前者）：同名重载/多个类里同名方法会合并成一个，
    # 属于正则方案的已知精度损失。
    by_name = {row.name: row for row in symbol_rows}
    for path in checkout_path.rglob("*"):
        if not path.is_file() or any(part in SKIP_DIRS for part in path.relative_to(checkout_path).parts) or not _is_text_candidate(path):
            continue
        text = _read_text(path, max_bytes=180_000)
        if not text:
            continue
        relative_path = path.relative_to(checkout_path).as_posix()
        # 三个 import 分组只有一个非空，next(...) 取第一个非空值，空串会在下一行被过滤。
        import_targets = [next((part for part in match.groups() if part), "") for match in IMPORT_RE.finditer(text)]
        for target in [item for item in import_targets if item]:
            db.add(
                CodeRelation(
                    repo_id=repo.id,
                    # imports 关系是「文件 → 模块名」，没有具体符号，所以 source/target 用名字而非 id。
                    source_name=relative_path,
                    target_name=target,
                    relation_type="imports",
                    path=relative_path,
                    branch=branch,
                    pr_id=pr_id,
                    pr_number=pr_number,
                    meta={},
                )
            )
        # 取本文件的第一个符号当作 calls 关系的「来源符号」；
        # 一个文件多个符号时只能挂到第一个上，是这套简化模型的粗糙之处。
        local_symbols = [row for row in symbol_rows if row.path == relative_path]
        source = local_symbols[0] if local_symbols else None
        seen_calls: set[str] = set()
        for match in CALL_RE.finditer(text):
            target_name = match.group(1)
            # 去重（同一函数多次调用只留一条关系）并过滤语言关键字/内建函数，
            # 否则 "if ("、"print(" 会制造大量无意义的关系边。
            if target_name in seen_calls or target_name in {"if", "for", "while", "return", "print", "len", "str", "int"}:
                continue
            seen_calls.add(target_name)
            target = by_name.get(target_name)
            db.add(
                CodeRelation(
                    repo_id=repo.id,
                    # target 可能为 None（调用了第三方/未索引的函数），此时只留名字，
                    # 保留这条边对「谁在调用什么」仍然有价值。
                    source_symbol_id=source.id if source else None,
                    target_symbol_id=target.id if target else None,
                    source_name=source.name if source else relative_path,
                    target_name=target_name,
                    relation_type="calls",
                    path=relative_path,
                    branch=branch,
                    pr_id=pr_id,
                    pr_number=pr_number,
                    # 记录调用点行号，便于从关系直接跳到代码位置。
                    meta={"line": _line_for_offset(text, match.start())},
                )
            )
    return len(symbol_rows)


def purge_legacy_code_documents(db: Session, repo_id: uuid.UUID | None = None) -> int:
    """清理历史遗留的 code_file 类型文档（旧版把代码当知识库文档存）。

    现在代码走 CodeSymbol/CodeRelation 图谱，不再进 Document 表；
    这些旧文档如果不删，会和新的向量检索重复召回同一份代码。
    不传 repo_id 就全库清理。返回删除行数。
    """
    query = db.query(Document).filter(Document.source_type == "code_file")
    if repo_id is not None:
        query = query.filter(Document.repo_id == repo_id)
    return query.delete(synchronize_session=False)


async def sync_repository_code_analysis(db: Session, repo: Repository) -> int:
    """同步仓库代码并重建默认分支的代码图谱，返回符号数。

    用 asyncio.to_thread 把阻塞的 git / 文件扫描丢到线程池：
    这是个 async 函数，直接在事件循环里跑 subprocess 会卡住整个服务。
    """
    token = decrypt_token(repo.github_token_encrypted, settings.token_encryption_key) if repo.github_token_encrypted else None
    checkout_path = await asyncio.to_thread(_sync_checkout, repo, token)
    # 分支优先用配置里的默认分支；没配就问 git 当前分支；再不行退回 main。
    branch = repo.default_branch or _git_text(["branch", "--show-current"], checkout_path) or "main"
    commit_sha = _git_text(["rev-parse", "HEAD"], checkout_path)
    symbol_count = await asyncio.to_thread(
        _sync_code_graph,
        db,
        repo,
        checkout_path,
        branch=branch,
        commit_sha=commit_sha,
    )
    # 顺手清掉旧版遗留的代码文档，避免和图谱重复。
    purge_legacy_code_documents(db, repo.id)
    # 回填 clone_url：老数据可能没存，补上便于后续增量更新和前端展示。
    repo.clone_url = repo.clone_url or _clone_url(repo)
    # 整个流程只有这一次提交：图谱重建 + 清理旧文档是一个原子动作，
    # 任一步失败都不会留下「图谱已换、旧文档还在」的中间状态。
    db.commit()
    return symbol_count


async def sync_repository_code_analysis_by_id(repo_id: uuid.UUID) -> int | None:
    """按 ID 触发同步（后台任务用），失败只记日志并返回 None。

    自己建会话：后台任务没有请求上下文，不能用 get_db。
    不向上抛异常：批量调用时一个仓库失败不该打断其余仓库。
    """
    db = SessionLocal()
    try:
        repo = db.get(Repository, repo_id)
        if repo is None:
            return None
        try:
            return await sync_repository_code_analysis(db, repo)
        except Exception as exc:
            logger.warning("Failed to sync repository code analysis for %s: %s", repo_id, exc)
            return None
    finally:
        # 保证连接归还，否则后台反复调用会耗尽连接池。
        db.close()
