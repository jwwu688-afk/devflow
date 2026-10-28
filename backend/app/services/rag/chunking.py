"""切块（chunking）：RAG 链路的第一步，把文档/代码切成可向量化的片段。

位置：**切块 → 向量化 → 检索 → 重排 → 生成 → 答案门禁**。
本模块的产物是带完整元数据的 chunk 列表，下游 indexing.py 直接拿去向量化。

为什么切块质量决定整个 RAG 的上限：向量化之后原文的结构信息就丢了，
检索只能靠片段自身的语义。如果切出来的片段是半句话、或者脱离了所在章节，
那无论后面用多好的 embedding 和重排模型都救不回来。

本文件的核心设计：

1. **两阶段切分**。先按结构切出 _Block（段落 / 标题 / 代码块 / JSON 值 / CSV 行），
   再由 _pack_blocks 把 block 装箱成不超过 max_chars 的 chunk。
   分开做的好处是「结构边界」和「长度预算」两件事可以各自独立调整。

2. **每个 chunk 自带上下文前缀**。见 _section_prefix：把所在标题层级
   （## 一级 / ### 二级）重新拼到 chunk 正文前面。这样单独看一个片段也知道
   它属于哪一节，语义完整度大幅提升。

3. **父文档 + 子块两级 ID**。同一节切出的多个子块共享 parent_id，
   便于「先按小块检索、再按父块扩上下文」的小-大检索策略。

4. **ID 稳定可复现**。全部由内容哈希得出（_stable_id），同样的输入永远得到
   同样的 chunk_id，所以重复索引是幂等的，不会产生重复向量。

5. **按文件类型分派**：Markdown/PDF/DOCX 走标题结构，JSON 按顶层 key，
   CSV 按行并重复表头，代码按 AST/正则出的符号边界，其他走纯文本窗口。
"""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# 走 Markdown 结构解析的后缀。注意 .pdf / .docx 也在这里——它们在上游
# document_extraction 里已经被转成带 `#` 标题和 `<!-- page: N -->` 标记的
# Markdown 文本，所以能复用同一套解析器
MARKDOWN_SUFFIXES = {".md", ".markdown", ".mdx", ".pdf", ".docx"}
# 结构化数据后缀：JSON 按顶层 key 切，CSV 按行切
STRUCTURED_DATA_SUFFIXES = {".json", ".csv"}


@dataclass
class _Block:
    """切分过程中的最小中间单元（内部类型）。

    一个 block 通常对应「一段正文」「一个标题行」「一整块代码围栏」或
    「一个 JSON 顶层值」。它还不是最终 chunk——多个 block 会被 _pack_blocks
    按长度预算装进同一个 chunk。

    字段说明（元数据都会向下传递，最终进入向量库的 metadata）：
    - start_line / end_line：1 基行号，用于溯源与前端高亮；None 表示无行号概念
      （例如 PDF 转出的文本、JSON 反序列化后的值）
    - section_path：标题层级路径，如 ("部署", "环境变量")，用于拼上下文前缀
    - section_instance：带同名序号（occurrence）的路径。同一篇文档里可能出现
      两个 "## 配置"，靠 occurrence 区分，保证 parent_id 不会撞车
    - page：PDF 页码
    """

    text: str
    start_line: int | None
    end_line: int | None
    kind: str = "paragraph"
    section_path: tuple[str, ...] = ()
    section_instance: tuple[tuple[str, int], ...] = ()
    page: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# 单文档 chunk 数上限。这是一个防「事故」的闸门而不是业务限制：
# 如果有人误把一个几百 MB 的日志当知识文件上传，没有这道闸就会生成几十万个
# 向量，把 embedding API 的额度和 Milvus 一起打爆。触发时直接抛错，不静默截断
MAX_CHUNKS_PER_DOCUMENT = 5000


def _validate_limits(max_chars: int, overlap: int) -> None:
    """校验切块参数合法性。

    overlap 必须严格小于 max_chars：否则每次滑动都不前进，切块循环会变成死循环
    （或产生无限多的重复 chunk）。
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must be between zero and max_chars")


def _normalize_text(text: str) -> str:
    """统一换行符为 \\n。

    Windows 来源的文件是 CRLF，若不归一化，行号计算（count("\\n")）与
    按行切分都会出现错位或残留 \\r。
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _stable_id(*parts: object) -> str:
    """由若干字段生成稳定的短 ID（sha256 前 24 位十六进制）。

    两个细节：
    - 用 \\x1f（Unit Separator）而不是普通分隔符拼接：这是控制字符，
      不会出现在正常的路径/正文里，避免 "a-b"+"c" 与 "a"+"b-c" 撞成同一个串。
    - 截断到 24 个十六进制字符（96 bit）：碰撞概率足够低，同时 ID 短、
      便于放进 Milvus 主键与日志。

    稳定性是刻意保证的——同样的输入永远得到同样的 ID，所以重新索引时
            upsert 会覆盖同一条记录，而不是产生重复向量。
    """
    raw = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _line_for_offset(text: str, offset: int, first_line: int | None) -> int | None:
    """把「文本内字符偏移」换算成「1 基行号」。

    first_line 为 None 时直接返回 None（无行号概念的数据源）。
    offset 兜底为 0，防止负偏移让 count 的区间反了（Python 里会静默返回 0，
    但语义上不对）。
    """
    if first_line is None:
        return None
    return first_line + text.count("\n", 0, max(0, offset))


def _split_long_text(block: _Block, max_chars: int) -> list[_Block]:
    """把超长的普通段落按句子边界拆成多个 block。

    切分优先级（越靠前越自然）：
    1. 中文句末标点后（。！？）
    2. 英文句末标点 + 空白后（. ! ? 后跟空格，避免把 3.14 拆开）
    3. 换行处

    如果整段连一个句子边界都没有（比如一长串无标点的中文），退化为
    按 max_chars 硬切——这是下策，但总比切不动好。

    拆分后要用 find(piece, cursor) 反查每段在原 block 里的偏移，才能算出
    正确的行号；cursor 单调前进，保证同一段文本不会被重复定位到前面的位置。
    """
    if len(block.text) <= max_chars:
        return [block]

    # 三个分支用 | 串联：先按中文句末、再按英文句末（后随空白）、最后按换行
    units = [
        unit
        for unit in re.split(r"(?<=[。！？!?])|(?<=[.!?])\s+|\n+", block.text)
        if unit and unit.strip()
    ]
    # 一个边界都没有（或只有一个单元）→ 无法按语义切，直接按字符数硬切
    if len(units) <= 1:
        units = [block.text[index : index + max_chars] for index in range(0, len(block.text), max_chars)]

    pieces: list[str] = []
    current = ""
    for unit in units:
        unit = unit.strip()
        if not unit:
            continue
        # 单个句子本身就超长：先把已攒的吐出，再把这个长句按长度硬切
        if len(unit) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(unit[index : index + max_chars] for index in range(0, len(unit), max_chars))
            continue
        # 用 \n 拼接而不是空格：保留原文档的换行结构，行号换算也更准
        candidate = f"{current}\n{unit}" if current else unit
        if current and len(candidate) > max_chars:
            pieces.append(current)
            current = unit
        else:
            current = candidate
    if current:
        pieces.append(current)

    output: list[_Block] = []
    cursor = 0
    for piece in pieces:
        offset = block.text.find(piece, cursor)
        # find 失败说明原文里有前后空白被 strip 掉了，退化为「从上次位置继续」，
        # 保证行号至少是单调的而不是乱跳
        if offset < 0:
            offset = cursor
        end_offset = min(len(block.text), offset + len(piece))
        output.append(
            _Block(
                text=piece,
                start_line=_line_for_offset(block.text, offset, block.start_line),
                # end 用 end_offset - 1：区间是闭区间，避免与下一段的起始行重叠一行
                end_line=_line_for_offset(block.text, max(offset, end_offset - 1), block.start_line),
                kind=block.kind,
                section_path=block.section_path,
                section_instance=block.section_instance,
                page=block.page,
                # 浅拷贝 metadata：拆出的每段都能独立追加自己的键而不影响兄弟
                metadata=dict(block.metadata),
            )
        )
        cursor = end_offset
    return output


def _split_long_code_fence(block: _Block, max_chars: int) -> list[_Block]:
    """拆超长代码围栏，并保证**每一片都仍是合法的 Markdown 代码块**。

    这是本文件里最琐碎的一段，核心难点在于长度预算：
    每片都要额外加上开围栏（```）+ 换行 + 闭围栏（```），
    所以正文的可用长度是 max_chars - wrapper_size，而不是 max_chars。
    漏算这一步就会让某个 chunk 实测超过 max_chars，被 _pack_blocks 末尾的
    断言拦住。

    max_chars <= 8 时连围栏都放不下，降级为普通文本切分并标注 code_fragment。
    """
    if len(block.text) <= max_chars:
        return [block]
    lines = block.text.splitlines()
    if not lines:
        return []

    if max_chars <= 8:
        # 预算太小，围栏开销（6 字符 + 2 个换行）已经占满，放弃保持代码块语义
        plain = _Block(
            text=block.text,
            start_line=block.start_line,
            end_line=block.end_line,
            kind="code_fragment",
            section_path=block.section_path,
            section_instance=block.section_instance,
            page=block.page,
            metadata=dict(block.metadata),
        )
        return _split_long_text(plain, max_chars)

    # 保留原围栏的标记类型（``` 或 ~~~），取前 3 个字符做标记；
    # 原文首行不是围栏时兜底用 ```
    original_opener = lines[0] if lines[0].lstrip().startswith(("```", "~~~")) else "```"
    marker = original_opener.lstrip()[:3]
    opener = marker
    # 有闭合围栏才排除最后一行；没有闭合的（被上游截断的块）保留全部内容
    has_closer = len(lines) > 1 and lines[-1].lstrip().startswith(marker)
    body = lines[1:-1] if has_closer else lines[1:]
    closing = marker
    # +2 是两个换行（围栏与正文之间各一个）
    wrapper_size = len(opener) + len(closing) + 2
    body_limit = max_chars - wrapper_size

    if not body:
        # 空代码块（只有 ``` 和 ```），保持原样输出一片
        return [
            _Block(
                text=f"{opener}\n{closing}",
                start_line=block.start_line,
                end_line=block.end_line,
                kind=block.kind,
                section_path=block.section_path,
                section_instance=block.section_instance,
                page=block.page,
                metadata=dict(block.metadata),
            )
        ]

    output: list[_Block] = []
    current: list[str] = []
    current_size = 0
    # 正文从原围栏的下一行开始
    first_body_line = (block.start_line + 1) if block.start_line is not None else None
    current_start = first_body_line

    def flush(end_line: int | None) -> None:
        """把累积的正文行吐成一片，每片都重新套上开/闭围栏。

        metadata 里额外记录 fence_start_line / fence_end_line，
        让下游知道这片来自哪个原始代码块（排查时很有用）。
        """
        nonlocal current, current_size, current_start
        if not current:
            return
        body_text = "\n".join(current)
        output.append(
            _Block(
                text=f"{opener}\n{body_text}\n{closing}",
                start_line=current_start,
                end_line=end_line,
                kind=block.kind,
                section_path=block.section_path,
                section_instance=block.section_instance,
                page=block.page,
                metadata={
                    **block.metadata,
                    "fence_start_line": block.start_line,
                    "fence_end_line": block.end_line if has_closer else None,
                },
            )
        )
        current = []
        current_size = 0

    for body_index, line in enumerate(body):
        index = (first_body_line + body_index) if first_body_line is not None else None
        # 单行就超预算（压缩过的 JS、超长 JSON 行）：必须按字符硬切这一行，
        # 否则它永远装不进任何一片
        if len(line) > body_limit:
            flush((index - 1) if index is not None else None)
            for offset in range(0, len(line), body_limit):
                piece = line[offset : offset + body_limit]
                output.append(
                    _Block(
                        text=f"{opener}\n{piece}\n{closing}",
                        start_line=index,
                        end_line=index,
                        kind=block.kind,
                        section_path=block.section_path,
                        section_instance=block.section_instance,
                        page=block.page,
                        metadata={
                            **block.metadata,
                            "fence_start_line": block.start_line,
                            "fence_end_line": block.end_line if has_closer else None,
                        },
                    )
                )
            current_start = (index + 1) if index is not None else None
            continue
        # 换行符也算 1 个字符（除第一行外），保证拼出来的正文长度估算准确
        line_size = len(line) + (1 if current else 0)
        if current and current_size + line_size > body_limit:
            flush((index - 1) if index is not None else None)
            current_start = index
        if not current:
            current_start = index
        current.append(line)
        current_size += line_size
    last_body_line = None
    if block.end_line is not None:
        # 有闭合围栏时，正文最后一行是 end_line - 1
        last_body_line = block.end_line - (1 if has_closer else 0)
    flush(last_body_line)
    return output


def _expand_block(block: _Block, max_chars: int) -> list[_Block]:
    """按 block 类型选择拆分策略：代码围栏走专用逻辑（保持围栏合法），其余走句子切分。"""
    if block.kind == "code_fence":
        return _split_long_code_fence(block, max_chars)
    return _split_long_text(block, max_chars)


def _overlap_blocks(blocks: list[_Block], overlap: int) -> list[_Block]:
    """从上一个 chunk 的**尾部**取出不超过 overlap 字符的内容，作为下一个 chunk 的开头。

    为什么需要重叠：句子被切断时，后半句单独看往往语义不完整。带上上一块的
    尾巴能让上下文连续，显著改善「跨块提问」的召回效果。

    三个实现细节：
    1. 跳过 code_fence：代码块带围栏标记，截断后拼到下一个 chunk 开头会破坏
       代码块结构，宁可丢掉这段重叠。
    2. 从后往前累加，直到预算用完——保证取到的是紧邻边界的最近内容。
    3. 单块装不下时，只取它的最后 allowance 个字符，并且**尽力在句子/单词
       边界处下刀**（lookback 窗口 + 正则回溯），避免从半个词开始。
       实在找不到边界（比如一长串中文），退而求其次向后找边界，仍找不到就直接切。
    """
    if overlap <= 0 or not blocks:
        return []
    selected: list[_Block] = []
    remaining = overlap
    for block in reversed(blocks):
        if not block.text or block.kind == "code_fence":
            continue
        # 多个 block 拼接时要有 \n\n 分隔（2 个字符），第一块不需要
        separator = 2 if selected else 0
        allowance = remaining - separator
        if allowance <= 0:
            break
        if len(block.text) <= allowance:
            selected.append(block)
            remaining -= len(block.text) + separator
            continue
        offset = len(block.text) - allowance
        # 往前多看一段（16~64 字符）找句子边界；窗口大小随 allowance 放大，
        # 但夹在 16~64 之间，避免无谓的大范围搜索
        lookback_start = max(0, offset - min(64, max(16, allowance * 2)))
        lookback = block.text[lookback_start:offset]
        sentence_boundaries = list(re.finditer(r"[。！？.!?；;]\s*", lookback))
        word_boundaries = list(re.finditer(r"\s+", lookback))
        if sentence_boundaries:
            # 取最后一个句末 → 从下一句开始，语义最完整
            fragment_offset = lookback_start + sentence_boundaries[-1].end()
        elif word_boundaries:
            # 没有句末就退到词边界
            fragment_offset = lookback_start + word_boundaries[-1].end()
        else:
            # 连空白都没有（纯中文长句）：改为**向后**找第一个边界，
            # 窗口 8~48 字符，宁可少一点重叠也不要从半个词开始
            boundary_window = block.text[offset : offset + min(48, max(8, allowance // 3))]
            boundary = re.search(r"\s+|[。！？.!?；;]\s*", boundary_window)
            fragment_offset = offset + boundary.end() if boundary else offset
        selected.append(
            _Block(
                text=block.text[fragment_offset:],
                start_line=_line_for_offset(block.text, fragment_offset, block.start_line),
                end_line=block.end_line,
                kind=block.kind,
                section_path=block.section_path,
                section_instance=block.section_instance,
                page=block.page,
                # 打标记，便于统计/排查「这段是重叠带过来的」
                metadata={**block.metadata, "overlap_fragment": True},
            )
        )
        remaining = 0
        break
    # 上面是倒序遍历，这里反转回来恢复文档顺序
    return list(reversed(selected))


def _section_prefix(section_path: tuple[str, ...], page: int | None, limit: int | None = None) -> str:
    """把标题层级渲染成 Markdown 标题前缀，例如 "## 部署\\n### 环境变量"。

    为什么要有这个前缀：chunk 被切出来之后，就失去了「我在哪一节」的信息。
    把标题路径重新拼回正文开头，等于给每段话补上了上下文标签，
    对「这段代码属于哪个模块」这类问题的检索效果提升很明显。

    层级用 min(index + 1, 6) 限制在 6 级以内（Markdown 最多支持 6 个 #）。
    limit 不为空且超长时截断并加 "..."；limit <= 3 时连 "..." 都放不下，直接硬切。
    """
    parts = [f"{'#' * min(index + 1, 6)} {title}" for index, title in enumerate(section_path)]
    if page is not None:
        # PDF 页码也作为上下文的一部分，方便判断内容在第几页
        parts.append(f"Page {page}")
    value = "\n".join(parts)
    if limit is None or len(value) <= limit:
        return value
    if limit <= 3:
        return value[:limit]
    return value[: limit - 3].rstrip() + "..."


def _prefix_for_chunk(
    section_path: tuple[str, ...],
    page: int | None,
    max_chars: int,
    *,
    has_body: bool,
) -> str:
    """计算某个 chunk 允许使用的标题前缀长度上限。

    预算分配规则：
    - 没有标题也没有页码 → 不加前缀（省下全部预算给正文）；
    - 只有前缀没有正文 → 前缀可以吃掉整个 max_chars；
    - 前缀 + 正文 → 前缀最多占 **三分之一**，且硬上限 512 字符。

    为什么要截前缀：一份层级很深的文档（6 级标题 + 长标题名）可能让前缀
    就有好几百字符，若不加限制，正文空间会被挤成个位数，chunk 全是标题没有内容。
    """
    if not section_path and page is None:
        return ""
    if not has_body:
        limit = max_chars
    elif max_chars <= 3:
        # 预算小到连章节前缀都放不下（例如 max_chars=2），干脆不加
        limit = 0
    else:
        # -3 是给 "..." 预留的空间，保证截断后总长仍不超 max_chars
        limit = min(512, max_chars // 3, max_chars - 3)
    return _section_prefix(section_path, page, limit=limit) if limit > 0 else ""


def _join_block_text(blocks: list[_Block]) -> str:
    """把多个 block 拼成 chunk 正文。

    分隔符用两个换行（空行）而不是单个 \\n：这样 Markdown 里相邻的两段
    仍然保持独立段落，代码块前后也符合 CommonMark 要求。
    空 block 被过滤掉，避免拼出多余的空行；整体 strip 去掉首尾空白。
    """
    return "\n\n".join(block.text for block in blocks if block.text).strip()


def _emit_chunk(
    blocks: list[_Block],
    *,
    source_path: str,
    chunk_type: str,
    child_index: int,
    max_chars: int,
) -> dict[str, Any]:
    """把一组 block 组装成最终的 chunk 字典（下游 indexing 直接消费这个结构）。

    产出的字段分三类：
    - 检索用：content（实际被向量化的文本，**含标题前缀**）
    - 溯源用：chunk_id / parent_id / child_index / start_line / end_line /
      section_path / section_title / page
    - 长度校验用：max_chars 只用于计算前缀预算，不落库

    ID 设计：
    - parent_id 只由「源路径 + 章节实例 + 页码」决定 → 同一节的多个子块共享父 ID，
      支撑小-大检索（先命中细粒度子块，再取父块的全部内容）。
    - chunk_id 额外混入 child_index 与 content → 内容变了 ID 就变，
      避免旧向量残留在库里被误检索。
    section_instance 为空时退化成 tuple((title, 0), ...)，保证 parent_id 始终可算。

    注意 effective_type：只要这组 block 里含代码围栏，chunk_type 就升级为
    markdown_code——下游可以据此对代码片段采用不同的展示/高亮策略。
    """
    first = blocks[0]
    body = _join_block_text(blocks)
    prefix = _prefix_for_chunk(first.section_path, first.page, max_chars, has_body=bool(body))
    # 三种情况：有前缀+有正文 → 中间空一行；只有其一时直接取非空的那一个
    content = f"{prefix}\n\n{body}" if prefix and body else (prefix or body)
    parent_id = _stable_id(
        "parent",
        source_path,
        first.section_instance or tuple((title, 0) for title in first.section_path),
        first.page,
    )
    effective_type = "markdown_code" if any(block.kind == "code_fence" for block in blocks) else chunk_type
    # 逐个 update：后出现的 block 的 metadata 会覆盖先前的同名键。
    # 这是有意的——例如重叠片段带的 overlap_fragment 标记会保留下来
    metadata: dict[str, Any] = {}
    for block in blocks:
        metadata.update(block.metadata)
    return {
        "content": content,
        "chunk_id": _stable_id("chunk", parent_id, child_index, content),
        "parent_id": parent_id,
        "child_index": child_index,
        "chunk_type": effective_type,
        # 落库时用 list，因为 JSON 不支持 tuple
        "section_path": list(first.section_path),
        "section_title": first.section_path[-1] if first.section_path else None,
        "page": first.page,
        # 取所有 block 的最小起始行与最大结束行；都无行号时 default=None
        "start_line": min((block.start_line for block in blocks if block.start_line is not None), default=None),
        "end_line": max((block.end_line for block in blocks if block.end_line is not None), default=None),
        **metadata,
    }


def _pack_blocks(
    blocks: list[_Block],
    *,
    source_path: str,
    max_chars: int,
    overlap: int,
    chunk_type: str,
) -> list[dict[str, Any]]:
    """装箱：把 block 序列按长度预算装成 chunk。

    算法是一次贪心扫描，规则如下：
    - 遇到**章节或页码变化**（key 变化）立即 flush：一个 chunk 绝不跨章节，
      否则前缀只能写一个标题，另一半内容就失去了上下文；
    - 先对单个 block 做 _expand_block（超长的先拆小），再尝试并入当前 chunk；
    - 装不下时 flush 当前 chunk，然后用 _overlap_blocks 从刚 flush 的内容尾部
      取一段重叠作为新 chunk 的开头；由于重叠部分加上新 block 可能又超预算，
      这里有个 while 循环不断丢弃重叠段最前面的 block，直到装得下；
    - 每 flush 一次，同一 key 下的 child_index 自增，保证子块编号连续。

    child_counts 用 key 做字典而不是简单的计数器，是因为文档会来回跨节
    （虽然正常 Markdown 不会，但 PDF 转出的文本可能），按 key 记数才不会串号。

    最后一个断言是「不该发生」的兜底：如果某处长度算漏了（比如前缀预算
    与正文预算不一致），这里会立刻炸出来，而不是让超长 chunk 悄悄进向量库
    ——超长 chunk 会导致 embedding API 报错或内容被静默截断。
    """
    chunks: list[dict[str, Any]] = []
    current: list[_Block] = []
    current_key: tuple[tuple[tuple[str, int], ...], int | None] | None = None
    child_counts: dict[tuple[tuple[tuple[str, int], ...], int | None], int] = {}

    def flush() -> None:
        """输出当前累积的 block 为一个 chunk，并推进该章节的子块序号。"""
        nonlocal current
        if not current or current_key is None:
            return
        child_index = child_counts.get(current_key, 0)
        chunks.append(
            _emit_chunk(
                current,
                source_path=source_path,
                chunk_type=chunk_type,
                child_index=child_index,
                max_chars=max_chars,
            )
        )
        child_counts[current_key] = child_index + 1

    for raw_block in blocks:
        key = (raw_block.section_instance, raw_block.page)
        # 先按「最坏情况」估算前缀长度（此时还不知道会并入哪些 block），
        # 据此得到正文可用的长度上限，再去拆超长 block
        prefix = _prefix_for_chunk(
            raw_block.section_path,
            raw_block.page,
            max_chars,
            has_body=bool(raw_block.text),
        )
        prefix_size = len(prefix)
        # max(1, ...) 保证 body_limit 至少为 1，否则 _expand_block 会死循环
        # -2 是为前缀与正文之间那个空行预留
        body_limit = max(1, max_chars - prefix_size - (2 if prefix_size else 0))
        for block in _expand_block(raw_block, body_limit):
            key = (block.section_instance, block.page)
            # 章节/页码切换 → 必须先收口，chunk 不跨节
            if current and key != current_key:
                flush()
                current = []
            current_key = key
            # 用「如果把这个 block 加进去」的实际拼装结果来算真实大小，
            # 因为前缀是按合并后的第一条 block 重新计算的，可能比预估的短
            candidate_body = _join_block_text([*current, block])
            prefix = _prefix_for_chunk(
                block.section_path,
                block.page,
                max_chars,
                has_body=bool(candidate_body),
            )
            candidate_size = len(candidate_body) + len(prefix) + (2 if prefix else 0)
            if current and candidate_size > max_chars:
                previous = list(current)
                flush()
                current = _overlap_blocks(previous, overlap)
                # 重叠段可能太长，导致「重叠 + 新 block」仍超预算；
                # 从最老的重叠块开始丢，直到装得下（最差会丢空，等价于无重叠）
                while current:
                    candidate_body = _join_block_text([*current, block])
                    if len(candidate_body) + len(prefix) + (2 if prefix else 0) <= max_chars:
                        break
                    current.pop(0)
            current.append(block)
    flush()
    if any(len(chunk["content"]) > max_chars for chunk in chunks):
        raise AssertionError("chunk content exceeded max_chars")
    return chunks


def _plain_blocks(text: str) -> list[_Block]:
    """把纯文本按空行切成段落 block（无标题结构）。

    行号处理有一个易错点：段落内容会 strip 掉首尾空行，但记录的
    start_line / end_line 仍是**原始行号**，所以前端能准确高亮原文。
    flush 传的是「空行的前一行」（line_number - 1），也就是段落真正的末行。
    """
    lines = text.splitlines()
    blocks: list[_Block] = []
    current: list[str] = []
    start_line = 1

    def flush(end_line: int) -> None:
        nonlocal current
        content = "\n".join(current).strip()
        if content:
            # max(start_line, end_line) 防御只有一个空格的「假段落」导致 end < start
            blocks.append(_Block(content, start_line, max(start_line, end_line)))
        current = []

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            flush(line_number - 1)
            continue
        if not current:
            start_line = line_number
        current.append(line)
    # 文件末尾没有空行时也要 flush 最后一个段落
    flush(len(lines))
    return blocks


def _plain_chunks(
    text: str,
    *,
    source_path: str,
    max_chars: int,
    overlap: int,
) -> list[dict[str, Any]]:
    """纯文本的滑动窗口切分（不走 block 装箱，因为纯文本没有结构可言）。

    关键点：
    - **优先在边界处收尾**：从「半窗之后」开始向前找 \\n\\n 或 \\n，
      找到就把 end 挪到边界。只从 max_chars//2 之后开始找，是为了保证
      每块至少有半个窗口的内容，不会因为边界太靠前而切出一堆短块。
    - 一级边界（空行/段落）优先于二级边界（换行）：max 取更大的偏移，
      \\n\\n 的位置天然比 \\n 靠前或相等，取 max 即优先选它。
    - overlap 回退时用 max(cursor + 1, ...) 兜底，保证 cursor 一定前进，
      否则 overlap >= 窗口时会造成死循环。
    - chunk 数是全局递增的 child_index，与 _plain_blocks 的章节无关。
    """
    chunks: list[dict[str, Any]] = []
    parent_id = _stable_id("parent", source_path, "plain")
    cursor = 0
    text_length = len(text)
    while cursor < text_length:
        end = min(text_length, cursor + max_chars)
        if end < text_length:
            search_start = cursor + max(1, max_chars // 2)
            candidates = [
                text.rfind("\n\n", search_start, end),
                text.rfind("\n", search_start, end),
            ]
            boundary = max(candidates)
            # boundary > cursor 才采用：否则边界落在上一块里，会让本块倒着走
            if boundary > cursor:
                # 命中空行时要把两个 \n 都算进本块，命中单换行则只算一个
                end = boundary + (2 if text.startswith("\n\n", boundary) else 1)

        raw = text[cursor:end]
        # 用「左 trim 长度」和「右 trim 后的长度」还原 trim 前后的偏移，
        # 从而保证行号是基于原始文本算的，而不是 trim 后的片段
        left_trim = len(raw) - len(raw.lstrip())
        right_trim = len(raw.rstrip())
        content = raw[left_trim:right_trim]
        if content:
            start_offset = cursor + left_trim
            end_offset = cursor + right_trim
            child_index = len(chunks)
            chunks.append(
                {
                    "content": content,
                    "chunk_id": _stable_id("chunk", parent_id, child_index, content),
                    "parent_id": parent_id,
                    "child_index": child_index,
                    "chunk_type": "text",
                    "section_path": [],
                    "section_title": None,
                    "page": None,
                    # 纯文本首行固定为 1
                    "start_line": _line_for_offset(text, start_offset, 1),
                    "end_line": _line_for_offset(text, max(start_offset, end_offset - 1), 1),
                }
            )
            if len(chunks) > MAX_CHUNKS_PER_DOCUMENT:
                raise ValueError(f"document exceeds the {MAX_CHUNKS_PER_DOCUMENT} chunk limit")
        if end >= text_length:
            break
        # 回退 overlap 个字符形成重叠；+1 兜底保证严格前进
        cursor = max(cursor + 1, end - overlap)
    return chunks


def _markdown_blocks(text: str) -> list[_Block]:
    """把 Markdown 解析成 block 序列，同时维护「当前所处的标题层级」。

    这是全文件最重要的解析器，Markdown 与 PDF/DOCX（已转成 Markdown）都走它。

    三个状态机变量：
    - headings：标题栈，按层级维护。遇到新标题时不断 pop 掉层级 >= 当前层级
      的旧标题（while headings[-1][0] >= level），这样栈内容始终是「从根到当前
      标题的路径」，直接就是 section_path。
    - fence_marker：非 None 表示正处于代码围栏内部。围栏内的 # 不能被当成标题、
      空行也不能断开 block，否则会把代码切碎。
    - current_kind：当前累积中的 block 类型，用于标记代码块。

    同名标题的处理：heading_occurrences 以「父路径 + 层级 + 标题」为键计数。
    一份文档里出现两个 "## 配置" 时，第二个的 occurrence=1，于是两个章节的
    section_instance 不同，parent_id 不会撞车——这是避免「不同章节的子块被合并成
    同一父文档」这类隐蔽 bug 的关键。

    标题行本身也产出一个 text 为空、kind="heading" 的 block：因为它在
    _pack_blocks 里会触发 section 切换，但没有正文需要向量化。
    """
    lines = text.splitlines()
    blocks: list[_Block] = []
    headings: list[tuple[int, str, int]] = []
    heading_occurrences: dict[tuple[tuple[tuple[str, int], ...], int, str], int] = {}
    current_page: int | None = None
    current: list[str] = []
    current_start = 1
    current_kind = "paragraph"
    fence_marker: str | None = None
    heading_re = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
    # PDF 转 Markdown 时插入的页码标记，例如 <!-- page: 3 -->
    page_re = re.compile(r"^<!--\s*page\s*:\s*(\d+)\s*-->$", re.IGNORECASE)

    def section_path() -> tuple[str, ...]:
        """当前标题栈的纯标题路径（用于展示与生成前缀）。"""
        return tuple(title for _, title, _ in headings)

    def section_instance() -> tuple[tuple[str, int], ...]:
        """当前标题栈带序号的形式（用于生成稳定的 parent_id）。"""
        return tuple((title, occurrence) for _, title, occurrence in headings)

    def flush(end_line: int) -> None:
        """把累积行收成一个 block，并带上当前标题层级与页码。"""
        nonlocal current
        content = "\n".join(current).strip()
        if content:
            blocks.append(
                _Block(
                    content,
                    current_start,
                    max(current_start, end_line),
                    kind=current_kind,
                    section_path=section_path(),
                    section_instance=section_instance(),
                    page=current_page,
                )
            )
        current = []

    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        # —— 状态 1：在代码围栏内部 ——
        # 逐行累积，只有遇到同类围栏标记才收口。中间的空行、# 全部当普通内容
        if fence_marker is not None:
            current.append(line)
            if stripped.startswith(fence_marker):
                flush(line_number)
                fence_marker = None
                current_kind = "paragraph"
            continue

        # 页码标记：先收口前文，再更新页码。页码标记自身不产生 block
        page_match = page_re.match(stripped)
        if page_match:
            flush(line_number - 1)
            current_page = int(page_match.group(1))
            continue

        # 标题：弹出层级 >= 自身的旧标题，再压入新标题
        heading_match = heading_re.match(line)
        if heading_match:
            flush(line_number - 1)
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            while headings and headings[-1][0] >= level:
                headings.pop()
            # 序号要在「父路径确定之后」取，这样同名但不同父章节的标题互不干扰
            parent_instance = section_instance()
            occurrence_key = (parent_instance, level, title)
            occurrence = heading_occurrences.get(occurrence_key, 0)
            heading_occurrences[occurrence_key] = occurrence + 1
            headings.append((level, title, occurrence))
            # 空 text 的 heading block：只为标记章节切换点
            blocks.append(
                _Block(
                    "",
                    line_number,
                    line_number,
                    kind="heading",
                    section_path=section_path(),
                    section_instance=section_instance(),
                    page=current_page,
                    metadata={"heading_line": line_number},
                )
            )
            continue

        # 代码围栏开始：把开围栏本身也放进 current，后续 _split_long_code_fence
        # 依赖「首行是 ``` 」来识别围栏
        if stripped.startswith(("```", "~~~")):
            flush(line_number - 1)
            current_start = line_number
            current_kind = "code_fence"
            fence_marker = stripped[:3]
            current = [line]
            continue

        # 空行即段落边界
        if not stripped:
            flush(line_number - 1)
            continue
        if not current:
            current_start = line_number
        current.append(line)
    flush(len(lines))
    return blocks


def _json_blocks(text: str, max_chars: int) -> list[_Block]:
    """把 JSON 拆成「一个顶层 key / 一个数组元素 = 一个 block」。

    为什么按顶层切开：JSON 的语义单位就是顶层键值。把整个对象塞进一个 chunk，
    向量会被几十个无关字段平均掉；按 key 切开后每个 chunk 都是「一个字段的
    完整内容」，检索精度高得多。section_path 直接用 JSONPath（$.a.b 或 $[3]），
    既能展示又能溯源。

    两个降级路径：
    - 解析失败 → 当纯文本处理（可能不是合法 JSON，或文件被截断）；
    - 格式化后整体不超预算 → 干脆不拆，保持单个 chunk（小文件没必要碎化）。

    注意 indent=2 + ensure_ascii=False：中文保持原字符，缩进便于人读，
    也让向量模型看到清晰的结构。
    """
    try:
        value = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return _plain_blocks(text)

    formatted = json.dumps(value, ensure_ascii=False, indent=2)
    if len(formatted) <= max_chars:
        return [
            _Block(
                formatted,
                1,
                max(1, len(text.splitlines())),
                kind="json_value",
                metadata={"json_path": "$"},
            )
        ]

    blocks: list[_Block] = []
    if isinstance(value, dict):
        # 保留 key 本身：{"a": ...} 而不是只取 value，因为键名本身携带语义
        items = [(f"$.{key}", {key: item}) for key, item in value.items()]
    elif isinstance(value, list):
        items = [(f"$[{index}]", item) for index, item in enumerate(value)]
    else:
        # 顶层是标量（罕见）：整体作为一个块
        items = [("$", value)]
    for json_path, item in items:
        blocks.append(
            _Block(
                json.dumps(item, ensure_ascii=False, indent=2),
                # JSON 反序列化后已无行号对应关系，置 None
                None,
                None,
                kind="json_value",
                section_path=(json_path,),
                section_instance=((json_path, 0),),
                metadata={"json_path": json_path},
            )
        )
    return blocks


def _csv_chunks(text: str, source_path: str, max_chars: int, overlap: int) -> list[dict[str, Any]]:
    """CSV 切块：按行装箱，并且**每个 chunk 都重复表头**。

    为什么必须重复表头：一行数据 "2,500,2026-01-03,已完成" 脱离表头之后
    完全无法理解。把表头跟在每块前面，chunk 才是自解释的——
    这也是 CSV 这类表格数据能进语义检索的前提。

    处理顺序里有几个不太直观的地方：

    1. csv.field_size_limit 动态放大：Python 默认单字段上限 128KB，超了会
       抛 csv.Error。这里按文本长度临时调大（这是进程级全局设置，属于
       已知的副作用，但在这条离线索引路径上可以接受）。

    2. 表头本身超预算：先按 max_chars 硬切表头并单独成块（chunk_type=
       csv_header_part），之后正文不再带前缀。否则每块都被表头塞满。

    3. 单行超预算：把这一行按 row_budget 切成多片，用 row_part / row_parts
       标记这是同一行的第几片，便于下游还原。

    4. 结尾断言：CSV 这条路径的长度计算最绕（表头 + 多行 + 换行），
       一旦算错就直接炸掉，不能把超长 chunk 放进向量库。
    """
    try:
        # 把单字段上限提到整个文本长度，保证不会因为某个超长字段直接失败
        if len(text) > csv.field_size_limit():
            csv.field_size_limit(len(text))
        rows = list(csv.reader(io.StringIO(text)))
    except csv.Error:
        # CSV 格式错误（如引号未闭合）→ 退回纯文本切分，至少内容不丢
        return _pack_blocks(
            _plain_blocks(text),
            source_path=source_path,
            max_chars=max_chars,
            overlap=overlap,
            chunk_type="text",
        )
    if not rows:
        return []

    def render(row: list[str]) -> str:
        """把一行 list 还原成 CSV 文本。

        lineterminator="" 去掉行尾换行，因为这里要手动控制拼接换行，
        让长度计算与最终内容严格一致。
        借 csv.writer 而不是 ",".join，是为了正确处理含逗号/引号的字段。
        """
        buffer = io.StringIO()
        csv.writer(buffer, lineterminator="").writerow(row)
        return buffer.getvalue()

    header = render(rows[0])
    chunks: list[dict[str, Any]] = []
    parent_id = _stable_id("parent", source_path, "csv")

    def append_chunk(
        content: str,
        *,
        row_start: int,
        row_end: int,
        chunk_type: str = "csv_rows",
        row_part: int | None = None,
        row_parts: int | None = None,
        precise_lines: bool = True,
    ) -> None:
        """追加一个 CSV chunk。

        precise_lines 表示「这个 chunk 的 start_line/end_line 是否可信」：
        行内被切成多片时，内容与原 CSV 行不再一一对应，所以行号置 None，
        只保留 row_start / row_end 这两个**逻辑行号**。
        """
        child_index = len(chunks)
        chunks.append(
            {
                "content": content,
                "chunk_id": _stable_id("chunk", parent_id, child_index, content),
                "parent_id": parent_id,
                "child_index": child_index,
                "chunk_type": chunk_type,
                "section_path": [],
                "section_title": None,
                "page": None,
                "start_line": row_start if precise_lines else None,
                "end_line": row_end if precise_lines else None,
                # 逻辑行号始终保留，便于按「第 N 行」定位
                "row_start": row_start,
                "row_end": row_end,
                # 只有行被切开时才带这两个字段（用字典展开实现条件键）
                **({"row_part": row_part, "row_parts": row_parts} if row_part is not None else {}),
            }
        )
        if len(chunks) > MAX_CHUNKS_PER_DOCUMENT:
            raise ValueError(f"document exceeds the {MAX_CHUNKS_PER_DOCUMENT} chunk limit")

    # 表头单独就超预算：切成多片各自成块，且后续正文不再带表头前缀
    if header and len(header) + 1 >= max_chars:
        header_parts = [header[index : index + max_chars] for index in range(0, len(header), max_chars)] or [""]
        for part_index, part in enumerate(header_parts):
            append_chunk(
                part,
                row_start=1,
                row_end=1,
                chunk_type="csv_header_part",
                row_part=part_index + 1,
                row_parts=len(header_parts),
                # 切开的片里若不含换行，说明是单物理行，行号仍可精确
                precise_lines="\n" not in part,
            )
        header_prefix = ""
    else:
        header_prefix = header

    # 只有表头没有数据行
    if not rows[1:]:
        if not chunks:
            append_chunk(header, row_start=1, row_end=1, precise_lines="\n" not in header)
        return chunks

    # +1 是表头与第一行数据之间的换行
    prefix_size = len(header_prefix) + (1 if header_prefix else 0)
    row_budget = max_chars - prefix_size
    # records 把每行展开成 (文本片, 行号, 片号, 总片数, 是否单物理行)
    records: list[tuple[str, int, int | None, int | None, bool]] = []
    for row_number, row in enumerate(rows[1:], start=2):
        rendered = render(row)
        parts = [rendered[index : index + row_budget] for index in range(0, len(rendered), row_budget)] or [""]
        for part_index, part in enumerate(parts):
            records.append((part, row_number, part_index + 1 if len(parts) > 1 else None, len(parts) if len(parts) > 1 else None, "\n" not in rendered))

    # 贪心装箱：一个 chunk 尽量多装几行，装不下就换下一块
    start = 0
    while start < len(records):
        selected: list[tuple[str, int, int | None, int | None, bool]] = []
        index = start
        while index < len(records):
            candidate_rows = [item[0] for item in selected]
            candidate_rows.append(records[index][0])
            candidate = "\n".join([header_prefix, *candidate_rows]) if header_prefix else "\n".join(candidate_rows)
            # selected 非空才 break：保证至少装一条，否则单条超预算时会死循环
            if selected and len(candidate) > max_chars:
                break
            selected.append(records[index])
            index += 1
        selected_text = [item[0] for item in selected]
        content = "\n".join([header_prefix, *selected_text]) if header_prefix else "\n".join(selected_text)
        # 一块里可能装多行，行号取范围的两端
        row_start = min(item[1] for item in selected)
        row_end = max(item[1] for item in selected)
        # 只有「这一块恰好是一片行片段」时才能标 row_part；多行混装无法表达
        row_part = selected[0][2] if len(selected) == 1 else None
        row_parts = selected[0][3] if len(selected) == 1 else None
        append_chunk(
            content,
            row_start=row_start,
            row_end=row_end,
            row_part=row_part,
            row_parts=row_parts,
            # 全部由单物理行组成时，行号才可信
            precise_lines=all(item[4] for item in selected),
        )
        if index >= len(records):
            break
        if overlap <= 0:
            start = index
            continue
        # 计算重叠：从本块尾部倒着数，累计到 overlap 字符为止，
        # 数进去几行就回退几行（overlap 以整行为单位，不切半行）
        size = 0
        overlap_records = 0
        for record in reversed(selected):
            record_size = len(record[0])
            # 已经数了至少一行、再加一行就超预算 → 停（保证至少重叠一行）
            if overlap_records and size + record_size > overlap:
                break
            size += record_size
            overlap_records += 1
            if size >= overlap:
                break
        # +1 兜底：即使 overlap_records 等于已消费行数，也要保证 start 前进
        start = max(start + 1, index - overlap_records)
    if any(len(chunk["content"]) > max_chars for chunk in chunks):
        raise AssertionError("CSV chunk content exceeded max_chars")
    return chunks


def chunk_document(
    text: str,
    source_path: str = "document.txt",
    max_chars: int = 1600,
    overlap: int = 160,
) -> list[dict[str, Any]]:
    """文档切块总入口：按文件后缀分派到对应的解析器。

    分派表（source_path 的后缀决定走哪条路）：
    - .csv                    → _csv_chunks（按行 + 重复表头）
    - Markdown 系列/PDF/DOCX  → _markdown_blocks + _pack_blocks
    - .json                   → _json_blocks + _pack_blocks
    - 其他（.txt / 无后缀等）  → _plain_chunks（滑动窗口）

    默认 max_chars=1600 / overlap=160 是经验值：1600 字符约 400~800 token，
    能装下 2~4 个自然段，既保证单块语义完整，又不至于让向量过于笼统；
    overlap 取 1/10，足够衔接跨块的句子。

    返回空列表的两种情况：空文本、纯空白文本——都不必生成任何 chunk。

    Split a document on structural boundaries and return retrieval metadata.
    """

    _validate_limits(max_chars, overlap)
    if not text:
        return []
    normalized = _normalize_text(text)
    # 全空白（只有空格/换行）的文档没有检索价值，直接返回空
    if not normalized.strip():
        return []

    suffix = Path(source_path).suffix.lower()
    if suffix == ".csv":
        return _csv_chunks(normalized, source_path, max_chars, overlap)
    if suffix in MARKDOWN_SUFFIXES:
        blocks = _markdown_blocks(normalized)
        chunk_type = "markdown_section"
    elif suffix == ".json":
        blocks = _json_blocks(normalized, max_chars)
        chunk_type = "json_value"
    else:
        # 注意这里直接 return，不再走 _pack_blocks（纯文本没有 block 结构）
        return _plain_chunks(
            normalized,
            source_path=source_path,
            max_chars=max_chars,
            overlap=overlap,
        )
    chunks = _pack_blocks(
        blocks,
        source_path=source_path,
        max_chars=max_chars,
        overlap=overlap,
        chunk_type=chunk_type,
    )
    if len(chunks) > MAX_CHUNKS_PER_DOCUMENT:
        raise ValueError(f"document exceeds the {MAX_CHUNKS_PER_DOCUMENT} chunk limit")
    return chunks


def chunk_text(text: str, max_chars: int = 1600, overlap: int = 160) -> list[str]:
    """向后兼容的纯文本接口：只返回正文列表，丢掉全部元数据。

    老代码只要字符串列表；新代码应该直接用 chunk_document 拿完整结构。
    这里固定用 "document.txt" 作为 source_path，保证走纯文本分支。

    Backward-compatible plain-text API used by older indexing paths.
    """

    return [
        chunk["content"]
        for chunk in chunk_document(text, "document.txt", max_chars=max_chars, overlap=overlap)
    ]


def _line_windows(
    lines: list[str],
    start_line: int,
    end_line: int,
    max_chars: int,
    overlap_lines: int = 6,
) -> list[tuple[str, int, int]]:
    """在指定行区间内做「按行」的滑动窗口切分，返回 (内容, 起行, 止行) 列表。

    这是代码切块的**最后一道兜底**：当某个符号（函数/类）的完整文本超过
    max_chars 时，仍然必须按长度切开，否则整个符号无法成为一个 chunk。

    与 _plain_chunks 的差异：
    - 以「行」为单位而不是字符，保证不会把一行代码从中间切断；
    - 单行本身超过 max_chars 时才按字符硬切（压缩后的 JS 常见）；
    - overlap_lines 默认 6 行，让前后片段有重叠，避免跨片段的理解断裂。

    参数是 1 基行号，内部转 0 基下标（start_line - 1）；stop 用 min 夹到
    实际行数，防止 end_line 越界（AST 的 end_lineno 有时会比实际行数大）。
    """
    output: list[tuple[str, int, int]] = []
    cursor = max(0, start_line - 1)
    stop = min(len(lines), end_line)
    while cursor < stop:
        # 单行超长：按字符切，每片都记同一行号（因为都来自这一行）
        if len(lines[cursor]) > max_chars:
            for offset in range(0, len(lines[cursor]), max_chars):
                output.append((lines[cursor][offset : offset + max_chars], cursor + 1, cursor + 1))
            cursor += 1
            continue
        size = 0
        end = cursor
        while end < stop:
            # 除首行外每行多算 1 个换行符，长度估算才准
            line_size = len(lines[end]) + (1 if end > cursor else 0)
            # end > cursor 才 break：保证至少装一行，否则空行会卡死
            if end > cursor and size + line_size > max_chars:
                break
            size += line_size
            end += 1
        if end > cursor:
            # strip("\n") 而不是 strip()：保留行首缩进，代码缩进有意义
            content = "\n".join(lines[cursor:end]).strip("\n")
            if content:
                output.append((content, cursor + 1, end))
        if end >= stop:
            break
        # +1 兜底防死循环（overlap_lines 接近窗口大小时）
        cursor = max(cursor + 1, end - overlap_lines)
    return output


def _python_regions(text: str, max_chars: int) -> list[tuple[int, int, str, str]]:
    """用 AST 把 Python 源码切成「符号区间」：(起行, 止行, 符号名, 类型)。

    比正则可靠得多：能正确处理装饰器、嵌套定义、多行字符串里的假 def。
    AST 解析失败（语法错误、文件是 py2、递归过深）时返回空列表，
    由调用方降级成「整文件一个 module 区间」。

    几个关键处理：
    - 符号名用点号连接父子（Class.method），最终写进 metadata.symbol，
      让检索结果能直接显示「这是哪个类的方法」。
    - 装饰器要算进函数起始行：node.lineno 指向 def 行，不含 @decorator，
      若不带装饰器，切片会丢掉 @app.get(...) 这类关键信息。
    - 超长类会被拆成「类头 + 每个方法一个区间」：因为类整体放不进预算时，
      按方法切是最合理的语义单位；类头（含 docstring 与属性）单独成一块。
    - 顶层定义之间的空隙（import、模块级常量）归入 "module" 区间，
      保证源码没有一行被漏掉。
    """
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return []
    lines = text.splitlines()
    regions: list[tuple[int, int, str, str]] = []
    cursor = 1

    def node_start(node: ast.AST) -> int:
        """节点真正开始的行号：取 def/class 行与所有装饰器行的最小值。"""
        decorators = getattr(node, "decorator_list", []) or []
        positions = [getattr(node, "lineno", 1), *(getattr(item, "lineno", 1) for item in decorators)]
        return min(positions)

    def append_node(node: ast.AST, parent: str | None = None) -> None:
        """把一个顶层定义转成区间；超长的类再展开成方法级区间。"""
        start = node_start(node)
        end = int(getattr(node, "end_lineno", start) or start)
        name = str(getattr(node, "name", "module"))
        symbol = f"{parent}.{name}" if parent else name
        kind = "class" if isinstance(node, ast.ClassDef) else "function"
        # 只对「超长的类」做方法级拆分，普通函数保持整体（函数本身就是一个好单位）
        if isinstance(node, ast.ClassDef) and len("\n".join(lines[start - 1 : end])) > max_chars:
            methods = [
                item
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            if methods:
                method_cursor = start
                first_method_start = node_start(methods[0])
                # 第一个方法之前的部分（class 声明 + 类级属性 + docstring）单独成块
                if first_method_start > method_cursor:
                    regions.append((method_cursor, first_method_start - 1, symbol, "class_header"))
                for index, method in enumerate(methods):
                    method_start = node_start(method)
                    # 下一个方法的起点减一就是本方法的终点；最后一个方法到类结尾
                    next_start = node_start(methods[index + 1]) if index + 1 < len(methods) else end + 1
                    regions.append((method_start, next_start - 1, f"{symbol}.{method.name}", "method"))
                return
        regions.append((start, end, symbol, kind))

    top_level = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    for node in top_level:
        start = node_start(node)
        end = int(getattr(node, "end_lineno", start) or start)
        # 与上一个定义之间的空隙（import/常量/注释）归入 module 区间
        if start > cursor:
            regions.append((cursor, start - 1, "module", "module"))
        append_node(node)
        cursor = max(cursor, end + 1)
    # 最后一个定义之后到文件末尾
    if cursor <= len(lines):
        regions.append((cursor, len(lines), "module", "module"))
    # 过滤 start > end 的异常区间（AST 行号偶尔会给出反直觉的值）
    return [region for region in regions if region[0] <= region[1]]


def _generic_symbol(line: str, suffix: str) -> tuple[str, str] | None:
    """用正则从单行代码里识别符号定义（非 Python 语言的近似方案）。

    为什么不用真正的解析器：JS/Go/Rust/Java/SQL/Shell 各需要一套 parser 和
    依赖，成本远超收益。这些语言的定义语句都有很强的行首特征，够用的正则
    准确率已经足够——即使偶有误判，最坏结果也只是切块边界不理想，不影响正确性。

    每种语言返回 (符号名, 类型)，类型会写进 metadata.symbol_kind。
    匹配顺序即优先级，命中第一个就返回。
    """
    patterns: list[tuple[re.Pattern[str], str]] = []
    if suffix in {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}:
        patterns = [
            # function / class / interface / type / enum，可带 export default / async
            (
                re.compile(
                    r"^\s*(?:export\s+(?:default\s+)?)?(?:(?:async\s+)?function|class|interface|type|enum)\s+([A-Za-z_$][\w$]*)"
                ),
                "symbol",
            ),
            # const foo = function / const foo = () => / const foo = x =>
            (
                re.compile(
                    r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:function\b|\([^)]*\)\s*=>|[A-Za-z_$][\w$]*\s*=>)"
                ),
                "function",
            ),
        ]
    elif suffix == ".go":
        # func Name / func (r Recv) Name / type Name
        patterns = [(re.compile(r"^\s*(?:func\s+(?:\([^)]*\)\s*)?|type\s+)([A-Za-z_]\w*)"), "symbol")]
    elif suffix == ".rs":
        patterns = [
            (re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:fn|struct|enum|trait|impl)\s+([A-Za-z_]\w*)"), "symbol")
        ]
    elif suffix in {".java", ".kt", ".kts", ".cs", ".cpp", ".c", ".h", ".hpp", ".php", ".rb", ".swift", ".scala"}:
        # 只看类型声明；这些语言的函数定义形态差异太大，正则容易误伤
        patterns = [
            (re.compile(r"^\s*(?:(?:public|private|protected|internal|static|final|abstract|sealed|data|open)\s+)*(?:class|interface|enum|struct|trait|record)\s+([A-Za-z_]\w*)"), "type")
        ]
    elif suffix == ".sql":
        # CREATE [OR REPLACE] FUNCTION/PROCEDURE/TABLE/VIEW
        patterns = [(re.compile(r"^\s*CREATE\s+(?:OR\s+REPLACE\s+)?(?:FUNCTION|PROCEDURE|TABLE|VIEW)\s+([\w.]+)", re.IGNORECASE), "sql_object")]
    elif suffix in {".sh", ".bash", ".zsh", ".ps1"}:
        # name() { ... }  或  function name { ... }
        patterns = [(re.compile(r"^\s*(?:function\s+)?([A-Za-z_]\w*)\s*(?:\(\))?\s*\{"), "function")]
    for pattern, kind in patterns:
        match = pattern.match(line)
        if match:
            return match.group(1), kind
    return None


def _generic_regions(text: str, suffix: str) -> list[tuple[int, int, str, str]]:
    """基于行首符号定义，把非 Python 源码切成区间。

    算法很简单：先扫出所有「符号起始行」，然后用「下一个符号的上一行」
    作为当前符号的结束行。这意味着一个函数的区间会一直延伸到下一个定义之前
    ——含它后面的所有内容（包括下一个定义前的注释）。这种粗糙度是可以接受的，
    因为切块的目的只是让一个 chunk 尽量对应一个语义单位。

    没有任何符号时（如纯配置文件），整个文件作为一个 module 区间。
    """
    lines = text.splitlines()
    starts: list[tuple[int, str, str]] = []
    for line_number, line in enumerate(lines, start=1):
        symbol = _generic_symbol(line, suffix)
        if symbol:
            starts.append((line_number, symbol[0], symbol[1]))
    if not starts:
        return [(1, len(lines), "module", "module")] if lines else []

    regions: list[tuple[int, int, str, str]] = []
    # 第一个符号之前的内容（import/常量/许可证头）归 module
    if starts[0][0] > 1:
        regions.append((1, starts[0][0] - 1, "module", "module"))
    for index, (start, name, kind) in enumerate(starts):
        end = starts[index + 1][0] - 1 if index + 1 < len(starts) else len(lines)
        regions.append((start, end, name, kind))
    return regions


def _language_for_path(path: str) -> str:
    """由文件后缀推断语言标签（去掉点）。无后缀时退回完整文件名（如 Makefile）。"""
    suffix = Path(path).suffix.lower().lstrip(".")
    return suffix or Path(path).name.lower()


def chunk_code_text(text: str, path: str, max_chars: int = 2200) -> list[dict[str, Any]]:
    """代码切块入口：按符号切分，超长符号再按行窗口二次切分。

    输出比文档 chunk 多几个字段：symbol / symbol_kind / language / header。
    header 是「File / Language / Symbol / Lines」四行摘要，会被一起向量化
    ——代码检索里「这个函数在哪个文件的哪个类里」往往比函数体本身更关键。

    流程：
    1. 文档类后缀（md/pdf/docx/json/csv/txt）不按符号切，直接转交 chunk_document，
       只是把结果补上 symbol 等字段；
    2. 其余按语言取区间（Python 走 AST，其他走正则）；
    3. 每个符号区间用 _line_windows 切成不超过 max_chars 的片段，
       片段序号写成 child_index，父 ID 由「路径 + 符号 + 同名序号」决定。

    symbol_occurrences 的存在是因为同一文件里可能有同名函数（不同作用域、
    或 if TYPE_CHECKING 分支里重复定义），加上序号才能保证 parent_id 唯一。

    Split source on language symbols, with parser and line-window fallbacks.
    """

    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.strip():
        return []
    suffix = Path(path).suffix.lower()
    language = _language_for_path(path)

    # 文档类文件：复用文档切块，然后补齐代码侧字段，让下游处理逻辑统一
    if suffix in MARKDOWN_SUFFIXES | STRUCTURED_DATA_SUFFIXES | {".txt"}:
        # 文档重叠取 min(160, max_chars//10)：预算小时按比例缩小，避免重叠吃掉半个 chunk
        document_chunks = chunk_document(normalized, path, max_chars=max_chars, overlap=max(0, min(160, max_chars // 10)))
        output: list[dict[str, Any]] = []
        for chunk in document_chunks:
            # 文档没有「符号」概念，用章节标题代替，退化到 "document"
            symbol = chunk.get("section_title") or "document"
            header = f"File: {path}\nSection: {symbol}\nLines: {chunk['start_line']}-{chunk['end_line']}"
            output.append({**chunk, "symbol": symbol, "symbol_kind": chunk["chunk_type"], "language": language, "header": header})
        return output

    lines = normalized.splitlines()
    regions = _python_regions(normalized, max_chars) if suffix == ".py" else _generic_regions(normalized, suffix)
    # 解析失败或空文件 → 整个文件当一个 module 区间，保证内容不丢
    if not regions:
        regions = [(1, len(lines), "module", "module")]

    chunks: list[dict[str, Any]] = []
    symbol_occurrences: dict[str, int] = {}
    for start_line, end_line, symbol, symbol_kind in regions:
        symbol_occurrence = symbol_occurrences.get(symbol, 0)
        symbol_occurrences[symbol] = symbol_occurrence + 1
        parent_id = _stable_id("code-parent", path, symbol, symbol_occurrence)
        for part_index, (content, part_start, part_end) in enumerate(
            _line_windows(lines, start_line, end_line, max_chars=max_chars)
        ):
            # 纯空白片段（如连续空行）不产生向量
            if not content.strip():
                continue
            chunk_id = _stable_id("code", parent_id, part_index, content)
            # 只有第 2 片起才标 part 号，单片时不加后缀，保持 header 干净
            part_suffix = f" (part {part_index + 1})" if part_index else ""
            chunks.append(
                {
                    "content": content,
                    "start_line": part_start,
                    "end_line": part_end,
                    "symbol": symbol,
                    "symbol_kind": symbol_kind,
                    "language": language,
                    # module 区间单独给一个类型，前端可以据此淡化展示
                    "chunk_type": "code_symbol" if symbol != "module" else "code_module",
                    "chunk_id": chunk_id,
                    "parent_id": parent_id,
                    "child_index": part_index,
                    "symbol_occurrence": symbol_occurrence,
                    "header": (
                        f"File: {path}\nLanguage: {language}\nSymbol: {symbol}{part_suffix}\n"
                        f"Lines: {part_start}-{part_end}"
                    ),
                }
            )
    return chunks


def chunk_pr_files(files: list[dict[str, Any]], max_chars: int = 1800) -> list[dict[str, Any]]:
    """把 PR 的文件变更（diff patch）切成可检索的片段。

    输入是 GitHub /pulls/{n}/files 返回的列表，每项含 filename 与 patch。
    与文档切块的关键差别：**overlap=0**。

    为什么 diff 不要重叠：patch 是按 hunk 组织的，重叠会把两个不连续的
    hunk 拼在一起，产生一段「现实中不存在的代码」，既误导模型也让引用无法定位。
    宁可少一点上下文，也不能编造出错误的代码片段。

    patch 为空的文件（二进制、纯重命名、改动过大被 GitHub 省略）会被跳过，
    最终可能返回空列表——这是正常的，不代表出错。
    """
    chunks: list[dict[str, Any]] = []
    for file in files:
        patch = file.get("patch") or ""
        filename = str(file.get("filename") or "patch.diff")
        for index, content in enumerate(chunk_text(patch, max_chars=max_chars, overlap=0)):
            chunks.append(
                {
                    "filename": filename,
                    "chunk_index": index,
                    # chunk_id 由文件名 + 内容决定，不含 index：
                    # 这样同一个 hunk 在 PR 更新后位置变了，ID 仍然稳定
                    "chunk_id": _stable_id("patch", filename, content),
                    "content": content,
                }
            )
    return chunks
