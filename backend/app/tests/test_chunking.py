"""RAG 文档切块（chunking）与文本抽取的测试。

切块是 RAG 索引的地基：切得太碎会丢上下文，切得太大则检索精度下降、
还会超出 embedding 模型的输入上限。这个文件验证四类行为：
1. 通用文本切块遵守 max_chars 上限，并保留段落/行边界与相邻块 overlap；
2. Markdown 切块保留标题层级（section_path）、代码围栏完整性和行号映射；
3. 代码切块按 AST / 正则识别符号（函数、类、导出函数），非法源码要能降级；
4. CSV / JSON / 超长字段 / 压缩炸弹 / 超页 PDF 等极端输入必须有上限保护且不丢数据。
"""

import io
import zipfile

import pytest

from app.services.rag import chunking, document_extraction
from app.services.rag.chunking import chunk_code_text, chunk_document, chunk_text


def test_chunk_text_keeps_paragraphs_and_respects_size_limit() -> None:
    """场景：三段各重复 8 次的段落，用 max_chars=190、overlap=20 切块。

    期望：至少切成 2 块、每块都不超过 190 字，且第一块里仍保留空行（\n\n）段落分隔。
    为什么重要：单纯按字数硬切会把段落拦腰截断，检索出来的片段可读性差、语义不完整。
    """
    text = "第一段说明登录流程。" * 8 + "\n\n" + "第二段说明刷新令牌。" * 8 + "\n\n" + "第三段说明退出流程。" * 8

    chunks = chunk_text(text, max_chars=190, overlap=20)

    assert len(chunks) >= 2
    assert all(len(chunk) <= 190 for chunk in chunks)
    assert "\n\n" in chunks[0]


def test_chunk_text_rejects_invalid_overlap() -> None:
    """场景：overlap 等于 max_chars（相邻块完全重叠，切分将无法前进）。

    期望：抛 ValueError，且异常信息里点名 "overlap"，方便调用方定位参数问题。
    为什么重要：这是个死循环式的参数错误，必须在入口拦掉而不是让切块卡死。
    """
    try:
        chunk_text("content", max_chars=100, overlap=100)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_markdown_chunking_preserves_heading_context_and_code_fences() -> None:
    """场景：一份带一级/二级标题和 python 代码围栏的 Markdown，按 180 字上限切块。

    期望：section_path 能还原成 ["Authentication", "Refresh tokens"] 的层级；
    代码块被单独识别为 markdown_code 类型，围栏 ``` 成对保留（出现 2 次），
    并且 start_line / end_line 自洽。
    为什么重要：标题路径决定模型能否理解片段出处；围栏被切断会让片段变成乱码，
    行号错乱则会让「引用出处」功能指到错误位置。
    """
    text = """# Authentication

The service verifies access tokens before every request.

## Refresh tokens

Refresh tokens are rotated after a successful exchange.

```python
def rotate_token(token: str) -> str:
    return token + "-rotated"
```
"""

    chunks = chunk_document(text, "auth.md", max_chars=180, overlap=20)

    assert chunks
    assert all(len(chunk["content"]) <= 180 for chunk in chunks)
    # section_path 是「标题栈」，二级标题块要能带上一级标题，否则模型不知道 Refresh tokens 属于哪个主题
    assert any(chunk["section_path"] == ["Authentication", "Refresh tokens"] for chunk in chunks)
    # 代码块单独成形，且起止围栏都还在（count == 2 才算完整）
    code_chunk = next(chunk for chunk in chunks if chunk["chunk_type"] == "markdown_code")
    assert code_chunk["content"].count("```") == 2
    assert code_chunk["section_title"] == "Refresh tokens"
    assert code_chunk["start_line"] <= code_chunk["end_line"]


def test_markdown_overlap_starts_at_a_readable_boundary() -> None:
    """场景：max_chars 只有 55，必须切成 2 块，overlap=20。

    期望：第二块错开重叠部分后的正文，要从 "charlie" 或 "delta" 这样的完整单词开始，
    而不是从单词中间截断。
    为什么重要：从半个单词开始的重叠会让检索命中率下降，也会污染 embedding 语义。
    """
    text = "# A\n\nalpha bravo charlie delta echo foxtrot\n\nsecond evidence"

    chunks = chunk_document(text, "overlap.md", max_chars=55, overlap=20)

    assert len(chunks) == 2
    # 去掉重叠前缀后再看正文首词，验证切点落在词边界
    overlap_body = chunks[1]["content"].split("\n\n", 1)[1]
    assert overlap_body.split()[0] in {"charlie", "delta"}


def test_markdown_chunk_ids_are_stable_for_the_same_content() -> None:
    """场景：同一份内容切两次。

    期望：chunk_id 与 parent_id 完全一致（确定性）。
    为什么重要：增量索引靠 id 判断「这块内容是否已存在」，id 不稳定会导致每次同步都全量重建向量库。
    """
    text = "# Deploy\n\nRun the migration before restarting the API."

    first = chunk_document(text, "runbook.md", max_chars=200, overlap=20)
    second = chunk_document(text, "runbook.md", max_chars=200, overlap=20)

    assert [chunk["chunk_id"] for chunk in first] == [chunk["chunk_id"] for chunk in second]
    assert [chunk["parent_id"] for chunk in first] == [chunk["parent_id"] for chunk in second]


def test_python_chunking_uses_ast_symbol_boundaries() -> None:
    """场景：一段含模块级 import、一个函数、一个带方法的类。

    期望：切块按 AST 符号边界产出 module / load_settings / TokenService；
    函数块的 symbol_kind 是 "function"，start_line 精确指向 def 那一行（第 3 行），
    并在 header 里带上文件路径，方便检索结果标注出处。
    为什么重要：按符号切才能保证「一个问题对应一个完整函数」，而不是半截代码。
    """
    source = """import os

def load_settings() -> dict:
    return {"environment": os.getenv("ENV", "dev")}

class TokenService:
    def issue(self, user_id: str) -> str:
        return f"token:{user_id}"
"""

    chunks = chunk_code_text(source, "app/token_service.py", max_chars=220)

    assert {chunk["symbol"] for chunk in chunks} >= {"module", "load_settings", "TokenService"}
    function = next(chunk for chunk in chunks if chunk["symbol"] == "load_settings")
    assert function["symbol_kind"] == "function"
    assert function["start_line"] == 3
    assert function["header"].startswith("File: app/token_service.py")


def test_typescript_chunking_recognizes_exported_functions_and_arrow_functions() -> None:
    """场景：TS 文件里有 export async function 和 export const 箭头函数两种写法。

    期望：两种符号都能被识别（loadUser / saveUser）且被标记为 module 之外的独立块，
    所有块的 language 统一是 "ts"。
    为什么重要：前端代码大量使用箭头函数，只认 function 关键字会漏切一半代码。
    """
    source = """import { api } from './api';

export async function loadUser(id: string) {
  return api.get(`/users/${id}`);
}

export const saveUser = async (id: string) => {
  return api.post(`/users/${id}`);
};
"""

    chunks = chunk_code_text(source, "src/users.ts", max_chars=220)

    assert {chunk["symbol"] for chunk in chunks} >= {"module", "loadUser", "saveUser"}
    assert all(chunk["language"] == "ts" for chunk in chunks)


def test_invalid_python_falls_back_without_losing_content() -> None:
    """场景：语法错误的 Python 源码（AST 解析必然失败）。

    期望：不抛异常，降级成整文件（symbol=module）单块切分，且原文 "def broken" 仍在内容里。
    为什么重要：仓库里存在半成品/生成代码，索引流水线遇到坏文件不能整体崩掉，只能丢结构化信息、不能丢内容。
    """
    source = "def broken(:\n    return 1\n"

    chunks = chunk_code_text(source, "broken.py", max_chars=80)

    assert chunks
    assert "def broken" in "\n".join(chunk["content"] for chunk in chunks)
    assert chunks[0]["symbol"] == "module"


def test_csv_chunking_repeats_header_and_tracks_row_ranges() -> None:
    """场景：5 行 CSV，max_chars=25 逼它切成多块。

    期望：每块都重复表头 "id,name"（模型才能看懂每列含义）、chunk_type 统一为 csv_rows，
    且 row_start <= row_end 记录行号区间供溯源。
    为什么重要：CSV 丢掉表头后就是一堆无意义的值；行号区间是「引用到具体数据行」的前提。
    """
    text = "id,name\n1,Alice\n2,Bob\n3,Carol\n4,David"

    chunks = chunk_document(text, "users.csv", max_chars=25, overlap=5)

    assert len(chunks) >= 2
    assert all(chunk["content"].splitlines()[0] == "id,name" for chunk in chunks)
    assert all(chunk["chunk_type"] == "csv_rows" for chunk in chunks)
    assert all(chunk["row_start"] <= chunk["row_end"] for chunk in chunks)


def test_small_json_manifest_stays_in_one_chunk() -> None:
    """场景：一个很小的 package.json。

    期望：只切成 1 块，json_path 为根 "$"，且内容被美化（"name": "demo" 带空格）。
    为什么重要：小 JSON 拆开会破坏结构；json_path 让小片段仍能表达「这是 dependencies 下的键」。
    """
    chunks = chunk_document('{"name":"demo","version":"1.0.0"}', "package.json", max_chars=200, overlap=20)

    assert len(chunks) == 1
    assert chunks[0]["json_path"] == "$"
    assert '"name": "demo"' in chunks[0]["content"]


def test_plain_text_overlap_is_present_and_line_numbers_keep_leading_blanks() -> None:
    """场景：纯文本开头有两个空行，后面是 100 个三位数字，max_chars=60、overlap=12。

    期望：切成多块；第一块的 start_line 是 3（说明空行被计入行号，没有在清洗时被吃掉）；
    相邻块严格满足「前一块末尾 12 字符 == 后一块开头 12 字符」。
    为什么重要：行号偏移会让所有引用定位漂移；overlap 不精确则跨块语义会断裂。
    """
    text = "\n\n" + "".join(f"{index:03d}" for index in range(100))

    chunks = chunk_document(text, "notes.txt", max_chars=60, overlap=12)

    assert len(chunks) > 2
    assert chunks[0]["start_line"] == 3
    assert all(previous["content"][-12:] == current["content"][:12] for previous, current in zip(chunks, chunks[1:]))


def test_csv_long_rows_are_split_without_data_loss() -> None:
    """场景：第 2 行有 80 个 x 的超长字段，max_chars 只给 20。

    期望：单行被拆块后每块仍 <= 20 字符，所有块里 x 的总数正好 80（一个字符不丢），
    被拆的碎片用 row_part 标记且都归属第 2 行。
    为什么重要：日志/数据导出里常见超长单行，拆块时丢字符就等于数据损坏。
    """
    text = "id,data\n1," + "x" * 80 + "\n2,ok"

    chunks = chunk_document(text, "long.csv", max_chars=20, overlap=2)

    assert all(len(chunk["content"]) <= 20 for chunk in chunks)
    assert sum(chunk["content"].count("x") for chunk in chunks) == 80
    assert {chunk["row_start"] for chunk in chunks if chunk.get("row_part")} == {2}


def test_csv_fields_larger_than_the_stdlib_default_remain_structured() -> None:
    """场景：单个字段 140000 字符，超过 csv 模块默认的字段长度上限（131072）。

    期望：切块不报错，去掉每块首行表头后拼起来与原始行完全相等，chunk_type 仍是 csv_rows。
    为什么重要：不显式调大 csv.field_size_limit，Python 标准库会直接抛
    "field larger than field limit"，真实仓库里的压缩数据/长文本字段很容易触发。
    """
    rendered_row = "1," + "x" * 140_000

    chunks = chunk_document(f"id,data\n{rendered_row}", "large-field.csv", max_chars=5000, overlap=0)

    reconstructed = "".join(chunk["content"].split("\n", 1)[1] for chunk in chunks)
    assert reconstructed == rendered_row
    assert all(chunk["chunk_type"] == "csv_rows" for chunk in chunks)


def test_csv_oversized_header_is_lossless_and_size_limited() -> None:
    """场景：表头本身就有 64 字符，超过 max_chars=20。

    期望：表头被切成 csv_header_part 碎片，拼回来与原始表头完全一致；
    数据行 "value" 仍能单独成块；所有块长度都 <= 20。
    为什么重要：表头超长时如果直接丢弃，后续块就失去了列语义；如果直接超限，又会撑爆 embedding 输入。
    """
    header = "column_" * 8

    chunks = chunk_document(f"{header}\nvalue", "header.csv", max_chars=20, overlap=2)

    header_chunks = [chunk for chunk in chunks if chunk["chunk_type"] == "csv_header_part"]
    assert "".join(chunk["content"] for chunk in header_chunks) == header
    assert any("value" in chunk["content"] for chunk in chunks)
    assert all(len(chunk["content"]) <= 20 for chunk in chunks)


def test_repeated_and_empty_markdown_sections_remain_distinct() -> None:
    """场景：同一份文档里出现两个同名二级标题 "Auth"，外加一个没有正文的空标题 "Empty"。

    期望：两个 Auth 各自成块（共 2 块）但 parent_id 不同；空标题也能产出块；
    所有 chunk_id 全局唯一。
    为什么重要：重名标题在长文档里很常见，若按标题名做主键会互相覆盖、丢内容。
    """
    text = "# API\n\n## Auth\n\nfirst\n\n## Auth\n\nsecond\n\n### Empty"

    chunks = chunk_document(text, "api.md", max_chars=100, overlap=10)

    auth_chunks = [chunk for chunk in chunks if chunk["section_title"] == "Auth"]
    assert len(auth_chunks) == 2
    assert auth_chunks[0]["parent_id"] != auth_chunks[1]["parent_id"]
    assert any(chunk["section_title"] == "Empty" for chunk in chunks)
    assert len({chunk["chunk_id"] for chunk in chunks}) == len(chunks)


def test_long_markdown_prefix_and_fence_obey_limit_with_exact_body_lines() -> None:
    """场景：200 个 H 的超长一级标题 + 带超长 info string 的代码围栏，max_chars 只有 30。

    期望：所有块 <= 30 字符；含 line2=2222 的块，其 start_line/end_line 都精确等于 5
    （不因为前面塞了标题前缀就把行号算错），并且仍记得围栏起始行是 3。
    为什么重要：为了塞进「标题上下文前缀」很容易把真实行号搞乱，
    而引用出处（file:line）错一行，用户点过去就找不到代码。
    """
    fence = "`" * 3
    text = "\n".join(
        [
            "# " + "H" * 200,
            "",
            fence + "python-with-a-very-long-info-string",
            "line1=1111",
            "line2=2222",
            "line3=3333",
            fence,
        ]
    )

    chunks = chunk_document(text, "fence.md", max_chars=30, overlap=2)

    assert all(len(chunk["content"]) <= 30 for chunk in chunks)
    line2_chunk = next(chunk for chunk in chunks if "line2=2222" in chunk["content"])
    assert line2_chunk["start_line"] == 5
    assert line2_chunk["end_line"] == 5
    assert line2_chunk["fence_start_line"] == 3


def test_tiny_markdown_budget_falls_back_without_zero_width_fence_split() -> None:
    """场景：max_chars=8，比 "```" 加上正文还紧。

    期望：仍然有产出且每块 <= 8 字符，不会出现「宽度为 0」的切分（否则会死循环）。
    为什么重要：极端小预算必须走降级分支，而不是卡死索引任务。
    """
    text = "# A\n\n```\nvalue\n```"

    chunks = chunk_document(text, "tiny.md", max_chars=8, overlap=1)

    assert chunks
    assert all(len(chunk["content"]) <= 8 for chunk in chunks)


def test_duplicate_code_symbols_receive_unique_stable_ids() -> None:
    """场景：同一个文件里定义了两个同名的 same() 函数（后者覆盖前者，但源码里都在）。

    期望：两个同名块得到不同的 chunk_id（靠位置区分），且两次切分结果完全一致。
    为什么重要：同名符号（重载/条件定义/复制粘贴）很常见，id 撞车会导致向量库里一块把另一块覆盖掉。
    """
    source = "def same():\n    pass\n\ndef same():\n    pass"

    first = chunk_code_text(source, "same.py", max_chars=100)
    second = chunk_code_text(source, "same.py", max_chars=100)
    same_chunks = [chunk for chunk in first if chunk["symbol"] == "same"]

    assert len(same_chunks) == 2
    assert same_chunks[0]["chunk_id"] != same_chunks[1]["chunk_id"]
    assert [chunk["chunk_id"] for chunk in first] == [chunk["chunk_id"] for chunk in second]


def test_large_json_does_not_claim_invented_source_lines() -> None:
    """场景：JSON 被重新排版后切块（原文与美化后文本行号不再一一对应）。

    期望：确实切成多块，但 start_line / end_line 都为 None，即「不谎报行号」。
    为什么重要：JSON 切块前会做缩进美化，算出来的行号与源文件不对应；
    报一个假行号比报 None 更糟——用户跳过去会看到完全不相关的内容。
    """
    text = '{\n  "first": "' + "a" * 40 + '",\n  "second": "' + "b" * 40 + '"\n}'

    chunks = chunk_document(text, "data.json", max_chars=30, overlap=3)

    assert len(chunks) > 1
    assert all(chunk["start_line"] is None and chunk["end_line"] is None for chunk in chunks)


def test_text_extraction_rejects_binary_and_excessive_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """场景：把 MAX_EXTRACTED_CHARS 压到 8，然后分别喂「超长文本」和「含 \x00 的二进制」。

    期望：超长文本抛 "character limit"，二进制抛 "binary"。
    为什么重要：用户随手把 .zip/.png 当文档上传很常见，二进制强行解码会变成
    满屏乱码并浪费 embedding 额度；字符上限则防止一份超大日志拖垮索引任务。
    """
    # 调小上限常量，用几字节的样本就能触发分支，不用真的构造几十 MB 数据
    monkeypatch.setattr(document_extraction, "MAX_EXTRACTED_CHARS", 8)

    with pytest.raises(ValueError, match="character limit"):
        document_extraction.decode_text(b"more than eight characters")
    with pytest.raises(ValueError, match="binary"):
        document_extraction.decode_text(b"abc\x00def")


def test_docx_archive_expansion_is_checked_before_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    """场景：docx 本质是 zip。这里造假一个「32 字节内容」的压缩包，
    并把 MAX_DOCX_UNCOMPRESSED_BYTES 压到 16，模拟 zip 炸弹。

    期望：在真正解析 XML 之前就抛 "expanded size"。
    为什么重要：解压炸弹可以把几百 KB 的 docx 解成几个 GB，必须按「声明解压后大小」先拦截。
    """
    buffer = io.BytesIO()
    # 用 DEFLATED 压缩：重复字符会被压得很小，解压后却膨胀，正是炸弹的典型形态
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", "x" * 32)
    monkeypatch.setattr(document_extraction, "MAX_DOCX_UNCOMPRESSED_BYTES", 16)

    with pytest.raises(ValueError, match="expanded size"):
        document_extraction.extract_document_text("bomb.docx", buffer.getvalue())


def test_pdf_page_limit_is_checked_before_page_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    """场景：现场生成一个 2 页 PDF，把 MAX_PDF_PAGES 设为 1。

    期望：抛 "page limit" 而不是慢慢把两页都解析完。
    为什么重要：PDF 解析贵且慢，页数上限要在抽取之前判，否则一份千页手册能把 worker 占满。
    """
    # 延迟导入：pypdf 只在需要时加载，避免整个测试模块在 import 阶段就依赖它
    from pypdf import PdfWriter

    buffer = io.BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.add_blank_page(width=100, height=100)
    writer.write(buffer)
    monkeypatch.setattr(document_extraction, "MAX_PDF_PAGES", 1)

    with pytest.raises(ValueError, match="page limit"):
        document_extraction.extract_document_text("many-pages.pdf", buffer.getvalue())


def test_document_chunk_count_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """场景：把 MAX_CHUNKS_PER_DOCUMENT 压到 2，用 "abcdef"、max_chars=2 去切（必然超过 2 块）。

    期望：抛 "chunk limit"。
    为什么重要：切块数直接决定向量条数与 embedding 花费，
    没有上限时一份畸形文档就能让一次同步写入几十万条向量。
    """
    monkeypatch.setattr(chunking, "MAX_CHUNKS_PER_DOCUMENT", 2)

    with pytest.raises(ValueError, match="chunk limit"):
        chunking.chunk_document("abcdef", "tiny.txt", max_chars=2, overlap=0)
