"""文档抽取：把上传的二进制文件转成纯文本（切块之前的一步）。

位置：**上传 → 抽取成本文 → 切块 → 向量化 → …**

本模块只负责「解码成 UTF-8 文本」，不做切块。它把 PDF / DOCX 都转成
**带 Markdown 结构标记**的文本（PDF 插 `<!-- page: N -->`，DOCX 把标题样式
转成 `#`、表格转成 Markdown 表格），这样下游 chunking 就能用同一套
Markdown 解析器处理三种来源，不需要为每种格式各写一套切块逻辑。

整份文件贯穿一条主题：**所有输入都是不可信的外部文件**。
因此每一个入口都有明确的资源上限（体积、页数、条目数、解压后大小、字符数），
并且全部选择「超限就报错」而不是「截断后继续」——截断会让用户以为文件
已经完整入库，实际上后半部分永远检索不到。
"""

from __future__ import annotations

import re
import zipfile
from io import BytesIO
from pathlib import Path


# 原始文件体积上限 12MB：覆盖绝大多数文档，同时防止把大文件读进内存
MAX_DOCUMENT_BYTES = 12 * 1024 * 1024
# 抽取后的字符上限 400 万：约 2000 个 chunk 的量级，避免单文件耗尽索引资源
MAX_EXTRACTED_CHARS = 4_000_000
# PDF 页数上限：pypdf 逐页抽取是纯 Python 实现，页数多时耗时会线性膨胀
MAX_PDF_PAGES = 500
# DOCX（本质是 zip）内部条目数上限
MAX_DOCX_ENTRIES = 10_000
# DOCX 解压后总大小上限 64MB：防 zip 炸弹（几 KB 的压缩包解压出几十 GB）
MAX_DOCX_UNCOMPRESSED_BYTES = 64 * 1024 * 1024


def _ensure_text_limit(text: str) -> str:
    """统一出口的长度校验，所有抽取函数返回前都必须过这一关。"""
    if len(text) > MAX_EXTRACTED_CHARS:
        raise ValueError(f"extracted text exceeds the {MAX_EXTRACTED_CHARS} character limit")
    return text


def decode_text(data: bytes) -> str:
    """把任意字节解码成文本，按常见中文环境编码依次尝试。

    顺序是刻意的：
    - utf-8-sig：带 BOM 的 UTF-8（Windows 记事本常有），不先处理会残留 \\ufeff；
    - utf-8：标准 UTF-8；
    - gb18030：兼容 GBK/GB2312，中文 Windows 的老文件多是这个；
    - latin-1：**永远能成功**的解码器（单字节一一映射），放在最后兜底，
      保证任何字节都不会抛异常，代价是乱码。

    开头的 NUL 字节检测用来识别二进制文件：文本文件前 8KB 里几乎不可能出现
    \\x00，而图片/压缩包/可执行文件必然有。若不拦，latin-1 会把它们"成功"
    解码成一堆乱码并入库，污染检索结果。
    """
    if b"\x00" in data[:8192]:
        raise ValueError("binary content cannot be indexed as text")
    for encoding in ["utf-8-sig", "utf-8", "gb18030", "latin-1"]:
        try:
            return _ensure_text_limit(data.decode(encoding))
        except UnicodeDecodeError:
            continue
    return ""


def _extract_pdf(data: bytes) -> str:
    """抽取 PDF 文本，每页前面插入页码标记。

    页码标记 `<!-- page: N -->` 有两个作用：
    1. 下游 _markdown_blocks 能识别它并写入 block.page，最终进 chunk 的 metadata，
       回答时可以精确引用「第 3 页」；
    2. 页码变化会触发切块分隔，chunk 不会跨页。

    延迟 import pypdf：它不是所有部署都装，放在函数内可以让不处理 PDF 的
    环境照常启动（真正用到时才会 ImportError）。

    空白页（扫描页、纯图片页）会被跳过——没有文字就没有可抽取的内容，
    这类 PDF 需要 OCR，不在本模块能力范围内。

    extracted_chars 是**累加式**预算检查：逐页累计，一旦超限立即中断，
    而不是等全部抽完再判断——这样恶意超大 PDF 不会先把内存吃满。
    """
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(data))
    if len(reader.pages) > MAX_PDF_PAGES:
        raise ValueError(f"PDF exceeds the {MAX_PDF_PAGES} page limit")
    pages: list[str] = []
    extracted_chars = 0
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            content = f"<!-- page: {page_number} -->\n{text}"
            # +2 是页与页之间的 "\n\n" 分隔符（第一页不加）
            extracted_chars += len(content) + (2 if pages else 0)
            if extracted_chars > MAX_EXTRACTED_CHARS:
                raise ValueError(f"PDF text exceeds the {MAX_EXTRACTED_CHARS} character limit")
            pages.append(content)
    return _ensure_text_limit("\n\n".join(pages))


def _paragraph_as_markdown(paragraph) -> str:
    """把 Word 段落转成等价的 Markdown 行。

    只能靠样式名推断语义——docx 的段落本身不带「这是标题」的标记，
    信息藏在 style.name 里：
    - "Heading 1"~"Heading 6" → 对应数量的 #；层级夹在 1~6（Markdown 上限）
    - "List Bullet" → "- "
    - "List Number" → "1. "（统一写 1.，具体序号由渲染器处理）
    - 其他 → 原样返回

    这种基于样式名的启发式在多语言 Word 里可能失效（中文版样式名可能是
    "标题 1"），失效的后果只是丢失标题层级、退化成普通段落，不会出错。
    """
    text = (paragraph.text or "").strip()
    if not text:
        return ""
    style_name = str(getattr(getattr(paragraph, "style", None), "name", "") or "")
    heading = re.match(r"heading\s+(\d+)", style_name, re.IGNORECASE)
    if heading:
        level = min(6, max(1, int(heading.group(1))))
        return f"{'#' * level} {text}"
    lowered = style_name.lower()
    if "list bullet" in lowered:
        return f"- {text}"
    if "list number" in lowered:
        return f"1. {text}"
    return text


def _table_as_markdown(table) -> str:
    """把 Word 表格转成 Markdown 表格。

    两个必要处理：
    1. **补齐列数**：Word 允许合并单元格，导致各行列数不一致；Markdown 表格
       要求每行列数相同，所以按最大列数用空串补齐。
    2. **转义竖线**：单元格内容里的 | 会破坏表格结构，必须先转成 \\|。
    另外把单元格内的换行替换成空格——Markdown 表格里不能有裸换行。

    第一行当表头，第二行插入分隔行（---），符合 GFM 表格语法。
    """
    rows = [[cell.text.strip().replace("\n", " ") for cell in row.cells] for row in table.rows]
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    normalized = [row + [""] * (width - len(row)) for row in rows]

    def render(row: list[str]) -> str:
        return "| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |"

    return "\n".join([render(normalized[0]), render(["---"] * width), *(render(row) for row in normalized[1:])])


def _extract_docx(data: bytes) -> str:
    """抽取 DOCX 文本，按文档流顺序把段落与表格转成 Markdown。

    为什么要先做一轮 zip 检查（在真正用 python-docx 打开之前）：
    DOCX 就是一个 zip 包，python-docx 打开时会解压并解析整个 XML。
    如果是个 zip 炸弹，等到解析时内存已经爆了。所以先用标准库 zipfile
    读目录信息（不解压），检查条目数与解压后总大小，两个门槛都过才继续。

    还会直接拒绝加密文件：zip 的 flag_bits 最低位表示加密，加密的 docx
    无法解析，早报错比后面抛一堆看不懂的异常好。

    遍历用 document.element.body.iterchildren() 而不是 document.paragraphs：
    后者只返回段落，会丢掉表格，而且会打乱「段落-表格-段落」的原始顺序。
    这里按底层 XML 子元素的真实顺序遍历，段落和表格各自处理。
    """
    try:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_DOCX_ENTRIES:
                raise ValueError(f"DOCX exceeds the {MAX_DOCX_ENTRIES} entry limit")
            # file_size 是 zip 头里声明的解压后大小，读目录即可拿到，不需要真正解压
            uncompressed_size = sum(info.file_size for info in infos)
            if uncompressed_size > MAX_DOCX_UNCOMPRESSED_BYTES:
                raise ValueError(
                    f"DOCX expanded size exceeds the {MAX_DOCX_UNCOMPRESSED_BYTES} byte limit"
                )
            # 0x1 位 = 该条目被加密
            if any(info.flag_bits & 0x1 for info in infos):
                raise ValueError("encrypted DOCX files are not supported")
    except zipfile.BadZipFile as exc:
        # 统一转成 ValueError：调用方只需要处理「文件不合法」这一种语义，
        # 不用区分是 zip 层还是 docx 层的问题
        raise ValueError("invalid DOCX archive") from exc

    import docx
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = docx.Document(BytesIO(data))
    blocks: list[str] = []
    extracted_chars = 0
    for child in document.element.body.iterchildren():
        # 判断依据是底层 XML 标签类型，CT_P=段落、CT_Tbl=表格
        if isinstance(child, CT_P):
            content = _paragraph_as_markdown(Paragraph(child, document))
        elif isinstance(child, CT_Tbl):
            content = _table_as_markdown(Table(child, document))
        else:
            # sectPr 等排版元素没有文本，跳过
            content = ""
        if content:
            extracted_chars += len(content) + (2 if blocks else 0)
            if extracted_chars > MAX_EXTRACTED_CHARS:
                raise ValueError(f"DOCX text exceeds the {MAX_EXTRACTED_CHARS} character limit")
            blocks.append(content)
    return _ensure_text_limit("\n\n".join(blocks))


def extract_document_text(filename: str, data: bytes) -> str:
    """按扩展名分派抽取器，未知类型一律当纯文本处理。

    只看后缀不看 magic number：这是有意的简化——真正的类型嗅探会让
    「用户把 .md 改名成 .txt」这种常见情况变复杂，而按后缀处理的最坏结果
    只是走纯文本解码，仍然能用。

    注意这里没有校验 MAX_DOCUMENT_BYTES，体积检查由上传入口负责；
    本函数假定传入的 data 已经在限制之内。
    """
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(data)
    if suffix == ".docx":
        return _extract_docx(data)
    return decode_text(data)
