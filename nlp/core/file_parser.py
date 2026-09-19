# -*- coding: utf-8 -*-
"""
通用文件解析模块
================

将各种格式的文件解析为纯文本，供 LLM 理解或向量数据库入库使用。

支持格式：
- .txt / .md：直接读取
- .pdf：pdfplumber 文字提取 + OCR 降级（扫描版）+ 表格跨页合并
- .docx / .doc：python-docx 段落提取
- .csv：pandas 读取，转自然语言
- .xlsx / .xls：pandas 读取，转自然语言
- .json：直接读取

两类路径，读的时候别混：
1. 纯文本类（txt/md/json）直接读；结构化文档（pdf/docx/csv/xlsx）交给第三方库解析成文本。
2. OCR（光学字符识别：从图片里认出文字）**只服务 PDF**，且只是降级手段：
   扫描版 PDF 没有文字层，得把页渲染成图片再交给 Tesseract（OCR 引擎，外部程序，需另装）。
   **本模块不解析图片文件**（.png/.jpg 不在支持列表里）—— 图片输入是 vqa_agent 那条路，
   不经过这里。

pdfplumber / python-docx / pandas 都是可选依赖：缺失时对应格式返回空字符串并记 error 日志，
不向上抛异常。所以"解析出空文本"有两种可能 —— 文件本身是空的，或者依赖没装。

本模块从 init_vector_db.py 中提取而来，统一前后端的文件解析逻辑。
"""

import os
import re
import sys
import logging
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)


# ========== PDF 相关 ==========

def has_garbled_text(text: str) -> bool:
    """
    检测文本是否为乱码（伪文字层）

    原理：扫描版PDF有时会嵌入一层伪文本，提取出来全是乱码。
    通过统计特殊字符（非中英文、非数字、非常见标点）的占比来判断。

    Args:
        text: 待检测文本

    Returns:
        True=乱码, False=正常文本
    """
    if not text or len(text.strip()) < 10:
        return False

    cleaned = text.strip()
    normal_pattern = re.compile(
        r'[\u4e00-\u9fff'       # 中文
        r'a-zA-Z'               # 英文
        r'0-9'                  # 数字
        r'\s'                   # 空白
        r'，。！？、；：""''（）《》【】'  # 中文标点
        r',.!?;:\'\"()\[\]{}\-+=/\\@#$%&*~`<>]'  # 英文标点
    )
    normal_count = len(normal_pattern.findall(cleaned))
    total_count = len(cleaned)

    if total_count == 0:
        return False

    abnormal_ratio = 1 - (normal_count / total_count)
    return abnormal_ratio > 0.3


def is_scanned_pdf(file_path: str, sample_pages: int = 3) -> bool:
    """
    文件级检测：判断PDF是扫描版还是文字版

    为什么采样而不是逐页全查：大 PDF 逐页解析很慢，而"整份是扫描件还是文字版"
    通常前几页就能看出来。代价是极端情况（如前面几页恰好正常、后面全是扫描页）会判错。

    检测策略（采样前N页）：
    1. 文本层检测：提取文本，长度 < 50 字符视为无文本
    2. 乱码检测：特殊字符占比 > 30% 视为乱码（伪文字层）
    3. 图片占比检测：图片面积 > 页面80% 视为扫描件

    综合判断：超过一半的采样页无有效文本 → 扫描版

    Args:
        file_path: PDF文件路径
        sample_pages: 采样页数（默认前3页）

    Returns:
        True=扫描版, False=文字版
    """
    try:
        import pdfplumber

        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            pages_to_check = min(sample_pages, total_pages)
            scanned_count = 0

            for i in range(pages_to_check):
                page = pdf.pages[i]
                page_text = page.extract_text()

                # 检查1：无文本或文本极少
                if not page_text or len(page_text.strip()) < 50:
                    scanned_count += 1
                    continue

                # 检查2：乱码检测
                if has_garbled_text(page_text):
                    scanned_count += 1
                    continue

                # 检查3：图片占比检测
                images = page.images
                if images:
                    page_area = page.width * page.height
                    img_area = sum(
                        abs(img.get("x1", 0) - img.get("x0", 0)) *
                        abs(img.get("bottom", 0) - img.get("top", 0))
                        for img in images
                    )
                    if page_area > 0 and img_area / page_area > 0.8:
                        scanned_count += 1
                        continue

            is_scanned = scanned_count > pages_to_check / 2
            logger.info(
                f"  PDF类型检测: 采样{pages_to_check}页, "
                f"{scanned_count}页无有效文本 → "
                f"{'扫描版' if is_scanned else '文字版'}"
            )
            return is_scanned

    except Exception as e:
        logger.warning(f"  PDF类型检测失败: {e}，默认按文字版处理")
        return False


def _ocr_page(page, page_num: int, pytesseract) -> str:
    """
    对单个PDF页面执行OCR

    先把页渲染成 300dpi 的图片，再交给 Tesseract 识别（分辨率太低中文识别率会明显掉）。
    异常一律吞掉返回空串 —— 包括"引擎没装"这类环境问题，所以调用方分不清
    "这页真没字"和"OCR 根本用不了"。

    Args:
        page: pdfplumber page对象
        page_num: 页码
        pytesseract: pytesseract模块

    Returns:
        OCR识别的文本
    """
    try:
        img = page.to_image(resolution=300)
        pil_img = img.original
        ocr_text = pytesseract.image_to_string(pil_img, lang='chi_sim+eng')
        if ocr_text.strip():
            logger.info(f"    OCR成功提取 {len(ocr_text)} 字符")
            return ocr_text
        return ""
    except Exception as e:
        logger.warning(f"    OCR失败: {e}")
        return ""


def is_table_continuation(table, prev_header) -> bool:
    """
    判断表格是否是上一页表格的延续（跨页表格检测）

    Args:
        table: 当前表格原始数据
        prev_header: 上一页表格的表头

    Returns:
        是否是跨页延续
    """
    if not prev_header or not table or len(table) < 1:
        return False

    first_row = table[0]

    # 检查1：列数是否一致
    if len(first_row) != len(prev_header):
        return False

    # 检查2：第一行是否和上一页表头完全一致（重复表头）
    if first_row == prev_header:
        return True

    # 检查3：第一行是否像数据（含数字多）而非表头
    numeric_count = sum(1 for cell in first_row if cell and re.search(r'\d', str(cell)))
    if numeric_count > len(first_row) * 0.4:
        return True

    return False


def table_to_natural_language(df, source_name: str = "表格") -> List[str]:
    """
    将DataFrame转换为自然语言描述

    策略：每行转换为一句话，包含所有列的信息
    例如：产品A的价格是100元，销量是50件，库存是200件

    为什么要转：喂给 LLM 和检索时，"列名是X"这种完整句子比制表符分隔的原始表格
    更容易被理解、也更容易按语义召回；代价是文本膨胀（列名每行重复一次）。

    Args:
        df: pandas DataFrame（pandas 的二维表格对象，有行索引和列名）
        source_name: 数据来源名称

    Returns:
        自然语言文本列表
    """
    import pandas as pd

    texts = []

    # 添加表格概述
    summary = f"这是一个包含{len(df)}行数据的{source_name}，列名包括：{', '.join(df.columns.tolist())}"
    texts.append(summary)

    # 转换每一行
    for idx, row in df.iterrows():
        parts = []
        for col in df.columns:
            value = row[col]
            if pd.notna(value):
                parts.append(f"{col}是{value}")

        if parts:
            sentence = "，".join(parts) + "。"
            texts.append(sentence)

    return texts


def extract_pdf_with_ocr(file_path: str) -> Tuple[str, str]:
    """
    智能提取PDF内容（文件级检测 + 页级OCR回退 + 跨页表格合并）

    策略：
    1. 先用 is_scanned_pdf() 做文件级检测
    2. 扫描版 → 全部页面统一OCR（保证一致性）
    3. 文字版 → 直接提取文本 + 页级OCR回退（兼顾性能和可靠性）
    4. 表格统一用 pdfplumber 提取，支持跨页合并

    两条分支的取舍：扫描版整本走 OCR 是为了输出口径一致（避免一半文字层一半 OCR）；
    文字版默认不 OCR 是为了省时间（OCR 一页要几百毫秒起），只在单页文本缺失或乱码时补。
    has_ocr 为 False（Python 包 pytesseract 都没装）时这两条 OCR 路径全跳过。
    但要注意 has_ocr 只看 `import pytesseract` 成不成功，**不检查 Tesseract 程序本身在不在** ——
    本项目当前环境正是"包在、程序不在"：has_ocr=True，于是每页都真去调 OCR、每次都在
    _ocr_page 里抛 TesseractNotFoundError 被吞掉返回空串，白等一遍仍拿不到文本。
    所以扫描版 PDF 解析出空文本时，先分清是"OCR 没跑"还是"跑了但引擎缺失"。

    Args:
        file_path: PDF文件路径

    Returns:
        (文本内容, 表格文本)
    """
    try:
        import pdfplumber
        import pandas as pd

        # 尝试加载OCR（可选）
        try:
            import pytesseract
            # 尝试加载项目中的Tesseract配置
            try:
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                ocr_config_path = os.path.join(project_root, 'ocr_tools')
                if os.path.exists(ocr_config_path):
                    if ocr_config_path not in sys.path:
                        sys.path.insert(0, project_root)
                    from ocr_tools.ocr_config import setup_tesseract
                    setup_tesseract()
            except Exception:
                pass
            has_ocr = True
        except ImportError:
            has_ocr = False
            logger.info("  pytesseract未安装，跳过OCR功能")

        # 文件级检测
        use_ocr_for_all = has_ocr and is_scanned_pdf(file_path)

        text_content = []
        tables_content = []
        prev_page_last_header = None
        pending_df = None

        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                logger.info(f"  处理第 {page_num}/{len(pdf.pages)} 页")

                if use_ocr_for_all:
                    # 扫描版：统一OCR
                    page_text = _ocr_page(page, page_num, pytesseract)
                else:
                    # 文字版：直接提取 + 页级回退
                    page_text = page.extract_text()

                    # 页级回退：提取失败或内容少或乱码 → 对该页使用OCR
                    if has_ocr and (not page_text
                                    or len(page_text.strip()) < 50
                                    or has_garbled_text(page_text)):
                        logger.info(f"    页面{page_num}文本异常，回退到OCR...")
                        ocr_text = _ocr_page(page, page_num, pytesseract)
                        if ocr_text:
                            page_text = ocr_text

                if page_text:
                    text_content.append(f"[第{page_num}页]\n{page_text}")

                # 表格提取（支持跨页）
                tables = page.extract_tables()
                if tables:
                    logger.info(f"    发现 {len(tables)} 个表格")
                    for table_idx, table in enumerate(tables, 1):
                        if not table or len(table) < 1:
                            continue

                        # 检查是否是跨页表格
                        is_cont = (table_idx == 1 and
                                   is_table_continuation(table, prev_page_last_header))

                        if is_cont:
                            logger.info(f"    检测到跨页表格，与上一页合并")
                            if table[0] == prev_page_last_header:
                                data_rows = table[1:]
                            else:
                                data_rows = table

                            df = pd.DataFrame(data_rows, columns=prev_page_last_header)
                            if pending_df is not None:
                                pending_df = pd.concat([pending_df, df], ignore_index=True)
                            else:
                                pending_df = df
                        else:
                            # 先处理之前待合并的表格
                            if pending_df is not None:
                                table_texts = table_to_natural_language(
                                    pending_df,
                                    f"第{page_num-1}-{page_num}页跨页表格"
                                )
                                tables_content.extend(table_texts)
                                pending_df = None

                            # 处理当前表格
                            if len(table) > 1:
                                df = pd.DataFrame(table[1:], columns=table[0])
                                table_texts = table_to_natural_language(
                                    df,
                                    f"第{page_num}页表格{table_idx}"
                                )
                                tables_content.extend(table_texts)

                                if table_idx == len(tables):
                                    prev_page_last_header = table[0]
                            else:
                                prev_page_last_header = None
                else:
                    prev_page_last_header = None

        # 处理最后一个待合并的表格
        if pending_df is not None:
            table_texts = table_to_natural_language(pending_df, "跨页表格")
            tables_content.extend(table_texts)

        full_text = "\n\n".join(text_content)
        tables_text = "\n\n".join(tables_content)

        return full_text, tables_text

    except ImportError as e:
        logger.error(f"缺少依赖库: {e}")
        return "", ""
    except Exception as e:
        logger.error(f"PDF处理失败: {e}")
        return "", ""


# ========== Word 解析 ==========

def extract_docx(file_path: str) -> str:
    """
    解析 Word 文档（.docx / .doc）

    支持三种内容来源：
    1. 普通段落（doc.paragraphs）
    2. 表格（doc.tables）
    3. 文本框（TextBox）—— 许多简历/排版文档将内容放在文本框中

    对于文本框内容会自动去重（Word 的 mc:AlternateContent 机制
    常导致同一文本框内容出现两份：Fallback + Choice）。

    三段内容来源顺序即最终文本顺序，且不去重跨来源的重复 —— 表格里和段落里出现同一句话，
    两处都会保留。文本框那段是按"整块文本"去重的，若两个真正不同的文本框恰好内容一字不差，
    也会被并成一份（实际几乎不会遇到，知道边界在哪即可）。

    Args:
        file_path: 文件路径

    Returns:
        文本内容
    """
    try:
        from docx import Document
        doc = Document(file_path)

        all_parts = []

        # ---- 1. 普通段落 ----
        para_texts = [p.text for p in doc.paragraphs if p.text.strip()]
        if para_texts:
            all_parts.extend(para_texts)

        # ---- 2. 表格 ----
        for table in doc.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if cells:
                    all_parts.append(" | ".join(cells))

        # ---- 3. 文本框（TextBox） ----
        # Word 文本框的 XML 结构：
        #   w:body -> ... -> mc:AlternateContent -> mc:Choice/mc:Fallback
        #     -> w:drawing -> ... -> wps:txbx -> w:txbxContent -> w:p -> w:r -> w:t
        # 由于 AlternateContent 会产生重复（Choice 和 Fallback 各一份），
        # 这里提取后通过集合去重。
        try:
            ns_w = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
            body = doc.element.body
            txbx_contents = body.findall(f'.//{{{ns_w}}}txbxContent')

            seen_texts = set()  # 用于去重
            for txbx in txbx_contents:
                paras = txbx.findall(f'.//{{{ns_w}}}p')
                lines = []
                for p in paras:
                    runs = p.findall(f'.//{{{ns_w}}}r/{{{ns_w}}}t')
                    line = ''.join(r.text or '' for r in runs)
                    if line.strip():
                        lines.append(line.strip())

                if lines:
                    block_text = "\n".join(lines)
                    # 去重：同一文本框内容可能因 mc:AlternateContent 出现两次
                    if block_text not in seen_texts:
                        seen_texts.add(block_text)
                        all_parts.append(block_text)
        except Exception as e:
            logger.warning(f"文本框提取时出错（不影响其他内容）: {e}")

        return "\n\n".join(all_parts)

    except ImportError:
        logger.error("缺少 python-docx 库")
        return ""
    except Exception as e:
        logger.error(f"Word解析失败: {e}")
        return ""


# ========== Excel / CSV 解析 ==========

def _smart_sheet_summary(df, sheet_name: str, sample_rows: int = 5) -> str:
    """
    生成单个 Sheet 的智能摘要（用于大表场景）

    Sheet（工作表）= Excel 文件里的一张表；一个 .xlsx 可以有多张。
    摘要而非全量，是为了在"大表塞不进 token 预算"时至少让 LLM 知道表的结构。

    包含：
    1. Sheet 概述（名称、行列数、列名）
    2. 前 N 行样例数据
    3. 数值列的基础统计（min/max/mean）

    列名里的 "Unnamed:" 是 pandas 对没有表头的列自动起的名字（Excel 常有合并单元格
    导致表头识别不出来），这里把它们改写成「列N」再展示，避免 LLM 看到一串无意义的键名。

    Args:
        df: DataFrame
        sheet_name: Sheet名称
        sample_rows: 抽样行数

    Returns:
        摘要文本
    """
    import pandas as pd

    rows, cols = len(df), len(df.columns)
    # 清理 "Unnamed:" 开头的列名
    col_names = [str(c) for c in df.columns]
    display_cols = [c if not c.startswith('Unnamed') else f'列{i+1}' for i, c in enumerate(col_names)]

    parts = [f"Sheet「{sheet_name}」：共{rows}行 x {cols}列，字段包括：{', '.join(display_cols)}"]

    # 抽样前 N 行
    sample_df = df.head(sample_rows)
    parts.append(f"前{min(sample_rows, rows)}行样例数据：")
    for _, row in sample_df.iterrows():
        items = []
        for c, dc in zip(df.columns, display_cols):
            v = row[c]
            if pd.notna(v):
                sv = str(v).strip()
                if len(sv) > 80:
                    sv = sv[:77] + "..."
                items.append(f"{dc}={sv}")
        if items:
            parts.append("  " + "，".join(items))

    # 数值列基础统计
    num_cols = df.select_dtypes(include='number').columns.tolist()
    if num_cols:
        stats = []
        for c in num_cols[:5]:  # 最多展示5个数值列
            dc = display_cols[col_names.index(c)]
            try:
                stats.append(f"{dc}：范围[{df[c].min():.4g} ~ {df[c].max():.4g}]，均值={df[c].mean():.4g}")
            except Exception:
                pass
        if stats:
            parts.append("数值列统计：" + "；".join(stats))

    return "\n".join(parts)


def extract_excel(file_path: str, max_chars: int = 0) -> str:
    """
    解析 Excel 文件，智能分级转换

    策略：
    1. 先尝试全量自然语言转换（每行一句话）
    2. 如果超过 max_chars 限制，自动降级为「智能摘要」模式：
       - 每个 Sheet 输出概述（行列数、列名）
       - 抽样前5行数据
       - 数值列基础统计
    3. max_chars=0 表示不限制，始终全量转换（向量数据库入库场景）

    Args:
        file_path: Excel文件路径
        max_chars: 最大字符数限制（0=不限制）

    Returns:
        自然语言文本
    """
    try:
        import pandas as pd

        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names

        # ---- 第一步：尝试全量转换 ----
        full_texts = []
        for sheet_name in sheet_names:
            logger.info(f"  处理Sheet: {sheet_name}")
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            texts = table_to_natural_language(df, f"Sheet '{sheet_name}'")
            full_texts.extend(texts)

        full_result = "\n\n".join(full_texts)

        # 不限制 或 未超限 → 直接返回全量
        if max_chars <= 0 or len(full_result) <= max_chars:
            return full_result

        # ---- 第二步：超限 → 降级为智能摘要 ----
        logger.info(f"  Excel全量转换{len(full_result)}字符超限({max_chars})，降级为智能摘要模式")
        summary_parts = [
            f"该Excel文件包含 {len(sheet_names)} 个Sheet：{', '.join(sheet_names)}",
            f"（原始数据量较大，以下为各Sheet的结构概要和样例数据）",
        ]

        for sheet_name in sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            summary_parts.append(_smart_sheet_summary(df, sheet_name))

        return "\n\n".join(summary_parts)

    except ImportError:
        logger.error("缺少pandas或openpyxl库")
        return ""
    except Exception as e:
        logger.error(f"Excel处理失败: {e}")
        return ""


def extract_csv(file_path: str, max_chars: int = 0) -> str:
    """
    解析 CSV 文件，智能分级转换

    策略同 extract_excel：先全量，超限则降级为智能摘要。

    Args:
        file_path: CSV文件路径
        max_chars: 最大字符数限制（0=不限制）

    Returns:
        自然语言文本
    """
    try:
        import pandas as pd

        df = pd.read_csv(file_path)
        texts = table_to_natural_language(df, "CSV文件")
        full_result = "\n\n".join(texts)

        # 不限制 或 未超限 → 直接返回全量
        if max_chars <= 0 or len(full_result) <= max_chars:
            return full_result

        # 降级为智能摘要
        logger.info(f"  CSV全量转换{len(full_result)}字符超限({max_chars})，降级为智能摘要模式")
        summary_parts = [
            f"该CSV文件数据量较大，以下为结构概要和样例数据：",
            _smart_sheet_summary(df, os.path.basename(file_path)),
        ]
        return "\n\n".join(summary_parts)

    except ImportError:
        logger.error("缺少pandas库")
        return ""
    except Exception as e:
        logger.error(f"CSV处理失败: {e}")
        return ""


# ========== 统一入口 ==========

def parse_file(file_path: str, max_chars: int = 0) -> dict:
    """
    统一文件解析入口

    将各种格式的文件解析为纯文本。供 API 接口（用户上传）和向量数据库入库共用。

    按扩展名分派（不是按 MIME type 探测）：所以文件后缀写错就走错分支。
    失败统一用返回值表达（success=False + error），不抛异常 —— 调用方不必包 try。

    max_chars 有两条不同的落地方式：PDF/DOCX 是先解析完再在末尾整体截断（truncated=True），
    Excel/CSV 则是在各自函数内部直接降级成摘要（不截断）。同一个参数，行为并不一致。

    Args:
        file_path: 文件路径
        max_chars: 最大字符数限制（0=不限制）

    Returns:
        {
            "success": True/False,
            "text": "解析后的文本",
            "truncated": False,  # 是否被截断
            "char_count": 1234,
            "error": "错误信息"  # 仅失败时
        }
    """
    if not os.path.exists(file_path):
        return {"success": False, "error": f"文件不存在: {file_path}"}

    ext = os.path.splitext(file_path)[1].lower()

    try:
        text = ""

        # 纯文本类：不做任何加工，原样读出
        if ext in (".txt", ".md"):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

        # PDF：走"文字层 + OCR 降级 + 跨页表格合并"那条重路径，正文与表格拼成一段
        elif ext == ".pdf":
            text_content, tables_content = extract_pdf_with_ocr(file_path)
            if text_content or tables_content:
                text = f"{text_content}\n\n[表格数据]\n{tables_content}" if tables_content else text_content
            else:
                return {"success": False, "error": "PDF 内容为空，无法提取有效文本"}

        # Word：按"段落 / 表格 / 文本框"三处收集，注意 .doc（老二进制格式）实际会被 python-docx 拒掉
        elif ext in (".docx", ".doc"):
            text = extract_docx(file_path)
            if not text:
                return {"success": False, "error": "Word 文档内容为空或解析失败"}

        # CSV / Excel：带 max_chars 的智能分级（全量 → 超限降级为摘要），所以把参数透传下去
        elif ext == ".csv":
            text = extract_csv(file_path, max_chars=max_chars)
            if not text:
                return {"success": False, "error": "CSV 文件内容为空或解析失败"}

        elif ext in (".xlsx", ".xls"):
            text = extract_excel(file_path, max_chars=max_chars)
            if not text:
                return {"success": False, "error": "Excel 文件内容为空或解析失败"}

        # JSON：整体美化成缩进文本，不做字段裁剪 —— 深层嵌套会原样展开，可能很长
        elif ext == ".json":
            import json
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                text = json.dumps(data, ensure_ascii=False, indent=2)

        else:
            return {"success": False, "error": f"不支持的文件格式: {ext}"}

        if not text.strip():
            return {"success": False, "error": "文件内容为空"}

        # 截断处理
        truncated = False
        if max_chars > 0 and len(text) > max_chars:
            text = text[:max_chars]
            truncated = True

        return {
            "success": True,
            "text": text,
            "truncated": truncated,
            "char_count": len(text),
        }

    except Exception as e:
        logger.error(f"文件解析失败: {file_path}, 错误: {e}")
        return {"success": False, "error": f"文件解析失败: {str(e)}"}
