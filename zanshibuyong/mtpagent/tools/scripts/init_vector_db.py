# -*- coding: utf-8 -*-
"""
向量数据库初始化脚本
==================

支持多种文档格式：
- Markdown (.md)
- PDF (.pdf) - 支持OCR和表格提取
- Word (.docx)
- Excel (.xlsx) - 转换为自然语言
- JSON (.json)
- CSV (.csv) - 转换为自然语言
- Text (.txt)
"""

import os
import sys
import logging
from pathlib import Path
import json
import pandas as pd  # 添加pandas导入

# 设置Windows控制台UTF-8编码
if sys.platform == 'win32':
    import codecs
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    if hasattr(sys.stderr, 'buffer'):
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# 加载.env文件
def load_env():
    """加载.env文件中的环境变量"""
    env_path = '.env'
    if os.path.exists(env_path):
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

load_env()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from rag_core.api_embedder import APIEmbedder
from rag_core.chroma_store import ChromaStore


def table_to_natural_language(df, source_name="表格"):
    """
    将DataFrame转换为自然语言描述

    策略：每行转换为一句话，包含所有列的信息
    例如：产品A的价格是100元，销量是50件，库存是200件

    Args:
        df: pandas DataFrame
        source_name: 数据来源名称

    Returns:
        自然语言文本列表
    """
    texts = []

    # 添加表格概述
    summary = f"这是一个包含{len(df)}行数据的{source_name}，列名包括：{', '.join(df.columns.tolist())}"
    texts.append(summary)

    # 转换每一行
    for idx, row in df.iterrows():
        # 构建自然语言描述
        parts = []
        for col in df.columns:
            value = row[col]
            if pd.notna(value):  # 跳过空值
                parts.append(f"{col}是{value}")

        if parts:
            sentence = "，".join(parts) + "。"
            texts.append(sentence)

    return texts


def is_table_continuation(table, prev_header):
    """
    判断表格是否是上一页表格的延续

    Args:
        table: 当前表格原始数据
        prev_header: 上一页表格的表头

    Returns:
        bool: 是否是跨页延续
    """
    if not prev_header or not table or len(table) < 1:
        return False

    first_row = table[0]

    # 检查1：列数是否一致
    if len(first_row) != len(prev_header):
        return False

    # 检查2：第一行是否和上一页表头完全一致（重复表头）
    if first_row == prev_header:
        return True  # 是重复表头，也算延续

    # 检查3：第一行是否像数据（含数字多）而非表头
    import re
    numeric_count = sum(1 for cell in first_row if cell and re.search(r'\d', str(cell)))
    if numeric_count > len(first_row) * 0.4:  # 超过40%含数字 → 是数据行
        return True

    return False


def is_scanned_pdf(file_path, sample_pages=3):
    """
    文件级检测：判断PDF是扫描版还是文字版

    检测策略（采样前N页）：
    1. 文本层检测：提取文本，长度 < 50 字符视为无文本
    2. 乱码检测：特殊字符占比 > 30% 视为乱码（伪文字层）
    3. 图片占比检测：图片面积 > 页面80% 视为扫描件

    综合判断：超过一半的采样页无有效文本 → 扫描版

    Args:
        file_path: PDF文件路径
        sample_pages: 采样页数（默认前3页）

    Returns:
        bool: True=扫描版, False=文字版
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

                # 检查2：乱码检测（有文本层但内容是乱码）
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


def has_garbled_text(text):
    """
    检测文本是否为乱码（伪文字层）

    原理：扫描版PDF有时会嵌入一层伪文本，提取出来全是乱码。
    通过统计特殊字符（非中英文、非数字、非常见标点）的占比来判断。

    Args:
        text: 待检测文本

    Returns:
        bool: True=乱码, False=正常文本
    """
    import re
    if not text or len(text.strip()) < 10:
        return False

    # 去除空白字符后统计
    cleaned = text.strip()
    # 正常字符：中文、英文字母、数字、常见中英文标点、空白
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


def extract_pdf_with_ocr(file_path):
    """
    智能提取PDF内容（方案A：文件级检测 + 页级回退）

    策略：
    1. 先用 is_scanned_pdf() 做文件级检测
    2. 扫描版 → 全部页面统一OCR（保证一致性）
    3. 文字版 → 直接提取文本 + 页级OCR回退（兼顾性能和可靠性）
    4. 表格统一用 pdfplumber 提取，支持跨页合并

    Args:
        file_path: PDF文件路径

    Returns:
        (文本内容, 表格列表)
    """
    try:
        import pdfplumber
        import pytesseract
        from PIL import Image
        import io

        # 使用项目中的Tesseract配置
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from ocr_tools.ocr_config import setup_tesseract
        setup_tesseract()

        # ===== 文件级检测 =====
        use_ocr_for_all = is_scanned_pdf(file_path)

        text_content = []
        tables_content = []
        prev_page_last_header = None  # 记录上一页最后一个表格的表头
        pending_df = None  # 记录待合并的跨页表格

        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                logger.info(f"  处理第 {page_num}/{len(pdf.pages)} 页")

                if use_ocr_for_all:
                    # ===== 扫描版：统一OCR =====
                    page_text = _ocr_page(page, page_num, pytesseract)
                else:
                    # ===== 文字版：直接提取 + 页级回退 =====
                    page_text = page.extract_text()

                    # 页级回退：提取失败或内容少或乱码 → 对该页使用OCR
                    if (not page_text
                            or len(page_text.strip()) < 50
                            or has_garbled_text(page_text)):
                        logger.info(f"    页面{page_num}文本异常，回退到OCR...")
                        ocr_text = _ocr_page(page, page_num, pytesseract)
                        if ocr_text:
                            page_text = ocr_text

                if page_text:
                    text_content.append(f"[第{page_num}页]\n{page_text}")

                # ===== 表格提取（支持跨页）=====
                tables = page.extract_tables()
                if tables:
                    logger.info(f"    发现 {len(tables)} 个表格")
                    for table_idx, table in enumerate(tables, 1):
                        import pandas as pd
                        if not table or len(table) < 1:
                            continue

                        # 检查是否是跨页表格（只对本页第一个表格检测）
                        is_continuation = (table_idx == 1 and
                                         is_table_continuation(table, prev_page_last_header))

                        if is_continuation:
                            logger.info(f"    检测到跨页表格，与上一页合并")
                            # 处理重复表头
                            if table[0] == prev_page_last_header:
                                data_rows = table[1:]  # 去掉重复表头
                            else:
                                data_rows = table  # 第一行就是数据

                            # 用上一页的表头创建DataFrame
                            df = pd.DataFrame(data_rows, columns=prev_page_last_header)

                            # 合并到待处理的DataFrame
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

                            # 处理当前表格（正常表格）
                            if len(table) > 1:
                                df = pd.DataFrame(table[1:], columns=table[0])
                                table_texts = table_to_natural_language(
                                    df,
                                    f"第{page_num}页表格{table_idx}"
                                )
                                tables_content.extend(table_texts)

                                # 记录本页最后一个表格的表头
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
        logger.error("请安装: pip install pdfplumber pytesseract pillow pandas")
        logger.error("并安装Tesseract OCR: https://github.com/tesseract-ocr/tesseract")
        return None, None
    except Exception as e:
        logger.error(f"PDF处理失败: {e}")
        return None, None


def _ocr_page(page, page_num, pytesseract):
    """
    对单个页面执行OCR

    Args:
        page: pdfplumber page对象
        page_num: 页码
        pytesseract: pytesseract模块

    Returns:
        OCR识别的文本
    """
    try:
        # 将页面转为图片
        img = page.to_image(resolution=300)
        pil_img = img.original

        # OCR识别
        ocr_text = pytesseract.image_to_string(pil_img, lang='chi_sim+eng')
        if ocr_text.strip():
            logger.info(f"    OCR成功提取 {len(ocr_text)} 字符")
            return ocr_text
        return ""
    except Exception as e:
        logger.warning(f"    OCR失败: {e}")
        return ""


def load_excel_as_natural_language(file_path):
    """
    加载Excel文件并转换为自然语言

    Args:
        file_path: Excel文件路径

    Returns:
        自然语言文本
    """
    try:
        import pandas as pd

        # 读取所有sheet
        excel_file = pd.ExcelFile(file_path)
        all_texts = []

        for sheet_name in excel_file.sheet_names:
            logger.info(f"  处理Sheet: {sheet_name}")
            df = pd.read_excel(file_path, sheet_name=sheet_name)

            # 转换为自然语言
            texts = table_to_natural_language(df, f"Sheet '{sheet_name}'")
            all_texts.extend(texts)

        return "\n\n".join(all_texts)

    except ImportError:
        logger.error("缺少pandas或openpyxl库")
        logger.error("请安装: pip install pandas openpyxl")
        return None
    except Exception as e:
        logger.error(f"Excel处理失败: {e}")
        return None


def load_csv_as_natural_language(file_path):
    """
    加载CSV文件并转换为自然语言

    Args:
        file_path: CSV文件路径

    Returns:
        自然语言文本
    """
    try:
        import pandas as pd

        df = pd.read_csv(file_path)
        texts = table_to_natural_language(df, f"CSV文件")

        return "\n\n".join(texts)

    except ImportError:
        logger.error("缺少pandas库")
        logger.error("请安装: pip install pandas")
        return None
    except Exception as e:
        logger.error(f"CSV处理失败: {e}")
        return None


def read_documents(docs_dir: str = "data/knowledge"):
    """
    读取文档文件夹中的所有文档（支持多种格式）

    支持格式：
    - .md, .txt: 直接读取
    - .pdf: OCR + 表格提取
    - .docx: LangChain加载
    - .xlsx: 转换为自然语言
    - .csv: 转换为自然语言
    - .json: LangChain加载

    Args:
        docs_dir: 文档目录

    Returns:
        文档列表
    """
    documents = []
    docs_path = Path(docs_dir)

    if not docs_path.exists():
        logger.error(f"文档目录不存在: {docs_dir}")
        return documents

    # 尝试导入依赖
    try:
        import docx
        has_docx = True
    except ImportError:
        has_docx = False
        logger.warning("python-docx未安装，Word文件可能无法处理")

    # 支持的文件扩展名
    supported_extensions = ['.md', '.txt', '.pdf', '.docx', '.xlsx', '.csv', '.json']

    # 遍历文档目录
    for file_path in docs_path.rglob('*'):
        if not file_path.is_file():
            continue

        ext = file_path.suffix.lower()
        if ext not in supported_extensions:
            continue

        # 跳过samples目录
        if 'samples' in file_path.parts:
            continue

        try:
            content = None
            doc_type = ext[1:]  # 去掉点号

            logger.info(f"处理文档: {file_path.name} ({doc_type})")

            # 处理不同格式
            if ext in ['.md', '.txt']:
                # 文本文件
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

            elif ext == '.pdf':
                # PDF文件（OCR + 表格）
                text_content, tables_content = extract_pdf_with_ocr(file_path)
                if text_content or tables_content:
                    content = f"{text_content}\n\n[表格数据]\n{tables_content}" if tables_content else text_content

            elif ext == '.docx':
                # Word文件（使用python-docx）
                if has_docx:
                    doc = docx.Document(str(file_path))
                    content = "\n\n".join([para.text for para in doc.paragraphs if para.text.strip()])
                else:
                    logger.warning(f"跳过 {file_path.name}: 需要安装python-docx")

            elif ext == '.xlsx':
                # Excel文件（转自然语言）
                content = load_excel_as_natural_language(file_path)

            elif ext == '.csv':
                # CSV文件（转自然语言）
                content = load_csv_as_natural_language(file_path)

            elif ext == '.json':
                # JSON文件（直接读取）
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    content = json.dumps(data, ensure_ascii=False, indent=2)

            if content:
                # 提取标题
                lines = content.split('\n')
                title = lines[0].strip('#').strip()[:100] if lines else file_path.stem

                documents.append({
                    "content": content,
                    "metadata": {
                        "source": str(file_path),
                        "title": title,
                        "doc_id": file_path.stem,
                        "doc_type": doc_type
                    }
                })
                logger.info(f"  ✓ 成功加载 ({len(content)} 字符)")

        except Exception as e:
            logger.error(f"  ✗ 处理失败: {e}")

    return documents


def chunk_document(content: str, chunk_size: int = 800, overlap: int = 100):
    """
    文档分块（使用 RecursiveCharacterTextSplitter）

    Args:
        content: 文档内容
        chunk_size: 最大块大小（字符数）
        overlap: 重叠大小（字符数）

    Returns:
        文档块列表
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", "，", ",", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
    )
    return splitter.split_text(content)


def init_vector_db():
    """初始化向量数据库"""
    logger.info("="*60)
    logger.info("开始初始化向量数据库")
    logger.info("="*60)

    # 1. 初始化Embedder
    logger.info("\n[步骤1] 初始化Embedder...")
    api_key = os.getenv('DASHSCOPE_API_KEY')
    if not api_key:
        logger.error("未找到DASHSCOPE_API_KEY环境变量")
        return

    embedder = APIEmbedder(
        api_key=api_key,
        provider='dashscope',
        model='text-embedding-v2'
    )
    logger.info("Embedder初始化完成")

    # 2. 初始化ChromaDB
    logger.info("\n[步骤2] 初始化ChromaDB...")

    # 先删除旧集合（如果存在）
    import chromadb
    from chromadb.config import Settings
    client = chromadb.PersistentClient(
        path="./vector_db/chroma_db",
        settings=Settings(anonymized_telemetry=False, allow_reset=True)
    )
    try:
        client.delete_collection("rag_documents")
        logger.info("已删除旧集合")
    except:
        pass

    chroma_store = ChromaStore(
        dimension=1536,  # DashScope text-embedding-v2 实际返回1536维
        collection_name="rag_documents",
        persist_directory="./vector_db/chroma_db"
    )
    logger.info("ChromaDB初始化完成")

    # 3. 读取文档
    logger.info("\n[步骤3] 读取文档（支持多种格式）...")
    documents = read_documents("data/knowledge")
    logger.info(f"共读取 {len(documents)} 个文档")

    if not documents:
        logger.warning("没有找到文档，请确保data/knowledge文件夹中有支持的文档")
        return

    # 统计文档类型
    doc_types = {}
    for doc in documents:
        doc_type = doc["metadata"].get("doc_type", "unknown")
        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1

    logger.info("文档类型统计:")
    for doc_type, count in doc_types.items():
        logger.info(f"  {doc_type}: {count} 个")

    # 4. 分块并向量化
    logger.info("\n[步骤4] 文档分块...")
    all_chunks = []
    all_metadatas = []

    for doc in documents:
        content = doc["content"]
        metadata = doc["metadata"]

        # 分块（使用语义分块策略）
        chunks = chunk_document(content, chunk_size=800)  # 增大chunk_size，因为语义分块更智能
        logger.info(f"  {metadata['title'][:50]}: {len(chunks)} 个块")

        # 为每个块添加元数据
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_id"] = i
            all_metadatas.append(chunk_metadata)

    logger.info(f"总共 {len(all_chunks)} 个文档块")

    # 5. 批量向量化
    logger.info("\n[步骤5] 批量向量化文档...")
    batch_size = 10
    for i in range(0, len(all_chunks), batch_size):
        batch_chunks = all_chunks[i:i+batch_size]
        batch_metadatas = all_metadatas[i:i+batch_size]

        logger.info(f"  处理批次 {i//batch_size + 1}/{(len(all_chunks)-1)//batch_size + 1}")

        # 向量化
        embeddings = embedder.encode_texts(batch_chunks)

        # 存入ChromaDB
        chroma_store.add_vectors(
            vectors=embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings,
            chunks=batch_chunks,
            metadata=batch_metadatas
        )

    logger.info("向量化完成")

    # 6. 测试检索
    logger.info("\n[步骤6] 测试检索...")
    test_query = "什么是向量数据库"
    query_embedding = embedder.encode_query(test_query)
    results = chroma_store.search(
        query_vector=query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding,
        top_k=3
    )

    logger.info(f"测试查询: {test_query}")
    logger.info(f"检索到 {len(results)} 个结果:")
    for i, (chunk, score) in enumerate(results):
        # chunk 可能是字符串或对象
        content = chunk if isinstance(chunk, str) else str(chunk)
        logger.info(f"  {i+1}. 相似度: {score:.3f}")
        logger.info(f"     内容: {content[:80]}...")

    logger.info("\n" + "="*60)
    logger.info("向量数据库初始化完成！")
    logger.info("="*60)
    logger.info(f"\n数据库位置: ./vector_db/chroma_db")
    logger.info(f"文档总数: {len(documents)}")
    logger.info(f"文档块总数: {len(all_chunks)}")
    logger.info("\n支持的文档格式:")
    logger.info("  ✓ Markdown (.md)")
    logger.info("  ✓ Text (.txt)")
    logger.info("  ✓ PDF (.pdf) - 支持OCR和表格提取")
    logger.info("  ✓ Word (.docx)")
    logger.info("  ✓ Excel (.xlsx) - 转换为自然语言")
    logger.info("  ✓ CSV (.csv) - 转换为自然语言")
    logger.info("  ✓ JSON (.json)")
    logger.info("\n现在可以运行 main.py 开始使用RAG系统")


if __name__ == "__main__":
    init_vector_db()
