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

# 设置Windows控制台UTF-8编码
if sys.platform == 'win32':
    import codecs
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    if hasattr(sys.stderr, 'buffer'):
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# 定位项目根目录（脚本在 tools/scripts/ 下，根目录在上两级）
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# 加载.env文件
def load_env():
    """加载.env文件中的环境变量（基于项目根目录定位）"""
    env_path = os.path.join(PROJECT_ROOT, '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ.setdefault(key.strip(), value.strip())
    else:
        print(f"警告: .env 文件不存在于 {env_path}")

# 切换工作目录到项目根目录（确保所有相对路径正常）
os.chdir(PROJECT_ROOT)

# 将项目根目录加入 sys.path（确保能 import 项目模块）
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

load_env()

# 配置日志
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from rag_core.api_embedder import APIEmbedder
from rag_core.chroma_store import ChromaStore

# 导入公共文件解析模块（与 api.py 共用同一套解析逻辑）
from core.file_parser import (
    parse_file,
    table_to_natural_language,
    is_table_continuation,
    has_garbled_text,
    is_scanned_pdf,
    extract_pdf_with_ocr,
    extract_excel,
    extract_csv,
    extract_docx,
)


# ========== 文档白名单 ==========
# 仅索引与核心问题相关的文档（doc_id = 文件名不含扩展名）
# 不在此列表中的文档将被跳过，不会写入向量数据库
#
# 当前场景：电商售后助手（云集优选）
# 售后助手要能回答的问题：
#   1. 七天无理由怎么算？哪些商品不支持？
#   2. 退货的运费谁承担？
#   3. 退款多久到账？
#   4. 商品有质量问题怎么办？时限和普通退货有什么不同？
#   5. 会员等级对售后权益有什么影响？
#
# ⚠️ 换知识库时必须全量重建（python main.py --init-db）。
#    增量脚本 incremental_update.py 不做白名单过滤，会把被排除的文档也索引进去。
#
# 如需索引全部文档，将此变量设为 None
ALLOWED_DOC_IDS = {
    # 退货与换货
    "seven_day_return_policy",   # 七天无理由退货：起算时间、商品完好标准、例外商品
    "return_exchange_process",   # 退换货申请流程：怎么申请、审核时效、寄回要求
    "quality_issue_policy",      # 质量问题处理：15日退货/30日换货，运费平台承担

    # 费用与资金
    "shipping_fee_policy",       # 退换货运费承担规则：谁出运费、会员补贴
    "refund_timeline",           # 退款到账时间：按支付渠道区分
    "price_protection_policy",   # 价格保护：下单后降价可退差价

    # 物流与发货
    "logistics_delivery_policy", # 物流时效与配送范围：各省时效、偏远地区
    "late_shipment_policy",      # 超时未发货处理：发货时限、超时赔付

    # 会员与票据
    "member_benefits_policy",    # 会员等级与售后权益：四级会员的差异化待遇
    "invoice_policy",            # 发票开具与管理：开票、换开、随货发票
}



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

    # 支持的文件扩展名
    supported_extensions = ['.md', '.txt', '.pdf', '.docx', '.xlsx', '.csv', '.json']

    # 遍历文档目录
    for file_path in docs_path.rglob('*'):
        if not file_path.is_file():
            continue

        ext = file_path.suffix.lower()
        if ext not in supported_extensions:
            continue

        # 跳过隐藏目录（.claude 等）和 samples 目录
        if any(part.startswith('.') for part in file_path.relative_to(docs_path).parts[:-1]):
            continue
        if 'samples' in file_path.parts:
            continue

        # 白名单过滤：仅索引指定的文档
        if ALLOWED_DOC_IDS is not None:
            doc_id = file_path.stem
            if doc_id not in ALLOWED_DOC_IDS:
                logger.debug(f"  跳过（不在白名单中）: {file_path.name}")
                continue

        try:
            content = None
            doc_type = ext[1:]  # 去掉点号

            logger.debug(f"处理文档: {file_path.name} ({doc_type})")

            # 处理不同格式（统一使用 core.file_parser 中的解析函数）
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
                # Word文件（使用 extract_docx 统一解析）
                content = extract_docx(str(file_path))

            elif ext == '.xlsx':
                # Excel文件（转自然语言）
                content = extract_excel(str(file_path))

            elif ext == '.csv':
                # CSV文件（转自然语言）
                content = extract_csv(str(file_path))

            elif ext == '.json':
                # JSON文件（直接读取）
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    content = json.dumps(data, ensure_ascii=False, indent=2)

            if content:
                # 提取标题（针对不同格式优化）
                lines = content.split('\n')
                title = file_path.stem  # 默认使用文件名

                if lines:
                    first_line = lines[0].strip()
                    # PDF 内容以 [第X页] 开头，跳过页码标记，寻找真正的标题
                    if ext == '.pdf':
                        for line in lines:
                            line = line.strip()
                            # 跳过空行和页码标记
                            if not line or line.startswith('[第') and line.endswith('页]'):
                                continue
                            title = line.strip('#').strip()[:100]
                            break
                    else:
                        title = first_line.strip('#').strip()[:100]

                documents.append({
                    "content": content,
                    "metadata": {
                        "source": str(file_path),
                        "title": title,
                        "doc_id": file_path.stem,
                        "doc_type": doc_type
                    }
                })
                logger.debug(f"  ✓ 成功加载 ({len(content)} 字符)")

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
    print("\n" + "=" * 60)
    print("  向量数据库初始化")
    print("=" * 60)

    # 1. 初始化Embedder
    api_key = os.getenv('DASHSCOPE_API_KEY')
    if not api_key:
        print("[ERROR] 未找到 DASHSCOPE_API_KEY 环境变量")
        return

    embedder = APIEmbedder(
        api_key=api_key,
        provider='dashscope',
        model='text-embedding-v2'
    )

    # 2. 初始化ChromaDB（清空旧集合）
    import chromadb
    from chromadb.config import Settings
    client = chromadb.PersistentClient(
        path="./vector_db/chroma_db",
        settings=Settings(anonymized_telemetry=False, allow_reset=True)
    )
    try:
        client.delete_collection("knowledge_base")
    except:
        pass

    chroma_store = ChromaStore(
        dimension=1536,
        collection_name="knowledge_base",
        persist_directory="./vector_db/chroma_db"
    )

    # 3. 读取文档
    print("\n[1/4] 读取文档...")
    if ALLOWED_DOC_IDS is not None:
        print(f"  白名单模式：仅索引 {len(ALLOWED_DOC_IDS)} 个指定文档")
    documents = read_documents("data/knowledge")

    if not documents:
        print("[ERROR] 未找到文档，请检查 data/knowledge 目录")
        return

    # 4. 分块
    print("[2/4] 文档分块...")
    all_chunks = []
    all_metadatas = []
    file_stats = []  # 收集每个文件的统计信息

    for doc in documents:
        content = doc["content"]
        metadata = doc["metadata"]
        chunks = chunk_document(content, chunk_size=800)

        file_stats.append({
            "source": metadata.get("source", ""),
            "title": metadata.get("title", ""),
            "doc_type": metadata.get("doc_type", ""),
            "char_count": len(content),
            "chunk_count": len(chunks),
        })

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_id"] = i
            all_metadatas.append(chunk_metadata)

    # 5. 批量向量化
    total_batches = (len(all_chunks) - 1) // 10 + 1
    print(f"[3/4] 向量化 {len(all_chunks)} 个块（{total_batches} 批）...")
    batch_size = 10
    for i in range(0, len(all_chunks), batch_size):
        batch_chunks = all_chunks[i:i+batch_size]
        batch_metadatas = all_metadatas[i:i+batch_size]
        embeddings = embedder.encode_texts(batch_chunks)
        chroma_store.add_vectors(
            vectors=embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings,
            chunks=batch_chunks,
            metadata=batch_metadatas
        )

    # 6. 测试检索
    # 用一个真实的售后问题做冒烟测试 —— 如果知识库换对了，这条应该能命中政策文档。
    # 换了知识库场景的话，记得把这里也改成新场景的问题，否则分数会很低、看不出效果。
    print("[4/4] 验证检索...")
    test_query = "七天无理由退货的时间怎么计算"
    query_embedding = embedder.encode_query(test_query)
    results = chroma_store.search(
        query_vector=query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding,
        top_k=3
    )

    # ===== 输出统计报表 =====
    print("\n" + "=" * 60)
    print("  初始化完成 - 统计报表")
    print("=" * 60)

    # 按类型分组统计
    type_stats = {}
    for fs in file_stats:
        t = fs["doc_type"]
        if t not in type_stats:
            type_stats[t] = {"count": 0, "chunks": 0, "chars": 0}
        type_stats[t]["count"] += 1
        type_stats[t]["chunks"] += fs["chunk_count"]
        type_stats[t]["chars"] += fs["char_count"]

    print(f"\n  数据库路径: ./vector_db/chroma_db")
    print(f"  文档总数:   {len(documents)}")
    print(f"  Chunk总数:  {len(all_chunks)}")
    print(f"  Embedding:  text-embedding-v2 (1536维)")

    # 类型统计
    print(f"\n  {'类型':<6} {'文件数':>6} {'Chunks':>8} {'字符数':>10}")
    print(f"  {'-'*6} {'-'*6} {'-'*8} {'-'*10}")
    for t, s in sorted(type_stats.items()):
        print(f"  {t:<6} {s['count']:>6} {s['chunks']:>8} {s['chars']:>10,}")

    # 文件明细表
    print(f"\n  {'序号':>4}  {'文件名':<40} {'类型':<5} {'字符':>8} {'Chunks':>6}")
    print(f"  {'─'*4}  {'─'*40} {'─'*5} {'─'*8} {'─'*6}")
    for idx, fs in enumerate(file_stats, 1):
        fname = Path(fs["source"]).name
        # 截断过长的文件名
        if len(fname) > 38:
            fname = fname[:35] + "..."
        print(f"  {idx:>4}  {fname:<40} {fs['doc_type']:<5} {fs['char_count']:>8,} {fs['chunk_count']:>6}")

    # 测试检索结果
    print(f"\n  检索验证 (query=\"{test_query}\"):")
    for i, item in enumerate(results):
        chunk = item[0]
        score = item[1]
        meta = item[2] if len(item) >= 3 else {}
        content = chunk if isinstance(chunk, str) else str(chunk)
        source = Path(meta.get("source", "")).name if meta.get("source") else "unknown"
        print(f"    {i+1}. score={score:.3f}  {source}  \"{content[:60]}...\"")

    print(f"\n{'=' * 60}")
    print(f"  Done! 可运行 python main.py 或 python api.py 使用系统")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    init_vector_db()
