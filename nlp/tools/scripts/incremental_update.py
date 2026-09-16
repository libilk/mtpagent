# -*- coding: utf-8 -*-
"""
向量数据库增量更新脚本
====================

支持增量添加新文档，无需重新初始化整个数据库
"""

import os
import sys
import logging
from pathlib import Path
import json
import hashlib

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

# 导入文档处理函数
import importlib.util
spec = importlib.util.spec_from_file_location(
    "init_vector_db",
    os.path.join(os.path.dirname(__file__), "init_vector_db.py")
)
init_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(init_module)

read_documents = init_module.read_documents
chunk_document = init_module.chunk_document
extract_pdf_with_ocr = init_module.extract_pdf_with_ocr
load_excel_as_natural_language = init_module.load_excel_as_natural_language
load_csv_as_natural_language = init_module.load_csv_as_natural_language


class IncrementalIndexer:
    """增量索引器"""

    def __init__(self, docs_dir="data/knowledge", index_file="vector_db/indexed_files.json"):
        self.docs_dir = docs_dir
        self.index_file = index_file
        self.indexed_files = self._load_indexed_files()

    def _load_indexed_files(self):
        """加载已索引文件记录"""
        if os.path.exists(self.index_file):
            with open(self.index_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_indexed_files(self):
        """保存已索引文件记录"""
        os.makedirs(os.path.dirname(self.index_file), exist_ok=True)
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(self.indexed_files, f, ensure_ascii=False, indent=2)

    def _get_file_hash(self, file_path):
        """计算文件哈希值（用于检测文件变化）"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def get_new_or_modified_files(self):
        """获取新增或修改的文件"""
        new_files = []
        modified_files = []

        docs_path = Path(self.docs_dir)
        if not docs_path.exists():
            logger.error(f"文档目录不存在: {self.docs_dir}")
            return new_files, modified_files

        supported_extensions = ['.md', '.txt', '.pdf', '.docx', '.xlsx', '.csv', '.json']

        for file_path in docs_path.rglob('*'):
            if not file_path.is_file():
                continue

            ext = file_path.suffix.lower()
            if ext not in supported_extensions:
                continue

            # 跳过samples目录
            if 'samples' in file_path.parts:
                continue

            file_path_str = str(file_path)
            file_hash = self._get_file_hash(file_path)

            if file_path_str not in self.indexed_files:
                # 新文件
                new_files.append((file_path, file_hash))
            elif self.indexed_files[file_path_str] != file_hash:
                # 文件已修改
                modified_files.append((file_path, file_hash))

        return new_files, modified_files

    def mark_as_indexed(self, file_path, file_hash):
        """标记文件为已索引"""
        self.indexed_files[str(file_path)] = file_hash
        self._save_indexed_files()


def add_documents_to_index(documents, embedder, chroma_store):
    """将文档添加到向量索引"""
    if not documents:
        logger.info("没有需要添加的文档")
        return

    logger.info(f"开始处理 {len(documents)} 个文档...")

    # 文档分块
    all_chunks = []
    all_metadatas = []

    for doc in documents:
        content = doc["content"]
        metadata = doc["metadata"]

        # 分块
        chunks = chunk_document(content, chunk_size=800)
        logger.info(f"  {metadata['title'][:50]}: {len(chunks)} 个块")

        # 为每个块添加元数据
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_id"] = i
            all_metadatas.append(chunk_metadata)

    logger.info(f"总共 {len(all_chunks)} 个文档块")

    # 批量向量化
    logger.info("批量向量化文档...")
    batch_size = 10
    total_added = 0

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

        total_added += len(batch_chunks)

    logger.info(f"成功添加 {total_added} 个文档块到向量索引")


def incremental_update():
    """增量更新向量数据库"""
    logger.info("="*60)
    logger.info("开始增量更新向量数据库")
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

    # 2. 加载ChromaDB
    logger.info("\n[步骤2] 加载ChromaDB...")
    chroma_store = ChromaStore(
        dimension=1536,
        collection_name="knowledge_base",
        persist_directory="./vector_db/chroma_db"
    )
    current_count = chroma_store.collection.count()
    logger.info(f"ChromaDB加载完成，当前文档数: {current_count}")

    # 3. 检查新增或修改的文件
    logger.info("\n[步骤3] 检查新增或修改的文件...")
    indexer = IncrementalIndexer()
    new_files, modified_files = indexer.get_new_or_modified_files()

    logger.info(f"发现 {len(new_files)} 个新文件")
    logger.info(f"发现 {len(modified_files)} 个修改的文件")

    if not new_files and not modified_files:
        logger.info("没有需要更新的文件")
        return

    # 4. 处理新增和修改的文件
    logger.info("\n[步骤4] 处理文件...")
    all_files_to_process = new_files + modified_files

    documents = []
    for file_path, file_hash in all_files_to_process:
        try:
            logger.info(f"处理: {file_path.name}")

            # 读取文档内容（复用init_vector_db的逻辑）
            ext = file_path.suffix.lower()
            content = None
            doc_type = ext[1:]

            if ext in ['.md', '.txt']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

            elif ext == '.pdf':
                text_content, tables_content = extract_pdf_with_ocr(file_path)
                if text_content or tables_content:
                    content = f"{text_content}\n\n[表格数据]\n{tables_content}" if tables_content else text_content

            elif ext == '.docx':
                import docx
                doc = docx.Document(str(file_path))
                content = "\n\n".join([para.text for para in doc.paragraphs if para.text.strip()])

            elif ext == '.xlsx':
                content = load_excel_as_natural_language(file_path)

            elif ext == '.csv':
                content = load_csv_as_natural_language(file_path)

            elif ext == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    content = json.dumps(data, ensure_ascii=False, indent=2)

            if content:
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

                # 标记为已索引
                indexer.mark_as_indexed(file_path, file_hash)
                logger.info(f"  ✓ 成功处理 ({len(content)} 字符)")

        except Exception as e:
            logger.error(f"  ✗ 处理失败: {e}")

    # 5. 添加到向量索引
    logger.info(f"\n[步骤5] 添加 {len(documents)} 个文档到向量索引...")
    add_documents_to_index(documents, embedder, chroma_store)

    # 6. 验证
    new_count = chroma_store.collection.count()
    logger.info(f"\n[步骤6] 更新完成")
    logger.info(f"更新前文档数: {current_count}")
    logger.info(f"更新后文档数: {new_count}")
    logger.info(f"新增文档块数: {new_count - current_count}")

    logger.info("\n" + "="*60)
    logger.info("增量更新完成！")
    logger.info("="*60)


if __name__ == "__main__":
    incremental_update()
