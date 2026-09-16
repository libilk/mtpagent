"""
向量存储模块 - Chroma 实现
============================

使用 ChromaDB 实现高效的向量存储和相似度搜索。
支持持久化、元数据过滤和高级搜索策略。
"""

import os
import logging
from typing import List, Tuple, Optional, Dict, Any
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

__all__ = ['ChromaStore']


class ChromaStore:
    """
    向量存储：使用 ChromaDB 实现高效的向量检索

    支持的功能：
    1. 自动持久化（无需手动保存）
    2. 元数据过滤和管理
    3. 批量添加和增量更新
    4. 余弦相似度搜索
    5. 集合管理
    """

    def __init__(
        self,
        dimension: int,
        collection_name: str = "knowledge_base",
        persist_directory: str = "./chroma_db",
        metric: str = "cosine"
    ):
        """
        初始化 Chroma 向量存储

        Args:
            dimension: 向量维度（Chroma 会自动验证）
            collection_name: 集合名称
            persist_directory: 持久化目录
            metric: 距离度量 ('cosine', 'l2', 'ip')
        """
        self.dimension = dimension
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.metric = metric

        # 创建持久化目录
        os.makedirs(persist_directory, exist_ok=True)

        # 初始化 Chroma 客户端（持久化模式）
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,  # 禁用遥测
                allow_reset=True
            )
        )

        # 距离函数映射
        distance_map = {
            'cosine': 'cosine',
            'l2': 'l2',
            'ip': 'ip'
        }

        # 获取或创建集合
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=None  # 我们手动提供 embeddings
            )
            logger.info(f"加载已存在的集合: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": distance_map.get(metric, 'cosine')},
                embedding_function=None
            )
            logger.info(f"创建新集合: {collection_name}")

        # 存储文档块（用于返回完整信息）
        self.chunks = []
        self.metadata = []
        self.chunk_id_to_idx = {}

        logger.info(f"ChromaDB 初始化完成，持久化目录: {persist_directory}")

    def add_vectors(
        self,
        vectors: List[List[float]],
        chunks: List[Any],
        metadata: Optional[List[Dict]] = None
    ) -> None:
        """
        添加向量到 Chroma

        Args:
            vectors: 向量列表（可能是 numpy array 或 list）
            chunks: 文档块列表
            metadata: 元数据列表
        """
        # 处理 numpy array
        import numpy as np
        if isinstance(vectors, np.ndarray):
            if vectors.size == 0:
                logger.warning("没有向量需要添加")
                return
            vectors = vectors.tolist()
        elif not vectors or len(vectors) == 0:
            logger.warning("没有向量需要添加")
            return

        n = len(vectors)
        if metadata is None:
            metadata = [{} for _ in range(n)]

        # 生成唯一 ID
        start_idx = len(self.chunks)
        ids = [f"doc_{start_idx + i}" for i in range(n)]

        # 准备文档内容（Chroma 需要 documents 参数）
        documents = [chunk.text if hasattr(chunk, 'text') else str(chunk) for chunk in chunks]

        # 转换元数据为 Chroma 兼容格式（所有值必须是字符串、数字或布尔）
        chroma_metadata = []
        for meta in metadata:
            clean_meta = {}
            for key, value in meta.items():
                if isinstance(value, (str, int, float, bool)):
                    clean_meta[key] = value
                else:
                    clean_meta[key] = str(value)
            chroma_metadata.append(clean_meta)

        # 批量添加到 Chroma
        self.collection.add(
            ids=ids,
            embeddings=vectors,
            documents=documents,
            metadatas=chroma_metadata
        )

        # 更新本地缓存
        for i, chunk in enumerate(chunks):
            self.chunks.append(chunk)
            self.metadata.append(metadata[i])
            self.chunk_id_to_idx[ids[i]] = start_idx + i

        logger.info(f"添加了 {n} 个向量到 ChromaDB")

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict] = None,
        min_score: float = 0.0
    ) -> List[Tuple[Any, float]]:
        """
        搜索最相似的向量

        Args:
            query_vector: 查询向量
            top_k: 返回结果数量
            filter_dict: 元数据过滤条件
            min_score: 最低相似度阈值，低于此分数的文档将被过滤掉

        Returns:
            [(chunk, score, meta), ...] 列表
        """
        if self.collection.count() == 0:
            logger.warning("集合为空，无法搜索")
            return []

        # 执行查询
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, self.collection.count()),
            where=filter_dict,
            include=["distances", "metadatas", "documents"]
        )

        # 解析结果
        output = []
        filtered_count = 0
        if results['ids'] and len(results['ids'][0]) > 0:
            ids = results['ids'][0]
            distances = results['distances'][0]
            documents = results['documents'][0] if results.get('documents') else []
            metadatas = results['metadatas'][0] if results.get('metadatas') else []

            for i, (doc_id, distance) in enumerate(zip(ids, distances)):
                # 优先从 self.chunks 获取，如果不存在则从查询结果获取
                if doc_id in self.chunk_id_to_idx:
                    idx = self.chunk_id_to_idx[doc_id]
                    chunk = self.chunks[idx]
                elif i < len(documents):
                    chunk = documents[i]
                else:
                    continue

                # 获取元数据
                meta = metadatas[i] if i < len(metadatas) else {}

                # 转换距离为相似度分数
                if self.metric == 'cosine':
                    score = 1 - distance
                elif self.metric == 'l2':
                    score = 1 / (1 + distance)
                else:  # ip
                    score = distance

                # 过滤低于阈值的文档
                if score < min_score:
                    filtered_count += 1
                    continue

                output.append((chunk, score, meta))

        if filtered_count > 0:
            logger.info(f"[向量检索] 过滤了 {filtered_count} 个低相似度文档 (阈值: {min_score})")

        return output

    def search_by_metadata(
        self,
        filter_dict: Dict,
        top_k: int = 10
    ) -> List[Tuple[Any, float]]:
        """
        根据元数据过滤搜索

        Args:
            filter_dict: 过滤条件，如 {"source": "doc1.pdf"}
            top_k: 返回结果数量

        Returns:
            [(chunk, score), ...] 列表
        """
        results = self.collection.get(
            where=filter_dict,
            limit=top_k,
            include=["metadatas", "documents"]
        )

        output = []
        if results['ids']:
            for doc_id in results['ids']:
                if doc_id in self.chunk_id_to_idx:
                    idx = self.chunk_id_to_idx[doc_id]
                    output.append((self.chunks[idx], 1.0))

        return output

    def delete_by_metadata(self, filter_dict: Dict) -> int:
        """
        根据元数据删除文档

        Args:
            filter_dict: 过滤条件

        Returns:
            删除的文档数量
        """
        # 先查询要删除的 ID
        results = self.collection.get(
            where=filter_dict,
            include=[]
        )

        if not results['ids']:
            return 0

        # 删除
        self.collection.delete(ids=results['ids'])

        # 更新本地缓存
        deleted_count = 0
        for doc_id in results['ids']:
            if doc_id in self.chunk_id_to_idx:
                idx = self.chunk_id_to_idx[doc_id]
                del self.chunk_id_to_idx[doc_id]
                deleted_count += 1

        logger.info(f"删除了 {deleted_count} 个文档")
        return deleted_count

    def clear(self) -> None:
        """清空所有数据"""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.metric},
            embedding_function=None
        )
        self.chunks = []
        self.metadata = []
        self.chunk_id_to_idx = {}
        logger.info("已清空向量存储")

    def save(self, save_dir: str) -> None:
        """
        保存向量存储（Chroma 自动持久化，此方法保持接口兼容）

        Args:
            save_dir: 保存目录（忽略，使用初始化时的 persist_directory）
        """
        logger.info(f"ChromaDB 自动持久化到 {self.persist_directory}")

    @classmethod
    def load(cls, save_dir: str, collection_name: str = "knowledge_base") -> 'ChromaStore':
        """
        加载向量存储

        Args:
            save_dir: 加载目录
            collection_name: 集合名称

        Returns:
            ChromaStore 实例
        """
        # 创建客户端
        client = chromadb.PersistentClient(path=save_dir)

        # 获取集合
        try:
            collection = client.get_collection(name=collection_name)
        except:
            raise ValueError(f"集合 {collection_name} 不存在于 {save_dir}")

        # 获取集合信息
        count = collection.count()

        # 创建实例（维度从第一个向量推断）
        if count > 0:
            sample = collection.get(limit=1, include=["embeddings"])
            dimension = len(sample['embeddings'][0]) if sample['embeddings'] else 768
        else:
            dimension = 768  # 默认维度

        store = cls(
            dimension=dimension,
            collection_name=collection_name,
            persist_directory=save_dir
        )

        # 重建本地缓存
        all_data = collection.get(include=["metadatas", "documents"])
        if all_data['ids']:
            for i, doc_id in enumerate(all_data['ids']):
                # 创建简单的 chunk 对象
                class SimpleChunk:
                    def __init__(self, text, metadata):
                        self.text = text
                        self.metadata = metadata

                chunk = SimpleChunk(
                    all_data['documents'][i],
                    all_data['metadatas'][i]
                )
                store.chunks.append(chunk)
                store.metadata.append(all_data['metadatas'][i])
                store.chunk_id_to_idx[doc_id] = i

        logger.info(f"从 {save_dir} 加载了向量存储，包含 {count} 个向量")
        return store

    def get_statistics(self) -> Dict:
        """
        获取向量存储统计信息

        Returns:
            统计信息字典
        """
        return {
            'total_vectors': self.collection.count(),
            'dimension': self.dimension,
            'index_type': 'HNSW',  # Chroma 默认使用 HNSW
            'metric': self.metric,
            'persist_directory': self.persist_directory,
            'collection_name': self.collection_name
        }
