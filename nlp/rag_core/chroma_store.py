"""
向量存储模块 - Chroma 实现
============================

使用 ChromaDB 实现高效的向量存储和相似度搜索。
支持持久化、元数据过滤和高级搜索策略。

向量库解决的是什么：向量化（embedding）把文本变成一串数字，但"找出最像的那几条"
如果每次都拿查询去和全库逐个算一遍，规模一大就没法用。向量数据库把向量**存下来**
并预先建好索引，让相似度搜索只算一小部分候选 —— 本文件就是这层"存储 + 检索"能力。

ChromaDB（向量数据库）= 存向量并做相似度搜索的库；collection（集合）约等于一张表，
本项目的库落在 ./vector_db/chroma_db，集合名 knowledge_base。

现役状态：langgraph_orchestrator/enhanced_entry.py 和 tools/scripts/ 下的建库脚本
都在用它 —— 这是**在跑**的代码，不是历史遗留。
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

    collection（集合）≈ 一张表，本类只操作一个集合。
    metadata（元数据）= 附在每条向量上的结构化字段（本项目存 doc_id / title 等），
    既用于展示，也能在查询时用 where 过滤缩小候选范围 —— 但现役调用没用到过滤。
    索引是 HNSW（近邻图索引）：用"近似"换速度，所以这里搜出来的是近似最近邻，不是精确穷举。
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
            persist_directory: 持久化（persistence，落盘后进程重启仍在）目录
            metric: 距离度量 ('cosine' 余弦, 'l2' 欧氏, 'ip' 内积) —— 只对新建的集合生效
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
                embedding_function=None  # 显式传 None：Chroma 自带一个默认嵌入模型，不传就会去加载它；传 None 表示向量由本项目自己算好传进来
            )
            logger.info(f"加载已存在的集合: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": distance_map.get(metric, 'cosine')},  # 度量在建集合时就定死。已有集合会走上面那个分支，这里的 metric 参数被静默忽略 —— 换 metric 要重建库
                embedding_function=None
            )
            logger.info(f"创建新集合: {collection_name}")

        # 进程内缓存：记录 ID → chunk（文本块，检索的最小单位）对象。
        # 只有本进程 add_vectors 过、或调用过 load()，缓存才有内容；
        # __init__ 直接挂上一个已有集合时这里是空的 —— 那时 search 只能退回用 Chroma 返回的纯文本。
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

        # 生成唯一 ID。序号取自本地缓存长度，缓存为空时会从 doc_0 重新开始，
        # 所以对"已有数据的库"直接用本方法追加会和旧记录重名（现役建库脚本是整库重建，没踩到）。
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

        这是全项目真正在跑的向量检索入口。query（查询）在这里已经是向量，
        本函数不做向量化 —— 算向量是调用方（embedder，向量化模型）的事。

        Args:
            query_vector: 查询向量
            top_k: 取前 k 条，是返回条数的上限
            filter_dict: 元数据过滤条件（如 {"doc_id": "xxx"}），None = 不过滤；现役调用没传
            min_score: 最低相似度阈值，低于此分数的文档将被过滤掉；现役调用传 0.0，改由上层再筛

        Returns:
            [(chunk, score, meta), ...] 列表
        """
        if self.collection.count() == 0:
            logger.warning("集合为空，无法搜索")
            return []

        # 执行查询
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, self.collection.count()),  # 不夹这一下的话，请求条数超过库里总数 Chroma 会直接报错
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

            # 循环里的 doc_id 是 Chroma 的记录 ID（doc_0 这种），不是元数据里那个知识库 doc_id，别混
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

                # 转换距离为相似度分数。反直觉点：Chroma 返回的是"距离"（越小越像），
                # 这里统一翻成 score（分数，越大越像），所以余弦相似度 = 1 - 距离。
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

        本仓库无调用方（和 delete_by_metadata 一样）：现役检索只走 search()。
        它做的是"只按 where 挑记录、不算相似度"，所以返回的 score 恒为 1.0，不是相关性。

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
        """清空所有数据（Chroma 没有"留集合、清内容"的操作，只能连集合一起删了再同名重建）"""
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

        # 重建本地缓存 —— 这是 load() 和 __init__ 的关键差别：load 会把已有记录读回来填进
        # chunks，之后 search 才能返回 chunk 对象；而 __init__ 的缓存是空的。
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
