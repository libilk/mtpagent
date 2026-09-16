# -*- coding: utf-8 -*-
"""
混合检索器
==========

结合向量检索和BM25检索，使用RRF算法融合结果
"""

import logging
from typing import List, Dict, Tuple, Optional
import numpy as np

# 从重排序模块导入（保持向后兼容）
from rag_core.reranker import SimpleReranker, RerankerFactory

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    混合检索器：结合向量检索和BM25检索

    策略：
    1. 向量检索：捕捉语义相似度
    2. BM25检索：捕捉关键词匹配
    3. RRF融合：Reciprocal Rank Fusion
    4. 重排序：使用交叉编码器精细打分
    """

    def __init__(
        self,
        vector_retriever=None,
        bm25_retriever=None,
        vector_weight: float = 0.5,
        use_rerank: bool = False,
        reranker=None
    ):
        """
        初始化混合检索器

        Args:
            vector_retriever: 向量检索器
            bm25_retriever: BM25检索器
            vector_weight: 向量检索权重（0-1）
            use_rerank: 是否使用重排序
            reranker: 重排序器
        """
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.vector_weight = vector_weight
        self.use_rerank = use_rerank
        self.reranker = reranker

        logger.info(f"混合检索器初始化完成 (向量权重: {vector_weight})")

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        vector_top_k: int = 20,
        bm25_top_k: int = 20
    ) -> List[Dict]:
        """
        混合检索

        Args:
            query: 查询文本
            top_k: 最终返回结果数量
            vector_top_k: 向量检索召回数量
            bm25_top_k: BM25检索召回数量

        Returns:
            检索结果列表
        """
        logger.info(f"[混合检索] 查询: '{query}'")

        # 第一阶段：多路召回
        vector_results = []
        bm25_results = []

        if self.vector_retriever:
            logger.info(f"  [向量检索] 召回 top-{vector_top_k}")
            vector_results = self.vector_retriever.retrieve(query, top_k=vector_top_k)

        if self.bm25_retriever:
            logger.info(f"  [BM25检索] 召回 top-{bm25_top_k}")
            bm25_results = self.bm25_retriever.retrieve(query, top_k=bm25_top_k)

        # 第二阶段：RRF融合
        logger.info(f"  [RRF融合] 融合检索结果")
        fused_results = self._rrf_fusion(
            vector_results,
            bm25_results,
            top_k=top_k * 2 if self.use_rerank else top_k
        )

        # 第三阶段：重排序（可选）
        if self.use_rerank and self.reranker and len(fused_results) > 0:
            logger.info(f"  [重排序] 使用交叉编码器重排序")
            fused_results = self._rerank(query, fused_results, top_k=top_k)
        else:
            fused_results = fused_results[:top_k]

        logger.info(f"[混合检索] 返回 {len(fused_results)} 个结果")
        return fused_results

    def _rrf_fusion(
        self,
        vector_results: List[Dict],
        bm25_results: List[Dict],
        top_k: int = 10,
        k: int = 60
    ) -> List[Dict]:
        """
        RRF (Reciprocal Rank Fusion) 融合算法

        公式：RRF_score = Σ 1/(k + rank_i)

        Args:
            vector_results: 向量检索结果
            bm25_results: BM25检索结果
            top_k: 返回结果数量
            k: RRF参数（通常为60）

        Returns:
            融合后的结果
        """
        # 使用文档内容作为唯一标识
        doc_scores = {}
        doc_map = {}

        # 向量检索结果打分
        for rank, doc in enumerate(vector_results):
            content = doc.get("content", "")
            doc_id = content[:100]  # 使用前100字符作为ID

            score = self.vector_weight / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + score
            doc_map[doc_id] = doc

        # BM25检索结果打分
        for rank, doc in enumerate(bm25_results):
            content = doc.get("content", "")
            doc_id = content[:100]

            score = (1 - self.vector_weight) / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + score

            if doc_id not in doc_map:
                doc_map[doc_id] = doc

        # 按分数排序
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # 返回top-k结果
        results = []
        for doc_id, score in sorted_docs[:top_k]:
            doc = doc_map[doc_id].copy()
            doc["rrf_score"] = score
            results.append(doc)

        return results

    def _rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 3
    ) -> List[Dict]:
        """
        使用重排序模型对候选结果重新打分

        Args:
            query: 查询文本
            candidates: 候选结果
            top_k: 返回结果数量

        Returns:
            重排序后的结果
        """
        if not self.reranker or not candidates:
            return candidates[:top_k]

        # 提取文档内容
        docs = [doc.get("content", "") for doc in candidates]

        # 重排序打分
        scores = self.reranker.rank(query, docs)

        # 更新分数并排序
        for i, doc in enumerate(candidates):
            doc["rerank_score"] = scores[i]

        # 按重排序分数排序
        reranked = sorted(candidates, key=lambda x: x.get("rerank_score", 0), reverse=True)

        return reranked[:top_k]

# SimpleReranker 已迁移到 rag_core/reranker.py
# 通过顶部 import 保持向后兼容：from rag_core.reranker import SimpleReranker
