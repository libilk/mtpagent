# -*- coding: utf-8 -*-
"""
混合检索器
==========

结合向量检索和BM25检索，使用RRF算法融合结果

hybrid retrieval（混合检索）= 向量检索 + 关键词检索一起用，再合成一个列表。
为什么要两路：它们的强项正好互补 —— 向量管"意思相近"，BM25 管精确字符串
（型号/单号/条款编号）。任何一路单独用，都会在对方的强项上翻车。

RRF（倒数排名融合）= 只看"排第几"、不用原始分数来合成，理由见 _rrf_fusion。

⚠️ 接线现状（别高估本类的作用）：本类实例在 knowledge_agent 里被 CachedRetriever
包着，而那条链只在**降级路径** _handle_with_optimizations 才被走到；主路径 ReAct
的混合检索是自己内联实现 RRF 的（knowledge_agent.hybrid_search），不经过本类。
customer_service_agent 也建了本类，但显式 use_rerank=False。
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

    第 4 步的 reranker（重排序器）理想情况下是 cross-encoder（交叉编码器）：把
    「查询+文档」拼成一对一起送进模型打分，比"各自算出向量再比距离"更准，代价是
    慢很多 —— 所以只对第 3 步召回的少量候选做精排。注意 auto 档可能降级成纯规则
    打分（见 reranker.py），此时这一步并不真的"交叉编码"。
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
            vector_weight: 向量检索权重（0-1）—— 注意它不是"分数权重"：在 RRF 里
                只按排名倒数分配比例，BM25 那一路拿的是 1-vector_weight。
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

        第一阶段是 recall（召回）：两路各自多取一些（默认各 20）。这一步别急着砍，
        漏掉的东西后面重排序也救不回来；要缩小范围靠的是第二阶段融合、第三阶段精排。

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

        为什么用"排名倒数"而不是把两路分数直接相加：两路的分数量纲根本不同 ——
        BM25 分数无上界（可能到 8 分），向量余弦相似度被压在 0~1，直接相加会让 BM25
        整个把向量那一路压死。换成"排第几"后两边都成了 0~1 的排名分，天然可比。
        k（默认 60）= 平滑项，压低第 1 名（1/1）相对第 2 名（1/2）的悬殊程度。

        Args:
            vector_results: 向量检索结果
            bm25_results: BM25检索结果
            top_k: 返回结果数量
            k: RRF参数（通常为60）

        Returns:
            融合后的结果
        """
        # 使用文档内容作为唯一标识
        # 用"正文前 100 字符"而非 doc_id 来对齐两路结果：因为两路返回的字段里只有
        # content 一定都有。代价是前 100 字符相同的两篇文档会被当成同一篇合并掉。
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

        注意调用约定：reranker.rank() 返回的是"与入参同序"的分数列表，不是排好序的结果，
        所以这里要自己把分数写回文档、再重新排序（见下）。

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
