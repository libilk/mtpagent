# -*- coding: utf-8 -*-
"""
向量相似度计算
==============

用于Agent路由的向量相似度匹配
"""

import logging
import numpy as np
from typing import List, Tuple, Dict, Any

logger = logging.getLogger(__name__)


class Embedder:
    """向量嵌入器"""

    def __init__(self, api_embedder=None):
        """
        初始化

        Args:
            api_embedder: API嵌入器实例（来自rag_core.api_embedder）
        """
        self.api_embedder = api_embedder

    def embed(self, text: str) -> List[float]:
        """
        文本转向量

        Args:
            text: 输入文本

        Returns:
            向量
        """
        if self.api_embedder:
            return self.api_embedder.embed_query(text)
        else:
            raise ValueError("未配置嵌入器")

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        批量文本转向量

        Args:
            texts: 文本列表

        Returns:
            向量列表
        """
        if self.api_embedder:
            return self.api_embedder.embed_documents(texts)
        else:
            raise ValueError("未配置嵌入器")


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    计算余弦相似度

    Args:
        vec1: 向量1
        vec2: 向量2

    Returns:
        相似度（0-1）
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    # 防御性处理：维度不一致时截断到较短维度
    if vec1.shape != vec2.shape:
        logger.warning(f"向量维度不一致: {vec1.shape} vs {vec2.shape}，截断到较短维度")
        min_dim = min(len(vec1), len(vec2))
        vec1 = vec1[:min_dim]
        vec2 = vec2[:min_dim]

    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))


def compute_similarities(
    query_vec: List[float],
    candidate_vecs: List[List[float]]
) -> List[float]:
    """
    计算查询向量与候选向量的相似度

    Args:
        query_vec: 查询向量
        candidate_vecs: 候选向量列表

    Returns:
        相似度列表
    """
    similarities = []
    for candidate_vec in candidate_vecs:
        sim = cosine_similarity(query_vec, candidate_vec)
        similarities.append(sim)
    return similarities


def rank_by_similarity(
    query_vec: List[float],
    candidates: List[Dict[str, Any]],
    candidate_vecs: List[List[float]],
    top_k: int = 3
) -> List[Tuple[Dict[str, Any], float]]:
    """
    根据相似度排序候选项

    Args:
        query_vec: 查询向量
        candidates: 候选项列表
        candidate_vecs: 候选项向量列表
        top_k: 返回前k个

    Returns:
        排序后的候选项列表（候选项, 相似度）
    """
    similarities = compute_similarities(query_vec, candidate_vecs)

    # 组合候选项和相似度
    ranked = list(zip(candidates, similarities))

    # 按相似度降序排序
    ranked.sort(key=lambda x: x[1], reverse=True)

    # 返回top-k
    return ranked[:top_k]


class VectorMatcher:
    """向量匹配器"""

    def __init__(self, embedder: Embedder):
        """
        初始化

        Args:
            embedder: 嵌入器实例
        """
        self.embedder = embedder
        self.index = {}  # {id: {"data": data, "vector": vector}}

    def add(self, id: str, data: Dict[str, Any], text: str):
        """
        添加项目

        Args:
            id: 项目ID
            data: 项目数据
            text: 用于嵌入的文本
        """
        vector = self.embedder.embed(text)
        self.index[id] = {
            "data": data,
            "vector": vector
        }
        logger.debug(f"添加项目: {id}")

    def search(self, query: str, top_k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        """
        搜索最相似的项目

        Args:
            query: 查询文本
            top_k: 返回前k个

        Returns:
            排序后的项目列表（项目数据, 相似度）
        """
        if not self.index:
            return []

        # 查询向量
        query_vec = self.embedder.embed(query)

        # 计算相似度
        results = []
        for id, item in self.index.items():
            sim = cosine_similarity(query_vec, item["vector"])
            results.append((item["data"], sim))

        # 排序
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    def clear(self):
        """清空索引"""
        self.index.clear()
