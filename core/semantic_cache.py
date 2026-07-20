# -*- coding: utf-8 -*-
"""
语义相似缓存
============

基于向量相似度的缓存匹配，提高命中率
"""

import logging
import numpy as np
from typing import Any, Optional, List, Tuple

logger = logging.getLogger(__name__)


class SemanticCache:
    """
    语义相似缓存

    不同于精确匹配，使用向量相似度判断缓存命中
    "iPhone 15 Pro的价格" 和 "iPhone 15 Pro多少钱" 可以命中同一缓存
    """

    def __init__(
        self,
        embedder,
        similarity_threshold: float = 0.95,
        max_size: int = 500,
        ttl: int = 3600
    ):
        """
        初始化语义缓存

        Args:
            embedder: 向量化器
            similarity_threshold: 相似度阈值（0-1），超过此值视为命中
            max_size: 最大缓存条目数
            ttl: 缓存过期时间（秒）
        """
        self.embedder = embedder
        self.threshold = similarity_threshold
        self.max_size = max_size
        self.ttl = ttl

        # 缓存结构：[(query_vector, query_text, value, timestamp)]
        self.cache: List[Tuple[np.ndarray, str, Any, float]] = []

        self.stats = {"hits": 0, "misses": 0}

        logger.info(f"语义缓存初始化完成 (阈值={similarity_threshold}, max_size={max_size})")

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def get(self, query: str) -> Optional[Any]:
        """
        获取缓存（基于语义相似度）

        Args:
            query: 查询文本

        Returns:
            缓存值，未命中返回None
        """
        import time

        # 向量化查询
        query_vec = np.array(self.embedder.embed(query))

        # 遍历缓存，找最相似的
        best_similarity = 0
        best_value = None
        best_idx = None

        for idx, (cached_vec, cached_query, value, timestamp) in enumerate(self.cache):
            # 检查是否过期
            if time.time() - timestamp > self.ttl:
                continue

            # 计算相似度
            similarity = self._cosine_similarity(query_vec, cached_vec)

            if similarity > best_similarity:
                best_similarity = similarity
                best_value = value
                best_idx = idx

        # 判断是否命中
        if best_similarity >= self.threshold:
            self.stats["hits"] += 1
            logger.debug(f"[语义缓存命中] 相似度={best_similarity:.3f}, 查询='{query[:50]}'")
            return best_value
        else:
            self.stats["misses"] += 1
            logger.debug(f"[语义缓存未命中] 最高相似度={best_similarity:.3f}")
            return None

    def set(self, query: str, value: Any):
        """
        设置缓存

        Args:
            query: 查询文本
            value: 缓存值
        """
        import time

        # 向量化查询
        query_vec = np.array(self.embedder.embed(query))

        # 添加到缓存
        self.cache.append((query_vec, query, value, time.time()))

        # LRU淘汰：超过容量删除最旧的
        if len(self.cache) > self.max_size:
            self.cache.pop(0)

    def clear(self):
        """清空缓存"""
        self.cache.clear()
        logger.info("语义缓存已清空")

    def get_stats(self):
        """获取统计信息"""
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total if total > 0 else 0

        return {
            "type": "semantic",
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "hit_rate": f"{hit_rate:.2%}",
            "threshold": self.threshold
        }
