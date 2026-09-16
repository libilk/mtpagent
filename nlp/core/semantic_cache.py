# -*- coding: utf-8 -*-
"""
语义相似缓存
============

基于向量相似度的缓存匹配，提高命中率
支持可选 Redis 持久化（重启后可恢复缓存）
"""

import json
import logging
import numpy as np
from typing import Any, Optional, List, Tuple

logger = logging.getLogger(__name__)


class SemanticCache:
    """
    语义相似缓存

    不同于精确匹配，使用向量相似度判断缓存命中
    "iPhone 15 Pro的价格" 和 "iPhone 15 Pro多少钱" 可以命中同一缓存

    支持可选 Redis 持久化：
    - 写入时同步到 Redis（向量 + 文本 + 值 序列化存储）
    - 启动时从 Redis 恢复全部缓存条目
    - Redis 不可用时自动降级为纯内存
    """

    def __init__(
        self,
        embedder,
        similarity_threshold: float = 0.95,
        max_size: int = 500,
        ttl: int = 3600,
        redis_client=None,
    ):
        """
        初始化语义缓存

        Args:
            embedder: 向量化器
            similarity_threshold: 相似度阈值（0-1），超过此值视为命中
            max_size: 最大缓存条目数
            ttl: 缓存过期时间（秒）
            redis_client: Redis 客户端实例（可选，为 None 则纯内存）
        """
        self.embedder = embedder
        self.threshold = similarity_threshold
        self.max_size = max_size
        self.ttl = ttl

        # Redis 持久化
        self.redis = redis_client
        self.redis_enabled = redis_client is not None
        self._redis_key_prefix = "semantic_cache"
        self._redis_index_key = f"{self._redis_key_prefix}:index"

        # 缓存结构：[(query_vector, query_text, value, timestamp)]
        self.cache: List[Tuple[np.ndarray, str, Any, float]] = []

        self.stats = {"hits": 0, "misses": 0}

        # 从 Redis 恢复缓存
        if self.redis_enabled:
            self._restore_from_redis()

        mode = "Redis持久化" if self.redis_enabled else "纯内存"
        logger.info(f"语义缓存初始化完成 (阈值={similarity_threshold}, max_size={max_size}, 模式={mode})")

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        n1, n2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (n1 * n2))

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
        timestamp = time.time()

        # 添加到内存缓存
        self.cache.append((query_vec, query, value, timestamp))

        # LRU淘汰：超过容量删除最旧的
        if len(self.cache) > self.max_size:
            self.cache.pop(0)

        # 同步到 Redis
        if self.redis_enabled:
            self._persist_entry_to_redis(query_vec, query, value, timestamp)

    def _persist_entry_to_redis(self, vec: np.ndarray, query: str, value: Any, timestamp: float):
        """将单条缓存条目持久化到 Redis"""
        try:
            import hashlib
            # 用查询文本的 MD5 作为条目 key
            entry_id = hashlib.md5(query.encode()).hexdigest()
            entry_key = f"{self._redis_key_prefix}:entry:{entry_id}"

            entry_data = {
                "vector": vec.tolist(),
                "query": query,
                "value": value if isinstance(value, str) else json.dumps(value, ensure_ascii=False),
                "value_is_str": isinstance(value, str),
                "timestamp": timestamp,
            }

            self.redis.setex(
                entry_key,
                self.ttl,
                json.dumps(entry_data, ensure_ascii=False)
            )

            # 维护索引集合（用于启动时恢复）
            self.redis.sadd(self._redis_index_key, entry_id)

        except Exception as e:
            logger.warning(f"[语义缓存] Redis 持久化失败: {e}")

    def _restore_from_redis(self):
        """启动时从 Redis 恢复缓存"""
        try:
            import time

            # 获取所有条目 ID
            entry_ids = self.redis.smembers(self._redis_index_key)
            if not entry_ids:
                logger.info("[语义缓存] Redis 无历史缓存")
                return

            restored = 0
            expired_ids = []

            for entry_id in entry_ids:
                entry_key = f"{self._redis_key_prefix}:entry:{entry_id}"
                raw = self.redis.get(entry_key)

                if not raw:
                    # 条目已被 Redis TTL 淘汰，从索引中移除
                    expired_ids.append(entry_id)
                    continue

                try:
                    data = json.loads(raw)
                    vec = np.array(data["vector"])
                    query = data["query"]
                    timestamp = data["timestamp"]

                    # 检查是否过期（双重保险）
                    if time.time() - timestamp > self.ttl:
                        expired_ids.append(entry_id)
                        continue

                    # 恢复值
                    if data.get("value_is_str", True):
                        value = data["value"]
                    else:
                        value = json.loads(data["value"])

                    self.cache.append((vec, query, value, timestamp))
                    restored += 1

                except Exception as e:
                    logger.warning(f"[语义缓存] 恢复条目失败: {e}")
                    expired_ids.append(entry_id)

            # 清理过期索引
            if expired_ids:
                self.redis.srem(self._redis_index_key, *expired_ids)

            logger.info(f"[语义缓存] 从 Redis 恢复了 {restored} 条缓存")

        except Exception as e:
            logger.warning(f"[语义缓存] Redis 恢复失败，降级为空缓存: {e}")

    def clear(self):
        """清空缓存（内存 + Redis）"""
        self.cache.clear()

        if self.redis_enabled:
            try:
                # 删除所有条目
                entry_ids = self.redis.smembers(self._redis_index_key)
                if entry_ids:
                    entry_keys = [f"{self._redis_key_prefix}:entry:{eid}" for eid in entry_ids]
                    self.redis.delete(*entry_keys, self._redis_index_key)
                logger.info("[语义缓存] Redis 缓存已清空")
            except Exception as e:
                logger.warning(f"[语义缓存] Redis 清空失败: {e}")

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
            "threshold": self.threshold,
            "redis_enabled": self.redis_enabled,
        }
