# -*- coding: utf-8 -*-
"""
缓存管理器
==========

为检索器和向量化器提供缓存包装
"""

import os
import logging
from typing import List, Dict, Any, Optional
from core.unified_cache import RetrievalCache, EmbeddingCache, QueryCache

logger = logging.getLogger(__name__)


def _create_redis_client():
    """尝试创建Redis客户端，失败则返回None"""
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    try:
        import redis
        client = redis.from_url(redis_url, decode_responses=True)
        client.ping()
        logger.info(f"Redis连接成功: {redis_url}")
        return client
    except ImportError:
        logger.warning("redis库未安装，降级为纯内存缓存")
        return None
    except Exception as e:
        logger.warning(f"Redis连接失败({e})，降级为纯内存缓存")
        return None


class CacheManager:
    """
    统一缓存管理器

    管理三层缓存：检索缓存、向量缓存、查询缓存
    支持Redis持久化，连接失败自动降级为纯内存缓存
    """

    def __init__(self, max_size: int = 1000, ttl: int = 3600, enable_redis: bool = True):
        """
        初始化缓存管理器

        Args:
            max_size: 缓存最大条目数
            ttl: 缓存过期时间（秒）
            enable_redis: 是否启用Redis持久化
        """
        # 尝试连接Redis
        redis_client = _create_redis_client() if enable_redis else None

        if redis_client:
            from core.persistent_cache import PersistentCache
            # 使用Redis持久化缓存
            self.retrieval_cache = PersistentCache(
                redis_client=redis_client, cache_type='retrieval', max_size=max_size, ttl=ttl
            )
            self.embedding_cache = PersistentCache(
                redis_client=redis_client, cache_type='embedding', max_size=max_size * 2, ttl=ttl * 2
            )
            self.query_cache = PersistentCache(
                redis_client=redis_client, cache_type='query', max_size=max_size, ttl=ttl // 2
            )
            logger.info("缓存管理器初始化完成（Redis持久化模式）")
        else:
            # 降级为纯内存缓存
            self.retrieval_cache = RetrievalCache(max_size=max_size, ttl=ttl)
            self.embedding_cache = EmbeddingCache(max_size=max_size * 2, ttl=ttl * 2)
            self.query_cache = QueryCache(max_size=max_size, ttl=ttl // 2)
            logger.info("缓存管理器初始化完成（纯内存模式）")

    def get_stats(self) -> Dict:
        """获取所有缓存的统计信息"""
        return {
            "retrieval": self.retrieval_cache.get_stats(),
            "embedding": self.embedding_cache.get_stats(),
            "query": self.query_cache.get_stats()
        }

    def clear_all(self):
        """清空所有缓存"""
        self.retrieval_cache.clear()
        self.embedding_cache.clear()
        self.query_cache.clear()
        logger.info("所有缓存已清空")


class CachedRetriever:
    """
    带缓存的检索器包装

    在底层检索器外包装一层缓存，相同查询直接返回缓存结果
    """

    def __init__(self, retriever, cache_manager: CacheManager):
        """
        初始化

        Args:
            retriever: 底层检索器（HybridRetriever等）
            cache_manager: 缓存管理器
        """
        self.retriever = retriever
        self.cache = cache_manager.retrieval_cache

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        检索文档（带缓存）

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            文档列表
        """
        # 生成缓存键（包含query和top_k）
        cache_key = f"{query}|{top_k}"

        # 尝试从缓存获取
        cached = self.cache.get(cache_key)
        if cached is not None:
            logger.debug(f"[检索缓存命中] {query[:50]}...")
            return cached

        # 缓存未命中，调用底层检索器
        logger.debug(f"[检索缓存未命中] {query[:50]}...")
        results = self.retriever.retrieve(query, top_k=top_k)

        # 存入缓存
        self.cache.set(cache_key, results)
        return results


class CachedEmbedder:
    """
    带缓存的向量化器包装

    在底层向量化器外包装一层缓存，相同文本直接返回缓存向量
    """

    def __init__(self, embedder, cache_manager: CacheManager):
        """
        初始化

        Args:
            embedder: 底层向量化器（APIEmbedder等）
            cache_manager: 缓存管理器
        """
        self.embedder = embedder
        self.cache = cache_manager.embedding_cache

    def embed(self, text: str) -> List[float]:
        """
        向量化文本（带缓存）

        Args:
            text: 文本

        Returns:
            向量
        """
        # 尝试从缓存获取
        cached = self.cache.get(text)
        if cached is not None:
            logger.debug(f"[向量缓存命中] {text[:50]}...")
            return cached

        # 缓存未命中，调用底层向量化器
        logger.debug(f"[向量缓存未命中] {text[:50]}...")
        embedding = self.embedder.embed(text)

        # 存入缓存
        self.cache.set(text, embedding)
        return embedding

    def encode(self, text: str) -> List[float]:
        """兼容不同的API命名（encode/embed）"""
        return self.embed(text)

    def encode_query(self, text: str) -> List[float]:
        """兼容 encode_query 调用"""
        return self.embed(text)
