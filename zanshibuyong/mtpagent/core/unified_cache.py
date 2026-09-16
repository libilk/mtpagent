# -*- coding: utf-8 -*-
"""
统一缓存管理模块
================

提供统一的缓存管理，支持多种缓存类型
"""
import re
import hashlib
import time
import json
import logging
from typing import Any, Optional, Dict
from collections import OrderedDict

logger = logging.getLogger(__name__)


class UnifiedCache:
    """统一缓存基类（LRU + TTL策略）"""

    def __init__(self, cache_type: str = 'general', max_size: int = 1000, ttl: int = 3600):
        """
        初始化缓存

        Args:
            cache_type: 缓存类型（general/llm/query/embedding/retrieval/supplier）
            max_size: 最大缓存条目数
            ttl: 缓存过期时间（秒）
        """
        self.cache_type = cache_type
        self.max_size = max_size
        self.ttl = ttl
        self.cache = OrderedDict()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0
        }

    def _normalize_query(self, query: str) -> str:
        """
        查询归一化，提高缓存命中率

        处理：
        1. 转小写
        2. 去除多余空格
        3. 去除常见标点
        """
        # 转小写
        query = query.lower()
        # 去除多余空格
        query = re.sub(r'\s+', ' ', query).strip()
        # 去除常见标点（保留语义）
        query = re.sub(r'[,，.。!！?？、]', '', query)
        return query

    def _make_key(self, key: Any) -> str:
        """生成缓存键"""
        if isinstance(key, str):
            # 字符串类型做归一化处理
            return self._normalize_query(key)
        elif isinstance(key, dict):
            return hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()
        else:
            return hashlib.md5(str(key).encode()).hexdigest()

    def get(self, key: Any) -> Optional[Any]:
        """
        获取缓存

        Args:
            key: 缓存键

        Returns:
            缓存值，如果不存在或过期返回None
        """
        cache_key = self._make_key(key)

        if cache_key not in self.cache:
            self.stats["misses"] += 1
            return None

        # 支持两种格式：(value, timestamp) 或 (value, timestamp, ttl)
        cached_data = self.cache[cache_key]
        if len(cached_data) == 3:
            value, timestamp, ttl = cached_data
        else:
            value, timestamp = cached_data
            ttl = self.ttl

        # 检查是否过期
        if time.time() - timestamp > ttl:
            del self.cache[cache_key]
            self.stats["misses"] += 1
            return None

        # LRU：移到末尾
        self.cache.move_to_end(cache_key)
        self.stats["hits"] += 1
        return value

    def set(self, key: Any, value: Any, custom_ttl: Optional[int] = None):
        """
        设置缓存

        Args:
            key: 缓存键
            value: 缓存值
            custom_ttl: 自定义TTL（秒），不指定则使用默认TTL
        """
        cache_key = self._make_key(key)

        # 如果已存在，先删除
        if cache_key in self.cache:
            del self.cache[cache_key]

        # 使用自定义TTL或默认TTL
        ttl = custom_ttl if custom_ttl is not None else self.ttl

        # 添加新缓存（存储value, timestamp, ttl）
        self.cache[cache_key] = (value, time.time(), ttl)

        # LRU淘汰
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
            self.stats["evictions"] += 1

    def set_with_cost(self, key: Any, value: Any, cost_ms: float):
        """
        根据查询成本动态设置TTL

        Args:
            key: 缓存键
            value: 缓存值
            cost_ms: 查询耗时（毫秒）
        """
        # 根据成本计算TTL倍数
        if cost_ms > 5000:  # 超过5秒
            ttl_multiplier = 3
        elif cost_ms > 2000:  # 超过2秒
            ttl_multiplier = 2
        else:
            ttl_multiplier = 1

        custom_ttl = int(self.ttl * ttl_multiplier)
        self.set(key, value, custom_ttl=custom_ttl)

        logger.debug(f"[{self.cache_type}] 智能TTL: 耗时={cost_ms:.0f}ms, TTL={custom_ttl}s")

    def clear(self):
        """清空缓存"""
        self.cache.clear()
        logger.info(f"[{self.cache_type}] 缓存已清空")

    def get_stats(self) -> Dict:
        """获取缓存统计"""
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total if total > 0 else 0

        return {
            "type": self.cache_type,
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "evictions": self.stats["evictions"],
            "hit_rate": f"{hit_rate:.2%}"
        }


# ========== 专用缓存类 ==========

class LLMCache(UnifiedCache):
    """LLM调用缓存"""

    def __init__(self, max_size: int = 500, ttl: int = 3600):
        super().__init__(cache_type='llm', max_size=max_size, ttl=ttl)

    def get_cached_response(self, messages: list, model: str = None) -> Optional[str]:
        """获取缓存的LLM响应"""
        key = {
            'messages': messages,
            'model': model
        }
        return self.get(key)

    def cache_response(self, messages: list, response: str, model: str = None):
        """缓存LLM响应"""
        key = {
            'messages': messages,
            'model': model
        }
        self.set(key, response)


class QueryCache(UnifiedCache):
    """查询结果缓存"""

    def __init__(self, max_size: int = 1000, ttl: int = 1800):
        super().__init__(cache_type='query', max_size=max_size, ttl=ttl)


class EmbeddingCache(UnifiedCache):
    """向量嵌入缓存"""

    def __init__(self, max_size: int = 2000, ttl: int = 7200):
        super().__init__(cache_type='embedding', max_size=max_size, ttl=ttl)


class RetrievalCache(UnifiedCache):
    """检索结果缓存"""

    def __init__(self, max_size: int = 500, ttl: int = 1800):
        super().__init__(cache_type='retrieval', max_size=max_size, ttl=ttl)


class SupplierCache(UnifiedCache):
    """供应商历史缓存"""

    def __init__(self, max_size: int = 200, ttl: int = 3600):
        super().__init__(cache_type='supplier', max_size=max_size, ttl=ttl)


# ========== 向后兼容：旧的CacheManager接口 ==========

class CacheManager(UnifiedCache):
    """
    缓存管理器（向后兼容）

    保持与旧代码的兼容性
    """

    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        super().__init__(cache_type='general', max_size=max_size, ttl=ttl)


# ========== 全局缓存实例 ==========

# 为常用缓存提供全局实例
_global_llm_cache = None
_global_query_cache = None
_global_embedding_cache = None


def get_llm_cache() -> LLMCache:
    """获取全局LLM缓存实例"""
    global _global_llm_cache
    if _global_llm_cache is None:
        _global_llm_cache = LLMCache()
    return _global_llm_cache


def get_query_cache() -> QueryCache:
    """获取全局查询缓存实例"""
    global _global_query_cache
    if _global_query_cache is None:
        _global_query_cache = QueryCache()
    return _global_query_cache


def get_embedding_cache() -> EmbeddingCache:
    """获取全局向量缓存实例"""
    global _global_embedding_cache
    if _global_embedding_cache is None:
        _global_embedding_cache = EmbeddingCache()
    return _global_embedding_cache
