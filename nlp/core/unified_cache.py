# -*- coding: utf-8 -*-
"""
统一缓存管理模块
================

提供统一的缓存管理，支持多种缓存类型。

【四个 cache 文件是什么关系】读者最容易被这一堆同义文件名绕晕，先看这里：
- 本文件 = **统一入口**：定义基类 UnifiedCache（LRU + TTL）和几个具体缓存类。
  真正在跑的就是这里 —— llm_client 用 LLMCache，rag_core/cache_manager 用
  RetrievalCache / EmbeddingCache / QueryCache。
- [persistent_cache.py] = 继承本文件的 UnifiedCache，只是给 get/set 多加一层 Redis，
  换来"进程重启缓存还在"。rag_core/cache_manager 在 Redis 可用时选它，否则退回本文件的纯内存类。
- [semantic_cache.py] = **另一套独立实现**（不继承本文件），靠向量相似度命中"问法不同但意思一样"。
- [cache_manager.py] = 存根（stub），只转发 import，不干活。

【Redis 是可选的，不是必需依赖】外部传 redis_client=None 时，这里所有类都退化为纯内存，
进程重启缓存即丢。是否创建 Redis 由 enhanced_entry._create_redis_client() 读 REDIS_URL 决定，
读不到就传 None —— 所以没装 Redis 也能跑，只是重启后缓存冷掉。
"""

import hashlib
import re
import time
import json
import logging
from typing import Any, Optional, Dict
from collections import OrderedDict

logger = logging.getLogger(__name__)


class UnifiedCache:
    """统一缓存基类（LRU + TTL策略）

    LRU（最近最少使用）= 容量满时淘汰"最久没被访问"的那条，靠 OrderedDict 记访问顺序；
    TTL（存活时间，秒）= 一条缓存写入后最多留多久，过期即当作未命中。
    两者各管一头：TTL 管"太旧"，LRU 管"太多"。
    """

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

        为什么值得做：不归一化的话，"iPhone 15 多少钱？" 和 "iphone 15 多少钱"
        会各占一条键，缓存命中率直接腰斩。
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

        注意：命中后会把条目移到队尾 —— 这就是 LRU 的"刚刚用过"记号，
        淘汰时才能从队头挑出最久没用的那条。
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

        为什么按耗时定 TTL：一次查询越慢（越贵），越值得多缓存一会儿，
        用更长的 TTL 把这次成本摊薄。

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
    """LLM调用缓存（支持可选 Redis 持久化）

    两层：内存为准，Redis 兜底。内存未命中才查 Redis，查到后**回填内存** ——
    否则每次命中都得走一次网络，缓存反而变慢。
    redis_client 为 None 时只剩内存这一层（见模块头的降级说明）。
    """

    def __init__(self, max_size: int = 500, ttl: int = 3600, redis_client=None):
        super().__init__(cache_type='llm', max_size=max_size, ttl=ttl)
        self.redis = redis_client
        self.redis_enabled = redis_client is not None
        if self.redis_enabled:
            logger.info("[LLMCache] Redis 持久化已启用")

    def get_cached_response(self, messages: list, model: str = None) -> Optional[str]:
        """获取缓存的LLM响应"""
        key = {
            'messages': messages,
            'model': model
        }
        # 先查内存
        result = self.get(key)
        if result is not None:
            return result

        # 内存未命中，查 Redis
        if self.redis_enabled:
            try:
                cache_key = self._make_key(key)
                redis_value = self.redis.get(f"llm:{cache_key}")
                if redis_value:
                    # 回填内存
                    self.set(key, redis_value)
                    self.stats["hits"] += 1  # 修正统计（set 不计 hit）
                    logger.debug("[LLMCache] Redis 命中，回填内存")
                    return redis_value
            except Exception as e:
                logger.warning(f"[LLMCache] Redis 读取失败: {e}")

        return None

    def cache_response(self, messages: list, response: str, model: str = None):
        """缓存LLM响应"""
        key = {
            'messages': messages,
            'model': model
        }
        # 写入内存
        self.set(key, response)

        # 写入 Redis
        if self.redis_enabled:
            try:
                cache_key = self._make_key(key)
                self.redis.setex(
                    f"llm:{cache_key}",
                    self.ttl,
                    response  # LLM 响应是纯字符串，无需 JSON 序列化
                )
            except Exception as e:
                logger.warning(f"[LLMCache] Redis 写入失败: {e}")


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

    保持与旧代码的兼容性。

    事实核对：本仓库已无任何 import 方 —— 现在真正被用的是
    rag_core/cache_manager.py 里的同名 CacheManager（它内部组合了下面那几个具体缓存类）。
    这里留一份只为旧调用路径不报错。
    """

    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        super().__init__(cache_type='general', max_size=max_size, ttl=ttl)


# ========== 全局缓存实例 ==========

# 为常用缓存提供全局实例
# 事实核对：下面三个 getter 本仓库暂无调用方（预留的全局单例入口）——
# 现有代码都是各自 new 一个实例并自己传入 redis_client。
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
