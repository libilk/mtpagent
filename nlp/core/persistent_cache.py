# -*- coding: utf-8 -*-
"""
持久化缓存
==========

基于Redis的持久化缓存，防止重启丢失
"""

import json
import logging
from typing import Any, Optional
from core.unified_cache import UnifiedCache

logger = logging.getLogger(__name__)


class PersistentCache(UnifiedCache):
    """
    持久化缓存（内存 + Redis 两层）

    策略：
    1. 读取：先查内存，未命中再查Redis，命中后回填内存
    2. 写入：同时写内存和Redis
    3. Redis作为持久化层，重启后可恢复
    """

    def __init__(self, redis_client=None, **kwargs):
        """
        初始化持久化缓存

        Args:
            redis_client: Redis客户端实例（可选）
            **kwargs: 传递给UnifiedCache的参数
        """
        super().__init__(**kwargs)
        self.redis = redis_client
        self.redis_enabled = redis_client is not None

        if self.redis_enabled:
            logger.info(f"[{self.cache_type}] 持久化缓存已启用（Redis）")
        else:
            logger.warning(f"[{self.cache_type}] Redis未配置，降级为纯内存缓存")

    def get(self, key: Any) -> Optional[Any]:
        """
        获取缓存（先内存后Redis）

        Args:
            key: 缓存键

        Returns:
            缓存值
        """
        # 先查内存
        cached = super().get(key)
        if cached is not None:
            return cached

        # 内存未命中，查Redis
        if self.redis_enabled:
            try:
                cache_key = self._make_key(key)
                redis_value = self.redis.get(f"{self.cache_type}:{cache_key}")

                if redis_value:
                    value = json.loads(redis_value)
                    # 回填内存缓存
                    super().set(key, value)
                    logger.debug(f"[{self.cache_type}] Redis命中，回填内存")
                    return value
            except Exception as e:
                logger.error(f"[{self.cache_type}] Redis读取失败: {e}")

        return None

    def set(self, key: Any, value: Any, custom_ttl: Optional[int] = None):
        """
        设置缓存（同时写内存和Redis）

        Args:
            key: 缓存键
            value: 缓存值
            custom_ttl: 自定义TTL（秒），不指定则使用默认TTL
        """
        # 写入内存
        super().set(key, value, custom_ttl=custom_ttl)

        # 写入Redis
        if self.redis_enabled:
            try:
                cache_key = self._make_key(key)
                ttl = custom_ttl if custom_ttl is not None else self.ttl
                self.redis.setex(
                    f"{self.cache_type}:{cache_key}",
                    ttl,
                    json.dumps(value, ensure_ascii=False)
                )
            except Exception as e:
                logger.error(f"[{self.cache_type}] Redis写入失败: {e}")

    def clear(self):
        """清空缓存（内存 + Redis）"""
        super().clear()

        if self.redis_enabled:
            try:
                # 删除所有该类型的缓存
                pattern = f"{self.cache_type}:*"
                keys = self.redis.keys(pattern)
                if keys:
                    self.redis.delete(*keys)
                logger.info(f"[{self.cache_type}] Redis缓存已清空")
            except Exception as e:
                logger.error(f"[{self.cache_type}] Redis清空失败: {e}")
