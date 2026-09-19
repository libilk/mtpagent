# -*- coding: utf-8 -*-
"""
持久化缓存
==========

基于Redis的持久化缓存，防止重启丢失。

【位置】四个 cache 文件里唯一的"继承者"：继承 unified_cache.UnifiedCache，
只重写 get/ set/ clear 三个方法，在父类的内存字典外再加一层 Redis。
选它还是选父类由 rag_core/cache_manager.py 决定 —— 有 Redis 用本类（跨重启保留），
没有就用父类的纯内存类，行为一致、只是重启后要重算。

【Redis 可选】redis_client=None 时本类退化成父类，只是多打一条 warning（见 __init__）。
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
        # 先查内存（内存命中率远高于 Redis，先问它最省事）
        cached = super().get(key)
        if cached is not None:
            return cached

        # 内存未命中，查Redis
        if self.redis_enabled:
            try:
                cache_key = self._make_key(key)
                redis_value = self.redis.get(f"{self.cache_type}:{cache_key}")

                if redis_value:
                    # 反直觉点：JSON 存进去再取出来，tuple 会变 list、int 键会变 str。
                    # 所以"Redis 命中"返回的值，类型可能和当初内存里算出来的不完全一样。
                    value = json.loads(redis_value)
                    # 回填内存缓存（让下次命中不必再走网络）
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
                # 反直觉点：Redis 的 keys(pattern) 会遍历整个库并阻塞其他请求，
                # 生产环境大库不宜这么清缓存；这里数据量小才可接受。
                pattern = f"{self.cache_type}:*"
                keys = self.redis.keys(pattern)
                if keys:
                    self.redis.delete(*keys)
                logger.info(f"[{self.cache_type}] Redis缓存已清空")
            except Exception as e:
                logger.error(f"[{self.cache_type}] Redis清空失败: {e}")
