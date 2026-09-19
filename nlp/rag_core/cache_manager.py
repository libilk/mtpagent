# -*- coding: utf-8 -*-
"""
缓存管理器
==========

为检索器和向量化器提供缓存包装

【三份"缓存"文件的分工 —— 先看这里，别在名字里迷路】
- **本文件 = 有逻辑的那份**：CacheManager 决定"用 Redis 持久化版还是纯内存版"，
  CachedRetriever / CachedEmbedder 是两个包装器。缓存算法本身不在本文件。
- [core/unified_cache.py] = 纯内存缓存类的真正实现（LRU 淘汰 + TTL 过期）。
  Redis 可用时改用 [core/persistent_cache.py]（继承前者、多一层 Redis，
  换来"进程重启缓存还在"）。
- [core/cache_manager.py] = 存根，只把本文件的三个类转发出去（兼容旧 import 路径）；
  它自己另外定义的同名 LLMCache/RetrievalCache/SupplierCache 是**空壳**（get 恒返回 None）。

【调用状态 —— 三层缓存冷热不均，已核实】
- `CacheManager` 主路径真接线：knowledge_agent（agent.py:278）、database_agent（117）、
  customer_service_agent（151）都用它的 `query_cache` 在 handle 开头查/写答案。
- `CachedRetriever` 只用于降级路径：agent.py:283 构造，唯一调用在 1212。
- `CachedEmbedder` 装进了主路径（agent.py:289-290 把 retriever 内部的 embedder
  原地替换成它），**但缓存实际被绕过** —— 原因见 CachedEmbedder 的类说明。
"""

import os
import logging
from typing import List, Dict, Any, Optional
from core.unified_cache import RetrievalCache, EmbeddingCache, QueryCache

logger = logging.getLogger(__name__)


def _create_redis_client():
    """尝试创建Redis客户端，失败则返回None

    这里**不把 Redis 当必需依赖**：连不上就返回 None，由调用方退回纯内存缓存。
    Redis（内存数据库）= 常用作缓存，重启不一定丢（可持久化）。
    """
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    try:
        import redis
        # decode_responses=True：取出来的值是 str 而不是 bytes —— 否则缓存键的
        # 比较、拼接都会踩 bytes/str 混用的坑
        client = redis.from_url(redis_url, decode_responses=True)
        # 必须 ping：from_url 只是构造客户端、不真连，错误会推迟到第一次用时才炸。
        # 提前 ping 一下，才能在这一刻就决定"到底用不用 Redis"
        client.ping()
        logger.info(f"Redis连接成功: {redis_url}")
        return client
    except ImportError:
        # 分了两种异常是为了让日志能区分病因：没装库 vs 装了但连不上
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

    为什么分三层而不是一个大缓存：三层的"贵"和"新鲜度要求"不一样。
    向量化最贵（要调远程 API）→ 给最大容量和最长 TTL；查询答案最易过时
    （知识库一更新就作废）→ TTL 最短。分层的意义是让每层各自过期，
    而不是互相牵连。
    """

    def __init__(self, max_size: int = 1000, ttl: int = 3600, enable_redis: bool = True,
                 redis_client=None):
        """
        初始化缓存管理器

        Args:
            max_size: 缓存最大条目数
            ttl: 缓存过期时间（秒）
            enable_redis: 是否启用Redis持久化
            redis_client: 外部传入的 Redis 客户端（优先使用，避免重复连接）
        """
        # 优先使用外部传入的 redis_client，否则尝试自建。
        # 之所以允许外部传：多个 Agent 各建一条 Redis 连接纯属浪费，
        # 上层建一条、大家共用就行（enhanced_entry 就是这么做的）
        if redis_client:
            self.redis_client = redis_client
        else:
            self.redis_client = _create_redis_client() if enable_redis else None

        if self.redis_client:
            from core.persistent_cache import PersistentCache
            # 使用Redis持久化缓存
            # 三层参数故意不一致（下面纯内存分支同款）：向量条目更大更值钱 →
            # 容量与 TTL 都 ×2；查询结果最易过时 → TTL 减半。这些差值就是设计意图本身
            self.retrieval_cache = PersistentCache(
                redis_client=self.redis_client, cache_type='retrieval', max_size=max_size, ttl=ttl
            )
            self.embedding_cache = PersistentCache(
                redis_client=self.redis_client, cache_type='embedding', max_size=max_size * 2, ttl=ttl * 2
            )
            self.query_cache = PersistentCache(
                redis_client=self.redis_client, cache_type='query', max_size=max_size, ttl=ttl // 2
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

    ⚠️ 缓存不会随知识库更新自动失效：新文档入库后，同样的查询仍会命中旧结果，
    得靠 api.py 的清缓存入口（最终调 cache_manager.clear_all）手动清。
    做"用装饰器自动缓存"式的改造时，这是最容易漏的一条。
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
        # top_k 必须进键：同一查询取 3 条和取 10 条是两份不同结果，
        # 键里不带就会互相"串味"（返回条数与请求不符）
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

    ⚠️ 两条反直觉的事实（已核实，读这个类必须知道）：
    1. **只有 embed / encode 单条走缓存**。encode_query / encode_texts 会直接转发
       给底层 embedder（只要它有同名方法 —— APIEmbedder 两个都有），批量路径、
       查询路径**完全不过缓存**。
    2. 现役接线里，检索器调用的恰好是 encode_query（enhanced_entry.py:847 的
       SimpleRetriever.retrieve），所以 agent.py:290 那次替换是"装上了但缓存不生效"。

    为什么每个方法名都要单独转发、而不是统一走 embed：不同 embedder 的 API 命名
    不一（embed / encode / encode_query / encode_texts）。包装后必须"长得和底层一样"，
    否则替换掉别人的对象之后，调用方按原方法名调用会直接 AttributeError。
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
        # 同一件事两个名字都留：调用方习惯用哪个都能跑通，包装才不会被察觉
        return self.embed(text)

    def encode_query(self, query: str):
        """编码单个查询（委托给底层embedder，保留原始返回类型）

        ⚠️ 转发给底层 = 不过缓存（见类说明）；走 embed 才会缓存。
        保留原始返回类型（numpy 数组 vs list）也是刻意的：上层可能直接对它调 .tolist()
        """
        if hasattr(self.embedder, 'encode_query'):
            return self.embedder.encode_query(query)
        return self.embed(query)

    def encode_texts(self, texts, **kwargs):
        """批量编码文本（委托给底层embedder，保留原始返回类型）

        ⚠️ 同上，这条路也不经缓存；底层的批量接口通常自带批处理优化，
        逐条丢给 self.embed 反而更慢，所以这里优先转发
        """
        if hasattr(self.embedder, 'encode_texts'):
            return self.embedder.encode_texts(texts, **kwargs)
        import numpy as np
        return np.array([self.embed(t) for t in texts])
