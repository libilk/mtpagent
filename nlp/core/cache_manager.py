# -*- coding: utf-8 -*-
"""core.cache_manager 存根 — 委托给 rag_core.cache_manager

【为什么有这个文件】旧代码写的是 `from core.cache_manager import ...`，
实现后来搬到了 rag_core/cache_manager.py。为了不改动那些旧 import 路径、
又不维护两份实现，这里只做转发。

【真正干活的是谁】**第一行的 import**：CacheManager / CachedRetriever / CachedEmbedder
都来自 rag_core/cache_manager.py，那份才是有逻辑的（组合 unified_cache 里的类 + 可选 Redis）。

【本文件里下面三个类不是转发】**它们是自己定义的空壳**：构造什么都不做，
get 恒返回 None、set 什么都不存 —— 即"用了也不会缓存"。document_agent 曾 import 它们，
那些调用实际全部无效（其注释里也标了"事实核对"）。别把它们和第一行的同名转发混为一谈。
"""
from rag_core.cache_manager import CacheManager, CachedRetriever, CachedEmbedder


# ↓↓↓ 以下三个类是本文件自带的空壳，与上面转发的 rag_core 版本同名但无关系 ↓↓↓


class LLMCache:
    def __init__(self, cache_manager=None): pass
    def get(self, key): return None
    def set(self, key, value): pass


class RetrievalCache:
    def __init__(self, cache_manager=None): pass
    def get(self, key): return None
    def set(self, key, value): pass


class SupplierCache:
    def __init__(self, cache_manager=None): pass
    def get(self, key): return None
    def set(self, key, value): pass
