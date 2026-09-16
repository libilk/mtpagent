# -*- coding: utf-8 -*-
"""core.cache_manager 存根 — 委托给 rag_core.cache_manager"""
from rag_core.cache_manager import CacheManager, CachedRetriever, CachedEmbedder


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
