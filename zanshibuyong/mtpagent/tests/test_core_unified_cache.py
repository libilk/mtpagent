import time
import pytest
from unittest.mock import patch

from core.unified_cache import (
    UnifiedCache,
    LLMCache,
    QueryCache,
    EmbeddingCache,
    RetrievalCache,
    SupplierCache,
    CacheManager,
    get_llm_cache,
    get_query_cache,
    get_embedding_cache,
)


class TestUnifiedCache:
    def test_set_and_get(self):
        cache = UnifiedCache()
        cache.set("key", "value")
        assert cache.get("key") == "value"

    def test_query_normalization_spaces(self):
        cache = UnifiedCache()
        cache.set("Hello  World!", "v")
        assert cache.get("hello world") == "v"

    def test_query_normalization_chinese_punctuation(self):
        cache = UnifiedCache()
        # 标点被移除后，"你好，世界！" 归一化为 "你好世界"
        cache.set("你好，世界！", "v")
        assert cache.get("你好世界") == "v"

    def test_dict_key_same_hash(self):
        cache = UnifiedCache()
        cache.set({"a": 1}, "v")
        assert cache.get({"a": 1}) == "v"

    def test_ttl_expiry(self):
        cache = UnifiedCache(ttl=0.01)
        cache.set("key", "value")
        time.sleep(0.02)
        assert cache.get("key") is None

    def test_custom_ttl(self):
        cache = UnifiedCache(ttl=3600)
        cache.set("k", "v", custom_ttl=0.01)
        time.sleep(0.02)
        assert cache.get("k") is None

    def test_lru_eviction(self):
        cache = UnifiedCache(max_size=2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    def test_lru_access_refreshes(self):
        cache = UnifiedCache(max_size=2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.get("a")  # 访问 a，a 变"新"
        cache.set("c", 3)  # 淘汰 b
        assert cache.get("a") == 1
        assert cache.get("b") is None
        assert cache.get("c") == 3

    def test_set_with_cost_low(self):
        cache = UnifiedCache(ttl=100)
        cache.set_with_cost("k", "v", cost_ms=100)
        cached = cache.cache[cache._make_key("k")]
        assert cached[2] == 100  # ttl unchanged

    def test_set_with_cost_medium(self):
        cache = UnifiedCache(ttl=100)
        cache.set_with_cost("k", "v", cost_ms=3000)
        cached = cache.cache[cache._make_key("k")]
        assert cached[2] == 200  # ttl * 2

    def test_set_with_cost_high(self):
        cache = UnifiedCache(ttl=100)
        cache.set_with_cost("k", "v", cost_ms=6000)
        cached = cache.cache[cache._make_key("k")]
        assert cached[2] == 300  # ttl * 3

    def test_get_stats(self):
        cache = UnifiedCache()
        cache.get("nonexistent1")
        cache.get("nonexistent2")
        cache.set("k", "v")
        cache.get("k")
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert "33.33%" in stats["hit_rate"]

    def test_clear(self):
        cache = UnifiedCache()
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()
        assert cache.get_stats()["size"] == 0

    def test_non_string_key(self):
        cache = UnifiedCache()
        cache.set(123, "number")
        assert cache.get(123) == "number"


class TestLLMCache:
    def test_cache_response_and_retrieve(self):
        cache = LLMCache()
        messages = [{"role": "user", "content": "hi"}]
        cache.cache_response(messages, "hello", model="qwen-plus")
        result = cache.get_cached_response(messages, model="qwen-plus")
        assert result == "hello"


class TestSpecializedCaches:
    def test_cache_types(self):
        assert QueryCache().cache_type == "query"
        assert EmbeddingCache().cache_type == "embedding"
        assert RetrievalCache().cache_type == "retrieval"
        assert SupplierCache().cache_type == "supplier"

    def test_default_params(self):
        assert QueryCache().ttl == 1800
        assert EmbeddingCache().ttl == 7200
        assert RetrievalCache().max_size == 500
        assert SupplierCache().max_size == 200


class TestCacheManager:
    def test_is_unified_cache(self):
        cm = CacheManager()
        assert cm.cache_type == "general"
        cm.set("k", "v")
        assert cm.get("k") == "v"


class TestGlobalCacheFactories:
    def test_llm_cache_singleton(self):
        import core.unified_cache as uc
        uc._global_llm_cache = None
        a = get_llm_cache()
        b = get_llm_cache()
        assert a is b

    def test_query_cache_singleton(self):
        import core.unified_cache as uc
        uc._global_query_cache = None
        a = get_query_cache()
        b = get_query_cache()
        assert a is b

    def test_embedding_cache_singleton(self):
        import core.unified_cache as uc
        uc._global_embedding_cache = None
        a = get_embedding_cache()
        b = get_embedding_cache()
        assert a is b
