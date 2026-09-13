import time
import pytest
from unittest.mock import MagicMock

from core.semantic_cache import SemanticCache


class TestSemanticCache:
    def test_exact_match_hit(self, mock_embedder):
        cache = SemanticCache(embedder=mock_embedder)
        cache.set("iPhone 价格", "答案是9999元")
        result = cache.get("iPhone 价格")
        assert result == "答案是9999元"
        assert cache.stats["hits"] == 1

    def test_semantic_similar_hit(self, mock_embedder):
        mock_embedder.embed.return_value = [0.1, 0.2, 0.3]
        mock_embedder.embed.side_effect = None  # 清除 side_effect，统一用 return_value
        cache = SemanticCache(embedder=mock_embedder, similarity_threshold=0.9)
        cache.set("iPhone 15 Pro的价格", "9999元")
        result = cache.get("iPhone 15 Pro多少钱")
        assert result == "9999元"

    def test_dissimilar_miss(self, mock_embedder):
        cache = SemanticCache(embedder=mock_embedder, similarity_threshold=0.95)
        cache.set("苹果价格", "5元")
        # 让 get 时返回完全不同的向量，确保不命中
        mock_embedder.embed.return_value = [-0.9, -0.8, -0.7]
        result = cache.get("电脑配置")
        assert result is None
        assert cache.stats["misses"] >= 1

    def test_custom_threshold(self, mock_embedder):
        cache = SemanticCache(embedder=mock_embedder, similarity_threshold=0.999)
        cache.set("hello world", "answer")
        # Same query but cosine distance from floating point can't be exactly 1.0
        result = cache.get("hello world")
        assert result == "answer"  # exact match should still work

    def test_ttl_expiry(self, mock_embedder):
        cache = SemanticCache(embedder=mock_embedder, ttl=0.01)
        cache.set("query", "answer")
        time.sleep(0.02)
        result = cache.get("query")
        assert result is None

    def test_lru_eviction(self, mock_embedder):
        cache = SemanticCache(embedder=mock_embedder, max_size=2)
        mock_embedder.embed.side_effect = None
        mock_embedder.embed.return_value = [0.1, 0.2, 0.3]
        cache.set("q1", "v1")
        # 使用不同向量以避免相似度命中（如果向量相同则可能命中第一条）
        mock_embedder.embed.return_value = [0.4, 0.5, 0.6]
        cache.set("q2", "v2")
        mock_embedder.embed.return_value = [0.7, 0.8, 0.9]
        cache.set("q3", "v3")
        assert len(cache.cache) == 2

    def test_get_stats(self, mock_embedder):
        cache = SemanticCache(embedder=mock_embedder)
        cache.get("miss1")
        cache.get("miss2")
        mock_embedder.embed.return_value = [0.1, 0.2, 0.3]
        cache.set("hit", "v")
        cache.get("hit")
        stats = cache.get_stats()
        assert stats["type"] == "semantic"
        assert stats["hits"] == 1
        assert stats["misses"] == 2

    def test_clear(self, mock_embedder):
        cache = SemanticCache(embedder=mock_embedder)
        cache.set("q", "v")
        cache.clear()
        assert len(cache.cache) == 0

    def test_empty_cache_get(self, mock_embedder):
        mock_embedder.embed.return_value = [0.1, 0.2, 0.3]
        cache = SemanticCache(embedder=mock_embedder)
        result = cache.get("anything")
        assert result is None
        assert cache.stats["misses"] == 1
