import json
import pytest
from unittest.mock import patch, MagicMock

from core.persistent_cache import PersistentCache


class TestPersistentCache:
    def test_without_redis(self):
        pc = PersistentCache(redis_client=None)
        assert pc.redis_enabled is False

    def test_with_redis(self, mock_redis):
        pc = PersistentCache(redis_client=mock_redis)
        assert pc.redis_enabled is True

    def test_get_memory_hit(self, mock_redis):
        pc = PersistentCache(redis_client=mock_redis)
        pc.set("k", "v")
        # set 写入了 Redis
        assert mock_redis.setex.called
        mock_redis.reset_mock()
        result = pc.get("k")
        assert result == "v"
        # 内存命中，不再访问 Redis
        mock_redis.get.assert_not_called()

    def test_get_redis_hit_backfill(self, mock_redis):
        mock_redis.get.return_value = json.dumps("redis_value")
        pc = PersistentCache(redis_client=mock_redis)
        result = pc.get("k")
        assert result == "redis_value"
        # Memory backfill: second get should hit memory
        mock_redis.reset_mock()
        result2 = pc.get("k")
        assert result2 == "redis_value"
        mock_redis.get.assert_not_called()

    def test_get_both_miss(self, mock_redis):
        pc = PersistentCache(redis_client=mock_redis)
        result = pc.get("nonexistent")
        assert result is None

    def test_set_writes_both(self, mock_redis):
        pc = PersistentCache(redis_client=mock_redis)
        pc.set("k", "v")
        mock_redis.setex.assert_called_once()
        # memory also set
        assert pc.get("k") == "v"

    def test_set_custom_ttl(self, mock_redis):
        pc = PersistentCache(redis_client=mock_redis)
        pc.set("k", "v", custom_ttl=60)
        args = mock_redis.setex.call_args[0]
        assert args[1] == 60  # ttl

    def test_set_without_redis_no_error(self):
        pc = PersistentCache(redis_client=None)
        pc.set("k", "v")  # should not raise

    def test_get_redis_error_returns_none(self, mock_redis):
        mock_redis.get.side_effect = Exception("Redis down")
        pc = PersistentCache(redis_client=mock_redis)
        result = pc.get("k")
        assert result is None

    def test_set_redis_error_no_crash(self, mock_redis):
        mock_redis.setex.side_effect = Exception("Redis down")
        pc = PersistentCache(redis_client=mock_redis)
        pc.set("k", "v")  # should not raise
        # memory still set
        assert pc.get("k") == "v"

    def test_clear_both(self, mock_redis):
        mock_redis.keys.return_value = [b"persistent:a", b"persistent:b"]
        pc = PersistentCache(redis_client=mock_redis, cache_type="persistent")
        pc.set("a", 1)
        pc.set("b", 2)
        pc.clear()
        mock_redis.delete.assert_called_once()
        assert pc.get("a") is None

    def test_clear_without_redis_no_error(self):
        pc = PersistentCache(redis_client=None)
        pc.set("a", 1)
        pc.clear()
        assert pc.get("a") is None
