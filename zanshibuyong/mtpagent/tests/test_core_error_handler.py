import pytest
from unittest.mock import MagicMock, call

from core.error_handler import (
    retry_llm_call,
    RetryStrategy,
    ErrorHandler,
    GracefulErrorHandler,
    FallbackStrategy,
)


class TestRetryLlmCall:
    def test_success_first_try(self):
        call_count = 0

        @retry_llm_call
        def ok_func():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = ok_func()
        assert result == "ok"
        assert call_count == 1

    def test_retry_then_success(self):
        call_count = 0

        @retry_llm_call
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("network error")
            return "recovered"

        result = flaky_func()
        assert result == "recovered"
        assert call_count == 3

    def test_retry_exhausted(self):
        call_count = 0

        @retry_llm_call
        def always_fail():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("always down")

        with pytest.raises(ConnectionError):
            always_fail()
        assert call_count == 3

    def test_non_retryable_error(self):
        call_count = 0

        @retry_llm_call
        def value_error_func():
            nonlocal call_count
            call_count += 1
            raise ValueError("bad input")

        with pytest.raises(ValueError):
            value_error_func()
        assert call_count == 1


class TestRetryStrategy:
    def test_execute_success(self):
        rs = RetryStrategy()
        result = rs.execute(lambda x: x * 2, 3)
        assert result == 6

    def test_custom_retries(self):
        rs = RetryStrategy(max_retries=2)
        call_count = 0

        def fail_twice():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ConnectionError()
            return "ok"

        result = rs.execute(fail_twice)
        assert result == "ok"
        assert call_count == 3

    def test_custom_backoff(self):
        rs = RetryStrategy(initial_delay=0.1, max_delay=0.5)
        call_count = 0

        def fail_once():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError()
            return "ok"

        result = rs.execute(fail_once)
        assert result == "ok"


class TestErrorHandler:
    def test_connection_error_fallback(self):
        eh = ErrorHandler()
        result = eh.handle(ConnectionError(), fallback_value="降级")
        assert result == "降级"

    def test_timeout_friendly_message(self):
        eh = ErrorHandler()
        result = eh.handle(TimeoutError())
        assert result is not None

    def test_value_error_fallback(self):
        eh = ErrorHandler()
        result = eh.handle(ValueError())
        assert result is not None

    def test_key_error_fallback(self):
        eh = ErrorHandler()
        result = eh.handle(KeyError())
        assert result is not None

    def test_unknown_error_fallback(self):
        eh = ErrorHandler()
        result = eh.handle(RuntimeError())
        assert result is not None

    def test_context_key_matches_friendly(self):
        eh = ErrorHandler()
        result = eh.handle(Exception(), context="timeout")
        assert result is not None

    def test_error_stats_accumulation(self):
        eh = ErrorHandler()
        eh.handle(ConnectionError(), context="query")
        eh.handle(ConnectionError(), context="query")
        eh.handle(ValueError(), context="query")
        stats = eh.get_error_stats()
        assert stats["query_ConnectionError"] == 2
        assert stats["query_ValueError"] == 1

    def test_graceful_error_handler_alias(self):
        assert GracefulErrorHandler is ErrorHandler


class TestFallbackStrategy:
    def test_first_succeeds(self):
        fs = FallbackStrategy()
        fs.add_fallback(lambda: "first")
        fs.add_fallback(lambda: "second")
        result = fs.execute()
        assert result == "first"

    def test_second_fallback(self):
        fs = FallbackStrategy()
        fs.add_fallback(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        fs.add_fallback(lambda: "second")
        result = fs.execute()
        assert result == "second"

    def test_all_fail(self):
        fs = FallbackStrategy()
        fs.add_fallback(lambda: (_ for _ in ()).throw(RuntimeError("fail1")))
        fs.add_fallback(lambda: (_ for _ in ()).throw(RuntimeError("fail2")))
        with pytest.raises(Exception, match="所有降级方案都失败"):
            fs.execute()
