# -*- coding: utf-8 -*-
"""
错误处理与重试机制（统一版）
============================

基于 tenacity 提供统一的重试、降级和错误处理。
合并了 core/ 和 rag_core/ 两个版本，移除 Windows 不兼容的 SIGALRM 超时。
"""

import logging
from typing import Callable, Any, Optional, List
from functools import wraps

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

logger = logging.getLogger(__name__)


# ============================================================================
# 基于 tenacity 的重试装饰器
# ============================================================================

def retry_llm_call(func: Callable) -> Callable:
    """LLM 调用重试装饰器

    指数退避，最多 3 次重试，适用于 LLM API 调用。
    """
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


# ============================================================================
# RetryStrategy（兼容已有调用方式）
# ============================================================================

class RetryStrategy:
    """重试策略（基于 tenacity）"""

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
        max_delay: float = 10.0,
    ):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay

    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """执行函数（带重试）"""

        @retry(
            stop=stop_after_attempt(self.max_retries + 1),
            wait=wait_exponential(
                multiplier=self.initial_delay,
                min=self.initial_delay,
                max=self.max_delay,
            ),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        def _call():
            return func(*args, **kwargs)

        return _call()


# ============================================================================
# ErrorHandler（合并两个版本：按异常类型分发 + 用户友好消息）
# ============================================================================

class ErrorHandler:
    """统一错误处理器

    合并了 core/ 版本（异常类型分发 + 统计）和
    rag_core/ GracefulErrorHandler 版本（用户友好消息）。
    """

    # 用户友好的错误消息
    _FRIENDLY_MESSAGES = {
        "timeout": "抱歉，系统响应超时，请稍后再试。",
        "api_error": "抱歉，服务暂时不可用，请稍后再试。",
        "not_found": "抱歉，我在知识库中没有找到相关信息。",
        "invalid_input": "抱歉，您的输入格式不正确，请检查后重试。",
        "unknown": "抱歉，系统遇到了未知错误，请联系管理员。",
    }

    def __init__(self):
        self.error_counts: dict = {}

    def handle(
        self,
        error: Exception,
        context: str = "unknown",
        fallback_value: Any = None,
    ) -> Any:
        """处理错误：记录统计 + 按异常类型返回降级值

        Args:
            error: 异常对象
            context: 错误上下文 / 错误类型键
            fallback_value: 降级返回值

        Returns:
            降级返回值或用户友好消息
        """
        error_type = type(error).__name__
        key = f"{context}_{error_type}"
        self.error_counts[key] = self.error_counts.get(key, 0) + 1

        logger.error(f"[错误处理] 上下文={context}, 类型={error_type}, 消息={error}")

        # 如果 context 匹配友好消息键（兼容 GracefulErrorHandler 用法）
        if context in self._FRIENDLY_MESSAGES:
            return fallback_value or self._FRIENDLY_MESSAGES[context]

        # 按异常类型分发
        if isinstance(error, (ConnectionError, TimeoutError)):
            logger.warning("[错误处理] 网络错误，返回降级值")
            return fallback_value or self._FRIENDLY_MESSAGES["timeout"]
        elif isinstance(error, ValueError):
            logger.warning("[错误处理] 值错误，返回降级值")
            return fallback_value or "数据格式错误"
        elif isinstance(error, KeyError):
            logger.warning("[错误处理] 键错误，返回降级值")
            return fallback_value or "缺少必要字段"
        else:
            logger.warning("[错误处理] 未知错误，返回降级值")
            return fallback_value or self._FRIENDLY_MESSAGES["unknown"]

    def get_error_stats(self) -> dict:
        """获取错误统计"""
        return self.error_counts.copy()


# 兼容别名：原 rag_core 版本的 GracefulErrorHandler
GracefulErrorHandler = ErrorHandler


# ============================================================================
# FallbackStrategy（链式降级，从 rag_core 版本迁移）
# ============================================================================

class FallbackStrategy:
    """降级策略：按顺序尝试多个降级方案，返回第一个成功的结果。"""

    def __init__(self):
        self.fallback_chain: List[Callable] = []

    def add_fallback(self, func: Callable):
        """添加降级方案"""
        self.fallback_chain.append(func)

    def execute(self, *args, **kwargs) -> Any:
        """执行降级链"""
        last_exception = None

        for i, fallback_func in enumerate(self.fallback_chain):
            try:
                logger.info(f"[降级] 尝试降级方案 {i + 1}/{len(self.fallback_chain)}")
                result = fallback_func(*args, **kwargs)
                logger.info(f"[降级] 降级方案 {i + 1} 成功")
                return result
            except Exception as e:
                last_exception = e
                logger.warning(f"[降级] 降级方案 {i + 1} 失败: {e}")

        logger.error("[降级] 所有降级方案都失败")
        raise Exception("所有降级方案都失败") from last_exception
