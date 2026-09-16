# -*- coding: utf-8 -*-
"""
监控模块（轻量级存根实现）
"""
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class StructuredLogger:
    def log_query(self, query: str = "", user_id: str = "", **kwargs):
        logger.debug(f"[query] user={user_id} query={query[:50]}")

    def log_retrieval(self, query: str = "", num_results: int = 0, latency_ms: float = 0, **kwargs):
        logger.debug(f"[retrieval] results={num_results} latency={latency_ms:.0f}ms")

    def log_generation(self, query: str = "", answer: str = "", latency_ms: float = 0, **kwargs):
        logger.debug(f"[generation] latency={latency_ms:.0f}ms")

    def log_error(self, error_type: str = "", message: str = "", **kwargs):
        logger.warning(f"[error] {error_type}: {message}")


class MetricsCollector:
    def __init__(self):
        self._counters: Dict[str, int] = {}
        self._values: Dict[str, float] = {}

    def increment(self, key: str, value: int = 1):
        self._counters[key] = self._counters.get(key, 0) + value

    def record(self, key: str, value: float):
        self._values[key] = value

    def record_metric(self, key: str, value: Any):
        pass

    def get_all_stats(self) -> Dict:
        return {"counters": self._counters, "values": self._values}


class AlertManager:
    def check_alerts(self, metrics: Dict):
        pass

    def get_alert_history(self, limit: int = 10):
        return []


structured_logger = StructuredLogger()
metrics_collector = MetricsCollector()
alert_manager = AlertManager()
