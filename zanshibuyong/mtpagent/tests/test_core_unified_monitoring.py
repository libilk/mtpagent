import pytest
from unittest.mock import patch, MagicMock

from core.unified_monitoring import (
    PerformanceMonitor,
    CostTracker,
    QualityMonitor,
    StructuredLogger,
    AlertManager,
    MetricsCollector,
    UnifiedMonitoring,
)


class TestPerformanceMonitor:
    def test_record_and_get_stats(self):
        pm = PerformanceMonitor()
        pm.record("latency", 1.5)
        stats = pm.get_stats("latency")
        assert stats["count"] == 1
        assert stats["mean"] == 1.5
        assert stats["min"] == 1.5
        assert stats["max"] == 1.5

    def test_window_size_limit(self):
        pm = PerformanceMonitor(window_size=3)
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            pm.record("latency", v)
        stats = pm.get_stats("latency")
        assert stats["count"] == 3
        assert stats["min"] == 3.0
        assert stats["max"] == 5.0

    def test_increment(self):
        pm = PerformanceMonitor()
        pm.increment("errors")
        pm.increment("errors", 2)
        assert pm.counters["errors"] == 3

    def test_get_stats_nonexistent(self):
        pm = PerformanceMonitor()
        assert pm.get_stats("missing") == {}

    def test_get_stats_empty(self):
        pm = PerformanceMonitor()
        pm.metrics["empty"]  # access to create empty deque
        assert pm.get_stats("empty") == {}


class TestCostTracker:
    def test_track_llm_call(self):
        ct = CostTracker()
        ct.track_llm_call("qwen-plus", 1000, 500)
        assert ct.token_usage["qwen-plus"]["input"] == 1000
        assert ct.token_usage["qwen-plus"]["output"] == 500
        assert ct.costs["qwen-plus"] > 0

    def test_unknown_model_defaults(self):
        ct = CostTracker()
        ct.track_llm_call("unknown-model", 1000, 500)
        assert "qwen-plus" in ct.costs  # defaults to qwen-plus

    def test_get_summary(self):
        ct = CostTracker()
        ct.track_llm_call("qwen-plus", 1000, 500)
        ct.track_llm_call("qwen-max", 1000, 500)
        summary = ct.get_summary()
        assert "total_cost" in summary
        assert "by_model" in summary
        assert "qwen-plus" in summary["by_model"]
        assert "qwen-max" in summary["by_model"]


class TestQualityMonitor:
    def test_record_extraction(self):
        qm = QualityMonitor()
        qm.record_extraction_quality(0.9, 0.85)
        metrics = qm.get_metrics()
        assert metrics["extraction_completeness"]["mean"] == 0.9
        assert metrics["extraction_accuracy"]["mean"] == 0.85

    def test_record_risk_coverage(self):
        qm = QualityMonitor()
        qm.record_risk_coverage(0.8)
        metrics = qm.get_metrics()
        assert metrics["risk_coverage"]["mean"] == 0.8

    def test_empty_metrics(self):
        qm = QualityMonitor()
        assert qm.get_metrics() == {}


class TestStructuredLogger:
    def test_log_output(self):
        sl = StructuredLogger("test_svc")
        with patch("core.unified_monitoring.logger.info") as mock_info:
            sl.log("test_event", extra_field="extra_val")
            assert mock_info.called

    def test_log_query(self):
        sl = StructuredLogger()
        with patch("core.unified_monitoring.logger.info") as mock_info:
            sl.log_query("search query", user_id="u1")
            assert mock_info.called

    def test_log_retrieval(self):
        sl = StructuredLogger()
        with patch("core.unified_monitoring.logger.info") as mock_info:
            sl.log_retrieval("q", num_results=5, latency=0.5)
            assert mock_info.called


class TestAlertManager:
    def test_alert_triggered(self):
        am = AlertManager()
        am.add_rule("high_latency", lambda m: m.get("latency", 0) > 100, "延迟过高", "critical")
        alerts = am.check_alerts({"latency": 150})
        assert len(alerts) == 1
        assert alerts[0]["name"] == "high_latency"

    def test_no_alert(self):
        am = AlertManager()
        am.add_rule("high_latency", lambda m: m.get("latency", 0) > 100, "延迟过高")
        alerts = am.check_alerts({"latency": 50})
        assert len(alerts) == 0

    def test_get_active_alerts(self):
        am = AlertManager()
        am.add_rule("rule1", lambda m: True, "msg1")
        am.add_rule("rule2", lambda m: True, "msg2")
        am.check_alerts({})
        assert len(am.get_active_alerts()) == 2


class TestMetricsCollector:
    def test_collect_and_get(self):
        mc = MetricsCollector()
        mc.collect("cpu", 80, tags={"host": "srv1"})
        items = mc.get_metrics("cpu")
        assert len(items) == 1
        assert items[0]["value"] == 80
        assert items[0]["tags"] == {"host": "srv1"}

    def test_get_missing_metric(self):
        mc = MetricsCollector()
        assert mc.get_metrics("missing") == []


class TestUnifiedMonitoring:
    def test_get_dashboard_keys(self):
        um = UnifiedMonitoring()
        dashboard = um.get_dashboard()
        assert "performance" in dashboard
        assert "cost" in dashboard
        assert "quality" in dashboard
        assert "alerts" in dashboard
