from core.unified_monitoring import StructuredLogger, MetricsCollector, AlertManager,UnifiedMonitoring

structured_logger = StructuredLogger()
metrics_collector = MetricsCollector()
alert_manager = AlertManager()
global_monitor = UnifiedMonitoring()