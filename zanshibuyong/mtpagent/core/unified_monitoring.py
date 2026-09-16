import time
import logging
import json
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
from datetime import datetime
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """性能监控器"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = defaultdict(lambda: deque(maxlen=window_size))
        self.counters = defaultdict(int)

    def record(self, metric_name: str, value: float):
        """记录指标"""
        self.metrics[metric_name].append({
            "value": value,
            "timestamp": time.time()
        })

    def increment(self, counter_name: str, value: int = 1):
        """增加计数器"""
        self.counters[counter_name] += value

    def get_stats(self, metric_name: str) -> Dict:
        """获取指标统计"""
        if metric_name not in self.metrics:
            return {}

        values = [m["value"] for m in self.metrics[metric_name]]
        if not values:
            return {}

        return {
            "count": len(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }


# ========== 成本追踪 ==========

class CostTracker:
    """成本追踪器"""

    # Token价格（元/1000 tokens）
    PRICES = {
        "qwen-max": {"input": 0.04, "output": 0.12},
        "qwen-plus": {"input": 0.004, "output": 0.012},
        "qwen-turbo": {"input": 0.002, "output": 0.006},
    }

    def __init__(self):
        self.costs = defaultdict(float)
        self.token_usage = defaultdict(lambda: {"input": 0, "output": 0})

    def track_llm_call(self, model: str, input_tokens: int, output_tokens: int):
        """追踪LLM调用成本"""
        if model not in self.PRICES:
            model = "qwen-plus"  # 默认

        input_cost = (input_tokens / 1000) * self.PRICES[model]["input"]
        output_cost = (output_tokens / 1000) * self.PRICES[model]["output"]
        total_cost = input_cost + output_cost

        self.costs[model] += total_cost
        self.token_usage[model]["input"] += input_tokens
        self.token_usage[model]["output"] += output_tokens

    def get_summary(self) -> Dict:
        """获取成本汇总"""
        total_cost = sum(self.costs.values())
        return {
            "total_cost": f"¥{total_cost:.4f}",
            "by_model": {
                model: {
                    "cost": f"¥{cost:.4f}",
                    "tokens": self.token_usage[model]
                }
                for model, cost in self.costs.items()
            }
        }


# ========== 质量监控 ==========

class QualityMonitor:
    """质量监控器"""

    def __init__(self):
        self.metrics = defaultdict(list)

    def record_extraction_quality(self, completeness: float, accuracy: float):
        """记录提取质量"""
        self.metrics["extraction_completeness"].append(completeness)
        self.metrics["extraction_accuracy"].append(accuracy)

    def record_risk_coverage(self, coverage: float):
        """记录风险覆盖度"""
        self.metrics["risk_coverage"].append(coverage)

    def get_metrics(self) -> Dict:
        """获取质量指标"""
        result = {}
        for metric_name, values in self.metrics.items():
            if values:
                result[metric_name] = {
                    "mean": sum(values) / len(values),
                    "count": len(values)
                }
        return result


# ========== 结构化日志 ==========

class StructuredLogger:
    """结构化日志记录器"""

    def __init__(self, service_name: str = "rag_system"):
        self.service_name = service_name

    def log(self, event: str, level: str = "INFO", **kwargs):
        """记录结构化日志"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "service": self.service_name,
            "event": event,
            "level": level,
            **kwargs
        }

        log_message = json.dumps(log_entry, ensure_ascii=False)

        if level == "ERROR":
            logger.error(log_message)
        elif level == "WARNING":
            logger.warning(log_message)
        else:
            logger.info(log_message)

    def log_error(self, event: str, **kwargs):
        """记录错误日志"""
        self.log(event, level="ERROR", **kwargs)

    def log_warning(self, event: str, **kwargs):
        """记录警告日志"""
        self.log(event, level="WARNING", **kwargs)

    def log_query(self, query: str, user_id: Optional[str] = None, **kwargs):
        """记录查询日志"""
        self.log("query", query=query, user_id=user_id, **kwargs)

    def log_retrieval(self, query: str, num_results: int, latency: float, **kwargs):
        """记录检索日志"""
        self.log("retrieval", query=query, num_results=num_results, latency=latency, **kwargs)


# ========== 告警管理 ==========

class AlertManager:
    """告警管理器"""

    def __init__(self):
        self.rules = []
        self.active_alerts = []

    def add_rule(self, name: str, condition: callable, message: str, severity: str = "warning"):
        """添加告警规则"""
        self.rules.append({
            "name": name,
            "condition": condition,
            "message": message,
            "severity": severity
        })

    def check_alerts(self, metrics: Dict) -> List[Dict]:
        """检查告警"""
        new_alerts = []
        for rule in self.rules:
            if rule["condition"](metrics):
                alert = {
                    "name": rule["name"],
                    "message": rule["message"],
                    "severity": rule["severity"],
                    "timestamp": datetime.utcnow().isoformat()
                }
                new_alerts.append(alert)
                logger.warning(f"[ALERT] {rule['name']}: {rule['message']}")

        self.active_alerts.extend(new_alerts)
        return new_alerts

    def get_active_alerts(self) -> List[Dict]:
        """获取活跃告警"""
        return self.active_alerts


# ========== 指标收集器 ==========

class MetricsCollector:
    """指标收集器"""

    def __init__(self):
        self.metrics = defaultdict(list)

    def collect(self, metric_name: str, value: Any, tags: Optional[Dict] = None):
        """收集指标"""
        self.metrics[metric_name].append({
            "value": value,
            "timestamp": time.time(),
            "tags": tags or {}
        })

    def record(self, metric_name: str, value: Any, tags: Optional[Dict] = None):
        """记录指标（collect 的别名）"""
        self.collect(metric_name, value, tags)

    def increment(self, counter_name: str, value: int = 1):
        """增加计数器"""
        if counter_name not in self.metrics:
            self.metrics[counter_name] = 0
        self.metrics[counter_name] += value

    def get_metrics(self, metric_name: str) -> List:
        """获取指标"""
        return self.metrics.get(metric_name, [])


# ========== 统一监控系统 ==========

class UnifiedMonitoring:
    """统一监控系统"""

    def __init__(self):
        # 性能监控
        self.performance = PerformanceMonitor()

        # 成本追踪
        self.cost = CostTracker()

        # 质量监控
        self.quality = QualityMonitor()

        # 结构化日志
        self.logger = StructuredLogger()

        # 告警管理
        self.alerts = AlertManager()

        # 指标收集
        self.metrics = MetricsCollector()

        logger.info("统一监控系统初始化完成")

    def get_dashboard(self) -> Dict:
        """获取监控仪表板"""
        return {
            "performance": {
                "llm_latency": self.performance.get_stats("llm_latency"),
                "retrieval_latency": self.performance.get_stats("retrieval_latency"),
            },
            "cost": self.cost.get_summary(),
            "quality": self.quality.get_metrics(),
            "alerts": self.alerts.get_active_alerts()
        }


# ========== 全局监控实例 ==========

global_monitor = UnifiedMonitoring()


# ========== 向后兼容 ==========

# 为旧代码提供兼容接口
structured_logger = global_monitor.logger
metrics_collector = global_monitor.metrics
alert_manager = global_monitor.alerts
