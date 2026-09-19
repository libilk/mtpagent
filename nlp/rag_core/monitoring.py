# -*- coding: utf-8 -*-
"""
监控模块（轻量级存根实现）

存根（stub）= 只有方法签名、没有真正实现的占位代码：先让调用点能跑通，
真正的对接（Prometheus、告警平台）留给以后。**所以这里的方法被调到，
不等于"监控在工作"。**

【逐项调用状态 —— 已核实，三个类冷热完全不同】
- `StructuredLogger`：**主路径真被调** —— knowledge_agent（agent.py:849 记查询、
  874 记错误）、database_agent（589/608）、customer_service_agent（1022/1041）。
  但实现只是把内容丢给 `logger.debug`，默认日志级别（INFO）下**一条都看不到**。
- `MetricsCollector`：**主路径真被调且真记账** —— `increment` / `record` 往内存字典
  累加，knowledge_agent 里大量调用（工具耗时、错误数、ReAct 轮次…），
  最后由 `get_all_stats` 汇总进 get_stats 报表。注意 `record_metric` 是 `pass`。
- `AlertManager`：只在降级路径 `_handle_with_optimizations` 末尾 check_alerts 一次
  （agent.py:1275），而方法体是空的 —— **等于不告警**。
"""
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class StructuredLogger:
    """结构化日志（存根）

    全部落到 logger.debug。默认级别 INFO 看不到它们，调试时得显式把
    本模块的日志级别调到 DEBUG —— 否则会误判"日志系统没生效"。
    **kwargs 的作用：调用方可以多传字段（cost、model 等）而不报错，
    "结构化"字段目前只被丢掉、没有真正写进日志。
    """

    def log_query(self, query: str = "", user_id: str = "", **kwargs):
        logger.debug(f"[query] user={user_id} query={query[:50]}")

    def log_retrieval(self, query: str = "", num_results: int = 0, latency_ms: float = 0, **kwargs):
        logger.debug(f"[retrieval] results={num_results} latency={latency_ms:.0f}ms")

    def log_generation(self, query: str = "", answer: str = "", latency_ms: float = 0, **kwargs):
        logger.debug(f"[generation] latency={latency_ms:.0f}ms")

    def log_error(self, error_type: str = "", message: str = "", **kwargs):
        logger.warning(f"[error] {error_type}: {message}")


class MetricsCollector:
    """指标收集器（半真半存根）

    increment / record 是真实现（内存字典累加，用 get_all_stats 取）；
    record_metric / AlertManager.* 是空实现 —— 调用方看着像记了指标，
    实际是静默丢弃。想知道某个指标有没有值，先看它走的是哪个方法。
    """

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
    """告警管理器（纯存根）

    两个方法都是空壳：check_alerts 收到指标后什么都不做（不判断阈值、不发通知），
    get_alert_history 恒返回空列表。调用方传进来的指标是被悄悄丢掉的 ——
    系统**目前没有任何告警能力**。
    """

    def check_alerts(self, metrics: Dict):
        pass

    def get_alert_history(self, limit: int = 10):
        return []


# 模块级单例：三个实例在 import 时就建好，全项目共用同一份。
# 为什么用单例而不是各 Agent 各 new 一个：指标要能跨 Agent 汇总成一份全局视图
# （get_all_stats 出总数），各记各的就没法统计"整个系统的错误率"了。
structured_logger = StructuredLogger()
metrics_collector = MetricsCollector()
alert_manager = AlertManager()
