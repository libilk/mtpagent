# -*- coding: utf-8 -*-
"""core.monitoring 存根

【为什么有这个文件】旧代码写 `from core.monitoring import global_monitor`，
但监控的**真正实现**在 rag_core/monitoring.py（结构化日志 + 计数器，各 agent 在用）。
这里留一个同名变量接住旧 import 路径，方法体全空 —— 只保证"不报 ImportError"。

【谁在用它】document_agent 仍从这里取 global_monitor。但注意下面这个类只实现了
record / increment / get_stats 三个空方法，**没有 performance / cost / quality 属性**；
document_agent 里 `self.monitor.performance.record(...)` 这类调用会抛 AttributeError，
被外层 except 吞掉。想要真监控得换 rag_core 那份，或接上 core/unified_monitoring.py。
"""
import logging

logger = logging.getLogger(__name__)


class GlobalMonitor:
    # 三个方法都是空操作：不计数、不记录，get_stats 恒为 {}。
    def record(self, *a, **kw): pass
    def increment(self, *a, **kw): pass
    def get_stats(self): return {}


global_monitor = GlobalMonitor()
