# -*- coding: utf-8 -*-
"""core.monitoring 存根"""
import logging

logger = logging.getLogger(__name__)


class GlobalMonitor:
    def record(self, *a, **kw): pass
    def increment(self, *a, **kw): pass
    def get_stats(self): return {}


global_monitor = GlobalMonitor()
