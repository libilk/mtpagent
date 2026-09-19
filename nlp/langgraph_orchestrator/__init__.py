# -*- coding: utf-8 -*-
"""
LangGraph Orchestrator（图编排器）
=================================

LangGraph（图编排框架）：把多步骤流程画成图的框架 —— 节点干活，边决定下一步走谁。
本包就是这套多 Agent RAG（检索增强生成）系统的主编排层：一次查询进来后，
由 enhanced_graph 连成的图决定走哪个 Agent、要不要重试、要不要停下来等真人。

这里导出的三样东西是一条线：
- EnhancedLangGraphRAGSystem：对外的门面，外部只跟它打交道（enhanced_entry.py）
- build_enhanced_graph：连图函数，返回编译好的图（enhanced_graph.py）
- EnhancedGraphState：全图共享的状态结构（enhanced_state.py）

别和隔壁的 orchestrator/ 包搞混：那个装的是 planner / router / registry 这类
被本包的图节点调用的零件，它自己不连图、不跑流程。
"""

from langgraph_orchestrator.enhanced_entry import EnhancedLangGraphRAGSystem  # 门面：持有图，对外提供 handle/resume
from langgraph_orchestrator.enhanced_graph import build_enhanced_graph  # 连图：唯一产出可执行图的地方
from langgraph_orchestrator.enhanced_state import EnhancedGraphState  # 状态：图里所有节点共享的数据结构

# __all__ = 显式导出清单：决定 `from langgraph_orchestrator import *` 会带出什么，
# 同时也是一句声明 —— 这个包的公开 API 就这三个，其余都算内部实现。
__all__ = [
    "EnhancedLangGraphRAGSystem",
    "build_enhanced_graph",
    "EnhancedGraphState",
]
