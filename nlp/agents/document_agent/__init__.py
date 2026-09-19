# -*- coding: utf-8 -*-
"""
Document Agent
==============

企业级合同审核Agent

注意：本 Agent 已在电商售后改造（阶段 4）时从编排层摘除注册，图里没有它的节点，
本包当前不参与线上运行 —— 原因与恢复方式见 agent.py 的文件头，别在这里找调用点。

本文件只是个"导出壳"，作用有两个：
- `from .agent import DocumentAgent` 把类提到包这一层，外部才能写
  `from agents.document_agent import DocumentAgent`，不必知道类其实在 agent.py 里。
- `__all__` 声明 `from agents.document_agent import *` 时会导出哪些名字。
"""

from .agent import DocumentAgent  # 包的对外出口：只做转发，自身不含任何逻辑

__all__ = ['DocumentAgent']
