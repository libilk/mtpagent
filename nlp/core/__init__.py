# -*- coding: utf-8 -*-
"""
core —— 与业务无关的基础设施层
=============================

本包放的是"任何 Agent 都能用、但不属于任何具体 Agent"的东西，所以它不 import agents/、
也不 import langgraph_orchestrator/，依赖方向是单向的：上层依赖 core，core 不反向依赖。

五个模块，各管一件事：
- protocol.py：Agent 接口契约（AgentProtocol）与结构化消息模型。是"接口长什么样"的唯一出处。
- write_ops.py：写操作记录器（线程本地）。让"Agent 改了数据"这个信号能越过
  handle() -> str 的返回值限制，传给编排层触发人工审批。
- file_parser.py：通用文件解析，把 pdf/docx/csv/xlsx/json 等读成纯文本。
- filesystem_service.py：文件系统访问封装（名为 MCP 服务，当前实际走 Python 原生实现）。
- tracing.py：LangSmith 链路追踪开关（默认关闭）。必须在 langchain 被 import 之前调用，
  调用点固定在两个入口的 load_dotenv() 之后 —— 时机比代码本身更重要。

本文件刻意保持为空（仅文档字符串）：没有需要包级导出的符号，
调用方一律 `from core.protocol import ...` 这样按模块取，别在这里加 import 制造隐式依赖。
"""
