# -*- coding: utf-8 -*-
"""客服 Agent 包（customer_service_agent）。

对外只提供一个类：`CustomerServiceAgent` —— 全系统的业务主角，
既能答（政策咨询）、也能办事（建工单、提交退换货申请）。

注意这个 `__init__.py` 里**没有** re-export 那个类，调用方一律写成
`from agents.customer_service_agent.agent import CustomerServiceAgent`
（见 langgraph_orchestrator/enhanced_entry.py 的注册处）。
这不算疏漏：agent.py 顶部会连带加载 LLM 封装、CRM、检索器、写操作记录器等
一串依赖，放进包的入口会让"只想读某个子模块"的人被迫全量加载。

- agent（智能体）= 能自己决定"调哪个工具、调几次"的 LLM 程序，
  入口方法固定叫 handle（handle=处理入口）。
- ReAct（推理-行动循环）= 它内部的工作方式：想 → 调工具 → 看结果 → 再想。
"""
