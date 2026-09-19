# -*- coding: utf-8 -*-
"""database_agent 包：Text-to-SQL（自然语言转 SQL）形态的数据库查询 Agent。

实现都在 agent.py 的 DatabaseAgent 里 —— 它注册 4 个工具，让 LLM 自己生成 SQL
查订单/物流/退款/会员数据，再把结果翻译成自然语言。包本身不做导入聚合，
要用就直接 from agents.database_agent.agent import DatabaseAgent。
"""
