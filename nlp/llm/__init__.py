# -*- coding: utf-8 -*-
"""
llm 包 —— 与"模型"打交道的那一层
================================

把"怎么调模型"和"拿到模型输出后怎么收拾"集中在这里，上层 Agent 不直接碰 HTTP：

- llm_client.py       统一 LLM 接口（generate / chat），所有模型调用的唯一出口。
- langchain_tools.py  工具注册（ToolRegistry）：把 Python 函数登记成 LLM 可调用的
                      tool（工具），并负责参数校验。
- output_parser.py    手写的容错 JSON 解析：LLM 的文本常带 markdown 围栏等噪声，
                      不是严格 JSON，得先洗干净再抠出来。
- langchain_parser.py 借 LangChain 的 OutputParser 解析成 Pydantic 对象。
- mcp_client.py       按 MCP（模型上下文协议）连接外部工具服务。
- embedder.py         文本向量化（embedding：把文本变成一串数字，语义相近则向量相近）。

**本文件刻意不放 import**：上面各模块一律按完整路径单独导入
（如 `from llm.llm_client import LLM`），没有"包级再导出"这一层。
好处是改单个模块不会牵连整个包 —— 也正因如此，本文件改动不影响任何调用方。
"""
