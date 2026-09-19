"""knowledge_agent 包：RAG（检索增强生成）形态的知识库问答 Agent。

实现全在 agent.py 的 KnowledgeAgent 里：把用户问题向量化后去 Chroma
（向量数据库）召回文档，再用 BM25（关键词检索算法）补一路，两路结果融合后
交给 LLM 生成带引用溯源的回答。本文件不做导入聚合，用就直接
from agents.knowledge_agent.agent import KnowledgeAgent。
"""
