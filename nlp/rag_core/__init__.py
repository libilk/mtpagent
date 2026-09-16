"""
RAG 核心组件
===================

RAG 系统的核心组件包括：
- 文档处理和分块
- 基于 FAISS 的向量存储
- 使用句子转换器的语义检索
- BM25 关键词检索
- 使用 Qwen 模型的文本生成
- 提示词模板和优化
- 综合评估系统
- 重排序模块
"""

__version__ = "2.0.0"
__author__ = "RAG System"

# 核心模块 - 只导入存在的模块
try:
    from .api_embedder import APIEmbedder
except ImportError:
    APIEmbedder = None

__all__ = [
    "APIEmbedder",
]
