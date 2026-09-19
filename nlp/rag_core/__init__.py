"""
RAG 核心组件
===================

RAG（检索增强生成）= 先检索资料、再让 LLM 基于资料回答，而不是全靠模型自身记忆。
本包就是「检索」这一半的零件库。

下面这份清单是**历史描述，和当前实现有出入**，读之前先对齐：
- 「基于 FAISS 的向量存储」实际是 Chroma（见 chroma_store.py），FAISS 在这个仓库里根本没用到；
- 「使用句子转换器的语义检索」实际是调远程 API 做向量化（见 api_embedder.py），
  requirements.txt 里的 sentence-transformers 是注释掉的。
- 其余几条（BM25 关键词检索、重排序、评估）确实存在，各自成文件。

注意本文件只重导出了 APIEmbedder，别以为「核心组件」都在这里能 import 到 ——
其余组件由调用方各自 `from rag_core.xxx import ...`。不在此聚合是为了避免
`import rag_core` 一次就把 Chroma、jieba 这些重依赖全拉起来。
"""

__version__ = "2.0.0"
__author__ = "RAG System"

# 核心模块 - 只导入存在的模块
# 用 try/except 兜底：让「装不上某个可选依赖」不至于让整个包的 import 失败（降级为 None）。
try:
    from .api_embedder import APIEmbedder
except ImportError:
    APIEmbedder = None

__all__ = [
    "APIEmbedder",
]
