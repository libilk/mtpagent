# -*- coding: utf-8 -*-
"""
向量相似度计算
==============

用于Agent路由的向量相似度匹配

本文件有两半：
1. embedder（向量化模型）：把文本变成 vector（向量）—— 但 Embedder 只是个**薄包装**，
   真正干活的是外部传进来的 api_embedder（来自 rag_core.api_embedder.APIEmbedder）。
   包一层的目的是让上层面向统一接口（embed / embed_batch），不直接依赖 rag_core。
2. cosine_similarity（余弦相似度）及基于它的排序工具 —— 这部分是纯数学。

**最容易误解的一点（先看这条再读代码）**：api_embedder 为 None 时，
Embedder **没有**任何本地备用方案 —— 它直接抛 ValueError("未配置嵌入器")。
也就是说这个对象看起来像"兜底构造"，实际是个**不可用的空壳**。
而 enhanced_entry._create_embedder 的 except 分支返回的恰恰就是这种裸 Embedder()：
构造能过，一旦真要用向量（路由召回、重复检测）就在调用点才炸。
所以看到 `Embedder()` 别以为"至少能跑"，它是"能构造、不能用"。
"""

import logging
import numpy as np
from typing import List, Tuple, Dict, Any

logger = logging.getLogger(__name__)


class Embedder:
    """向量嵌入器（转发给 api_embedder，自身不实现向量化算法）"""

    def __init__(self, api_embedder=None):
        """
        初始化

        Args:
            api_embedder: API嵌入器实例（来自rag_core.api_embedder）

        传 None 也能构造成功，但一调用就抛错 —— 见文件头"最容易误解的一点"。
        """
        self.api_embedder = api_embedder

    def embed(self, text: str) -> List[float]:
        """
        文本转向量

        Args:
            text: 输入文本

        Returns:
            向量
        """
        if self.api_embedder:
            # embed_query 是"单条"接口，与下面批量的 embed_documents 成对 ——
            # 有些嵌入服务对两者用不同前缀，所以别混用
            return self.api_embedder.embed_query(text)
        else:
            # 注意：这不是 fallback（降级兜底），是直接失败 —— 没有 api_embedder 就无路可走
            raise ValueError("未配置嵌入器")

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        批量文本转向量

        Args:
            texts: 文本列表

        Returns:
            向量列表
        """
        if self.api_embedder:
            # 走批量接口（一次请求多条），比循环调 embed 省往返 —— 它不保证与 embed 等价
            return self.api_embedder.embed_documents(texts)
        else:
            raise ValueError("未配置嵌入器")


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    计算余弦相似度（比较两个向量方向有多接近，不考虑长短）

    Args:
        vec1: 向量1
        vec2: 向量2

    Returns:
        相似度（0-1）

    数值上余弦取值是 -1~1；此处注释所言 0~1 是实际嵌入向量的常见落点，
    并非代码做了截断 —— 负数结果会原样返回（调用方做阈值判断时需留意）。
    """
    # 注意：这两行把入参名覆盖成了 numpy 数组，之后 vec1/vec2 已是数组而非 List
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    # 防御性处理：维度不一致时截断到较短维度
    # 这是"能算下去"的权宜之计、不是正确比较 —— 不同维度的向量本就不该比，结果无意义
    if vec1.shape != vec2.shape:
        logger.warning(f"向量维度不一致: {vec1.shape} vs {vec2.shape}，截断到较短维度")
        min_dim = min(len(vec1), len(vec2))
        vec1 = vec1[:min_dim]
        vec2 = vec2[:min_dim]

    # 余弦相似度 = 点积 ÷ 两模长之积；除以模长是为了消掉向量长度的影响，只比方向
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    # 零向量没有方向，做除会得到 nan，所以单独短路返回 0
    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))


def compute_similarities(
    query_vec: List[float],
    candidate_vecs: List[List[float]]
) -> List[float]:
    """
    计算查询向量与候选向量的相似度

    Args:
        query_vec: 查询向量
        candidate_vecs: 候选向量列表

    Returns:
        相似度列表

    逐条 Python 循环，没有向量化 —— 候选多时这里是性能点（并非慢在数学，而是慢在循环）。
    """
    similarities = []
    for candidate_vec in candidate_vecs:
        sim = cosine_similarity(query_vec, candidate_vec)
        similarities.append(sim)
    return similarities


def rank_by_similarity(
    query_vec: List[float],
    candidates: List[Dict[str, Any]],
    candidate_vecs: List[List[float]],
    top_k: int = 3
) -> List[Tuple[Dict[str, Any], float]]:
    """
    根据相似度排序候选项

    Args:
        query_vec: 查询向量
        candidates: 候选项列表
        candidate_vecs: 候选项向量列表
        top_k: 取前 k 条（只保留得分最高的 k 个）

    Returns:
        排序后的候选项列表（候选项, 相似度）
    """
    similarities = compute_similarities(query_vec, candidate_vecs)

    # 组合候选项和相似度
    ranked = list(zip(candidates, similarities))

    # 按相似度降序排序
    ranked.sort(key=lambda x: x[1], reverse=True)

    # 返回top-k
    return ranked[:top_k]


class VectorMatcher:
    """向量匹配器（纯内存的最近邻索引，没有 ANN/近似索引 —— 数据量小才适用）"""

    def __init__(self, embedder: Embedder):
        """
        初始化

        Args:
            embedder: 嵌入器实例
        """
        self.embedder = embedder
        # 向量在 add 时就预先算好存下，避免每次查询重复调嵌入接口（省时省钱）
        self.index = {}  # {id: {"data": data, "vector": vector}}

    def add(self, id: str, data: Dict[str, Any], text: str):
        """
        添加项目（每 add 一次就真的调一次嵌入接口，是会产生网络开销的操作）

        Args:
            id: 项目ID
            data: 项目数据
            text: 用于嵌入的文本
        """
        vector = self.embedder.embed(text)
        self.index[id] = {
            "data": data,
            "vector": vector
        }
        logger.debug(f"添加项目: {id}")

    def search(self, query: str, top_k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        """
        搜索最相似的项目

        Args:
            query: 查询文本
            top_k: 返回前k个

        Returns:
            排序后的项目列表（项目数据, 相似度）
        """
        if not self.index:
            return []

        # 查询向量
        # 每次 search 都要先把 query 向量化一次（一次嵌入接口调用），再与索引里
        # **所有**向量逐一算余弦 —— 故耗时随条目数线性增长
        query_vec = self.embedder.embed(query)

        # 计算相似度
        results = []
        for id, item in self.index.items():
            sim = cosine_similarity(query_vec, item["vector"])
            results.append((item["data"], sim))

        # 排序
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    def clear(self):
        """清空索引"""
        self.index.clear()
