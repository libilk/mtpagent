# -*- coding: utf-8 -*-
"""
查询优化器
==========

优化用户查询，提高检索效果
"""

import logging
from typing import List, Optional
import re

logger = logging.getLogger(__name__)


class QueryOptimizer:
    """
    查询优化器

    功能：
    1. 查询改写（Query Rewriting）
    2. 查询扩展（Query Expansion）
    3. HyDE（Hypothetical Document Embeddings）
    4. 查询纠错
    """

    def __init__(self, llm=None, enable_hyde: bool = False):
        """
        初始化查询优化器

        Args:
            llm: LLM实例（用于查询改写和HyDE）
            enable_hyde: 是否启用HyDE
        """
        self.llm = llm
        self.enable_hyde = enable_hyde

        logger.info(f"查询优化器初始化完成 (HyDE: {enable_hyde})")

    def optimize(self, query: str) -> List[str]:
        """
        优化查询，返回多个查询变体

        Args:
            query: 原始查询

        Returns:
            查询列表（包含原始查询和优化后的查询）
        """
        queries = [query]  # 始终包含原始查询

        # 1. 查询清洗
        cleaned_query = self._clean_query(query)
        if cleaned_query != query:
            queries.append(cleaned_query)

        # 2. 查询扩展
        expanded_queries = self._expand_query(cleaned_query)
        queries.extend(expanded_queries)

        # 3. HyDE（可选）
        if self.enable_hyde and self.llm:
            hyde_query = self._hyde(cleaned_query)
            if hyde_query:
                queries.append(hyde_query)

        # 去重
        queries = list(dict.fromkeys(queries))

        logger.info(f"[查询优化] 原始查询: '{query}'")
        logger.info(f"[查询优化] 生成 {len(queries)} 个查询变体")

        return queries

    def _clean_query(self, query: str) -> str:
        """
        清洗查询

        - 去除多余空格
        - 去除特殊字符
        - 统一标点符号

        Args:
            query: 原始查询

        Returns:
            清洗后的查询
        """
        # 去除多余空格
        query = re.sub(r'\s+', ' ', query).strip()

        # 去除特殊字符（保留中英文、数字、常用标点）
        query = re.sub(r'[^\w\s\u4e00-\u9fff，。！？、；：""''（）]', '', query)

        return query

    def _expand_query(self, query: str) -> List[str]:
        """
        查询扩展

        策略：
        1. 添加同义词
        2. 添加相关词
        3. 改写为不同表达方式

        Args:
            query: 查询文本

        Returns:
            扩展后的查询列表
        """
        expanded = []

        # 简单的同义词替换（可以扩展为使用词典）
        synonyms = {
            "价格": ["多少钱", "售价", "定价"],
            "怎么": ["如何", "怎样"],
            "什么": ["啥", "哪些"],
        }

        for word, syns in synonyms.items():
            if word in query:
                for syn in syns:
                    expanded_query = query.replace(word, syn)
                    if expanded_query != query:
                        expanded.append(expanded_query)

        return expanded[:2]  # 最多返回2个扩展查询

    def _hyde(self, query: str) -> Optional[str]:
        """
        HyDE (Hypothetical Document Embeddings)

        策略：
        1. 让LLM生成一个假设性的答案
        2. 用这个答案的向量去检索
        3. 通常比直接用问题检索效果更好

        Args:
            query: 查询文本

        Returns:
            假设性文档
        """
        if not self.llm:
            return None

        try:
            prompt = f"""请根据以下问题，生成一个简短的假设性答案（100字以内）。
不需要完全准确，只需要生成一个可能的答案即可。

问题：{query}

假设性答案："""

            hyde_doc = self.llm.generate(prompt, temperature=0.7, max_tokens=200)

            logger.info(f"[HyDE] 生成假设性文档: {hyde_doc[:50]}...")

            return hyde_doc.strip()

        except Exception as e:
            logger.error(f"HyDE生成失败: {e}")
            return None

    def rewrite_with_llm(self, query: str) -> List[str]:
        """
        使用LLM改写查询

        Args:
            query: 原始查询

        Returns:
            改写后的查询列表
        """
        if not self.llm:
            return [query]

        try:
            prompt = f"""请将以下用户查询改写为3个不同的表达方式，保持原意不变。
每个改写占一行，不要编号。

原始查询：{query}

改写："""

            response = self.llm.generate(prompt, temperature=0.7, max_tokens=200)

            # 解析改写结果
            rewritten = [line.strip() for line in response.split('\n') if line.strip()]
            rewritten = [q for q in rewritten if q and q != query]

            logger.info(f"[LLM改写] 生成 {len(rewritten)} 个改写查询")

            return [query] + rewritten[:2]  # 原始查询 + 最多2个改写

        except Exception as e:
            logger.error(f"LLM改写失败: {e}")
            return [query]
