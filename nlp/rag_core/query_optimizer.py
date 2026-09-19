# -*- coding: utf-8 -*-
"""
查询优化器
==========

优化用户查询，提高检索效果

【调用状态 —— 只在降级路径生效】唯一调用方是 knowledge_agent/agent.py:1204
（`_handle_with_optimizations` 第 1 步）。ReAct 主路径不调它：主路径要么原样检索，
要么由 ReAct 的 LLM 自己换个关键词重试（那是模型临场决定，不是这里的固定流程）。

【为什么需要它，尤其 HyDE】用户的问题通常很短、很口语（"能退吗"），
而知识库文档的措辞是书面、完整的（"七天无理由退货政策说明…"）。两边字面几乎不重叠，
直接拿问题去比向量，语义距离很远、容易检索不到。

HyDE（假设文档嵌入）= 先让 LLM 编一段"看起来像答案"的假文本，再拿**这段假文本**
的向量去检索，而不是拿问题本身 —— 假答案的用词和真文档同属"书面语域"，语义距离
一下被拉近，命中率明显提高。注意假答案**不需要正确**，它只是当"翻译器"用：
把口语问题翻译成书面语措辞。代价是多一次 LLM 调用，且假答案如果跑偏，
会把整轮检索带偏。
"""

import logging
from typing import List, Optional
import re

logger = logging.getLogger(__name__)


class QueryOptimizer:
    """
    查询优化器

    功能：
    1. 查询改写（Query Rewriting）—— 同一问题换 3 种说法，多路检索提高召回
    2. 查询扩展（Query Expansion）—— ⚠️ 只列在这里，无对应实现
    3. HyDE（Hypothetical Document Embeddings，假设文档嵌入）—— 见文件头
    4. 查询纠错 —— ⚠️ 只列在这里，无对应实现（_clean_query 只去空格和特殊符号）
    """

    def __init__(self, llm=None, enable_hyde: bool = False):
        """
        初始化查询优化器

        Args:
            llm: LLM实例（用于查询改写和HyDE）
            enable_hyde: 是否启用HyDE（knowledge_agent 里默认 True，见 agent.py:146/269）
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
        # 1. 查询清洗
        cleaned_query = self._clean_query(query)

        # 2. LLM 改写
        queries = self.rewrite_with_llm(cleaned_query)

        # 3. HyDE（可选）
        if self.enable_hyde and self.llm:
            hyde_query = self._hyde(cleaned_query)
            if hyde_query:
                queries.append(hyde_query)

        # 去重：dict 从 Python 3.7 起保留插入顺序，所以既能去重又能保序，
        # 原始查询必定排在最前面（下游按顺序检索，顺序影响最终结果的排名）
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
        # 去除多余空格（多个空格压成一个，再去首尾）
        query = re.sub(r'\s+', ' ', query).strip()

        # 去除特殊字符（保留中英文、数字、常用标点）
        query = re.sub(r'[^\w\s\u4e00-\u9fff，。！？、；：""''（）]', '', query)

        return query

    def _hyde(self, query: str) -> Optional[str]:
        """
        HyDE (Hypothetical Document Embeddings)

        策略：
        1. 让LLM生成一个假设性的答案
        2. 用这个答案的向量去检索（本方法只负责产出文本，向量化在调用方的检索器里）
        3. 通常比直接用问题检索效果更好

        prompt 里特意写了"不需要完全准确"：要的是**像文档、像答案的措辞**，
        不是正确答案本身 —— 编得越像文档，向量越靠近目标。这一点最容易读反
        （以为是"生成候选答案再验证"）。温度取 0.7 也是为多样性，不是求准。

        返回 None 表示"这次没造出来"（未配 LLM 或调用失败），
        调用方会跳过追加、退回原查询，而不是中断整个流程。

        Args:
            query: 查询文本

        Returns:
            假设性文档（假答案文本），失败时为 None
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

        # 有意的降级设计：查询改写是"锦上添花"，失败绝不该拖垮整轮问答 ——
        # 所以下面 catch 一切异常，最差也只是原查询单跑，功能不缺失只是效果打折。
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

            # 原始查询 + 最多2个改写：每个变体都要独立跑一遍向量+BM25 检索，
            # 只留 2 个是拿"召回提升"换"检索耗时"的折中（prompt 里虽要了 3 个）
            return [query] + rewritten[:2]

        except Exception as e:
            logger.error(f"LLM改写失败: {e}")
            return [query]
