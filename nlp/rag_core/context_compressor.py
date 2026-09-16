# -*- coding: utf-8 -*-
"""
上下文压缩器
============

压缩检索到的文档，减少LLM输入token
"""

import logging
from typing import List, Dict
import re

logger = logging.getLogger(__name__)


class ContextCompressor:
    """
    上下文压缩器

    策略：
    1. 提取与查询相关的句子
    2. 过滤无关信息
    3. 保留关键上下文
    """

    def __init__(self, llm=None, max_length: int = 2000):
        """
        初始化上下文压缩器

        Args:
            llm: LLM实例（用于智能压缩）
            max_length: 压缩后的最大长度
        """
        self.llm = llm
        self.max_length = max_length

        logger.info(f"上下文压缩器初始化完成 (max_length={max_length})")

    def compress(
        self,
        query: str,
        documents: List[Dict],
        use_llm: bool = False
    ) -> List[Dict]:
        """
        压缩文档上下文

        Args:
            query: 查询文本
            documents: 文档列表
            use_llm: 是否使用LLM压缩

        Returns:
            压缩后的文档列表
        """
        if not documents:
            return documents

        logger.info(f"[上下文压缩] 压缩 {len(documents)} 个文档")

        # 计算原始长度
        original_length = sum(len(doc.get("content", "")) for doc in documents)

        if use_llm and self.llm:
            compressed_docs = self._compress_with_llm(query, documents)
        else:
            compressed_docs = self._compress_with_rules(query, documents)

        # 计算压缩后长度
        compressed_length = sum(len(doc.get("content", "")) for doc in compressed_docs)

        compression_ratio = (1 - compressed_length / original_length) if original_length > 0 else 0

        logger.info(f"[上下文压缩] 原始长度: {original_length}, 压缩后: {compressed_length}, 压缩率: {compression_ratio:.1%}")

        return compressed_docs

    def _compress_with_rules(
        self,
        query: str,
        documents: List[Dict]
    ) -> List[Dict]:
        """
        基于规则的压缩

        策略：
        1. 提取包含查询关键词的句子
        2. 保留句子的上下文（前后各一句）
        3. 按相关性排序

        Args:
            query: 查询文本
            documents: 文档列表

        Returns:
            压缩后的文档列表
        """
        compressed_docs = []
        query_lower = query.lower()
        query_words = set(query_lower.split())

        for doc in documents:
            content = doc.get("content", "")

            # 按句子分割
            sentences = self._split_sentences(content)

            # 计算每个句子的相关性分数
            sentence_scores = []
            for i, sentence in enumerate(sentences):
                score = self._calculate_relevance(sentence, query_words)
                sentence_scores.append((i, sentence, score))

            # 按分数排序
            sentence_scores.sort(key=lambda x: x[2], reverse=True)

            # 选择top句子
            selected_indices = set()
            for i, sentence, score in sentence_scores:
                if score > 0:  # 只选择有相关性的句子
                    # 添加当前句子及其上下文
                    selected_indices.add(i)
                    if i > 0:
                        selected_indices.add(i - 1)  # 前一句
                    if i < len(sentences) - 1:
                        selected_indices.add(i + 1)  # 后一句

                # 检查长度限制
                current_length = sum(len(sentences[idx]) for idx in selected_indices)
                if current_length > self.max_length:
                    break

            # 按原始顺序重组句子
            selected_sentences = [sentences[i] for i in sorted(selected_indices)]
            compressed_content = "".join(selected_sentences)

            # 创建压缩后的文档
            compressed_doc = doc.copy()
            compressed_doc["content"] = compressed_content
            compressed_doc["compressed"] = True

            compressed_docs.append(compressed_doc)

        return compressed_docs

    def _compress_with_llm(
        self,
        query: str,
        documents: List[Dict]
    ) -> List[Dict]:
        """
        基于LLM的压缩

        让LLM提取与查询相关的关键信息

        Args:
            query: 查询文本
            documents: 文档列表

        Returns:
            压缩后的文档列表
        """
        compressed_docs = []

        for doc in documents:
            content = doc.get("content", "")

            # 如果内容已经很短，不需要压缩
            if len(content) <= 500:
                compressed_docs.append(doc)
                continue

            try:
                prompt = f"""请从以下文档中提取与查询相关的关键信息，保持原文表达，不要添加额外内容。

查询：{query}

文档：
{content}

提取的关键信息："""

                compressed_content = self.llm.generate(prompt, temperature=0.3, max_tokens=500)

                # 创建压缩后的文档
                compressed_doc = doc.copy()
                compressed_doc["content"] = compressed_content.strip()
                compressed_doc["compressed"] = True

                compressed_docs.append(compressed_doc)

            except Exception as e:
                logger.error(f"LLM压缩失败: {e}")
                # 失败时使用原文档
                compressed_docs.append(doc)

        return compressed_docs

    def _split_sentences(self, text: str) -> List[str]:
        """
        分割句子

        Args:
            text: 文本

        Returns:
            句子列表
        """
        # 按句号、问号、感叹号分割
        sentences = re.split(r'([。！？.!?]+)', text)

        # 重新组合句子和标点
        combined = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                combined.append(sentences[i] + sentences[i + 1])
            else:
                combined.append(sentences[i])

        # 过滤空句子
        combined = [s.strip() for s in combined if s.strip()]

        return combined

    def _calculate_relevance(self, sentence: str, query_words: set) -> float:
        """
        计算句子与查询的相关性

        Args:
            sentence: 句子
            query_words: 查询词集合

        Returns:
            相关性分数
        """
        sentence_lower = sentence.lower()
        sentence_words = set(sentence_lower.split())

        # 计算交集
        intersection = query_words & sentence_words

        # 相关性分数 = 匹配词数 / 查询词数
        score = len(intersection) / len(query_words) if query_words else 0

        return score
