# -*- coding: utf-8 -*-
"""
答案生成器
==========

优化答案生成，支持Self-Consistency和Citation

【调用状态 —— 唯一调用方是降级路径】knowledge_agent/agent.py:1257
（`_handle_with_optimizations` 第 4 步）。ReAct 主路径的答案由 LLM 直接产出，
不经过本类 —— 别把它当成"每次问答都会跑的答案层"。

【两个开关的默认值】knowledge_agent 构造时 `enable_self_consistency` 默认 True
（agent.py:147 定义、328 传入），即默认会跑 3 次采样投票，成本也是 3 倍。
"""

import logging
from typing import List, Dict, Tuple
from collections import Counter
import re

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """
    答案生成器

    功能：
    1. Self-Consistency（自洽性：同一问题生成多次，取多数一致的那个答案）
    2. Citation（引用溯源：答案里标出"这句来自哪篇文档"）
    3. 答案验证

    ⚠️ 第 3 项"答案验证"只写在这里，本类并没有对应的校验方法（无实现），
       实际只有 generate 一条产出路径。
    """

    def __init__(self, llm, enable_self_consistency: bool = False, num_samples: int = 3):
        """
        初始化答案生成器

        Args:
            llm: LLM实例
            enable_self_consistency: 是否启用自洽性检查
            num_samples: 采样次数
        """
        self.llm = llm
        self.enable_self_consistency = enable_self_consistency
        # num_samples 越大投票越稳，成本也按倍数涨。另外自洽性只有在
        # temperature（采样温度：控制输出随机性，越高越发散）> 0 时才有意义 ——
        # 温度若为 0，多次采样答案几乎完全一样，"投票"等于白花 3 倍钱。
        self.num_samples = num_samples

        logger.info(f"答案生成器初始化完成 (Self-Consistency: {enable_self_consistency})")

    def generate(
        self,
        query: str,
        documents: List[Dict],
        add_citation: bool = True
    ) -> Dict:
        """
        生成答案

        Args:
            query: 查询文本
            documents: 检索到的文档
            add_citation: 是否添加引用

        Returns:
            答案字典 {answer, citations, confidence}
        """
        if not documents:
            return {
                "answer": "抱歉，我在知识库中没有找到与您问题相关的信息。",
                "citations": [],
                "confidence": 0.0
            }

        # 构建上下文
        context = self._build_context(documents)

        # 生成答案
        if self.enable_self_consistency:
            answer, confidence = self._generate_with_consistency(query, context)
        else:
            answer = self._generate_single(query, context)
            # ⚠️ 这里的 1.0 不是"非常确信"，而是"只生成了一次、无从估计"的占位值。
            # 读 confidence 时得先知道走的是哪条分支，否则会被误导
            confidence = 1.0

        # 添加引用
        citations = []
        if add_citation:
            answer, citations = self._add_citations(answer, documents)

        logger.info(f"[答案生成] 置信度: {confidence:.2f}, 引用数: {len(citations)}")

        return {
            "answer": answer,
            "citations": citations,
            "confidence": confidence
        }

    def _build_context(self, documents: List[Dict]) -> str:
        """
        构建上下文

        Args:
            documents: 文档列表

        Returns:
            上下文文本

        context（上下文）= 喂给 LLM 的参考资料。这里给每篇文档标上 [文档i] 编号，
        是为了让模型能指着编号说"依据第几篇" —— 但这套编号与 _add_citations 产出的
        [1][2] **不是同一套**（后者按实际命中顺序重新编号），别指望能直接对上。
        """
        context_parts = []

        for i, doc in enumerate(documents):
            content = doc.get("content", "")
            source = doc.get("metadata", {}).get("source", "未知来源")

            context_parts.append(f"[文档{i+1}] 来源: {source}\n{content}")

        return "\n\n".join(context_parts)

    def _generate_single(self, query: str, context: str) -> str:
        """
        单次生成答案

        Args:
            query: 查询文本
            context: 上下文

        Returns:
            答案
        """
        prompt = f"""请基于以下文档回答用户问题。

文档：
{context}

用户问题：{query}

请给出简洁准确的回答："""

        # temperature 0.3（偏低）：问答求稳 —— 同一个问题问两遍不该给出两套说法
        answer = self.llm.generate(prompt, temperature=0.3)

        return answer.strip()

    def _generate_with_consistency(self, query: str, context: str) -> Tuple[str, float]:
        """
        使用Self-Consistency生成答案

        策略：
        1. 多次采样生成答案
        2. 提取关键信息
        3. 投票选择最一致的答案

        为什么必须先"增加多样性"再投票：三次生成如果一字不差，投票就没有任何信息量。
        所以采样温度要调高（见下方 0.7），温度低时这套机制等于白跑三遍。

        代价与局限：投票能压掉**随机性**错误（模型每次都换个地方错），
        但压不掉**系统性**错误 —— 如果模型每次都错在同一个点上，三票一致反而
        会得到一个虚高的 confidence。

        Args:
            query: 查询文本
            context: 上下文

        Returns:
            (答案, 置信度)
        """
        logger.info(f"[Self-Consistency] 生成 {self.num_samples} 个答案")

        answers = []

        for i in range(self.num_samples):
            prompt = f"""请基于以下文档回答用户问题。

文档：
{context}

用户问题：{query}

请给出简洁准确的回答："""

            answer = self.llm.generate(prompt, temperature=0.7)  # 使用较高温度增加多样性
            answers.append(answer.strip())

        # 提取关键信息并投票
        final_answer, confidence = self._vote_answers(answers)

        logger.info(f"[Self-Consistency] 最终答案置信度: {confidence:.2f}")

        return final_answer, confidence

    def _vote_answers(self, answers: List[str]) -> Tuple[str, float]:
        """
        对多个答案投票

        策略：
        1. 提取每个答案的关键信息
        2. 统计最常见的答案
        3. 计算置信度

        Args:
            answers: 答案列表

        Returns:
            (最终答案, 置信度)
        """
        # 简单策略：选择最常见的答案
        # ⚠️ 这里是**整串精确比较**，不是语义比较：两次回答只差一个标点或一个"的"字
        #    就算两票不同，所以 confidence 偏低是常态，别据此判断答案不可信。
        answer_counter = Counter(answers)
        most_common = answer_counter.most_common(1)[0]

        final_answer = most_common[0]
        count = most_common[1]
        confidence = count / len(answers)

        return final_answer, confidence

    def _add_citations(self, answer: str, documents: List[Dict]) -> Tuple[str, List[Dict]]:
        """
        为答案添加引用

        策略：
        1. 找到答案中引用的文档
        2. 添加引用标记 [1], [2], ...
        3. 返回引用列表

        Args:
            answer: 答案文本
            documents: 文档列表

        Returns:
            (带引用的答案, 引用列表)
        """
        citations = []
        cited_answer = answer

        # 为每个文档检查是否被引用
        for i, doc in enumerate(documents):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})

            # 简单策略：检查答案中是否包含文档的关键片段
            # ⚠️ 拿"文档开头 50 字"原样去答案里找。LLM 极少逐字照抄文档开头，
            #    所以 citations 命中率天然很低、经常为空。这是已知的粗糙近似，
            #    不代表模型没用上文档 —— 用 citation 判断"有无依据"会低估。
            # 提取文档的关键短语（前50字符）
            key_phrase = content[:50].strip()

            if key_phrase and key_phrase in answer:
                citation_num = len(citations) + 1
                citations.append({
                    "id": citation_num,
                    "source": metadata.get("source", "未知来源"),
                    "title": metadata.get("title", "未命名文档"),
                    "content_preview": content[:100]
                })

        # 如果有引用，在答案末尾添加引用标记
        if citations:
            citation_marks = ", ".join([f"[{c['id']}]" for c in citations])
            cited_answer = f"{answer} {citation_marks}"

        return cited_answer, citations
