# -*- coding: utf-8 -*-
"""
RAG评估体系
===========

提供离线和在线评估指标

【调用状态 —— 核实自 knowledge_agent/agent.py，别把它当成主链路的质量把关】
- 真接线：`OnlineMetrics.record_query`（agent.py:851，主路径 handle 开头）；
  `RAGEvaluator.get_summary` / `OnlineMetrics.get_metrics` 只被 get_stats 报表调（1664）。
- 只在降级路径 `_handle_with_optimizations` 生效：`RAGEvaluator.evaluate_latency`（1271）。
- **完全没有调用方**：`evaluate_retrieval` / `evaluate_generation` —— 全仓库只有定义，
  没有任何地方调用（离线评估需要人工标注，线上请求拿不到）。

⚠️ 命名易混：本文件的 `RAGEvaluator` 只统计延迟这类运行指标，**不给答案打质量分**。
编排层那个"不过关就打回重试"的 evaluator（评估器）是另一个东西，别对号入座。
"""

import logging
from typing import List, Dict, Tuple
import time
from collections import defaultdict

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """
    RAG（检索增强生成：先检索资料、再让 LLM 基于资料回答）评估器

    功能（第 1、2 项目前无调用方，见文件头）：
    1. 检索评估（Recall 召回 / Precision 精确率 / MRR 平均倒数排名）
    2. 生成评估（ROUGE 与参考答案的文本重合度 / Faithfulness 忠实度）
    3. 端到端评估（Latency 延迟 / Throughput 吞吐量 —— 后者未实现）
    """

    def __init__(self):
        """初始化评估器"""
        # defaultdict(list)：指标名 → 历史值的列表。之所以留全量历史而不是只存均值，
        # 是因为 get_summary 要算 p95/p99 —— 只留均值就再也算不出分位数了。
        self.metrics = defaultdict(list)

        logger.info("RAG评估器初始化")

    def evaluate_retrieval(
        self,
        retrieved_docs: List[Dict],
        relevant_docs: List[str],
        k: int = 3
    ) -> Dict:
        """
        评估检索质量

        Args:
            retrieved_docs: 检索到的文档列表
            relevant_docs: 相关文档ID列表（ground truth = 人工标注的标准答案）；
                           有标注才判得出"检索得对不对"，线上请求没有标注，
                           这正是本方法无调用方的原因
            k: 评估前k个结果（top_k 取前 k 条）

        Returns:
            评估指标字典
        """
        # doc_id（文档 ID）= 每篇文档的唯一标识；这里只取 id 不取正文——判相关性只需要 id
        retrieved_ids = [doc.get("metadata", {}).get("doc_id") for doc in retrieved_docs[:k]]

        # Recall@k: 召回率 —— 全部相关文档里捞回来了几成，衡量"漏没漏"
        relevant_retrieved = set(retrieved_ids) & set(relevant_docs)
        recall = len(relevant_retrieved) / len(relevant_docs) if relevant_docs else 0

        # Precision@k: 精确率 —— 取回的 k 条里几条真相关，衡量"准不准"。
        # 分母写死 k，不足 k 条时等价于按"没凑够数"扣分
        precision = len(relevant_retrieved) / k if k > 0 else 0

        # MRR (Mean Reciprocal Rank): 平均倒数排名 —— 只看第一条命中的位置：
        # 排第 1 得 1、第 2 得 1/2、第 3 得 1/3，反映"用户要往下翻多久"
        mrr = 0.0
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_docs:
                mrr = 1.0 / (i + 1)
                break

        metrics = {
            f"recall@{k}": recall,
            f"precision@{k}": precision,
            "mrr": mrr
        }

        logger.info(f"[检索评估] Recall@{k}={recall:.3f}, Precision@{k}={precision:.3f}, MRR={mrr:.3f}")

        return metrics

    def evaluate_generation(
        self,
        generated_answer: str,
        reference_answer: str,
        retrieved_docs: List[Dict]
    ) -> Dict:
        """
        评估生成质量

        Args:
            generated_answer: 生成的答案
            reference_answer: 参考答案（ground truth）
            retrieved_docs: 检索到的文档

        Returns:
            评估指标字典
        """
        # ROUGE-L: 最长公共子序列 —— 与参考答案的重合度。只比字面、不判对错，
        # 换个同义词重述就会被判成不重合，所以低分不等于答错
        rouge_l = self._calculate_rouge_l(generated_answer, reference_answer)

        # Faithfulness: 答案是否忠实于检索文档 —— 用来抓"脱离资料自己编"（幻觉）
        faithfulness = self._calculate_faithfulness(generated_answer, retrieved_docs)

        # Answer Relevance: 答案与问题的相关性（简化版）
        relevance = self._calculate_relevance(generated_answer, reference_answer)

        metrics = {
            "rouge_l": rouge_l,
            "faithfulness": faithfulness,
            "relevance": relevance
        }

        logger.info(f"[生成评估] ROUGE-L={rouge_l:.3f}, Faithfulness={faithfulness:.3f}, Relevance={relevance:.3f}")

        return metrics

    def evaluate_latency(self, start_time: float, end_time: float) -> Dict:
        """
        评估延迟

        Args:
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            延迟指标
        """
        latency_ms = (end_time - start_time) * 1000

        self.metrics["latency"].append(latency_ms)

        metrics = {
            "latency_ms": latency_ms,
            "avg_latency_ms": sum(self.metrics["latency"]) / len(self.metrics["latency"])
        }

        logger.info(f"[延迟评估] 当前={latency_ms:.1f}ms, 平均={metrics['avg_latency_ms']:.1f}ms")

        return metrics

    def get_summary(self) -> Dict:
        """
        获取评估摘要

        Returns:
            评估摘要字典
        """
        summary = {}

        if self.metrics["latency"]:
            # p50/p95/p99（百分位数）：延迟从小到大排，95% 的请求都快于 p95 这个值。
            # 均值会被个别慢请求拉偏，分位数才看得出"长尾拖到多长"
            latencies = self.metrics["latency"]
            summary["latency"] = {
                "avg_ms": sum(latencies) / len(latencies),
                "min_ms": min(latencies),
                "max_ms": max(latencies),
                "p50_ms": self._percentile(latencies, 50),
                "p95_ms": self._percentile(latencies, 95),
                "p99_ms": self._percentile(latencies, 99)
            }

        return summary

    def _calculate_rouge_l(self, generated: str, reference: str) -> float:
        """
        计算ROUGE-L分数（最长公共子序列）

        Args:
            generated: 生成的文本
            reference: 参考文本

        Returns:
            ROUGE-L分数
        """
        # 简化版：计算字符级别的最长公共子序列
        # 用动态规划而不是直接比字符串：允许中间插字、跳字，只要求先后顺序一致。
        # "字符级"对中文正好合适 —— 不用先分词，按字对齐就行
        m, n = len(generated), len(reference)

        if m == 0 or n == 0:
            return 0.0

        # 动态规划计算LCS（最长公共子序列）
        # dp[i][j] = 前 i 个生成字符与前 j 个参考字符的最长公共子序列长度
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if generated[i - 1] == reference[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        lcs_length = dp[m][n]

        # ROUGE-L = LCS / reference_length
        rouge_l = lcs_length / n if n > 0 else 0

        return rouge_l

    def _calculate_faithfulness(self, answer: str, documents: List[Dict]) -> float:
        """
        计算忠实度（答案是否基于检索文档）

        Args:
            answer: 生成的答案
            documents: 检索到的文档

        Returns:
            忠实度分数
        """
        if not documents:
            return 0.0

        # 简化版：检查答案中的关键词是否出现在文档中
        # ⚠️ 靠空白切词，中文句子没有空格 → 整句被当成一个"词"，
        #    交集几乎恒为空。所以中文场景这个分数基本恒为 0，不代表答案不忠实。
        #    同一处粗糙也存在于 context_compressor._calculate_relevance。
        answer_words = set(answer.lower().split())

        # 统计有多少答案词出现在文档中
        doc_words = set()
        for doc in documents:
            content = doc.get("content", "").lower()
            doc_words.update(content.split())

        if not answer_words:
            return 0.0

        overlap = answer_words & doc_words
        faithfulness = len(overlap) / len(answer_words)

        return faithfulness

    def _calculate_relevance(self, generated: str, reference: str) -> float:
        """
        计算相关性（简化版）

        Args:
            generated: 生成的文本
            reference: 参考文本

        Returns:
            相关性分数
        """
        # 简化版：计算词汇重叠度
        gen_words = set(generated.lower().split())
        ref_words = set(reference.lower().split())

        if not gen_words or not ref_words:
            return 0.0

        overlap = gen_words & ref_words
        relevance = len(overlap) / len(ref_words)

        return relevance

    def _percentile(self, data: List[float], percentile: int) -> float:
        """
        计算百分位数

        Args:
            data: 数据列表
            percentile: 百分位（0-100）

        Returns:
            百分位值
        """
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]


class OnlineMetrics:
    """
    在线指标收集器

    收集用户反馈和系统指标

    接线现状：只有 `record_query` 在 knowledge_agent.handle 主路径被调（agent.py:851），
    `get_metrics` 在 get_stats 报表被调（1665）；`record_feedback / record_click /
    record_impression` **全仓库无调用方** —— 没有前端回调把它们接进来，
    所以 satisfaction_rate / ctr 现在恒为 0。
    """

    def __init__(self):
        """初始化在线指标收集器"""
        self.metrics = {
            "total_queries": 0,
            "thumbs_up": 0,
            "thumbs_down": 0,
            "clicks": 0,
            "impressions": 0
        }

        logger.info("在线指标收集器初始化")

    def record_query(self):
        """记录查询"""
        self.metrics["total_queries"] += 1

    def record_feedback(self, is_positive: bool):
        """
        记录用户反馈

        Args:
            is_positive: 是否为正面反馈
        """
        if is_positive:
            self.metrics["thumbs_up"] += 1
        else:
            self.metrics["thumbs_down"] += 1

    def record_click(self):
        """记录点击"""
        self.metrics["clicks"] += 1

    def record_impression(self):
        """记录展示"""
        self.metrics["impressions"] += 1

    def get_metrics(self) -> Dict:
        """
        获取指标

        Returns:
            指标字典
        """
        total_feedback = self.metrics["thumbs_up"] + self.metrics["thumbs_down"]
        satisfaction = self.metrics["thumbs_up"] / total_feedback if total_feedback > 0 else 0

        # CTR（点击率）= 点击 / 曝光；分母为 0 时返回 0，避免除零
        ctr = self.metrics["clicks"] / self.metrics["impressions"] if self.metrics["impressions"] > 0 else 0

        return {
            "total_queries": self.metrics["total_queries"],
            "satisfaction_rate": f"{satisfaction:.2%}",
            "ctr": f"{ctr:.2%}",
            "raw_metrics": self.metrics
        }
