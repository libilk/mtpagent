# -*- coding: utf-8 -*-
"""
重排序器模块
============

提供三种重排序策略，按优先级自动降级：
1. DashScopeReranker: 调用阿里云百炼 Rerank API（推荐，无需本地模型）
2. CrossEncoderReranker: 本地 Cross-Encoder 模型（需要下载模型）
3. SimpleReranker: 基于规则的轻量方案（兜底）

使用 RerankerFactory.create_reranker() 创建实例。
"""

import os
import json
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


# ==========================================================
# 方式一：DashScope Rerank API（推荐）
# ==========================================================

class DashScopeReranker:
    """
    阿里云百炼 Rerank API 重排序器

    调用 DashScope 的 gte-rerank 模型，通过 API 实现交叉编码重排序。
    无需下载模型，无需 GPU，成本极低（0.0008元/千tokens）。

    使用方式：
        reranker = DashScopeReranker()
        scores = reranker.rank(query, documents)
    """

    def __init__(
        self,
        model: str = "gte-rerank",
        api_key: Optional[str] = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        top_n: Optional[int] = None,
    ):
        """
        初始化 DashScope 重排序器

        Args:
            model: 重排序模型名称（gte-rerank / gte-rerank-v2）
            api_key: DashScope API Key（默认读取 DASHSCOPE_API_KEY 环境变量）
            base_url: API 地址
            top_n: 返回前 N 个结果（None 表示返回全部）
        """
        self.model = model
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.base_url = base_url
        self.top_n = top_n

        if not self.api_key:
            raise ValueError(
                "DashScope API Key 未设置，请设置 DASHSCOPE_API_KEY 环境变量"
            )

        logger.info(f"DashScope 重排序器初始化完成 (model={model})")

    def rank(self, query: str, documents: List[str]) -> List[float]:
        """
        对文档列表进行重排序打分

        Args:
            query: 查询文本
            documents: 文档内容列表

        Returns:
            分数列表（与 documents 顺序一一对应）
        """
        import requests

        if not documents:
            return []

        try:
            # 构造请求
            url = f"{self.base_url}/rerank"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": self.model,
                "query": query,
                "documents": documents,
            }
            if self.top_n is not None:
                payload["top_n"] = self.top_n

            # 调用 API
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()

            result = response.json()

            # 解析结果：API 返回按相关性排序的结果列表
            # 每个结果包含 index（原始位置）和 relevance_score
            results = result.get("results", [])

            # 构建分数数组（按原始文档顺序）
            scores = [0.0] * len(documents)
            for item in results:
                idx = item.get("index", 0)
                score = item.get("relevance_score", 0.0)
                if 0 <= idx < len(documents):
                    scores[idx] = score

            logger.debug(
                f"[DashScope Rerank] 完成 | "
                f"文档数={len(documents)}, "
                f"最高分={max(scores):.4f}, 最低分={min(scores):.4f}"
            )

            return scores

        except Exception as e:
            logger.error(f"[DashScope Rerank] API 调用失败: {e}")
            # 降级：返回均匀分数，不影响后续流程
            return [0.5] * len(documents)


# ==========================================================
# 方式二：本地 Cross-Encoder 模型
# ==========================================================

class CrossEncoderReranker:
    """
    本地 Cross-Encoder 重排序器

    下载并加载交叉编码器模型到本地运行。
    无需 API 调用，但需要下载模型文件（约 400MB）。

    推荐模型：BAAI/bge-reranker-base（中文效果好）

    使用方式：
        reranker = CrossEncoderReranker(model_path="BAAI/bge-reranker-base")
        scores = reranker.rank(query, documents)
    """

    def __init__(
        self,
        model_path: str = "BAAI/bge-reranker-base",
        device: Optional[str] = None,
        max_length: int = 512,
    ):
        """
        初始化本地 Cross-Encoder 重排序器

        Args:
            model_path: 模型路径（HuggingFace 模型名或本地路径）
            device: 计算设备（'cuda' / 'cpu'，默认自动检测）
            max_length: 最大序列长度
        """
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError:
            raise ImportError(
                "本地 Cross-Encoder 需要安装 torch 和 transformers，"
                "请运行: pip install torch transformers"
            )

        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"加载 Cross-Encoder 模型: {model_path} (device={self.device})")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Cross-Encoder 重排序器初始化完成")

    def rank(self, query: str, documents: List[str]) -> List[float]:
        """
        对文档列表进行重排序打分

        一次推理处理所有文档，速度快。

        Args:
            query: 查询文本
            documents: 文档内容列表

        Returns:
            分数列表（与 documents 顺序一一对应）
        """
        import torch

        if not documents:
            return []

        try:
            # 构造 query-document 对
            pairs = [[query, doc] for doc in documents]

            # Tokenize
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            # 推理
            with torch.no_grad():
                outputs = self.model(**inputs)
                # logits shape: (num_docs,) 或 (num_docs, 1)
                logits = outputs.logits.squeeze(-1)

                # Sigmoid 归一化到 0-1
                scores = torch.sigmoid(logits)

            scores_list = scores.cpu().tolist()

            # 确保返回的是列表（单个文档时 tolist() 可能返回标量）
            if isinstance(scores_list, float):
                scores_list = [scores_list]

            logger.debug(
                f"[CrossEncoder Rerank] 完成 | "
                f"文档数={len(documents)}, "
                f"最高分={max(scores_list):.4f}, 最低分={min(scores_list):.4f}"
            )

            return scores_list

        except Exception as e:
            logger.error(f"[CrossEncoder Rerank] 推理失败: {e}")
            return [0.5] * len(documents)


# ==========================================================
# 方式三：规则重排序（兜底方案）
# ==========================================================

class SimpleReranker:
    """
    简单的重排序器（基于规则）

    不依赖任何模型或 API，纯规则打分。
    当 API 和本地模型都不可用时作为兜底方案。
    """

    def __init__(self):
        logger.info("使用简单重排序器（基于规则）")

    def rank(self, query: str, documents: List[str]) -> List[float]:
        """
        基于简单规则打分

        规则：
        1. 文档长度适中（300-1000字符）→ +0.3
        2. 包含查询关键词 → +0.5
        3. 查询词出现位置靠前 → +0.2

        Args:
            query: 查询文本
            documents: 文档列表

        Returns:
            分数列表
        """
        scores = []
        query_lower = query.lower()

        for doc in documents:
            score = 0.0
            doc_lower = doc.lower()

            # 规则1：长度适中（300-1000字符）
            length = len(doc)
            if 300 <= length <= 1000:
                score += 0.3
            elif length < 300:
                score += 0.1
            else:
                score += 0.2

            # 规则2：包含查询关键词
            query_words = query_lower.split()
            matched_words = sum(1 for word in query_words if word in doc_lower)
            score += (matched_words / len(query_words)) * 0.5 if query_words else 0

            # 规则3：查询词出现位置靠前
            first_occurrence = doc_lower.find(query_lower)
            if first_occurrence != -1:
                position_score = max(0, 1 - first_occurrence / len(doc))
                score += position_score * 0.2

            scores.append(score)

        return scores


# ==========================================================
# 工厂类：统一创建入口
# ==========================================================

class RerankerFactory:
    """
    重排序器工厂

    根据类型创建对应的重排序器实例，支持自动降级：
    - api: DashScopeReranker（优先推荐）
    - cross_encoder: CrossEncoderReranker（需要本地模型）
    - rule: SimpleReranker（兜底）
    - auto: 按 api → cross_encoder → rule 顺序尝试

    使用方式：
        reranker = RerankerFactory.create_reranker("api")
        reranker = RerankerFactory.create_reranker("auto")
    """

    @staticmethod
    def create_reranker(
        reranker_type: str = "auto",
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        max_length: int = 512,
        api_key: Optional[str] = None,
        rerank_model: str = "gte-rerank",
    ):
        """
        创建重排序器

        Args:
            reranker_type: 类型（'api' / 'cross_encoder' / 'rule' / 'auto'）
            model_path: 本地模型路径（cross_encoder 模式）
            device: 计算设备（cross_encoder 模式）
            max_length: 最大序列长度（cross_encoder 模式）
            api_key: DashScope API Key（api 模式）
            rerank_model: API 重排序模型名（api 模式）

        Returns:
            重排序器实例
        """
        if reranker_type == "api":
            return DashScopeReranker(
                model=rerank_model,
                api_key=api_key,
            )

        elif reranker_type == "cross_encoder":
            return CrossEncoderReranker(
                model_path=model_path or "BAAI/bge-reranker-base",
                device=device,
                max_length=max_length,
            )

        elif reranker_type == "rule":
            return SimpleReranker()

        elif reranker_type == "auto":
            return RerankerFactory._auto_create(
                model_path=model_path,
                device=device,
                max_length=max_length,
                api_key=api_key,
                rerank_model=rerank_model,
            )

        else:
            logger.warning(f"未知的重排序器类型: {reranker_type}，降级为规则重排序")
            return SimpleReranker()

    @staticmethod
    def _auto_create(**kwargs):
        """
        自动选择最佳可用的重排序器

        尝试顺序：API → Cross-Encoder → Rule
        """
        # 1. 优先尝试 API（最方便，无需本地资源）
        api_key = kwargs.get("api_key") or os.getenv("DASHSCOPE_API_KEY")
        if api_key:
            try:
                reranker = DashScopeReranker(
                    model=kwargs.get("rerank_model", "gte-rerank"),
                    api_key=api_key,
                )
                logger.info("[Reranker Auto] 选择 DashScope API 重排序")
                return reranker
            except Exception as e:
                logger.warning(f"[Reranker Auto] API 重排序不可用: {e}")

        # 2. 尝试本地 Cross-Encoder
        try:
            model_path = kwargs.get("model_path") or "BAAI/bge-reranker-base"
            reranker = CrossEncoderReranker(
                model_path=model_path,
                device=kwargs.get("device"),
                max_length=kwargs.get("max_length", 512),
            )
            logger.info("[Reranker Auto] 选择本地 Cross-Encoder 重排序")
            return reranker
        except Exception as e:
            logger.warning(f"[Reranker Auto] 本地 Cross-Encoder 不可用: {e}")

        # 3. 兜底：规则重排序
        logger.info("[Reranker Auto] 降级为规则重排序")
        return SimpleReranker()
