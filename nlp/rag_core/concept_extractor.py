# -*- coding: utf-8 -*-
"""
跨文档语义概念提取器
==================

从多个检索到的文档中提取共同的语义概念，构建概念关系图
"""

import logging
import json
import re
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, Counter
import numpy as np

logger = logging.getLogger(__name__)


class ConceptExtractor:
    """
    跨文档语义概念提取器

    功能：
    1. 从多个文档中提取核心概念（实体、主题、术语）
    2. 使用语义相似度聚类相似概念
    3. 构建概念-文档映射关系
    4. 计算概念重要性权重
    5. 提取概念间的关系
    """

    def __init__(
        self,
        llm,
        embedder=None,
        similarity_threshold: float = 0.8,
        min_concept_freq: int = 1,
        max_concepts_per_doc: int = 5
    ):
        """
        初始化概念提取器

        Args:
            llm: LLM实例（用于概念提取）
            embedder: 嵌入器实例（用于概念聚类）
            similarity_threshold: 概念相似度阈值（默认0.8）
            min_concept_freq: 最小概念频率（默认1）
            max_concepts_per_doc: 每个文档最多提取的概念数（默认5）
        """
        self.llm = llm
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.min_concept_freq = min_concept_freq
        self.max_concepts_per_doc = max_concepts_per_doc

        logger.info(f"概念提取器初始化完成 (相似度阈值: {similarity_threshold})")

    def extract(
        self,
        query: str,
        documents: List[Dict],
        extract_relations: bool = True
    ) -> Dict:
        """
        从多个文档中提取跨文档语义概念

        Args:
            query: 用户查询（用于提取相关概念）
            documents: 文档列表，每个文档格式：{"content": "...", "metadata": {...}}
            extract_relations: 是否提取概念间关系（默认True）

        Returns:
            概念提取结果：
            {
                "concepts": [
                    {
                        "name": "概念名称",
                        "type": "概念类型",
                        "weight": 0.95,
                        "doc_indices": [0, 2, 5],
                        "mentions": 3
                    }
                ],
                "relations": [
                    {
                        "source": "概念A",
                        "relation": "关系类型",
                        "target": "概念B",
                        "weight": 0.8
                    }
                ],
                "concept_map": {
                    "概念A": [0, 1, 3],  # 出现在文档0, 1, 3中
                    "概念B": [1, 2]
                },
                "summary": "概念摘要"
            }
        """
        if not documents:
            logger.warning("文档列表为空，无法提取概念")
            return self._empty_result()

        logger.info(f"[概念提取] 开始从 {len(documents)} 个文档中提取概念")
        start_time = time.time()

        # 第一步：从每个文档中提取概念
        doc_concepts = self._extract_concepts_from_documents(query, documents)

        # 第二步：概念聚类（合并相似概念）
        clustered_concepts = self._cluster_concepts(doc_concepts)

        # 第三步：计算概念权重
        weighted_concepts = self._calculate_concept_weights(
            clustered_concepts,
            len(documents)
        )

        # 第四步：构建概念-文档映射
        concept_map = self._build_concept_map(clustered_concepts)

        # 第五步：提取概念间关系（可选）
        relations = []
        if extract_relations:
            relations = self._extract_concept_relations(
                weighted_concepts,
                documents
            )

        # 第六步：生成概念摘要
        summary = self._generate_concept_summary(
            query,
            weighted_concepts,
            relations
        )

        elapsed = time.time() - start_time
        logger.info(f"[概念提取] 完成，提取 {len(weighted_concepts)} 个概念，耗时 {elapsed:.2f}s")

        return {
            "concepts": weighted_concepts,
            "relations": relations,
            "concept_map": concept_map,
            "summary": summary,
            "metadata": {
                "num_documents": len(documents),
                "num_concepts": len(weighted_concepts),
                "num_relations": len(relations),
                "elapsed_time": elapsed
            }
        }

    def _extract_concepts_from_documents(
        self,
        query: str,
        documents: List[Dict]
    ) -> List[List[Dict]]:
        """
        从每个文档中提取概念

        Returns:
            每个文档的概念列表：[[doc0_concepts], [doc1_concepts], ...]
        """
        all_doc_concepts = []

        for idx, doc in enumerate(documents):
            content = doc.get("content", "")
            if not content or len(content.strip()) < 10:
                all_doc_concepts.append([])
                continue

            # 截断过长的文档（避免超过LLM上下文限制）
            if len(content) > 1500:
                content = content[:1500] + "..."

            try:
                concepts = self._extract_concepts_with_llm(query, content, idx)
                all_doc_concepts.append(concepts)
                logger.debug(f"  文档 {idx}: 提取 {len(concepts)} 个概念")
            except Exception as e:
                logger.error(f"  文档 {idx} 概念提取失败: {e}")
                all_doc_concepts.append([])

        return all_doc_concepts

    def _extract_concepts_with_llm(
        self,
        query: str,
        content: str,
        doc_idx: int
    ) -> List[Dict]:
        """
        使用LLM从单个文档中提取概念

        Returns:
            概念列表：[{"name": "...", "type": "...", "doc_idx": 0}, ...]
        """
        prompt = f"""任务：从以下文档中提取与查询相关的核心概念。

查询：{query}

文档：
{content}

要求：
1. 提取 {self.max_concepts_per_doc} 个最重要的概念（实体、主题、关键术语）
2. 概念应该是名词或名词短语
3. 优先提取与查询相关的专业术语和核心主题
4. 为每个概念标注类型（实体/主题/术语/技术/产品/组织/人物等）

输出格式（JSON）：
{{
  "concepts": [
    {{"name": "概念名称", "type": "概念类型"}},
    {{"name": "概念名称", "type": "概念类型"}}
  ]
}}

只输出JSON，不要其他内容。

输出："""

        try:
            response = self.llm.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=300
            ).strip()

            # 解析JSON
            result = self._parse_llm_json(response)

            if not result or "concepts" not in result:
                logger.warning(f"LLM返回格式错误: {response[:100]}")
                return []

            # 添加文档索引
            concepts = []
            for concept in result["concepts"][:self.max_concepts_per_doc]:
                if "name" in concept and concept["name"].strip():
                    concepts.append({
                        "name": concept["name"].strip(),
                        "type": concept.get("type", "未知").strip(),
                        "doc_idx": doc_idx
                    })

            return concepts

        except Exception as e:
            logger.error(f"LLM概念提取失败: {e}")
            return []

    def _cluster_concepts(
        self,
        doc_concepts: List[List[Dict]]
    ) -> Dict[str, List[Dict]]:
        """
        使用语义相似度聚类概念（合并相似概念）

        Args:
            doc_concepts: 每个文档的概念列表

        Returns:
            聚类后的概念：{"聚类名": [概念1, 概念2, ...]}
        """
        # 展平所有概念
        all_concepts = []
        for concepts in doc_concepts:
            all_concepts.extend(concepts)

        if not all_concepts:
            return {}

        # 如果没有embedder，直接按名称分组（不聚类）
        if not self.embedder:
            return self._group_by_name(all_concepts)

        # 提取概念名称
        concept_names = [c["name"] for c in all_concepts]

        try:
            # 向量化概念名称
            embeddings = self.embedder.embed_batch(concept_names)

            # 计算相似度矩阵
            similarity_matrix = np.dot(embeddings, embeddings.T)

            # 聚类：相似度 > threshold 的概念合并
            clusters = {}
            visited = set()

            for i, concept in enumerate(all_concepts):
                if i in visited:
                    continue

                # 找到相似概念
                similar_indices = np.where(
                    similarity_matrix[i] > self.similarity_threshold
                )[0]

                # 收集相似概念
                cluster_concepts = [all_concepts[j] for j in similar_indices]

                # 使用最常见的概念名作为聚类名
                cluster_names = [c["name"] for c in cluster_concepts]
                cluster_name = Counter(cluster_names).most_common(1)[0][0]

                clusters[cluster_name] = cluster_concepts
                visited.update(similar_indices)

            logger.info(f"  概念聚类: {len(all_concepts)} 个概念 → {len(clusters)} 个聚类")
            return clusters

        except Exception as e:
            logger.error(f"概念聚类失败: {e}，降级为按名称分组")
            return self._group_by_name(all_concepts)

    def _group_by_name(self, concepts: List[Dict]) -> Dict[str, List[Dict]]:
        """按概念名称分组（不使用语义聚类）"""
        groups = defaultdict(list)
        for concept in concepts:
            groups[concept["name"]].append(concept)
        return dict(groups)

    def _calculate_concept_weights(
        self,
        clustered_concepts: Dict[str, List[Dict]],
        total_docs: int
    ) -> List[Dict]:
        """
        计算概念权重

        权重计算公式：
        weight = (mentions / total_docs) * (1 + type_bonus)

        Args:
            clustered_concepts: 聚类后的概念
            total_docs: 文档总数

        Returns:
            带权重的概念列表（按权重降序）
        """
        weighted_concepts = []

        # 类型权重加成
        type_weights = {
            "实体": 1.2,
            "主题": 1.3,
            "术语": 1.1,
            "技术": 1.2,
            "产品": 1.1,
            "组织": 1.0,
            "人物": 1.0,
            "未知": 1.0
        }

        for cluster_name, concepts in clustered_concepts.items():
            # 统计出现次数和文档索引
            doc_indices = list(set(c["doc_idx"] for c in concepts))
            mentions = len(concepts)

            # 统计最常见的类型
            types = [c["type"] for c in concepts]
            most_common_type = Counter(types).most_common(1)[0][0]

            # 计算权重
            doc_coverage = len(doc_indices) / total_docs
            type_bonus = type_weights.get(most_common_type, 1.0)
            weight = doc_coverage * type_bonus

            weighted_concepts.append({
                "name": cluster_name,
                "type": most_common_type,
                "weight": round(weight, 3),
                "doc_indices": sorted(doc_indices),
                "mentions": mentions,
                "doc_coverage": round(doc_coverage, 2)
            })

        # 按权重降序排序
        weighted_concepts.sort(key=lambda x: x["weight"], reverse=True)

        # 过滤低频概念
        weighted_concepts = [
            c for c in weighted_concepts
            if c["mentions"] >= self.min_concept_freq
        ]

        return weighted_concepts

    def _build_concept_map(
        self,
        clustered_concepts: Dict[str, List[Dict]]
    ) -> Dict[str, List[int]]:
        """
        构建概念-文档映射

        Returns:
            {"概念名": [文档索引列表]}
        """
        concept_map = {}
        for cluster_name, concepts in clustered_concepts.items():
            doc_indices = sorted(set(c["doc_idx"] for c in concepts))
            concept_map[cluster_name] = doc_indices
        return concept_map

    def _extract_concept_relations(
        self,
        concepts: List[Dict],
        documents: List[Dict]
    ) -> List[Dict]:
        """
        提取概念间的关系

        策略：
        1. 找到共现的概念对（出现在同一文档中）
        2. 使用LLM提取关系类型

        Returns:
            关系列表：[{"source": "A", "relation": "包含", "target": "B", "weight": 0.8}]
        """
        if len(concepts) < 2:
            return []

        # 只提取top概念的关系（避免组合爆炸）
        top_concepts = concepts[:min(10, len(concepts))]

        # 找到共现的概念对
        co_occurring_pairs = self._find_co_occurring_concepts(
            top_concepts,
            documents
        )

        if not co_occurring_pairs:
            return []

        # 使用LLM提取关系（批量处理）
        relations = self._extract_relations_with_llm(
            co_occurring_pairs,
            documents
        )

        return relations

    def _find_co_occurring_concepts(
        self,
        concepts: List[Dict],
        documents: List[Dict]
    ) -> List[Tuple[str, str, List[int]]]:
        """
        找到共现的概念对

        Returns:
            [(概念A, 概念B, [共现文档索引]), ...]
        """
        pairs = []

        for i, concept_a in enumerate(concepts):
            for concept_b in concepts[i+1:]:
                # 找到共同出现的文档
                common_docs = set(concept_a["doc_indices"]) & set(concept_b["doc_indices"])

                if common_docs:
                    pairs.append((
                        concept_a["name"],
                        concept_b["name"],
                        sorted(common_docs)
                    ))

        return pairs

    def _extract_relations_with_llm(
        self,
        pairs: List[Tuple[str, str, List[int]]],
        documents: List[Dict]
    ) -> List[Dict]:
        """
        使用LLM提取概念对之间的关系
        """
        relations = []

        # 限制提取数量（避免过多LLM调用）
        max_pairs = 5
        for concept_a, concept_b, doc_indices in pairs[:max_pairs]:
            # 获取共现文档的内容片段
            context = self._get_context_for_concepts(
                concept_a,
                concept_b,
                doc_indices,
                documents
            )

            if not context:
                continue

            try:
                relation = self._extract_relation_with_llm(
                    concept_a,
                    concept_b,
                    context
                )

                if relation:
                    relations.append({
                        "source": concept_a,
                        "relation": relation,
                        "target": concept_b,
                        "weight": round(len(doc_indices) / len(documents), 2),
                        "doc_indices": doc_indices
                    })
            except Exception as e:
                logger.error(f"提取关系失败 ({concept_a} - {concept_b}): {e}")

        return relations

    def _get_context_for_concepts(
        self,
        concept_a: str,
        concept_b: str,
        doc_indices: List[int],
        documents: List[Dict]
    ) -> str:
        """获取包含两个概念的上下文片段"""
        contexts = []

        for idx in doc_indices[:2]:  # 最多取2个文档
            if idx >= len(documents):
                continue

            content = documents[idx].get("content", "")

            # 找到包含两个概念的句子
            sentences = content.split('。')
            for sentence in sentences:
                if concept_a in sentence and concept_b in sentence:
                    contexts.append(sentence.strip() + '。')
                    break

        return ' '.join(contexts)

    def _extract_relation_with_llm(
        self,
        concept_a: str,
        concept_b: str,
        context: str
    ) -> Optional[str]:
        """使用LLM提取两个概念之间的关系"""
        prompt = f"""任务：分析"{concept_a}"和"{concept_b}"之间的关系。

上下文：
{context}

要求：
1. 用一个简短的词或短语描述关系（如：包含、属于、使用、依赖、相关、对比等）
2. 只输出关系词，不要其他内容
3. 如果没有明确关系，输出"相关"

关系："""

        try:
            relation = self.llm.generate(
                prompt=prompt,
                temperature=0.1,
                max_tokens=20
            ).strip()

            # 清理输出
            relation = re.sub(r'[^\u4e00-\u9fa5a-zA-Z]', '', relation)

            if relation and len(relation) <= 10:
                return relation
            else:
                return "相关"
        except:
            return "相关"

    def _generate_concept_summary(
        self,
        query: str,
        concepts: List[Dict],
        relations: List[Dict]
    ) -> str:
        """
        生成概念摘要

        Returns:
            概念摘要文本
        """
        if not concepts:
            return "未提取到有效概念。"

        # 构建摘要
        top_concepts = concepts[:5]
        concept_names = [c["name"] for c in top_concepts]

        summary_parts = []
        summary_parts.append(f"针对查询「{query}」，从文档中提取到 {len(concepts)} 个核心概念。")
        summary_parts.append(f"最重要的概念包括：{', '.join(concept_names)}。")

        if relations:
            summary_parts.append(f"发现 {len(relations)} 个概念间关系。")

        return ' '.join(summary_parts)

    def _parse_llm_json(self, text: str) -> Optional[Dict]:
        """解析LLM返回的JSON"""
        try:
            # 尝试直接解析
            return json.loads(text)
        except:
            # 尝试提取JSON块
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except:
                    pass
        return None

    def _empty_result(self) -> Dict:
        """返回空结果"""
        return {
            "concepts": [],
            "relations": [],
            "concept_map": {},
            "summary": "文档列表为空，无法提取概念。",
            "metadata": {
                "num_documents": 0,
                "num_concepts": 0,
                "num_relations": 0,
                "elapsed_time": 0
            }
        }


# 导入time模块（补充）
import time
