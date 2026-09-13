# -*- coding: utf-8 -*-
"""
Agent路由器
===========

使用向量相似度+LLM精排选择最合适的Agent
"""

import json
import logging
from typing import List, Dict, Optional, Tuple
from orchestrator.error_handler import AgentErrorHandler
from orchestrator.intent_validator import IntentValidator

logger = logging.getLogger(__name__)


class AgentRouter:
    """Agent路由器"""

    def __init__(self, llm, embedder, enable_validation=True):
        """
        初始化

        Args:
            llm: LLM实例（Qwen-Max）
            embedder: 向量嵌入器
            enable_validation: 是否启用意图验证
        """
        self.llm = llm
        self.embedder = embedder
        self.agent_vectors = {}  # {agent_id: vector}
        self.enable_validation = enable_validation
        self.validator = IntentValidator(llm) if enable_validation else None
        self.error_handler = AgentErrorHandler()

    def index_agents(self, agents: List[Dict]):
        """
        为Agent建立向量索引

        Args:
            agents: Agent列表
        """
        for agent in agents:
            # 组合描述和能力作为索引文本
            text = f"{agent['description']} {' '.join(agent['capabilities'])}"
            vector = self.embedder.embed(text)
            self.agent_vectors[agent['id']] = vector

        logger.info(f"已为 {len(agents)} 个Agent建立索引")

    def route(
        self,
        task: Dict,
        agents: List[Dict],
        use_llm_rerank: bool = True,
        enable_auto_correct: bool = True
    ) -> str:
        """
        路由任务到Agent（带容错和纠正）

        Args:
            task: 任务信息
            agents: 可用的Agent列表
            use_llm_rerank: 是否使用LLM精排
            enable_auto_correct: 是否启用自动纠正

        Returns:
            选中的Agent ID
        """
        if not agents:
            raise ValueError("没有可用的Agent")

        # 如果只有一个Agent，直接返回
        if len(agents) == 1:
            return agents[0]['id']

        # 第一步：向量相似度召回top-3
        candidates = self._vector_recall(task, agents, top_k=3)

        # 如果只有一个候选，直接返回
        if len(candidates) == 1:
            return candidates[0][0]['id']

        # 第二步：LLM精排
        if use_llm_rerank:
            selected_agent_id = self._llm_rerank(task, candidates)
        else:
            # 直接返回相似度最高的
            selected_agent_id = candidates[0][0]['id']

        # 第三步：意图验证与纠正
        if enable_auto_correct and self.enable_validation:
            selected_agent = next((a for a in agents if a['id'] == selected_agent_id), None)
            if selected_agent:
                is_correct, suggested_id, reason = self.validator.validate_routing(
                    task.get('description', ''),
                    selected_agent,
                    agents
                )

                if not is_correct and suggested_id:
                    logger.warning(f"路由纠正: {selected_agent_id} -> {suggested_id}")
                    logger.warning(f"纠正原因: {reason}")
                    selected_agent_id = suggested_id

        logger.info(f"路由任务到Agent: {selected_agent_id}")
        return selected_agent_id

    def _vector_recall(
        self,
        task: Dict,
        agents: List[Dict],
        top_k: int = 3
    ) -> List[Tuple[Dict, float]]:
        """
        向量相似度召回

        Args:
            task: 任务信息
            agents: Agent列表
            top_k: 召回数量

        Returns:
            候选Agent列表（agent, similarity）
        """
        # 任务描述向量化
        task_desc = task.get('description', '')
        logger.info(f"[RAG步骤1] 用户问题向量化: '{task_desc}'")
        task_vector = self.embedder.embed(task_desc)
        logger.info(f"[RAG步骤1] 向量化完成，维度: {len(task_vector)}")

        # 计算相似度
        similarities = []
        for agent in agents:
            agent_id = agent['id']
            if agent_id not in self.agent_vectors:
                # 如果没有索引，现场计算
                text = f"{agent['description']} {' '.join(agent['capabilities'])}"
                agent_vector = self.embedder.embed(text)
                self.agent_vectors[agent_id] = agent_vector
            else:
                agent_vector = self.agent_vectors[agent_id]

            # 计算余弦相似度
            from llm.embedder import cosine_similarity
            sim = cosine_similarity(task_vector, agent_vector)
            similarities.append((agent, sim))

        # 排序并返回top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        candidates = similarities[:top_k]

        logger.info(f"[RAG步骤2] Agent路由完成，候选: {[(c[0]['name'], f'{c[1]:.3f}') for c in candidates]}")
        logger.debug(f"向量召回候选: {[(c[0]['id'], c[1]) for c in candidates]}")
        return candidates

    def _llm_rerank(
        self,
        task: Dict,
        candidates: List[Tuple[Dict, float]]
    ) -> str:
        """
        LLM精排

        Args:
            task: 任务信息
            candidates: 候选Agent列表

        Returns:
            选中的Agent ID
        """
        # 构建提示词
        task_desc = task.get('description', '')

        candidates_desc = "\n".join([
            f"{i+1}. {agent['id']}: {agent['name']}\n"
            f"   描述: {agent['description']}\n"
            f"   能力: {', '.join(agent['capabilities'])}\n"
            f"   相似度: {sim:.3f}"
            for i, (agent, sim) in enumerate(candidates)
        ])

        prompt = f"""请选择最适合执行以下任务的Agent。

任务: {task_desc}

候选Agent:
{candidates_desc}

请分析任务需求，选择最合适的Agent。只输出Agent的ID，不要有其他内容。

选择的Agent ID:"""

        try:
            response = self.llm.generate(prompt, temperature=0.1, max_tokens=50)
            selected_id = response.strip()

            # 验证ID是否有效
            valid_ids = [agent['id'] for agent, _ in candidates]
            if selected_id in valid_ids:
                logger.debug(f"LLM选择: {selected_id}")
                return selected_id
            else:
                logger.warning(f"LLM返回无效ID: {selected_id}，使用相似度最高的")
                return candidates[0][0]['id']

        except Exception as e:
            logger.error(f"LLM精排失败: {e}，使用相似度最高的")
            return candidates[0][0]['id']

    def route_simple(self, query: str, agents: List[Dict]) -> str:
        """
        简单路由（基于关键词）

        Args:
            query: 查询文本
            agents: Agent列表

        Returns:
            Agent ID
        """
        query_lower = query.lower()

        # 关键词映射
        keyword_mapping = {
            "code_agent": ["代码", "编程", "code", "python", "java", "函数", "算法"],
            "customer_agent": ["客服", "工单", "订单", "投诉", "咨询", "用户"],
            "knowledge_agent": ["什么是", "如何", "为什么", "解释", "原理"]
        }

        # 匹配关键词
        for agent_id, keywords in keyword_mapping.items():
            if any(kw in query_lower for kw in keywords):
                # 检查Agent是否可用
                if any(a['id'] == agent_id for a in agents):
                    logger.debug(f"关键词路由: {agent_id}")
                    return agent_id

        # 默认返回knowledge_agent
        return "knowledge_agent"
