# -*- coding: utf-8 -*-
"""
Agent路由器
===========

使用向量相似度+LLM精排选择最合适的Agent
"""

import json
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class AgentRouter:
    """Agent路由器"""

    def __init__(self, llm, embedder):
        """
        初始化

        Args:
            llm: LLM实例（Qwen-Max）
            embedder: 向量嵌入器
        """
        self.llm = llm
        self.embedder = embedder
        self.agent_vectors = {}  # {agent_id: vector}

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
        路由任务到Agent（优化版：精排+验证合并为一次LLM调用）

        流程：
        0. 前置拦截：闲聊/问候/极短查询直接路由到 chat_agent
        1. 向量相似度召回 top-3 候选
        2. LLM 一次调用完成精排+验证（输出选择、置信度、理由）

        Args:
            task: 任务信息
            agents: 可用的Agent列表
            use_llm_rerank: 是否使用LLM精排
            enable_auto_correct: 是否启用自动纠正（保留参数兼容性，已融入精排）

        Returns:
            选中的Agent ID
        """
        if not agents:
            raise ValueError("没有可用的Agent")

        # 如果只有一个Agent，直接返回
        if len(agents) == 1:
            return agents[0]['id']

        # ===== 前置拦截：闲聊/问候/极短查询直接路由到 chat_agent =====
        task_desc = task.get('description', '').strip()
        chat_agent_available = any(a['id'] == 'chat_agent' for a in agents)
        if chat_agent_available and self._is_chitchat(task_desc):
            logger.info(f"[路由] 前置拦截: 闲聊/问候 → chat_agent (query='{task_desc}')")
            return "chat_agent"

        # 第一步：向量相似度召回top-3
        candidates = self._vector_recall(task, agents, top_k=3)

        # 如果只有一个候选，直接返回
        if len(candidates) == 1:
            return candidates[0][0]['id']

        # 第二步：LLM精排+验证（合并为一次调用）
        if use_llm_rerank:
            selected_agent_id = self._llm_select(task, candidates, agents)
        else:
            # 直接返回相似度最高的
            selected_agent_id = candidates[0][0]['id']

        logger.info(f"路由任务到Agent: {selected_agent_id}")
        return selected_agent_id

    @staticmethod
    def _is_chitchat(query: str) -> bool:
        """
        判断是否为闲聊/问候/无实质内容的查询

        Args:
            query: 用户查询文本

        Returns:
            是否为闲聊
        """
        # 极短查询（<=5字符）且不含技术关键词，视为闲聊
        if len(query) <= 5:
            tech_keywords = ["RAG", "SQL", "Agent", "API", "检索", "查询", "数据", "分析", "对比", "合同"]
            if not any(kw.lower() in query.lower() for kw in tech_keywords):
                return True

        # 常见问候和闲聊模式
        chitchat_patterns = [
            "hi", "hello", "hey", "你好", "嗨", "哈喽", "在吗", "在不在",
            "你是谁", "你叫什么", "谢谢", "感谢", "再见", "拜拜", "bye",
            "早上好", "下午好", "晚上好", "早安", "晚安", "good morning",
            "good afternoon", "good evening", "good night",
            "ok", "好的", "嗯", "哦", "呵呵", "哈哈", "666", "厉害",
        ]
        query_lower = query.lower().strip()
        if query_lower in chitchat_patterns:
            return True

        return False

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

    def _llm_select(
        self,
        task: Dict,
        candidates: List[Tuple[Dict, float]],
        all_agents: List[Dict]
    ) -> str:
        """
        LLM精排+验证（合并为一次调用）

        将原来的 _llm_rerank() + IntentValidator.validate_routing()
        合并为一次 LLM 调用，同时完成选择和验证。

        Args:
            task: 任务信息
            candidates: 向量召回的候选Agent列表
            all_agents: 所有可用Agent列表（用于验证候选之外是否有更合适的）

        Returns:
            选中的Agent ID
        """
        task_desc = task.get('description', '')

        # 构建候选Agent描述
        candidates_desc = "\n".join([
            f"{i+1}. {agent['id']}: {agent['name']}\n"
            f"   描述: {agent['description']}\n"
            f"   能力: {', '.join(agent['capabilities'])}\n"
            f"   相似度: {sim:.3f}"
            for i, (agent, sim) in enumerate(candidates)
        ])

        # 构建所有Agent描述（用于验证是否有候选之外更合适的）
        candidate_ids = {agent['id'] for agent, _ in candidates}
        other_agents = [a for a in all_agents if a['id'] not in candidate_ids]

        other_agents_desc = ""
        if other_agents:
            other_agents_desc = "\n\n其他可用Agent（未进入候选）:\n" + "\n".join([
                f"- {agent['id']}: {agent['name']}\n"
                f"  描述: {agent['description']}\n"
                f"  能力: {', '.join(agent['capabilities'])}"
                for agent in other_agents
            ])

        prompt = (
            f"请为以下任务选择最合适的Agent。\n\n"
            f"任务: {task_desc}\n\n"
            f"候选Agent（按相似度排序）:\n"
            f"{candidates_desc}{other_agents_desc}\n\n"
            "选择规则：\n"
            "1. 涉及数据库操作（查询表、SQL、统计数据库中的记录/数据）-> database_agent\n"
            "2. 涉及知识概念解释、原理、对比分析 -> knowledge_agent\n"
            "3. 涉及已上传的文件分析（Excel/CSV/PDF/Word/TXT等文档）-> document_agent\n"
            "4. 涉及客户投诉、咨询、工单 -> customer_service_agent\n"
            "5. 涉及合同审核、文档风险分析 -> document_agent\n"
            "6. 涉及图片分析、图表识别 -> vqa_agent\n"
            "7. 闲聊、问候、意图不明确 -> chat_agent\n\n"
            "请仔细分析任务需求与每个Agent的能力匹配度，选择最合适的Agent。\n"
            "如果候选列表中的Agent都不合适，可以从其他可用Agent中选择。\n\n"
            '只输出JSON格式：\n'
            '{"selected_agent_id": "agent_id", "confidence": "high/medium/low", "reason": "选择理由"}'
        )

        try:
            response = self.llm.generate(prompt, temperature=0.1, max_tokens=200)

            from pydantic import BaseModel, Field
            from llm.langchain_parser import parse_llm_output

            class RouteResult(BaseModel):
                selected_agent_id: str = Field(description="选择的Agent ID")
                confidence: str = Field(default="medium", description="置信度")
                reason: str = Field(default="", description="选择理由")

            result = parse_llm_output(response, RouteResult, fallback=None)

            if result and result.selected_agent_id:
                selected_id = result.selected_agent_id
                confidence = result.confidence
                reason = result.reason

                # 验证ID是否有效（候选 + 全部Agent）
                all_valid_ids = {a['id'] for a in all_agents}
                if selected_id in all_valid_ids:
                    if selected_id not in candidate_ids:
                        logger.info(f"[路由] LLM精排修正: 从候选外选择了更合适的 {selected_id}，理由: {reason}")
                    else:
                        logger.info(f"[路由] LLM选择: {selected_id}（置信度: {confidence}）")
                    return selected_id

            # JSON 解析成功但 ID 无效，降级
            logger.warning(f"LLM返回无效结果，使用相似度最高的")
            return candidates[0][0]['id']

        except Exception as e:
            logger.error(f"LLM精排+验证失败: {e}，使用相似度最高的")
            return candidates[0][0]['id']

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

        prompt = (
            f"请选择最适合执行以下任务的Agent。\n\n"
            f"任务: {task_desc}\n\n"
            f"候选Agent:\n{candidates_desc}\n\n"
            f"请分析任务需求，选择最合适的Agent。只输出Agent的ID，不要有其他内容。\n\n"
            f"选择的Agent ID:"
        )

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

        # 默认兜底返回chat_agent
        fallback = "chat_agent" if any(a['id'] == "chat_agent" for a in agents) else "knowledge_agent"
        return fallback
