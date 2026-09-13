# -*- coding: utf-8 -*-
"""
Orchestrator主模块
==================

协调任务规划、Agent路由和执行
"""

import logging
from typing import Dict, Optional

from orchestrator.planner import TaskPlanner
from orchestrator.router import AgentRouter
from orchestrator.executor import ReactExecutor
from orchestrator.registry import AgentRegistry
from orchestrator.error_handler import AgentErrorHandler

logger = logging.getLogger(__name__)


class Orchestrator:
    """主Agent - 协调器"""

    def __init__(self, llm_max, llm_plus, embedder):
        """
        初始化

        Args:
            llm_max: Qwen-Max实例（用于规划和路由）
            llm_plus: Qwen-Plus实例（备用）
            embedder: 向量嵌入器
        """
        self.llm_max = llm_max
        self.llm_plus = llm_plus
        self.embedder = embedder

        # 初始化组件
        self.registry = AgentRegistry()
        self.planner = TaskPlanner(llm_max)
        self.router = AgentRouter(llm_max, embedder)
        self.executor = ReactExecutor(llm_max, self.registry)
        self.error_handler = AgentErrorHandler()

        logger.info("Orchestrator初始化完成")

    def register_agent(
        self,
        agent_id: str,
        name: str,
        description: str,
        capabilities: list,
        agent_instance,
        model: str = "qwen-plus"
    ):
        """
        注册Agent

        Args:
            agent_id: Agent ID
            name: Agent名称
            description: Agent描述
            capabilities: 能力列表
            agent_instance: Agent实例
            model: 使用的模型
        """
        self.registry.register(
            agent_id=agent_id,
            name=name,
            description=description,
            capabilities=capabilities,
            agent_instance=agent_instance,
            model=model
        )

        # 更新路由器索引
        agents = self.registry.get_all_agents()
        self.router.index_agents(agents)

    def handle(self, query: str, use_planning: Optional[bool] = None) -> Dict:
        """
        处理用户查询

        Args:
            query: 用户查询
            use_planning: 是否使用任务规划（None表示自动判断）

        Returns:
            处理结果
        """
        logger.info(f"收到查询: {query}")

        try:
            # 获取可用的Agent
            agents = self.registry.get_all_agents()
            if not agents:
                return {"error": "没有可用的Agent"}

            # 判断是否需要任务规划
            if use_planning is None:
                use_planning = self.planner.should_plan(query)

            if use_planning:
                # 复杂查询：使用任务规划
                return self._handle_with_planning(query, agents)
            else:
                # 简单查询：直接路由
                return self._handle_simple(query, agents)

        except Exception as e:
            logger.error(f"处理查询失败: {e}", exc_info=True)
            return {"error": str(e)}

    def _handle_with_planning(self, query: str, agents: list) -> Dict:
        """
        使用任务规划处理查询

        Args:
            query: 用户查询
            agents: 可用的Agent列表

        Returns:
            处理结果
        """
        logger.info("使用任务规划模式")

        # 生成任务计划
        agent_descriptions = self.registry.get_agent_descriptions()
        plan = self.planner.plan(query, agent_descriptions)

        logger.info(f"任务计划: {plan.get('reasoning', '')}")
        logger.info(f"任务数量: {len(plan.get('tasks', []))}")

        # 执行任务计划
        result = self.executor.execute(plan, query)

        return {
            "mode": "planning",
            "plan": plan,
            "result": result.get("result", ""),
            "task_results": result.get("task_results", {}),
            "success": result.get("success", False)
        }

    def _handle_simple(self, query: str, agents: list) -> Dict:
        """
        简单路由处理查询（带容错和重试）

        Args:
            query: 用户查询
            agents: 可用的Agent列表

        Returns:
            处理结果
        """
        logger.info("使用简单路由模式")

        # 路由到Agent
        task = {"description": query}
        agent_id = self.router.route(task, agents)

        # 获取Agent实例
        agent_info = self.registry.get_agent(agent_id)
        if not agent_info:
            return {"error": f"Agent不存在: {agent_id}"}

        agent_instance = agent_info["instance"]

        # 执行（带容错和重试）
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                result = agent_instance.handle(query, {})

                # 验证响应格式
                validated_result = self.error_handler.validate_agent_response(result)

                # 如果响应失败且有fallback，尝试其他Agent
                if not validated_result.get("success", True) and attempt < max_retries:
                    logger.warning(f"Agent {agent_id} 响应异常，尝试fallback")
                    fallback_agent_id = self._get_fallback_agent(agent_id, agents)
                    if fallback_agent_id:
                        agent_id = fallback_agent_id
                        agent_info = self.registry.get_agent(agent_id)
                        agent_instance = agent_info["instance"]
                        continue

                return {
                    "mode": "simple",
                    "agent_id": agent_id,
                    "agent_name": agent_info["name"],
                    "result": validated_result,
                    "success": validated_result.get("success", True),
                    "attempts": attempt + 1
                }

            except Exception as e:
                logger.error(f"Agent执行失败 (尝试 {attempt + 1}/{max_retries + 1}): {e}")

                if attempt < max_retries:
                    # 尝试fallback
                    fallback_agent_id = self._get_fallback_agent(agent_id, agents)
                    if fallback_agent_id:
                        logger.info(f"切换到fallback Agent: {fallback_agent_id}")
                        agent_id = fallback_agent_id
                        agent_info = self.registry.get_agent(agent_id)
                        agent_instance = agent_info["instance"]
                        continue

                # 最后一次尝试失败
                error_response = self.error_handler.handle_agent_error(e, agent_id, query)
                return {
                    "mode": "simple",
                    "agent_id": agent_id,
                    "error": error_response,
                    "success": False,
                    "attempts": attempt + 1
                }

    def _get_fallback_agent(self, failed_agent_id: str, agents: list) -> Optional[str]:
        """
        获取fallback Agent

        Args:
            failed_agent_id: 失败的Agent ID
            agents: 可用Agent列表

        Returns:
            fallback Agent ID或None
        """
        # 简单策略：返回knowledge_agent作为通用fallback
        for agent in agents:
            if agent['id'] != failed_agent_id and agent['id'] == 'knowledge_agent':
                return agent['id']

        # 如果没有knowledge_agent，返回第一个不同的Agent
        for agent in agents:
            if agent['id'] != failed_agent_id:
                return agent['id']

        return None

    def list_agents(self) -> str:
        """列出所有Agent"""
        return self.registry.list_agents()

    def get_agent_info(self, agent_id: str) -> Optional[Dict]:
        """获取Agent信息"""
        return self.registry.get_agent(agent_id)
