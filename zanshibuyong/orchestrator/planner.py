# -*- coding: utf-8 -*-
"""
任务规划器
==========

使用LLM生成任务DAG
"""

import json
import logging
from typing import List, Dict, Any, Optional
from llm.output_parser import parse_llm_json

logger = logging.getLogger(__name__)


class TaskPlanner:
    """任务规划器"""

    def __init__(self, llm):
        """
        初始化

        Args:
            llm: LLM实例（Qwen-Max）
        """
        self.llm = llm

    def plan(self, query: str, available_agents: List[Dict]) -> Dict:
        """
        生成任务DAG

        Args:
            query: 用户查询
            available_agents: 可用的Agent列表

        Returns:
            任务DAG
        """
        # 构建提示词
        prompt = self._build_prompt(query, available_agents)

        try:
            # 调用LLM生成计划
            response = self.llm.generate(
                prompt,
                temperature=0.3,  # 降低温度以获得更确定的输出
                max_tokens=2048
            )

            # 解析JSON
            plan = self._parse_plan(response)

            # 验证计划
            self._validate_plan(plan, available_agents)

            logger.info(f"生成任务计划: {len(plan.get('tasks', []))} 个任务")
            return plan

        except Exception as e:
            logger.error(f"任务规划失败: {e}")
            # 返回简单的单任务计划
            return self._create_simple_plan(query, available_agents)

    def _build_prompt(self, query: str, available_agents: List[Dict]) -> str:
        """构建规划提示词"""
        agents_desc = "\n".join([
            f"- {agent['id']}: {agent['name']}\n"
            f"  描述: {agent['description']}\n"
            f"  能力: {', '.join(agent['capabilities'])}"
            for agent in available_agents
        ])

        prompt = f"""你是一个任务规划专家。请将用户问题拆解成任务DAG（有向无环图）。

可用的Agent:
{agents_desc}

用户问题: {query}

请分析问题并生成任务计划。规则:
1. 如果问题简单，只需一个Agent即可完成，则生成单个任务
2. 如果问题复杂，需要多个步骤，则拆解成多个任务
3. 每个任务必须对应一个Agent的能力
4. 任务之间可以有依赖关系（depends_on字段）
5. 并行任务不设置依赖关系

输出JSON格式:
{{
  "tasks": [
    {{
      "task_id": "task_1",
      "description": "任务描述",
      "agent_id": "knowledge_agent",
      "depends_on": []
    }}
  ],
  "reasoning": "规划思路说明"
}}

请直接输出JSON，不要有其他内容。"""

        return prompt

    def _parse_plan(self, response: str) -> Dict:
        """解析LLM响应"""
        plan = parse_llm_json(response, fallback=None)
        if plan is None:
            logger.error("JSON解析失败")
            logger.debug(f"响应内容: {response}")
            raise ValueError("无法解析任务计划")
        return plan

    def _validate_plan(self, plan: Dict, available_agents: List[Dict]):
        """验证任务计划"""
        if "tasks" not in plan:
            raise ValueError("任务计划缺少tasks字段")

        tasks = plan["tasks"]
        if not tasks:
            raise ValueError("任务列表为空")

        agent_ids = {agent["id"] for agent in available_agents}
        task_ids = {task["task_id"] for task in tasks}

        for task in tasks:
            # 检查必需字段
            required_fields = ["task_id", "description", "agent_id"]
            for field in required_fields:
                if field not in task:
                    raise ValueError(f"任务缺少字段: {field}")

            # 检查Agent是否存在
            if task["agent_id"] not in agent_ids:
                raise ValueError(f"未知的Agent: {task['agent_id']}")

            # 检查依赖是否存在
            depends_on = task.get("depends_on", [])
            for dep in depends_on:
                if dep not in task_ids:
                    raise ValueError(f"未知的依赖任务: {dep}")

        # 检查是否有循环依赖
        self._check_circular_dependency(tasks)

    def _check_circular_dependency(self, tasks: List[Dict]):
        """检查循环依赖"""
        # 构建依赖图
        graph = {task["task_id"]: task.get("depends_on", []) for task in tasks}

        # DFS检测环
        visited = set()
        rec_stack = set()

        def has_cycle(node):
            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for task_id in graph:
            if task_id not in visited:
                if has_cycle(task_id):
                    raise ValueError("任务计划存在循环依赖")

    def _create_simple_plan(self, query: str, available_agents: List[Dict]) -> Dict:
        """创建简单的单任务计划（回退方案）"""
        # 默认使用knowledge_agent
        agent_id = "knowledge_agent"

        # 尝试根据关键词选择Agent
        query_lower = query.lower()
        if any(kw in query_lower for kw in ["代码", "编程", "code", "python", "java"]):
            agent_id = "code_agent"
        elif any(kw in query_lower for kw in ["客服", "工单", "订单", "投诉"]):
            agent_id = "customer_agent"

        return {
            "tasks": [
                {
                    "task_id": "task_1",
                    "description": query,
                    "agent_id": agent_id,
                    "depends_on": []
                }
            ],
            "reasoning": "简单查询，使用单个Agent处理"
        }

    def should_plan(self, query: str) -> bool:
        """
        判断是否需要任务规划（使用LLM智能判断）

        Args:
            query: 用户查询

        Returns:
            是否需要规划
        """
        prompt = f"""你是一个任务复杂度分析专家。请判断以下用户问题是否需要多步骤规划。

用户问题: {query}

判断标准:
- 简单问题：单一明确的问题，可以直接回答（如"什么是RAG"、"介绍下向量数据库"）
- 复杂问题：需要多个步骤、对比分析、综合多方面信息（如"对比A和B"、"分析优缺点"、"先做X再做Y"）

请只回答"简单"或"复杂"，不要有其他内容。

判断结果:"""

        try:
            logger.info(f"[任务规划判断] 使用LLM分析问题复杂度: '{query}'")
            response = self.llm.generate(prompt, temperature=0.1, max_tokens=10)
            result = response.strip().lower()

            # 判断是否需要规划
            need_planning = "复杂" in result or "complex" in result

            logger.info(f"[任务规划判断] LLM判断结果: {result} -> {'需要规划' if need_planning else '简单路由'}")
            return need_planning

        except Exception as e:
            logger.error(f"LLM判断失败: {e}，使用启发式规则")
            # 降级到启发式规则
            return self._should_plan_heuristic(query)

    def _should_plan_heuristic(self, query: str) -> bool:
        """
        启发式规则判断（作为LLM判断的降级方案）

        Args:
            query: 用户查询

        Returns:
            是否需要规划
        """
        # 简单启发式规则
        complex_keywords = [
            "对比", "比较", "区别", "优缺点", "分析",
            "先", "然后", "接着", "最后",
            "和", "以及", "还有"
        ]

        query_lower = query.lower()
        for keyword in complex_keywords:
            if keyword in query_lower:
                logger.info(f"[任务规划判断] 启发式规则: 检测到关键词 '{keyword}' -> 需要规划")
                return True

        # 查询长度超过50字符，可能较复杂
        if len(query) > 50:
            logger.info(f"[任务规划判断] 启发式规则: 查询长度 {len(query)} > 50 -> 需要规划")
            return True

        logger.info(f"[任务规划判断] 启发式规则: 简单查询 -> 简单路由")
        return False
