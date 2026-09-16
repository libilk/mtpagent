# -*- coding: utf-8 -*-
"""
任务规划器
==========

使用LLM生成任务DAG
"""

import json
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from llm.langchain_parser import parse_llm_output

logger = logging.getLogger(__name__)


class TaskPlan(BaseModel):
    """任务计划模型"""
    reasoning: str = Field(description="规划思路")
    tasks: List[Dict[str, Any]] = Field(description="任务列表")


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

请先在 reasoning 字段中分析规划思路，再生成 tasks 列表。规则:
1. 如果问题简单，只需一个Agent即可完成，则生成单个任务
2. 如果问题复杂，需要多个步骤，则拆解成多个任务
3. 每个任务必须对应一个Agent的能力
4. 任务之间可以有依赖关系（depends_on字段）
5. 互相不依赖的任务不设置依赖关系，它们会被并行执行
6. 有依赖关系的任务必须声明 input_schema（需要哪些字段）和 parameter_mapping（字段从哪个上游任务获取）
7. 所有任务都应声明 output_schema（承诺输出哪些字段）
8. 重要路由规则（这是电商售后场景）：
   - 查询订单状态、物流轨迹、退款记录、会员数据、交易统计 → 用 database_agent（查数据）
   - 售后政策咨询、退换货办理、投诉受理、工单 → 用 customer_service_agent（问规则或办事）
   - 与售后无关的通用知识、概念解释 → 用 knowledge_agent
   - 图片分析（破损商品照、快递单、发票截图）→ 用 vqa_agent

示例一（有依赖的任务链）：用户问题="订单 SO20260909001 的耳机坏了，帮我办退货"
{{
  "reasoning": "退货要两步且有先后依赖：必须先查出订单的真实状态和签收时间（database_agent），才能判断是否在售后时限内、并据此办理（customer_service_agent）。后者的输入来自前者的输出，所以是串行而非并行。",
  "tasks": [
    {{
      "task_id": "task_1",
      "description": "查询订单 SO20260909001 的详情：订单状态、下单时间、商品名称与数量、物流签收时间",
      "agent_id": "database_agent",
      "depends_on": [],
      "input_schema": {{
        "required_fields": [],
        "field_types": {{}}
      }},
      "output_schema": {{
        "required_fields": ["order_detail"],
        "field_types": {{"order_detail": "str"}}
      }}
    }},
    {{
      "task_id": "task_2",
      "description": "依据订单详情办理退货，并引用售后政策条款说明运费由谁承担",
      "agent_id": "customer_service_agent",
      "depends_on": ["task_1"],
      "input_schema": {{
        "required_fields": ["order_detail"],
        "field_types": {{"order_detail": "str"}}
      }},
      "output_schema": {{
        "required_fields": ["handling_result"],
        "field_types": {{"handling_result": "str"}}
      }},
      "parameter_mapping": {{
        "order_detail": "task_1.order_detail"
      }}
    }}
  ]
}}

示例二（并行+汇总）：用户问题="帮我看看还有哪些订单没发货，另外超时未发货平台怎么赔"
{{
  "reasoning": "这是两件互不相干的事：一件是查数据（统计未发货订单），一件是问规则（超时赔付标准）。前者要读数据库，后者要查政策文档，彼此没有依赖，可以并行执行。",
  "tasks": [
    {{
      "task_id": "task_1",
      "description": "统计当前所有状态为「待发货」的订单，列出订单号和下单时间",
      "agent_id": "database_agent",
      "depends_on": [],
      "input_schema": {{
        "required_fields": [],
        "field_types": {{}}
      }},
      "output_schema": {{
        "required_fields": ["pending_orders"],
        "field_types": {{"pending_orders": "str"}}
      }}
    }},
    {{
      "task_id": "task_2",
      "description": "检索超时未发货的判定标准和赔付规则",
      "agent_id": "customer_service_agent",
      "depends_on": [],
      "input_schema": {{
        "required_fields": [],
        "field_types": {{}}
      }},
      "output_schema": {{
        "required_fields": ["late_shipment_policy"],
        "field_types": {{"late_shipment_policy": "str"}}
      }}
    }}
  ]
}}

请直接输出JSON，不要有其他内容。"""

        return prompt

    def _parse_plan(self, response: str) -> Dict:
        """解析LLM响应（使用LangChain）"""
        plan = parse_llm_output(response, TaskPlan, fallback=None)
        if plan is None:
            logger.error("JSON解析失败")
            logger.debug(f"响应内容: {response}")
            raise ValueError("无法解析任务计划")
        return plan.dict()

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

            # 补全 Schema 字段（LLM 可能漏掉）
            if "input_schema" not in task:
                task["input_schema"] = {"required_fields": [], "field_types": {}}
            if "output_schema" not in task:
                task["output_schema"] = {"required_fields": [], "field_types": {}}
            if "parameter_mapping" not in task and depends_on:
                task["parameter_mapping"] = {}

            # 验证 parameter_mapping 引用的上游任务存在
            for field, source_path in task.get("parameter_mapping", {}).items():
                if "." in source_path:
                    source_task_id = source_path.split(".", 1)[0]
                    if source_task_id not in task_ids:
                        logger.warning(
                            f"任务 {task['task_id']} 的 parameter_mapping "
                            f"引用了不存在的任务: {source_task_id}，已移除"
                        )
                        task["parameter_mapping"].pop(field)

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
        """创建简单的单任务计划（回退方案）

        注意这里的关键词必须能对应上**真实注册的** agent_id，
        否则路由会指向一个不存在的 Agent，静默失败。
        （原代码用的是 code_agent / customer_agent，两个都已经不存在了。）
        """
        # 默认落到售后客服 —— 它是能力最全的一个（既有检索又有办理工具），
        # 在电商售后场景下做兜底最不容易答错。
        agent_id = "customer_service_agent"

        # 按关键词再细化一下
        query_lower = query.lower()
        if any(kw in query_lower for kw in ["订单", "物流", "快递", "退款记录", "统计", "多少"]):
            agent_id = "database_agent"
        elif any(kw in query_lower for kw in ["图片", "照片", "截图", "图里"]):
            agent_id = "vqa_agent"

        return {
            "tasks": [
                {
                    "task_id": "task_1",
                    "description": query,
                    "agent_id": agent_id,
                    "depends_on": [],
                    "input_schema": {"required_fields": [], "field_types": {}},
                    "output_schema": {"required_fields": [], "field_types": {}},
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
