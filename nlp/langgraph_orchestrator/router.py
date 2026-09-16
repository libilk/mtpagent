# -*- coding: utf-8 -*-
"""
LangGraph 路由逻辑
==================

条件边函数，用于控制图的执行流向。
所有函数接收 GraphState，返回字符串键（映射到节点名）或 Send 列表。
"""

from __future__ import annotations

import logging
from typing import List, Union

from langgraph.types import Send

from langgraph_orchestrator.enhanced_state import EnhancedGraphState as GraphState

logger = logging.getLogger(__name__)


def route_by_complexity(state: GraphState) -> str:
    """
    根据 complexity_classifier 节点的判断结果路由。

    Returns:
        ``"simple"`` — 简单查询，走 router 节点
        ``"complex"`` — 复杂查询，走 planner 节点
    """
    if state.get("is_complex", False):
        logger.info("[LangGraph] 复杂查询 -> Planner")
        return "complex"
    logger.info("[LangGraph] 简单查询 -> Router")
    return "simple"


def route_to_agent(state: GraphState) -> str:
    """
    根据 router 节点选择的 Agent 路由到对应的 Agent 节点。

    Returns:
        agent_id 字符串，对应已注册的 Agent 节点名
    """
    agent_id = state.get("selected_agent")
    if agent_id:
        logger.info("[LangGraph] 路由到 Agent: %s", agent_id)
        return agent_id
    # 回退到 knowledge_agent
    logger.warning("[LangGraph] 未选择 Agent，默认 knowledge_agent")
    return "knowledge_agent"


def make_fan_out_dag_tasks(agent_ids: List[str]):
    """
    创建 DAG 并行分发函数（闭包，捕获已注册的 agent_ids）。
    只发送 root tasks（无依赖的任务）。

    Args:
        agent_ids: 所有已注册的 Agent ID 列表

    Returns:
        fan_out_dag_tasks 条件边函数
    """

    def fan_out_dag_tasks(state: GraphState) -> List[Send]:
        """将 DAG 计划中的 root 任务并行分发到对应 Agent 节点。"""
        plan = state.get("plan") or {}
        tasks = plan.get("tasks", [])

        if not tasks:
            logger.warning("[LangGraph] DAG 计划为空，回退到 knowledge_agent")
            return [Send("knowledge_agent", {**state})]

        # 只发送 root tasks（depends_on 为空）
        sends: List[Send] = []
        for task in tasks:
            depends_on = task.get("depends_on", [])
            if depends_on:
                continue  # 跳过有依赖的任务

            agent_id = task.get("agent_id", "knowledge_agent")
            task_id = task.get("task_id", agent_id)

            if agent_id not in agent_ids:
                logger.warning(
                    "[LangGraph] Agent %s 未注册，跳过任务: %s",
                    agent_id,
                    task.get("description", ""),
                )
                continue

            sends.append(
                Send(
                    agent_id,
                    {**state, "query": task["description"], "current_task_id": task_id},
                )
            )

        if not sends:
            logger.warning("[LangGraph] 无有效任务，回退到 knowledge_agent")
            return [Send("knowledge_agent", {**state})]

        logger.info("[LangGraph] 第一波分发 %d 个 root 任务", len(sends))
        return sends

    return fan_out_dag_tasks


def make_wave_scheduler(agent_ids: List[str]):
    """
    创建波次调度器（闭包，捕获已注册的 agent_ids）。

    Args:
        agent_ids: 所有已注册的 Agent ID 列表

    Returns:
        wave_scheduler 条件边函数
    """

    def wave_scheduler(state: GraphState) -> Union[str, List[Send]]:
        """
        检查是否有就绪任务（依赖全部完成且自身未完成），有则发送下一波，无则返回 "done"。
        """
        plan = state.get("plan")
        if not plan:
            return "done"

        tasks = plan.get("tasks", [])
        completed = set(state.get("completed_task_ids", []))

        ready_tasks = []
        for task in tasks:
            task_id = task.get("task_id", task.get("agent_id", ""))
            if task_id in completed:
                continue

            depends_on = task.get("depends_on", [])
            if all(dep in completed for dep in depends_on):
                ready_tasks.append(task)

        if not ready_tasks:
            logger.info("[LangGraph] 无更多就绪任务，结束")
            return "done"

        sends: List[Send] = []
        for task in ready_tasks:
            agent_id = task.get("agent_id", "knowledge_agent")
            task_id = task.get("task_id", agent_id)

            if agent_id not in agent_ids:
                logger.warning(
                    "[LangGraph] Agent %s 未注册，跳过任务: %s",
                    agent_id,
                    task.get("description", ""),
                )
                continue

            sends.append(
                Send(
                    agent_id,
                    {**state, "query": task["description"], "current_task_id": task_id},
                )
            )

        if not sends:
            return "done"

        logger.info("[LangGraph] 下一波分发 %d 个就绪任务", len(sends))
        return sends

    return wave_scheduler


def check_quality(state: GraphState) -> str:
    """
    检查 evaluator 的质量评分，决定结束或重试。

    Returns:
        ``"pass"`` — 质量合格或达到最大迭代次数，流向 END
        ``"retry"`` — 质量不足且仍有重试配额，流向 router 重试
    """
    score = state.get("quality_score", 0.0)
    iteration = state.get("iteration", 0)

    if score >= 0.7:
        logger.info("[LangGraph] 质量通过 (%.2f)，结束", score)
        return "pass"

    if iteration >= 3:
        logger.info(
            "[LangGraph] 达到最大迭代次数 (%d)，结束", iteration
        )
        return "pass"

    logger.info(
        "[LangGraph] 质量不足 (%.2f)，重试 (迭代 %d)", score, iteration
    )
    return "retry"
