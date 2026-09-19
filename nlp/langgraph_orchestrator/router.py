# -*- coding: utf-8 -*-
"""
LangGraph 路由逻辑
==================

条件边（conditional edge）函数，用于控制图的执行流向。
所有函数接收 GraphState，返回字符串键（映射到节点名）或 Send 列表。

返回值怎么变成"下一个节点"——两种返回类型，机制完全不同：

1. 返回字符串（route_by_complexity / route_to_agent / check_quality）
   函数返回一个 key，LangGraph 拿它去 add_conditional_edges 时给的映射表里查，
   查到的那一项才是真正的下一站。所以返回的字符串本身不是节点名，而是"映射表的键"；
   映射表里没有这个键就会报错。
   （route_to_agent 的映射表写成 {agent_id: agent_id}，键恰好等于节点名，
   容易让人误以为函数直接返回了节点名。）

2. 返回 List[Send]（fan_out_dag_tasks / wave_scheduler 的部分分支）
   Send（分发指令）= "把这份数据送给某个节点"，是对象不是字符串，不查映射表。
   列表里放 N 个 Send，就等于让当前节点扇出成 N 个并行分支。
   注意每个 Send 都带一份 state 副本（本文件用 {**state} 浅拷贝）：分支各自独立跑，
   结果最后由挂在 state 字段上的 reducer 合并回主流程 —— 这也是节点必须按 reducer
   的预期返回列表的原因。

   同一个函数可以两种都返回（见 wave_scheduler 的 Union[str, List[Send]]）：
   有活干就发 Send，没活干就返回 "done" 去查映射表。
"""

from __future__ import annotations

import logging
from typing import List, Union

from langgraph.types import Send  # 扇出并行分支用；它是什么见上面模块 docstring 第 2 点

from langgraph_orchestrator.enhanced_state import EnhancedGraphState as GraphState

logger = logging.getLogger(__name__)


def route_by_complexity(state: GraphState) -> str:
    """
    条件边函数：按 complexity_classifier 节点的判断结果分流。

    只读 is_complex（由复杂度分类器写入），不在这里重新判断 —— 判断与分流分离。

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

    返回值取自 router 写下的 selected_agent；因为映射表是 {agent_id: agent_id}
    （见 enhanced_graph.py），它同时充当"映射表的键"和"节点名"。

    Returns:
        agent_id 字符串，对应已注册的 Agent 节点名
    """
    agent_id = state.get("selected_agent")
    if agent_id:
        logger.info("[LangGraph] 路由到 Agent: %s", agent_id)
        return agent_id
    # 回退到 knowledge_agent；它必须出现在 agent_ids 里，否则映射表查不到这个键
    logger.warning("[LangGraph] 未选择 Agent，默认 knowledge_agent")
    return "knowledge_agent"


def make_fan_out_dag_tasks(agent_ids: List[str]):
    """
    创建 DAG 并行分发函数（闭包，捕获已注册的 agent_ids）。
    只发送 root tasks（无依赖的任务）。

    也是工厂函数：节点函数只能收 state，agent_ids 这种配置得靠闭包提前捕获。
    它只管"第一波"—— 把没有依赖的任务一次全发出去并行跑；带依赖的任务不归它管，
    留给 aggregator 之后的 wave_scheduler 一波波推。

    Args:
        agent_ids: 所有已注册的 Agent ID 列表

    Returns:
        fan_out_dag_tasks 条件边函数 —— 返回 List[Send]，不查映射表
    """

    def fan_out_dag_tasks(state: GraphState) -> List[Send]:
        """将 DAG 计划中的 root 任务并行分发到对应 Agent 节点。"""
        plan = state.get("plan") or {}
        tasks = plan.get("tasks", [])

        if not tasks:
            logger.warning("[LangGraph] DAG 计划为空，回退到 knowledge_agent")
            return [Send("knowledge_agent", {**state})]

        # 只发送 root tasks（depends_on 为空）
        # depends_on = 该任务依赖的 task_id 列表，也就是 DAG 的边；非空表示要等上游先跑完
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

            # 每个 Send 拿一份 state 副本，并把 query 换成该任务的描述、记下 current_task_id：
            # Agent 节点完成后靠 current_task_id 才知道"完成的是哪个 task"（波次调度据此算依赖）
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
    创建波次调度器（wave scheduler：有依赖的 DAG 不能全并发，每轮只发"依赖已全部完成"的那批）。

    它在 aggregator 之后被调用 —— 同一个聚合节点既是收尾点也是中转站：
    还有任务没跑完就发下一波（Send），全跑完了才返回 "done" 走向 evaluator。
    这也解释了 aggregator 里那句"任务没跑完就不汇总、直接透传 {}"。

    Args:
        agent_ids: 所有已注册的 Agent ID 列表

    Returns:
        wave_scheduler 条件边函数 —— 返回 Union[str, List[Send]]，两种返回类型都要处理
    """

    def wave_scheduler(state: GraphState) -> Union[str, List[Send]]:
        """
        检查是否有就绪任务（依赖全部完成且自身未完成），有则发送下一波，无则返回 "done"。

        "有没有活"分两层检查，顺序不能反：先挑出依赖都满足的 ready 任务，
        再逐个确认 agent 已注册。若全被跳过必须显式返回 "done" ——
        返回空列表等于一个分支都不发，映射表接不上，流程就到不了 evaluator。
        """
        plan = state.get("plan")
        if not plan:
            return "done"

        tasks = plan.get("tasks", [])
        completed = set(state.get("completed_task_ids", []))

        # completed_task_ids 是个累加字段，记着所有已完成任务的 id；下面用它做两件事：
        # 排除已完成的，判断依赖是否全部满足
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

            # 与 fan_out 同样：查不到就跳过；若一个都没剩，下面会返回 "done" 而不是空列表
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
    检查 evaluator（评估器）的质量评分，决定结束或重试。

    两道闸：quality_score >= 0.7 直接放行；否则看 iteration（迭代轮次），
    到 3 轮就返回 "pass" 强行收尾，防止"评估不过 → 重试"来回转成死循环。

    注意：下面 Returns 里"流向 router"是旧描述 —— 实际连图时 "retry" 映射到
    complexity_classifier（见 enhanced_graph.py，重试前要重新判复杂度）。此处只标注，未改逻辑。

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
