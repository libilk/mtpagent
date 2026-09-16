# -*- coding: utf-8 -*-
"""
增强版 LangGraph 图构建
=======================

集成所有高级功能：
- 参数验证与重试（仅多 Agent DAG 场景，基于 Schema 校验）
- 参数对齐
- Critic 验证
- 重复检测
- 人工介入

职责划分：
- 参数验证：多 Agent DAG 中，下游检测上游输出是否缺少必需字段（Schema 校验，不调 LLM）
- Evaluator：所有场景的最终答案质量把关（相关性、完整性、准确性）
- 单 Agent 场景不走参数验证（parameter_validator 内部短路），质量问题交给 Evaluator
"""

import logging
from typing import Dict, Any

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from langgraph_orchestrator.enhanced_state import EnhancedGraphState
from langgraph_orchestrator.nodes import GraphNodes, make_agent_node
from langgraph_orchestrator.router import route_by_complexity, route_to_agent, make_fan_out_dag_tasks, make_wave_scheduler, check_quality
from langgraph_orchestrator.clarification import (
    parameter_validator_node,
    upstream_retry_node,
    should_retry_validation
)
from langgraph_orchestrator.enhanced_nodes import (
    make_duplicate_detection_node,
    human_intervention_check_node,
    human_intervention_execute_node,
)

logger = logging.getLogger(__name__)


def _route_after_duplicate(state: Dict[str, Any]) -> str:
    """
    duplicate_detection 之后的条件路由

    Returns:
        "duplicate" — 检测到重复，跳过后续直接进入 evaluator
        "continue"  — 未检测到重复，继续正常流程
    """
    if state.get("has_duplicate", False):
        logger.info("[DuplicateRoute] 检测到重复结果，跳过 critic/aggregator，直接进入 evaluator")
        return "duplicate"
    return "continue"


def build_enhanced_graph(
    agents_dict: Dict[str, Dict[str, Any]],
    planner,
    router,
    registry,
    llm,
    shared_memory=None,
    memory_store=None,
    embedder=None,
    enable_checkpointer: bool = True,
    enable_parameter_validation: bool = True,
    enable_critic: bool = True,
):
    """
    构建增强版 LangGraph 图

    Args:
        agents_dict: Agent 字典
        planner: TaskPlanner 实例
        router: AgentRouter 实例
        registry: AgentRegistry 实例
        llm: LLM 实例
        shared_memory: 共享记忆实例（已废弃，保留向后兼容，优先使用 memory_store）
        memory_store: 按 thread_id 隔离的记忆管理器（MemoryStore 实例）
        enable_checkpointer: 是否启用状态持久化
        enable_parameter_validation: 是否启用参数验证与重试
        enable_critic: 是否启用 Critic 验证
    """

    nodes = GraphNodes(planner, router, registry, llm)
    builder = StateGraph(EnhancedGraphState)

    # ========== 核心节点 ==========
    builder.add_node("complexity_classifier", nodes.complexity_classifier_node)
    builder.add_node("router", nodes.router_node)
    builder.add_node("planner", nodes.planner_node)
    builder.add_node("aggregator", nodes.aggregator_node)
    builder.add_node("evaluator", nodes.evaluator_node)

    # ========== Agent 节点 ==========
    agent_ids = list(agents_dict.keys())
    for agent_id, agent_info in agents_dict.items():
        node_fn = make_agent_node(agent_info["instance"], agent_id, shared_memory=shared_memory, memory_store=memory_store)
        builder.add_node(agent_id, node_fn)

    # ========== 高级功能节点 ==========
    if enable_parameter_validation:
        builder.add_node("parameter_validator", parameter_validator_node)
        builder.add_node("upstream_retry", upstream_retry_node)

    if enable_critic:
        from langgraph_orchestrator.critic import critic_validation_node
        builder.add_node("critic", lambda s: critic_validation_node(s, registry))

    builder.add_node("duplicate_detection", make_duplicate_detection_node(embedder))
    builder.add_node("human_intervention_check", human_intervention_check_node)
    builder.add_node("human_intervention_execute", human_intervention_execute_node)

    # ========== 边连接 ==========
    builder.add_edge(START, "complexity_classifier")

    # complexity_classifier -> router/planner
    builder.add_conditional_edges(
        "complexity_classifier",
        route_by_complexity,
        {"simple": "router", "complex": "planner"}
    )

    # router -> Agent
    builder.add_conditional_edges(
        "router",
        route_to_agent,
        {aid: aid for aid in agent_ids}
    )

    # planner -> 并行执行
    fan_out_fn = make_fan_out_dag_tasks(agent_ids)
    builder.add_conditional_edges("planner", fan_out_fn)

    # Agent -> 参数验证 (可选) -> Critic (可选) -> Duplicate Detection -> Aggregator
    #
    # 参数验证职责划分：
    # - 简单路由（单 Agent）：parameter_validator 内部检测到无 plan，直接短路跳过
    #   质量问题由下游 Evaluator 外环处理
    # - 复杂路由（多 Agent DAG）：Schema 校验上游输出是否满足下游 required_fields
    #   缺少字段则触发 upstream_retry 重新执行上游 Agent

    # 每个 Agent 完成后 → 下一个处理节点（边从 agent 出发，可重复添加）
    for agent_id in agent_ids:
        if enable_parameter_validation:
            builder.add_edge(agent_id, "parameter_validator")
        elif enable_critic:
            builder.add_edge(agent_id, "critic")
        else:
            builder.add_edge(agent_id, "duplicate_detection")

    # parameter_validator 的条件分支（只添加一次，放在循环外）
    if enable_parameter_validation:
        builder.add_conditional_edges(
            "parameter_validator",
            should_retry_validation,
            {"reexecute": "upstream_retry", "continue": "duplicate_detection"}
        )

        # upstream_retry 根据 state 中的 validation_target 动态路由回对应 Agent
        def _route_retry_target(state: Dict[str, Any]) -> str:
            target = state.get("validation_target")
            if target and target in agent_ids:
                return target
            return agent_ids[0] if agent_ids else "duplicate_detection"

        builder.add_conditional_edges(
            "upstream_retry",
            _route_retry_target,
            {aid: aid for aid in agent_ids}
        )

    if enable_critic and not enable_parameter_validation:
        # Agent → critic → duplicate_detection
        builder.add_edge("critic", "duplicate_detection")
    elif enable_parameter_validation and enable_critic:
        # parameter_validator → duplicate_detection，通过后继续走 critic
        builder.add_edge("critic", "aggregator")

    # ========== duplicate_detection 条件分支（核心修复） ==========
    # 重复 → 跳过后续直接进入 evaluator（熔断）
    # 未重复 → 继续正常流程
    duplicate_continue_target = "critic" if (enable_parameter_validation and enable_critic) else "aggregator"
    builder.add_conditional_edges(
        "duplicate_detection",
        _route_after_duplicate,
        {"duplicate": "evaluator", "continue": duplicate_continue_target}
    )

    # aggregator -> wave_scheduler -> (done: evaluator 或下一波 Agent)
    wave_scheduler_fn = make_wave_scheduler(agent_ids)
    builder.add_conditional_edges(
        "aggregator",
        wave_scheduler_fn,
        {"done": "evaluator", **{aid: aid for aid in agent_ids}}
    )

    # evaluator -> human_intervention_check -> (execute 或 END/retry)
    builder.add_edge("evaluator", "human_intervention_check")

    # check 节点根据是否需要人工介入，决定下一步走向
    builder.add_conditional_edges(
        "human_intervention_check",
        lambda s: "need_human" if s.get("human_intervention_required") else check_quality(s),
        {
            "need_human": "human_intervention_execute",  # 需要介入 → 走 execute（会被 interrupt_before 拦住）
            "pass": END,                                  # 质量合格 → 直接结束
            "retry": "complexity_classifier",             # 质量不合格 → 重新分类后重试（修复：原来回 router 会跳过复杂度判断）
        }
    )

    # execute 节点执行完后（人工放行后），重新走质量检查
    builder.add_conditional_edges(
        "human_intervention_execute",
        check_quality,
        {"pass": END, "retry": "complexity_classifier"}
    )

    # ========== 编译 ==========
    if enable_checkpointer:
        checkpointer = MemorySaver()
        logger.info("[EnhancedGraph] 启用状态持久化 (MemorySaver)")
    else:
        checkpointer = None

    compiled = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_intervention_execute"] if checkpointer else None,
    )

    logger.info(
        "[EnhancedGraph] 图编译完成 | Agents: %d | ParameterValidation: %s | Critic: %s | Checkpointer: %s",
        len(agent_ids),
        enable_parameter_validation,
        enable_critic,
        enable_checkpointer
    )

    return compiled
