

# -*- coding: utf-8 -*-
"""
增强版 LangGraph 图构建
=======================

LangGraph = 把多步骤流程画成一张"图"的编排框架：节点(node) 是干活的函数，
边(edge) 决定这一步做完之后该走谁。本文件只负责"连图"，
每个节点内部具体做什么，在 nodes.py / enhanced_nodes.py / clarification.py 里。

集成所有高级功能：
- 参数验证与重试（仅多 Agent DAG 场景，基于 Schema 校验）
  · DAG（有向无环图）= 多个 Agent 按依赖顺序串成的流程，"无环"指不能绕回自己
  · Schema（结构契约）= 一份字段清单，声明"这个节点的输出必须包含哪些字段"
- 参数对齐（把上游产出的字段整理、补齐成下游直接能用的入参，见 orchestrator/parameter_aligner.py）
- Critic 验证（critic = 评审：检查某个 Agent 的输出能不能接上后续任务）
- 重复检测
- 人工介入（human intervention：流程暂停，等真人确认后再继续）

职责划分：
- 参数验证：多 Agent DAG 中，下游检测上游输出是否缺少必需字段（Schema 校验，不调 LLM）
- Evaluator（评估器）：所有场景的最终答案质量把关（相关性、完整性、准确性）
- 单 Agent 场景不走参数验证（parameter_validator 检测到无 plan 会直接短路跳过），
  质量问题交给 Evaluator
"""

import logging
from typing import Dict, Any

from langgraph.graph import StateGraph, START, END  # START/END = 图的虚拟起点、终点
from langgraph.checkpoint.memory import MemorySaver  # 内置存档器：把图状态存进内存，支持暂停后继续

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
    duplicate_detection 之后的条件路由（conditional edge）

    "条件边"= 下一步走谁不写死，而是调用一个函数、看它返回哪个字符串，再去映射表里查。
    本函数就是被调用的那个函数，返回的字符串必须和 add_conditional_edges 里给的键一致。

    Args:
        state: 图的全局状态字典。各节点靠"读它 / 把自己的返回值合进去"来传递数据，
               字段清单见 enhanced_state.py

    Returns:
        "duplicate" — 检测到重复，跳过后续直接进入 evaluator（评估器：最终质量把关）
        "continue"  — 未检测到重复，继续正常流程
    """
    if state.get("has_duplicate", False):  # has_duplicate：重复检测节点写下的标记位
        # aggregator（聚合器）= 把多个 Agent 的结果汇总成一份的节点
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
        agents_dict: Agent 字典，形如 {agent_id: {"instance": Agent 实例}}
        planner: TaskPlanner 实例 —— 复杂问题拆任务（planner = 规划器）
        router: AgentRouter 实例 —— 简单问题挑一个 Agent（router = 路由器）
        registry: AgentRegistry 实例 —— Agent 名册，按名字取实例（registry = 注册表）
        llm: LLM 实例
        shared_memory: 共享记忆实例（已废弃，保留向后兼容，优先使用 memory_store）
        memory_store: 按 thread_id 隔离的记忆管理器（MemoryStore 实例）
                      thread_id = 会话 ID，不同对话窗口/用户互不串记忆
        embedder: 向量化模型实例，只有重复检测节点用它算语义相似度；
                  传 None 时重复检测退化为"只看前 500 字符是否完全一致"
        enable_checkpointer: 是否启用状态持久化（checkpointer = 检查点，见文件末尾编译处）
        enable_parameter_validation: 是否启用参数验证与重试
        enable_critic: 是否启用 Critic 验证

    注意：下面的 enable_xxx 开关会改变"连图"的方式 —— 开关不同，边就不一样。
    读边的时候先确认是哪套开关组合，否则会看不懂为什么某条边时有时无。
    """

    nodes = GraphNodes(planner, router, registry, llm)  # 所有节点函数的集合（定义在 nodes.py）
    builder = StateGraph(EnhancedGraphState)  # StateGraph = 图构建器；EnhancedGraphState = 全图共享的状态结构

    # ========== 核心节点 ==========
    # add_node(节点名, 处理函数)：节点名 = 后面连边时引用的字符串；处理函数 = 真正干活的函数
    builder.add_node("complexity_classifier", nodes.complexity_classifier_node)  # 复杂度分类器：判断简单还是复杂
    builder.add_node("router", nodes.router_node)  # 路由器：简单问题 → 挑一个 Agent
    builder.add_node("planner", nodes.planner_node)  # 规划器：复杂问题 → 拆成带依赖的任务列表
    builder.add_node("aggregator", nodes.aggregator_node)  # 聚合器：把多个 Agent 的结果汇总成一份
    builder.add_node("evaluator", nodes.evaluator_node)  # 评估器：给最终答案打分，不过关就打回重试

    # ========== Agent 节点 ==========
    # 每个业务 Agent（售后 / 知识库 / 数据库 / 文档 …）在这里变成图上的一个节点，
    # 节点名就是 agent_id 本身。
    agent_ids = list(agents_dict.keys())
    for agent_id, agent_info in agents_dict.items():
        # make_agent_node 是"工厂函数"：给一份 Agent 实例，返回一个符合图要求的节点函数
        node_fn = make_agent_node(agent_info["instance"], agent_id, shared_memory=shared_memory, memory_store=memory_store)
        builder.add_node(agent_id, node_fn)

    # ========== 高级功能节点 ==========
    if enable_parameter_validation:
        # 参数验证器：检查上游 Agent 的输出够不够下游用（只查字段，不调 LLM，很便宜）
        builder.add_node("parameter_validator", parameter_validator_node)
        # upstream_retry（上游重试）：字段缺了，就把缺字段的那个上游 Agent 再跑一遍
        builder.add_node("upstream_retry", upstream_retry_node)

    if enable_critic:
        from langgraph_orchestrator.critic import critic_validation_node
        # 评审节点；要带上 registry，因为它得从名册里取 critic_agent 实例
        builder.add_node("critic", lambda s: critic_validation_node(s, registry))

    # 重复检测：结果和之前某次相同就没必要再重试了，直接熔断
    builder.add_node("duplicate_detection", make_duplicate_detection_node(embedder))
    # 人工介入拆成两步：check 只判断"要不要拦"，execute 处理"收到人工答复后怎么走"。
    # 拆开是因为中间要暂停等真人输入 —— 暂停点就卡在这两个节点之间。
    builder.add_node("human_intervention_check", human_intervention_check_node)
    builder.add_node("human_intervention_execute", human_intervention_execute_node)

    # ========== 边连接 ==========
    # add_edge(A, B)                       = 固定边：A 做完一定走 B
    # add_conditional_edges(A, 函数, 映射表) = 条件边：函数返回的字符串 → 查映射表 → 决定走谁
    builder.add_edge(START, "complexity_classifier")

    # complexity_classifier -> router/planner
    # 返回 "simple" 走 router（问题简单，不需要拆解），返回 "complex" 走 planner
    builder.add_conditional_edges(
        "complexity_classifier",
        route_by_complexity,
        {"simple": "router", "complex": "planner"}
    )

    # router -> Agent
    # 映射表写成 {agent_id: agent_id}，是因为 route_to_agent 直接返回选中的 agent_id 当节点名
    builder.add_conditional_edges(
        "router",
        route_to_agent,
        {aid: aid for aid in agent_ids}
    )

    # planner -> 并行执行
    # fan_out（扇出）：把计划里"没有依赖"的任务一次性全发出去，让多个 Agent 并行跑。
    # 注意它返回的是 Send 列表（Send = "把这份数据送给某个节点"），而不是单个字符串 ——
    # 这是 LangGraph 里"一个节点扇出成多个并行分支"的写法，也是它不写映射表的原因。
    fan_out_fn = make_fan_out_dag_tasks(agent_ids)
    builder.add_conditional_edges("planner", fan_out_fn)

    # Agent -> 参数验证 (可选) -> Critic (可选) -> Duplicate Detection -> Aggregator
    #
    # 参数验证职责划分：
    # - 简单路由（单 Agent）：parameter_validator 内部检测到无 plan，直接短路跳过
    #   质量问题由下游 Evaluator 外环处理
    # - 复杂路由（多 Agent DAG）：Schema 校验上游输出是否满足下游 required_fields
    #   缺少字段则触发 upstream_retry 重新执行上游 Agent

    # 每个 Agent 完成后 → 下一个处理节点（边从 agent 出发，所以这行在循环里加 N 次）
    for agent_id in agent_ids:
        if enable_parameter_validation:
            builder.add_edge(agent_id, "parameter_validator")
        elif enable_critic:
            builder.add_edge(agent_id, "critic")
        else:
            builder.add_edge(agent_id, "duplicate_detection")

    # parameter_validator 的条件分支（只添加一次，放在循环外）
    if enable_parameter_validation:
        # should_retry_validation 返回 "reexecute"（校验没过且重试配额没用完）或 "continue"
        builder.add_conditional_edges(
            "parameter_validator",
            should_retry_validation,
            {"reexecute": "upstream_retry", "continue": "duplicate_detection"}
        )

        # upstream_retry 之后要回到"该重跑的那个 Agent"，映射表同样是 {agent_id: agent_id}
        def _route_retry_target(state: Dict[str, Any]) -> str:
            # 注意：这里读的 validation_target 全项目没有任何地方写入过
            # （真正被写入的是 clarification.py 的 retry_target_task），
            # 所以目前恒定回退到 agent_ids[0]。如实注释，未改动逻辑。
            target = state.get("validation_target")
            if target and target in agent_ids:
                return target
            return agent_ids[0] if agent_ids else "duplicate_detection"

        builder.add_conditional_edges(
            "upstream_retry",
            _route_retry_target,
            {aid: aid for aid in agent_ids}
        )

    # ---- critic 和 duplicate_detection 谁先谁后，取决于开关组合 ----
    # 两个都开 : Agent → parameter_validator → duplicate_detection → critic → aggregator
    # 只开 critic: Agent → critic → duplicate_detection → aggregator
    # 都关      : Agent → duplicate_detection → aggregator
    # 规律：duplicate_detection 永远排在 critic 前面 —— 先用便宜的办法拦掉重复，
    #       再决定要不要花 LLM 去评审。
    if enable_critic and not enable_parameter_validation:
        # 参数验证关着，Agent 直接进 critic，所以这里接 critic 的下一站
        builder.add_edge("critic", "duplicate_detection")
    elif enable_parameter_validation and enable_critic:
        # 参数验证开着，critic 排在重复检测之后，所以 critic 的下一站是聚合
        builder.add_edge("critic", "aggregator")

    # ========== duplicate_detection 条件分支（核心修复） ==========
    # 重复 → 跳过 critic/aggregator 直接进 evaluator（熔断：结果都是重复的，再评审也没意义）
    # 未重复 → 继续正常流程，去哪由上面的开关组合决定
    duplicate_continue_target = "critic" if (enable_parameter_validation and enable_critic) else "aggregator"
    builder.add_conditional_edges(
        "duplicate_detection",
        _route_after_duplicate,
        {"duplicate": "evaluator", "continue": duplicate_continue_target}
    )

    # aggregator -> wave_scheduler -> (done: evaluator 或下一波 Agent)
    # wave scheduler（波次调度）：DAG 里的任务有依赖关系，不能一次性全并发。
    # 每轮只发"依赖已经全部完成"的那一批（一波），一批批往前推；
    # 找不到就绪任务时返回 "done"，流程才继续往下走。
    wave_scheduler_fn = make_wave_scheduler(agent_ids)
    builder.add_conditional_edges(
        "aggregator",
        wave_scheduler_fn,
        {"done": "evaluator", **{aid: aid for aid in agent_ids}}
    )

    # evaluator -> human_intervention_check -> (execute 或 END/retry)
    builder.add_edge("evaluator", "human_intervention_check")

    # check 节点根据是否需要人工介入，决定下一步走向
    # check_quality = 看 evaluator 打的分：分够（≥0.7）或重试次数用完 → "pass"，否则 → "retry"
    builder.add_conditional_edges(
        "human_intervention_check",
        # 要人工介入就返回 "need_human"，否则把决定权交给 check_quality
        lambda s: "need_human" if s.get("human_intervention_required") else check_quality(s),
        {
            "need_human": "human_intervention_execute",  # 需要介入 → 走 execute（会被 interrupt_before 拦住）
            "pass": END,                                  # 质量合格 → 走到 END，流程结束
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
    # 上面只是"画好图纸"，compile() 之后才是一张能真正跑的图。
    if enable_checkpointer:
        # checkpointer（检查点）：每走完一步就把 state 存一份快照。
        # "人工介入要暂停、暂停后还能接着跑"全靠它 —— 没有快照就无法恢复现场。
        # MemorySaver 把快照存在内存里，进程一重启就没了。
        checkpointer = MemorySaver()
        logger.info("[EnhancedGraph] 启用状态持久化 (MemorySaver)")
    else:
        # 没有 checkpointer 就无从暂停/恢复，下面的 interrupt_before 也只能是 None
        checkpointer = None

    compiled = builder.compile(
        checkpointer=checkpointer,
        # interrupt_before（执行前中断）：走到 human_intervention_execute 之前先停下，
        # 把控制权交回调用方；等人工给出 human_feedback，再从断点继续往下走。
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
