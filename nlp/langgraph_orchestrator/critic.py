# -*- coding: utf-8 -*-
"""
Critic Agent 验证节点
====================

验证任务衔接的合理性

critic（评审）最容易猜错：它**不是给输出打分**，而是校验"这段输出能不能接上
后续任务"（给答案打分的是 evaluator，两套阈值也各不相同）。

本文件是个薄壳：自己不实现任何校验逻辑，而是把上下文打包交给注册中心里的
critic_agent（一个 LLM Agent）去判，再把它的结论落成 state 字段。

三条容易忽略的行为：
- 只在多 Agent DAG 场景里起作用：单 Agent 路径不跑规划器，state 里 plan 是 None
  （不是空字典），本函数对 None 取 .get 会抛异常 —— 今天因 critic_agent 未注册而走不到这条。
- 凡是"没法判"的情况（取不到 Agent、没有结果、找不到任务、调用抛异常）一律
  判通过 —— fail-open：宁可放过，也不要因为评审环节把主流程卡死。
- 它只是"标记 + 展示"：判定结果写进 critic_passed / critic_issues，
  真正决定流程走向的是 human_intervention_check 里读这两个字段的那条通道。
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def critic_validation_node(state: Dict[str, Any], registry) -> dict:
    """验证任务衔接

    Args:
        state: 全局状态；读 agent_results（最新一条）和 plan（任务依赖关系）
        registry: 注册中心 —— Agent 名册，按名字取实例
    """

    # 获取 Critic Agent
    critic_agent_info = registry.get_agent("critic_agent")

    if not critic_agent_info:
        logger.debug("[Critic] Critic Agent 未注册，跳过验证")
        return {"critic_passed": True}

    critic_agent = critic_agent_info["instance"]
    agent_results = state.get("agent_results", [])

    if not agent_results:
        return {"critic_passed": True}

    latest_result = agent_results[-1]  # 只审刚跑完的这一条，不回头翻历史轮次
    plan = state.get("plan", {})
    # 默认值 {} 不生效：plan 这个键是存在的，只是值为 None（见 enhanced_entry 的初始 state），
    # 所以单 Agent 路径走到下一行就是 None.get(...)。今天碰不到，是因为 critic_agent 从未注册、
    # 上面已提前 return；一旦注册了 critic_agent 并开启 enable_critic，这里会抛 AttributeError。
    tasks = plan.get("tasks", [])

    # 找到当前任务
    current_task = next((t for t in tasks if t["agent_id"] == latest_result["agent"]), None)
    if not current_task:
        return {"critic_passed": True}

    # 找到后续任务：依赖当前任务的那些任务，也就是"这段输出要接上的对象"
    next_tasks = [t for t in tasks if current_task["task_id"] in t.get("depends_on", [])]

    # 构建上下文：交给 critic_agent 判断的全部依据
    # dependencies 取 agent_results[:-1]（除最新一条外的历史结果），按 Agent 名建索引，
    # 同名 Agent 只保留最后一条
    critic_context = {
        "task": current_task,
        "result": latest_result,
        "dependencies": {r["agent"]: r for r in agent_results[:-1]},
        "next_tasks": next_tasks
    }

    # 调用 Critic
    try:
        # handle（处理入口）= Agent 对外统一的方法签名；这里传"要它做什么"和判断依据
        validation = critic_agent.handle("验证任务衔接", critic_context)
        # 三个默认值一律取"通过"那一侧：Agent 少返回字段时不至于凭空判失败
        is_valid = validation.get("is_valid", True)
        quality_score = validation.get("quality_score", 1.0)
        issues = validation.get("issues", [])

        logger.info(f"[Critic] 验证: {'通过' if is_valid else '失败'}, 质量: {quality_score:.2f}")

        if not is_valid:
            logger.warning(f"[Critic] 问题: {issues}")

        # 这里的局部变量 quality_score 是"任务衔接质量"，与 state 里 evaluator 写的
        # 同名字段（最终答案质量）不是一回事；阈值 0.6 也和 evaluator 的 0.7 各自独立
        return {
            "critic_passed": is_valid and quality_score >= 0.6,
            "critic_validation": validation,
            "critic_issues": issues
        }

    except Exception as e:
        logger.error(f"[Critic] 验证失败: {e}")
        return {"critic_passed": True}  # 降级（fail-open）：评审环节出错就放行，不让它把流程卡死
