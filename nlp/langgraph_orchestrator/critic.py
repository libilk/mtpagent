# -*- coding: utf-8 -*-
"""
Critic Agent 验证节点
====================

验证任务衔接的合理性
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def critic_validation_node(state: Dict[str, Any], registry) -> dict:
    """验证任务衔接"""

    # 获取 Critic Agent
    critic_agent_info = registry.get_agent("critic_agent")

    if not critic_agent_info:
        logger.debug("[Critic] Critic Agent 未注册，跳过验证")
        return {"critic_passed": True}

    critic_agent = critic_agent_info["instance"]
    agent_results = state.get("agent_results", [])

    if not agent_results:
        return {"critic_passed": True}

    latest_result = agent_results[-1]
    plan = state.get("plan", {})
    tasks = plan.get("tasks", [])

    # 找到当前任务
    current_task = next((t for t in tasks if t["agent_id"] == latest_result["agent"]), None)
    if not current_task:
        return {"critic_passed": True}

    # 找到后续任务
    next_tasks = [t for t in tasks if current_task["task_id"] in t.get("depends_on", [])]

    # 构建上下文
    critic_context = {
        "task": current_task,
        "result": latest_result,
        "dependencies": {r["agent"]: r for r in agent_results[:-1]},
        "next_tasks": next_tasks
    }

    # 调用 Critic
    try:
        validation = critic_agent.handle("验证任务衔接", critic_context)
        is_valid = validation.get("is_valid", True)
        quality_score = validation.get("quality_score", 1.0)
        issues = validation.get("issues", [])

        logger.info(f"[Critic] 验证: {'通过' if is_valid else '失败'}, 质量: {quality_score:.2f}")

        if not is_valid:
            logger.warning(f"[Critic] 问题: {issues}")

        return {
            "critic_passed": is_valid and quality_score >= 0.6,
            "critic_validation": validation,
            "critic_issues": issues
        }

    except Exception as e:
        logger.error(f"[Critic] 验证失败: {e}")
        return {"critic_passed": True}  # 降级：验证失败时放行
