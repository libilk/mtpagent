# -*- coding: utf-8 -*-
"""
Agent注册中心
=============

管理所有Agent的注册和发现。
注册时强制校验 Agent 是否实现 AgentProtocol 接口。
"""

import logging
from typing import Dict, List, Optional, Any

from core.protocol import AgentProtocol

logger = logging.getLogger(__name__)


class AgentRegistry:
    """Agent注册中心"""

    def __init__(self):
        self.agents = {}  # {agent_id: agent_info}

    def register(
        self,
        agent_id: str,
        name: str,
        description: str,
        capabilities: List[str],
        agent_instance: Any,
        model: str = "qwen-plus",
        enabled: bool = True
    ):
        """
        注册Agent

        Args:
            agent_id: Agent ID
            name: Agent名称
            description: Agent描述
            capabilities: 能力列表
            agent_instance: Agent实例（必须实现 AgentProtocol）
            model: 使用的模型
            enabled: 是否启用

        Raises:
            TypeError: Agent实例未实现 AgentProtocol（缺少 handle 方法）
        """
        # ========== Harness 契约校验 ==========
        if not isinstance(agent_instance, AgentProtocol):
            raise TypeError(
                f"Agent '{agent_id}' ({type(agent_instance).__name__}) 未实现 AgentProtocol。"
                f"请确保该类定义了 handle(self, query: str, context: Dict[str, Any]) -> str 方法。"
            )
        self.agents[agent_id] = {
            "id": agent_id,
            "name": name,
            "description": description,
            "capabilities": capabilities,
            "instance": agent_instance,
            "model": model,
            "enabled": enabled,
            "status": "active"
        }
        logger.info(f"注册Agent: {agent_id} - {name}")

    def unregister(self, agent_id: str):
        """
        注销Agent

        Args:
            agent_id: Agent ID
        """
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info(f"注销Agent: {agent_id}")

    def get_agent(self, agent_id: str) -> Optional[Dict]:
        """
        获取Agent信息

        Args:
            agent_id: Agent ID

        Returns:
            Agent信息
        """
        return self.agents.get(agent_id)

    def get_all_agents(self, enabled_only: bool = True) -> List[Dict]:
        """
        获取所有Agent

        Args:
            enabled_only: 是否只返回启用的Agent

        Returns:
            Agent列表
        """
        agents = list(self.agents.values())

        if enabled_only:
            agents = [a for a in agents if a.get("enabled", True)]

        return agents

    def get_agent_descriptions(self, enabled_only: bool = True) -> List[Dict]:
        """
        获取Agent描述（用于任务规划）

        Args:
            enabled_only: 是否只返回启用的Agent

        Returns:
            Agent描述列表
        """
        agents = self.get_all_agents(enabled_only)

        descriptions = []
        for agent in agents:
            descriptions.append({
                "id": agent["id"],
                "name": agent["name"],
                "description": agent["description"],
                "capabilities": agent["capabilities"]
            })

        return descriptions

    def set_agent_status(self, agent_id: str, status: str):
        """
        设置Agent状态

        Args:
            agent_id: Agent ID
            status: 状态（active/inactive/error）
        """
        if agent_id in self.agents:
            self.agents[agent_id]["status"] = status
            logger.info(f"设置Agent状态: {agent_id} -> {status}")

    def enable_agent(self, agent_id: str):
        """启用Agent"""
        if agent_id in self.agents:
            self.agents[agent_id]["enabled"] = True
            logger.info(f"启用Agent: {agent_id}")

    def disable_agent(self, agent_id: str):
        """禁用Agent"""
        if agent_id in self.agents:
            self.agents[agent_id]["enabled"] = False
            logger.info(f"禁用Agent: {agent_id}")

    def get_agent_by_capability(self, capability: str) -> List[Dict]:
        """
        根据能力查找Agent

        Args:
            capability: 能力描述

        Returns:
            匹配的Agent列表
        """
        matching_agents = []

        for agent in self.get_all_agents():
            if capability.lower() in [c.lower() for c in agent["capabilities"]]:
                matching_agents.append(agent)

        return matching_agents

    def list_agents(self) -> str:
        """
        列出所有Agent（格式化输出）

        Returns:
            格式化的Agent列表
        """
        agents = self.get_all_agents(enabled_only=False)

        if not agents:
            return "没有注册的Agent"

        lines = ["已注册的Agent:"]
        for agent in agents:
            status = "✓" if agent["enabled"] else "✗"
            lines.append(
                f"  {status} {agent['id']}: {agent['name']} "
                f"({agent['status']})"
            )
            lines.append(f"     描述: {agent['description']}")
            lines.append(f"     能力: {', '.join(agent['capabilities'])}")

        return "\n".join(lines)
