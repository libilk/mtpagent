# -*- coding: utf-8 -*-
"""
Agent注册中心
=============

管理所有Agent的注册和发现。
注册时强制校验 Agent 是否实现 AgentProtocol 接口。

registry（注册中心）= Agent 名册：按名字取实例。图上的 Agent 节点、路由器建索引、
评审节点取实例，全都从这里拿 —— 它是"有哪些 Agent 可选"的唯一事实来源。

两件容易读错的事：
1. 这里的 isinstance 校验就是 Harness（约束层）的落点 —— 把"Agent 必须符合统一接口"
   从口头约定变成注册那一刻的硬拦截（校验规则本身见 core/protocol.py）。
   它只保证接口形状，不保证 Agent 干得对。
2. 名册里有什么 ≠ agents/ 目录下有什么。现役 5 个：knowledge / database /
   customer_service / vqa / chat（注册处见 enhanced_entry._register_agents）——
   注意 chat_agent 的注册代码写在 _register_vqa_agent 里面，函数名和内容对不上。
   另有 document_agent 有类但注册调用已摘掉、critic_agent 有类但从未注册，都别当现役 Agent 读。
"""

import logging
from typing import Dict, List, Optional, Any

from core.protocol import AgentProtocol

logger = logging.getLogger(__name__)


class AgentRegistry:
    """Agent注册中心

    名册就是一个普通 dict：没有并发保护、不落盘，重启后靠 _register_agents 重填一遍。
    存的是"信息字典"而不是 Agent 实例本身，实例在字典的 "instance" 键里，
    取值时要多剥一层 —— 这是读本文件最容易栽的地方。
    """

    def __init__(self):
        self.agents = {}  # {agent_id: agent_info}；agent_info 是描述字典，不是 Agent 对象

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

        校验不过就直接抛错、不留半条记录：宁可启动时就炸，也别让接口不合规的 Agent
        混进名册、等到流程跑到某一步才暴露。这是约束层的取舍。
        """
        # ========== Harness 契约校验 ==========
        # 配 runtime_checkable 的 Protocol 做 isinstance，等价于"有没有 handle 方法"的
        # 鸭子类型检查：只查方法在不在，签名对不对、返回什么类型一概不管。
        # 因此校验通过 ≠ 调用一定成功，别把这条当强类型保证。
        if not isinstance(agent_instance, AgentProtocol):
            raise TypeError(
                f"Agent '{agent_id}' ({type(agent_instance).__name__}) 未实现 AgentProtocol。"
                f"请确保该类定义了 handle(self, query: str, context: Dict[str, Any]) -> str 方法。"
            )
        # 注意 enabled 与 status 是两套独立开关：enabled 决定"要不要被选用"（见 get_all_agents），
        # status 只是运行状态标记，不影响过滤结果。
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

        规划器（planner）拿到的就是这份裁剪过的视图：只有 id/name/description/capabilities，
        没有 instance 也没有 model。即规划器是靠"文字描述"决定派谁，而不是靠代码探测能力。

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

        改的只是上面那个 status 标记位，不碰 enabled —— 所以把 Agent 置成 error
        也不会让它退出被选中的范围，要停用得另外调 disable_agent。

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

        是精确匹配（大小写不敏感地逐项相等），不是模糊/包含匹配 ——
        名字叫"根据能力查找"容易让人以为传个关键词就能搜到，实际传 "退款" 匹配不到
        "退款处理"。本方法当前无调用方。

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
