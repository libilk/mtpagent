# -*- coding: utf-8 -*-
"""
意图验证器
==========

验证Agent选择是否正确，支持自动纠正
"""

import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class IntentValidator:
    """意图验证器"""

    def __init__(self, llm):
        self.llm = llm

    def validate_routing(
        self,
        query: str,
        selected_agent: Dict,
        all_agents: list
    ) -> Tuple[bool, Optional[str], str]:
        """
        验证路由是否正确

        Args:
            query: 用户查询
            selected_agent: 选中的Agent
            all_agents: 所有可用Agent

        Returns:
            (是否正确, 建议的Agent ID, 原因)
        """
        # 构建验证提示词
        prompt = f"""请验证以下Agent选择是否正确。

用户查询: {query}

已选择的Agent:
- ID: {selected_agent['id']}
- 名称: {selected_agent['name']}
- 描述: {selected_agent['description']}
- 能力: {', '.join(selected_agent['capabilities'])}

所有可用Agent:
{self._format_agents(all_agents)}

请分析：
1. 选择是否合理？
2. 如果不合理，应该选择哪个Agent？

只输出JSON格式：
{{
  "is_correct": true/false,
  "suggested_agent_id": "agent_id或null",
  "reason": "原因说明"
}}"""

        try:
            response = self.llm.generate(prompt, temperature=0.1, max_tokens=200)

            # 解析响应
            from orchestrator.error_handler import AgentErrorHandler
            result = AgentErrorHandler.parse_json_safe(response)

            if result:
                return (
                    result.get('is_correct', True),
                    result.get('suggested_agent_id'),
                    result.get('reason', '')
                )
        except Exception as e:
            logger.error(f"意图验证失败: {e}")

        # 默认认为正确
        return True, None, ""

    def _format_agents(self, agents: list) -> str:
        """格式化Agent列表"""
        lines = []
        for agent in agents:
            lines.append(
                f"- {agent['id']}: {agent['name']}\n"
                f"  描述: {agent['description']}\n"
                f"  能力: {', '.join(agent['capabilities'])}"
            )
        return "\n".join(lines)
