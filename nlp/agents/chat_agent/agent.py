# -*- coding: utf-8 -*-
"""
Chat Agent
==========

通用闲聊兜底 Agent，处理意图不明确或不属于其他 Agent 的对话。
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ChatAgent:
    """通用闲聊兜底 Agent"""

    def __init__(self, llm):
        self.llm = llm
        logger.info("ChatAgent 初始化完成")

    def handle(self, query: str, context: Dict[str, Any] = None) -> str:
        return self.run(query, context)

    def run(self, query: str, context: Dict[str, Any] = None) -> str:
        context = context or {}
        history = context.get("conversation_history", [])

        messages = []
        messages.append({
            "role": "system",
            "content": (
                "你是一个友好、专业的 AI 助手。"
                "你可以回答各种问题、进行闲聊、提供建议。"
                "如果用户的问题涉及专业领域（数据库查询、文档分析、数据可视化等），"
                "请告知用户可以使用对应的专业功能。"
            )
        })

        for turn in history[-6:]:
            if turn.get("role") in ("user", "assistant"):
                messages.append({"role": turn["role"], "content": turn["content"]})

        messages.append({"role": "user", "content": query})

        try:
            response = self.llm.chat(messages, temperature=0.7, max_tokens=2048)
            return response
        except Exception as e:
            logger.error("ChatAgent 调用失败: %s", e)
            return "抱歉，我暂时无法回答，请稍后再试。"
