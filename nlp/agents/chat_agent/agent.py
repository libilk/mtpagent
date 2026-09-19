# -*- coding: utf-8 -*-
"""
Chat Agent
==========

通用闲聊兜底 Agent，处理意图不明确或不属于其他 Agent 的对话。

形态：最轻的一个 Agent —— 零工具、零检索、零 ReAct 循环，就是把历史消息拼一拼
发给 LLM 拿一句回复。它的价值不在能力，而在"路由需要有兜底"：路由器挑不出
专业 Agent 时，总得有个节点接住，否则图会走进死路。

找它的注册代码时注意：它注册在 enhanced_entry.py 的 _register_vqa_agent 里
（函数名只提 VQA，实际末尾挂了 chat_agent 的注册块）—— 函数名和内容对不上，
别因为搜不到 _register_chat_agent 就以为它没被注册。它确实是现役 Agent。
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
        # handle（处理入口）：图节点只认这个名字，内部转给自己更语义化的 run()。
        return self.run(query, context)

    def run(self, query: str, context: Dict[str, Any] = None) -> str:
        context = context or {}
        # 这里读的键是 "conversation_history"，而 database_agent / knowledge_agent
        # 等读的是 "history"，图节点 make_agent_node 注入的键名是后者 ——
        # 也就是说在 LangGraph 路径下这里恒取到空列表，闲聊 Agent 拿不到历史。
        # 事实性说明，未改动逻辑。
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

        # 只回带最近 6 条：闲聊不需要长记忆，多带只会挤占提示词预算
        for turn in history[-6:]:
            if turn.get("role") in ("user", "assistant"):
                messages.append({"role": turn["role"], "content": turn["content"]})

        messages.append({"role": "user", "content": query})

        try:
            # temperature=0.7 比调工具场景（0.3）高：闲聊要的是自然、有变化，
            # 不需要可复现；这里不走 ReAct，所以没有 tools 参数。
            response = self.llm.chat(messages, temperature=0.7, max_tokens=2048)
            return response
        except Exception as e:
            logger.error("ChatAgent 调用失败: %s", e)
            return "抱歉，我暂时无法回答，请稍后再试。"
