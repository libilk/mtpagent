import logging
from typing import List, Dict, Optional
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

# 摘要提示词模板
SUMMARY_PROMPT = """请将以下对话历史逐步压缩为简洁摘要。包含关键信息、决策和上下文：

{chat_history}

摘要："""


class ConversationMemory:
    """
    对话记忆管理器

    特性：
    - 使用 InMemoryChatMessageHistory 存储消息
    - 每 compress_every 轮自动压缩历史对话
    - 保留最近消息 + 历史摘要
    """

    def __init__(self, llm: BaseChatModel, max_history: int = 10, compress_every: int = 8):
        """
        初始化Memory

        Args:
            llm: LLM实例（用于生成摘要）
            max_history: 最大保留的对话轮数
            compress_every: 每N轮触发一次压缩（默认8轮）
        """
        self.llm = llm
        self.max_history = max_history
        self.compress_every = compress_every
        self.turn_count = 0
        self.summary = ""  # 历史摘要

        
        self.chat_memory = InMemoryChatMessageHistory()

        logger.info(f"ConversationMemory初始化完成（每{compress_every}轮压缩）")

    def add_message(self, role: str, content: str):
        """
        添加消息

        Args:
            role: 角色（user/assistant）
            content: 消息内容
        """
        if role == "user":
            self.chat_memory.add_user_message(content)
        elif role == "assistant":
            self.chat_memory.add_ai_message(content)

        logger.debug(f"添加消息: {role} - {content[:50]}...")

    def add_user_message(self, content: str):
        """添加用户消息"""
        self.add_message("user", content)
        self.turn_count += 1
        self._check_and_compress()

    def add_assistant_message(self, content: str):
        """添加助手消息"""
        self.add_message("assistant", content)

    def _check_and_compress(self):
        """检查是否需要压缩历史"""
        if self.turn_count % self.compress_every == 0:
            logger.info(f"[记忆压缩] 已达到{self.turn_count}轮，触发历史压缩")
            self._compress_history()
            self._log_memory_status()

    def _compress_history(self):
        """将当前消息压缩为摘要，保留最近 max_history 条消息"""
        messages = self.chat_memory.messages
        total = len(messages)

        # 保留最近 max_history*2 条消息（每轮=用户+助手=2条）
        keep_count = self.max_history * 2
        if total <= keep_count:
            return

        # 取要被压缩的旧消息
        old_messages = messages[:total - keep_count]
        # 保留最近的消息
        recent_messages = messages[total - keep_count:]

        # 用LLM生成摘要
        try:
            old_text = "\n".join(
                f"{'用户' if isinstance(m, HumanMessage) else '助手'}: {m.content}"
                for m in old_messages
            )
            prompt = SUMMARY_PROMPT.format(chat_history=old_text)
            response = self.llm.invoke(prompt)
            new_summary = response.content if hasattr(response, 'content') else str(response)

            if self.summary:
                self.summary = self.summary + "\n" + new_summary
            else:
                self.summary = new_summary

            # 更新 chat_memory：摘要 + 最近消息
            self.chat_memory.clear()
            if self.summary:
                self.chat_memory.add_message(
                    SystemMessage(content=f"[对话历史摘要]\n{self.summary}")
                )
            for msg in recent_messages:
                self.chat_memory.add_message(msg)

            logger.info(f"[记忆压缩] 压缩 {len(old_messages)} 条旧消息为摘要，保留 {len(recent_messages)} 条最近消息")
        except Exception as e:
            logger.warning(f"压缩历史失败: {e}，跳过本次压缩")

    def _log_memory_status(self):
        """记录记忆状态"""
        try:
            messages = self.chat_memory.messages
            logger.info(f"[记忆状态] 当前消息数: {len(messages)}，摘要长度: {len(self.summary)}")
        except Exception as e:
            logger.debug(f"记录记忆状态失败: {e}")

    def get_messages(self) -> List[Dict]:
        """
        获取对话历史

        Returns:
            消息列表
        """
        messages = self.chat_memory.messages
        return [
            {
                "role": "user" if isinstance(msg, HumanMessage)
                else "assistant" if isinstance(msg, AIMessage)
                else "system",
                "content": msg.content,
            }
            for msg in messages
        ]

    def get_context(self) -> str:
        """
        获取对话上下文（包含摘要）

        Returns:
            上下文字符串
        """
        parts = []
        if self.summary:
            parts.append(f"[历史摘要]\n{self.summary}")
        for msg in self.chat_memory.messages:
            if isinstance(msg, SystemMessage):
                continue  # 摘要已经在上面了
            role = "用户" if isinstance(msg, HumanMessage) else "助手"
            parts.append(f"{role}: {msg.content}")
        return "\n".join(parts)

    def clear(self):
        """清空记忆"""
        self.chat_memory.clear()
        self.summary = ""
        self.turn_count = 0
        logger.info("对话记忆已清空")


class ConversationSummaryMemory(ConversationMemory):
    """
    对话摘要记忆（兼容旧API）

    继承自ConversationMemory，提供相同功能
    """

    def __init__(self, llm: BaseChatModel, max_history: int = 10, summary_threshold: int = 20):
        """
        初始化

        Args:
            llm: LLM实例
            max_history: 最大保留的对话轮数
            summary_threshold: 触发摘要的消息数阈值（已废弃，使用compress_every）
        """
        compress_every = max(5, summary_threshold // 2)
        super().__init__(llm, max_history, compress_every)
        logger.info(f"ConversationSummaryMemory初始化完成（每{compress_every}轮压缩）")


class ContextMemory:
    """
    上下文记忆管理器

    用于跟踪实体、主题等结构化信息
    """

    def __init__(self):
        """初始化上下文记忆"""
        self.context = {}
        logger.info("ContextMemory初始化完成")

    def set(self, key: str, value: any):
        """
        设置上下文

        Args:
            key: 键
            value: 值
        """
        self.context[key] = value
        logger.debug(f"设置上下文: {key} = {value}")

    def get(self, key: str, default=None):
        """
        获取上下文

        Args:
            key: 键
            default: 默认值

        Returns:
            上下文值
        """
        return self.context.get(key, default)

    def update(self, updates: Dict):
        """
        批量更新上下文

        Args:
            updates: 更新字典
        """
        self.context.update(updates)
        logger.debug(f"批量更新上下文: {list(updates.keys())}")

    def clear(self):
        """清空上下文"""
        self.context.clear()
        logger.info("上下文记忆已清空")

    def __repr__(self):
        return f"ContextMemory({len(self.context)} items)"