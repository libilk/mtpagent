# -*- coding: utf-8 -*-
"""
Memory系统
==========

对话历史管理，支持自动压缩。
纯Python实现（不依赖 langchain.memory，兼容新版 LangChain）。

重要：MemoryStore 按 thread_id 隔离记忆实例，
      不同浏览器窗口/会话之间互不干扰。
"""

import time
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class ConversationMemory:
    """
    对话记忆管理器

    特性：
    - 存储最近的对话消息
    - 超过 max_messages 条后自动截断旧消息
    - 兼容 make_agent_node 中的 shared_memory 接口
    """

    def __init__(self, llm=None, max_token_limit: int = 2000, max_messages: int = 20):
        """
        初始化Memory

        Args:
            llm: LLM实例（预留，兼容旧接口）
            max_token_limit: token数量上限（预留参数，兼容旧接口）
            max_messages: 最大保存消息条数（超过后截断最旧的）
        """
        self.llm = llm
        self.max_token_limit = max_token_limit
        self.max_messages = max_messages
        self._messages: List[Dict[str, str]] = []

        logger.info(f"ConversationMemory初始化完成（最大消息数: {max_messages}）")

    def add_message(self, role: str, content: str):
        """
        添加消息

        Args:
            role: 角色（user/assistant）
            content: 消息内容
        """
        self._messages.append({"role": role, "content": content})

        # 超过上限时截断旧消息
        if len(self._messages) > self.max_messages:
            self._messages = self._messages[-self.max_messages:]

        logger.debug(f"添加消息: {role} - {content[:50]}...")

    def add_user_message(self, content: str):
        """添加用户消息"""
        self.add_message("user", content)

    def add_assistant_message(self, content: str):
        """添加助手消息"""
        self.add_message("assistant", content)

    def get_messages(self) -> List[Dict]:
        """
        获取对话历史

        Returns:
            消息列表，每项 {"role": "user"|"assistant", "content": str}
        """
        return list(self._messages)

    def get_context(self) -> str:
        """
        获取对话上下文（文本格式）

        Returns:
            上下文字符串
        """
        parts = []
        for msg in self._messages:
            role_label = "用户" if msg["role"] == "user" else "助手"
            parts.append(f"{role_label}: {msg['content']}")
        return "\n".join(parts)

    def clear(self):
        """清空记忆"""
        self._messages.clear()
        logger.info("对话记忆已清空")


class ConversationSummaryMemory(ConversationMemory):
    """
    对话摘要记忆（兼容旧API）

    继承自ConversationMemory，提供相同功能
    """

    def __init__(self, llm=None, max_token_limit: int = 2000, **kwargs):
        """
        初始化

        Args:
            llm: LLM实例
            max_token_limit: token数量上限（兼容旧参数）
            **kwargs: 兼容旧参数（max_history, summary_threshold等）
        """
        super().__init__(llm, max_token_limit)
        logger.info(f"ConversationSummaryMemory初始化完成（token上限: {max_token_limit}）")


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


class MemoryStore:
    """
    按 thread_id 隔离的记忆管理器

    每个 thread_id（对应一个浏览器窗口/会话）拥有独立的 ConversationMemory 实例。
    不同窗口/用户之间的对话记忆完全隔离，互不干扰。

    特性：
    - 按需创建：首次访问某 thread_id 时自动创建记忆实例
    - 自动清理：定期清理超过 ttl 未活跃的记忆实例，防止内存泄漏
    - 线程安全：使用 threading.Lock 保护共享数据结构
    """

    def __init__(
        self,
        llm=None,
        max_token_limit: int = 2000,
        max_messages: int = 20,
        ttl_seconds: int = 7200,
    ):
        """
        初始化 MemoryStore

        Args:
            llm: LLM实例（传递给每个 ConversationMemory）
            max_token_limit: 每个记忆实例的 token 上限
            max_messages: 每个记忆实例的最大消息条数
            ttl_seconds: 记忆实例的存活时间（秒），超过未活跃则自动清理，默认2小时
        """
        import threading

        self._llm = llm
        self._max_token_limit = max_token_limit
        self._max_messages = max_messages
        self._ttl = ttl_seconds

        self._store: Dict[str, ConversationMemory] = {}
        self._last_access: Dict[str, float] = {}
        self._lock = threading.Lock()

        logger.info(
            f"MemoryStore 初始化完成（max_messages={max_messages}, ttl={ttl_seconds}s）"
        )

    def get(self, thread_id: str) -> ConversationMemory:
        """
        获取指定 thread_id 的记忆实例（不存在则自动创建）

        Args:
            thread_id: 会话ID

        Returns:
            该会话专属的 ConversationMemory 实例
        """
        with self._lock:
            # 定期清理过期实例（每次访问时检查）
            self._cleanup_expired()

            if thread_id not in self._store:
                self._store[thread_id] = ConversationMemory(
                    llm=self._llm,
                    max_token_limit=self._max_token_limit,
                    max_messages=self._max_messages,
                )
                logger.info(f"[MemoryStore] 创建新记忆实例: {thread_id}")

            self._last_access[thread_id] = time.time()
            return self._store[thread_id]

    def clear(self, thread_id: str) -> None:
        """清空指定会话的记忆"""
        with self._lock:
            if thread_id in self._store:
                self._store[thread_id].clear()
                logger.info(f"[MemoryStore] 清空记忆: {thread_id}")

    def remove(self, thread_id: str) -> None:
        """删除指定会话的记忆实例"""
        with self._lock:
            self._store.pop(thread_id, None)
            self._last_access.pop(thread_id, None)
            logger.info(f"[MemoryStore] 删除记忆实例: {thread_id}")

    def _cleanup_expired(self) -> None:
        """清理超过 TTL 未活跃的记忆实例（需在锁内调用）"""
        now = time.time()
        expired = [
            tid for tid, ts in self._last_access.items()
            if now - ts > self._ttl
        ]
        for tid in expired:
            self._store.pop(tid, None)
            self._last_access.pop(tid, None)
            logger.info(f"[MemoryStore] 自动清理过期记忆: {tid}")

    def active_count(self) -> int:
        """返回当前活跃的记忆实例数量"""
        with self._lock:
            return len(self._store)

    def __repr__(self):
        return f"MemoryStore({self.active_count()} active sessions)"
