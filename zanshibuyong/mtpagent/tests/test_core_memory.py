import pytest
from unittest.mock import patch, MagicMock

from core.memory import ConversationMemory, ConversationSummaryMemory, ContextMemory


class TestConversationMemory:
    def test_init_defaults(self, mock_llm):
        m = ConversationMemory(llm=mock_llm)
        assert m.turn_count == 0
        assert m.summary == ""
        assert m.chat_memory is not None

    def test_add_user_message(self, mock_llm):
        m = ConversationMemory(llm=mock_llm)
        m.add_user_message("你好")
        assert m.turn_count == 1
        assert len(m.chat_memory.messages) == 1

    def test_add_assistant_message(self, mock_llm):
        m = ConversationMemory(llm=mock_llm)
        m.add_user_message("你好")
        m.add_assistant_message("你好，有什么可以帮助你的")
        assert len(m.chat_memory.messages) == 2

    def test_get_messages_format(self, mock_llm):
        m = ConversationMemory(llm=mock_llm)
        m.add_user_message("测试消息")
        msgs = m.get_messages()
        assert msgs == [{"role": "user", "content": "测试消息"}]

    def test_get_context_no_summary(self, mock_llm):
        m = ConversationMemory(llm=mock_llm)
        m.add_user_message("你好")
        context = m.get_context()
        assert "你好" in context

    def test_get_context_with_summary(self, mock_llm):
        m = ConversationMemory(llm=mock_llm, max_history=1, compress_every=1)
        m.add_user_message("msg1")
        m.add_assistant_message("resp1")
        m.add_user_message("msg2")
        m.add_assistant_message("resp2")
        m.add_user_message("msg3")
        context = m.get_context()
        assert "[历史摘要]" in context or m.summary != ""

    def test_trigger_compression(self, mock_llm):
        m = ConversationMemory(llm=mock_llm, max_history=1, compress_every=2)
        for i in range(4):
            m.add_user_message(f"msg{i}")
            m.add_assistant_message(f"resp{i}")
        # 第4次 add_user_message 触发压缩
        assert m.summary != ""

    def test_keep_recent_messages_after_compress(self, mock_llm):
        m = ConversationMemory(llm=mock_llm, max_history=1, compress_every=1)
        for i in range(8):
            m.add_user_message(f"msg{i}")
            m.add_assistant_message(f"resp{i}")
        msgs = m.get_messages()
        # 压缩保留 2 条最近消息 + 摘要 SystemMessage = 3，但末尾还有一条未配对的 assistant 消息
        assert len(msgs) <= 5

    def test_clear_resets_all(self, mock_llm):
        m = ConversationMemory(llm=mock_llm)
        m.add_user_message("hello")
        m.add_assistant_message("hi")
        m.clear()
        assert m.turn_count == 0
        assert m.summary == ""
        assert len(m.chat_memory.messages) == 0

    def test_compress_handles_llm_failure(self, mock_llm):
        mock_llm.invoke.side_effect = RuntimeError("LLM down")
        m = ConversationMemory(llm=mock_llm, max_history=1, compress_every=1)
        for i in range(5):
            m.add_user_message(f"msg{i}")
            m.add_assistant_message(f"resp{i}")
        # Should not raise
        assert True

    def test_add_message_direct(self, mock_llm):
        m = ConversationMemory(llm=mock_llm)
        m.add_message("user", "hello")
        m.add_message("assistant", "world")
        assert len(m.chat_memory.messages) == 2


class TestConversationSummaryMemory:
    def test_inheritance(self, mock_llm):
        m = ConversationSummaryMemory(llm=mock_llm, summary_threshold=20)
        assert m.compress_every == 10

    def test_summary_threshold_minimum(self, mock_llm):
        m = ConversationSummaryMemory(llm=mock_llm, summary_threshold=8)
        assert m.compress_every == 5


class TestContextMemory:
    def test_set_and_get(self):
        cm = ContextMemory()
        cm.set("key", "value")
        assert cm.get("key") == "value"

    def test_get_default(self):
        cm = ContextMemory()
        assert cm.get("missing", "default") == "default"

    def test_update_batch(self):
        cm = ContextMemory()
        cm.update({"a": 1, "b": 2})
        assert cm.get("a") == 1
        assert cm.get("b") == 2

    def test_clear(self):
        cm = ContextMemory()
        cm.set("key", "value")
        cm.clear()
        assert cm.get("key") is None

    def test_repr(self):
        cm = ContextMemory()
        assert "0 items" in repr(cm)
        cm.set("k", "v")
        assert "1 items" in repr(cm)
