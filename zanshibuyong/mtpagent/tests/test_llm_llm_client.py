import os
import json
import pytest
from unittest.mock import patch, MagicMock

from llm.llm_client import LLM


@pytest.fixture
def llm_with_key():
    return LLM(api_key="test-api-key")


class TestLLMInit:
    def test_no_api_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="API密钥未设置"):
                LLM(api_key=None)

    def test_with_api_key_creates_client(self, llm_with_key):
        assert llm_with_key.client is not None

    def test_default_values(self, llm_with_key):
        assert llm_with_key.model_name == "qwen-plus"
        assert llm_with_key.temperature == 0.7
        assert llm_with_key.max_tokens == 1024
        assert llm_with_key.timeout == 60

    def test_enable_cache_attribute(self):
        """enable_cache 属性应被正确记录"""
        llm = LLM(api_key="test-key", enable_cache=True)
        assert llm.enable_cache is True
        llm2 = LLM(api_key="test-key", enable_cache=False)
        assert llm2.enable_cache is False
