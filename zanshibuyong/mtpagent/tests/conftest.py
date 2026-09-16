import sys
from pathlib import Path

# 将项目根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_llm():
    """Mock LLM for memory and cache tests."""
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="[摘要] 历史对话摘要内容")
    return llm


@pytest.fixture
def mock_embedder():
    """Mock embedder returning a fixed vector."""
    emb = MagicMock()
    emb.embed.return_value = [0.1, 0.2, 0.3]
    emb.embed_query.return_value = [0.1, 0.2, 0.3]
    emb.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    return emb


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    redis = MagicMock()
    redis.get.return_value = None
    redis.keys.return_value = []
    return redis
