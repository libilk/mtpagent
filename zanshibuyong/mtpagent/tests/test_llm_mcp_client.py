import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from llm.mcp_client import MCPClient, MCPClientSync


@pytest.fixture
def mock_session():
    session = AsyncMock()
    session.initialize = AsyncMock()
    session.list_tools = AsyncMock(return_value=MagicMock(tools=[{"name": "t1"}]))
    session.call_tool = AsyncMock(return_value={"result": "ok"})
    session.close = AsyncMock()
    return session


@pytest.fixture
def mock_stdio():
    with patch("llm.mcp_client.stdio_client") as mock:
        async def _mock_stdio(*a, **kw):
            read = AsyncMock()
            write = AsyncMock()
            return (read, write)
        mock.side_effect = _mock_stdio
        yield mock


class TestMCPClient:
    @pytest.mark.asyncio
    async def test_connect_server_success(self, mock_stdio, mock_session):
        with patch("llm.mcp_client.ClientSession", return_value=mock_session):
            client = MCPClient()
            await client.connect_server("test_server", "python", ["-m", "test_mod"])
            assert "test_server" in client.sessions

    @pytest.mark.asyncio
    async def test_connect_server_failure(self, mock_stdio):
        async def raise_err(*a, **kw):
            raise RuntimeError("connection failed")
        mock_stdio.side_effect = raise_err
        client = MCPClient()
        with pytest.raises(RuntimeError, match="connection failed"):
            await client.connect_server("bad", "cmd", [])

    @pytest.mark.asyncio
    async def test_list_tools(self, mock_session):
        client = MCPClient()
        client.sessions["test"] = mock_session
        tools = await client.list_tools("test")
        assert tools == [{"name": "t1"}]
        mock_session.list_tools.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_list_tools_not_connected(self):
        client = MCPClient()
        with pytest.raises(ValueError, match="未连接"):
            await client.list_tools("missing")

    @pytest.mark.asyncio
    async def test_call_tool_success(self, mock_session):
        client = MCPClient()
        client.sessions["test"] = mock_session
        result = await client.call_tool("test", "do_thing", {"arg": 1})
        assert result == {"result": "ok"}
        mock_session.call_tool.assert_awaited_once_with("do_thing", {"arg": 1})

    @pytest.mark.asyncio
    async def test_call_tool_not_connected(self):
        client = MCPClient()
        with pytest.raises(ValueError, match="未连接"):
            await client.call_tool("missing", "tool", {})

    @pytest.mark.asyncio
    async def test_call_tool_exception(self, mock_session):
        mock_session.call_tool.side_effect = RuntimeError("tool error")
        client = MCPClient()
        client.sessions["test"] = mock_session
        with pytest.raises(RuntimeError, match="tool error"):
            await client.call_tool("test", "tool", {})

    @pytest.mark.asyncio
    async def test_close(self, mock_session):
        session2 = AsyncMock()
        session2.close = AsyncMock()
        client = MCPClient()
        client.sessions["s1"] = mock_session
        client.sessions["s2"] = session2
        await client.close()
        mock_session.close.assert_awaited_once()
        session2.close.assert_awaited_once()
        assert len(client.sessions) == 0

    @pytest.mark.asyncio
    async def test_close_handles_session_error(self, mock_session):
        mock_session.close.side_effect = RuntimeError("close failed")
        client = MCPClient()
        client.sessions["s1"] = mock_session
        await client.close()
        assert len(client.sessions) == 0


class TestMCPClientSync:
    def test_connect_server(self):
        sync_client = MCPClientSync()
        mock_async = MagicMock()
        async def _connect(server_name, command, args, env=None):
            return None
        mock_async.connect_server = _connect
        sync_client.client = mock_async
        sync_client.connect_server("srv", "cmd", ["a"])  # should not raise

    def test_list_tools(self):
        sync_client = MCPClientSync()
        mock_async = MagicMock()
        async def _list_tools(server_name):
            return [{"name": "t1"}]
        mock_async.list_tools = _list_tools
        sync_client.client = mock_async
        result = sync_client.list_tools("srv")
        assert result == [{"name": "t1"}]

    def test_call_tool(self):
        sync_client = MCPClientSync()
        mock_async = MagicMock()
        async def _call_tool(server_name, tool_name, arguments):
            return {"result": "ok"}
        mock_async.call_tool = _call_tool
        sync_client.client = mock_async
        result = sync_client.call_tool("srv", "tool", {"a": 1})
        assert result == {"result": "ok"}

    def test_close(self):
        sync_client = MCPClientSync()
        mock_async = MagicMock()
        async def _close():
            pass
        mock_async.close = _close
        sync_client.client = mock_async
        sync_client.close()  # should not raise
