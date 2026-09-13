"""
MCP 客户端 —— 连接 FastMCP 服务端，获取工具/资源并调用

运行方式：
  python learn/fast_mcp_client.py
"""

import asyncio
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters


class FastMCPClient:
    """
    MCP 客户端：启动服务端子进程，通过 stdio 通信。

    职责：
      - 启动服务端子进程
      - 初始化会话（握手）
      - 列出工具 / 列出资源
      - 调用工具 / 读取资源
    """

    def __init__(self, command: str, args: list[str]):
        self.server_params = StdioServerParameters(command=command, args=args)
        self.session: ClientSession | None = None
        self._read = None
        self._write = None
        self._ctx = None

    async def connect(self):
        """启动 MCP 服务端子进程并建立连接"""
        self._ctx = stdio_client(self.server_params)
        self._read, self._write = await self._ctx.__aenter__()
        self.session = ClientSession(self._read, self._write)
        await self.session.initialize()
        print("[客户端] 已连接 MCP 服务端")

    async def list_tools(self) -> list[dict]:
        """获取服务端暴露的工具列表"""
        result = await self.session.list_tools()
        return [
            {"name": t.name, "description": t.description, "inputSchema": t.inputSchema}
            for t in result.tools
        ]

    async def list_resources(self) -> list[dict]:
        """获取服务端暴露的资源列表"""
        result = await self.session.list_resources()
        return [
            {"uri": r.uri, "name": r.name, "description": r.description}
            for r in result.resources
        ]

    async def call_tool(self, name: str, arguments: dict) -> str:
        """调用指定工具"""
        result = await self.session.call_tool(name, arguments)
        if result.content:
            return result.content[0].text
        return str(result)

    async def read_resource(self, uri: str) -> str:
        """读取指定资源"""
        result = await self.session.read_resource(uri)
        if result.contents:
            return result.contents[0].text
        return str(result)

    async def close(self):
        if self._ctx:
            await self._ctx.__aexit__(None, None, None)


# ---- 使用示例 ----
async def main():
    client = FastMCPClient(
        command="python",
        args=["learn/fast_mcp_server.py"],
    )
    await client.connect()

    # 1. 列出工具
    tools = await client.list_tools()
    print(f"\n=== 工具列表 ({len(tools)} 个) ===")
    for t in tools:
        print(f"  • {t['name']}: {t['description']}")

    # 2. 调用工具
    print("\n=== 调用工具 ===")
    result = await client.call_tool("add", {"a": 3, "b": 5})
    print(f"  add(3, 5) = {result}")

    result = await client.call_tool("get_weather", {"city": "北京"})
    print(f"  get_weather('北京') = {result}")

    # 3. 列出资源
    resources = await client.list_resources()
    print(f"\n=== 资源列表 ({len(resources)} 个) ===")
    for r in resources:
        print(f"  • {r['uri']}: {r['description']}")

    # 4. 读取资源
    print("\n=== 读取资源 ===")
    content = await client.read_resource("greeting://World")
    print(f"  greeting://World → {content}")

    content = await client.read_resource("config://app")
    print(f"  config://app → {content}")

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
