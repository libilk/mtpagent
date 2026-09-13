"""
使用 MCP 工具的 Agent —— 完整示例

和 simple_agent.py 的唯一区别：
  工具不是代码里写死的 @tool，而是从 MCP 服务端运行时发现的。

流程：
  1. 启动 MCP 服务端（mcp_server.py，子进程）
  2. 客户端连接 → 调用 list_tools() 获取工具列表
  3. 把 MCP 工具列表 → 转成 LangChain StructuredTool
  4. bind_tools → 进入正常的 Agent 循环

运行前设置环境变量：
  export DASHSCOPE_API_KEY="your-key"

运行：
  python learn/mcp_agent.py
"""

import os
import asyncio
import logging
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

from pydantic import BaseModel, Field, create_model

from langchain.chat_models import init_chat_model
from langchain_core.tools import StructuredTool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

logging.basicConfig(level=logging.WARNING)  # 减少 MCP 内部日志噪音

# ============================================================
# 第一部分：MCP 工具 → LangChain 工具的转换
# ============================================================

class MCPToolBridge:
    """
    连接 MCP 服务端，把它的工具变成 LangChain 能用的格式。

    核心就两步：
      1. list_tools() → 拿到 MCP 侧的工具描述
      2. 每个工具包一层 call_tool → LangChain StructuredTool
    """

    def __init__(self, server_command: str, server_args: list[str]):
        self.server_params = StdioServerParameters(
            command=server_command,
            args=server_args,
        )
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
        print(f"[MCP] 已连接服务端")

    async def list_tools(self) -> list[dict]:
        """从 MCP 服务端获取工具列表"""
        result = await self.session.list_tools()
        # result.tools 是 MCP Tool 对象列表
        return [
            {
                "name": t.name,
                "description": t.description or "",
                "inputSchema": t.inputSchema,
            }
            for t in result.tools
        ]

    async def call_tool(self, name: str, args: dict) -> str:
        """调用 MCP 工具"""
        result = await self.session.call_tool(name, args)
        # 解析 MCP 返回的 content
        if result.content:
            return result.content[0].text
        return str(result)

    async def close(self):
        """断开连接"""
        if self._ctx:
            await self._ctx.__aexit__(None, None, None)


# ---- JSON Schema 类型 → Python 类型 ----
_TYPE_MAP = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _json_schema_to_pydantic(name: str, schema: dict) -> type[BaseModel]:
    """
    把 MCP 的 inputSchema 转成动态 Pydantic 模型。

    为什么需要这一步：
      bind_tools() 通过 Pydantic 模型生成 OpenAI 的 function schema。
      如果不用 Pydantic（用 **kwargs），LLM 就拿不到参数定义。

    输入:
      {"type":"object", "properties":{"user_id":{"type":"string"}}, "required":["user_id"]}

    输出:
      class get_user_info_args(BaseModel):
          user_id: str = Field(description="...")
    """
    fields: dict[str, tuple[type, Any]] = {}
    properties = schema.get("properties", {})
    required: list[str] = schema.get("required", [])

    for field_name, field_info in properties.items():
        json_type = field_info.get("type", "string")
        py_type = _TYPE_MAP.get(json_type, str)

        is_required = field_name in required
        default = ... if is_required else None  # ... = Pydantic 的 required 标记

        fields[field_name] = (
            py_type,
            Field(default=default, description=field_info.get("description", "")),
        )

    return create_model(f"{name}_args", **fields)


def _make_mcp_tool_fn(name: str, bridge: MCPToolBridge):
    """工厂函数：避免循环闭包陷阱，每个工具捕获自己的 name"""
    async def fn(**kwargs) -> str:
        return await bridge.call_tool(name, kwargs)
    return fn


def mcp_tools_to_langchain(bridge: MCPToolBridge, mcp_tools: list[dict]) -> list[StructuredTool]:
    """
    把 MCP 工具转成 LangChain StructuredTool。

    闭合链路：
      LLM 调 tool → StructuredTool.invoke() → bridge.call_tool() → MCP 服务端 → 结果返回
    """
    langchain_tools = []

    for mt in mcp_tools:
        name = mt["name"]

        # 1. 动态生成 Pydantic args_schema（让 LLM 知道参数格式）
        ArgsModel = _json_schema_to_pydantic(name, mt["inputSchema"])

        # 2. 创建 StructuredTool
        t = StructuredTool.from_function(
            coroutine=_make_mcp_tool_fn(name, bridge),
            name=name,
            description=mt["description"],
            args_schema=ArgsModel,
        )
        langchain_tools.append(t)

    return langchain_tools


# ============================================================
# 第二部分：Agent（和 simple_agent.py 几乎一模一样）
# ============================================================

class Agent:
    """Agent 本身不变——它只管消息列表 + 调用 LLM + 执行工具"""

    def __init__(self, tools: list, tools_map: dict):
        llm = init_chat_model(
            model=os.getenv("LLM_MODEL", "qwen-plus"),
            api_key=os.getenv("DASHSCOPE_API_KEY", os.getenv("OPENAI_API_KEY")),
            base_url=os.getenv(
                "OPENAI_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            model_provider="openai",
        )
        self.llm_with_tools = llm.bind_tools(tools)
        self.tools_map = tools_map
        self.memory: list = []

    async def run(self, user_input: str) -> str:
        messages = [
            SystemMessage(content="你是客服助手。用中文回复。"),
            *self.memory,
            HumanMessage(content=user_input),
        ]

        for _ in range(5):
            ai_msg = self.llm_with_tools.invoke(messages)
            messages.append(ai_msg)

            if not ai_msg.tool_calls:
                self._save_to_memory(user_input, ai_msg)
                return ai_msg.content or ""

            for tc in ai_msg.tool_calls:
                fn = self.tools_map[tc["name"]]
                result = await fn.ainvoke(tc["args"])  # await 因为 MCP 工具是异步的
                messages.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tc["id"],
                ))

        return "达到最大轮次"

    def _save_to_memory(self, user_input, ai_msg):
        from langchain_core.messages import HumanMessage as HM, AIMessage as AIM
        self.memory.append(HM(content=user_input))
        self.memory.append(ai_msg)
        if len(self.memory) > 20:
            self.memory = self.memory[-20:]


# ============================================================
# 第三部分：主流程 —— 连接 MCP → 创建 Agent → 交互
# ============================================================

async def main():
    # 1. 启动 MCP 服务端并连接
    bridge = MCPToolBridge(
        server_command="python",
        server_args=["learn/mcp_server.py"],
    )
    await bridge.connect()

    # 2. 获取 MCP 工具列表
    mcp_tools = await bridge.list_tools()
    print(f"[MCP] 发现工具: {[t['name'] for t in mcp_tools]}")

    # 3. MCP 工具 → LangChain 工具
    lc_tools = mcp_tools_to_langchain(bridge, mcp_tools)
    tools_map = {t.name: t for t in lc_tools}

    # 4. 创建 Agent
    agent = Agent(tools=lc_tools, tools_map=tools_map)

    print("=" * 50)
    print("MCP Agent 就绪，输入 quit 退出")
    print("=" * 50)

    # 5. 交互循环
    while True:
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break

        reply = await agent.run(user_input)
        print(f"\nAgent: {reply}")

    await bridge.close()


if __name__ == "__main__":
    asyncio.run(main())
