"""
MCP 主机（Agent）—— 通过客户端连接 MCP 服务端，使用 MCP 工具回答问题。

主机 = 最终应用程序。它不直接和服务端通信，而是通过客户端。

  服务端 ←—stdio—→ 客户端 ←—Python调用—→ 主机(Agent)

运行前设置环境变量：
  export DASHSCOPE_API_KEY="your-key"
  export LLM_MODEL="qwen-plus"   # 可选

运行方式：
  python learn/fast_mcp_host.py
"""

import os
import json
import asyncio
import logging

from openai import OpenAI

from learn.fast_mcp_client import FastMCPClient

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ============================================================
# 核心：MCP 工具 → OpenAI tool 格式
# ============================================================

def mcp_tools_to_openai(mcp_tools: list[dict]) -> list[dict]:
    """
    MCP inputSchema 和 OpenAI parameters 都是 JSON Schema，
    字段名都不用改，直接映射即可。
    """
    openai_tools = []
    for t in mcp_tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["inputSchema"],
            }
        })
    return openai_tools


# ============================================================
# MCP 主机：通过客户端使用 MCP 工具
# ============================================================

class MCPHost:
    """
    MCP 主机 —— 使用 MCP 工具的 Agent。

    职责：
      1. 通过 FastMCPClient 连接服务端
      2. 获取工具并转为 OpenAI 格式
      3. 调用 LLM，让 LLM 决定调哪些工具
      4. 执行工具，把结果返回给 LLM
      5. 循环直到 LLM 生成最终回答
    """

    def __init__(self):
        self.client: FastMCPClient | None = None
        self.tools_openai: list[dict] = []
        self.llm = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY", os.getenv("OPENAI_API_KEY", "sk-placeholder")),
            base_url=os.getenv(
                "OPENAI_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
        )
        self.model = os.getenv("LLM_MODEL", "qwen-plus")

    async def connect(self):
        """连接 MCP 服务端，获取工具列表"""
        self.client = FastMCPClient(
            command="python",
            args=["learn/fast_mcp_server.py"],
        )
        await self.client.connect()

        mcp_tools = await self.client.list_tools()
        self.tools_openai = mcp_tools_to_openai(mcp_tools)
        print(f"[主机] 已加载 {len(self.tools_openai)} 个 MCP 工具")
        for t in self.tools_openai:
            print(f"  • {t['function']['name']}")

    async def chat(self, user_message: str) -> str:
        """处理用户消息，自动调用 MCP 工具"""
        messages = [
            {"role": "system", "content": "你是一个有用的助手。使用中文回复。必要时调用工具获取信息。"},
            {"role": "user", "content": user_message},
        ]

        # Agent 循环：最多 5 轮
        for _ in range(5):
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools_openai or None,  # 如果没有工具则传 None
            )

            choice = response.choices[0]
            message = choice.message

            # 如果没有 tool_calls，说明 LLM 已经给出最终回答
            if not message.tool_calls:
                return message.content or ""

            # 有 tool_calls，逐一执行
            messages.append(message.model_dump())  # 把 AI 消息加入历史

            for tc in message.tool_calls:
                tool_name = tc.function.name
                tool_args = json.loads(tc.function.arguments)

                print(f"  [主机] 调用工具: {tool_name}({tool_args})")

                # 调用 MCP 工具（通过客户端 → stdio → 服务端子进程）
                result = await self.client.call_tool(tool_name, tool_args)

                print(f"  [主机] 工具返回: {result}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        return "已达到最大轮次，未能完成请求。"

    async def close(self):
        if self.client:
            await self.client.close()


# ============================================================
# 交互式运行
# ============================================================

async def main():
    host = MCPHost()
    await host.connect()

    print("\n" + "=" * 50)
    print("MCP 主机就绪，输入问题或 quit 退出")
    print("=" * 50)

    # 预设几个示例
    examples = [
        "帮我算一下 123 + 456 等于多少？",
        "北京今天天气怎么样？",
        "深圳的天气呢？",
    ]

    for q in examples:
        print(f"\n你: {q}")
        answer = await host.chat(q)
        print(f"Agent: {answer}")

    # 交互循环
    while True:
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break

        answer = await host.chat(user_input)
        print(f"Agent: {answer}")

    await host.close()


if __name__ == "__main__":
    asyncio.run(main())
