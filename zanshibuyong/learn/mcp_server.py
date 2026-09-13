"""
MCP 服务端 —— 把 CRM 工具暴露为 MCP 协议

这个文件独立运行，不依赖 Agent。它通过 stdio 和客户端通信。

运行方式：
  python learn/mcp_server.py

或者用 uv：
  uv run python learn/mcp_server.py
"""

import json
import asyncio
import logging

from mcp.server import Server
from mcp.types import Tool, TextContent
from mcp.server.stdio import stdio_server

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-crm-server")

# ---- 模拟数据库 ----
_users = {
    "user_001": {"name": "张三", "vip": "gold", "orders": 25},
    "user_002": {"name": "李四", "vip": "silver", "orders": 10},
}
_tickets: dict = {}
_ticket_counter = 0

# ---- 创建 MCP Server ----
app = Server("crm-server")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """告诉客户端：我有哪些工具"""
    return [
        Tool(
            name="get_user_info",
            description="查询用户信息，根据用户ID获取姓名、VIP等级、订单数",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "用户ID，如 user_001"}
                },
                "required": ["user_id"],
            },
        ),
        Tool(
            name="create_ticket",
            description="创建客服工单",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "用户ID"},
                    "issue": {"type": "string", "description": "问题简述"},
                    "priority": {
                        "type": "string",
                        "enum": ["low", "normal", "high", "urgent"],
                        "description": "优先级",
                    },
                },
                "required": ["user_id", "issue"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """客户端调用工具时执行"""
    logger.info(f"收到调用: {name}({arguments})")

    if name == "get_user_info":
        user = _users.get(arguments["user_id"])
        if user:
            text = json.dumps(user, ensure_ascii=False)
        else:
            text = f"未找到用户 {arguments['user_id']}"
        return [TextContent(type="text", text=text)]

    elif name == "create_ticket":
        global _ticket_counter
        _ticket_counter += 1
        tid = f"T{_ticket_counter:05d}"
        _tickets[tid] = {
            "ticket_id": tid,
            "user_id": arguments["user_id"],
            "issue": arguments["issue"],
            "priority": arguments.get("priority", "normal"),
        }
        text = f"工单已创建: {tid} - {arguments['issue']}"
        return [TextContent(type="text", text=text)]

    else:
        return [TextContent(type="text", text=f"未知工具: {name}")]


async def main():
    """启动 MCP 服务端（stdio 模式）"""
    logger.info("CRM MCP Server 启动中...")
    async with stdio_server() as (read, write):
        await app.run(
            read,
            write,
            app.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
