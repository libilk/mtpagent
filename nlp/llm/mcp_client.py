# -*- coding: utf-8 -*-
"""
MCP 客户端 v2
============

通过 stdio 协议与 MCP Server 通信
"""

import os
import json
import logging
import subprocess
import asyncio
from typing import Dict, Any, List, Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)


class MCPClient:
    """MCP 客户端（支持 stdio 协议）"""

    def __init__(self):
        """初始化"""
        self.sessions: Dict[str, ClientSession] = {}
        self._cm_stack: Dict[str, Any] = {}  # 保持 context manager 引用，防止被 GC
        logger.info("MCPClient 初始化完成")

    async def connect_server(
        self,
        server_name: str,
        command: str,
        args: List[str],
        env: Optional[Dict[str, str]] = None
    ):
        """
        连接 MCP Server

        Args:
            server_name: 服务器名称
            command: 启动命令
            args: 命令参数
            env: 环境变量
        """
        try:
            logger.info(f"连接 MCP Server: {server_name}")

            # 合并环境变量
            server_env = os.environ.copy()
            if env:
                server_env.update(env)

            # 创建 server parameters
            server_params = StdioServerParameters(
                command=command,
                args=args,
                env=server_env
            )

            # 连接 server（mcp >= 1.2 使用 async context manager）
            cm = stdio_client(server_params)
            transport = await cm.__aenter__()
            self._cm_stack[server_name] = cm  # 保持引用，防止连接被关闭
            read, write = transport

            session = ClientSession(read, write)
            await session.initialize()

            self.sessions[server_name] = session
            logger.info(f"✓ {server_name} 连接成功")

        except Exception as e:
            logger.error(f"连接 {server_name} 失败: {e}")
            raise

    async def list_tools(self, server_name: str) -> List[Dict]:
        """
        列出 MCP Server 的所有工具

        Args:
            server_name: 服务器名称

        Returns:
            工具列表
        """
        if server_name not in self.sessions:
            raise ValueError(f"Server {server_name} 未连接")

        session = self.sessions[server_name]
        result = await session.list_tools()
        return result.tools

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Any:
        """
        调用 MCP 工具

        Args:
            server_name: 服务器名称
            tool_name: 工具名称
            arguments: 参数

        Returns:
            工具执行结果
        """
        if server_name not in self.sessions:
            raise ValueError(f"Server {server_name} 未连接")

        try:
            logger.info(f"调用 MCP 工具: {server_name}.{tool_name}({arguments})")

            session = self.sessions[server_name]
            result = await session.call_tool(tool_name, arguments)

            logger.info(f"✓ 工具调用成功")
            return result

        except Exception as e:
            logger.error(f"工具调用失败: {e}")
            raise

    async def close(self):
        """关闭所有连接"""
        for server_name, session in self.sessions.items():
            try:
                await session.close()
                logger.info(f"✓ {server_name} session 已关闭")
            except Exception as e:
                logger.error(f"关闭 {server_name} session 失败: {e}")

        # 关闭 stdio context manager（释放子进程）
        for server_name, cm in self._cm_stack.items():
            try:
                await cm.__aexit__(None, None, None)
                logger.info(f"✓ {server_name} transport 已关闭")
            except Exception as e:
                logger.error(f"关闭 {server_name} transport 失败: {e}")

        self.sessions.clear()
        self._cm_stack.clear()


class MCPClientSync:
    """MCP 客户端同步封装（方便在同步代码中使用）"""

    def __init__(self):
        self.client = MCPClient()
        self.loop = None

    def _get_loop(self):
        """获取或创建事件循环"""
        if self.loop is None:
            try:
                # 检查是否已有运行中的事件循环
                self.loop = asyncio.get_running_loop()
            except RuntimeError:
                # 没有运行中的循环，尝试获取或创建新的
                try:
                    self.loop = asyncio.get_event_loop()
                    if self.loop.is_running():
                        # 如果循环正在运行（如在Jupyter中），使用nest_asyncio
                        import nest_asyncio
                        nest_asyncio.apply()
                except RuntimeError:
                    self.loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(self.loop)
        return self.loop

    def connect_server(
        self,
        server_name: str,
        command: str,
        args: List[str],
        env: Optional[Dict[str, str]] = None
    ):
        """连接 MCP Server（同步）"""
        loop = self._get_loop()
        return loop.run_until_complete(
            self.client.connect_server(server_name, command, args, env)
        )

    def list_tools(self, server_name: str) -> List[Dict]:
        """列出工具（同步）"""
        loop = self._get_loop()
        return loop.run_until_complete(
            self.client.list_tools(server_name)
        )

    def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Any:
        """调用工具（同步）"""
        loop = self._get_loop()
        return loop.run_until_complete(
            self.client.call_tool(server_name, tool_name, arguments)
        )

    def close(self):
        """关闭连接（同步）"""
        loop = self._get_loop()
        return loop.run_until_complete(self.client.close())
