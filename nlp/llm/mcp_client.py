# -*- coding: utf-8 -*-
"""
MCP 客户端 v2
============

通过 stdio 协议与 MCP Server 通信

MCP（模型上下文协议）= 让 LLM 按一套统一协议调用外部工具的规范。原本每个工具一套接法，
MCP 把它标准化成固定的两步：list_tools（列出有哪些工具）→ call_tool（调用某个工具）。
stdio（标准输入输出）= 其中一种通信方式：把 MCP Server 起成一个**子进程**，
用它的 stdin/stdout 管道收发消息 —— 所以 server 是本机进程，不是 HTTP 服务。

**事实核对（读之前先知道）**：本文件今天**没有活的调用方**。
唯二 import MCPClientSync 的地方是 core/filesystem_service.py 与 core/sqlite_mcp_service.py，
但都在 `if not skip_mcp:` 分支内；而 skip_mcp 默认 True，全仓库无人传 False
（sqlite 那个 npm 包已下架，filesystem 的未核实）。所以这是**保留能力，不是现役路径** ——
别按"MCP 正在跑"的预期去读上面那两个服务类，它们实际走的是原生降级分支。

实现分两层：MCPClient（异步原生，直接照 MCP 官方 SDK 的异步 API 写）+ MCPClientSync
（把异步包成同步，方便同步代码调用，代价是 _get_loop 里那一堆事件循环兼容处理）。
"""

import os
import json
import logging
import subprocess
import asyncio
from typing import Dict, Any, List, Optional
# mcp 是 Anthropic 官方的 Python SDK；ClientSession 是"客户端这一侧的会话对象"
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)


class MCPClient:
    """MCP 客户端（支持 stdio 协议）"""

    def __init__(self):
        """初始化"""
        self.sessions: Dict[str, ClientSession] = {}
        # 保持 context manager 引用，防止被 GC
        # stdio_client 给的是"异步上下文管理器"，不能只取它 __aenter__ 的返回值就丢掉管理器：
        # 丢掉后对象被垃圾回收，连带把子进程和连接一起关掉 —— 所以必须自己留个引用。
        self._cm_stack: Dict[str, Any] = {}
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
            command: 启动命令（即"用哪个程序当 MCP Server"）
            args: 命令参数（如 npx 要拉哪个包、传哪些目录）
            env: 环境变量（叠加在继承的进程环境之上）
        """
        try:
            logger.info(f"连接 MCP Server: {server_name}")

            # 合并环境变量：先复制当前进程环境再叠加，因为子进程默认要靠 PATH 等才能找到命令
            server_env = os.environ.copy()
            if env:
                server_env.update(env)

            # 创建 server parameters
            # command + args 就是"怎么拉起这个子进程"，等价于在命令行敲 command 后跟 args
            server_params = StdioServerParameters(
                command=command,
                args=args,
                env=server_env
            )

            # 连接 server（mcp >= 1.2 使用 async context manager）
            # 这里手工展开 async with：因为连接要跨越整个对象生命周期，
            # 不能在一个 with 块里进出（with 一退出子进程就没了），所以手动 __aenter__/__aexit__
            cm = stdio_client(server_params)
            transport = await cm.__aenter__()
            self._cm_stack[server_name] = cm  # 保持引用，防止连接被关闭
            read, write = transport

            # initialize() 是 MCP 握手：交换协议版本/能力；不握手后面 list/call 都调不动
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
            工具列表（含 name/description/参数结构）—— 通常就是喂给 LLM，
            让它知道"现在有哪些工具可调"；有了它模型才可能发起 Function Calling。
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
            arguments: 参数（dict；形状必须符合 list_tools 给出的参数结构）

        Returns:
            工具执行结果

        注意：真正执行工具的是 server 那侧的子进程，本方法只是**发指令、取回结果**。
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
        """关闭所有连接

        要关两步且顺序不能反：先关 session（协议层告别），再 __aexit__ 释放 transport
        （真正收掉子进程）。只关一半会让子进程变成孤儿留在后台。
        """
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
    """MCP 客户端同步封装（方便在同步代码中使用）

    MCP 官方 SDK 只提供异步 API，而本项目大部分代码是同步的，所以包这一层：
    每个方法都用 run_until_complete 把协程跑完再返回，调用方完全不用写 async/await。
    """

    def __init__(self):
        self.client = MCPClient()
        self.loop = None  # 事件循环：asyncio 的调度器，异步任务都挂在它上面跑

    def _get_loop(self):
        """获取或创建事件循环

        本文件"同步封装存在兼容性问题"的根源就在这：`run_until_complete` 要求
        手上有**一个没有在运行**的循环，但代码可能跑在已有循环的环境里（Jupyter、
        另一个 async 框架），此时再起循环会报 "already running"。三种情况依次兜：
        有运行中的循环 → 用 nest_asyncio 打补丁允许嵌套；有闲置循环 → 直接用；
        都没有 → 新建一个。
        """
        if self.loop is None:
            try:
                # 检查是否已有运行中的事件循环（能拿到说明当前就在 async 上下文里）
                self.loop = asyncio.get_running_loop()
            except RuntimeError:
                # 没有运行中的循环，尝试获取或创建新的
                try:
                    self.loop = asyncio.get_event_loop()
                    if self.loop.is_running():
                        # 如果循环正在运行（如在Jupyter中），使用nest_asyncio
                        # nest_asyncio 是第三方补丁包，允许事件循环嵌套执行
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
        """连接 MCP Server（同步）

        run_until_complete 会**阻塞**直到协程结束；下面几个同步方法都是这个套路。
        """
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
