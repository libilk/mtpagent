# -*- coding: utf-8 -*-
"""
Filesystem MCP 服务
==================

封装 Filesystem MCP Server，提供文件系统访问能力

**事实核对（读本文件前先知道，否则会按错误的预期读）：今天这个类不建立任何 MCP 连接。**
skip_mcp 默认 True，而唯一的构造处 agents/document_agent/agent.py 用的是
`FilesystemService()`（一个参数都没传），所以 _connect() 从不执行，self.connected 恒为 False。
结果就是：list_directory / read_file 每次都走 `*_fallback` 的 Python 原生分支，
文件系统访问靠的是 os / open，跟 MCP 没关系。

所以本文件的名字与现状有落差 —— MCP（模型上下文协议）那条路是**保留能力**，不是现役路径。
真要走 MCP 得显式传 skip_mcp=False，且要求本机 npx 能拉到
@modelcontextprotocol/server-filesystem。仓库里没有记录这个 filesystem 包是否可获取
（README 与 study.md 只对 sqlite 那个包给出"已下架"的结论），所以别把"下架"直接套到本文件。

读数技巧：把每个方法里 `if not self.connected:` 理解为**恒真**，后面的 MCP 分支就是死代码；
真正干活的函数都在文件末尾的 `_*_fallback` 里。
"""

import os
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class FilesystemService:
    """Filesystem 服务（现状：纯 Python 原生文件操作，见文件头事实核对）"""

    def __init__(self, allowed_directory: str = None, skip_mcp: bool = True):
        """
        初始化

        skip_mcp 默认 True 意味着**默认路径与 MCP 无关**：不建连接，直接走 fallback。
        想启用 MCP 必须显式传 False；即便传了，_connect() 失败也会自动落回 fallback
        （self.connected 保持 False），所以调用方不需要为 MCP 存不存在做分支。

        Args:
            allowed_directory: 允许访问的目录路径；不传则用 <cwd>/data
                （注意用的是 os.getcwd()，即进程启动目录，随启动位置而变）
            skip_mcp: 是否跳过 MCP 连接（默认 True，因 MCP 同步封装存在兼容性问题）
        """
        self.allowed_directory = allowed_directory or os.path.join(
            os.getcwd(), "data"
        )
        self.mcp_client = None
        self.server_name = "filesystem"
        # 单一状态位：MCP 是否可用。所有 public 方法都以它决定走 MCP 还是走 fallback，
        # 因此它就是"本实例实际在用什么"的唯一开关。
        self.connected = False

        if not skip_mcp:
            self._connect()

        if not self.connected:
            logger.info(f"Filesystem 使用 Python 原生模式，目录: {self.allowed_directory}")

    def _connect(self):
        """连接 Filesystem MCP Server（当前默认路径不会执行到这里）

        连法是标准 MCP 做法：用 npx 拉起一个子进程当 MCP Server，通过
        stdio（标准输入输出，即子进程管道）传消息 —— 所以它依赖本机能联网拉包、
        也依赖 npx 存在。任一环节失败都会被下面的 except 兜住，降级为原生文件操作。
        """
        try:
            from llm.mcp_client import MCPClientSync
            self.mcp_client = MCPClientSync()
            logger.info(f"连接 Filesystem MCP，允许目录: {self.allowed_directory}")

            self.mcp_client.connect_server(
                server_name=self.server_name,
                command="npx",
                args=[
                    "-y",
                    "@modelcontextprotocol/server-filesystem",
                    self.allowed_directory
                ]
            )
            self.connected = True
            logger.info("✓ Filesystem MCP 连接成功")

        except Exception as e:
            logger.warning(f"Filesystem MCP 连接失败（{e}），降级为原生文件操作")
            self.mcp_client = None
            self.connected = False

    def list_directory(self, path: str = "") -> List[str]:
        """
        列出目录内容

        两条实现路径同名同签名：connected 时走 MCP 工具调用，否则走 os.listdir。
        注意 MCP 分支内部出错也会兜回 fallback，所以这个方法**几乎不会失败**，
        拿到的可能只是空列表 —— 别把"空列表"当成"目录确实为空"。

        Args:
            path: 相对路径（相对于 allowed_directory）

        Returns:
            文件列表
        """
        if not self.connected:
            return self._list_directory_fallback(path)

        try:
            full_path = os.path.join(self.allowed_directory, path)
            logger.info(f"列出目录: {full_path}")

            result = self.mcp_client.call_tool(
                server_name=self.server_name,
                tool_name="list_directory",
                arguments={"path": full_path}
            )

            return self._parse_list_result(result)

        except Exception as e:
            logger.error(f"列出目录失败: {e}")
            return self._list_directory_fallback(path)

    def read_file(self, path: str) -> str:
        """
        读取文件内容

        同 list_directory：失败静默降级。文件不存在、编码不是 UTF-8、MCP 调用出错，
        最终都表现为返回空字符串 —— 调用方无法从返回值区分失败原因，需要原因得看日志。

        Args:
            path: 相对路径

        Returns:
            文件内容
        """
        if not self.connected:
            return self._read_file_fallback(path)

        try:
            full_path = os.path.join(self.allowed_directory, path)
            logger.info(f"读取文件: {full_path}")

            result = self.mcp_client.call_tool(
                server_name=self.server_name,
                tool_name="read_file",
                arguments={"path": full_path}
            )

            return self._parse_read_result(result)

        except Exception as e:
            logger.error(f"读取文件失败: {e}")
            return self._read_file_fallback(path)

    def _parse_list_result(self, result: Any) -> List[str]:
        """解析列出目录结果"""
        try:
            if hasattr(result, 'content'):
                content = result.content
                if isinstance(content, list) and len(content) > 0:
                    text = content[0].text if hasattr(content[0], 'text') else str(content[0])
                    # 解析文件列表
                    import json
                    data = json.loads(text)
                    return data if isinstance(data, list) else []

            return []

        except Exception as e:
            logger.error(f"解析列表结果失败: {e}")
            return []

    def _parse_read_result(self, result: Any) -> str:
        """解析读取文件结果"""
        try:
            if hasattr(result, 'content'):
                content = result.content
                if isinstance(content, list) and len(content) > 0:
                    return content[0].text if hasattr(content[0], 'text') else str(content[0])

            return str(result)

        except Exception as e:
            logger.error(f"解析读取结果失败: {e}")
            return ""

    def _list_directory_fallback(self, path: str) -> List[str]:
        """降级方案：直接使用 Python os 模块

        现状下这是真正生效的那条。返回的是**目录项名字**（不含 path 前缀），
        且不区分文件和子目录；也不做任何路径越界检查 —— 传 "../" 能读到
        allowed_directory 之外，安全性完全依赖调用方自觉。
        """
        try:
            full_path = os.path.join(self.allowed_directory, path)
            if os.path.exists(full_path):
                return os.listdir(full_path)
            return []
        except Exception as e:
            logger.error(f"降级列出目录失败: {e}")
            return []

    def _read_file_fallback(self, path: str) -> str:
        """降级方案：直接使用 Python 读取文件

        按 UTF-8 严格解码（不传 errors 参数），遇到 GBK 编码的老文件会抛
        UnicodeDecodeError，被 except 兜住后返回空串 —— 表现为"文件读出来是空的"。
        """
        try:
            full_path = os.path.join(self.allowed_directory, path)
            if os.path.exists(full_path):
                with open(full_path, 'r', encoding='utf-8') as f:
                    return f.read()
            return ""
        except Exception as e:
            logger.error(f"降级读取文件失败: {e}")
            return ""

    def close(self):
        """关闭连接

        默认路径（skip_mcp=True）下 connected 恒为 False，所以这是个空操作 ——
        调用它不会报错，也不会释放任何东西，因为本来就没建立过连接。
        """
        if self.connected and self.mcp_client:
            self.mcp_client.close()
