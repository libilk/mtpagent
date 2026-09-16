# -*- coding: utf-8 -*-
"""
Filesystem MCP 服务
==================

封装 Filesystem MCP Server，提供文件系统访问能力
"""

import os
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class FilesystemService:
    """Filesystem 服务"""

    def __init__(self, allowed_directory: str = None, skip_mcp: bool = True):
        """
        初始化

        Args:
            allowed_directory: 允许访问的目录路径
            skip_mcp: 是否跳过 MCP 连接（默认 True，因 MCP 同步封装存在兼容性问题）
        """
        self.allowed_directory = allowed_directory or os.path.join(
            os.getcwd(), "data"
        )
        self.mcp_client = None
        self.server_name = "filesystem"
        self.connected = False

        if not skip_mcp:
            self._connect()

        if not self.connected:
            logger.info(f"Filesystem 使用 Python 原生模式，目录: {self.allowed_directory}")

    def _connect(self):
        """连接 Filesystem MCP Server"""
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
        """降级方案：直接使用 Python os 模块"""
        try:
            full_path = os.path.join(self.allowed_directory, path)
            if os.path.exists(full_path):
                return os.listdir(full_path)
            return []
        except Exception as e:
            logger.error(f"降级列出目录失败: {e}")
            return []

    def _read_file_fallback(self, path: str) -> str:
        """降级方案：直接使用 Python 读取文件"""
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
        """关闭连接"""
        if self.connected and self.mcp_client:
            self.mcp_client.close()
