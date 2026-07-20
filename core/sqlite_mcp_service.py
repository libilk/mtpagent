# -*- coding: utf-8 -*-
"""
SQLite MCP 服务
===============

封装 SQLite MCP Server，提供数据库查询能力
"""

import os
import logging
from typing import Dict, Any, List
from llm.mcp_client import MCPClientSync

logger = logging.getLogger(__name__)


class SQLiteMCPService:
    """SQLite MCP 服务"""

    def __init__(self, database_path: str = None):
        """
        初始化

        Args:
            database_path: SQLite数据库文件路径
        """
        self.database_path = database_path or os.path.join(
            os.getcwd(), "database", "chinook.db"
        )
        self.mcp_client = MCPClientSync()
        self.server_name = "sqlite"
        self.connected = False
        self._connect()

    def _connect(self):
        """连接 SQLite MCP Server"""
        try:
            logger.info(f"连接 SQLite MCP，数据库: {self.database_path}")

            # 连接 MCP Server
            self.mcp_client.connect_server(
                server_name=self.server_name,
                command="npx",
                args=[
                    "-y",
                    "@modelcontextprotocol/server-sqlite",
                    self.database_path
                ]
            )
            self.connected = True
            logger.info("✓ SQLite MCP 连接成功")

        except Exception as e:
            logger.error(f"连接 SQLite MCP 失败: {e}")
            self.connected = False

    def query(self, sql: str) -> Dict[str, Any]:
        """
        执行SQL查询

        Args:
            sql: SQL查询语句

        Returns:
            查询结果
        """
        if not self.connected:
            return self._query_fallback(sql)

        try:
            logger.info(f"执行SQL查询: {sql}")

            result = self.mcp_client.call_tool(
                server_name=self.server_name,
                tool_name="query",
                arguments={"sql": sql}
            )

            return self._parse_result(result)

        except Exception as e:
            logger.error(f"SQL查询失败: {e}")
            return self._query_fallback(sql)

    def list_tables(self) -> List[str]:
        """
        列出所有表

        Returns:
            表名列表
        """
        sql = "SELECT name FROM sqlite_master WHERE type='table'"
        result = self.query(sql)

        if result.get("success"):
            data = result.get("data", [])
            return [row[0] for row in data]
        return []

    def describe_table(self, table_name: str) -> Dict[str, Any]:
        """
        获取表结构

        Args:
            table_name: 表名

        Returns:
            表结构信息
        """
        sql = f"PRAGMA table_info({table_name})"
        return self.query(sql)

    def _parse_result(self, result: Any) -> Dict[str, Any]:
        """解析MCP返回结果"""
        try:
            if hasattr(result, 'content'):
                content = result.content
                if isinstance(content, list) and len(content) > 0:
                    text = content[0].text if hasattr(content[0], 'text') else str(content[0])

                    # 解析JSON结果
                    import json
                    data = json.loads(text)

                    return {
                        "success": True,
                        "data": data.get("rows", []),
                        "columns": data.get("columns", [])
                    }

            return {"success": True, "data": str(result)}

        except Exception as e:
            logger.error(f"解析结果失败: {e}")
            return {"success": False, "error": str(e)}

    def _query_fallback(self, sql: str) -> Dict[str, Any]:
        """降级方案：直接使用 Python sqlite3"""
        try:
            import sqlite3

            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            cursor.execute(sql)

            # 获取列名
            columns = [desc[0] for desc in cursor.description] if cursor.description else []

            # 获取数据
            rows = cursor.fetchall()

            conn.close()

            return {
                "success": True,
                "data": rows,
                "columns": columns
            }

        except Exception as e:
            logger.error(f"降级查询失败: {e}")
            return {"success": False, "error": str(e)}

    def close(self):
        """关闭连接"""
        if self.connected:
            self.mcp_client.close()
