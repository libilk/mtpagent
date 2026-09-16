# -*- coding: utf-8 -*-
"""
SQLite MCP 服务
===============

封装 SQLite 数据库查询能力。
优先尝试 MCP Server（需 Node.js + npm），连接失败自动降级为 Python sqlite3。
"""

import os
import re
import sqlite3
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class SQLiteMCPService:
    """SQLite 数据库服务（MCP 优先，sqlite3 降级）"""

    # 表名 → 中文描述映射
    #
    # 这个映射会拼进 list_tables 工具的结果里给 LLM 看。
    # 有了中文描述，LLM 在「哪些表跟退款有关」这类语义判断上会准很多，
    # 不用先去读每个表的列名。表名对不上也没关系 —— 下面的
    # _get_table_description() 会退化成「从 DDL 解析列名」。
    #
    # 当前对应 database/ecommerce.db（电商售后演示库）。
    KNOWN_TABLE_DESCRIPTIONS = {
        "customers": "会员表，存储会员等级、注册时间、历史消费总额，用于判断售后权益",
        "orders": "订单主表，存储订单号、下单时间、订单状态、实付金额、收货信息",
        "order_items": "订单明细表，记录每笔订单买了什么商品、数量、单价、小计",
        "logistics": "物流表，存储承运商、运单号、物流状态、预计到达和实际签收时间",
        "refunds": "退款表，记录退款类型（退货退款/仅退款/换货）、金额和处理状态",
        "tickets": "售后工单表，记录客服工单的分类、优先级、情绪、处理状态",
    }

    def __init__(self, database_path: str = None, mcp_server_package: str = None,
                 skip_mcp: bool = True):
        """
        初始化

        Args:
            database_path: SQLite数据库文件路径
            mcp_server_package: MCP Server npm 包名（默认 @modelcontextprotocol/server-sqlite）
            skip_mcp: 是否跳过 MCP 连接尝试（默认 True，因为 npm 包已下架）
        """
        self.database_path = database_path or os.path.join(
            os.getcwd(), "database", "ecommerce.db"
        )
        self.mcp_client = None
        self.server_name = "sqlite"
        self.connected = False  # MCP 是否连接
        self._mcp_server_package = mcp_server_package or "@modelcontextprotocol/server-sqlite"

        if not skip_mcp:
            # 尝试 MCP 连接（需要 Node.js + npm 上可用的包）
            self._try_mcp_connect()

        # 验证 sqlite3 可用
        if not self.connected:
            self._verify_sqlite3()

    def _try_mcp_connect(self):
        """尝试连接 MCP Server，失败则静默降级"""
        try:
            from llm.mcp_client import MCPClientSync
            self.mcp_client = MCPClientSync()
            logger.info(f"尝试连接 SQLite MCP，数据库: {self.database_path}")
            self.mcp_client.connect_server(
                server_name=self.server_name,
                command="npx",
                args=["-y", self._mcp_server_package, self.database_path]
            )
            self.connected = True
            logger.info("✓ SQLite MCP 连接成功")
        except Exception as e:
            logger.warning(f"SQLite MCP 连接失败（{e}），降级为 sqlite3 直连模式")
            self.mcp_client = None
            self.connected = False

    def _verify_sqlite3(self):
        """验证 sqlite3 直连可用"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            conn.close()
            logger.info(f"✓ SQLite3 直连模式就绪: {self.database_path}")
        except Exception as e:
            logger.error(f"SQLite3 直连也失败: {e}，数据库文件可能不存在")

    def query(self, sql: str) -> Dict[str, Any]:
        """
        执行SQL查询（MCP优先，sqlite3降级）

        Args:
            sql: SQL查询语句

        Returns:
            {"success": bool, "data": List[tuple], "columns": List[str]}
        """
        if self.connected and self.mcp_client:
            try:
                result = self.mcp_client.call_tool(
                    server_name=self.server_name,
                    tool_name="query",
                    arguments={"sql": sql}
                )
                return self._parse_result(result)
            except Exception as e:
                logger.warning(f"MCP 查询失败，降级到 sqlite3: {e}")

        return self._query_sqlite3(sql)

    def _query_sqlite3(self, sql: str) -> Dict[str, Any]:
        """使用 Python sqlite3 直接查询"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            cursor.execute(sql)

            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
            conn.close()

            return {"success": True, "data": rows, "columns": columns}

        except Exception as e:
            logger.error(f"sqlite3 查询失败: {e}")
            return {"success": False, "error": str(e)}

    def list_tables(self) -> List[Dict[str, Any]]:
        """
        列出所有表，附带中文描述和关键列名，帮助LLM快速定位目标表。

        Returns:
            包含表名、中文描述、关键列名的列表
        """
        sql = "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        result = self.query(sql)

        tables_info = []
        if result.get("success"):
            data = result.get("data", [])
            for row in data:
                name = row[0]
                ddl = row[1] if row[1] else ""
                cn_desc = self._get_table_description(name, ddl)
                tables_info.append({
                    "table_name": name,
                    "description": cn_desc,
                })
            return tables_info
        return []

    def _get_table_description(self, table_name: str, ddl: str) -> str:
        """
        为表生成中文描述：优先用已知映射，否则从DDL解析列名。

        Args:
            table_name: 表名
            ddl: CREATE TABLE 语句

        Returns:
            中文描述字符串，如 "曲目/歌曲表，列: TrackId, Name, AlbumId..."
        """
        # 1. 从DDL中提取列名
        columns = self._extract_columns_from_ddl(ddl)
        col_str = ", ".join(columns[:8])
        if len(columns) > 8:
            col_str += f" 等共{len(columns)}列"

        # 2. 查已知映射
        key = table_name.lower()
        known = self.KNOWN_TABLE_DESCRIPTIONS.get(key, "")

        # 3. 获取行数
        row_count = self._get_row_count(table_name)
        count_str = f"，约{row_count}行" if row_count is not None else ""

        if known:
            return f"{known}{count_str}。列: [{col_str}]"
        return f"列: [{col_str}]{count_str}"

    @staticmethod
    def _extract_columns_from_ddl(ddl: str) -> List[str]:
        """从 CREATE TABLE DDL 中提取列名列表"""
        match = re.search(r'\((.+)\)', ddl, re.DOTALL)
        if not match:
            return []
        body = match.group(1)

        # 先剥掉 SQL 行注释（-- 到行尾）再解析。
        # SQLite 会把建表语句的原始文本（含注释）原样存进 sqlite_master，
        # 而下面的逻辑是「按逗号切分，取每段第一个词」。带注释的写法如
        #     user_id  TEXT PRIMARY KEY,   -- 会员ID
        #     name     TEXT NOT NULL,      -- 姓名
        # 切出来第二段的第一个词会变成 "--"，列名就丢了。
        body = re.sub(r'--[^\n]*', '', body)

        columns = []
        for part in body.split(','):
            part = part.strip()
            if not part:
                continue
            # 跳过约束行（FOREIGN KEY / PRIMARY KEY / UNIQUE 等）
            upper = part.upper()
            if any(upper.startswith(kw) for kw in
                   ['FOREIGN', 'PRIMARY', 'UNIQUE', 'CHECK', 'CONSTRAINT']):
                continue
            # 第一个 token 就是列名，去掉可能的引号/方括号
            token = part.split()[0]
            token = token.strip('"[]`')
            if token:
                columns.append(token)
        return columns

    def _get_row_count(self, table_name: str):
        """快速获取表行数，失败返回 None"""
        try:
            r = self.query(f"SELECT COUNT(*) FROM [{table_name}]")
            if r.get("success") and r.get("data"):
                return r["data"][0][0]
        except Exception:
            pass
        return None

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

    def close(self):
        """关闭连接"""
        if self.connected and self.mcp_client:
            self.mcp_client.close()
