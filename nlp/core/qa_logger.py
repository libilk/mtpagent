# -*- coding: utf-8 -*-
"""
问答日志记录器
==============

使用 SQLite 持久化存储每次用户问答记录，包括：
- 用户问题、系统答案
- 使用的 Agent、质量评分、执行模式
- 耗时、引用源文档
- 时间戳

用于上线后收集用户真实使用数据，分析系统效果。
"""

import os
import json
import time
import sqlite3
import logging
import threading
from typing import Optional, List, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# 默认数据库路径
DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "database", "qa_logs.db"
)


class QALogger:
    """
    问答日志记录器

    线程安全的 SQLite 写入，支持分页查询、统计、导出。
    """

    def __init__(self, db_path: str = None):
        """
        初始化日志记录器

        Args:
            db_path: SQLite 数据库文件路径，默认 database/qa_logs.db
        """
        self.db_path = db_path or DEFAULT_DB_PATH

        # 确保目录存在
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        # 线程锁（SQLite 单写）
        self._lock = threading.Lock()

        # 初始化表结构
        self._init_db()

        logger.info(f"[QALogger] 初始化完成，数据库: {self.db_path}")

    def _get_conn(self) -> sqlite3.Connection:
        """获取数据库连接（每次新建，线程安全）"""
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")  # 提高并发性能
        return conn

    def _init_db(self):
        """初始化数据库表结构"""
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS qa_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        thread_id TEXT DEFAULT 'default',
                        query TEXT NOT NULL,
                        answer TEXT,
                        mode TEXT,
                        quality_score REAL DEFAULT 0,
                        success INTEGER DEFAULT 1,
                        duration_ms INTEGER DEFAULT 0,
                        agents TEXT,
                        sources TEXT,
                        iterations INTEGER DEFAULT 0,
                        has_files INTEGER DEFAULT 0,
                        has_images INTEGER DEFAULT 0,
                        error TEXT
                    )
                """)

                # 创建索引加速查询
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_qa_timestamp
                    ON qa_logs(timestamp DESC)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_qa_thread
                    ON qa_logs(thread_id)
                """)

                conn.commit()
            finally:
                conn.close()

    def log(
        self,
        query: str,
        answer: str,
        mode: str = "",
        quality_score: float = 0.0,
        success: bool = True,
        duration_ms: int = 0,
        thread_id: str = "default",
        agent_results: List[Dict] = None,
        sources: List[Dict] = None,
        iterations: int = 0,
        has_files: bool = False,
        has_images: bool = False,
        error: str = None,
    ):
        """
        记录一条问答日志

        Args:
            query: 用户问题
            answer: 系统答案
            mode: 执行模式 (simple / planning / error)
            quality_score: 质量评分
            success: 是否成功
            duration_ms: 总耗时（毫秒）
            thread_id: 会话 ID
            agent_results: Agent 结果列表
            sources: 引用源文档列表
            iterations: 迭代次数
            has_files: 是否附带文件
            has_images: 是否附带图片
            error: 错误信息
        """
        try:
            # 提取参与的 Agent 名称列表
            agents_str = ""
            if agent_results:
                agent_names = [r.get("agent", "") for r in agent_results if r.get("agent")]
                agents_str = json.dumps(agent_names, ensure_ascii=False)

            # 序列化 sources
            sources_str = ""
            if sources:
                sources_str = json.dumps(sources, ensure_ascii=False)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with self._lock:
                conn = self._get_conn()
                try:
                    conn.execute(
                        """
                        INSERT INTO qa_logs
                        (timestamp, thread_id, query, answer, mode, quality_score,
                         success, duration_ms, agents, sources, iterations,
                         has_files, has_images, error)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            timestamp,
                            thread_id,
                            query,
                            answer[:5000] if answer else "",  # 限制答案长度
                            mode,
                            quality_score,
                            1 if success else 0,
                            duration_ms,
                            agents_str,
                            sources_str,
                            iterations,
                            1 if has_files else 0,
                            1 if has_images else 0,
                            error,
                        ),
                    )
                    conn.commit()
                    logger.debug(f"[QALogger] 记录成功: {query[:50]}...")
                finally:
                    conn.close()

        except Exception as e:
            # 日志记录失败不应影响主流程
            logger.warning(f"[QALogger] 记录失败: {e}")

    def query_logs(
        self,
        page: int = 1,
        page_size: int = 20,
        keyword: str = None,
        mode: str = None,
        success_only: bool = None,
        start_date: str = None,
        end_date: str = None,
    ) -> Dict[str, Any]:
        """
        分页查询问答日志

        Args:
            page: 页码（从 1 开始）
            page_size: 每页条数
            keyword: 关键词搜索（搜索 query 和 answer）
            mode: 筛选模式 (simple / planning)
            success_only: 仅成功的记录
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)

        Returns:
            {"total": int, "page": int, "page_size": int, "data": list}
        """
        conditions = []
        params = []

        if keyword:
            conditions.append("(query LIKE ? OR answer LIKE ?)")
            params.extend([f"%{keyword}%", f"%{keyword}%"])
        if mode:
            conditions.append("mode = ?")
            params.append(mode)
        if success_only is not None:
            conditions.append("success = ?")
            params.append(1 if success_only else 0)
        if start_date:
            conditions.append("timestamp >= ?")
            params.append(f"{start_date} 00:00:00")
        if end_date:
            conditions.append("timestamp <= ?")
            params.append(f"{end_date} 23:59:59")

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        conn = self._get_conn()
        try:
            # 总数
            total = conn.execute(
                f"SELECT COUNT(*) FROM qa_logs WHERE {where_clause}", params
            ).fetchone()[0]

            # 分页数据
            offset = (page - 1) * page_size
            rows = conn.execute(
                f"""
                SELECT * FROM qa_logs
                WHERE {where_clause}
                ORDER BY id DESC
                LIMIT ? OFFSET ?
                """,
                params + [page_size, offset],
            ).fetchall()

            data = [dict(row) for row in rows]

            return {
                "total": total,
                "page": page,
                "page_size": page_size,
                "pages": (total + page_size - 1) // page_size,
                "data": data,
            }
        finally:
            conn.close()

    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计概览

        Returns:
            总数、成功率、平均评分、平均耗时、Agent 使用频次等
        """
        conn = self._get_conn()
        try:
            row = conn.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as success_count,
                    AVG(quality_score) as avg_score,
                    AVG(duration_ms) as avg_duration_ms,
                    SUM(CASE WHEN mode = 'planning' THEN 1 ELSE 0 END) as complex_count,
                    SUM(CASE WHEN mode = 'simple' THEN 1 ELSE 0 END) as simple_count,
                    MIN(timestamp) as first_query,
                    MAX(timestamp) as last_query
                FROM qa_logs
            """).fetchone()

            stats = dict(row)
            stats["success_rate"] = (
                f"{stats['success_count'] / stats['total'] * 100:.1f}%"
                if stats["total"] > 0 else "N/A"
            )
            stats["avg_score"] = round(stats["avg_score"] or 0, 3)
            stats["avg_duration_ms"] = int(stats["avg_duration_ms"] or 0)

            # 今日统计
            today = datetime.now().strftime("%Y-%m-%d")
            today_row = conn.execute(
                "SELECT COUNT(*) as cnt FROM qa_logs WHERE timestamp >= ?",
                (f"{today} 00:00:00",),
            ).fetchone()
            stats["today_count"] = today_row["cnt"]

            return stats
        finally:
            conn.close()

    def export_all(self) -> List[Dict]:
        """导出全部记录（用于 CSV 下载）"""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM qa_logs ORDER BY id DESC"
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()
