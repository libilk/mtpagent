# -*- coding: utf-8 -*-
"""
电商售后 CRM 服务
================

读写 `database/ecommerce.db` 的 `tickets`（工单）和 `customers`（会员）两张表，
供客服 Agent 的 create_ticket / query_ticket / query_user_info 三个工具调用。

**它替代了 `core/crm_mock.py`** —— 那个实现把用户和工单硬编码在内存、落盘到
`data/crm/*.json`，是纯粹为了跑通流程的假数据：工单号从 T00001 顺序排，
重启机器也不影响，但数据永远就那么几条，改了也不会反映到任何真实业务库。

换成本实现之后：
- 工单落在真实的 SQLite 表里，可以被 SQL 查询、被其他 Agent 读到
- 会员等级从 `customers` 表读，售后权益能按等级区分

**接口是刻意和 MockCRM 保持一致的**（get_user_info / create_ticket / get_ticket /
update_ticket_status / get_user_tickets），这样 Agent 那边只需要换一行实例化。
"""

import os
import sqlite3
import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# 项目根目录：本文件在 core/ 下，根目录是上一级
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_DB_PATH = os.path.join(PROJECT_ROOT, 'database', 'ecommerce.db')

# 工单状态的合法取值，和建库脚本里的默认值保持一致
TICKET_STATUS_PENDING = "待处理"
TICKET_STATUS_PROCESSING = "处理中"
TICKET_STATUS_RESOLVED = "已解决"
TICKET_STATUS_CLOSED = "已关闭"


class EcommerceCRM:
    """电商售后 CRM（SQLite 直连）"""

    def __init__(self, db_path: str = None):
        """
        初始化

        Args:
            db_path: SQLite 数据库路径，默认用项目根目录下的 database/ecommerce.db
        """
        self.db_path = db_path or DEFAULT_DB_PATH

        if not os.path.exists(self.db_path):
            # 不抛异常，只警告 —— 让上层 Agent 能优雅地返回错误信息给用户，
            # 而不是整个服务起不来。重建命令写在提示里，方便排查。
            logger.warning(
                f"数据库不存在: {self.db_path}，"
                f"请先运行 python tools/scripts/init_ecommerce_db.py"
            )
        else:
            logger.info(f"EcommerceCRM 就绪，数据库: {self.db_path}")

    # ----------------------------------------------------------
    # 内部工具
    # ----------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """
        建立连接。

        每次操作都新开连接、用完就关：SQLite 的连接不能跨线程共享，
        而 Agent 可能在多个线程里被调用，共享连接会踩坑。
        开连接的开销很小，不值得为此做连接池。
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row   # 让查询结果能按列名取值，比下标可读
        return conn

    def _next_ticket_id(self, conn: sqlite3.Connection) -> str:
        """
        生成工单号，格式 TK + 日期 + 3位序号，如 TK20260916001。

        取当天已有工单的最大序号 +1。用 MAX 而不是 COUNT：
        如果中间删过工单，COUNT 会算出重复的号。
        """
        today = datetime.now().strftime("%Y%m%d")
        prefix = f"TK{today}"

        row = conn.execute(
            "SELECT MAX(ticket_id) AS max_id FROM tickets WHERE ticket_id LIKE ?",
            (f"{prefix}%",)
        ).fetchone()

        max_id = row["max_id"] if row else None
        if max_id:
            # 取末三位当序号；解析失败就从头开始，不让一个脏数据卡死整个流程
            try:
                seq = int(max_id[len(prefix):]) + 1
            except ValueError:
                seq = 1
        else:
            seq = 1

        return f"{prefix}{seq:03d}"

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict:
        """sqlite3.Row 转普通 dict，方便直接当 JSON 返回给 LLM。"""
        return dict(row) if row else {}

    # ----------------------------------------------------------
    # 对外接口（与 MockCRM 保持一致）
    # ----------------------------------------------------------

    def get_user_info(self, user_id: str) -> Optional[Dict]:
        """
        查询会员信息。

        Returns:
            会员信息字典（含 member_level 会员等级、total_spent 历史消费），
            查不到返回 None。
        """
        if not user_id:
            return None

        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM customers WHERE user_id = ?", (user_id,)
            ).fetchone()

        if not row:
            logger.info(f"[CRM] 未找到会员: {user_id}")
            return None

        user = self._row_to_dict(row)
        # 顺带把该会员的历史工单也带上，客服回答时能直接说"您之前有 N 个工单"
        user["tickets"] = self.get_user_tickets(user_id)
        return user

    def create_ticket(
        self,
        user_id: Optional[str],
        issue: str,
        priority: str = "normal",
        description: str = "",
        order_id: Optional[str] = None,
        category: Optional[str] = None,
        sentiment: Optional[str] = None,
    ) -> Dict:
        """
        创建售后工单。

        Args:
            user_id: 会员ID，可为空（匿名咨询）
            issue: 问题描述（同时写入 content 字段）
            priority: 优先级，必须是 low/normal/high/urgent 之一
            description: 详细描述，默认与 issue 相同
            order_id: 关联订单号，可空
            category: 工单分类（退货/换货/物流/发票/投诉/咨询）
            sentiment: 情绪倾向（positive/neutral/negative）

        Returns:
            新建的工单信息字典
        """
        # 优先级兜底：LLM 偶尔会给出枚举外的值，落到 normal 比写进库更好
        if priority not in ("low", "normal", "high", "urgent"):
            logger.warning(f"[CRM] 非法优先级 {priority!r}，回退为 normal")
            priority = "normal"

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with self._connect() as conn:
            ticket_id = self._next_ticket_id(conn)
            conn.execute(
                """
                INSERT INTO tickets
                    (ticket_id, user_id, order_id, category, priority,
                     sentiment, content, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ticket_id,
                    user_id or "anonymous",
                    order_id,
                    category or "咨询",
                    priority,
                    sentiment,
                    description or issue,
                    TICKET_STATUS_PENDING,
                    now,
                    now,
                ),
            )
            conn.commit()

            row = conn.execute(
                "SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,)
            ).fetchone()

        logger.info(f"[CRM] 已创建工单 {ticket_id}（{category or '咨询'} / {priority}）")
        return self._row_to_dict(row)

    def get_ticket(self, ticket_id: str) -> Optional[Dict]:
        """按工单号查询，查不到返回 None。"""
        if not ticket_id:
            return None

        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,)
            ).fetchone()

        return self._row_to_dict(row) if row else None

    def update_ticket_status(self, ticket_id: str, status: str) -> bool:
        """
        更新工单状态。

        Returns:
            True 表示确实改动了某一行；工单不存在或状态本来就相同都返回 False。
        """
        if not ticket_id:
            return False

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE tickets SET status = ?, updated_at = ? WHERE ticket_id = ?",
                (status, now, ticket_id),
            )
            conn.commit()
            changed = cursor.rowcount > 0

        if changed:
            logger.info(f"[CRM] 工单 {ticket_id} 状态更新为 {status}")
        return changed

    def get_user_tickets(self, user_id: str) -> List[Dict]:
        """查某个会员的全部工单，按创建时间倒序（最新的在前）。"""
        if not user_id:
            return []

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM tickets WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,),
            ).fetchall()

        return [self._row_to_dict(r) for r in rows]
