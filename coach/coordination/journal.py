"""append-only 事件日志(work.md §2.1 / §5.6)。

**先记后做**:消息一到就先落 journal,再执行业务。
这样进程在任何时刻挂掉,重启后都能从 journal 捞出「已记未完成」的事件补做。

journal 只增不改业务字段:唯一的变更就是给 done_at 盖时间戳。
"""

import json
import sqlite3
import time
from typing import List, Optional

from coach.events.schema import Event

DDL = """
CREATE TABLE IF NOT EXISTS journal (
    event_id    TEXT PRIMARY KEY,
    type        TEXT NOT NULL,
    trace_id    TEXT,
    learner_id  TEXT,
    payload     TEXT,
    recorded_at REAL NOT NULL,
    done_at     REAL
);
CREATE INDEX IF NOT EXISTS idx_journal_pending ON journal(done_at, recorded_at);
"""


class Journal:
    """基于 SQLite 的 append-only 日志。与图/画像同一个库文件。"""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.conn.executescript(DDL)
        self.conn.commit()

    def record(self, event: Event, recorded_at: Optional[float] = None) -> bool:
        """先记后做。已记录过则原样保留(重放不覆盖),返回 False。"""
        cur = self.conn.execute(
            """
            INSERT OR IGNORE INTO journal
                (event_id, type, trace_id, learner_id, payload, recorded_at, done_at)
            VALUES (:event_id, :type, :trace_id, :learner_id, :payload, :recorded_at, NULL)
            """,
            {
                "event_id": event.event_id,
                "type": event.type,
                "trace_id": event.trace_id,
                "learner_id": event.learner_id,
                "payload": json.dumps(event.payload, ensure_ascii=False),
                "recorded_at": time.time() if recorded_at is None else recorded_at,
            },
        )
        self.conn.commit()
        return cur.rowcount > 0

    def mark_done(self, event_id: str, done_at: Optional[float] = None) -> None:
        self.conn.execute(
            "UPDATE journal SET done_at = ? WHERE event_id = ? AND done_at IS NULL",
            (time.time() if done_at is None else done_at, event_id),
        )
        self.conn.commit()

    def is_recorded(self, event_id: str) -> bool:
        return (
            self.conn.execute(
                "SELECT 1 FROM journal WHERE event_id = ?", (event_id,)
            ).fetchone()
            is not None
        )

    def is_done(self, event_id: str) -> bool:
        row = self.conn.execute(
            "SELECT done_at FROM journal WHERE event_id = ?", (event_id,)
        ).fetchone()
        return bool(row and row["done_at"] is not None)

    def get(self, event_id: str) -> Optional[Event]:
        row = self.conn.execute(
            "SELECT * FROM journal WHERE event_id = ?", (event_id,)
        ).fetchone()
        return self._row_to_event(row) if row else None

    def pending(self) -> List[Event]:
        """「已记未完成」的事件,按记录顺序返回——这就是重启后要补做的清单。"""
        rows = self.conn.execute(
            "SELECT * FROM journal WHERE done_at IS NULL ORDER BY recorded_at, event_id"
        ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def counts(self) -> dict:
        row = self.conn.execute(
            """
            SELECT COUNT(*) AS total,
                   SUM(CASE WHEN done_at IS NULL THEN 1 ELSE 0 END) AS pending
            FROM journal
            """
        ).fetchone()
        total = row["total"] or 0
        pending = row["pending"] or 0
        return {"total": total, "pending": pending, "done": total - pending}

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> Event:
        return Event(
            event_id=row["event_id"],
            type=row["type"],
            ts=row["recorded_at"],
            learner_id=row["learner_id"] or "",
            payload=json.loads(row["payload"]) if row["payload"] else {},
            trace_id=row["trace_id"] or "",
        )
