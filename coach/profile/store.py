"""画像持久化(work.md §4.2)。与图共用一个 SQLite 文件,但只建自己的表。

这一层只做存取,不含算法:掌握度怎么算在 bkt.py,间隔怎么推在 sm2.py(P3)。
"""

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from coach import config
from coach.knowledge import schema

DDL = """
CREATE TABLE IF NOT EXISTS learner_mastery (
    learner_id TEXT, kp_id TEXT,
    p_known REAL, updated_at REAL,
    PRIMARY KEY (learner_id, kp_id)
);

CREATE TABLE IF NOT EXISTS learner_memory (          -- SM-2 状态
    learner_id TEXT, kp_id TEXT,
    ease REAL DEFAULT 2.5,             -- EF
    interval_days REAL DEFAULT 0,      -- I
    reps INTEGER DEFAULT 0,
    lapses INTEGER DEFAULT 0,
    last_review REAL, due_at REAL,
    PRIMARY KEY (learner_id, kp_id)
);

CREATE TABLE IF NOT EXISTS learner_error (
    learner_id TEXT, kp_id TEXT, error_type TEXT,
    count INTEGER DEFAULT 0,
    PRIMARY KEY (learner_id, kp_id, error_type)
);

CREATE TABLE IF NOT EXISTS learner_profile (
    learner_id TEXT PRIMARY KEY,
    goal_kp_id TEXT, daily_minutes INTEGER DEFAULT 30,
    preference TEXT, created_at REAL
);

-- ★ 根因解释缓存:由 planner_worker 异步生成(要调 LLM),
--   GET /gap 直接读这里,保证入口层不碰 LLM
CREATE TABLE IF NOT EXISTS gap_explanations (
    learner_id TEXT, kp_id TEXT,
    explanation TEXT, generated_at REAL,
    PRIMARY KEY (learner_id, kp_id)
);

CREATE TABLE IF NOT EXISTS answers (
    answer_id TEXT PRIMARY KEY, learner_id TEXT, problem_id TEXT,
    kp_ids TEXT, correct INTEGER, answer_text TEXT,
    elapsed_ms INTEGER, ts REAL
);
CREATE INDEX IF NOT EXISTS idx_answers_learner ON answers(learner_id, ts);

-- ★ 幂等表
CREATE TABLE IF NOT EXISTS processed_events (
    event_id TEXT PRIMARY KEY, ts REAL
);
CREATE INDEX IF NOT EXISTS idx_memory_due ON learner_memory(learner_id, due_at);
"""


class ProfileStore:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.conn.executescript(DDL)
        self.conn.commit()

    @classmethod
    def open(
        cls,
        path: Optional[Union[str, Path]] = None,
        check_same_thread: bool = True,
    ) -> "ProfileStore":
        conn = schema.connect(path, check_same_thread=check_same_thread)
        schema.init_schema(conn)  # 图与画像同库,建表互不干扰
        return cls(conn)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "ProfileStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---------------- 掌握度 ----------------

    def get_mastery(self, learner_id: str, kp_id: str) -> float:
        row = self.conn.execute(
            "SELECT p_known FROM learner_mastery WHERE learner_id = ? AND kp_id = ?",
            (learner_id, kp_id),
        ).fetchone()
        return row["p_known"] if row else config.BKT_P_INIT

    def get_mastery_map(
        self, learner_id: str, kp_ids: Optional[Sequence[str]] = None
    ) -> Dict[str, float]:
        if kp_ids is None:
            rows = self.conn.execute(
                "SELECT kp_id, p_known FROM learner_mastery WHERE learner_id = ?",
                (learner_id,),
            ).fetchall()
            return {r["kp_id"]: r["p_known"] for r in rows}

        ids = list(kp_ids)
        if not ids:
            return {}
        # 一次批量查,避免 N+1(§5.5 的 M 就靠这个)
        placeholders = ",".join("?" for _ in ids)
        rows = self.conn.execute(
            f"SELECT kp_id, p_known FROM learner_mastery "
            f"WHERE learner_id = ? AND kp_id IN ({placeholders})",
            (learner_id, *ids),
        ).fetchall()
        known = {r["kp_id"]: r["p_known"] for r in rows}
        return {kp: known.get(kp, config.BKT_P_INIT) for kp in ids}

    def set_mastery(
        self, learner_id: str, kp_id: str, p_known: float, ts: Optional[float] = None
    ) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO learner_mastery (learner_id, kp_id, p_known, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(learner_id, kp_id) DO UPDATE SET
                    p_known = excluded.p_known, updated_at = excluded.updated_at
                """,
                (learner_id, kp_id, p_known, time.time() if ts is None else ts),
            )

    # ---------------- 间隔重复状态(P3 用)----------------

    def get_memory(self, learner_id: str, kp_id: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM learner_memory WHERE learner_id = ? AND kp_id = ?",
            (learner_id, kp_id),
        ).fetchone()
        return dict(row) if row else None

    def set_memory(
        self,
        learner_id: str,
        kp_id: str,
        ease: float,
        interval_days: float,
        reps: int,
        lapses: int,
        last_review: Optional[float],
        due_at: Optional[float],
    ) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO learner_memory
                    (learner_id, kp_id, ease, interval_days, reps, lapses, last_review, due_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(learner_id, kp_id) DO UPDATE SET
                    ease = excluded.ease,
                    interval_days = excluded.interval_days,
                    reps = excluded.reps,
                    lapses = excluded.lapses,
                    last_review = excluded.last_review,
                    due_at = excluded.due_at
                """,
                (learner_id, kp_id, ease, interval_days, reps, lapses, last_review, due_at),
            )

    def due_reviews(self, learner_id: str, now: Optional[float] = None) -> List[str]:
        """到期待复习的知识点,最急的排前面。"""
        rows = self.conn.execute(
            "SELECT kp_id FROM learner_memory "
            "WHERE learner_id = ? AND due_at IS NOT NULL AND due_at <= ? "
            "ORDER BY due_at",
            (learner_id, time.time() if now is None else now),
        ).fetchall()
        return [r["kp_id"] for r in rows]

    # ---------------- 易错模式 ----------------

    def bump_error(self, learner_id: str, kp_id: str, error_type: str) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO learner_error (learner_id, kp_id, error_type, count)
                VALUES (?, ?, ?, 1)
                ON CONFLICT(learner_id, kp_id, error_type) DO UPDATE SET
                    count = count + 1
                """,
                (learner_id, kp_id, error_type),
            )

    def list_errors(self, learner_id: str) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT kp_id, error_type, count FROM learner_error "
            "WHERE learner_id = ? ORDER BY count DESC",
            (learner_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ---------------- 答题记录 ----------------

    def record_answer(
        self,
        answer_id: str,
        learner_id: str,
        problem_id: str,
        kp_ids: Sequence[str],
        correct: bool,
        answer_text: str,
        elapsed_ms: int = 0,
        ts: Optional[float] = None,
    ) -> bool:
        """写答题记录。answer_id 已存在则返回 False——上层据此跳过 BKT 更新。

        这是**幂等的第二道闸**:processed_events 挡住重复消费,
        这里挡住「恢复重放」时把同一次答题二次计入画像。
        """
        cur = self.conn.execute(
            """
            INSERT OR IGNORE INTO answers
                (answer_id, learner_id, problem_id, kp_ids, correct, answer_text, elapsed_ms, ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                answer_id,
                learner_id,
                problem_id,
                json.dumps(list(kp_ids), ensure_ascii=False),
                1 if correct else 0,
                answer_text,
                elapsed_ms,
                time.time() if ts is None else ts,
            ),
        )
        self.conn.commit()
        return cur.rowcount > 0

    def has_answer(self, answer_id: str) -> bool:
        return (
            self.conn.execute(
                "SELECT 1 FROM answers WHERE answer_id = ?", (answer_id,)
            ).fetchone()
            is not None
        )

    def list_answers(self, learner_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM answers WHERE learner_id = ? ORDER BY ts DESC LIMIT ?",
            (learner_id, limit),
        ).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            item["kp_ids"] = json.loads(item["kp_ids"]) if item["kp_ids"] else []
            item["correct"] = bool(item["correct"])
            out.append(item)
        return out

    # ---------------- 档案 ----------------

    def upsert_profile(
        self,
        learner_id: str,
        goal_kp_id: Optional[str] = None,
        daily_minutes: int = 30,
        preference: Optional[str] = None,
    ) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO learner_profile (learner_id, goal_kp_id, daily_minutes, preference, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(learner_id) DO UPDATE SET
                    goal_kp_id = excluded.goal_kp_id,
                    daily_minutes = excluded.daily_minutes,
                    preference = excluded.preference
                """,
                (learner_id, goal_kp_id, daily_minutes, preference, time.time()),
            )

    def get_profile(self, learner_id: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM learner_profile WHERE learner_id = ?", (learner_id,)
        ).fetchone()
        return dict(row) if row else None

    # ---------------- 根因解释缓存 ----------------

    def set_gap_explanation(
        self, learner_id: str, kp_id: str, explanation: str, ts: Optional[float] = None
    ) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO gap_explanations (learner_id, kp_id, explanation, generated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(learner_id, kp_id) DO UPDATE SET
                    explanation = excluded.explanation,
                    generated_at = excluded.generated_at
                """,
                (learner_id, kp_id, explanation, time.time() if ts is None else ts),
            )

    def get_gap_explanation(self, learner_id: str, kp_id: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT explanation, generated_at FROM gap_explanations "
            "WHERE learner_id = ? AND kp_id = ?",
            (learner_id, kp_id),
        ).fetchone()
        return dict(row) if row else None

    # ---------------- 幂等 ----------------

    def mark_processed(self, event_id: str, ts: Optional[float] = None) -> bool:
        """标记事件已处理。返回 True 表示本次是新标记;False 表示早就处理过。"""
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO processed_events (event_id, ts) VALUES (?, ?)",
            (event_id, time.time() if ts is None else ts),
        )
        self.conn.commit()
        return cur.rowcount > 0

    def is_processed(self, event_id: str) -> bool:
        return (
            self.conn.execute(
                "SELECT 1 FROM processed_events WHERE event_id = ?", (event_id,)
            ).fetchone()
            is not None
        )
