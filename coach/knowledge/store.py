"""知识图谱的存取与遍历(work.md §5.1 递归 CTE)。

只封装 SQLite,不含业务判断:
- 环检测归 governance(P0.5)
- 根因定位归 queries(P2)

所有写入走 upsert,反复执行结果一致(建图可重跑)。
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from coach import config
from coach.domain.models import EDGE_TYPES, Edge, KnowledgePoint, Problem
from coach.knowledge import schema

_ANCESTORS_SQL = """
WITH RECURSIVE prereqs(kp_id, depth) AS (
    SELECT :x, 0
    UNION
    SELECT e.from_id, p.depth + 1
    FROM edges e JOIN prereqs p ON e.to_id = p.kp_id
    WHERE e.type = :edge_type AND p.depth < :max_depth
)
SELECT kp_id, MIN(depth) AS depth
FROM prereqs WHERE kp_id != :x
GROUP BY kp_id ORDER BY depth
"""

_DESCENDANTS_SQL = """
WITH RECURSIVE deps(kp_id, depth) AS (
    SELECT :x, 0
    UNION
    SELECT e.to_id, p.depth + 1
    FROM edges e JOIN deps p ON e.from_id = p.kp_id
    WHERE e.type = :edge_type AND p.depth < :max_depth
)
SELECT kp_id, MIN(depth) AS depth
FROM deps WHERE kp_id != :x
GROUP BY kp_id ORDER BY depth
"""


def _row_to_concept(row: sqlite3.Row) -> KnowledgePoint:
    return KnowledgePoint(
        id=row["id"],
        name=row["name"],
        subject=row["subject"],
        difficulty=row["difficulty"] if row["difficulty"] is not None else 3.0,
        description=row["description"],
        created_at=row["created_at"] if row["created_at"] is not None else 0.0,
    )


def _row_to_edge(row: sqlite3.Row) -> Edge:
    return Edge(
        from_id=row["from_id"],
        to_id=row["to_id"],
        type=row["type"],
        weight=row["weight"] if row["weight"] is not None else 1.0,
        confidence=row["confidence"] if row["confidence"] is not None else 1.0,
        source=row["source"],
    )


def _row_to_problem(row: sqlite3.Row) -> Problem:
    return Problem(
        id=row["id"],
        title=row["title"],
        source=row["source"],
        difficulty=row["difficulty"],
        judge_type=row["judge_type"] or "exact_output",
        test_cases=json.loads(row["test_cases"]) if row["test_cases"] else [],
        kp_ids=json.loads(row["kp_ids"]) if row["kp_ids"] else [],
    )


class KnowledgeStore:
    """SQLite 知识图谱的读写门面。"""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    @classmethod
    def open(
        cls,
        path: Optional[Union[str, Path]] = None,
        check_same_thread: bool = True,
    ) -> "KnowledgeStore":
        conn = schema.connect(path, check_same_thread=check_same_thread)
        schema.init_schema(conn)
        return cls(conn)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "KnowledgeStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---------------- 知识点 ----------------

    def upsert_concept(self, kp: KnowledgePoint) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO concepts (id, name, subject, difficulty, description, created_at)
                VALUES (:id, :name, :subject, :difficulty, :description, :created_at)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    subject = excluded.subject,
                    difficulty = excluded.difficulty,
                    description = excluded.description
                """,
                {
                    "id": kp.id,
                    "name": kp.name,
                    "subject": kp.subject,
                    "difficulty": kp.difficulty,
                    "description": kp.description,
                    "created_at": kp.created_at,
                },
            )

    def get_concept(self, kp_id: str) -> Optional[KnowledgePoint]:
        row = self.conn.execute(
            "SELECT * FROM concepts WHERE id = ?", (kp_id,)
        ).fetchone()
        return _row_to_concept(row) if row else None

    def list_concepts(self, subject: Optional[str] = None) -> List[KnowledgePoint]:
        if subject is None:
            rows = self.conn.execute("SELECT * FROM concepts ORDER BY id").fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM concepts WHERE subject = ? ORDER BY id", (subject,)
            ).fetchall()
        return [_row_to_concept(r) for r in rows]

    def has_concept(self, kp_id: str) -> bool:
        return (
            self.conn.execute(
                "SELECT 1 FROM concepts WHERE id = ?", (kp_id,)
            ).fetchone()
            is not None
        )

    def delete_concept(self, kp_id: str) -> None:
        """删知识点,连带删掉与它相关的边(保持图一致)。"""
        with self.conn:
            self.conn.execute(
                "DELETE FROM edges WHERE from_id = ? OR to_id = ?", (kp_id, kp_id)
            )
            self.conn.execute("DELETE FROM concepts WHERE id = ?", (kp_id,))

    # ---------------- 边 ----------------

    def upsert_edge(self, edge: Edge) -> None:
        if edge.type not in EDGE_TYPES:
            raise ValueError(f"未知关系类型:{edge.type},只允许 {EDGE_TYPES}")
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO edges (from_id, to_id, type, weight, confidence, source)
                VALUES (:from_id, :to_id, :type, :weight, :confidence, :source)
                ON CONFLICT(from_id, to_id, type) DO UPDATE SET
                    weight = excluded.weight,
                    confidence = excluded.confidence,
                    source = excluded.source
                """,
                {
                    "from_id": edge.from_id,
                    "to_id": edge.to_id,
                    "type": edge.type,
                    "weight": edge.weight,
                    "confidence": edge.confidence,
                    "source": edge.source,
                },
            )

    def has_edge(self, from_id: str, to_id: str, edge_type: str) -> bool:
        return (
            self.conn.execute(
                "SELECT 1 FROM edges WHERE from_id = ? AND to_id = ? AND type = ?",
                (from_id, to_id, edge_type),
            ).fetchone()
            is not None
        )

    def get_edge(self, from_id: str, to_id: str, edge_type: str) -> Optional[Edge]:
        row = self.conn.execute(
            "SELECT * FROM edges WHERE from_id = ? AND to_id = ? AND type = ?",
            (from_id, to_id, edge_type),
        ).fetchone()
        return _row_to_edge(row) if row else None

    def list_edges(self, edge_type: Optional[str] = None) -> List[Edge]:
        if edge_type is None:
            rows = self.conn.execute(
                "SELECT * FROM edges ORDER BY from_id, to_id, type"
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM edges WHERE type = ? ORDER BY from_id, to_id",
                (edge_type,),
            ).fetchall()
        return [_row_to_edge(r) for r in rows]

    def delete_edge(self, from_id: str, to_id: str, edge_type: str) -> None:
        with self.conn:
            self.conn.execute(
                "DELETE FROM edges WHERE from_id = ? AND to_id = ? AND type = ?",
                (from_id, to_id, edge_type),
            )

    def prerequisite_pairs(self) -> List[Tuple[str, str]]:
        """所有 PREREQUISITE 边,形如 [(from, to)]。给 §5.2 环检测用。"""
        rows = self.conn.execute(
            "SELECT from_id, to_id FROM edges WHERE type = 'PREREQUISITE'"
        ).fetchall()
        return [(r["from_id"], r["to_id"]) for r in rows]

    def prerequisite_adjacency(self) -> Dict[str, List[str]]:
        """{知识点: [它的直接前置]}。**一次查完**。

        给需要"自己看后一步前置"的调用方用(`next_to_learn` 就在循环里做这件事)。
        在循环里逐个 `ancestors(kp, 1)` 是 N+1 —— 22 个点看不出来,
        但 P5 评测 30 个学生 × 每步都在跑它。
        """
        adjacency: Dict[str, List[str]] = {}
        for from_id, to_id in self.prerequisite_pairs():
            adjacency.setdefault(to_id, []).append(from_id)
        return adjacency

    # ---------------- 题目 ----------------

    def upsert_problem(self, problem: Problem) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO problems (id, title, source, difficulty, judge_type, test_cases, kp_ids)
                VALUES (:id, :title, :source, :difficulty, :judge_type, :test_cases, :kp_ids)
                ON CONFLICT(id) DO UPDATE SET
                    title = excluded.title,
                    source = excluded.source,
                    difficulty = excluded.difficulty,
                    judge_type = excluded.judge_type,
                    test_cases = excluded.test_cases,
                    kp_ids = excluded.kp_ids
                """,
                {
                    "id": problem.id,
                    "title": problem.title,
                    "source": problem.source,
                    "difficulty": problem.difficulty,
                    "judge_type": problem.judge_type,
                    "test_cases": json.dumps(problem.test_cases, ensure_ascii=False),
                    "kp_ids": json.dumps(problem.kp_ids, ensure_ascii=False),
                },
            )

    def get_problem(self, problem_id: str) -> Optional[Problem]:
        row = self.conn.execute(
            "SELECT * FROM problems WHERE id = ?", (problem_id,)
        ).fetchone()
        return _row_to_problem(row) if row else None

    def list_problems(self) -> List[Problem]:
        rows = self.conn.execute("SELECT * FROM problems ORDER BY id").fetchall()
        return [_row_to_problem(r) for r in rows]

    def kp_ids_with_problems(self) -> set:
        """有题目可做的知识点集合。

        ★ 计划必须依赖它:推荐一个**没有题**的知识点,学生无从练起,那一步就空转了。
        P5 实测:不加这道过滤时,计划 360 步里约 300 步(82%)是白费的。

        注意 `problems.kp_ids` 是 JSON 列,**建不了索引** —— 现在 13 道题直接全扫,
        上千道题时应该拆一张 `problem_kp` 关联表。
        """
        covered = set()
        for problem in self.list_problems():
            covered.update(problem.kp_ids)
        return covered

    def delete_problem(self, problem_id: str) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM problems WHERE id = ?", (problem_id,))

    # ---------------- 遍历(§5.1)----------------

    def ancestors(
        self,
        kp_id: str,
        depth: int = config.MAX_PREREQ_DEPTH,
        edge_type: str = "PREREQUISITE",
    ) -> Dict[str, int]:
        """前置闭包:{kp_id: 最短深度},不含自身。

        只沿 from_id → to_id 的反方向(e.to_id = 当前节点)走,即"我要学 X,先要会谁"。
        """
        rows = self.conn.execute(
            _ANCESTORS_SQL,
            {"x": kp_id, "edge_type": edge_type, "max_depth": depth},
        ).fetchall()
        return {r["kp_id"]: r["depth"] for r in rows}

    def descendants(
        self,
        kp_id: str,
        depth: int = config.MAX_PREREQ_DEPTH,
        edge_type: str = "PREREQUISITE",
    ) -> Dict[str, int]:
        """后继闭包:{kp_id: 最短深度},不含自身。即"学会 X 之后能学谁"。"""
        rows = self.conn.execute(
            _DESCENDANTS_SQL,
            {"x": kp_id, "edge_type": edge_type, "max_depth": depth},
        ).fetchall()
        return {r["kp_id"]: r["depth"] for r in rows}
