"""知识图谱建表 DDL(work.md §4.1)。

约定:
- 图与画像共用一个 SQLite 文件,但**各层只建自己的表**:本文件只建知识层的表,
  画像层的表由 `coach/profile/store.py` 建。
- `init_schema` 幂等,可反复调用。
"""

import sqlite3
from pathlib import Path
from typing import Optional, Union

from coach import config

DDL = """
-- 知识点
CREATE TABLE IF NOT EXISTS concepts (
    id          TEXT PRIMARY KEY,      -- "algo.dp"
    name        TEXT NOT NULL,         -- "动态规划"
    subject     TEXT NOT NULL,         -- "algorithms"
    difficulty  REAL DEFAULT 3.0,      -- 1.0 ~ 5.0
    description TEXT,
    created_at  REAL
);

-- 边:前置关系(from 是 to 的前置)
-- type 列保留、值域只剩 PREREQUISITE。为什么不一并删列:它是递归 CTE 的过滤键、
-- 也是 idx_edges_to 的一部分,删列要动项目最核心且测试最密的代码,收益只是少一列。
-- 这是有意的取舍(见 work.md §11.1 P2#8)。
CREATE TABLE IF NOT EXISTS edges (
    from_id     TEXT NOT NULL,
    to_id       TEXT NOT NULL,
    type        TEXT NOT NULL,         -- 目前只有 PREREQUISITE
    weight      REAL DEFAULT 1.0,
    confidence  REAL DEFAULT 1.0,      -- 抽取置信度
    source      TEXT,                  -- 来源:llm_extract|manual|outline|golden
    PRIMARY KEY (from_id, to_id, type)
);
CREATE INDEX IF NOT EXISTS idx_edges_to ON edges(to_id, type);   -- ★ 反向遍历用

-- 题目
CREATE TABLE IF NOT EXISTS problems (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    source      TEXT,                  -- "leetcode:70"
    difficulty  REAL,
    judge_type  TEXT DEFAULT 'exact_output',
    test_cases  TEXT,                  -- JSON: [{"input":...,"expected":...}]
    kp_ids      TEXT                   -- JSON 数组
);

-- ★ 写入治理:观察 → 提案
CREATE TABLE IF NOT EXISTS observations (
    id          TEXT PRIMARY KEY,
    kind        TEXT,                  -- "edge" | "concept"
    payload     TEXT,                  -- JSON
    source      TEXT,                  -- 从哪抽的
    observed_at REAL
);

CREATE TABLE IF NOT EXISTS proposals (
    id             TEXT PRIMARY KEY,
    observation_id TEXT,
    kind           TEXT,               -- "edge" | "concept"
    payload        TEXT,               -- 待入库的变更(JSON)
    status         TEXT,               -- pending|accepted|rejected
    reason         TEXT,               -- 拒绝原因
    created_at     REAL
);
CREATE INDEX IF NOT EXISTS idx_proposals_status ON proposals(status, created_at);
"""


def connect(
    path: Optional[Union[str, Path]] = None,
    check_same_thread: bool = True,
) -> sqlite3.Connection:
    """打开(必要时创建)SQLite 连接。WAL + 外键 + busy_timeout。"""
    target = Path(path) if path is not None else config.db_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target), check_same_thread=check_same_thread)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """建表(幂等)。"""
    conn.executescript(DDL)
    conn.commit()


def open_db(
    path: Optional[Union[str, Path]] = None,
    check_same_thread: bool = True,
) -> sqlite3.Connection:
    """连接 + 建表,一步到位。

    **新代码请用这个,别直接用 `connect()`** —— `connect` 只连接不建表,
    对着一个全新的库文件直接写会报 "no such table"。这个坑踩过两次
    (demo 和 run_all 各自在全新库上挂过),所以在这里统一掉。
    """
    conn = connect(path, check_same_thread=check_same_thread)
    init_schema(conn)
    return conn
