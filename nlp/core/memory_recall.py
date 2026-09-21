# -*- coding: utf-8 -*-
"""
长期记忆 · 查询层
================

读 `database/memory.db` 的**只读视图 `v_memory`**，把历史案例记忆取出来。

**记忆的对象是商品 / 政策，不是用户** —— 这里回答的是"这类问题以前是怎么处理的"，
不是"这个用户以前做过什么"。为什么必须是后者，见 study.md §7.1.1。

## 为什么只读视图

`v_memory` 的定义里已经固化了 `WHERE status = 'verified'`，
所以**应用层不可能漏写状态过滤**把未核对的候选记忆取出来用 —— 与知识图谱
（`v_rel`）同一手法。本模块只读，不提供任何写入函数。

## 多域隔离 = 按类别分区

每条记忆归属一个 `category_name`，取自知识图谱的类别体系。
`recall_by_category()` 按类别过滤，就是"多域隔离"在查询侧的实现 ——
问「耳机」不会把「生鲜」的历史案例捞出来。

**Phase 1 只做 SQL 层过滤。** Phase 2 会在此基础上叠加向量检索 + RRF 融合，
那时候这个函数仍然是"过滤"那一环（先按域收窄，再做语义排序）。

## 用法

    from core.memory_recall import recall_by_category, count_memories
    rows = recall_by_category("无线耳机", issue_type="QUALITY_ISSUE")
"""

import os
import sqlite3
from typing import Any, Dict, List, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DB_PATH = os.path.join(PROJECT_ROOT, 'database', 'memory.db')


def _connect(db_path: Optional[str] = None) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path or DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def load_issue_types(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """读出问题类型词表（code + label）。

    用途：把用户话里的说法映射到封闭枚举上；以及给调用方呈现人话标签。
    """
    conn = _connect(db_path)
    try:
        return [
            {"code": r["code"], "label": r["label"]}
            for r in conn.execute("SELECT code, label FROM issue_types ORDER BY code")
        ]
    finally:
        conn.close()


def recall_by_category(
    category_name: str,
    issue_type: Optional[str] = None,
    limit: int = 5,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """按类别（+ 可选的问题类型）取历史案例记忆。

    Args:
        category_name: 知识图谱里的类别名，如「无线耳机」。**必填** ——
            不传就退化成"捞所有类别的记忆"，那等于没有多域隔离
        issue_type: 问题类型 code；不传则该类别下所有类型都返回
        limit: 最多返回几条（默认 5，够模型参考即可，不要塞满上下文）

    Returns:
        记忆列表，每项含 category_name / product_name / issue_type /
        conclusion / summary / importance / access_count
        —— **不含 source_qa_log_id 与 source_quote**：血缘是给人工核对用的，
        不应进入给模型看的上下文（避免"用户原话"被反复带进 prompt）
    """
    if not category_name:
        return []

    sql = """
        SELECT category_name, product_name, issue_type, conclusion, summary,
               importance, access_count
        FROM v_memory
        WHERE category_name = ?
    """
    params: List[Any] = [category_name]
    if issue_type:
        sql += " AND issue_type = ?"
        params.append(issue_type)
    # 重要度高的优先，其次是被命中过的 —— 治理（Phase 3）会给这两个字段赋值
    sql += " ORDER BY importance DESC, access_count DESC, id DESC LIMIT ?"
    params.append(limit)

    conn = _connect(db_path)
    try:
        return [dict(r) for r in conn.execute(sql, params)]
    finally:
        conn.close()


def count_memories(db_path: Optional[str] = None) -> Dict[str, int]:
    """统计各类状态下的记忆条数（含未生效的，供运维脚本与核对时看进度）"""
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT status, COUNT(*) AS n FROM memory_events GROUP BY status"
        ).fetchall()
        stats = {r["status"]: r["n"] for r in rows}
        for key in ("proposed", "verified", "rejected", "superseded"):
            stats.setdefault(key, 0)
        return stats
    finally:
        conn.close()


def format_memories(rows: List[Dict[str, Any]]) -> str:
    """把记忆列表渲染成给人/给模型看的文本。

    Phase 2 的"叙事化注入"会在这里扩展成三段式；Phase 1 先给一个朴素版本，
    让查询层可独立测试。
    """
    if not rows:
        return ""
    lines = []
    for r in rows:
        prod = f"（{r['product_name']}）" if r.get("product_name") else ""
        lines.append(f"- [{r['category_name']}{prod}] {r['conclusion']}")
    return "\n".join(lines)
