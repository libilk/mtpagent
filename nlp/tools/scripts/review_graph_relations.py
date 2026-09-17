# -*- coding: utf-8 -*-
"""
知识图谱 · 逐条核对抽取结果
==========================

把 `extract_graph_relations.py` 抽出来的候选关系展示出来（带原文引文和上下文），
人工逐条判断后 approve / reject / 修正，写回图谱。

**为什么必须有人工环节：** 抽取是 LLM 做的，它会犯错 —— 实测中就出现了
"把一般商品的时限安到家电头上""把运费规则当成退货排除"这类错误。
规则校验只能挡掉格式问题（引文对不上、节点不在词表里），**语义对不对只能靠人看**。

## 用法

    # 1. 列出来看（带原文上下文）
    .venv/Scripts/python.exe tools/scripts/review_graph_relations.py --list

    # 2. 按决策文件批量应用
    .venv/Scripts/python.exe tools/scripts/review_graph_relations.py --apply <决策文件.json>

决策文件格式：

    {
      "approve": [34, 35],
      "reject":  [{"id": 25, "reason": "与 23 完全重复"}],
      "edit":    [{"id": 36, "condition": null, "reason": "原文是无条件排除"}],
      "add":     [{"subject": "家电", "predicate": "APPLIES_TO",
                   "object": "家电质量问题退货（7日）", "condition": "QUALITY_ISSUE",
                   "doc_id": "quality_issue_policy", "quote": "...", "note": "人工补充"}]
    }

`add` 里如果 object 是个还不存在的实体名，脚本会**自动创建**，
但会打印提醒 —— 记得把新节点同步回 `init_knowledge_graph.py` 的种子，
否则重新建库时它会消失。
"""

import os
import sys
import json
import sqlite3
import argparse
from datetime import datetime
from typing import Dict, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DB_PATH = os.path.join(PROJECT_ROOT, 'database', 'knowledge_graph.db')
DOCS_DIR = os.path.join(PROJECT_ROOT, 'data', 'knowledge')

if sys.platform == 'win32' and hasattr(sys.stdout, 'buffer'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

REVIEWER = "manual-review"


def fetch_pending(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    return list(conn.execute("""
        SELECT r.id, s.name AS subj, r.predicate, o.name AS obj,
               r.condition_code, r.source_doc_id, r.source_quote, r.note
        FROM relations r
        JOIN entities s ON s.id = r.subject_id
        JOIN entities o ON o.id = r.object_id
        WHERE r.status = 'proposed'
        ORDER BY r.source_doc_id, r.id
    """))


def cmd_list(conn: sqlite3.Connection) -> None:
    rows = fetch_pending(conn)
    if not rows:
        print("\n没有待核对的关系。\n")
        return

    print(f"\n待核对 {len(rows)} 条（带原文引文，供逐条判断）\n" + "=" * 74)
    cur_doc = None
    for r in rows:
        if r["source_doc_id"] != cur_doc:
            cur_doc = r["source_doc_id"]
            print(f"\n── {cur_doc} ──")
        cond = r["condition_code"] or "NULL(无条件)"
        print(f"\n  [{r['id']:>2}] {r['subj']}  --{r['predicate']}-->  {r['obj']}   [{cond}]")
        print(f"       引文: {r['source_quote']}")
        if r["note"]:
            print(f"       备注: {r['note']}")
    print("\n" + "=" * 74 + "\n")


def resolve_entity(conn: sqlite3.Connection, name: str, created: List[str]) -> int:
    """按名字取实体 id；没有就建（会记进 created 提醒同步种子）。"""
    row = conn.execute("SELECT id FROM entities WHERE name = ?", (name,)).fetchone()
    if row:
        return row[0]

    etype = "policy" if ("退货" in name or "换货" in name or "运费" in name
                         or "无理由" in name or "发货" in name or "赔付" in name
                         or "退款" in name or "次数" in name or "补贴" in name) else "category"
    cur = conn.execute(
        "INSERT INTO entities (name, entity_type, description) VALUES (?,?,?)",
        (name, etype, "审核阶段人工新增"),
    )
    created.append(f"{name} ({etype})")
    return cur.lastrowid


def cmd_apply(conn: sqlite3.Connection, decisions: Dict) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    created: List[str] = []
    approved = rejected = edited = added = 0

    for rid in decisions.get("approve", []):
        conn.execute(
            "UPDATE relations SET status='verified', reviewed_by=?, reviewed_at=? WHERE id=?",
            (REVIEWER, now, rid))
        approved += 1

    for item in decisions.get("reject", []):
        conn.execute(
            "UPDATE relations SET status='rejected', reviewed_by=?, reviewed_at=?, "
            "note = COALESCE(note,'') || ' | 驳回: ' || ? WHERE id=?",
            (REVIEWER, now, item.get("reason", ""), item["id"]))
        rejected += 1

    for item in decisions.get("edit", []):
        conn.execute(
            "UPDATE relations SET condition_code=?, reviewed_by=?, reviewed_at=?, "
            "note = COALESCE(note,'') || ' | 修正: ' || ? WHERE id=?",
            (item.get("condition"), REVIEWER, now, item.get("reason", ""), item["id"]))
        edited += 1

    for item in decisions.get("add", []):
        subj_id = resolve_entity(conn, item["subject"], created)
        obj_id = resolve_entity(conn, item["object"], created)
        conn.execute(
            "INSERT INTO relations (subject_id, predicate, object_id, condition_code, "
            "  status, source_doc_id, source_quote, batch_id, reviewed_by, reviewed_at, note) "
            "VALUES (?,?,?,?, 'verified', ?,?,?,?,?,?)",
            (subj_id, item["predicate"], obj_id, item.get("condition"),
             item.get("doc_id"), item.get("quote", ""), "manual",
             REVIEWER, now, item.get("note", "审核阶段人工补充")))
        added += 1

    conn.commit()

    print(f"\n核对完成：批准 {approved} | 驳回 {rejected} | 修正 {edited} | 补充 {added}")

    if created:
        print(f"\n⚠ 自动创建了 {len(created)} 个新实体：")
        for c in created:
            print(f"    - {c}")
        print("  记得同步回 init_knowledge_graph.py 的种子，否则重新建库会丢。")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="列出待核对的关系")
    ap.add_argument("--apply", help="按决策文件批量应用")
    args = ap.parse_args()

    if not os.path.exists(DB_PATH):
        print(f"图谱不存在: {DB_PATH}，请先跑 init_knowledge_graph.py")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        if args.apply:
            with open(args.apply, encoding='utf-8') as f:
                decisions = json.load(f)
            cmd_apply(conn, decisions)
        else:
            cmd_list(conn)

        verified = conn.execute(
            "SELECT COUNT(*) FROM relations WHERE status='verified'").fetchone()[0]
        proposed = conn.execute(
            "SELECT COUNT(*) FROM relations WHERE status='proposed'").fetchone()[0]
        rejected_n = conn.execute(
            "SELECT COUNT(*) FROM relations WHERE status='rejected'").fetchone()[0]
        print(f"\n图谱现状：已核实 {verified} | 待核对 {proposed} | 已驳回 {rejected_n}\n")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
