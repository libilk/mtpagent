# -*- coding: utf-8 -*-
"""
长期记忆 · 逐条核对候选记忆
==========================

把 `extract_memory_events.py` 沉淀出来的候选记忆展示出来（**同时显示引文和它的原始
qa_log**），人工逐条判断后 approve / reject / 修正，写回记忆库。

**为什么必须有人工环节：** 总结是 LLM 做的。规则校验只能挡掉格式问题
（引文对不上原文、类别不在图谱、含身份信息），**"这个总结说得对不对"只能靠人看**。
而且这里是长期记忆 —— 错的东西一旦 `verified`，以后每一轮检索都会把它当成"历史经验"推荐出去。

## 用法

    # 1. 列出来看（连同原始问答一起显示，供对照）
    .venv/Scripts/python.exe tools/scripts/review_memory_events.py --list

    # 2. 按决策文件批量应用
    .venv/Scripts/python.exe tools/scripts/review_memory_events.py --apply <决策文件.json>

决策文件格式：

    {
      "approve": [1, 2],
      "reject":  [{"id": 3, "reason": "总结把运费规则说成了退款时限"}],
      "edit":    [{"id": 4, "issue_type": "SHIPPING_FEE", "reason": "类型选错了"}],
      "add":     [{"category_name": "无线耳机", "issue_type": "QUALITY_ISSUE",
                   "conclusion": "...", "summary": "...",
                   "source_qa_log_id": 12, "source_quote": "...", "note": "人工补充"}]
    }

`edit` 可改的字段：`category_name` / `issue_type` / `conclusion` / `summary`（只改传了的那些）。
`add` 直接以 `verified` 入库（人工补充的记忆不需要再核一遍自己）。

## 幂等性 / 破坏性

- `--list` **只读**，随便跑，看的是 `status='proposed'` 的那批。
- `--apply` **永久改库且没有 undo**：状态一改，`--list` 就再也看不到它了。
  唯一的回退是重跑 `init_memory_db.py` 全量重建 —— 但那会把**别的核对结果也一起清掉**。
  所以规矩是：**决策文件先落盘，再 --apply**。

它是记忆流水线的第 ③ 步：① 建库词表 → ② 抽取（写 proposed）→ ③ 本脚本核对（改 verified）。
"""

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime
from typing import Dict, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DB_PATH = os.path.join(PROJECT_ROOT, 'database', 'memory.db')
QA_DB = os.path.join(PROJECT_ROOT, 'database', 'qa_logs.db')

if sys.platform == 'win32' and hasattr(sys.stdout, 'buffer'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

REVIEWER = "manual-review"


def fetch_pending(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    """取出待核对的候选记忆，并连带把它的原始 qa_log 一起拉出来。

    为什么一定要带原文：核对的唯一依据就是"这条总结与那一轮问答对得上吗"。
    只显示总结，人工无从判断 —— 那就成了走过场。
    """
    return list(conn.execute("""
        SELECT m.id, m.category_name, m.product_name, m.issue_type,
               m.conclusion, m.summary, m.source_quote, m.note,
               m.source_qa_log_id,
               q.query  AS qa_query,
               q.answer AS qa_answer
        FROM main.memory_events m
        LEFT JOIN qa.qa_logs q ON q.id = m.source_qa_log_id
        WHERE m.status = 'proposed'
        ORDER BY m.id
    """))


def cmd_list(conn: sqlite3.Connection) -> None:
    rows = fetch_pending(conn)
    if not rows:
        print("\n没有待核对的候选记忆。\n")
        print("  （下一步跑 extract_memory_events.py 从 qa_logs.db 沉淀候选）")
        return

    print(f"\n待核对 {len(rows)} 条（带原始问答，供逐条判断）\n" + "=" * 74)
    for r in rows:
        prod = f" / 商品 {r['product_name']}" if r["product_name"] else ""
        print(f"\n  [{r['id']:>2}] {r['category_name']}{prod}  ▸  {r['issue_type']}")
        print(f"       结论: {r['conclusion']}")
        print(f"       总结: {r['summary']}")
        print(f"       引文: {r['source_quote']}")
        if r["note"]:
            print(f"       ⚠ 备注: {r['note']}")
        print(f"       ── 原始 qa#{r['source_qa_log_id']} ──")
        q = (r["qa_query"] or "").replace("\n", " ")
        a = (r["qa_answer"] or "").replace("\n", " ")
        print(f"       问: {q[:110]}{'...' if len(q) > 110 else ''}")
        print(f"       答: {a[:110]}{'...' if len(a) > 110 else ''}")
    print("\n" + "=" * 74 + "\n")


def cmd_apply(conn: sqlite3.Connection, decisions: Dict) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    approved = rejected = edited = added = 0

    for mid in decisions.get("approve", []):
        conn.execute(
            "UPDATE main.memory_events SET status='verified', reviewed_by=?, reviewed_at=? WHERE id=?",
            (REVIEWER, now, mid))
        approved += 1

    for item in decisions.get("reject", []):
        # COALESCE(note,'') 把 NULL 当空串 —— 否则 note 本来是 NULL 时整段拼接也会变 NULL，驳回理由就丢了
        conn.execute(
            "UPDATE main.memory_events SET status='rejected', reviewed_by=?, reviewed_at=?, "
            "note = COALESCE(note,'') || ' | 驳回: ' || ? WHERE id=?",
            (REVIEWER, now, item.get("reason", ""), item["id"]))
        rejected += 1

    for item in decisions.get("edit", []):
        # 只更新传了的字段：核对时经常只改一处（比如类型选错），不该要求把整条重写
        editable = ("category_name", "issue_type", "conclusion", "summary")
        sets, params = [], []
        for field in editable:
            if field in item:
                sets.append(f"{field}=?")
                params.append(item[field])
        if not sets:
            continue
        sets += ["reviewed_by=?", "reviewed_at=?", "note = COALESCE(note,'') || ' | 修正: ' || ?"]
        params += [REVIEWER, now, item.get("reason", ""), item["id"]]
        conn.execute(f"UPDATE main.memory_events SET {', '.join(sets)} WHERE id=?", params)
        edited += 1

    for item in decisions.get("add", []):
        conn.execute(
            "INSERT INTO main.memory_events "
            "(category_name, product_name, issue_type, conclusion, summary, "
            " status, source_qa_log_id, source_quote, batch_id, reviewed_by, reviewed_at, note) "
            "VALUES (?,?,?,?,?, 'verified', ?,?,?,?,?,?)",
            (item["category_name"], item.get("product_name"), item["issue_type"],
             item["conclusion"], item["summary"], item.get("source_qa_log_id"),
             item.get("source_quote", ""), "manual", REVIEWER, now,
             item.get("note", "审核阶段人工补充")))
        added += 1

    conn.commit()

    print(f"\n核对完成：批准 {approved} | 驳回 {rejected} | 修正 {edited} | 补充 {added}")
    verified = conn.execute("SELECT COUNT(*) FROM main.v_memory").fetchone()[0]
    print(f"当前已生效的记忆：{verified} 条")
    if added:
        print("\n  提示：人工补充的记忆若以后要重建，记得把它的来源 qa_log 也留在库里")


def main() -> None:
    ap = argparse.ArgumentParser(description="逐条核对候选记忆")
    ap.add_argument("--list", action="store_true", help="只读：列出 status='proposed' 的候选")
    ap.add_argument("--apply", type=str, default=None, help="按决策文件批量应用（永久改库，无 undo）")
    args = ap.parse_args()

    if not args.list and not args.apply:
        ap.print_help()
        return

    if not os.path.exists(QA_DB):
        print(f"✗ 缺少 {QA_DB}（核对要看原始问答，没有它无从对照）")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("ATTACH DATABASE ? AS qa", (QA_DB,))
    try:
        if args.list:
            cmd_list(conn)
            return

        with open(args.apply, encoding='utf-8') as f:
            decisions = json.load(f)
        cmd_apply(conn, decisions)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
