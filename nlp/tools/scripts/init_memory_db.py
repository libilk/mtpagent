# -*- coding: utf-8 -*-
"""
长期记忆 · 建库与词表种子
========================

建出 `database/memory.db` —— 放**案例记忆**：把一轮售后会话沉淀成
「商品/类别 + 问题类型 + 处理结论」，以后遇到类似问题能检索出来参考。

**记忆的对象是"这类问题该怎么处理"，不是"这个用户做过什么"。**
为什么必须是后者 —— 见 study.md §7.1.1：记用户行为会撞三条合规红线
（不记情绪历史 / 不长期留存投诉原文 / 不用于对用户不利的决策）。
换成商品/政策之后"没人被画像"，技术点却一个不少。

## 本脚本只铺骨架

- 建表：`memory_events` / `issue_types` / `memory_links` + 视图 `v_memory`
- 种下**问题类型词表**（封闭枚举，LLM 之后只能从里面选）

**不负责抽取。** 记忆条目由 `extract_memory_events.py` 产出，且只写
`status='proposed'`，要经 `review_memory_events.py` 人工核对后才生效。

## 破坏性 + 它在流水线里的位置（重跑前必读）

**每次运行都会 `DROP` 掉 1 个视图 + 3 张表重建，库里的东西全部清空** ——
包括已经人工核对通过的 `status='verified'` 记忆。**重跑本脚本等于前面抽取和核对全白做。**

    ① init_memory_db.py         建库 + 词表（本脚本）→ database/memory.db
    ② extract_memory_events.py  读 qa_logs.db，LLM 总结成候选记忆（status='proposed'）
    ③ review_memory_events.py   人工逐条核对，approve 后 status 变 'verified'，才生效

## 三个关键设计（改代码前先读）

### 1. 记忆域不含 `user_id` / `thread_id`

表里只有 `source_qa_log_id`（指向 qa_logs 表的行号）用于血缘追溯。
**诚实说明**：通过 qa_logs 仍可回溯到会话 —— 防护不靠"删字段"，靠
「**不进检索、不参与任何决策**」（见 memory_events 表的注释）。

### 2. 问题类型必须外置成封闭枚举

「耳机坏了想退」里的"质量问题"不是一个自由文本标签，而是一个 `code`
（`QUALITY_ISSUE`）加一组关键词。这样问题类型是**可校验的枚举**，
LLM 只能从里面选、不能自己发明新类型 —— 与知识图谱的 `condition_atoms` 同一手法。

### 3. `status='verified'` 固化进视图 `v_memory`

查询层只读视图，应用层不可能漏写 `WHERE status='verified'`。
`rejected` / `superseded` 都**保留不删** —— 前者用于打回重复出现的同一条幻觉，
后者是治理（合并/遗忘）的结果，删了就无从复查。

## 用法

    .venv/Scripts/python.exe tools/scripts/init_memory_db.py
"""

import os
import sys
import json
import sqlite3

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DB_PATH = os.path.join(PROJECT_ROOT, 'database', 'memory.db')

if sys.platform == 'win32' and hasattr(sys.stdout, 'buffer'):
    import codecs
    # 只在直接执行时重绑 stdout —— 被 import 时（回归测试要读 SCHEMA_SQL 建临时库）
    # 绝不能动全局 stdout，否则会把调用方的输出流弄坏
    if __name__ == '__main__':
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')


# ============================================================
# 建表
# ============================================================

SCHEMA_SQL = """
-- 问题类型词表：封闭枚举，抽取时只能从里面选
CREATE TABLE issue_types (
    code     TEXT PRIMARY KEY,
    label    TEXT NOT NULL,              -- 人话标签，如「质量问题退换」
    patterns TEXT NOT NULL               -- JSON 数组，关键词
);

-- 案例记忆：一轮售后会话沉淀成一条
CREATE TABLE memory_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,

    -- ---- 记忆对象：商品 / 政策（不是人）----
    category_name TEXT NOT NULL,         -- 必须在知识图谱 entities 里真实存在（抽取时校验）
    product_name  TEXT,                  -- 原始商品名；纯政策咨询可空
    issue_type    TEXT NOT NULL REFERENCES issue_types(code),
    conclusion    TEXT NOT NULL,         -- 处理结论（一句话）
    summary       TEXT NOT NULL,         -- 事件总结（两三句）

    -- ---- 治理用（Phase 3 的相似合并 / 遗忘判据）----
    importance       REAL NOT NULL DEFAULT 0.5,   -- 重要度 0~1，初值由规则算，不调 LLM
    access_count     INTEGER NOT NULL DEFAULT 0,  -- 被检索命中过几次
    last_accessed_at TEXT,

    -- ---- 状态机（照抄知识图谱 relations 表的做法）----
    status        TEXT NOT NULL DEFAULT 'proposed'
                  CHECK(status IN ('proposed', 'verified', 'rejected', 'superseded')),

    -- ---- 血缘（人工核对时要靠它对照原文）----
    -- 注意：只有这两样，没有 user_id / thread_id。
    -- 「不参与检索、不参与决策」是记忆域与用户身份之间的隔离带。
    source_qa_log_id INTEGER,            -- 指向 qa_logs.db 的 qa_logs.id
    source_quote     TEXT,               -- query/answer 的原文摘录（限长 300 字）

    -- ---- 审计 ----
    batch_id    TEXT,
    reviewed_by TEXT,
    reviewed_at TEXT,
    note        TEXT
);

-- 记忆之间的关系：相似（治理时发现）/ 已合并
CREATE TABLE memory_links (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    from_id    INTEGER NOT NULL REFERENCES memory_events(id),
    to_id      INTEGER NOT NULL REFERENCES memory_events(id),
    kind       TEXT NOT NULL CHECK(kind IN ('similar', 'merged_into')),
    score      REAL,                     -- 相似度（kind='similar' 时有值）
    created_at TEXT,
    note       TEXT
);

-- 只读视图：把「只取已核实记忆」固化在库侧，应用层不可能漏写状态过滤
CREATE VIEW v_memory AS
    SELECT * FROM memory_events WHERE status = 'verified';
"""


# ============================================================
# 种子：问题类型词表
# ============================================================
# 与 data/knowledge/ 下 10 篇政策文档 + 投诉 对齐，一条记忆必落其中之一。
# patterns 只用于"从对话里认出问题类型"的兜底匹配 —— 主路径是让 LLM 从这份
# 封闭清单里选，选不出来就丢（不发明新类型）。
ISSUE_TYPES = [
    ("RETURN_NO_REASON", "七天无理由退货",
     ["七天无理由", "无理由退货", "7天无理由", "不想要了", "后悔了"]),
    ("RETURN_PROCESS", "退换货流程",
     ["怎么退", "退货流程", "换货流程", "如何申请", "怎么申请"]),
    ("QUALITY_ISSUE", "质量问题退换",
     ["坏了", "故障", "质量问题", "破损", "有裂痕", "不能用", "开不了机"]),
    ("SHIPPING_FEE", "运费承担",
     ["运费", "谁出运费", "自己付", "运费谁承担", "包邮"]),
    ("REFUND_TIMELINE", "退款到账",
     ["退款多久", "退款到账", "钱什么时候", "几天到账", "还没退"]),
    ("PRICE_PROTECTION", "价保",
     ["价保", "降价", "买贵了", "差价", "便宜了"]),
    ("LOGISTICS", "物流时效",
     ["快递", "物流", "到哪了", "什么时候到", "运单号"]),
    ("LATE_SHIPMENT", "超时未发货",
     ["没发货", "还没发", "超时未发货", "一直不发货", "怎么还不发"]),
    ("MEMBER_BENEFIT", "会员权益",
     ["会员", "金卡", "银卡", "等级", "权益", "补贴"]),
    ("INVOICE", "发票",
     ["发票", "开票", "报销", "税号"]),
    ("COMPLAINT", "投诉受理",
     ["投诉", "太差", "曝光", "催过", "服务差"]),
]


# ============================================================
# 建库
# ============================================================

def build(conn: sqlite3.Connection) -> dict:
    """建表 + 灌词表。返回统计。"""
    # 先 DROP 再建，保证脚本可反复跑（幂等）；代价是已有记忆一并清空 —— 见文件头警告
    conn.executescript("""
        DROP VIEW  IF EXISTS v_memory;
        DROP TABLE IF EXISTS memory_links;
        DROP TABLE IF EXISTS memory_events;
        DROP TABLE IF EXISTS issue_types;
    """)
    conn.executescript(SCHEMA_SQL)

    for code, label, patterns in ISSUE_TYPES:
        conn.execute(
            "INSERT INTO issue_types (code, label, patterns) VALUES (?,?,?)",
            (code, label, json.dumps(patterns, ensure_ascii=False)),
        )

    conn.commit()
    return {"issue_types": len(ISSUE_TYPES)}


def main() -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    try:
        stats = build(conn)
        print(f"\n记忆库已生成: {DB_PATH}\n")
        print(f"  问题类型词表  {stats['issue_types']}")

        pending = conn.execute(
            "SELECT COUNT(*) FROM memory_events WHERE status = 'proposed'"
        ).fetchone()[0]
        verified = conn.execute("SELECT COUNT(*) FROM v_memory").fetchone()[0]
        print(f"\n  待核对的候选记忆: {pending}")
        print(f"  已生效的记忆    : {verified}")

        if pending == 0 and verified == 0:
            print("\n    （下一步跑 extract_memory_events.py，从 qa_logs.db 沉淀候选记忆）")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
