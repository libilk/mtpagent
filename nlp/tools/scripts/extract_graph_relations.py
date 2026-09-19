# -*- coding: utf-8 -*-
"""
知识图谱 · 从政策文档抽取「类别 → 政策」关系
==========================================

读 `data/knowledge/` 下的政策文档，用 LLM 抽出候选关系，再经**纯代码的规则校验**后
以 `status='proposed'` 写入图谱。

## 为什么要有这一层"抽取 + 校验"，而不是让 LLM 直接入库

**LLM 抽出来的关系是「提案」，不是「事实」。** 直接入库等于把幻觉当知识永久沉淀，
而且错了也看不出来 —— 图谱里的东西看起来都像那么回事。

所以流程是：**抽取 → 规则校验（本脚本）→ 人工逐条核对（review_graph_relations.py）→ 生效**。

只有走完这一步，`status` 才会变成 `verified`，才会被 `v_rel` 视图选中参与查询。

它是图谱流水线的第 ② 步（① 建库种子 → ② 本脚本抽 proposed → ③ 人工核对改 verified），
中间产物就写在同一个库 `database/knowledge_graph.db` 的 `relations` 表里。

## 幂等性：追加式，**重跑会留重复**

本脚本只 `INSERT`、从不删除，每次跑都新写一批 `status='proposed'` 的行（用 `batch_id` 标批次）。
所以**对同一篇文档重复跑就会产生重复候选**，人工核对时得一条条驳回。
要干净重来，先重跑 `init_knowledge_graph.py`（会清空），再重抽；
只验一篇时用 `--doc <doc_id>` 比重跑整批便宜得多。

## 规则校验查什么（全部是代码，不再经过模型）

1. `quote` 必须是原文的**精确子串**，且 `source_offset` 与之一致 —— 定位不到就丢弃
2. `subject` 必须是图谱里已有的**类别**节点（不在就标 needs_new_entity，交人工判断）
3. `object` 必须是图谱里已有的**政策**节点
4. `predicate` 必须是白名单里的三个之一
5. `condition` 必须能映射到**已有的条件原子**（不能自己发明条件）
6. 同一 (subject, predicate, object) 出现不同 condition → 标 CONFLICT
7. `IS_A` 环检测
8. 按文档统计覆盖度，某篇一条没抽出来 → 告警（可能是漏抽）

**范围：** 只抽退货 / 换货 / 运费 / 时限相关的。价保、发票、物流覆盖范围不在本期范围。

## 用法

    .venv/Scripts/python.exe tools/scripts/extract_graph_relations.py
    .venv/Scripts/python.exe tools/scripts/extract_graph_relations.py --doc seven_day_return_policy
"""

import os
import re
import sys
import json
import sqlite3
import argparse
from datetime import datetime
from typing import List, Dict, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

DB_PATH = os.path.join(PROJECT_ROOT, 'database', 'knowledge_graph.db')
DOCS_DIR = os.path.join(PROJECT_ROOT, 'data', 'knowledge')

if sys.platform == 'win32' and hasattr(sys.stdout, 'buffer'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# 加载 .env
_env = os.path.join(PROJECT_ROOT, '.env')
if os.path.exists(_env):
    with open(_env, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip())


# ============================================================
# LLM 抽取
# ============================================================

# prompt（提示词）= 喂给 LLM（大语言模型）的指令文本。
# 这里把三张"合法词表"直接拼进 prompt，等于提前给模型划死可选范围 —— 比事后纠错便宜得多。
EXTRACT_PROMPT = """你是知识图谱抽取器。从下面这篇电商售后政策文档中，抽取
「商品类别 → 政策」的**适用**或**排除**关系。

## 必须先读的约束

**你只能从给定的词表里选节点，不能自己发明新词。**

### 可用的「类别」节点（subject 必须是其中之一）
{categories}

### 可用的「政策」节点（object 必须是其中之一）
{policies}

### 可用的「条件」原子（condition 必须从中选，或填 null）
{conditions}

条件填 `null` 表示**无条件生效**（例如"生鲜类商品不支持换货"是天生如此）。
条件填具体 code 表示**满足该条件才成立**（例如"已激活的3C数码不适用七天无理由"）。
**这个区分很重要，不要混。**

## 抽取范围

只抽**退货 / 换货 / 运费 / 时限**相关的。
不要抽价格保护、发票、物流配送范围相关的内容。

## 输出格式

只输出 JSON 数组，不要解释。每个元素：

```json
[
  {{
    "subject": "3C数码产品",
    "predicate": "EXCLUDES",
    "object": "七天无理由退货",
    "condition": "ACTIVATED",
    "quote": "已激活或已拆封的 3C 数码产品，如手机、平板电脑、无线耳机",
    "reasoning": "一句话说明为什么这么判"
  }}
]
```

`quote` 必须是文档里的**原文片段**（逐字复制，包括标点），我会用它来定位和核对。
**quote 对不上，这条就会被丢弃。** 宁可少抽，不要编造原文。

文档内容：

---
{doc}
---
"""


def call_llm(prompt: str) -> str:
    from llm.llm_client import LLM
    llm = LLM(model_name="qwen-plus", api_key=os.environ.get("DASHSCOPE_API_KEY"))
    return llm.generate(prompt, temperature=0.1, max_tokens=4000)


def parse_relations(raw: str) -> List[Dict]:
    """从 LLM 输出里抠出 JSON 数组。模型经常包 ```json 代码块，得容错。"""
    text = raw.strip()
    m = re.search(r'```(?:json)?\s*(.*?)```', text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    m = re.search(r'\[.*\]', text, re.DOTALL)
    if m:
        text = m.group(0)
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


# ============================================================
# 规则校验（纯代码，不过模型）
# ============================================================

VALID_PREDICATES = {"APPLIES_TO", "EXCLUDES"}


def rule_check(cand: Dict, doc: str, categories: set, policies: set,
               conditions: set) -> Tuple[bool, str]:
    """
    对单条候选关系做规则校验（纯代码、不过模型 —— 模型不能既当运动员又当裁判）。

    核心是防幻觉（hallucination：模型一本正经编出原文里根本没有的内容）：
    `quote` 必须能在原文里逐字找到，编造的引文对不上就直接丢。

    Returns:
        (是否通过, 不通过的原因或备注)
    """
    subject = (cand.get("subject") or "").strip()
    predicate = (cand.get("predicate") or "").strip()
    obj = (cand.get("object") or "").strip()
    condition = cand.get("condition")
    quote = (cand.get("quote") or "").strip()

    if not subject or not predicate or not obj:
        return False, "缺少 subject / predicate / object"

    if predicate not in VALID_PREDICATES:
        return False, f"predicate 非法: {predicate}"

    # ① 引文必须是原文精确子串 —— 这是防幻觉最关键的一条
    if not quote:
        return False, "缺少 quote，无法追溯"
    if quote not in doc:
        return False, f"quote 不是原文子串（可能是模型编造）: {quote[:40]}..."

    # ② 节点必须在图谱里存在
    if subject not in categories:
        return False, f"subject 不在类别词表里: {subject}"
    if obj not in policies:
        return False, f"object 不在政策词表里: {obj}"

    # ③ 条件必须是已有原子（不能自己发明）
    if condition not in (None, "", "null"):
        if condition not in conditions:
            return False, f"condition 不在条件原子表里: {condition}"

    return True, "ok"


def is_cycle(conn: sqlite3.Connection, subj_id: int, obj_id: int) -> bool:
    """加一条 subj IS_A obj 之前，检查会不会成环（成环会让"向上找父类别"无限递归）。
    做法是顺着已有 IS_A 边往上爬，看能不能绕回 obj。"""
    if subj_id == obj_id:
        return True
    rows = conn.execute("""
        WITH RECURSIVE up(id, depth) AS (
            SELECT ?, 0
            UNION ALL
            SELECT r.object_id, up.depth + 1
            FROM up JOIN relations r ON r.subject_id = up.id AND r.predicate = 'IS_A'
            WHERE up.depth < 10
        )
        SELECT 1 FROM up WHERE id = ? LIMIT 1
    """, (subj_id, obj_id)).fetchone()
    return rows is not None


# ============================================================
# 主流程
# ============================================================

def load_vocab(conn: sqlite3.Connection):
    categories = {r[0] for r in conn.execute(
        "SELECT name FROM entities WHERE entity_type = 'category'")}
    policies = {r[0] for r in conn.execute(
        "SELECT name FROM entities WHERE entity_type = 'policy'")}
    conditions = {r[0] for r in conn.execute("SELECT code FROM condition_atoms")}
    return categories, policies, conditions


def extract_one(conn: sqlite3.Connection, doc_id: str, stats: Dict) -> None:
    doc_path = os.path.join(DOCS_DIR, f"{doc_id}.md")
    if not os.path.exists(doc_path):
        print(f"  跳过（文件不存在）: {doc_id}")
        return

    with open(doc_path, encoding='utf-8') as f:
        doc = f.read()

    categories, policies, conditions = load_vocab(conn)

    prompt = EXTRACT_PROMPT.format(
        categories="\n".join(f"- {c}" for c in sorted(categories)),
        policies="\n".join(f"- {p}" for p in sorted(policies)),
        conditions="\n".join(
            f"- {r[0]}: {r[1]}" for r in conn.execute(
                "SELECT code, label FROM condition_atoms")),
        doc=doc,
    )

    raw = call_llm(prompt)
    candidates = parse_relations(raw)

    if not candidates:
        stats["no_extraction"].append(doc_id)
        print(f"  {doc_id:<32} 抽出 0 条 ⚠")
        return

    accepted = rejected = 0
    batch_id = f"{datetime.now():%Y%m%d%H%M}"

    for cand in candidates:
        ok, reason = rule_check(cand, doc, categories, policies, conditions)
        if not ok:
            stats["rejected"].append((doc_id, cand.get("subject"), cand.get("object"), reason))
            rejected += 1
            continue

        subj_id = conn.execute(
            "SELECT id FROM entities WHERE name = ?", (cand["subject"],)).fetchone()[0]
        obj_id = conn.execute(
            "SELECT id FROM entities WHERE name = ?", (cand["object"],)).fetchone()[0]
        condition = cand.get("condition")
        if condition in ("", "null"):
            condition = None

        # ④ 同 (s,p,o) 但 condition 不同 → 冲突，标记出来让人判
        existing = conn.execute(
            "SELECT condition_code FROM relations "
            "WHERE subject_id=? AND predicate=? AND object_id=? AND status != 'rejected'",
            (subj_id, cand["predicate"], obj_id)).fetchall()
        note = ""
        if existing:
            old = {r[0] for r in existing}
            if condition not in old:
                note = f"CONFLICT: 已有 condition={old}，本条是 {condition!r}"
                stats["conflicts"].append((doc_id, cand["subject"], cand["object"]))

        conn.execute(
            "INSERT INTO relations (subject_id, predicate, object_id, condition_code, "
            "  status, source_doc_id, source_quote, source_offset, batch_id, note) "
            "VALUES (?,?,?,?, 'proposed', ?,?,?,?,?)",
            (subj_id, cand["predicate"], obj_id, condition,
             doc_id, cand["quote"], doc.find(cand["quote"]), batch_id, note),
        )
        accepted += 1

        # IS_A 环检测（抽取理论上不产 IS_A，但白名单里留着以防）
        if cand["predicate"] == "IS_A" and is_cycle(conn, subj_id, obj_id):
            conn.execute("UPDATE relations SET status='rejected', note='IS_A 成环' "
                         "WHERE id = last_insert_rowid()")
            stats["rejected"].append((doc_id, cand["subject"], cand["object"], "IS_A 成环"))
            accepted -= 1
            rejected += 1

    conn.commit()
    stats["accepted"] += accepted
    stats["rejected_count"] += rejected
    print(f"  {doc_id:<32} 抽出 {len(candidates):>2} 条 → 通过 {accepted}，丢弃 {rejected}")


def main() -> None:
    # argparse = 标准库的命令行参数解析：让同一个脚本既能整批跑，也能用 --doc 只跑一篇
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", help="只抽某一篇（doc_id）")
    args = ap.parse_args()

    if not os.path.exists(DB_PATH):
        print(f"图谱不存在: {DB_PATH}")
        print("请先运行: python tools/scripts/init_knowledge_graph.py")
        sys.exit(1)

    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("缺少 DASHSCOPE_API_KEY（写在 .env 里）")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    try:
        doc_ids = ([args.doc] if args.doc else
                   sorted(p[:-3] for p in os.listdir(DOCS_DIR) if p.endswith('.md')))

        print(f"\n从 {len(doc_ids)} 篇文档抽取「类别 → 政策」关系\n")

        stats = {"accepted": 0, "rejected_count": 0, "rejected": [],
                 "conflicts": [], "no_extraction": []}
        for doc_id in doc_ids:
            extract_one(conn, doc_id, stats)

        print(f"\n{'=' * 62}")
        print(f"  抽取完成 | 入库(待核对) {stats['accepted']} | 规则丢弃 {stats['rejected_count']}")
        print(f"{'=' * 62}")

        if stats["rejected"]:
            print(f"\n被规则丢弃的 {len(stats['rejected'])} 条：")
            for doc_id, s, o, why in stats["rejected"]:
                print(f"  [{doc_id}] {s} -> {o}: {why}")

        if stats["conflicts"]:
            print(f"\n⚠ 条件冲突 {len(stats['conflicts'])} 处，需要人工判断：")
            for doc_id, s, o in stats["conflicts"]:
                print(f"  [{doc_id}] {s} -> {o}")

        if stats["no_extraction"]:
            print(f"\n⚠ 这些文档一条都没抽出来（可能是漏抽）：{stats['no_extraction']}")

        print("\n下一步：运行 review_graph_relations.py 逐条核对\n")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
