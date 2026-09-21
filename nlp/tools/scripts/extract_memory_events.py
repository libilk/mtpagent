# -*- coding: utf-8 -*-
"""
长期记忆 · 从问答流水沉淀候选记忆
=================================

读 `database/qa_logs.db` 里已经落好的每一轮问答，让 LLM 总结成一条**案例记忆**
（商品/类别 + 问题类型 + 处理结论），经纯代码规则校验后写入
`database/memory.db`，状态为 `status='proposed'`。

## 为什么原料用 qa_logs，而不是在请求路径上加钩子

`qa_logs` 表每一轮都已经写好了（query / answer / quality_score / agents / ...）。
从这里离线读料，意味着**LLM 总结的开销不落在用户等待里** —— 请求路径一行不改。

代价是**记忆有延迟**（要等脚本跑）。这是刻意的取舍：长期记忆不是实时功能。

## 只产提案，不产事实

和知识图谱的抽取脚本同一立场：**LLM 抽出来的东西是提案，不是事实。**
所以这里只写 `status='proposed'`，必须经 `review_memory_events.py` 人工核对才生效。

**模型不能既当运动员又当裁判** —— 所以校验全部用纯代码：

    1. 必填字段齐（category / issue_type / conclusion / summary / source_quote）
    2. issue_type 必须来自封闭词表（不能自己发明问题类型）
    3. ★ source_quote 必须是该条 qa_log 的 query 或 answer 的**精确子串**
    4. ★ category_name 必须是知识图谱里**真实存在的类别**
    5. source_quote 长度 ≤ 300 字（不长期留存投诉原文 —— study.md §7.1 红线）
    6. ★ source_quote 不得包含身份标识（用户ID / 手机号 / 身份证 / 邮箱）—— 同一条红线
    7. 与已有记忆重复的，打 CONFLICT 备注交人工，不静默产生重复

第 3 条是最狠的一道：模型编造引文的唯一防线就是"引文必须真的出现在原文里"。
第 4 条沿用知识图谱的立场 —— **对不上就丢并告警，绝不静默兜底**到某个泛类别。
第 6 条是提示词的**硬约束版本**：提示词里已经写了"不要记录用户个人信息"，
但**提示词是软约束，模型可以不听**（本项目反复踩过的坑），所以这里再用代码拦一道。
实测 qa_logs 里真的存在「我是张伟 U10001，订单……」这种原话 —— 引文一旦选中它就会带出身份信息。

## 幂等性

**部分幂等**：已经产出过记忆的 qa_log 不会被重复处理（按 `source_qa_log_id` 去重）。
但**重跑不会删除已有候选** —— 想重来请先重跑 `init_memory_db.py`（会清空）。
`--limit` 默认 20，避免一次跑满 70 条带来意外的模型开销。

## 用法

    .venv/Scripts/python.exe tools/scripts/extract_memory_events.py
    .venv/Scripts/python.exe tools/scripts/extract_memory_events.py --limit 50
    .venv/Scripts/python.exe tools/scripts/extract_memory_events.py --since 2026-09-20
"""

import argparse
import json
import os
import re
import sqlite3
import sys
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# 把项目根加进 sys.path：脚本直接跑时 Python 只把「脚本所在目录」加进去，
# 而下面要 import 项目的 llm.llm_client（与 extract_graph_relations.py 同样的处理）
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MEMORY_DB = os.path.join(PROJECT_ROOT, 'database', 'memory.db')
QA_DB = os.path.join(PROJECT_ROOT, 'database', 'qa_logs.db')
KG_DB = os.path.join(PROJECT_ROOT, 'database', 'knowledge_graph.db')
ENV_PATH = os.path.join(PROJECT_ROOT, '.env')

# source_quote 的长度上限。超了就丢 —— 这是"不长期留存投诉原文"红线的落地方式：
# 摘录只够核对结论与原文是否对得上，不足以还原整段用户陈述。
SOURCE_QUOTE_MAX = 300

# 引文里出现这些形态就整条丢掉。
# 「不长期留存投诉原文」这条红线（study.md §7.1）在数据上的具体形态就是：
# 用户的陈述里夹着身份标识，而引文是原话的忠实摘录 —— 忠实反而会把身份信息带进记忆域。
# 所以拦在入口：宁可丢一条候选记忆，也不让记忆库开始积累"谁说过什么"。
PII_PATTERNS = [
    (re.compile(r"U\d{5,}"), "用户ID"),
    (re.compile(r"1[3-9]\d{9}"), "手机号"),
    (re.compile(r"\d{17}[\dXx]"), "身份证号"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "邮箱"),
]

# 送进提示词的 answer 截断长度（原料侧的限制，与上面的落库限制是两件事）
ANSWER_INPUT_MAX = 1200

if sys.platform == 'win32' and hasattr(sys.stdout, 'buffer'):
    import codecs
    # 只在直接执行时重绑 stdout —— 被 import 时（比如回归测试要测 rule_check）
    # 绝不能动全局 stdout，否则会把调用方的输出流弄坏
    if __name__ == '__main__':
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')


# ============================================================
# 环境与模型
# ============================================================

def load_env() -> None:
    """把 nlp/.env 读进环境变量（手写解析，与仓库其它脚本一致；只 setdefault 不覆盖）"""
    if not os.path.exists(ENV_PATH):
        return
    with open(ENV_PATH, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            k, v = line.split('=', 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def call_llm(prompt: str) -> str:
    from llm.llm_client import LLM
    llm = LLM(model_name="qwen-plus", api_key=os.environ.get("DASHSCOPE_API_KEY"))
    # temperature=0.1：总结要稳，不要发挥
    return llm.generate(prompt, temperature=0.1, max_tokens=1500)


# ============================================================
# 提示词
# ============================================================

EXTRACT_PROMPT = """从下面这一轮电商售后问答里，总结出一条**可复用的案例记忆**。

## 这一轮问答

用户问：{query}

助手答：{answer}

## 要求

只输出一个 JSON 对象，字段如下：

- `category_name`：**必须从下面这份类别清单里原样选一个**，不许自己造。
  如果这一轮不涉及具体商品（纯政策咨询），选「全部商品」。
- `product_name`：涉及的商品名。**只能从下面「已知商品」里选**，选不出就填 null。
- `issue_type`：**必须从下面的问题类型清单里原样选 code**，不许自己造。
- `conclusion`：一句话结论（"这类问题最终怎么处理"）。≤ 60 字。
- `summary`：两三句话总结这件事和结论。≤ 200 字。
- `retrieval_questions`：3~5 条**用户可能问出来的话**，要能让这条记忆被检索到。
  **用用户的口吻写**（口语、简短），不要用书面总结的口气。
  例：一条"已拆封耳机走质量问题通道"的记忆，问题应该是
  「耳机拆开了还能退吗」「用了几天坏了怎么退」「激活过的能退货吗」。
  每条 ≤ 30 字。
- `source_quote`：**从上面「用户问」或「助手答」里原样复制的一段**（≤ 300 字），
  必须能支撑你的 conclusion。**不许改写、不许拼接** —— 会被逐字校验，对不上就整条作废。

## 可选的问题类型 code

{issue_types}

## 可选的类别名

{categories}

## 已知商品（选了它就不必再填 category，会按锚点自动对齐）

{products}

## 注意

- 这是**案例记忆**，记的是"这类问题该怎么处理"，**不要记录任何用户个人信息**
  （姓名、电话、地址、会员号一律不要出现在 conclusion 和 summary 里）
- 结论要基于助手的回答，**不要自己补充政策判断**

只输出 JSON，不要解释，不要代码块："""


def build_prompt(query: str, answer: str, issue_types: list, categories: list, products: list) -> str:
    it_lines = "\n".join(f"- {code}：{label}" for code, label, _ in issue_types)
    cat_lines = "\n".join(f"- {c}" for c in categories)
    prod_lines = "\n".join(f"- {p}（→ {c}）" for p, c in products) or "（无）"
    return EXTRACT_PROMPT.format(
        query=query,
        answer=answer[:ANSWER_INPUT_MAX],
        issue_types=it_lines,
        categories=cat_lines,
        products=prod_lines,
    )


def parse_memory(raw: str) -> dict:
    """从模型回复里抠出 JSON（容忍代码块围栏与前后杂字）"""
    if not raw:
        return {}
    text = raw.strip()
    # 剥 ```json ... ``` 围栏
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    # 抠第一个平衡的 {...}
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        data = json.loads(text[start:end + 1])
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def normalize_questions(raw) -> list:
    """把模型给的 retrieval_questions 规整成字符串列表。

    **宽松处理，不拦**：这只是"额外的召回线索"，缺了记忆仍然可用（靠 summary 检索）。
    为它单独丢一条总结质量很好的记忆，不划算。
    """
    if not isinstance(raw, list):
        return []
    out = []
    for q in raw:
        s = str(q).strip()
        if s:
            out.append(s[:60])
    return out[:5]


# ============================================================
# 规则校验（纯代码 —— 模型不能当自己的裁判）
# ============================================================

def rule_check(item: dict, qa_row: dict, issue_codes: set, kg_categories: set) -> tuple:
    """校验一条候选记忆。

    Returns:
        (ok: bool, reason: str) —— ok=False 时 reason 说明被哪条规则拦下，用于打日志
    """
    # ---- 规则 1：必填字段齐 ----
    required = ["category_name", "issue_type", "conclusion", "summary", "source_quote"]
    missing = [f for f in required if not (item.get(f) or "").strip()]
    if missing:
        return False, f"缺少必填字段: {', '.join(missing)}"

    quote = item["source_quote"].strip()

    # ---- 规则 2：issue_type 必须来自封闭词表 ----
    if item["issue_type"] not in issue_codes:
        return False, f"问题类型不在词表里: {item['issue_type']}（不许自己发明类型）"

    # ---- 规则 3：引文必须是原文精确子串（防编造原文 —— 本项目最狠的一招）----
    query = (qa_row.get("query") or "").strip()
    answer = (qa_row.get("answer") or "").strip()
    if quote not in query and quote not in answer:
        return False, f"引文不是原文的精确子串（疑似编造）: {quote[:40]}..."

    # ---- 规则 4：类别必须是知识图谱里真实存在的类别 ----
    if item["category_name"] not in kg_categories:
        return False, f"类别不在知识图谱里: {item['category_name']}（不静默兜底到泛类别）"

    # ---- 规则 5：引文长度上限（不长期留存投诉原文）----
    if len(quote) > SOURCE_QUOTE_MAX:
        return False, f"引文超长 {len(quote)} 字 > {SOURCE_QUOTE_MAX}（摘录只够核对，不该还原整段陈述）"

    # ---- 规则 6：引文不得带身份标识（提示词的硬约束版本）----
    for pattern, pii_name in PII_PATTERNS:
        hit = pattern.search(quote)
        if hit:
            return False, f"引文含{pii_name}「{hit.group(0)}」（记忆域不得积累身份信息）"

    return True, ""


# ============================================================
# 抽取
# ============================================================

def fetch_pending(conn: sqlite3.Connection, limit: int, since: str = None) -> list:
    """取还没沉淀过记忆的问答流水。

    ATTACH 之后可以跨库查：qa = qa_logs.db，kg = knowledge_graph.db。
    """
    sql = """
        SELECT q.id, q.timestamp, q.query, q.answer, q.quality_score, q.agents
        FROM qa.qa_logs q
        WHERE q.success = 1
          AND q.answer IS NOT NULL AND length(trim(q.answer)) > 20
          AND q.query  IS NOT NULL AND length(trim(q.query))  > 2
          AND q.id NOT IN (
              SELECT source_qa_log_id FROM main.memory_events
              WHERE source_qa_log_id IS NOT NULL
          )
    """
    params: list = []
    if since:
        sql += " AND q.timestamp >= ?"
        params.append(since)
    sql += " ORDER BY q.id LIMIT ?"
    params.append(limit)

    rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def is_duplicate(conn: sqlite3.Connection, category: str, issue_type: str, conclusion: str) -> bool:
    """规则 6 用：同 (类别, 问题类型, 结论) 是否已存在（含 proposed/rejected —— 打回过的也要认出来）"""
    row = conn.execute(
        "SELECT 1 FROM memory_events WHERE category_name=? AND issue_type=? AND conclusion=? LIMIT 1",
        (category, issue_type, conclusion),
    ).fetchone()
    return row is not None


def extract_one(conn, qa_row, issue_types, issue_codes, kg_categories, products, batch_id) -> str:
    """处理一条 qa_log。返回一句结果描述（给日志用）。"""
    qa_id = qa_row["id"]
    prompt = build_prompt(qa_row["query"], qa_row["answer"], issue_types, kg_categories, products)

    try:
        raw = call_llm(prompt)
    except Exception as e:
        return f"[qa#{qa_id}] ✗ 调用模型失败: {e}"

    item = parse_memory(raw)
    if not item:
        return f"[qa#{qa_id}] ✗ 模型没吐出可解析的 JSON"

    ok, reason = rule_check(item, qa_row, issue_codes, kg_categories)
    if not ok:
        return f"[qa#{qa_id}] ✗ 规则拦下 —— {reason}"

    category = item["category_name"]
    issue = item["issue_type"]
    conclusion = item["conclusion"].strip()

    # 规则 7：重复的打标记交人工，但**仍然入库** —— 让核对者看到"这条和那条很像"
    note = None
    if is_duplicate(conn, category, issue, conclusion):
        note = "CONFLICT: 与已有记忆的 (类别, 问题类型, 结论) 完全相同，核对时请确认是否重复"

    product = item.get("product_name")
    if product is not None:
        product = str(product).strip() or None

    questions = normalize_questions(item.get("retrieval_questions"))

    conn.execute(
        """
        INSERT INTO memory_events
            (category_name, product_name, issue_type, conclusion, summary,
             retrieval_questions, status, source_qa_log_id, source_quote, batch_id, note)
        VALUES (?,?,?,?,?,?, 'proposed', ?,?,?,?)
        """,
        (category, product, issue, conclusion, item["summary"].strip(),
         json.dumps(questions, ensure_ascii=False) if questions else None,
         qa_id, item["source_quote"].strip(), batch_id, note),
    )
    conn.commit()

    flag = "⚠ 重复" if note else "✓"
    qflag = f" +{len(questions)}问" if questions else " (无检索问题)"
    return f"[qa#{qa_id}] {flag} {category} / {issue}{qflag} —— {conclusion[:30]}"


# ============================================================
# 入口
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="从 qa_logs 沉淀候选案例记忆")
    ap.add_argument("--limit", type=int, default=20, help="最多处理多少条（默认 20，控成本）")
    ap.add_argument("--since", type=str, default=None, help="只处理该时间之后的记录，如 2026-09-20")
    args = ap.parse_args()

    load_env()
    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("✗ 未找到 DASHSCOPE_API_KEY（检查 nlp/.env）")
        sys.exit(1)

    for p, name in ((MEMORY_DB, "memory.db"), (QA_DB, "qa_logs.db"), (KG_DB, "knowledge_graph.db")):
        if not os.path.exists(p):
            print(f"✗ 缺少 {name}: {p}")
            print("  （memory.db 跑 init_memory_db.py；qa_logs.db 跑服务产生；knowledge_graph.db 跑 init_knowledge_graph.py）")
            sys.exit(1)

    conn = sqlite3.connect(MEMORY_DB)
    conn.row_factory = sqlite3.Row
    # ATTACH 两个只读库：qa_logs 是原料，knowledge_graph 提供合法类别与商品锚点
    conn.execute("ATTACH DATABASE ? AS qa", (QA_DB,))
    conn.execute("ATTACH DATABASE ? AS kg", (KG_DB,))
    try:
        issue_types = [tuple(r) for r in conn.execute(
            "SELECT code, label, patterns FROM main.issue_types ORDER BY code")]
        issue_codes = {t[0] for t in issue_types}
        kg_categories = {r[0] for r in conn.execute(
            "SELECT name FROM kg.entities WHERE entity_type='category'")}
        products = [(r[0], r[1]) for r in conn.execute(
            """SELECT pc.pattern, e.name FROM kg.product_categories pc
               JOIN kg.entities e ON e.id = pc.category_id""")]

        pending = fetch_pending(conn, args.limit, args.since)
        print("=" * 60)
        print(f"待沉淀的问答流水: {len(pending)} 条（上限 {args.limit}）")
        print(f"合法问题类型: {len(issue_codes)} | 合法类别: {len(kg_categories)} | 已知商品: {len(products)}")
        print("=" * 60)

        if not pending:
            print("\n没有需要处理的记录。")
            print("  （要么已全部沉淀过，要么 qa_logs 里还没有成功的问答）")
            return

        batch_id = f"{datetime.now():%Y%m%d%H%M}"
        passed = failed = 0
        print()
        for row in pending:
            result = extract_one(conn, row, issue_types, issue_codes,
                                 kg_categories, products, batch_id)
            print(" ", result)
            if "✓" in result or "⚠" in result:
                passed += 1
            else:
                failed += 1

        print()
        print("=" * 60)
        print(f"入库 {passed} 条 | 被拦下 {failed} 条 | batch_id={batch_id}")
        print()
        print("  下一步：人工核对这些提案（只会列出 status='proposed' 的）")
        print("    .venv/Scripts/python.exe tools/scripts/review_memory_events.py --list")
        print("  核对结果写成 JSON 决策文件后再 --apply（先落盘，再改库，没有 undo）")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
