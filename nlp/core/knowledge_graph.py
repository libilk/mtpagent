# -*- coding: utf-8 -*-
"""
知识图谱查询层
============

把 `database/knowledge_graph.db` 里存的「商品 → 类别 → 政策适用性」关系查出来，
回答一类问题：**这件商品，能走哪条售后通道？**

RAG（检索增强生成）= 先检索资料、再让 LLM 基于资料作答。

## 它和 RAG 检索的区别

| | 向量检索（rag_core） | 图谱查询（本模块） |
|---|---|---|
| 回答 | "哪段文字和问题相关" | "沿着关系，这件商品适用/不适用哪些政策" |
| 特点 | 相似度匹配，能容忍模糊 | 确定性推导，**每一步都有依据** |
| 弱点 | 答不出"不适用"（相似度只会说"相关"） | 只能答图谱里存了的 |

**特别地：负向关系（EXCLUDES）是向量检索的盲区。**
向量检索能告诉你「3C数码」和「七天无理由政策」相关，但**说不出"不适用"**。
而这恰恰是售后判责最关键的一步。

## 三值逻辑（必须理解，否则会判反）

| 情况 | 含义 | 输出 |
|---|---|---|
| `condition_code IS NULL` | **无条件生效**（如"生鲜不支持换货"，天生如此） | 直接判定 |
| 条件为 `True` | 条件满足 | 直接判定 |
| 条件为 `False` | 条件明确不满足 | **反向判定**（排除不成立 → 该政策适用） |
| 条件为 `unknown` / 未提供 | 不确定满不满足 | 输出「待确认」，**绝不能默认成"适用"** |

**把 unknown 当成"适用"，会把本该被排除的商品判成可退 —— 错误方向最危险。**

## 为什么"商品没登记锚点"要当场报错，而不是猜个默认类别

猜默认 = 把"我们漏配了映射"悄悄翻译成"这件商品按某类别处理"：政策照常返回、答案
看着完整，漏配的 bug 被吞掉 —— 而且**错的方向可能是把不该退的判成能退**。报错则让它
当场暴露（`resolve_category` 返回 None → 调用方显式处理）。

## 用法

    from core.knowledge_graph import query_policy_applicability

    r = query_policy_applicability("云听 Pro 主动降噪无线耳机", {"ACTIVATED": True})
    # r["excluded"]  -> [{'policy': '七天无理由退货', 'via': '3C数码产品', ...}]
    # r["applies"]   -> [...]
"""

import os
import re
import json
import sqlite3
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DB_PATH = os.path.join(PROJECT_ROOT, 'database', 'knowledge_graph.db')

# 三值常量。用字符串而不是 True/False/None，是为了让"未提供"和"明确为假"能分开 ——
# 这两者含义完全不同：不知道 ≠ 不成立。
TRUE, FALSE, UNKNOWN = "true", "false", "unknown"


# ----------------------------------------------------------
# 条件匹配：从用户话语推出条件状态
# ----------------------------------------------------------

def load_condition_atoms(db_path: str = None) -> List[Dict]:
    """读出全部条件原子（condition atom：可校验的封闭枚举，关系只引用它的 code）。"""
    conn = sqlite3.connect(db_path or DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT code, label, patterns FROM condition_atoms").fetchall()
    finally:
        conn.close()
    return [{"code": r["code"], "label": r["label"],
             "patterns": json.loads(r["patterns"])} for r in rows]


def match_conditions(text: str, db_path: str = None) -> Dict[str, str]:
    """
    从一段用户话语里，用关键词匹配推出「哪些条件成立」。

    **为什么用关键词而不是让 LLM 判：** 这里是分类不是推理，关键词足够且**完全可复现**。
    匹配不到的会留在 unknown —— 那是诚实的，比猜一个结果好。

    真实系统里这里会再加一层 LLM 兜底（把用户话语分类到条件枚举），
    但**本层保持确定性**，LLM 兜底是可选的外挂，不影响这个函数的可复现性。

    Returns:
        {condition_code: 'true' / 'unknown'}

        **只产出 true / unknown，从不产出 false** —— 关键词没命中 ≠ 条件不成立。
        「明确为假」只能由调用方（人工确认 / LLM 兜底）给出，不能由本函数代劳。
    """
    if not text:
        return {}

    result = {}
    for atom in load_condition_atoms(db_path):
        hit = any(p and p in text for p in atom["patterns"])
        result[atom["code"]] = TRUE if hit else UNKNOWN
    return result


# ----------------------------------------------------------
# 类别解析
# ----------------------------------------------------------

def resolve_category(conn: sqlite3.Connection, product_name: str) -> Optional[str]:
    """
    商品名 → 类别名。用 product_categories 里的锚点（anchor：营销名到类别的显式映射 ——
    「云听 Pro 主动降噪无线耳机」这种营销名和文档里的类别词「无线耳机」对不上）做子串匹配。

    **返回 None 表示没匹配上，这时候调用方必须显式处理** ——
    静默兜底到"一般商品"会掩盖漏配锚点的问题，导致查不到政策却看不出来。
    """
    if not product_name:
        return None
    rows = conn.execute(
        "SELECT p.pattern, e.name FROM product_categories p "
        "JOIN entities e ON e.id = p.category_id").fetchall()
    # 取最长匹配（越具体的锚点越优先）
    best = None
    for pattern, cat_name in rows:
        if pattern in product_name or product_name in pattern:
            if best is None or len(pattern) > len(best[0]):
                best = (pattern, cat_name)
    return best[1] if best else None


# ----------------------------------------------------------
# 主查询
# ----------------------------------------------------------

# 递归 CTE（公用表表达式：SQL 里的临时命名子查询，RECURSIVE 版可自引用）：
# `anc` 从商品所在类别沿 IS_A 逐级向上，得到"它以及它所有祖先类别"；
# `hit` 把每个祖先类别上挂的 APPLIES_TO（适用）/ EXCLUDES（排除）关系摊平。
# 关系以 subject_id / predicate / object_id 三元组存储，predicate 即"哪种关系"；
# 只读 v_rel 视图（已滤掉未核实的抽取结果）；查不到 ≠ 都适用，而是图谱没覆盖。
_QUERY_SQL = """
WITH RECURSIVE
anc(id, depth, path) AS (
    -- 起点：商品所属的类别
    SELECT e.id, 0, ',' || e.id || ','
    FROM entities e WHERE e.name = :seed
    UNION ALL
    -- 沿 IS_A 向上找父类别；instr 那行是环检测（重复出现在路径上就停）
    SELECT r.object_id, a.depth + 1, a.path || r.object_id || ','
    FROM anc a
    JOIN v_rel r ON r.subject_id = a.id AND r.predicate = 'IS_A'
    WHERE a.depth < 5 AND instr(a.path, ',' || r.object_id || ',') = 0
),
hit AS (
    SELECT r.predicate      AS predicate,
           p.name           AS policy,
           cat.name         AS via,
           a.depth          AS depth,
           r.condition_code AS cond,
           p.source_doc_id  AS doc_id,
           p.family         AS family      -- 政策族，用于"更具体的覆盖更泛的"
    FROM anc a
    JOIN v_rel r   ON r.subject_id = a.id AND r.predicate IN ('APPLIES_TO', 'EXCLUDES')
    JOIN entities p   ON p.id = r.object_id
    JOIN entities cat ON cat.id = a.id
)
SELECT * FROM hit ORDER BY depth ASC, policy
"""


def query_policy_applicability(product_name: str, facts: Dict[str, bool] = None,
                               db_path: str = None) -> Dict:
    """
    查这件商品适用 / 不适用哪些售后政策。

    Args:
        product_name: 商品名（用演示库里的名字，如「云听 Pro 主动降噪无线耳机」）
        facts:        已知条件状态，如 {"ACTIVATED": True}。
                      **没给的条件一律视为 unknown，不会被当成"不成立"。**

    Returns:
        {
          "product": ...,
          "category": ... 或 None（没匹配到锚点时会显式说明）,
          "excluded":  [{"policy","via","condition","reason"}],
          "applies":   [...],
          "uncertain": [{"policy","via","condition","need"}]   ← 需要向用户追问的
        }

        （注：实现返回的条目实为 {"policy","verdict","source_doc","reasons"}，
        via/why 收在 reasons 里；另会返回上面未列的 "conflicts"。括号内键名为旧版残留。）
    """
    facts = facts or {}
    # 把调用方的 Python 布尔归一成三值常量（True→"true"、False→"false"）。
    # 用 `is True`/`is False` 而非真值判断：1、"yes" 这类会被 str() 原样保留，
    # 之后对不上 TRUE/FALSE 自然落进 unknown —— 不会把随手传的 1 误当成条件成立。
    facts = {k: (TRUE if v is True else FALSE if v is False else str(v))
             for k, v in facts.items()}

    conn = sqlite3.connect(db_path or DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        category = resolve_category(conn, product_name)
        if not category:
            return {
                "product": product_name,
                "category": None,
                "error": (f"商品「{product_name}」在图谱里没有对应的类别锚点。"
                          f"需要在 product_categories 表里补一条映射，"
                          f"否则查不到任何政策（不会静默按'一般商品'处理）。"),
                "excluded": [], "applies": [], "uncertain": [],
            }

        rows = conn.execute(_QUERY_SQL, {"seed": category}).fetchall()
    finally:
        conn.close()

    # ---- 第一步：同族覆盖（越具体的类别越优先）----
    #
    # 通用政策挂在「全部商品」上（如「质量问题退货（15日）」），
    # 而某些类别有**自己的时限**（家电是 7 日）。
    # 如果不做覆盖，家电会同时拿到 15日 和 7日 两条 —— 看起来像"两条都适用"，
    # 实际上是"更具体的覆盖了更泛的"。
    #
    # 做覆盖的依据是 `family`（政策族）：同一族里只保留 depth 最小（离商品最近、
    # 最具体）的那条。没有 family 的政策（如「七天无理由退货」）不参与覆盖。
    by_family: Dict[str, sqlite3.Row] = {}
    keep: List[sqlite3.Row] = []
    for r in rows:
        fam = r["family"]
        if not fam:
            keep.append(r)
            continue
        cur = by_family.get(fam)
        if cur is None or r["depth"] < cur["depth"]:
            by_family[fam] = r
    rows = keep + list(by_family.values())

    # ---- 第二步：按政策合并 ----
    #
    # 同一个政策可能从多个类别被判定到（例如「七天无理由退货」既挂在 3C数码 上，
    # 也挂在 全部商品 上）。如果逐条输出，用户会看到同一个政策出现好几遍，
    # 而且"确定被排除"和"待确认"混在一起，反而看不出结论。
    #
    # 合并规则：**更确定的结论优先** —— 排除/适用 > 待确认。
    # 同一政策同时被判"排除"和"适用"属于真冲突，单独标出来人工看。
    # verdict（判定结果）= 每条政策最终的 excluded / applies / uncertain / conflict。
    CERTAINTY = {"excluded": 3, "applies": 3, "uncertain": 1}
    merged: Dict[str, Dict] = {}

    for r in rows:
        cond = r["cond"]
        is_exclude = r["predicate"] == "EXCLUDES"
        via = r["via"]

        entry = {"policy": r["policy"], "via": via, "condition": cond,
                 "source_doc": r["doc_id"], "why": None}

        if cond is None:
            entry["verdict"] = "excluded" if is_exclude else "applies"
            entry["why"] = f"「{via}」无条件{'排除' if is_exclude else '适用'}该政策"

        else:
            state = facts.get(cond, UNKNOWN)

            if state == TRUE:
                entry["verdict"] = "excluded" if is_exclude else "applies"
                entry["why"] = f"「{via}」在该条件下{'排除' if is_exclude else '适用'}"

            elif state == FALSE:
                if is_exclude:
                    # 排除条件不成立 → 该政策反而适用
                    entry["verdict"] = "applies"
                    entry["why"] = f"排除条件「{cond}」不成立，因此该政策适用"
                else:
                    # 适用条件不成立 → 不做任何输出。
                    # 「不适用」和「被排除」是两回事，这里不替用户下结论。
                    continue

            else:
                # ★ unknown：绝不能默认成"适用"
                entry["verdict"] = "uncertain"
                entry["why"] = f"「{via}」的{'排除' if is_exclude else '适用'}取决于「{cond}」，当前无法确定"

        # 每条依据都带上它自己的结论 —— 后面要按最终结论筛掉无关的依据。
        # 否则会出现「判成'适用'，依据里却列着一条'待确认'」这种误导。
        reason = {"via": via, "why": entry["why"], "verdict": entry["verdict"]}

        name = entry["policy"]
        cur = merged.get(name)
        if cur is None:
            merged[name] = {
                "policy": name, "verdict": entry["verdict"],
                "source_doc": entry["source_doc"],
                "reasons": [reason],
            }
        else:
            cur["reasons"].append(reason)
            if entry["verdict"] != cur["verdict"]:
                if (CERTAINTY[entry["verdict"]] == 3 and CERTAINTY[cur["verdict"]] == 3):
                    cur["verdict"] = "conflict"          # 两个都很确定 → 真冲突
                elif CERTAINTY[entry["verdict"]] > CERTAINTY[cur["verdict"]]:
                    cur["verdict"] = entry["verdict"]    # 一强一弱 → 取强的

    # 只保留支持最终结论的依据；被压下去的单独放一边，便于排查
    for v in merged.values():
        final = v["verdict"]
        keep = [r for r in v["reasons"]
                if r["verdict"] == final or (final == "conflict")]
        drop = [r for r in v["reasons"] if r not in keep]
        v["reasons"] = keep or v["reasons"]
        if drop:
            v["overridden"] = [
                {"via": r["via"], "verdict": r["verdict"]} for r in drop
            ]

    excluded = [v for v in merged.values() if v["verdict"] == "excluded"]
    applies = [v for v in merged.values() if v["verdict"] == "applies"]
    uncertain = [v for v in merged.values() if v["verdict"] == "uncertain"]
    conflicts = [v for v in merged.values() if v["verdict"] == "conflict"]

    return {
        "product": product_name,
        "category": category,
        "excluded": excluded,
        "applies": applies,
        "uncertain": uncertain,
        "conflicts": conflicts,
    }


def format_result(result: Dict) -> str:
    """
    把查询结果整理成给 Agent / 用户看的一段话。

    刻意**保留推导依据**（via 字段）—— 这样答复里能说清"因为耳机属于 3C 数码，
    该类别的无理由退货被排除"，而不是干巴巴一句"不能退"。
    """
    if result.get("error"):
        return result["error"]

    lines = [f"商品「{result['product']}」归属类别：{result['category']}"]

    def render(title: str, items: List[Dict]) -> None:
        if not items:
            return
        lines.append(f"\n【{title}】")
        for e in items:
            via = "、".join(f"「{r['via']}」" for r in e["reasons"])
            lines.append(f"  - {e['policy']}（依据：{via}）")

    render("不适用", result["excluded"])
    render("适用", result["applies"])
    render("⚠ 图谱冲突 —— 需要人工核对", result.get("conflicts", []))

    if result["uncertain"]:
        lines.append("\n【待确认 —— 这些需要向用户问清楚，不能假设】")
        for e in result["uncertain"]:
            need = "、".join(r["via"] for r in e["reasons"])
            lines.append(f"  - {e['policy']}（取决于：{need}）")

    if not any(result.get(k) for k in ("excluded", "applies", "uncertain", "conflicts")):
        lines.append("\n（没有查到任何与该商品相关的政策，可能是图谱覆盖不足）")

    return "\n".join(lines)
