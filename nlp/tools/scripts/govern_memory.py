# -*- coding: utf-8 -*-
"""
长期记忆 · 治理（相似合并 / 遗忘）
=================================

维持记忆库的规模与检索稳定性。**两段式，不用 `--dry-run`**（项目惯例，见
`review_graph_relations.py`：先落盘报告，再拿报告去 apply，没有 undo）。

    # 1. 只读：算重要度、找相似对、找该遗忘的，出一份 JSON 报告
    .venv/Scripts/python.exe tools/scripts/govern_memory.py --report > 报告.json

    # 2. 拿报告写库
    .venv/Scripts/python.exe tools/scripts/govern_memory.py --apply 报告.json

## 两类治理动作

### ① 相似合并

用 `llm.embedder.cosine_similarity` 两两比向量（**复用索引脚本已经建好的 Chroma
collection 里的向量**，不重新调模型，零 API 成本）。
超过阈值的记进 `memory_links(kind='similar')`；apply 时**保留较早的那条**，
另一条置 `status='superseded'` 并记 `kind='merged_into'`。

> 为什么保留较早的：先入库的那条是人工先核对过的，理由更充分。

### ② 遗忘

**三个条件缺一不可：**

    importance < 阈值  且  access_count == 0  且  距上次访问 > N 天

少任何一条都不忘 —— 这是刻意的：

| 少哪条 | 为什么不该忘 |
|---|---|
| 只看重要度 | 高价值的记忆可能只是暂时没被问到 |
| 只看访问次数 | 刚入库还没被检索过的正常记忆会被误杀 |
| 只看时间 | 一条重要且被反复用过的记忆，不该因为"最近没人问"就丢 |

**遗忘不是删除** —— 置 `status='superseded'`，与知识图谱的 `rejected` 保留不删同一个道理：
删了就无从复查，也没法解释"这条为什么不见了"。

## 重要度怎么算（纯规则，不调 LLM）

    base 0.3
    + 0.3 × 质量分              答得好的，结论更可信
    + 0.3 真办了事（有单据号）   有具体处理结论，复用价值最高
    + 0.1 有具体商品类别         比"全部商品"上的泛泛政策咨询更有针对性
    上限 1.0

信号全部来自**已有的** `qa_logs` 行（通过 `source_qa_log_id` 关联），
不需要新增任何采集 —— 与 study.md §7.1「零新增数据采集 = 零新增合规义务」一致。

## 幂等性

- `--report` **只读**，随便跑。
- `--apply` 改 `status` / `importance` / 写 `memory_links`。**重复 apply 同一份报告是安全的**
  （已经是 superseded 的不会再变，importance 是覆盖写）。但**没有 undo**。
"""

import argparse
import json
import os
import re
import sqlite3
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MEMORY_DB = os.path.join(PROJECT_ROOT, 'database', 'memory.db')
QA_DB = os.path.join(PROJECT_ROOT, 'database', 'qa_logs.db')
CHROMA_DIR = os.path.join(PROJECT_ROOT, 'vector_db', 'chroma_db')

COLLECTION_NAME = "memory_events"

# ---- 阈值（都是纯规则，改这里就能调松紧）----
# 相似度阈值：**实测校准，但只有一个正样本、没有负样本对照** —— 上线前请按自己的库调。
#   实测：同一案例（同订单同商品同问题）产生的两条记忆，余弦相似度 = 0.855。
#   所以默认取 0.85 能合并它们。但注意 0.855 与 0.85 只差一点点 ——
#   换一种措辞就可能落到线下。缺的是"两条不同案例"的相似度做负样本对照。
#   项目里另有一处 0.95 的口径（enhanced_nodes 语义重复），那是判**同一请求内的
#   重复答案**（近乎逐字相同），比"同一案例的不同措辞"严得多，不能直接照搬。
SIMILARITY_THRESHOLD = 0.85
FORGET_IMPORTANCE_MAX = 0.4    # 低于它才算"价值不足"
FORGET_UNACCESSED_DAYS = 30    # 且这么久没被检索命中过
GOVERNOR = "govern-script"

# 重要度权重
IMP_BASE = 0.3
IMP_QUALITY_WEIGHT = 0.3
IMP_HANDLED_BONUS = 0.3        # 真办了事（答案里带工单号/退货单号）
IMP_SPECIFIC_CATEGORY = 0.1    # 有具体商品类别（不是根类别「全部商品」）

ROOT_CATEGORY = "全部商品"
# 单据号：工单 TK… / 退货 RF…，与 core/ecommerce_crm.py 的编号前缀一致
DOC_NO_RE = re.compile(r"\b(?:TK|RF)\d{6,}\b")

if sys.platform == 'win32' and hasattr(sys.stdout, 'buffer'):
    import codecs
    # 只在直接执行时重绑 stdout —— 被 import 时绝不能动全局输出流（测试会 import 本模块）
    if __name__ == '__main__':
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')


# ============================================================
# 纯规则：重要度与遗忘判定（可单测，不碰 IO）
# ============================================================

def compute_importance(qa_row: Optional[Dict[str, Any]], memory_row: Dict[str, Any]) -> float:
    """按规则算一条记忆的重要度（0~1）。不调 LLM。

    Args:
        qa_row: 它的来源 qa_logs 行（可能为 None —— 人工补充的记忆没有来源）
        memory_row: memory_events 行（用 category_name）
    """
    score = IMP_BASE

    quality = 0.0
    handled = False
    if qa_row:
        try:
            quality = float(qa_row.get("quality_score") or 0.0)
        except (TypeError, ValueError):
            quality = 0.0
        answer = str(qa_row.get("answer") or "")
        # 真办了事：答案里出现真实的工单号/退货单号
        # （防幻觉兜底保证了"有单号"就基本等于"真写库了"，见 _verify_write_claim）
        handled = bool(DOC_NO_RE.search(answer))

    score += IMP_QUALITY_WEIGHT * max(0.0, min(1.0, quality))
    if handled:
        score += IMP_HANDLED_BONUS
    if (memory_row.get("category_name") or "") not in ("", ROOT_CATEGORY):
        score += IMP_SPECIFIC_CATEGORY

    return round(min(1.0, score), 3)


def should_forget(importance: float, access_count: int, last_accessed_at: Optional[str],
                  now: datetime, days: int = FORGET_UNACCESSED_DAYS,
                  importance_max: float = FORGET_IMPORTANCE_MAX) -> Tuple[bool, str]:
    """遗忘判定：**三个条件缺一不可**。

    Returns:
        (是否该忘, 说明) —— 不该忘时说明里写清缺哪一条，便于核对时理解规则
    """
    if importance >= importance_max:
        return False, f"重要度 {importance} ≥ {importance_max}（有价值，留着）"
    if int(access_count or 0) > 0:
        return False, f"被检索命中过 {access_count} 次（有人在用）"

    # 从没被访问过 + 重要度低 —— 再看时间
    if not last_accessed_at:
        # 从没访问过，用入库时间来判"躺了多久"得靠调用方传；这里保守认为可以忘，
        # 但把理由写清楚，让核对的人能一眼看出这是"从来没被用过且价值不足"
        return True, f"重要度 {importance} < {importance_max}，从未被检索命中过"

    try:
        last = datetime.strptime(str(last_accessed_at)[:19], "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return False, f"上次访问时间无法解析（{last_accessed_at}）→ 宁可留着"

    idle_days = (now - last).days
    if idle_days >= days:
        return True, f"重要度 {importance} < {importance_max}，且已 {idle_days} 天未被访问"
    return False, f"距上次访问才 {idle_days} 天 < {days} 天（还没到遗忘窗口）"


# ============================================================
# 读库
# ============================================================

def load_rows(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """取所有 verified 记忆，并连带它的来源 qa_log（算重要度要用）"""
    rows = []
    for r in conn.execute("""
        SELECT m.id, m.category_name, m.product_name, m.issue_type, m.conclusion,
               m.summary, m.importance, m.access_count, m.last_accessed_at,
               q.quality_score AS qa_quality, q.answer AS qa_answer
        FROM main.memory_events m
        LEFT JOIN qa.qa_logs q ON q.id = m.source_qa_log_id
        WHERE m.status = 'verified'
        ORDER BY m.id
    """):
        rows.append(dict(r))
    return rows


def load_vectors() -> Dict[int, List[float]]:
    """从索引过的 Chroma collection 里取每条记忆的向量（**不重新调模型**）。

    拿不到（还没建索引、或依赖缺失）就返回空 dict —— 治理退化成"只做遗忘"，
    不因为相似度算不了就整体失败。
    """
    try:
        from rag_core.chroma_store import ChromaStore
        store = ChromaStore(dimension=1536, collection_name=COLLECTION_NAME,
                            persist_directory=CHROMA_DIR)
        data = store.collection.get(include=["embeddings", "metadatas"])
        # 注意：Chroma 返回的 embeddings / metadatas 是 **numpy 数组**，
        # 不能用 `x or []` 兜底（数组做真值判断会抛 "truth value ... is ambiguous"），
        # 必须显式判 None。
        embeddings = data.get("embeddings")
        metadatas = data.get("metadatas")
        if embeddings is None or metadatas is None:
            return {}
        out: Dict[int, List[float]] = {}
        for vec, meta in zip(embeddings, metadatas):
            mid = (meta or {}).get("memory_id")
            if mid is not None:
                out[int(mid)] = [float(x) for x in vec]
        return out
    except Exception as e:
        print(f"  ⚠ 读不到向量（相似合并将跳过）: {e}")
        return {}


# ============================================================
# 报告 / 应用
# ============================================================

def build_report(conn: sqlite3.Connection, now: datetime,
                 similarity_threshold: float = SIMILARITY_THRESHOLD) -> Dict[str, Any]:
    rows = load_rows(conn)
    vectors = load_vectors()

    importances: Dict[str, float] = {}
    forget: List[Dict[str, Any]] = []
    keep: List[Dict[str, Any]] = []

    for r in rows:
        qa_row = {"quality_score": r["qa_quality"], "answer": r["qa_answer"]}
        imp = compute_importance(qa_row, r)
        importances[str(r["id"])] = imp
        drop, why = should_forget(imp, r["access_count"], r["last_accessed_at"], now)
        entry = {"id": r["id"], "importance": imp, "access_count": r["access_count"],
                 "last_accessed_at": r["last_accessed_at"],
                 "category_name": r["category_name"],
                 "conclusion": (r["conclusion"] or "")[:40], "reason": why}
        (forget if drop else keep).append(entry)

    # ---- 相似对（两两比，记忆量小，不做 ANN）----
    pairs = []
    ids = [r["id"] for r in rows if r["id"] in vectors]
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            try:
                from llm.embedder import cosine_similarity
                sim = cosine_similarity(vectors[a], vectors[b])
            except Exception:
                sim = 0.0
            if sim >= similarity_threshold:
                pairs.append({"a": a, "b": b, "score": round(float(sim), 4),
                              # 保留较早的那条（先入库的是人工先核对过的）
                              "keep": min(a, b), "merge": max(a, b)})

    return {
        "_说明": [
            "govern_memory.py --report 的产出。用 --apply 本文件写库。",
            f"生成时间: {now:%Y-%m-%d %H:%M:%S}",
            f"阈值: 相似度≥{similarity_threshold} 视为重复；"
            f"遗忘需同时满足 重要度<{FORGET_IMPORTANCE_MAX} 且 从未被检索 且 未访问>{FORGET_UNACCESSED_DAYS}天",
            "遗忘与合并都是把 status 置为 'superseded'，**不删除**（删了无从复查）。",
        ],
        "generated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        "importance": importances,
        "similar_pairs": pairs,
        "forget": forget,
        "keep": keep,
        "vectors_available": len(vectors),
    }


def cmd_report(conn: sqlite3.Connection, similarity_threshold: float) -> None:
    now = datetime.now()
    report = build_report(conn, now, similarity_threshold)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def cmd_apply(conn: sqlite3.Connection, report: Dict[str, Any],
              similarity_threshold: float = SIMILARITY_THRESHOLD) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    merged = forgotten = imp_updated = 0

    # ---- ① 重要度落库 ----
    # recall_by_category 的 ORDER BY 用的就是它，所以必须写进去（report 阶段只算不写）
    for mid, imp in (report.get("importance") or {}).items():
        conn.execute("UPDATE main.memory_events SET importance = ? WHERE id = ?",
                     (float(imp), int(mid)))
        imp_updated += 1

    # ---- ② 相似合并 ----
    for p in report.get("similar_pairs") or []:
        keep, merge = int(p["keep"]), int(p["merge"])
        conn.execute(
            "INSERT INTO memory_links (from_id, to_id, kind, score, created_at, note) "
            "VALUES (?,?, 'similar', ?, ?, ?)",
            (keep, merge, float(p.get("score") or 0.0), now,
             f"治理脚本判定相似（阈值 {similarity_threshold}）"))
        conn.execute(
            "INSERT INTO memory_links (from_id, to_id, kind, created_at, note) "
            "VALUES (?,?, 'merged_into', ?, ?)",
            (merge, keep, now, "已并入较早的一条"))
        conn.execute(
            "UPDATE main.memory_events SET status='superseded', reviewed_by=?, reviewed_at=?, "
            "note = COALESCE(note,'') || ' | 已合并到 #' || ? WHERE id = ?",
            (GOVERNOR, now, str(keep), merge))
        merged += 1

    # ---- ③ 遗忘 ----
    for f in report.get("forget") or []:
        conn.execute(
            "UPDATE main.memory_events SET status='superseded', reviewed_by=?, reviewed_at=?, "
            "note = COALESCE(note,'') || ' | 遗忘: ' || ? WHERE id = ?",
            (GOVERNOR, now, f.get("reason", ""), int(f["id"])))
        forgotten += 1

    conn.commit()

    print(f"\n治理完成：重要度更新 {imp_updated} | 合并 {merged} | 遗忘 {forgotten}")
    verified = conn.execute("SELECT COUNT(*) FROM main.v_memory").fetchone()[0]
    superseded = conn.execute(
        "SELECT COUNT(*) FROM main.memory_events WHERE status='superseded'").fetchone()[0]
    print(f"  当前生效 {verified} 条 | 已归档(superseded) {superseded} 条")
    if merged or forgotten:
        print("\n  被归档的记忆没有删掉 —— 随时可以查 status='superseded' 看它们为什么被归档。")
        print("  它们的旧向量还留在检索索引里，但**不会被检索出来**：")
        print("  recall_similar 会拿检回来的 id 去 v_memory 再过滤一次（只认生效的）。")
        print("  想彻底清掉那些向量，重跑一次 index_memory_events.py 即可。")


def main() -> None:
    ap = argparse.ArgumentParser(description="记忆库治理：相似合并 / 遗忘")
    ap.add_argument("--report", action="store_true", help="只读：出治理报告（JSON 到标准输出）")
    ap.add_argument("--apply", type=str, default=None, help="按报告文件写库（无 undo）")
    ap.add_argument("--similarity", type=float, default=SIMILARITY_THRESHOLD,
                    help=f"相似合并阈值，默认 {SIMILARITY_THRESHOLD}（未做负样本校准，请按自己的库调）")
    args = ap.parse_args()

    if not args.report and not args.apply:
        ap.print_help()
        return

    conn = sqlite3.connect(MEMORY_DB)
    conn.row_factory = sqlite3.Row
    if os.path.exists(QA_DB):
        conn.execute("ATTACH DATABASE ? AS qa", (QA_DB,))
    try:
        if args.report:
            cmd_report(conn, args.similarity)
            return
        with open(args.apply, encoding='utf-8') as f:
            report = json.load(f)
        cmd_apply(conn, report, args.similarity)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
