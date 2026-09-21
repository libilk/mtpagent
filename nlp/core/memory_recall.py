# -*- coding: utf-8 -*-
"""
长期记忆 · 查询与检索层
======================

读 `database/memory.db` 的**只读视图 `v_memory`**，把历史案例记忆取出来。

**记忆的对象是商品 / 政策，不是用户** —— 这里回答的是"这类问题以前是怎么处理的"，
不是"这个用户以前做过什么"。为什么必须是后者，见 study.md §7.1.1。

## 为什么只读视图

`v_memory` 的定义里已经固化了 `WHERE status = 'verified'`，
所以**应用层不可能漏写状态过滤**把未核对的候选记忆取出来用 —— 与知识图谱
（`v_rel`）同一手法。本模块的 `recall_*` 全是只读；唯一会写的是 `mark_accessed()`
（单独一个显式函数，不和检索混在一起）。

## 多域隔离 = 按类别分区

每条记忆归属一个 `category_name`，取自知识图谱的类别体系。
检索时按类别过滤，就是"多域隔离"在查询侧的实现 ——
问「耳机」不会把「生鲜」的历史案例捞出来。

## 两阶段检索

    recall_by_category()  阶段一：SQL 层按类别收窄（域隔离）
    recall_similar()      阶段二：向量 + 关键词两路召回 → RRF 融合排序

两路结果的分数量纲完全不同（余弦相似度 0~1，BM25 无上界），
所以用**倒数排名融合（RRF）**——只看"排第几"，不比分数。
这是项目主路径同款的融合方式（见 agents/knowledge_agent 内联那份 RRF）。

## ⚠️ 索引文本的唯一真相源在本文件

`build_index_text()` 定义在**这里**，而不是在索引脚本里 ——
因为**建索引和查询必须对同一段文本向量化**，两处各写一份迟早会漂移。

## 用法

    from core.memory_recall import recall_similar, format_narrative
    rows = recall_similar("耳机拆开了还能退吗", product_name="云听 Pro 主动降噪无线耳机")
    print(format_narrative(rows))
"""

import json
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DB_PATH = os.path.join(PROJECT_ROOT, 'database', 'memory.db')
KG_DB = os.path.join(PROJECT_ROOT, 'database', 'knowledge_graph.db')
CHROMA_DIR = os.path.join(PROJECT_ROOT, 'vector_db', 'chroma_db')
COLLECTION_NAME = "memory_events"
EMBEDDING_DIM = 1536
EMBEDDING_MODEL = "text-embedding-v2"

# RRF 的平滑常数，与项目主路径一致（k=60：防止排名第 1 的分数过于压倒性）
RRF_K = 60

# ★ 相关性下限：融合之前，每一路各自先按阈值筛掉不相关的。
# 为什么必须有这道：**RRF 只看排名、不看分数** —— 于是"什么都不相关"时，
# 排名第 0 的那条照样拿到 1/(k+1) 的分、照样被返回。
# 实测症状：「生鲜坏了怎么办」在只有耳机记忆的库里，也返回了耳机那两条。
# 没有下限，工具就永远不会"查不到"，模型也就永远会把无关案例当参考。
# 两个阈值都沿用项目已有的口径（rag_core 里 SimpleRetriever / BM25 那两个常量）。
MIN_VECTOR_SCORE = 0.45
MIN_BM25_SCORE = 0.5

# BM25 索引的进程内缓存。键是 (记忆条数, 最后一条 id) —— 记忆库变了就重建。
# 为什么需要缓存：BM25 要 fit 整个语料，每次都建太浪费；而记忆是低频变动的。
_BM25_CACHE: Dict[str, Any] = {"key": None, "rows": None, "texts": None, "retriever": None}


def _connect(db_path: Optional[str] = None) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path or DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ============================================================
# 索引文本（建索引与查询共用的唯一真相源）
# ============================================================

def build_index_text(row: Dict[str, Any]) -> str:
    """拼出代表一条记忆的文本：**反生成的问题 + 叙述**。

    为什么问题要放在前面：用户的问法和记忆的叙述写法天然不一样 ——

        用户说：  「耳机拆开了还能退吗」
        记忆写：  「已拆封3C数码不支持无理由退货，可走质量问题通道（15日、运费平台承担）」

    直接拿叙述检索，这两种说法之间的语义距离很远。把反生成的问题一起索引，
    等于**替用户把话先问了一遍**，能显著拉近两种说法的距离。

    注意：Chroma 只取整体向量，先后顺序不影响相似度；放前面是为了读日志时
    先看到"用户会怎么问"。
    """
    questions = row.get("questions") or []
    if isinstance(questions, str):
        try:
            questions = json.loads(questions)
        except (json.JSONDecodeError, TypeError):
            questions = []
    parts = []
    if questions:
        parts.append("可能的问题：" + "；".join(questions))
    parts.append(f"【{row.get('issue_type', '')}】{row.get('summary', '')}")
    return "\n".join(parts)


# ============================================================
# 读取
# ============================================================

def load_verified(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """取所有已生效的记忆（含索引需要的字段）。索引脚本与 BM25 缓存都用它。"""
    conn = _connect(db_path)
    try:
        rows = []
        for r in conn.execute("""
            SELECT id, category_name, product_name, issue_type, conclusion,
                   summary, retrieval_questions, importance, access_count
            FROM v_memory ORDER BY id
        """):
            row = dict(r)
            try:
                row["questions"] = json.loads(row.get("retrieval_questions") or "[]")
            except (json.JSONDecodeError, TypeError):
                row["questions"] = []
            rows.append(row)
        return rows
    finally:
        conn.close()


def load_issue_types(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """读出问题类型词表（code + label），用于把 code 呈现成人话标签。"""
    conn = _connect(db_path)
    try:
        return [{"code": r["code"], "label": r["label"]}
                for r in conn.execute("SELECT code, label FROM issue_types ORDER BY code")]
    finally:
        conn.close()


def recall_by_category(
    category_name: str,
    issue_type: Optional[str] = None,
    limit: int = 5,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """【阶段一】按类别（+ 可选的问题类型）取历史案例 —— 不做语义排序。

    Args:
        category_name: 知识图谱里的类别名。**必填** ——
            不传就退化成"捞所有类别的记忆"，那等于没有多域隔离
        issue_type: 问题类型 code；不传则该类别下所有类型都返回
        limit: 最多返回几条

    Returns:
        记忆列表（**不含 source_qa_log_id / source_quote** —— 血缘是给人工核对用的，
        不该进入给模型看的上下文，避免"用户原话"被反复带进 prompt）
    """
    if not category_name:
        return []
    sql = """
        SELECT category_name, product_name, issue_type, conclusion, summary,
               importance, access_count
        FROM v_memory WHERE category_name = ?
    """
    params: List[Any] = [category_name]
    if issue_type:
        sql += " AND issue_type = ?"
        params.append(issue_type)
    sql += " ORDER BY importance DESC, access_count DESC, id DESC LIMIT ?"
    params.append(limit)

    conn = _connect(db_path)
    try:
        return [dict(r) for r in conn.execute(sql, params)]
    finally:
        conn.close()


# ============================================================
# 两阶段检索：向量 + 关键词 → RRF
# ============================================================

def _rrf_fuse(ranked_id_lists: List[List[int]], k: int = RRF_K) -> List[tuple]:
    """倒数排名融合：得分 = Σ 1/(k + 排名)。

    只吃"排名"，不吃原始分数 —— 所以余弦相似度和 BM25 分数**量纲不同也不怕**，
    不需要先归一化。这正是选它的理由。
    """
    scores: Dict[int, float] = {}
    for ranked in ranked_id_lists:
        for rank, mid in enumerate(ranked):
            scores[mid] = scores.get(mid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def _get_bm25(rows: List[Dict[str, Any]]):
    """懒建 BM25 索引并缓存（记忆库变了才重建）。

    BM25Retriever 直接用原始 retrieve —— 注意它返回的是 `[(语料下标, 分数)]`，
    **没有文本**。这里只要下标就够了（下标 → rows[i] → memory_id）。
    项目另一处（enhanced_entry）用 monkey patch 换成了返回 dict 的版本，
    那是为了给 Agent 的工具用；本模块不需要。
    """
    key = (len(rows), rows[-1]["id"] if rows else 0)
    if _BM25_CACHE["key"] != key:
        from rag_core.bm25_retriever import BM25Retriever
        texts = [build_index_text(r) for r in rows]
        retriever = BM25Retriever()
        retriever.fit(texts)
        _BM25_CACHE.update(key=key, rows=rows, texts=texts, retriever=retriever)
    return _BM25_CACHE["retriever"], _BM25_CACHE["texts"]


def _build_vector_store():
    from rag_core.chroma_store import ChromaStore
    return ChromaStore(dimension=EMBEDDING_DIM, collection_name=COLLECTION_NAME,
                       persist_directory=CHROMA_DIR)


def recall_similar(
    query: str,
    product_name: Optional[str] = None,
    category_name: Optional[str] = None,
    top_k: int = 3,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """【阶段二】按语义 + 关键词两路召回，RRF 融合后返回最相关的历史案例。

    Args:
        query: 用户的问题（用来检索）
        product_name: 商品名 —— 给了就先解析成类别当过滤条件（多域隔离）
        category_name: 直接指定类别（与 product_name 二选一，给了就以它为准）
        top_k: 最终返回几条

    Returns:
        记忆列表（含 conclusion / summary / importance 等，同样不含血缘字段）

    检索不到任何东西时返回 `[]`，**不抛异常** —— 记忆只是"参考"，
    没有历史案例是正常情况，不该让 Agent 的调用失败。
    """
    if not query or not query.strip():
        return []

    # ---- 多域隔离：先把范围收窄到某个类别 ----
    # 注意：商品名解析不到类别时**不兜底成"全部类别"** —— 那等于关掉域隔离。
    # 与图谱查询层同一个立场（resolve_category 返回 None 就该收手，不猜）。
    if not category_name and product_name:
        category_name = resolve_category(product_name)
        if not category_name:
            return []

    rows = load_verified(db_path)
    if not rows:
        return []
    if category_name:
        rows = [r for r in rows if r["category_name"] == category_name]
        if not rows:
            return []

    id_to_row = {r["id"]: r for r in rows}
    ids = set(id_to_row)

    # ---- 路 1：向量召回 ----
    vector_ranked: List[int] = []
    try:
        from rag_core.api_embedder import APIEmbedder
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if api_key and os.path.exists(CHROMA_DIR):
            embedder = APIEmbedder(api_key=api_key, provider="dashscope",
                                   model=EMBEDDING_MODEL)
            qv = embedder.encode_query(query)
            if hasattr(qv, "tolist"):
                qv = qv.tolist()
            store = _build_vector_store()
            # 多召回一些，融合后再截断
            hits = store.search(query_vector=qv, top_k=max(top_k * 4, 10))
            for item in hits:
                meta = item[2] if len(item) >= 3 else {}
                score = item[1] if len(item) >= 2 else 0.0
                mid = meta.get("memory_id") if isinstance(meta, dict) else None
                # 相关性下限：低于它的直接不要（见文件头 MIN_VECTOR_SCORE 的说明）
                if score < MIN_VECTOR_SCORE:
                    continue
                if mid is not None and int(mid) in ids:
                    vector_ranked.append(int(mid))
    except Exception as e:  # 向量路坏了不该整体失败 —— 关键词路还能兜
        import logging
        logging.getLogger(__name__).warning("[记忆检索] 向量路失败，只用关键词路: %s", e)

    # ---- 路 2：关键词召回（BM25）----
    bm25_ranked: List[int] = []
    try:
        retriever, texts = _get_bm25(rows)
        for idx, score in retriever.retrieve(query, top_k=max(top_k * 4, 10)):
            # 同样先过下限 —— BM25 分数无上界，< 0.5 基本等于没有关键词重合
            if score < MIN_BM25_SCORE:
                continue
            if 0 <= idx < len(rows):
                bm25_ranked.append(rows[idx]["id"])
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("[记忆检索] 关键词路失败: %s", e)

    if not vector_ranked and not bm25_ranked:
        return []

    # ---- RRF 融合 ----
    fused = _rrf_fuse([vector_ranked, bm25_ranked])

    out = []
    for mid, _score in fused[:top_k]:
        row = id_to_row.get(mid)
        if not row:
            continue
        out.append({
            # id 是主键、不是血缘 —— 带上它是为了让调用方能够 mark_accessed()
            "id": row["id"],
            "category_name": row["category_name"],
            "product_name": row["product_name"],
            "issue_type": row["issue_type"],
            "conclusion": row["conclusion"],
            "summary": row["summary"],
            "importance": row["importance"],
            "access_count": row["access_count"],
        })
    return out


def resolve_category(product_name: str) -> Optional[str]:
    """商品名 → 知识图谱类别（复用图谱的锚点表，不另建一套映射）。

    解析不到返回 None（不兜底到泛类别）—— 与图谱查询层同一个立场：
    静默兜底会把"漏配锚点"翻译成"按某个类别处理"，方向可能是错的。
    """
    if not product_name:
        return None
    try:
        import sqlite3 as _sq
        conn = _sq.connect(KG_DB)
        try:
            from core.knowledge_graph import resolve_category as _resolve
            return _resolve(conn, product_name)
        finally:
            conn.close()
    except Exception:
        return None


# ============================================================
# 呈现：叙事化三段式
# ============================================================

def format_memories(rows: List[Dict[str, Any]]) -> str:
    """朴素列表渲染（给运维/调试看）"""
    if not rows:
        return ""
    lines = []
    for r in rows:
        prod = f"（{r['product_name']}）" if r.get("product_name") else ""
        lines.append(f"- [{r['category_name']}{prod}] {r['conclusion']}")
    return "\n".join(lines)


def format_narrative(rows: List[Dict[str, Any]], issue_labels: Dict[str, str] = None) -> str:
    """**叙事化三段式注入**：把检索到的记忆组织成模型能读的一段叙事。

        ① 情境   —— 这是什么问题（类别 + 问题类型）
        ② 类似案例 —— 历史上怎么处理的（逐条）
        ③ 参考   —— 边界声明：这是历史经验，不是政策依据

    **第三段不能省。** 没有它，模型很容易把"历史上这么处理的"当成"政策这么规定的" ——
    那就把记忆从"参考"变成了"依据"，等于让历史案例替代政策判断。
    记忆只提供线索，判断仍须回到政策条款与这笔订单的事实。
    """
    if not rows:
        return ""

    labels = issue_labels or {}
    first = rows[0]
    cat = first["category_name"]
    prod = f"商品「{first['product_name']}」" if first.get("product_name") else ""
    issue = labels.get(first["issue_type"], first["issue_type"])

    parts = [f"【情境】这是「{cat}」{prod}上的「{issue}」类问题，库里有 {len(rows)} 条类似案例。"]
    parts.append("【类似案例】")
    for i, r in enumerate(rows, 1):
        summary = (r.get("summary") or "").strip().replace("\n", " ")
        if len(summary) > 80:
            summary = summary[:80] + "…"
        parts.append(f"  {i}. 结论：{r['conclusion']}")
        if summary:
            parts.append(f"     过程：{summary}")
    parts.append(
        "【参考】以上是**历史处理经验**，不是政策依据。"
        "请仍以当前政策条款和这笔订单的事实为准；若与本单情况不同，不要照搬。"
    )
    return "\n".join(parts)


# ============================================================
# 唯一会写的地方：访问计数
# ============================================================

def mark_accessed(memory_ids: List[int], db_path: Optional[str] = None) -> int:
    """记录这些记忆被检索命中过。

    **单独一个函数、不在 recall_* 里顺手做** —— 检索路径要保持只读，
    写入必须是显式的，这样调用方一眼能看出"这里改了数据"。

    `access_count` / `last_accessed_at` 是 Phase 3 治理（遗忘）的判据之一：
    「从没被命中过 + 重要度低 + 长期未访问」才该被遗忘。
    """
    ids = [int(i) for i in (memory_ids or []) if i is not None]
    if not ids:
        return 0
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = _connect(db_path)
    try:
        conn.executemany(
            "UPDATE memory_events SET access_count = access_count + 1, last_accessed_at = ? WHERE id = ?",
            [(now, i) for i in ids],
        )
        conn.commit()
        return len(ids)
    finally:
        conn.close()


def count_memories(db_path: Optional[str] = None) -> Dict[str, int]:
    """统计各类状态下的记忆条数（含未生效的，供运维脚本与核对时看进度）"""
    conn = _connect(db_path)
    try:
        stats = {r["status"]: r["n"] for r in conn.execute(
            "SELECT status, COUNT(*) AS n FROM memory_events GROUP BY status")}
        for key in ("proposed", "verified", "rejected", "superseded"):
            stats.setdefault(key, 0)
        return stats
    finally:
        conn.close()
