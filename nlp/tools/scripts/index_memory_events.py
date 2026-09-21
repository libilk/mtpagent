# -*- coding: utf-8 -*-
"""
长期记忆 · 建检索索引
====================

把 `database/memory.db` 里 `status='verified'` 的记忆向量化，写进
**独立的 Chroma collection `memory_events`**，供 `core/memory_recall.py` 检索。

## 为什么是独立 collection

同一个 Chroma 目录（`vector_db/chroma_db`）下用不同 collection 名，
就是两张互不干扰的表。**不碰 `knowledge_base`** —— 否则政策检索会被历史案例污染。

## 索引什么文本：「叙述 + 生成的可检索问题」

    summary + retrieval_questions

**这是「生成可检索问题」这个能力的用途所在。** 用户的问法和记忆的叙述写法天然不一样：

    用户说：  「耳机拆开了还能退吗」
    记忆写：  「已拆封3C数码不支持无理由退货，可走质量问题通道（15日、运费平台承担）」

直接拿叙述去检索，这两种说法之间的语义距离很远、召回率低。
把反生成的问题一起索引，等于**替用户把话先问了一遍**。

## 破坏性：整库重建

**每次运行都会清空 `memory_events` 这个 collection 再重建。**
原因是 `ChromaStore` 没有 upsert —— `add_vectors` 的 ID 取自本地缓存长度，
对已有数据的库直接追加会从 `doc_0` 重新编号、和旧记录重名。
所以只能清空重建（与 `init_*` 系列的 DROP+CREATE 是同一个道理）。

代价：**没进库的新记忆要重跑本脚本才会被检索到**。这符合"记忆是离线沉淀"的整体设计。

## 用法

    .venv/Scripts/python.exe tools/scripts/index_memory_events.py
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 「索引什么文本」与「查询时对什么文本向量化」必须是同一份实现 ——
# 所以 build_index_text 定义在 core/memory_recall.py 里，这里只引用不重写。
# 两处各写一份，迟早会漂移成"索引了一套、查询另一套"，而且查不出来。
from core.memory_recall import build_index_text, load_verified  # noqa: E402

MEMORY_DB = os.path.join(PROJECT_ROOT, 'database', 'memory.db')
CHROMA_DIR = os.path.join(PROJECT_ROOT, 'vector_db', 'chroma_db')
ENV_PATH = os.path.join(PROJECT_ROOT, '.env')

COLLECTION_NAME = "memory_events"
# 与 knowledge_base 一致：text-embedding-v2 是 1536 维
EMBEDDING_DIM = 1536
EMBEDDING_MODEL = "text-embedding-v2"

if sys.platform == 'win32' and hasattr(sys.stdout, 'buffer'):
    import codecs
    # 只在直接执行时重绑 stdout —— 被 import 时绝不能动全局输出流
    if __name__ == '__main__':
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')


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


def main() -> None:
    load_env()
    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("✗ 未找到 DASHSCOPE_API_KEY（检查 nlp/.env）")
        sys.exit(1)

    if not os.path.exists(MEMORY_DB):
        print(f"✗ 缺少 {MEMORY_DB}（先跑 init_memory_db.py 建库、review_memory_events.py 核对）")
        sys.exit(1)

    rows = load_verified()
    print("=" * 60)
    print(f"已生效的记忆: {len(rows)} 条")
    if not rows:
        print()
        print("  没有可索引的记忆。")
        print("  检查：是不是还没有 verified 的记忆？")
        print("    review_memory_events.py --list     看有哪些待核对")
        return

    with_q = sum(1 for r in rows if r["questions"])
    by_cat = {}
    for r in rows:
        by_cat[r["category_name"]] = by_cat.get(r["category_name"], 0) + 1
    print(f"其中带可检索问题的: {with_q} 条")
    print("按类别分布: " + " / ".join(f"{k}×{v}" for k, v in sorted(by_cat.items())))
    print("=" * 60)

    from rag_core.api_embedder import APIEmbedder
    from rag_core.chroma_store import ChromaStore

    embedder = APIEmbedder(api_key=os.environ["DASHSCOPE_API_KEY"],
                           provider="dashscope", model=EMBEDDING_MODEL)
    store = ChromaStore(dimension=EMBEDDING_DIM, collection_name=COLLECTION_NAME,
                        persist_directory=CHROMA_DIR)

    # 整库重建：ChromaStore 没有 upsert，追加会与旧记录重名（见文件头）
    store.clear()

    texts = [build_index_text(r) for r in rows]
    print(f"\n正在向量化 {len(texts)} 条记忆...")
    vectors = embedder.encode_texts(texts, batch_size=10, show_progress=False)

    metadata = [{
        "memory_id": r["id"],                 # ★ 回 SQLite 取完整字段靠它
        "category_name": r["category_name"],
        "issue_type": r["issue_type"],
        "importance": float(r["importance"] or 0.5),
    } for r in rows]

    store.add_vectors(vectors, texts, metadata)

    print(f"\n索引完成：{COLLECTION_NAME} 共 {store.collection.count()} 条")
    print(f"  向量库目录: {CHROMA_DIR}")
    print("\n  下一步：Agent 侧的工具 recall_similar_cases 会用这个 collection")


if __name__ == "__main__":
    main()
