# -*- coding: utf-8 -*-
"""
一次性脚本：从 ChromaDB 向量数据库中删除与核心问题不相关的文档。

保留的文档（与4个核心问题相关）：
  - rag_optimization_strategies    — RAG优化策略
  - hybrid_search_deep_dive        — 混合检索深度解析
  - retrieval_methods_comparison   — 检索方法对比
  - rag_embedding_and_vectordb     — Embedding与向量数据库
  - rag_introduction               — RAG入门介绍
  - chinook_database_schema        — Chinook数据库表结构
  - customer_satisfaction_guide    — 客户满意度提升策略
  - customer_retention_best_practices — 客户留存最佳实践

删除的文档：
  - multi_agent_orchestration      — 系统编排文档（含污染检索的示例查询）
  - system_architecture            — 系统架构文档
  - langgraph_workflow             — LangGraph工作流说明
  - product_sales_data             — Excel销售数据
  - sales_report_with_tables       — PDF销售报表
"""

import os
import sys

# 设置Windows控制台UTF-8编码
if sys.platform == 'win32':
    import codecs
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    if hasattr(sys.stderr, 'buffer'):
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# 定位项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(PROJECT_ROOT)

import chromadb
from chromadb.config import Settings

# 要删除的 doc_id 列表
DOCS_TO_DELETE = [
    "multi_agent_orchestration",
    "system_architecture",
    "langgraph_workflow",
    "product_sales_data",
    "sales_report_with_tables",
    "hybrid_search_deep_dive",
    "retrieval_methods_comparison",
    "chinook_database_schema",
    # 已删除的文件，如果还残留在向量库里也清理
    "customer_loyalty_operations",
]

# 要删除的物理文件（相对于 data/knowledge/）
FILES_TO_DELETE = [
    "product_sales_data.xlsx",
    "sales_report_with_tables.pdf",
    "hybrid_search_deep_dive.md",
    "langgraph_workflow.md",
    "multi_agent_orchestration.md",
    "retrieval_methods_comparison.md",
    "system_architecture.md",
    "chinook_database_schema.md",
]

def main():
    print("=" * 60)
    print("  清理向量数据库 — 删除不相关文档")
    print("=" * 60)

    db_path = os.path.join(PROJECT_ROOT, "vector_db", "chroma_db")
    if not os.path.exists(db_path):
        print(f"[ERROR] 数据库目录不存在: {db_path}")
        return

    client = chromadb.PersistentClient(
        path=db_path,
        settings=Settings(anonymized_telemetry=False, allow_reset=True)
    )

    try:
        collection = client.get_collection(
            name="knowledge_base",
            embedding_function=None
        )
    except Exception as e:
        print(f"[ERROR] 无法获取集合 knowledge_base: {e}")
        return

    total_before = collection.count()
    print(f"\n  清理前向量总数: {total_before}")

    # 逐个删除
    total_deleted = 0
    for doc_id in DOCS_TO_DELETE:
        try:
            results = collection.get(
                where={"doc_id": doc_id},
                include=[]
            )
            count = len(results['ids']) if results['ids'] else 0
            if count > 0:
                collection.delete(ids=results['ids'])
                print(f"  ✓ 删除 {doc_id}: {count} 个 chunks")
                total_deleted += count
            else:
                print(f"  - 跳过 {doc_id}: 不存在")
        except Exception as e:
            print(f"  ✗ 删除 {doc_id} 失败: {e}")

    total_after = collection.count()
    print(f"\n  清理后向量总数: {total_after}")
    print(f"  共删除: {total_deleted} 个 chunks")

    # 显示剩余文档
    print(f"\n  剩余文档列表:")
    try:
        all_data = collection.get(include=["metadatas"])
        remaining_docs = set()
        for meta in all_data['metadatas']:
            if meta and 'doc_id' in meta:
                remaining_docs.add(meta['doc_id'])
        for i, doc_id in enumerate(sorted(remaining_docs), 1):
            print(f"    {i}. {doc_id}")
    except Exception as e:
        print(f"    (查询失败: {e})")

    # 删除物理文件
    print(f"\n  删除物理文件:")
    knowledge_dir = os.path.join(PROJECT_ROOT, "data", "knowledge")
    for filename in FILES_TO_DELETE:
        filepath = os.path.join(knowledge_dir, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"    ✓ 删除 {filename}")
        else:
            print(f"    - 跳过 {filename}: 不存在")

    # 清理 document_metadata.json 中的残留条目
    print(f"\n  清理元数据:")
    metadata_path = os.path.join(PROJECT_ROOT, "data", "metadata", "document_metadata.json")
    if os.path.exists(metadata_path):
        import json
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        removed_keys = []
        for doc_id in DOCS_TO_DELETE:
            if doc_id in metadata:
                del metadata[doc_id]
                removed_keys.append(doc_id)
        # 同时清理已无物理文件的旧条目
        existing_files = set()
        if os.path.isdir(knowledge_dir):
            for fn in os.listdir(knowledge_dir):
                stem = os.path.splitext(fn)[0]
                existing_files.add(stem)
        orphans = [k for k in metadata if k not in existing_files]
        for k in orphans:
            del metadata[k]
            removed_keys.append(k)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        if removed_keys:
            print(f"    ✓ 清理了 {len(removed_keys)} 个元数据条目: {removed_keys}")
        else:
            print(f"    - 无需清理")

    print(f"\n{'=' * 60}")
    print(f"  Done!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
