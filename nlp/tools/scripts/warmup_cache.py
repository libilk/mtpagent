# -*- coding: utf-8 -*-
"""
面试演示缓存预热脚本
====================

在面试前运行此脚本，将常用 demo 查询预跑一遍，
自然填充所有缓存层（查询缓存、语义缓存、检索缓存、向量缓存、LLM缓存）。
后续相同或语义相似的查询可秒回。

用法:
    python tools/scripts/warmup_cache.py          # 预热所有 demo 查询
    python tools/scripts/warmup_cache.py --check   # 仅检查缓存状态
    python tools/scripts/warmup_cache.py --clear   # 清空所有缓存
"""

import os
import sys
import time
import logging
from typing import List, Dict

# 设置项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

# 配置日志（预热时只显示关键信息）
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("warmup")
logger.setLevel(logging.INFO)

# ========== 面试演示查询 ==========
# 按类型分组，方便选择性预热

DEMO_QUERIES: List[Dict] = [
    # --- 简单路由：知识检索 ---
    {
        "label": "🔍 知识检索（简单路由）",
        "query": "RAG系统有哪些核心优化策略？",
        "type": "simple",
    },
    # --- 简单路由：数据库查询 ---
    {
        "label": "🗄️ 数据库查询（SQL生成）",
        "query": "从数据库查询销售额最高的前5位客户及其购买的曲目数量",
        "type": "simple",
    },
    # --- 复杂任务：DAG 编排 ---
    {
        "label": "🧩 复杂任务（DAG编排）#1",
        "query": "从数据库查询客户Eduardo Martins的消费记录和购买偏好，并从知识库检索客户满意度提升策略，给出综合建议",
        "type": "complex",
    },
    {
        "label": "🧩 复杂任务（DAG编排）#2",
        "query": "从数据库查询销售额前3的客户信息，并从知识库检索客户留存最佳实践，给出针对性策略",
        "type": "complex",
    },
]


def warmup(queries: List[Dict] = None):
    """
    执行缓存预热

    将每个 demo 查询实际跑一遍，自然填充所有缓存层：
    - knowledge_agent: query_cache + semantic_cache + retrieval_cache + embedding_cache
    - database_agent: 内部 LLM 缓存
    - 编排层: LLM 缓存（复杂度分类、任务规划）
    """
    from langgraph_orchestrator.enhanced_entry import EnhancedLangGraphRAGSystem

    queries = queries or DEMO_QUERIES

    print("\n" + "=" * 60)
    print("  🔥 面试演示缓存预热")
    print("=" * 60)
    print(f"\n待预热查询: {len(queries)} 个\n")

    # 初始化系统
    print("⏳ 正在初始化 RAG 系统...")
    init_start = time.time()
    system = EnhancedLangGraphRAGSystem(
        auto_update_index=False,
        enable_checkpointer=True,
        enable_parameter_validation=True,
        enable_critic=False,
    )
    init_time = time.time() - init_start
    print(f"✅ 系统初始化完成 ({init_time:.1f}s)\n")

    # 逐个预热
    total_start = time.time()
    results = []

    for i, demo in enumerate(queries, 1):
        label = demo["label"]
        query = demo["query"]
        query_type = demo["type"]

        print(f"[{i}/{len(queries)}] {label}")
        print(f"    查询: {query[:60]}{'...' if len(query) > 60 else ''}")

        start = time.time()
        try:
            # 使用非流式方式执行（更简洁）
            result = system.handle_query(
                query=query,
                thread_id=f"warmup_{i}",  # 独立 thread 避免干扰
            )

            elapsed = time.time() - start
            success = result.get("success", False)
            score = result.get("quality_score", 0)
            answer_preview = (result.get("result", "") or "")[:80]

            status = "✅" if success else "⚠️"
            print(f"    {status} 完成 | 耗时: {elapsed:.1f}s | 评分: {score:.2f}")
            print(f"    预览: {answer_preview}...")

            results.append({
                "label": label,
                "success": success,
                "time": elapsed,
                "score": score,
            })

        except Exception as e:
            elapsed = time.time() - start
            print(f"    ❌ 失败 | 耗时: {elapsed:.1f}s | 错误: {e}")
            results.append({
                "label": label,
                "success": False,
                "time": elapsed,
                "score": 0,
            })

        print()

    # 汇总
    total_time = time.time() - total_start
    success_count = sum(1 for r in results if r["success"])

    print("=" * 60)
    print(f"  📊 预热完成")
    print(f"  成功: {success_count}/{len(results)}")
    print(f"  总耗时: {total_time:.1f}s")
    print(f"  平均耗时: {total_time / len(results):.1f}s / 查询")
    print("=" * 60)

    # 验证缓存命中
    print("\n🔄 验证缓存命中效果...\n")
    verify_start = time.time()

    for i, demo in enumerate(queries, 1):
        start = time.time()
        try:
            result = system.handle_query(
                query=demo["query"],
                thread_id=f"warmup_verify_{i}",
            )
            elapsed = time.time() - start
            cached = elapsed < 2.0  # 缓存命中通常 < 2s
            icon = "⚡" if cached else "🐢"
            print(f"  {icon} [{elapsed:.2f}s] {demo['label']}")
        except Exception as e:
            elapsed = time.time() - start
            print(f"  ❌ [{elapsed:.2f}s] {demo['label']} - {e}")

    verify_time = time.time() - verify_start
    print(f"\n  验证总耗时: {verify_time:.1f}s (预热前约 {total_time:.0f}s)")
    print(f"  加速比: {total_time / max(verify_time, 0.1):.1f}x 🚀")
    print()


def check_cache_stats():
    """检查当前缓存状态"""
    from langgraph_orchestrator.enhanced_entry import EnhancedLangGraphRAGSystem

    print("\n⏳ 正在初始化系统以读取缓存状态...\n")
    system = EnhancedLangGraphRAGSystem(
        auto_update_index=False,
        enable_checkpointer=False,
        enable_parameter_validation=False,
        enable_critic=False,
    )

    print("=" * 60)
    print("  📊 缓存状态")
    print("=" * 60)

    # Agent 级别缓存
    try:
        agents = system.registry.get_all_agents()
        for agent_info in agents:
            agent = agent_info.get("instance")
            if agent and hasattr(agent, "get_optimization_stats"):
                stats = agent.get_optimization_stats()
                if stats.get("optimization_enabled"):
                    cache_stats = stats.get("cache_stats", {})
                    print(f"\n  [{agent_info['id']}]")
                    for layer, layer_stats in cache_stats.items():
                        if isinstance(layer_stats, dict):
                            size = layer_stats.get("size", "?")
                            hits = layer_stats.get("hits", 0)
                            misses = layer_stats.get("misses", 0)
                            hit_rate = layer_stats.get("hit_rate", "N/A")
                            print(f"    {layer}: size={size}, hits={hits}, misses={misses}, rate={hit_rate}")

            # 语义缓存
            if agent and hasattr(agent, "semantic_cache") and agent.semantic_cache:
                sc_stats = agent.semantic_cache.get_stats()
                print(f"    semantic: size={sc_stats['size']}, hits={sc_stats['hits']}, rate={sc_stats['hit_rate']}")
    except Exception as e:
        print(f"  ⚠️ 获取 Agent 缓存失败: {e}")

    # LLM 缓存
    for name, llm in [("llm_max", system.llm_max), ("llm_plus", system.llm_plus)]:
        if hasattr(llm, "llm_cache") and llm.llm_cache:
            stats = llm.llm_cache.get_stats()
            print(f"\n  [{name}] LLM缓存")
            print(f"    size={stats.get('size', '?')}, hits={stats.get('hits', 0)}, rate={stats.get('hit_rate', 'N/A')}")

    # Redis 状态
    redis_status = "未配置"
    if hasattr(system, "redis_client") and system.redis_client:
        try:
            system.redis_client.ping()
            redis_status = "已连接 ✅"
        except Exception:
            redis_status = "连接失败 ❌"
    print(f"\n  Redis: {redis_status}")
    print()


def clear_cache():
    """清空所有缓存"""
    from langgraph_orchestrator.enhanced_entry import EnhancedLangGraphRAGSystem

    print("\n⏳ 正在初始化系统...\n")
    system = EnhancedLangGraphRAGSystem(
        auto_update_index=False,
        enable_checkpointer=False,
        enable_parameter_validation=False,
        enable_critic=False,
    )

    # 清空 Agent 缓存
    try:
        agents = system.registry.get_all_agents()
        for agent_info in agents:
            agent = agent_info.get("instance")
            if agent:
                if hasattr(agent, "cache_manager"):
                    agent.cache_manager.clear_all()
                    print(f"  ✅ {agent_info['id']} cache_manager 已清空")
                if hasattr(agent, "semantic_cache") and agent.semantic_cache:
                    agent.semantic_cache.clear()
                    print(f"  ✅ {agent_info['id']} semantic_cache 已清空")
    except Exception as e:
        print(f"  ⚠️ 清空 Agent 缓存失败: {e}")

    # 清空 LLM 缓存
    for name, llm in [("llm_max", system.llm_max), ("llm_plus", system.llm_plus)]:
        if hasattr(llm, "llm_cache") and llm.llm_cache:
            llm.llm_cache.clear()
            print(f"  ✅ {name} LLM 缓存已清空")

    print("\n  所有缓存已清空 🗑️\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="面试演示缓存预热")
    parser.add_argument("--check", action="store_true", help="仅查看缓存状态")
    parser.add_argument("--clear", action="store_true", help="清空所有缓存")
    args = parser.parse_args()

    if args.check:
        check_cache_stats()
    elif args.clear:
        clear_cache()
    else:
        warmup()
