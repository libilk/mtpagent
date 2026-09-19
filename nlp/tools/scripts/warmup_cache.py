# -*- coding: utf-8 -*-
"""
面试演示缓存预热脚本
====================

要解决的问题：一次查询要串起「向量化 → 检索 → LLM 生成」好几步远程调用，
冷启动时**第一个真实用户要独自承担全部耗时**（还有对应的 token 成本）。
预热的思路是"自己先当一次那个用户"：把 demo 查询预跑一遍，
让各层缓存（查询缓存 cache、语义缓存 semantic cache、检索缓存、向量缓存、LLM缓存）
里提前有货，轮到真人提问时直接命中。

⚠️ 但这个办法生效有个前提：**缓存得能跨进程活下来。**
   这几层缓存只有在 Redis 可用时才会落盘、才对"下一个进程"可见。
   本机 .env 里 REDIS_URL 是注释掉的、本地也没有 Redis 在跑 ——
   此时缓存是纯内存的：本脚本自己看着很快（预热和验证跑在同一个进程里，缓存自然还在），
   但你随后另起的那个服务进程拿到的是空缓存，预热等于白做。
   到底连没连上 Redis，看 `--check` 输出末行的「Redis: ...」。

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
# （"type" 里的 simple/complex 只是给人看的标签，不会强制走哪条路径 ——
#   实际走简单还是复杂由复杂度分类器现场判断，这里别当成路由开关。）

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

    注意这里"填充"是被动发生的 —— 没有一行代码去写缓存，
    是各层原有的"查不到就算一次、把结果顺手存下"逻辑在跑真实查询时自己填的。
    所以只要某条链路这次没被走到（例如问题没触发数据库 Agent），那层就是空的。
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
    # auto_update_index=False：预热只读不写知识库。若开着，初始化时会去碰
    #   incremental_update（见该脚本文件头的坑），可能顺手把不该索引的文档写进向量库。
    # enable_checkpointer=True：保持和真实服务一致，否则跑出来的耗时不代表线上。
    # enable_critic=False：critic 是死代码（类在、但从未注册进图），开着也不会有节点。
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
            # thread_id（会话 ID）每个查询给一个独立的：避免多个 demo 共享同一份
            # 对话记忆互相污染。反过来说，缓存的命中**不依赖** thread ——
            # 各缓存层是跨会话共享的，这正是预热能对别的会话生效的原因。
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
    # 读法：下面这个"加速比"量的是**同一个进程内**第二遍比第一遍快多少。
    # 它偏乐观 —— 这里换了 thread_id，但缓存跨会话共享所以仍会命中；
    # 真正决定"另起一个服务进程还有没有缓存"的是 Redis，不是这个数字。
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
            # 这只是"耗时 < 2s"的近似判断，不是真的缓存命中计数 ——
            # 网络偶然变快、答案变短都可能低于 2s。想看真实命中率/命中次数用 --check。
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
    """检查当前缓存状态

    输出按"谁的缓存"分组：每个 Agent 一段（逐层列 size/hits/misses/rate），
    再单独列 LLM 缓存和 Redis 连接状态。
    读法：hits>0 才说明这层真被命中过；size 大但 hits=0 只说明"填过、没用上"。
    因为统计归属进程，这个命令本身会新建一个进程 —— 拿它看"上一个进程填的缓存"
    只有在 Redis 可用时才有意义。
    """
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
    """清空所有缓存

    注意"清空"的范围就是**本进程刚 new 出来的这堆实例**。
    Redis 可用时这些 clear 会连带清掉 Redis 里的键，影响后续所有进程；
    Redis 不可用（本机默认情况）时就只是清了份内存副本、进程一退全没，
    不会影响任何别的东西 —— 所以"清完再看还是空的"是正常的，不是没清掉。
    """
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
