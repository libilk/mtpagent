# -*- coding: utf-8 -*-
"""
增强节点函数
============

包含重复检测、人工介入等高级功能
"""

import logging
from typing import Dict, Any, Optional
import time

logger = logging.getLogger(__name__)

_SEMANTIC_DUPLICATE_THRESHOLD = 0.95


def _cosine_similarity(vec1, vec2) -> float:
    import numpy as np
    v1, v2 = np.array(vec1), np.array(vec2)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def _emit(agent: str, msg: str, **extra):
    try:
        from langgraph.config import get_stream_writer
        writer = get_stream_writer()
        if writer:
            event = {"event": "progress", "agent": agent, "msg": msg, "ts": time.time()}
            event.update(extra)
            writer(event)
    except Exception:
        pass


def make_duplicate_detection_node(embedder=None):
    """
    工厂函数：创建重复检测节点

    检测策略（分层）：
      第1层 - 精确匹配：前500字符完全一致 → 零成本快速拦截
      第2层 - 语义相似度：Embedding 余弦相似度 >= 0.95 → 拦截换了措辞但内容相同的重复

    Args:
        embedder: Embedder 实例（来自 llm.embedder），为 None 时退化为仅精确匹配
    """

    def duplicate_detection_node(state: Dict[str, Any]) -> dict:
        agent_results = state.get("agent_results", [])
        if not agent_results:
            return {"has_duplicate": False}

        latest_result = agent_results[-1]
        result_text = str(latest_result.get("result", ""))
        result_fingerprint = result_text[:500]

        previous_results = state.get("previous_results", [])
        previous_embeddings = state.get("previous_result_embeddings", [])

        _emit("duplicate_detection", "正在检测重复结果...", stage="start")

        # ---- 第1层：精确匹配（零成本快速拦截） ----
        if result_fingerprint in previous_results:
            logger.info("[DuplicateDetection] 第1层命中：精确匹配重复，停止重试")
            _emit("duplicate_detection", "精确匹配命中，跳过重复结果", stage="done")
            return {
                "has_duplicate": True,
                "stop_reason": "duplicate_exact_match"
            }

        # ---- 第2层：语义相似度（Embedding） ----
        new_embedding = None
        if embedder:
            try:
                new_embedding = embedder.embed(result_fingerprint)
                for i, prev_emb in enumerate(previous_embeddings):
                    sim = _cosine_similarity(new_embedding, prev_emb)
                    if sim >= _SEMANTIC_DUPLICATE_THRESHOLD:
                        logger.info(
                            "[DuplicateDetection] 第2层命中：语义相似度 %.4f >= %.2f，判定为重复",
                            sim, _SEMANTIC_DUPLICATE_THRESHOLD
                        )
                        _emit("duplicate_detection", f"语义相似度 {sim:.2f} 命中，跳过重复结果", stage="done")
                        return {
                            "has_duplicate": True,
                            "stop_reason": f"duplicate_semantic(similarity={sim:.4f})"
                        }
            except Exception as e:
                logger.warning("[DuplicateDetection] 语义检测异常，跳过第2层: %s", e)

        # ---- 未命中：记录指纹和向量，继续执行 ----
        previous_results.append(result_fingerprint)
        if new_embedding is not None:
            previous_embeddings.append(new_embedding)

        _emit("duplicate_detection", "无重复，继续执行", stage="done")
        return {
            "has_duplicate": False,
            "previous_results": previous_results,
            "previous_result_embeddings": previous_embeddings,
        }

    return duplicate_detection_node


def human_intervention_check_node(state: Dict[str, Any]) -> dict:
    quality_score = state.get("quality_score", 1.0)
    iteration = state.get("iteration", 0)
    critic_passed = state.get("critic_passed", True)
    selected_agent = state.get("selected_agent") or ""
    plan = state.get("plan", {})

    _emit("human_intervention_check", "检查是否需要人工介入...", stage="start")

    if plan and plan.get("complexity") == "complex" and not state.get("plan_approved"):
        _emit("human_intervention_check", "触发: 复杂任务规划需审核", stage="triggered")
        return {
            "human_intervention_required": True,
            "intervention_reason": "任务规划完成，等待人工审核",
            "intervention_data": {"plan": plan, "options": {"approved": "确认规划，继续执行", "abort": "终止任务"}}
        }

    # ---------- 写操作审批（阶段 3 新增）----------
    #
    # 【为什么需要这条通道】
    # 下面那段"数据库写操作审核"覆盖不到真正的写操作，两个原因：
    #   1. 它要求 Agent 名里含 "database"，但建工单/提交退货发生在售后 Agent 内部；
    #   2. 它检查的是**用户那句中文**里有没有 INSERT/UPDATE 关键字，
    #      而用户说的是"我要退货"，永远不可能命中。
    #
    # 所以新增这条：Agent 执行真正的写操作时通过 core/write_ops.py 登记，
    # Agent 节点把登记结果写进 state["write_operations"]，这里据此触发审批。
    # 它不依赖用户措辞、也不依赖 Agent 的名字 —— 只看"数据是不是真的被改了"。
    write_ops = state.get("write_operations") or []
    if write_ops and not state.get("write_approved"):
        _emit("human_intervention_check", "触发: 检测到写操作需人工确认", stage="triggered")

        # 把写操作整理成人看得懂的一句话，展示给审批人
        summaries = []
        for op in write_ops:
            detail = op.get("detail", {}) or {}
            if op.get("type") == "submit_return_request":
                summaries.append(
                    f"提交{detail.get('refund_type', '退货')}申请："
                    f"订单 {detail.get('order_id')}，"
                    f"金额 {detail.get('amount')} 元，"
                    f"单号 {detail.get('refund_id')}"
                )
            elif op.get("type") == "create_ticket":
                summaries.append(
                    f"创建售后工单 {detail.get('ticket_id')}，"
                    f"分类 {detail.get('category')}，优先级 {detail.get('priority')}"
                )
            else:
                summaries.append(f"{op.get('type')}: {detail}")

        return {
            "human_intervention_required": True,
            "intervention_reason": "已完成写操作，等待人工确认",
            "intervention_data": {
                "operations": write_ops,
                "summary": summaries,
                "options": {
                    "approved": "确认无误，放行给用户",
                    "abort": "不认可，终止本次回复",
                },
            },
        }

    # 数据库写操作审核（仅当查询中包含写操作关键词时触发，SELECT查询无需审核）
    if selected_agent and "database" in selected_agent.lower() and not state.get("db_operation_approved"):
        query = state.get("query", "").upper()
        write_keywords = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE"]
        is_write_operation = any(kw in query for kw in write_keywords)
        if is_write_operation:
            _emit("human_intervention_check", "触发: 数据库写操作需审核", stage="triggered")
            return {
                "human_intervention_required": True,
                "intervention_reason": "即将执行数据库写入操作",
                "intervention_data": {"agent": selected_agent, "options": {"approved": "确认执行数据库操作", "abort": "取消操作"}}
            }

    if quality_score < 0.4 and iteration >= 2:
        _emit("human_intervention_check", f"触发: 质量过低({quality_score:.2f})且已重试{iteration}次", stage="triggered")
        return {
            "human_intervention_required": True,
            "intervention_reason": "质量评估未通过且已重试多次",
            "intervention_data": {"score": quality_score, "iteration": iteration, "result": state.get("final_answer", ""), "options": {"accept": "接受当前结果", "retry": "重新执行", "abort": "终止任务"}}
        }

    if not critic_passed:
        _emit("human_intervention_check", "触发: Critic验证失败", stage="triggered")
        return {
            "human_intervention_required": True,
            "intervention_reason": "Critic 验证失败",
            "intervention_data": {"issues": state.get("critic_issues", []), "options": {"override": "忽略冲突，强制通过", "retry": "重新执行有问题的 Agent", "abort": "终止任务"}}
        }

    _emit("human_intervention_check", "无需人工介入，正常放行", stage="done")
    return {"human_intervention_required": False}


def human_intervention_execute_node(state: Dict[str, Any]) -> dict:
    human_feedback = state.get("human_feedback", "").strip().lower()
    if human_feedback:
        logger.info("[HumanIntervention] 收到人工反馈: %s", human_feedback)

    if human_feedback == "abort":
        decision = "abort"
    elif human_feedback in ("approved", "override", "accept"):
        decision = "continue"
    elif human_feedback == "retry":
        decision = "retry"
    else:
        decision = "continue"

    _emit("human_intervention_execute", f"人工反馈已处理: {human_feedback or '默认继续'}", stage="done")
    return {
        "human_intervention_required": False,
        "intervention_reason": None,
        "intervention_data": None,
        "human_decision": decision
    }
