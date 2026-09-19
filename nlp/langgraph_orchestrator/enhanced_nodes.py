# -*- coding: utf-8 -*-
"""
增强节点函数
============

包含重复检测、人工介入等高级功能 —— 三类节点都在 enhanced_graph.py 里被连进图：
- duplicate_detection（重复检测）
- human_intervention_check：只判断"要不要拦人"，不碰数据
- human_intervention_execute：拿到真人答复（human_feedback）后决定继续 / 重试 / 终止

为什么人工介入要拆成 check / execute 两个节点：暂停是在「节点边界」上生效的
（见 enhanced_graph.py 的 interrupt_before），停不进一个节点的内部 ——
所以"判断要不要拦"和"收到答复后怎么走"只能分家，暂停点正好卡在两者之间。
"""

import logging
from typing import Dict, Any, Optional
import time

logger = logging.getLogger(__name__)

# 判定"语义重复"的余弦相似度（cosine similarity）阈值。定得这么高（0.95）是因为
# 误判代价不对称：把不重复的判成重复会直接熔断、丢掉本可用的答案，宁漏勿错。
_SEMANTIC_DUPLICATE_THRESHOLD = 0.95


def _cosine_similarity(vec1, vec2) -> float:
    """余弦相似度：比较两个向量方向有多接近，取值 0~1，越接近 1 越像。手写，不引第三方库。

    vec 里全是 0（零向量）时方向无意义、且会除零，约定返回 0.0 表示"不相似"。
    """
    import numpy as np
    v1, v2 = np.array(vec1), np.array(vec2)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def _emit(agent: str, msg: str, **extra):
    """往流式通道推一条进度事件（StreamWriter = 流式事件写入器），前端靠它显示"正在检测重复…"。

    取不到 writer 就静默跳过：非流式调用下根本没有这个通道，不该因此报错、更不该打断流程。
    """
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

    检测策略（分层，从便宜到贵）：
      第1层 - 精确匹配：前500字符完全一致 → 零成本，不碰任何模型
      第2层 - 语义相似度：embedding（向量化）后算余弦相似度 >= 0.95
              → 拦的是"换了措辞但内容相同"的重复，精确匹配抓不到这种

    分层是为了省钱：第 1 层能定案就直接返回，只有它没命中才值得去调向量模型。
    为什么要检测重复：重试会把同一问题再跑一遍，产出若和上一轮一致，
    再评审、再重试都只是白花钱，直接熔断更划算。

    Args:
        embedder: 向量化模型实例（来自 llm.embedder），为 None 时只剩第 1 层
    """

    def duplicate_detection_node(state: Dict[str, Any]) -> dict:
        # agent_results 是累加字段（reducer 见 enhanced_state.py），所以这里翻得到历史轮次的结果
        agent_results = state.get("agent_results", [])
        if not agent_results:
            return {"has_duplicate": False}  # 还没跑出结果，谈不上重复

        latest_result = agent_results[-1]  # 只比最近一次，不和全部历史两两比
        result_text = str(latest_result.get("result", ""))
        # 指纹（fingerprint）= 取前 500 字符当廉价标识。截断是有意的：既省比较开销，
        # 又让"只有尾部长短/措辞不同"的两段结果仍算重复。
        result_fingerprint = result_text[:500]

        previous_results = state.get("previous_results", [])
        previous_embeddings = state.get("previous_result_embeddings", [])
        # 两个历史列表长度未必相等：第 2 层没跑成（embedder 为 None 或抛异常）时只留了指纹

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
        # embedder 为 None 时整层消失：此时只剩精确匹配，抓不住"换说法"的重复
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
                # 降级而不中断：检测不出重复的代价只是"当不重复继续跑"，比整轮报错划算
                logger.warning("[DuplicateDetection] 语义检测异常，跳过第2层: %s", e)

        # ---- 未命中：记录指纹和向量，继续执行 ----
        # 这两个字段没挂 reducer，累积靠"读出来 → 原地改 → 随返回值写回"；
        # 少了这步，下一轮就查不到历史，重复检测形同虚设。
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
    """判断本轮要不要暂停、等真人介入。只读 state，不执行任何动作。

    里面串着五条互不相干的触发通道，按顺序判断、命中一条就返回（后面的不再看）：
      1. 复杂任务的规划还没被审过（plan 为 complex 且 plan_approved 未置位）
      2. Agent 本轮真改了数据（write_operations 非空且未批准）—— 唯一可靠的写操作闸门
      3. 用户原话里含数据库写关键字、且当前 Agent 名带 database（几乎不命中，见下方注释）
      4. 质量分过低且已重试多轮
      5. critic 校验失败
    返回 required=True 后，图会在 execute 节点前停下等人工输入。

    每条通道给的 intervention_data["options"] 是给前端渲染按钮的，真人点哪个，
    由前端翻译成 human_feedback 传回来。
    """
    # 默认值一律取"无需介入"那一侧：字段缺失时宁可放行，也别凭空拦一道把人叫来
    quality_score = state.get("quality_score", 1.0)
    iteration = state.get("iteration", 0)
    critic_passed = state.get("critic_passed", True)
    selected_agent = state.get("selected_agent") or ""
    plan = state.get("plan", {})

    _emit("human_intervention_check", "检查是否需要人工介入...", stage="start")

    # 通道 1：复杂规划先过目。plan_approved 由人工批准后置位（见 enhanced_entry.py 的 resume 分支）
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
    # 读者视角提醒：文件里两条"写操作"通道长得很像，认准这条 ——
    # 它看的是程序"实际做了什么"（write_ops.py 登记的真实改动），下面那条看的是"用户说了什么"。
    # 命中与否只取决于"数据到底有没有被改"，跟用户措辞、Agent 叫什么名字都无关。
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
    #
    # 通道 3，实际很难命中：它查的是用户**中文原话**里有没有 INSERT/UPDATE 等词，
    # 而真人只会说"我要退货"。真正兜住写操作的是上面 write_operations 那条。
    if selected_agent and "database" in selected_agent.lower() and not state.get("db_operation_approved"):
        # .upper() 只是为了对齐英文关键字的大小写，对中文原话没有任何作用
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

    # 通道 4a：两个条件都要满足 —— 分数过低(<0.4) 且已经重试到第 2 轮以上
    if quality_score < 0.4 and iteration >= 2:
        _emit("human_intervention_check", f"触发: 质量过低({quality_score:.2f})且已重试{iteration}次", stage="triggered")
        return {
            "human_intervention_required": True,
            "intervention_reason": "质量评估未通过且已重试多次",
            "intervention_data": {"score": quality_score, "iteration": iteration, "result": state.get("final_answer", ""), "options": {"accept": "接受当前结果", "retry": "重新执行", "abort": "终止任务"}}
        }

    # 通道 4b：critic（评审）不通过。注意 critic 查的是"这段输出能不能接上后续任务"，
    # 不是给答案对错打分 —— 别和 evaluator 的 quality_score 混为一谈。
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
    """处理人工答复：把前端传回的 human_feedback 归一成"继续 / 重试 / 终止"三件事。

    能走到这里，说明 interrupt_before 刚把图放行；human_feedback 是
    resume_after_human_input 在恢复前写进 state 的（前端五种按钮值 → 这里收敛成三种）。

    事实说明：归一的结果写在 human_decision 字段上，但全项目没有任何地方读它 ——
    真正改变走向的是 resume 时按 intervention_reason 置下的审批标志
    （plan_approved / write_approved / db_operation_approved），以及这里清空
    intervention_reason 后，下一轮 check_quality 的判定。如实标注，未改动逻辑。
    """
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
        # 兜底：认不出的答复（含空值）一律当"继续"，避免误终止把结果丢掉
        decision = "continue"

    _emit("human_intervention_execute", f"人工反馈已处理: {human_feedback or '默认继续'}", stage="done")
    # 顺手清空中断现场：state 在同一 thread_id 下是续用的，不清会把上一次的中断原因留给下一轮
    return {
        "human_intervention_required": False,
        "intervention_reason": None,
        "intervention_data": None,
        "human_decision": decision
    }
