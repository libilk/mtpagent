# -*- coding: utf-8 -*-
"""
LangGraph 节点函数定义
======================

包装现有 Agent 为 LangGraph 节点，不修改任何现有 Agent 代码。
所有节点函数接收 GraphState，返回需要更新的状态字段字典。

支持 StreamWriter 实时推送进度事件（stream_mode="custom"）。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import AIMessage
from langgraph.config import get_stream_writer

from langgraph_orchestrator.enhanced_state import EnhancedGraphState as GraphState
from core.protocol import AgentProtocol
from core.write_ops import consume_write_ops, clear_write_ops

logger = logging.getLogger(__name__)


def _get_writer() -> Optional[Callable]:
    """安全获取 StreamWriter，非流式调用时返回 None"""
    try:
        return get_stream_writer()
    except Exception:
        return None


def _emit(writer: Optional[Callable], agent: str, msg: str, **extra):
    """通过 StreamWriter 发射一条进度事件"""
    if writer is None:
        return
    event = {"event": "progress", "agent": agent, "msg": msg, "ts": time.time()}
    event.update(extra)
    writer(event)


# ---------------------------------------------------------------------------
# Agent 节点工厂
# ---------------------------------------------------------------------------

def make_agent_node(agent_instance: Any, agent_name: str, shared_memory=None, memory_store=None):
    """
    将任意 handle(query, context) -> str 的 Agent 包装为 LangGraph 节点。

    Args:
        agent_instance: Agent 实例，必须实现 AgentProtocol（有 handle 方法）
        agent_name: Agent 标识名（同时也是节点名）
        shared_memory: 共享记忆实例（已废弃，保留向后兼容）
        memory_store: 按 thread_id 隔离的记忆管理器（优先使用）

    Returns:
        LangGraph 节点函数（支持 StreamWriter 实时进度推送）

    Raises:
        TypeError: Agent实例未实现 AgentProtocol
    """
    # ========== Harness 契约校验（注册阶段拦截，避免运行时崩溃） ==========
    if not isinstance(agent_instance, AgentProtocol):
        raise TypeError(
            f"Agent '{agent_name}' ({type(agent_instance).__name__}) 未实现 AgentProtocol。"
            f"请确保该类定义了 handle(self, query: str, context: Dict[str, Any]) -> str 方法。"
        )

    def node_fn(state: GraphState) -> dict:
        writer = _get_writer()
        query = state["query"]

        _emit(writer, agent_name, f"开始执行", stage="start")

        # ========== 解析当前会话的记忆实例 ==========
        # 优先使用 memory_store（按 thread_id 隔离），兼容旧的 shared_memory
        session_memory = None
        thread_id = state.get("thread_id")
        if memory_store and thread_id:
            session_memory = memory_store.get(thread_id)
        elif shared_memory:
            session_memory = shared_memory

        # ========== 共享记忆：指代消解 ==========
        if session_memory:
            history = session_memory.get_messages()
            if history:
                _emit(writer, agent_name, "正在进行指代消解...", stage="anaphora")

                # 取最近4条消息构造上下文
                history_text = "\n".join([
                    f"{'用户' if msg['role'] == 'user' else '助手'}: {msg['content']}"
                    for msg in history[-4:]
                ])

                # 构造消解 prompt
                resolve_prompt = f"""请根据对话历史，判断当前查询是否存在指代不明或省略的情况，并进行消解。

对话历史：
{history_text}

当前查询：{query}

要求：
1. 如果查询中有代词（它、这个、那个、同款等），替换为对话历史中具体指代的内容
2. 如果查询是省略句（缺少主语或宾语），根据对话历史补全完整问题
3. 如果查询已经完整明确，直接返回原查询
4. 只返回最终的查询，不要解释

消解后的查询："""

                try:
                    resolved = agent_instance.llm.generate(
                        resolve_prompt, temperature=0.1, max_tokens=100
                    ).strip()
                    resolved = resolved.strip('"\'""''')
                    if resolved and resolved != query:
                        logger.info(f"[共享记忆消解] {agent_name}: {query} → {resolved}")
                        _emit(writer, agent_name, f"指代消解: {query} → {resolved}", stage="anaphora_done")
                        query = resolved
                except Exception as e:
                    logger.warning(f"[共享记忆消解] 失败: {e}")

        # 构建 context —— 与原始 executor._act() 保持兼容
        context: Dict[str, Any] = {
            "messages": state.get("messages", []),
            "dependencies": {
                r["agent"]: r for r in state.get("agent_results", [])
            },
            "original_query": state["query"],
        }

        # 多模态：将图片字段传入 context（供 VQA Agent 使用）
        for img_key in ("image_urls", "image_paths", "image_base64_list"):
            img_val = state.get(img_key)
            if img_val:
                context[img_key] = img_val

        # 文件上传：将文件路径传入 context（供 Excel Agent 等使用）
        file_paths = state.get("file_paths")
        if file_paths:
            context["file_paths"] = file_paths

        # 如有反馈（重试场景），注入到 query 中
        quality = state.get("quality_score", 1.0)
        iteration = state.get("iteration", 0)
        if quality < 0.7 and iteration > 0:
            feedback = state.get("feedback", "")
            if feedback:
                query = (
                    f"{query}\n\n"
                    f"【改进要求】请针对以下反馈改进：{feedback}"
                )
                _emit(writer, agent_name, f"重试(第{iteration}次)，改进中...", stage="retry")

        _emit(writer, agent_name, "正在调用Agent处理...", stage="executing")

        # 将历史注入 context，供 Agent ReAct 循环使用
        if session_memory:
            context["history"] = session_memory.get_messages()

        # 将 stream_writer 和 agent_name 注入 context，供 Agent ReAct 循环发射中间步骤事件
        context["_stream_writer"] = writer
        context["_agent_name"] = agent_name

        # 取出本次 Agent 调用期间登记的所有写操作（详见 core/write_ops.py）。
        # 必须在 handle() 之后、且在本线程内读取 —— 记录器是线程本地的。
        write_ops: List[Dict[str, Any]] = []
        try:
            result = agent_instance.handle(query, context)
            write_ops = consume_write_ops()
            _emit(writer, agent_name, "执行完成", stage="done")
            # 写入共享记忆
            if session_memory:
                session_memory.add_user_message(query)
                session_memory.add_assistant_message(str(result) if not isinstance(result, str) else result)
        except Exception as e:
            logger.error(f"Agent {agent_name} 执行失败: {e}")
            _emit(writer, agent_name, f"执行失败: {e}", stage="error")
            result = f"Agent执行出错: {e}"
            # 异常路径也要清空，否则残留的写操作会误触下一个请求的审批
            clear_write_ops()

        if write_ops:
            logger.info(f"[写操作] {agent_name} 本轮执行了 {len(write_ops)} 个写操作")

        return {
            "agent_results": [
                {
                    "agent": agent_name,
                    "result": result,
                    "iteration": iteration,
                }
            ],
            "messages": [
                AIMessage(content=str(result)[:8000], name=agent_name)
            ],
            "completed_task_ids": [state.get("current_task_id") or agent_name],
            "write_operations": write_ops,
        }

    # 让函数名在调试时更有辨识度
    node_fn.__name__ = f"{agent_name}_node"
    return node_fn


# ---------------------------------------------------------------------------
# 核心节点类
# ---------------------------------------------------------------------------

class GraphNodes:
    """
    图节点工厂，持有对现有编排组件的引用。

    使用方法::

        nodes = GraphNodes(planner, router, registry, llm)
        builder.add_node("planner", nodes.planner_node)
        builder.add_node("router", nodes.router_node)
        ...
    """

    def __init__(self, planner, router, registry, llm):
        """
        Args:
            planner: TaskPlanner 实例（来自 orchestrator/planner.py）
            router:  AgentRouter  实例（来自 orchestrator/router.py）
            registry: AgentRegistry 实例（来自 orchestrator/registry.py）
            llm: LLM 实例（来自 llm/llm_client.py，用于聚合和评估）
        """
        self.planner = planner
        self.router = router
        self.registry = registry
        self.llm = llm

    # ---- Complexity Classifier 节点（职责分离：只判断复杂度）----

    def complexity_classifier_node(self, state: GraphState) -> dict:
        """
        唯一职责：判断问题复杂度（使用 LLM）
        """
        writer = _get_writer()
        query = state["query"]

        _emit(writer, "complexity_classifier", "正在分析问题复杂度...", stage="start")

        # LLM 判断
        prompt = f"""判断以下问题是否需要多个**不同类型**的Agent协作完成。

问题：{query}

判断标准：
- 简单：一个Agent就能独立完成的任务。包括：
  · 知识问答（即使问题很长或涉及对比分析）
  · 数据库查询（即使涉及多表关联、聚合统计）
  · 文档解析、图片分析、Excel分析等
- 复杂：必须由多个**不同类型**的Agent协作才能完成。例如：
  · "查询数据库中的客户信息，并结合知识库分析改进建议"（需要数据库Agent + 知识Agent）
  · "分析这张图表并查询相关数据"（需要视觉Agent + 数据库Agent）

只回答"简单"或"复杂"："""

        try:
            result = self.llm.generate(prompt, temperature=0.1, max_tokens=10).strip()
            is_complex = "复杂" in result
            label = "复杂" if is_complex else "简单"
            logger.info("[LangGraph] 复杂度判断（LLM）: %s", label)
            _emit(writer, "complexity_classifier", f"判断结果: {label}查询", stage="done")
            return {"is_complex": is_complex}
        except Exception as e:
            logger.warning("[LangGraph] 复杂度判断失败，默认为简单: %s", e)
            _emit(writer, "complexity_classifier", "判断失败，默认简单查询", stage="error")
            return {"is_complex": False}

    # ---- Planner 节点（职责分离：只生成DAG计划）----

    def planner_node(self, state: GraphState) -> dict:
        """
        唯一职责：生成 DAG 任务计划（仅在复杂查询时调用）
        """
        writer = _get_writer()
        _emit(writer, "planner", "正在生成任务执行计划...", stage="start")

        # 构建规划用的 query：附加文件/图片上下文，帮助规划器安排正确的 Agent
        plan_query = state["query"]
        file_paths = state.get("file_paths", [])
        image_paths = state.get("image_paths", [])
        image_urls = state.get("image_urls", [])

        if file_paths:
            import os
            file_names = [os.path.basename(fp) for fp in file_paths]
            file_exts = [os.path.splitext(fn)[1].lower() for fn in file_names]
            plan_query += f"\n[用户已上传文件: {', '.join(file_names)}]"
            if any(ext in ('.xlsx', '.xls', '.csv') for ext in file_exts):
                plan_query += "\n[文件类型: Excel/CSV数据文件，需要数据分析]"
            elif any(ext in ('.pdf', '.docx', '.doc', '.md', '.txt') for ext in file_exts):
                plan_query += "\n[文件类型: 文档文件，需要文档解析分析]"
        if image_paths or image_urls:
            plan_query += "\n[用户已上传图片，必须安排 vqa_agent 进行图片内容识别和分析]"

        agent_descriptions = self.registry.get_agent_descriptions()
        plan = self.planner.plan(plan_query, agent_descriptions)

        task_count = len(plan.get("tasks", []))
        logger.info("[LangGraph] DAG规划: %d 个任务", task_count)

        task_summary = "; ".join(
            f"{t.get('agent_id', '?')}: {t.get('description', '')[:30]}"
            for t in plan.get("tasks", [])
        )
        _emit(writer, "planner", f"规划完成: {task_count}个任务 [{task_summary}]", stage="done")

        return {"plan": plan}

    def router_node(self, state: GraphState) -> dict:
        """唯一职责：选择最佳 Agent（仅在简单查询时调用）"""
        writer = _get_writer()
        _emit(writer, "router", "正在选择最佳Agent...", stage="start")

        # 构建任务描述：附加文件/图片上下文，帮助路由器做出更准确的判断
        task_desc = state["query"]
        file_paths = state.get("file_paths", [])
        image_paths = state.get("image_paths", [])
        image_urls = state.get("image_urls", [])

        if file_paths:
            import os
            file_names = [os.path.basename(fp) for fp in file_paths]
            file_exts = [os.path.splitext(fn)[1].lower() for fn in file_names]
            task_desc += f"\n[用户已上传文件: {', '.join(file_names)}]"
            # 明确提示文件类型以辅助路由（所有文档类型统一由 document_agent 处理）
            if any(ext in ('.xlsx', '.xls', '.csv') for ext in file_exts):
                task_desc += "\n[文件类型: Excel/CSV数据文件，需要文档处理Agent解析分析]"
            elif any(ext in ('.pdf', '.docx', '.doc') for ext in file_exts):
                task_desc += "\n[文件类型: 文档文件，需要文档处理Agent解析分析]"

        if image_paths or image_urls:
            task_desc += "\n[用户已上传图片，需要图片分析/识别]"

        task = {"description": task_desc}
        agents = self.registry.get_all_agents()
        agent_id = self.router.route(task, agents)

        logger.info("[LangGraph] Router 选择 Agent: %s", agent_id)
        _emit(writer, "router", f"已选择: {agent_id}", stage="done")

        return {"selected_agent": agent_id}

    # ---- Aggregator 节点 ----

    def aggregator_node(self, state: GraphState) -> dict:
        """汇总多个 Agent 的结果为最终答案。"""
        writer = _get_writer()
        results = state.get("agent_results", [])

        if not results:
            return {"final_answer": "未获得任何结果"}

        # ========== 波次检查：只在所有任务完成时才汇总 ==========
        plan = state.get("plan")
        if plan:
            tasks = plan.get("tasks", [])
            completed = set(state.get("completed_task_ids", []))
            if len(completed) < len(tasks):
                # 还有未完成任务，不汇总，直接透传（避免中间波次浪费 LLM 调用）
                _emit(writer, "aggregator", f"中间波次检查点 ({len(completed)}/{len(tasks)} 完成)", stage="checkpoint")
                return {}

        # ========== 所有任务完成或简单路由，执行汇总 ==========
        # 取最新一轮迭代的结果
        max_iter = max(r.get("iteration", 0) for r in results)
        latest = [r for r in results if r.get("iteration", 0) == max_iter]

        # 单结果直接返回
        if len(latest) == 1:
            answer = latest[0].get("result", "")
            if isinstance(answer, dict):
                answer = answer.get("result", str(answer))
            _emit(writer, "aggregator", "单Agent结果，直接输出", stage="done")
            return {"final_answer": str(answer)}

        # 多结果用 LLM 综合
        _emit(writer, "aggregator", f"正在综合{len(latest)}个Agent的结果...", stage="merging")

        # 提取并保留各 Agent 结果中的引用源标记（避免 LLM 综合时丢失）
        import re
        all_sources = []
        all_retrieval_stats = {}  # 保留检索统计（多个 Agent 的统计合并）
        clean_results = []
        for r in latest:
            result_text = str(r.get('result', ''))
            sources_match = re.search(r'<!-- SOURCES:(.*?):SOURCES -->', result_text, re.DOTALL)
            if sources_match:
                try:
                    import json as _json
                    sources_data = _json.loads(sources_match.group(1))
                    # 兼容新格式 {"sources": [...], "retrieval_stats": {...}} 和旧格式 [...]
                    if isinstance(sources_data, dict):
                        all_sources.extend(sources_data.get("sources", []))
                        # 保留检索统计（多个 Agent 时取第一个非空的）
                        stats = sources_data.get("retrieval_stats", {})
                        if stats and not all_retrieval_stats:
                            all_retrieval_stats = stats
                    elif isinstance(sources_data, list):
                        all_sources.extend(sources_data)
                except Exception:
                    pass
                # 从结果中剥离 SOURCES 标记，避免干扰 LLM 综合
                result_text = re.sub(r'\n*<!-- SOURCES:.*?:SOURCES -->', '', result_text, flags=re.DOTALL)
            clean_results.append({"agent": r["agent"], "result": result_text})

        # 限制每个 Agent 结果长度，防止综合 prompt 过长导致 LLM 调用失败
        MAX_PER_AGENT = 2000
        parts = "\n\n".join(
            f"【{r['agent']}】:\n{r['result'][:MAX_PER_AGENT]}" for r in clean_results
        )
        prompt = (
            f"请根据以下子任务的结果，综合回答用户的问题。\n\n"
            f"用户问题: {state['query']}\n\n"
            f"子任务结果:\n{parts}\n\n"
            f"请综合以上信息，给出完整、连贯的回答:"
        )
        try:
            answer = self.llm.generate(prompt, temperature=0.5, max_tokens=4096)
            # 重新附加引用源标记（去重、排序、取 top 3），保留检索统计
            if all_sources:
                import json as _json
                seen = set()
                unique_sources = []
                for s in sorted(all_sources, key=lambda x: x.get("score", 0), reverse=True):
                    key = s.get("source", "")
                    if key and key not in seen:
                        seen.add(key)
                        unique_sources.append(s)
                if unique_sources:
                    # 使用新格式（包含 retrieval_stats），保持与 _extract_sources 兼容
                    sources_payload = {"sources": unique_sources[:3]}
                    if all_retrieval_stats:
                        sources_payload["retrieval_stats"] = all_retrieval_stats
                    sources_json = _json.dumps(sources_payload, ensure_ascii=False)
                    answer = f"{answer}\n\n<!-- SOURCES:{sources_json}:SOURCES -->"
            _emit(writer, "aggregator", "结果综合完成", stage="done")
        except Exception as e:
            logger.error("LLM 综合失败: %s", e)
            _emit(writer, "aggregator", f"LLM综合失败，拼接输出", stage="error")
            answer = "\n\n".join(
                f"{r['agent']}: {r.get('result', '')}" for r in latest
            )

        return {"final_answer": answer}

    # ---- Evaluator 节点 ----

    def evaluator_node(self, state: GraphState) -> dict:
        """LLM 语义质量评估，复用 executor.py 中的评估 prompt 逻辑。"""
        writer = _get_writer()
        answer = state.get("final_answer", "")
        query = state.get("query", "")
        iteration = state.get("iteration", 0)

        if not answer:
            return {
                "quality_score": 0.0,
                "iteration": iteration + 1,
                "feedback": "未生成任何答案",
            }

        _emit(writer, "evaluator", "正在评估答案质量...", stage="start")

        try:
            prompt = (
                f"请评估以下回答的质量。\n\n"
                f"用户问题: {query}\n"
                f"回答内容: {str(answer)[:1000]}\n\n"
                f"评估维度（每项0-10分）：\n"
                f"1. 相关性：回答是否针对用户问题\n"
                f"2. 完整性：是否覆盖问题的关键方面\n"
                f"3. 准确性：是否有明显错误或矛盾\n\n"
                f'请只输出JSON格式：'
                f'{{"relevance": 分数, "completeness": 分数, '
                f'"accuracy": 分数, "feedback": "具体改进建议"}}'
            )

            from llm.output_parser import parse_llm_json

            response = self.llm.generate(
                prompt, temperature=0.1, max_tokens=200
            )
            scores = parse_llm_json(
                response,
                fallback={
                    "relevance": 7,
                    "completeness": 7,
                    "accuracy": 7,
                    "feedback": "",
                },
            )

            # 加权平均（与 executor.py _llm_evaluate 一致）
            total = (
                scores.get("relevance", 7) * 0.4
                + scores.get("completeness", 7) * 0.3
                + scores.get("accuracy", 7) * 0.3
            ) / 10

            feedback_parts = []
            if scores.get("relevance", 10) < 6:
                feedback_parts.append("回答偏离了任务要求，请更聚焦于问题本身")
            if scores.get("completeness", 10) < 6:
                feedback_parts.append("回答不够完整，请补充遗漏的方面")
            if scores.get("accuracy", 10) < 6:
                feedback_parts.append("回答可能存在不准确的内容，请验证后修正")
            custom = scores.get("feedback", "")
            if custom:
                feedback_parts.append(custom)

            feedback = "；".join(feedback_parts)
            logger.info("[LangGraph] Evaluator 评分: %.2f", total)
            _emit(writer, "evaluator", f"质量评分: {total:.2f}", stage="done", score=total)

        except Exception as e:
            logger.warning("LLM 评估失败: %s，使用基础评分", e)
            total = 0.7 if len(str(answer)) > 50 else 0.5
            feedback = ""
            _emit(writer, "evaluator", f"评估降级，基础评分: {total:.2f}", stage="fallback")

        return {
            "quality_score": total,
            "iteration": iteration + 1,
            "feedback": feedback,
        }
