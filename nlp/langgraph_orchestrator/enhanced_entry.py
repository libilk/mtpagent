# -*- coding: utf-8 -*-
"""
LangGraph（图编排框架）RAG（检索增强生成）系统主入口
====================================================

本文件是整个系统的门面（facade）：外部（api.py / main.py）只跟
EnhancedLangGraphRAGSystem 打交道，不直接碰图、Agent（智能体）或检索器。
一个实例的生命周期就三步，按这个顺序读：
  1. __init__ 注册 Agent（_register_agents）并拼装依赖（LLM / 向量化模型 / 记忆）
  2. 把注册结果交给 build_enhanced_graph 连成图（连边逻辑在 enhanced_graph.py）
  3. handle_query 系列跑图；被 interrupt_before 拦下时，
     由 resume_after_human_input 从断点接着跑

基于 LangGraph 框架的多 Agent 编排系统。
支持状态持久化、断点续传、人机共驾、参数对齐验证等高级功能。

为什么查询入口有四个：跑图用的 API 不同（invoke / stream / astream），分别对应
非流式接口、CLI 同步流式、FastAPI 异步流式；resume 是共用的人工恢复通道。
四者的字段与流程高度重复，改一个记得同步另外两个，否则 CLI 与 Web 会悄悄分叉。
"""

import os
import re
import time
import json as _json
import logging
from typing import Dict, Any, AsyncIterator, Iterator, List, Tuple

import yaml
from langchain_core.messages import HumanMessage, BaseMessage

from langgraph_orchestrator.enhanced_graph import build_enhanced_graph

logger = logging.getLogger(__name__)


def _sanitize_output(obj):
    """
    递归清洗输出，将 LangChain Message 对象转为可 JSON 序列化的格式

    AIMessage / HumanMessage 等 BaseMessage 子类无法被 json.dumps 直接序列化，
    需要转为普通 dict 或 str。

    只在"出口"调用：图内部存的是 LangChain 对象，Web 层最后要 json 序列化，这里做边界转换。
    """
    if isinstance(obj, BaseMessage):
        return {"role": getattr(obj, "type", "unknown"), "content": obj.content}
    if isinstance(obj, dict):
        return {k: _sanitize_output(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_output(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_sanitize_output(item) for item in obj)
    # 其他不可序列化对象兜底转 str
    try:
        import json
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


# ===== 引用源标记提取 =====
# 机制：引用溯源（citation）不单独走 state 字段，而是塞进回答文本末尾的 HTML 注释
# <!-- SOURCES:{...}:SOURCES -->，跟着 final_answer 一起流转，到出口才剥掉、不漏给用户。
_SOURCES_RE = re.compile(r'\n*<!-- SOURCES:(.*?):SOURCES -->', re.DOTALL)


def _extract_sources(text: str) -> Tuple[str, List[Dict], Dict]:
    """
    从文本中提取并剥离 <!-- SOURCES:...:SOURCES --> 标记

    支持两种格式：
    - 新格式：{"sources": [...], "retrieval_stats": {...}}
    - 旧格式（向后兼容）：[...]

    返回 (纯净文本, 引用源列表, 检索统计)。
    """
    if not text or '<!-- SOURCES:' not in text:
        return text, [], {}

    sources: List[Dict] = []
    retrieval_stats: Dict = {}
    match = _SOURCES_RE.search(text)
    if match:
        try:
            parsed = _json.loads(match.group(1))
            if isinstance(parsed, dict):
                sources = parsed.get("sources", [])
                retrieval_stats = parsed.get("retrieval_stats", {})
            elif isinstance(parsed, list):
                sources = parsed  # 旧格式兼容
        except Exception:
            pass
        text = _SOURCES_RE.sub('', text)

    return text, sources, retrieval_stats


# 流式事件的初始状态模板（避免重复定义）
# 注：handle_query（非流式）与两个流式接口都调用它，并非只有流式用。
def _make_initial_state(
    query: str,
    image_urls: list = None,
    image_paths: list = None,
    image_base64_list: list = None,
    file_paths: list = None,
    thread_id: str = None,
) -> dict:
    """
    构建查询初始状态

    为什么要手写一整个 dict：state（状态）是**全图共享的字典**，节点靠 `state.get(字段)` 读、
    靠返回小 dict 合入来写。凡是条件路由函数可能读到的字段，这里必须先给初始值 ——
    缺键会让路由里的比较/取用直接抛异常，而不是安静地走默认分支。
    特别注意 write_operations 这类挂了 reducer（归并函数，这里是 operator.add）的字段：
    多个并行分支会各自往同一字段追加，没有初始值在合并时同样会报错，空列表不是可有可无的。
    previous_results / has_duplicate 是重复检测要用的跨轮历史位，每条新 query 都清空重来。

    Args:
        query: 用户查询文本
        image_urls: 图片公网URL列表（VQA 视觉问答：让模型看着图回答问题）
        image_paths: 本地图片路径列表（VQA多模态）
        image_base64_list: base64编码图片列表（VQA多模态）
        file_paths: 已上传文件路径列表（Excel/CSV等）
        thread_id: 会话ID（用于记忆隔离）
    """
    return {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "thread_id": thread_id,
        "plan": None,
        "is_complex": False,
        "selected_agent": None,
        "agent_results": [],
        "final_answer": "",
        "quality_score": 0.0,
        "iteration": 0,
        "feedback": "",
        "need_parameter_validation": False,
        "validation_target": None,
        "validation_info": None,
        "validation_reason": None,
        "parameter_retry_count": 0,
        "parameters_aligned": True,
        "parameter_alignment_errors": [],
        "critic_passed": True,
        "critic_validation": None,
        "critic_issues": [],
        "previous_results": [],
        "previous_result_embeddings": [],
        "has_duplicate": False,
        "stop_reason": None,
        "human_intervention_required": False,
        "human_feedback": None,
        "intervention_reason": None,
        "intervention_data": None,
        # 写操作审计（这些字段用 operator.add 聚合，必须给初始值，否则图会报错）
        "write_operations": [],
        "write_approved": False,
        # 多模态（VQA）
        "image_urls": image_urls or [],
        "image_paths": image_paths or [],
        "image_base64_list": image_base64_list or [],
        # 文件上传（Excel/CSV等）
        "file_paths": file_paths or [],
    }


class EnhancedLangGraphRAGSystem:
    """增强版 LangGraph RAG 系统"""

    def __init__(
        self,
        auto_update_index: bool = True,
        enable_checkpointer: bool = True,
        enable_parameter_validation: bool = True,
        enable_critic: bool = False,  # 需要 critic_agent
    ):
        """
        初始化系统

        Args:
            auto_update_index: 是否自动更新向量索引
            enable_checkpointer: 是否启用状态持久化
            enable_parameter_validation: 是否启用参数验证与重试
            enable_critic: 是否启用 Critic 验证
                           （critic=评审：校验"这段输出能不能接上后续任务"，不是给答案打分）
        """
        logger.info("初始化增强版 LangGraph RAG 系统...")

        self.enable_checkpointer = enable_checkpointer
        self.enable_parameter_validation = enable_parameter_validation
        self.enable_critic = enable_critic

        # 事实：_register_agents 并没有注册 critic_agent，
        # 所以即便 enable_critic=True，critic 节点也取不到实例、会直接判定通过 ——
        # 默认 False 的真正原因是"当前没人实现它"，不只是"怕报错"。

        # 自动更新索引
        if auto_update_index:
            self._auto_update_vector_index()

        # 加载配置
        # 事实：config/agents.yaml 虽被读进 self.config["agents"]，但全项目无人消费 ——
        # 真正的 Agent 清单/描述是下面 _register_* 里硬编码的，改 yaml 不产生任何效果。
        self.config = self._load_config()

        # ========== 统一创建 Redis（内存数据库）连接（全局唯一） ==========
        self.redis_client = self._create_redis_client()

        # 初始化 LLM（传入 redis_client）
        from llm.llm_client import LLM
        self.llm_max = self._create_llm("qwen-max")
        self.llm_plus = self._create_llm("qwen-plus")

        # 初始化 Embedder（向量化模型）：路由召回与重复检测都靠它算语义相似度
        self.embedder = self._create_embedder()

        # 初始化记忆管理器（按 thread_id 隔离，不同窗口/用户互不干扰）
        from core.memory import MemoryStore
        self.memory_store = MemoryStore(
            llm=self.llm_plus,
            max_token_limit=2000,
            max_messages=20,
            ttl_seconds=7200,  # 2小时无活动自动清理
        )
        # 向后兼容：保留 shared_memory 属性指向 None（不再使用全局单例）
        self.shared_memory = None
        logger.info("记忆管理器初始化完成（按 thread_id 隔离）")

        # 初始化编排组件：registry（注册中心，Agent 名册）/ planner（规划器）/ router（路由器）
        from orchestrator.planner import TaskPlanner
        from orchestrator.router import AgentRouter
        from orchestrator.registry import AgentRegistry

        self.registry = AgentRegistry()
        self.planner = TaskPlanner(self.llm_max)
        self.router = AgentRouter(self.llm_max, self.embedder)

        # 注册 Agents
        # 顺序有约束：必须先注册完才能建图 —— 图上的 Agent 节点就是从这里逐个取出来加的。
        self._register_agents()

        # 构建增强图
        # 这里把注册中心的记录原样交出去；build_enhanced_graph 认的是每条记录里的
        # "instance" 键（真正能 handle 的 Agent 实例），缺了它建图时会 KeyError。
        agents_dict = {agent["id"]: agent for agent in self.registry.get_all_agents()}

        self.graph = build_enhanced_graph(
            agents_dict=agents_dict,
            planner=self.planner,
            router=self.router,
            registry=self.registry,
            llm=self.llm_max,
            shared_memory=None,
            memory_store=self.memory_store,
            embedder=self.embedder,
            enable_checkpointer=enable_checkpointer,
            enable_parameter_validation=enable_parameter_validation,
            enable_critic=enable_critic,
        )

        # 初始化问答日志记录器
        from core.qa_logger import QALogger
        self.qa_logger = QALogger()

        logger.info("增强版 LangGraph RAG 系统初始化完成")

    def handle_query(
        self,
        query: str,
        thread_id: str = "default",
        image_urls: list = None,
        image_paths: list = None,
        image_base64_list: list = None,
        file_paths: list = None,
    ) -> dict:
        """
        处理查询（支持 interrupt_before（执行前中断）原生中断）

        机制：跑完后不能只看 invoke 的返回值 —— 若图被 interrupt_before 拦下，invoke 会提前
        返回，返回值只是半成品。必须再查 get_state(config).next 才能区分
        "真跑完了" 与 "停在中途等人"：next 非空说明还停在断点上（下面据此返回人工介入分支）。

        Args:
            query: 用户查询
            thread_id: 线程ID（会话 ID，用于状态持久化和中断恢复）
            image_urls: 图片公网URL列表（VQA多模态）
            image_paths: 本地图片路径列表（VQA多模态）
            image_base64_list: base64编码图片列表（VQA多模态）
            file_paths: 已上传文件路径列表（Excel/CSV等）

        Returns:
            dict: 包含结果或人工介入（human intervention）信息。
                  如果 human_intervention_required=True，
                  调用方应调用 resume_after_human_input() 恢复执行。
        """

        # checkpointer（检查点）靠 config 里的 thread_id（会话 ID）定位"这一份存档"。
        # 恢复时必须传一模一样的 thread_id，否则接不回断点、会另起一份新存档。
        config = {"configurable": {"thread_id": thread_id}}
        _start_time = time.time()

        try:
            result = self.graph.invoke(
                _make_initial_state(query, image_urls, image_paths, image_base64_list, file_paths, thread_id=thread_id),
                config,
            )

            # 检查图是否被 interrupt_before 暂停
            current_state = self.graph.get_state(config)

            if current_state.next:
                # current_state.next 不为空 = 图还没执行完 = 被 interrupt 暂停了
                # 此时 final_answer 多半还是空的/半成品，真正有用的是 intervention_reason
                # 与 intervention_data（前端据此渲染"批准/重试/终止"按钮，再走 resume）。
                state_values = current_state.values

                return {
                    "mode": "human_intervention",
                    "result": state_values.get("final_answer", ""),
                    "success": False,
                    "human_intervention_required": True,
                    "intervention_reason": state_values.get("intervention_reason", ""),
                    "intervention_data": state_values.get("intervention_data", {}),
                    "quality_score": state_values.get("quality_score", 0.0),
                    "iterations": state_values.get("iteration", 0),
                    "plan": state_values.get("plan"),
                    "thread_id": thread_id,
                }

            # 正常完成
            final_answer = result.get("final_answer", "")

            # 提取引用源标记（避免 <!-- SOURCES:...:SOURCES --> 泄露到用户侧）
            final_answer, sources, retrieval_stats = _extract_sources(final_answer)

            # 保存本轮对话到会话记忆（按 thread_id 隔离）
            if self.memory_store and final_answer and thread_id:
                session_mem = self.memory_store.get(thread_id)
                session_mem.add_user_message(query)
                session_mem.add_assistant_message(final_answer[:500])
                logger.info(f"[会话记忆] 保存本轮对话 (thread={thread_id})")

            _mode = "planning" if result.get("is_complex") else "simple"
            _success = result.get("quality_score", 0.0) >= 0.5 or bool(final_answer)
            _duration_ms = int((time.time() - _start_time) * 1000)

            self.qa_logger.log(
                query=query,
                answer=final_answer,
                mode=_mode,
                quality_score=result.get("quality_score", 0.0),
                success=_success,
                duration_ms=_duration_ms,
                thread_id=thread_id,
                agent_results=result.get("agent_results", []),
                sources=sources,
                iterations=result.get("iteration", 0),
                has_files=bool(file_paths),
                has_images=bool(image_urls or image_paths or image_base64_list),
            )

            return {
                "mode": _mode,
                "result": final_answer,
                "success": _success,
                "agent_results": _sanitize_output(result.get("agent_results", [])),
                "quality_score": result.get("quality_score", 0.0),
                "iterations": result.get("iteration", 0),
                "plan": _sanitize_output(result.get("plan")),
                "critic_validation": _sanitize_output(result.get("critic_validation")),
                "human_intervention_required": False,
                "sources": sources,
                "retrieval_stats": retrieval_stats,
            }

        except Exception as e:
            logger.error("查询处理失败: %s", e, exc_info=True)
            self.qa_logger.log(
                query=query, answer="", mode="error", success=False,
                duration_ms=int((time.time() - _start_time) * 1000),
                thread_id=thread_id, error=str(e),
            )
            return {"mode": "error", "result": "", "success": False, "error": str(e)}

    def resume_after_human_input(self, thread_id: str, human_feedback: str,
                                  extra_state: dict = None) -> dict:
        """
        人工审核后恢复执行

        机制（两段式，顺序不能反）：
          1. update_state：只往暂停中的 state 补写几个标志位，不跑图；此时 state 是"活的"。
          2. invoke(None, config)：第一个参数传 None = "别给我新输入，接着上次的断点跑"。
             若误传一份新 state，图会从 START 重新开始，人工的批准就白做了。
        断点是 enhanced_graph.py 编译时用 interrupt_before 钉在 human_intervention_execute
        之前的；下面写进 state 的 human_feedback 就是喂给那个节点的输入。

        Args:
            thread_id: 与 handle_query 时相同的 thread_id（必须一致，否则找不到断点）
            human_feedback: 用户输入（"approved" / "accept" / "retry" / "override" / "abort"）
            extra_state: 额外要注入的状态
        """

        config = {"configurable": {"thread_id": thread_id}}

        # 1. 把人工反馈注入到当前暂停的 State 中
        update_values = {"human_feedback": human_feedback}

        # 根据反馈类型自动设置对应的审批标志
        # 图的逻辑只认 state 里的标志位（plan_approved / db_operation_approved / write_approved），
        # 并不认识 "approved" 这个词；这一步就是把人的答复翻译成标志位，否则恢复后
        # check 节点会再次命中同一条通道、又被拦一次。
        # 判断依据是 intervention_reason 里的中文字样 —— 属于约定式耦合，
        # 改 enhanced_nodes.py 里那几条 reason 文案时，这里必须同步改。
        if human_feedback == "approved":
            current_state = self.graph.get_state(config)
            # 注意用 `or ""`：intervention_reason 可能是显式的 None，
            # 而 `.get(key, "")` 只在键不存在时才给默认值，键存在但值为 None 时会返回 None，
            # 下一行的 `in` 判断就会抛 TypeError。这是修掉的一个既有 bug。
            reason = current_state.values.get("intervention_reason") or ""
            if "任务规划" in reason:
                update_values["plan_approved"] = True
            elif "数据库" in reason:
                update_values["db_operation_approved"] = True
            elif "写操作" in reason:
                # 阶段 3 新增的写操作审批通道（见 enhanced_nodes.py 的
                # human_intervention_check_node）。批准后置位，避免再次触发。
                update_values["write_approved"] = True

        if extra_state:
            update_values.update(extra_state)

        self.graph.update_state(config, values=update_values)

        # 2. 从暂停点继续执行（invoke(None) = 接着断点跑，不是开新一轮）
        try:
            result = self.graph.invoke(None, config)

            # 检查是否再次被中断（例如恢复后又触发了另一个中断条件）
            current_state = self.graph.get_state(config)
            if current_state.next:
                state_values = current_state.values
                return {
                    "mode": "human_intervention",
                    "result": state_values.get("final_answer", ""),
                    "success": False,
                    "human_intervention_required": True,
                    "intervention_reason": state_values.get("intervention_reason", ""),
                    "intervention_data": state_values.get("intervention_data", {}),
                    "quality_score": state_values.get("quality_score", 0.0),
                    "iterations": state_values.get("iteration", 0),
                    "plan": state_values.get("plan"),
                    "thread_id": thread_id,
                }

            final_answer_raw = result.get("final_answer", "")
            final_answer_clean, sources, retrieval_stats = _extract_sources(final_answer_raw)

            return {
                "mode": "planning" if result.get("is_complex") else "simple",
                "result": final_answer_clean,
                "success": result.get("quality_score", 0.0) >= 0.5 or bool(final_answer_raw),
                "agent_results": result.get("agent_results", []),
                "quality_score": result.get("quality_score", 0.0),
                "iterations": result.get("iteration", 0),
                "plan": result.get("plan"),
                "human_intervention_required": False,
                "sources": sources,
                "retrieval_stats": retrieval_stats,
            }

        except Exception as e:
            logger.error("恢复执行失败: %s", e, exc_info=True)
            return {"mode": "error", "result": "", "success": False, "error": str(e)}

    def handle_query_stream(
        self,
        query: str,
        thread_id: str = "default",
        image_urls: list = None,
        image_paths: list = None,
        image_base64_list: list = None,
        file_paths: list = None,
    ) -> Iterator[dict]:
        """
        同步流式处理查询（stream=流式输出：结果分片吐出，不用等全部生成完）

        使用 stream_mode=["updates", "custom"] 实现：
        - "custom" 事件：节点内部通过 StreamWriter（流式事件写入器）推送的实时进度
        - "updates" 事件：节点完成时的状态更新

        为什么要两种模式：custom 让长节点中途也能吐进度（否则用户干等），
        updates 才带"这一步是谁跑完了"的结构；chunk 是 (模式名, 数据) 二元组，故按 mode 分流。

        Yields:
            dict: 流式事件，格式:
                {"type": "progress", "agent": str, "msg": str, ...}  — 实时进度
                {"type": "node_done", "node": str, "output": dict}   — 节点完成
                {"type": "final", "result": dict}                     — 最终结果
        """
        initial_state = _make_initial_state(query, image_urls, image_paths, image_base64_list, file_paths, thread_id=thread_id)
        config = {"configurable": {"thread_id": thread_id}}
        _start_time = time.time()
        final_state = None

        try:
            for chunk in self.graph.stream(
                initial_state,
                config,
                stream_mode=["custom", "updates"],
            ):
                mode, data = chunk

                if mode == "custom":
                    # StreamWriter 推送的自定义进度事件
                    yield _sanitize_output({"type": "progress", **data})

                elif mode == "updates":
                    # 节点完成时的状态更新
                    for node_name, node_output in data.items():
                        yield {
                            "type": "node_done",
                            "node": node_name,
                            "output": _sanitize_output(node_output),
                        }
                        final_state = node_output

            # 流结束后获取最终状态
            current_state = self.graph.get_state(config)

            if current_state.next:
                # 被 interrupt 暂停
                state_values = current_state.values
                yield {
                    "type": "human_intervention",
                    "reason": state_values.get("intervention_reason", ""),
                    "data": state_values.get("intervention_data", {}),
                    "thread_id": thread_id,
                }
            else:
                state_values = current_state.values
                final_answer = state_values.get("final_answer", "")

                # 提取引用源标记
                final_answer, sources, retrieval_stats = _extract_sources(final_answer)

                # 保存到会话记忆（按 thread_id 隔离）
                if self.memory_store and final_answer and thread_id:
                    session_mem = self.memory_store.get(thread_id)
                    session_mem.add_user_message(query)
                    session_mem.add_assistant_message(final_answer[:500])

                _mode = "planning" if state_values.get("is_complex") else "simple"
                _success = state_values.get("quality_score", 0.0) >= 0.5 or bool(final_answer)
                self.qa_logger.log(
                    query=query, answer=final_answer, mode=_mode,
                    quality_score=state_values.get("quality_score", 0.0),
                    success=_success,
                    duration_ms=int((time.time() - _start_time) * 1000),
                    thread_id=thread_id,
                    agent_results=state_values.get("agent_results", []),
                    sources=sources,
                    iterations=state_values.get("iteration", 0),
                    has_files=bool(file_paths),
                    has_images=bool(image_urls or image_paths or image_base64_list),
                )

                yield {
                    "type": "final",
                    "result": {
                        "mode": _mode,
                        "result": final_answer,
                        "success": _success,
                        "quality_score": state_values.get("quality_score", 0.0),
                        "plan": state_values.get("plan"),
                        "agent_results": _sanitize_output(state_values.get("agent_results", [])),
                        "sources": sources,
                        "retrieval_stats": retrieval_stats,
                    },
                }

        except Exception as e:
            logger.error("流式查询处理失败: %s", e, exc_info=True)
            yield {"type": "error", "error": str(e)}

    async def handle_query_stream_async(
        self,
        query: str,
        thread_id: str = "default",
        image_urls: list = None,
        image_paths: list = None,
        image_base64_list: list = None,
        file_paths: list = None,
    ) -> AsyncIterator[dict]:
        """
        异步流式处理查询

        使用 astream + stream_mode=["updates", "custom"] 实现非阻塞流式输出。
        适用于 FastAPI / WebSocket 等异步场景。

        与 handle_query_stream 逻辑一一对应，只是 stream 换成 astream、for 换成 async for。
        两份代码不是冗余：FastAPI 的路由是 async 的，在里面跑同步 stream 会阻塞事件循环。

        Yields:
            dict: 流式事件（格式同 handle_query_stream）
        """
        initial_state = _make_initial_state(query, image_urls, image_paths, image_base64_list, file_paths, thread_id=thread_id)
        config = {"configurable": {"thread_id": thread_id}}
        _start_time = time.time()

        try:
            async for chunk in self.graph.astream(
                initial_state,
                config,
                stream_mode=["custom", "updates"],
            ):
                mode, data = chunk

                if mode == "custom":
                    yield _sanitize_output({"type": "progress", **data})

                elif mode == "updates":
                    for node_name, node_output in data.items():
                        yield _sanitize_output({
                            "type": "node_done",
                            "node": node_name,
                            "output": node_output,
                        })

            # 流结束后获取最终状态
            current_state = self.graph.get_state(config)

            if current_state.next:
                state_values = current_state.values
                yield _sanitize_output({
                    "type": "human_intervention",
                    "reason": state_values.get("intervention_reason", ""),
                    "data": state_values.get("intervention_data", {}),
                    "thread_id": thread_id,
                })
            else:
                state_values = current_state.values
                final_answer = state_values.get("final_answer", "")

                # 提取引用源标记
                final_answer, sources, retrieval_stats = _extract_sources(final_answer)

                if self.memory_store and final_answer and thread_id:
                    session_mem = self.memory_store.get(thread_id)
                    session_mem.add_user_message(query)
                    session_mem.add_assistant_message(final_answer[:500])

                _mode = "planning" if state_values.get("is_complex") else "simple"
                _success = state_values.get("quality_score", 0.0) >= 0.5 or bool(final_answer)
                self.qa_logger.log(
                    query=query, answer=final_answer, mode=_mode,
                    quality_score=state_values.get("quality_score", 0.0),
                    success=_success,
                    duration_ms=int((time.time() - _start_time) * 1000),
                    thread_id=thread_id,
                    agent_results=state_values.get("agent_results", []),
                    sources=sources,
                    iterations=state_values.get("iteration", 0),
                    has_files=bool(file_paths),
                    has_images=bool(image_urls or image_paths or image_base64_list),
                )

                yield _sanitize_output({
                    "type": "final",
                    "result": {
                        "mode": _mode,
                        "result": final_answer,
                        "success": _success,
                        "quality_score": state_values.get("quality_score", 0.0),
                        "plan": state_values.get("plan"),
                        "agent_results": state_values.get("agent_results", []),
                        "sources": sources,
                        "retrieval_stats": retrieval_stats,
                    },
                })

        except Exception as e:
            logger.error("异步流式查询处理失败: %s", e, exc_info=True)
            yield {"type": "error", "error": str(e)}

    def list_agents(self) -> str:
        """列出所有 Agent"""
        return self.registry.list_agents()

    # ========== 私有方法（复用 entry.py 逻辑）==========

    def _auto_update_vector_index(self):
        """自动更新向量索引"""
        try:
            from tools.scripts.incremental_update import IncrementalIndexer, incremental_update
            indexer = IncrementalIndexer()
            new_files, modified_files = indexer.get_new_or_modified_files()
            if new_files or modified_files:
                logger.info("检测到 %d 个新文档, %d 个修改的文档", len(new_files), len(modified_files))
                incremental_update()
                logger.info("向量索引更新完成")
        except Exception as e:
            logger.warning("自动更新向量索引失败: %s", e)

    def _load_config(self) -> dict:
        """加载配置：读 models 与 agents 两个 yaml（路径是相对的，依赖进程 CWD 在 nlp/ 下）

        事实：返回值里只有 config["models"] 被真正取用（见 _create_llm）；
        config["agents"] 读进来了却全项目无人消费 —— 名册实际由 _register_* 硬编码。
        """
        config = {}
        for path, key in [("config/models.yaml", "models"), ("config/agents.yaml", "agents")]:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    config[key] = yaml.safe_load(f).get(key, {})
        return config

    def _create_llm(self, model_name: str):
        """创建 LLM（传入 redis_client 实现 LLM 缓存持久化）"""
        from llm.llm_client import LLM
        model_config = self.config.get("models", {}).get(model_name, {})
        api_key = os.getenv(model_config.get("api_key_env", "DASHSCOPE_API_KEY"))
        return LLM(
            model_name=model_config.get("model_name", model_name),
            api_key=api_key,
            base_url=model_config.get("base_url"),
            max_tokens=model_config.get("max_tokens", 4096),
            temperature=model_config.get("temperature", 0.7),
            timeout=model_config.get("timeout", 120),
            redis_client=self.redis_client,
        )

    def _create_embedder(self):
        """创建 Embedder"""
        from llm.embedder import Embedder
        try:
            from rag_core.api_embedder import APIEmbedder
            api_key = os.getenv("DASHSCOPE_API_KEY")
            api_embedder = APIEmbedder(api_key=api_key, provider="dashscope", model="text-embedding-v2")
            return Embedder(api_embedder=api_embedder)
        except Exception as e:
            # 注意：这个 except 返回的 Embedder() 没有 api_embedder，
            # 真正调用时 llm/embedder.py 会抛 "未配置嵌入器" —— 它不是可用的降级兜底，
            # 只保证"构造不炸"。检索/路由一旦真用到向量，错误会在调用点才暴露。
            logger.warning("无法初始化API Embedder: %s", e)
            return Embedder()

    def _create_redis_client(self):
        """
        创建全局 Redis 连接（唯一入口）

        所有需要 Redis 的组件共享同一个连接，避免重复连接。
        连接失败时返回 None，所有组件自动降级为纯内存模式。
        """
        redis_url = os.getenv("REDIS_URL")
        if not redis_url:
            logger.info("[Redis] 未配置 REDIS_URL，全部缓存使用纯内存模式")
            return None

        try:
            import redis
            client = redis.from_url(redis_url, decode_responses=True)
            client.ping()
            logger.info(f"[Redis] 连接成功: {redis_url}")
            return client
        except ImportError:
            logger.warning("[Redis] redis 库未安装，降级为纯内存缓存（pip install redis）")
            return None
        except Exception as e:
            logger.warning(f"[Redis] 连接失败（{e}），降级为纯内存缓存")
            return None

    def _register_agents(self):
        """注册 Agents（复用 entry.py 逻辑）

        登记顺序 ≠ 执行顺序：这里只决定"有哪几个 Agent 可选"，
        谁先谁后由 enhanced_graph.py 的边决定。
        """
        # 向量检索器：retriever（检索器）负责从知识库召回文档
        retriever = self._init_retriever()
        bm25_retriever = self._init_bm25_retriever(retriever)  # BM25（关键词检索算法）：按词频+稀有度打分

        # 混合检索（hybrid retrieval）= 语义检索 + 关键词检索两路一起用，
        # 所以 knowledge / customer_service 两个 Agent 都要同时拿到上面两个检索器。
        #
        # 注册 Agents（文档角色过滤已禁用，所有用户可查询所有文档）
        #
        # 注意：document_agent 已移除（阶段 4）。
        # 它那 9 个工具全是合同审核专用的（风险识别、违约金计算、按供应商检索），
        # 与电商售后无关。原 Agent 代码保留在 agents/document_agent/ 未删除，
        # 只是不再注册 —— 如果以后要恢复，把 _register_document_agent 调用加回来即可。
        # 第三个参数 doc_filter 恒传 None = 完全不做角色过滤（KnowledgeAgent 内部
        # 只在 document_filter 为真时才启用过滤逻辑）。该形参保留但已无其它调用方。
        self._register_knowledge_agent(retriever, bm25_retriever, None)
        self._register_database_agent()
        self._register_customer_service_agent(retriever, bm25_retriever)
        self._register_vqa_agent()

        # 更新路由器索引：把 Agent 的描述向量化后建索引，供 router 做语义召回
        agents = self.registry.get_all_agents()
        if agents:
            self.router.index_agents(agents)

    def _init_retriever(self):
        """初始化向量检索器"""
        try:
            from rag_core.chroma_store import ChromaStore  # Chroma（向量数据库）：存向量、按相似度检索
            from rag_core.api_embedder import APIEmbedder

            chroma_db_path = "./vector_db/chroma_db"
            if not os.path.exists(chroma_db_path):
                return None

            api_embedder = APIEmbedder(
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                provider="dashscope",
                model="text-embedding-v2"
            )
            chroma_store = ChromaStore(
                dimension=api_embedder.embedding_dim,
                collection_name="knowledge_base",
                persist_directory=chroma_db_path
            )

            class SimpleRetriever:
                """简单向量检索器，封装 APIEmbedder + ChromaStore

                就地定义在方法里是因为只在文件内用一次；作用是给 Agent 一个统一接口
                （retrieve(query, top_k) → 结果列表），好让向量/BM25 两路结果被同一套代码消费。
                """

                # 向量相似度最低阈值
                # 余弦相似度（cosine similarity）范围 0~1，中文 embedding（向量化）场景下：
                #   > 0.7  高度相关
                #   0.5~0.7 较相关
                #   0.45~0.55 弱相关
                #   < 0.45 基本不相关
                MIN_VECTOR_SCORE = 0.45

                def __init__(self, embedder, chroma_store):
                    self.embedder = embedder
                    self.chroma_store = chroma_store

                def retrieve(self, query: str, top_k: int = 3):
                    query_embedding = self.embedder.encode_query(query)
                    if hasattr(query_embedding, 'tolist'):
                        query_vector = query_embedding.tolist()
                    else:
                        query_vector = list(query_embedding)
                    # 知识库总 chunk（文本块）数 = 文档切分后的最小检索单位，下面统计都以它计数
                    corpus_size = self.chroma_store.collection.count()
                    # 先不过滤获取总数，再按阈值过滤（top_k=取前 k 条，是"取多少候选"的上限）
                    # 召回（recall）= 先把可能相关的都捞出来这一步，阈值过滤是在召回之后再做的
                    all_results = self.chroma_store.search(query_vector=query_vector, top_k=top_k, min_score=0.0)
                    total_count = len(all_results)
                    passed = [r for r in all_results if r[1] >= self.MIN_VECTOR_SCORE]
                    filtered_count = total_count - len(passed)
                    if filtered_count > 0:
                        logger.info(f"[向量检索] 知识库 {corpus_size} 个chunk，召回 {total_count}，过滤低相似度 {filtered_count} 个 (阈值: {self.MIN_VECTOR_SCORE})，通过 {len(passed)} 个")
                    output = []
                    for item in passed:
                        chunk, score = item[0], item[1]
                        meta = dict(item[2]) if len(item) >= 3 else {}
                        meta["_vector_corpus_size"] = corpus_size
                        meta["_vector_total"] = total_count
                        meta["_vector_filtered"] = filtered_count
                        meta["_vector_score"] = round(score, 4)
                        output.append({"content": chunk, "score": score, "metadata": meta})
                    return output

            return SimpleRetriever(api_embedder, chroma_store)
        except Exception as e:
            logger.error("向量检索器初始化失败: %s", e)
            return None

    def _init_bm25_retriever(self, retriever):
        """初始化 BM25 检索器（从 ChromaDB 加载语料库并 fit）"""
        try:
            from rag_core.bm25_retriever import BM25Retriever
            if not retriever:
                return None

            bm25 = BM25Retriever()

            # 从 ChromaDB 加载所有文档内容，作为 BM25 的语料库
            # 为什么复用 Chroma 的原文：BM25 只是换一套打分方式（词频 vs 向量），
            # 语料还是同一批文档，不必再维护第二份数据源。
            try:
                chroma_store = retriever.chroma_store
                all_data = chroma_store.collection.get(include=["documents", "metadatas"])
                if all_data and all_data.get("documents"):
                    corpus = all_data["documents"]
                    # 保存元数据供检索时使用
                    bm25._metadatas = all_data.get("metadatas", [{} for _ in corpus])
                    # fit = 用整个语料库建索引；BM25 的"训练"只是统计词频，不涉及模型参数
                    bm25.fit(corpus)
                    logger.info("BM25检索器从 ChromaDB 加载了 %d 个文档块", len(corpus))

                    # 包装 retrieve 方法，返回与向量检索器一致的格式
                    # 这里改的是实例属性（monkey patch），不是改类定义：为了在拿到语料库后
                    # 顺手统一出参格式，让上层能用同一套代码消费向量/关键词两路结果。
                    original_retrieve = bm25.retrieve

                    # BM25 最低分数阈值
                    # BM25 分数无上界，但 < 0.5 说明查询词与文档几乎不匹配：
                    #   > 3.0  高度匹配（多个查询词高频出现）
                    #   1.0~3.0 较匹配
                    #   0.5~1.0 弱匹配（仅个别词低频命中）
                    #   < 0.5  基本不匹配
                    MIN_BM25_SCORE = 0.5
                    # 查询词命中率阈值：文档必须命中查询中至少这个比例的关键词
                    # 例如查询 8 个关键词，MIN_HIT_RATIO=0.2 → 至少命中 2 个才保留
                    # 防止只靠 1 个通用词（如"推荐""适合"）就蒙混过关
                    MIN_HIT_RATIO = 0.2

                    def retrieve_with_metadata(query: str, top_k: int = 5, **kwargs):
                        idx_results = original_retrieve(query, top_k=top_k)
                        corpus_size = bm25.corpus_size
                        matched_count = sum(1 for _, s in idx_results if s > 0)

                        # 计算查询关键词列表（去重），用于命中率检查
                        query_tokens = set(bm25.tokenize(query))
                        num_query_tokens = len(query_tokens)

                        results = []
                        filtered_count = 0
                        for idx, score in idx_results:
                            if score >= MIN_BM25_SCORE and idx < len(bm25.corpus):
                                # 检查该文档命中了查询中多少个关键词
                                doc_tokens = set(bm25.doc_tokens[idx]) if idx < len(bm25.doc_tokens) else set()
                                hit_tokens = query_tokens & doc_tokens
                                hit_ratio = len(hit_tokens) / num_query_tokens if num_query_tokens > 0 else 0

                                if num_query_tokens >= 3 and hit_ratio < MIN_HIT_RATIO:
                                    # 查询词足够多但命中率太低 → 视为不相关
                                    filtered_count += 1
                                    logger.debug(f"[BM25] chunk {idx} 被命中率过滤: 命中 {len(hit_tokens)}/{num_query_tokens} = {hit_ratio:.2f} < {MIN_HIT_RATIO}, 命中词={hit_tokens}")
                                    continue

                                meta = dict(bm25._metadatas[idx]) if idx < len(bm25._metadatas) else {}
                                meta["_bm25_corpus_size"] = corpus_size
                                meta["_bm25_matched"] = matched_count
                                meta["_bm25_filtered"] = 0
                                meta["_bm25_score"] = round(float(score), 4)
                                meta["_bm25_hit_tokens"] = list(hit_tokens)
                                meta["_bm25_hit_ratio"] = round(hit_ratio, 2)
                                results.append({
                                    "content": bm25.corpus[idx],
                                    "score": float(score),
                                    "metadata": meta
                                })
                            elif score > 0 and score < MIN_BM25_SCORE:
                                filtered_count += 1
                        for r in results:
                            r["metadata"]["_bm25_filtered"] = filtered_count
                        if filtered_count > 0:
                            logger.info(f"[BM25检索] 知识库 {corpus_size} 个chunk，关键词命中 {matched_count}，过滤 {filtered_count} 个 (分数阈值: {MIN_BM25_SCORE}, 命中率阈值: {MIN_HIT_RATIO})")
                        return results

                    bm25.retrieve = retrieve_with_metadata
                    return bm25
                else:
                    logger.warning("ChromaDB 中没有文档，BM25 检索器为空")
                    return None
            except Exception as e:
                logger.warning("从 ChromaDB 加载 BM25 语料库失败: %s", e)
                return None
        except Exception as e:
            logger.warning("BM25检索器初始化失败: %s", e)
        return None

    def _register_knowledge_agent(self, retriever, bm25_retriever, doc_filter):
        """注册 Knowledge Agent（doc_filter 目前恒为 None，见 _register_agents 的说明）"""
        try:
            from agents.knowledge_agent.agent import KnowledgeAgent
            agent = KnowledgeAgent(
                llm=self.llm_plus,
                retriever=retriever,
                bm25_retriever=bm25_retriever,
                document_filter=doc_filter,
                embedder=self.embedder,
                enable_optimizations=True,
                enable_cache=False,  # 禁用业务缓存，确保每次展示完整 RAG 流程
                redis_client=self.redis_client,
            )
            # 描述会被向量化，用于路由召回 —— 所以要写清楚"什么该找我、什么不该找我"，
            # 并且尽量覆盖用户真实会用的词。
            # register 内部有 Harness（约束层）校验：实例若没有 handle 方法（不符合
            # AgentProtocol=Agent 接口契约）会直接抛 TypeError，而不是默默注册进去。
            self.registry.register(
                agent_id="knowledge_agent",
                name="知识检索Agent",
                description="通用知识问答，适合概念解释、原理说明、技术问题解答。"
                            "不涉及售后业务；售后政策、退换货规则、退款时效属于客服Agent的职责。",
                capabilities=["语义检索", "关键词检索", "混合检索", "查询优化",
                              "概念解释", "原理解读", "网络搜索"],
                agent_instance=agent,
            )
            logger.info("Knowledge Agent注册成功")
        except Exception as e:
            logger.error("Knowledge Agent注册失败: %s", e)

    def _register_database_agent(self):
        """注册 Database Agent"""
        try:
            from agents.database_agent.agent import DatabaseAgent
            agent = DatabaseAgent(llm=self.llm_plus, enable_optimizations=True, enable_cache=False)
            self.registry.register(
                agent_id="database_agent",
                name="订单数据查询Agent",
                description="查询交易数据：订单状态与详情、物流轨迹与签收时间、退款记录与进度、"
                            "会员等级与消费统计、按条件筛选和统计订单。适合「我的订单到哪了」"
                            "「查一下这个订单」「退款到账了没」这类**查数据**的问题。",
                capabilities=["订单查询", "物流查询", "退款记录查询", "会员数据查询",
                              "SQL生成", "表结构探索", "数据统计"],
                agent_instance=agent,
            )
            logger.info("Database Agent注册成功")
        except Exception as e:
            logger.error("Database Agent注册失败: %s", e)

    def _register_customer_service_agent(self, retriever, bm25_retriever):
        """注册 CustomerService Agent"""
        try:
            from agents.customer_service_agent.agent import CustomerServiceAgent
            agent = CustomerServiceAgent(
                llm=self.llm_plus,
                retriever=retriever,
                bm25_retriever=bm25_retriever,
                embedder=self.embedder,
                enable_optimizations=True,
                enable_cache=False,  # 禁用业务缓存，确保每次展示完整流程
            )
            self.registry.register(
                agent_id="customer_service_agent",
                name="售后客服Agent",
                description="处理电商售后业务：退换货政策咨询（几天内能退、运费谁承担、"
                            "哪些商品不支持无理由退货）、提交退货换货申请、退款规则、"
                            "会员售后权益、投诉与工单受理。适合「我要退货」「耳机坏了"
                            "怎么办」「退货运费谁出」这类**问规则或要办理**的问题。",
                capabilities=["售后政策问答", "退换货办理", "退款规则", "运费规则",
                              "投诉受理", "工单创建", "工单查询", "情感分析", "会员权益查询"],
                agent_instance=agent,
            )
            logger.info("CustomerService Agent注册成功")
        except Exception as e:
            logger.error("CustomerService Agent注册失败: %s", e)

    def _register_document_agent(self, retriever):
        """注册 Document Agent（当前无调用方：阶段 4 已把该 Agent 下线，方法保留备恢复）"""
        try:
            from agents.document_agent.agent import DocumentAgent
            agent = DocumentAgent(llm=self.llm_plus, retriever=retriever, embedder=self.embedder)
            self.registry.register(
                agent_id="document_agent",
                name="文档处理Agent",
                description="统一文档处理Agent，支持PDF（含OCR扫描件识别）、Word（.docx/.doc）、Excel/CSV、纯文本（.txt/.md）等多格式文档的解析与智能分析。",
                capabilities=["文档解析", "文档分析", "文档总结"],
                agent_instance=agent,
            )
            logger.info("Document Agent注册成功")
        except Exception as e:
            logger.error("Document Agent注册失败: %s", e)

    def _register_vqa_agent(self):
        """注册 VQA（视觉问答）Agent

        方法名只提了 VQA，但 chat_agent 也在这里注册（历史原因跟 VQA 挂在了一起）——
        找 chat_agent 的注册处时别漏了下面第二个 try 块。
        """
        try:
            from agents.vqa_agent.agent import VQAAgent
            agent = VQAAgent(
                llm=self.llm_plus,
                model_name="qwen-vl-max",
            )
            self.registry.register(
                agent_id="vqa_agent",
                name="视觉问答Agent",
                description="基于千问VL多模态模型的视觉问答Agent，擅长图表数据趋势分析、单图/多图对比分析、图片内容描述、OCR文字识别。需要用户提供图片。",
                capabilities=["图像问答", "图表趋势分析", "多图对比分析", "图片描述", "OCR文字识别", "图表数据提取"],
                agent_instance=agent,
                model="qwen-vl-max",
            )
            logger.info("VQA Agent注册成功")
        except Exception as e:
            logger.error("VQA Agent注册失败: %s", e)

        try:
            from agents.chat_agent.agent import ChatAgent
            agent = ChatAgent(llm=self.llm_plus)
            self.registry.register(
                agent_id="chat_agent",
                name="闲聊兜底Agent",
                description="通用对话兜底Agent，处理闲聊、问候、意图不明确或不属于其他专业Agent的问题。当用户意图模糊、其他Agent没有把握时，优先选择此Agent。",
                capabilities=["闲聊对话", "问候回复", "通用问答", "意图不明确处理", "引导用户使用专业功能"],
                agent_instance=agent,
            )
            logger.info("Chat Agent注册成功")
        except Exception as e:
            logger.error("Chat Agent注册失败: %s", e)
