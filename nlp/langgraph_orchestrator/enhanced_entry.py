# -*- coding: utf-8 -*-
"""
LangGraph RAG 系统主入口
========================

基于 LangGraph 框架的多 Agent 编排系统。
支持状态持久化、断点续传、人机共驾、参数对齐验证等高级功能。
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

    Args:
        query: 用户查询文本
        image_urls: 图片公网URL列表（VQA多模态）
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
        """
        logger.info("初始化增强版 LangGraph RAG 系统...")

        self.enable_checkpointer = enable_checkpointer
        self.enable_parameter_validation = enable_parameter_validation
        self.enable_critic = enable_critic

        # 自动更新索引
        if auto_update_index:
            self._auto_update_vector_index()

        # 加载配置
        self.config = self._load_config()

        # ========== 统一创建 Redis 连接（全局唯一） ==========
        self.redis_client = self._create_redis_client()

        # 初始化 LLM（传入 redis_client）
        from llm.llm_client import LLM
        self.llm_max = self._create_llm("qwen-max")
        self.llm_plus = self._create_llm("qwen-plus")

        # 初始化 Embedder
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

        # 初始化编排组件
        from orchestrator.planner import TaskPlanner
        from orchestrator.router import AgentRouter
        from orchestrator.registry import AgentRegistry

        self.registry = AgentRegistry()
        self.planner = TaskPlanner(self.llm_max)
        self.router = AgentRouter(self.llm_max, self.embedder)

        # 注册 Agents
        self._register_agents()

        # 构建增强图
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
        处理查询（支持 interrupt_before 原生中断）

        Args:
            query: 用户查询
            thread_id: 线程ID（用于状态持久化和中断恢复）
            image_urls: 图片公网URL列表（VQA多模态）
            image_paths: 本地图片路径列表（VQA多模态）
            image_base64_list: base64编码图片列表（VQA多模态）
            file_paths: 已上传文件路径列表（Excel/CSV等）

        Returns:
            dict: 包含结果或人工介入信息。
                  如果 human_intervention_required=True，
                  调用方应调用 resume_after_human_input() 恢复执行。
        """

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

        Args:
            thread_id: 与 handle_query 时相同的 thread_id
            human_feedback: 用户输入（"approved" / "accept" / "retry" / "override" / "abort"）
            extra_state: 额外要注入的状态
        """

        config = {"configurable": {"thread_id": thread_id}}

        # 1. 把人工反馈注入到当前暂停的 State 中
        update_values = {"human_feedback": human_feedback}

        # 根据反馈类型自动设置对应的审批标志
        if human_feedback == "approved":
            current_state = self.graph.get_state(config)
            reason = current_state.values.get("intervention_reason", "")
            if "任务规划" in reason:
                update_values["plan_approved"] = True
            elif "数据库" in reason:
                update_values["db_operation_approved"] = True

        if extra_state:
            update_values.update(extra_state)

        self.graph.update_state(config, values=update_values)

        # 2. 从暂停点继续执行
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
        同步流式处理查询

        使用 stream_mode=["updates", "custom"] 实现：
        - "custom" 事件：节点内部通过 StreamWriter 推送的实时进度
        - "updates" 事件：节点完成时的状态更新

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
        """加载配置"""
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
        """注册 Agents（复用 entry.py 逻辑）"""
        # 向量检索器
        retriever = self._init_retriever()
        bm25_retriever = self._init_bm25_retriever(retriever)

        # 注册 Agents（文档角色过滤已禁用，所有用户可查询所有文档）
        self._register_knowledge_agent(retriever, bm25_retriever, None)
        self._register_database_agent()
        self._register_customer_service_agent(retriever, bm25_retriever)
        self._register_document_agent(retriever)
        self._register_vqa_agent()

        # 更新路由器索引
        agents = self.registry.get_all_agents()
        if agents:
            self.router.index_agents(agents)

    def _init_retriever(self):
        """初始化向量检索器"""
        try:
            from rag_core.chroma_store import ChromaStore
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
                """简单向量检索器，封装 APIEmbedder + ChromaStore"""

                # 向量相似度最低阈值
                # 余弦相似度范围 0~1，中文 embedding 场景下：
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
                    # 知识库总 chunk 数
                    corpus_size = self.chroma_store.collection.count()
                    # 先不过滤获取总数，再按阈值过滤
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
            try:
                chroma_store = retriever.chroma_store
                all_data = chroma_store.collection.get(include=["documents", "metadatas"])
                if all_data and all_data.get("documents"):
                    corpus = all_data["documents"]
                    # 保存元数据供检索时使用
                    bm25._metadatas = all_data.get("metadatas", [{} for _ in corpus])
                    bm25.fit(corpus)
                    logger.info("BM25检索器从 ChromaDB 加载了 %d 个文档块", len(corpus))

                    # 包装 retrieve 方法，返回与向量检索器一致的格式
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
        """注册 Knowledge Agent"""
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
            self.registry.register(
                agent_id="knowledge_agent",
                name="知识检索Agent",
                description="专门处理通用知识查询，擅长技术概念、原理解释、文档检索。",
                capabilities=["语义检索", "关键词检索", "混合检索", "查询优化", "网络搜索"],
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
                name="数据库查询Agent",
                description="专门处理数据库查询任务，擅长SQL生成、表结构探索、数据分析。",
                capabilities=["数据库查询", "SQL生成", "数据分析"],
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
                name="客服Agent",
                description="专门处理客户咨询、投诉、工单管理。",
                capabilities=["情感分析", "工单创建", "用户信息查询"],
                agent_instance=agent,
            )
            logger.info("CustomerService Agent注册成功")
        except Exception as e:
            logger.error("CustomerService Agent注册失败: %s", e)

    def _register_document_agent(self, retriever):
        """注册 Document Agent"""
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
        """注册 VQA（视觉问答）Agent"""
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
