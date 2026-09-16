# -*- coding: utf-8 -*-
"""
Customer Service Agent
======================

专门处理客户咨询、投诉、工单管理。支持情感分析、工单创建与跟进、用户信息查询。
可检索知识库回答客户问题。
"""

import logging
import time
import json
from typing import Dict, Any, List
from llm.output_parser import parse_llm_json, parse_tool_calls
from llm.langchain_tools import ToolRegistry
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class CustomerServiceAgent:
    """客服Agent（ReAct模式）"""

    def _emit_progress(self, context: Dict, msg: str, **extra):
        """通过 StreamWriter 发射中间步骤事件（供前端 trace 面板展示）"""
        writer = context.get("_stream_writer")
        if writer is None:
            return
        agent_name = context.get("_agent_name", "customer_service_agent")
        import time as _time
        event = {"event": "progress", "agent": agent_name, "msg": msg, "ts": _time.time()}
        event.update(extra)
        writer(event)

    def __init__(
        self,
        llm,
        retriever=None,
        bm25_retriever=None,
        tool_registry=None,
        embedder=None,
        crm=None,
        enable_optimizations: bool = True,
        enable_cache: bool = True,
        cache_size: int = 500,
        cache_ttl: int = 3600,
        max_retries: int = 3,
        vector_weight: float = 0.8,
    ):
        """
        初始化

        Args:
            llm: LLM实例
            retriever: 语义检索器
            bm25_retriever: BM25检索器
            function_calling: Function Calling管理器
            embedder: 嵌入器实例
            crm: CRM系统实例
            enable_optimizations: 是否启用优化（默认True）
            enable_memory: 是否启用对话记忆（默认True）
            max_history: 对话历史最大轮数（默认10）
            cache_size: 缓存大小（默认500）
            cache_ttl: 缓存TTL秒数（默认3600）
            max_retries: 最大重试次数（默认3）
            vector_weight: 向量检索权重（默认0.5）
        """
        self.llm = llm
        self.retriever = retriever
        self.bm25_retriever = bm25_retriever
        self.tool_registry = tool_registry or ToolRegistry()
        self.embedder = embedder
        self.enable_optimizations = enable_optimizations
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        self.cache_ttl = cache_ttl
        self.max_retries = max_retries
        self.vector_weight = vector_weight

        # 初始化 CRM
        self._init_customer_service_modules(crm)

        # 初始化优化模块
        if enable_optimizations:
            self._init_optimization_modules()
        else:
            self.cache_manager = None
            self.retry_strategy = None
            self.error_handler = None
            self.hybrid_retriever = None

        # 注册工具
        self._register_tools()

        logger.info(f"CustomerServiceAgent初始化完成 (优化: {enable_optimizations})")

    def _init_customer_service_modules(self, crm):
        """初始化客服模块"""
        try:
            from core.crm_mock import MockCRM
            self.crm = crm or MockCRM()
            logger.info("✓ 客服系统初始化完成")
        except Exception as e:
            logger.error(f"客服系统初始化失败: {e}", exc_info=True)
            logger.warning("客服功能将被禁用")
            self.crm = None

    def _init_optimization_modules(self):
        """初始化优化模块（子集）"""
        try:
            # 1. 缓存机制
            from rag_core.cache_manager import CacheManager
            self.cache_manager = CacheManager(
                max_size=self.cache_size,
                ttl=self.cache_ttl
            )
            logger.info("✓ 缓存管理器初始化完成")

            # 2. 错误处理
            from core.error_handler import RetryStrategy, GracefulErrorHandler
            self.retry_strategy = RetryStrategy(max_retries=self.max_retries)
            self.error_handler = GracefulErrorHandler()
            logger.info("✓ 错误处理器初始化完成")

            # 3. 监控
            from rag_core.monitoring import structured_logger, metrics_collector
            self.structured_logger = structured_logger
            self.metrics_collector = metrics_collector
            logger.info("✓ 监控系统初始化完成")

            # 4. 混合检索
            if self.retriever:
                from rag_core.hybrid_retriever import HybridRetriever, SimpleReranker
                self.hybrid_retriever = HybridRetriever(
                    vector_retriever=self.retriever,
                    bm25_retriever=self.bm25_retriever,
                    vector_weight=self.vector_weight,
                    use_rerank=False,
                    reranker=None
                )
                logger.info("✓ 混合检索器初始化完成")
            else:
                self.hybrid_retriever = None

            logger.info("=" * 50)
            logger.info("CustomerServiceAgent 优化模块加载完成")
            logger.info("=" * 50)

        except Exception as e:
            logger.error(f"优化模块初始化失败: {e}", exc_info=True)
            logger.warning("降级为基础模式运行")
            self.enable_optimizations = False
            self.cache_manager = None
            self.retry_strategy = None
            self.error_handler = None
            self.hybrid_retriever = None

    def _register_tools(self):
        """注册工具（7个）"""
        # ========== 知识检索工具 ==========

        # 1. 混合检索
        self.tool_registry.register_tool(
            name="hybrid_search",
            description="结合向量检索和关键词检索，使用RRF算法融合结果",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "查询文本"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回结果数量",
                        "default": 3
                    },
                    "vector_weight": {
                        "type": "number",
                        "description": "向量检索权重（0-1）",
                        "default": 0.5
                    }
                },
                "required": ["query"]
            },
            function=self.hybrid_search
        )

        # 2. 向量检索
        self.tool_registry.register_tool(
            name="vector_search",
            description="在知识库中进行语义相似度检索",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "查询文本"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回结果数量",
                        "default": 3
                    }
                },
                "required": ["query"]
            },
            function=self.vector_search
        )

        # 3. 关键词检索
        self.tool_registry.register_tool(
            name="keyword_search",
            description="基于BM25算法的关键词精确匹配检索",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "查询文本"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回结果数量",
                        "default": 3
                    }
                },
                "required": ["query"]
            },
            function=self.keyword_search
        )

        # ========== 客服工具 ==========

        # 4. 情感分析
        self.tool_registry.register_tool(
            name="analyze_sentiment",
            description="分析用户情绪（正面/负面/中性），适合客服场景判断用户满意度",
            parameters={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "待分析的文本"
                    }
                },
                "required": ["text"]
            },
            function=self.analyze_sentiment
        )

        # 5. 创建工单
        self.tool_registry.register_tool(
            name="create_ticket",
            description="为用户创建客服工单，适合投诉、问题反馈、需要人工跟进的场景",
            parameters={
                "type": "object",
                "properties": {
                    "issue": {
                        "type": "string",
                        "description": "问题描述"
                    },
                    "user_id": {
                        "type": "string",
                        "description": "用户ID（可选）"
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "normal", "high", "urgent"],
                        "default": "normal",
                        "description": "优先级"
                    }
                },
                "required": ["issue"]
            },
            function=self.create_ticket
        )

        # 6. 查询工单
        self.tool_registry.register_tool(
            name="query_ticket",
            description="查询工单状态，适合用户跟进问题处理进度",
            parameters={
                "type": "object",
                "properties": {
                    "ticket_id": {
                        "type": "string",
                        "description": "工单ID"
                    }
                },
                "required": ["ticket_id"]
            },
            function=self.query_ticket
        )

        # 7. 查询用户信息
        self.tool_registry.register_tool(
            name="query_user_info",
            description="查询用户基本信息和历史记录，适合提供个性化服务",
            parameters={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "用户ID"
                    }
                },
                "required": ["user_id"]
            },
            function=self.query_user_info
        )

    @staticmethod
    def _truncate_tool_result(result_str: str, max_len: int = 4000) -> str:
        """截断过长的工具结果"""
        if len(result_str) <= max_len:
            return result_str
        keep = max_len // 2
        truncated = len(result_str) - max_len
        logger.info(f"[ReAct] 工具结果截断了 {truncated} 个字符")
        return result_str[:keep] + f"\n...[截断了 {truncated} 个字符]...\n" + result_str[-keep:]

    def _execute_tool_calls(self, tool_calls: List[Dict], retrieval_count: int,
                            seen_tool_calls: set = None) -> List[Dict]:
        """执行工具调用（支持去重 + 并行 + JSON容错）"""
        if seen_tool_calls is None:
            seen_tool_calls = set()

        unique_calls = []
        for tc in tool_calls:
            func_name = tc["function"]["name"]
            args_str = tc["function"]["arguments"]
            call_key = (func_name, args_str)
            if call_key in seen_tool_calls:
                logger.info(f"[ReAct] 跳过重复调用: {func_name}")
                continue
            seen_tool_calls.add(call_key)
            unique_calls.append(tc)

        if not unique_calls:
            return []

        def _exec_one(tool_call):
            fn = tool_call["function"]["name"]
            try:
                arguments = json.loads(tool_call["function"]["arguments"])
            except json.JSONDecodeError:
                arguments = parse_llm_json(tool_call["function"]["arguments"], fallback={})

            logger.info(f"[ReAct工具调用] 工具名称: {fn}")
            logger.info(f"[ReAct工具参数] {json.dumps(arguments, ensure_ascii=False)}")

            tool_start = time.time()
            try:
                result = self.tool_registry.call_tool(fn, arguments)
                tool_latency = (time.time() - tool_start) * 1000

                if isinstance(result, list):
                    logger.info(f"[ReAct工具结果] 返回 {len(result)} 条结果")
                elif isinstance(result, dict):
                    logger.info(f"[ReAct工具结果] 返回字典，键: {list(result.keys())}")
                elif isinstance(result, str):
                    logger.info(f"[ReAct工具结果] 返回文本，长度: {len(result)} 字符")
                else:
                    logger.info(f"[ReAct工具结果] 返回类型: {type(result)}")

                if self.enable_optimizations:
                    self.metrics_collector.record(f"tool_{fn}_latency_ms", tool_latency)

                logger.info(f"[ReAct] 工具执行完成，耗时 {tool_latency:.1f}ms")
                return {
                    "tool_call_id": tool_call.get("id", fn),
                    "tool": fn,
                    "arguments": arguments,
                    "result": result,
                    "success": True
                }

            except Exception as e:
                tool_latency = (time.time() - tool_start) * 1000
                logger.error(f"[ReAct] 工具执行失败: {fn}, 错误: {e}")
                if self.enable_optimizations:
                    self.metrics_collector.increment(f"tool_{fn}_errors")
                return {
                    "tool_call_id": tool_call.get("id", fn),
                    "tool": fn,
                    "arguments": arguments,
                    "result": f"工具执行失败: {str(e)}",
                    "success": False,
                    "error": str(e)
                }

        tool_results = []
        if len(unique_calls) > 1:
            with ThreadPoolExecutor(max_workers=min(len(unique_calls), 4)) as pool:
                futures = {pool.submit(_exec_one, tc): tc for tc in unique_calls}
                for fut in as_completed(futures):
                    tool_results.append(fut.result())
        else:
            tool_results.append(_exec_one(unique_calls[0]))

        return tool_results

    def _build_system_prompt(self, context: Dict) -> str:
        """构建系统提示词（客服专用）"""
        prompt = """你是一个专业的客服助手，具备知识检索和客户服务能力。你需要通过调用工具来帮助用户解决问题。

工作流程：
1. 分析用户问题，判断是知识咨询还是需要客服介入
2. 如果是知识问题，先用 hybrid_search 检索知识库
3. 如果用户情绪明显，调用 analyze_sentiment 分析情感
4. 如果需要人工跟进，调用 create_ticket 创建工单
5. 给出温暖、专业的回答

【知识检索工具】
- hybrid_search: 混合检索知识库，回答用户的知识类问题
- vector_search: 语义检索
- keyword_search: 关键词检索

【客服工具】
- analyze_sentiment: 分析用户情绪（正面/负面/中性）
- create_ticket: 创建客服工单（投诉、问题反馈）
- query_ticket: 查询工单状态
- query_user_info: 查询用户信息和历史记录

【决策策略】
- 知识咨询：hybrid_search → 回答
- 投诉/反馈：analyze_sentiment → create_ticket → 安抚回复
- 工单跟进：query_ticket → 告知进度
- 个性化服务：query_user_info → 针对性回答

【语气要求】
- 使用同理心语言，表达理解
- 情绪负面时优先安抚，再解决问题
- 提供明确的解决方案或时间表
- 避免推诿或模糊回答

【注意事项】
- 每次调用1-3个工具，不要过多
- 最多5轮迭代，避免过度检索
- 信息充足时立即给出答案
- 情绪负面时，优先考虑创建工单
"""

        # 添加情感上下文
        if context.get("urgent"):
            sentiment = context.get("sentiment", {})
            prompt += f"""
【重要提示】
当前用户情绪：{sentiment.get('sentiment', 'negative')}（评分：{sentiment.get('score', -0.7):.2f}）
用户情绪强烈负面，请：
1. 优先考虑创建工单（create_ticket）
2. 使用同理心语言，表达理解和歉意
3. 提供明确的解决方案或时间表
4. 避免推诿或模糊回答
"""

        return prompt

    def _handle_react(self, query: str, context: Dict, start_time: float) -> str:
        """ReAct模式处理查询"""
        logger.info("[ReAct] 开始ReAct循环")
        logger.info(f"[ReAct查询] {query}")

        tools = self.tool_registry.get_tools_schema()

        if not tools:
            logger.warning("[ReAct] 无可用工具")
            return "客服工具未初始化，无法处理查询。"

        logger.info(f"[ReAct工具] 可用工具数量: {len(tools)}")

        system_prompt = self._build_system_prompt(context)

        # 获取对话历史（由 LangGraph 共享记忆层注入）
        history_messages = context.get("history", [])

        messages = [
            {"role": "system", "content": system_prompt},
            *history_messages,
            {"role": "user", "content": query}
        ]

        max_iterations = 5
        retrieval_count = 0
        seen_tool_calls = set()

        for iteration in range(max_iterations):
            logger.info(f"[ReAct] 第 {iteration + 1}/{max_iterations} 轮")
            self._emit_progress(context, f"ReAct 第{iteration + 1}轮：正在思考...", stage="thinking")

            try:
                response = self.llm.chat(messages, tools=tools, temperature=0.3)
                logger.info(f"[ReAct思考] LLM响应内容: {response[:500]}{'...' if len(response) > 500 else ''}")

                tool_calls = parse_tool_calls(response)

                if not tool_calls:
                    logger.info("[ReAct] Agent决定给出最终答案")
                    self._emit_progress(context, f"ReAct 第{iteration + 1}轮：生成最终答案", stage="answering")

                    if self.enable_optimizations:
                        total_time = time.time() - start_time
                        self.metrics_collector.record("react_total_latency_ms", total_time * 1000)
                        self.metrics_collector.record("react_iterations", iteration + 1)

                    if response.startswith("[内部]"):
                        messages.append({
                            "role": "user",
                            "content": "请直接给出用户友好的答案，不要输出工具调用信息。"
                        })
                        response = self.llm.chat(messages, temperature=0.5)

                    return response

                # 发射工具调用事件
                tool_names = [tc["function"]["name"] for tc in tool_calls]
                self._emit_progress(context, f"ReAct 第{iteration + 1}轮：调用工具 {', '.join(tool_names)}", stage="tool_call")

                tool_results = self._execute_tool_calls(tool_calls, retrieval_count, seen_tool_calls)

                if not tool_results:
                    logger.info("[ReAct] 所有工具调用均为重复，强制收敛")
                    messages.append({"role": "user", "content": "请基于已有信息给出最终答案。"})
                    return self.llm.chat(messages, temperature=0.5)

                for tr in tool_results:
                    if tr.get("tool") in ["vector_search", "keyword_search", "hybrid_search"]:
                        retrieval_count += 1

                # 发射工具结果事件
                result_summary = ", ".join([tr["tool"] for tr in tool_results])
                self._emit_progress(context, f"工具执行完成: {result_summary}", stage="tool_result")

                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tr["tool_call_id"],
                            "type": "function",
                            "function": {
                                "name": tr["tool"],
                                "arguments": json.dumps(tr["arguments"], ensure_ascii=False)
                            }
                        }
                        for tr in tool_results
                    ]
                })

                for tr in tool_results:
                    result_str = str(tr["result"]) if not isinstance(tr["result"], str) else tr["result"]
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tr["tool_call_id"],
                        "content": self._truncate_tool_result(result_str)
                    })

                if iteration >= max_iterations - 2:
                    messages.append({
                        "role": "user",
                        "content": "这是最后一轮机会。请基于已有信息给出最佳答案，不要再调用工具。"
                    })

            except Exception as e:
                logger.error(f"[ReAct] 第{iteration + 1}轮执行失败: {e}", exc_info=True)

                if iteration == 0:
                    logger.warning("[ReAct] 首轮失败")
                    return f"客服处理失败: {str(e)}"

                messages.append({
                    "role": "user",
                    "content": "工具执行出现问题，请基于已有信息给出答案。"
                })

                try:
                    return self.llm.chat(messages, temperature=0.5)
                except:
                    return "抱歉，处理您的请求时遇到了问题。请稍后再试。"

        logger.warning(f"[ReAct] 达到最大迭代次数 {max_iterations}，强制生成答案")
        messages.append({
            "role": "user",
            "content": "请基于目前已有的所有信息给出最终答案，不要再调用工具。如果信息不足，请诚实说明。"
        })

        try:
            return self.llm.chat(messages, temperature=0.5)
        except Exception as e:
            logger.error(f"[ReAct] 生成最终答案失败: {e}")
            return "抱歉，我无法完成您的请求。请尝试重新表述您的问题。"

    def handle(self, query: str, context: Dict) -> str:
        """
        处理查询（ReAct模式 + 对话记忆）

        Args:
            query: 用户查询
            context: 上下文信息

        Returns:
            处理结果
        """
        start_time = time.time()

        try:
            # 缓存查询（demo模式下跳过，确保面试官看到完整流程）
            if self.enable_cache and self.enable_optimizations and self.cache_manager:
                cached_answer = self.cache_manager.query_cache.get(query)
                if cached_answer:
                    logger.info("[业务缓存命中] 直接返回缓存答案")
                    return cached_answer

            # 记录查询
            if self.enable_optimizations:
                self.structured_logger.log_query(query=query, user_id=context.get("user_id", "anonymous"))
                self.metrics_collector.increment("total_queries")

            logger.info("[CustomerServiceAgent] 开始ReAct模式处理查询")

            # ReAct循环
            result = self._handle_react(query, context, start_time)

            # 缓存结果
            if self.enable_optimizations and self.cache_manager:
                query_cost_ms = (time.time() - start_time) * 1000
                self.cache_manager.query_cache.set_with_cost(query, result, query_cost_ms)

            return result

        except Exception as e:
            logger.error(f"处理查询失败: {e}", exc_info=True)

            if self.enable_optimizations and self.error_handler:
                self.structured_logger.log_error("query_processing_error", str(e))
                self.metrics_collector.increment("errors")
                return self.error_handler.handle(e, "unknown")
            else:
                return f"处理失败: {str(e)}"

    # ==================== 检索工具实现 ====================

    def vector_search(self, query: str, top_k: int = 3) -> List[Dict]:
        """向量检索"""
        if not self.retriever:
            return []

        try:
            logger.info(f"[向量检索] 查询: '{query}', top_k: {top_k}")
            results = self.retriever.retrieve(query, top_k=top_k)

            return [
                {
                    "content": doc.get("content", ""),
                    "score": doc.get("score", 0.0),
                    "metadata": doc.get("metadata", {})
                }
                for doc in results
            ]
        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            return []

    def keyword_search(self, query: str, top_k: int = 3) -> List[Dict]:
        """关键词检索"""
        if not self.bm25_retriever:
            return []

        try:
            results = self.bm25_retriever.retrieve(query, top_k=top_k)

            return [
                {
                    "content": doc.get("content", ""),
                    "score": doc.get("score", 0.0),
                    "metadata": doc.get("metadata", {})
                }
                for doc in results
            ]
        except Exception as e:
            logger.error(f"关键词检索失败: {e}")
            return []

    def hybrid_search(
        self,
        query: str,
        top_k: int = 3,
        vector_weight: float = 0.8
    ) -> List[Dict]:
        """混合检索（RRF融合）"""
        vector_results = self.vector_search(query, top_k=top_k * 2)
        keyword_results = self.keyword_search(query, top_k=top_k * 2)

        # RRF融合
        k = 60
        scores = {}
        doc_map = {}

        for rank, doc in enumerate(vector_results):
            doc_id = doc.get("id") or doc.get("content", "")[:50]
            scores[doc_id] = scores.get(doc_id, 0) + vector_weight / (k + rank + 1)
            if doc_id not in doc_map:
                doc_map[doc_id] = doc

        for rank, doc in enumerate(keyword_results):
            doc_id = doc.get("id") or doc.get("content", "")[:50]
            scores[doc_id] = scores.get(doc_id, 0) + (1 - vector_weight) / (k + rank + 1)
            if doc_id not in doc_map:
                doc_map[doc_id] = doc

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in sorted_docs[:top_k]:
            doc = doc_map.get(doc_id)
            if doc:
                results.append({
                    "content": doc.get("content", ""),
                    "score": score,
                    "metadata": doc.get("metadata", {})
                })

        return results

    # ==================== 客服工具实现 ====================

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """情感分析：分析用户情绪（正面/负面/中性）"""
        try:
            logger.info("[情感分析] 开始分析")

            prompt = f"""请分析以下文本的情感倾向。

文本：{text}

请输出JSON格式：
{{
    "sentiment": "positive/negative/neutral",
    "score": -1到1之间的分数（-1最负面，0中性，1最正面），
    "keywords": ["关键词1", "关键词2"],
    "reason": "简短说明"
}}

只输出JSON："""

            response = self.llm.generate(prompt, temperature=0.1, max_tokens=200)
            result = parse_llm_json(response, fallback=None)

            if result is None:
                negative_keywords = ["差", "烂", "垃圾", "投诉", "退款", "骗", "坑", "失望", "愤怒", "不满"]
                positive_keywords = ["好", "棒", "优秀", "满意", "感谢", "赞", "喜欢", "推荐"]

                neg_count = sum(1 for kw in negative_keywords if kw in text)
                pos_count = sum(1 for kw in positive_keywords if kw in text)

                if neg_count > pos_count:
                    sentiment = "negative"
                    score = -0.7
                elif pos_count > neg_count:
                    sentiment = "positive"
                    score = 0.7
                else:
                    sentiment = "neutral"
                    score = 0.0

                result = {
                    "sentiment": sentiment,
                    "score": score,
                    "keywords": [],
                    "reason": "基于关键词的简单判断"
                }

            logger.info(f"[情感分析] 结果: {result.get('sentiment')}, 分数: {result.get('score')}")
            return result

        except Exception as e:
            logger.error(f"情感分析失败: {e}")
            return {
                "sentiment": "neutral",
                "score": 0.0,
                "keywords": [],
                "reason": f"分析失败: {str(e)}"
            }

    def create_ticket(self, issue: str, user_id: str = None, priority: str = "normal") -> Dict:
        """创建客服工单"""
        try:
            if not self.crm:
                return {"error": "CRM系统未初始化"}

            logger.info(f"[创建工单] 问题: {issue}, 用户: {user_id}, 优先级: {priority}")

            ticket = self.crm.create_ticket(
                user_id=user_id,
                issue=issue,
                priority=priority,
                description=issue
            )

            logger.info(f"[创建工单] 成功创建工单: {ticket.get('ticket_id')}")
            return ticket

        except Exception as e:
            logger.error(f"创建工单失败: {e}")
            return {"error": f"创建工单失败: {str(e)}"}

    def query_ticket(self, ticket_id: str) -> Dict:
        """查询工单状态"""
        try:
            if not self.crm:
                return {"error": "CRM系统未初始化"}

            logger.info(f"[查询工单] 工单ID: {ticket_id}")

            ticket = self.crm.get_ticket(ticket_id)

            if ticket:
                logger.info(f"[查询工单] 找到工单: {ticket_id}, 状态: {ticket.get('status')}")
                return ticket
            else:
                logger.warning(f"[查询工单] 未找到工单: {ticket_id}")
                return {"error": f"未找到工单: {ticket_id}"}

        except Exception as e:
            logger.error(f"查询工单失败: {e}")
            return {"error": f"查询工单失败: {str(e)}"}

    def query_user_info(self, user_id: str) -> Dict:
        """查询用户信息"""
        try:
            if not self.crm:
                return {"error": "CRM系统未初始化"}

            logger.info(f"[查询用户] 用户ID: {user_id}")

            user = self.crm.get_user_info(user_id)

            if user:
                logger.info(f"[查询用户] 找到用户: {user_id}, VIP等级: {user.get('vip_level')}")
                return user
            else:
                logger.warning(f"[查询用户] 未找到用户: {user_id}")
                return {"error": f"未找到用户: {user_id}"}

        except Exception as e:
            logger.error(f"查询用户失败: {e}")
            return {"error": f"查询用户失败: {str(e)}"}
