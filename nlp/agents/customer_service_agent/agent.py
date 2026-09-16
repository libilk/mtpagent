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
from core.write_ops import record_write_op
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
        """初始化客服模块

        工单和会员数据来自 database/ecommerce.db（真实 SQLite 表），
        不再用 crm_mock.py 里那套硬编码的假数据。
        """
        try:
            from core.ecommerce_crm import EcommerceCRM
            self.crm = crm or EcommerceCRM()
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
        """注册工具（11 个 = 3 检索 + 4 客服 + 4 售后办理）"""
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
            description="为用户创建售后工单，适合投诉、问题反馈、需要人工跟进的场景",
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
                        "description": "优先级。应依据 analyze_sentiment 的结果来定："
                                       "negative 且 score<=-0.7 用 urgent，negative 用 high，"
                                       "neutral 用 normal，positive 用 low"
                    },
                    "order_id": {
                        "type": "string",
                        "description": "关联的订单号（可选，但建议带上，便于客服定位问题）"
                    },
                    "category": {
                        "type": "string",
                        "enum": ["退货", "换货", "物流", "发票", "投诉", "咨询"],
                        "default": "咨询",
                        "description": "工单分类"
                    },
                    "sentiment": {
                        "type": "string",
                        "enum": ["positive", "neutral", "negative"],
                        "description": "analyze_sentiment 返回的情绪结果，一并留档"
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

        # ========== 售后办理工具（阶段 3 新增）==========
        # 这四个把 Agent 从"只能答问题"变成"能办事"。
        # 前三个是只读查询，最后一个是写操作（会触发人工审批）。

        # 8. 查订单
        self.tool_registry.register_tool(
            name="query_order",
            description="按订单号查询订单详情（状态、金额、商品明细、收货信息）。"
                        "办理退货/换货前必须先调用本工具确认订单真实存在且状态允许售后。",
            parameters={
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "订单号，形如 SO20260909001"
                    }
                },
                "required": ["order_id"]
            },
            function=self.query_order
        )

        # 9. 查物流
        self.tool_registry.register_tool(
            name="query_logistics",
            description="按订单号查询物流轨迹（承运商、运单号、当前状态、签收时间）。"
                        "订单存在但查不到物流，说明尚未发货。",
            parameters={
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "订单号，形如 SO20260909001"
                    }
                },
                "required": ["order_id"]
            },
            function=self.query_logistics
        )

        # 10. 查退款进度
        self.tool_registry.register_tool(
            name="query_refund_status",
            description="查询退款/退货申请的处理进度。按订单号或退款单号查皆可。",
            parameters={
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "订单号（可选）"
                    },
                    "refund_id": {
                        "type": "string",
                        "description": "退款单号，形如 RF20260916001（可选）"
                    }
                },
                "required": []
            },
            function=self.query_refund_status
        )

        # 11. 提交退换货申请
        self.tool_registry.register_tool(
            name="submit_return_request",
            description="★写操作★ 为用户提交退换货申请。调用前必须先 query_order 确认订单"
                        "归属和状态，并用 hybrid_search 检索政策确认时限与运费规则。"
                        "本操作会改动资金相关数据，将触发人工审批。",
            parameters={
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "订单号"
                    },
                    "user_id": {
                        "type": "string",
                        "description": "申请人的用户ID，必须与订单归属一致"
                    },
                    "reason": {
                        "type": "string",
                        "description": "申请原因，如「耳机右耳无声，属质量问题」"
                    },
                    "refund_type": {
                        "type": "string",
                        "enum": ["退货退款", "仅退款", "换货"],
                        "default": "退货退款",
                        "description": "售后类型"
                    }
                },
                "required": ["order_id", "user_id", "reason"]
            },
            function=self.submit_return_request
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

        # ★ 写操作登记必须在这里做（调用线程），不能放进工具函数内部：
        # 上面那个并行分支会让工具跑在 worker 线程里，而写操作记录器是**线程本地**的。
        # 在 worker 线程里登记，信号会丢失；更糟的是会残留在被复用的线程上，
        # 污染下一个恰好调度到该线程的请求。
        self._record_write_operations(tool_results)

        return tool_results

    @staticmethod
    def _parse_tool_result(result) -> Dict:
        """把工具返回值还原成字典。

        tool_registry.call_tool() 统一返回 str()，字典会被转成 Python repr，
        这里用 ast.literal_eval 安全地还原（不用 eval，避免执行任意代码）。
        """
        if isinstance(result, dict):
            return result
        if isinstance(result, str):
            try:
                import ast
                parsed = ast.literal_eval(result)
                if isinstance(parsed, dict):
                    return parsed
            except (ValueError, SyntaxError):
                pass
        return {"raw": str(result)[:300]}

    def _record_write_operations(self, tool_results: List[Dict]) -> None:
        """把本轮**成功执行**的写操作登记到当前线程，供编排层判断是否需要人工审批。"""
        for tr in tool_results:
            if not tr.get("success") or tr.get("tool") not in self._WRITE_TOOLS:
                continue

            detail = self._parse_tool_result(tr.get("result"))
            if detail.get("error"):
                continue        # 写失败了不算，别拿失败去触发审批

            record_write_op(tr["tool"], detail)

    def _build_system_prompt(self, context: Dict) -> str:
        """构建系统提示词（电商售后专用）"""
        prompt = """你是「云集优选」电商平台的售后客服助手。你的职责是解答售后政策、
查询订单与物流、受理退换货申请、处理投诉。

【最高优先级的三条铁律】
★ 政策条款必须来自知识库检索结果。禁止凭印象背条文 —— 售后政策有大量例外
  （哪些商品不支持七天无理由、运费谁承担、时限算几天），记错会直接引发纠纷。
★ 订单号、物流状态、退款进度必须来自工具返回的真实数据。查不到就如实说查不到，
  绝对不要编造一个"看起来合理"的状态或时间。
★ 涉及收货人手机号、详细地址等个人信息时，只回必要的部分，不要整条复述。
★ **不要臆造 ID。** 用户ID（形如 U10001）、订单号、单号都必须来自用户原话或工具返回。
  拿不到就留空或向用户索取，**绝对不要自己编一个"看起来像"的编号**。
  从订单查用户是最稳的做法：query_order 的返回里就有 user_id 字段，直接用它。
★ **办理类操作必须真的执行工具，不能只在回答里描述。** 如果你对用户说
  "已为您提交退货申请""已为您创建工单"，那么在这一轮对话里**必须真的调用过**
  submit_return_request 或 create_ticket，并且拿到它们返回的单号。
  **只描述而不调用，等于欺骗用户，是严重事故。**
  错误示范：查完订单和政策后直接写"已为您提交退货申请，单号 RFxxx"（其实没调用）。
  正确做法：查完信息后**继续调用** submit_return_request，用工具返回的单号来回答。
  经验值：完整退货流程通常需要 4~5 轮，别在第 3 轮就急着收工。

【工作流程】
1. 先判断用户意图属于下面哪一类，再选对应工具：

   政策咨询（"能退吗""运费谁出""几天到账"）
     → hybrid_search 检索政策 → 依据检索结果回答

   查订单 / 查物流（"我的货到哪了""订单什么状态"）
     → query_order 确认订单情况 → query_logistics 查物流

   申请退货 / 换货（"我要退货""东西坏了"）
     → ① query_order 确认订单存在、归属正确、状态允许售后
     → ② hybrid_search 检索政策，确认时限和运费规则
     → ③ submit_return_request 提交申请，并把政策依据一并告诉用户

   投诉 / 情绪激烈（"太差了""我要投诉"）
     → analyze_sentiment 判断情绪 → create_ticket 建工单（必要时提高优先级）→ 安抚

   工单跟进（"我那个工单怎么样了"）
     → query_ticket 查询进度

2. 用户没给订单号却又需要查订单时，**先向用户要订单号**，不要猜。

【工具清单】
知识检索：hybrid_search / vector_search / keyword_search
订单售后：query_order（查订单）/ query_logistics（查物流）/
          submit_return_request（提交退换货）/ query_refund_status（查退款进度）
客服办理：analyze_sentiment / create_ticket / query_ticket / query_user_info

【必须注意的业务陷阱】
- **不是所有商品都能七天无理由退货**。已激活的 3C 数码、贴身衣物、生鲜、定制商品等
  属于法定例外，这类商品只能走「质量问题」通道（时限更长、运费由平台承担）。
  遇到这类商品，务必检索政策确认，不要想当然地按七天无理由答复。
- **「无理由退货」和「质量问题退货」的成本完全不同**：前者运费用户自付、时限短；
  后者运费平台承担、时限长。判断错误会直接影响用户的钱包，要先分清是哪一种。
- **会员等级会影响售后权益**（免费退货次数、运费补贴额度）。用户提到会员身份或
  需要判断补贴时，用 query_user_info 查一下等级。
- 订单状态为「待发货」时**不存在退货问题**，应引导用户走取消订单。

【语气要求】
- 先共情、再办事。用户带着情绪来时，第一句要先表达理解，不要直接抛条款
- 给明确的时间表和下一步动作，不说"尽快""也许""可能"
- 政策对用户不利时，要解释清楚依据（引用检索到的条款），而不是生硬拒绝
- 不确定就说不确定，并给出人工介入的路径（建工单）

【注意事项】
- 每次可并行调用多个工具（如同时 query_order + query_logistics），减少轮次
- 信息够了就立刻回答，但**办理类操作没调用工具就不算办完**
- 一次能办完的事不要拆成多次追问
"""

        # 情绪 → 工单优先级的联动规则。
        #
        # 原先这里是一段 `if context.get("urgent")` 的分支，但**全项目没有任何地方
        # 写过 `urgent` 这个 key**，所以那段提示词从来没生效过（死代码）。
        #
        # 改成写死在提示词里的规则：让 LLM 先调 analyze_sentiment，再把情绪结果
        # 传给 create_ticket 的 priority 参数。这样不需要跨请求保存状态，
        # 也不会因为 Agent 实例被多线程共享而出错。
        prompt += """
【情绪与工单优先级的对应关系】
调过 analyze_sentiment 之后，把它返回的结果传给 create_ticket：
- sentiment = negative 且 score <= -0.7  →  priority = "urgent"
- sentiment = negative                    →  priority = "high"
- sentiment = neutral                     →  priority = "normal"
- sentiment = positive                    →  priority = "low"
同时把 sentiment 字段一起传进去，工单里会留档，便于后续分析。
如果用户明确提到"要投诉""要曝光""已经催过很多次"，即使情感分析没到负值，也按 high 起。
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

        # 迭代上限 8 轮。原先的 5 轮对完整售后流程不够用：
        # query_order → query_logistics → hybrid_search → query_user_info
        # → submit_return_request 就要 5 轮，再加一轮生成答案 = 6 轮。
        # 上限卡在 5 会把它逼到"没办成却编一个办成了的答案"。
        max_iterations = 8
        retrieval_count = 0
        seen_tool_calls = set()
        executed_tools = set()      # 本轮真正执行过的工具名，用于事后核验（见 _verify_write_claim）

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

                    return self._verify_write_claim(response, executed_tools)

                # 发射工具调用事件
                tool_names = [tc["function"]["name"] for tc in tool_calls]
                self._emit_progress(context, f"ReAct 第{iteration + 1}轮：调用工具 {', '.join(tool_names)}", stage="tool_call")

                tool_results = self._execute_tool_calls(tool_calls, retrieval_count, seen_tool_calls)

                if not tool_results:
                    logger.info("[ReAct] 所有工具调用均为重复，强制收敛")
                    messages.append({"role": "user", "content": "请基于已有信息给出最终答案。"})
                    return self._verify_write_claim(
                        self.llm.chat(messages, temperature=0.5), executed_tools
                    )

                for tr in tool_results:
                    if tr.get("tool") in ["vector_search", "keyword_search", "hybrid_search"]:
                        retrieval_count += 1
                    if tr.get("success"):
                        executed_tools.add(tr["tool"])

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
            return self._verify_write_claim(
                self.llm.chat(messages, temperature=0.5), executed_tools
            )
        except Exception as e:
            logger.error(f"[ReAct] 生成最终答案失败: {e}")
            return "抱歉，我无法完成您的请求。请尝试重新表述您的问题。"

    # ==================== 防幻觉核验 ====================

    # 会改动数据的工具。声称办过事，就必须真的调用过其中之一。
    _WRITE_TOOLS = {"submit_return_request", "create_ticket"}

    # 出现这些说法 = 在向用户宣称"事情已经办好了"
    _CLAIM_MARKERS = (
        "已为您提交", "已提交", "已受理", "已为您创建", "已创建工单",
        "申请单号", "退货单号", "退款单号", "工单号",
    )

    def _verify_write_claim(self, answer: str, executed_tools: set) -> str:
        """
        核验"声称已办理"的答复，是否真的有对应的工具调用。

        **为什么要做这件事：** ReAct 循环有个危险特性 —— LLM 可以在**没有真正调用**
        写工具的情况下，编出一个"已为您提交申请，单号 RF2026xxxx"的答复。
        对用户来说这和真办了没区别，直到他发现系统里查无此单。

        实测就踩到了：迭代次数用尽时，模型会照着"流程应该是怎样"补全一个结果，
        连单号都是编的（真实是 RF...004，它写了 RF...001）。

        提示词里虽然写了硬规则，但**提示词是软约束，模型可以不听**。
        所以这里加一道确定性兜底：核验不过就改写答复，明确告诉用户"没办成"。

        Args:
            answer: LLM 生成的最终答复
            executed_tools: 本轮真正成功执行过的工具名集合

        Returns:
            核验通过的原文，或修正后的诚实答复
        """
        # 真的调用过写工具 → 正常放行
        if executed_tools & self._WRITE_TOOLS:
            return answer

        # 没声称办过事 → 放行（例如只是回答政策咨询）
        hit = next((m for m in self._CLAIM_MARKERS if m in answer), None)
        if hit is None:
            return answer

        logger.error(
            f"[防幻觉] 答复中出现「{hit}」，但本轮未成功执行任何写操作"
            f"（已执行工具: {sorted(executed_tools)}），判定为编造，已改写答复"
        )

        return (
            "【重要】本次售后申请**并未成功提交**，请注意：\n"
            "上面若有单号，均为无效内容，请勿据此操作。\n\n"
            "建议您：\n"
            "1. 重新描述一次您的需求，我再为您办理；\n"
            "2. 或直接联系人工客服协助处理。\n\n"
            "---------- 以下为未经验证的生成内容，仅供了解政策参考 ----------\n"
            + answer
        )

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
                # LLM 解析失败时的兜底判定。词表是按**电商售后场景**挑的 ——
                # 原先是通用客服词表（差/烂/垃圾），抓不住售后用户真正的表达方式。
                #
                # 挑选原则：优先用不容易误伤的多字词。
                # 反例：「拖」能命中「拖延」（负面），也会命中「拖鞋」（中性），所以不取。
                # 同理不把单独的「退款」当负面信号 —— 用户问「退款多久到账」是正常咨询。
                negative_keywords = [
                    # 商品问题
                    "破损", "损坏", "坏了", "发错", "错发", "漏发", "少发", "缺件",
                    # 物流问题
                    "没收到", "未收到", "丢失", "一直没", "滞留",
                    # 资金问题
                    "退款慢", "没到账", "迟迟不", "还没退",
                    # 情绪与升级
                    "投诉", "差评", "垃圾", "骗", "太差", "失望", "愤怒", "不满",
                    "催了", "催过", "什么破", "曝光", "维权",
                ]
                positive_keywords = [
                    "满意", "感谢", "谢谢", "赞", "好评", "推荐",
                    "及时", "贴心", "很快", "服务好", "喜欢",
                ]

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

    def create_ticket(
        self,
        issue: str,
        user_id: str = None,
        priority: str = "normal",
        order_id: str = None,
        category: str = "咨询",
        sentiment: str = None,
    ) -> Dict:
        """创建售后工单"""
        try:
            if not self.crm:
                return {"error": "CRM系统未初始化"}

            logger.info(
                f"[创建工单] 问题: {issue}, 用户: {user_id}, "
                f"优先级: {priority}, 分类: {category}"
            )

            ticket = self.crm.create_ticket(
                user_id=user_id,
                issue=issue,
                priority=priority,
                description=issue,
                order_id=order_id,
                category=category,
                sentiment=sentiment,
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
                logger.info(f"[查询用户] 找到用户: {user_id}, 会员等级: {user.get('member_level')}")
                return user
            else:
                logger.warning(f"[查询用户] 未找到用户: {user_id}")
                return {"error": f"未找到用户: {user_id}"}

        except Exception as e:
            logger.error(f"查询用户失败: {e}")
            return {"error": f"查询用户失败: {str(e)}"}

    # ==================== 售后办理工具实现（阶段 3 新增）====================

    def query_order(self, order_id: str) -> Dict:
        """查订单详情（含商品明细）。"""
        try:
            if not self.crm:
                return {"error": "CRM系统未初始化"}

            logger.info(f"[查订单] 订单号: {order_id}")

            order = self.crm.query_order(order_id)
            if order:
                return order

            logger.warning(f"[查订单] 未找到订单: {order_id}")
            return {"error": f"未找到订单: {order_id}。请确认订单号是否正确。"}

        except Exception as e:
            logger.error(f"查订单失败: {e}")
            return {"error": f"查询订单失败: {str(e)}"}

    def query_logistics(self, order_id: str) -> Dict:
        """查物流轨迹。"""
        try:
            if not self.crm:
                return {"error": "CRM系统未初始化"}

            logger.info(f"[查物流] 订单号: {order_id}")

            logistics = self.crm.query_logistics(order_id)
            if logistics:
                return logistics

            # 区分"订单不存在"和"订单存在但没发货" —— 这两种情况该给用户的回答完全不同
            order = self.crm.query_order(order_id)
            if order:
                return {
                    "message": f"订单 {order_id} 当前状态为「{order['status']}」，"
                               f"尚未产生物流记录，暂无物流信息。",
                    "order_status": order["status"],
                }

            return {"error": f"未找到订单: {order_id}"}

        except Exception as e:
            logger.error(f"查物流失败: {e}")
            return {"error": f"查询物流失败: {str(e)}"}

    def query_refund_status(self, order_id: str = None, refund_id: str = None) -> Dict:
        """查退款进度。"""
        try:
            if not self.crm:
                return {"error": "CRM系统未初始化"}

            logger.info(f"[查退款] 订单号: {order_id}, 退款单号: {refund_id}")

            refunds = self.crm.query_refund(order_id=order_id, refund_id=refund_id)
            if refunds:
                # 一个订单可能有多笔退款，全返回让 LLM 自己归纳
                return {"refunds": refunds, "count": len(refunds)}

            target = refund_id or order_id
            return {"message": f"没有查到 {target} 对应的退款记录。"}

        except Exception as e:
            logger.error(f"查退款失败: {e}")
            return {"error": f"查询退款失败: {str(e)}"}

    def submit_return_request(
        self,
        order_id: str,
        user_id: str,
        reason: str,
        refund_type: str = "退货退款",
    ) -> Dict:
        """
        提交退换货申请（★写操作）。

        校验逻辑放在 CRM 层（订单存在 / 归属正确 / 状态允许），
        这里只负责调用和日志 —— Agent 层保持"薄"，业务规则不散落在两处。
        """
        try:
            if not self.crm:
                return {"error": "CRM系统未初始化"}

            logger.info(
                f"[提交退换货] 订单: {order_id}, 用户: {user_id}, 类型: {refund_type}"
            )

            result = self.crm.submit_return_request(
                order_id=order_id,
                user_id=user_id,
                reason=reason,
                refund_type=refund_type,
            )

            if result.get("error"):
                logger.warning(f"[提交退换货] 失败: {result['error']}")
            else:
                logger.info(f"[提交退换货] 成功: {result.get('refund_id')}")

            return result

        except Exception as e:
            logger.error(f"提交退换货失败: {e}")
            return {"error": f"提交退换货申请失败: {str(e)}"}
