# -*- coding: utf-8 -*-
"""
Database Agent
==============

专门处理数据库查询任务，擅长SQL生成、表结构探索、数据分析。

形态：这是一个"工具式"Agent —— 它自己不含数据，靠 ReAct（推理-行动循环：
想一步 → 调工具 → 看结果 → 再想，直到给出答案）调用自己在 _register_tools 里
登记的 4 个工具，其中最核心的是生成一段 SQL 交给 SQLite 执行，再把结果翻译成
自然语言。和 vqa_agent 那种"直接调多模态模型看图、不注册工具"的形态正好相反。

一个必须知道的事实：代码里到处写着 "SQLite MCP"（MCP = 模型上下文协议，
让 LLM 按统一协议调用外部工具的规范），但 core/sqlite_mcp_service.py 的
skip_mcp 默认就是 True（MCP Server 的 npm 包已下架）—— 所以今天真正生效的是
Python sqlite3（SQLite 的 Python 驱动）直连这条降级路径，MCP 分支根本进不去。
读下面的代码时请按 sqlite3 理解，否则会去找并不存在的 MCP 调用。
"""

import logging
import time
import json
from typing import Dict, List
from llm.output_parser import parse_llm_json, parse_tool_calls
from llm.langchain_tools import ToolRegistry
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class DatabaseAgent:
    """数据库查询Agent（ReAct模式）"""

    def _emit_progress(self, context: Dict, msg: str, **extra):
        """通过 StreamWriter（流式事件写入器）发射中间步骤事件（供前端 trace 面板展示）

        writer 不是 Agent 自己的东西，是图节点在调 handle() 之前塞进 context 的
        （见 nodes.py make_agent_node）—— Agent 内部拿不到图上下文，只能靠转交。
        这里 writer 为 None 就直接返回，所以脱离 LangGraph 单独调也不报错。
        """
        writer = context.get("_stream_writer")
        if writer is None:
            return
        agent_name = context.get("_agent_name", "database_agent")
        import time as _time
        event = {"event": "progress", "agent": agent_name, "msg": msg, "ts": _time.time()}
        event.update(extra)
        writer(event)

    def __init__(
        self,
        llm,
        tool_registry=None,
        enable_optimizations: bool = True,
        enable_cache: bool = True,
        cache_size: int = 500,
        cache_ttl: int = 3600,
        max_retries: int = 3,
    ):
        """
        初始化

        Args:
            llm: LLM实例
            function_calling: Function Calling管理器
            enable_optimizations: 是否启用优化（默认True）
            enable_memory: 是否启用对话记忆（默认True）
            max_history: 对话历史最大轮数（默认10）
            cache_size: 缓存大小（默认500）
            cache_ttl: 缓存TTL秒数（默认3600）
            max_retries: 最大重试次数（默认3）

        事实性说明（未改动代码）：上面 function_calling / enable_memory / max_history
        三项在下面的真实签名里并不存在，是早期版本的残留说明；可传的入参以 def __init__
        的签名为准。cache（缓存）= 把算过的答案存下来下次直接复用；
        TTL = 这条缓存最多存活多少秒，过期即失效。
        """
        self.llm = llm
        # tool_registry（工具注册中心）：Agent 名下的工具名册。
        # register_tool 按名字登记，call_tool 按名字取用 —— LLM 全程只见到名字和描述。
        self.tool_registry = tool_registry or ToolRegistry()
        self.enable_optimizations = enable_optimizations
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        self.cache_ttl = cache_ttl
        self.max_retries = max_retries

        # 初始化 SQLite 服务。名字里的 MCP 是历史遗留，实际落到 sqlite3 直连（见文件头）
        from core.sqlite_mcp_service import SQLiteMCPService
        try:
            self.sqlite_mcp = SQLiteMCPService()
            logger.info("✓ SQLite MCP 初始化完成")
        except Exception as e:
            logger.warning(f"SQLite MCP 初始化失败，将使用降级方案: {e}")
            self.sqlite_mcp = None

        # 初始化优化模块
        if enable_optimizations:
            self._init_optimization_modules()
        else:
            self.cache_manager = None
            self.retry_strategy = None
            self.error_handler = None

        # 表结构缓存
        self._table_schema_cache = {}

        # 注册工具
        self._register_tools()

        logger.info(f"DatabaseAgent初始化完成 (优化: {enable_optimizations})")

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

            logger.info("=" * 50)
            logger.info("DatabaseAgent 优化模块加载完成")
            logger.info("=" * 50)

        except Exception as e:
            logger.error(f"优化模块初始化失败: {e}", exc_info=True)
            logger.warning("降级为基础模式运行")
            self.enable_optimizations = False
            self.cache_manager = None
            self.retry_strategy = None
            self.error_handler = None

    def _register_tools(self):
        """注册工具（4个）

        这几个工具的 description 不是给人看的注释，而是提示词的一部分：
        模型只能靠"工具名 + 描述 + 参数结构（声明入参字段与类型）"来决定调哪个、
        怎么填参。所以描述里才反复强调"必须先调 list_tables / describe_table"。
        """
        if not self.sqlite_mcp:
            logger.warning("SQLite MCP 未初始化，跳过数据库工具注册")
            return

        # 1. 查询数据库
        self.tool_registry.register_tool(
            name="query_database",
            description="""查询SQLite数据库，执行SELECT语句并返回结果。

在使用此工具前，请先：
1. 调用 list_tables 了解数据库中有哪些表
2. 调用 describe_table 了解目标表的结构（列名、类型）
3. 根据表结构生成正确的SQL查询

注意：表名和字段名区分大小写，使用时请保持大小写一致。""",
            parameters={
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "SQL查询语句（SELECT语句），注意表名和字段名的大小写"
                    }
                },
                "required": ["sql"]
            },
            function=self.query_database
        )

        # 2. 列出数据库所有表
        self.tool_registry.register_tool(
            name="list_tables",
            description="列出数据库中所有表名。在查询数据库前应先调用此工具了解有哪些表。",
            parameters={
                "type": "object",
                "properties": {},
                "required": []
            },
            function=self.list_database_tables
        )

        # 3. 查看表结构
        self.tool_registry.register_tool(
            name="describe_table",
            description="查看指定表的结构（列名、类型、是否主键）。在生成SQL前应先了解表结构。",
            parameters={
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "要查看结构的表名（区分大小写）"
                    }
                },
                "required": ["table_name"]
            },
            function=self.describe_database_table
        )

        # 4. 查看样例数据
        self.tool_registry.register_tool(
            name="get_sample_data",
            description="查看指定表的前5行样例数据，帮助理解数据内容后生成更准确的SQL。",
            parameters={
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "要查看样例数据的表名（区分大小写）"
                    }
                },
                "required": ["table_name"]
            },
            function=self.get_sample_data
        )

    @staticmethod
    def _truncate_tool_result(result_str: str, max_len: int = 4000) -> str:
        """截断过长的工具结果，保留头尾

        工具结果会被原样塞回消息列表再喂给模型，不截断会撑爆上下文窗口、
        也白烧 token（词元：模型处理文本的最小单位，长度和计费都按它算）。
        保留头尾是因为中间通常是大段重复行，头尾才是列名和汇总。
        """
        if len(result_str) <= max_len:
            return result_str
        keep = max_len // 2
        truncated = len(result_str) - max_len
        logger.info(f"[ReAct] 工具结果截断了 {truncated} 个字符")
        return result_str[:keep] + f"\n...[截断了 {truncated} 个字符]...\n" + result_str[-keep:]

    def _execute_tool_calls(self, tool_calls: List[Dict], retrieval_count: int,
                            seen_tool_calls: set = None) -> List[Dict]:
        """
        执行工具调用（支持去重 + 并行 + JSON容错）

        三处非直观的地方：
        - 去重：seen_tool_calls 由调用方跨轮次传进来，模型反复要调同一个工具时会命中，
          直接跳过，省下一次真实查询。
        - 并行：多个调用同时发（下面的 ThreadPoolExecutor）；结果用 as_completed 收集，
          所以返回顺序和入参顺序并不一致，别按位置一一对应。
        - JSON 容错：模型偶尔吐出非法 JSON，json.loads 失败就退到 parse_llm_json
          （fallback = 降级兜底：主路径失败时退到备用方案）。
        - retrieval_count 这个形参在本函数内自始至终没被用到，属死参数，保持原样未动。
        """
        if seen_tool_calls is None:
            seen_tool_calls = set()

        # 去重
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

        # 并行/串行执行
        tool_results = []
        if len(unique_calls) > 1:
            with ThreadPoolExecutor(max_workers=min(len(unique_calls), 4)) as pool:
                futures = {pool.submit(_exec_one, tc): tc for tc in unique_calls}
                for fut in as_completed(futures):
                    tool_results.append(fut.result())
        else:
            tool_results.append(_exec_one(unique_calls[0]))

        return tool_results

    def _build_system_prompt(self) -> str:
        """构建系统提示词（增强表名选择逻辑）

        下面这段是原文，一字未改。读 few-shot（少量示例）时注意三个刻意的设计：
        - 示例二专门示范 JOIN（联表查询）：订单在 orders、物流在 logistics，
          两张表靠 order_id 关联。不示范的话模型很容易只查一张表就作答。
        - 防幻觉约束：示例末尾写明"查不到物流记录 = 订单还没发货，不要编造轨迹" ——
          模型凭空造数据的毛病，只能靠提示词里的禁令配合示例来压。
        - 示例给的是完整工具链路（list_tables → describe_table → query_database），
          目的是让它学"调用次序"，而不是背下示例里那条 SQL。
        """
        return """你是一个专业的数据库查询助手。你通过调用工具来查询SQLite数据库并回答用户问题。

【最重要的规则】
★ 你必须调用 `query_database` 工具来执行SQL并获取真实数据，然后根据查询结果给出自然语言回答。
★ 绝对禁止只输出SQL语句而不执行。用户需要的是查询结果，不是SQL代码。
★ 可以在一次请求中同时调用多个工具（如同时 describe_table 多张表），以减少轮次。

【工作流程】
1. **获取概览**: 调用 `list_tables` 获取所有表名及用途描述。
2. **了解结构**: 调用 `describe_table` 确认目标表的列名和类型。可同时查看多个表。
3. **执行查询**: 生成 SQL 并调用 `query_database` 执行，获取真实数据。
4. **回答用户**: 根据查询结果，用自然语言清晰地回答用户问题。

【工具规范】
- `list_tables`: 列出表名及其用途（第一步必调）。
- `describe_table`: 查看表列信息（生成SQL前必调，可一次调用多个表）。
- `query_database`: 执行 SELECT 查询并返回结果。★必须调用此工具获取数据★
- `get_sample_data`: 查看 5 条样例数据（不确定字段含义时使用）。

【SQL 生成规范】
- 表名和字段名区分大小写。
- 优先使用 LIMIT 限制数量。
- 涉及关联查询时明确指定 JOIN 条件。

【Few-shot 示例】

示例一（单表查询 —— 查订单状态）:
用户: "订单 SO20260909001 现在什么状态？"
→ 调用 `list_tables` → 看到 orders（订单主表）
→ 调用 `describe_table("orders")` → 确认有 order_id、status、order_date 字段
→ 调用 `query_database("SELECT order_id, status, order_date FROM orders WHERE order_id = 'SO20260909001'")` → 返回「已签收」
→ 最终回答: "订单 SO20260909001 状态为「已签收」，下单时间是 2026-09-09。"

示例二（关联查询 —— 查物流，必须 JOIN）:
用户: "SO20260909001 这个订单的快递到哪了？"
→ 订单状态在 orders 表，物流轨迹在 logistics 表，两张表用 order_id 关联
→ 调用 `describe_table("logistics")` → 确认有 order_id、carrier、tracking_no、status、delivered_at
→ 调用 `query_database("SELECT o.order_id, l.carrier, l.tracking_no, l.status, l.delivered_at FROM orders o JOIN logistics l ON o.order_id = l.order_id WHERE o.order_id = 'SO20260909001'")`
→ 最终回答: "该订单由顺丰速运承运，运单号 SF1234567890，已于 2026-09-11 15:42 签收。"

注意事项: 查不到某个订单的物流记录，通常意味着订单还没发货 ——
这时要如实回答「该订单尚未发货，暂无物流信息」，不要编造轨迹。

【注意事项】
- 每次可调用 1-3 个工具，尽量并行调用以减少轮次。
- 必须拿到 `query_database` 的真实结果后，才能给出最终答案。
"""

    def _handle_react(self, query: str, context: Dict, start_time: float) -> str:
        """
        ReAct 模式处理查询

        每一轮固定动作：把消息发给模型 → 若它返回 tool_calls 就执行工具、把结果
        作为 role="tool" 消息追回去 → 下一轮；若它没返回 tool_calls，说明模型认为
        信息够了，这一轮的文本就是最终答案。所以循环只有两个出口：模型主动收口，
        或者撞上 max_iterations 上限被强制收口。
        """
        logger.info("[ReAct] 开始ReAct循环")
        logger.info(f"[ReAct查询] {query}")

        tools = self.tool_registry.get_tools_schema()

        if not tools:
            logger.warning("[ReAct] 无可用工具")
            return "数据库工具未初始化，无法处理查询。"

        logger.info(f"[ReAct工具] 可用工具数量: {len(tools)}")

        system_prompt = self._build_system_prompt()

        # 获取对话历史（由 LangGraph 共享记忆层注入）
        # 键名固定是 "history"，写入方是 nodes.py make_agent_node（不是本 Agent 自己填的）；
        # 脱离图单独调用时这里是空列表，历史就丢了。
        history_messages = context.get("history", [])

        messages = [
            {"role": "system", "content": system_prompt},
            *history_messages,
            {"role": "user", "content": query}
        ]

        max_iterations = 8  # 循环上限：防模型反复调工具不收敛，同时也是成本兜底
        retrieval_count = 0
        seen_tool_calls = set()

        for iteration in range(max_iterations):
            logger.info(f"[ReAct] 第 {iteration + 1}/{max_iterations} 轮")
            self._emit_progress(context, f"ReAct 第{iteration + 1}轮：正在思考...", stage="thinking")

            try:
                # temperature（采样温度）：越低输出越确定、越高越发散。调工具这一轮
                # 要的是可复现的 SQL，所以用 0.3；比闲聊场景的 0.7 明显更冷。
                response = self.llm.chat(messages, tools=tools, temperature=0.3)
                if response is None:
                    response = ""
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

                retrieval_count += len(tool_results)

                # 发射工具结果事件
                for tr in tool_results:
                    tool_name = tr["tool"]
                    args = tr.get("arguments", {})
                    if tool_name == "query_database":
                        sql_text = args.get("sql", "")
                        self._emit_progress(context, f"执行SQL: {sql_text[:80]}", stage="tool_result")
                    elif tool_name == "describe_table":
                        self._emit_progress(context, f"查看表结构: {args.get('table_name', '')}", stage="tool_result")
                    elif tool_name == "list_tables":
                        self._emit_progress(context, f"获取数据库表列表", stage="tool_result")
                    elif tool_name == "get_sample_data":
                        self._emit_progress(context, f"查看样例数据: {args.get('table_name', '')}", stage="tool_result")
                    else:
                        self._emit_progress(context, f"工具 {tool_name} 执行完成", stage="tool_result")

                # 标准 tool calling 消息格式，顺序和 ID 都不能错：
                # assistant 那条要把 content 置 None、用 tool_calls 声明"我要调这几个"；
                # 紧跟其后的 role="tool" 消息靠 tool_call_id 与之一一对应。
                # 一旦顺序错开或 ID 对不上，OpenAI 兼容接口会直接报错。
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

                # 提前两轮就打招呼：不留缓冲的话，模型常常在最后一轮被截断，
                # 答案会草草收尾甚至半句。这句相当于给它一个主动收口的机会。
                if iteration >= max_iterations - 2:
                    messages.append({
                        "role": "user",
                        "content": "这是最后一轮查询机会。请基于已有信息给出最佳答案，不要再调用工具。"
                    })

            except Exception as e:
                logger.error(f"[ReAct] 第{iteration + 1}轮执行失败: {e}", exc_info=True)

                if iteration == 0:
                    logger.warning("[ReAct] 首轮失败")
                    return f"数据库查询处理失败: {str(e)}"

                messages.append({
                    "role": "user",
                    "content": "工具执行出现问题，请基于已有信息给出答案。"
                })

                try:
                    return self.llm.chat(messages, temperature=0.5)
                except:
                    return "抱歉，处理您的请求时遇到了问题。请稍后再试。"

        # 达到最大迭代次数
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
        handle（处理入口）：Agent 对外统一的方法签名，图节点只认它。

        处理查询（ReAct模式 + 对话记忆）

        Args:
            query: 用户查询
            context: 上下文信息

        Returns:
            处理结果

        注意缓存（cache）的键就是 query 原文、不含 history —— 同一句话在不同对话
        上下文里会命中同一条缓存。这是刻意的取舍：省下一整轮 ReAct 的开销。
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

            logger.info("[DatabaseAgent] 开始ReAct模式处理查询")

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

    # ==================== 工具实现 ====================

    def query_database(self, sql: str) -> str:
        """
        查询数据库（通过 SQLite MCP）

        事实性更正（未改动原描述）：实际走的是 sqlite3 直连而非 MCP —— 原注释里的
        "通过 SQLite MCP"已不成立，默认 skip_mcp=True，详见文件头。
        """
        try:
            if not self.sqlite_mcp:
                return "SQLite MCP 未初始化，无法查询数据库"

            logger.info(f"[数据库查询] SQL: {sql}")

            # 安全检查：只允许SELECT语句
            # 这是写操作（会改数据的操作）的第一道闸门。真正要改数据的动作不走这里，
            # 而是走 core/write_ops.py 登记 + 人工审批那条独立通道。
            if not sql.strip().upper().startswith("SELECT"):
                return "安全限制：只允许执行SELECT查询"

            result = self.sqlite_mcp.query(sql)

            if result.get("success"):
                data = result.get("data", [])
                columns = result.get("columns", [])

                if not data:
                    return "查询成功，但没有找到数据"

                formatted = "【查询结果】\n\n"

                if columns:
                    formatted += " | ".join(columns) + "\n"
                    formatted += "-" * (len(" | ".join(columns))) + "\n"

                # 只回显前 10 行：结果集会整段回到提示词里，不限制就会挤掉历史对话。
                # 但总数故意如实告知 —— 让模型知道"这是被截断的"，避免它以为只有 10 条。
                for i, row in enumerate(data[:10]):
                    formatted += " | ".join(str(cell) for cell in row) + "\n"

                if len(data) > 10:
                    formatted += f"\n... 还有 {len(data) - 10} 行数据"

                formatted += f"\n\n总计: {len(data)} 条记录"

                logger.info(f"[数据库查询] 查询成功，返回 {len(data)} 条记录")
                return formatted
            else:
                error = result.get("error", "查询失败")
                logger.error(f"[数据库查询] {error}")
                return f"查询失败: {error}"

        except Exception as e:
            logger.error(f"数据库查询失败: {e}")
            return f"数据库查询失败: {str(e)}"

    def list_database_tables(self, _input: str = "") -> str:
        """列出数据库中所有表名及其概览"""
        try:
            if not self.sqlite_mcp:
                return "SQLite MCP 未初始化，无法查询数据库"

            logger.info("[数据库] 获取所有表及其概览")
            tables_info = self.sqlite_mcp.list_tables()

            if not tables_info:
                return "数据库中没有找到任何表"

            formatted = "【数据库表及概览列表】\n\n"
            for i, info in enumerate(tables_info, 1):
                name = info.get("table_name", "?")
                desc = info.get("description", "无描述")
                formatted += f"{i}. **{name}**: {desc}\n"
            formatted += f"\n共 {len(tables_info)} 张表"

            logger.info(f"[数据库] 找到 {len(tables_info)} 张表及概览信息")
            return formatted

        except Exception as e:
            logger.error(f"列出数据库表失败: {e}")
            return f"列出数据库表失败: {str(e)}"

    def describe_database_table(self, table_name: str) -> str:
        """查看指定表的结构"""
        try:
            if not self.sqlite_mcp:
                return "SQLite MCP 未初始化，无法查询数据库"

            # 检查缓存
            # 表结构在一个进程生命周期内不会变，所以这份缓存没有 TTL（对比 __init__ 里
            # 那个带 cache_ttl 的答案缓存）—— 只要进程活着就一直有效。
            if table_name in self._table_schema_cache:
                logger.info(f"[数据库] 表结构缓存命中: {table_name}")
                return self._table_schema_cache[table_name]

            logger.info(f"[数据库] 查看表结构: {table_name}")
            result = self.sqlite_mcp.describe_table(table_name)

            if result.get("success"):
                data = result.get("data", [])

                if not data:
                    return f"表 '{table_name}' 不存在或没有列信息"

                formatted = f"【表 {table_name} 的结构】\n\n"
                formatted += "列名 | 类型 | 是否主键\n"
                formatted += "--- | --- | ---\n"

                # 下面的下标不是随便取的偏移量：这张表来自 SQLite 的 PRAGMA
                # table_info（SQLite 的配置指令），它的列序固定为
                # cid(0) / name(1) / type(2) / notnull(3) / dflt_value(4) / pk(5)，
                # 所以 row[1] 一定是列名、row[2] 是类型、row[5] 才是"是不是主键"。
                for row in data:
                    col_name = row[1] if len(row) > 1 else "?"
                    col_type = row[2] if len(row) > 2 else "?"
                    is_pk = "是" if (len(row) > 5 and row[5]) else "否"
                    formatted += f"{col_name} | {col_type} | {is_pk}\n"

                # 缓存表结构
                self._table_schema_cache[table_name] = formatted

                logger.info(f"[数据库] 表 {table_name} 有 {len(data)} 列")
                return formatted
            else:
                error = result.get("error", "查询失败")
                logger.error(f"[数据库] 查看表结构失败: {error}")
                return f"查看表结构失败: {error}"

        except Exception as e:
            logger.error(f"查看表结构失败: {e}")
            return f"查看表结构失败: {str(e)}"

    def get_sample_data(self, table_name: str) -> str:
        """
        查看表的前5行样例数据

        Args:
            table_name: 表名

        Returns:
            格式化的样例数据
        """
        try:
            if not self.sqlite_mcp:
                return "SQLite MCP 未初始化，无法查询数据库"

            logger.info(f"[数据库] 查看样例数据: {table_name}")

            sql = f"SELECT * FROM [{table_name}] LIMIT 5"
            result = self.sqlite_mcp.query(sql)

            if result.get("success"):
                data = result.get("data", [])
                columns = result.get("columns", [])

                if not data:
                    return f"表 '{table_name}' 没有数据"

                formatted = f"【表 {table_name} 的样例数据（前5行）】\n\n"

                if columns:
                    formatted += " | ".join(columns) + "\n"
                    formatted += "-" * (len(" | ".join(columns))) + "\n"

                for row in data:
                    formatted += " | ".join(str(cell) for cell in row) + "\n"

                formatted += f"\n共显示 {len(data)} 行样例数据"

                logger.info(f"[数据库] 表 {table_name} 返回 {len(data)} 行样例数据")
                return formatted
            else:
                error = result.get("error", "查询失败")
                logger.error(f"[数据库] 查看样例数据失败: {error}")
                return f"查看样例数据失败: {error}"

        except Exception as e:
            logger.error(f"查看样例数据失败: {e}")
            return f"查看样例数据失败: {str(e)}"
