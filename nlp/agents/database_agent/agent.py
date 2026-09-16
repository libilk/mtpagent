# -*- coding: utf-8 -*-
"""
Database Agent
==============

专门处理数据库查询任务，擅长SQL生成、表结构探索、数据分析。
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
        """通过 StreamWriter 发射中间步骤事件（供前端 trace 面板展示）"""
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
        """
        self.llm = llm
        self.tool_registry = tool_registry or ToolRegistry()
        self.enable_optimizations = enable_optimizations
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        self.cache_ttl = cache_ttl
        self.max_retries = max_retries

        # 初始化 SQLite MCP 服务
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
        """注册工具（4个）"""
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
        """截断过长的工具结果，保留头尾"""
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
        """构建系统提示词（增强表名选择逻辑）"""
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
用户: "有多少歌曲？"
→ 调用 `list_tables` → 看到 tracks 表
→ 调用 `describe_table("tracks")` → 确认有 TrackId 字段
→ 调用 `query_database("SELECT COUNT(*) AS total FROM tracks")` → 返回 3503
→ 最终回答: "数据库中共有 3503 首歌曲。"

【注意事项】
- 每次可调用 1-3 个工具，尽量并行调用以减少轮次。
- 必须拿到 `query_database` 的真实结果后，才能给出最终答案。
"""

    def _handle_react(self, query: str, context: Dict, start_time: float) -> str:
        """
        ReAct模式处理查询
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
        history_messages = context.get("history", [])

        messages = [
            {"role": "system", "content": system_prompt},
            *history_messages,
            {"role": "user", "content": query}
        ]

        max_iterations = 8
        retrieval_count = 0
        seen_tool_calls = set()

        for iteration in range(max_iterations):
            logger.info(f"[ReAct] 第 {iteration + 1}/{max_iterations} 轮")
            self._emit_progress(context, f"ReAct 第{iteration + 1}轮：正在思考...", stage="thinking")

            try:
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

                # 标准 tool calling 消息格式
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
        """
        try:
            if not self.sqlite_mcp:
                return "SQLite MCP 未初始化，无法查询数据库"

            logger.info(f"[数据库查询] SQL: {sql}")

            # 安全检查：只允许SELECT语句
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
