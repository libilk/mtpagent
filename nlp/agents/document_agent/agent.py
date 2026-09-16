# -*- coding: utf-8 -*-
"""
Document Agent
==============

统一文档处理Agent（ReAct模式）

支持多种文件格式的解析与智能分析：
- PDF（含OCR扫描件、跨页表格合并）
- Word（.docx/.doc，含表格和文本框提取）
- Excel/CSV（读取并转换为文本）
- 纯文本（.txt/.md）
- 合同审核（风险识别、历史对比、评分）
"""

import os
import re
import logging
import time
import json
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from llm.output_parser import parse_llm_json, parse_tool_calls
from llm.langchain_tools import ToolRegistry
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


# ==================== Pydantic 参数 Schema ====================

# ---------- 文档解析工具 ----------

class ParseDocumentArgs(BaseModel):
    """parse_document 工具参数"""
    file_path: str = Field(..., description="文档路径")

class ExtractStructureArgs(BaseModel):
    """extract_structure 工具参数"""
    document_text: str = Field(..., description="文档文本内容")

# ---------- 合同审核工具 ----------

class IdentifyRisksArgs(BaseModel):
    """identify_risks 工具参数"""
    contract_data: str = Field(..., description="结构化合同数据（JSON字符串）")

class CompareWithHistoryArgs(BaseModel):
    """compare_with_history 工具参数"""
    current_contract: str = Field(..., description="当前合同数据（JSON字符串）")
    supplier: str = Field(..., description="供应商名称")

class CalculateRiskScoreArgs(BaseModel):
    """calculate_risk_score 工具参数"""
    contract_data: str = Field(..., description="合同数据（JSON字符串）")
    risk_analysis: str = Field(..., description="风险分析结果")

class SearchSimilarContractsArgs(BaseModel):
    """search_similar_contracts 工具参数"""
    query: str = Field(..., description="检索查询（如：软件开发合同、XX供应商）")
    top_k: int = Field(default=3, description="返回结果数量")

class SearchBySupplierArgs(BaseModel):
    """search_by_supplier 工具参数"""
    supplier: str = Field(..., description="供应商名称")

class ListContractsArgs(BaseModel):
    """list_contracts 工具参数"""
    directory: str = Field(default="contracts", description="目录路径（相对于data目录）")

class ReadContractFileArgs(BaseModel):
    """read_contract_file 工具参数"""
    file_path: str = Field(..., description="文件路径（相对于data目录）")


class DocumentAgent:
    """文档处理Agent（ReAct自主决策模式）"""

    def _emit_progress(self, context: Dict, msg: str, **extra):
        """通过 StreamWriter 发射中间步骤事件（供前端 trace 面板展示）"""
        writer = context.get("_stream_writer")
        if writer is None:
            return
        agent_name = context.get("_agent_name", "document_agent")
        import time as _time
        event = {"event": "progress", "agent": agent_name, "msg": msg, "ts": _time.time()}
        event.update(extra)
        writer(event)

    def __init__(
        self,
        llm,
        retriever=None,
        tool_registry=None,
        embedder=None,
        enable_optimizations=True
    ):
        """
        初始化

        Args:
            llm: LLM实例
            retriever: 向量检索器（用于检索历史合同）
            function_calling: Function Calling管理器
            embedder: 嵌入器实例
            enable_optimizations: 是否启用优化模块（默认True）
        """
        self.llm = llm
        self.retriever = retriever
        self.tool_registry = tool_registry or ToolRegistry()
        self.embedder = embedder
        self.enable_optimizations = enable_optimizations

        # 初始化 Filesystem MCP 服务
        from core.filesystem_service import FilesystemService
        try:
            self.filesystem = FilesystemService()
            logger.info("✓ Filesystem MCP 初始化完成")
        except Exception as e:
            logger.warning(f"Filesystem MCP 初始化失败，将使用降级方案: {e}")
            self.filesystem = None

        # ========== 初始化优化模块 ==========
        if enable_optimizations:
            self._init_optimization_modules()
        else:
            self.cache_manager = None
            self.llm_cache = None
            self.retrieval_cache = None
            self.supplier_cache = None
            self.standards_calculator = None
            self.retry_strategy = None
            self.error_handler = None
            self.monitor = None

        # 注册工具
        self._register_tools()

        logger.info(f"DocumentAgent初始化完成 (优化模式: {enable_optimizations})")

    def _init_optimization_modules(self):
        """初始化优化模块"""
        try:
            # 1. 缓存管理
            from core.cache_manager import CacheManager, LLMCache, RetrievalCache, SupplierCache
            self.cache_manager = CacheManager(max_size=1000, ttl=3600)
            self.llm_cache = LLMCache(self.cache_manager)
            self.retrieval_cache = RetrievalCache(self.cache_manager)
            self.supplier_cache = SupplierCache(self.cache_manager)
            logger.info("✓ 缓存管理器初始化完成")

            # 2. 行业标准计算器
            from core.industry_standards import IndustryStandardsCalculator
            self.standards_calculator = IndustryStandardsCalculator(self.retriever)
            logger.info("✓ 行业标准计算器初始化完成")

            # 3. 错误处理
            from core.error_handler import RetryStrategy, ErrorHandler
            self.retry_strategy = RetryStrategy(max_retries=3, initial_delay=1.0)
            self.error_handler = ErrorHandler()
            logger.info("✓ 错误处理器初始化完成")

            # 4. 监控
            from core.monitoring import global_monitor
            self.monitor = global_monitor
            logger.info("✓ 监控系统初始化完成")

            logger.info("=" * 50)
            logger.info("🚀 优化模块全部加载完成")
            logger.info("=" * 50)

        except Exception as e:
            logger.error(f"优化模块初始化失败: {e}", exc_info=True)
            logger.warning("降级为基础模式运行")
            self.enable_optimizations = False

    def _register_tools(self):
        """注册所有工具（使用 StructuredTool + Pydantic Schema，支持多参数调用）"""

        # ========== 文档解析工具 ==========

        self.tool_registry.register_structured_tool(
            name="parse_document",
            func=self.parse_document,
            description="解析文档，自动识别类型（PDF/Word/TXT），提取文本内容。支持OCR扫描件识别和表格提取。适合第一步阅读合同",
            args_schema=ParseDocumentArgs,
        )

        self.tool_registry.register_structured_tool(
            name="extract_structure",
            func=self.extract_structure,
            description="从合同文本中提取结构化信息：甲乙方、金额、日期、付款方式、违约金、质保条款等。适合理解合同要素",
            args_schema=ExtractStructureArgs,
        )

        # ========== 智能分析工具 ==========

        self.tool_registry.register_structured_tool(
            name="identify_risks",
            func=self.identify_risks,
            description="基于LLM识别合同风险点：违约金过高、无限责任、单方解约权、缺失条款等。适合发现潜在问题",
            args_schema=IdentifyRisksArgs,
        )

        self.tool_registry.register_structured_tool(
            name="compare_with_history",
            func=self.compare_with_history,
            description="对比历史合同，分析条款变化、风险趋势。需要先知道供应商名称",
            args_schema=CompareWithHistoryArgs,
        )

        self.tool_registry.register_structured_tool(
            name="calculate_risk_score",
            func=self.calculate_risk_score,
            description="综合计算合同风险评分（0-100分），基于金额、风险点数量、供应商历史等维度",
            args_schema=CalculateRiskScoreArgs,
        )

        # ========== 知识检索工具 ==========

        self.tool_registry.register_structured_tool(
            name="search_similar_contracts",
            func=self.search_similar_contracts,
            description="在历史合同库中检索相似合同，用于参考和对比。可以按合同类型、供应商等检索",
            args_schema=SearchSimilarContractsArgs,
        )

        self.tool_registry.register_structured_tool(
            name="search_by_supplier",
            func=self.search_by_supplier,
            description="查询某供应商的历史合同记录，了解合作历史、违约记录、常用条款等",
            args_schema=SearchBySupplierArgs,
        )

        # ========== MCP 外部工具 ==========

        # Filesystem MCP 工具
        if self.filesystem:
            self.tool_registry.register_structured_tool(
                name="list_contracts",
                func=self.list_contracts,
                description="列出指定目录下的所有合同文件，用于批量审核",
                args_schema=ListContractsArgs,
            )

            self.tool_registry.register_structured_tool(
                name="read_contract_file",
                func=self.read_contract_file,
                description="通过MCP读取合同文件内容",
                args_schema=ReadContractFileArgs,
            )

    def handle(self, query: str, context: Dict) -> str:
        """
        处理查询（ReAct模式）

        Args:
            query: 用户查询
            context: 上下文信息（包含file_path等）

        Returns:
            处理结果
        """
        start_time = time.time()

        try:
            logger.info(f"[DocumentAgent] 开始处理查询: {query}")

            # 使用ReAct模式（自主决策调用工具）
            return self._handle_react(query, context, start_time)

        except Exception as e:
            logger.error(f"处理查询失败: {e}", exc_info=True)
            return f"处理失败: {str(e)}"

    @staticmethod
    def _truncate_tool_result(result_str: str, max_len: int = 4000) -> str:
        """截断过长的工具结果，保留头尾"""
        if len(result_str) <= max_len:
            return result_str
        keep = max_len // 2
        truncated = len(result_str) - max_len
        logger.info(f"[ReAct] 工具结果截断了 {truncated} 个字符")
        return result_str[:keep] + f"\n...[截断了 {truncated} 个字符]...\n" + result_str[-keep:]

    def _handle_react(self, query: str, context: Dict, start_time: float) -> str:
        """
        ReAct模式处理查询
        Agent自主决策调用哪些工具、调用顺序、何时停止

        Args:
            query: 用户查询
            context: 上下文信息
            start_time: 开始时间

        Returns:
            最终答案
        """
        logger.info("[ReAct] 开始ReAct循环")

        # 获取工具schema
        tools = self.tool_registry.get_tools_schema()

        if not tools:
            return "错误：无可用工具"

        # 构建系统提示词（定义Agent的能力和决策策略）
        system_prompt = """你是一个企业级文档处理Agent。你支持多种文档格式（PDF、Word、Excel/CSV、TXT/Markdown）的解析与智能分析，同时具备合同审核能力。

【核心能力】
1. 文档解析：支持PDF（含OCR扫描件）、Word、Excel/CSV、TXT/Markdown等多格式
2. 文档分析：提取结构化信息、生成摘要、回答用户关于文档内容的问题
3. 合同审核：风险识别、历史对比、风险评分

【文档解析工具】
- parse_document: 解析文档，自动识别格式并提取文本（第一步必须调用）
- extract_structure: 从文档中提取结构化信息（合同场景：甲乙方、金额、条款等）

【智能分析工具】
- identify_risks: 识别合同风险点（违约金、无限责任、缺失条款等）
- compare_with_history: 对比历史合同（需要先知道供应商名称）
- calculate_risk_score: 计算合同风险评分（0-100分）

【知识检索工具】
- search_similar_contracts: 检索相似合同
- search_by_supplier: 查询供应商历史

【MCP外部工具】
- list_contracts: 列出目录下的所有文件（批量处理）
- read_contract_file: 读取文件内容

【决策策略】
1. 通用文档分析（用户上传文档并提问）：
   parse_document → 根据用户问题直接回答

2. 合同审核任务：
   parse_document → extract_structure → identify_risks
   → search_by_supplier → compare_with_history
   → calculate_risk_score → 给出结论

3. 供应商查询任务：
   search_by_supplier → 分析历史

4. 文档对比任务：
   parse_document → extract_structure → search_similar_contracts → 对比分析

【重要原则】
- 根据用户意图灵活选择工具，不要硬编码流程
- 如果用户只是想了解文档内容，parse_document后直接回答即可，不必走完合同审核全流程
- 每次调用1-3个工具，避免过度调用
- 最多5轮迭代
- 合同审核场景：必须先parse_document才能extract_structure
- 给出明确的风险等级：低风险/中风险/高风险
"""

        # 初始化对话
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        max_iterations = 5  # 最多5轮ReAct循环
        seen_tool_calls = set()  # 重复调用检测

        for iteration in range(max_iterations):
            logger.info(f"[ReAct] 第 {iteration + 1}/{max_iterations} 轮")
            self._emit_progress(context, f"ReAct 第{iteration + 1}轮：正在思考...", stage="thinking")

            try:
                # 调用LLM（支持function calling）
                response = self.llm.chat(messages, tools=tools, temperature=0.3)

                # 解析工具调用
                tool_calls = parse_tool_calls(response)

                if not tool_calls:
                    # 没有工具调用，说明Agent认为可以直接回答了
                    logger.info("[ReAct] Agent决定给出最终答案")
                    self._emit_progress(context, f"ReAct 第{iteration + 1}轮：生成最终答案", stage="answering")
                    return response

                # 发射工具调用事件
                tool_names = [tc["function"]["name"] for tc in tool_calls]
                self._emit_progress(context, f"ReAct 第{iteration + 1}轮：调用工具 {', '.join(tool_names)}", stage="tool_call")

                # ---------- 去重 ----------
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
                    logger.info("[ReAct] 所有工具调用均为重复，强制收敛")
                    messages.append({"role": "user", "content": "请基于已有信息给出最终分析结论。"})
                    return self.llm.chat(messages, temperature=0.5)

                # ---------- 执行工具（并行/串行） ----------
                def _exec_one(tc):
                    fn = tc["function"]["name"]
                    try:
                        args = json.loads(tc["function"]["arguments"])
                    except json.JSONDecodeError:
                        args = parse_llm_json(tc["function"]["arguments"], fallback={})
                    logger.info(f"[ReAct] 调用工具: {fn}({args})")
                    t0 = time.time()
                    res = self.tool_registry.call_tool(fn, args)
                    logger.info(f"[ReAct] 工具执行完成，耗时 {(time.time()-t0)*1000:.1f}ms")
                    return tc, str(res) if not isinstance(res, str) else res

                results_map = {}  # tool_call_id -> result_str
                if len(unique_calls) > 1:
                    with ThreadPoolExecutor(max_workers=min(len(unique_calls), 4)) as pool:
                        futures = {pool.submit(_exec_one, tc): tc for tc in unique_calls}
                        for fut in as_completed(futures):
                            tc, res = fut.result()
                            results_map[tc["id"]] = self._truncate_tool_result(res)
                else:
                    tc, res = _exec_one(unique_calls[0])
                    results_map[tc["id"]] = self._truncate_tool_result(res)

                # ---------- 标准 tool calling 消息格式 ----------
                # 发射工具结果事件
                result_summary = ", ".join([tc["function"]["name"] for tc in unique_calls])
                self._emit_progress(context, f"工具执行完成: {result_summary}", stage="tool_result")

                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["function"]["name"],
                                "arguments": tc["function"]["arguments"]
                            }
                        }
                        for tc in unique_calls
                    ]
                })

                for tc in unique_calls:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": results_map[tc["id"]]
                    })

                # 仅最后一轮追加 user 消息强制收敛
                if iteration >= max_iterations - 2:
                    messages.append({
                        "role": "user",
                        "content": "这是最后一轮。请基于已有信息给出最终分析结论，不要再调用工具。"
                    })

            except Exception as e:
                logger.error(f"[ReAct] 第{iteration + 1}轮执行失败: {e}", exc_info=True)

                # 如果是第一轮就失败，返回错误
                if iteration == 0:
                    return f"处理失败: {str(e)}"

                # 否则尝试基于已有信息生成答案
                messages.append({
                    "role": "user",
                    "content": "工具执行出现问题，请基于已有信息给出答案。"
                })

                try:
                    final_answer = self.llm.chat(messages, temperature=0.5)
                    return final_answer
                except:
                    return "抱歉，处理您的请求时遇到了问题。"

        # 达到最大迭代次数，强制要求给出答案
        logger.warning(f"[ReAct] 达到最大迭代次数 {max_iterations}，强制生成答案")
        messages.append({
            "role": "user",
            "content": "请基于目前已有的所有信息给出最终分析结论。"
        })

        try:
            final_answer = self.llm.chat(messages, temperature=0.5)
            return final_answer
        except Exception as e:
            logger.error(f"[ReAct] 生成最终答案失败: {e}")
            return "抱歉，我无法完成文档分析。请检查文档格式或重试。"

    # ========== 工具实现 ==========

    @staticmethod
    def _resolve_file_path(file_path: str) -> Optional[str]:
        """
        智能解析文件路径：处理 LLM 只传文件名（缺少上传目录前缀）的情况。

        搜索顺序：
        1. 原始路径（绝对路径或相对路径直接可达）
        2. data/uploads/ 下精确匹配
        3. data/uploads/ 下模糊匹配（忽略 UUID 前缀，如 'f8f66baf_文件名.md'）

        Args:
            file_path: LLM 传入的文件路径（可能是纯文件名）

        Returns:
            实际存在的文件路径，找不到返回 None
        """
        # 1. 原始路径直接存在
        if os.path.exists(file_path):
            return file_path

        # 提取纯文件名（去掉可能的目录前缀）
        basename = os.path.basename(file_path)

        # 2. 在 data/uploads/ 下查找
        upload_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "uploads")
        if os.path.isdir(upload_dir):
            # 2a. 精确匹配
            exact_path = os.path.join(upload_dir, basename)
            if os.path.exists(exact_path):
                return exact_path

            # 2b. 模糊匹配：上传文件带 UUID 前缀（格式 '{8位UUID}_{原始文件名}'）
            for fname in os.listdir(upload_dir):
                # 跳过 UUID 前缀（8位hex + 下划线）后比较
                if "_" in fname:
                    original_name = fname.split("_", 1)[1]
                    if original_name == basename:
                        logger.info(f"[文件路径解析] 模糊匹配成功: {file_path} -> {fname}")
                        return os.path.join(upload_dir, fname)

        return None

    def parse_document(self, file_path: str) -> str:
        """
        解析文档（统一调用 core/file_parser.py）

        支持 PDF（含OCR扫描件、跨页表格合并）、Word（含表格和文本框）、TXT/MD。

        Args:
            file_path: 文档路径

        Returns:
            文档文本内容
        """
        try:
            logger.info(f"[解析文档] 文件: {file_path}")

            # 智能路径解析：处理 LLM 只传文件名、缺少 uploads 目录前缀的情况
            resolved_path = self._resolve_file_path(file_path)
            if resolved_path is None:
                return f"错误：文件不存在 {file_path}"
            if resolved_path != file_path:
                logger.info(f"[解析文档] 路径解析: {file_path} -> {resolved_path}")
            file_path = resolved_path

            # 统一使用 core/file_parser.py（支持OCR、表格合并、乱码检测等）
            from core.file_parser import parse_file
            result = parse_file(file_path, max_chars=8000)

            if not result["success"]:
                return f"解析失败: {result.get('error', '未知错误')}"

            text = result["text"]
            truncated = result.get("truncated", False)

            info = f"[解析文档] 成功，文本长度: {len(text)}"
            if truncated:
                info += "（已截断）"
            logger.info(info)

            return text

        except Exception as e:
            logger.error(f"解析文档失败: {e}")
            return f"解析失败: {str(e)}"

    def extract_structure(self, document_text: str) -> str:
        """
        提取结构化信息（基于LLM）

        Args:
            document_text: 文档文本

        Returns:
            结构化信息（JSON字符串）
        """
        try:
            logger.info("[提取结构] 开始提取结构化信息")

            # 截断过长的文本
            text_preview = document_text[:3000] if len(document_text) > 3000 else document_text

            prompt = f"""请从以下合同中提取结构化信息，输出JSON格式。

合同内容：
{text_preview}

请提取以下信息（输出JSON格式）：
{{
    "contract_number": "合同编号",
    "contract_type": "合同类型",
    "party_a": "甲方名称",
    "party_b": "乙方（供应商）名称",
    "amount": 合同金额（数字），
    "payment_terms": "付款方式描述",
    "delivery_time": "交付时间",
    "warranty_period": "质保期",
    "warranty_deposit": "质保金比例",
    "penalty_clause": "违约金条款",
    "key_terms": ["关键条款1", "关键条款2"]
}}

只输出JSON，不要其他内容："""

            response = self.llm.generate(prompt, temperature=0.1, max_tokens=800)

            # 提取JSON
            result = parse_llm_json(response, fallback={"error": "无法提取结构化信息"})
            logger.info("[提取结构] 成功")
            return json.dumps(result, ensure_ascii=False)

        except Exception as e:
            logger.error(f"提取结构失败: {e}")
            return f'{{"error": "{str(e)}"}}'

    def identify_risks(self, contract_data: str) -> str:
        """
        识别风险点（基于LLM + 行业标准）

        Args:
            contract_data: 结构化合同数据（JSON字符串）

        Returns:
            风险分析结果
        """
        try:
            logger.info("[风险识别] 开始识别风险")

            # ========== 优化1：计算行业标准 ==========
            standards_text = ""
            if self.enable_optimizations and self.standards_calculator:
                try:
                    # 尝试从合同数据中提取合同类型
                    contract_dict = json.loads(contract_data)
                    contract_type = contract_dict.get("contract_type")

                    # 计算行业标准
                    standards = self.standards_calculator.calculate_standards(contract_type)
                    standards_text = self.standards_calculator.format_standards_for_prompt(standards)
                    logger.info("[风险识别] 已加载行业标准")
                except Exception as e:
                    logger.warning(f"[风险识别] 行业标准计算失败: {e}")

            # 构建Prompt
            prompt = f"""请分析以下合同的风险点。

合同数据：
{contract_data}

{standards_text}

请从以下维度识别风险（输出JSON格式）：
{{
    "risks": [
        {{
            "type": "风险类型（如：违约金过高、无限责任、缺失条款等）",
            "description": "具体描述（对比行业标准）",
            "level": "风险等级（高/中/低）",
            "suggestion": "建议"
        }}
    ],
    "overall_assessment": "总体风险评估"
}}

常见风险类型：
1. 违约金过高（对比行业标准）
2. 预付款比例异常（对比行业标准）
3. 质保期过短（对比行业标准）
4. 无限责任条款
5. 单方解约权
6. 知识产权不明确
7. 缺少必备条款（质保、验收标准等）

只输出JSON："""

            # ========== 优化2：检查LLM缓存 ==========
            if self.enable_optimizations and self.llm_cache:
                cached_response = self.llm_cache.get(prompt, temperature=0.2, max_tokens=1000)
                if cached_response:
                    logger.info("[风险识别] 使用缓存结果")
                    return cached_response

            # ========== 优化3：调用LLM（带重试和监控） ==========
            def call_llm():
                start_time = time.time()
                response = self.llm.generate(prompt, temperature=0.2, max_tokens=1000)
                latency = (time.time() - start_time) * 1000

                # 记录性能
                if self.enable_optimizations and self.monitor:
                    self.monitor.performance.record("identify_risks_latency_ms", latency)
                    self.monitor.performance.increment("identify_risks_calls")

                return response

            # 使用重试策略
            if self.enable_optimizations and self.retry_strategy:
                response = self.retry_strategy.execute(call_llm)
            else:
                response = call_llm()

            # ========== 优化4：缓存结果 ==========
            if self.enable_optimizations and self.llm_cache:
                self.llm_cache.set(prompt, response, temperature=0.2, max_tokens=1000)

            # 提取JSON
            result = parse_llm_json(response, fallback={"risks": [], "overall_assessment": "无法识别风险"})
            logger.info("[风险识别] 成功")
            return json.dumps(result, ensure_ascii=False)

        except Exception as e:
            logger.error(f"风险识别失败: {e}")

            # 使用错误处理器
            if self.enable_optimizations and self.error_handler:
                return self.error_handler.handle(e, "identify_risks", '{"error": "识别失败"}')
            else:
                return f'{{"error": "{str(e)}"}}'

    def compare_with_history(self, current_contract: str, supplier: str) -> str:
        """
        对比历史合同

        Args:
            current_contract: 当前合同数据（JSON字符串）
            supplier: 供应商名称

        Returns:
            对比结果
        """
        try:
            logger.info(f"[历史对比] 供应商: {supplier}")

            # 检索该供应商的历史合同
            history_results = self.search_by_supplier(supplier)

            if "未找到" in history_results or "错误" in history_results:
                return f"无法对比：{history_results}"

            # 使用LLM对比
            prompt = f"""请对比当前合同与历史合同，分析变化和风险。

当前合同：
{current_contract}

历史合同：
{history_results}

请分析（输出JSON格式）：
{{
    "changes": [
        {{
            "aspect": "变化方面（如：付款条件、违约金、质保期等）",
            "before": "历史情况",
            "after": "当前情况",
            "impact": "影响评估（有利/不利/中性）"
        }}
    ],
    "risk_trend": "风险趋势分析",
    "recommendation": "建议"
}}

只输出JSON："""

            response = self.llm.generate(prompt, temperature=0.2, max_tokens=1000)

            # 提取JSON
            result = parse_llm_json(response, fallback={"changes": [], "risk_trend": "无法对比"})
            logger.info("[历史对比] 成功")
            return json.dumps(result, ensure_ascii=False)

        except Exception as e:
            logger.error(f"历史对比失败: {e}")
            return f'{{"error": "{str(e)}"}}'

    def calculate_risk_score(self, contract_data: str, risk_analysis: str) -> str:
        """
        计算风险评分

        Args:
            contract_data: 合同数据
            risk_analysis: 风险分析结果

        Returns:
            风险评分结果
        """
        try:
            logger.info("[风险评分] 开始计算")

            # 解析数据
            try:
                contract_dict = json.loads(contract_data)
                risk_dict = json.loads(risk_analysis)
            except:
                contract_dict = {}
                risk_dict = {}

            # 基于规则计算基础分数
            score = 0
            reasons = []

            # 1. 金额维度（0-30分）
            amount = contract_dict.get("amount", 0)
            if isinstance(amount, str):
                # 提取数字
                amount = int(re.sub(r'[^\d]', '', amount)) if re.search(r'\d', amount) else 0

            if amount > 1000000:
                score += 30
                reasons.append("合同金额超过100万")
            elif amount > 500000:
                score += 20
                reasons.append("合同金额超过50万")
            elif amount > 100000:
                score += 10
                reasons.append("合同金额超过10万")

            # 2. 风险点维度（0-50分）
            risks = risk_dict.get("risks", [])
            high_risks = [r for r in risks if r.get("level") == "高"]
            medium_risks = [r for r in risks if r.get("level") == "中"]

            risk_score = len(high_risks) * 15 + len(medium_risks) * 5
            risk_score = min(risk_score, 50)  # 最多50分
            score += risk_score

            if high_risks:
                reasons.append(f"存在{len(high_risks)}个高风险条款")
            if medium_risks:
                reasons.append(f"存在{len(medium_risks)}个中风险条款")

            # 3. 历史维度（0-20分）
            # 这里简化处理，实际应该从历史对比中获取
            if "违约" in str(risk_analysis):
                score += 20
                reasons.append("供应商有违约记录")

            # 确定风险等级
            if score >= 70:
                level = "高风险"
            elif score >= 40:
                level = "中风险"
            else:
                level = "低风险"

            result = {
                "risk_score": score,
                "risk_level": level,
                "reasons": reasons,
                "recommendation": self._get_recommendation(level, score)
            }

            logger.info(f"[风险评分] 完成，分数: {score}, 等级: {level}")
            return json.dumps(result, ensure_ascii=False)

        except Exception as e:
            logger.error(f"风险评分失败: {e}")
            return f'{{"error": "{str(e)}"}}'

    def _get_recommendation(self, level: str, score: int) -> str:
        """获取审批建议"""
        if level == "高风险":
            return "建议：需要CEO审批，法务总监重点审核，建议与供应商重新协商关键条款"
        elif level == "中风险":
            return "建议：需要部门经理和法务审核，关注重点风险条款"
        else:
            return "建议：部门经理审批即可，风险可控"

    def search_similar_contracts(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        检索相似合同

        Args:
            query: 检索查询
            top_k: 返回结果数量

        Returns:
            检索结果
        """
        if not self.retriever:
            return []

        try:
            logger.info(f"[检索相似合同] 查询: {query}")
            results = self.retriever.retrieve(query, top_k=top_k)
            logger.info(f"[检索相似合同] 找到 {len(results)} 个结果")
            return results
        except Exception as e:
            logger.error(f"检索失败: {e}")
            return []

    def search_by_supplier(self, supplier: str) -> str:
        """
        按供应商检索历史合同（带缓存和重试）

        Args:
            supplier: 供应商名称

        Returns:
            检索结果（文本格式）
        """
        if not self.retriever:
            return "错误：检索器未初始化"

        try:
            logger.info(f"[按供应商检索] 供应商: {supplier}")

            # ========== 优化1：检查缓存 ==========
            if self.enable_optimizations and self.supplier_cache:
                cached_result = self.supplier_cache.get(supplier)
                if cached_result:
                    logger.info("[按供应商检索] 使用缓存结果")
                    return cached_result

            # ========== 优化2：检索（带重试和监控） ==========
            def retrieve():
                start_time = time.time()
                results = self.retriever.retrieve(supplier, top_k=5)
                latency = (time.time() - start_time) * 1000

                # 记录性能
                if self.enable_optimizations and self.monitor:
                    self.monitor.performance.record("supplier_search_latency_ms", latency)
                    self.monitor.performance.increment("supplier_search_calls")

                return results

            # 使用重试策略
            if self.enable_optimizations and self.retry_strategy:
                results = self.retry_strategy.execute(retrieve)
            else:
                results = retrieve()

            if not results:
                return f"未找到供应商 {supplier} 的历史合同"

            # 格式化结果
            output = f"供应商 {supplier} 的历史合同：\n\n"
            for i, doc in enumerate(results, 1):
                content = doc.get("content", "")
                # 提取关键信息
                output += f"【合同{i}】\n{content[:500]}...\n\n"

            # ========== 优化3：缓存结果 ==========
            if self.enable_optimizations and self.supplier_cache:
                self.supplier_cache.set(supplier, output)

            logger.info(f"[按供应商检索] 找到 {len(results)} 份合同")
            return output

        except Exception as e:
            logger.error(f"按供应商检索失败: {e}")

            # 使用错误处理器
            if self.enable_optimizations and self.error_handler:
                return self.error_handler.handle(e, "search_by_supplier", f"检索失败: {str(e)}")
            else:
                return f"检索失败: {str(e)}"

    def list_contracts(self, directory: str = "contracts") -> str:
        """
        列出目录下的所有合同文件（通过 Filesystem MCP）

        Args:
            directory: 目录路径（相对于data目录）

        Returns:
            文件列表（JSON字符串）
        """
        try:
            if not self.filesystem:
                return json.dumps({"error": "Filesystem MCP 未初始化"}, ensure_ascii=False)

            logger.info(f"[MCP] 列出目录: {directory}")

            # 调用 Filesystem MCP
            files = self.filesystem.list_directory(directory)

            logger.info(f"[MCP] 找到 {len(files)} 个文件")
            return json.dumps({"files": files}, ensure_ascii=False)

        except Exception as e:
            logger.error(f"[MCP] 列出目录失败: {e}")
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    def read_contract_file(self, file_path: str) -> str:
        """
        读取合同文件内容（通过 Filesystem MCP）

        Args:
            file_path: 文件路径（相对于data目录）

        Returns:
            文件内容
        """
        try:
            if not self.filesystem:
                return "Filesystem MCP 未初始化"

            logger.info(f"[MCP] 读取文件: {file_path}")

            # 调用 Filesystem MCP
            content = self.filesystem.read_file(file_path)

            logger.info(f"[MCP] 读取成功，长度: {len(content)}")
            return content

        except Exception as e:
            logger.error(f"[MCP] 读取文件失败: {e}")
            return f"读取文件失败: {str(e)}"

    def get_optimization_stats(self) -> Dict:
        """
        获取优化统计信息

        Returns:
            统计信息字典
        """
        if not self.enable_optimizations:
            return {"optimization_enabled": False}

        try:
            stats = {
                "optimization_enabled": True,
                "cache_stats": self.cache_manager.get_stats() if self.cache_manager else {},
                "error_stats": self.error_handler.get_error_stats() if self.error_handler else {},
                "performance_stats": self.monitor.performance.get_all_stats() if self.monitor else {},
                "cost_stats": self.monitor.cost.get_stats() if self.monitor else {},
                "quality_stats": self.monitor.quality.get_stats() if self.monitor else {}
            }
            return stats
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {"error": str(e)}

    def print_optimization_stats(self):
        """打印优化统计信息"""
        if not self.enable_optimizations:
            print("\n优化模式未启用")
            return

        stats = self.get_optimization_stats()

        print("\n" + "=" * 60)
        print("Document Agent 优化统计")
        print("=" * 60)

        # 缓存统计
        if "cache_stats" in stats and stats["cache_stats"]:
            cache = stats["cache_stats"]
            print("\n【缓存统计】")
            print(f"  缓存大小: {cache.get('size', 0)}/{cache.get('max_size', 0)}")
            print(f"  命中次数: {cache.get('hits', 0)}")
            print(f"  未命中次数: {cache.get('misses', 0)}")
            print(f"  命中率: {cache.get('hit_rate', 0):.2%}")
            print(f"  淘汰次数: {cache.get('evictions', 0)}")

        # 性能统计
        if "performance_stats" in stats and stats["performance_stats"]:
            perf = stats["performance_stats"]
            print("\n【性能统计】")

            # 风险识别
            if "identify_risks_latency_ms" in perf:
                risk_stats = perf["identify_risks_latency_ms"]
                print(f"  风险识别延迟:")
                print(f"    均值: {risk_stats.get('mean', 0):.2f}ms")
                print(f"    P95: {risk_stats.get('p95', 0):.2f}ms")

            # 供应商检索
            if "supplier_search_latency_ms" in perf:
                search_stats = perf["supplier_search_latency_ms"]
                print(f"  供应商检索延迟:")
                print(f"    均值: {search_stats.get('mean', 0):.2f}ms")
                print(f"    P95: {search_stats.get('p95', 0):.2f}ms")

            # 计数器
            if "counters" in perf:
                print(f"\n  调用次数:")
                for counter_name, value in perf["counters"].items():
                    print(f"    {counter_name}: {value}")

        # 错误统计
        if "error_stats" in stats and stats["error_stats"]:
            errors = stats["error_stats"]
            print("\n【错误统计】")
            for error_key, count in errors.items():
                print(f"  {error_key}: {count}次")

        print("\n" + "=" * 60)

