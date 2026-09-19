# -*- coding: utf-8 -*-
"""
Knowledge Agent
===============

处理知识检索相关任务（已集成8大生产级优化）

在售后系统里扮演什么角色
------------------------
这是**通用知识型** Agent，不碰售后业务数据：它只回答"知识库里写了什么 /
网上说了什么"，不查订单、不办退款、不写库。因为它纯只读（不产生 write
operation 写操作），所以不经过人工审批那道闸门。
编排层里它通常作为 DAG 的下游节点被调用（例如"退货政策怎么规定的"），
与 customer_service / database / document 等业务 Agent 并列注册在同一个名册里。

⚠️ 事实核对（下面这句"8大"是旧口径，阅读时以代码为准）：
- `_init_optimization_modules` 实际编号到第 9 项（还夹了一个 3.5）。
- 其中数项只在 `_handle_with_optimizations`（ReAct 失败后的降级路径）里被调用，
  主路径 ReAct 走的是另一套自带的工具实现 —— 哪些真接线、哪些没接线，
  在对应初始化块和方法 docstring 上逐一标注了。
"""

import logging
import time
import json
import os
from typing import Dict, Any, List, Optional
from llm.output_parser import parse_llm_json, parse_tool_calls
# parse_tool_calls：从 LLM 的文本回复里解析出"工具调用（tool call）"，是 ReAct 循环的入口
# parse_llm_json：LLM 吐出的 JSON 常带 markdown 围栏等噪声，用它做容错解析
from llm.langchain_tools import ToolRegistry  # registry（注册中心）：Agent 名册，按名字取工具实例
from concurrent.futures import ThreadPoolExecutor, as_completed  # 多工具并行执行；as_completed = 谁先完成谁先取
from pydantic import BaseModel, Field  # Pydantic（数据校验库）：用类型标注声明数据结构，运行时自动校验


# ========== 工具参数 Schema（Pydantic 自动校验） ==========
# schema（结构契约）= 一份字段清单；这里每个类声明一个工具的入参长什么样。
# 用处：LLM 传参不符合契约时，Pydantic 在函数被调用前就拦下报错，
# 而不是让工具方法内部崩在半路 —— 错误信息还能回喂给 LLM 让它下一轮改对。

class QueryOnlyArgs(BaseModel):
    """仅需 query 参数的工具"""
    query: str = Field(description="查询文本")

class HybridSearchArgs(BaseModel):
    query: str = Field(description="查询文本")
    top_k: int = Field(default=3, description="返回结果数量")
    vector_weight: float = Field(default=0.8, description="向量检索权重（0-1）")

class SummarizeArgs(BaseModel):
    text: str = Field(description="待摘要的文本")
    max_length: int = Field(default=200, description="摘要最大长度")

class QueryExpansionArgs(BaseModel):
    query: str = Field(description="原始查询")
    num_variants: int = Field(default=3, description="生成变体数量")

class ExtractKeywordsArgs(BaseModel):
    query: str = Field(description="查询文本")
    max_keywords: int = Field(default=5, description="最大关键词数量")

class CheckCompletenessArgs(BaseModel):
    information: str = Field(description="已获取的信息")
    query: str = Field(description="原始查询")

class ExtractConceptsArgs(BaseModel):
    query: str = Field(description="用户查询")
    documents: str = Field(description="文档列表（JSON字符串）")
    extract_relations: bool = Field(default=True, description="是否提取概念间关系")

class MultiQuerySearchArgs(BaseModel):
    queries: List[str] = Field(description="查询列表")
    top_k: int = Field(default=3, description="每个查询返回的结果数")
    strategy: str = Field(default="parallel", description="检索策略（parallel/sequential）")

class FilteredSearchArgs(BaseModel):
    query: str = Field(description="查询文本")
    filters: dict = Field(default_factory=dict, description="过滤条件（如 audience, doc_type 等）")
    top_k: int = Field(default=5, description="返回结果数量")

class RerankResultsArgs(BaseModel):
    results: str = Field(description="检索结果（JSON字符串）")
    query: str = Field(description="查询文本")
    method: str = Field(default="llm", description="重排序方法（llm/cross_encoder）")

class MergeDocumentsArgs(BaseModel):
    documents: str = Field(description="文档列表（JSON字符串）")
    strategy: str = Field(default="summarize", description="合并策略（concat/summarize/extract）")

class CompareDocumentsArgs(BaseModel):
    documents: str = Field(description="文档列表（JSON字符串）")
    aspects: List[str] = Field(default_factory=list, description="对比的方面（如性能、功能、价格等）")

class WebSearchArgs(BaseModel):
    query: str = Field(description="搜索查询文本")
    max_results: int = Field(default=5, description="返回结果数量（1-10）")


logger = logging.getLogger(__name__)


class KnowledgeAgent:
    """
    知识检索Agent（生产级优化版）

    "生产级优化版"是自我描述，真实接线情况见模块头的事实核对。
    agent（智能体）= 能自己决定调哪个工具、调几次的 LLM 程序；
    本类对外只暴露 handle(query, context) 一个入口（handle = 处理入口），
    内部主循环是 ReAct（推理-行动循环）：想一步 → 调工具 → 看结果 → 再想，
    直到 LLM 不再要求调工具、直接给出自然语言答案为止。
    """

    def _emit_progress(self, context: Dict, msg: str, **extra):
        """
        通过 StreamWriter 发射中间步骤事件（供前端 trace 面板展示）

        StreamWriter（流式事件写入器）= 节点执行过程中往外推进度事件的接口，
        由调用方塞在 context["_stream_writer"] 里传进来；取不到就静默跳过 ——
        这样本 Agent 不依赖"有没有前端在听"，离线/脚本调用也能照常跑。
        trace = 前端的调用链面板，靠这些 progress 事件把每轮 ReAct 画出来。
        """
        writer = context.get("_stream_writer")
        if writer is None:
            return
        agent_name = context.get("_agent_name", "knowledge_agent")
        import time as _time
        event = {"event": "progress", "agent": agent_name, "msg": msg, "ts": _time.time()}
        event.update(extra)
        writer(event)

    def __init__(
        self,
        llm,
        retriever=None,        # retriever（检索器）：语义检索，负责从向量库召回文档
        bm25_retriever=None,   # BM25（关键词检索算法）：按词频+稀有度打分，擅长精确匹配
        tool_registry=None,
        document_filter=None,  # 按 audience（受众角色）过滤文档，避免越权看到不该看的
        embedder=None,         # embedder（向量化模型）：把文本转成向量，语义缓存靠它算相似度

        # 功能开关（全部默认启用）
        enable_optimizations: bool = True,
        enable_cache: bool = True,

        # 高级功能开关（全部默认启用）
        enable_rerank: bool = True,
        enable_hyde: bool = True,
        enable_self_consistency: bool = True,

        # 可调参数
        max_history: int = 10,
        cache_size: int = 1000,
        cache_ttl: int = 3600,
        max_retries: int = 3,
        vector_weight: float = 0.8,
        max_context_length: int = 2000,

        # Redis（内存数据库）支持：传了它缓存可跨进程共享；不传则退化为进程内内存缓存
        redis_client=None,
    ):
        """
        初始化

        Args:
            llm: LLM实例
            retriever: 语义检索器（来自rag_core.retriever）
            bm25_retriever: BM25检索器（来自rag_core.bm25_retriever）
            tool_registry: 工具注册器（LangChain Tool）
            document_filter: 文档过滤器
            embedder: 嵌入器实例

            enable_optimizations: 是否启用生产级优化（默认True）
            enable_memory: 是否启用对话记忆（默认True）

            enable_rerank: 是否启用重排序（默认True）
            enable_hyde: 是否启用HyDE（默认True）
            enable_self_consistency: 是否启用Self-Consistency（默认True）

            max_history: 对话历史最大轮数（默认10）
            cache_size: 缓存大小（默认1000）
            cache_ttl: 缓存TTL秒数（默认3600）
            max_retries: 最大重试次数（默认3）
            vector_weight: 向量检索权重（默认0.5）
            max_context_length: 上下文最大长度（默认2000）

        Note:
            上面 Args 与真实签名有两处对不上（只记录，未改代码）：
            1) 并不存在 "enable_memory" 参数 —— 对话历史是编排层经
               context["history"] 注入的，本类没有自己的记忆开关；
            2) vector_weight 真实默认值是 0.8，不是这里写的 0.5。
        """
        self.llm = llm
        self.retriever = retriever
        self.bm25_retriever = bm25_retriever
        self.tool_registry = tool_registry or ToolRegistry()
        self.document_filter = document_filter
        self.embedder = embedder
        self.enable_optimizations = enable_optimizations
        self.enable_cache = enable_cache

        # 保存配置参数
        self.enable_rerank = enable_rerank
        self.enable_hyde = enable_hyde
        self.enable_self_consistency = enable_self_consistency
        self.cache_size = cache_size
        self.cache_ttl = cache_ttl
        self.max_retries = max_retries
        self.vector_weight = vector_weight
        self.max_context_length = max_context_length
        self.redis_client = redis_client

        # 初始化优化模块
        if enable_optimizations:
            self._init_optimization_modules()
        else:
            self.hybrid_retriever = None
            self.query_optimizer = None
            self.cached_retriever = None
            self.compressor = None
            self.answer_generator = None
            self.retry_strategy = None
            self.evaluator = None

        # 注册工具
        self._register_tools()

        logger.info(f"KnowledgeAgent初始化完成 (优化: {enable_optimizations})")

    def _init_optimization_modules(self):
        """
        初始化"8大优化模块"（实际编号到第 9 项，见下方各块）

        接线现状速查 —— 这才是读这个文件最容易踩空的地方：
        - 主路径 ReAct 真会用到：缓存/语义缓存（handle 开头判断）、监控 metrics_collector
        - 只在降级路径 _handle_with_optimizations 用到：查询优化、HybridRetriever、
          上下文压缩、答案生成、重试策略、评估器、告警
        - 只在被 LLM 当工具调用时才用到：概念提取器
        - 初始化了但全文件无调用方：fallback_strategy
        这些模块来自 rag_core / core 包，也就是 RAG（检索增强生成：
        先检索资料、再让 LLM 基于资料回答）流水线的各个组件。
        """
        try:
            # 1. 混合检索 + 重排序
            # hybrid retrieval（混合检索）= 向量检索 + 关键词检索一起用；
            # reranker（重排序器）= 对召回结果做精排，把最相关的顶上来。
            # ⚠️ self.hybrid_retriever 只被下面的 cached_retriever 包着，
            #    而 cached_retriever 只在降级路径 _handle_with_optimizations 里被调；
            #    ReAct 主路径调用的 hybrid_search 工具是自己内联实现 RRF 的（见该函数）。
            from rag_core.hybrid_retriever import HybridRetriever
            from rag_core.reranker import RerankerFactory
            if self.enable_rerank:
                reranker = RerankerFactory.create_reranker("auto")
            else:
                reranker = None
            self.hybrid_retriever = HybridRetriever(
                vector_retriever=self.retriever,
                bm25_retriever=self.bm25_retriever,
                vector_weight=self.vector_weight,  # 使用参数
                use_rerank=self.enable_rerank,  # 使用参数
                reranker=reranker
            )
            logger.info("✓ 混合检索器初始化完成")

            # 2. 查询优化
            # HyDE（假设文档嵌入）= 先让 LLM 编一个"假答案"，再拿它去检索。
            # 适合"问题太短、直接检索命中率低"的场景，代价是多一次 LLM 调用。
            from rag_core.query_optimizer import QueryOptimizer
            self.query_optimizer = QueryOptimizer(
                llm=self.llm,
                enable_hyde=self.enable_hyde  # 使用参数
            )
            logger.info("✓ 查询优化器初始化完成")

            # 3. 缓存机制
            # cache（缓存）= 把算过的结果存起来，下次直接用。
            # 这里一次挂三层：答案层（CachedRetriever）、向量层（CachedEmbedder）、
            # 加上下面 3.5 的语义层 —— 分层是为了"问题换个说法"也少算几次。
            from rag_core.cache_manager import CacheManager, CachedRetriever, CachedEmbedder
            self.cache_manager = CacheManager(
                max_size=self.cache_size,  # 使用参数
                ttl=self.cache_ttl,  # 使用参数
                redis_client=self.redis_client,  # 传入 Redis 连接
            )
            self.cached_retriever = CachedRetriever(self.hybrid_retriever, self.cache_manager)

            # 向量层缓存：替换 retriever 内部的 embedder 为 cached_embedder
            # 注意这是"改别人的对象"：直接把传进来的 retriever 实例的 embedder 换掉了。
            # 属于副作用式接线 —— 外部若还持有同一个 retriever，也会一起被换掉。
            if self.embedder and self.retriever:
                self.cached_embedder = CachedEmbedder(self.embedder, self.cache_manager)
                self.retriever.embedder = self.cached_embedder
                logger.info("✓ 向量缓存已接入 retriever")

            logger.info("✓ 缓存管理器初始化完成")

            # 3.5 语义缓存（业务层第二道防线）
            # semantic cache（语义缓存）= 问法不同但意思一样也能命中；
            # 靠 cosine similarity（余弦相似度）比向量，阈值 0.95 卡得很严，
            # 就是怕把"意思相近但答案不同"的问题误判成同一个。
            if self.embedder:
                from core.semantic_cache import SemanticCache
                self.semantic_cache = SemanticCache(
                    embedder=self.embedder,
                    similarity_threshold=0.95,
                    max_size=self.cache_size,
                    ttl=self.cache_ttl,
                    redis_client=self.redis_client,  # 传入 Redis 连接
                )
                logger.info("✓ 语义缓存初始化完成")
            else:
                self.semantic_cache = None

            # 4. 上下文压缩
            # context compression（上下文压缩）= 把检索到的长文档压短，省 token
            # （词元：LLM 处理文本的最小单位，也是计费和长度单位）；不压会撑爆上下文窗口。
            from rag_core.context_compressor import ContextCompressor
            self.compressor = ContextCompressor(
                llm=self.llm,
                max_length=self.max_context_length  # 使用参数
            )
            logger.info("✓ 上下文压缩器初始化完成")

            # 5. 答案生成优化
            # Self-Consistency（自洽性）= 同一问题让模型生成多次，取多数一致的那个答案。
            # num_samples=3 就是生成 3 次做投票：换稳定性，代价是成本也翻三倍。
            from rag_core.answer_generator import AnswerGenerator
            self.answer_generator = AnswerGenerator(
                llm=self.llm,
                enable_self_consistency=self.enable_self_consistency,  # 使用参数
                num_samples=3
            )
            logger.info("✓ 答案生成器初始化完成")

            # 6. 错误处理
            # retry（重试）= 失败就回头再跑；fallback（降级兜底）= 主路径不行就退到备用方案。
            # ⚠️ fallback_strategy 在这里被创建，但全文件再无任何调用方 —— "定义了没接线"。
            from core.error_handler import RetryStrategy, FallbackStrategy, GracefulErrorHandler
            self.retry_strategy = RetryStrategy(max_retries=self.max_retries)  # 使用参数
            self.fallback_strategy = FallbackStrategy()
            self.error_handler = GracefulErrorHandler()
            logger.info("✓ 错误处理器初始化完成")

            # 7. 评估体系
            # ⚠️ 别和编排层的 evaluator（评估器：给最终答案做质量把关、不过关就打回重试）搞混：
            #    这里的 RAGEvaluator 只统计延迟等运行指标（evaluate_latency），不打质量分。
            from rag_core.evaluator import RAGEvaluator, OnlineMetrics
            self.evaluator = RAGEvaluator()
            self.online_metrics = OnlineMetrics()
            logger.info("✓ 评估器初始化完成")

            # 8. 监控告警
            # 这三个是模块级单例：metrics_collector 在 ReAct 主路径里被大量调用（真接线）；
            # alert_manager 只在降级路径 _handle_with_optimizations 末尾 check_alerts 一次。
            from rag_core.monitoring import structured_logger, metrics_collector, alert_manager
            self.structured_logger = structured_logger
            self.metrics_collector = metrics_collector
            self.alert_manager = alert_manager
            logger.info("✓ 监控系统初始化完成")

            # 9. 跨文档概念提取器
            # 只在 LLM 主动调用 extract_concepts 工具时才走到，不参与 handle 的主流程。
            from rag_core.concept_extractor import ConceptExtractor
            self.concept_extractor = ConceptExtractor(
                llm=self.llm,
                embedder=self.embedder,
                similarity_threshold=0.8,
                min_concept_freq=1,
                max_concepts_per_doc=5
            )
            logger.info("✓ 概念提取器初始化完成")

            logger.info("=" * 50)
            logger.info("🚀 生产级优化模块全部加载完成（含概念提取）")
            logger.info("=" * 50)

        except Exception as e:
            # 任一优化模块导入/初始化失败 → 整体关掉开关，降级为基础模式。
            # 注意：关掉后不少属性（cache_manager / metrics_collector / semantic_cache …）
            # 干脆不会被赋值；之所以没炸，是因为调用点都先判断了 enable_optimizations。
            logger.error(f"优化模块初始化失败: {e}", exc_info=True)
            logger.warning("降级为基础模式运行")
            self.enable_optimizations = False

    def _register_tools(self):
        """
        注册工具（基于 StructuredTool + Pydantic 自动参数校验）

        StructuredTool（LangChain 工具类）= 用 args_schema（参数结构）声明入参，
        调用前自动做参数校验和默认值填充 —— 所以方法签名上的默认值不一定生效，
        真正生效的是 args_schema 里的 default（两者不一致时以 Schema 为准）。
        registry 里注册的名字，就是 LLM 在 tool_calls 中要写的函数名，必须与之一致。
        """
        # 向量检索
        self.tool_registry.register_structured_tool(
            name="vector_search",
            func=self.vector_search,
            description="在知识库中进行语义相似度检索",
            args_schema=QueryOnlyArgs
        )

        # 关键词检索
        self.tool_registry.register_structured_tool(
            name="keyword_search",
            func=self.keyword_search,
            description="基于BM25算法的关键词精确匹配检索",
            args_schema=QueryOnlyArgs
        )

        # 混合检索
        self.tool_registry.register_structured_tool(
            name="hybrid_search",
            func=self.hybrid_search,
            description="结合向量检索和关键词检索，使用RRF算法融合结果",
            args_schema=HybridSearchArgs
        )

        # 文本摘要
        self.tool_registry.register_structured_tool(
            name="summarize",
            func=self.summarize,
            description="对长文本进行摘要提取",
            args_schema=SummarizeArgs
        )

        # ========== 扩展工具（优先级1：基于LLM） ==========
        # 下面这些工具多数要额外调一次 LLM（展开/分解/提取），是"花钱换命中率"的手段。
        # 工具描述（description）会原样进提示词，是 LLM 挑工具的主要依据。

        # 查询扩展
        self.tool_registry.register_structured_tool(
            name="query_expansion",
            func=self.query_expansion,
            description="生成查询的多个变体，提升召回率。适合查询词较少或需要多角度检索的场景",
            args_schema=QueryExpansionArgs
        )

        # 查询分解
        self.tool_registry.register_structured_tool(
            name="query_decomposition",
            func=self.query_decomposition,
            description="将复杂查询分解为多个子查询。适合对比、分析、多步骤推理等复杂问题",
            args_schema=QueryOnlyArgs
        )

        # 关键词提取
        self.tool_registry.register_structured_tool(
            name="extract_keywords",
            func=self.extract_keywords,
            description="从查询中提取核心关键词。适合长查询或需要精确匹配的场景",
            args_schema=ExtractKeywordsArgs
        )

        # 跨文档概念提取
        self.tool_registry.register_structured_tool(
            name="extract_concepts",
            func=self.extract_concepts,
            description="从多个检索到的文档中提取跨文档语义概念，构建概念关系图。适合需要理解文档间共同主题、对比分析、知识整合的场景",
            args_schema=ExtractConceptsArgs
        )

        # ========== 扩展工具（优先级2：需要适度开发） ==========

        # 多查询检索
        self.tool_registry.register_structured_tool(
            name="multi_query_search",
            func=self.multi_query_search,
            description="并行检索多个查询并融合结果。适合需要多角度信息的复杂问题",
            args_schema=MultiQuerySearchArgs
        )

        # 过滤检索
        self.tool_registry.register_structured_tool(
            name="filtered_search",
            func=self.filtered_search,
            description="带元数据过滤的检索。可以根据文档类型、时间、作者等过滤",
            args_schema=FilteredSearchArgs
        )

        # 结果重排序
        self.tool_registry.register_structured_tool(
            name="rerank_results",
            func=self.rerank_results,
            description="对检索结果重新排序，提升相关性。适合初次检索结果质量不高的情况",
            args_schema=RerankResultsArgs
        )

        # 文档合并
        self.tool_registry.register_structured_tool(
            name="merge_documents",
            func=self.merge_documents,
            description="合并多个文档的信息，去重并整合。适合从多个来源获取信息后的整合",
            args_schema=MergeDocumentsArgs
        )

        # 文档对比
        self.tool_registry.register_structured_tool(
            name="compare_documents",
            func=self.compare_documents,
            description="对比两个或多个文档的异同。适合对比类问题",
            args_schema=CompareDocumentsArgs
        )

        # 网络搜索
        self.tool_registry.register_structured_tool(
            name="web_search",
            func=self.web_search,
            description="搜索互联网获取最新信息。当知识库中没有相关内容、需要实时数据或最新资讯时使用",
            args_schema=WebSearchArgs
        )

    def _collect_sources(self, results: List[Dict]):
        """
        收集检索结果中的源文档信息（按文件去重，保留最高分）

        metadata（元数据）= 附在文档/向量上的结构化信息，这里靠它过滤和分层；
        score（分数）= 相关性得分；doc_id（文档 ID）= 每篇文档的唯一标识。
        同一 source（来源文件）常被切成多个 chunk（文本块），命中时会有好几条，
        所以按 source 去重、只留最高分。

        同时提取 metadata 中的检索统计信息：
        - _vector_score / _bm25_score: 原始检索分数
        - _vector_total / _vector_filtered: 向量检索总数与过滤数
        - _bm25_total / _bm25_filtered: BM25 检索总数与过滤数
        - _rrf_score: RRF 融合分数（由 hybrid_search 注入）

        Note:
            当 _suppress_source_collection 标志为 True 时（hybrid_search 内部调用
            vector_search/keyword_search 期间），跳过收集，避免不同评分尺度的分数
            混入同一列表。
        """
        if getattr(self, '_suppress_source_collection', False):
            return

        if not hasattr(self, '_retrieved_sources'):
            self._retrieved_sources = []

        # 收集全局检索统计（只取第一个文档的即可，因为同一批结果统计相同）
        if not hasattr(self, '_retrieval_stats'):
            self._retrieval_stats = {}
        if results:
            first_meta = results[0].get("metadata", {})
            if "_vector_total" in first_meta:
                self._retrieval_stats["vector_corpus_size"] = first_meta.get("_vector_corpus_size", first_meta["_vector_total"])
                self._retrieval_stats["vector_total"] = first_meta["_vector_total"]
                self._retrieval_stats["vector_filtered"] = first_meta["_vector_filtered"]
            if "_bm25_matched" in first_meta:
                self._retrieval_stats["bm25_corpus_size"] = first_meta.get("_bm25_corpus_size", 0)
                self._retrieval_stats["bm25_matched"] = first_meta["_bm25_matched"]
                self._retrieval_stats["bm25_filtered"] = first_meta["_bm25_filtered"]

        source_index = {s.get("source", ""): i for i, s in enumerate(self._retrieved_sources)}

        for doc in results:
            meta = doc.get("metadata", {})
            source = meta.get("source", "")
            if not source:
                continue

            new_score = round(doc.get("score", 0.0), 4)

            # 提取各检索阶段的原始分数
            vector_score = meta.get("_vector_score")
            bm25_score = meta.get("_bm25_score")
            rrf_score = meta.get("_rrf_score")

            if source in source_index:
                idx = source_index[source]
                if new_score > self._retrieved_sources[idx].get("score", 0):
                    self._retrieved_sources[idx]["score"] = new_score
                    self._retrieved_sources[idx]["snippet"] = doc.get("content", "")[:150]
                # 合并分数（取各自最高值）
                if vector_score is not None:
                    old_vs = self._retrieved_sources[idx].get("vector_score")
                    self._retrieved_sources[idx]["vector_score"] = max(vector_score, old_vs) if old_vs is not None else vector_score
                if bm25_score is not None:
                    old_bs = self._retrieved_sources[idx].get("bm25_score")
                    self._retrieved_sources[idx]["bm25_score"] = max(bm25_score, old_bs) if old_bs is not None else bm25_score
                if rrf_score is not None:
                    old_rs = self._retrieved_sources[idx].get("rrf_score")
                    self._retrieved_sources[idx]["rrf_score"] = max(rrf_score, old_rs) if old_rs is not None else rrf_score
            else:
                source_index[source] = len(self._retrieved_sources)
                entry = {
                    "source": source,
                    "title": meta.get("title", ""),
                    "doc_id": meta.get("doc_id", ""),
                    "score": new_score,
                    "snippet": doc.get("content", "")[:150],
                }
                if vector_score is not None:
                    entry["vector_score"] = vector_score
                if bm25_score is not None:
                    entry["bm25_score"] = bm25_score
                if rrf_score is not None:
                    entry["rrf_score"] = rrf_score
                self._retrieved_sources.append(entry)

    def _append_sources(self, response: str) -> str:
        """
        在回答末尾附加引用源文档信息（使用特殊标记，前端解析展示）

        过滤策略：
        - 按分数降序排列
        - 只保留分数 > 0 的文档
        - 过滤掉与最高分差距过大的文档（低于最高分 25% 的不展示）
        - 最多展示5个引用源

        同时附加检索统计信息（总候选数、过滤数）供前端展示。

        Args:
            response: LLM生成的回答文本

        Returns:
            附加了引用信息和检索统计的回答

        隐式契约：返回值末尾用 `<!-- SOURCES:{json}:SOURCES -->` 包一段 JSON，
        由前端反向解析出 citation（引用溯源：标出"这句来自哪篇文档"）。
        后端与前端就靠这个字符串标记耦合 —— 改格式等于改接口，所以在此单独说明。
        """
        sources = getattr(self, '_retrieved_sources', [])
        if not sources:
            return response

        candidates = sorted(
            [s for s in sources if s.get("score", 0) > 0],
            key=lambda x: x.get("score", 0),
            reverse=True
        )

        if not candidates:
            return response

        candidates = [s for s in candidates if s.get("score", 0) >= 0.3]

        if not candidates:
            return response

        max_score = candidates[0].get("score", 0)
        if max_score > 0:
            threshold = max_score * 0.25
            candidates = [s for s in candidates if s.get("score", 0) >= threshold]

        filtered = candidates[:5]

        if not filtered:
            return response

        # 构建最终数据：sources + retrieval_stats
        import json as _json
        output_data = {
            "sources": filtered,
            "retrieval_stats": getattr(self, '_retrieval_stats', {})
        }
        data_json = _json.dumps(output_data, ensure_ascii=False)
        return f"{response}\n\n<!-- SOURCES:{data_json}:SOURCES -->"

    @staticmethod
    def _truncate_tool_result(result_str: str, max_len: int = 4000) -> str:
        """
        截断过长的工具结果，保留头尾

        为什么留头尾而不是只留头：检索结果的"开头是正文摘要、结尾是来源/统计"，
        砍掉任一头都会丢信息，所以从中间挖掉一段。
        """
        if len(result_str) <= max_len:
            return result_str
        keep = max_len // 2
        truncated = len(result_str) - max_len
        logger.info(f"[ReAct] 工具结果截断了 {truncated} 个字符")
        return result_str[:keep] + f"\n...[截断了 {truncated} 个字符]...\n" + result_str[-keep:]

    def _handle_tool_not_found(self, tool_name: str, tool_call_id: str) -> Dict:
        """
        处理工具不存在的情况（含拼写纠错）

        用 difflib 做模糊匹配给出候选名 —— LLM 偶尔会凭记忆把工具名拼错，
        直接回"不存在"它下轮多半照错，给候选名更容易自我纠正。

        Args:
            tool_name: 不存在的工具名称
            tool_call_id: 工具调用ID

        Returns:
            错误信息字典
        """
        from difflib import get_close_matches

        available_tools = self.tool_registry.get_all_tool_names()
        suggestions = get_close_matches(tool_name, available_tools, n=3, cutoff=0.6)

        if suggestions:
            error_msg = f"工具'{tool_name}'不存在。你可能想使用：{', '.join(suggestions)}"
        else:
            error_msg = f"工具'{tool_name}'不存在。可用工具：{', '.join(available_tools[:5])}"

        logger.warning(f"[工具不存在] {error_msg}")

        return {
            "tool_call_id": tool_call_id,
            "tool": tool_name,
            "arguments": {},
            "result": error_msg,
            "success": False,
            "error_type": "tool_not_found"
        }

    def _execute_tool_calls(self, tool_calls: List[Dict], retrieval_count: int,
                            seen_tool_calls: set = None) -> List[Dict]:
        """
        执行工具调用（支持去重 + 并行 + JSON容错）

        三件事为什么凑在一起 —— 都在 ReAct 循环里，省一次往返就省一次 LLM 调用：
        - 去重：LLM 常在同一轮里重复请求同一个调用，重复执行纯属浪费
        - 并行：同一轮多个互不依赖的工具丢线程池一起跑
        - JSON 容错：arguments 是 LLM 现生成的字符串，格式不合法时用宽松解析兜底

        Args:
            tool_calls: 工具调用列表
            retrieval_count: 当前检索次数（用于日志）
            seen_tool_calls: 跨轮去重集合

        Returns:
            工具执行结果列表
        """
        if seen_tool_calls is None:
            seen_tool_calls = set()

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
            return []

        # ---------- 单个工具执行逻辑 ----------
        def _exec_one(tool_call):
            fn = tool_call["function"]["name"]
            tool_call_id = tool_call.get("id", fn)

            try:
                arguments = json.loads(tool_call["function"]["arguments"])
            except json.JSONDecodeError:
                arguments = parse_llm_json(tool_call["function"]["arguments"], fallback={})

            logger.info(f"[ReAct工具调用] 工具名称: {fn}")
            logger.info(f"[ReAct工具参数] {json.dumps(arguments, ensure_ascii=False)}")

            # 检查工具是否存在
            if not self.tool_registry.has_tool(fn):
                return self._handle_tool_not_found(fn, tool_call_id)

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
                    "tool_call_id": tool_call_id,
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
                    "tool_call_id": tool_call_id,
                    "tool": fn,
                    "arguments": arguments,
                    "result": f"工具执行失败: {str(e)}",
                    "success": False,
                    "error": str(e)
                }

        # ---------- 并行 / 串行执行 ----------
        tool_results = []
        if len(unique_calls) > 1:
            with ThreadPoolExecutor(max_workers=min(len(unique_calls), 4)) as pool:
                futures = {pool.submit(_exec_one, tc): tc for tc in unique_calls}
                for fut in as_completed(futures):
                    tool_results.append(fut.result())
        else:
            tool_results.append(_exec_one(unique_calls[0]))

        return tool_results


    def handle(self, query: str, context: Dict) -> str:
        """
        处理查询（ReAct模式 + 生产级优化 + 对话记忆 + 情感分析）

        说明：标题里的"情感分析"在本类里没有实现（旧描述未同步，情感分析在
        customer_service_agent 那边）。本方法实际只做三件事：查缓存 → 跑 ReAct → 回写缓存。

        两道缓存都摆在最前面：命中就直接 return，连 ReAct 循环都不进 ——
        所以调优时"为什么没看到工具调用"的常见答案就是缓存命中了。

        Args:
            query: 用户查询
            context: 上下文信息

        Returns:
            处理结果
        """
        start_time = time.time()
        user_id = context.get("user_id", "anonymous")

        try:
            # 业务层缓存：第一道防线 - 精确匹配（demo模式下跳过，确保面试官看到完整流程）
            if self.enable_cache and self.enable_optimizations and self.cache_manager:
                cached_answer = self.cache_manager.query_cache.get(query)
                if cached_answer:
                    logger.info(f"[业务精确缓存命中] 直接返回缓存答案")
                    return cached_answer

            # 业务层缓存：第二道防线 - 语义相似匹配
            if self.enable_cache and self.enable_optimizations and self.semantic_cache:
                cached_answer = self.semantic_cache.get(query)
                if cached_answer:
                    logger.info(f"[业务语义缓存命中] 直接返回缓存答案")
                    return cached_answer

            # 记录查询
            if self.enable_optimizations:
                self.structured_logger.log_query(query=query, user_id=user_id)
                self.metrics_collector.increment("total_queries")
                self.online_metrics.record_query()

            logger.info(f"[KnowledgeAgent] 开始ReAct模式处理查询")

            # 使用ReAct模式（自主决策调用工具）
            result = self._handle_react(query, context, start_time, user_id)

            # 业务层缓存：同时写入精确缓存 + 语义缓存
            if self.enable_optimizations and self.cache_manager:
                query_cost_ms = (time.time() - start_time) * 1000
                self.cache_manager.query_cache.set_with_cost(query, result, query_cost_ms)
                logger.debug(f"[业务精确缓存] 答案已缓存（耗时={query_cost_ms:.0f}ms）")

            if self.enable_optimizations and self.semantic_cache:
                self.semantic_cache.set(query, result)
                logger.debug(f"[业务语义缓存] 答案已缓存")

            return result

        except Exception as e:
            logger.error(f"处理查询失败: {e}", exc_info=True)

            if self.enable_optimizations:
                self.structured_logger.log_error("query_processing_error", str(e))
                self.metrics_collector.increment("errors")
                return self.error_handler.handle(e, "unknown")
            else:
                return f"处理失败: {str(e)}"

    def _build_system_prompt(self, context: Dict) -> str:
        """
        构建系统提示词（知识检索专用）

        prompt（提示词）= 喂给 LLM 的指令文本。这份提示词就是本 Agent 的"行为策略书"：
        强制先检索后回答、规定每个工具的适用场景、约定检索空结果时必须转 web_search。
        它是一套通用的知识检索策略，不含任何售后业务规则 —— 业务口径不在这里。
        整段提示词是本文件最长的字符串字面量，改它等于改 Agent 行为，要当代码对待。

        Args:
            context: 上下文信息

        Returns:
            系统提示词
        """
        prompt = """你是一个专业的智能助手，具备知识检索和网络搜索能力。你需要通过调用工具来收集信息，然后回答用户问题。

【核心原则】
⚠️ 你必须先调用检索工具获取相关文档，再基于检索结果回答问题。
⚠️ 禁止不调用任何工具就直接回答——即使你认为自己已经知道答案。
⚠️ 第一轮必须至少调用一个检索工具（hybrid_search、vector_search 或 keyword_search）。

工作流程：
1. 分析用户问题，判断查询类型和复杂度
2. 调用检索工具获取知识库中的相关文档（必做）
3. 评估结果质量，必要时调整策略
4. 当信息充足时，基于检索到的文档给出最终答案

【基础检索工具】
- vector_search: 语义相似度检索，适合概念性问题
- keyword_search: 关键词精确匹配，适合查找特定术语
- hybrid_search: 混合检索（推荐），结合语义和关键词
- summarize: 文本摘要

【查询优化工具】
- query_expansion: 生成查询变体，提升召回率（查询词少时使用）
- query_decomposition: 分解复杂查询（对比、分析、多步骤问题）
- extract_keywords: 提取核心关键词（长查询或需要精确匹配时）

【高级检索工具】
- multi_query_search: 并行检索多个查询（需要多角度信息时）
- filtered_search: 带过滤条件的检索（需要特定类型文档时）
- rerank_results: 重排序结果（初次检索质量不高时）

【结果处理工具】
- merge_documents: 合并多个文档信息（多来源信息整合）
- compare_documents: 对比文档异同（对比类问题）

【网络搜索工具】
- web_search: 搜索互联网最新信息（知识库中没有的内容时使用）

【决策策略】
1. 知识问答任务：
   - 简单查询：hybrid_search → 直接回答
   - 查询词少：query_expansion → multi_query_search → 回答
   - 复杂查询：query_decomposition → multi_query_search → merge_documents → 回答
   - 对比问题：query_decomposition → 分别检索 → compare_documents → 回答
   - 结果不佳：自行判断原因，换用其他检索工具（如 query_expansion、rerank_results）重新检索
   - 知识库无结果：⚠️ 必须调用 web_search 搜索互联网 → 基于搜索结果回答
   - 涉及公司、人物、时事、最新动态等实时信息：优先使用 web_search

"""

        # 未检索到结果时的处理
        prompt += """
【重要：未检索到结果时的处理】
如果检索工具返回空结果（0个文档）或结果完全不相关：
1. ⚠️ 必须先调用 web_search 工具搜索互联网，尝试获取最新信息
2. 如果 web_search 返回了有效结果，基于搜索结果回答，并注明信息来源
3. 只有当 web_search 也无法获取有效信息时，才降级使用通用知识回答：
   - 在回答开头加上"⚠️ 知识库和网络搜索均未找到相关信息，以下是基于通用知识的回答："
   - 基于通用知识提供简洁回答（控制在200字以内）
   - 建议用户换一种方式提问或确认信息来源
如果知识库检索到了任何文档（哪怕只有1个），请直接基于这些文档回答，不要输出"未找到"警告。

【注意事项】
- 每次调用1-3个工具，不要过多
- 每次检索后，自行评估结果与问题的相关性：
  - 结果高度相关且信息充足 → 直接回答
  - 结果部分相关或信息不足 → 换用其他检索工具补充（如 query_expansion、rerank_results、web_search）
  - 结果完全不相关 → 调用 web_search 搜索互联网
- 最多5轮迭代，避免过度检索
- 信息充足时立即给出答案
- 知识库检索不到结果时，必须先尝试 web_search，不要直接使用通用知识回答
"""

        return prompt

    def _handle_react(self, query: str, context: Dict, start_time: float, user_id: str) -> str:
        """
        ReAct模式处理查询（推理-行动循环：想一步 → 调工具 → 看结果 → 再想）
        Agent自主决策调用哪些工具、调用顺序、何时停止

        循环上限 max_iterations（最大迭代次数）= 5，写死在函数体内（不是构造参数），
        防止 LLM 反复调工具停不下来；到顶就注入"别再调工具、直接回答"来强制收敛。
        消息格式遵循 function calling 约定（role=assistant 带 tool_calls、
        role=tool 带 tool_call_id），这是与 LLM 接口之间的隐式契约，顺序不能乱。

        Args:
            query: 用户查询
            context: 上下文信息
            start_time: 开始时间
            user_id: 用户ID

        Returns:
            最终答案
        """
        logger.info("[ReAct] 开始ReAct循环")
        logger.info(f"[ReAct查询] {query}")

        # 收集检索过程中命中的源文档信息
        self._retrieved_sources = []

        # 获取工具schema
        tools = self.tool_registry.get_tools_schema()

        if not tools:
            # 降级到优化流程
            logger.warning("[ReAct] 无可用工具，降级到优化流程")
            return self._handle_with_optimizations(query, context, start_time, user_id)

        logger.info(f"[ReAct工具] 可用工具数量: {len(tools)}")

        # 构建系统提示词（根据启用的功能动态生成）
        system_prompt = self._build_system_prompt(context)
        logger.info(f"[ReAct系统提示] 提示词长度: {len(system_prompt)} 字符")

        # 获取对话历史（由 LangGraph 共享记忆层注入）
        history_messages = context.get("history", [])

        # 初始化对话
        messages = [
            {"role": "system", "content": system_prompt},
            *history_messages,  # 插入对话历史
            {"role": "user", "content": query}
        ]

        max_iterations = 5  # 最多5轮ReAct循环；局部常量，写了5，提示词里也写了"最多5轮"，改要两边一起改
        retrieval_count = 0  # 检索次数统计
        seen_tool_calls = set()  # 重复调用检测
        logger.info(f"[ReAct配置] 最大迭代次数: {max_iterations}")

        for iteration in range(max_iterations):
            logger.info(f"[ReAct] 第 {iteration + 1}/{max_iterations} 轮")
            self._emit_progress(context, f"ReAct 第{iteration + 1}轮：正在思考...", stage="thinking")

            try:
                # 调用LLM（支持function calling）
                response = self.llm.chat(messages, tools=tools, temperature=0.3)

                # 记录LLM的思考过程
                logger.info(f"[ReAct思考] LLM响应内容: {response[:500]}{'...' if len(response) > 500 else ''}")

                # 解析工具调用
                tool_calls = parse_tool_calls(response)

                if not tool_calls:
                    # 没有工具调用，说明Agent认为可以直接回答了
                    logger.info("[ReAct] Agent决定给出最终答案")
                    logger.info(f"[ReAct最终答案] {response[:300]}{'...' if len(response) > 300 else ''}")
                    self._emit_progress(context, f"ReAct 第{iteration + 1}轮：生成最终答案", stage="answering")

                    # 记录指标
                    if self.enable_optimizations:
                        total_time = time.time() - start_time
                        self.metrics_collector.record("react_total_latency_ms", total_time * 1000)
                        self.metrics_collector.record("react_iterations", iteration + 1)
                        self.metrics_collector.record("react_retrieval_count", retrieval_count)

                    # 过滤掉内部工具调用信息
                    if response.startswith("[内部]"):
                        messages.append({
                            "role": "user",
                            "content": "请直接给出用户友好的答案，不要输出工具调用信息。"
                        })
                        response = self.llm.chat(messages, temperature=0.5)

                    # 附加引用源文档信息
                    response = self._append_sources(response)

                    return response

                # 执行工具调用（带去重 + 并行）
                tool_names = [tc["function"]["name"] for tc in tool_calls]
                self._emit_progress(context, f"ReAct 第{iteration + 1}轮：调用工具 {', '.join(tool_names)}", stage="tool_call")

                tool_results = self._execute_tool_calls(tool_calls, retrieval_count, seen_tool_calls)

                if not tool_results:
                    logger.info("[ReAct] 所有工具调用均为重复，强制收敛")
                    messages.append({"role": "user", "content": "请基于已有信息给出最终答案。"})
                    final = self.llm.chat(messages, temperature=0.5)
                    return self._append_sources(final or "")

                # 更新检索次数
                for tr in tool_results:
                    if tr.get("tool") in ["vector_search", "keyword_search", "hybrid_search"]:
                        retrieval_count += 1

                # 发射工具结果事件
                result_summary = ", ".join([tr["tool"] for tr in tool_results])
                self._emit_progress(context, f"工具执行完成: {result_summary}", stage="tool_result")

                # ---------- 标准 tool calling 消息格式 ----------
                tool_call_summary = ", ".join([f"{tr['tool']}({tr['arguments']})" for tr in tool_results])
                logger.info(f"[ReAct工具总结] 本轮调用: {tool_call_summary}")

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

                logger.info(f"[ReAct工具结果摘要] 共 {len(tool_results)} 个工具返回结果")

                # ========== 硬保障：知识库检索空结果时强制调用 web_search ==========
                retrieval_tools = {"hybrid_search", "vector_search", "keyword_search"}
                web_search_called = "web_search" in seen_tool_calls
                empty_retrieval = any(
                    tr["tool"] in retrieval_tools and (
                        not tr["result"]
                        or tr["result"] == "[]"
                        or "未找到" in str(tr["result"])
                        or "没有找到" in str(tr["result"])
                        or "无相关" in str(tr["result"])
                        or str(tr["result"]).strip() == "[]"
                    )
                    for tr in tool_results
                )
                if empty_retrieval and not web_search_called:
                    logger.info("[ReAct硬保障] 知识库检索返回空结果且未调用web_search，注入强制指令")
                    messages.append({
                        "role": "user",
                        "content": (
                            "知识库检索未找到相关信息。"
                            "请立即调用 web_search 工具搜索互联网获取相关信息，不要直接回答。"
                        )
                    })

                # 仅最后一轮追加 user 消息强制收敛
                if iteration >= max_iterations - 2:
                    logger.info(f"[ReAct提示] 最后一轮，要求给出答案")
                    messages.append({
                        "role": "user",
                        "content": "这是最后一轮检索机会。请基于已有信息给出最佳答案，不要再调用工具。"
                    })

            except Exception as e:
                logger.error(f"[ReAct] 第{iteration + 1}轮执行失败: {e}", exc_info=True)

                # 如果是第一轮就失败，降级到优化流程
                if iteration == 0:
                    logger.warning("[ReAct] 首轮失败，降级到优化流程")
                    return self._handle_with_optimizations(query, context, start_time, user_id)

                # 否则尝试基于已有信息生成答案
                messages.append({
                    "role": "user",
                    "content": "工具执行出现问题，请基于已有信息给出答案。"
                })

                try:
                    final_answer = self.llm.chat(messages, temperature=0.5)
                    return self._append_sources(final_answer or "")
                except:
                    return "抱歉，处理您的请求时遇到了问题。请稍后再试。"

        # 达到最大迭代次数，强制要求给出答案
        logger.warning(f"[ReAct] 达到最大迭代次数 {max_iterations}，强制生成答案")
        messages.append({
            "role": "user",
            "content": "请基于目前已有的所有信息给出最终答案，不要再调用工具。如果信息不足，请诚实说明。"
        })

        try:
            final_answer = self.llm.chat(messages, temperature=0.5)
            logger.info(f"[ReAct强制答案] {final_answer[:300]}{'...' if len(final_answer) > 300 else ''}")

            # 记录指标
            if self.enable_optimizations:
                total_time = time.time() - start_time
                self.metrics_collector.record("react_total_latency_ms", total_time * 1000)
                self.metrics_collector.record("react_iterations", max_iterations)
                self.metrics_collector.record("react_retrieval_count", retrieval_count)
                self.metrics_collector.increment("react_max_iterations_reached")

            return self._append_sources(final_answer or "")
        except Exception as e:
            logger.error(f"[ReAct] 生成最终答案失败: {e}")
            return "抱歉，我无法完成您的请求。请尝试重新表述您的问题。"

    def _handle_with_optimizations(self, query: str, context: Dict, start_time: float, user_id: str) -> str:
        """
        使用优化模块处理查询 —— 注意：**这是降级路径，不是主路径**

        只有两种情况会进来：ReAct 发现无可用工具，或 ReAct 首轮就抛异常。
        所以这里列的"查询优化 / HybridRetriever / 上下文压缩 / 答案生成 / 重试 /
        评估 / 告警"虽然在 _init_optimization_modules 里都初始化了，
        日常请求并不会走到它们 —— 读初始化代码时别以为它们在主链路上。

        另：本方法没有判断 enable_optimizations，开关关闭时这里会因属性为 None 而抛异常，
        再被自己的 except 兜住转去 _handle_basic（结果见那个方法的说明）。
        """
        try:
            # 1. 查询优化
            logger.info("[优化] 步骤1: 查询优化")
            optimized_queries = self.query_optimizer.optimize(query)
            logger.info(f"[优化] 生成 {len(optimized_queries)} 个查询变体")

            # 2. 混合检索（带缓存和重试）
            logger.info("[优化] 步骤2: 混合检索")
            def retrieve_with_retry():
                all_results = []
                for q in optimized_queries:
                    results = self.cached_retriever.retrieve(q, top_k=10)
                    all_results.extend(results)
                # 去重并取top-5
                seen = set()
                unique_results = []
                for r in all_results:
                    content_key = r.get("content", "")[:100]
                    if content_key not in seen:
                        seen.add(content_key)
                        unique_results.append(r)
                return unique_results[:5]

            retrieved_docs = self.retry_strategy.execute(retrieve_with_retry)

            # 记录检索
            retrieval_time = time.time()
            retrieval_latency = (retrieval_time - start_time) * 1000
            self.structured_logger.log_retrieval(
                query=query,
                num_results=len(retrieved_docs),
                latency_ms=retrieval_latency
            )
            self.metrics_collector.record("retrieval_latency_ms", retrieval_latency)

            # 检查检索结果
            if not retrieved_docs:
                logger.warning("[优化] 未检索到相关文档，尝试使用LLM通用知识回答")
                fallback_answer = self._fallback_to_llm_knowledge(query, user_id)

                # 空结果缓存：缓存"未找到"的标记，防止缓存穿透
                if self.cache_manager:
                    self.cache_manager.query_cache.set(query, fallback_answer)
                    logger.debug("[空结果缓存] 已缓存未找到结果")

                return fallback_answer

            logger.info(f"[优化] 检索到 {len(retrieved_docs)} 个文档")

            # 3. 上下文压缩
            logger.info("[优化] 步骤3: 上下文压缩")
            compressed_docs = self.compressor.compress(query, retrieved_docs, use_llm=False)
            logger.info(f"[优化] 压缩后文档数: {len(compressed_docs)}")

            # 4. 答案生成（带引用）
            logger.info("[优化] 步骤4: 答案生成")
            result = self.answer_generator.generate(query, compressed_docs, add_citation=True)

            # 记录生成
            generation_time = time.time()
            generation_latency = (generation_time - retrieval_time) * 1000
            self.structured_logger.log_generation(
                query=query,
                answer=result["answer"],
                latency_ms=generation_latency
            )
            self.metrics_collector.record("generation_latency_ms", generation_latency)

            # 5. 评估
            total_latency = (generation_time - start_time) * 1000
            self.evaluator.evaluate_latency(start_time, generation_time)
            logger.info(f"[优化] 总延迟: {total_latency:.1f}ms")

            # 6. 检查告警
            self.alert_manager.check_alerts({
                "retrieval_latency_ms": retrieval_latency,
                "generation_latency_ms": generation_latency
            })

            # 格式化输出
            answer = result["answer"]
            if result.get("citations"):
                answer += "\n\n参考来源："
                for citation in result["citations"]:
                    answer += f"\n[{citation['id']}] {citation['source']}"

            return answer

        except Exception as e:
            logger.error(f"优化流程失败: {e}", exc_info=True)
            # 降级到基础流程
            logger.warning("降级到基础流程")
            return self._handle_basic(query, context)

    def _handle_basic(self, query: str, context: Dict) -> str:
        """
        基础处理流程（无优化）

        ⚠️ 事实性说明（只记录，未改逻辑）：本方法内调用 self.function_calling.execute_function，
        但 self.function_calling 在本类里从未被赋值 —— 所以这段代码只要执行到那一步
        就必然抛 AttributeError，被自己的 except 吞掉后返回"处理失败: ..."。
        换言之它是一条实际跑不通的死路径，只剩下日志痕迹。
        """
        try:
            # 获取工具schema
            tools = self.tool_registry.get_tools_schema()

            # 构建提示词
            system_prompt = """你是一个知识检索助手。你可以使用以下工具来回答用户问题：
- vector_search: 语义相似度检索
- keyword_search: 关键词精确匹配
- hybrid_search: 混合检索
- summarize: 文本摘要

请根据用户问题选择合适的工具。"""

            # 调用LLM
            logger.info(f"[RAG步骤4] 调用LLM分析问题并选择检索工具")
            response = self.llm.generate(
                query,
                system_prompt=system_prompt,
                tools=tools
            )

            # 解析工具调用
            tool_calls = parse_tool_calls(response)

            if tool_calls:
                # 执行工具调用
                logger.info(f"[RAG步骤5] 检测到工具调用: {[tc['function']['name'] for tc in tool_calls]}")
                results = []
                for tool_call in tool_calls:
                    func_name = tool_call["function"]["name"]
                    try:
                        arguments = json.loads(tool_call["function"]["arguments"])
                    except json.JSONDecodeError:
                        arguments = parse_llm_json(tool_call["function"]["arguments"], fallback={})

                    logger.info(f"[RAG步骤5] 执行工具: {func_name}, 参数: {arguments}")
                    result = self.function_calling.execute_function(func_name, arguments)
                    results.append(result)

                # 检查检索结果是否为空
                has_valid_results = False
                for result in results:
                    if isinstance(result, list) and len(result) > 0:
                        has_valid_results = True
                        break
                    elif isinstance(result, str) and len(result.strip()) > 0:
                        has_valid_results = True
                        break

                if not has_valid_results:
                    logger.warning(f"[RAG步骤6] 未检索到相关文档，尝试使用LLM通用知识回答")
                    return self._fallback_to_llm_knowledge(query, context.get("user_id", "anonymous"))

                # 基于工具结果生成最终回答
                tool_results_text = "\n\n".join([str(r) for r in results])
                final_prompt = f"""基于以下检索结果回答用户问题。

用户问题: {query}

检索结果:
{tool_results_text}

请给出简洁准确的回答:"""

                logger.info(f"[RAG步骤7] 基于检索结果生成最终答案")
                final_answer = self.llm.generate(final_prompt, temperature=0.5)
                logger.info(f"[RAG步骤8] 答案生成完成")
                return final_answer
            else:
                logger.info(f"[RAG步骤8] 未使用工具，直接使用LLM回答")
                return response

        except Exception as e:
            logger.error(f"基础流程失败: {e}", exc_info=True)
            return f"处理失败: {str(e)}"

    def vector_search(self, query: str, top_k: int = 3, audience: Optional[str] = None) -> List[Dict]:
        """
        向量检索（语义相似度路线）

        与 keyword_search 的差别：这条只走向量库，靠 embedding 的语义接近来找，
        问题换个说法也能命中，但精确术语（型号、编号）容易漏。
        """
        if not self.retriever:
            return []

        try:
            logger.info(f"[RAG步骤5.1] 向量检索 - 查询: '{query}', top_k: {top_k}")

            # 推断用户角色：document_filter 会按 audience 卡可见范围，
            # 所以先猜"用户是谁"再过滤 —— 猜错就会多滤或漏滤，是这套过滤的固有风险
            if not audience and self.document_filter:
                from core.document_filter import infer_audience_from_query
                audience = infer_audience_from_query(query)
                logger.info(f"推断用户角色: {audience}")

            # 检索
            logger.info(f"[RAG步骤5.2] 在向量数据库中检索相关文档")
            # top_k（取前 k 条）= 只保留得分最高的 k 个。这里先取 2k 条是留过滤余量：
            # 按 audience 过滤后常剩不够 k 条（recall 召回 = 先把可能相关的都捞出来）
            results = self.retriever.retrieve(query, top_k=top_k * 2)  # 多检索一些，过滤后可能不够
            logger.info(f"[RAG步骤5.3] 检索到 {len(results)} 个候选文档")

            # 文档过滤
            if self.document_filter and audience:
                logger.info(f"[RAG步骤6] 根据用户角色过滤文档: {audience}")
                filtered_results = []
                for doc in results:
                    doc_id = doc.get("metadata", {}).get("doc_id")
                    if doc_id:
                        # 检查是否匹配受众
                        from core.document_filter import Audience
                        if self.document_filter.filter_by_audience([doc_id], Audience(audience)):
                            filtered_results.append(doc)
                    else:
                        # 没有doc_id的文档默认保留
                        filtered_results.append(doc)

                results = filtered_results[:top_k]
                logger.info(f"[RAG步骤6] 文档过滤完成: {len(results)} 个结果匹配受众 {audience}")
            else:
                results = results[:top_k]

            final_results = [
                {
                    "content": doc.get("content", ""),
                    "score": doc.get("score", 0.0),
                    "metadata": doc.get("metadata", {})
                }
                for doc in results
            ]

            # 收集引用源信息
            self._collect_sources(final_results)

            return final_results
        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            return []

    def keyword_search(self, query: str, top_k: int = 3, audience: Optional[str] = None) -> List[Dict]:
        """关键词检索（BM25 精确匹配路线）—— 不走向量，纯靠词面命中"""
        if not self.bm25_retriever:
            return []

        try:
            # 推断用户角色
            if not audience and self.document_filter:
                from core.document_filter import infer_audience_from_query
                audience = infer_audience_from_query(query)

            # 检索
            results = self.bm25_retriever.retrieve(query, top_k=top_k * 2)

            # 文档过滤
            if self.document_filter and audience:
                filtered_results = []
                for doc in results:
                    doc_id = doc.get("metadata", {}).get("doc_id")
                    if doc_id:
                        from core.document_filter import Audience
                        if self.document_filter.filter_by_audience([doc_id], Audience(audience)):
                            filtered_results.append(doc)
                    else:
                        filtered_results.append(doc)

                results = filtered_results[:top_k]
            else:
                results = results[:top_k]

            kw_results = [
                {
                    "content": doc.get("content", ""),
                    "score": doc.get("score", 0.0),
                    "metadata": doc.get("metadata", {})
                }
                for doc in results
            ]

            # 收集引用源信息
            self._collect_sources(kw_results)

            return kw_results
        except Exception as e:
            logger.error(f"关键词检索失败: {e}")
            return []

    def hybrid_search(
        self,
        query: str,
        top_k: int = 3,
        vector_weight: float = 0.8
    ) -> List[Dict]:
        """
        混合检索（RRF融合）

        RRF（倒数排名融合）= 不看原始分数、只看"排第几"，把多路结果按 1/(k+排名) 相加
        融合成一个列表 —— 好处是向量分与 BM25 分量纲不同也不怕。
        hybrid retrieval（混合检索）= 向量检索 + 关键词检索一起用，这里两路各取 2k 条。

        与 self.hybrid_retriever 的区别（易混点）：那个昂贵的 HybridRetriever 只用在
        降级路径；本方法是自己内联实现的 RRF，不经过 self.hybrid_retriever，也不带 reranker。

        Args:
            query: 查询文本
            top_k: 返回结果数量
            vector_weight: 向量检索权重

        Returns:
            融合后的结果
        """
        # 抑制子调用的 _collect_sources，避免不同尺度分数混入同一列表
        # 用实例属性当"开关"是隐式契约：_collect_sources 开头会 getattr 它。
        # 副作用：多查询并行时多个线程共用这一个属性，可能互相把开关改回 False（竞态）
        self._suppress_source_collection = True
        try:
            vector_results = self.vector_search(query, top_k=top_k * 2)
            keyword_results = self.keyword_search(query, top_k=top_k * 2)
        finally:
            self._suppress_source_collection = False

        # ========== 建立 doc_id -> 原始分数 的映射 ==========
        vector_score_map = {}   # doc_id -> 向量相似度（0-1）
        bm25_score_map = {}     # doc_id -> BM25 分数
        for doc in vector_results:
            did = doc.get("id") or doc.get("content", "")[:50]
            vector_score_map[did] = doc.get("metadata", {}).get("_vector_score") or round(doc.get("score", 0.0), 4)
        for doc in keyword_results:
            did = doc.get("id") or doc.get("content", "")[:50]
            bm25_score_map[did] = doc.get("metadata", {}).get("_bm25_score") or round(doc.get("score", 0.0), 4)

        # RRF融合
        k = 60  # RRF参数：平滑常数，取 60 是原论文推荐值，作用是压低头部排名的差距
        scores = {}
        doc_map = {}  # doc_id -> 原始文档

        # 向量检索结果
        for rank, doc in enumerate(vector_results):
            doc_id = doc.get("id") or doc.get("content", "")[:50]
            scores[doc_id] = scores.get(doc_id, 0) + vector_weight / (k + rank + 1)
            if doc_id not in doc_map:
                doc_map[doc_id] = doc

        # 关键词检索结果
        for rank, doc in enumerate(keyword_results):
            doc_id = doc.get("id") or doc.get("content", "")[:50]
            scores[doc_id] = scores.get(doc_id, 0) + (1 - vector_weight) / (k + rank + 1)
            if doc_id not in doc_map:
                doc_map[doc_id] = doc

        # 排序
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # ========== 按源文件去重：同一文件只保留最高分 chunk ==========
        # chunk（文本块）= 文档切分后的片段，检索的最小单位。同一份文档被切成很多块，
        # 命中时会有多条同源结果 —— 不去重的话 top_k 会被同一篇文档占满。
        seen_files = {}
        deduped_docs = []
        for doc_id, rrf_score in sorted_docs:
            doc = doc_map.get(doc_id)
            if not doc:
                continue
            source_file = doc.get("metadata", {}).get("source", doc_id)
            if source_file not in seen_files:
                seen_files[source_file] = (rrf_score, doc_id)
                deduped_docs.append((doc_id, rrf_score))

        # ========== 绝对阈值过滤：RRF 分数过低说明不相关 ==========
        max_rrf = deduped_docs[0][1] if deduped_docs else 0.0
        RRF_MIN_THRESHOLD = 0.005
        if max_rrf < RRF_MIN_THRESHOLD:
            logger.info(f"[混合检索] 最高RRF分数 {max_rrf:.6f} 低于阈值 {RRF_MIN_THRESHOLD}，无相关结果")
            return []

        # 将 RRF 分数归一化到 0-1 范围（min-max）
        # 为什么要归一化：下游 _append_sources 按 0.3 绝对阈值和"最高分 25%"过滤，
        # 不归一化的话 RRF 的原始分（几十到几百量级）会让这些过滤条件全部失效
        min_rrf = deduped_docs[-1][1] if len(deduped_docs) > 1 else 0.0
        rrf_range = max_rrf - min_rrf

        # 提取向量和BM25的全局统计（从各自结果的第一个文档元数据中取）
        vector_stats = {}
        if vector_results:
            vm = vector_results[0].get("metadata", {})
            if "_vector_total" in vm:
                vector_stats = {
                    "_vector_corpus_size": vm.get("_vector_corpus_size", vm["_vector_total"]),
                    "_vector_total": vm["_vector_total"],
                    "_vector_filtered": vm["_vector_filtered"],
                }
        bm25_stats = {}
        if keyword_results:
            bm = keyword_results[0].get("metadata", {})
            if "_bm25_matched" in bm:
                bm25_stats = {
                    "_bm25_corpus_size": bm.get("_bm25_corpus_size", 0),
                    "_bm25_matched": bm["_bm25_matched"],
                    "_bm25_filtered": bm["_bm25_filtered"],
                }

        # 返回top-k，注入三种原始分数到 metadata
        results = []
        for doc_id, raw_rrf in deduped_docs[:top_k]:
            doc = doc_map.get(doc_id)
            if doc:
                if rrf_range > 0:
                    norm_score = (raw_rrf - min_rrf) / rrf_range
                else:
                    norm_score = 1.0
                meta = dict(doc.get("metadata", {}))
                # 注入三种原始分数（前端展示用）
                if doc_id in vector_score_map:
                    meta["_vector_score"] = vector_score_map[doc_id]
                if doc_id in bm25_score_map:
                    meta["_bm25_score"] = bm25_score_map[doc_id]
                meta["_rrf_score"] = round(raw_rrf, 6)
                # 注入全局统计（_collect_sources 从 results[0] 读取）
                meta.update(vector_stats)
                meta.update(bm25_stats)
                results.append({
                    "content": doc.get("content", ""),
                    "score": round(norm_score, 4),
                    "metadata": meta
                })

        # 收集引用源信息（使用归一化后的分数，与 vector_search 分数尺度一致）
        self._collect_sources(results)

        return results

    def summarize(self, text: str, max_length: int = 200) -> str:
        """文本摘要"""
        try:
            prompt = f"""请对以下文本进行简洁摘要，不超过{max_length}字：

{text}

摘要："""

            summary = self.llm.generate(prompt, temperature=0.3, max_tokens=max_length)
            return summary
        except Exception as e:
            logger.error(f"摘要生成失败: {e}")
            return text[:max_length]

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
                "cache_stats": self.cache_manager.get_stats(),
                "metrics": self.metrics_collector.get_all_stats(),
                "evaluator_summary": self.evaluator.get_summary(),
                "online_metrics": self.online_metrics.get_metrics(),
                "alert_history": self.alert_manager.get_alert_history(limit=5)
            }
            return stats
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {"error": str(e)}

    # ========== 扩展工具实现（优先级1：基于LLM） ==========

    def query_expansion(self, query: str, num_variants: int = 3) -> List[str]:
        """
        查询扩展：生成查询的多个变体（多花一次 LLM 调用，换更多命中）

        本方法只产出变体文本、自己不做检索；要和 multi_query_search 配合
        才形成"多路召回"。返回列表里第一个元素始终是原始查询，保证兜底。

        Args:
            query: 原始查询
            num_variants: 生成变体数量

        Returns:
            查询变体列表（包含原始查询）
        """
        try:
            prompt = f"""请为以下查询生成{num_variants}个语义相似但表达不同的变体，用于提升检索召回率。

原始查询：{query}

要求：
1. 保持原始查询的核心意图
2. 使用不同的表达方式、同义词、相关术语
3. 每个变体一行，不要编号

变体："""

            response = self.llm.generate(prompt, temperature=0.7, max_tokens=200)

            # 解析变体
            variants = [line.strip() for line in response.strip().split('\n') if line.strip()]
            variants = [query] + variants[:num_variants]  # 包含原始查询

            logger.info(f"[查询扩展] 生成了 {len(variants)} 个查询变体")
            return variants

        except Exception as e:
            logger.error(f"查询扩展失败: {e}")
            return [query]

    def query_decomposition(self, query: str) -> List[str]:
        """
        查询分解：将复杂查询分解为多个子查询

        Args:
            query: 复杂查询

        Returns:
            子查询列表
        """
        try:
            prompt = f"""请将以下复杂查询分解为多个简单的子查询，每个子查询关注一个具体方面。

查询：{query}

要求：
1. 识别查询中的多个主题或方面
2. 每个子查询应该独立且具体
3. 适合对比、分析、多步骤推理等复杂问题
4. 每个子查询一行，不要编号

子查询："""

            response = self.llm.generate(prompt, temperature=0.3, max_tokens=300)

            # 解析子查询
            sub_queries = [line.strip() for line in response.strip().split('\n') if line.strip()]

            if not sub_queries:
                logger.warning("[查询分解] 未能分解查询，返回原始查询")
                return [query]

            logger.info(f"[查询分解] 分解为 {len(sub_queries)} 个子查询")
            return sub_queries

        except Exception as e:
            logger.error(f"查询分解失败: {e}")
            return [query]

    def extract_keywords(self, query: str, max_keywords: int = 5) -> List[str]:
        """
        关键词提取：从查询中提取核心关键词

        Args:
            query: 查询文本
            max_keywords: 最大关键词数量

        Returns:
            关键词列表
        """
        try:
            prompt = f"""请从以下查询中提取最多{max_keywords}个核心关键词。

查询：{query}

要求：
1. 提取最重要的名词、术语、概念
2. 去除停用词和无意义词
3. 保留专业术语和领域词汇
4. 每个关键词一行，不要编号

关键词："""

            response = self.llm.generate(prompt, temperature=0.1, max_tokens=100)

            # 解析关键词
            keywords = [line.strip() for line in response.strip().split('\n') if line.strip()]
            keywords = keywords[:max_keywords]

            logger.info(f"[关键词提取] 提取了 {len(keywords)} 个关键词: {keywords}")
            return keywords

        except Exception as e:
            logger.error(f"关键词提取失败: {e}")
            return [query]

    def evaluate_result_quality(self, results: str, query: str) -> Dict[str, Any]:
        """
        结果质量评估：评估检索结果的质量和相关性

        Args:
            results: 检索结果（JSON字符串）
            query: 原始查询

        Returns:
            评估结果 {quality_score, relevance, completeness, issues, suggestions}
        """
        try:
            import json

            # 解析结果
            try:
                results_data = json.loads(results) if isinstance(results, str) else results
            except:
                results_data = results

            # 截断过长的结果
            results_preview = str(results_data)[:1000]

            prompt = f"""请评估以下检索结果的质量。

查询：{query}

检索结果（预览）：
{results_preview}

请从以下维度评估（输出JSON格式）：
1. quality_score: 总体质量分数（0-1）
2. relevance: 相关性（high/medium/low）
3. completeness: 完整性（complete/partial/insufficient）
4. issues: 存在的问题列表
5. suggestions: 改进建议列表

只输出JSON："""

            response = self.llm.generate(prompt, temperature=0.1, max_tokens=300)

            # 解析JSON
            evaluation = parse_llm_json(response, fallback={
                "quality_score": 0.7,
                "relevance": "medium",
                "completeness": "partial",
                "issues": ["无法解析LLM评估结果"],
                "suggestions": ["重新检索"]
            })

            logger.info(f"[质量评估] 分数: {evaluation.get('quality_score', 0)}, 相关性: {evaluation.get('relevance', 'unknown')}")
            return evaluation

        except Exception as e:
            logger.error(f"质量评估失败: {e}")
            return {
                "quality_score": 0.5,
                "relevance": "unknown",
                "completeness": "unknown",
                "issues": [str(e)],
                "suggestions": ["重试"]
            }

    def extract_concepts(
        self,
        query: str,
        documents: str,
        extract_relations: bool = True
    ) -> Dict[str, Any]:
        """
        跨文档概念提取：从多个文档中提取语义概念和关系

        接线说明：底层 concept_extractor 只在优化模块初始化成功时才存在，
        所以下面要用 hasattr 探一下；本方法不在主流程里，靠 LLM 主动调工具触发。

        Args:
            query: 用户查询
            documents: 文档列表（JSON字符串）
            extract_relations: 是否提取概念间关系

        Returns:
            概念提取结果 {concepts, relations, concept_map, summary}
        """
        try:
            # 解析文档列表
            if isinstance(documents, str):
                docs = json.loads(documents)
            else:
                docs = documents

            if not docs:
                logger.warning("[概念提取] 文档列表为空")
                return {
                    "concepts": [],
                    "relations": [],
                    "concept_map": {},
                    "summary": "文档列表为空，无法提取概念。"
                }

            logger.info(f"[概念提取] 从 {len(docs)} 个文档中提取概念")

            # 调用概念提取器
            if self.enable_optimizations and hasattr(self, 'concept_extractor'):
                result = self.concept_extractor.extract(
                    query=query,
                    documents=docs,
                    extract_relations=extract_relations
                )

                # 记录指标
                if hasattr(self, 'metrics_collector'):
                    self.metrics_collector.record_metric(
                        "concept_extraction",
                        {
                            "num_documents": len(docs),
                            "num_concepts": len(result.get("concepts", [])),
                            "num_relations": len(result.get("relations", [])),
                            "elapsed_time": result.get("metadata", {}).get("elapsed_time", 0)
                        }
                    )

                logger.info(f"[概念提取] 提取到 {len(result.get('concepts', []))} 个概念")
                return result
            else:
                logger.warning("[概念提取] 概念提取器未初始化，返回空结果")
                return {
                    "concepts": [],
                    "relations": [],
                    "concept_map": {},
                    "summary": "概念提取器未启用。"
                }

        except json.JSONDecodeError as e:
            logger.error(f"[概念提取] JSON解析失败: {e}")
            return {
                "concepts": [],
                "relations": [],
                "concept_map": {},
                "summary": "文档格式错误。"
            }
        except Exception as e:
            logger.error(f"[概念提取] 失败: {e}", exc_info=True)
            return {
                "concepts": [],
                "relations": [],
                "concept_map": {},
                "summary": f"概念提取失败: {str(e)}"
            }

    # ========== 扩展工具实现（优先级2：需要适度开发） ==========

    def multi_query_search(
        self,
        queries: List[str],
        top_k: int = 3,
        strategy: str = "parallel"
    ) -> List[Dict]:
        """
        多查询检索：并行检索多个查询并融合结果

        融合只是"按分数排序 + 按内容前缀去重"，不是 RRF —— 分数来自各次 hybrid_search
        的归一化分，量纲一致所以能直接比。并行跑线程池，注意 hybrid_search 那个抑制开关
        是实例级共享的（见该函数注释里的竞态说明）。

        Args:
            queries: 查询列表
            top_k: 每个查询返回的结果数
            strategy: 检索策略（parallel/sequential）

        Returns:
            融合后的检索结果
        """
        try:
            logger.info(f"[多查询检索] 检索 {len(queries)} 个查询，策略: {strategy}")

            all_results = []
            seen_ids = set()

            def _search_one(q):
                return self.hybrid_search(q, top_k=top_k)

            # 根据 strategy 决定并行还是串行
            if strategy == "parallel" and len(queries) > 1:
                logger.info(f"[多查询检索] 并行执行 {len(queries)} 个查询")
                with ThreadPoolExecutor(max_workers=min(len(queries), 4)) as pool:
                    futures = {pool.submit(_search_one, q): q for q in queries}
                    for i, fut in enumerate(as_completed(futures)):
                        q = futures[fut]
                        logger.info(f"[多查询检索] 第 {i+1}/{len(queries)} 个查询完成: {q}")
                        results = fut.result()
                        for doc in results:
                            doc_id = doc.get("id") or doc.get("content", "")[:100]
                            if doc_id not in seen_ids:
                                seen_ids.add(doc_id)
                                all_results.append(doc)
            else:
                for i, q in enumerate(queries):
                    logger.info(f"[多查询检索] 第 {i+1}/{len(queries)} 个查询: {q}")
                    results = self.hybrid_search(q, top_k=top_k)
                    for doc in results:
                        doc_id = doc.get("id") or doc.get("content", "")[:100]
                        if doc_id not in seen_ids:
                            seen_ids.add(doc_id)
                            all_results.append(doc)

            # 按分数排序
            all_results.sort(key=lambda x: x.get("score", 0), reverse=True)

            logger.info(f"[多查询检索] 共检索到 {len(all_results)} 个去重结果")
            return all_results[:top_k * len(queries)]

        except Exception as e:
            logger.error(f"多查询检索失败: {e}")
            return []

    def filtered_search(
        self,
        query: str,
        filters: Optional[Dict] = None,
        top_k: int = 5
    ) -> List[Dict]:
        """
        过滤检索：带元数据过滤的检索

        Args:
            query: 查询文本
            filters: 过滤条件（如 audience, doc_type 等）
            top_k: 返回结果数量

        Returns:
            过滤后的检索结果
        """
        try:
            logger.info(f"[过滤检索] 查询: {query}, 过滤条件: {filters}")

            # 先检索更多结果
            results = self.hybrid_search(query, top_k=top_k * 3)

            if not filters:
                return results[:top_k]

            # 应用过滤：只检查 metadata 里**存在**的键 —— filters 给了某字段、
            # 而文档没有该字段时算通过（放行）。这与"必须匹配所有条件"的直觉相反，如实记录
            filtered_results = []
            for doc in results:
                metadata = doc.get("metadata", {})

                # 检查所有过滤条件
                match = True
                for key, value in filters.items():
                    if key in metadata:
                        if metadata[key] != value:
                            match = False
                            break

                if match:
                    filtered_results.append(doc)

                if len(filtered_results) >= top_k:
                    break

            logger.info(f"[过滤检索] 过滤后剩余 {len(filtered_results)} 个结果")
            return filtered_results

        except Exception as e:
            logger.error(f"过滤检索失败: {e}")
            return []

    def rerank_results(
        self,
        results: str,
        query: str,
        method: str = "auto"
    ) -> List[Dict]:
        """
        结果重排序：对检索结果重新排序

        两条打分路线：method="llm" 逐个调 API 打分（慢、花钱、灵活）；
        auto/api/cross_encoder/rule 走专用重排序器。cross-encoder（交叉编码器）=
        把"问题+文档"拼在一起送进模型打分，比向量比对更准但更慢。
        ⚠️ 默认值不一致：args_schema（RerankResultsArgs）里写的是 "llm"，
        方法签名里写的是 "auto" —— 经 registry 调用时以 Schema 里的为准。

        Args:
            results: 检索结果（JSON字符串）
            query: 查询文本
            method: 重排序方法（auto/api/cross_encoder/llm/rule）

        Returns:
            重排序后的结果
        """
        try:
            import json

            # 解析结果
            try:
                results_data = json.loads(results) if isinstance(results, str) else results
            except:
                results_data = results

            if not isinstance(results_data, list):
                logger.warning("[重排序] 结果格式错误")
                return []

            logger.info(f"[重排序] 使用 {method} 方法重排序 {len(results_data)} 个结果")

            if method == "llm":
                # 使用LLM重排序（逐个调API，慢但灵活）
                reranked = self._rerank_with_llm(results_data, query)
            elif method in ("auto", "api", "cross_encoder", "rule"):
                # 使用专用重排序器（API/本地模型/规则）
                reranked = self._rerank_with_reranker(results_data, query, method)
            else:
                # 默认按原始分数排序
                reranked = sorted(results_data, key=lambda x: x.get("score", 0), reverse=True)

            logger.info(f"[重排序] 完成")
            return reranked

        except Exception as e:
            logger.error(f"重排序失败: {e}")
            return []

    def _rerank_with_reranker(self, results: List[Dict], query: str, method: str = "auto") -> List[Dict]:
        """
        使用专用重排序器（API/本地模型/规则）

        每次调用都新建一个 reranker 实例（若加载本地模型，这一步很贵）——
        本 Agent 主路径不经过这里，但被 LLM 反复当工具调用时会有重复加载开销。
        """
        try:
            from rag_core.reranker import RerankerFactory

            reranker = RerankerFactory.create_reranker(method)

            # 提取文档内容
            docs = [doc.get("content", "") for doc in results]

            # 批量打分
            scores = reranker.rank(query, docs)

            # 更新分数
            for i, doc in enumerate(results):
                doc["rerank_score"] = scores[i]

            # 按重排序分数排序
            results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

            return results

        except Exception as e:
            logger.error(f"专用重排序器失败: {e}，降级为原始排序")
            return sorted(results, key=lambda x: x.get("score", 0), reverse=True)

    def _rerank_with_llm(self, results: List[Dict], query: str) -> List[Dict]:
        """使用LLM重排序"""
        try:
            # 为每个文档生成相关性分数
            scored_results = []

            for i, doc in enumerate(results[:10]):  # 最多重排序前10个
                content_preview = doc.get("content", "")[:300]

                prompt = f"""评估以下文档与查询的相关性，给出0-1的分数。

查询：{query}

文档：{content_preview}

只输出分数（0-1的小数）："""

                try:
                    response = self.llm.generate(prompt, temperature=0.1, max_tokens=10)
                    score = float(response.strip())
                    score = max(0.0, min(1.0, score))  # 限制在0-1之间
                except:
                    score = doc.get("score", 0.5)

                doc["rerank_score"] = score
                scored_results.append(doc)

            # 按重排序分数排序
            scored_results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

            # 添加未重排序的结果
            scored_results.extend(results[10:])

            return scored_results

        except Exception as e:
            logger.error(f"LLM重排序失败: {e}")
            return results

    def merge_documents(
        self,
        documents: str,
        strategy: str = "summarize"
    ) -> str:
        """
        文档合并：合并多个文档的信息

        Args:
            documents: 文档列表（JSON字符串）
            strategy: 合并策略（concat/summarize/extract）

        Returns:
            合并后的文本
        """
        try:
            import json

            # 解析文档
            try:
                docs_data = json.loads(documents) if isinstance(documents, str) else documents
            except:
                docs_data = documents

            if not isinstance(docs_data, list):
                return str(docs_data)

            logger.info(f"[文档合并] 合并 {len(docs_data)} 个文档，策略: {strategy}")

            # 提取内容
            contents = []
            for doc in docs_data:
                if isinstance(doc, dict):
                    content = doc.get("content", "")
                else:
                    content = str(doc)
                if content:
                    contents.append(content)

            if not contents:
                return ""

            if strategy == "concat":
                # 简单拼接
                merged = "\n\n".join(contents)

            elif strategy == "summarize":
                # 摘要合并
                all_text = "\n\n".join(contents)
                prompt = f"""请将以下多个文档的信息合并为一个连贯的摘要，去除重复内容。

文档内容：
{all_text[:2000]}

合并摘要："""

                merged = self.llm.generate(prompt, temperature=0.3, max_tokens=500)

            elif strategy == "extract":
                # 提取关键信息
                all_text = "\n\n".join(contents)
                prompt = f"""请从以下文档中提取关键信息，去重并整理。

文档内容：
{all_text[:2000]}

关键信息："""

                merged = self.llm.generate(prompt, temperature=0.3, max_tokens=500)

            else:
                merged = "\n\n".join(contents)

            logger.info(f"[文档合并] 合并完成，长度: {len(merged)}")
            return merged

        except Exception as e:
            logger.error(f"文档合并失败: {e}")
            return ""

    def compare_documents(
        self,
        documents: str,
        aspects: Optional[List[str]] = None
    ) -> str:
        """
        文档对比：对比两个或多个文档的异同

        Args:
            documents: 文档列表（JSON字符串）
            aspects: 对比的方面

        Returns:
            对比结果
        """
        try:
            import json

            # 解析文档
            try:
                docs_data = json.loads(documents) if isinstance(documents, str) else documents
            except:
                docs_data = documents

            if not isinstance(docs_data, list) or len(docs_data) < 2:
                return "需要至少2个文档才能对比"

            logger.info(f"[文档对比] 对比 {len(docs_data)} 个文档")

            # 提取内容
            contents = []
            for i, doc in enumerate(docs_data[:5]):  # 最多对比5个
                if isinstance(doc, dict):
                    content = doc.get("content", "")
                else:
                    content = str(doc)
                contents.append(f"文档{i+1}:\n{content[:500]}")

            all_text = "\n\n".join(contents)

            # 构建对比提示
            if aspects:
                aspects_text = "、".join(aspects)
                prompt = f"""请对比以下文档在【{aspects_text}】方面的异同。

{all_text}

对比分析："""
            else:
                prompt = f"""请对比以下文档的主要异同点。

{all_text}

对比分析："""

            comparison = self.llm.generate(prompt, temperature=0.3, max_tokens=600)

            logger.info(f"[文档对比] 对比完成")
            return comparison

        except Exception as e:
            logger.error(f"文档对比失败: {e}")
            return f"对比失败: {str(e)}"

    def web_search(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Tavily 网络搜索：从互联网检索最新信息

        需要环境变量 TAVILY_API_KEY；缺 key 或缺包都返回一条"不可用"的假结果
        （而不是抛异常）—— 这样 ReAct 循环能继续跑，不会因外部依赖缺失整体中断。
        返回结构与检索工具一致（content/title/url/score），下游处理可以直接复用。

        Args:
            query: 搜索查询文本
            max_results: 返回结果数量

        Returns:
            搜索结果列表，每项包含 title、url、content、score
        """
        try:
            from tavily import TavilyClient
        except ImportError:
            logger.error("[WebSearch] tavily-python 未安装，请运行: pip install tavily-python")
            return [{"content": "网络搜索不可用：请先安装 tavily-python", "title": "", "url": "", "score": 0.0}]

        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            logger.warning("[WebSearch] TAVILY_API_KEY 未设置，跳过网络搜索")
            return [{"content": "网络搜索不可用：请在 .env 中配置 TAVILY_API_KEY", "title": "", "url": "", "score": 0.0}]

        try:
            client = TavilyClient(api_key=api_key)
            response = client.search(
                query=query,
                max_results=max_results,
                search_depth="basic"
            )
            results = []
            for item in response.get("results", []):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                    "score": item.get("score", 0.0),
                    "metadata": {"source": item.get("url", ""), "title": item.get("title", "")}
                })
            logger.info(f"[WebSearch] 搜索完成，返回 {len(results)} 条结果")
            return results
        except Exception as e:
            logger.error(f"[WebSearch] 搜索失败: {e}")
            return [{"content": f"网络搜索失败: {str(e)}", "title": "", "url": "", "score": 0.0}]

    def _fallback_to_llm_knowledge(self, query: str, user_id: str) -> str:
        """
        降级到LLM通用知识回答（fallback = 主路径失败时退到备用方案）
        当向量数据库未检索到结果时，使用LLM的通用知识回答

        ⚠️ 定位提醒：ReAct 主路径不走这里 —— 主路径的"知识库没有就转 web_search"
        写死在系统提示词里。本方法只服务于降级路径 _handle_with_optimizations。

        Args:
            query: 用户查询
            user_id: 用户ID

        Returns:
            基于LLM通用知识的回答
        """
        try:
            logger.info("[降级] 使用LLM通用知识回答")

            # 记录降级事件
            if self.enable_optimizations:
                self.metrics_collector.increment("fallback_to_llm_knowledge")
                self.structured_logger.log_query(
                    query=query,
                    user_id=user_id,
                    metadata={"fallback": True}
                )

            # 构建提示词
            prompt = f"""【重要提示】知识库中未找到相关信息，请基于你的通用知识回答。

用户问题：{query}

回答要求：
1. 首先明确说明"知识库中未找到相关信息，以下是基于通用知识的回答"
2. 提供简洁、准确的回答（控制在200字以内）
3. 如果不确定，诚实说明
4. 建议用户提供更多上下文或换一种方式提问

回答："""

            # 调用LLM（使用较低的temperature保证准确性）
            answer = self.llm.generate(prompt, temperature=0.3, max_tokens=300)

            # 格式化输出
            formatted_answer = f"""⚠️ 知识库中未找到相关信息

{answer}

💡 提示：
- 如果您需要更准确的信息，请尝试换一种方式提问
- 或者确认知识库中是否包含相关文档"""

            logger.info("[降级] LLM通用知识回答完成")
            return formatted_answer

        except Exception as e:
            logger.error(f"[降级] LLM通用知识回答失败: {e}")
            return """⚠️ 知识库中未找到相关信息

抱歉，我在知识库中没有找到与您问题相关的信息，且无法提供通用知识回答。

💡 建议：
- 请尝试换一种方式提问
- 或者确认知识库中是否包含相关文档
- 如果问题持续，请联系管理员"""



