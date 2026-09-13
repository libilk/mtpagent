# -*- coding: utf-8 -*-
"""
Knowledge Agent
===============

处理知识检索相关任务（已集成8大生产级优化）
"""

import logging
import time
import json
from typing import Dict, Any, List, Optional
from llm.output_parser import parse_llm_json

logger = logging.getLogger(__name__)


class KnowledgeAgent:
    """知识检索Agent（生产级优化版）"""

    def __init__(
        self,
        llm,
        retriever=None,
        bm25_retriever=None,
        function_calling=None,
        document_filter=None,
        embedder=None,
        crm=None,

        # 功能开关（全部默认启用）
        enable_optimizations: bool = True,
        enable_customer_service: bool = True,
        enable_memory: bool = True,
        enable_sentiment: bool = True,

        # 高级功能开关（全部默认启用）
        enable_rerank: bool = True,
        enable_hyde: bool = True,
        enable_self_consistency: bool = True,

        # 可调参数
        max_history: int = 10,
        cache_size: int = 1000,
        cache_ttl: int = 3600,
        max_retries: int = 3,
        vector_weight: float = 0.5,
        max_context_length: int = 2000,
    ):
        """
        初始化

        Args:
            llm: LLM实例
            retriever: 语义检索器（来自rag_core.retriever）
            bm25_retriever: BM25检索器（来自rag_core.bm25_retriever）
            function_calling: Function Calling管理器
            document_filter: 文档过滤器
            embedder: 嵌入器实例
            crm: CRM系统实例（用于客服功能）

            enable_optimizations: 是否启用生产级优化（默认True）
            enable_customer_service: 是否启用客服功能（默认True）
            enable_memory: 是否启用对话记忆（默认True）
            enable_sentiment: 是否启用情感分析（默认True）

            enable_rerank: 是否启用重排序（默认True）
            enable_hyde: 是否启用HyDE（默认True）
            enable_self_consistency: 是否启用Self-Consistency（默认True）

            max_history: 对话历史最大轮数（默认10）
            cache_size: 缓存大小（默认1000）
            cache_ttl: 缓存TTL秒数（默认3600）
            max_retries: 最大重试次数（默认3）
            vector_weight: 向量检索权重（默认0.5）
            max_context_length: 上下文最大长度（默认2000）
        """
        self.llm = llm
        self.retriever = retriever
        self.bm25_retriever = bm25_retriever
        self.function_calling = function_calling
        self.document_filter = document_filter
        self.embedder = embedder
        self.enable_optimizations = enable_optimizations

        # 保存配置参数
        self.enable_customer_service = enable_customer_service
        self.enable_memory = enable_memory
        self.enable_sentiment = enable_sentiment
        self.enable_rerank = enable_rerank
        self.enable_hyde = enable_hyde
        self.enable_self_consistency = enable_self_consistency
        self.max_history = max_history
        self.cache_size = cache_size
        self.cache_ttl = cache_ttl
        self.max_retries = max_retries
        self.vector_weight = vector_weight
        self.max_context_length = max_context_length

        # 初始化 Tavily Search 服务
        from core.tavily_search_service import TavilySearchService
        try:
            self.tavily_search = TavilySearchService()
            logger.info("✓ Tavily Search 初始化完成")
        except Exception as e:
            logger.warning(f"Tavily Search 初始化失败，将使用模拟数据: {e}")
            self.tavily_search = None

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
            self.hybrid_retriever = None
            self.query_optimizer = None
            self.cached_retriever = None
            self.compressor = None
            self.answer_generator = None
            self.retry_strategy = None
            self.evaluator = None

        # 初始化客服模块
        if enable_customer_service:
            self._init_customer_service_modules(crm)
        else:
            self.crm = None

        # 初始化对话记忆
        if enable_memory:
            self._init_memory_modules()
        else:
            self.memory = None

        # 注册工具
        if function_calling:
            self._register_tools()
            if enable_customer_service:
                self._register_customer_tools()

        logger.info(f"KnowledgeAgent初始化完成 (优化: {enable_optimizations}, 客服: {enable_customer_service}, 记忆: {enable_memory})")

    def _init_optimization_modules(self):
        """初始化8大优化模块"""
        try:
            # 1. 混合检索 + 重排序
            from rag_core.hybrid_retriever import HybridRetriever, SimpleReranker
            self.hybrid_retriever = HybridRetriever(
                vector_retriever=self.retriever,
                bm25_retriever=self.bm25_retriever,
                vector_weight=self.vector_weight,  # 使用参数
                use_rerank=self.enable_rerank,  # 使用参数
                reranker=SimpleReranker() if self.enable_rerank else None
            )
            logger.info("✓ 混合检索器初始化完成")

            # 2. 查询优化
            from rag_core.query_optimizer import QueryOptimizer
            self.query_optimizer = QueryOptimizer(
                llm=self.llm,
                enable_hyde=self.enable_hyde  # 使用参数
            )
            logger.info("✓ 查询优化器初始化完成")

            # 3. 缓存机制
            from rag_core.cache_manager import CacheManager, CachedRetriever, CachedEmbedder
            self.cache_manager = CacheManager(
                max_size=self.cache_size,  # 使用参数
                ttl=self.cache_ttl  # 使用参数
            )
            self.cached_retriever = CachedRetriever(self.hybrid_retriever, self.cache_manager)

            # 向量层缓存：替换 retriever 内部的 embedder 为 cached_embedder
            if self.embedder and self.retriever:
                self.cached_embedder = CachedEmbedder(self.embedder, self.cache_manager)
                self.retriever.embedder = self.cached_embedder
                logger.info("✓ 向量缓存已接入 retriever")

            logger.info("✓ 缓存管理器初始化完成")

            # 3.5 语义缓存（业务层第二道防线）
            if self.embedder:
                from core.semantic_cache import SemanticCache
                self.semantic_cache = SemanticCache(
                    embedder=self.embedder,
                    similarity_threshold=0.95,
                    max_size=self.cache_size,
                    ttl=self.cache_ttl
                )
                logger.info("✓ 语义缓存初始化完成")
            else:
                self.semantic_cache = None

            # 4. 上下文压缩
            from rag_core.context_compressor import ContextCompressor
            self.compressor = ContextCompressor(
                llm=self.llm,
                max_length=self.max_context_length  # 使用参数
            )
            logger.info("✓ 上下文压缩器初始化完成")

            # 5. 答案生成优化
            from rag_core.answer_generator import AnswerGenerator
            self.answer_generator = AnswerGenerator(
                llm=self.llm,
                enable_self_consistency=self.enable_self_consistency,  # 使用参数
                num_samples=3
            )
            logger.info("✓ 答案生成器初始化完成")

            # 6. 错误处理
            from core.error_handler import RetryStrategy, FallbackStrategy, GracefulErrorHandler
            self.retry_strategy = RetryStrategy(max_retries=self.max_retries)  # 使用参数
            self.fallback_strategy = FallbackStrategy()
            self.error_handler = GracefulErrorHandler()
            logger.info("✓ 错误处理器初始化完成")

            # 7. 评估体系
            from rag_core.evaluator import RAGEvaluator, OnlineMetrics
            self.evaluator = RAGEvaluator()
            self.online_metrics = OnlineMetrics()
            logger.info("✓ 评估器初始化完成")

            # 8. 监控告警
            from rag_core.monitoring import structured_logger, metrics_collector, alert_manager
            self.structured_logger = structured_logger
            self.metrics_collector = metrics_collector
            self.alert_manager = alert_manager
            logger.info("✓ 监控系统初始化完成")

            # 9. 跨文档概念提取器
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
            logger.error(f"优化模块初始化失败: {e}", exc_info=True)
            logger.warning("降级为基础模式运行")
            self.enable_optimizations = False

    def _init_customer_service_modules(self, crm):
        """初始化客服模块"""
        try:
            from core.crm_mock import MockCRM

            self.crm = crm or MockCRM()

            logger.info("=" * 50)
            logger.info("✓ 客服系统初始化完成")
            logger.info("=" * 50)

        except Exception as e:
            logger.error(f"客服系统初始化失败: {e}", exc_info=True)
            logger.warning("客服功能将被禁用")
            self.crm = None
            self.enable_customer_service = False

    def _init_memory_modules(self):
        """初始化对话记忆模块"""
        try:
            from core.memory import ConversationMemory

            self.memory = ConversationMemory(
                llm=self.llm,
                max_history=self.max_history,
                compress_every=8  # 每8轮压缩一次
            )

            logger.info("=" * 50)
            logger.info("✓ 对话记忆初始化完成")
            logger.info("=" * 50)

        except Exception as e:
            logger.error(f"对话记忆初始化失败: {e}", exc_info=True)
            logger.warning("对话记忆功能将被禁用")
            self.memory = None
            self.enable_memory = False

    def _register_tools(self):
        """注册工具"""
        # 向量检索
        self.function_calling.register_function(
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

        # 关键词检索
        self.function_calling.register_function(
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

        # 混合检索
        self.function_calling.register_function(
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

        # 文本摘要
        self.function_calling.register_function(
            name="summarize",
            description="对长文本进行摘要提取",
            parameters={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "待摘要的文本"
                    },
                    "max_length": {
                        "type": "integer",
                        "description": "摘要最大长度",
                        "default": 200
                    }
                },
                "required": ["text"]
            },
            function=self.summarize
        )

        # ========== 扩展工具（优先级1：基于LLM） ==========

        # 查询扩展
        self.function_calling.register_function(
            name="query_expansion",
            description="生成查询的多个变体，提升召回率。适合查询词较少或需要多角度检索的场景",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "原始查询"
                    },
                    "num_variants": {
                        "type": "integer",
                        "description": "生成变体数量",
                        "default": 3
                    }
                },
                "required": ["query"]
            },
            function=self.query_expansion
        )

        # 查询分解
        self.function_calling.register_function(
            name="query_decomposition",
            description="将复杂查询分解为多个子查询。适合对比、分析、多步骤推理等复杂问题",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "复杂查询"
                    }
                },
                "required": ["query"]
            },
            function=self.query_decomposition
        )

        # 关键词提取
        self.function_calling.register_function(
            name="extract_keywords",
            description="从查询中提取核心关键词。适合长查询或需要精确匹配的场景",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "查询文本"
                    },
                    "max_keywords": {
                        "type": "integer",
                        "description": "最大关键词数量",
                        "default": 5
                    }
                },
                "required": ["query"]
            },
            function=self.extract_keywords
        )

        # 结果质量评估
        self.function_calling.register_function(
            name="evaluate_result_quality",
            description="评估检索结果的质量和相关性。用于判断是否需要重新检索或调整策略",
            parameters={
                "type": "object",
                "properties": {
                    "results": {
                        "type": "string",
                        "description": "检索结果（JSON字符串）"
                    },
                    "query": {
                        "type": "string",
                        "description": "原始查询"
                    }
                },
                "required": ["results", "query"]
            },
            function=self.evaluate_result_quality
        )

        # 信息完整性检查
        self.function_calling.register_function(
            name="check_information_completeness",
            description="检查信息是否完整，是否足以回答问题。用于判断是否需要补充检索",
            parameters={
                "type": "object",
                "properties": {
                    "information": {
                        "type": "string",
                        "description": "已获取的信息"
                    },
                    "query": {
                        "type": "string",
                        "description": "原始查询"
                    }
                },
                "required": ["information", "query"]
            },
            function=self.check_information_completeness
        )

        # 跨文档概念提取
        self.function_calling.register_function(
            name="extract_concepts",
            description="从多个检索到的文档中提取跨文档语义概念，构建概念关系图。适合需要理解文档间共同主题、对比分析、知识整合的场景",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用户查询"
                    },
                    "documents": {
                        "type": "string",
                        "description": "文档列表（JSON字符串）"
                    },
                    "extract_relations": {
                        "type": "boolean",
                        "description": "是否提取概念间关系",
                        "default": True
                    }
                },
                "required": ["query", "documents"]
            },
            function=self.extract_concepts
        )

        # ========== 扩展工具（优先级2：需要适度开发） ==========

        # 多查询检索
        self.function_calling.register_function(
            name="multi_query_search",
            description="并行检索多个查询并融合结果。适合需要多角度信息的复杂问题",
            parameters={
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "查询列表"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "每个查询返回的结果数",
                        "default": 3
                    },
                    "strategy": {
                        "type": "string",
                        "enum": ["parallel", "sequential"],
                        "description": "检索策略",
                        "default": "parallel"
                    }
                },
                "required": ["queries"]
            },
            function=self.multi_query_search
        )

        # 过滤检索
        self.function_calling.register_function(
            name="filtered_search",
            description="带元数据过滤的检索。可以根据文档类型、时间、作者等过滤",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "查询文本"
                    },
                    "filters": {
                        "type": "object",
                        "description": "过滤条件（如 audience, doc_type 等）"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回结果数量",
                        "default": 5
                    }
                },
                "required": ["query"]
            },
            function=self.filtered_search
        )

        # 结果重排序
        self.function_calling.register_function(
            name="rerank_results",
            description="对检索结果重新排序，提升相关性。适合初次检索结果质量不高的情况",
            parameters={
                "type": "object",
                "properties": {
                    "results": {
                        "type": "string",
                        "description": "检索结果（JSON字符串）"
                    },
                    "query": {
                        "type": "string",
                        "description": "查询文本"
                    },
                    "method": {
                        "type": "string",
                        "enum": ["llm", "cross_encoder"],
                        "description": "重排序方法",
                        "default": "llm"
                    }
                },
                "required": ["results", "query"]
            },
            function=self.rerank_results
        )

        # 文档合并
        self.function_calling.register_function(
            name="merge_documents",
            description="合并多个文档的信息，去重并整合。适合从多个来源获取信息后的整合",
            parameters={
                "type": "object",
                "properties": {
                    "documents": {
                        "type": "string",
                        "description": "文档列表（JSON字符串）"
                    },
                    "strategy": {
                        "type": "string",
                        "enum": ["concat", "summarize", "extract"],
                        "description": "合并策略",
                        "default": "summarize"
                    }
                },
                "required": ["documents"]
            },
            function=self.merge_documents
        )

        # 文档对比
        self.function_calling.register_function(
            name="compare_documents",
            description="对比两个或多个文档的异同。适合对比类问题",
            parameters={
                "type": "object",
                "properties": {
                    "documents": {
                        "type": "string",
                        "description": "文档列表（JSON字符串）"
                    },
                    "aspects": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "对比的方面（如性能、功能、价格等）"
                    }
                },
                "required": ["documents"]
            },
            function=self.compare_documents
        )

    def _register_customer_tools(self):
        """注册客服工具（4个）"""

        # 1. 情感分析
        self.function_calling.register_function(
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

        # 2. 创建工单
        self.function_calling.register_function(
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

        # 3. 查询工单
        self.function_calling.register_function(
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

        # 4. 查询用户信息
        self.function_calling.register_function(
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

        # 5. Tavily Search 网络搜索
        if self.tavily_search:
            self.function_calling.register_function(
                name="web_search",
                description="通过Tavily Search搜索互联网最新信息，适合查询最新法规、行业标准、实时数据等知识库中没有的内容",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "搜索查询"
                        },
                        "count": {
                            "type": "integer",
                            "default": 5,
                            "description": "返回结果数量"
                        }
                    },
                    "required": ["query"]
                },
                function=self.web_search
            )

        # 6. SQLite MCP 数据库查询
        if self.sqlite_mcp:
            self.function_calling.register_function(
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

            # 7. 列出数据库所有表
            self.function_calling.register_function(
                name="list_tables",
                description="列出数据库中所有表名。在查询数据库前应先调用此工具了解有哪些表。",
                parameters={
                    "type": "object",
                    "properties": {},
                    "required": []
                },
                function=self.list_database_tables
            )

            # 8. 查看表结构
            self.function_calling.register_function(
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

    def _resolve_coreference(self, query: str) -> str:
        """
        指代消解：有对话历史时，由LLM判断是否需要消解并一次完成

        仅以"是否存在对话历史"作为唯一前置条件，避免关键词规则漏判。
        LLM在同一次调用中完成判断和消解：需要则返回重写后的查询，不需要则原样返回。

        Args:
            query: 用户查询

        Returns:
            消解后的查询
        """
        if not self.memory:
            return query

        # 唯一前置条件：有对话历史才调用LLM
        history = self.memory.get_messages()
        if not history:
            return query

        # 取最近4轮对话作为消解上下文
        history_text = "\n".join([
            f"{'用户' if msg['role'] == 'user' else '助手'}: {msg['content']}"
            for msg in history[-4:]
        ])

        prompt = f"""请根据对话历史，判断当前查询是否存在指代不明或省略的情况，并进行消解。

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
            resolved = self.llm.generate(prompt, temperature=0.1, max_tokens=100).strip()
            # 去除可能的引号
            resolved = resolved.strip('"\'""''')
            return resolved if resolved else query
        except Exception as e:
            logger.warning(f"指代消解失败: {e}")
            return query

    def _execute_tool_calls(self, tool_calls: List[Dict], retrieval_count: int) -> List[Dict]:
        """
        执行工具调用（使用函数映射表方式）

        Args:
            tool_calls: 工具调用列表
            retrieval_count: 当前检索次数（用于日志）

        Returns:
            工具执行结果列表
        """
        tool_results = []

        for tool_call in tool_calls:
            func_name = tool_call["function"]["name"]
            arguments = json.loads(tool_call["function"]["arguments"])

            logger.info(f"[ReAct工具调用] 工具名称: {func_name}")
            logger.info(f"[ReAct工具参数] {json.dumps(arguments, ensure_ascii=False)}")

            # 执行工具（通过函数映射表）
            tool_start = time.time()
            try:
                result = self.function_calling.execute_function(func_name, arguments)
                tool_latency = (time.time() - tool_start) * 1000

                # 记录工具执行结果摘要
                if isinstance(result, list):
                    logger.info(f"[ReAct工具结果] 返回 {len(result)} 条结果")
                elif isinstance(result, dict):
                    logger.info(f"[ReAct工具结果] 返回字典，键: {list(result.keys())}")
                elif isinstance(result, str):
                    logger.info(f"[ReAct工具结果] 返回文本，长度: {len(result)} 字符")
                else:
                    logger.info(f"[ReAct工具结果] 返回类型: {type(result)}")

                # 记录工具执行
                if self.enable_optimizations:
                    self.metrics_collector.record(f"tool_{func_name}_latency_ms", tool_latency)

                tool_results.append({
                    "tool": func_name,
                    "arguments": arguments,
                    "result": result,
                    "success": True
                })

                logger.info(f"[ReAct] 工具执行完成，耗时 {tool_latency:.1f}ms")

            except Exception as e:
                tool_latency = (time.time() - tool_start) * 1000
                logger.error(f"[ReAct] 工具执行失败: {func_name}, 错误: {e}")

                # 记录错误
                if self.enable_optimizations:
                    self.metrics_collector.increment(f"tool_{func_name}_errors")

                tool_results.append({
                    "tool": func_name,
                    "arguments": arguments,
                    "result": f"工具执行失败: {str(e)}",
                    "success": False,
                    "error": str(e)
                })

        return tool_results


    def handle(self, query: str, context: Dict) -> str:
        """
        处理查询（ReAct模式 + 生产级优化 + 对话记忆 + 情感分析）

        Args:
            query: 用户查询
            context: 上下文信息

        Returns:
            处理结果
        """
        start_time = time.time()
        user_id = context.get("user_id", "anonymous")

        try:
            # 指代消解
            if self.memory:
                original_query = query
                query = self._resolve_coreference(query)
                if query != original_query:
                    logger.info(f"[指代消解] {original_query} → {query}")

            # 业务层缓存：第一道防线 - 精确匹配
            if self.enable_optimizations and self.cache_manager:
                cached_answer = self.cache_manager.query_cache.get(query)
                if cached_answer:
                    logger.info(f"[业务精确缓存命中] 直接返回缓存答案")
                    if self.memory:
                        self.memory.add_user_message(query)
                        self.memory.add_assistant_message(cached_answer)
                    return cached_answer

            # 业务层缓存：第二道防线 - 语义相似匹配
            if self.enable_optimizations and self.semantic_cache:
                cached_answer = self.semantic_cache.get(query)
                if cached_answer:
                    logger.info(f"[业务语义缓存命中] 直接返回缓存答案")
                    if self.memory:
                        self.memory.add_user_message(query)
                        self.memory.add_assistant_message(cached_answer)
                    return cached_answer

            # 添加到对话记忆
            if self.memory:
                self.memory.add_user_message(query)

            # 情感分析（如果启用）
            sentiment = None
            if self.enable_sentiment and self.enable_customer_service:
                sentiment = self.analyze_sentiment(query)
                logger.info(f"[情感分析] {sentiment}")

                # 如果情绪强烈负面，标记为紧急
                if sentiment["sentiment"] == "negative" and sentiment["score"] < -0.6:
                    context["urgent"] = True
                    context["sentiment"] = sentiment
                    logger.warning("⚠️ 检测到强烈负面情绪")

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

            # 添加到对话记忆
            if self.memory:
                self.memory.add_assistant_message(result)

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
        构建系统提示词（根据启用的功能动态生成）

        Args:
            context: 上下文信息（包含情感、紧急标记等）

        Returns:
            系统提示词
        """
        # 基础提示词
        prompt = """你是一个专业的智能助手，具备知识检索"""

        # 如果启用客服功能，添加客服能力描述
        if self.enable_customer_service:
            prompt += "和客服"

        prompt += """能力。你需要通过调用工具来收集信息，然后回答用户问题。

工作流程：
1. 分析用户问题，判断查询类型和复杂度
2. 选择合适的工具组合获取信息
3. 评估结果质量，必要时调整策略
4. 当信息充足时，给出最终答案

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

【质量控制工具】
- evaluate_result_quality: 评估检索结果质量（判断是否需要重新检索）
- check_information_completeness: 检查信息完整性（判断是否需要补充）
"""

        # 如果启用数据库查询功能，添加数据库工具说明
        if self.sqlite_mcp:
            prompt += """
【数据库工具】
- list_tables: 列出数据库中所有表名（查询数据库前必须先调用）
- describe_table: 查看指定表的结构（列名、类型、是否主键）
- query_database: 执行SQL查询（仅支持SELECT语句）
"""

        # 如果启用客服功能，添加客服工具说明
        if self.enable_customer_service:
            prompt += """
【客服工具】
- analyze_sentiment: 分析用户情绪（正面/负面/中性），适合客服场景
- create_ticket: 为用户创建客服工单，适合投诉、问题反馈场景
- query_ticket: 查询工单状态，适合用户跟进问题
- query_user_info: 查询用户基本信息和历史记录，适合个性化服务
"""

        # 决策策略
        prompt += """
【决策策略】
1. 知识问答任务：
   - 简单查询：hybrid_search → 直接回答
   - 查询词少：query_expansion → multi_query_search → 回答
   - 复杂查询：query_decomposition → multi_query_search → merge_documents → 回答
   - 对比问题：query_decomposition → 分别检索 → compare_documents → 回答
   - 结果不佳：evaluate_result_quality → 调整策略 → 重新检索
"""

        # 数据库查询决策策略
        if self.sqlite_mcp:
            prompt += """
2. 数据库查询任务：
   - 推荐流程：list_tables → describe_table → query_database → 回答
   - 先调用 list_tables 了解有哪些表
   - 再调用 describe_table 了解目标表的列名和类型
   - 最后根据表结构生成正确的SQL并调用 query_database
   - 注意表名和字段名大小写敏感
"""

        if self.enable_customer_service:
            prompt += """
3. 客服咨询任务：
   - 先分析情感：analyze_sentiment（如果用户情绪明显）
   - 如果需要查询信息：query_user_info
   - 如果需要创建工单：create_ticket
   - 如果需要跟进工单：query_ticket

4. 混合任务（知识 + 客服）：
   - 先检索知识：hybrid_search
   - 如果知识库无答案，查询用户历史：query_user_info
   - 如果仍无法解决，创建工单：create_ticket
"""

        # 添加情感上下文
        if context.get("urgent"):
            sentiment = context.get("sentiment", {})
            prompt += f"""
【⚠️ 重要提示】
当前用户情绪：{sentiment.get('sentiment', 'negative')}（评分：{sentiment.get('score', -0.7):.2f}）
用户情绪强烈负面，请：
1. 优先考虑创建工单（create_ticket）
2. 使用同理心语言，表达理解和歉意
3. 提供明确的解决方案或时间表
4. 避免推诿或模糊回答
"""

        # 未检索到结果时的处理
        prompt += """
【重要：未检索到结果时的处理】
如果检索工具返回空结果或结果很少（少于2个文档）：
1. 首先明确告知用户："知识库中未找到相关信息"
2. 然后基于你的通用知识提供简洁回答（控制在200字以内）
3. 在回答开头加上"⚠️ 知识库中未找到相关信息，以下是基于通用知识的回答："
4. 建议用户换一种方式提问或确认知识库内容

【注意事项】
- 每次调用1-3个工具，不要过多
- 使用质量控制工具判断是否需要继续
- 最多5轮迭代，避免过度检索
- 信息充足时立即给出答案
"""

        if self.enable_customer_service:
            prompt += "- 情绪负面时，优先考虑创建工单\n"

        prompt += "- 检索不到结果时，使用通用知识回答"

        return prompt

    def _handle_react(self, query: str, context: Dict, start_time: float, user_id: str) -> str:
        """
        ReAct模式处理查询
        Agent自主决策调用哪些工具、调用顺序、何时停止

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

        # 获取工具schema
        tools = self.function_calling.get_tools_schema() if self.function_calling else None

        if not tools:
            # 降级到优化流程
            logger.warning("[ReAct] 无可用工具，降级到优化流程")
            return self._handle_with_optimizations(query, context, start_time, user_id)

        logger.info(f"[ReAct工具] 可用工具数量: {len(tools)}")

        # 构建系统提示词（根据启用的功能动态生成）
        system_prompt = self._build_system_prompt(context)
        logger.info(f"[ReAct系统提示] 提示词长度: {len(system_prompt)} 字符")

        # 获取对话历史
        if self.memory:
            history_messages = self.memory.get_messages()
            logger.info(f"[ReAct对话历史] 加载 {len(history_messages)} 条历史消息")
        else:
            history_messages = []
            logger.info(f"[ReAct对话历史] 无历史消息")

        # 初始化对话
        messages = [
            {"role": "system", "content": system_prompt},
            *history_messages,  # 插入对话历史
            {"role": "user", "content": query}
        ]

        max_iterations = 5  # 最多5轮ReAct循环
        retrieval_count = 0  # 检索次数统计
        logger.info(f"[ReAct配置] 最大迭代次数: {max_iterations}")

        for iteration in range(max_iterations):
            logger.info(f"[ReAct] 第 {iteration + 1}/{max_iterations} 轮")

            try:
                # 调用LLM（支持function calling）
                response = self.llm.chat(messages, tools=tools, temperature=0.3)

                # 记录LLM的思考过程
                logger.info(f"[ReAct思考] LLM响应内容: {response[:500]}{'...' if len(response) > 500 else ''}")

                # 解析工具调用
                from llm.function_calling import parse_tool_calls
                tool_calls = parse_tool_calls(response)

                if not tool_calls:
                    # 没有工具调用，说明Agent认为可以直接回答了
                    logger.info("[ReAct] Agent决定给出最终答案")
                    logger.info(f"[ReAct最终答案] {response[:300]}{'...' if len(response) > 300 else ''}")

                    # 记录指标
                    if self.enable_optimizations:
                        total_time = time.time() - start_time
                        self.metrics_collector.record("react_total_latency_ms", total_time * 1000)
                        self.metrics_collector.record("react_iterations", iteration + 1)
                        self.metrics_collector.record("react_retrieval_count", retrieval_count)

                    # 过滤掉内部工具调用信息
                    if response.startswith("[内部]"):
                        # 如果响应以[内部]开头，说明是工具调用信息，需要重新生成答案
                        messages.append({
                            "role": "user",
                            "content": "请直接给出用户友好的答案，不要输出工具调用信息。"
                        })
                        response = self.llm.chat(messages, temperature=0.5)

                    return response

                # 执行工具调用（使用函数映射表方式）
                tool_results = self._execute_tool_calls(tool_calls, retrieval_count)

                # 更新检索次数
                for tr in tool_results:
                    if tr.get("tool") in ["vector_search", "keyword_search", "hybrid_search"]:
                        retrieval_count += 1

                # 将工具调用和结果添加到对话历史
                # 格式化工具调用信息（内部使用，不显示给用户）
                tool_call_summary = ", ".join([f"{tr['tool']}({tr['arguments']})" for tr in tool_results])
                logger.info(f"[ReAct工具总结] 本轮调用: {tool_call_summary}")

                messages.append({
                    "role": "assistant",
                    "content": f"[内部] 已调用工具: {tool_call_summary}"
                })

                # 格式化工具结果
                import json
                tool_results_text = json.dumps(tool_results, ensure_ascii=False, indent=2)

                # 记录工具结果摘要到日志
                logger.info(f"[ReAct工具结果摘要] 共 {len(tool_results)} 个工具返回结果")

                # 根据迭代次数调整提示
                if iteration < max_iterations - 2:
                    follow_up = "请分析检索结果。如果信息足够，直接给出答案；如果不足，可以继续调用工具补充信息。"
                    logger.info(f"[ReAct提示] 允许继续调用工具")
                else:
                    follow_up = "这是最后一轮检索机会。请基于已有信息给出最佳答案，不要再调用工具。"
                    logger.info(f"[ReAct提示] 最后一轮，要求给出答案")

                messages.append({
                    "role": "user",
                    "content": f"工具执行结果：\n{tool_results_text}\n\n{follow_up}"
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
                    return final_answer
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

            return final_answer
        except Exception as e:
            logger.error(f"[ReAct] 生成最终答案失败: {e}")
            return "抱歉，我无法完成您的请求。请尝试重新表述您的问题。"

    def _handle_with_optimizations(self, query: str, context: Dict, start_time: float, user_id: str) -> str:
        """使用优化模块处理查询"""
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
        """基础处理流程（无优化）"""
        try:
            # 获取工具schema
            tools = self.function_calling.get_tools_schema() if self.function_calling else None

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
            from llm.function_calling import parse_tool_calls
            tool_calls = parse_tool_calls(response)

            if tool_calls:
                # 执行工具调用
                logger.info(f"[RAG步骤5] 检测到工具调用: {[tc['function']['name'] for tc in tool_calls]}")
                results = []
                for tool_call in tool_calls:
                    func_name = tool_call["function"]["name"]
                    import json
                    arguments = json.loads(tool_call["function"]["arguments"])

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
        """向量检索"""
        if not self.retriever:
            return []

        try:
            logger.info(f"[RAG步骤5.1] 向量检索 - 查询: '{query}', top_k: {top_k}")

            # 推断用户角色
            if not audience and self.document_filter:
                from core.document_filter import infer_audience_from_query
                audience = infer_audience_from_query(query)
                logger.info(f"推断用户角色: {audience}")

            # 检索
            logger.info(f"[RAG步骤5.2] 在向量数据库中检索相关文档")
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

    def keyword_search(self, query: str, top_k: int = 3, audience: Optional[str] = None) -> List[Dict]:
        """关键词检索"""
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
        vector_weight: float = 0.5
    ) -> List[Dict]:
        """
        混合检索（RRF融合）

        Args:
            query: 查询文本
            top_k: 返回结果数量
            vector_weight: 向量检索权重

        Returns:
            融合后的结果
        """
        # 获取两种检索结果
        vector_results = self.vector_search(query, top_k=top_k * 2)
        keyword_results = self.keyword_search(query, top_k=top_k * 2)

        # RRF融合
        k = 60  # RRF参数
        scores = {}

        # 向量检索结果
        for rank, doc in enumerate(vector_results):
            doc_id = doc.get("content", "")[:50]  # 使用内容前50字符作为ID
            scores[doc_id] = scores.get(doc_id, 0) + vector_weight / (k + rank + 1)

        # 关键词检索结果
        for rank, doc in enumerate(keyword_results):
            doc_id = doc.get("content", "")[:50]
            scores[doc_id] = scores.get(doc_id, 0) + (1 - vector_weight) / (k + rank + 1)

        # 排序
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # 返回top-k
        results = []
        for doc_id, score in sorted_docs[:top_k]:
            # 找到原始文档
            for doc in vector_results + keyword_results:
                if doc.get("content", "")[:50] == doc_id:
                    results.append({
                        "content": doc.get("content", ""),
                        "score": score,
                        "metadata": doc.get("metadata", {})
                    })
                    break

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
        查询扩展：生成查询的多个变体

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

    def check_information_completeness(self, information: str, query: str) -> Dict[str, Any]:
        """
        信息完整性检查：检查信息是否完整，是否足以回答问题

        Args:
            information: 已获取的信息
            query: 原始查询

        Returns:
            检查结果 {is_complete, confidence, missing_aspects, suggestions}
        """
        try:
            # 截断过长的信息
            info_preview = information[:1500]

            prompt = f"""请检查以下信息是否足以回答用户问题。

用户问题：{query}

已获取的信息：
{info_preview}

请评估（输出JSON格式）：
1. is_complete: 信息是否完整（true/false）
2. confidence: 置信度（0-1）
3. missing_aspects: 缺失的方面列表
4. suggestions: 补充建议列表

只输出JSON："""

            response = self.llm.generate(prompt, temperature=0.1, max_tokens=300)

            # 解析JSON
            check_result = parse_llm_json(response, fallback={
                "is_complete": len(information) > 100,
                "confidence": 0.6,
                "missing_aspects": [],
                "suggestions": []
            })

            logger.info(f"[完整性检查] 完整: {check_result.get('is_complete', False)}, 置信度: {check_result.get('confidence', 0)}")
            return check_result

        except Exception as e:
            logger.error(f"完整性检查失败: {e}")
            return {
                "is_complete": False,
                "confidence": 0.5,
                "missing_aspects": ["检查失败"],
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
            seen_contents = set()

            for i, q in enumerate(queries):
                logger.info(f"[多查询检索] 第 {i+1}/{len(queries)} 个查询: {q}")

                # 使用混合检索
                results = self.hybrid_search(q, top_k=top_k)

                # 去重
                for doc in results:
                    content_key = doc.get("content", "")[:100]
                    if content_key not in seen_contents:
                        seen_contents.add(content_key)
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

            # 应用过滤
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
        method: str = "llm"
    ) -> List[Dict]:
        """
        结果重排序：对检索结果重新排序

        Args:
            results: 检索结果（JSON字符串）
            query: 查询文本
            method: 重排序方法（llm/cross_encoder）

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
                # 使用LLM重排序
                reranked = self._rerank_with_llm(results_data, query)
            else:
                # 默认按原始分数排序
                reranked = sorted(results_data, key=lambda x: x.get("score", 0), reverse=True)

            logger.info(f"[重排序] 完成")
            return reranked

        except Exception as e:
            logger.error(f"重排序失败: {e}")
            return []

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

    def _fallback_to_llm_knowledge(self, query: str, user_id: str) -> str:
        """
        降级到LLM通用知识回答
        当向量数据库未检索到结果时，使用LLM的通用知识回答

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

    # ==================== 客服工具实现 ====================

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        情感分析：分析用户情绪（正面/负面/中性）

        Args:
            text: 用户输入文本

        Returns:
            情感分析结果 {sentiment, score, keywords}
        """
        try:
            logger.info("[情感分析] 开始分析")

            # 构建情感分析prompt
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

            # 解析JSON
            result = parse_llm_json(response, fallback=None)

            # 如果解析失败，降级：基于关键词的简单判断
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
        """
        创建客服工单

        Args:
            issue: 问题描述
            user_id: 用户ID（可选）
            priority: 优先级（low/normal/high/urgent）

        Returns:
            工单信息
        """
        try:
            if not self.crm:
                return {"error": "CRM系统未初始化"}

            logger.info(f"[创建工单] 问题: {issue}, 用户: {user_id}, 优先级: {priority}")

            # 调用CRM系统创建工单
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
        """
        查询工单状态

        Args:
            ticket_id: 工单ID

        Returns:
            工单信息
        """
        try:
            if not self.crm:
                return {"error": "CRM系统未初始化"}

            logger.info(f"[查询工单] 工单ID: {ticket_id}")

            # 调用CRM系统查询工单
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
        """
        查询用户信息

        Args:
            user_id: 用户ID

        Returns:
            用户信息
        """
        try:
            if not self.crm:
                return {"error": "CRM系统未初始化"}

            logger.info(f"[查询用户] 用户ID: {user_id}")

            # 调用CRM系统查询用户信息
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

    def web_search(self, query: str, count: int = 5) -> str:
        """
        网络搜索（通过 Tavily Search）

        Args:
            query: 搜索查询
            count: 返回结果数量

        Returns:
            搜索结果
        """
        try:
            if not self.tavily_search:
                return "Tavily Search 未初始化，无法进行网络搜索"

            logger.info(f"[网络搜索] 查询: {query}")

            # 调用 Tavily Search
            result = self.tavily_search.web_search(query, count)

            if result.get("success"):
                logger.info(f"[网络搜索] 搜索成功")
                return result.get("results", "")
            else:
                error = result.get("error", "搜索失败")
                logger.error(f"[网络搜索] {error}")
                return f"搜索失败: {error}"

        except Exception as e:
            logger.error(f"网络搜索失败: {e}")
            return f"网络搜索失败: {str(e)}"

    def query_database(self, sql: str) -> str:
        """
        查询数据库（通过 SQLite MCP）

        Args:
            sql: SQL查询语句

        Returns:
            查询结果
        """
        try:
            if not self.sqlite_mcp:
                return "SQLite MCP 未初始化，无法查询数据库"

            logger.info(f"[数据库查询] SQL: {sql}")

            # 安全检查：只允许SELECT语句
            if not sql.strip().upper().startswith("SELECT"):
                return "安全限制：只允许执行SELECT查询"

            # 调用 SQLite MCP
            result = self.sqlite_mcp.query(sql)

            if result.get("success"):
                data = result.get("data", [])
                columns = result.get("columns", [])

                if not data:
                    return "查询成功，但没有找到数据"

                # 格式化结果
                formatted = "【查询结果】\n\n"

                # 添加列名
                if columns:
                    formatted += " | ".join(columns) + "\n"
                    formatted += "-" * (len(" | ".join(columns))) + "\n"

                # 添加数据行（最多显示10行）
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

    def list_database_tables(self) -> str:
        """
        列出数据库中所有表名

        Returns:
            格式化的表名列表字符串
        """
        try:
            if not self.sqlite_mcp:
                return "SQLite MCP 未初始化，无法查询数据库"

            logger.info("[数据库] 列出所有表")
            tables = self.sqlite_mcp.list_tables()

            if not tables:
                return "数据库中没有找到任何表"

            formatted = "【数据库表列表】\n\n"
            for i, table in enumerate(tables, 1):
                formatted += f"{i}. {table}\n"
            formatted += f"\n共 {len(tables)} 张表"

            logger.info(f"[数据库] 找到 {len(tables)} 张表")
            return formatted

        except Exception as e:
            logger.error(f"列出数据库表失败: {e}")
            return f"列出数据库表失败: {str(e)}"

    def describe_database_table(self, table_name: str) -> str:
        """
        查看指定表的结构

        Args:
            table_name: 表名

        Returns:
            格式化的列信息字符串
        """
        try:
            if not self.sqlite_mcp:
                return "SQLite MCP 未初始化，无法查询数据库"

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
                    # PRAGMA table_info 返回: cid, name, type, notnull, dflt_value, pk
                    col_name = row[1] if len(row) > 1 else "?"
                    col_type = row[2] if len(row) > 2 else "?"
                    is_pk = "是" if (len(row) > 5 and row[5]) else "否"
                    formatted += f"{col_name} | {col_type} | {is_pk}\n"

                logger.info(f"[数据库] 表 {table_name} 有 {len(data)} 列")
                return formatted
            else:
                error = result.get("error", "查询失败")
                logger.error(f"[数据库] 查看表结构失败: {error}")
                return f"查看表结构失败: {error}"

        except Exception as e:
            logger.error(f"查看表结构失败: {e}")
            return f"查看表结构失败: {str(e)}"


