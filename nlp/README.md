# 基于LangGraph的多Agent RAG编排系统

## 项目概述

基于LangGraph构建的多Agent RAG系统，以StateGraph状态机驱动，实现从问题分类、任务规划、并行执行到质量闭环的全链路自动化编排，通过FastAPI暴露SSE流式接口，支持节点级实时进度推送、异步执行与人工介入，覆盖知识检索、数据库查询、多格式文档处理、智能客服、图表智能解读等企业场景。

---

## 技术亮点

### 一、LangGraph全局编排

- **难度分类节点：** LLM对用户问题进行简单/复杂二分类，复杂问题进入DAG规划
- **DAG规划节点：** LLM根据问题语义生成含依赖关系的任务有向无环图（DAG），每个任务声明输入输出内容与上下游参数传递规则
- **Agent并行执行节点：** 基于LangGraph的Send机制，按照任务依赖情况实现DAG波次调度
- **Agent通信质量保障节点：** 检测上游输出是否满足下游所需字段与类型、检测Agent重跑结果是否重复、检测不同Agent输出结果是否不一致
- **结果汇总节点：** 单Agent结果直接输出，多Agent场景由LLM综合各子任务结果生成融合答案
- **质量评估节点：** LLM从相关性、完整性、准确性三维度打分，生成针对性反馈，注入改进要求后路由回Agent重做

### 二、LangChain框架集成

- **工具体系：** 基于LangChain框架实现工具统一注册，生成Function Calling格式的描述供LLM调用，Pydantic对LLM输出的工具调用参数校验，最后统一执行返回结果
- **输出解析：** 任务规划和Agent路由中，通过Few-Shot引导LLM输出标准JSON格式，再通过PydanticOutputParser解析并验证字段，确保输出符合预期结构
- **对话记忆：** 基于ConversationSummaryBufferMemory实现跨Agent共享记忆层，早期对话LLM自动摘要压缩（超过2000tokens触发），Agent执行前读取历史完成指代消解
- **MCP协议集成：** 通过stdio协议接入SQLite，数据库查询Agent自主探索表结构（有哪些表→表有什么字段→数据长什么样），生成SQL执行查询，结果回注上下文生成答案

### 三、单Agent ReAct自主决策

- **Think-Act-Observe循环：** LLM综合系统指令、对话历史和用户问题后，自主判断需要调的工具；多个工具并行执行，结果反馈给LLM，驱动下一轮思考与决策
- **渐进式收敛策略：** 第4轮追加"最后工具调用机会"提示词引导LLM收敛，第5轮强制输出最终答案；同时记录每轮已调用过的工具和参数，重复调用则跳过，避免浪费Token
- **20+注册工具：** 覆盖查询扩展、查询分解、并行检索、多文档合并去重、元数据过滤检索、提取结构化信息等能力
- **三层缓存：** 业务层（查询→答案）、检索层（查询→文档缓存）、LLM层（messages→响应），命中缓存时直接返回，支持Redis持久化

### 四、文档处理及其他

- **多格式文档处理：** PDF/Word/Excel/CSV/TXT/Markdown等格式统一解析，跨页表格识别与合并，异常页面自动切换OCR；基于角色的文档权限过滤，增量索引自动检测文档变更，避免全量重建
- **语义分块：** 多级分割策略（段落→句子→标点→字符逐级回退），控制块大小同时保留语义完整性
- **双路召回融合：** ChromaDB语义检索 + BM25关键词检索，加权RRF算法（k=60, 向量权重0.8）融合排序
- **三层阈值过滤：** 向量检索余弦相似度 < 0.45 过滤 → BM25分数 < 0.5 或关键词命中率 < 20% 过滤 → RRF融合分 < 0.005 过滤，源文件级去重（同源保留最高分chunk）
- **检索统计可视化：** 前端实时展示每次检索的粗排数量、过滤数量、通过数量，每个文档卡片显示三种原始分数（向量/BM25/RRF）

---

## 项目结构

```
RAG_Project/
├── main.py                                     # 系统主入口，CLI交互模式
├── api.py                                      # FastAPI Web服务入口（SSE流式/非流式聊天接口）
├── requirements.txt                            # Python依赖清单
├── .env.example                                # 环境变量配置模板
├── .gitignore                                  # Git忽略规则
│
├── config/                                     # ========== 配置文件 ==========
│   ├── agents.yaml                             # Agent注册配置（ID、能力、优化开关）
│   ├── models.yaml                             # LLM模型配置（Qwen-Max/Plus参数）
│   └── skills.yaml                             # 工具技能配置
│
├── langgraph_orchestrator/                     # ========== LangGraph编排核心 ==========
│   ├── enhanced_entry.py                       # 系统初始化入口（LLM/Redis/Agent统一创建）
│   ├── enhanced_graph.py                       # StateGraph构建（全部节点注册与边连接）
│   ├── enhanced_state.py                       # 全局状态定义（TypedDict，30+字段）
│   ├── enhanced_nodes.py                       # 增强节点（重复检测、人工介入）
│   ├── nodes.py                                # 核心节点（Agent执行、评估、聚合）
│   ├── router.py                               # 路由与调度（复杂度分类、DAG波次调度）
│   ├── clarification.py                        # 参数校验与对齐（上下游字段类型检查）
│   └── critic.py                               # Critic一致性验证（跨Agent输出冲突检测）
│
├── orchestrator/                               # ========== 编排组件 ==========
│   ├── planner.py                              # 任务规划器（LLM生成DAG任务图）
│   ├── router.py                               # 智能路由器（向量相似度+LLM精排选Agent）
│   ├── registry.py                             # Agent注册表（统一发现与管理）
│   └── parameter_aligner.py                    # 参数对齐器（跨Agent字段映射与填槽）
│
├── agents/                                     # ========== Agent实现 ==========
│   ├── knowledge_agent/                        # 知识检索Agent
│   │   ├── agent.py                            #   ReAct循环、20+工具、混合检索、缓存
│   │   └── openapi.yaml                        #   工具接口描述
│   ├── database_agent/                         # 数据库查询Agent
│   │   └── agent.py                            #   MCP协议接入SQLite，自主探索表结构
│   ├── document_agent/                         # 文档处理Agent
│   │   ├── agent.py                            #   多格式文档解析（PDF/Word/Excel/CSV/TXT/MD）、智能分析
│   │   └── openapi.yaml                        #   工具接口描述
│   ├── customer_service_agent/                 # 智能客服Agent
│   │   └── agent.py                            #   情感分析、工单管理、知识库检索
│   ├── vqa_agent/                              # 视觉问答Agent（VQA）
│   │   ├── __init__.py                         #   模块入口
│   │   └── agent.py                            #   千问VL多模态：图表分析、多图对比、OCR
│   ├── chat_agent/                             # 闲聊兜底Agent
│   │   └── agent.py                            #   处理问候、闲聊、意图不明确的查询
│   └── critic_agent/                           # 评论家Agent
│       └── __init__.py                         #   跨Agent输出一致性审计
│
├── rag_core/                                   # ========== RAG检索核心 ==========
│   ├── hybrid_retriever.py                     # 混合检索器（向量+BM25双路召回，RRF融合）
│   ├── reranker.py                             # 重排序器（DashScope API/本地CrossEncoder/规则）
│   ├── chroma_store.py                         # ChromaDB向量数据库（存储、检索、元数据过滤）
│   ├── bm25_retriever.py                       # BM25关键词检索（jieba中文分词）
│   ├── retriever.py                            # 语义检索器（Sentence-BERT编码、查询扩展）
│   ├── query_optimizer.py                      # 查询优化器（扩展、分解、HyDE）
│   ├── cache_manager.py                        # 缓存管理器（三层缓存统一管理，Redis持久化）
│   ├── monitoring.py                           # 检索性能监控（延迟、命中率统计）
│   ├── api_embedder.py                         # API向量化器（DashScope Embedding接口）
│   ├── answer_generator.py                     # 答案生成器（基于检索结果生成回答）
│   ├── context_compressor.py                   # 上下文压缩（去除冗余信息）
│   ├── concept_extractor.py                    # 概念提取器（关键概念识别）
│   └── evaluator.py                            # 检索质量评估
│
├── llm/                                        # ========== LLM调用层 ==========
│   ├── llm_client.py                           # 统一LLM接口（Qwen-Max/Plus，Function Calling）
│   ├── embedder.py                             # Embedding统一封装
│   ├── function_calling.py                     # Function Calling格式转换
│   ├── output_parser.py                        # LLM输出解析（JSON提取、容错处理）
│   ├── langchain_parser.py                     # LangChain PydanticOutputParser集成
│   ├── langchain_tools.py                      # LangChain工具注册与Schema生成
│   └── mcp_client.py                           # MCP协议客户端（stdio方式接入SQLite）
│
├── core/                                       # ========== 核心基础设施 ==========
│   ├── memory.py                               # 对话记忆（ConversationSummaryBufferMemory）
│   ├── protocol.py                             # Agent通信协议（Pydantic强类型约束）
│   ├── unified_cache.py                        # 统一缓存基类（LRU+TTL，RetrievalCache等）
│   ├── cache_manager.py                        # 业务缓存管理（LLM/检索/供应商三级缓存）
│   ├── persistent_cache.py                     # Redis持久化缓存（内存+Redis双层读写）
│   ├── semantic_cache.py                       # 语义缓存（余弦相似度匹配，Redis持久化）
│   ├── unified_monitoring.py                   # 统一监控（性能追踪、成本统计、日志）
│   ├── monitoring.py                           # 全局监控实例（性能指标采集）
│   ├── qa_logger.py                            # 问答日志（SQLite持久化，质量追踪）
│   ├── error_handler.py                        # 全局错误处理与恢复策略
│   ├── document_filter.py                      # 文档权限过滤（基于角色的访问控制）
│   ├── sqlite_mcp_service.py                   # SQLite MCP服务（数据库Agent后端）
│   ├── crm_mock.py                             # CRM模拟数据（客服Agent测试用）
│   ├── filesystem_service.py                   # 文件系统服务
│   ├── file_parser.py                          # 统一文档解析（PDF/Word/Excel/CSV/TXT/MD，供document_agent与init共用）
│   └── industry_standards.py                   # 行业标准数据（合同审核参考）
│
├── data/                                       # ========== 数据文件 ==========
│   ├── knowledge/                              # 知识库文档（12个, ALLOWED_DOC_IDS 白名单管理）
│   │   ├── rag_optimization_strategies.md      #   RAG核心优化策略
│   │   ├── hybrid_search_deep_dive.md          #   混合检索深度解析
│   │   ├── retrieval_methods_comparison.md     #   检索方法对比与选型
│   │   ├── rag_embedding_and_vectordb.md       #   向量嵌入与向量数据库
│   │   ├── rag_introduction.pdf                #   RAG系统介绍
│   │   ├── rag_evaluation_and_monitoring.md    #   RAG评估与监控体系
│   │   ├── semantic_cache_and_performance.md   #   语义缓存与性能优化
│   │   ├── document_chunking_best_practices.md #   文档分块与数据预处理
│   │   ├── llm_prompt_engineering.md           #   LLM提示工程
│   │   ├── customer_satisfaction_guide.md      #   客户满意度提升策略
│   │   ├── customer_retention_best_practices.md#   客户留存最佳实践
│   │   └── customer_data_analytics.md          #   客户数据分析方法论
│   ├── contracts/                              # 合同文档（审核Agent测试数据）
│   │   ├── 待审核合同_2026_00x_*.txt           #   待审核合同（高/中/低风险）
│   │   └── 历史合同_202x_00x.txt               #   历史合同（对比参考）
│   ├── crm/                                    # CRM数据（客服Agent测试数据）
│   │   ├── tickets.json                        #   工单数据
│   │   └── users.json                          #   用户数据
│   ├── uploads/                                # 用户上传文件（文档/图片，UUID前缀防冲突）
│   ├── charts/                                 # 生成的图表图片（VQA Agent输出）
│   └── metadata/                               # 文档元数据
│       ├── document_metadata.json              #   元数据配置
│       └── document_metadata_template.json     #   元数据模板
│
├── database/                                   # ========== 数据库 ==========
│   ├── chinook.db                              # SQLite示例数据库（数据库Agent）
│   └── qa_logs.db                              # 问答日志数据库（质量追踪、统计分析）
│
├── vector_db/                                  # ========== 向量存储 ==========
│   └── chroma_db/                              # ChromaDB持久化数据
│
├── tools/                                      # ========== 运维脚本 ==========
│   └── scripts/
│       ├── init_vector_db.py                   # 向量数据库初始化（全量重建, 白名单过滤）
│       ├── incremental_update.py               # 向量数据库增量更新（MD5哈希检测变更）
│       ├── clean_vector_db.py                  # 向量数据库清理
│       ├── diagnose_db_agent.py                # 数据库Agent诊断工具
│       ├── warmup_cache.py                     # 缓存预热脚本
│       └── init_metadata.py                    # 文档元数据自动生成
│
├── ocr_tools/                                  # ========== OCR工具 ==========
│   ├── tesseract.exe                           # Tesseract OCR引擎
│   ├── ocr_config.py                           # OCR配置
│   ├── tessdata/                               # OCR语言模型
│   ├── scripts/                                # OCR测试脚本
│   └── docs/                                   # OCR使用文档
│
├── frontend/                                   # ========== 前端界面 ==========
│   ├── index.html                              # Web聊天界面（SSE流式、文件上传、图表展示）
│   └── monitor.html                            # 系统监控面板
│
├── tests/                                      # ========== 测试 ==========
│   └── test_system_prelaunch.py                # 上线前系统测试（25个用例，覆盖全Agent）
│
└── docs/                                       # ========== 项目文档 ==========
    ├── README.md                               # 项目说明
    ├── 项目核心架构图.md                         # 完整架构图
    ├── 项目核心架构图_精简.md                     # 精简版架构图
    ├── 整体代码架构清单.md                       # 代码结构说明
    ├── 知识检索Agent执行流程.md                   # KnowledgeAgent流程详解
    ├── 知识检索Agent执行流程_架构图.md             # KnowledgeAgent架构图
    ├── 子Agent及对应工具.md                      # Agent工具清单
    ├── LangGraph相关概念.md                     # LangGraph学习笔记
    ├── LangChain相关概念.md                     # LangChain学习笔记
    └── 异步执行_流式输出_调用工具容错.md           # 异步与容错机制说明
```

---

## 快速开始

### 环境准备

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，填入你的 API Key：
#   DASHSCOPE_API_KEY=sk-your-key    （必填，通义千问 + 千问VL）
#   REDIS_URL=redis://localhost:6379/0  （可选，缓存持久化）

# 3. 初始化向量数据库
python tools/scripts/init_vector_db.py
```

### 启动服务

```bash
# 方式一：CLI交互模式
python main.py

# 方式二：Web API服务
python api.py
# 访问 http://localhost:8000/docs 查看接口文档
```

### API接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | Web 聊天界面（前端页面） |
| `/chat/stream` | POST | 流式聊天（SSE实时推送节点进度） |
| `/chat` | POST | 非流式聊天（等待完成后返回） |
| `/upload` | POST | 上传单个文件（PDF/Word/Excel/CSV/图片/文档） |
| `/upload/multiple` | POST | 批量上传文件 |
| `/resume` | POST | 人工介入后恢复执行 |
| `/agents` | GET | 查询已注册Agent列表 |
| `/health` | GET | 健康检查（含LLM + Redis状态） |
| `/static/charts/*` | GET | 图表图片静态文件服务 |

---

## 系统架构流程图

### 1. LangGraph 主编排流程

```
                                 START
                                   │
                                   ▼
                     ┌───────────────────────┐
                     │ complexity_classifier │  ← LLM 判断复杂度
                     └───────────┬───────────┘
                                 │
                       route_by_complexity
                        ╱                ╲
                   simple              complex
                     ╱                      ╲
                    ▼                        ▼
           ┌──────────────┐        ┌─────────────────┐
           │    router    │        │    planner      │  ← 生成 DAG 任务计划
           │ 选择最佳Agent│        │ {tasks:详情如下}  │
           └──────┬───────┘        └────────┬────────┘
                  │                         │
            route_to_agent          fan_out_dag_tasks
            (单个Agent)            (只send没有 depends_on 的任务)
                  │                         │
                  │                ┌────────┴────────┐
                  ▼                ▼                 ▼
                  └───────┬───────┘                  │
                          │                          │
              ┌───────────▼──────────────────────────▼───┐
              │        跨Agent共享记忆层（指代消解）         │
      ┌──────►│  LLM 判断是否有指代词，有则替换后执行        │
      │       └──────────────────┬───────────────────────┘
      │                     ┌────┴────┐
      │                     ▼         ▼
      │               agent_A    agent_B    ← Send 机制实现 DAG 波次调度，并行执行
      │                     │         │
      │                     │  完成时返回 已完成的任务id
      │                     └────┬────┘
      │                          │  Fan-in合并（列表拼接）
      │                          ▼
      │               ┌─────────────────────┐
      │               │ parameter_validator │  ← 参数校验与对齐（动态填槽、字段类型检查）
      │               └──────────┬──────────┘
      │                     ╱         ╲
      │               reexecute     continue
      │                  │              │
      │          upstream_retry         │
      │             → 重跑 agent       │
      │                          ┌─────▼──────────────┐
      │                          │ duplicate_detection │ ← 语义相似度检测重跑结果是否重复
      │                          └─────────┬──────────┘
      │                                    ▼
      │                          ┌─────────────────┐
      │                          │ critic_validation│ ← LLM语义一致性检查（跨Agent输出冲突）
      │                          └────────┬─────────┘
      │                                ╱     ╲
      │                          passed    failed → 触发人工介入
      │                             │
      │                             ▼
      │                 ┌──────────────────────────┐
      │                 │ human_intervention_check │ ← 统一检查人工介入场景
      │                 │ 1.任务规划审核            │
      │                 │ 2.数据库写入操作          │
      │                 │ 3.质量低且重试多次        │
      │                 │ 4.Critic验证失败          │
      │                 └────────────┬─────────────┘
      │                         ╱    │    ╲
      │                   override retry  abort
      │                   (强制通过)(重跑) (终止)
      │                       │      │      │
      │                       │  循环回共享  END
      │                       │   记忆层
      │                       ▼
      │                ┌─────────────┐
      │                │ aggregator  │  ← 汇总（中间波次跳过，最后LLM综合生成答案）
      │                └──────┬──────┘
      │                       │
      │              ┌────────┴────────┐
      │              │ wave_scheduler  │  ← 找就绪任务（依赖全满足且未完成）
      │              └────────┬────────┘
      │                       │
      │          ┌────────────┼────────────┐
      │          │                         │
      │     有就绪任务                无就绪任务
      │     (Send下一波)             (所有任务完成)
      │          │                         │
      └──────────┘                         ▼
       下一波Agent执行              ┌─────────────┐
       （循环回共享记忆层）          │  evaluator  │  ← LLM 三维度打分（相关性/完整性/准确性）
                                   └──────┬──────┘
                                          │
                                    ╱          ╲
                              质量达标        质量不达标
                                │               │
                               END         注入反馈，路由回Agent重做
```

### 2. 单Agent ReAct执行流程（以知识检索Agent为例）

```
                                    START
                                      │
                        用户查询: "RAG 和传统搜索引擎有什么区别？"
                                      │
                                      ▼
                    ╔═════════════════════════════════════════╗
                    ║   第一阶段：入口预处理 (handle)          ║
                    ╚═════════════════════════════════════════╝
                                      │
                                      ▼
                         ┌────────────────────────┐
                         │  ① 指代消解             │
                         │  检查 self.memory      │
                         │  有代词/省略 → 调 LLM  │
                         │  查询完整 → 原样返回    │
                         └───────────┬────────────┘
                                     │
                                     ▼
                         ┌────────────────────────┐
                         │ ② 精确缓存查询          │
                         │ cache_manager.         │
                         │ query_cache.get()      │
                         └───────────┬────────────┘
                                ╱         ╲
                           命中              未命中
                            ╱                  ╲
                           ▼                    ▼
                    ┌──────────┐      ┌────────────────────────┐
                    │ 直接返回  │      │ ③ 语义缓存查询          │
                    │ 缓存结果  │      │ semantic_cache.get()   │
                    │   END    │      │ 余弦相似度 >= 0.95?     │
                    └──────────┘      └───────────┬────────────┘
                                             ╱         ╲
                                        命中              未命中
                                         ╱                  ╲
                                        ▼                    ▼
                                 ┌──────────┐      ┌────────────────────────┐
                                 │ 直接返回  │      │ ④ 记录到对话记忆        │
                                 │ 缓存结果  │      │ memory.add_user_       │
                                 │   END    │      │ message(query)         │
                                 └──────────┘      └───────────┬────────────┘
                                                                │
                                                                ▼
                                                   ┌────────────────────────┐
                                                   │ ⑤ 进入 ReAct 循环      │
                                                   │ _handle_react()        │
                                                   └───────────┬────────────┘
                                                               │
                                                               ▼
                    ╔═════════════════════════════════════════════════════════╗
                    ║   第二阶段：ReAct 循环初始化                              ║
                    ╚═════════════════════════════════════════════════════════╝
                                                               │
                                                               ▼
                                                   ┌────────────────────────┐
                                                   │ 构建 messages          │
                                                   │ [0] system: 决策策略   │
                                                   │ [1] user: 用户查询     │
                                                   └───────────┬────────────┘
                                                               │
                                                               ▼
                                                   ┌────────────────────────┐
                                                   │ 获取工具 schema        │
                                                   │ tool_registry.         │
                                                   │ get_tools_schema()     │
                                                   │ → 14 个工具 JSON       │
                                                   └───────────┬────────────┘
                                                               │
                                                               ▼
                                                   ┌────────────────────────┐
                                                   │ 初始化控制变量          │
                                                   │ max_iterations = 5     │
                                                   │ retrieval_count = 0    │
                                                   │ seen_tool_calls = {}   │
                                                   └───────────┬────────────┘
                                                               │
                                                               ▼
                    ╔═════════════════════════════════════════════════════════╗
                    ║   第三阶段：ReAct 循环执行（最多5轮）                     ║
                    ╚═════════════════════════════════════════════════════════╝
                                                               │
                      ┌────────────────────────────────────────┘
                      │
                      ▼
          ┌───────────────────────────┐
          │ LLM.chat(messages, tools) │  ← LLM 自主推理决策
          │ temperature=0.3           │
          └───────────┬───────────────┘
                      │
                      ▼
          ┌───────────────────────────┐
          │ parse_tool_calls(response)│  ← 解析 LLM 返回的 JSON
          └───────────┬───────────────┘
                      │
                 ╱         ╲
            有 tool_calls   无 tool_calls
               ╱                ╲
              ▼                  ▼
    ┌──────────────────┐   ┌─────────────────────┐
    │_execute_tool_calls│   │ LLM 认为信息充足     │
    └─────────┬────────┘   │ 返回最终答案         │
              │            │ 退出 ReAct 循环      │
              ▼            └──────────┬──────────┘
    ┌──────────────────┐             │
    │  去重检测         │             │
    │ seen_tool_calls  │             │
    └─────────┬────────┘             │
         ╱         ╲                 │
      重复          新调用             │
       ╱              ╲               │
      ▼                ▼              │
 ┌─────────┐   ┌──────────────────┐  │
 │跳过该调用│   │  参数验证         │  │
 └────┬────┘   │_validate_tool_    │  │
      │        │  parameters()     │  │
      │        └─────────┬─────────┘  │
      │             ╱         ╲        │
      │          无效          有效     │
      │           ╱              ╲     │
      │          ▼                ▼    │
      │   ┌─────────────┐  ┌──────────────────┐
      │   │返回错误信息  │  │ tool_registry.   │
      │   │给 LLM       │  │ call_tool()      │
      │   └──────┬──────┘  │ 执行工具          │
      │          │         └─────────┬────────┘
      │          │                   │
      └──────────┴───────────────────┘
                 │
                 ▼
      ┌──────────────────────┐
      │ 结果回填 messages     │
      │ [n] assistant:       │
      │     tool_calls       │
      │ [n+1] tool: 执行结果 │
      └──────────┬───────────┘
                 │
                 ▼
      ┌──────────────────────┐
      │    收敛检查           │
      └──────────┬───────────┘
            ╱    │    ╲
           ╱     │     ╲
    所有调用  倒数第2轮  继续
    均重复      │        │
       │        │        │
       ▼        ▼        │
    ┌─────┐ ┌─────┐     │
    │强制  │ │追加  │     │
    │收敛  │ │提示  │     │
    └──┬──┘ └──┬──┘     │
       │       │        │
       └───┬───┴────────┘
           │
           │ iteration++
           │
           ▼
      ┌──────────────────────┐
      │ iteration < max (5)? │
      └──────────┬───────────┘
            ╱         ╲
          是            否
         ╱               ╲
        │                 ▼
        │        ┌──────────────────┐
        │        │ 强制生成最终答案  │
        │        │ "请基于已有信息   │
        │        │  给出最终答案"    │
        │        └─────────┬────────┘
        │                  │
        └──────────────────┘
                 │
                 ▼
                    ╔═════════════════════════════════════════════════════════╗
                    ║   第四阶段：后处理（回到 handle）                         ║
                    ╚═════════════════════════════════════════════════════════╝
                                                               │
                                                               ▼
                                                   ┌────────────────────────┐
                                                   │ ① 写入精确缓存          │
                                                   │ cache_manager.         │
                                                   │ query_cache.set()      │
                                                   └───────────┬────────────┘
                                                               │
                                                               ▼
                                                   ┌────────────────────────┐
                                                   │ ② 写入语义缓存          │
                                                   │ semantic_cache.set()   │
                                                   │ 存储 embedding + 答案  │
                                                   └───────────┬────────────┘
                                                               │
                                                               ▼
                                                   ┌────────────────────────┐
                                                   │ ③ 记录对话记忆          │
                                                   │ memory.add_assistant_  │
                                                   │ message(result)        │
                                                   └───────────┬────────────┘
                                                               │
                                                               ▼
                                                   ┌────────────────────────┐
                                                   │ ④ return result        │
                                                   └───────────┬────────────┘
                                                               │
                                                               ▼
                    ╔═════════════════════════════════════════════════════════╗
                    ║   第五阶段：回到 LangGraph 外层                           ║
                    ╚═════════════════════════════════════════════════════════╝
                                                               │
                                                               ▼
                                                   ┌────────────────────────┐
                                                   │ make_agent_node        │
                                                   │ 收到 result            │
                                                   └───────────┬────────────┘
                                                               │
                                                               ▼
                                                   ┌────────────────────────┐
                                                   │ 写入 State             │
                                                   │ agent_results: [{      │
                                                   │   agent: "knowledge_   │
                                                   │   agent",              │
                                                   │   result: "...",       │
                                                   │   iteration: 0        │
                                                   │ }]                     │
                                                   └───────────┬────────────┘
                                                               │
                                                               ▼
                                                   ┌────────────────────────┐
                                                   │ 流转到下游节点          │
                                                   │ → parameter_validator  │
                                                   │ → duplicate_detection  │
                                                   │ → critic               │
                                                   │ → aggregator           │
                                                   │ → evaluator            │
                                                   └───────────┬────────────┘
                                                               │
                                                               ▼
                                                             END
```

---

## 技术栈

| 类别 | 技术 |
|------|------|
| 语言 | Python >= 3.11（LangGraph async StreamWriter 依赖 contextvar 传播） |
| 编排框架 | LangGraph (StateGraph + Send API 并行调度) |
| LLM | 通义千问 Qwen-Max / Qwen-Plus (DashScope API，OpenAI 兼容接口) |
| 多模态 | 千问VL Qwen-VL-Max（图表分析、多图对比、OCR） |
| 向量数据库 | ChromaDB |
| 关键词检索 | BM25 + jieba中文分词 |
| 检索融合 | 加权RRF (向量权重0.8) + 三层阈值过滤 |
| 重排序 | DashScope gte-rerank API / Cross-Encoder / 规则兜底 |
| 向量化 | DashScope text-embedding-v2 (1536维) |
| Web框架 | FastAPI + Uvicorn + SSE流式推送 |
| 前端 | 单文件HTML（原生JS + EventSource，暗色主题） |
| 缓存 | 三层缓存（内存LRU + Redis持久化） |
| 数据库协议 | MCP (Model Context Protocol) + SQLite |
| 文档解析 | pdfplumber + python-docx + pandas + Tesseract OCR |
| 网络搜索 | Tavily API（补充知识库盲区） |
| 数据验证 | Pydantic v2（跨Agent通信强类型约束） |
| 质量追踪 | SQLite问答日志（成功率、评分、延迟统计） |

