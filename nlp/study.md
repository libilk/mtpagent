# 不能偷懒

> 这份文档记录 `nlp/` 从「通用多 Agent RAG 系统」改造为「**电商售后助手**」的路线。
> 写作日期：2026-09-16　　当前代码基线：仓库 commit `1ed39f1` 之后的 nlp/ 目录
> 用途：动手前先读这个，避免走弯路；每完成一个阶段回来勾掉。
>
> **配套文档**：[idea.md](idea.md) —— 记录每一处改动「怎么想的 / 为什么这么改 / 人类该从中学到什么」。
> 本文件管**路线和进度**，idea.md 管**理解**，两者同步更新。

---

## 0. 一句话目标

把它改成一个能**答售后问题**（退换货政策、物流时效、退款规则）、能**查订单**、能**办工单**（提交退换货申请）的电商售后助手。

**关键判断：** 这套系统的骨架（LangGraph 编排、Agent 协议、RAG 检索、缓存/记忆/监控）**完全通用、不用动**。改造量几乎全部在「换业务知识和数据源」上。四个必改点没有一个是改架构。

---

## 1. 已确认的决策（别再回头问第二遍）

| 决策项 | 结论 |
|---|---|
| 改造范围 | **完整改造**为电商售后系统，不只做最小闭环 |
| 业务能力 | **答问题 + 办工单**（写操作，需要保留人工介入环节） |
| 数据来源 | **我合成一套 demo 数据**（政策文档 + 订单 SQLite 库） |
| 模型 | 沿用 **DashScope 通义千问**，key 已验证可用 |
| Python | **3.12.13**（由 uv 管理；3.14 太新，chromadb/langchain 的 wheel 还没跟上） |
| 依赖工具 | **uv** + 清华镜像（`pypi.tuna.tsinghua.edu.cn`），不用 pip |

---

## 2. 环境基线

### 已完成
- [x] `nlp/.env` 建好，含真实 `DASHSCOPE_API_KEY`，已被 [.gitignore:5](.gitignore#L5) 覆盖
- [x] `.git/hooks/pre-commit` 防泄漏钩子 —— 拦截 `sk-` / `tvly-` / `AKIA` 三类密钥特征，实测生效
- [x] **key 实测有效**：HTTP 200，返回正常（2026-09-16）
- [x] `nlp/.venv` 建好，Python **3.12.13**（`uv venv --python 3.12 .venv`）
- [x] 依赖装完，143 个包；项目自身模块导入全部通过
- [x] 根目录那个空的 `.venv`（Python 3.14.5，0 个包）**弃用但保留**，不删

> **为什么用 uv 而不是 pip、为什么是 3.12 而不是更新版本** —— 见 [idea.md](idea.md) 阶段 0 的「为什么这么改」。

### ⚠️ 实测装到的版本，比需求写的新很多

`requirements.txt` 只写了下限（`chromadb>=0.4.0`、`openai>=1.0.0`），实际解析到：

| 包 | 需求写的 | 实际装到 |
|---|---|---|
| chromadb | `>=0.4.0` | **1.5.9** |
| openai | `>=1.0.0` | **3.14.1** |
| pandas | `>=2.0.0` | **3.0.5** |
| langgraph | `>=0.1.0` | 见 `uv pip list` |

两者都有过大版本破坏性变更。**已验证**：项目自身模块（`core.file_parser` / `llm.llm_client` / `rag_core.chroma_store` / `rag_core.hybrid_retriever` / `langgraph_orchestrator.enhanced_entry`）**导入全部正常**，说明没有 import 层的 API 漂移。

但**运行时**的 API 差异还没验证过（比如 Chroma collection 的方法签名）。阶段 1 重建向量库时会第一次真正调用，届时如果报错，优先怀疑是这个原因 —— 处理方式是把这两个包钉到代码写作年代的版本。

> 另注：项目**不依赖 `dashscope` 包**，它用 `openai` SDK 走 DashScope 的 OpenAI 兼容接口。别被 README 的措辞误导。

### 防泄漏三道防线（不要破坏）
1. `.gitignore` 覆盖 `.env`（根目录 + nlp/ 各一份）
2. `.git/hooks/pre-commit` 提交时扫描暂存区（**本地文件，不在版本控制内，删掉就失效**）
3. 手动：只 `git add` 具体文件，永远不用 `git add -A` / `git add .`

---

## 3. 改造前的骨架（现状，改动时对照）

**入口**：`api.py`（FastAPI + SSE）、`main.py`（CLI）

**编排总线** `langgraph_orchestrator/`
```
START → complexity_classifier ─┬─ simple  → router ──→ [单个 Agent]
                               └─ complex → planner → Send 并行发 DAG 根任务
各 Agent 汇合 → parameter_validator → duplicate_detection → aggregator
              → wave_scheduler(还有就绪任务就发下一波) → evaluator → 人工介入检查 → END
```
波次调度没有 `current_wave` 字段，靠 `completed_task_ids ⊇ depends_on` 推算（[nodes.py:191](langgraph_orchestrator/nodes.py#L191)）。

**Agent 层**：接口极简，鸭子类型，只要实现 `handle(query, context) -> str`（[protocol.py:20-54](core/protocol.py#L20-L54)）。
实际注册 6 个（[enhanced_entry.py:888-978](langgraph_orchestrator/enhanced_entry.py#L888-L978)）：
`knowledge_agent` / `database_agent` / `customer_service_agent` / `document_agent` / `vqa_agent` / `chat_agent`

**选 Agent 是两级**：先按 description 向量召回 top-3，再 LLM 精排（[router.py:132-276](orchestrator/router.py#L132-L276)）

---

## 4. 分阶段路线

> 规则：**每个阶段结束 → 验证里程碑 → commit + push**。不要攒着一堆改动一起提交。

### 阶段 0：环境就位　✅ 已完成（2026-09-16）

建 venv 装依赖（uv + Python 3.12 + 清华镜像）。

**里程碑**：`python -c "import langgraph, chromadb, openai, fastapi"` 无报错　✅
补充验证：项目自身 5 个核心模块导入通过，无 API 漂移。
详情见 [idea.md](idea.md) 阶段 0。

### 阶段 1：合成数据基座　✅ 已完成（2026-09-16）

- [x] 写 **10 篇**售后政策文档，替换原有 12 篇 RAG 技术文档
      （7天无理由 / 退换货流程 / 质量问题 / 运费承担 / 退款到账 / 价保 / 物流时效 / 超时未发货 / 会员权益 / 发票）
- [x] 新建 [tools/scripts/init_ecommerce_db.py](tools/scripts/init_ecommerce_db.py) → `database/ecommerce.db`
      6 张表：`customers` / `orders` / `order_items` / `logistics` / `refunds` / `tickets`
- [x] 改白名单 `ALLOWED_DOC_IDS`（[init_vector_db.py:91](tools/scripts/init_vector_db.py#L91)）→ 10 个新 doc_id
- [x] 重写 `data/metadata/document_metadata.json`（10 条，`domain` 全为 `customer_service`）
- [x] 全量重建：`python tools/scripts/init_vector_db.py` → **10 文档 / 34 chunk**
- [x] 修掉脚本里过时的硬编码测试问题（「什么是向量数据库」→ 售后问题）
- [x] `database/ecommerce.db` 加进 `.gitignore`（生成物不入库）

**里程碑**：6 个真实售后问题检索，**top-1 全部命中正确文档**，分数 0.575~0.827，
高于 `MIN_VECTOR_SCORE=0.45` 阈值　✅

**附带结论**：chromadb 1.5.9 全程无 API 报错 —— 阶段 0 记下的版本漂移隐患**已解除**。

> 这里踩到的坑、以及"为什么白名单和删除要同时做"等取舍，见 [idea.md](idea.md) 阶段 1。

**主场景订单**（后续阶段 2/3 都围绕它验证）：

| 字段 | 值 |
|---|---|
| 订单号 | `SO20260909001` |
| 会员 | 张伟（U10001，**金卡**） |
| 商品 | 云听 Pro 主动降噪无线耳机 |
| 状态 | 已签收（2026-09-11 签收，9-16 来投诉即"签收 5 天"） |

> 这个场景有个刻意的设计：**已激活的 3C 数码不适用七天无理由**（知识库里写了这条例外），
> 所以"耳机坏了要退货"必须走**质量问题**通道（15 日、运费平台承担），
> 而不是无理由通道（7 日、运费自付）。这逼着系统必须真的读懂政策，而不是套模板。

### 阶段 2：数据接入层　✅ 已完成（2026-09-16）

- [x] 表名中文映射换成电商 6 张表（[sqlite_mcp_service.py](core/sqlite_mcp_service.py)）
- [x] 默认库路径 `chinook.db` → `ecommerce.db`
- [x] `database_agent` 的 few-shot 换成 2 个订单/物流示例（含 **JOIN** 示范 + 防幻觉约束）
- [x] 新建 [core/ecommerce_crm.py](core/ecommerce_crm.py)：`EcommerceCRM` 读写真实 `tickets` / `customers` 表，
      接口与 `MockCRM` 完全一致，Agent 侧只改一行实例化
- [x] 修掉 `vip_level` → `member_level` 字段名（那是旧 CRM 的字段名）
- [x] **修 bug**：DDL 解析器把中文行注释当成了列名，导致 `list_tables` 输出的列名变成 `--`
      （详见 [idea.md](idea.md) 阶段 2 第 4 节 —— 本阶段最有价值的一条教训）

**里程碑**：数据库 Agent 端到端答对 2 个问题，**答案来自真实 SQL 而非幻觉**　✅

```
问: 订单 SO20260909001 现在什么状态？
答: 状态为「已签收」，下单时间 2026-09-09 14:23。

问: SO20260909001 这个订单的快递到哪了？   ← 必须 JOIN orders 和 logistics 才能答对
答: 顺丰速运承运，运单号 SF1234567890，已于 2026-09-11 15:42 签收。
```

> **遗留在原地的东西**：`core/crm_mock.py` 现在没有任何地方引用了（Agent 已切到
> `EcommerceCRM`）。它没有坏处，但确实是死代码 —— 需要的话可以删掉，连同 `data/crm/*.json`。
> **暂未删除，标记在此以免日后困惑。**

### 阶段 3：售后 Agent 重写　🔄 进行中（2026-09-16）

**Agent 本体　✅ 已完成**

- [x] 重写系统提示词 → 电商售后话术 + **按意图组织的决策树** + 4 个业务陷阱
- [x] 重写情感词表 → 售后场景用词（破损/发错/没收到/退款慢…）
- [x] 删掉 `context["urgent"]` 死代码，改成提示词里的情绪 → 工单优先级映射
- [x] 新增 **4 个**工具：`query_order` / `query_logistics` / `query_refund_status` / `submit_return_request`
      （计划里写的是 3 个，`query_order` 是第 4 个 —— 退货前必须先校验订单归属和状态）
- [x] `EcommerceCRM` 扩 4 个业务方法，`submit_return_request` 带**三重校验**
      （订单存在 / 归属正确 / 状态允许售后）
- [x] `max_iterations` 5 → 8
- [x] **加防幻觉兜底** `_verify_write_claim()`

**三个必须记下来的问题（详见 [idea.md](idea.md) 阶段 3）：**

1. **Agent 会"演"** —— 一口气描述了完整的退货流程，但 `submit_return_request` 从没被调用，
   数据库里什么都没有。**LLM Agent 最危险的失败模式：描述没发生的事。**
2. **迭代次数卡死会诱发编造** —— `max_iterations=5` 装不下 5 轮工具 + 1 轮回答，
   被强制收敛后模型顺着"流程应该走到哪"补完了结果，连单号都是编的。
   **`max_iterations` 是正确性参数，不是性能参数。**
3. **项目级潜伏 bug** —— `register_tool()` 用的 `Tool`（即 LangChain 的 `SimpleTool`）
   **硬编码只接受 1 个参数**，加 `args_schema` 也无效。项目原有工具恰好都只传 1 个参数，
   所以从没暴露；`create_ticket` 其实一直是坏的。已改用 `StructuredTool` 修复。

**里程碑**：连跑 3 次「耳机坏了要退货」，**3/3 成功**，答复中的单号与真实落库单号一致　✅

```
写操作: ['RF20260916004']   答复中的单号: {'RF20260916004'}
写操作: ['RF20260916005']   答复中的单号: {'RF20260916005'}
写操作: ['RF20260916006']   答复中的单号: {'RF20260916006'}
```

**写操作审批闸门　✅ 已完成**

- [x] State 新增 `write_operations` / `write_approved`
- [x] Agent 节点在 `handle()` 后读线程本地的写操作登记，写入 State
- [x] 人工介入节点新增写操作审批分支（带可读摘要 + approved/abort 选项）
- [x] `resume` 支持放行写操作；初始 State 补默认值
- [x] 修既有 bug：`intervention_reason` 为 None 时 `resume` 会 crash
- [x] **踩到并修掉一个条件触发的坑**：工具并行执行时跑在线程池里，
      写操作登记若放在工具内部会丢失信号、甚至跨请求串号。登记已挪到调用线程。

**闸门验证**：

```
handle_query → 闸门拦截  mode=human_intervention
               摘要: ['创建售后工单 TK20260916004，分类 退货，优先级 high']
resume("approved") → success=True，答复引用真实工单号 TK20260916004
```

> ⚠️ 测这个时发现**路由有随机性**：同一句提问连跑 3 次，前两次走复杂路径（规划审批）、
> 第三次才走简单路径触发写操作闸门。**单次通过 ≠ 稳定通过。**

**⚠️ 两个已知问题，留给阶段 6：**

1. 建表语句里的 `FOREIGN KEY` **默认不生效** —— SQLite 需要显式 `PRAGMA foreign_keys = ON`，
   所以现在能往 `tickets` 里塞不存在的 user_id。
2. Agent 实测**臆造过用户ID**（填了 `U87654321`，库里没这人）。已加提示词约束，
   但根治要靠外键约束。

### 阶段 4：路由与编排适配　✅ 已完成（2026-09-16）

- [x] 重写 LLM 选择规则 → 售后意图分类（政策咨询 / 查数据 / 办理 / 图片 / 闲聊）
- [x] 重写 **Agent 注册描述**（[enhanced_entry.py](langgraph_orchestrator/enhanced_entry.py)）——
      向量召回是把 description + capabilities 做 embedding 比相似度，**描述就是路由依据**
- [x] 重写 **planner 提示词**的规则与两个示例（原来讲的是 Chinook 和向量数据库对比）
- [x] 摘掉 `document_agent` 注册（9 个工具全是合同审核专用）
- [x] 修 `planner._create_simple_plan()` 里的过期 id（`code_agent` / `customer_agent`），
      默认兜底从 `knowledge_agent` 改为 `customer_service_agent`
- [x] 更新 `nodes.py` 里给路由用的文件类型提示（原来硬指 document_agent）
- [x] **删除** `router.route_simple()` —— 见下方更正

**里程碑**：7 类问题路由全部正确　✅

```
OK [政策问答] 七天无理由退货的时间怎么计算  -> customer_service_agent
OK [政策问答] 退货运费由谁来承担            -> customer_service_agent
OK [查订单]   订单 SO20260909001 现在什么状态 -> database_agent
OK [查物流]   我的快递到哪了                -> database_agent
OK [投诉]     你们服务太差了，我要投诉！      -> customer_service_agent
OK [闲聊]     你好                          -> chat_agent
OK [通用知识] 什么是向量数据库               -> knowledge_agent
```

注册表最终形态：`knowledge_agent` / `database_agent` / `customer_service_agent` /
`vqa_agent` / `chat_agent`（5 个，document_agent 已摘除，critic 本来就没注册）。

> **⚠️ 更正一条我之前记错的信息：** 我在阶段 4 的准备阶段说过
> 「`router.py` 里 `customer_agent` 是过期 id，导致关键词路由整条失效、静默降级」。
> 实际去查证后发现——**那个函数 `route_simple()` 全项目无人调用，是死代码**，
> 它从来没有参与过路由，所以谈不上"失效"。
> 我当时只看了 id 对不上就下了结论，**没有先确认代码路径是否被执行**。
> 这次已把死函数删除。`_create_simple_plan()` 里的同类问题倒是真在回退路径上，已修。

### 阶段 5：前端与文案　✅ 已完成（HTTP 层面验证）

- [x] `frontend/index.html`：标题 / 顶栏 / 欢迎屏 / placeholder 改为售后场景
- [x] **重写示例问题** —— 按「执行路径」分组，5 个示例分别走 5 条不同路径
- [x] `frontend/monitor.html` 标题同步
- [x] `api.py` 的 title / description / health 服务名同步（对外身份统一）

**里程碑**：跑通一轮完整多轮对话（含指代消解）　✅

| HTTP 请求 | 结果 |
|---|---|
| `POST /chat` 问订单状态 | `mode=simple` → database_agent → 答对 |
| 同一 thread 追问「那它是什么时候签收的？」 | **指代消解成功**（"它"→订单）→ 答对 |
| `POST /chat` 要求退货 | `mode=human_intervention`，摘要含真实单号 `RF20260916005` |
| `POST /resume` 传 approved | `success=True`，答复引用同一单号 |

> **最后一问顺带验证了业务正确性**：答复判定为**质量问题**（15日、运费平台承担），
> 而非七天无理由 —— 阶段 1 埋的 3C 例外设计在真实链路上生效了。

**⚠️ 验证边界（必须知道）：** 以上是 **HTTP 层面**的验证，
**没有在浏览器里点过界面**（当前环境未配置浏览器调试工具）。
**未覆盖**：页面渲染效果、SSE 流式在前端的显示、前端多轮上下文传递、
审批弹窗交互、文件/图片上传流程。**这三件事需要你开浏览器亲自确认一次**（见 [idea.md](idea.md) 阶段 5 §4）。

**两个踩到的坑**（详见 [idea.md](idea.md)）：
1. curl 内联传中文 JSON 会解析失败 → 改用 `--data-binary @文件`
2. 改完 `api.py` 后不重启，服务仍跑旧代码（`/health` 返回旧服务名）；
   另外 `taskkill` 父进程不会杀掉 uvicorn 的**子进程**，端口会一直被占着

### 阶段 6：回归验证
- 参照 `tests/test_system_prelaunch.py` 的用例风格，写一组售后场景用例并跑通

**里程碑**：全部用例通过，作为交付基线。

---

## 5. 已知的坑（动手前必读）

| # | 坑 | 位置 | 后果 |
|---|---|---|---|
| 1 | **`config/agents.yaml` 根本没被消费** | [enhanced_entry.py:637-644](langgraph_orchestrator/enhanced_entry.py#L637-L644) 只读进内存，无人使用 | 改 yaml 不会有任何效果，必须改 `_register_*` 里的字面量 |
| 2 | **`critic_agent` 是死代码** | 类存在但从没注册，且 `enable_critic` 默认 `False`（[enhanced_entry.py:151](langgraph_orchestrator/enhanced_entry.py#L151)） | 图里压根没这个节点，别为它花时间 |
| 3 | **增量脚本不走白名单** | [incremental_update.py](tools/scripts/incremental_update.py) 未做 `ALLOWED_DOC_IDS` 过滤 | 换知识库**必须全量重建**，别用 `--update-db` |
| 4 | **MCP 其实是关的** | [sqlite_mcp_service.py](core/sqlite_mcp_service.py) 的 `skip_mcp` 默认 `True`（npm 包已下架），实际走 sqlite3 降级 | README 说的「MCP 自主探索表结构」并不生效，别按那个预期调 |
| 5 | **角色过滤没接线** | [enhanced_entry.py:705](langgraph_orchestrator/enhanced_entry.py#L705) 把 `doc_filter` 硬编码传 `None` | Agent 端有权限过滤逻辑，但运行时不会触发 |
| 6 | **过期 agent id** | `router.route_simple()`（已删）及 `planner._create_simple_plan()` 里的 `customer_agent` / `code_agent` | 前者是死代码从未执行；**后者在回退路径上，会指向不存在的 Agent** —— 已修 |

---

## 6. 代码规范约定（用户明确要求）

1. **生成的代码要方便人类阅读** —— 可读性优先于简洁，宁可多几行也不写技巧性的一行流
2. **自带注释** —— 关键逻辑写中文注释，解释**为什么**这么做，而不是复述代码在做什么
3. **新文件头写用途说明** —— 一小段，说清这个模块负责什么
4. **不用过度抽象** —— 函数职责单一、命名说人话
5. **每次修改后 git 推送** —— 每完成一个可验证的阶段就 commit + push，不要攒着

---

## 7. 明确不做的事

- 不引入新的编排框架（LangGraph 够用）
- 不接真实支付/物流/CRM 的第三方 API —— demo 一律用合成数据
- 不动 `llm/` 调用层和 `rag_core/` 的检索算法
- 不重构 LangGraph 编排骨架

### 7.1 待评估：知识图谱 + 用户画像

> 提出于 2026-09-16，**结论是先不做**，但要留下判断依据，避免以后重复讨论。
> 决策理由见 [idea.md](idea.md) 「待评估项」一节。

#### 知识图谱（从内容抽实体关系构图）

| | |
|---|---|
| **想法** | 从内容里抽取实体和关系建图，支持「和 X 相关的 Y 有哪些」这类**关系推理**，补足向量检索只能"找相似段落"的短板 |
| **现状** | [rag_core/concept_extractor.py](rag_core/concept_extractor.py) **已经在抽概念和关系**，但它是**纯内存、每次查询现算、不落盘**的 —— 是查询时的一次性分析结果，**不是图谱** |
| **为什么不做** | RAG 尚未表现出失败。阶段 1 的 6 个检索问题、阶段 4 的 7 个路由用例**全部命中**，没有真实失败案例支撑这次扩张 |
| **触发条件** | 满足其一再评估：① 压测中出现一批失败案例，共性是需要**跨文档关系推理**（例如"这个商品属不属于某条例外"）；② 问题量增长到向量召回质量明显下降 |

**如果要做，已知的设计要点**（来自归档的 coach 项目 —— 同类问题的现成经验）：

- 图存储用 **SQLite + 递归 CTE**，不引入图数据库
- 关系类型必须**收敛成有限几类**，不做开放式
- **LLM 抽取的关系不直接入库** —— 走 `observation → proposal → aggregator`。
  LLM 抽出来的东西是**提案，不是事实**，直接入库等于把幻觉当知识沉淀下来

#### 用户画像（记录用户习惯）

| | |
|---|---|
| **现状** | **完全没有**。`ConversationMemory` 只存最近 20 条原始消息，**不区分用户**、重启即失；`qa_logs` 表的键是 `thread_id` **不是 `user_id`**，连"这个用户问过几次"都关联不起来 |
| **轻量做法** | 如果要起步，**不必新建子系统**：`customers` 表已有 `member_level` / `total_orders` / `total_spent`，`tickets` 表有历史工单，从现有数据聚合就能得到最低限度画像 |
| **必须同时考虑** | 售后场景记录用户习惯涉及**个人信息保护**。这是合规问题，不是技术问题，绕不过去 |

#### 一个重要的边界：两者不要捆在一起做

| | 知识图谱 | 用户画像 |
|---|---|---|
| 数据源 | 文档内容 | 用户行为 |
| 更新时机 | 内容变更时（低频） | 每次交互（高频） |
| 风险 | 抽取质量、图谱腐烂 | **隐私合规** |

数据源、更新频率、风险都不同 —— 捆在一起做，两个都做不好。

#### 结论

**两个都先不做。** 当前优先级是把阶段 5/6 做完、跑通闭环，然后**用一批真实售后问题压测，
把答错的案例收集起来**：

- 失败案例里有一批需要跨文档关系推理 → 图谱有依据，值得做
- 一个都没有 → 加了也是白加

**不要因为"这个东西听起来高级"就加。** 架构扩张要有失败案例作为依据 ——
先有证据，再有方案。

---

## 8. 改造后的目标架构（带注释）

**图例**：【原有】不动 ·【改造】接口不变只换内部逻辑 ·【新增】新做的东西

### 8.1 分层总览

```
┌──────────────────────────────────────────────────────────────────────────┐
│ 层 0 · 接入层                                                    【原有】 │
│                                                                          │
│   浏览器 frontend/index.html          ← 阶段 5 只改标题/欢迎语/示例问题   │
│        │                                                                 │
│        │ EventSource（SSE 流式，逐节点推进度）                            │
│        ▼                                                                 │
│   api.py（FastAPI）                                                       │
│    ├ /chat/stream   流式对话，前端 trace 面板靠它      api.py:291         │
│    ├ /chat          非流式                             api.py:354         │
│    ├ /resume        人工介入后恢复执行                 api.py:385         │
│    ├ /upload(.multiple)  上传文件/图片                 api.py:467/547     │
│    └ /health /agents /cache/stats /logs ...                              │
│        │                                                                 │
│        ▼ 以 thread_id 隔离会话                                            │
│   会话状态 = MemorySaver(进程内) + MemoryStore(内存字典)                  │
│   ⚠️ 两者都不持久化，重启即丢；Redis 只缓存、不存会话                     │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 层 1 · 编排层 langgraph_orchestrator/                     【原有·骨架不动】│
│         唯一的改动：节点总数少 1 个（document_agent 被摘掉）              │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 层 2 · Agent 层                                  6 个 → 5 个，有增有减   │
│   knowledge_agent 【原有】零业务知识，原样复用                            │
│   database_agent  【改造】查询目标换成 ecommerce.db                       │
│   after_sales_agent 【改造】由 customer_service_agent 重写而来（核心工作）│
│   vqa_agent       【原有】保留 —— 售后要高品照片/快递单/发票识别          │
│   chat_agent      【原有】兜底，原样保留                                  │
│   document_agent  【删除】9 个工具全为合同审核而写                        │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 层 3 · 能力层                                                             │
│   检索：rag_core/ 混合检索（向量 + BM25 → RRF 融合）      【原有】        │
│   数据：database_agent 内 SQL 生成      【改造】表名映射 + few-shot 示例  │
│   审批：写操作闸门                                         【新增】★     │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 层 4 · 数据层                                                             │
│   data/knowledge/   12 篇技术文档 → 8~10 篇售后政策       【全换】        │
│   database/         chinook.db(音乐店) → ecommerce.db      【全换】        │
│                     customers / orders / order_items /                   │
│                     logistics / refunds / tickets                        │
│   vector_db/        ChromaDB collection "knowledge_base"  【重建】        │
└──────────────────────────────────────────────────────────────────────────┘

横切能力（全部【原有】，不用动）：三层缓存 · 对话记忆 · 性能监控 · 问答日志
```

### 8.2 编排主干（改造后）

```
                              START
                                │
                                ▼
                ┌───────────────────────────────┐
                │     complexity_classifier      │  LLM 二分类：判断是否需要
                │     nodes.py:230               │  「多个不同类型」Agent 协作
                └───────────────┬───────────────┘  解析失败 → 降级为 simple
                                │
                          route_by_complexity
                          ╱                  ╲
                    simple                  complex
                          │                        │
                          ▼                        ▼
            ┌──────────────────────┐   ┌──────────────────────────┐
            │       router         │   │        planner           │
            │  两级选择：           │   │  LLM 生成 DAG 任务图      │
            │  ①向量召回 top-3      │   │  每个 task 声明：         │
            │  ②LLM 精排            │   │   agent_id / depends_on   │
            │  router.py:132-276   │   │   input_schema / 参数映射 │
            └──────────┬───────────┘   └────────────┬─────────────┘
                       │                            │
                 route_to_agent            fan_out_dag_tasks（Send）
                 （单个 Agent）            ★只发 depends_on 为空的任务
                       │                            │
                       └──────────┬─────────────────┘
                                  ▼
                  ┌───────────────────────────────┐
                  │   各 Agent 执行                │  ★Send 载荷带
                  │   （DAG 分波次并行）           │   current_task_id
                  │   完成后回写 completed_task_ids│
                  └───────────────┬───────────────┘
                                  ▼
                  ┌───────────────────────────────┐
                  │    parameter_validator         │  检查上游字段/类型是否
                  │    clarification.py            │  满足下游所需，动态填槽
                  └───────────────┬───────────────┘
                              ╱         ╲
                        reexecute     continue
                              │             │
                    upstream_retry          │   ← 重跑的是上游那个 Agent
                    （重跑上游 Agent）      │
                                            ▼
                  ┌───────────────────────────────┐
                  │    duplicate_detection         │  语义相似度检测
                  │    enhanced_nodes.py:39        │  重跑结果是否与上次重复
                  └───────────────┬───────────────┘
                                  ▼
                  ┌───────────────────────────────┐
                  │        aggregator              │  ★中间波次直接 return {}
                  │        nodes.py:344            │   空转，不打扰下游；
                  └───────────────┬───────────────┘   全部完成后 LLM 综合
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │      wave_scheduler            │  ★就绪判据 =
                  │      router.py:120-169         │   depends_on ⊆ completed
                  └───────────────┬───────────────┘
                            ╱           ╲
                     还有就绪任务     全部完成
                            │             │
                  Send 下一波 │             ▼
                  （回到 Agent）│  ┌───────────────────────────┐
                            └───→│        evaluator           │  LLM 三维打分：
                                 │        nodes.py:447        │  相关性/完整性/准确性
                                 └─────────────┬─────────────┘
                                          ╱         ╲
                                     达标           不达标
                                       │              │
                                       │      注入反馈，路由回 Agent 重做
                                       ▼
                        ╔══════════════════════════════════╗
                        ║   human_intervention_check       ║  统一人工介入闸门
                        ║   enhanced_nodes.py:109          ║  触发条件见 8.4
                        ╚════════════════┬═════════════════╝
                                    ╱     │     ╲
                              need_human  pass  retry
                                  │        │      │
                                  ▼        │      └→ 回到 complexity_classifier
                    ┌──────────────────┐   │         （整条链重跑，注意是重跑不是续跑）
                    │ human_intervene  │   │
                    │ _execute         │   │  ★interrupt_before 在这里挂起
                    │ （/resume 唤醒） │   │
                    └────────┬─────────┘   │
                        ╱         ╲       │
                    override     retry    │
                   (强制通过)   (重跑)     │
                        │         │       │
                        ▼         └───────┤
                      END                 │
                                          ▼
                                        END
```

### 8.3 Agent 层改造明细

| Agent | 状态 | 具体动作 |
|---|---|---|
| `knowledge_agent` | 【原有】 | 15 个工具全是通用检索能力，**零业务硬编码**，接口不动。改的只是它检索的语料（阶段 1 换知识库） |
| `database_agent` | 【改造】 | 换库 + 改表名中文映射（[sqlite_mcp_service.py:23-39](../core/sqlite_mcp_service.py#L23-L39)）+ few-shot 从「多少首歌」改成订单场景。**接口仍是 4 个工具** |
| `after_sales_agent` | 【改造·重点】 | 由 `customer_service_agent` 重写。提示词、情感词表、工具集全换（详见下表） |
| `vqa_agent` | 【原有】 | 千问 VL，售后场景要它认破损商品照/快递单/发票，**保留** |
| `chat_agent` | 【原有】 | 兜底闲聊，原样保留 |
| `document_agent` | 【删除】 | 从 [enhanced_entry.py:938](../langgraph_orchestrator/enhanced_entry.py#L938) 摘掉注册。它的 9 个工具（风险识别/违约金计算/按供应商检索）全是合同审核专用 |
| `critic_agent` | 无需处理 | 本来就是死代码（从未注册 + `enable_critic` 默认 False），**别为它花时间** |

**`after_sales_agent` 的工具集变化：**

```
【沿用】hybrid_search      混合检索售后政策
【沿用】vector_search      语义检索
【沿用】keyword_search     BM25 精确匹配（「7天无理由」这类固定表述）
【沿用】analyze_sentiment  情绪识别 —— 改造后要把「情绪激动」真正接到工单优先级上
【沿用】create_ticket      建工单（后端从 MockCRM 假数据换成 tickets 表）
【沿用】query_ticket       查工单进度
【沿用】query_user_info    查用户信息（改成查会员等级 + 历史订单）

【新增】query_logistics        查物流轨迹（transport → logistics 表）
【新增】submit_return_request  提交退换货申请（★写操作，走审批闸门）
【新增】query_refund_status    查退款进度（→ refunds 表）
```

### 8.4 ★ 新增：写操作审批闸门（必须补的设计）

**为什么必须补：** 现有代码里的「数据库写操作审核」拦不住办工单，两个原因 ——

1. 它要求 `"database" in selected_agent`（[enhanced_nodes.py:127](../langgraph_orchestrator/enhanced_nodes.py#L127)），
   但工单是 `after_sales_agent` 内部建的，压根不满足这个条件
2. 它检查**用户那句中文**里有没有 `INSERT`/`UPDATE`（[enhanced_nodes.py:129](../langgraph_orchestrator/enhanced_nodes.py#L129)），
   而用户说的是「我要退货」，永远不可能命中

另外 `selected_agent` 只在 simple 单 Agent 路径上有值，DAG 路径上大概率是空的。

**改造后设计：**

```
  Agent 执行时，给写类工具打标记
  （submit_return_request / create_ticket / 任何改数据的工具）
              │
              ▼
  agent_results 里出现 write 类工具调用
              │
              ▼
  human_intervention_check 新增一个判断分支
  → human_intervention_required = True
  → intervention_reason = "即将提交退换货申请，等待人工确认"
              │
              ▼
  interrupt_before 挂起 → 前端弹确认 → /resume 唤醒
```

这样「办工单」才真正带审批。**落在阶段 3。**

### 8.5 一次典型请求的完整数据流

用户说：**「我上周买的耳机坏了，想退货」**

```
① complexity_classifier
   → complex（要同时查订单、查政策、建工单，涉及 3 个不同类型 Agent）

② planner 生成 DAG
   ├─ task_1  「查询该用户最近购买的耳机订单」      → database_agent
   ├─ task_2  「质量问题退货运费由谁承担」          → knowledge_agent
   └─ task_3  「创建退换货工单」                   → after_sales_agent
              depends_on: [task_1, task_2]      ← 必须等前两个回填参数

③ 第 1 波：Send 只发 task_1、task_2（依赖都为空），并行执行
   task_1 → SQL 查 ecommerce.db 命中订单 SO20260909001（张伟的耳机，已签收）
   task_2 → 混合检索命中《质量问题退换货政策》第 3 条

④ 汇合 → parameter_validator 检查 task_3 需要的 order_id / policy_basis
   —— task_1/task_2 的输出正好填上，continue

⑤ aggregator 空转（任务没全完成）→ wave_scheduler 发现 task_3 就绪
   → Send 第 2 波：task_3 执行，调 submit_return_request 建工单

⑥ ★ 写操作闸门触发 → human_intervention_check 置 True
   → 图在 human_intervention_execute 前挂起（interrupt_before）

⑦ 前端弹确认框，客服点「通过」→ POST /resume
   → 图继续 → aggregator 汇总 → evaluator 打分（三维）→ END

⑧ 返回用户：「已为您创建退换货工单 TK20260916001，运费由商家承担
   （质量问题），预计 3 个工作日内上门取件」
```

对比一下：**同样一句话在改造前会走成什么样** —— `task_2` 检索不到任何售后政策（知识库里只有 RAG 技术文档），`task_3` 的 `create_ticket` 写进的是 `MockCRM` 的假数据，工单号在下次重启后消失，而且全程**没有任何审批**。

