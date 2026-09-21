# 不能偷懒

> 这份文档记录 `nlp/` 从「通用多 Agent RAG 系统」改造为「**电商售后助手**」的路线。
> 写作日期：2026-09-16　　当前代码基线：仓库 commit `1ed39f1` 之后的 nlp/ 目录
> 用途：动手前先读这个，避免走弯路；每完成一个阶段回来勾掉。
>
> **配套文档**：[idea.md](idea.md) —— 记录每一处改动「怎么想的 / 为什么这么改 / 人类该从中学到什么」。
> [术语表.md](术语表.md) —— 英文术语的统一译名与一句话解释，读代码卡住时按词查。
> 本文件管**路线和进度**，idea.md 管**理解**，术语表管**口径**，三者同步更新。

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

### 阶段 3：售后 Agent 重写　✅ 已完成（2026-09-16）

**Agent 本体**

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

**⚠️ 两个已知问题 → 已在阶段 6 修复：**

1. ~~建表语句里的 `FOREIGN KEY` 默认不生效~~ → 已加 `PRAGMA foreign_keys = ON`，
   并补齐 `tickets.order_id`、`refunds.user_id` 两处漏声明
2. ~~Agent 臆造用户ID（填过 `U87654321`）~~ → 现在被外键直接拒绝，返回可读错误

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

**✅ 浏览器实测（后续补完）：** 配好浏览器调试能力后真的点了一遍 ——
首屏渲染、SSE 流式、trace 面板、复杂路径的 DAG 全部正常，
控制台只有 `favicon.ico` 404（无害）。

**🔴 浏览器实测抓到一个 HTTP 测试漏掉的真 bug：**

模型在浏览器里答复「我已为您**立即创建了**售后工单并提交了退货申请」，
但**数据库里什么都没有** —— 它撒谎了，而阶段 3 的防幻觉兜底**没有拦住**。

根因：最初的兜底用的是**固定子串匹配**（`"已为您创建" in answer`），
而模型在中间插了副词「立即」，9 条规则全部落空。已改成允许插入副词的**正则**。

**更值得记的是：为什么原来的回归测试没抓到？**
因为 `test_05` 的测试数据是我**自己手写的**那句话，恰好命中了子串。
**用自己的样例测自己的检测逻辑，是结构性盲区** —— 你的样例只覆盖你想得到的说法。
逃逸的原句现已固化进 `test_05`。

详见 [idea.md](idea.md) 阶段 5 §4.4 / §4.5。

**第二轮浏览器验证（补完）：**

| 验证项 | 结果 |
|---|---|
| 审批弹窗渲染 | ✅ 摘要显示为一行人话，折叠区正常 |
| 点「批准继续」→ 恢复 | ✅ 答复引用同一个真实工单号 TK20260917001 |
| 文件上传 UI | ✅ 预览显示文件名/大小/缩略图 |
| VQA 识图 | ✅ 正确读出图中内容并判定为功能性缺陷 |
| 多模态 DAG | ✅ planner 拆成 vqa → 售后处理两个任务 |

**至此界面渲染 / SSE 流式 / trace 面板 / 审批弹窗交互 / 上传流程全部实测通过。**

**🟡 一个未复现的偶发问题（已做防御）：**

图片那次查询，服务端正常完成（Evaluator 评分 0.87），但浏览器端没有渲染出答复。
- 排查中我曾误判为「DAG 的 task_2 从未执行」，**打桩后证明是错的**，调度完全正常（§4.7）
- 静态排查发现前端存在"静默失败"组合：`removeElement` 在渲染之前、
  异常只 `console.warn` 不上界面、`appendBotMessage` 假定入参是字符串 ——
  合起来正好产生"进度条消失 + 无答复 + 无报错"。**已修三处**（§4.8）
- 充值后重跑同一场景，**故障未复现**，答复正常渲染、控制台无任何输出、
  两波 DAG 都执行了（§4.9）。**根因无法确认**，最可能是某个一次性异常被静默吞掉了。

**现在的状态：故障本身没解决，但它的"不可诊断性"解决了** ——
下次再发生会在界面上显示一条带事件类型的错误，而不是什么都没有。

**⚠️ 环境提醒：DashScope 账户已欠费**（返回 `Arrearage` 错误码），
所有 LLM / Embedding 调用会 400 失败。**充值前无法跑任何依赖模型的测试。**

**两个踩到的坑**（详见 [idea.md](idea.md)）：
1. curl 内联传中文 JSON 会解析失败 → 改用 `--data-binary @文件`
2. 改完 `api.py` 后不重启，服务仍跑旧代码（`/health` 返回旧服务名）；
   另外 `taskkill` 父进程不会杀掉 uvicorn 的**子进程**，端口会一直被占着

### 阶段 6：回归验证　✅ 已完成（2026-09-16）

- [x] 新建 [tests/test_after_sales.py](tests/test_after_sales.py) —— 15 条用例，两段式（纯逻辑 / 端到端）
- [x] 移除 `tests/test_system_prelaunch.py`（25 条用例全指向旧系统，已失效 —— 详见下方）
- [x] **修已知问题 1**：SQLite 外键不生效 → 显式 `PRAGMA foreign_keys = ON`，
      并补齐 `tickets.order_id` 与 `refunds.user_id` 两处漏掉的外键声明
- [x] **修已知问题 2**：Agent 臆造 user_id → 现在被数据库外键直接拒绝并返回可读错误
- [x] `tickets.user_id` 改为可空（匿名咨询），不再写入假用户 `"anonymous"`

**里程碑**：15/15 通过，耗时 168s　✅

```
[第一段] 数据层与纯逻辑（不调 LLM）
  PASS  01 数据库就位          (36 行 / 6 张表)
  PASS  02 外键约束生效        (臆造 user_id / order_id 均被拒绝)
  PASS  03 匿名工单允许        (TK20260916004)
  PASS  04 退货三重校验        (RF20260916004)
  PASS  05 防幻觉兜底          (4 种场景判断正确)
  PASS  06 知识库白名单        (10 篇售后政策)

[第二段] 端到端（调 LLM，较慢）
  PASS  07 系统初始化          PASS  08 Agent 注册集合 (5 个)
  PASS  09 路由分派 (6/6)      PASS  10 政策问答（3C 例外）质量分 0.97
  PASS  11 订单查询            PASS  12 物流查询（JOIN）
  PASS  13 多轮指代消解        PASS  14 写操作闸门
  PASS  15 边界输入

  总计 15 | 通过 15 | 失败 0 | 耗时 168.4s
```

> **测试过程中防幻觉兜底又拦了一次** —— 日志里能看到模型试图写出"已为您提交"
> 但实际只调了 `hybrid_search` 和 `query_order`。**那道护栏不是摆设，它在上线前的测试里就发挥作用了。**

**关于移除旧测试文件：** `test_system_prelaunch.py` 的 25 条用例全部指向改造前的系统
（期望 `document_agent` 存在、问"什么是 RAG"、断言 Chinook 的销售数据），
在新系统下会**全部失败**。保留一个永远红的测试文件只会误导，
所以移除它 —— 新套件覆盖了同样的方面（初始化 / Agent 集合 / 路由 / 多轮 / 边界），
只是换成了售后场景。需要找回可以 `git checkout 1c5cd18 -- nlp/tests/`。

### 阶段 7：文档收尾（顺手补的）

改造完成后发现一处容易被忽略但影响很坏的东西：**`nlp/README.md` 还是改造前那份**
（614 行，描述的是 `RAG_Project` 结构、`document_agent`、Chinook 数据库）。
那是别人 clone 之后**第一个读的文件**，写的是另一个系统。

- [x] 重写 `nlp/README.md` —— 面向售后助手：能力表、快速开始、架构分层、
      5 个 Agent、三个值得说的设计、目录结构、测试、改造记录
- [x] 新增仓库根 `README.md` —— 一句话说清 `nlp/`（活跃）与 `zanshibuyong/`（归档请忽略）
- [x] 顺带在根 README 里说明「根目录那个空的 `.venv` 已弃用，别用」

> **写文档时的数量全部是数出来的，不是估的。**
> 初稿我写 `knowledge_agent` 有 15 个工具，核对后发现实际是 **14 个**，已修正。
> README 里的具体数字最容易被读者当成事实，也最容易因为"凭印象写"而失真。

**⚠️ 这条教训在 2026-09-19 复发了（见 [idea.md](idea.md) 附记）：** 阶段 8 之后
README 的 4 处数字又漂了（工具数 11→12、测试 15→16 两处、study 阶段数 7→9），
已重新按源码数过并修正。**复发的原因和当初一样：报数字时用的是印象而非源码。**

> 顺带厘清一个区别：**README 必须"当前正确"，study.md 允许"当时正确"。**
> study.md 里「15 条用例 / 15-15 通过」（阶段 6）**不修** —— 那是对当时状态的忠实记录，
> 阶段 8 把总数推到 16 是后来发生的事，两处并存才是准确的历史。
> 分不清这个，就会把"日志"当"现状"去改，反而破坏记录的可信度。

### 阶段 8：知识图谱　✅ 已完成（2026-09-17）

把「商品 → 类别 → 政策适用性」显式存成关系，让「能不能退」从**语义猜测**变成**沿关系推导**。

- [x] 建库 + 种子：[init_knowledge_graph.py](tools/scripts/init_knowledge_graph.py)
      （4 表 + 1 视图；类别层级、政策节点、条件词表、商品锚点）
- [x] 抽取：[extract_graph_relations.py](tools/scripts/extract_graph_relations.py)
      （LLM 抽取 + 纯代码规则校验）
- [x] 核对：[review_graph_relations.py](tools/scripts/review_graph_relations.py)
      —— **16 批准 / 15 驳回 / 1 修正 / 2 补充**，决定记录在 [graph_review_decisions.json](tools/scripts/graph_review_decisions.json)
- [x] 查询层：[core/knowledge_graph.py](core/knowledge_graph.py)
      （递归 CTE + 三值逻辑 + 政策族覆盖）
- [x] 接入售后 Agent：新增第 12 个工具 `query_policy_applicability`，提示词引导优先用图谱拿依据
- [x] 回归测试：新增 test_06（三值逻辑 / 同族覆盖 / 确定性 / 未登记锚点必须报错）

**里程碑**：回归 **16/16** 通过。

```
耳机（3C）+ 已激活 + 质量问题：
  【不适用】七天无理由退货（依据：「3C数码产品」）
  【适用】  质量问题退货（15日）、质量问题换货（30日）、质量问题退货运费由平台承担
```

Agent 答复里出现了可解释的依据：「该商品属于「3C数码产品」，已激活/拆封后不适用「七天无理由退货」」。

**★ 这一阶段挖出六个问题，详见 [idea.md](idea.md) 阶段 8 §3：**

1. 递归 CTE 第一次跑就暴露缺一个根类别（**问题靠想不出来，必须跑**）
2. 规则校验拦下 **2 条编造原文**的引文
3. ★★ **引文是真的，但引文不支持结论** —— 这类错误规则挡不住，只能靠人读（4 条）
4. ★ **词表不够细 → 抽取必然出错**（家电的 7 日节点缺失，LLM 只能选"最接近"的 15 日）
5. ★ **通用政策挂在兄弟类别上 → Agent 顺手编了个数字**（15 日被说成 30 天）
6. 抽取偏向抓"字面词"而非"概念层级"（抽了内衣/泳衣，漏了父类贴身用品）

> **最值得记的一条**：图谱的**覆盖缺口不会表现为"回答缺失"，而会伪装成"回答错误"** ——
> 查不到政策时，模型会自己补一个看起来合理的答案。这让它比"功能没实现"更难发现。

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
| 7 | **`validation_target` 是空字段** | [enhanced_graph.py:206](langgraph_orchestrator/enhanced_graph.py#L206) 读它，但**全项目无人写入**（真正在写的是 [clarification.py](langgraph_orchestrator/clarification.py) 的 `retry_target_task`） | 参数校验失败要重试时，**恒定回跑 `agent_ids[0]`**，而不是真正缺字段的那个上游 Agent —— 未修，仅在代码里如实注释 |
| 8 | **critic 节点整个是空跑的** | `critic_agent` 类存在（[agents/critic_agent/](agents/critic_agent/)），但 `_register_agents` **从未注册它**；[critic.py:40](langgraph_orchestrator/critic.py#L40) 取不到实例即提前判"通过" | `enable_critic` 开不开都一样：**没有任何输出真的被评审过**。另有潜伏问题：真注册上之后，[critic.py:51](langgraph_orchestrator/critic.py#L51) 的 `state.get("plan", {})` 默认值不生效（键存在、值为 `None`），单 Agent 路径会抛 `AttributeError` —— 未修，仅在代码里如实注释 |
| 9 | **两个"参数对齐"字段是空的** | `parameters_aligned` / `parameter_alignment_errors` 只在 [enhanced_entry.py:121-122](langgraph_orchestrator/enhanced_entry.py#L121-L122) 初始化，**全项目无人读、无人写** | 不影响运行；但读 state 字段表时别以为它们接了什么逻辑 |

---

## 6. 代码规范约定（用户明确要求）

1. **生成的代码要方便人类阅读** —— 可读性优先于简洁，宁可多几行也不写技巧性的一行流
2. **自带注释** —— 关键逻辑写中文注释，解释**为什么**这么做，而不是复述代码在做什么
3. **新文件头写用途说明** —— 一小段，说清这个模块负责什么
4. **不用过度抽象** —— 函数职责单一、命名说人话
5. **每次修改后 git 推送** —— 每完成一个可验证的阶段就 commit + push，不要攒着
6. **英文术语在首次出现处标注中文**（2026-09-19 新增）—— 形式：`critic（评审）= ...`，
   **同一文件只标一次**（次次都标会把代码淹掉）；标注前**必须去源码核过语义** ——
   `critic` 看着像"给输出打分"，实际是"校验任务衔接"，猜错的注释比没有注释更糟
   - **译名口径一律以 [术语表.md](术语表.md) 为准**，新词先补进表再使用
   - **只加注释，不改逻辑**；改完用「剥离注释/文档字符串后比对 AST」自证
   - 状态：`nlp/` 下 **85 个 `.py` 文件已全部标注完成**（2026-09-19，净增 3074 行，逻辑零改动）

---

## 7. 明确不做的事

- 不引入新的编排框架（LangGraph 够用）
- 不接真实支付/物流/CRM 的第三方 API —— demo 一律用合成数据
- 不动 `llm/` 调用层和 `rag_core/` 的检索算法
- 不重构 LangGraph 编排骨架

### 7.1 知识图谱（已实施）与用户画像（明确不做）

> 提出于 2026-09-16。**图谱已于阶段 8 实施**；**用户画像决定不做**。
> 决策理由见 [idea.md](idea.md) 「待评估项」一节。

#### 知识图谱 —— ✅ 已实施（阶段 8）

| | |
|---|---|
| **当初的判断** | RAG 尚未表现出失败（阶段 1 六个检索问题、阶段 4 七个路由用例全部命中），**没有失败案例支撑这次扩张** |
| **为什么最终还是做了** | 它要解决的不是"答错"，而是**答不出「不适用」** —— 向量检索只会说"相关"，说不出"排除"，而这恰恰是售后判责最关键的一步。另外它还解决了「讲不出依据」 |
| **验证方式因此变了** | 不是"变好了没有"，而是**推导链是否确定、可解释、可复现** —— 见阶段 8 |
| **实施结果** | 见 §4 阶段 8；过程中挖出 6 个问题，见 §9 |

> **这个决定值得回头看：** 当初说"没有失败案例就别做"是对的，
> 但后来发现**这类功能的价值不在于修错，而在于把软约束换成硬约束** ——
> 判断标准应该多一条：*"即使现在答对了，这个答对是可靠的吗？"*

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

##### 🔴 决定：不做（2026-09-17 明确）

用户明确表示**不做用户画像**。这个决定记在这里，避免以后重复讨论。

**理由（当时分析的结论，依然成立）：**

1. **它是合规问题，不是技术问题** —— 售后系统记录"这个用户常退什么货"，
   属于对用户行为的画像，涉及个人信息保护。做之前要先回答"为什么必须记"，
   而这个问题**现在答不上来**
2. **收益不明确** —— 唯一想清楚的价值点是"审批时给审批人上下文"，
   但那用一次查询就够了，**不需要建画像子系统**
3. **有替代方案** —— `customers` / `tickets` 表里已有的数据，
   用只读查询就能满足当前所有已知需求，**零新增数据采集 = 零新增合规义务**

**如果将来要做，红线先立好：**

| 不记 | 为什么 |
|---|---|
| **情绪历史** | `analyze_sentiment` 每次都算，但**算完即弃**。存下来就是"给用户贴情绪标签" |
| **投诉原文长期留存** | 用户自己的陈述里含大量个人信息 |
| **退货率用于风控** | 来自已有数据、聚合很自然，但**一旦用于对用户做不利决策，性质就变了**（歧视性判定） |

> 现在只把退货次数当"事实"展示（"您历史有 3 次退货"），**不拿它做自动决策**。

#### 一个重要的边界：两者不要捆在一起做

| | 知识图谱 | 用户画像 |
|---|---|---|
| 数据源 | 文档内容 | 用户行为 |
| 更新时机 | 内容变更时（低频） | 每次交互（高频） |
| 风险 | 抽取质量、图谱腐烂 | **隐私合规** |

数据源、更新频率、风险都不同 —— 捆在一起做，两个都做不好。

#### 结论（2026-09-17 更新）

| | 决定 | 状态 |
|---|---|---|
| **知识图谱** | ✅ **做了** | 阶段 8 已完成，回归 16/16 |
| **用户画像** | ❌ **不做** | 明确决定，理由与红线见上 |

**当初定的规则是「先有证据，再有方案」，后来对图谱做了调整 ——**
判据从"有没有失败案例"扩成两条：

1. 有没有失败案例？（修复性架构）
2. **即使现在答对了，这个答对可靠吗、讲得出依据吗？**（约束性架构）

图谱是按**第 2 条**做的。这个补充是对的 —— 因为它最终解决的问题
（答不出"不适用"、讲不出依据）本来就不会表现为"答错"，
**用一个只认失败案例的判据，永远等不到做它的理由。**

画像则两条都不满足：既没有失败案例，收益也不明确，还多一层合规负担。

#### 7.1.1 后续需求：长期记忆系统（2026-09-21）

> 有人提出加一套**长期记忆系统**：「事件总结、实体/关系抽取、多域隔离实现存储结构化沉淀并支撑
> 多场景复用；生成可检索问题、多阶段混合检索排序、检索结果叙事化三段式检索注入；
> 重复归档、相似合并、低重要度+长期未访问遗忘的定期治理」。
>
> **结论：按原形态不做；改为「换记忆对象」做 —— 见下方决定。**

**为什么原形态不做 —— 它不是"用户画像"，是画像的加强版：**

| §7.1 的"用户画像"是 | 这个需求是 |
|---|---|
| 从 `customers` / `tickets` **只读聚合** | **新增采集**：对话 → 总结 → 抽取 → 存储 |
| 零新增数据采集 = **零新增合规义务** | **每轮对话都在沉淀用户信息** |
| — | 还要"治理遗忘"，隐含"这些记忆将来会被用" |

**逐条撞 §7.1 已立的三条红线：**

| 需求做的事 | 撞哪条 |
|---|---|
| **事件总结**（对话总结成事件存下来） | 「**投诉原文长期留存**」—— 总结是用户陈述的派生，范围更难界定 |
| **实体/关系抽取**（从对话抽"张伟-退货-耳机"） | 「**给用户贴标签**」—— 抽出的实体关系是用户行为的**结构化标签**，比画像更结构化 |
| **重要度评分** | 按重要度决定"记住谁"，逆向就是**对用户分级** |
| **定期治理维持检索稳定性** | 隐含"记忆会被反复检索使用" —— 而用在哪儿？ |

**§7.1 的三条理由今天依然成立**：它是合规问题不是技术问题；收益仍不明确（唯一想清的
价值点"审批时给审批人上下文"用一次只读查询就够）；而**零新增采集 = 零新增合规义务**这条
恰恰被这个需求破坏掉。

**技术可行性（不是障碍）：9 个能力里项目已有一半。**

| 完全已有 | 部分已有 | 完全没有 |
|---|---|---|
| 实体/关系抽取的整套链路（抽取脚本 + 规则校验 + **人工核对** + 提案转事实）<br>多阶段混合检索排序（向量 + BM25 + RRF + 重排） | 多域隔离（现仅 `thread_id`）<br>结构化沉淀（`tickets`/`customers`）<br>叙事化注入（现仅注入原始 `history`）<br>重复/相似处理（`duplicate_detection` 的思路，但不持久化） | 事件总结（`ConversationSummaryMemory` 的摘要能力**没实装**）<br>生成可检索问题<br>重要度定义 + 定期治理的**执行机制**（项目无调度器） |

##### ✅ 决定：做，但换记忆对象（用户 → 商品 / 政策）

**理由：技术点一个不少，而"没人被画像"—— §7.1 那条决定的合规理由直接不成立。**

| 模块 | 做法 | 复用什么 |
|---|---|---|
| 事件总结 | 一轮售后会话结束后，总结成「商品 X + 问题类型 Y + 处理结论」 | 新增一个节点 |
| 实体/关系抽取 | 抽 `商品 → 类别 → 常见问题 → 对应政策` | ✅ 整套抽取链（proposal → 规则校验 → 人工核对 → 入库） |
| 多域隔离 | 按**商品类别**分区（不按用户） | 图谱已有类别层级 |
| 多阶段检索 | 向量 + BM25 + RRF | ✅ 直接复用 `hybrid_search` |
| 生成可检索问题 | 给每条记忆反生成"它该被什么问题召回" | 新增 |
| 叙事化三段式注入 | 检索结果组织成叙事再注入 prompt | 新增组装逻辑 |
| 相似合并 / 遗忘 | 相似度用现成 embedding；**治理用脚本**（`tools/scripts/` 已有运维脚本传统，**不必引入调度器**） | ✅ 复用 embedding |

**保留的红线（换了对象也要守）：**

| 不记 | 为什么 |
|---|---|
| 任何**与用户身份绑定**的内容（谁问的、谁退过几次） | 换了对象就不要再滑回去 —— 一旦能关联到人，§7.1 的合规理由立刻重新成立 |
| 情绪历史、投诉原文 | 同 §7.1，与记忆对象无关 |
| 用记忆做**对用户不利的自动决策** | 同 §7.1 |

> **状态：Phase 1 已完成（2026-09-21）。** 与已有知识图谱**同域**（都是商品/政策），
> 所以抽取链、类别层级、混合检索都能直接复用 —— 新增代码量比看起来小很多。

#### 实施进度（分三阶段，每阶段独立可验证）

**Phase 1 · 记忆库 + 离线沉淀管线　✅ 已完成（2026-09-21）**

| 新增 | 作用 |
|---|---|
| [tools/scripts/init_memory_db.py](tools/scripts/init_memory_db.py) | 建 `database/memory.db`：`memory_events` / `issue_types` / `memory_links` + 视图 `v_memory`；灌 11 个问题类型 |
| [tools/scripts/extract_memory_events.py](tools/scripts/extract_memory_events.py) | 读 `qa_logs` → LLM 总结 → **7 条纯代码规则** → 写 `status='proposed'` |
| [tools/scripts/review_memory_events.py](tools/scripts/review_memory_events.py) | `--list`（**连同原始问答一起显示**，供对照）/ `--apply <决策.json>` |
| [core/memory_recall.py](core/memory_recall.py) | 查询层：只读 `v_memory`，按类别过滤（= 多域隔离） |

**两个关键取舍：**

1. **原料用现成的 `qa_logs` 表，不在 `enhanced_entry.py` 加实时钩子** ——
   LLM 总结的开销**不落在用户等待里**（请求路径一行不改）。代价是记忆有延迟，这是刻意的。
2. **加了第 6 条规则：引文不得含身份标识** —— 这是实测逼出来的。
   `qa_logs` 里真的存在「我是张伟 U10001，订单…」这种原话，而规则 3 要求引文忠实于原文，
   **忠实反而会把身份信息带进记忆域**。提示词里已写"不要记录个人信息"，
   但**提示词是软约束、模型可以不听**（本项目反复踩过的坑），所以代码再拦一道。

**Phase 2 · 检索与注入　✅ 已完成（2026-09-21）**

| 新增 / 扩展 | 作用 |
|---|---|
| [tools/scripts/index_memory_events.py](tools/scripts/index_memory_events.py) | 把 `verified` 记忆向量化，写**独立 Chroma collection `memory_events`**（不碰 `knowledge_base`） |
| [core/memory_recall.py](core/memory_recall.py) 扩展 | 两阶段检索：`recall_by_category`（域收窄）→ `recall_similar`（向量 + BM25 → RRF 融合）；`format_narrative` 三段式；`mark_accessed` |
| [agents/customer_service_agent/agent.py](agents/customer_service_agent/agent.py) | 第 13 个工具 `recall_similar_cases` + 提示词【工具清单】 |

**三个关键点：**

1. **`retrieval_questions`（生成可检索问题）是真实起作用的一环** ——
   用户的问法与记忆的叙述写法天然不同（「耳机拆开了还能退吗」vs「已拆封3C数码不支持无理由…」），
   把它一起索引，等于**替用户把话先问了一遍**，显著拉近两种说法的语义距离。

2. **★ RRF 没有"相关性下限"，必须自己补** —— 实测症状：「生鲜坏了怎么办」
   在只有耳机记忆的库里**也返回了耳机那两条**。因为 RRF 只看排名、不看分数，
   排名第 0 那条照样拿到 `1/(k+1)`。修法是融合前**每一路各自过阈值**
   （沿用项目已有的 0.45 / 0.5 两个口径）。**这是端到端跑才暴露出来的。**

3. **叙事第三段（"这是历史经验、不是政策依据"）不能省** ——
   没有它，模型容易把"历史上这么处理的"当成"政策这么规定的"，
   等于让历史案例替代政策判断，而记忆**不参与任何决策**是 §7.1.1 的红线。

**Phase 3 · 治理（相似合并 / 遗忘）　✅ 已完成（2026-09-21）**

| 新增 | 作用 |
|---|---|
| [tools/scripts/govern_memory.py](tools/scripts/govern_memory.py) | 两段式：`--report`（只读出 JSON）/ `--apply <报告.json>`（写库，无 undo） |

**两类动作：**

1. **相似合并** —— 复用索引里已建好的向量（**不重新调模型，零 API 成本**）两两比余弦；
   超阈值记 `memory_links(kind='similar')`，apply 时保留**较早**的那条
   （先入库的是人工先核对过的），另一条 `superseded` + `merged_into`。
2. **遗忘** —— **三条件缺一不可**：`importance < 0.4` **且** `access_count == 0`
   **且** 距上次访问 > 30 天。少任何一条都不忘（每条都能单独造成误杀，见 test_06f）。
   **归档不是删除**（`superseded`），与图谱的 `rejected` 保留不删同一个道理。

**重要度是纯规则算的**（`base 0.3 + 0.3×质量分 + 0.3 真办了事 + 0.1 具体类别`，上限 1.0），
信号全部来自**已有的** `qa_logs` 行 —— 不需要新增任何采集。

**两个如实记录的弱点：**

1. **相似度阈值没有负样本校准** —— 实测"同一案例的两条记忆"相似度 = **0.855**，
   默认阈值就定在 0.85。但**这是唯一一个正样本，缺"两条不同案例"的负样本对照**，
   而 0.855 与 0.85 只差一点点、换个措辞就可能落到线下。所以做成了 `--similarity` 可覆盖。
   项目里另有 0.95 的口径（`enhanced_nodes` 的语义重复），但那是判**同一请求内近乎逐字相同
   的重复答案**，比"同一案例的不同措辞"严得多，不能直接照搬。
2. **Chroma 返回的 embeddings 是 numpy 数组**，`x or []` 会抛
   "truth value of an array is ambiguous" —— 必须显式判 `None`。
   这个坑只在真跑治理脚本时才会踩到。

### 7.2 已知局限：申请原因不受政策校验

> 发现于 2026-09-16，阶段 5 补充验证时。

**现象：** 走退货流程时，`refunds.reason` 被写成了 **「七天无理由退货」**。
但这笔订单的商品是**已激活的无线耳机**，按知识库里的政策属于七天无理由的**例外**，
本不该走无理由通道（应该走质量问题）。

**原因：** `submit_return_request` 的校验只有三条 —— 订单存在、归属正确、状态允许售后。
它**不校验"申请原因与商品类别是否匹配"**，`reason` 是个自由文本字段，模型写什么就存什么。

**为什么不做：** 要做到"校验原因合规"，需要给商品打上**政策适用性标记**
（比如 `supports_seven_day_return: false`），而那意味着一套**政策规则引擎** ——
这是比当前 demo 大一个量级的工程。

**当前的影响范围有限：**
- **对用户可见的回答是对的** —— 助手在正文里正确说明了这是质量问题（15 日内、运费平台承担），
  例外判断在**提示词 + 检索**这一层是生效的
- **写进库的 `reason` 字段不严谨** —— 它只影响数据记录，不影响用户拿到的结论

**如果要做（触发条件）：** 当售后记录需要用于**统计或风控**时（比如"统计无理由退货率"），
`reason` 的准确性就变成硬需求了。届时最小改法是：
- `order_items` 增加政策标记列，`submit_return_request` 按标记校验 `refund_type`
- 或者把原因收敛成**枚举**，从自由文本改成受控值

> 这类问题值得单独记，因为它**不是 bug** —— 代码按设计正常工作，
> 是**设计边界**。区分"实现错了"和"当初就没打算做"，决定了你该修代码还是该改设计。

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


---

## 9. 问题台账

> 改造全程踩过的坑，按**类型**归类（不是按时间）—— 这样能看出"该防哪几类"。
> 详细分析在 [idea.md](idea.md) 对应章节，这里保证**读 study.md 一份就能看全**。

**合计 37 项。** 最值得注意的分布：**LLM 行为类 10 项**（占三分之一），
而且这一类**没有一个能靠"写好提示词"解决**。

### 9.1 🤖 LLM 行为类（10 项）

模型不是"答错"，而是会**编造没发生的事**、**被措辞绕过**、**在词表不全时被迫选错**。

| # | 阶段 | 问题 | 根因 | 修法 |
|---|---|---|---|---|
| 1 | 3 | Agent 描述了完整退货流程，但从未调用写工具，数据库里什么都没有 | LLM 会"演"——描述动作而不执行动作 | 加防幻觉兜底（输出后核验） |
| 2 | 3 | `max_iterations=5` 装不下「5 步工具 + 1 轮回答」，被强制收敛后**编了单号** | 迭代上限是**正确性参数**，不是性能参数 | 上限改 8 |
| 3 | 3 | **Agent 臆造 user_id**（填过 `U87654321`，库里没这人） | 无约束，模型自由生成 | 外键 + 提示词约束 |
| 4 | 5 | 答复说「已为您**立即**创建了工单」，实际没写 | 兜底用**固定子串**匹配，模型插了副词就绕过 | 改成正则（允许插入副词） |
| 5 | 5 | **图谱/工单的"原因"字段写错**（3C 商品的原因写成"七天无理由"） | `reason` 是自由文本，无校验 | 记为设计边界（study §7.2） |
| 6 | 8 | 抽取时 **LLM 编造了 2 条原文引文** | 模型会补一个"看起来像原文"的句子 | 规则校验：引文必须是原文精确子串 |
| 7 | 8 | **引文是真的，但引文不支持结论**（4 条） | 模型会用真引文说假话 | **规则挡不住，只能人工读** |
| 8 | 8 | **词表不完整 → 抽取被迫选错**（家电 7 日 → 抽成 15 日） | 封闭词表防幻觉，但不全时会"被迫出错" | 补节点 + 人工修正 |
| 9 | 8 | 图谱查不到时，**Agent 自己编了个数字**（15 日 → 说成 30 天） | 覆盖缺口伪装成"回答错误" | 通用政策挂根类别 |
| 10 | 8 | 抽取偏向抓**字面词**而非**概念层级**（抽了内衣/泳衣，漏了父类贴身用品） | LLM 抽取的固有倾向 | 人工核对时专看这一类 |

> **这一类的共同点：提示词只能"改善概率"，不能"保证正确"。**
> 关键路径必须靠代码级校验（防幻觉兜底、引文比对、封闭词表），
> 而**语义层面的对错最终只能靠人读**。

### 9.2 🔧 代码 bug 类（10 项）

| # | 阶段 | 问题 | 根因 | 修法 |
|---|---|---|---|---|
| 11 | 2 | `list_tables` 输出的列名全变成 `--` | SQLite 把建表语句**原文含注释**存进 `sqlite_master`，解析器按逗号切分取到了注释符 | 解析前先剥注释 |
| 12 | 3 | **`register_tool` 生成的工具无法接受多参数** | LangChain 的 `Tool` 就是 `SimpleTool`，源码里硬编码只收 1 个参数；**加 args_schema 也无效** | 改用 `StructuredTool.from_function` |
| 13 | 3 | 写操作登记在工具并行执行时丢失，还会**跨请求串号** | 工具跑在**线程池**里，而记录器是线程本地的 | 登记挪到 Agent 的调用线程 |
| 14 | 5 | 前端**静默失败**：进度条消失、无答复、无报错 | `removeElement` 在渲染**之前** + 异常只 `console.warn` + `appendBotMessage` 假定入参是字符串 | 调换顺序 + 异常上界面 + 入参兜底 |
| 15 | 6 | **建表语句里的 `FOREIGN KEY` 不生效** | SQLite 默认 `PRAGMA foreign_keys = OFF`，声明等于注释 | 连接时显式开启 |
| 16 | 6 | `tickets.order_id` / `refunds.user_id` **漏声明外键** | 打开校验时**顺带照出来的** | 补声明 |
| 17 | 6 | 空 `user_id` 被写成字符串 `"anonymous"` | 用假值冒充空值 | 改存 NULL |
| 18 | 8 | `entities` 缺根类别 → 通用政策查不到 | 设计时没想到"没专属政策的商品" | 加根类别「全部商品」 |
| 31 | 阅读 | **参数校验失败后的重试恒回退到 `agent_ids[0]`**（永远是 `knowledge_agent`） | 边函数 `_route_retry_target` 读的是 `validation_target`，而**全项目没有任何地方写入过它**（真正被写入的是 `clarification.py` 的 `retry_target_task`） | 改读 `retry_target_task`（**未修**，先记） |
| 35 | 阅读 | **`task_id` 重复不会被发现 → 任务被静默跳过** | `_validate_plan` 用 `{task["task_id"] for task in tasks}` 收集 id，**set 会静默折叠重复值**，依赖检查也照样通过。而 `wave_scheduler` 靠 `task_id in completed` 判断"已完成" —— 两个任务共用同一个 `task_id` 时，**第二个会被当作已完成静默跳过** | 加一条"`task_id` 必须唯一"的校验（**未修**，成本很低） |

> **#12 潜伏了很久**：项目原有工具**恰好都只传 1 个参数**，
> 所以这个 bug 从没暴露过 —— 但 `create_ticket` 只要同时传 `issue`+`user_id` 就是坏的。
> **"一直能跑"不等于"是对的"。**

### 9.3 🧠 我的判断失误类（3 项）

这三项是**我（辅助者）自己犯的错**，不是代码问题。留下的教训最通用。

| # | 阶段 | 我看到 | 我下的结论 | 真相 | 漏掉的一步 |
|---|---|---|---|---|---|
| 19 | 4 | `customer_agent` 是个过期 id | 「关键词路由整条失效」 | 那个函数是**没人调用的死代码** | **没确认代码是否被执行** |
| 20 | 5 | 日志里只有 `vqa_agent`、没有 `customer_service_agent` | 「DAG 的 task_2 从未执行」 | 打桩后证明**调度完全正常** | **没确认日志是否完整** |
| 21 | 6 | 我写的 test_14 变红了 | 「防幻觉兜底没拦住」 | 兜底**正确工作**了，只是它的实现是把原文**附在**改写后的答复后面，我的检测没识别这种状态 | **没理解被测机制的实现方式** |

> **三次都是"从观察直接跳到结论，跳过了验证那一步"。**
> #19/#20 是同一类：**"没看到" ≠ "没发生"**。
> #21 是另一类：**测试代码也需要被怀疑** —— 它红了不代表被测对象坏了。

### 9.4 📐 设计缺陷类（10 项）

| # | 阶段 | 问题 | 说明 |
|---|---|---|---|
| 22 | 3 | `context["urgent"]` 是一段**从没生效过的死代码** | 全项目没有任何地方给这个 key 赋过值。危险不在"没用"，而在**"看起来有用"** |
| 23 | 3 | test_14 **依赖模型的随机决策** | 模型有时会先要求用户补凭证再提交（合理行为），测试却断言"必须写" → 偶发失败 |
| 24 | 6 | 政策节点**时限编码在名字里**，导致"更具体的覆盖更泛的"需要额外机制 | 引入 `family`（政策族）解决 |
| 25 | 8 | 通用政策挂在**兄弟类别**而非父类别上 | 3C数码拿不到「质量问题退货15日」 |
| 26 | 8 | `route_simple` / `_create_simple_plan` 里的**过期 agent id** | 前者是死代码该删，后者在回退路径上该修 —— **同样是过期 id，处理方式不同** |
| 32 | 阅读 | **指代消解被放在了"最不需要它的那一步"** | 它写在 `make_agent_node` 的 `node_fn` 里（每个 Agent 一份）。但 `complexity_classifier` / `router` / `planner` **全在它之前、全用未消解的 query** —— 而这三步恰恰最需要消解。DAG 路径下更糟：跑 N 次、消解的是 planner 写的**任务描述**（本就自包含）、结果**不写回 state**、还把任务描述当"用户消息"塞进记忆。**simple 路径下它是正确且必需的**（test_13 靠它过），所以测试全绿也看不出来 |
| 33 | 阅读 | **`context` 这个隐式契约里 4 个键对不上** | 只有写没有读：`dependencies`、`original_query`（后者还在契约 docstring 里专门解释过）。只有读没有写：`user_id`（5 个 Agent 在读，节点从未设置 → 恒为 `"anonymous"`）。键名不匹配：`chat_agent` 读 `conversation_history`，节点注入的是 `history` → **恒拿空历史**。见下方 §9.4.1 |
| 34 | 阅读 → ✅**已修** | ★★ **`dependencies` / `parameter_mapping` 声明了"上游结果传给下游"，但没有任何环节真正传递** | 节点构造了 `context["dependencies"]`，但**全项目 0 个 Agent 读它**；三个 Agent 的提示词构造函数**收了 `context` 却不用**（`database_agent` 连参数都没有）；派发时 `Send` 的 `query` 只是 planner **事先**写好的描述，不含上游结果。唯一读取方是 `critic.py`，而 **critic 从未进图**（`enable_critic=False`）。结果：`depends_on` 只实现了"执行顺序"，`parameter_mapping` 声明的那套数据交接**从未接线**。**详见 §9.4.2**<br>**修法（2026-09-21）：** 派发任务时把上游结果拼进 `query`（下游 Agent 唯一必然读到的入口），并给 `agent_results` 补 `task_id` 以便按任务取结果（agent 名在同一 Agent 跑两次时分不清）。`fan_out` 那处是 no-op（root 任务无依赖），但两处共用同一个拼装函数。<br>**仍未解决的：** `parameter_mapping` 的**字段级**取值 —— 见 **#37**：那条路不是"没接线"，而是**整段被守卫挡死、从不执行** |
| 36 | 阅读 | **架构不支持"带着中间结果回到同一个 Agent"** | 「共享状态 + 单向无环 DAG」的固有代价：`wave_scheduler` 只发 `depends_on ⊆ completed` 的任务，而已完成集合只增不减 → **没有单任务回退能力**（参数级有 `upstream_retry`、全局级有质量重试环，**中间这一层缺**）；Agent 之间不互调（契约只有 `handle`），所以"A 检索 → 需要 B 的数据 → 回到 A 回答"这种形态**拆得出来、跑得起来，但语义是错的**（A 的第二次执行看不到 B 的产出）。这是**设计边界**而非缺陷 —— 换来的是"必然终止、不会静默卡死"。同类见 §7.2 |
| 37 | 阅读 | ★ **参数校验 + 上游重试这一整条链路，在当前实现下从不执行** | `parameter_validator_node` 有一道守卫：**「Agent 返回纯文本（非 dict）→ 跳过字段校验」**（[clarification.py:154](langgraph_orchestrator/clarification.py#L154)）。而契约规定 `handle() -> str`，**AST 实测 5 个 Agent 的 `handle` 没有任何一个 return dict**（全是字符串/调用/拼接）→ **守卫永远触发** → `_validate_with_aligner` 与降级用的 `_validate_simple` **都到不了**，`upstream_retry` / `should_retry_validation` / `parameter_retry_count` 全是死机器。<br>**连带影响 #31**：`validation_target` 那个无人写入的字段，正因为**根本没走到那条路**才一直没人发现。两条是一条因果链 |

#### 9.4.1 为什么这类问题专挑 `context` 出现（#33 的根因）

**因为 `context` 是一份"隐式契约"：有文档，没有类型检查。**

| | 有编译器检查吗 |
|---|---|
| `handle(query, context)` 的**签名** | ✅ 有 —— `AgentProtocol` 注册时校验 |
| `context` **里面有哪些键** | ❌ **完全没有** —— 类型就是 `Dict[str, Any]` |

所以：
- 塞什么、取什么**都不会报错**
- `.get("不存在的键", 默认值)` **静默通过**，拿到默认值继续跑
- 契约 docstring（[protocol.py:59-66](core/protocol.py#L59-L66)）是唯一的"文档"，而 **docstring 不会被验证** —— 这就是 #33 里那个键能"**写在契约里却没人读**"的原因

**★ 发现方法（可复用）：对每个键做「谁写 / 谁读」双向核对。**

```bash
grep -rn '"键名"' --include=*.py agents/ core/ langgraph_orchestrator/
```

| 核对结果 | 含义 |
|---|---|
| **写方为零** | **"读了但拿不到"** —— 最隐蔽，因为代码看起来是通的（`user_id`） |
| **读方为零** | **死数据** —— 写了、有文档，但没人用（`dependencies`） |
| 两边都有但**键名不同** | 静默拿默认值（`conversation_history` vs `history`） |

> **通用结论：用 `Dict[str, Any]` 跨模块传数据，等于放弃了编译器帮你查错的能力。**
> 能拦住它的只有人去核对 —— 没有工具、没有类型系统会提醒你。

#### 9.4.2 #34 为什么是这本台账里最严重的一条

> **✅ 已修（2026-09-21）。** 修法：在**派发任务时**把上游结果拼进下游任务的 `query`。
> 为什么选 query 而不是补 `context["dependencies"]` 的读取方 —— **query 是每个 Agent 必然
> 读到的唯一入口**（`messages` 里的 user 消息），改一处所有 Agent 都受益，**零 Agent 改动**；
> 而让 `dependencies` 生效得改 3 个 Agent 的提示词构造函数。这也与 `clarification.py` 的
> `upstream_retry` 保持同一种"改写 query"的做法。
>
> 配套改动：`agent_results` 每条补 `task_id`（`depends_on` 用 task_id，而 agent 名在
> "同一 Agent 跑两次"时分不清是哪个任务 —— 那段场景恰恰需要它）。
>
> 新增回归用例 `test_06c`（第一段纯逻辑，不调 LLM）：root 任务原样返回 / 上游结果注入 /
> 取最新轮次 / 按声明顺序 / 截断 / 依赖缺失容错。
>
> **仍未解决：** `parameter_mapping` 的**字段级**语义 —— 本次交接的是**上游结果原文**
> （带 1500 字截断），不是"命名字段"。追这条时又发现了 **#37**：那条字段级校验链路
> **整段被守卫挡死、从不执行**，所以它不是"差一步"，而是"压根没走过"。


**它是最齐全的那种"看起来在工作"—— 所有该有的痕迹都有：**

| 该有的东西 | 有没有 |
|---|---|
| 字段 | ✅ 节点构造了 `context["dependencies"]` |
| 声明 | ✅ `parameter_mapping` 明写 `"order_detail": "task_1.order_detail"` |
| 校验 | ✅ `parameter_validator` 检查字段够不够，缺了还**会重跑上游** |
| 示例 | ✅ planner 提示词示例一就是串行链（`database → customer_service`） |
| 日志 | ✅ "下游任务 X 校验失败，回退目标 Y" 打得好好的 |
| **把数据送过去** | ❌ **唯一缺的，也是唯一重要的** |

**为什么测试也测不出来：** 项目里多 Agent 的用例（示例二那种）正好是"**两个独立子问题**"的形态 —— 各答各的、最后汇总，**不需要交接也能答对**。而真正需要交接的形态（示例一那种串行链），**恰好没被测试覆盖**。

**四道证据（可复现）：**

```bash
grep -rn "dependencies" --include=*.py agents/     # 预期：无输出
```

| 环节 | 结论 |
|---|---|
| ① 节点构造 | [nodes.py:159](langgraph_orchestrator/nodes.py#L159) 写了 |
| ② Agent 读 | **0 个** |
| ③ 提示词注入 | `customer_service_agent` / `knowledge_agent` 的提示词构造函数**签名收了 `context`、函数体里一次都没用**；`database_agent` 的**连参数都没有** |
| ④ 派发注入 | `Send` 的 `query` 就是 planner **事先**写好的任务描述，不可能包含还没发生的上游结果 |
| ⑤ 唯一读取方 | [critic.py:71](langgraph_orchestrator/critic.py#L71)，而 **critic 从未进图** |

**所以"最终答案看起来完整"是从哪来的？** 靠 `aggregator` 最后把 `agent_results` 全读一遍做 LLM 综合 —— **完整来自汇总，不来自任务间交接。**

> **通用结论（比这条问题本身更值钱）：**
> **"声明、校验、示例、日志全在，但关键动作没做"是最难发现的一类缺陷。**
> 唯一的破法是**追一条数据的完整路径**：谁写 → 谁读 → 读到的内容从哪来。
> 中间断任何一环，整套看起来完整的机制就都是空的。

> **⚠️ 对外的措辞提醒（写简历/答辩时）：**
> 可以说"**按依赖关系分波次调度执行**"；**不能说**"上游结果自动传递给下游""支持链式协作""任务间参数自动对齐" —— 那三样都没接线。
> 被追问时的正确答法：**"各任务独立执行、最后由聚合环节统一综合；任务间的字段声明和校验做了，但传递环节没有实现 —— 这是已知的设计缺口。"**

### 9.5 🌐 环境 / 工具类（4 项）

| # | 阶段 | 问题 | 修法 |
|---|---|---|---|
| 27 | 0/1 | `requirements.txt` 只写下限，实测装到 chromadb 1.5.9 / openai 3.14.1（远新于代码写作年代） | 已验证无 import 层破裂；建议上锁文件 |
| 28 | 5 | `curl` **内联传中文 JSON** 解析失败（Windows shell 转了一层编码） | 改用 `--data-binary @文件` |
| 29 | 5 | 改完 `api.py` 不重启，服务仍跑旧代码（`/health` 返回旧服务名） | 进程已加载旧代码 —— 这个现象反而能**证明服务在跑启动时那份** |
| 30 | 5 | `taskkill` 父进程**不杀子进程**，uvicorn 的 worker 成孤儿继续占端口 | 用 `taskkill /F /T` 杀进程树 |

### 9.6 一项未解决

| 问题 | 状态 |
|---|---|
| 浏览器里图片查询那次**没渲染出答复**（服务端正常完成、评分 0.87、无 API 报错） | **根因未查明**，充值后重跑未复现。但已修掉"静默失败"的隐患 —— 下次再发生会显示**带事件类型的可见错误**，而不是什么都没有 |

> 这是唯一一项**没有闭环**的。如实记在这里。
