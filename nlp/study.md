# 不能偷懒

> 这份文档记录 `nlp/` 从「通用多 Agent RAG 系统」改造为「**电商售后助手**」的路线。
> 写作日期：2026-09-16　　当前代码基线：仓库 commit `1ed39f1` 之后的 nlp/ 目录
> 用途：动手前先读这个，避免走弯路；每完成一个阶段回来勾掉。

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
| Python | **3.13**（机器上没有 3.12；3.14 太新，chromadb/langchain 轮子有风险） |

---

## 2. 环境基线

### 已完成
- [x] `nlp/.env` 建好，含真实 `DASHSCOPE_API_KEY`，已被 [.gitignore:5](.gitignore#L5) 覆盖
- [x] `.git/hooks/pre-commit` 防泄漏钩子 —— 拦截 `sk-` / `tvly-` / `AKIA` 三类密钥特征，实测生效
- [x] **key 实测有效**：HTTP 200，返回正常（2026-09-16）

### 待办
- [ ] 建 `nlp/.venv`（Python 3.13）：`py -3.13 -m venv .venv`
- [ ] 装依赖：`.venv/Scripts/python.exe -m pip install -r requirements.txt`
- [ ] 根目录那个空的 `.venv`（Python 3.14.5，0 个包）**不要用**，留着别管

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

### 阶段 0：环境就位
建 venv 装依赖。
**里程碑**：`python -c "import langgraph, chromadb, dashscope, fastapi"` 无报错。

### 阶段 1：合成数据基座
- 写 8–10 篇售后政策文档，替换 `data/knowledge/` 里原有的 12 篇（那些全是 RAG 技术文档，没有一条电商业务内容）
  覆盖：7 天无理由、退换货流程、运费谁承担、物流时效、退款到账时间、质量问题处理、发票、价格保护、超时未发货
- 建订单库 `database/ecommerce.db`，表：`customers` / `orders` / `order_items` / `logistics` / `refunds` / `tickets`
- 改白名单 [init_vector_db.py:91](tools/scripts/init_vector_db.py#L91) 的 `ALLOWED_DOC_IDS`
- 同步 `data/metadata/document_metadata.json`（`domain` 用 `customer_service`）
- 全量重建：`python main.py --init-db`

**里程碑**：检索「7天无理由」能命中政策文档，且返回的 chunk 来自新文档。

### 阶段 2：数据接入层
- 表名中文映射改成电商表（[sqlite_mcp_service.py:23-39](core/sqlite_mcp_service.py#L23-L39)）
- 默认库路径指向 `ecommerce.db`（[sqlite_mcp_service.py:51-53](core/sqlite_mcp_service.py#L51-L53)）
- `database_agent` 的 few-shot 示例从「多少首歌」改成订单查询场景
- `customer_service_agent` 的工单后端从 `MockCRM` 假数据（[crm_mock.py:42-81](core/crm_mock.py#L42-L81)）换成读写 `tickets` 表

**里程碑**：问「订单 SO2024001 现在什么状态」能查对，答案来自真实 SQL 查询而非幻觉。

### 阶段 3：售后 Agent 重写（工作量最大）
- 重写系统提示词（[agent.py:404-458](agents/customer_service_agent/agent.py#L404-L458)）→ 电商售后话术
- 重写情感词表（[agent.py:733-772](agents/customer_service_agent/agent.py#L733-L772)）→ 售后场景负面词：破损/发错/没收到/退款慢/漏发
- 扩工具集，现有 7 个之外补：**查物流轨迹**、**提交退换货申请**、**查退款进度**
- 保留 `analyze_sentiment`，把「情绪激动」接到工单优先级上（现在是死代码，字段没人写）

**里程碑**：「我买的东西坏了要退货」→ 能走完识别意图 → 建工单 → 返回工单号。

### 阶段 4：路由与编排适配
- 重写选择规则（[router.py:229-236](orchestrator/router.py#L229-L236)）→ 售后意图分类
- **修 bug**：[router.py:345](orchestrator/router.py#L345) 的 `"customer_agent"` 是过期 id（真实 id 是 `customer_service_agent`），整条关键词规则失效；同时把「订单」从客服划给 `database_agent`
- 摘掉 `document_agent` 注册（[enhanced_entry.py:938](langgraph_orchestrator/enhanced_entry.py#L938)）—— 它 9 个工具全是为合同审核写的

**里程碑**：5 类典型问题（政策问答 / 查订单 / 查物流 / 投诉 / 闲聊）路由全部正确。

### 阶段 5：前端与文案
- `frontend/index.html` 的标题、欢迎语、示例问题改成售后场景

**里程碑**：浏览器里跑通一轮完整多轮对话（含指代消解："那它什么时候到"）。

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
| 6 | **过期 agent id** | [router.py:345](orchestrator/router.py#L345) 写的是 `customer_agent` | 关键词路由整条失效，静默降级 |

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
