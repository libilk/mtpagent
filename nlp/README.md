# 云集优选 · 售后助手

> 一个基于 **LangGraph 多 Agent 编排**的电商售后系统。
> 能答售后政策、能查订单物流、能**真的办理退换货**，办理前会停下来请人确认。

不是问答机器人 —— 它带工具、会查数据、会写数据，并且在动数据之前有一道人工审批。

---

## 它能做什么

| 能力 | 用户怎么说 | 系统内部怎么走 |
|---|---|---|
| **售后政策问答** | 「七天无理由怎么算？」 |
 路由 → 售后 Agent → 混合检索政策库 → 依据条款回答 |
| **订单 / 物流查询** | 「我的快递到哪了？」 |
 路由 → 数据 Agent → 生成 SQL → 查订单库 → 回答 |
| **退换货办理** | 「耳机坏了，我要退货」 |
 复杂路径 → DAG（查订单 → 检索政策 → 提交申请）→ **人工审批** |
| **投诉受理** | 「我要投诉！」 | 
情绪分析 → 建工单（优先级按情绪定）→ **人工审批** |
| **图片识别** | 上传破损商品照片 |
 `vqa_agent` 识图 → 售后 Agent 据图判断能否退 |

**业务设定**：虚构平台「云集优选」，10 篇售后政策文档 + 一个含订单/物流/退款/工单的 SQLite 库。

---

## 快速开始

```bash
cd nlp

# 1. 建虚拟环境并装依赖（用 uv，走清华镜像）
uv venv --python 3.12 .venv
uv pip install -r requirements.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 2. 配置 API Key
cp .env.example .env      # 编辑 .env，填入 DASHSCOPE_API_KEY（必填）

# 3. 初始化数据（两个都要跑）
python tools/scripts/init_ecommerce_db.py    # 订单/物流/退款/工单库
python tools/scripts/init_vector_db.py       # 售后政策 → 向量库

# 4. 启动
python api.py     # Web 界面 + API：http://localhost:8000/
python main.py    # 或者用命令行交互
```

> **Windows CMD 用户注意**：路径要用反斜杠（`.venv\Scripts\python.exe`），
> CMD 不把 `/` 当路径分隔符。

---

## 架构

### 分层

```
接入层    api.py（FastAPI + SSE 流式）· frontend/index.html
   ↓
编排层    langgraph_orchestrator/  ← StateGraph 状态机，DAG 波次调度
   ↓
Agent 层  knowledge · database · customer_service · vqa · chat
   ↓
能力层    混合检索（向量+BM25+RRF）· Text-to-SQL · 工单读写
   ↓
数据层    data/knowledge/（10 篇政策）· database/ecommerce.db（6 张表）
```

### 编排主干

```
START → complexity_classifier ─┬─ simple  → router ──→ 单个 Agent
                               └─ complex → planner → Send 并行发 DAG 根任务
各 Agent 汇合 → parameter_validator → duplicate_detection → aggregator
              → wave_scheduler（还有就绪任务就发下一波）
              → evaluator（三维打分）→ 人工介入检查 → END
```

**DAG 是波次调度的**：`fan_out_dag_tasks` 只发依赖为空的根任务，
每波跑完由 `wave_scheduler` 找"依赖全满足且未完成"的任务再发下一波。
就绪判据是 `depends_on ⊆ completed_task_ids`。

### Agent 一览

| Agent | 职责 | 工具数 |
|---|---|---|
| `knowledge_agent` | 与售后无关的通用知识、概念解释 | 14 |
| `database_agent` | 订单/物流/退款/会员数据查询（Text-to-SQL） | 4 |
| `customer_service_agent` | 售后政策问答 + 退换货办理 + 投诉工单 | 12 |
| `vqa_agent` | 破损商品照、快递单、发票截图识别（三种模式，非工具式） | — |
| `chat_agent` | 闲聊兜底（无工具） | — |

> Agent 的接口极简：**鸭子类型**，只要实现 `handle(query, context) -> str`。
> 不要求继承基类，注册时用 `isinstance` 校验。

---

## 三个值得说的设计

### 1. 写操作审批闸门（人机协同）

系统能做写操作（提退货申请、建工单），但**动手之前会停下来等人确认**。

```
Agent 执行写工具 → 登记 → Agent 节点写入 state["write_operations"]
                → 人工介入节点据此挂起（interrupt_before）
                → 前端弹审批框 → 点「批准继续」→ POST /resume → 继续
```

判据是「**数据是不是真的被改了**」，而不是「用户说了什么关键词」——
后者容易被措辞绕过。

### 2. 防幻觉兜底

LLM Agent 最危险的失败模式不是答错，而是**描述没发生的事**。
实测中模型会写出「已为您提交退货申请，单号 RF2026xxx」，而数据库里什么都没有。

所以加了输出后校验：**声称办过事，就必须真调用过写工具**，否则改写答复、
明确告诉用户"并未提交"。

### 3. 检索与查询的分工

「退货政策是什么」→ Agent 检索文档；「我的订单什么状态」→ Agent 写 SQL。
两者都靠 LLM，但**一个查规则、一个查数据**，路由时按这个边界分派。

---

## 目录结构

```
nlp/
├── api.py                          FastAPI 入口（SSE 流式 / 人工恢复 / 上传）
├── main.py                         CLI 入口
├── 术语表.md                       英文术语统一译名（读代码卡住时按词查）
├── langgraph_orchestrator/         编排核心（图、状态、节点、路由）
├── orchestrator/                   planner（DAG 生成）· router（选 Agent）· registry
├── agents/                         5 个 Agent 的实现
├── rag_core/                       混合检索、重排、向量库、查询优化
├── llm/                            模型调用层、工具注册、MCP 客户端
├── core/                           基础设施：缓存/记忆/监控/日志 + CRM + 写操作审计
├── config/                         配置（注意：agents.yaml 当前未被消费，见 study.md）
├── data/knowledge/                 10 篇售后政策文档（向量库语料）
├── data/metadata/                  文档元数据
├── database/                       ecommerce.db（脚本生成，不入库）
├── tools/scripts/                  建库、建向量库、增量更新等运维脚本
├── frontend/                       单文件 HTML（原生 JS + EventSource）
└── tests/                          回归测试套件（16 条）
```

---

## 测试

```bash
python tests/test_after_sales.py
```

16 条用例，分两段跑：

- **第一段（不调 LLM，秒级）** 数据层与纯逻辑：外键约束、退货三重校验、
  防幻觉兜底、知识图谱推导、知识库白名单
- **第二段（调 LLM，约 3 分钟）** 端到端：路由分派、政策问答、订单/物流查询、
  多轮指代消解、写操作登记链路、边界输入

退出码 0 = 全通过，可直接接 CI。

---

## 改造记录

这个项目是从一个**通用多 Agent RAG 系统**改造成电商售后助手的，过程完整记录在两份文档里：

| 文档 | 内容 |
|---|---|
| [study.md](study.md) | **路线图** —— 9 个阶段（0~8）的计划、里程碑、进度，以及待评估项与已知局限 |
| [idea.md](idea.md) | **学习笔记** —— 每一处改动「怎么想的 / 为什么这么改 / 人类该从中学到什么」 |
| [术语表.md](术语表.md) | **英文术语口径** —— 同一个英文词全项目统一译名，读代码卡住时按词查 |

改造过程中挖出并修复的几个真问题（都记在 idea.md 里）：

- **工具注册无法传多参数** —— `register_tool` 用的 LangChain `Tool` 硬编码只接受
  1 个参数，加 `args_schema` 也无效。原有工具恰好都只传 1 个参数所以从未暴露，
  `create_ticket` 其实一直是坏的
- **防幻觉兜底被措辞绕过** —— 固定子串匹配挡不住模型在中间插副词
- **SQLite 外键默认不生效** —— 建表语句里写了 `FOREIGN KEY` 只是声明，不开
  `PRAGMA foreign_keys = ON` 就没有执行力，导致臆造的 user_id 能写进库
- **前端静默失败** —— 渲染抛异常时进度条已移除、异常只打日志，
  用户看到的是"什么都没有"

---

## 环境要求

| 项 | 版本 |
|---|---|
| Python | **3.12**（3.14 太新，chromadb/langchain 的 wheel 还没跟上） |
| 模型 | 通义千问 Qwen-Plus / Qwen-VL-Max（DashScope，走 OpenAI 兼容接口） |
| 向量库 | ChromaDB（本地文件，无需服务） |
| 可选 | Redis（不配就降级为纯内存缓存） |
