# mainconten.md — 项目总纲

> **这份文档是什么:** 调研同类项目后的最终定位与设计总纲。它**取代并整合**了
> [LEARNING_COACH.md](LEARNING_COACH.md)(方向论证)和 [LEARNING_COACH_DESIGN.md](LEARNING_COACH_DESIGN.md)(技术设计)。
> 保留那两份作过程记录,但**以本文为准**。
>
> **目标:** 一个能拿去找 AI 应用 / Agent 开发实习的作品集项目。
> **状态:** 方向待确认。

---

# 第一部分:调研

## 1. 两个同类项目

### 1.1 WeSmartFlow(Tencent,开源,MIT)

**定位:** Agent-native 自适应学习框架。约 **627 个文件**,Python 3.10+ / Vue 3。

**五种能力:** 理解目标 · 陪伴练习 · 记住成长 · 调整路径 · 进入情境

**与我们最相关的部分是它的知识图谱(`backend/kg/`,20 个文件):**

```
backend/kg/
├── models.py                 # 图数据模型
├── relation_semantics.py     # ★ 关系语义
├── embedder.py / vector_store.py / text_for_embedding.py
├── repositories/
│   ├── concept_repo.py       # 概念节点
│   ├── edge_repo.py          # 边
│   ├── facet_repo.py
│   ├── observation_repo.py   # ★ 观察(待处理的信号)
│   └── proposal_repo.py      # ★ 提案(待审核的变更)
└── services/
    ├── aggregator_service.py # ★ 聚合(合并进图)
    ├── proposal_service.py
    └── retrieve_service.py
```

**三个必须记住的设计点:**

1. **四类知识关系**(不只是"前置"):`prerequisite` / `related` / `extends` / `contrasts`
2. **间隔重复用 SM-2**,不是 FSRS(更简单,经典,够用)
3. **★ 知识写入有治理流程**:LLM 抽取的知识**不直接写图**,而是
   `observation(观察到) → proposal(提案) → aggregator(聚合审核后入库)`。
   —— 这直接解决了我之前在设计中担心的"LLM 抽前置依赖质量不可控"问题。

**其他关键模块:**

| 模块 | 内容 |
|---|---|
| `agent_core/` | 自研 agent 基础库:ReAct / Reflection / Plan-and-Solve / Tool / MCP / Skills |
| `services/` | `profile_service`(画像)、`memory_service`、`node_service`、`quiz_service`、`daily_plan_service`、`daily_brief_service`、**`apply_worker`**、`kg_background` |
| `services/immersive/` | `node_extractor`、`profile_updater`、`persistence`、`sse`、`suggestions` |
| `repositories/` | `node_repo` / `profile_repo` / `quiz_repo` / `daily_plan_repo` / `daily_brief_repo` / `session_repo` |
| `channels/` | 微信接入(clawbridge 长轮询 + async httpx 协程池) |

**技术栈:** FastAPI · SQLite(WAL) · **sqlite-vec** · Pydantic · Uvicorn · Vue 3

> **注意**:它用 **sqlite-vec** —— 图和向量**都在 SQLite 里**,不需要独立的图库和向量库。
> 这印证了我之前"SQLite 优先"的判断,而且可以更进一步。

### 1.2 DeepTutor(HKUDS,开源,Apache 2.0,20k+ stars,arXiv 论文)

**定位:** Lifelong Personalized Tutoring。**3559 个文件,~200k 行**,Python 3.11+ / Next.js 16。

**它是平台级的**,能力包括:Chat / Ask Questions / Quiz / Research / Visualize / Solve /
Course Study / **Mastery Path** / Immersive Reading / Immersive Watching。

**与我们最相关的三个模块:**

**① `deeptutor/learning/`(39 个文件,含 18 个测试)** —— 学习闭环的完整实现

```
mastery.py          掌握度
scheduler.py        ★ 间隔重复调度
grading.py          判分
event_hub.py        ★ 事件中心
navigation.py       路径导航
policy.py           学习策略
pending.py          待处理队列
question_card.py    题目卡片
storage.py / service.py / models.py / identity.py / prompts.py
tests/              ★ 18 个测试文件(test_scheduler / test_mastery_* / test_grading ...)
```

**② `deeptutor/runtime/coordination/`** —— **这解决了"消息队列"该怎么做**

```
redis.py        ★ Redis 协调
journal.py      ★ 日志式持久化(append-only)
recovery.py     ★ 故障恢复
protocol.py     协调协议
memory.py / settings.py / types.py
```

> **关键洞察:成熟项目不用 Kafka,用 Redis 做轻量协调 + journal 做持久化 + recovery 做恢复。**
> 这正是我在设计里说的"用 Redis Stream 而不是重型 MQ",而且他们做得更完整——
> **journal + recovery 才是"消息不丢"的真正实现**。

**③ `deeptutor/capabilities/`(84 个文件)** —— 能力插件模型

每个能力一个目录,结构完全一致:

```
capabilities/mastery/
├── capability.py    能力入口
├── loop.py          ★ 该能力的 agent loop
├── tools.py         该能力的工具
├── binding.py       绑定/装配
├── mode.py          模式
├── choices.py
├── pipeline.py
└── (prompts/en|zh/system.md)
```

**其他值得记的:**

| 模块 | 内容 |
|---|---|
| `events/event_bus.py` | 事件总线 |
| `runtime/agentic/` | `loop.py` / `tool_dispatch.py` / `tool_arg_guard.py` / `think_stream.py` / `usage.py` |
| `runtime/` | `background_leader.py` / `isolated_worker.py` / `process.py` / `memory_probe.py` |
| Memory | **三层可检查记忆**:L1 原始 trace / L2 各面事实 / L3 跨面综合。**明确说"不是隐藏的向量库"** |
| `docs/adr/` | ★ 架构决策记录,含 `0003-idempotent-turn-commands.md`(**幂等 turn 命令**) |

## 1.3 对比

| | WeSmartFlow | DeepTutor | 我们的目标 |
|---|---|---|---|
| 规模 | 627 文件 | 3559 文件 / 200k 行 | **≤ 5k 行** |
| 定位 | 学习框架 | 学习平台 | **单点做深** |
| 图 | 自建 kg 模块 + SQLite | GraphRAG/LightRAG 多引擎 | **SQLite 表 + 递归 CTE** |
| 记忆 | 图谱 + 画像 | L1/L2/L3 三层 | **BKT + SM-2** |
| 队列 | 后台任务 + worker | Redis 协调 + journal | **Redis 协调(简化版)** |
| 编排 | 自研 agent_core | capability 插件 + 统一 runtime | **单 loop + 线性工作流** |
| 评测 | 未公开数字 | 未公开数字 | **★ 有量化数字** |

---

# 第二部分:调研结论

## 2.1 好消息:场景被验证了

这不是伪需求。**腾讯和港大实验室都在投入**,说明"个性化学习"是真问题。
我之前的担忧(§"场景是不是伪需求")可以打消。

## 2.2 坏消息:不能重复造轮子

如果我们做"又一个学习教练平台",结果会很难看:
- 规模比不过:**人家 200k 行,我们 5k 行**
- 功能比不过:他们覆盖了视频、语音、IM、多用户、几十种 RAG 引擎
- 面试官会问:"这不就是重造 WeSmartFlow 吗?"

**照抄这两个项目的功能列表 = 自杀。**

## 2.3 真正的机会在哪

我把两份 README 通读下来,发现一个共同点:

> **两个项目都在讲"我有什么能力",但都没有公开"我的效果有多好"。**

- WeSmartFlow 说"记录掌握度变化",**没说掌握度预测准不准**
- DeepTutor 说"Mastery Path",**没说它比随便推题好多少**
- 两个都提到 GraphRAG,**都没有说图相比向量检索到底提升多少**

**这就是我们的位置:不做平台,做"把某一个点做到可测量"。**

## 2.4 从两个项目学到的最有价值的三件事

这三条修正了我之前的设计:

| 学到 | 原设计 | 修正后 |
|---|---|---|
| **① 知识写入要治理** | LLM 抽完直接写图 | `observation → proposal → aggregator` 三阶段,LLM 的输出是**提案**不是事实 |
| **② 队列 = Redis 协调 + journal + recovery** | Redis Stream,重试 + 死信 | 加上 **journal(append-only 日志)** 和 **recovery(重启恢复)**,这才是"不丢消息" |
| **③ 关系不止"前置"** | 只有 `prerequisite` | 四类:`prerequisite` / `related` / `extends` / `contrasts` |

---

# 第三部分:定位

## 3.1 实习项目 ≠ 创业项目

找实习的作品集,面试官看的**不是规模**,是四件事:

1. **能不能把一件事做完整**(从数据到 API 到能跑)
2. **懂不懂原理**(不是调 API,是知道 BKT 为什么这么算)
3. **有没有工程意识**(异步、幂等、恢复、测试)
4. **能不能讲清楚取舍**(为什么这么做,为什么不那么做)

> 200k 行的项目对实习**没有加分**——面试官只会觉得"这不是你写的"。
> **5k 行但每一行都能解释清楚,才是加分。**

## 3.2 收窄策略

**不做"学习平台",做"一个可测量的学习闭环"。**

具体收窄:

| 维度 | 收窄为 |
|---|---|
| 学科 | **数据结构 + 算法**(已确认) |
| 规模 | **30~50 个知识点**(不是几千) |
| 用户 | **单用户**(不做多租户、不做权限) |
| 入口 | **只用 Swagger**,不做前端 |
| 能力 | **只做闭环**,不做课程生成 / 视频 / 语音 / IM |
| 判分 | **纯规则**(已确认) |

## 3.3 差异化:量化评测(这是全部理由)

我们比两个大项目多的**只有一样东西:数字**。

| 评测 | 指标 | 为什么两个大项目都没做/没公开 |
|---|---|---|
| **掌握度预测准不准** | AUC / Brier | 需要 ground truth,平台不做实验 |
| **遗忘预测准不准** | 校准曲线 | 同上 |
| **图相比向量的增益** ★ | 根因定位准确率 | 需要构造对照实验 |
| **规划相比随机的增益** | 后续正确率提升 | 需要 A/B |

**最后两项是核心。** 它们能回答"你为什么要用图 / 为什么要用 BKT"——
这是面试官**一定会问**的问题,而大多数人答不上来。

---

# 第四部分:最终方案

## 4.1 一句话定位

> **一个把"学不会的根因"定位出来的学习 Agent:它用知识图谱做多跳推理找根因,
> 用 BKT/SM-2 建模掌握度和遗忘,用 Redis 协调把长流程异步化,
> 并且用公开可复现的评测证明——图推理相比扁平检索到底强多少。**

## 4.2 做什么 / 不做什么

| ✅ 做 | ❌ 不做 |
|---|---|
| 30~50 个知识点的小图(数据结构+算法) | 上千知识点的全量图 |
| 四类关系:prerequisite / related / extends / contrasts | 只做 prerequisite |
| BKT 掌握度 + SM-2 间隔重复 | FSRS / 复杂参数拟合 |
| **根因定位**(多跳图推理) | 泛泛的"推荐相关题" |
| Redis 协调 + journal + recovery | Kafka / RabbitMQ |
| 事件总线 + 常驻 worker | 多进程集群部署 |
| FastAPI + Swagger | Vue / Next.js 前端 |
| **四张评测表** | 堆功能 |
| 模拟学生(可控制 ground truth) | 收集真实用户数据 |

## 4.3 架构(借鉴两个项目的骨架)

```
┌──────────────────────────────────────────────────────────┐
│ api/            FastAPI:只校验 + 入队 + 查询,无 LLM 调用   │
│                 POST /answer · GET /plan · GET /gap · /profile │
└───────────────────────┬──────────────────────────────────┘
                        │ publish
                        ▼
┌──────────────────────────────────────────────────────────┐
│ events/         event_bus.py        (借鉴 DeepTutor)       │
│ coordination/   redis.py   ← Redis Stream / Pub-Sub      │
│                 journal.py ← append-only 事件日志 ★       │
│                 recovery.py← 重启后从 journal 恢复 ★      │
└───────────────────────┬──────────────────────────────────┘
                        │ consume
                        ▼
┌──────────────────────────────────────────────────────────┐
│ workers/        常驻消费者(asyncio)                       │
│   profile_worker   判分(规则) → BKT → SM-2                │
│   planner_worker   ★ GraphRAG 根因定位 / 路径规划          │
│   scheduler_worker SM-2 到期扫描 → 生成复习计划            │
└──────┬───────────────────────┬───────────────────────────┘
       │                       │
       ▼                       ▼
┌───────────────┐   ┌──────────────────────────┐
│ knowledge/    │   │ profile/                 │
│ 图谱(SQLite)  │   │ BKT 掌握度 / SM-2 记忆    │
│ 递归 CTE 遍历 │   │ 易错模式 / 学习者档案      │
│ ★ 写入治理:   │   └──────────────────────────┘
│  observation  │
│  → proposal   │   ┌──────────────────────────┐
│  → aggregator │   │ vector(SQLite + vec 扩展) │
└───────────────┘   └──────────────────────────┘
```

**分层职责:**

| 层 | 职责 | 借鉴来源 |
|---|---|---|
| `api/` | 入口,只入队 | DeepTutor |
| `events/` | 事件总线 | DeepTutor `events/event_bus.py` |
| `coordination/` | Redis 协调 + journal + recovery | DeepTutor `runtime/coordination/` |
| `workers/` | 常驻消费 | WeSmartFlow `apply_worker` / `kg_background` |
| `knowledge/` | 图谱 + 写入治理 | WeSmartFlow `backend/kg/` |
| `profile/` | 画像 + 建模 | DeepTutor `learning/` |
| `domain/` | 领域模型 | 两者都有的 `models.py` |

## 4.4 知识图谱设计(含写入治理)

**节点与边:**

```
节点: Concept {id, name, subject, difficulty, description}
      Problem {id, title, source, judge_type, test_cases}
      Topic   {id, name}

边:   PREREQUISITE  (前置,★ 多跳推理用)
      RELATED       (相关)
      EXTENDS       (延伸)
      CONTRASTS     (对比)          ← 四类关系,借鉴 WeSmartFlow
      ASSESSED_BY   (知识点 ←→ 题目)
```

**★ 写入治理流程(借鉴 WeSmartFlow 的 observation/proposal/aggregator):**

```
[教材/课程大纲]
      │
      ▼
[1] LLM 抽取  ──►  Observation(原始观察,未验证)
                     │  例: "动态规划 可能依赖 递归"
                     ▼
[2] 规则 + LLM 校验 ──► Proposal(提案)
                     │  拒自环 / 合并重复 / confidence < 0.6 丢弃
                     ▼
[3] 环检测(DFS)   ──► 有环则拒绝或降级
                     ▼
[4] Aggregator     ──► 写入图(PREREQUISITE 边)
```

**为什么这个流程重要:** 它把"LLM 可能抽错"这件事**显式化**了。
面试时可以讲:"LLM 的输出我不当事实,当提案,经过校验才入库"——这是工程成熟度的体现。

**多跳查询(核心,SQLite 递归 CTE):**

```sql
-- X 的前置闭包,深度 ≤ 3
WITH RECURSIVE prereqs(kp_id, depth) AS (
    SELECT $x, 0
    UNION
    SELECT e.from_id, p.depth + 1
    FROM edges e JOIN prereqs p ON e.to_id = p.kp_id
    WHERE e.type = 'PREREQUISITE' AND p.depth < 3
)
SELECT kp_id, MIN(depth) AS depth FROM prereqs
WHERE kp_id != $x GROUP BY kp_id;
```

**根因定位算法:**

```
输入: learner_id, 目标知识点 X
1. P ← 前置闭包(X, depth ≤ 3)
2. 取画像 { p.mastery : p ∈ P }
3. Gaps ← { p ∈ P : p.mastery < 0.4 }
4. 排序:按 (depth 升序, mastery 升序) ← 最浅且最弱
5. root_causes ← 前 1~3 个
6. LLM 把 (根因 + 依赖路径 + 证据题) 组织成人话
```

## 4.5 事件与协调(不是 Kafka,是 Redis)

**事件信封:**

```json
{
  "event_id": "01HXYZ...",     // ULID,幂等键
  "type": "answer.submitted",
  "ts": 1757500000.123,
  "learner_id": "u_001",
  "trace_id": "01HABC...",
  "attempt": 1,
  "payload": { ... }
}
```

**事件类型:**

| 事件 | 生产者 | 消费者 |
|---|---|---|
| `answer.submitted` | api | profile_worker |
| `profile.updated` | profile_worker | planner_worker |
| `tick.scheduled` | scheduler_worker | planner_worker |
| `plan.updated` | planner_worker | api(推送) |

**消费契约(每个 worker 必须遵守,借鉴 DeepTutor 的 ADR 思路):**

```
1. 从 Redis Stream 取(consumer group)
2. ★ 幂等检查(event_id 是否已处理)
3. 写 journal(append-only)——先记后做
4. 业务处理
5. ACK
6. 失败 → attempt += 1
   ├─ ≤ 3 → 重新投递
   └─ > 3 → 死信 + 告警
7. ★ 进程重启 → recovery 扫 journal + pending,补做未完成的事件
```

> **第 3 步和第 7 步是"不丢消息"的真正实现。** 光有 Redis Stream 不够——
> 必须先把事件落到 append-only 日志,重启后能重放。

## 4.6 工作流

**先不引入 LangGraph。** P0~P3 用线性 async 函数链,**P4 再替换**。

```
提交答案
  → [1] 判分(规则比对 test_cases)
  → [2] BKT 更新掌握度
  → [3] SM-2 更新记忆状态 + 重算到期时间
  → [4] ★ 根因检测(多跳图查询)
        ├─ 有缺口 → 生成补救路径
        └─ 无缺口 → 推进下一节点
  → [5] 生成/更新今日计划
  → [6] 异步:出题(入队,不阻塞)
  → [7] 推送(事件)
```

**P4 换 LangGraph 的理由:** 断点恢复。迁移本身就是面试素材——
"我先用 50 行手写打通,后来发现需要断点恢复才引入 LangGraph"。

---

# 第五部分:内容大纲

## 5.1 第一批知识点(数据结构 + 算法,30~50 个)

**数据结构(约 15 个):**
数组 · 链表 · 栈 · 队列 · 哈希表 · 二叉搜索树 · 堆 · 图(邻接表/矩阵) ·
并查集 · 前缀树 · 线段树 · 布隆过滤器 · 跳表 · 单调栈 · 滑动窗口

**算法(约 20 个):**
二分查找 · 双指针 · 排序(快排/归并) · 递归 · 分治 · 回溯 · 贪心 ·
动态规划(线性/区间/树形) · 记忆化搜索 · BFS · DFS · 拓扑排序 ·
最短路径(Dijkstra) · 最小生成树 · 字符串匹配(KMP) · 位运算

**前置依赖示例(建图时的金标准样例):**

```
函数调用  ──PREREQUISITE──►  递归
递归      ──PREREQUISITE──►  分治
递归      ──PREREQUISITE──►  回溯
递归 + 数组 ─PREREQUISITE──►  动态规划
数组      ──PREREQUISITE──►  双指针
双指针    ──PREREQUISITE──►  滑动窗口
数组      ──PREREQUISITE──►  二分查找
队列      ──PREREQUISITE──►  BFS
栈        ──PREREQUISITE──►  DFS
DFS + 队列 ─PREREQUISITE──►  拓扑排序
贪心      ──PREREQUISITE──►  Dijkstra
排序      ──RELATED──────►   二分查找
快排      ──CONTRASTS────►   归并排序
动态规划  ──CONTRASTS────►   贪心
```

**数据来源:**
- 知识点清单:公开课程大纲(数据结构/算法课程目录)、LeetCode tag 体系
- 前置关系:**用 LLM 从教材/大纲抽取**(走 §4.4 的治理流程)+ 人工校验上面这些金标准
- 题目:LeetCode 公开题(带 `test_cases`)+ 少量自建

## 5.2 目录结构

```
coach/
├── config.py
├── domain/models.py              # 领域模型
├── knowledge/                    # 知识图谱
│   ├── schema.py                 # 建表 DDL
│   ├── store.py                  # 存取 + 递归 CTE 查询
│   ├── queries.py                # ★ 多跳查询 / 根因定位
│   ├── governance.py             # ★ observation → proposal → aggregator
│   └── builder.py                # 建图流程
├── profile/                      # 学习者画像
│   ├── bkt.py                    # 掌握度
│   ├── sm2.py                    # 间隔重复
│   ├── errors.py                 # 易错模式
│   └── store.py
├── events/
│   ├── bus.py                    # 事件总线
│   └── schema.py                 # 事件定义
├── coordination/
│   ├── redis.py                  # Redis Stream 封装
│   ├── journal.py                # ★ append-only 日志
│   └── recovery.py               # ★ 重启恢复
├── workers/
│   ├── profile_worker.py
│   ├── planner_worker.py
│   ├── scheduler_worker.py
│   └── run_all.py
├── workflow/
│   └── pipeline.py               # 线性 async 链(P4 换 LangGraph)
├── evaluation/                   # ★ 差异化所在
│   ├── simulated_student.py      # 模拟学生(可控制 ground truth)
│   ├── metrics.py                # AUC / Brier / 校准
│   └── run_eval.py
└── api/
    ├── main.py
    ├── routes.py
    └── schemas.py
```

## 5.3 API 契约

```yaml
POST /answer
  body: {learner_id, problem_id, answer_text, elapsed_ms}
  resp: 202 {accepted, event_id, trace_id}     # < 50ms,不含 LLM 调用

GET /profile/{learner_id}
  resp: {goal, mastery_summary, due_now, error_patterns}

GET /plan/{learner_id}
  resp: {date, items: [{kp_id, action: review|learn|remedial, reason, est_minutes}]}

GET /gap/{learner_id}/{kp_id}          # ★ 核心:根因定位
  resp: {root_causes: [{kp_id, name, mastery, depth, path}], explanation}

GET /graph/{kp_id}?depth=3             # 图谱可视化数据
  resp: {node, prereq_tree, mastered, gaps}
```

---

# 第六部分:阶段路线

| 阶段 | 交付 | 验收标准 |
|---|---|---|
| **P0** | 小图(15 个点)+ 治理流程 + SQLite CTE | 图无环;`ancestors("算法.动态规划")` 返回正确闭包 |
| **P1** | FastAPI + Redis 协调 + journal + profile_worker | `POST /answer` < 50ms;BKT 单测过;kill 后 journal 能重放 |
| **P2** | ★ 根因定位 `/gap/{learner}/{kp}` | 能指出前置缺口;`/graph` 返回前置树 |
| **P3** | SM-2 + scheduler_worker + 事件推送 | 到期知识点生成复习计划 |
| **P4** | 线性链 → LangGraph + checkpointer | kill -9 后能恢复,不重跑 |
| **P5** | ★ 四张评测表 | README 有结果表 |
| **P6** | README + demo + docker-compose | 别人 clone 能跑 |

**P0 的验收线(最重要的里程碑):**
- `POST /answer` **< 50ms 返回** —— 证明是真异步
- 答错某题后,`GET /gap/...` 能定位到一个前置知识点
- 图中的前置关系**经过治理流程**(能演示 observation → proposal → aggregator)

---

# 第七部分:面试叙事

## 2 分钟版

> "市面上有腾讯的 WeSmartFlow、港大的 DeepTutor 这类学习平台,都很完整。我做的是
> **反过来的一件事:它们都在讲'我有什么能力',我想回答'这些能力到底有没有用'。**
>
> 我选了一个具体问题:**学生反复做错一类题,根因往往不在这个知识点本身,而在前置知识。**
> 扁平检索只能推荐'相似题目',做不到这一点——**只有沿着知识图谱反向遍历前置依赖,
> 才能定位到'你真正缺的是哪个基础'**。
>
> 工程上有三件事我做得比较认真:
> 一是 **LLM 抽取的前置关系不当事实用**,走 observation → proposal → aggregator 的
> 治理流程,有环检测和置信度过滤;
> 二是 **不用 Kafka**,用 Redis 协调 + append-only journal + 重启恢复,
> 因为消息不丢的关键在 journal 和 recovery,不在中间件有多重;
> 三是**建了评测**:用可控制 ground truth 的模拟学生,量化掌握度预测的 AUC,
> 以及**图推理相比向量检索在根因定位上到底强多少**——这个数字我诚实汇报。"

## 三个支点

| 支点 | 内容 |
|---|---|
| **取舍** | 不做平台,做一个可测量的点;承认同类项目的存在而不是装作没有 |
| **数字** | 四张评测表,尤其"图 vs 向量"的对照实验 |
| **工程成熟度** | 幂等 + journal + recovery + 写入治理,不是简单调 API |

## 会被追问的问题(提前准备)

| 问题 | 答案要点 |
|---|---|
| "这跟 Anki 有什么区别?" | Anki 是间隔重复,不管知识结构;我做的是**结构 + 根因** |
| "为什么不用 Neo4j?" | 规模小,SQLite 递归 CTE 够用;少一个依赖;接口封在 store.py 可平移 |
| "模拟学生不是自欺欺人吗?" | 承认局限;它验证的是**算法正确性**(BKT 能否收敛到真值),不是用户体验 |
| "为什么用 SM-2 不用 FSRS?" | SM-2 够用且参数可解释;FSRS 是优化项不是必需项 |
| "图真的比向量强吗?" | **这是我要测的,不是我要假设的**;如果没强很多,我会写进 Limitations |
| "为什么不上消息队列中间件?" | Redis 协调 + journal + recovery 已经解决了不丢消息的问题;Kafka 是过度设计 |

---

# 第八部分:待确认

| # | 决策 | 建议 | 状态 |
|---|---|---|---|
| 1 | 学科范围 | 数据结构 + 算法 | ✅ 已确认 |
| 2 | 判分方式 | 纯规则(比对 test_cases) | ✅ 已确认 |
| 3 | 代码位置 | 现有 repo 加 `coach/` | ✅ 已确认 |
| 4 | 图存储 | SQLite 表 + 递归 CTE(不上 Kuzu) | ✅ 已确认 |
| 5 | 关系类型 | 四类(prerequisite/related/extends/contrasts) | ✅ 已确认 |
| 6 | 间隔重复 | SM-2(不用 FSRS) | ✅ 已确认 |
| 7 | 写入治理 | 做 observation→proposal→aggregator 简化版 | ✅ 已确认 |
| 8 | 队列 | Redis Stream + journal + recovery(不上 Kafka) | ✅ 已确认 |
| 9 | 工作流 | P0~P3 线性链,P4 换 LangGraph | ✅ 已确认 |
| 10 | 评测规模 | 模拟学生 + 四张表 | ✅ 已确认 |

---

# 附:这份总纲相对之前的修正

| 项 | 之前的设计 | 修正后 | 原因 |
|---|---|---|---|
| 定位 | 学习教练(泛) | 单点做深 + 量化评测 | 同类项目已有两个,不能重复 |
| 图存储 | Kuzu → 后来改 SQLite | **确定 SQLite + CTE** | WeSmartFlow 用 sqlite-vec 印证 |
| 关系 | 只有 prerequisite | **四类关系** | 借鉴 WeSmartFlow |
| 抽取 | LLM 直接写图 | **三步治理流程** | 借鉴 WeSmartFlow 的 observation/proposal |
| 队列 | Redis Stream + 重试 + 死信 | **+ journal + recovery** | 借鉴 DeepTutor 的 coordination/journal/recovery |
| 间隔重复 | FSRS | **SM-2** | WeSmartFlow 用 SM-2,更简单够用 |
| 编排 | 一开始考虑 LangGraph | **P4 才引入** | 迁移本身是面试素材 |
| 差异化 | 未明确 | **"图 vs 向量"的对照实验** | 两个大项目都没公开这类数字 |
