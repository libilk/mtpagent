# 个性化学习教练 Agent — 详细技术设计

> 前置文档:[LEARNING_COACH.md](LEARNING_COACH.md)(方向论证)
> 领域:**编程 / 算法** | 位置:`coach/`(现有 repo 内)| 状态:**设计待确认**
>
> 本文只定设计,不写实现。确认后再进入 P0。

---

## 0. 设计原则(先立规矩)

1. **API 层不含任何 LLM 调用。** 一切慢操作异步化,`/answer` 必须毫秒级返回。
2. **每个副作用幂等。** 消息可能重复投递,画像更新必须能安全重放。
3. **状态可重建。** 画像由答题事件流推导而来,丢了能重算。
4. **图是无环的(DAG)。** 前置依赖有环 = 整个规划逻辑崩溃,建图时必须校验。
5. **可评测。** 每个核心组件都要有"怎么算它对"的定义。

---

## 1. 目录结构

```
coach/
├── __init__.py
├── config.py                 # 配置(Redis/Kuzu/LLM/路径)
├── domain/
│   └── models.py             # 领域模型(KnowledgePoint / LearnerState / Answer)
├── graph/                    # ── 知识图谱层 ──
│   ├── schema.py             # DDL + 约束
│   ├── store.py              # Kuzu 封装(增删查)
│   ├── queries.py            # ★ 多跳查询(前置闭包 / 缺口检测)
│   └── builder.py            # 建图:LLM 抽取 + 校验 + 写入
├── profile/                  # ── 学习者画像层 ──
│   ├── store.py              # SQLite 持久化
│   ├── bkt.py                # 掌握度模型
│   ├── fsrs.py               # 遗忘曲线模型
│   └── errors.py             # 易错模式
├── mq/                       # ── 消息层 ──
│   ├── streams.py            # Redis Stream 封装(publish/consume/ack/dlq)
│   ├── events.py             # 事件 schema
│   └── consumer.py           # 通用消费循环(consumer group + 重试 + 死信)
├── workers/                  # ── 常驻 worker ──
│   ├── profile_worker.py     # ① 画像更新
│   ├── planner_worker.py     # ② 规划(GraphRAG)
│   ├── scheduler_worker.py   # ③ 遗忘扫描(定时)
│   ├── generator_worker.py   # ④ 出题(慢)
│   └── run_all.py            # 单进程拉起全部 worker(开发用)
├── workflow/                 # ── 工作流层 ──
│   ├── state.py              # LangGraph State 定义
│   ├── nodes.py              # 节点实现
│   └── graph.py              # 图编排 + checkpointer
└── api/                      # ── 入口层 ──
    ├── main.py               # FastAPI app
    ├── routes.py             # REST 端点
    ├── ws.py                 # WebSocket 连接管理
    └── schemas.py            # 请求/响应 Pydantic 模型
```

---

## 2. 数据模型

### 2.1 领域模型(代码内)

```python
# coach/domain/models.py
@dataclass
class KnowledgePoint:
    id: str                  # "algo.recursion"
    name: str                # "递归"
    subject: str             # "algorithms"
    difficulty: float        # 1.0 ~ 5.0
    description: str         # 一句话说明
    refs: dict               # {"leetcode_tags": [...], "outline": "..."}

@dataclass
class PrereqEdge:
    from_kp: str             # 前置
    to_kp: str               # 后继
    weight: float = 1.0      # 依赖强度
    confidence: float = 1.0  # 抽取置信度(LLM 给的分)

@dataclass
class Answer:
    answer_id: str
    learner_id: str
    problem_id: str
    kp_ids: list[str]        # 该题考察的知识点(可多个)
    correct: bool
    answer_text: str
    elapsed_ms: int
    ts: float

@dataclass
class MemoryState:           # FSRS 状态
    kp_id: str
    difficulty: float        # D ∈ [1, 10]
    stability: float         # S,单位:天
    last_review: float       # 时间戳
    due_at: float            # 下次到期时间戳
    reps: int
    lapses: int
```

### 2.2 知识图谱(Kuzu)

Kuzu 用 **Cypher** 语法,嵌入式(像 SQLite),零运维。

```cypher
-- 节点表
CREATE NODE TABLE KnowledgePoint(
    id STRING,
    name STRING,
    subject STRING,
    difficulty DOUBLE,
    description STRING,
    PRIMARY KEY(id)
);

CREATE NODE TABLE Problem(
    id STRING,
    title STRING,
    source STRING,              -- "leetcode:70" / "custom"
    difficulty DOUBLE,
    judge_type STRING,          -- "exact_output"(纯规则判题,已确认方案)
    test_cases STRING,          -- JSON: [{"input":..., "expected":..., "hidden":false}]
    PRIMARY KEY(id)
);

CREATE NODE TABLE Topic(id STRING, name STRING, PRIMARY KEY(id));

-- 边表
CREATE REL TABLE PREREQ_OF(
    FROM KnowledgePoint TO KnowledgePoint,
    weight DOUBLE, confidence DOUBLE
);

CREATE REL TABLE RELATED_TO(FROM KnowledgePoint TO KnowledgePoint, weight DOUBLE);
CREATE REL TABLE PART_OF(FROM KnowledgePoint TO Topic);
CREATE REL TABLE ASSESSED_BY(FROM KnowledgePoint TO Problem);
```

**约束(在 builder 里用代码保证,不靠数据库):**
- `PREREQ_OF` 必须是 **DAG**。写入前做环检测(DFS),有环则拒绝。
- 自环(`X → X`)直接拒绝。
- 重复边合并,取 `max(confidence)`。

### 2.3 画像存储(SQLite)

```sql
-- 掌握度(BKT 状态)
CREATE TABLE learner_mastery (
    learner_id  TEXT,
    kp_id       TEXT,
    p_known     REAL,        -- BKT 的 P(L_n),0~1
    updated_at  REAL,
    PRIMARY KEY (learner_id, kp_id)
);

-- 记忆状态(FSRS)
CREATE TABLE learner_memory (
    learner_id  TEXT,
    kp_id       TEXT,
    difficulty  REAL,
    stability   REAL,
    last_review REAL,
    due_at      REAL,
    reps        INTEGER,
    lapses      INTEGER,
    PRIMARY KEY (learner_id, kp_id)
);

-- 易错模式
CREATE TABLE learner_error (
    learner_id  TEXT,
    kp_id       TEXT,
    error_type  TEXT,        -- concept_confusion / careless / missed_condition
    count       INTEGER,
    PRIMARY KEY (learner_id, kp_id, error_type)
);

-- 学习者档案
CREATE TABLE learner_profile (
    learner_id    TEXT PRIMARY KEY,
    goal_kp_id    TEXT,      -- 目标知识点
    daily_minutes INTEGER,
    preference    TEXT,      -- "example_first" / "theory_first"
    created_at    REAL
);

-- 答题明细(事件流的物化,用于评测与重放)
CREATE TABLE answers (
    answer_id   TEXT PRIMARY KEY,
    learner_id  TEXT,
    problem_id  TEXT,
    kp_ids      TEXT,        -- JSON 数组
    correct     INTEGER,
    answer_text TEXT,
    elapsed_ms  INTEGER,
    ts          REAL
);
CREATE INDEX idx_answers_learner ON answers(learner_id, ts);
```

> **在现有 repo 里**:可以复用 `data/` 目录约定,DB 放 `data/coach/coach.db`,图放 `data/coach/graph.kuzu`。

---

## 3. 核心算法

### 3.1 BKT — 掌握度(贝叶斯知识追踪)

每个知识点 4 个参数,初值可调:

| 参数 | 含义 | 建议初值 |
|---|---|---|
| `p_init` | 初始已掌握概率 | 0.1 |
| `p_transit` | 一次练习后学会的概率 | 0.15 |
| `p_guess` | 没学会但猜对的概率 | 0.2 |
| `p_slip` | 学会了但答错的概率 | 0.1 |

**更新公式:**

```
观测答对:
  P(Lₙ | correct) = P(Lₙ)(1 − S) / [ P(Lₙ)(1 − S) + (1 − P(Lₙ))·G ]

观测答错:
  P(Lₙ | wrong)   = P(Lₙ)·S / [ P(Lₙ)·S + (1 − P(Lₙ))(1 − G) ]

学习转移:
  P(Lₙ₊₁) = P(Lₙ | 观测) + (1 − P(Lₙ | 观测)) · T
```

**为什么用 BKT 而不是"答对 +0.1"这种土办法:** BKT 区分了"猜对"和"真会",并且有明确的概率语义——**可以直接被评测**(见 §9)。

### 3.2 FSRS — 遗忘曲线

用 **FSRS-4.5** 的幂函数遗忘曲线:

```
回忆概率:  R(t, S) = (1 + F · t / S)^D
          其中 D = −0.5,  F = 19/81 ≈ 0.2346

反解间隔:  I(R_d, S) = (S / F) · ( R_d^(1/D) − 1 )
```

**状态:** 每个知识点维护 `(difficulty, stability, last_review, due_at)`。

**复习时的更新(简化描述):**
- 答对 → `stability` 增长(增长率受 `difficulty` 和当前 `retrievability` 影响)
- 答错 → `stability` 大幅回退(`lapses += 1`),`difficulty` 上升
- 按目标保留率(如 `R_d = 0.9`)反解下次 `due_at`

> **实现建议**:直接用开源 `fsrs` Python 包(Anki 生态),避免自己调参。**但要在文档里写清楚公式**,否则面试问"FSRS 是什么"答不上来。

### 3.3 缺口检测 — GraphRAG 的核心

这是整个项目**最能讲**的一段。三个查询:

**Q1: 知识点 X 的前置闭包(深度 ≤ 3)**

```cypher
MATCH (x:KnowledgePoint {id: $x})
MATCH (p:KnowledgePoint)-[:PREREQ_OF*1..3]->(x)
RETURN DISTINCT p.id, p.name, p.difficulty
```

**Q2: "他为什么学不会 X?"** ← 根因定位

```
输入: learner_id, kp_id = X
1. 图遍历  P ← 前置闭包(X, max_depth=3)          # Q1
2. 批量查   {p.known : p ∈ P}                     # 画像
3. 缺口集  Gaps ← { p ∈ P : p.known < 0.4 }
4. 排序    按 (depth 升序, p.known 升序)
5. 根因    ← Gaps 中深度最浅且掌握度最低的 1~3 个
6. 组织    LLM 把(根因 + 依赖路径 + 证据题目)写成一句人话
```

**为什么必须用图:**
扁平检索只能召回"和 X 相似的题目/文档",**没有任何机制能表达"X 依赖 Y"**。
"学不会 X 是因为 Y 没掌握" 这个结论,**只能靠反向遍历前置边得到**。

**Q3: "下一步学什么?"** ← 路径规划

```
1. 目标闭包  P ← 前置闭包(goal_kp, max_depth=∞)
2. 可推进    Ready ← { p ∈ P : ∀(q ∈ prereqs(p)) q.known ≥ 0.7 }
3. 排序      by (difficulty, 距 goal 的路径长度)  取 top-k
4. 结合 FSRS due_at 与 daily_minutes,排进今日计划
```

### 3.4 建图 — 前置依赖从哪来(最大工作量)

公开数据里题目标签多,**前置依赖几乎没有**。所以:

```
[教材/课程大纲]
     │
     ▼
[1] LLM 抽取候选知识点          → KnowledgePoint 列表
     │
     ▼
[2] LLM 抽取候选前置关系        → (from, to, confidence) 列表
     │   prompt 要点: 只输出"没有 Y 就学不了 X"的强依赖,排除"相关但可独立学"
     ▼
[3] 规则校验                     → 拒自环;合并重复;过滤 confidence < 0.6
     │
     ▼
[4] 环检测(DFS)                 → 有环则人工介入 / 丢弃低置信度边
     │
     ▼
[5] 写入 Kuzu                   → PREREQ_OF 边
```

**P0 阶段建议手工校验 30~50 个知识点的小图**(比如 "算法入门"),把管道跑通,不要一上来铺上千点。

---

## 4. 消息与事件

### 4.1 Stream 定义

| Stream | 生产者 | 消费者组 | 语义 |
|---|---|---|---|
| `coach:answer` | API | `profile-workers` | 答题事件 |
| `coach:tick` | scheduler 定时器 | `planner-workers` | 定时扫描触发 |
| `coach:generate` | workflow | `generator-workers` | 异步出题 |
| `coach:notify` | workflow | `notify-workers` | 推送任务 |
| `coach:{stream}:dlq` | consumer | — | 死信(重试 N 次后) |

### 4.2 事件信封(统一格式)

```json
{
  "event_id": "01HXYZ...",        // ULID,幂等键
  "type": "answer.submitted",
  "ts": 1757500000.123,
  "learner_id": "u_001",
  "trace_id": "01HABC...",        // 贯穿一次请求的全链路
  "attempt": 1,
  "payload": {
    "answer_id": "...",
    "problem_id": "p_042",
    "kp_ids": ["algo.recursion"],
    "correct": true,
    "elapsed_ms": 42000
  }
}
```

### 4.3 消费契约(每个 worker 必须遵守)

```
1. XREADGROUP 取消息        (consumer group,避免重复消费)
2. 幂等检查                 (event_id 是否已处理过)
3. 业务处理
4. XACK 确认
5. 失败 → attempt += 1
   ├─ attempt ≤ 3 → 重新投递(可带延迟)
   └─ attempt > 3 → XADD 到 dlq,并告警
```

**幂等键存哪:** `processed_events(event_id, ts)` 表,TTL 7 天。这是**必须**的——Redis Stream 至少投递一次,不保证恰好一次。

---

## 5. API 契约

```yaml
POST /answer
  body:  {learner_id, problem_id, answer_text, elapsed_ms}
  resp:  202 {accepted: true, event_id, trace_id}
  说明:  只校验 + 入队,立刻返回。不含任何 LLM 调用。

GET /profile/{learner_id}
  resp: {
    goal: "algo.dp",
    daily_minutes: 30,
    mastery_summary: {strong: [...], weak: [...]},
    due_now: [{kp_id, name, retrievability}],
    error_patterns: [{kp_id, type, count}]
  }

GET /plan/{learner_id}
  resp: {
    date: "2026-03-05",
    items: [
      {kp_id, name, action: "review"|"learn"|"remedial",
       reason: "FSRS 到期" | "前置已满足" | "前置缺口",
       est_minutes: 12, problem_ids: [...]}
    ]
  }

GET /graph/{kp_id}?depth=3                    # GraphRAG 演示:RAG 侧
  resp: {node, prereq_tree, mastered: [...], gaps: [...]}

GET /gap/{learner_id}/{kp_id}                 # ★ 核心:为什么学不会
  resp: {
    kp: {...},
    root_causes: [{kp_id, name, p_known, depth, path: [...]}],
    explanation: "你在「递归」上正确率 45%,根因是「函数调用栈」只有 0.3 ...",
    suggested_path: ["algo.func_stack", "algo.recursion_base"]
  }

WS /ws/{learner_id}
  server → client:
    {type: "review_due",  data: {kp_ids: [...], retrievability: [...]}}
    {type: "new_problem", data: {problem_id, kp_ids, difficulty}}
    {type: "plan_updated", data: {...}}
  client → server: {type: "ping"}     # 保活
```

---

## 6. 工作流(LangGraph)

### 6.1 State

```python
# coach/workflow/state.py
class CoachState(TypedDict):
    learner_id: str
    trace_id: str
    answer: dict                # 答题事件 payload
    kp_ids: list[str]
    correct: bool
    mastery_delta: dict         # {kp_id: {before, after}}
    gaps: list[dict]            # 缺口检测结果
    branch: str                 # "remedial" | "advance"
    plan: list[dict]
    notifications: list[dict]
    errors: list[str]
```

### 6.2 图结构

```
        ┌─────────┐
        │  grade  │  判分(规则/LLM 判主观题)
        └────┬────┘
             ▼
   ┌──────────────────┐
   │ update_mastery   │  BKT 更新
   └────────┬─────────┘
            ▼
   ┌──────────────────┐
   │ update_memory    │  FSRS 更新 + 重算 due_at
   └────────┬─────────┘
            ▼
   ┌──────────────────┐
   │   detect_gap     │  ★ GraphRAG 多跳查询
   └────────┬─────────┘
            ▼
      ╱─────────╲
     ╱ route_gap ╲      条件边
      ╲─────────╱
       │        │
  gaps │        │ 无 gaps
       ▼        ▼
 ┌──────────┐ ┌──────────┐
 │ remedial │ │ advance  │   生成补救路径 / 推进下一节点
 └────┬─────┘ └─────┬────┘
      └──────┬──────┘
             ▼
        ┌────────┐
        │  plan  │  合成今日计划(结合 FSRS due + daily_minutes)
        └───┬────┘
            ▼
   ┌─────────────────┐
   │ enqueue_generate│  异步出题 → publish(coach:generate)
   └────────┬────────┘
            ▼
        ┌──────┐
        │ push │  → publish(coach:notify) → WebSocket
        └──────┘
```

### 6.3 Checkpointer

用 LangGraph 的 **SqliteSaver**(或 RedisSaver)按 `trace_id` 存每一节点的状态。

**验收标准(P4):** 在 `detect_gap` 节点中途 `kill -9` 进程,重启后能从 `update_memory` 的产物继续,**不重跑前面的 LLM 调用**。

> 这正是上个项目手写 orchestrator 做不到的事,也是选 LangGraph 的核心理由。

---

## 7. 复用现有代码

| 能力 | 复用来源 | 说明 |
|---|---|---|
| LLM 调用 | [llm/llm_client.py](llm/llm_client.py) | 尤其新加的 `chat_structured()`(带 token usage) |
| JSON 解析 | [llm/output_parser.py](llm/output_parser.py) | 建图时解析 LLM 抽取结果 |
| 向量存储 | [rag_core/chroma_store.py](rag_core/chroma_store.py) | 题目/教材语义检索 |
| 向量化 | [rag_core/api_embedder.py](rag_core/api_embedder.py) | — |
| Redis 客户端模式 | [core/persistent_cache.py](core/persistent_cache.py) | Stream 封装可参考其连接与降级写法 |
| 重试 | [core/error_handler.py](core/error_handler.py) | worker 内的重试策略 |
| 监控 | [core/unified_monitoring.py](core/unified_monitoring.py) | ⚠️ 进程内单例,**多 worker 下需换成 Redis sink** |

**新增依赖(按阶段):**
```
# P0 只需要这些
fastapi, uvicorn[standard]     # API
redis                          # MQ —— 项目已有使用经验
aiosqlite / sqlite3            # 画像 + 图(标准库 sqlite3 即可)
httpx                          # worker 内异步 HTTP(如需)

# 后续按需
fsrs            # P3:遗忘模型
langgraph       # P4:工作流 + checkpointer
kuzu            # P2+:仅当 SQLite CTE 不够用
```

> **P0 不引入图库、不引入工作流框架。** 存储只用 SQLite,编排只用 async 函数。
> 换实现时只动 `graph/store.py` 和 `workflow/`,调用方不变。

---

## 8. 运行与部署

**开发(单进程拉起所有 worker):**
```bash
# 终端 1:API
uvicorn coach.api.main:app --reload --port 8000

# 终端 2:全部 worker
python -m coach.workers.run_all
```

**生产形态(docker-compose):**
```yaml
services:
  redis:      { image: redis:7-alpine }
  api:        { build: ., command: uvicorn coach.api.main:app --host 0.0.0.0 }
  worker-profile:   { build: ., command: python -m coach.workers.profile_worker, deploy: {replicas: 2} }
  worker-planner:   { build: ., command: python -m coach.workers.planner_worker }
  worker-scheduler: { build: ., command: python -m coach.workers.scheduler_worker }
  worker-generator: { build: ., command: python -m coach.workers.generator_worker, deploy: {replicas: 2} }
```

> **worker 之间不要共享内存状态。** 所有状态在 Redis / SQLite / Kuzu 里。这是能水平扩展的前提。

---

## 9. 评测设计

承接上个项目的思路——**没有评测的 agent 项目只是 demo**。

| 评测对象 | 方法 | 指标 | 数据来源 |
|---|---|---|---|
| **画像准确性** | 用前 80% 答题记录拟合,预测后 20% 的"下次答对概率" | **AUC / Brier score** | 模拟学生 or 真人记录 |
| **遗忘预测** | 预测"何时逾期未复习",与实际对比 | 校准曲线 / MAE | 同上 |
| **规划有效性** | A/B:C 图谱规划 vs B 随机推荐 vs A 顺序推进,比较后续正确率 | 正确率提升 | 模拟学生 |
| **GraphRAG 价值** ★ | 缺口检测任务:图遍历 vs 纯向量召回 | 根因定位准确率 | 人工标注的"某学生某题错因" |

**第四项是核心卖点。** 它能回答"你为什么要用图"。
**如果测出来图没比向量强多少,就诚实写进 Limitations** —— 这比编一个好看的数字可信得多。

**模拟学生怎么造:** 给每个知识点预设一个"真实掌握度",按掌握度概率生成答题结果(掌握了就有 90% 概率答对)。这样**你知道 ground truth**,可以精确评估画像模型有多准。

---

## 10. 测试策略

| 层 | 测什么 | 必测 |
|---|---|---|
| graph | 环检测拒绝;前置闭包深度正确;孤立节点 | ★ |
| profile/bkt | 连续答对 → p_known 单调上升;连续答错 → 下降 | ★ |
| profile/fsrs | due_at 随 stability 变化;答错后 stability 回退 | ★ |
| mq | 幂等(重复 event_id 只处理一次);超过重试上限进 DLQ | ★ |
| workflow | 中断后从 checkpointer 恢复 | ★ |
| api | 契约(字段、状态码);`/answer` 不含 LLM 调用(可用 mock 断言) | ★ |
| e2e | 提交答案 → 画像变化 → /plan 输出变化 | ★ |

---

## 11. 分阶段路线(精简版)

### 11.1 精简原则

原设计有 **4 个全新子系统**(图库、消息队列、常驻 worker、工作流引擎)。同时上四个新东西 = 四个新故障域,必崩。

三条原则:

1. **垂直切片** —— 先打通一条最窄的端到端链路,而不是按层做完再叠层
2. **先简后繁** —— 每个组件先用最简实现,跑通了再换重装
3. **接口封装,实现可换** —— 换实现时不动调用方

### 11.2 具体裁剪

| 组件 | 原设计 | 精简后 | 后加时机 | 理由 |
|---|---|---|---|---|
| **图存储** | Kuzu | **SQLite 递归 CTE** | P2 视需要迁 Kuzu | 少一个依赖;闭包查询 CTE 够用 |
| **worker** | 4 个独立进程 | **1 进程 + 2 个 async 任务** | P3 拆 | 开发期不用开 4 个终端 |
| **遗忘模型** | FSRS | **固定间隔(1/3/7 天)** | P3 换 FSRS | 先验证"知道该复习什么"成立 |
| **工作流** | LangGraph | **线性 async 函数链** | P4 换 LangGraph | 断点恢复是 P4 才需要的能力 |
| **推送** | WebSocket | **轮询 `GET /plan`** | P3 加 WS | 少一整套连接管理 |
| **定时** | 独立 scheduler | **手动触发** | P3 加定时 | — |
| **评测** | 4 张表 | **先做画像 AUC** | P5 补全 | 一张表就能证明核心 |
| **部署** | docker-compose 6 服务 | **2 个终端** | P6 容器化 | — |
| **知识点规模** | 30~50 | **10~15 先跑通** | 逐步扩 | 建图是最大工作量 |

> **最关键的一刀:SQLite 递归 CTE 替代 Kuzu。**
> 画像本来就在 SQLite,图也放进去,**存储引擎从 3 个(SQLite + Kuzu + Chroma)降到 2 个**。
> 递归 CTE 就是图遍历,GraphRAG 的方法论完全不变:
> ```sql
> WITH RECURSIVE prereqs(kp_id, depth) AS (
>     SELECT $x, 0
>     UNION
>     SELECT e.from_kp, p.depth + 1
>     FROM prereq_edge e JOIN prereqs p ON e.to_kp = p.kp_id
>     WHERE p.depth < 3
> )
> SELECT kp_id, MIN(depth) FROM prereqs WHERE kp_id != $x GROUP BY kp_id;
> ```

### 11.3 新的 P0 —— 最小可演示的垂直切片

```
[教材/大纲] → LLM 抽 10~15 个知识点 + 前置关系 → 环检测 → SQLite
                                                          │
POST /answer ──► Redis Stream ──► 1 个 worker ──► 判分(规则比对 test_cases)
                                                      │
                                                      ├─► BKT 更新掌握度 → SQLite
                                                      │
                                                      └─► 若答错:CTE 查前置闭包
                                                                ∩ 掌握度 < 0.4
                                                                → 定位缺口 → 返回
```

**跑通这条链,8 个功能里到位 6 个**:FastAPI、消息队列、异步、用户画像、GraphRAG、常驻 loop。
只差"工作流"和"主动推送"的正式版。

**P0 验收标准(这就是第一道验收线):**
- `POST /answer` **< 50ms 返回** —— 证明是真异步,不是假异步
- 答错某题后,`GET /gap/{learner}/{kp}` 能指出一个前置知识点
- `git clone` + 2 条命令能跑起来

### 11.4 完整路线

| 阶段 | 交付 | 新增能力 | 验收 |
|---|---|---|---|
| **P0** | 垂直切片(§11.3) | FastAPI + MQ + 异步 + 画像 + GraphRAG + loop | 答错能定位前置缺口 |
| **P1** | 扩到 30 点;补 BKT 单测 | — | 画像准确率有初步数字 |
| **P2** | `/plan` 路径规划;视需要迁 Kuzu | 路径规划 | 能给出"下一步学什么" |
| **P3** | FSRS + 定时 + WebSocket | 遗忘模型 + 主动推送 | 到期知识点主动推给前端 |
| **P4** | LangGraph + checkpointer | 正式工作流 | kill -9 后能恢复 |
| **P5** | 评测补全(4 张表) | 评测 | README 有结果表 |
| **P6** | docker-compose + demo | 交付 | 别人能跑 |

> **P0 就是第一道验收线,不用等全做完。**
> 迁移本身也是面试素材:"我先用 50 行手写打通,后来发现需要断点恢复才引入 LangGraph"——比"一上来就上框架"更有说服力。

---

## 12. 已知风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| **前置依赖数据要从零抽** | 建图慢、质量不稳 | P0 只做 30~50 点,人工校验;LLM 抽取 + 规则过滤 + 环检测 |
| **没有真实用户** | 评测无数据 | 用模拟学生(可控制 ground truth),§9 有方案 |
| **图存储选型** | 卡住 | P0 用 **SQLite 递归 CTE**(零新依赖);需要 Cypher 表达力时再迁 Kuzu,接口封在 `graph/store.py` |
| **LangGraph API 变动快** | 返工 | 只在 `workflow/` 内使用,不外泄;节点函数保持纯函数 |
| **图可能是过度设计** | 卖点站不住 | §9 第四项评测给出诚实答案 |
| **WebSocket 在多 worker 下要广播** | 消息推不到正确实例 | 用 Redis Pub/Sub 做 WS 广播层(worker → Redis → 各 API 实例) |
| **现有 repo 的 `global_monitor` 非线程安全** | 多 worker 指标丢失 | 换成 Redis 计数器,或直接引入 Prometheus |

---

## 13. 待确认的设计决策

| # | 决策 | 选项 | 我的建议 |
|---|---|---|---|
| 1 | 图存储 | SQLite CTE / Kuzu / Neo4j | ✅ **已改:P0 用 SQLite 递归 CTE** —— 零新依赖,存储引擎从 3 个降为 2 个;需要 Cypher 时再迁 Kuzu |
| 2 | FSRS 实现 | 开源 `fsrs` 包 / 自实现 | **开源包** — 别自己调参;但公式要写进文档 |
| 3 | 领域小图的第一批知识点 | ~~算法入门 / Python 基础 / 数据结构~~ | ✅ **已确认:数据结构 + 算法入门** |
| 4 | 判分方式 | ~~纯规则 / LLM 判主观 / 混合~~ | ✅ **已确认:纯规则判题**(输入比对 test_cases),判分确定、可重现,评测才可信 |
| 5 | 是否做 Web 前端 | 做 / 只用 Swagger + wscat | **先不做**,Swagger 演示足够;P6 再看 |
| 6 | 模拟学生是否进 repo | 进(作为测试工具)/ 不进 | **进** — 它是评测的基础设施,也是"我认真做了评测"的证据 |
| 7 | 体量控制 | 一次全上 / 垂直切片 | ✅ **已确认:垂直切片**(见 §11),P0 工作量约原设计 1/3 |

---

## 14. 一句话总结设计

> **用图管"知识之间的关系",用数学模型管"你和知识的关系",用消息队列把两者异步解耦,用工作流保证长流程可恢复,最后用评测证明这套东西真的有用。**

四个组件各司其职,缺一不可——这就是它区别于"又一个 RAG 问答机器人"的地方。
