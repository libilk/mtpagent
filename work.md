# work.md — 实施手册与进度台账

> **这份文档是什么:** 项目的**操作手册**。回答"怎么做、做到哪一步了、卡在哪"。
> [mainconten.md](mainconten.md) 回答**为什么做、做什么**(定位与设计总纲),本文回答**怎么落地**。
>
> **怎么用(重要):**
> 1. 每完成一个任务,把 `- [ ]` 改成 `- [x]`
> 2. 每完成一个阶段,更新 §7 进度台账,并在 §10 更新记录里加一行
> 3. 很久没碰之后,直接跳到 §9 恢复指南
>
> **最后更新:** 2026-09-13
> **当前阶段:** **P3 已完成**(计划接口可用),下一步 P4 工作流(LangGraph + checkpointer)

---

## 1. 项目速览

**一句话:** 用知识图谱多跳推理定位"学不会的根因"的学习 Agent,并以量化评测证明图推理相比扁平检索的增益。

**当前状态:** **P0~P3 已完成**,并已补齐跑通所需的零件(题目库、tick 生产者、LLM 装配、学习者建档)。
截至 2026-09-13:`coach/` 约 4k 行,coach 测试 186 项 / 全量 336 项。
未做:P4 工作流(换 LangGraph)、P5 评测、P6 交付。

**唯一还没验的:**
- **P1 的端到端验收欠一次真 Redis 实跑**(开发机 Docker 未启动)。目前 `<50ms` 是进程内
  总线实测,`kill -9` 恢复是逻辑验证 —— 见 §10 台账。
- LLM 抽取链路已用真实调用验证过(含 id 漂移修复的复验)。

**已有的可复用资产(来自上一个项目):**

| 资产 | 路径 | 用途 |
|---|---|---|
| LLM 客户端 | [llm/llm_client.py](llm/llm_client.py) | 含 `chat_structured()`(返回 tool_calls + usage) |
| JSON 解析 | [llm/output_parser.py](llm/output_parser.py) | 解析 LLM 抽取结果 |
| 重试 | [core/error_handler.py](core/error_handler.py) | worker 内重试 |
| 向量(暂不用) | [rag_core/chroma_store.py](rag_core/chroma_store.py) | P5 若需要再加 |
| 归档参考 | [zanshibuyong/](zanshibuyong/) | 旧 orchestrator / MCP 实现,可参考 |

**环境:** Python 虚拟环境在 `.venv/`(当前 3.14.5)。Redis 需单独启动。

---

## 2. 架构

### 2.1 分层图

```
┌──────────────────────────────────────────────────────────────┐
│ ① 入口层  coach/api/                                          │
│   FastAPI:只做「校验 + 入队 + 查询」,**不含任何 LLM 调用**       │
│   POST /answer · GET /plan · GET /gap · GET /profile · /graph │
└────────────────────────┬─────────────────────────────────────┘
                         │ publish(event)
                         ▼
┌──────────────────────────────────────────────────────────────┐
│ ② 事件与协调层  coach/events/ + coach/coordination/            │
│   bus.py       事件总线                                       │
│   redis.py     Redis Stream(消费组)                          │
│   journal.py   ★ append-only 事件日志(先记后做)               │
│   recovery.py  ★ 重启后扫 journal + pending,补做未完成事件     │
└────────────────────────┬─────────────────────────────────────┘
                         │ consume
                         ▼
┌──────────────────────────────────────────────────────────────┐
│ ③ 常驻 worker  coach/workers/  (asyncio,可水平扩展)            │
│   profile_worker    判分(规则) → BKT → SM-2                   │
│   planner_worker    ★ 根因定位(多跳图查询)/ 路径规划           │
│   scheduler_worker  SM-2 到期扫描 → 生成复习计划                │
│   run_all.py        开发期单进程拉起全部                        │
└───────┬────────────────────┬─────────────────────────────────┘
        │                    │
        ▼                    ▼
┌──────────────────┐  ┌──────────────────────────────────────┐
│ ④ 知识层          │  │ ⑤ 画像层  coach/profile/              │
│  coach/knowledge/ │  │   bkt.py    掌握度                     │
│   图(SQLite)      │  │   sm2.py    间隔重复                   │
│   递归 CTE 遍历   │  │   errors.py 易错模式                   │
│   ★ 写入治理:     │  │   store.py  SQLite 持久化              │
│   observation     │  └──────────────────────────────────────┘
│   → proposal      │
│   → aggregator    │  ┌──────────────────────────────────────┐
└──────────────────┘  │ ⑥ 编排层  coach/workflow/             │
                      │   pipeline.py 线性 async 链(P4 换      │
                      │   LangGraph + checkpointer)           │
                      └──────────────────────────────────────┘
```

### 2.2 一次答题的完整数据流

```
客户端
  │ POST /answer {learner_id, problem_id, answer_text, elapsed_ms}
  ▼
[api] 校验 → 生成 event_id(ULID)→ publish(coach:answer)→ 202 返回   ← 目标 < 50ms
  │
  ▼
[coordination] Redis Stream 消费组取消息
  │ → 幂等检查(processed_events)
  │ → 写 journal(append-only)
  ▼
[workflow] 线性链开始
  │
  ├─[1] grade          规则比对 test_cases → correct: bool
  ├─[2] update_mastery BKT 更新 P(known)
  ├─[3] update_memory  SM-2 更新 stability + due_at
  ├─[4] detect_gap     ★ 递归 CTE 查前置闭包 ∩ 掌握度 < 0.4
  │      ├─ 有缺口 → remedial(补救路径)
  │      └─ 无缺口 → advance(推进下一节点)
  ├─[5] plan           合成今日计划
  └─[6] enqueue        异步出题(publish coach:generate)
  │
  ▼
[coord] ACK;失败 → attempt+1;>3 → 死信 + 告警
  ▼
commit_offset → 完成
```

### 2.3 模块职责表

| 模块 | 职责 | 不负责 |
|---|---|---|
| `api/` | HTTP 入口、校验、入队 | LLM、判分、图查询 |
| `events/` | 事件定义、总线 | 持久化 |
| `coordination/` | Redis 收发、幂等、journal、recovery | 业务逻辑 |
| `workers/` | 常驻消费、编排调用 | HTTP |
| `knowledge/` | 图的存取、遍历、写入治理 | 画像 |
| `profile/` | 掌握度、记忆、易错、持久化 | 图 |
| `workflow/` | 步骤编排、状态流转 | 存储细节 |
| `evaluation/` | 模拟学生、指标、跑分 | 生产逻辑 |

**硬规矩:依赖只能向下,`api/` 不许 import `knowledge/` 或 `profile/`。**

---

## 3. 技术栈与选型

| 层 | 选型 | 为什么 | 备选(何时换) |
|---|---|---|---|
| 语言 | Python 3.11+ | 项目已有 `.venv` | — |
| Web | **FastAPI** + Uvicorn | 原生 async,与 worker 同构 | — |
| 图存储 | **SQLite + 递归 CTE** | 零新依赖;规模小够用;接口封装可平移 | Kuzu / Neo4j(P2 后若图变复杂) |
| 画像存储 | **SQLite**(同一文件) | 与图同库,少一个依赖 | Postgres(多用户时) |
| 队列 | **Redis Stream** | 项目已有 Redis 经验;不引入 Kafka | Kafka(真需要高吞吐时) |
| 可靠投递 | **journal + recovery**(自实现) | 保证"不丢"的关键 | — |
| 掌握度 | **BKT** | 可解释、可评测 | DKT(需要更多数据) |
| 间隔重复 | **SM-2** | WeSmartFlow 验证过,够用 | FSRS(优化项) |
| 工作流 | 线性 async 链 → **LangGraph** | 先跑通,断点恢复是 P4 的需求 | Temporal(工业级) |
| LLM | 复用 `llm/llm_client.py` | 已修好协议 | — |
| 测试 | pytest | 已有 | — |
| 新增依赖 | `fastapi` `uvicorn[standard]` `redis` | 仅此三样(已装) | P4 需要 `langgraph` —— 待确认是否加第 4 样 |

> **P0~P3 不引入**:图库、工作流框架、向量库、前端、Docker。

---

## 4. 数据模型(定稿)

### 4.1 知识图谱(SQLite)

```sql
-- 知识点
CREATE TABLE concepts (
    id          TEXT PRIMARY KEY,      -- "algo.dp"
    name        TEXT NOT NULL,         -- "动态规划"
    subject     TEXT NOT NULL,         -- "algorithms"
    difficulty  REAL DEFAULT 3.0,      -- 1.0 ~ 5.0
    description TEXT,
    created_at  REAL
);

-- 边:四类关系
CREATE TABLE edges (
    from_id     TEXT NOT NULL,
    to_id       TEXT NOT NULL,
    type        TEXT NOT NULL,         -- PREREQUISITE|RELATED|EXTENDS|CONTRASTS
    weight      REAL DEFAULT 1.0,
    confidence  REAL DEFAULT 1.0,      -- 抽取置信度
    source      TEXT,                  -- 来源:llm_extract|manual|outline
    PRIMARY KEY (from_id, to_id, type)
);
CREATE INDEX idx_edges_to ON edges(to_id, type);   -- ★ 反向遍历用

-- 题目
CREATE TABLE problems (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    source      TEXT,                  -- "leetcode:70"
    difficulty  REAL,
    judge_type  TEXT DEFAULT 'exact_output',
    test_cases  TEXT,                  -- JSON: [{"input":...,"expected":...}]
    kp_ids      TEXT                   -- JSON 数组
);

-- ★ 写入治理:观察 → 提案
CREATE TABLE observations (
    id          TEXT PRIMARY KEY,
    kind        TEXT,                  -- "edge" | "concept"
    payload     TEXT,                  -- JSON
    source      TEXT,                  -- 从哪抽的
    observed_at REAL
);

CREATE TABLE proposals (
    id          TEXT PRIMARY KEY,
    observation_id TEXT,
    payload     TEXT,                  -- 待入库的变更
    status      TEXT,                  -- pending|accepted|rejected
    reason      TEXT,                  -- 拒绝原因
    created_at  REAL
);
CREATE INDEX idx_proposals_status ON proposals(status, created_at);  -- aggregator 捞 pending 用(实现时补充)
```

### 4.2 画像(SQLite,同一文件)

```sql
CREATE TABLE learner_mastery (
    learner_id TEXT, kp_id TEXT,
    p_known REAL, updated_at REAL,
    PRIMARY KEY (learner_id, kp_id)
);

CREATE TABLE learner_memory (          -- SM-2 状态
    learner_id TEXT, kp_id TEXT,
    ease REAL DEFAULT 2.5,             -- EF
    interval_days REAL DEFAULT 0,      -- I
    reps INTEGER DEFAULT 0,
    lapses INTEGER DEFAULT 0,
    last_review REAL, due_at REAL,
    PRIMARY KEY (learner_id, kp_id)
);

CREATE TABLE learner_error (
    learner_id TEXT, kp_id TEXT, error_type TEXT,
    count INTEGER DEFAULT 0,
    PRIMARY KEY (learner_id, kp_id, error_type)
);

CREATE TABLE learner_profile (
    learner_id TEXT PRIMARY KEY,
    goal_kp_id TEXT, daily_minutes INTEGER DEFAULT 30,
    preference TEXT, created_at REAL
);

CREATE TABLE answers (
    answer_id TEXT PRIMARY KEY, learner_id TEXT, problem_id TEXT,
    kp_ids TEXT, correct INTEGER, answer_text TEXT,
    elapsed_ms INTEGER, ts REAL
);
CREATE INDEX idx_answers_learner ON answers(learner_id, ts);

-- ★ 根因解释缓存(P2 补充):planner_worker 异步调 LLM 生成,
--   GET /gap 只读这里,保证入口层不碰 LLM
CREATE TABLE gap_explanations (
    learner_id TEXT, kp_id TEXT,
    explanation TEXT, generated_at REAL,
    PRIMARY KEY (learner_id, kp_id)
);

-- ★ 幂等表
CREATE TABLE processed_events (
    event_id TEXT PRIMARY KEY, ts REAL
);

-- ★ append-only 事件日志(P1 补充,由 coordination/journal.py 建)
--   注意:这张表不在本节的"画像"范围内,归协调层所有
CREATE TABLE journal (
    event_id TEXT PRIMARY KEY, type TEXT NOT NULL, trace_id TEXT,
    learner_id TEXT, payload TEXT, recorded_at REAL NOT NULL, done_at REAL
);
```

### 4.3 事件信封

```json
{
  "event_id": "01HXYZ...",
  "type": "answer.submitted",
  "ts": 1757500000.123,
  "learner_id": "u_001",
  "trace_id": "01HABC...",
  "attempt": 1,
  "payload": {}
}
```

| type | 生产者 | 消费者 |
|---|---|---|
| `answer.submitted` | api | profile_worker |
| `profile.updated` | profile_worker | planner_worker |
| `tick.scheduled` | scheduler_worker | planner_worker |
| `plan.updated` | planner_worker | api(推送) |
| `problem.generate` | workflow | generator(P3+) |

---

## 5. 核心技术要点(实现时照抄)

### 5.1 前置闭包(递归 CTE)

```sql
WITH RECURSIVE prereqs(kp_id, depth) AS (
    SELECT :x, 0
    UNION
    SELECT e.from_id, p.depth + 1
    FROM edges e JOIN prereqs p ON e.to_id = p.kp_id
    WHERE e.type = 'PREREQUISITE' AND p.depth < :max_depth
)
SELECT kp_id, MIN(depth) AS depth
FROM prereqs WHERE kp_id != :x
GROUP BY kp_id ORDER BY depth;
```

### 5.2 环检测(DAG 校验)

```python
def has_cycle(edges: list[tuple[str, str]]) -> bool:
    """edges: [(from, to)] 仅 PREREQUISITE。DFS 三色标记"""
    graph = defaultdict(list)
    for a, b in edges:
        graph[a].append(b)
    WHITE, GRAY, BLACK = 0, 1, 2
    color = defaultdict(int)
    def dfs(u):
        color[u] = GRAY
        for v in graph[u]:
            if color[v] == GRAY:
                return True
            if color[v] == WHITE and dfs(v):
                return True
        color[u] = BLACK
        return False
    return any(color[u] == WHITE and dfs(u) for u in list(graph))
```

### 5.3 BKT 掌握度更新

```
参数(每知识点): P_init=0.1  T=0.15  G=0.2  S=0.1

答对:  P(L|correct) = P(L)(1−S) / [ P(L)(1−S) + (1−P(L))·G ]
答错:  P(L|wrong)   = P(L)·S   / [ P(L)·S   + (1−P(L))(1−G) ]
转移:  P_next = P(L|obs) + (1 − P(L|obs)) · T
```

### 5.4 SM-2 间隔重复

```
状态: EF(难度因子,默认2.5), I(间隔天数,默认0), reps, lapses

评分 q ∈ {0..5}(答对程度):
  q < 3(失败): reps=0, I=1, lapses+=1
  q ≥ 3(通过):
     reps == 1 → I = 1
     reps == 2 → I = 6
     else      → I = round(I_prev * EF)
     EF = EF + (0.1 − (5−q)·(0.08 + (5−q)·0.02))
     EF = max(1.3, EF)

下次到期: due_at = now + I 天
```

### 5.5 根因定位(★ 项目核心)

```
输入: learner_id, 目标知识点 X
1. P ← 前置闭包(X, max_depth=3)              # §5.1
2. M ← {p: mastery(p) for p in P}            # 一次性批量查,别 N+1
3. Gaps ← {p ∈ P : M[p] < 0.4}
4. 排序:按 (depth 升序, mastery 升序)         # 最浅且最弱 = 根因
5. root_causes ← 取前 1~3 个
6. explanation ← LLM 组织(根因 + 依赖路径 + 证据题)
```

### 5.6 幂等消费契约(每个 worker 必须遵守)

```
1. XREADGROUP 取消息(consumer group)
2. 查 processed_events:已存在 → 直接 ACK,跳过
3. 写 journal(append-only)——先记后做
4. 执行业务
   ★ 业务若有多步写,必须放在**同一个事务**里。否则"有一步写成功、后续失败"
     加上"恢复重放看到第一步已落库而跳过"= 该更新永久丢失(P1 实现时踩到,已修)
5. 写 processed_events + XACK
6. 失败 → attempt += 1
   ├─ ≤ 3 → 重新投递
   └─ > 3 → 移入 dlq + 告警
7. 重启 → recovery 扫 journal 中「已记未完成」的事件,补做
```

**幂等要做两道闸**(只做第 2 步不够):
- `processed_events` 挡「同一消息被消费两次」
- 业务侧的唯一键(如 `answers.answer_id`)挡「恢复重放把同一步执行两次」

---

## 6. 目录结构(目标形态)

```
coach/
├── config.py                    # ✅ 路径 + 全部算法默认参数
├── domain/
│   ├── models.py                # ✅ 领域模型
│   ├── ids.py                   # ✅ ULID(§4.3 要求)
│   └── grading.py               # ✅ 规则判分(P1.9)
├── knowledge/
│   ├── schema.py                # ✅ §4.1 DDL + connect()/init_schema()
│   ├── store.py                 # ✅ 存取 + §5.1 递归 CTE
│   ├── queries.py               # ✅ ★ §5.5 根因定位(纯函数)
│   ├── governance.py            # ✅ ★ observation → proposal → aggregator
│   └── builder.py               # ✅ LLM 抽取 + §5.2 环检测 + 种子建图
├── profile/
│   ├── bkt.py                   # ✅ §5.3
│   ├── sm2.py                   # ✅ §5.4
│   ├── errors.py                # —  P2+ 视需要再拆,当前易错计数在 store.py
│   └── store.py                 # ✅ §4.2 建表 + 读写 + 幂等表
├── events/
│   ├── schema.py                # ✅ 事件信封 + 类型
│   └── bus.py                   # —  暂无进程内总线需求(P3 用轮询)
├── coordination/
│   ├── redis.py                 # ✅ Redis Stream / 消费组 / 死信
│   ├── journal.py               # ✅ ★ append-only
│   └── recovery.py              # ✅ ★ 重启恢复(按事件类型分流)
├── workers/
│   ├── profile_worker.py        # ✅ 判分 → BKT → SM-2
│   ├── planner_worker.py        # ✅ ★ 根因定位 + LLM 解释
│   ├── scheduler_worker.py      # ✅ SM-2 到期扫描 → 计划
│   └── run_all.py               # ✅ 开发期单进程拉起
├── workflow/
│   ├── query.py                 # ✅ 只读门面(api 依赖它,不直接依赖图/画像)
│   └── pipeline.py              # —  P4 做
├── evaluation/                  # —  P5 做
│   ├── simulated_student.py
│   ├── metrics.py
│   └── run_eval.py
└── api/
    ├── main.py                  # ✅ create_app(bus=, query_service=)
    ├── routes.py                # ✅ /answer /gap /graph /plan /health
    └── schemas.py               # ✅
```

---

## 7. 分阶段任务清单

> 约定:每个阶段结束必须**可演示**。不要憋大招。

### P0 — 图底座 + 写入治理

**目标:** 建起 15 个知识点的小图,前置依赖关系经过治理流程,能查询前置闭包。

- [x] P0.1 建 `coach/` 目录骨架 + `config.py`
- [x] P0.2 `domain/models.py`:KnowledgePoint / Edge / Problem / Observation / Proposal
- [x] P0.3 `knowledge/schema.py`:§4.1 建表 DDL + 索引
- [x] P0.4 `knowledge/store.py`:
  - [x] 增删查(concept / edge / problem)
  - [x] `ancestors(kp_id, depth) -> dict[str,int]`(§5.1 递归 CTE)
  - [x] `descendants(kp_id, depth)`
- [x] P0.5 `knowledge/governance.py`:
  - [x] `observe(payload)` → 写 observations
  - [x] `propose(observation_id)` → 规则校验(自环/重复/confidence<0.6/引用不存在的知识点/同名归并)→ 写 proposals
  - [x] `aggregate(proposal_id)` → §5.2 环检测 → 通过则写 edges,否则 status=rejected
- [x] P0.6 `knowledge/builder.py`:LLM 从教材/大纲抽取(复用 `llm/llm_client.py`)
- [x] P0.7 录入 **15 个知识点 + 前置关系**(数据来源见 [mainconten.md](mainconten.md) §5.1)
- [x] P0.8 测试 `tests/coach/test_knowledge.py`:
  - [x] 环检测:构造 A→B→C→A,断言拒绝
  - [x] 自环拒绝
  - [x] 前置闭包:深度正确、去重、不含自身
  - [x] 治理:被拒提案不进 edges
  - [x] 悬空引用拒绝(既无 from 也无 to 的知识点)
  - [x] 种子图:无环 + `ancestors("algo.dp",3)` + 重跑幂等 + `--seed` 截断不留悬空边

**交付物:** 一个 SQLite 图文件 + 可查询的闭包接口
**验收标准:**
- `ancestors("algo.dp", 3)` 返回正确的前置集合
- 图中无环(用 §5.2 在测试里断言)
- 能演示一条 observation → proposal → aggregator 的完整链路

**依赖:** 无(起点)

---

### P1 — 入口 + 可靠队列 + 画像

**目标:** `POST /answer` 毫秒级返回,事件异步被消费,画像被更新,进程崩溃能恢复。

- [x] P1.1 `events/schema.py`:事件信封 + 类型定义
- [x] P1.2 `coordination/redis.py`:Stream 封装
  - [x] `publish(stream, event)` / `consume(...)` / `ack(...)`
  - [x] consumer group 初始化
  - [x] 死信转移(attempt > 3)
- [x] P1.3 `coordination/journal.py`:append-only 日志
  - [x] `record(event)`(先记后做)
  - [x] `mark_done(event_id)`
  - [x] `pending()` → 返回「已记未完成」的事件
- [x] P1.4 `coordination/recovery.py`:启动时扫 pending 补做
- [x] P1.5 `profile/bkt.py`:§5.3
- [x] P1.6 `profile/store.py`:§4.2 建表 + 读写
- [x] P1.7 `workers/profile_worker.py`:消费 → 幂等检查 → journal → 判分 → BKT → ACK
- [x] P1.8 `api/`:FastAPI app + `POST /answer` + Swagger
- [x] P1.9 判分:规则比对 `test_cases`(先做简单 equal 比较)
- [x] P1.10 测试:
  - [x] BKT:连续答对 → p_known 单调上升;连续答错 → 下降
  - [x] 幂等:同一 event_id 投两次,画像只更新一次
  - [x] recovery:模拟进程重启,pending 事件被补做
  - [x] `POST /answer` 不触发 LLM(用 mock 断言)
- [x] P1.11 `workers/run_all.py`:开发期单进程拉起(启动时先跑 recovery)
- [x] P1.12 额外测试:`domain/grading.py` 判分、`api/` 分层不越界(禁 import knowledge/profile/llm)

**交付物:** 可跑的 API + 队列 + worker
**验收标准:**
- `POST /answer` **< 50ms 返回**(实测,不是估计)
- 提交答题后,`learner_mastery` 表被异步更新
- `kill -9` worker 后重启,未完成事件被 recovery 补做

**依赖:** P0

---

### P2 — 根因定位(★ 核心卖点)

**目标:** 答错后能指出"你真正缺的是哪个前置知识点"。

- [x] P2.1 `knowledge/queries.py`:
  - [x] `prereq_closure(kp_id, depth)`
  - [x] `detect_gaps(...)` → §5.5(**签名改为收掌握度映射**:knowledge 不许 import profile)
  - [x] `root_causes(...)`(每个根因带 `path`)
  - [x] `next_to_learn(...)`(前置全满足、与目标路径最近)
  - [x] `shortest_prereq_path` / `explain_gaps`(模板兜底,不调 LLM)
- [x] P2.2 `workers/planner_worker.py`:消费 profile.updated / tick.scheduled
- [x] P2.3 API:
  - [x] `GET /gap/{learner_id}/{kp_id}` → §5.5 输出
  - [x] `GET /graph/{kp_id}?depth=3` → 前置树 + 已掌握/缺口标记
  - [x] `workflow/query.py` 只读门面(解 api 不能 import knowledge/profile 的约束)
- [x] P2.4 LLM 把根因组织成人话(复用 `llm/llm_client.py`,**异步生成 + 缓存**)
- [x] P2.5 测试:构造已知掌握度分布,断言根因定位到预期节点
- [x] P2.6 `recovery.replay(types=...)`:多 worker 共用一个 journal 时按类型分流

**交付物:** 根因定位接口
**验收标准:**
- 构造一个"递归掌握 0.9、函数调用掌握 0.2、动态规划掌握 0.3"的学生,
  `GET /gap/u/算法.动态规划` 必须把 **函数调用** 报为根因
- `/graph` 返回的前置树深度正确

**依赖:** P0、P1

---

### P3 — 遗忘模型 + 常驻调度

**目标:** 系统主动发现"该复习了",生成今日计划。

- [x] P3.1 `profile/sm2.py`:§5.4
- [x] P3.2 `workers/scheduler_worker.py`:定时(或手动触发)扫 `due_at <= now`
- [x] P3.3 `GET /plan/{learner_id}`:合并「到期复习」+「根因补救」+「可推进新知识点」
- [x] P3.4 推送(先用轮询,不上 WebSocket):scheduler 发 `plan.updated` 到 `coach:plan`
- [x] P3.5 测试:SM-2 的 I / EF 演化符合预期;到期知识点出现在计划里
- [x] P3.6 把 SM-2 接进 profile_worker(答题后更新 reps/interval/due_at)

**交付物:** 每日学习计划接口
**验收标准:** 把某知识点的 `due_at` 设成过去,调用 `/plan` 能看到它排进"复习"

**依赖:** P1、P2

---

### P4 — 工作流(引入 LangGraph)

**目标:** 把线性链换成带 checkpointer 的工作流,支持中断恢复。

- [ ] P4.1 抽出 `workflow/pipeline.py` 的 State 定义
- [ ] P4.2 LangGraph 图:节点 = §2.2 的 [1]~[6],条件边在 `detect_gap`
- [ ] P4.3 checkpointer(先 SqliteSaver 或 RedisSaver)
- [ ] P4.4 测试:`kill -9` 后从断点恢复,**不重跑已完成的 LLM 调用**(用调用计数断言)
- [ ] P4.5 记录迁移过程(面试素材)

**交付物:** 可恢复的工作流
**验收标准:** 在 `detect_gap` 中途杀进程,恢复后 trace 显示前序节点未重跑

**依赖:** P1~P3

---

### P5 — 评测(★ 差异化所在)

**目标:** 四张表,其中一张回答"图 vs 向量强多少"。

- [ ] P5.1 `evaluation/simulated_student.py`:
  - [ ] 给每个知识点预设"真实掌握度"(ground truth)
  - [ ] 按掌握度概率生成答题结果(掌握了有 90% 答对)
  - [ ] 支持生成 N 个学生 × M 次答题的轨迹
- [ ] P5.2 `evaluation/metrics.py`:
  - [ ] AUC / Brier(掌握度预测)
  - [ ] 校准曲线(遗忘预测)
  - [ ] 提升幅度(规划 vs 随机)
  - [ ] ★ 根因定位准确率(图遍历 vs 纯向量召回)
- [ ] P5.3 `evaluation/run_eval.py`:一键跑出四张表
- [ ] P5.4 **对照组实现**:纯向量版根因定位(用 Chroma 或简单相似度)
- [ ] P5.5 结果落 `evaluation/results/*.json` + 生成 markdown 表

**交付物:** 四张评测表
**验收标准:**
- `python -m coach.evaluation.run_eval` 能跑出结果
- 掌握度预测 AUC 有数字(多少不重要,**是不是真算出来的**重要)
- 图 vs 向量的对照有结论(哪怕结论是"图没强多少")

**依赖:** P0~P3

---

### P6 — 交付

- [ ] P6.1 README(结构参考 [mainconten.md](mainconten.md) §7 的叙事)
- [ ] P6.2 `docker-compose.yml`(redis + api + worker)
- [ ] P6.3 录 2 分钟 demo
- [ ] P6.4 把评测结果表放进 README 最前面

**验收标准:** 别人 `git clone` + 2 条命令能跑起来

---

## 8. 命令速查

```bash
# 环境
# 注意:项目 venv 里**没有 pip**(用 uv 建的),必须走 uv
uv pip install --python .venv/Scripts/python.exe fastapi "uvicorn[standard]" redis

# Redis(需自行安装或用 Docker)
docker run -d -p 6379:6379 --name coach-redis redis:7-alpine

# P0:建图 + 灌题目(22 知识点 / 27 关系 / 13 题)
.venv/Scripts/python.exe -m coach.knowledge.builder --build
.venv/Scripts/python.exe -m coach.knowledge.builder --build --seed 15   # 只建前 15 个点的子图

# P1~P3:建档 + 起 worker(定时 60 秒发一次 tick)
.venv/Scripts/python.exe -m coach.workers.run_all --init-learner u1 --goal algo.dp
.venv/Scripts/python.exe -m coach.workers.run_all --init-learner u1 --goal algo.dp --no-llm
.venv/Scripts/python.exe -m coach.workers.run_all --tick-once u1    # 只发一次 tick,手动触发

# 注意:python -m uvicorn 那条要等 Redis 起来才有意义

# P1:起服务
.venv/Scripts/python.exe -m uvicorn coach.api.main:app --reload --port 8000
.venv/Scripts/python.exe -m coach.workers.run_all        # 另开终端

# 测试
.venv/Scripts/python.exe -m pytest tests/coach/ -q

# P5:跑评测
.venv/Scripts/python.exe -m coach.evaluation.run_eval --n 50
```

**注意:** 已有的两个测试失败([test_llm_mcp_client.py](tests/test_llm_mcp_client.py))是历史遗留,与 coach 无关。

---

## 9. 恢复指南(很久没碰后从这里开始)

1. **读 §1 速览** —— 30 秒想起来这是干什么的
2. **看 §7 进度台账** —— 知道停在哪
3. **跑 `pytest tests/coach/ -q`** —— 确认当前代码是绿的
4. **看 §11 风险与卡点** —— 上次卡在哪
5. **从台账里第一个未完成的任务继续**

**如果连环境都忘了:**
```bash
cd d:/cursor/time_one_hour/mtpagent
.venv/Scripts/python.exe -m pytest tests/coach/ -q
docker start coach-redis || docker run -d -p 6379:6379 --name coach-redis redis:7-alpine
```

---

## 10. 进度台账

> 每完成一个阶段更新这里。这是"同步"的主要入口。

| 阶段 | 状态 | 开始 | 完成 | 备注 |
|---|---|---|---|---|
| 设计定稿 | ✅ 完成 | — | 2026-09-12 | mainconten.md 已推送 |
| P0 图底座 | ✅ 完成 | 2026-09-13 | 2026-09-13 | 22 点 / 27 边 / 无环;27 项测试通过 |
| P1 入口+队列+画像 | 🟡 代码完成 | 2026-09-13 | — | 78 项测试通过;**真 Redis 端到端未验**(Docker 未起) |
| P2 根因定位 ★ | ✅ 完成 | 2026-09-13 | 2026-09-13 | 123 项 coach 测试通过;验收场景(函数调用为根因)已断言 |
| P3 遗忘+调度 | ✅ 完成 | 2026-09-13 | 2026-09-13 | 156 项 coach 测试通过;到期项进计划已断言 |
| P4 工作流 | ⬜ 未开始 | — | — | — |
| P5 评测 ★ | ⬜ 未开始 | — | — | 差异化所在 |
| P6 交付 | ⬜ 未开始 | — | — | — |

**图例:** ⬜ 未开始 · 🟡 进行中 · ✅ 完成 · ⛔ 阻塞

---

## 11. 风险与卡点

| 风险 | 影响 | 对策 | 状态 |
|---|---|---|---|
| **前置依赖数据要从零抽** | P0 主要工作量 | LLM 抽取 + 治理流程 + 人工校验金标准 | 未验证 |
| 知识点数量贪多 | P0 拖长 | 严格卡 15 个,P1 再扩 | — |
| Redis 环境没装 | P1 阻塞 | Docker 一行起 | — |
| SM-2 参数凭感觉 | P3 效果差 | 先用经典默认值,评测阶段再调 | — |
| 模拟学生 = 循环论证 | P5 结果不可信 | 模拟学生的生成模型要与 BKT 假设**不同**;诚实写 Limitations | 需注意 |
| 图可能不比向量强 | 卖点站不住 | **这本身就是结论**,写进 Limitations | 待 P5 验证 |
| **BKT 全对时饱和到 1.0**(约 10 次) | P5 校准曲线/置信度会失真 | 照抄 §5.3 不改公式;P5 若校准差,再考虑加参数或改公式并记录 | 已发现 |
| **没作答记录的前置会被算成缺口** | 根因列表被"从没考过的点"挤占,学生见过的真缺口排到后面 | 这是 §5.5 的既定行为(未记录 = 掌握度初始值 0.1 < 0.4);P5 评测时要么用有作答轨迹的学生,要么显式区分"未观测"与"已观测且弱" | 需注意 |
| ~~LLM 抽取的 id 与人工金标准对不上~~ | 治理按 id 判重会让同一概念长成两个节点,污染前置闭包 | **已解决(2026-09-13)**:① 抽取时把已有概念清单喂给 LLM 要求复用 id;② 治理层按**名字**归并 —— 同名提案不新增节点,引用别名的边改指规范 id。真实调用复验:修复前 LLM 给 `algo.merge_sort` / 22→23 个节点;修复后给 `algo.mergesort` / 22→22 | ✅ 已解决 |
| 提示词塞不下全量概念清单 | 知识点上千后 `known_concepts` 会撑爆上下文 | 当前 22 个点无问题;真要扩容时改成"按文本相似度召回相关概念再喂",而不是全量塞 | 远期 |
| LLM 抽取已实测可用 | — | 单次真实调用(qwen-plus)抽 4 概念 / 3 关系,类型与置信度都合理,`PREREQUISITE` 判定正确 | 已验证 |

---

## 12. 更新记录

> 每次改动这份文档,在这里加一行。

| 日期 | 改动 |
|---|---|
| 2026-09-12 | 初版:定稿架构、数据模型、P0~P6 任务清单 |
| 2026-09-13 | P0.1/P0.2 完成:`coach/` 八层包骨架 + `config.py` + `domain/models.py` |
| 2026-09-13 | P0.3 完成:`knowledge/schema.py` 建表 DDL + `connect()`/`init_schema()`(WAL、幂等) |
| 2026-09-13 | 偏离记录:§4.1 增补 `idx_proposals_status(status, created_at)`(原 DDL 未写,aggregator 按 status 捞 pending 需要);已回填到 §4.1 |
| 2026-09-13 | P0.4 完成:`knowledge/store.py` 存取门面 + §5.1 递归 CTE 双向闭包(已冒烟验证深度/去重/不含自身/幂等) |
| 2026-09-13 | 偏离记录:新增 `domain/ids.py`(ULID 生成,§6 目录树未列)。§4.3 要求 event_id 为 ULID,P1 复用,故提前抽出 |
| 2026-09-13 | P0.5 完成:`knowledge/governance.py` 三阶段治理 + §5.2 环检测(9 项冒烟断言全过) |
| 2026-09-13 | P0.6 完成:`knowledge/builder.py` LLM 抽取(工具调用 + JSON 退化 + 脏数据丢弃)+ `ingest()` 走治理入库 + CLI(`--extract-file`/`--apply`,`--build` 待 P0.7 接线) |
| 2026-09-13 | 偏离记录:治理规则新增第 4 条「引用不存在的知识点 → 拒绝」(§4.4 原文只列自环/重复/置信度三条)。原因:LLM 可能抽到端点未入库的悬空边;该规则使 §4.4 的顺序约束「先概念后边」成为硬性要求,`ingest()` 已按此顺序执行 |
| 2026-09-13 | 偏离记录:种子规模取 **22 个知识点 / 27 条边**,非 P0 目标写的 15 个。原因:mainconten §5.1 的金标准样例覆盖到 22 个概念,砍到 15 会断链;`--seed N` 可截断 |
| 2026-09-13 | P0.7 完成:`builder.py` 录入金标准种子(四类关系齐全)+ `--build` 接线;图文件 `data/coach/coach.db` 生成(22 点/27 边/无环);`.gitignore` 加 `data/coach/` |
| 2026-09-13 | **P0 阶段完成**:`tests/coach/test_knowledge.py` 27 项全过;全量 `pytest tests/` 177 passed(历史 2 失败不变) |
| 2026-09-13 | P1 代码完成:events/coordination/profile/workers/api 五层 + 78 项测试;全量 228 passed。**待办:起 Redis 跑端到端验收**(Docker daemon 未启动) |
| 2026-09-13 | 偏离记录:新增 `domain/grading.py`(§6 目录树无此文件,判分逻辑原本没归属);新增 `tests/coach/conftest.py` 的 `FakeBus`,使测试不依赖真 Redis |
| 2026-09-13 | 偏离记录:`journal` / `learner_memory` 各加一个索引(`idx_journal_pending`、`idx_memory_due`);worker 的 SQLite 连接必须 `check_same_thread=False`(asyncio.to_thread 跨线程访问) |
| 2026-09-13 | 发现:BKT §5.3 公式在全对时约 10 次就饱和到 1.0(已入 §11 风险) |
| 2026-09-13 | **P2 完成**:根因定位 + `/gap` `/graph` + planner_worker 异步生成人话解释;coach 测试 123 项,全量 273 passed |
| 2026-09-13 | ★ 文档张力记录:硬约束「api/ 不许 import knowledge/profile」与 §5.3 的 `GET /gap`、`GET /graph` 冲突。**解法**:算法留在 `knowledge/queries.py`(纯函数,收掌握度映射而非 ProfileStore),新增 `workflow/query.py` 只读门面,api 只依赖门面。api 的源码扫描测试已锁死这条线 |
| 2026-09-13 | 偏离记录:`detect_gaps`/`root_causes`/`next_to_learn` 签名从 `(learner_id, kp_id)` 改为 `(mastery: dict, kp_id)` —— 否则 knowledge 层要 import profile |
| 2026-09-13 | 偏离记录:P2.4 的 LLM 解释由 planner_worker **异步生成并缓存**到新表 `gap_explanations`,而非在 `GET /gap` 里同步调 LLM(否则入口层破 50ms 且违反 P1 的「api 不碰 LLM」) |
| 2026-09-13 | 修复:`governance._has_open_proposal` 原本每次全表扫 proposals,建种子图退化成 O(n²);改为实例内 key 集合 + 增量维护,建图 0.27s |
| 2026-09-13 | 修复:`recovery.replay` 增加 `types` 过滤,两个 worker 共用一个 journal 时各自认领自己的事件类型 |
| 2026-09-13 | **P3 完成**:SM-2 + scheduler_worker + `GET /plan`(review/remedial/learn 三类合并);coach 156 项,全量 306 passed |
| 2026-09-13 | ★ 决策:§5.4 没规定二值判分怎么映射到 0~5 评分。**定为 答对=4、答错=1**。理由:q=4 时 EF 增量为 0,难度因子不漂移;q=5 会让 EF 无限上涨。改这个映射会直接改变复习间隔节奏 |
| 2026-09-13 | ★ 修复(潜在丢消息):`planner_worker` 原本同时消费 `coach:profile` 和 `coach:tick`。Redis 消费组里一条消息只投给组内**一个**消费者,两个 worker 抢同一个流会随机丢消息。**改为一个流只配一个 worker**:planner 只认 `coach:profile`,tick 归 scheduler |
| 2026-09-13 | 决策:`GET /plan` 的预算是**软的** —— 超预算就停,但至少留一项,不返回空计划 |
| 2026-09-13 | **修复(数据丢失)**:`profile_worker` 写「答题记录/掌握度/SM-2/易错」原本各自提交,进程半途被杀会让答题记录落了库而掌握度没更新;恢复重放看到记录已存在直接跳过 → **该次更新永久丢失**。已改为 `ProfileStore.transaction()` 包成一个事务;5 项原子性测试,把 `transaction()` 打回空操作后其中 3 项失败(已验证测试有效) |
| 2026-09-13 | **P4/P5 前置补齐**:① 题目种子 13 道(`GOLDEN_PROBLEMS`,覆盖 13 个知识点),`--build` 一并灌入;② `tick.scheduled` 有了生产者(run_all 定时 + `--tick-once` 手动);③ run_all 装配真 LLM(没 key 或 `--no-llm` 自动降级为模板);④ 学习者建档 `--init-learner`(只碰 SQLite,不依赖 Redis)。新增 `tests/coach/test_demo.py` 端到端测试:POST /answer → 判分 → profile.updated → planner 写解释 → GET /gap 读得到。全量 323 passed |
| 2026-09-13 | ★ 实测发现:真实 LLM 调用(qwen-plus)抽取可用,但 **id 与人工金标准对不上**(LLM 给 `algo.merge_sort`,金标准是 `algo.mergesort`)。治理按 id 判重 → 会产生重复概念。已入 §11,**未解决** |
| 2026-09-13 | **修复(id 漂移)**:① `builder.extract(known_concepts=...)` 把已有概念清单喂给 LLM 要求复用 id;② 治理层新增第 5 条规则「同名归并」(`normalize_name` 去空白+小写,同名提案拒绝入库并记别名,引用别名的边自动改指规范 id)。observation 保留 LLM 原始输出以便追溯。真实调用复验:22→22 个节点,无重复;新增 13 项测试(`test_id_drift.py`)。coach 186 项 / 全量 336 passed |
