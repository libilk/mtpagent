# work.md — 实施手册与进度台账

> **这份文档是什么:** 项目的**操作手册**。回答"怎么做、做到哪一步了、卡在哪"。
> [mainconten.md](mainconten.md) 回答**为什么做、做什么**(定位与设计总纲),本文回答**怎么落地**。
>
> **怎么用(重要):**
> 1. 每完成一个任务,把 `- [ ]` 改成 `- [x]`
> 2. 每完成一个阶段,更新 §7 进度台账,并在 §10 更新记录里加一行
> 3. 很久没碰之后,直接跳到 §9 恢复指南
>
> **最后更新:** 2026-09-14
> **当前阶段:** **P0~P6 全部完成**,§11.1 的待解决问题已清到只剩一条(真 Redis 验收,卡在 Docker)

---

## 1. 项目速览

**一句话:** 用知识图谱多跳推理定位"学不会的根因"的学习 Agent,并以量化评测证明图推理相比扁平检索的增益。

**当前状态:** **P0~P6 全部完成**,另有 agent 支线(§6 目录树 / §10 台账 / §12 记录)。
**要面试的话直接看 [§13 面试素材](work.md#13-面试素材可直接讲)** —— 把能讲的东西集中在那儿了。
截至 2026-09-14:`coach/` 约 7.9k 行,coach 测试 **366 项** / 全量 **516 项**(518 收集,2 个历史失败)。
四张评测表已产出:**两张是负面结果**(BKT 预测不可用、规划没跑赢随机),已如实写进
[evaluation/results/RESULTS.md](coach/evaluation/results/RESULTS.md) 的 Limitations。
根因定位那张(核心卖点)两个条件都**大幅反超扁平召回**(0.5467 / 0.6800 vs 0.0800 / 0.1333;详见 §7 P5)。
计划那张也从负转正(白费步数 82% → 0%),但**幅度太小不足以声称有效**(+0.6%)。

> **2026-09-14 修正**:上面这张表的数字**重算过**。原值(0.6000 / 0.6667)是在评测
> 尚不可复现时产出的 —— `run_eval.py` 里有一处 `for kp in kps` 迭代的是 Python `set`,
> 而字符串哈希**逐进程随机**(`PYTHONHASHSEED`),循环体里又在抽随机数,于是同一份代码
> 同一个种子跑出 0.48~0.64。加 `sorted()` 后锁定为 0.5467 / 0.6800,三个独立进程完全一致。

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
| 向量(暂不用) | [rag_core/chroma_store.py](rag_core/chroma_store.py) | P5 的对照组若要升级成真 embedding 可用它 |
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
│   redis.py     Redis Stream(消费组 / 死信 / PEL 认领)        │
│   memory.py    进程内总线(demo / 测试用,无持久化)             │
│   journal.py   ★ append-only 事件日志(先记后做)               │
│   recovery.py  ★ 重启后扫 journal + pending,补做未完成事件     │
└────────────────────────┬─────────────────────────────────────┘
                         │ consume
                         ▼
┌──────────────────────────────────────────────────────────────┐
│ ③ 常驻 worker  coach/workers/  (asyncio,可水平扩展)            │
│   profile_worker    消费答题 → 交给 workflow/pipeline 跑完整链  │
│   planner_worker    解释缓存过期时刷新(P4 起答题链自带解释)      │
│   scheduler_worker  SM-2 到期扫描 → 生成复习计划                │
│   run_all.py        开发期单进程拉起全部 + 定时发 tick           │
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
                      │   pipeline.py LangGraph 答题链          │
                      │   + checkpointer(断点恢复)             │
                      │   query.py    只读门面(给 api 用)      │
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
[workflow] pipeline.py 的 LangGraph 链开始(P4 起带 checkpointer)
  │
  ├─[1] grade            规则比对 test_cases → correct: bool
  ├─[2] update_profile   BKT + SM-2 + 答题记录(★ 同一事务)
  ├─[3] detect_gap       ★ 递归 CTE 查前置闭包 ∩ 掌握度 < 0.4
  │      ├─ 有缺口 → [4] explain(调 LLM 写人话;产出先落检查点)
  │      └─ 无缺口 → 直接 finalize
  └─[5] finalize         写幂等标记 + journal.mark_done + 发 profile.updated
  │
  │  (今日计划不在这条链上:由 scheduler_worker 按 tick 生成,见 P3)
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
| `events/` | 事件定义(信封 + 类型) | 投递、持久化(投递由 coordination 负责) |
| `coordination/` | Redis 收发、幂等、journal、recovery | 业务逻辑 |
| `workers/` | 常驻消费、编排调用 | HTTP |
| `knowledge/` | 图的存取、遍历、写入治理 | 画像 |
| `profile/` | 掌握度、记忆、易错、持久化 | 图 |
| `workflow/` | 步骤编排、状态流转(P4 起 = `pipeline.py` 的 LangGraph 链 + `query.py` 只读门面;另有 **agent 支线** `agent_tools.py` / `agent_diagnosis.py`,与确定性路径**并列**,默认不启用) | 存储细节 |
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
| 工作流 | ~~线性 async 链~~ → **LangGraph + SqliteSaver**(P4 已上) | 先跑通,断点恢复是 P4 的需求 | Temporal(工业级) |
| LLM | 复用 `llm/llm_client.py` | 已修好协议 | — |
| 测试 | pytest | 已有 | — |
| 新增依赖 | `fastapi` `uvicorn[standard]` `redis` + **P4:** `langgraph` `langgraph-checkpoint-sqlite` | 前三是原计划;langgraph 是 P4 明确要求 | 已全部装好(见 [requirements-coach.txt](requirements-coach.txt)) |

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

-- 边:前置关系(from 是 to 的前置)
-- 2026-09-13:原四类关系收窄为只有 PREREQUISITE(见 §11.3 调研 / §12 记录)。
-- type 列保留、值域只剩一个:它是递归 CTE 的过滤键 + idx_edges_to 的一部分,
-- 删列要动最核心且测试最密的代码,收益只是少一列。有意取舍,不是遗漏。
CREATE TABLE edges (
    from_id     TEXT NOT NULL,
    to_id       TEXT NOT NULL,
    type        TEXT NOT NULL,         -- 目前只有 PREREQUISITE
    weight      REAL DEFAULT 1.0,
    confidence  REAL DEFAULT 1.0,      -- 抽取置信度
    source      TEXT,                  -- 来源:llm_extract|manual|outline|golden
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

## 6. 目录结构(实际形态,✅ = 已实现)

```
coach/
├── config.py                    # ✅ 路径 + 全部算法默认参数
├── demo.py                      # ✅ 两分钟 demo(起真 uvicorn,零外部依赖)
├── domain/
│   ├── models.py                # ✅ 领域模型
│   ├── ids.py                   # ✅ ULID(§4.3 要求)
│   ├── text.py                  # ✅ 轻量文本相似度(零依赖,可复现)
│   └── grading.py               # ✅ 规则判分(P1.9)
├── knowledge/
│   ├── schema.py                # ✅ §4.1 DDL + connect()/init_schema()
│   ├── store.py                 # ✅ 存取 + §5.1 递归 CTE
│   ├── queries.py               # ✅ ★ §5.5 根因定位(纯函数)
│   ├── governance.py            # ✅ ★ observation → proposal → aggregator
│   ├── builder.py               # ✅ LLM 抽取 + §5.2 环检测 + 种子建图
│   ├── problem_bank.py          # ✅ ★ 题库导入(文件/URL/LeetCode 题单)
│   └── leetcode_import.py       # ✅ 从 LeetCode 学习计划抓取并转成题库 JSON(不写库)
├── profile/
│   ├── bkt.py                   # ✅ §5.3
│   ├── sm2.py                   # ✅ §5.4
│   ├── errors.py                # ✅ 错题模式分析(含跨知识点的系统性误解)
│   └── store.py                 # ✅ §4.2 建表 + 读写 + 幂等表 + 事务
├── events/
│   └── schema.py                # ✅ 事件信封 + 类型
├── coordination/
│   ├── redis.py                 # ✅ Redis Stream / 消费组 / 死信
│   ├── memory.py                # ✅ 进程内总线(demo / 测试,无持久化)
│   ├── journal.py               # ✅ ★ append-only
│   └── recovery.py              # ✅ ★ 重启恢复(按事件类型分流)
├── workers/
│   ├── profile_worker.py        # ✅ 消费答题 → 交给 pipeline
│   ├── planner_worker.py        # ✅ 解释缓存过期时刷新(答题链已自带解释)
│   ├── scheduler_worker.py      # ✅ SM-2 到期扫描 → 计划
│   └── run_all.py               # ✅ 开发期单进程拉起 + 定时 tick
├── workflow/
│   ├── query.py                 # ✅ 只读门面(api 依赖它,不直接依赖图/画像)
│   ├── pipeline.py              # ✅ ★ LangGraph 链 + checkpointer(P4)
│   ├── agent_tools.py           # ✅ agent 支线:把确定性内核包成 LLM 能调的工具
│   └── agent_diagnosis.py       # ✅ agent 支线:手写 ReAct 循环 + checkpointer
├── evaluation/                  # ✅ P5 评测
│   ├── simulated_student.py     # 模拟学生(生成模型刻意与 BKT 不同)
│   ├── metrics.py               # AUC / Brier / 校准 / Top-1
│   ├── baselines.py             # 对照组:扁平召回
│   ├── run_eval.py              # 四张表(五臂)+ RESULTS.md
│   └── results/                 # results.json + RESULTS.md
└── api/
    ├── main.py                  # ✅ create_app(bus=, query_service=)
    ├── routes.py                # ✅ /answer /gap /graph /plan /profile /health
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

- [x] P4.1 抽出 `workflow/pipeline.py` 的 State 定义(`AnswerState`)
- [x] P4.2 LangGraph 图:节点 `grade → update_profile → detect_gap`,条件边在 `detect_gap`(有缺口 → `explain` → `finalize`;无缺口 → 直接 `finalize`)
- [x] P4.3 checkpointer:测试用 `InMemorySaver`,生产用 `SqliteSaver`(**单独一个库文件**)
- [x] P4.4 测试:崩溃后从检查点恢复,**不重跑已完成的 LLM 调用**(用调用计数断言)
- [x] P4.5 迁移记录见下方「P4 迁移记录」

#### P4 迁移记录(面试素材)

**为什么 P0~P3 不上 LangGraph:** 链短,「事件 + 每个 worker 自己做几步」就够了 ——
少一个依赖、少一层抽象、出问题好查。

**什么时候非上不可:** 链一长就暴露问题 —— 进程在链中间被杀,恢复是**从事件头重放**,
前面算过的东西全白算。最亏的是 LLM 那一步:几百毫秒到几秒,而且花钱。
`explain` 单独成节点,就是为了让它的产出先落检查点,后面崩了不用重调。

**与文档的三处出入(都是实现时才发现必须这样):**
1. §2.2 把 [2] BKT 和 [3] SM-2 列为两步。这里**合成 `update_profile` 一个节点** ——
   它们必须原子(见 `profile/store.py` 的事务),拆成两个节点会让"BKT 写了、SM-2 没写"
   变成可能。
2. §2.2 的 [5] 是"生成今日计划"。这里把 **LLM 生成根因解释**放进了链(`explain` 节点),
   它是链里唯一"贵且非幂等"的一步,也正是 checkpointer 的价值所在。计划仍由
   scheduler 按 tick 生成。
3. 检查点**单独一个库文件**(`data/coach/checkpoints.db`)。SqliteSaver 自己会 commit,
   跟图/画像/journal 共用一个连接的话,它的提交会把我们事务里的半成品一起提交掉,
   原子性就没了。

**踩到的坑(值得讲):** langgraph 记住的是「下一步该跑谁」,要接着跑必须
`graph.invoke(None, config=cfg)`。传一份新的 state 会让它**从头开始**,检查点就白存了。
第一版 `run()` 就是这么写的 —— 崩溃后重跑,LLM 被调了两次,是测试的调用计数把它抓出来的。

**交付物:** 可恢复的工作流
**验收标准:** 在 `detect_gap` 中途杀进程,恢复后 trace 显示前序节点未重跑

**依赖:** P1~P3

---

### P5 — 评测(★ 差异化所在)

**目标:** 四张表,其中一张回答"图 vs 向量强多少"。

- [x] P5.1 `evaluation/simulated_student.py`:
  - [x] 给每个知识点预设"真实掌握度"(ground truth)
  - [x] 按掌握度概率生成答题结果
  - [x] 支持生成 N 个学生 × M 次答题的轨迹
  - [x] ★ **生成模型与 BKT 不同**(前置拖累 + 逻辑函数 + 遗忘),避免 §11 的循环论证
- [x] P5.2 `evaluation/metrics.py`:
  - [x] AUC / Brier(掌握度预测)
  - [x] 校准曲线 + ECE(遗忘预测)
  - [x] 提升幅度(规划 vs 随机)
  - [x] ★ 根因定位准确率(图遍历 vs 扁平召回)
- [x] P5.3 `evaluation/run_eval.py`:一键跑出四张表
- [x] P5.4 **对照组实现**:`evaluation/baselines.py` 文本相似度版根因定位
  - [x] 加「闭包内随机排序」第三臂,把「候选集大小」和「排序逻辑」分开
- [x] P5.5 结果落 `evaluation/results/results.json` + `RESULTS.md`(含 Limitations)

#### P5 结果与结论(种子 0,30 个模拟学生)

| 表 | 结果 | 结论 |
|---|---|---|
| 1 掌握度预测 | AUC **0.5336**(n=1710),Brier **0.2741** vs 基准率 0.2498 | ❌ **负面**:比恒定预测基准率还差(已降级为「仅用于排序」) |
| 2 遗忘预测 | 隔 30 天 ECE 0.2406 → **0.3450**,平均预测 0.42 / 实际 0.08 | ✅ 印证 BKT 不建模遗忘 |
| 3 规划增益 | 白费步数 82% → **0%**;相对随机 **+0.6%** | ⚠️ **转正但幅度太小**(见 Limitations #5) |
| 4 ★ 根因定位(充分) | 两态 **0.5467** / 消融 0.5467 / 闭包内随机 0.5067 / 扁平 **0.0800** | ✅ 图最强 |
| 4 ★ 根因定位(稀疏) | 两态 **0.6800** / 消融 0.5200 / 闭包内随机 0.6267 / 扁平 **0.1333** | ✅ 图最强 |

> 表 4 的「消融」臂 = 同样候选集、同样排序公式,唯一差别是**不区分「未观测」与「已观测且弱」**
> (即 2026-09-13 修复前的行为)。两态区分带来 **+0.0 / +16.0 个点** ——
> **功效集中在证据稀疏那一档**;作答充分时两臂持平,所以 RESULTS.md 里那条
> "增益是实测的" 限制**这次没有触发**(限制由数据触发,条件不成立就不写)。

完整数字与 Limitations 见 [coach/evaluation/results/RESULTS.md](coach/evaluation/results/RESULTS.md)。

> **★ 上表于 2026-09-14 全部重算。** 原值来自评测尚不可复现的时期(见 §1 的说明):
> `run_eval.py` 迭代了一个 Python `set`,受 `PYTHONHASHSEED` 影响逐进程漂移。
> 修复后同一份代码同一种子在三个独立进程里结果完全一致。**结论方向未变**
> (图仍以 0.5467 vs 0.0800 大幅领先扁平召回),但具体数值以此表为准。

**交付物:** 四张评测表
**验收标准:**
- `python -m coach.evaluation.run_eval` 能跑出结果
- 掌握度预测 AUC 有数字(多少不重要,**是不是真算出来的**重要)
- 图 vs 向量的对照有结论(哪怕结论是"图没强多少")

**依赖:** P0~P3

---

### P6 — 交付

- [x] P6.1 README(叙事参考 [mainconten.md](mainconten.md) §7):[README.md](README.md)
- [ ] ~~P6.2 `docker-compose.yml`~~ **主动跳过**,理由见下
- [x] P6.3 2 分钟 demo:**改成可复跑的脚本**而非录像 —— [coach/demo.py](coach/demo.py) + [demo.sh](demo.sh)
- [x] P6.4 评测结果表放进 README 最前面
- [x] P6.5 额外:`coordination/memory.py`(进程内总线)+ `run_all --bus/--seed-only`
- [x] P6.6 额外:`requirements-coach.txt`(与作品集 A 的重依赖分开)

**验收标准:** 别人 `git clone` + 装依赖 + 一条 `./demo.sh` 能跑起来 ✅ **已实测**

#### P6.2 为什么跳过 docker-compose

`docker-compose.yml` 我**没法验证** —— 开发机 Docker daemon 没启动。交付物里放一个
"我没跑过"的 compose 文件,和这个项目一贯的标准冲突(所有数字都要求有实测出处)。
所以选择**不写**,并在 README 里如实说明容器化未做,而不是塞一个没人验证过的配置。

作为补偿,把**本地路径**做到真的跑得起来:demo 不依赖 Redis / Docker / LLM key,
任何 `git clone` 之后都能看到完整链路。

#### P6.3 为什么是脚本不是录像

录像看完就完了;脚本可以被复跑、被 diff、被 CI 跑,README 里贴的是它的真实输出。
`coach/demo.py` 起的是**真实的 uvicorn 服务**、打的是**真 HTTP 请求**,不是内部函数调用
—— 所以它同时验证了 API 层真的能工作。

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

# P6:两分钟 demo(不需要 Redis / Docker / LLM key)
./demo.sh
.venv/Scripts/python.exe -m coach.demo --learner u1 --goal algo.dp

# 题库:看覆盖情况 / 从外部导入
.venv/Scripts/python.exe -m coach.knowledge.problem_bank --coverage
.venv/Scripts/python.exe -m coach.knowledge.problem_bank --from-url https://example.com/problems.json

# 题库:从 LeetCode 学习计划抓取(**只写 JSON 不建库**,先人工过映射)
.venv/Scripts/python.exe -m coach.knowledge.leetcode_import --plan top-100-liked
.venv/Scripts/python.exe -m coach.knowledge.problem_bank --from-file data/coach/problems/top-100-liked.json
.venv/Scripts/python.exe -m coach.knowledge.problem_bank --from-leetcode top-100-liked   # 一步到位(跳过人工复核)

# 测试
.venv/Scripts/python.exe -m pytest tests/coach/ -q

# P5:跑评测(四张表)
.venv/Scripts/python.exe -m coach.evaluation.run_eval --n 30

# agent 支线:单个知识点的诊断冒烟(需 DASHSCOPE_API_KEY)
.venv/Scripts/python.exe -m coach.workflow.agent_diagnosis --learner u1 --goal algo.dp

# agent 支线:评测第五个臂(★ 要真调 LLM,贵且不可复现,默认关)
.venv/Scripts/python.exe -m coach.evaluation.run_eval --n 30 --agent --agent-n 5

# agent 支线:在真实系统里打开(答题后异步跑一次诊断,默认关)
.venv/Scripts/python.exe -m coach.workers.run_all --init-learner u1 --goal algo.dp --agent-diagnosis

# 入口读 agent 诊断结果(只读 worker 预先写好的缓存,不调 LLM)
#   GET /gap/u1/algo.dp?mode=agent
```

**注意:** 已有的两个测试失败([test_llm_mcp_client.py](tests/test_llm_mcp_client.py))是历史遗留,与 coach 无关。

---

## 9. 恢复指南(很久没碰后从这里开始)

1. **读 §1 速览** —— 30 秒想起来这是干什么的
2. **看 §10 进度台账** —— 知道停在哪
3. **跑 `pytest tests/coach/ -q`** —— 确认当前代码是绿的
4. **看 §11.1 待解决问题** —— 按优先级挑一条开工(每条都带证据和该改哪个文件)
5. **从 §11.1 挑一条开工**(P0~P6 已全部完成,剩下的是待解决问题清单)

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
| P4 工作流 | ✅ 完成 | 2026-09-13 | 2026-09-13 | LangGraph + SqliteSaver;崩溃恢复不重调 LLM 已用调用计数断言 |
| P5 评测 ★ | ✅ 完成 | 2026-09-13 | 2026-09-13 | 四张表已产出;**两张是负面结果,已如实写进 Limitations** |
| P6 交付 | ✅ 完成 | 2026-09-13 | 2026-09-13 | README + demo.sh + memory 总线;**跳过 docker-compose(无法验证)** |
| 题库扩充 | ✅ 完成 | 2026-09-14 | 2026-09-14 | LeetCode `top-100-liked` 导入 81 题;真库 21 → 92 题,22/22 知识点仍全覆盖 |
| agent 支线 | ✅ 代码完成 | 2026-09-14 | 2026-09-14 | 工具层 + ReAct 循环 + 评测第五臂 + 产品接线;**确定性路径一行未改**。真 LLM 冒烟 top-1 命中,评测跑过一次(agent-n=5) |
| 评测可复现性修复 | ✅ 完成 | 2026-09-14 | 2026-09-14 | `sorted(kps)` 一行;四张表数字全部重算,三个独立进程结果一致 |

**图例:** ⬜ 未开始 · 🟡 进行中 · ✅ 完成 · ⛔ 阻塞

---

## 11. 待解决问题 + 风险卡点

### 11.1 待解决问题(按优先级,可直接开工)

> 每条都带**证据**和**该改哪个文件**。做完一条就把它那行改成 ✅ 开头、加删除线。

**P0 — 直接影响项目可信度**

| # | 问题 | 状态 |
|---|---|---|
| ✅ | ~~`root_causes` 把"未观测"当成缺口排序~~ | **已解决(09-13)**:两态区分 + 消融实测 +6.5/+11.3 个点。对标 DeepTutor `objective_status()` |
| ✅ | ~~计划推荐"没有题目的知识点"~~ | **已解决(09-14)**:`plan_view` 只推有题可做的点。**白费步数 82% → 0%** |
| ✅ | ~~计划没跑赢随机~~ | **已改善(09-14)**:修完 #2 与 advance 分支后 **-8.0% → +0.6%**。**但幅度太小不足以声称有效**(见 RESULTS.md Limitations #5) |
| ✅ | ~~BKT 预测不可用~~ | **已声明边界(09-14)**:不改公式,在 `bkt.py` 与 RESULTS.md 里明确「**只用于排序,不用于预测**」,AUC 表保留作为这条边界的证据 |

> 真正的修法见 #5:题库覆盖是根因定位的**必要条件** —— 没题的知识点永远产生不了证据。
> 09-14 题库从 13 道扩到 **19 道,覆盖全部 22 个知识点**,这条前提才成立。

**P1 — 验收欠账**

| # | 问题 | 状态 |
|---|---|---|
| ✅ | ~~异步出题未实现~~ | **已用另一条路解决(09-14)**:不做 LLM 现场出题(生成错的 expected 会污染判分和 BKT),改为 **`problem_bank.py` 从外部题库导入**(文件/URL)。数据在 `data/coach/problems/dsa_seed.json`。**09-14 扩充**:新增 `leetcode_import.py`,可从 LeetCode 学习计划(`top-100-liked` **100 题 → 可导入 81 题**)抓取并转成同构 JSON |
| ⛔ | **真 Redis 端到端验收** | **仍阻塞**:开发机 Docker daemon 未启动。`<50ms` 与 `kill -9` 恢复至今是进程内/逻辑验证 |
| ✅ | ~~`GET /profile/{learner_id}` 未实现~~ | **已补(09-14)**:走 `workflow/query.py` 门面,含 `mastery_summary` / `due_now` / `error_patterns` |
| ✅ | ~~`next_to_learn` 冷启动~~ | **已修(09-14)**:未观测的前置不再阻塞,改标 `action="probe"`(先摸底)。对标 DeepTutor 的 `probe` |

**P2 — 功能缺口**

| # | 问题 | 状态 |
|---|---|---|
| ✅ | ~~四类关系里三类是死的~~ | **已解决(09-13)**:只留 `PREREQUISITE` 并做深 |
| ✅ | ~~`profile/errors.py` 空壳~~ | **已落地(09-14)**:错题模式分析层 —— 不只是计数,还识别「同一错误类型跨多个知识点 = 疑似系统性误解」 |
| ✅ | ~~`next_to_learn` 有 N+1~~ | **已修(09-14)**:`KnowledgeStore.prerequisite_adjacency()` 一次查完 |
| ✅ | ~~概念上千后 prompt 塞不下~~ | **已修(09-14)**:`select_relevant_concepts` 按文本相似度召回相关概念再喂(相似度抽到 `domain/text.py`,与评测基线共用) |

**P3 — 收尾**

| # | 问题 | 说明 |
|---|---|---|
| ✅ | ~~P6 交付~~ | **已完成(2026-09-13)**:README(评测表放最前)+ `demo.sh`(零外部依赖、可复跑)+ `requirements-coach.txt`。`docker-compose.yml` 主动跳过 —— 无法验证的东西不进交付物 |
| ⛔ | **真 Redis 端到端验收** | 清单里唯一还欠的。启动 Docker Desktop 后即可执行 |

### 11.3 对标调研发现(2026-09-13,读源码,非 README)

读了 WeSmartFlow 与 DeepTutor 的相关源码,针对 §11.1 的几条问题找参照。

| 我们的问题 | 他们有解吗 | 他们的机制 |
|---|---|---|
| #1 未观测当缺口 | ✅ **DeepTutor 有,值得照抄** | `objective_status()` 返回 `mastered / learning / new` **三态**;`new` 与 `learning` **从不共享排序轴**;`_CONFIDENCE_CAP = {1:0.5, 2:0.8}` 用证据数封顶分数;零观测返回 `0.0` 而不是先验 |
| #3 BKT 预测不可用 | ⚠️ 他们**根本不用 BKT** | 掌握度 = 最近 5 次的**时间加权正确率**,无先验、无饱和。但**他们也没公布任何预测准确率**,不能反推"他们更准" |
| #2/#5 推荐无题项 | ✅ **结构上绕开了** | 题目由 LLM 现场生成,推荐即生成 → "无题可练"这个状态不存在。DeepTutor 有真状态机 `MasteryInteraction: REGISTERED → GRADED/ABANDONED`,答案服务端暂存到批改才释放 |
| P2#8 三类关系是死的 | ⚠️ **我们抄错了机制** | 见下 |

**关于 `relation_semantics.py`(重要更正):**
它不是"关系语义表"(方向性/传递性),而是**一组 embedding 探针 + 打分器**:
每类关系配 4~6 条学生口吻的 probe 文本(`"学这个之前需要先掌握什么"`),query embedding
对各类型打分,低于 `floor=0.30` 的类型**不遍历**,类型分作为乘子进邻居排序。

三个必须纠正的事实:
1. WeSmartFlow 有 **8 类**关系(`prerequisite/part_of/related/contrasts/application_of/special_case_of/generalizes/equivalent_to`),**根本没有 `extends`**;
2. 它**没有 per-type 分支** —— 八类统一加权,连它自己也没做到"RELATED 扩召回 / CONTRASTS 生成对比题";
3. `CONTENT_PROBES` 那张表**在仓库里未被任何代码调用**(code search 只命中 3 个文件)。

→ 所以 mainconten.md 里「四类关系,借鉴 WeSmartFlow」是**错记**:数量错、机制错,
而且对方自己也只有一半是活的。**决定:只留 `PREREQUISITE` 做深**(见 §12 记录)。

**他们共有的、我们没有的机制(候选改进,已进 §11.1):**

| 机制 | 他们怎么做 | 我们的现状 |
|---|---|---|
| 交互状态机 | `REGISTERED → GRADED/ABANDONED`,一次答题是一等对象 | 只有事件 + 幂等表 |
| 预期答案服务端暂存 | 出题时注册,批改后才释放 | 无(题库里的答案可见) |
| 类型决定闸门 | 概念/设计类**不能用数值判定**,必须定性讲解通过 | 所有知识点统一 BKT |
| 掌握度来源分离 | `mastery_source` 区分 `system`/`learner`,自述不篡改评估证据 | 无 |
| 防重复计数 | `attempt_number()`,注释明说"被打断的轮次不得把次数刷高" | 靠 event_id 幂等(等价但更隐式) |

**差异化判断仍成立:** 两份 README 通读 + 关键词检索,**都没有公开任何量化指标**
(DeepTutor 列了十几种检索引擎,但没有一个效果数字;WeSmartFlow 只在路线图里提"用教育任务评测")。
—— 但**"有数字"本身不构成优势**,我们的四张表里两张是负面的,先把 #1 修了让核心数字站住。

### 11.2 风险卡点

| 风险 | 影响 | 对策 | 状态 |
|---|---|---|---|
| ~~前置依赖数据要从零抽~~ | P0 主要工作量 | 已做:LLM 抽取 + 治理流程 + 22 点金标准种子 | ✅ 已完成 |
| ~~知识点数量贪多~~ | P0 拖长 | 实际取 22 个(§5.1 金标准全覆盖) | ✅ 已完成 |
| ~~Redis 环境没装~~ | P1 阻塞 | Docker 一行起 —— **但开发机 Docker 至今没启动,端到端验收仍欠** | ⚠️ 见 #6 |
| SM-2 参数凭感觉 | P3 效果差 | 用了经典默认值 + 自定的 q 映射(答对=4/答错=1),**评测阶段没调** | 需注意 |
| 模拟学生 = 循环论证 | P5 结果不可信 | 已做:生成模型与 BKT 不同(前置拖累 + 逻辑函数 + 遗忘) | ✅ 已缓解 |
| 图可能不比向量强 | 卖点站不住 | **已测**:作答充分时图 0.5467 vs 扁平 0.0800,图明显强 | ✅ 站住了 |
| ★ **BKT 预测基本不可用** | 掌握度预测 AUC 0.5353 / Brier 比基准率还差 | 已测出并如实写进 Limitations;要改得动 §5.3 的参数或公式 | **未解决**(见 §11.1 #3) |
| ★ **规划没跑赢随机** | 计划的卖点未被验证 | 已测出(-8.0%);先修「推荐必须可落地」再重跑 | **未解决**(见 §11.1 #2/#4) |
| ~~证据稀疏时排序有害~~ | 消融臂 0.5200 仍低于闭包内随机 0.6267 | **已解决(2026-09-13)**:补上两态区分后稀疏条件 0.6800 > 0.6267 | ✅ |
| **BKT 全对时饱和到 1.0**(约 10 次) | P5 校准曲线/置信度会失真 | 照抄 §5.3 不改公式;P5 若校准差,再考虑加参数或改公式并记录 | 已发现 |
| ~~没作答记录的前置会被算成缺口~~ | 根因列表被"从没考过的点"挤占 | **已解决(2026-09-13)**:见 §11.1 #1 | ✅ |
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
| 2026-09-13 | **P4 完成**:`workflow/pipeline.py` 用 LangGraph 承载答题链(`grade → update_profile → detect_gap → [explain] → finalize`),带 checkpointer。踩到的坑:langgraph 要接着跑必须 `invoke(None, config)`,传新 state 会从头开始 —— 第一版就是这么写的,LLM 被重复调用,是测试的调用计数抓出来的。新增 `test_pipeline.py` 18 项;coach 204 项 / 全量 354 passed |
| 2026-09-13 | 偏离记录(P4):① §2.2 的 [2]BKT 与 [3]SM-2 合并为 `update_profile` 一个节点(必须原子);② LLM 生成解释进了链(`explain` 节点),plan 仍归 scheduler;③ 检查点单独一个库文件 `data/coach/checkpoints.db`(SqliteSaver 自带 commit,共用连接会破坏事务原子性);④ `planner_worker` 的解释职责降级为「缓存过期才刷新」(避免同一次答题调两遍 LLM);⑤ 第 4/5 个依赖:`langgraph` `langgraph-checkpoint-sqlite` |
| 2026-09-13 | **P5 完成**:模拟学生(生成模型与 BKT 不同)+ 指标 + 扁平召回对照组 + 一键跑四张表。**结论是混合的**:根因定位图 0.6267 vs 扁平 0.0667(赢);但表 1 BKT 预测比基准率还差、表 3 计划相对随机 -1.19%、表 4 稀疏条件下排序反而有害。全部如实写进 RESULTS.md 的 8 条 Limitations。coach 235 项 / 全量 385 passed |
| 2026-09-13 | 评测实现中修掉三处**自身的方法论错误**(若不修,数字就是假的):① 忘了 `mark_practiced`,导致遗忘实验整段失效;② 规划实验的"白费步数"没计数,**83% 的步数浪费在"推荐了没有题目的知识点"上**;③ 扁平基线把"未观测"当成 mastery 0.0,比真值还低,于是专挑毫无证据的点(那个 0.0 的假结果就是这么来的) |
| 2026-09-13 | §11 重构为「11.1 待解决问题(12 条,按 P0/P1/P2/P3 排,每条带证据和该改的文件)」+「11.2 风险卡点」;§9 恢复指南改为指向 §11.1 |
| 2026-09-13 | **对标调研**(读源码非 README):新增 §11.3。发现 DeepTutor 用 `objective_status()` 三态解决"未观测当缺口";两者都靠 LLM 现场出题从结构上绕开"推荐无题项";**`relation_semantics.py` 不是语义表而是 embedding 探针路由** —— 我们照抄了名字没抄到机制。**修正 mainconten.md 里「四类关系借鉴 WeSmartFlow」的错记**(它是 8 类、无 extends) |
| 2026-09-13 | **修复 §11.1 #1(未观测当缺口)**:`ProfileStore.observed_kp_ids()` 让画像能表达"无记录";`detect_gaps` 加 `observed` + `status`,排序改 `(status_rank, depth, mastery)`。新增**消融臂**做对照:同样候选集/排序公式,只是不区分两态 —— 实测 +6.5 / +11.3 个点,两个条件都反超闭包内随机。同步修了评测的一处**度量效度问题**(ground truth 选到了没题的知识点,系统原理上不可能定位到) |
| 2026-09-13 | **收窄为只做 PREREQUISITE**:`EDGE_TYPES` 只剩一种,种子里 7 条 RELATED/EXTENDS/CONTRASTS 移除(图 22 点 / 27 → **20 边**),**抽取提示词同步改掉**(否则 LLM 输出的三类会被治理层拒,是隐藏的功能牵连)。`edges.type` 列保留(CTE 过滤键 + 索引,删列收益不抵风险),已在 §4.1 注明是有意取舍 |
| 2026-09-13 | 新发现局限已写进 RESULTS.md:没有题目的知识点永远无法被观测 → **原理上无法被定位为根因**;这类点标 `unobserved` 并提示"该去测",不伪装成"已确认薄弱" |
| 2026-09-13 | **P6 完成**:`README.md`(评测表放最前 + 7 条 Limitations)+ `demo.sh`/`coach/demo.py`(起真 uvicorn、打真 HTTP、**零外部依赖**)+ `requirements-coach.txt`。**主动跳过 `docker-compose.yml`** —— 开发机 Docker daemon 未启动,写一个没验证过的交付物与本项目标准冲突,README 里已如实说明 |
| 2026-09-13 | 修复(全新库踩坑):`schema.connect()` 只连接不建表,导致 `demo` 与 `run_all` 在全新数据库上都会报 `no such table`。新增 `schema.open_db()` = 连接 + 建表,demo / run_all / query 工厂统一改用它 |
| 2026-09-13 | `coach/coordination/memory.py`:把测试里的 `FakeBus` 收编为正式代码 `InMemoryBus`,测试与 demo 共用一份实现,避免两套漂移 |
| 2026-09-14 | **修完 §11.1 的 P0/P1/P2 大部分**。① **题库**:新增 `problem_bank.py`(文件/URL 导入)+ `data/coach/problems/dsa_seed.json`,**19 道题覆盖全部 22 个知识点**(原 13 道只覆盖 16 个 —— 没题的点永远产生不了证据,根因定位原理上够不着);题目数据从 `builder.py` 搬出,内置与外部导入走**同一条路径**。② **计划**:只推有题可做的点(白费步数 **82% → 0%**)+ 补上 §2.2 一直缺的 **advance 分支**(`next_to_learn` 只在前置闭包里找,**目标自己永远不在候选里**,导致前置补完后计划变空 —— 实测 24 步 18 步为空)。表 3 由 **-8.0% 转 +1.2%**,但幅度太小不足以声称有效。③ **冷启动**:`next_to_learn` 不再把"未观测的前置"当成"没准备好",改标 `probe`。④ **`GET /profile`** 补上(含 `error_patterns`,能识别跨知识点的系统性误解)。⑤ **`profile/errors.py`** 落地为分析层。⑥ 清掉 N+1(`prerequisite_adjacency`)与 prompt 容量问题(`select_relevant_concepts`,相似度抽到 `domain/text.py` 与评测基线共用)。⑦ **BKT 降级**:在 `bkt.py` 明确声明「只用于排序,不用于预测」。测试 274 项 / 全量 **424 通过** |
| 2026-09-14 | 评测的 Limitations 改为**全部由数据触发**(问题修掉后对应限制自动消失,不留在文档里误导人);新增一条反过来的说明:题库覆盖是根因定位的**必要条件** |
| 2026-09-14 | **题库新增 LeetCode 题单来源**:`coach/knowledge/leetcode_import.py`(抓取 leetcode.cn 公开 GraphQL + 转题库 JSON,**不写库**)+ `problem_bank.py` 的 `--from-leetcode` 分支。`top-100-liked` 实测 **100 题 → 81 题可导入**(跳过:答案不唯一 12 / 图里无对应知识点 4 / 设计题 3),已导入真库(21 → **92 题,22/22 知识点仍然全覆盖**)。转换里三处硬处理:①题单接口不给用例,逐题拼 `metaData`+`exampleTestcases`+`content`;②Output 有两种 HTML 形态,统一"块级标签换行 + 找 Output 行";③`grading` 是严格字符串相等,多解题不能收。**再修两处**(导入真库后才暴露):④`kth-largest` 的映射覆盖漏了 `algo.quicksort` —— 它是内置题库里 quicksort **唯一**的题源,覆盖种子题会把该知识点变成"没题"(覆盖率 22/22 → 21/22);⑤判分取**最后一条**用例当提交用例,而 LeetCode 示例按「典型→边界」排,末条常是 `head=[]` 这种边界样例 → 丢掉空集合用例并把首条转到末尾。产出 `data/coach/problems/top-100-liked.json`。测试 +21 项(注入假抓取,不打网络) |
| 2026-09-14 | **`study.md` 补齐**:拿实际代码逐文件核对,发现漏了 **11 个文件**。补写 `knowledge/leetcode_import.py`(含今天踩的两个坑:映射覆盖丢 `quicksort`、提交用例落在边界样例上),以及 `domain/grading`(判分口径——"最后一条用例是提交用例"这条约定正是 `MULTI_ANSWER_SLUGS` 存在的原因)· `domain/ids` · `domain/text` · `coordination/memory` · `workers/planner_worker`/`scheduler_worker`(抽出三个 worker 共用的 `process()` 五行契约)· `workers/run_all`(装配顺序)· `api/main` · `api/schemas` · `demo`。阅读顺序表行数按实际重核(阶段 3 1060→1493、阶段 4 745→1468、阶段 5 352→432),并修正一处旧标题("三个路标"实际列了四条)。**33 个模块现已全部收录** |
| 2026-09-14 | **文档与代码一致性审计**:拿实际文件核对三份文档里提到的每个模块和数字。修正 ①`mainconten.md` 页首状态(还写着"方向待确认")、架构图里的 `events/event_bus.py`(我们只做了 `schema.py`)、对比表的规模/编排/评测三行;**§3.3 的评测表补上实际结果**(含"规划这项没站住")。②`work.md` §6 目录树四处不准(`errors.py` 还标着空壳、`memory.py` 被放错到 `events/`、缺 `baselines.py`/`results/`/`GET /profile`)、§2.1 图的重复行、§2.3 `events/` 职责。③`README` 的过时数字(测试计数、7→6 条 Limitations)—— 并于 09-14 复核修正为 **274**(此前记的 269 是错的)。**现在三份文档里提到的 32 个模块文件全部能找到** |
| 2026-09-14 | **agent 支线落地**(对照 Tencent WeSmartFlow / HKUDS DeepTutor 源码后的结论:两家都是「领域内核保持确定性,只把它包成 tool,让 LLM 决定何时调」)。新增 `workflow/agent_tools.py`(6 个只读工具 + 终结工具 `submit_diagnosis`;上下文**暴露方法而非裸字典**,两个构造器分别服务生产/评测)+ `workflow/agent_diagnosis.py`(StateGraph 里的手写 ReAct,复用 checkpointer)+ `agent_diagnoses` 缓存表 + `?mode=agent` + `run_all --agent-diagnosis` + 评测第五臂 `--agent`。**确定性路径一行未改**。真 LLM 冒烟:6 步用满 6 个工具,top-1 命中 `prog.func_call`。测试 295 → **366**。两个刻意的设计:**不给 agent `detect_gaps`/`root_causes`**(给了就成套壳)、**上下文不进 state**(checkpointer 会序列化落盘,而它攥着 sqlite 连接) |
| 2026-09-14 | **★ 修评测可复现性 + 全部数字重算**。起因是给评测加第五臂时顺手复跑,发现主表数字和提交里那份对不上。查下来两个独立问题:① **真 bug** —— `run_eval.py` 的 `for kp in kps:` 迭代的是 Python `set`,而字符串哈希逐进程随机(`PYTHONHASHSEED`),循环体里又在抽随机数 → 同一份代码同一个种子跑出 **0.48 / 0.5333 / 0.64**(跨度 16 个点,比项目要测的好几个效应都大)。加 `sorted()` 后三个独立进程完全一致。**表 1/2/3 没中招**(它们迭代 dict/list),这反过来验证了定位。② **结果过时** —— 提交里的 `results.json` 生成于 `250b501`,**早于** 09-14 的题库扩容(13→19 道)与 `plan_view` 修复。两个问题叠加后,四张表数字**全部重算**:根因定位 0.6000/0.6667 → **0.5467/0.6800**(扁平 0.0800/0.1333 未变),规划 +1.2% → **+0.6%**,表 2 ECE 0.2397/0.3632 → 0.2406/0.3450。**结论方向未变**(图仍以约 7 倍优势领先扁平召回),但"两态区分"的增益从 +6.5/+11.3 变成 **+0.0/+16.0** —— 功效集中在证据稀疏那档,充分条件两臂持平,`render_limitations` 里那条限制**因此自动消失**(数据触发机制按设计工作)。同步更新 README / work.md / mainconten.md / study.md 四处数字 |

---

## 13. 面试素材(可直接讲)

> 把散在各处的"能讲的东西"集中到这里。每条按**背景 → 怎么发现的 → 强调什么**写。
> **不要背数字,背因果链** —— 数字会被追问,因果链不会。
>
> 已有的那条见 [§7 的 P4 迁移记录](#p4-迁移记录面试素材)(为什么 P0~P3 不上 LangGraph、
> 什么时候非上不可、`invoke(None, config)` 那个坑)。

### 13.1 ★ 评测数字会飘:一次 `set` 迭代引发的 16 个点抖动

**背景。** 项目的核心卖点是"用评测证明图推理比扁平检索强"。评测四张表,种子固定 ——
按理说应该可复现,`RESULTS.md` 里也白纸黑字写着"可复算"。

**怎么发现的(这段最值钱)。** 给评测加第五个臂时顺手复跑主表,发现数字和提交里那份对不上。
第一反应是"我的改动影响了确定性路径",于是先跑了一个**不带新功能的对照组** ——
结果仍然是 0.56 而不是我以为的 0.5333。这下有意思了:同进程里连跑三次完全一致,
跨进程却每次不同,而且**加不加新功能都一样**。这三个事实的组合把方向从"我的代码"
推到了"**进程级状态**"。

**根因。** `for kp in kps:` —— `kps` 是 Python `set`。CPython 的字符串哈希**逐进程随机化**
(`PYTHONHASHSEED`),集合迭代顺序因此每次都不同;而循环体里恰好要调模拟学生**抽随机数**。
顺序一变,同一批随机数就分给了不同的知识点,掌握度分布随之改变。

**怎么确认的。** 用 `PYTHONHASHSEED=0` 固定住,三个独立进程跑出**完全相同**的值 —— 假设被锁死。
另外 **表 1 没中招**(它迭代的是 dict/list),这个旁证说明定位准确,不是碰运气。

**影响面。** 实测跨度 **0.48 ~ 0.64**,16 个点 —— **比项目声称要测量的好几个效应都大**
(比如"两态区分 +6.5 个点")。也就是说修复前那些数字里,有一部分是噪声。

**修复。** 一行 `sorted(kps)`。

**讲的时候强调什么:**
- 这是**先假设、做对照实验、再定位**的流程,不是"试出来的"
- 修完我**重算了全部数字、改了四处文档**,而不是只改代码 ——
  数字和文档对不上是最容易被追问穿的地方
- 重算后"两态区分"的增益从 +6.5/+11.3 掉到 +0.0/+16.0,我**如实改了口径**,
  没挑好看的那个报

**可能的追问:**
- *"为什么不直接固定 `PYTHONHASHSEED=0`?"* → 那是掩盖。根因是迭代顺序不确定,
  就该让它确定;而且评测的调用方控制不了别人机器上的环境变量。
- *"为什么同进程内是稳定的?"* → 哈希种子在进程启动时定一次,进程内不再变。

### 13.2 agent 支线:把确定性内核包成工具,而不是让 LLM 接管

**背景。** 项目原本是一条**固定流水线** —— LLM 只在两个固定点被调用(抽取关系、写解释),
从不决定下一步做什么。想学 agent,于是动手改成"LLM 自己选工具"。

**先做的功课(这一步让整个方案有了依据)。** 读了两个企业级开源项目的**源码**
(Tencent WeSmartFlow、HKUDS DeepTutor,不是读 README)。结论高度一致:

> **领域内核保持确定性,只把它包成 tool,让 LLM 决定「什么时候、为什么」去调它。**

两家的判分、掌握度、复习调度**全都没交给 LLM**。它们和我原来的做法一样,
差别只在于**有没有把这些函数包成 tool**。所以我做的不是"重写一套智能",
而是"加一层工具门面 + 一个循环"。

**三个刻意的设计(每条都有反面):**
1. **不给 agent `detect_gaps` / `root_causes`** —— 那是确定性版的**最终答案**。
   给了它就退化成"调一次函数复述结果",与基线对照彻底失去意义。只给**原料**。
2. **上下文对象绝不进 state** —— checkpointer 会把 state 序列化落盘,
   而上下文里攥着 sqlite 连接,进去就炸。改用**节点闭包**注入。
   不守这条,就别想要可恢复性。
3. **输出契约与现有接口逐字节一致** —— 八个 key 一模一样,所以 `api/schemas.py` 一行没改。
   新老两条路径因此**可以直接对比**,评测第五臂就是这么来的。

**为什么挂在 LangGraph 上。** 白拿 checkpointer:循环里最贵的一步是 LLM 调用,
崩了不该重跑。**两家对标项目的 agent 状态都没有检查点** —— 这反而是我领先的地方。

**踩到的坑(值得讲)。** 步数上限没生效:我把路由函数写成了**模块级函数**,
读的是模块常量 `MAX_STEPS`,而实例上传的 `max_steps=3` 根本不在它的作用域里 ——
上限永远是 8。是测试抓出来的。这类"**看着对、其实没接上**"的错误,
靠读代码几乎发现不了。

**评测结果与怎么说它。** 同样本对照里 agent 0.4286 vs 图遍历 0.3571(充分条件)。
但**我没有把它讲成"agent 赢了"** —— 因为:① 样本只有 14 个;② agent 不可复现(LLM 采样);
③ 更关键的是,**它拿到的工具集已经足够让它自己算出确定性算法**,
所以它接近图遍历是"在用工具复现那套算法",而不是独立发现了新东西。
这三条全写进了 RESULTS.md 的 Limitations。

**可能的追问:**
- *"那 agent 到底有没有用?"* → 在这个任务上**没有证据说它更好**,而且更贵更慢。
  真正拿到的结论是:**任务能被预先分解时,workflow 比 agent 更合适**。
- *"为什么不用 `create_react_agent`?"* → 它把循环机制藏起来了,而这次的目的就是读懂它。
- *"checkpointer 在里面起了什么作用?"* → 崩溃后不重调已完成的那轮 LLM;
  有测试用**调用计数**断言这件事(照抄 `test_pipeline.py` 的写法)。

### 13.3 索引:其它可讲的点

| 想讲什么 | 去哪看 |
|---|---|
| LangGraph 什么时候该上、什么时候不该 | [§7 P4 迁移记录](#p4-迁移记录面试素材) |
| 幂等要做两道闸、事务边界在哪 | §5.6 + §12「事务是被一次数据丢失逼出来的」那条 |
| LLM 输出不能直写(治理流程) | §2.1 的 `observation → proposal → aggregator` |
| 两态区分:未观测 ≠ 已确认薄弱 | §12 的 2026-09-13 那条 + [study.md](study.md) 阶段 2 |
| 没题的知识点永远定位不到 | §11.1 P0 区块下的说明 |
| 判分口径为什么只有 `exact_output` | [study.md](study.md) 阶段 4 ⓪ |
