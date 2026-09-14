# study.md — 代码阅读指南

> **这份文档是什么:** 读懂 `coach/` 代码思路的**顺序与路标**。
> 不解释业务(那是 [mainconten.md](mainconten.md)),不记录进度(那是 [work.md](work.md)),
> 只回答一个问题:**"我想搞懂它是怎么做的,该按什么顺序读、每个文件在干什么?"**
>
> **怎么用:** 每个文件都按同一套格式写 ——
> **[意图]** 它为什么存在 · **[骨架]** 它长什么样 · **[细节]** 打开后看哪几处。
> 遇到看不懂的地方,回头查这一节,而不是从头重读。

**最后核对:** 2026-09-14(骨架由 `ast` 从实际代码抽取,非手写)

---

## 阅读顺序总览

**原则:按「设计推理发生的顺序」读,不按目录顺序。**

目录顺序会让人一上来就陷进 `store.py` 的 300 行 CRUD。真实的思考顺序是:
**数据长什么样 → 核心算法是什么 → 数据从哪来 → 一次请求怎么走 → 怎么保证不丢 → 数字怎么来**。

| 阶段 | 主题 | 文件 | 行数 | 读完能回答 |
|---|---|---|---|---|
| 1 | 数据形状 | `models` / `schema` / `config` | 261 | 系统里有哪些"东西"?表怎么建的? |
| 2 | **核心算法 ★** | `queries` / `store` | 552 | 怎么算出"他缺什么"? |
| 3 | 数据从哪来 | `governance` / `builder` / `problem_bank` | 1060 | LLM 抽的东西为什么不能直写? |
| 4 | **一次请求怎么走 ★** | `events` / `api` / `workers` / `pipeline` | 745 | 为什么 API 只要 30ms? |
| 5 | 可靠性 | `journal` / `recovery` / `redis` | 352 | 进程被杀为什么消息不丢? |
| 6 | 画像怎么算 | `bkt` / `sm2` / `store` / `errors` | 633 | 掌握度和复习间隔怎么更新? |
| 7 | 计划怎么排 | `workflow/query` | 397 | 今天该练什么? |
| 8 | 数字怎么来的 | `evaluation/*` | 1170 | 那些评测数字可信吗? |

---

# 阶段 1:数据形状(先搞清楚"东西长什么样")

## [domain/models.py](coach/domain/models.py) — 97 行

**[意图]**
定义系统里到底有哪几种"东西"。不读它,后面所有代码里的 `kp_id`、`payload`、`status` 都是猜的。

**[骨架]**
```
常量:EDGE_TYPES(只有 PREREQUISITE) / OBSERVATION_KINDS / PROPOSAL_STATUSES

KnowledgePoint  知识点      id, name, subject, difficulty, description
Edge            前置边      from_id → to_id(学 to 前须掌握 from)
Problem         题目        test_cases(JSON), kp_ids, judge_type
Observation     原始观察    LLM 抽出来的信号,**未经验证**
Proposal        提案        观察经规则校验后的产物,带 status + reason
```

**[细节]**
- 全部是 `@dataclass`,没有 ORM、没有基类 —— 这是刻意的:模型只描述形状,持久化交给 store
- 看 `Edge` 和 `Proposal` 的**字段差异**:`Edge` 是结果,`Proposal` 是待审的变更。**这个区分是整个治理流程的基础**
- 文件顶部有一大段注释解释"为什么只有一种关系" —— 那是设计决策的记录,不是废话

---

## [knowledge/schema.py](coach/knowledge/schema.py) — 107 行

**[意图]**
建表。**读完应该能画出 ER 图** —— 之后所有查询都是在这几张表上做文章。

**[骨架]**
```
常量 DDL             5 张表 + 2 个索引(一次 executescript)
connect()            打开连接:WAL + 外键 + busy_timeout
init_schema()        建表(幂等)
open_db()            连接 + 建表,一步到位 ← 实际都用这个
```

**[细节]**
- **五张表的关系**:
  ```
  concepts ──┐
             ├─ edges(from_id, to_id, type)   前置 DAG
  problems ──┘ (kp_ids 是 JSON,不建关联表 ← 记下,这是后话)
  observations → proposals → (通过后写入 concepts/edges)
  ```
- ⚠️ `connect()` **只连接不建表**。全新库上直接用它会报 `no such table` —— 这个坑踩过两次,所以才有了 `open_db()`
- `edges` 上 `idx_edges_to(to_id, type)` 是**反向遍历的命门** —— 递归 CTE 全靠它

---

## [config.py](coach/config.py) — 57 行

**[意图]**
所有魔法数字的集中地。**调参只看这一个文件**,不用到处 grep。

**[骨架]**
```
路径      REPO_ROOT / DATA_DIR / DB_PATH / db_path()
图遍历    MAX_PREREQ_DEPTH=3  GAP_THRESHOLD=0.4  ROOT_CAUSE_TOP_K=3
治理      MIN_EDGE_CONFIDENCE=0.6
BKT       P_INIT=0.1  TRANSIT=0.15  GUESS=0.2  SLIP=0.1
SM-2      DEFAULT_EF=2.5  MIN_EF=1.3  PASS_SCORE=3  QUALITY_CORRECT=4  QUALITY_WRONG=1
计划      PLAN_REVIEW_MINUTES / PLAN_LEARN_MINUTES_PER_DIFFICULTY / PLAN_REMEDIAL_MULTIPLIER
Redis     REDIS_URL / STREAM_* / GROUP_COACH / MAX_ATTEMPTS=3
```

**[细节]**
- `SM2_QUALITY_CORRECT=4` 有讲究:q=4 时 EF 增量正好为 0(不漂移);q=5 会指数膨胀,q=3 会持续下降
- `GAP_THRESHOLD=0.4` 同时被根因定位和计划使用 —— **改它会同时影响两张评测表**

---

# 阶段 2:核心算法 ★(项目存在的理由)

## [knowledge/queries.py](coach/knowledge/queries.py) — 229 行

**[意图]**
**这是整个项目的卖点。** 回答一个问题:"学生卡在动态规划,他真正缺的是哪块基础?"
扁平检索答不了,只有沿前置依赖反向遍历才能答。

**[骨架]**
```
prereq_closure()        前置闭包 {kp_id: 最短深度}     ← 转发递归 CTE
detect_gaps()     ★     闭包里的缺口 + 两态区分 + 排序
root_causes()           detect_gaps 取 top_k,每个附依赖路径
next_to_learn()         现在该学什么(含 probe / learn 两种动作)
shortest_prereq_path()  从目标回溯到某个前置的最短路径
explain_gaps()          不调 LLM 的兜底解释(措辞区分两态)
```

**[细节]** —— 按这个顺序读

1. **`detect_gaps` 的排序键是第一重点**:
   ```python
   gaps.sort(key=lambda g: (0 if g["status"] == "gap" else 1, g["depth"], g["mastery"]))
   ```
   为什么第一个键是"有没有证据"而不是掌握度?因为**没被观测过的点,掌握度只是初始值**,
   数值上可能比"考过且确实弱"的点还低。不区分就会让"从没见过的浅层点"挤掉真根因。
   —— 这一行是**实测出来的**(消融 +17.3 个点),注释里有完整推理

2. **`next_to_learn` 的三个分支**:
   已掌握 → 跳过;前置**已确认弱** → 跳过;前置**只是没测过** → 不阻塞但标 `probe`。
   这是冷启动修正

3. **`observed=None` 的退化路径** —— 不传就是旧行为,给不关心两态的调用方留的出口

4. **全是纯函数**:只收「图 + 掌握度映射」,不认识 `ProfileStore` —— 为了满足分层硬约束

---

## [knowledge/store.py](coach/knowledge/store.py) — 323 行

**[意图]**
把 SQLite 包一层。**大部分是 CRUD,不用逐行读** —— 只有三个方法值得看。

**[骨架]**
```
常量 _ANCESTORS_SQL / _DESCENDANTS_SQL     ← 递归 CTE 就在这

KnowledgeStore
├── concept / edge / problem 的增删查(约 20 个方法,跳读)
└── ★ 值得看的三个:
    .ancestors()                前置闭包(递归 CTE + UNION 去重 + MIN(depth))
    .descendants()              后继闭包(反向)
    .prerequisite_adjacency()   一次查完 {点: [直接前置]} —— 避免循环里 N+1
```

**[细节]**
- **递归 CTE 的两处关键**:`UNION`(不是 `UNION ALL`)负责去重,`MIN(depth)` 取最短路;
  `p.depth < :max_depth` 是唯一的终止条件
- `prerequisite_adjacency()` 的存在本身就是个故事:原来 `next_to_learn` 在循环里逐个
  调 `ancestors(kp, 1)`,22 个点看不出来,但评测里 30 个学生 × 每步都在跑它
- 写入一律是 **upsert** —— 所以建图可以反复重跑

---

# 阶段 3:数据从哪来(治理流程)

## [knowledge/governance.py](coach/knowledge/governance.py) — 387 行

**[意图]**
**LLM 抽出来的东西不能直接当事实。** 这个文件是那道闸门。
核心立场一句话:LLM 的输出是**提案**,不是知识。

**[骨架]**
```
has_cycle()           DFS 三色标记,PREREQUISITE 成环直接拒
normalize_name()      去空白 + 转小写(同名归并用)

Governance
├── ① observe(kind, payload, source) → Observation      写 observations 表
├── ② propose(observation_id) → Proposal                规则校验
│      _validate_edge()     5 条规则:自环/类型/置信度/悬空引用/重复
│      _validate_concept()  同名归并 → 拒绝但记别名
│      _resolve_id()        ★ 把别名 id 归到规范 id
└── ③ aggregate(proposal_id) → Proposal                 环检测后入库
       _aggregate_edge()     PREREQUISITE 才做环检测
```

**[细节]**
- **`propose` 只做规则校验,不碰图**;`aggregate` 才做环检测和写库。
  **这个分离很重要**:规则是"这条数据像不像话",环检测是"放进图里会不会坏"
- **被拒的提案留在库里**(带 `reason`)—— 这是可解释性:能回答"这条关系为什么没进图"
- **`_resolve_id` 是同名归并的实现**:LLM 给 `algo.merge_sort`、金标准是 `algo.mergesort`,
  只按 id 判重会长出两个节点。这里按**名字**匹配,并把别名记下来让后续的边自动改指
- `_load_open_keys()` 是判重的缓存 —— 原来每次校验都把全部提案 `json.loads` 一遍,是 O(n²)

---

## [knowledge/builder.py](coach/knowledge/builder.py) — 453 行

**[意图]**
两个入口:把**人工确认好的**种子数据灌进去;让 LLM 从教材里**抽候选**。

**[骨架]**
```
EXTRACT_SYSTEM_PROMPT / EXTRACT_TOOL    提示词 + function calling schema
select_relevant_concepts()   ★ 按文本相似度召回相关概念(不全塞 prompt)
extract(llm, text, known_concepts)       工具调用 → JSON 退化 → 解析
_parse_extraction()                      脏数据丢弃

IngestReport / ingest() / _run()         唯一入库通道:observe→propose→aggregate
seed_graph(governance, limit)            种子图
seed_problems(knowledge)                 转发给 problem_bank
main()                                   CLI
```

**[细节]**
- ⚠️ **453 行里约 300 行是种子数据常量**,**跳读**
- 看 `ingest()` 就够:它是唯一入库通道,**没有直写后门** —— builder 不做任何绕过治理的捷径
- `extract` 的两条路:优先取 `submit_knowledge` 工具调用;模型不走工具调用时,
  退化到解析 content 里的 JSON(含 ```json 围栏剥壳)

---

## [knowledge/problem_bank.py](coach/knowledge/problem_bank.py) — 220 行

**[意图]**
题库导入。**为什么不用 LLM 现场出题**:生成的 `expected` 一旦写错会直接污染判分和 BKT。

**[骨架]**
```
parse_bank(payload)        校验 + 规整,**脏数据跳过而不是抛异常**
load_from_file(path) / fetch_remote(url)    两条来源
import_problems(knowledge, problems)        跳过悬空引用
expected_answer()          取"正确提交"(= 最后一条用例的 expected)
coverage()                 ★ 哪些知识点没有题
seed_from_bank()           灌内置题库
main()                     CLI
```

**[细节]**
- **`parse_bank` 的容错策略是重点**:外部数据不该让整批失败 —— 跳过坏的、留下好的
- **`coverage()` 是这份代码最有含量的地方**:它把"题库覆盖率"变成一条硬约束 ——
  **没题的知识点永远产生不了作答证据,根因定位在原理上够不着它**。
  这不是洁癖,是 P5 实测到的

---

# 阶段 4:一次请求怎么走 ★(主线,按数据流读)

## ① [events/schema.py](coach/events/schema.py) — 111 行

**[意图]**
定义"消息"长什么样。

**[骨架]**
```
常量:ANSWER_SUBMITTED / PROFILE_UPDATED / TICK_SCHEDULED / PLAN_UPDATED / PROBLEM_GENERATE

Event(event_id, type, ts, learner_id, payload, trace_id, attempt)
  .to_dict() / .from_dict()
  .to_stream_fields() / .from_stream_fields()   ← Redis 字段必须是字符串
  .with_attempt()
new_event(type, learner_id, payload, trace_id)
```

**[细节]**
- **七个字段是固定的**,业务数据全在 `payload` 里
- `event_id` 用 **ULID**:既是幂等键,又保证"按 id 排序 = 按时间排序"
- `to_stream_fields` 把整个信封塞进一个 `data` 字段 —— 因为 Redis Stream 的字段值必须是字符串

---

## ② [api/routes.py](coach/api/routes.py) — 134 行

**[意图]**
**看它有多薄。** 这个"薄"是设计的结果,不是偷懒。

**[骨架]**
```
POST /answer                     校验 → new_event → bus.publish → 202
GET  /gap/{learner_id}/{kp_id}   根因定位
GET  /graph/{kp_id}              前置树
GET  /plan/{learner_id}          今日计划
GET  /profile/{learner_id}       学习者档案
GET  /health
```

**[细节]**
- **`submit_answer` 里没有一行判分、没有一行 LLM、没有一行 SQL** —— 只有校验和入队。
  那 30ms 就是这么来的
- 查询类端点全部走 `request.app.state.query_service`,**不 import 图/画像** ——
  有一条源码扫描测试把这条线锁死
- 用 `def` 而不是 `async def`:redis-py 是同步的,用 `def` 让 FastAPI 丢进线程池,不阻塞事件循环

---

## ③ [workers/profile_worker.py](coach/workers/profile_worker.py) — 114 行

**[意图]**
★ **这个文件只有 114 行,本身就是个信号** —— 它把活全交给 pipeline 了。

**[骨架]**
```
ProfileWorker
├── handled_types = (ANSWER_SUBMITTED,)    recovery 靠它分流
├── process(event)     幂等检查 → journal.record → handle → mark_processed → mark_done
├── handle(event)      按类型分发
├── handle_answer()    ★ 一行:self.pipeline.run(event)
├── run() / stop()     异步消费循环(asyncio.to_thread 包阻塞调用)
└── _process_with_retry()  失败 → attempt+1 重投;> 3 进死信
```

**[细节]**
- **`handle_answer` 只有一行** —— 这是 P4 的转折点。P1~P3 时它这里有判分、BKT、SM-2 一长串,
  P4 全搬进 `pipeline.py` 了。**读懂这个"变薄"就懂了 P4 在干什么**
- `process()` 里的**顺序**是关键:`journal.record` 在业务**之前**("先记后做"),
  `mark_processed` 和 `mark_done` 在业务**之后**
- `asyncio.to_thread` 包阻塞的 SQLite/Redis 调用 —— 这也是为什么连接要 `check_same_thread=False`

---

## ④ [workflow/pipeline.py](coach/workflow/pipeline.py) — 386 行 ★

**[意图]**
一次答题的完整处理链,用 LangGraph 承载。**为什么要 LangGraph**:链一长,
进程在中间被杀就得从头重放,最亏的是 LLM 那步(贵且非幂等)。

**[骨架]**
```
AnswerState(TypedDict)    链上流转的状态(每个节点只返回自己产出的字段)

explain_gaps_with_llm()   和 planner_worker 共用
refresh_explanation()     重算根因 + 写缓存(带 not_before 新鲜度判断)

AnswerPipeline
├── _build()              编译图:5 个节点 + 1 条条件边
├── run(event)            ★ 执行(含"从检查点恢复"的判断)
└── 节点:
    _grade()            规则判分
    _update_profile()   ★ BKT + SM-2 + 答题记录,一个事务
    _detect_gap()       查前置缺口
    _explain()          ★ 调 LLM —— 单独成节点,产出先落检查点
    _finalize()         幂等标记 + journal + 发 profile.updated

make_checkpointer()      测试用 InMemorySaver,生产用 SqliteSaver
```

**[细节]** —— 四件事必须看懂

1. **节点划分与文档不一致**:文档 §2.2 把 BKT 和 SM-2 列为两步,这里**合成一个节点** ——
   因为两者必须原子,拆开会让"BKT 写了、SM-2 没写"成为可能

2. **为什么 `explain` 单独成节点**:它贵且非幂等。单独成节点,它的产出才会先落检查点,
   后面崩了不用重调 LLM。**这是 checkpointer 全部价值的来源**

3. **`run()` 里的恢复判断**:
   ```python
   snapshot = self.graph.get_state(config_dict)
   if snapshot.next:
       return _summarize(self.graph.invoke(None, config=config_dict))
   ```
   ⚠️ 必须是 `invoke(None, ...)` —— 传新 state 会让它**从头开始**,检查点白存。
   第一版就是这么写的,LLM 被重复调用,是测试的**调用计数**抓出来的

4. **文件开头的注释**是面试叙事(为什么 P0~P3 不上 LangGraph)—— 值得细读

---

# 阶段 5:可靠性(工程能力的体现)

## [coordination/journal.py](coach/coordination/journal.py) — 113 行

**[意图]**
**"不丢消息"的真正实现。** 不是靠消息队列,是靠这份 append-only 日志。

**[骨架]**
```
Journal(conn)
├── record(event)       ★ 先记后做;已记过则原样保留(INSERT OR IGNORE),返回 False
├── mark_done(id)
├── is_recorded() / is_done() / get()
├── pending()           ★ 「已记未完成」—— 重启后要补做的清单
└── counts()            {total, pending, done}
```

**[细节]**
- 表只有一次变更:给 `done_at` 盖时间戳。**日志本身只增不改**
- `record` 用 `INSERT OR IGNORE` —— 重放不会覆盖原始记录
- **为什么这一个文件就能保证不丢**:任何时刻进程被杀,已 ACK 前的消息都在 journal 里留着

---

## [coordination/recovery.py](coach/coordination/recovery.py) — 90 行

**[意图]**
重启后把没做完的补上。**两条路**:journal(业务)+ PEL(投递)。

**[骨架]**
```
Recovery(journal, bus)
├── replay(handler, types)        ★ 重放「已记未完成」;handler 失败**不吞异常**,保持未完成
├── reclaim_stale(...)            认领超时未 ACK 的消息
└── run(handler, stream, ...)     启动时的完整恢复
```

**[细节]**
- **`types` 参数为什么必要**:多个 worker 共用一个 journal,
  planner 不能重放 profile_worker 的消息
- `replay` 里 `except: continue` 但**不 mark_done** —— 失败的事件保持"未完成",下次启动还会被捞出来。
  **不吞异常、不假装成功**
- 为什么有了 journal 还要 reclaim PEL:journal 兜住"业务没做完",PEL 兜住"ACK 丢了",两件事

---

## [coordination/redis.py](coach/coordination/redis.py) — 149 行

**[意图]**
Redis Stream 的封装。**只有两个方法值得细看。**

**[骨架]**
```
Bus(Protocol)               worker 依赖协议而非实现 → 测试可以注入内存实现

RedisBus
├── ensure_group()         建消费组,已存在则忽略(BUSYGROUP)
├── publish(stream, event)
├── consume(...)           xreadgroup 读新消息
├── reclaim_stale(...)     ★ xautoclaim 认领别人留下的未 ACK 消息
├── ack() / move_to_dlq() / read_dlq()
└── ping() / close()

dlq_name(stream)           死信流名 = f"{stream}:dlq"
```

**[细节]**
- **`Bus` 协议的意义**:worker 只依赖这个协议,所以测试和 demo 能塞 `InMemoryBus` 进去,
  不需要真 Redis。这是"测试不依赖外部服务"的关键设计
- `ensure_group` 要容忍 `BUSYGROUP` —— 否则每次重启都会因为"组已存在"而崩

---

# 阶段 6:画像怎么算

## [profile/bkt.py](coach/profile/bkt.py) — 86 行

**[意图]**
掌握度。⚠️ **先读开头那段"边界声明"**,它明确说了这个数**只用于排序、不用于预测**,以及为什么。

**[骨架]**
```
update(p_known, correct, ...) → float     §5.3 公式:答对/答错 → 后验 → 转移
update_many(mastery, kp_ids, correct)     一组各更新一次,不改入参
update_sequence(p, observations)          连续喂,评测和测试用
predict_correct(p_known)                  ★ 掌握度 → 下一题答对的概率
```

**[细节]**
- 只有 86 行,一个 `update()` 是主体 —— 公式照抄 §5.3,别改
- **`predict_correct` 就是那个"不可用"的函数**:P5 测出 AUC 0.5336(随机 0.5)。
  它还在,因为**它的失败本身是文档的一部分**
- 文件开头的两个原因值得记住:`P_init=0.1` 系统性低估 + 连续答对约 10 次饱和到 1.0

---

## [profile/sm2.py](coach/profile/sm2.py) — 96 行

**[意图]**
间隔重复。回答"什么时候该复习"。

**[骨架]**
```
SM2State(ease, interval_days, reps, lapses, last_review, due_at)
  .from_row(row)            没有记录就是全新状态

quality_from_correct(bool)  ★ 二值判分 → 0~5 评分
review(state, quality, now) ★ 核心:失败重置 reps/间隔,lapses+1;通过则推进间隔
is_due(state, now)
```

**[细节]**
- `_next_interval` 的关键:**`reps` 是"本次之后"的连续通过次数**(对齐经典 SM-2 的 n=1→1天, n=2→6天)
- `quality_from_correct` 的映射(答对=4/答错=1)是**我定的,文档 §5.4 没规定**。
  选 4 是因为它让 EF 增量正好为 0(不漂移)

---

## [profile/store.py](coach/profile/store.py) — 402 行

**[意图]**
画像持久化。**只挑两处看**,其余是 CRUD。

**[骨架]**
```
DDL:  learner_mastery / learner_memory / learner_error / learner_profile
      answers / processed_events / gap_explanations

ProfileStore
├── 掌握度:  get_mastery / get_mastery_map(批量) / set_mastery
├── ★ observed_kp_ids()    有证据的知识点 —— 让画像能表达"没有记录"
├── ★ transaction()        把多步写合成一个事务
├── _write()               块内不提交,块外立即提交
├── 记忆:    get_memory / set_memory / due_reviews
├── 答题:    record_answer(返回 False=已存在) / list_answers
├── 档案:    upsert_profile / get_profile / list_learners
└── 幂等:    mark_processed / is_processed
```

**[细节]**
- **`transaction()` 是被一次数据丢失逼出来的**:原本答题记录、掌握度、SM-2 各自 commit,
  进程在中间被杀 → 答题记录已落库、掌握度没更新 → 恢复重放看到记录存在就跳过 →
  **该次更新永久丢失**。现在全部包进一个事务
- ⚠️ `transaction()` 有个使用约束写在 docstring 里:`self.conn` 是**共用连接**,
  块内不要调 Journal 的写方法(它自带 commit 会提前提交)
- **`observed_kp_ids()` 是两态区分的数据基础** —— `get_mastery` 对没记录的点返回 0.1,
  和"考过确实弱"的 0.1 **数值一样、含义不同**,这个函数就是把两者分开

---

## [profile/errors.py](coach/profile/errors.py) — 49 行

**[意图]**
很短。**区别在于它不做计数,做分析。**

**[骨架]**
```
top_errors(store, learner, limit)        按次数排序
recurring_types(store, learner, min_kps) ★ 同一错误类型跨 ≥2 个知识点
summary(store, learner)                  {top, recurring}
```

**[细节]**
- `bump_error`(计数)留在 store 里,这个文件只做**读取和分析** —— 职责干净
- **`recurring_types` 的价值**:区分"某个知识点不会"和"某类错误在好多知识点上重复犯"。
  后者的对策不是补单点,是查系统性误解

---

# 阶段 7:计划怎么排

## [workflow/query.py](coach/workflow/query.py) — 397 行

**[意图]**
双重身份:① 计划生成;② **`api/` 唯一依赖的门面**(绕开"api 不许 import 图/画像")。

**[骨架]**
```
QueryService(knowledge, profile)
├── gap_view(learner, kp)        ★ 根因(读缓存解释,没有就模板兜底)
├── graph_view(kp, depth)        前置树 + 已掌握/缺口标记
├── plan_view(learner, now)      ★ 计划:review > remedial > probe > learn
│     _due_items()               到期复习(★ 过滤掉没题的)
│     _goal_items()              根因 + advance 分支 + probe 标记
├── profile_view(learner)        §5.3 契约:goal / mastery_summary / due_now / error_patterns
└── learn_next()                 转发 next_to_learn

build_default_query_service()    装配连真实库的门面
```

**[细节]** —— 两个"被逼出来的"设计

1. **`practicable()` 过滤**:计划只推有题可做的点。不加这道,P5 里 **82% 的推荐是白费的**
2. **advance 分支**:`next_to_learn` 只在**前置闭包**里找候选,**目标自己永远不在里面** ——
   前置补完之后计划会变成空的(实测 24 步里 18 步为空)。补上这一支后表 3 从 -8% 转 +1.2%
3. **"本层不调 LLM"** 是刻意的:`gap_view` 优先读 worker 预先算好的 `gap_explanations` 缓存 ——
   入口层才能保持 < 50ms

---

# 阶段 8:数字怎么来的

## [evaluation/simulated_student.py](coach/evaluation/simulated_student.py) — 171 行

**[意图]**
★ **最关键的设计约束:生成模型必须与 BKT 不同**,否则评测就是循环论证。

**[骨架]**
```
常量 SLOPE=6 / FLOOR=0.02 / CEIL=0.96 / PREREQ_DRAG=0.7 / PRACTICE_GAIN / DECAY_PER_DAY

SimulatedStudent(ability, last_practiced, planted_weakness)   ← ability 是真值
  .own(kp)

StudentSimulator(knowledge, seed)
├── make_student(weakness=...)      造学生,可埋短板(ground truth)
├── effective(student, kp)          ★ 前置拖累:own × (0.3 + 0.7 × min(eff(前置)))
├── p_correct / answer              逻辑函数,不是 BKT 的 slip/guess
├── practice / forget / mark_practiced
└── trace(...)
```

**[细节]** —— 这段是理解"评测为什么可信"的钥匙

- **`effective()` 里的前置拖累是核心假设**:一个学生可能"学过"动态规划,但函数调用很烂,
  于是动态规划也做不出来。**这正是图推理有用的前提**,
  而 BKT 假设各知识点独立,没有这一项
- **答题概率用逻辑函数**(`0.02 + 0.96 × σ(6·(eff−0.5))`),形式和参数都**刻意不与 BKT 重合**
- `forget()` 只对**记过练习时间**的点生效 —— 忘了 `mark_practiced` 整段遗忘实验会静默失效(踩过)

---

## [evaluation/metrics.py](coach/evaluation/metrics.py) — 116 行

**[意图]**
指标。**全是纯函数,好读也好测。**

**[骨架]**
```
auc(scores, labels)               秩和法,含并列名次处理;单类别返回 nan
brier / log_loss
calibration_curve(scores, labels, bins)   分箱:平均预测 vs 实际频率
expected_calibration_error(curve, total)
top1_accuracy(predicted, truth) / recall_at_k
mean(values)
```

**[细节]**
- `auc` 里的 `_average_ranks` 是**并列值取平均名次** —— 掌握度大量并列(都停在 0.1),
  不做这个处理 AUC 会算错
- 单类别时返回 `nan` 而不是抛异常 —— 评测里常有这种情况,不该让整轮失败

---

## [evaluation/baselines.py](coach/evaluation/baselines.py) — 88 行

**[意图]**
对照组。回答"不用图、只用文本相似度行不行"。

**[骨架]**
```
FlatRetrievalBaseline(knowledge)     只认 concepts,完全不认识 edges 表
├── similar(kp_id, top_k, include_self)     字符二元组余弦相似度
└── root_causes(mastery, kp_id, observed=)  ★ observed 参数是关键
```

**[细节]**
- **`observed` 参数为什么必要**:一个从没被观测过的点,`mastery.get` 会落到 0.0 ——
  **比任何真实观测值都低**,基线会去挑一堆毫无证据的点。实测踩过,结果是 0.0 这种没法看的数字
- 文件开头的注释列了**公平性上做的三件事** —— 这类"防止打稻草人"的说明,
  是评测可信度的证据

---

## [evaluation/run_eval.py](coach/evaluation/run_eval.py) — 795 行

**[意图]**
一键四张表。**795 行里一半是 markdown 渲染,跳读。**

**[骨架]**
```
Env                    临时库 + 种子图 + 种子题
exp_mastery_prediction     表 1:AUC / Brier / 校准
exp_forgetting_calibration 表 2:连续练习 vs 隔 30 天
exp_planning_gain          表 3:三臂 + 白费步数
exp_root_cause             表 4 ★:四臂(图 / 消融 / 闭包内随机 / 扁平)
run_all(...)               跑全部
render_markdown()          四张表
render_limitations()       ★ 由数据触发的 Limitations
```

**[细节]**
- **`exp_root_cause` 的四个臂是这份文件最有价值的部分**:只看"图 vs 扁平"会把
  **候选集小**误当成"图推理强"。所以加了「闭包内随机排序」(隔离候选集)和
  「消融」(隔离两态区分)
- **`render_limitations` 全部由数据触发** —— 条件不成立就不写。
  **问题修掉后对应限制自动消失**,不留在文档里误导人
- `Env.submit` 把模拟学生的作答**送进真实答题链**(`pipeline.run`),而不是直接改库 ——
  所以评测算的数字是真实链路产生的

---

# 附:三条捷径

| 如果你… | 读这些 | 时长 |
|---|---|---|
| **只想懂核心** | `queries.py` → `governance.py` → `pipeline.py` 的节点划分 | 40 min |
| **只想懂工程** | `journal.py` → `recovery.py` → `profile/store.py` 的 `transaction` | 40 min |
| **只想懂评测** | `simulated_student.py` 的 `effective()` → `run_eval.py` 的 `exp_root_cause` | 30 min |

---

# 附:读代码时的三个路标

1. **注释写的是「为什么」不是「做什么」**
   —— `bkt.py` 的边界声明、`pipeline.py` 的迁移理由、`schema.py` 的"为什么删列不划算"。
   **这些是设计思路本身,不是废话**

2. **不是所有文件都该逐行读**
   - `builder.py` 453 行里约 300 行是种子数据常量
   - `store.py` 402 行里大半是 CRUD
   - `run_eval.py` 795 行里一半是 markdown 渲染

3. **测试是第二份文档** —— 尤其这四个:
   - [test_queries.py::TestObservedVersusUnobserved](tests/coach/test_queries.py)(两态区分)
   - [test_pipeline.py::TestCheckpointResume](tests/coach/test_pipeline.py)(断点恢复不重跑 LLM)
   - [test_worker.py::TestAtomicity](tests/coach/test_worker.py)(事务)
   - [test_id_drift.py](tests/coach/test_id_drift.py)(同名归并)

4. **`work.md` §11.1 / §12 是"踩坑日志"**
   —— 读完代码再看,能知道哪些设计是被 bug 逼出来的
   (比如 `transaction()` 是被一次数据丢失逼出来的)
