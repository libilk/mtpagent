# coach — 把「学不会的根因」定位出来的学习 Agent

> 学生反复做错一类题,根因往往**不在这个知识点本身,而在前置知识**。
> 扁平检索只能推荐"相似题目";只有沿知识图谱**反向遍历前置依赖**,才能定位到
> "你真正缺的是哪个基础"。
>
> 这个项目就做这一件事,并且**用可复现的评测说明它到底有多大用**。

**先说结论,包括不好看的那些。**

---

## 评测结果

四张表,由 `python -m coach.evaluation.run_eval` 一键产出(种子 0,30 个模拟学生,
22 个知识点 / 13 道题)。完整数字在 [RESULTS.md](coach/evaluation/results/RESULTS.md)。

### ★ 表 4:根因定位 —— 图遍历 vs 扁平召回

| 条件 | 臂 | Top-1 | Recall@k |
|---|---|---|---|
| 作答覆盖充分 | **图遍历(本方案)** | **0.6774** | 0.8065 |
| | 图遍历 · 消融(不区分"有没有证据") | 0.6129 | 0.8065 |
| | 闭包内随机排序 | 0.6129 | 0.9839 |
| | 扁平召回(文本相似度) | 0.1290 | 0.3548 |
| 作答稀疏 | **图遍历(本方案)** | **0.7258** | 0.8548 |
| | 图遍历 · 消融(不区分"有没有证据") | 0.6129 | 0.8548 |
| | 闭包内随机排序 | 0.5484 | 0.9516 |
| | 扁平召回(文本相似度) | 0.1774 | 0.4677 |

> 无脑随机猜的 Top-1 是 0.0455(1/22)。
>
> **为什么有三个对照臂**:只看"图 vs 扁平"会把**候选集小**误当成"图推理强"
> —— 图的闭包只有 4~5 个点,而扁平要在 22 个里挑。所以加了
> 「闭包内随机排序」(同样候选集、只不排序)和「消融」(同样候选集与排序公式、
> 只是不区分「未观测」与「已观测且弱」),把变量一个个隔离掉。

### 表 1~3(两张是负面的)

| 表 | 指标 | 结果 | |
|---|---|---|---|
| 1 掌握度预测 | AUC / Brier | **0.5353** / **0.2688** | ❌ 比"恒定预测基准率"(Brier 0.25)还差 |
| 2 遗忘预测 | ECE(隔 30 天) | 0.2397 → **0.3632** | ✅ 印证 BKT 不建模遗忘:预测 0.44,实际 0.08 |
| 3 规划增益 | 相对随机 | **-8.0%** | ❌ 计划没跑赢随机 |

---

## 快速开始

```bash
git clone https://github.com/libilk/mtpagent.git && cd mtpagent
python -m venv .venv && .venv/bin/pip install -r requirements-coach.txt   # Windows: .venv\Scripts\pip
./demo.sh          # 唯一入口,跑完整链路
```

> **零外部依赖** —— 不需要 Redis、Docker 或任何 API key。
> 前三行是「装好环境」,真正的运行就是 `./demo.sh` 这一条。
> (仓库根目录的 `requirements.txt` 是另一个作品集项目的,跑 coach 不需要那些重依赖。)

它会依次展示:

```
① 建图         22 个知识点,前置关系经过「观察 → 提案 → 聚合」治理流程
② POST /answer 交一个正确答案 → HTTP 202,耗时几十毫秒(此时还没判分、没调 LLM)
③ 异步消费     profile_worker 判分 → BKT 掌握度 → SM-2 记忆状态
④ 根因定位     沿前置依赖反向遍历,给出根因 + 依赖路径 + 解释
⑤ 答错一次     掌握度下降、易错计数 +1
⑥ 定时触发     scheduler 扫到期 → 今日计划(复习 / 补根因 / 推进)
⑦ 收尾         journal 记账,同一条消息重复投递不会重复计分
```

## 它是什么 / 不是什么

| ✅ 做 | ❌ 不做 |
|---|---|
| 22 个知识点的小图(数据结构 + 算法) | 上千知识点的全量图 |
| **根因定位** —— 多跳图推理找薄弱基础 | 泛泛的"推荐相似题" |
| BKT 掌握度 + SM-2 间隔重复 | 参数拟合 / 深度学习 |
| Redis 协调 + append-only journal + 重启恢复 | Kafka / 消息队列中间件 |
| FastAPI + Swagger | 前端 |
| **四张量化评测表** | 堆功能 |

---

## 核心设计

### 1. 根因定位:不是"哪个知识点分低",而是"哪块基础没打牢"

```
输入:learner_id, 目标知识点 X
1. P ← 前置闭包(X, 深度 ≤ 3)          ← SQLite 递归 CTE
2. M ← {p: 掌握度(p) for p ∈ P}        ← 一次批量查,不做 N+1
3. 缺口 ← {p ∈ P : M[p] < 0.4}
4. ★ 排序:(有没有证据, 深度, 掌握度)
5. 取前 3 个作为根因,附上从目标走回去的依赖路径
```

**第 4 步的"有没有证据"是这一版最重要的修正。** 闭包里掌握度低的点有两种,
数值可能一样、含义完全不同:

- **有作答证据,确实弱** → 这是真根因
- **从没被观测过**,掌握度只是初始值 → 这是**没测过**,不是弱

早期版本只按 `(深度, 掌握度)` 排,于是大量"从没见过的浅层点"把真正的弱项挤到后面。
消融实验显示,区分这两态带来 **+6.5 / +11.3 个点**的 Top-1 提升。

### 2. LLM 抽的知识不当事实用

```
教材文本 → LLM 抽取 → Observation → 规则校验 → Proposal → DFS 环检测 → Aggregator → 入图
                          ↓ 拒:自环 / 重复 / 置信度 < 0.6 / 悬空引用 / 同名归并
                     被拒提案留在库里,带 reason
```

LLM 的输出是**提案**不是事实。被拒的提案留在 `proposals` 表里,能回答
"这条关系为什么没进图"。

> 一个实测踩到的坑:LLM 给归并排序起 id `algo.merge_sort`,而人工金标准是
> `algo.mergesort` —— 只按 id 判重会在图里长出两个节点。修法是两头:**抽取时把
> 已有概念清单喂给 LLM 要求复用 id**,同时**治理层按名字归并**(同名提案不入库,
> 引用别名的边自动改指规范 id)。真实调用复验:修复前 22 → 23 个节点,修复后 22 → 22。

### 3. "不丢消息"靠的是 journal,不是中间件

用的不是 Kafka,而是 **Redis Stream + append-only journal + 重启恢复**:

```
取消息 → 查 processed_events(已处理则跳过) → ★ 写 journal(先记后做)
       → 执行业务 → 写 processed_events + journal.mark_done + ACK
       → 失败重投;> 3 次进死信
       → 重启:扫 journal 中「已记未完成」的事件补做
```

**幂等做了两道闸**:`processed_events` 挡"同一消息被消费两次",
`answers.answer_id` 唯一键挡"恢复重放把同一步执行两次"。
只做第一道不够 —— 恢复重放会把同一次答题的掌握度更新重复施加。

### 4. 答题链用 LangGraph 承载,断点恢复不重跑已完成的步骤

```
START → grade → update_profile → detect_gap ─┬→ explain(调 LLM)→ finalize → END
                                             └────────────────→ finalize → END
```

`explain` 单独成节点是因为它**贵且非幂等** —— 单独成节点,它的产出才会先落检查点,
后面崩了不用重调 LLM。

> 踩过的坑:langgraph 记住的是「下一步该跑谁」,要接着跑必须 `invoke(None, config)`;
> 传一份新 state 会让它**从头开始**,检查点就白存了。第一版就是这么写的,LLM 被重复
> 调用 —— 是测试里的**调用计数**把它抓出来的。

---

## 架构

```
① 入口层  coach/api/          FastAPI:只做校验 + 入队 + 查询,不含任何 LLM 调用
② 事件与协调  coach/coordination/  Redis Stream · journal(append-only)· recovery
③ 常驻 worker  coach/workers/      profile / planner / scheduler
④ 知识层  coach/knowledge/    图(SQLite + 递归 CTE)· 写入治理 · 根因查询
⑤ 画像层  coach/profile/      BKT 掌握度 · SM-2 记忆 · 易错 · 持久化
⑥ 编排层  coach/workflow/     LangGraph 答题链 + 只读门面
⑦ 评测层  coach/evaluation/   模拟学生 · 指标 · 对照组
```

**分层硬约束:依赖只能向下,`api/` 不许 import `knowledge/` 或 `profile/`。**
查询走 `workflow/query.py` 只读门面,并且有一条**源码扫描测试**把这条线锁死。

---

## Limitations(必读)

诚实交代边界,完整版见 [RESULTS.md](coach/evaluation/results/RESULTS.md)。

1. **这是模拟学生,不是真人。** 生成模型与 BKT **刻意不同**(有前置拖累、逻辑函数、
   遗忘),所以不是循环论证;但结论只在"这套假设"里成立。它验证的是**算法能否收敛到
   一个未知的真实过程**,不是"学生用了会不会变好"。
2. **扁平基线偏弱。** 对照组用的是字符 n-gram 相似度,不是真正的向量 embedding。
   真 embedding 语义泛化更强,很可能显著缩小差距。
3. **BKT 的掌握度预测基本不可用**(表 1)。AUC 0.5353,Brier 比恒定基准率还差。
   原因:`P_init = 0.1` 系统性低估,且连续答对约 10 次后 `p_known` 饱和到 1.0。
   这是照抄公式的必然结果,不是实现 bug。
4. **计划没跑赢随机**(表 3,-8.0%)。这个功能的价值**没有被这次实验证实**。
5. **没有题目的知识点,根因定位在原理上做不到。** 它永远不会产生证据,因此永远无法
   被定位为根因。这类点现在会被标成 `status="unobserved"` 并提示"该去测一下",
   而不是伪装成"已确认薄弱"。
6. **规模小。** 22 个知识点、13 道题、30 个模拟学生,单个数字有随机波动。

---

## 用真实依赖跑

demo 用的是进程内总线。要跑真实的 Redis 链路:

```bash
docker run -d -p 6379:6379 --name coach-redis redis:7-alpine

# 建图 + 灌题 + 建档(只碰 SQLite,不依赖 Redis)
python -m coach.knowledge.builder --build
python -m coach.workers.run_all --seed-only --init-learner u1 --goal algo.dp

# 起服务
python -m uvicorn coach.api.main:app --port 8000      # Swagger: /docs
# 另开一个终端:起 worker + 定时 tick(默认 60 秒一次)
python -m coach.workers.run_all --tick-interval 60
```

配了 `DASHSCOPE_API_KEY` 时,根因解释会由 LLM 生成;没有则自动降级为模板文案。

**跑评测:**

```bash
python -m coach.evaluation.run_eval --n 30
```

**跑测试:**

```bash
python -m pytest tests/coach/ -q     # 242 项
```

**依赖**:`fastapi` `uvicorn[standard]` `redis` `langgraph` `langgraph-checkpoint-sqlite`
(见 [requirements-coach.txt](requirements-coach.txt))。LLM 复用仓库已有的 `llm/llm_client.py`。

---

## 对标

市面上有两个成熟的学习平台:[WeSmartFlow](https://github.com/Tencent/WeSmartFlow)(腾讯)
和 [DeepTutor](https://github.com/HKUDS/DeepTutor)(港大)。

**它们都在讲"我有什么能力",但都没有公开"我的效果有多好"。**
DeepTutor 列了十几种检索引擎,一个效果数字都没有。这个项目就补这一块 ——

> 不做平台,做**一个可测量的点**:不重复造轮子,把根因定位做到**有数字**。

读它们源码后主动砍掉/修正的东西(见 [work.md](work.md) §11.3):

- **写入治理流程**(observation → proposal → aggregator)借鉴自 WeSmartFlow;
- **journal + recovery** 借鉴自 DeepTutor 的 `runtime/coordination/`;
- **四类知识关系** —— 我原本照搬了这个说法,读源码后发现是**错记**:
  WeSmartFlow 实际有 8 类关系,而且 `relation_semantics.py` 不是"关系语义表",
  是一组 **embedding 探针**(每类关系配几句学生口吻的 probe,给 query 打分决定
  遍历哪些关系类型)。它自己也没有 per-type 分支逻辑。
  **所以我只保留 `PREREQUISITE` 并做深** —— 与其维持"四类关系"的说法被问倒,
  不如说清楚这是评估后的主动裁剪。

---

## 文档导航

| 文档 | 内容 |
|---|---|
| [mainconten.md](mainconten.md) | 定位与设计总纲(为什么做、做什么) |
| [work.md](work.md) | 实施手册与进度台账(怎么做、做到哪、卡在哪) |
| [coach/evaluation/results/RESULTS.md](coach/evaluation/results/RESULTS.md) | 完整评测数字 + 7 条 Limitations |
