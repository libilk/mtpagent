# 改造计划：合同风险审查 Agent + 评测驱动

> 目标：把这个 15k 行的"agent 大杂烩"重做成一个**有数字、可复现、能讲故事**的作品集项目。
> 主线：合同风险审查 Agent + CUAD 公开数据集评测 + 三种策略消融对比。

---

## 0. 成功标准（做完之后 repo 应该长什么样）

一个面试官 clone 下来，能在 15 分钟内得到这个：

```bash
pip install -r requirements.txt
python -m eval.runner --strategy all --n 50     # 跑三种策略
```

然后看到一张表：

| Strategy | Recall | Precision | F1 | Latency (p50) | Tokens/合同 |
|---|---|---|---|---|---|
| A. Single-shot prompt | 0.61 | 0.72 | 0.66 | 2.1s | 4.2k |
| B. RAG-only | 0.70 | 0.69 | 0.69 | 2.8s | 6.1k |
| C. ReAct agent (ours) | 0.88 | 0.71 | 0.78 | 6.4s | 15.3k |
| C+ cache/concurrency | 0.88 | 0.71 | 0.78 | 2.9s | 15.3k |

**过关线**：这张表是真实跑出来的，且 README 里写清楚了数据集、匹配规则、指标定义。没有这张表，改造就不算完成。

---

## 1. 现状诊断

### 1.1 现有结构

```
main.py                  RAGSystem 入口 → Orchestrator
orchestrator/            planner(DAG) + router(向量+LLM精排) + executor(ReAct) + registry
agents/
  knowledge_agent/       2724 行，通用知识 RAG
  document_agent/        1053 行，合同审核（ReAct），← 保留这条
rag_core/                混合检索/BM25/chroma/query优化/context压缩/evaluator(322行，非本次eval)
core/                    crm_mock / industry_standards / unified_cache / monitoring / error_handler
                         filesystem_service(MCP) / sqlite_mcp / tavily_search / semantic_cache
llm/                     llm_client / function_calling / embedder / mcp_client / output_parser
learn/                   MCP + ReAct 教学脚本
data/                    contracts(中文合成) / crm / knowledge / metadata
```

### 1.2 DocumentAgent 现有流程

`handle()` → `_handle_react()`：手写 ReAct 循环，最多 5 轮，用 `self.llm.chat(messages, tools=tools)` 调模型，`parse_tool_calls()` 解析，`function_calling.execute_function()` 执行。

注册了 9 个工具：`parse_document` / `extract_structure` / `identify_risks` / `compare_with_history` / `calculate_risk_score` / `search_similar_contracts` / `search_by_supplier` / `list_contracts` / `read_contract_file`。

### 1.3 阻塞评测的 5 个问题（必须先解决）

| # | 问题 | 位置 | 影响 |
|---|---|---|---|
| 1 | **输出是自由文本**，`handle()` 返回 `str`（"审核结论 + 建议"） | [agent.py:281](agents/document_agent/agent.py#L281) `handle()` | 无法与 gold 标签比对，**根本没法算指标** |
| 2 | **`LLM.chat()` 返回 string**，tool_calls 被 `json.dumps` 成字符串 | [llm_client.py:135](llm/llm_client.py#L135) | 拿不到 token usage，无法统计成本；结构化解析脆弱 |
| 3 | **缓存会污染延迟测量**，`identify_risks` 命中缓存直接返回 | [agent.py:639](agents/document_agent/agent.py#L639) | 消融实验的延迟数字不可信 |
| 4 | **文本被截断到 3000 字符** | [agent.py:543](agents/document_agent/agent.py#L543) | CUAD 合同普遍超长，关键条款被切掉 |
| 5 | **风险类型是固定 7 类硬编码 prompt** | [agent.py:627](agents/document_agent/agent.py#L627) | 与 CUAD 的 41 个条款类别对不上，没法映射打分 |

> 问题 1 是核心。改造的本质是：**把 Agent 从"写一段话"变成"输出一个可校验的 JSON 结构"**。

---

## 2. 删除清单

处置原则：**全部保留，不删代码**。凡是不进主线的，一律 `git mv` 到 `zanshibuyong/`（"暂时不用"），保留其可运行性，作为独立的学习/教学资料。

`zanshibuyong/` 的定位：这些是我学习和探索 agent 各技术点时写的代码，**代码本身是有价值的**（MCP 三种实现、ReAct 从零写、混合检索等），只是不属于"合同风险审查"这条主线。README 末尾用一节"相关探索"指过去即可——既保住了这些工作，又不让主线显得"什么都做"。

| 路径 | 行数 | 处置 | 原因 |
|---|---|---|---|
| `agents/knowledge_agent/` | 2724 | 移 `zanshibuyong/` | 通用知识 RAG，与合同风险审查无关，且是"什么都做"味道的来源 |
| `data/knowledge/` | — | 移 `zanshibuyong/` | 同上 |
| `data/crm/` | — | 移 `zanshibuyong/` | 客服 mock |
| `core/crm_mock.py` | 208 | 移 `zanshibuyong/` | 客服 mock |
| `core/document_filter.py` | 245 | 移 `zanshibuyong/` | 给 knowledge_agent 用的分层过滤 |
| `core/semantic_cache.py` | 137 | 移 `zanshibuyong/` | 与 unified_cache 重复，主线只留一套 |
| `core/persistent_cache.py` | 114 | 移 `zanshibuyong/` | 同上 |
| `core/sqlite_mcp_service.py` | 166 | 移 `zanshibuyong/` | 主线不需要 |
| `core/tavily_search_service.py` | 159 | 移 `zanshibuyong/` | 联网搜索，主线不需要 |
| `ocr_tools/` | — | 移 `zanshibuyong/` | 扫描件 OCR，可作为后续扩展，不进主线 |
| `learn/` | ~1500 | 移 `zanshibuyong/learn/` | MCP 三种实现 + ReAct 教学脚本，**保留原样可运行** |
| `core/agent_from_scratch.py` | 681 | 移 `zanshibuyong/` | 5 步 agent 教学，质量不错，和 `core/crm_mock.py` 一起移（它 import 了 crm_mock） |
| `tests/` 下现有 12 个测试 | — | 依赖已移模块的测试跟着移 `zanshibuyong/`；能过的留在主线 | 测试挂了比没测试更糟；移走的模块其测试也要能独立跑 |
| `orchestrator/` | ~1200 | **决策点，见 §7** | 单 Agent 主线不需要；但它是"多 Agent 编排"的能力证明 |
| `agents/document_agent` 里的 `compare_with_history` / `search_by_supplier` / `search_similar_contracts` / `list_contracts` / `read_contract_file` | — | **决策点，见 §7** | 依赖 retriever/MCP；CUAD 评测用不上（下面详述） |

**`data/contracts/` 里的中文合成合同**：保留 2–3 份做 demo 冒烟测试，但 README 里必须**明说这是合成数据用于演示**。别让面试官以为你没用真数据。真数据是 CUAD。

---

## 3. 新增清单

### 3.1 核心重构（不改功能，改接口）

**`agents/document_agent/agent.py` — 结构化输出**

新增一个方法，返回机器可读结构，`handle()` 的自由文本作为它的下游渲染：

```python
@dataclass
class RiskFinding:
    category: str      # CUAD 类别，如 "Uncapped Liability"
    evidence: str      # 原文片段（用于 span 匹配）
    level: str         # high/medium/low
    rationale: str     # 一句话理由

def review_structured(self, contract_text: str) -> list[RiskFinding]:
    """主入口：返回可打分的风险清单，而不是一段话"""
```

要点：
- 让 LLM 输出 JSON 数组，每个元素**必须带 `evidence` 原文片段**——这是能算 recall 的关键。
- `evidence` 要能在原文里定位到（后续做 span 匹配用）。

**`llm/llm_client.py` — 补 usage 统计**

`_chat_with_openai` 里把 `response.usage`（prompt_tokens / completion_tokens）透出来，`chat()` 返回结构化结果或通过回调上报。评测需要 token 成本。

**缓存旁路**：给 `DocumentAgent` 加 `enable_cache` 开关，评测时关掉（或让 `eval/runner` 用不同 cache key）。

### 3.2 `eval/` 模块（本次改造的核心产出）

```
eval/
  __init__.py
  datasets/
    cuad_loader.py     # 下载/读取 CUAD，返回 [(contract_text, [gold_annotations])]
    cuad_taxonomy.py   # 41 个类别 → 风险等级的映射表
  gold.py              # CUAD 原始标注 → 统一的 gold 结构
  scorer.py            # 预测 vs gold，算 P/R/F1
  strategies.py        # 三种策略的统一接口
  runner.py            # CLI：跑 N 份合同 × 各策略，收集指标
  report.py            # 输出 markdown 表格 + JSON 原始结果
```

各文件职责：

**`datasets/cuad_loader.py`**
- CUAD 来源：HuggingFace `theatticusproject/cuad`（CC BY 4.0），或 Zenodo 原始 zip。写死一个可复现的下载方式 + 本地缓存。
- 返回 `(contract_id, text, annotations)`；annotations 含 `category` + `answer_start`/`answer_text` 字符偏移。
- **先下 10 份跑通，再扩到 50–100 份**（NL 推理要花钱，别一上来 510 份）。

**`datasets/cuad_taxonomy.py`**
- CUAD 41 类里挑出**真正算"风险"的子集**，映射到 high/medium/low。例如：
  - high：`Uncapped Liability`、`Non-Compete`、`Exclusivity`、`Most Favored Nation`、`Termination for Convenience`
  - medium：`Anti-Assignment`、`Change of Control`、`IP Ownership Assignment`、`Renewal Term`
  - 其余归为 not-risk（不进 recall 分母，但留意 precision 的假阳性）
- **这张表要写进 README**，因为它定义了你说的"风险"是什么——面试会问。

**`eval/scorer.py`**
- 匹配规则（必须明写并保持稳定）：
  - **类别匹配 + span 匹配**：预测的 `evidence` 和 gold 的 `answer_text` 做字符级/词级 IoU，阈值 ≥ 0.5 且类别一致 → 算 TP。
  - 只对类别匹配、span 不匹配的，算"类别对但位置错"（单独统计，调试用）。
- 指标：micro P/R/F1 + macro F1 + 每类别 F1。**micro 是主指标**。
- 边界：一份合同一个类别可能有多处 gold → 用一对多匹配，避免重复计数。

**`eval/strategies.py`** —— 三种策略统一接口 `run(contract_text) -> (findings, usage)`：
- `A_single_shot`：一整段 prompt，一次性让模型吐出全部风险 JSON。无工具、无检索。
- `B_rag_only`：先检索相似条款/历史合同（用现有 retriever），再把检索结果塞进 prompt 让模型判断。**只做一轮，不循环**。
- `C_react_agent`：现有 DocumentAgent 的 ReAct 循环（`review_structured`）。

三种策略共用**同一套输出 schema 和同一个 scorer**——这是消融实验可比的前提。

**`eval/runner.py`**
- CLI 参数：`--strategy {A,B,C,all}`、`--n`、`--seed`、`--workers`（并发）、`--no-cache`。
- 每份合同记录：findings、latency、prompt/completion tokens、retry 次数。
- 结果落 `eval/results/{timestamp}.json`，保证可复查。

**`eval/report.py`**
- 读 json → 生成 §0 那张 markdown 表，可直接贴 README。

### 3.3 其他新增

| 新增 | 作用 |
|---|---|
| `tests/test_scorer.py` | 用构造的假预测/gold 测匹配逻辑，**这是最该写测试的地方**（打分器错了，全部数字都没意义） |
| `tests/test_eval_smoke.py` | 用 2 份 CUAD 合同跑通端到端，防止重构破坏 |
| `requirements.txt` 补 `datasets`（或 `requests`+本地解析） | 数据加载 |
| `README.md` 重写 | 见 §6 |

---

## 4. 分阶段流程

每阶段有**可验收产出**，做完一阶段才进下一阶段。

### Phase 0 — 冻结现状（0.5 天）
- `git tag pre-refactor`，保证任何时候能回退。
- 跑通现有 `main.py`，录一段现状输出（作为"改造前"对比素材）。

### Phase 1 — 裁 repo + 结构化输出（2–3 天）
- 按 §2 建 `zanshibuyong/`，`git mv` 相关目录。
- 修掉 §1.3 的问题 1/2/4：`review_structured()` + LLM usage 透出 + 去掉截断（改成分块/长上下文）。
- 重写 `main.py`：只保留合同审查 CLI，去掉 Orchestrator 依赖（若 §7 决定移走）。
- **验收**：`python -c "from agents.document_agent.agent import DocumentAgent; ..."` 能对一份合同输出 `RiskFinding` 列表，且 `evidence` 能在原文里 `find()` 到。

### Phase 2 — 数据（2 天）
- 写 `cuad_loader.py` + `cuad_taxonomy.py`。
- 下载 CUAD，随机抽 10 份，人工过一遍 gold 标注，确认解析正确。
- **验收**：`python -m eval.datasets.cuad_loader --n 10` 打印出合同文本长度 + 类别分布。

### Phase 3 — 打分器（1.5 天）
- 写 `scorer.py` + `tests/test_scorer.py`。
- **验收**：所有 scorer 单测通过；手工构造一个"完美预测"和一个"全错预测"，分别得 F1=1.0 和 0.0。

### Phase 4 — 三策略 + 跑分（2–3 天）
- 写 `strategies.py` / `runner.py` / `report.py`。
- 先跑 `--n 10` 调通，再跑 `--n 50` 出正式数字。
- **验收**：拿到 §0 那张表。若 C 没有明显优于 A/B，**不要粉饰**——去分析原因（这才是面试的加分点）。

### Phase 5 — 优化 + 可视化（2–3 天，可选）
- 针对 Phase 4 暴露的瓶颈做优化：缓存（评测时旁路、生产时开启）、并行工具调用、prompt 压缩。
- 加一个最小 trace 可视化（Streamlit 最省事）：每份合同 → ReAct 每轮的思考/工具/结果时间线。
- 录 2 分钟 demo 视频，放 README。

### Phase 6 — 收尾（1 天）
- 重写 README（§6），把表、方法论、复现步骤写清楚。
- 清理日志噪音（现在满屏 `✓ xxx 初始化完成`）。

---

## 5. 评测方法论（面试官一定会追问，先想清楚）

**Q: 你说的"风险"是什么？**
A: CUAD 的 41 个条款类别中，我选了 N 类定义为风险，映射到 high/medium/low，表在 README。这是我做的第一个主观决定，也是可争论的——我选择把它显式化而不是藏在 prompt 里。

**Q: 怎么算"识别对了"？**
A: 类别一致 **且** evidence 与 gold 文本片段的 token IoU ≥ 0.5。只类别对算部分正确，单独统计。

**Q: 为什么用 CUAD？**
A: 公开、有专家标注、510 份真实（SEC EDGAR）合同、13k+ 标注。可复现——别人能跑出同样的数字。

**Q: A/B/C 三种策略公平吗？**
A: 所有策略共用输出 schema、同一个 scorer、同一批合同、同一温度（0.0/0.2）。只有"给模型的信息和迭代方式"是变量。

**Q: 最大的坑？**
A: 数据是英文法律文本，而现有 prompt 是中文的 → Phase 1 要把 prompt 改英文（或双语）。这是 Phase 2 最容易踩的雷。

---

## 6. README 结构（重写后）

```
# Contract Risk Review Agent (evaluated)

一句话：一个能定位合同风险条款的 ReAct agent，在 CUAD 上做了评测。

## Results        ← §0 那张表，放最前面
## What it does   ← 一句话 + 一张流程图 + demo 截图
## Evaluation     ← 数据集 / 匹配规则 / 指标定义 / 复现命令
## Architecture   ← 只讲 Agent loop，一张图
## Ablation       ← A/B/C 差异分析 + 为什么 ReAct 更好（或不好）
## Limitations    ← 诚实写：英文数据、类别映射主观、未测长合同
## Quickstart     ← 3 行命令
```

**Limitations 一节不要省。** 写清楚局限反而显得可信，也是面试的谈资。

---

## 7. 需要你拍板的决策点

| 决策 | 选项 | 我的建议 |
|---|---|---|
| **要不要保留多 Agent 编排（orchestrator/）** | (a) 移走，单 Agent 主线最干净 (b) 保留，作为"能编排多 Agent"的额外证明 | **(a)**。作品集讲一个好故事 > 展示两个半成品。若想保留，在 README 末尾加一节"相关探索"指向 `zanshibuyong/`。 |
| **历史对比/供应商检索那几个工具留不留** | (a) 全移走，CUAD 评测用不到 (b) 保留，加一个"历史合同对比"的次要 demo | **(a)** 先移走。它们让"这个 agent 到底评测了什么"变模糊。Phase 5 有余力再加回来做扩展。 |
| **评测规模** | (a) 10 份快速迭代 (b) 50 份正式 (c) 全 510 份 | **先 10，再 50**。510 份 NL 推理烧钱且慢，portfolio 不需要。README 里写"抽了 50 份"。 |
| **CUAD 下载方式** | (a) `datasets` 库 (b) 手动下载 zip 写进 `eval/datasets/` | **(b)** 更可控、离线可复现；把 copyright/license 说明写进 `eval/datasets/README`。 |
| **模型** | 继续 DashScope qwen，还是换 Claude/GPT | 保持 qwen（有 key、便宜）。评测的意义是指标，不是模型。README 注明模型即可。 |

---

## 8. 面试叙事（10 分钟版）

> "我原本写了个大而全的 agent 项目，有 RAG、多 Agent 编排、一堆优化模块。但我发现一个问题：**我没法回答它到底好不好的**。
>
> 所以我砍掉了 80% 的代码，只留合同风险审查这一条线，然后建了评测：用 CUAD 公开数据集当测试集，定义了'识别对'的匹配规则，跑了三种策略的消融。
>
> 结果发现：单次 prompt 召回 61%，多轮 ReAct 到 88%——**但延迟涨了 3 倍**。于是我去查延迟花在哪，发现是串行工具调用和重复的 LLM 调用，加了缓存和并发压回到 1.4 倍。
>
> 这个过程教会我的不是'怎么调 API'，而是**没有评测的 agent 项目都只是 demo**。"

这条叙事的三个支点：**砍代码的克制** + **可复现的数字** + **针对数字做的优化**。这三点恰好是绝大多数 agent 作品集缺的。

---

## 附：立即可以开始的第一步

Phase 0 + Phase 1 的第一半：`git tag pre-refactor` → 建 `zanshibuyong/` → `git mv` 移走 knowledge_agent。这一步零风险、可回退，做完 repo 立刻从"大杂烩"变成"一个合同审查项目"的雏形。
