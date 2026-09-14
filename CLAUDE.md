# CLAUDE.md

> 本仓库有**两个独立项目 + 一个归档区**。开工前先读这里,再读 [work.md](work.md)。

## 仓库地图

| 目录 | 是什么 | 状态 |
|---|---|---|
| `agents/document_agent/`、`llm/`、`core/` | **作品集 A**:合同风险审查 Agent | 已完成重构,维护状态 |
| `coach/` | **作品集 B**:学习教练 Agent | **未开工,当前主线** |
| `zanshibuyong/` | 归档区:旧多 Agent 编排、MCP 教学脚本 | 只读,不要改动 |

## 文档导航

| 文档 | 回答什么 | 什么时候读 |
|---|---|---|
| [work.md](work.md) | **怎么做、做到哪了** | **每次开工先读 §9 §10** |
| [mainconten.md](mainconten.md) | 为什么做、做什么 | 需要理解定位时读 §4 |
| [study.md](study.md) | **代码怎么读**(意图→骨架→细节) | 要读懂代码思路时 |
| [PLAN.md](PLAN.md) | 作品集 A 的改造计划 | 只做 coach 时不用读 |
| `LEARNING_COACH*.md` | 过程记录,已被 mainconten.md 取代 | 一般不用读 |

## 开工第一步

1. 读 [work.md](work.md) §1(速览)、§10(进度台账)、§9(恢复指南)
2. 从台账里第一个未完成阶段的任务清单继续
3. 技术细节查 [work.md](work.md) §4(数据模型)、§5(核心要点)——**照抄,不要重新设计**

## 硬约束(coach 项目)

- `coach/api/` **禁止** import `coach/knowledge/` 或 `coach/profile/`
- P0~P3 **不引入**:图库、工作流框架、向量库、前端、Docker
- 图存储用 **SQLite + 递归 CTE**,不上 Kuzu
- 关系四类:`PREREQUISITE` / `RELATED` / `EXTENDS` / `CONTRASTS`
- 间隔重复用 **SM-2**,不用 FSRS
- LLM 抽取的关系**不直接入库**,走 `observation → proposal → aggregator`

## 环境

```bash
.venv/Scripts/python.exe ...                              # Windows,项目自带 venv
docker run -d -p 6379:6379 --name coach-redis redis:7-alpine   # Redis(P1 起需要)
.venv/Scripts/python.exe -m pytest tests/coach/ -q        # 跑 coach 测试
```

## 已知情况(不要"顺手修")

- `tests/test_llm_mcp_client.py` 有 2 个失败:历史遗留,与 coach 无关
- `ttext.py` 是 0 字节的误建文件
- 两个 Agent 的示例合同是**中文合成数据**(demo 用),真实评测数据是 CUAD

## 提交约定

- 只 add 具体文件,**不要 `git add -A`**
- `api.env` / `.env` 含密钥,已被 `.gitignore` 覆盖,**不要提交**
- 提交/推送前先问用户
