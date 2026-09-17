# mtpagent

这个仓库里有两块内容，**只有一块是活跃的**。

```
mtpagent/
├── nlp/              ← 活跃项目：电商售后助手（多 Agent 编排）
└── zanshibuyong/     ← 归档区：旧项目，不要读、不要参考
```

---

## nlp/ —— 电商售后助手

基于 LangGraph 的多 Agent 系统：能答售后政策、查订单物流、办理退换货，
办理写操作前有一道人工审批闸门。

> **详细介绍、架构说明、快速开始，见 [nlp/README.md](nlp/README.md)**

两份过程文档：

| 文档 | 内容 |
|---|---|
| [nlp/study.md](nlp/study.md) | 改造**路线图** —— 阶段计划、里程碑、进度、待评估项 |
| [nlp/idea.md](nlp/idea.md) | **学习笔记** —— 每处改动「思路 / 为什么 / 人类怎么学」 |

一句话上手：

```bash
cd nlp
uv venv --python 3.12 .venv
uv pip install -r requirements.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple
cp .env.example .env          # 填 DASHSCOPE_API_KEY
python tools/scripts/init_ecommerce_db.py
python tools/scripts/init_vector_db.py
python api.py                 # http://localhost:8000/
```

---

## zanshibuyong/ —— 归档区

旧项目（学习教练 Agent、合同风险审查 Agent 等探索性实现）整体归档在这里。

**请忽略里面的所有内容** —— 不再维护、不参与当前主线，读代码或做规划时一律跳过。
归档区自己的 [README](zanshibuyong/README.md) 开头也有同样的声明。

需要时可以整个删除；`git checkout` 也能随时找回。

---

## 关于根目录的 `.venv`

根目录有个 **空的 `.venv`（0 个包，Python 3.14）**，是早期遗留、已弃用。

**不要用它** —— 每个项目自己的依赖在自己的 `.venv` 里（`nlp/.venv`，Python 3.12）。
