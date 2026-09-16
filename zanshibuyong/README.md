# zanshibuyong —— 暂时不用的归档区

> **本目录内容全部忽略。** 归档区,不再维护、不再被引用;读代码、做规划、写文档时一律跳过,不要读取也不要参考。

这里是从主项目移出来的**教学 / 探索性代码**，暂时不参与"合同风险审查 Agent"这条主线。

**移出来不等于没价值**：这些是我在学习 agent 各个技术点时写的实现，代码本身是完整的。
只是它们不属于合同审查主线，留在根目录会让项目显得"什么都做了一点"。

## 目录

| 路径 | 内容 | 原用途 |
|---|---|---|
| `learn/` | MCP 三种实现（raw / fast / server）、从零写 ReAct、chunking demo | 学习 MCP 协议与 ReAct 循环 |
| `core/agent_from_scratch.py` | 5 步渐进式 agent 教学（裸 API → LangChain → Runnable → AgentExecutor → Memory） | 理解 tool-calling 底层机制 |
| `core/crm_mock.py` | Mock CRM（用户/工单） | `agent_from_scratch` 的示例工具集 |
| `agents/knowledge_agent/` | 通用知识 RAG Agent（2700+ 行，含混合检索、查询优化、上下文压缩） | 客服/知识问答场景 |
| `orchestrator/` | 多 Agent 编排：planner(任务规划) + router(向量+LLM 精排) + executor(ReAct) + registry | 多 Agent 协作调度 |
| `core/document_filter.py` | 基于角色的文档分层过滤 | knowledge_agent 的权限过滤 |
| `core/semantic_cache.py` | 语义缓存（基于向量相似度） | 问答缓存 |
| `core/tavily_search_service.py` | 联网搜索服务 | knowledge_agent 外部工具 |
| `core/sqlite_mcp_service.py` | SQLite MCP 服务 | knowledge_agent 数据查询 |
| `data/knowledge/`、`data/crm/` | 上述模块的示例数据 | — |

## 关于运行

这些模块**依赖主仓库的 `llm/` 和部分 `core/`**（如 `core/unified_cache`、`core/unified_monitoring`），
因此不能脱离根目录独立运行。`orchestrator/` 内部还有 `from orchestrator.xxx import ...` 的绝对导入。

如果要跑其中某一个，推荐做法：

1. 把需要的模块临时拷回根目录对应位置；或
2. 写个 shim 调整 `sys.path`，让 `llm/`、`core/` 和本目录都在路径上。

主线的 `main.py` 已不再依赖这里的任何模块。

## 为什么留下而不是删掉

保留是为了：

- **参考**：这些实现以后可能被主线复用（例如 `learn/` 里的 MCP 客户端，`rag_core` 的检索思路）
- **诚实记录**：探索过什么、踩过什么坑，本身是项目历史的一部分
- **可复现**：需要时可以随时取回，而不是从 git 历史里翻

若确认长期不用，可以整个目录删除（`git tag pre-refactor` 已保存改造前快照）。
