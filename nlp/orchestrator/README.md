# Orchestrator - 核心编排组件

## 📋 目录说明

本目录包含**框架无关**的核心编排逻辑，被 LangGraph 编排框架调用。

## 🎯 设计原则

- **框架无关**：不依赖任何特定编排框架
- **可复用**：所有组件都可独立调用
- **稳定**：核心业务逻辑，不轻易修改

## 📦 核心组件

### 1. TaskPlanner (`planner.py`)
- **职责**：LLM 任务规划，生成 DAG
- **接口**：`plan(query, agents) -> Dict`
- **调用者**：`langgraph_orchestrator/`

### 2. AgentRouter (`router.py`)
- **职责**：智能路由（向量相似度 + LLM 精排）
- **接口**：`route(task, agents) -> str`
- **调用者**：`langgraph_orchestrator/`

### 3. AgentRegistry (`registry.py`)
- **职责**：Agent 注册和管理
- **接口**：`register()`, `get_agent()`, `get_all_agents()`
- **调用者**：`langgraph_orchestrator/`

### 4. ParameterAligner (`parameter_aligner.py`)
- **职责**：参数对齐与动态填槽
- **接口**：`validate_input()`, `fill_parameters()`
- **调用者**：`langgraph_orchestrator/`

## 🔄 使用方式

### LangGraph 模式（唯一模式）
```python
from langgraph_orchestrator.enhanced_entry import EnhancedLangGraphRAGSystem

system = EnhancedLangGraphRAGSystem()
result = system.handle_query(query)
```

## ⚠️ 注意事项

- 本目录的代码**不应该**依赖 `langgraph_orchestrator/`
- 修改本目录代码时，需要确保向后兼容
- 新增功能优先考虑在 `langgraph_orchestrator/` 中实现
