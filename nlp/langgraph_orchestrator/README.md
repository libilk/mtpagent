# LangGraph 编排系统完整迁移方案

## 📋 目录结构

```
langgraph_orchestrator/
├── README.md                    # 本文档
├── __init__.py                  # 包初始化
├── enhanced_state.py            # 状态定义
├── nodes.py                     # 核心节点
├── enhanced_nodes.py            # 增强节点（重复检测、人工介入）
├── router.py                    # 路由逻辑
├── enhanced_graph.py            # 图构建
├── enhanced_entry.py            # 主入口
├── parameter_alignment.py       # 参数对齐节点
├── clarification.py             # 参数验证节点
└── critic.py                    # Critic 验证节点
```

## 🎯 功能对比

### LangGraph vs 手动编排

| 功能 | LangGraph（默认） | 手动编排（可选） |
|------|------------------|----------------|
| 任务规划 | ✅ | ✅ |
| 智能路由 | ✅ | ✅ |
| ReAct 执行 | ✅ | ✅ |
| 质量评估 | ✅ | ✅ |
| 并行执行 | ✅ | ❌ |
| 参数验证 | ✅ | ✅ |
| 参数对齐 | ✅ | ✅ |
| Critic 验证 | ✅ | ✅ |
| 重复检测 | ✅ | ✅ |
| 状态持久化 | ✅ | ❌ |
| 流式输出 | ✅ | ❌ |
| 可视化 | ✅ | ❌ |

## 🚀 快速开始

```bash
# 使用 LangGraph 编排（唯一模式）
python main.py
```

### 代码配置

```python
from langgraph_orchestrator.enhanced_entry import EnhancedLangGraphRAGSystem

system = EnhancedLangGraphRAGSystem(
    auto_update_index=True,
    enable_checkpointer=True,
    enable_parameter_validation=True,
    enable_critic=False,  # 需要 critic_agent
)
```

## 📦 依赖安装

```bash
# 基础依赖
pip install langgraph langchain-core

# 状态持久化
pip install aiosqlite  # 或 pip install redis（使用 RedisSaver）

# 可视化（可选）
pip install grandalf
```

## 🔧 配置说明

### 1. 启用/禁用功能

```python
system = EnhancedLangGraphRAGSystem(
    enable_checkpointer=True,        # 状态持久化
    enable_parameter_validation=True,       # 参数验证与重试
    enable_critic=True,              # Critic 验证（需要 critic_agent）
)
```

### 2. Checkpointer 配置

```python
# SQLite（默认）
checkpointer = SqliteSaver(conn)

# Redis（生产环境推荐）
from langgraph.checkpoint.redis import RedisSaver
checkpointer = RedisSaver(redis_client)
```

### 3. 断点恢复

```python
# 保存状态
result = system.handle_query("什么是RAG？", thread_id="user_123")

# 恢复状态（继续之前的会话）
result = system.handle_query("详细解释一下", thread_id="user_123")
```

## 📊 核心功能详解

### 1. 参数验证与重试

**触发条件**：下游 Agent 检测到上游输出不足

**Agent 返回格式**：
```json
{
  "message_type": "parameter_validation",
  "target_task": "task_1",
  "missing_info": "缺少用户ID",
  "reason": "需要用户ID才能查询订单"
}
```

**流程**：
1. `parameter_validator_node` 检测请求
2. `upstream_retry_node` 重新执行上游任务
3. 注入验证要求到上游任务描述
4. 最多重试 2 次

### 2. 参数对齐

**前提**：任务定义了 `input_schema` 和 `parameter_mapping`

**示例**：
```python
task = {
    "task_id": "task_2",
    "agent_id": "database_agent",
    "input_schema": {
        "required_fields": ["user_id"],
        "field_types": {"user_id": "str"}
    },
    "parameter_mapping": {
        "user_id": "task_1.extracted_user_id"
    }
}
```

**流程**：
1. `parameter_alignment_node` 验证参数
2. 从依赖任务提取参数（动态填槽）
3. 验证类型和完整性
4. 失败时返回错误，成功时注入 `filled_parameters`

### 3. Critic 验证

**前提**：注册了 `critic_agent`

**验证维度**：
- 参数完整性
- 数据类型匹配
- 语义一致性
- 质量评估
- 冲突检测

**流程**：
1. 每个 Agent 执行后调用 `critic_validation_node`
2. Critic Agent 分析任务衔接
3. 返回验证结果和改进建议
4. 质量低于 0.6 时触发重试

### 4. 重复检测

**机制**：
- 保存历史结果指纹（前 500 字符）
- 检测到重复时停止重试
- 避免无意义的循环

### 5. 人工介入

**触发条件**：
- 质量评分 < 0.4 且重试 >= 2 次
- Critic 验证失败

**流程**：
1. `human_intervention_node` 检测条件
2. 设置 `human_intervention_required = True`
3. 流程暂停，等待人工处理

## 🎨 可视化

```python
# 生成 Mermaid 图
from IPython.display import Image, display

display(Image(system.graph.get_graph().draw_mermaid_png()))
```

## 📈 性能对比

| 指标 | 手动编排 | LangGraph 基础版 | LangGraph 增强版 |
|------|---------|-----------------|-----------------|
| 简单查询延迟 | ~2s | ~2.5s | ~3s |
| 复杂查询延迟 | ~8s | ~6s | ~7s |
| 并行执行 | ❌ | ✅ | ✅ |
| 内存占用 | 低 | 中 | 中高 |
| 可调试性 | 中 | 高 | 高 |

## 🐛 调试技巧

### 1. 查看执行日志

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 2. 查看状态快照

```python
for event in system.handle_query_stream("测试"):
    print(f"[{event['node']}]", event['output'])
```

### 3. 检查 Checkpointer

```python
# 查看所有会话
import sqlite3
conn = sqlite3.connect("./data/langgraph_checkpoints.db")
cursor = conn.execute("SELECT * FROM checkpoints")
print(cursor.fetchall())
```

## ⚠️ 注意事项

1. **LangGraph 是唯一编排框架**：手动编排已废弃
2. **参数验证需要 Agent 支持**：Agent 需要返回特定格式
3. **Critic 需要注册**：在 `config/agents.yaml` 中启用 `critic_agent`
4. **Checkpointer 线程安全**：生产环境使用 Redis

## 🔄 迁移路线图

### 阶段1：基础迁移（已完成）
- ✅ 状态定义
- ✅ 基础节点
- ✅ 图构建
- ✅ 入口封装

### 阶段2：高级功能（已完成）
- ✅ 参数验证与重试
- ✅ 参数对齐
- ✅ Critic 验证
- ✅ 重复检测
- ✅ 状态持久化

### 阶段3：生产优化（待完成）
- ⏳ 性能监控
- ⏳ 错误恢复
- ⏳ 分布式执行
- ⏳ A/B 测试

## 📚 参考资料

- [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/)
- [Checkpointer 指南](https://langchain-ai.github.io/langgraph/how-tos/persistence/)
- [Human-in-the-Loop](https://langchain-ai.github.io/langgraph/how-tos/human-in-the-loop/)
