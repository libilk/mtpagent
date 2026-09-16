"""
结构化通信协议
解决问题3、6：消除智能体间通信歧义，实现强类型上下文传递

包含：
- AgentProtocol: Agent 接口契约（Harness 约束层）
- 结构化消息模型（AgentMessage、TaskSchema 等）
"""
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Protocol, runtime_checkable
from datetime import datetime
from enum import Enum


# ---------------------------------------------------------------------------
# Agent 接口契约（Harness 约束层）
# ---------------------------------------------------------------------------

@runtime_checkable
class AgentProtocol(Protocol):
    """
    所有 Agent 必须遵循的接口契约。

    Harness（make_agent_node）通过此协议约束 Agent：
    - 注册时校验：AgentRegistry.register() 检查实例是否符合协议
    - 包装时校验：make_agent_node() 检查实例是否符合协议
    - 运行时调用：统一通过 handle(query, context) 调用 Agent

    实现要求：
        class MyAgent:
            def handle(self, query: str, context: Dict[str, Any]) -> str:
                ...

    注意：使用 Protocol + runtime_checkable，无需继承基类，
    只要实现了 handle 方法且签名兼容即可自动满足约束（鸭子类型）。
    """

    def handle(self, query: str, context: Dict[str, Any]) -> str:
        """
        处理用户查询（Agent 统一入口）

        Args:
            query: 用户问题（已经过 Harness 指代消解和反馈注入）
            context: 上下文信息，包含：
                - messages: 对话消息列表
                - dependencies: 上游 Agent 结果（DAG 场景）
                - original_query: 原始用户查询
                - history: 对话历史（来自共享记忆）
                - image_urls/image_paths/image_base64_list: 图片数据（多模态场景）

        Returns:
            回答文本
        """
        ...


class MessageType(str, Enum):
    """消息类型"""
    TASK_REQUEST = "task_request"
    TASK_RESULT = "task_result"
    PARAMETER_VALIDATION = "parameter_validation"
    ERROR = "error"


class MessageMetadata(BaseModel):
    """消息元数据"""
    timestamp: datetime = Field(default_factory=datetime.now)
    reasoning: str = Field(description="推理过程")
    confidence: float = Field(ge=0.0, le=1.0, description="置信度")
    key_entities: List[str] = Field(default_factory=list, description="关键实体")
    token_usage: Optional[int] = None


class AgentMessage(BaseModel):
    """标准化智能体消息"""
    task_id: str
    agent_id: str
    message_type: MessageType
    content: Dict[str, Any] = Field(description="结构化内容")
    metadata: MessageMetadata
    dependencies: List[str] = Field(default_factory=list, description="依赖的任务ID")

    def get_field(self, field_name: str, default: Any = None) -> Any:
        """安全获取字段"""
        return self.content.get(field_name, default)


class TaskSchema(BaseModel):
    """任务输入输出Schema"""
    required_fields: List[str] = Field(description="必需字段")
    optional_fields: List[str] = Field(default_factory=list)
    field_types: Dict[str, str] = Field(description="字段类型映射")
    field_descriptions: Dict[str, str] = Field(default_factory=dict)


class TaskWithSchema(BaseModel):
    """带Schema的任务定义（问题7：参数预对齐）"""
    task_id: str
    agent_id: str
    description: str
    input_schema: TaskSchema
    output_schema: TaskSchema
    parameter_mapping: Dict[str, str] = Field(
        default_factory=dict,
        description="从依赖任务到本任务的字段映射，格式：{本任务字段: 依赖任务.字段}"
    )
    dependencies: List[str] = Field(default_factory=list)


class ValidationResult(BaseModel):
    """验证结果"""
    is_valid: bool
    missing_fields: List[str] = Field(default_factory=list)
    type_mismatches: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.is_valid and not self.missing_fields and not self.type_mismatches
