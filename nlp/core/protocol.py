"""
结构化通信协议
解决问题3、6：消除智能体间通信歧义，实现强类型上下文传递

本文件同时是两个层次的落点，别混着读：
- AgentProtocol：Agent 接口契约（Harness 约束层），规定 Agent 必须实现哪些方法
- 后半部分的结构化消息模型（AgentMessage、TaskSchema 等）：Agent 之间传数据的信封与结构契约

**先解决一个看起来自相矛盾的说法**（README 说 Agent 是"鸭子类型、不要求继承基类"，
orchestrator/registry.py 注册时却对实例做 isinstance 校验）：

两者并不矛盾 —— 对 runtime_checkable（运行时可检查）的 Protocol 做 isinstance，
**本身就是鸭子类型检查**：它按"有没有 handle 这个属性"判断，不看继承关系，
所以不继承基类、只自己写了 handle 的普通类也能通过。
代价是它**只查方法名在不在，不查签名**（实测 handle(self) 这种错签名同样能注册成功）——
校验挡的是"忘了写 handle"，挡不住"handle 写错了"。

谁在用本文件：orchestrator/registry.py 的 register() 与 langgraph_orchestrator/nodes.py 的
make_agent_node() 取 AgentProtocol 做注册期校验；orchestrator/parameter_aligner.py 与
agents/critic_agent 取后半部分的结构契约做参数对齐 / 衔接校验。
"""
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Protocol, runtime_checkable
from datetime import datetime
from enum import Enum


# ---------------------------------------------------------------------------
# Agent 接口契约（Harness 约束层）
# ---------------------------------------------------------------------------

# runtime_checkable（运行时可检查）：给 Protocol 开一个口子，允许对它用 isinstance。
# 不加这个装饰器，isinstance(agent, AgentProtocol) 会直接抛 TypeError。
@runtime_checkable
class AgentProtocol(Protocol):
    """
    所有 Agent 必须遵循的接口契约。

    Harness（约束层，make_agent_node）通过此协议约束 Agent：
    - 注册时校验：AgentRegistry.register() 检查实例是否符合协议
    - 包装时校验：make_agent_node() 检查实例是否符合协议
    - 运行时调用：统一通过 handle(query, context) 调用 Agent

    实现要求：
        class MyAgent:
            def handle(self, query: str, context: Dict[str, Any]) -> str:
                ...

    注意：使用 Protocol + runtime_checkable，无需继承基类，
    只要实现了 handle 方法且签名兼容即可自动满足约束（鸭子类型）。

    事实补充：isinstance 只核对 handle 这个属性在不在，**不核对签名**，
    所以上句里的"签名兼容"是宽泛说法 —— 写成 handle(self)（一个参数都没有）也能通过校验。
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

        返回值被限定为 str，多一个字段都放不下 —— 所以"我刚改了数据"这类结构化信号
        无法从返回值带出，只能改用线程本地登记（详见 core/write_ops.py）。
        这条约束是那套登记机制的起因，读到这里可以顺带记一下。
        """
        ...


# 以下为结构化消息模型：跨 Agent 传递数据时的"信封 + 结构契约"。
# 它们不参与接口校验，是参数对齐器与 critic 校验时用的数据形状约定。


class MessageType(str, Enum):
    """消息类型（信封的类别标签：任务请求 / 任务结果 / 参数校验 / 错误）

    继承 str 是为了能直接当字符串比较和序列化（JSON 里存的就是 "task_result" 这种值）。
    """
    TASK_REQUEST = "task_request"
    TASK_RESULT = "task_result"
    PARAMETER_VALIDATION = "parameter_validation"
    ERROR = "error"


class MessageMetadata(BaseModel):
    """消息元数据（附在消息上的结构化附加信息，用于过滤、分层与排查）

    confidence 上的 Field(ge=0.0, le=1.0) 是 Pydantic 的区间约束：越界值在构造对象时
    就报错，不会等到下游用出问题才暴露。token_usage 是词元的消耗量，可缺省。
    """
    timestamp: datetime = Field(default_factory=datetime.now)
    reasoning: str = Field(description="推理过程")
    confidence: float = Field(ge=0.0, le=1.0, description="置信度")
    key_entities: List[str] = Field(default_factory=list, description="关键实体")
    token_usage: Optional[int] = None


class AgentMessage(BaseModel):
    """标准化的 Agent 间消息信封（结构化内容 + 置信度 + 推理过程 + 依赖的任务 ID）

    读它的人：orchestrator/parameter_aligner.py（按 dependencies 取上游结果做对齐）、
    agents/critic_agent（校验上游输出能不能接上下游任务）。
    Agent 之间不直接传裸 dict，都包成这个信封，字段才有统一口径。
    """
    task_id: str
    agent_id: str
    message_type: MessageType
    content: Dict[str, Any] = Field(description="结构化内容")
    metadata: MessageMetadata
    dependencies: List[str] = Field(default_factory=list, description="依赖的任务ID")

    def get_field(self, field_name: str, default: Any = None) -> Any:
        """安全获取字段（只查 content，不查 metadata）

        取不到时返回 default 而不是抛 KeyError —— 这条"有 get_field 就够用"的约定
        本身就是一种鸭子类型接口：只要对象带这个方法就能被当作信封使用，
        所以 langgraph_orchestrator/clarification.py 里能用轻量包装类顶替这个模型。
        """
        return self.content.get(field_name, default)


class TaskSchema(BaseModel):
    """任务的结构契约：声明这个任务必需/可选哪些字段、各是什么类型

    它只描述"形状"，不含真实数据。参数验证器拿它去比对上游实际输出够不够下游用；
    注意它只查字段在不在、类型对不对，不调 LLM。
    """
    required_fields: List[str] = Field(description="必需字段")
    optional_fields: List[str] = Field(default_factory=list)
    field_types: Dict[str, str] = Field(description="字段类型映射")
    field_descriptions: Dict[str, str] = Field(default_factory=dict)


class TaskWithSchema(BaseModel):
    """带结构契约的任务定义（问题7：参数预对齐）

    相较 EnhancedGraphState 里那个裸 dict 形式的 task，这个模型多了 input/output
    两个契约和 parameter_mapping，是参数对齐器唯一认的任务类型
    （构造处在 langgraph_orchestrator/clarification.py）。
    """
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
    """验证结果（参数是否够用、缺哪些字段、类型错在哪、怎么补）

    passed 比 is_valid 更严：is_valid 只表示"验证流程本身跑通了"，
    passed 还额外要求缺字段列表与类型不匹配列表都为空 —— 别把两者当同一个意思。
    """
    is_valid: bool
    missing_fields: List[str] = Field(default_factory=list)
    type_mismatches: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.is_valid and not self.missing_fields and not self.type_mismatches
