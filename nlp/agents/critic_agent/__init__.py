"""
评论家智能体（问题6：审计任务衔接合理性）

名字里的 critic 按项目术语口径统一译作"评审"（不译"评论家"，更不是"批评家"）：
它审的是任务衔接，不是给输出打分。

⚠ 必须先知道的事实：本文件写的类当前完全没有参与运行。
enhanced_entry.py 的 _register_agents 里从头到尾没有注册 critic_agent，
所以即便 enhanced_graph 里建了 critic 节点、enable_critic 打开，
那节点从名册（registry）里也取不到实例，会直接判定"通过"放行。
读这个文件请当作"写好了但没接上电"的备件，别以为评审环节真的在工作。

它和 evaluator（评估器）的分工，是最容易混淆的一点：
- 评审：校验"上游这段输出能不能接上后续任务"（够不够下游用），是衔接问题。
- 评估器：给最终答案的质量打分（相关性/完整性/准确性），是质量问题。
两者都在 langgraph_orchestrator/ 下有对应的节点实现，可对照着读。
"""
from typing import Dict, Any
from core.protocol import AgentMessage, ValidationResult, TaskSchema
from llm.llm_client import LLMClient


class CriticAgent:
    """评审智能体 —— 校验任务衔接是否合理；注意它当前未被注册，不参与运行"""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.agent_id = "critic_agent"

    def validate_handover(self,
                         upstream_output: AgentMessage,
                         downstream_input_schema: TaskSchema,
                         downstream_description: str) -> ValidationResult:
        """校验上游输出是否满足下游需求

        AgentMessage（Agent 之间传的消息信封：内容 + 置信度 + 推理过程）、
        TaskSchema（任务结构契约：声明下游必需哪些字段）、
        ValidationResult（校验结论：是否通过 + 缺了哪些字段 + 改进建议）。

        三个入参合起来只回答一个问题：把 upstream_output 交给下游，字段够用吗？
        """

        prompt = f"""你是一个任务衔接审计员。请判断上游任务的输出是否能支撑下游任务的执行。

**下游任务需求**：
- 任务描述：{downstream_description}
- 必需字段：{', '.join(downstream_input_schema.required_fields)}
- 字段说明：{downstream_input_schema.field_descriptions}

**上游任务输出**：
- 智能体：{upstream_output.agent_id}
- 推理过程：{upstream_output.metadata.reasoning}
- 输出内容：{upstream_output.content}
- 置信度：{upstream_output.metadata.confidence}

请判断：
1. 上游输出是否包含下游所需的所有必需字段？
2. 字段的语义是否匹配？
3. 数据质量是否足够（基于置信度和推理过程）？

以JSON格式返回：
{{
    "is_valid": true/false,
    "missing_fields": ["字段1", "字段2"],
    "suggestions": ["建议1", "建议2"]
}}"""

        response = self.llm_client.chat([{"role": "user", "content": prompt}])

        # 让 LLM 直接吐 JSON 是脆的：它常会加 ```json 围栏或前后寒暄，
        # json.loads 一失败就走下面的降级分支。
        try:
            import json
            result = json.loads(response)
            return ValidationResult(
                is_valid=result.get("is_valid", False),
                missing_fields=result.get("missing_fields", []),
                suggestions=result.get("suggestions", [])
            )
        except:
            # 解析失败一律判"不通过"（fail-safe：拿不准就拦住，宁可误报不可放行）。
            # 事实性说明：这里是裸 except，会连 KeyboardInterrupt 这类异常一起吞掉，
            # 保持原样未改动。
            return ValidationResult(
                is_valid=False,
                suggestions=["审计过程出错，建议人工检查"]
            )
