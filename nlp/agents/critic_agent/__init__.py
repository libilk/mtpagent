"""
评论家智能体（问题6：审计任务衔接合理性）
"""
from typing import Dict, Any
from core.protocol import AgentMessage, ValidationResult, TaskSchema
from llm.llm_client import LLMClient


class CriticAgent:
    """审计智能体，验证任务衔接的合理性"""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.agent_id = "critic_agent"

    def validate_handover(self,
                         upstream_output: AgentMessage,
                         downstream_input_schema: TaskSchema,
                         downstream_description: str) -> ValidationResult:
        """验证上游输出是否满足下游需求"""

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

        try:
            import json
            result = json.loads(response)
            return ValidationResult(
                is_valid=result.get("is_valid", False),
                missing_fields=result.get("missing_fields", []),
                suggestions=result.get("suggestions", [])
            )
        except:
            return ValidationResult(
                is_valid=False,
                suggestions=["审计过程出错，建议人工检查"]
            )
