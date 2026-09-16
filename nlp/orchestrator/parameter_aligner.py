"""
参数对齐器（问题7：入参拆解与动态填槽）
"""
from typing import Dict, Any, List
from core.protocol import TaskWithSchema, AgentMessage, ValidationResult, TaskSchema


class ParameterAligner:
    """参数预对齐与动态填槽"""

    def validate_input(self, task: TaskWithSchema, input_data: Dict[str, Any]) -> ValidationResult:
        """验证输入是否满足Schema"""
        missing = [f for f in task.input_schema.required_fields if f not in input_data]
        type_mismatches = []

        for field, expected_type in task.input_schema.field_types.items():
            if field in input_data:
                actual_type = type(input_data[field]).__name__
                if actual_type != expected_type and expected_type != "Any":
                    type_mismatches.append(f"{field}: expected {expected_type}, got {actual_type}")

        suggestions = []
        if missing:
            suggestions.append(f"缺少必需字段: {', '.join(missing)}")

        return ValidationResult(
            is_valid=len(missing) == 0,
            missing_fields=missing,
            type_mismatches=type_mismatches,
            suggestions=suggestions
        )

    def fill_parameters(self,
                       task: TaskWithSchema,
                       dependency_results: Dict[str, AgentMessage]) -> Dict[str, Any]:
        """动态填槽：从依赖任务结果中提取参数"""
        filled_params = {}

        for target_field, source_path in task.parameter_mapping.items():
            # 解析路径：task_id.field_name
            if "." in source_path:
                dep_task_id, field_name = source_path.split(".", 1)
                if dep_task_id in dependency_results:
                    value = dependency_results[dep_task_id].get_field(field_name)
                    if value is not None:
                        filled_params[target_field] = value

        return filled_params

    def validate_output(self, task: TaskWithSchema, output: AgentMessage) -> ValidationResult:
        """验证输出是否满足承诺的Schema"""
        return self.validate_input(
            TaskWithSchema(
                task_id=task.task_id,
                agent_id=task.agent_id,
                description="",
                input_schema=task.output_schema,
                output_schema=TaskSchema(required_fields=[], field_types={})
            ),
            output.content
        )

    def identify_missing_source(self, task: TaskWithSchema, missing_fields: List[str]) -> Dict[str, str]:
        """识别缺失字段应该从哪个上游任务获取"""
        missing_sources = {}

        for field in missing_fields:
            # 从 parameter_mapping 中查找该字段的来源
            if field in task.parameter_mapping:
                source_path = task.parameter_mapping[field]
                if "." in source_path:
                    source_task_id = source_path.split(".", 1)[0]
                    missing_sources[field] = source_task_id

        return missing_sources
