"""
参数对齐器（问题7：入参拆解与动态填槽）
======================================

ParameterAligner（参数对齐器）= 按 parameter_mapping（参数映射：声明"本任务的某个
入参该从哪个上游任务的哪个字段取"）从上游结果里抽值、填成下游能直接用的入参。

**参数验证真正的实现体在这个文件里，不在编排层。**
langgraph_orchestrator/clarification.py 的 parameter_validator_node 只负责决定
"什么时候查、查出来怎么处置"；字段级的提取与比对，全部是调用本类的四个方法完成。
读"字段为什么被判缺失"这类问题，规则看这里，别看编排层。

调用方按这条链把方法串起来：fill_parameters（从上游抽值填进来）→ validate_input
（比"字段齐不齐、类型对不对"）→ identify_missing_source（缺的字段本该谁给）。
validate_output 不在这条链上，是独立的一条，用于检查自己的产出。
本类无状态（不存任何字段），每次调用现算。

schema（结构契约）= 一份"必须包含哪些字段"的清单，本文件消费它的 required_fields
（必需字段）与 field_types（字段类型表）。
"""
from typing import Dict, Any, List
from core.protocol import TaskWithSchema, AgentMessage, ValidationResult, TaskSchema


class ParameterAligner:
    """参数预对齐与动态填槽

    注意 fill_parameters 只"填能填的"，填不满不算错 —— 缺口留给 validate_input 报、
    再由 identify_missing_source 溯源。所以"填槽"和"校验"必须成对读。
    """

    def validate_input(self, task: TaskWithSchema, input_data: Dict[str, Any]) -> ValidationResult:
        """验证输入是否满足Schema

        只查两样：required_fields 缺不缺、field_types 里声明的类型名对不对。
        类型比对是拿 type(x).__name__ 跟 schema 里写的字符串（如 "str"/"int"）比，
        不是真做类型检查；声明成 "Any" 的字段跳过不比。

        两个"通过"口径不一致，容易看漏：本方法返回的 is_valid 只看 missing 是否为空，
        而 ValidationResult.passed 还额外要求 type_mismatches 也为空 ——
        调用方用的是哪个，决定"类型不对"算不算失败。
        """
        # required_fields（必需字段）：本任务的入参清单，由规划器写进 plan，不是运行时猜的
        missing = [f for f in task.input_schema.required_fields if f not in input_data]
        type_mismatches = []

        # expected_type 是 schema 里的类型名字符串，不是 Python 类型对象 —— 别指望它能识别子类
        for field, expected_type in task.input_schema.field_types.items():
            if field in input_data:
                actual_type = type(input_data[field]).__name__
                if actual_type != expected_type and expected_type != "Any":
                    type_mismatches.append(f"{field}: expected {expected_type}, got {actual_type}")

        # suggestions 只是给人看的可读描述，不参与 is_valid 判定
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
        """动态填槽：从依赖任务结果中提取参数

        这就是"动态填槽"在干的事：不靠人手摆参数，而是照 parameter_mapping 声明的
        "目标字段 ← 上游task_id.字段"逐条从上游结果里抽值，拼成本任务的入参。

        parameter_mapping 是下游任务自己声明的（写在 plan 里），所以"谁该提供这个字段"
        在规划阶段就定死了 —— 抽不到时才能顺着它反查出该重跑谁（见 identify_missing_source）。
        """
        filled_params = {}

        # 只处理形如 "task_id.field" 的路径；mapping 里没带点的条目会被静默跳过
        for target_field, source_path in task.parameter_mapping.items():
            # 解析路径：task_id.field_name
            if "." in source_path:
                dep_task_id, field_name = source_path.split(".", 1)
                if dep_task_id in dependency_results:
                    value = dependency_results[dep_task_id].get_field(field_name)
                    # 上游给了 None 等同于没给：只填真拿到值的字段，缺口交给校验环节报出来
                    if value is not None:
                        filled_params[target_field] = value

        return filled_params

    def validate_output(self, task: TaskWithSchema, output: AgentMessage) -> ValidationResult:
        """验证输出是否满足承诺的Schema

        技巧：把 task.output_schema 借位塞进 input_schema 的位置再复用 validate_input ——
        "输出该有哪些字段"和"输入该有哪些字段"结构一样，没必要写两遍比对逻辑。
        本方法当前没有调用方（编排层只查输入侧），留着是接口完整性。
        """
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
        """识别缺失字段应该从哪个上游任务获取

        答的是"这个字段本该谁给"（溯源），不是"现在谁没给"。依据仍是 parameter_mapping，
        只是反过来查：字段 → 声明的上游 task_id。

        反查得到的映射是"该重跑谁"的唯一线索：没登记在 mapping 里的字段查不到来源，
        结果 dict 里就不会有它 —— 调用方（clarification.py）据此退化成"重跑当前任务"。
        """
        missing_sources = {}

        for field in missing_fields:
            # 从 parameter_mapping 中查找该字段的来源
            if field in task.parameter_mapping:
                source_path = task.parameter_mapping[field]
                if "." in source_path:
                    source_task_id = source_path.split(".", 1)[0]
                    missing_sources[field] = source_task_id

        return missing_sources
