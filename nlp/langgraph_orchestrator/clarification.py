# -*- coding: utf-8 -*-
"""
参数校验与对齐节点（ParameterAligner 增强版）
=====================================================

职责：仅在多 Agent DAG 场景下，校验上游 Agent 输出是否满足下游任务需求。

校验能力（复用 ParameterAligner）：
- 动态填槽：按 parameter_mapping 从上游结果提取参数
- 字段存在性：检查 required_fields 是否齐全
- 类型匹配：检查字段类型是否符合 field_types 声明
- 来源追溯：定位缺失字段应该由哪个上游任务提供

单 Agent 场景的质量问题由 Evaluator 外环处理，不走参数校验。

---
本文件里的三件东西**不是一回事**，读的时候别混：

- parameter_validator_node（参数验证器）：只"查"。比对上游输出字段与下游声明，
  全程不调 LLM。schema（结构契约）= "这个节点的输出必须包含哪些字段"的清单。
  它是"在花 LLM 重跑之前先用便宜办法拦一道"的闸门。
- upstream_retry_node（上游重试）：只"改 prompt"。它自己不重跑 Agent，
  而是把缺失字段写进 query、指向目标 Agent，交由图去路由。
- should_retry_validation：条件边的判断函数。只回答"还能不能再试一次"，不干活。

DAG（有向无环图）= 多个 Agent 按依赖顺序串成的任务图，"无环"指不能绕回自己；
有 DAG 才有"上游/下游"可言，单 Agent 路径没有。

ParameterAligner（参数对齐器）= 按 parameter_mapping（参数映射：声明下游字段
从上游哪个字段取）从上游结果里提取、补齐下游要的入参，并提供 validate_input
（校验入参）与 identify_missing_source（定位缺失字段的来源）。
本文件不自己实现校验，都是复用它的能力。
"""

import logging
import time
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


def _emit(agent: str, msg: str, **extra):
    """往 StreamWriter（流式事件写入器）推一条进度事件，供前端展示"现在在干嘛"。

    整段包在 try/except 里且异常静默吞掉：进度是"有了更好、没有也能跑"的旁路，
    不能因为前端没连上就把主流程搞挂。只有图以流式方式运行时才取得到 writer。
    """
    try:
        from langgraph.config import get_stream_writer
        writer = get_stream_writer()
        if writer:
            event = {"event": "progress", "agent": agent, "msg": msg, "ts": time.time()}
            event.update(extra)
            writer(event)
    except Exception:
        pass


def _find_downstream_tasks(current_task_id: str, tasks: List[Dict]) -> List[Dict]:
    """查找依赖当前任务的所有下游任务

    判据是任务的 depends_on 里含不含 current_task_id —— 即"谁在等我跑完"。
    """
    return [
        t for t in tasks
        if current_task_id in t.get("depends_on", [])
    ]


def _build_mock_agent_message(result_content: Any, task_id: str, agent_id: str):
    """
    将 LangGraph state 中的 agent_results 条目包装为 ParameterAligner 可用的对象。

    ParameterAligner.fill_parameters() 期望 dependency_results 的 value
    有 get_field(field_name) 方法（即 AgentMessage 接口）。
    这里用轻量包装避免强依赖 protocol.AgentMessage 的完整构造。

    mock（模拟对象）= 用假的替身顶掉真实依赖。这里是为了满足接口形状，
    而非"测试替身"；少构造一层真实对象就少一处可能失败的地方。

    注意入参 task_id / agent_id 只用于签名对齐，实际没被用到 ——
    提取字段只认 content，包装对象里不保留这两个标识。
    """

    class _MockMsg:
        def __init__(self, content):
            self.content = content if isinstance(content, dict) else {"result": content}

        def get_field(self, field_name, default=None):
            return self.content.get(field_name, default)

    return _MockMsg(result_content)


def parameter_validator_node(state: Dict[str, Any]) -> dict:
    """
    增强版 Schema 校验：复用 ParameterAligner 检查上游输出是否满足下游需求。

    校验流程：
    1. 无 plan（简单路由）→ 直接跳过
    2. 找到当前 Agent 对应的 task 及其下游任务
    3. 对每个下游任务：
       a. fill_parameters()：按 parameter_mapping 从上游结果提取参数
       b. validate_input()：检查 required_fields + field_types
       c. identify_missing_source()：定位缺失字段的上游来源
    4. 有问题 → 触发参数重试回退

    返回的是"局部 state"：LangGraph 只把这里返回的键合进全局 state，
    没返回的键保持原样 —— 所以每个提前返回都显式带上 validation_failed，
    避免上一轮的失败标记残留下来误导条件边。

    只查最新的 agent_results[-1]（刚跑完的那次），不回头翻历史轮次；
    校验依据全部来自 plan 里的 schema 声明，因此没 plan 就无从查起。
    """

    _emit("parameter_validator", "正在校验参数...", stage="start")

    # 简单路由无 DAG，直接跳过 —— 这就是 enhanced_graph 文档里说的"短路"：
    # 单 Agent 场景下本节点整个不干活，质量交给 evaluator
    plan = state.get("plan")
    if not plan:
        _emit("parameter_validator", "无 DAG 规划，跳过校验", stage="skip")
        return {"validation_failed": False}

    tasks = plan.get("tasks", [])
    if not tasks:
        _emit("parameter_validator", "无任务列表，跳过校验", stage="skip")
        return {"validation_failed": False}

    agent_results = state.get("agent_results", [])
    if not agent_results:
        _emit("parameter_validator", "无 Agent 结果，跳过校验", stage="skip")
        return {"validation_failed": False}

    latest_result = agent_results[-1]
    current_agent = latest_result.get("agent")
    result_content = latest_result.get("result", {})

    # 找到当前 Agent 对应的 task
    current_task = next(
        (t for t in tasks if t.get("agent_id") == current_agent), None
    )
    if not current_task:
        return {"validation_failed": False}

    current_task_id = current_task["task_id"]

    # 找到依赖当前任务的下游任务
    downstream_tasks = _find_downstream_tasks(current_task_id, tasks)
    if not downstream_tasks:
        _emit("parameter_validator", f"任务 {current_task_id} 无下游依赖，跳过校验", stage="skip")
        return {"validation_failed": False}

    # 如果 Agent 返回纯文本（非 dict），无法做字段级校验，跳过
    if not isinstance(result_content, dict):
        _emit("parameter_validator", "Agent 返回纯文本，跳过字段校验", stage="skip")
        return {"validation_failed": False}

    _emit("parameter_validator", f"校验任务 {current_task_id} → 下游 {len(downstream_tasks)} 个任务", stage="checking")

    # ========== 尝试使用 ParameterAligner 做完整校验 ==========
    # 降级兜底（fallback）：拿不到 ParameterAligner 就退到 _validate_simple，
    # 宁可少查（只比字段名）也不让流程断掉。
    # 注意 except 只接 ImportError —— 校验过程中别的异常不会被这里兜住。
    try:
        from orchestrator.parameter_aligner import ParameterAligner
        from core.protocol import TaskWithSchema, TaskSchema
        result = _validate_with_aligner(
            state, current_task_id, result_content,
            downstream_tasks, agent_results
        )
    except ImportError:
        logger.debug("[ParameterValidation] ParameterAligner 不可用，降级为简单集合校验")
        result = _validate_simple(current_task_id, result_content, downstream_tasks)

    if result.get("validation_failed"):
        _emit("parameter_validator", f"校验失败: {result.get('validation_error_info', '')}", stage="failed")
    else:
        _emit("parameter_validator", "参数校验通过", stage="done")
    return result


def _validate_with_aligner(
    state: Dict[str, Any],
    current_task_id: str,
    result_content: dict,
    downstream_tasks: List[Dict],
    agent_results: List[Dict],
) -> dict:
    """使用 ParameterAligner 做完整校验（字段 + 类型 + 来源追溯）"""
    from orchestrator.parameter_aligner import ParameterAligner
    from core.protocol import TaskWithSchema, TaskSchema

    aligner = ParameterAligner()

    # 构建已完成结果的 mock 字典（task_id → MockMsg）
    # ParameterAligner 是按 task_id 查依赖结果的（一个 Agent 可能出现在多个任务里）。
    plan_tasks = state.get("plan", {}).get("tasks", [])
    agent_to_task = {t["agent_id"]: t["task_id"] for t in plan_tasks}

    dependency_results = {}
    for r in agent_results:
        agent_id = r.get("agent")
        # 优先用结果自带的 task_id（node_fn 写入，见 nodes.py）：
        # 按 agent 名反查会有塌陷 —— 同一个 Agent 出现在多个任务里时，
        # agent_to_task 只能保留其中一个 task_id，多条结果会挤到同一个键上互相覆盖。
        # 保留 agent_to_task 兜底，是为了兼容早期产生、没有 task_id 的结果。
        task_id = r.get("task_id") or agent_to_task.get(agent_id)
        if task_id:
            dependency_results[task_id] = _build_mock_agent_message(
                r.get("result", {}), task_id, agent_id
            )

    # 逐个校验下游任务；一旦某个下游不满足就立刻 return，
    # 即"一次只报第一个失败的下游" —— 避免多个下游同时缺字段时信息互相淹没
    for ds_task in downstream_tasks:
        input_schema_raw = ds_task.get("input_schema", {})
        output_schema_raw = ds_task.get("output_schema", {})
        param_mapping = ds_task.get("parameter_mapping", {})

        # 没有声明 input_schema 的下游任务，跳过
        # 没写 required_fields 视为"这个任务不挑输入"，不是错误，直接放行
        required_fields = input_schema_raw.get("required_fields", [])
        if not required_fields:
            continue

        # TaskWithSchema = 带结构契约的任务对象；aligner 的方法都只认这个类型，
        # 所以要先把 plan 里的原始 dict 转一道
        try:
            task_with_schema = TaskWithSchema(
                task_id=ds_task["task_id"],
                agent_id=ds_task["agent_id"],
                description=ds_task.get("description", ""),
                input_schema=TaskSchema(
                    required_fields=required_fields,
                    field_types=input_schema_raw.get("field_types", {}),
                ),
                output_schema=TaskSchema(
                    required_fields=output_schema_raw.get("required_fields", []),
                    field_types=output_schema_raw.get("field_types", {}),
                ),
                parameter_mapping=param_mapping,
                dependencies=ds_task.get("depends_on", []),
            )
        except Exception as e:
            logger.warning("[ParameterValidation] 构造 TaskWithSchema 失败: %s", e)
            continue

        # 1. 动态填槽（按 parameter_mapping 从上游结果里取字段，凑成这个任务的入参）
        filled_params = aligner.fill_parameters(task_with_schema, dependency_results)

        # 2. 完整验证（同时查字段齐不齐、类型对不对，结果装在 validation 对象里）
        validation = aligner.validate_input(task_with_schema, filled_params)

        if not validation.passed:
            # 3. 来源追溯
            missing_sources = aligner.identify_missing_source(
                task_with_schema, validation.missing_fields
            )

            missing_str = ", ".join(sorted(validation.missing_fields))
            type_mismatches = validation.type_mismatches

            info_parts = []
            if validation.missing_fields:
                info_parts.append(f"缺少字段: {missing_str}")
            if type_mismatches:
                info_parts.append(f"类型不匹配: {'; '.join(type_mismatches)}")

            # 确定回退目标：优先用 missing_sources 定位的上游，否则回退当前 task
            if missing_sources:
                # 取第一个缺失字段的来源作为回退目标
                # 注意只取第一个：这个字段有多个来源候选时，其余来源会被忽略
                target_task_id = list(missing_sources.values())[0]
            else:
                target_task_id = current_task_id

            logger.info(
                "[ParameterValidation] 下游任务 %s 校验失败 | %s | 回退目标: %s",
                ds_task["task_id"], "; ".join(info_parts), target_task_id
            )

            # 这几个键分两拨被消费：validation_failed 给条件边 should_retry_validation，
            # 其余四个给 upstream_retry_node 拼重试 prompt（见 enhanced_state.py 字段表）
            return {
                "validation_failed": True,
                "retry_target_task": target_task_id,
                "validation_error_info": "; ".join(info_parts),
                "retry_reason": (
                    f"下游任务 {ds_task['task_id']}"
                    f"（{ds_task.get('description', '')}）"
                    f"需要这些字段才能执行"
                ),
                "validation_missing_sources": missing_sources,
                "validation_type_mismatches": type_mismatches,
            }

    return {"validation_failed": False}


def _validate_simple(
    current_task_id: str,
    result_content: dict,
    downstream_tasks: List[Dict],
) -> dict:
    """降级方案：简单集合运算校验（ParameterAligner 不可用时）

    只比"字段名有没有"：下游 required_fields 减去上游实际输出的键名（集合差集）。
    比完整版少两样 —— 不查类型、不追来源，所以回退目标只能是当前 task 自己，
    没法像完整版那样精确指到缺字段的那个上游。
    """
    output_fields = set(result_content.keys())

    for ds_task in downstream_tasks:
        input_schema = ds_task.get("input_schema", {})
        required_fields = set(input_schema.get("required_fields", []))

        if not required_fields:
            continue

        missing = required_fields - output_fields

        if missing:
            missing_str = ", ".join(sorted(missing))
            logger.info(
                "[ParameterValidation] 任务 %s 输出缺少字段 [%s]，下游任务 %s 需要",
                current_task_id, missing_str, ds_task["task_id"]
            )

            return {
                "validation_failed": True,
                "retry_target_task": current_task_id,
                "validation_error_info": f"缺少字段: {missing_str}",
                "retry_reason": (
                    f"下游任务 {ds_task['task_id']}"
                    f"（{ds_task.get('description', '')}）"
                    f"需要这些字段才能执行"
                ),
                "validation_missing_sources": {},
                "validation_type_mismatches": [],
            }

    return {"validation_failed": False}


# ---------------------------------------------------------------------------
# upstream_retry + should_retry_validation
# ---------------------------------------------------------------------------

def upstream_retry_node(state: Dict[str, Any]) -> dict:
    """
    重新执行上游任务，补充缺失字段。

    增强版：利用 missing_sources 精确定位回退目标，
    在 prompt 中包含缺失字段名和类型要求。

    容易误解的一点：本节点**自己不调用 Agent、不重跑任何任务**。
    它只做两件事 —— 把"缺什么、为什么"拼进 query，把 selected_agent 指向目标
    Agent，然后靠图上的边把控制权交回那个 Agent 节点（改 prompt 而不是改流程）。
    副产品是给 parameter_retry_count +1，供 should_retry_validation 判上限。

    注意实际去向并不由这里的 selected_agent 决定，而是由 enhanced_graph.py 的
    _route_retry_target 决定；那里读的 validation_target 全项目无人写入，
    目前恒定回退到 agent_ids[0] —— 详见该文件的对应注释。
    """

    # 这五样都是参数验证器写下的（见 enhanced_state.py），本节点只读不写
    target_task = state.get("retry_target_task")
    missing_info = state.get("validation_error_info")
    reason = state.get("retry_reason")
    missing_sources = state.get("validation_missing_sources", {})
    type_mismatches = state.get("validation_type_mismatches", [])

    plan = state.get("plan", {})
    tasks = plan.get("tasks", [])
    target_task_obj = next((t for t in tasks if t["task_id"] == target_task), None)

    if not target_task_obj:
        logger.error(f"[ParameterValidation] 目标任务不存在: {target_task}")
        _emit("upstream_retry", f"目标任务不存在: {target_task}", stage="error")
        return {"validation_retry_failed": True}

    _emit("upstream_retry", f"重新执行上游任务: {target_task} | {missing_info}", stage="start")

    # 构建增强 prompt（提示词：喂给 LLM 的指令文本）
    # 原始任务描述 + 下游参数要求 + 缺失字段来源，合成一段话给重跑的 Agent 看
    enhancement_parts = [
        f"{target_task_obj['description']}",
        "",
        "【下游任务参数要求】",
        f"下游任务需要补充以下信息：{missing_info}",
        f"原因：{reason}",
    ]

    if missing_sources:
        fields_detail = ", ".join(
            f"{field}（来源: {src}）" for field, src in missing_sources.items()
        )
        enhancement_parts.append(f"缺失字段详情：{fields_detail}")

    if type_mismatches:
        enhancement_parts.append(f"类型问题：{'; '.join(type_mismatches)}")

    enhancement_parts.append("请在输出中明确包含这些信息，并确保字段类型正确。")

    enhanced_desc = "\n".join(enhancement_parts)

    logger.info(f"[ParameterValidation] 重新执行任务: {target_task}")
    _emit("upstream_retry", f"已构建增强 prompt，路由至 Agent: {target_task_obj['agent_id']}", stage="done")

    # query 是"这次要让 Agent 干什么"，所以增强描述走 query 而不是 description：
    # Agent 节点读的就是 query，重跑时会拿到这段带缺失字段要求的文本
    return {
        "query": enhanced_desc,
        "selected_agent": target_task_obj["agent_id"],
        "parameter_retry_count": state.get("parameter_retry_count", 0) + 1
    }


def should_retry_validation(state: Dict[str, Any]) -> str:
    """
    判断是否应该重试参数校验。

    限制最多重试 2 次，防止无限循环。

    这是条件边（conditional edge）：返回值不是给人看的，而是拿去 enhanced_graph.py
    的映射表 {"reexecute": "upstream_retry", "continue": "duplicate_detection"} 里
    查下一个节点 —— 字符串拼错就等于边断了，改这里必须同步改那边。

    计数只在真正重试时由 upstream_retry_node 递增；到了上限就返回 "continue"
    放行，让流程往下走（而不是继续卡在重试上）。
    """

    validation_failed = state.get("validation_failed", False)
    retry_count = state.get("parameter_retry_count", 0)

    if validation_failed and retry_count < 2:
        return "reexecute"

    return "continue"

