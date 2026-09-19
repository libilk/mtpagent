# -*- coding: utf-8 -*-
"""
LangGraph 增强状态定义
======================

扩展原有状态，支持参数校验与重试、参数对齐、Critic 验证等高级功能

state（状态）= 全图共享的数据字典，各节点靠"读它 / 把自己的返回值合进去"传数据。

本文件是字段清单。读它的重点不是字段名，而是**每个字段谁写、谁读** ——
所以下面每个字段后面都标了写入方和读取方；字段本身的意思反而次要。

reducer（归并函数）= 声明同一字段被多个节点并发写入时怎么合并；
Annotated[类型, reducer] 就是把 reducer 挂到字段上的写法。
本文件用了两种：add_messages（追加消息）、operator.add（列表拼接）。

最容易被读错的一点：**挂了 reducer 的字段是"追加"，没挂的是"后来者覆盖前者"。**
同一次超步里多个节点写同一个没挂 reducer 的字段，只有最后一个的写生效。
"""

from __future__ import annotations

import operator
from typing import TypedDict, Optional, Annotated, List, Dict, Any

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class EnhancedGraphState(TypedDict):
    """LangGraph 增强状态（向后兼容原有 GraphState）"""

    # ========== 原有字段 ==========
    messages: Annotated[List[AnyMessage], add_messages]  # 写：入口初始化、各 Agent 节点追加；读：Agent 节点取历史对话。add_messages 保证"追加"而非覆盖
    query: str  # 当前要处理的问题；写：入口、router（DAG 中每个任务各改写一次）、upstream_retry；读：Agent 节点、complexity_classifier
    thread_id: Optional[str]  # 会话ID（用于记忆隔离，不同窗口/用户互不干扰）写：入口；读：memory_store 按它隔离记忆
    plan: Optional[Dict[str, Any]]  # 规划器产出的任务列表（tasks/依赖/schema）；简单路径不跑规划器，此处为 None。读：参数验证器、critic、波次调度
    is_complex: bool  # 复杂度判断结果（由 complexity_classifier 设置）读：route_by_complexity 决定走 router 还是 planner
    selected_agent: Optional[str]  # 写：router（简单路径）、upstream_retry（重试路径）；读：Agent 节点、人工介入判断
    agent_results: Annotated[List[Dict[str, Any]], operator.add]  # 每个 Agent 跑完追加一条 {agent, result}；读：aggregator 与下游三个校验节点（都只取 [-1] 最近一条）
    final_answer: str  # 最终答案；写：aggregator（多 Agent 汇总）、evaluator（重试后覆盖）；读：入口输出、人工介入展示
    quality_score: float  # evaluator（评估器：最终质量把关）打的分，0~1；读：check_quality 决定 pass/retry、人工介入通道 4a
    iteration: int  # 迭代轮次：evaluator 每判一次不合格 +1；读：check_quality（重试上限）、人工介入
    feedback: str  # evaluator 给出的改进建议；读：Agent 节点重试时拼进 prompt

    # ========== 参数校验与重试 ==========
    # 下面这组是"参数验证器 ↔ 上游重试"两个节点之间的私有通道，
    # 只在多 Agent DAG（有向无环图：有依赖顺序、且不能绕回自己的任务图）路径生效；
    # 简单路径 plan 为 None，参数验证器直接短路，这些字段不会被碰。
    validation_failed: bool  # 写：参数验证器；读：should_retry_validation 决定要不要 reexecute
    retry_target_task: Optional[str]  # 缺字段的那个上游 task_id（由参数验证器定位）；读：upstream_retry_node 拿它找任务对象
    validation_error_info: Optional[str]  # 缺什么字段/类型错在哪的可读描述；读：upstream_retry_node 拼进重试 prompt
    retry_reason: Optional[str]  # 为什么要重跑（给 Agent 看的自然语言理由）；读：upstream_retry_node 拼进重试 prompt
    parameter_retry_count: int  # 写：upstream_retry_node 每次重试 +1；读：should_retry_validation（上限 2）。与 iteration 是两套独立计数，别混
    validation_missing_sources: Optional[Dict[str, str]]  # {缺失字段: 来源task_id}；写：参数验证器；读：upstream_retry_node 精确定位该重跑谁
    validation_type_mismatches: List[str]                  # 类型不匹配列表；写：参数验证器；读：upstream_retry_node（拼进 prompt）

    # ========== 参数对齐 ==========
    parameters_aligned: bool  # 写：仅 enhanced_entry 初始化；本仓库无读取方 —— 字段目前没被用起来
    parameter_alignment_errors: List[Dict[str, Any]]  # 同上，仅初始化、无读取方

    # ========== Critic 验证 ==========
    # critic（评审）= 校验"这段输出能不能接上后续任务"，不是给答案打分（打分是 evaluator）
    critic_passed: bool  # 写：critic 节点；未注册 critic_agent 或校验抛异常时默认 True 放行；读：human_intervention_check 通道 4b
    critic_validation: Optional[Dict[str, Any]]  # critic 返回的原始结果；读：enhanced_entry 输出给前端
    critic_issues: List[Dict[str, Any]]  # critic 发现的任务衔接问题；读：human_intervention_check 展示给人工

    # ========== 重复检测 ==========
    previous_results: List[str]  # 历史结果指纹（用于精确匹配快速拦截）；写：重复检测节点（读出来→原地改→写回，没挂 reducer）；读：同节点下一轮
    previous_result_embeddings: List[List[float]]  # 历史结果向量（用于语义相似度检测）；读写同上；embedder 不可用时长度会短于 previous_results
    has_duplicate: bool                             # 是否检测到重复结果；写：重复检测节点；读：enhanced_graph._route_after_duplicate 决定是否熔断
    stop_reason: Optional[str]                      # 重复检测命中原因（duplicate_exact_match / duplicate_semantic）；写：重复检测节点；代码内无读取方，仅留档排查

    # ========== 人工介入 ==========
    human_intervention_required: bool  # 写：human_intervention_check；读：enhanced_graph 条件边决定是否走 execute（execute 前被 interrupt_before 执行前中断拦住）
    human_feedback: Optional[str]  # 人工答复（批准/拒绝/选项）；写：enhanced_entry 的 resume 分支；读：human_intervention_execute
    intervention_reason: Optional[str]              # 中断原因（展示给用户）；写：human_intervention_check
    intervention_data: Optional[Dict[str, Any]]     # 中断时展示给用户的详细数据，含 options 按钮；真人点选后由前端翻译回 human_feedback
    write_operations: Annotated[List[Dict[str, Any]], operator.add]  # 本轮已执行的写操作（见 core/write_ops.py）；写：Agent 节点；读：human_intervention_check 的写操作闸门
    write_approved: bool                            # 写操作是否已获人工批准；写：enhanced_entry 的 resume 分支；读：human_intervention_check

    # ========== 多模态（图片输入） ==========
    image_urls: Optional[List[str]]                          # 图片公网URL列表；写：入口；读：Agent 节点、vqa_agent
    image_paths: Optional[List[str]]                         # 本地图片路径列表；读：vqa_agent
    image_base64_list: Optional[List[str]]                   # base64编码图片列表；读：vqa_agent

    # ========== 文件上传（Excel/CSV等） ==========
    file_paths: Optional[List[str]]                          # 已上传文件路径列表；写：入口；读：Agent 节点（只取文件名提示）

    # ========== DAG 波次执行 ==========
    completed_task_ids: Annotated[List[str], operator.add]  # 已完成的 task_id；写：Agent 节点；读：波次调度（判断每个任务的依赖是否已全部就绪）
    current_task_id: Optional[str]                           # 当前执行的 task_id；写：router 扇出时随任务注入；读：Agent 节点（回写进上面的 completed_task_ids）
