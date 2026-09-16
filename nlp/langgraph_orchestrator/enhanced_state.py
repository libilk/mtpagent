# -*- coding: utf-8 -*-
"""
LangGraph 增强状态定义
======================

扩展原有状态，支持参数校验与重试、参数对齐、Critic 验证等高级功能
"""

from __future__ import annotations

import operator
from typing import TypedDict, Optional, Annotated, List, Dict, Any

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class EnhancedGraphState(TypedDict):
    """LangGraph 增强状态（向后兼容原有 GraphState）"""

    # ========== 原有字段 ==========
    messages: Annotated[List[AnyMessage], add_messages]
    query: str
    thread_id: Optional[str]  # 会话ID（用于记忆隔离，不同窗口/用户互不干扰）
    plan: Optional[Dict[str, Any]]
    is_complex: bool  # 复杂度判断结果（由 complexity_classifier 设置）
    selected_agent: Optional[str]
    agent_results: Annotated[List[Dict[str, Any]], operator.add]
    final_answer: str
    quality_score: float
    iteration: int
    feedback: str

    # ========== 参数校验与重试 ==========
    validation_failed: bool
    retry_target_task: Optional[str]
    validation_error_info: Optional[str]
    retry_reason: Optional[str]
    parameter_retry_count: int
    validation_missing_sources: Optional[Dict[str, str]]  # {缺失字段: 来源task_id}
    validation_type_mismatches: List[str]                  # 类型不匹配列表

    # ========== 参数对齐 ==========
    parameters_aligned: bool
    parameter_alignment_errors: List[Dict[str, Any]]

    # ========== Critic 验证 ==========
    critic_passed: bool
    critic_validation: Optional[Dict[str, Any]]
    critic_issues: List[Dict[str, Any]]

    # ========== 重复检测 ==========
    previous_results: List[str]  # 历史结果指纹（用于精确匹配快速拦截）
    previous_result_embeddings: List[List[float]]  # 历史结果向量（用于语义相似度检测）
    has_duplicate: bool                             # 是否检测到重复结果
    stop_reason: Optional[str]                      # 重复检测命中原因（duplicate_exact_match / duplicate_semantic）

    # ========== 人工介入 ==========
    human_intervention_required: bool
    human_feedback: Optional[str]
    intervention_reason: Optional[str]              # 中断原因（展示给用户）
    intervention_data: Optional[Dict[str, Any]]     # 中断时展示给用户的详细数据

    # ========== 多模态（图片输入） ==========
    image_urls: Optional[List[str]]                          # 图片公网URL列表
    image_paths: Optional[List[str]]                         # 本地图片路径列表
    image_base64_list: Optional[List[str]]                   # base64编码图片列表

    # ========== 文件上传（Excel/CSV等） ==========
    file_paths: Optional[List[str]]                          # 已上传文件路径列表

    # ========== DAG 波次执行 ==========
    completed_task_ids: Annotated[List[str], operator.add]  # 已完成的 task_id
    current_task_id: Optional[str]                           # 当前执行的 task_id
