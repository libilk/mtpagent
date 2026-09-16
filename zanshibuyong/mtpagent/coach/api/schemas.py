"""API 请求/响应模型(work.md §5.3 契约)。"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AnswerRequest(BaseModel):
    learner_id: str = Field(min_length=1, description="学习者 id")
    problem_id: str = Field(min_length=1, description="题目 id")
    answer_text: str = Field(default="", description="学生提交的答案")
    elapsed_ms: int = Field(default=0, ge=0, description="作答耗时(毫秒)")


class AnswerAccepted(BaseModel):
    accepted: bool = True
    event_id: str = Field(description="ULID,幂等键")
    trace_id: str = Field(description="同一条链路的多个事件共享")


class HealthResponse(BaseModel):
    status: str
    redis: bool


class RootCause(BaseModel):
    kp_id: str
    name: str
    depth: int = Field(description="距目标知识点的前置深度,越小越基础")
    mastery: float
    status: str = Field(
        default="gap",
        description="gap=有作答证据、确实薄弱;unobserved=从没测过(低掌握度只是初始值)",
    )
    path: List[str] = Field(default_factory=list, description="从目标走回根因的依赖路径")


class GapResponse(BaseModel):
    learner_id: str
    kp_id: str
    name: str
    mastery: float = Field(description="目标知识点自身的掌握度")
    root_causes: List[RootCause] = Field(default_factory=list)
    explanation: str
    explanation_source: str = Field(
        default="template", description="llm=planner_worker 已生成;template=模板兜底"
    )
    explanation_pending: bool = Field(
        default=False, description="有根因但人话解释还没生成完(P2.4 是异步的)"
    )


class ProfileResponse(BaseModel):
    learner_id: str
    goal: Optional[Dict[str, Any]] = Field(
        default=None, description="目标知识点及其掌握度;没建档或没设目标时为 null"
    )
    daily_minutes: Optional[int] = None
    mastery_summary: Dict[str, Any] = Field(
        default_factory=dict, description="观测数 / 已掌握数 / 均值 / 最低的几个"
    )
    due_now: List[Dict[str, Any]] = Field(default_factory=list, description="当前到期复习项")
    error_patterns: Dict[str, Any] = Field(
        default_factory=dict,
        description="top=错误计数排行;recurring=跨多个知识点重复出现(疑似系统性误解)",
    )


class PlanItem(BaseModel):
    kp_id: str
    name: str
    action: str = Field(
        description="review=到期复习 / remedial=补根因 / probe=前置没测过先摸底 / learn=推进新知识点"
    )
    reason: str
    est_minutes: float


class PlanResponse(BaseModel):
    learner_id: str
    date: str
    goal_kp_id: Optional[str] = None
    daily_minutes: int
    planned_minutes: float
    items: List[PlanItem] = Field(default_factory=list)


class GraphNode(BaseModel):
    kp_id: str
    name: str
    depth: int
    mastery: Optional[float] = None
    is_gap: bool = False
    children: List["GraphNode"] = Field(default_factory=list)


class GraphResponse(BaseModel):
    node: Dict[str, Any]
    prereq_tree: GraphNode
    mastered: List[Dict[str, Any]] = Field(default_factory=list)
    gaps: List[Dict[str, Any]] = Field(default_factory=list)
    learner_id: Optional[str] = None


GraphNode.model_rebuild()
