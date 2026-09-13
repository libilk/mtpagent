"""HTTP 路由。这一层只负责:校验 → 入队 → 返回 / 读门面 → 返回。

**没有任何 LLM 调用,也没有直接 import 图或画像** —— 查询走 workflow 层的
只读门面(workflow/query.py)。POST /answer 的目标是 < 50ms 返回(§7 P1 验收线)。
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request, status

from coach import config
from coach.api.schemas import (
    AnswerAccepted,
    AnswerRequest,
    GapResponse,
    GraphResponse,
    HealthResponse,
    PlanResponse,
)
from coach.events import schema as events

router = APIRouter()


@router.post(
    "/answer",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=AnswerAccepted,
    summary="提交答题(异步处理)",
    description="校验并入队后立刻返回;判分、BKT 更新、根因检测由 worker 异步完成。",
)
def submit_answer(payload: AnswerRequest, request: Request) -> AnswerAccepted:
    bus = request.app.state.bus
    event = events.new_event(
        events.ANSWER_SUBMITTED,
        learner_id=payload.learner_id,
        payload={
            "problem_id": payload.problem_id,
            "answer_text": payload.answer_text,
            "elapsed_ms": payload.elapsed_ms,
        },
    )
    bus.publish(config.STREAM_ANSWER, event)
    return AnswerAccepted(event_id=event.event_id, trace_id=event.trace_id)


@router.get(
    "/gap/{learner_id}/{kp_id}",
    response_model=GapResponse,
    summary="★ 根因定位",
    description=(
        "沿知识图谱反向遍历前置依赖,找出学生真正欠缺的基础知识点。\n\n"
        "解释文本由 planner_worker 异步生成;若还没生成,返回模板文案并置 "
        "`explanation_pending=true`。"
    ),
)
def get_gap(
    learner_id: str,
    kp_id: str,
    request: Request,
    top_k: int = Query(default=config.ROOT_CAUSE_TOP_K, ge=1, le=10),
    depth: int = Query(default=config.MAX_PREREQ_DEPTH, ge=1, le=5),
) -> GapResponse:
    service = request.app.state.query_service
    try:
        return service.gap_view(learner_id, kp_id, top_k=top_k, depth=depth)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"知识点不存在:{kp_id}")


@router.get(
    "/graph/{kp_id}",
    response_model=GraphResponse,
    summary="前置依赖树",
    description="返回知识点的前置树。带 learner_id 时会标注哪些已掌握、哪些是缺口。",
)
def get_graph(
    kp_id: str,
    request: Request,
    depth: int = Query(default=config.MAX_PREREQ_DEPTH, ge=1, le=5),
    learner_id: Optional[str] = Query(default=None),
) -> GraphResponse:
    service = request.app.state.query_service
    try:
        return service.graph_view(kp_id, depth=depth, learner_id=learner_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"知识点不存在:{kp_id}")


@router.get(
    "/plan/{learner_id}",
    response_model=PlanResponse,
    summary="今日学习计划",
    description=(
        "合并三类待办,按优先级排序:\n\n"
        "1. `review` 到期复习(SM-2 算出的 due_at 已过)\n"
        "2. `remedial` 目标知识点的根因缺口\n"
        "3. `learn` 前置已备齐、可以推进的新知识点\n\n"
        "按 `daily_minutes` 预算裁剪。P3 用轮询刷新,不上 WebSocket。"
    ),
)
def get_plan(
    learner_id: str,
    request: Request,
    daily_minutes: Optional[int] = Query(default=None, ge=5, le=600),
) -> PlanResponse:
    service = request.app.state.query_service
    return service.plan_view(learner_id, daily_minutes=daily_minutes)


@router.get("/health", response_model=HealthResponse, summary="健康检查")
def health(request: Request) -> HealthResponse:
    bus = request.app.state.bus
    reachable = bool(bus.ping()) if hasattr(bus, "ping") else True
    return HealthResponse(status="ok" if reachable else "degraded", redis=reachable)
