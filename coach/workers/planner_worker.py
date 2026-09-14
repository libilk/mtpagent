"""planner_worker:根因定位 + 把根因说成人话(work.md P2.2 / P2.4)。

为什么解释要放在 worker 而不是 GET /gap:
- 生成解释要调 LLM,几百毫秒到几秒;
- 入口层的验收线是 < 50ms,且 P1 已立下「api 不碰 LLM」的规矩。

所以流程是:答题 → profile.updated → 本 worker 算根因 + 调 LLM 写解释 →
落 `gap_explanations` 缓存 → `GET /gap` 直接读缓存。
第一次查询时缓存可能还没生成,这时返回模板文案并置 `explanation_pending=true`。
"""

import asyncio
import logging
from typing import Any, Dict, Optional

from coach import config
from coach.events import schema as events
from coach.events.schema import Event
from coach.workflow.agent_diagnosis import refresh_agent_diagnosis
from coach.workflow.pipeline import refresh_explanation
from coach.workflow.query import QueryService

logger = logging.getLogger(__name__)


class PlannerWorker:
    #: 本 worker 认领的事件类型(recovery 按此分流)。
    #: 只认 PROFILE_UPDATED —— TICK_SCHEDULED 归 scheduler_worker。
    #: 原因:同一个消费组里,一条消息只会投给其中一个消费者,
    #: 两个 worker 抢同一个流会随机丢消息。所以**一个流只配一个 worker**。
    handled_types = (events.PROFILE_UPDATED,)

    def __init__(
        self,
        knowledge,
        profile,
        journal,
        llm=None,
        bus=None,
        service: Optional[QueryService] = None,
        stream: str = config.STREAM_PROFILE,
        group: str = config.GROUP_COACH,
        consumer: str = "planner-1",
        agent_diagnosis: bool = False,
    ):
        self.knowledge = knowledge
        self.profile = profile
        self.journal = journal
        self.llm = llm
        self.bus = bus
        self.service = service or QueryService(knowledge, profile)
        self.stream = stream
        self.group = group
        self.consumer = consumer
        # ★ 默认关。一次 agent 诊断要好几轮 LLM 调用(贵且慢),
        #   挂在每条 profile.updated 上会失控 —— 要开必须显式说。
        self.agent_diagnosis = agent_diagnosis
        self._stop = asyncio.Event()

    # ---------------- 业务 ----------------

    def process(self, event: Event) -> Dict[str, Any]:
        if self.profile.is_processed(event.event_id):
            return {"status": "skipped", "reason": "already_processed"}

        self.journal.record(event)
        result = self.handle(event)
        self.profile.mark_processed(event.event_id)
        self.journal.mark_done(event.event_id)
        return result

    def handle(self, event: Event) -> Dict[str, Any]:
        if event.type == events.PROFILE_UPDATED:
            return self.handle_profile_updated(event)
        if event.type == events.TICK_SCHEDULED:
            return self.handle_tick(event)
        logger.warning("planner_worker 不处理的事件类型:%s", event.type)
        return {"status": "ignored", "type": event.type}

    def handle_profile_updated(self, event: Event) -> Dict[str, Any]:
        kp_ids = list(event.payload.get("kp_ids") or [])
        # since = 原始答题时间。答题链(P4 的 pipeline)若已经写过解释,
        # 这里的缓存就是新的,直接跳过 —— 否则一次答题会调两遍 LLM。
        since = event.payload.get("since")
        refreshed = [
            kp for kp in kp_ids if self.refresh_explanation(event.learner_id, kp, not_before=since)
        ]
        result: Dict[str, Any] = {"status": "ok", "refreshed": refreshed, "checked": kp_ids}
        if self.agent_diagnosis:
            result["agent_diagnosed"] = self.refresh_agent(event.learner_id, not_before=since)
        return result

    def refresh_agent(self, learner_id: str, not_before: Optional[float] = None) -> Optional[str]:
        """对**学习者的目标知识点**跑一次 agent 诊断,写进缓存。

        只跑目标这一个:一次诊断要好几轮 LLM 调用,对 payload 里每个 kp 都跑会失控。
        没配目标就什么都不做。
        """
        goal = (self.profile.get_profile(learner_id) or {}).get("goal_kp_id")
        if not goal:
            return None
        wrote = refresh_agent_diagnosis(
            self.knowledge, self.profile, self.llm, learner_id, goal, not_before=not_before
        )
        return goal if wrote else None

    def handle_tick(self, event: Event) -> Dict[str, Any]:
        """定时巡检:对学习者目标知识点重算根因(P3 会在此基础上出计划)。"""
        kp_ids = list(event.payload.get("kp_ids") or [])
        if not kp_ids:
            profile = self.profile.get_profile(event.learner_id)
            goal = (profile or {}).get("goal_kp_id")
            kp_ids = [goal] if goal else []
        refreshed = [kp for kp in kp_ids if self.refresh_explanation(event.learner_id, kp)]
        return {"status": "ok", "refreshed": refreshed}

    def refresh_explanation(
        self, learner_id: str, kp_id: str, not_before: Optional[float] = None
    ) -> bool:
        """重算根因并(有缺口且配了 LLM 时)写入人话解释。返回是否写了缓存。

        not_before:答题链已经写过一次解释时,pipeline 会把事件时间戳带过来,
        这里看到缓存够新就直接跳过 —— 否则同一次答题会调两遍 LLM。
        """
        return refresh_explanation(
            self.service, self.profile, self.llm, learner_id, kp_id, not_before=not_before
        )

    # ---------------- 消费循环 ----------------

    async def run(self, block_ms: int = 1000, count: int = 10) -> None:
        logger.info("planner_worker 启动:stream=%s group=%s", self.stream, self.group)
        while not self._stop.is_set():
            messages = await asyncio.to_thread(
                self.bus.consume, self.stream, self.group, self.consumer, count, block_ms
            )
            for msg_id, event in messages:
                await asyncio.to_thread(self._process_with_retry, msg_id, event)

    def stop(self) -> None:
        self._stop.set()

    def _process_with_retry(self, msg_id: str, event: Event) -> None:
        try:
            self.process(event)
        except Exception as exc:  # noqa: BLE001
            logger.exception("处理失败:%s", event.event_id)
            attempt = event.attempt + 1
            if attempt > config.MAX_ATTEMPTS:
                self.bus.move_to_dlq(self.stream, event, f"{type(exc).__name__}: {exc}")
                self.journal.mark_done(event.event_id)
            else:
                self.bus.publish(self.stream, event.with_attempt(attempt))
        finally:
            self.bus.ack(self.stream, self.group, msg_id)
