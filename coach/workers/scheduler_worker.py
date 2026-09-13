"""scheduler_worker:定时扫描到期复习,生成/广播今日计划(work.md P3.2 / P3.4)。

消费 `tick.scheduled`,做两件事:
    1. 扫 `learner_memory.due_at <= now`,算出今日计划
    2. 发布 `plan.updated` —— 客户端**轮询** `GET /plan` 或订阅该事件
       (P3 明确不上 WebSocket)

注意:本 worker 独占 `coach:tick` 流。Redis 消费组里一条消息只会投给组内
**一个**消费者,所以两个 worker 抢同一个流会随机丢消息 —— 一个流只配一个 worker。
"""

import asyncio
import logging
from typing import Any, Dict, Optional

from coach import config
from coach.events import schema as events
from coach.events.schema import Event
from coach.workflow.query import QueryService

logger = logging.getLogger(__name__)


class SchedulerWorker:
    handled_types = (events.TICK_SCHEDULED,)

    def __init__(
        self,
        knowledge,
        profile,
        journal,
        bus=None,
        service: Optional[QueryService] = None,
        stream: str = config.STREAM_TICK,
        group: str = config.GROUP_COACH,
        consumer: str = "scheduler-1",
    ):
        self.knowledge = knowledge
        self.profile = profile
        self.journal = journal
        self.bus = bus
        self.service = service or QueryService(knowledge, profile)
        self.stream = stream
        self.group = group
        self.consumer = consumer
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
        if event.type != events.TICK_SCHEDULED:
            logger.warning("scheduler_worker 不处理的事件类型:%s", event.type)
            return {"status": "ignored", "type": event.type}
        return self.handle_tick(event)

    def handle_tick(self, event: Event) -> Dict[str, Any]:
        learner_id = event.learner_id
        plan = self.service.plan_view(learner_id)
        due = [i["kp_id"] for i in plan["items"] if i["action"] == "review"]

        self._publish_plan_updated(learner_id, plan, event.trace_id)
        logger.info(
            "生成计划 learner=%s 项数=%d 到期复习=%d",
            learner_id,
            len(plan["items"]),
            len(due),
        )
        return {
            "status": "ok",
            "items": len(plan["items"]),
            "due": due,
            "planned_minutes": plan["planned_minutes"],
        }

    def _publish_plan_updated(self, learner_id: str, plan: Dict, trace_id: str) -> None:
        if self.bus is None:
            return
        update = events.new_event(
            events.PLAN_UPDATED,
            learner_id=learner_id,
            payload={"date": plan["date"], "items": plan["items"]},
            trace_id=trace_id,
        )
        try:
            self.bus.publish(config.STREAM_PLAN, update)
        except Exception:  # noqa: BLE001 — 推送失败不该影响计划生成
            logger.exception("发布 plan.updated 失败")

    # ---------------- 消费循环 ----------------

    async def run(self, block_ms: int = 1000, count: int = 10) -> None:
        logger.info("scheduler_worker 启动:stream=%s group=%s", self.stream, self.group)
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
