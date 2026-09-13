"""profile_worker:判分 → BKT → 画像落库(work.md §2.2 / §5.6)。

消费契约(§5.6)在这里落地:
    1. 从流里取消息
    2. 查 processed_events,已处理过直接 ACK 跳过
    3. 写 journal —— 先记后做
    4. 执行业务
    5. 写 processed_events + journal.mark_done + XACK
    6. 失败 → attempt+1 重投;超过上限进死信
    7. 重启 → recovery 扫 journal 补做

**幂等做了两道闸**:
- processed_events 挡「同一消息被消费两次」
- answers.answer_id 唯一键挡「恢复重放导致画像被更新两次」
"""

import asyncio
import logging
from typing import Any, Dict, Optional

from coach import config
from coach.events import schema as events
from coach.events.schema import Event
from coach.profile.store import ProfileStore
from coach.workflow.pipeline import AnswerPipeline

logger = logging.getLogger(__name__)


class ProfileWorker:
    #: 本 worker 认领的事件类型。多个 worker 共用一个 journal 时,
    #: recovery 靠它分流,避免互相重放对方的消息。
    handled_types = (events.ANSWER_SUBMITTED,)

    def __init__(
        self,
        profile: ProfileStore,
        knowledge,
        journal,
        bus=None,
        llm=None,
        pipeline: Optional[AnswerPipeline] = None,
        checkpointer=None,
        stream: str = config.STREAM_ANSWER,
        group: str = config.GROUP_COACH,
        consumer: str = "profile-1",
    ):
        self.profile = profile
        self.knowledge = knowledge
        self.journal = journal
        self.bus = bus
        self.stream = stream
        self.group = group
        self.consumer = consumer
        self._stop = asyncio.Event()
        self.pipeline = pipeline or AnswerPipeline(
            knowledge, profile, journal, llm=llm, bus=bus, checkpointer=checkpointer
        )

    # ---------------- 业务 ----------------

    def process(self, event: Event) -> Dict[str, Any]:
        """处理一条事件。**幂等**:重复调用同一 event 不会二次更新画像。"""
        if self.profile.is_processed(event.event_id):
            return {"status": "skipped", "reason": "already_processed"}

        self.journal.record(event)  # 先记后做
        result = self.handle(event)
        self.profile.mark_processed(event.event_id)
        self.journal.mark_done(event.event_id)
        return result

    def handle(self, event: Event) -> Dict[str, Any]:
        if event.type == events.ANSWER_SUBMITTED:
            return self.handle_answer(event)
        logger.warning("profile_worker 不处理的事件类型:%s", event.type)
        return {"status": "ignored", "type": event.type}

    def handle_answer(self, event: Event) -> Dict[str, Any]:
        """整条处理链交给 workflow/pipeline.py(P4 起由 LangGraph 承载)。

        这一层只负责「从流里取消息 → 交给链 → 按契约收尾」,
        链内部的判分/BKT/SM-2/根因/解释/幂等全在 pipeline 里。
        """
        return self.pipeline.run(event)

    # ---------------- 消费循环 ----------------

    async def run(self, block_ms: int = 1000, count: int = 10) -> None:
        logger.info("profile_worker 启动:stream=%s group=%s", self.stream, self.group)
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
        except Exception as exc:  # noqa: BLE001 — 失败要按契约重投或进死信
            logger.exception("处理失败:%s", event.event_id)
            attempt = event.attempt + 1
            if attempt > config.MAX_ATTEMPTS:
                self.bus.move_to_dlq(self.stream, event, f"{type(exc).__name__}: {exc}")
                self.journal.mark_done(event.event_id)  # 不再重试,别让它被恢复重放
                logger.error("进入死信:%s", event.event_id)
            else:
                self.bus.publish(self.stream, event.with_attempt(attempt))
        finally:
            self.bus.ack(self.stream, self.group, msg_id)
