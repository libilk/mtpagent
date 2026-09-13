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
import time
from typing import Any, Dict, Optional

from coach import config
from coach.domain.grading import grade
from coach.events import schema as events
from coach.events.schema import Event
from coach.profile import bkt, sm2
from coach.profile.store import ProfileStore

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
        payload = event.payload
        learner_id = event.learner_id
        problem_id = payload.get("problem_id", "")
        answer_text = payload.get("answer_text", "")
        elapsed_ms = int(payload.get("elapsed_ms", 0))

        problem = self.knowledge.get_problem(problem_id)
        if problem is None:
            logger.warning("题目不存在:%s", problem_id)
            return {"status": "problem_not_found", "problem_id": problem_id}

        try:
            correct = grade(problem, answer_text)
        except ValueError as exc:
            logger.warning("无法判分:%s", exc)
            return {"status": "ungradeable", "reason": str(exc)}

        kp_ids = list(problem.kp_ids)
        # ★ 第二道幂等闸:answer_id 已存在说明这次答题已经计过画像
        inserted = self.profile.record_answer(
            answer_id=event.event_id,
            learner_id=learner_id,
            problem_id=problem_id,
            kp_ids=kp_ids,
            correct=correct,
            answer_text=answer_text,
            elapsed_ms=elapsed_ms,
        )
        if not inserted:
            return {"status": "duplicate_answer"}

        now = time.time()
        quality = sm2.quality_from_correct(correct)
        mastery = self.profile.get_mastery_map(learner_id, kp_ids)
        due_at = {}
        for kp_id in kp_ids:
            new_p = bkt.update(mastery.get(kp_id), correct)
            mastery[kp_id] = new_p
            self.profile.set_mastery(learner_id, kp_id, new_p)

            memory = sm2.review(
                sm2.SM2State.from_row(self.profile.get_memory(learner_id, kp_id)),
                quality,
                now,
            )
            self._save_memory(learner_id, kp_id, memory)
            due_at[kp_id] = memory.due_at

            if not correct:
                self.profile.bump_error(learner_id, kp_id, "wrong")

        self._publish_profile_updated(event, kp_ids, mastery)
        return {
            "status": "updated",
            "correct": correct,
            "mastery": {k: round(v, 4) for k, v in mastery.items()},
            "due_at": due_at,
        }

    def _save_memory(self, learner_id: str, kp_id: str, memory: sm2.SM2State) -> None:
        self.profile.set_memory(
            learner_id,
            kp_id,
            ease=memory.ease,
            interval_days=memory.interval_days,
            reps=memory.reps,
            lapses=memory.lapses,
            last_review=memory.last_review,
            due_at=memory.due_at,
        )

    def _publish_profile_updated(self, event: Event, kp_ids, mastery) -> None:
        if self.bus is None:
            return
        update = events.new_event(
            events.PROFILE_UPDATED,
            learner_id=event.learner_id,
            payload={"kp_ids": list(kp_ids), "mastery": mastery, "source_event": event.event_id},
            trace_id=event.trace_id,
        )
        try:
            self.bus.publish(config.STREAM_PROFILE, update)
        except Exception:  # noqa: BLE001 — 下游订阅失败不该影响答题处理
            logger.exception("发布 profile.updated 失败")

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
