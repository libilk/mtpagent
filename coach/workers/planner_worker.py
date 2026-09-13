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
from coach.workflow.query import QueryService

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "你是一位算法课老师。请用 2~3 句中文,直接告诉学生他真正薄弱的前置基础是什么,"
    "为什么这个基础会导致他当前学不下去,以及应该先补哪一个。"
    "像跟人说话一样,不要罗列数据,不要用表格,不要复述题目。"
)


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
        refreshed = [kp for kp in kp_ids if self.refresh_explanation(event.learner_id, kp)]
        return {"status": "ok", "refreshed": refreshed, "checked": kp_ids}

    def handle_tick(self, event: Event) -> Dict[str, Any]:
        """定时巡检:对学习者目标知识点重算根因(P3 会在此基础上出计划)。"""
        kp_ids = list(event.payload.get("kp_ids") or [])
        if not kp_ids:
            profile = self.profile.get_profile(event.learner_id)
            goal = (profile or {}).get("goal_kp_id")
            kp_ids = [goal] if goal else []
        refreshed = [kp for kp in kp_ids if self.refresh_explanation(event.learner_id, kp)]
        return {"status": "ok", "refreshed": refreshed}

    def refresh_explanation(self, learner_id: str, kp_id: str) -> bool:
        """重算根因并（有缺口且配了 LLM 时）写入人话解释。返回是否写了缓存。"""
        try:
            view = self.service.gap_view(learner_id, kp_id)
        except KeyError:
            logger.warning("知识点不存在,跳过:%s", kp_id)
            return False

        if not view["root_causes"]:
            return False  # 没有缺口就没什么可解释的
        if self.llm is None:
            return False  # 没配 LLM 时由 GET /gap 用模板兜底

        try:
            explanation = self._explain_with_llm(view)
        except Exception:  # noqa: BLE001 — LLM 失败不该拖垮消费循环
            logger.exception("生成根因解释失败:%s/%s", learner_id, kp_id)
            return False

        self.profile.set_gap_explanation(learner_id, kp_id, explanation)
        return True

    def _explain_with_llm(self, view: Dict) -> str:
        lines = [
            f"目标知识点:{view['name']}(当前掌握度 {view['mastery']:.2f})",
            "前置缺口(按重要度排序):",
        ]
        for i, gap in enumerate(view["root_causes"], 1):
            chain = " → ".join(gap.get("path") or [])
            lines.append(
                f"{i}. {gap['name']} 掌握度 {gap['mastery']:.2f}"
                + (f",依赖路径 {chain}" if chain else "")
            )
        return self.llm.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "\n".join(lines)},
            ]
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
