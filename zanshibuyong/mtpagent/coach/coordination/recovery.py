"""重启恢复(work.md §2.1 / §5.6 第 7 步)。

Redis Stream 的 PEL 能兜住「投递了但没 ACK」,但兜不住
「进程在写 journal 之后、ACK 之前被杀」。所以恢复要走两条路:

1. **journal** —— 扫「已记未完成」的事件,重新执行业务(先记后做的价值所在)
2. **PEL**    —— 认领超时未 ACK 的消息,补 ACK

业务侧的幂等由 processed_events + answers.answer_id 唯一键保证,
所以这里「重放」是安全的:重复执行不会把画像更新两次。
"""

import logging
from typing import Callable, Iterable, List, Optional

from coach.coordination.journal import Journal
from coach.events.schema import Event

logger = logging.getLogger(__name__)


class Recovery:
    def __init__(self, journal: Journal, bus: Optional[object] = None):
        self.journal = journal
        self.bus = bus

    def pending_events(self) -> List[Event]:
        return self.journal.pending()

    def replay(
        self,
        handler: Callable[[Event], None],
        types: Optional[Iterable[str]] = None,
    ) -> List[str]:
        """把「已记未完成」的事件重放一遍。返回成功补做的 event_id 列表。

        handler 抛异常时该事件保持未完成,下次启动还会被捞出来——
        不吞异常,不假装成功。

        types: 只重放这些事件类型。多个 worker 共用一个 journal 时,
        必须各自认领自己的类型,否则会互相重放对方的消息。
        """
        allowed = set(types) if types is not None else None
        replayed: List[str] = []
        for event in self.pending_events():
            if allowed is not None and event.type not in allowed:
                continue
            try:
                handler(event)
            except Exception:  # noqa: BLE001 — 单个事件失败不该拖垮整个恢复流程
                logger.exception("恢复重放失败,事件保留为未完成:%s", event.event_id)
                continue
            replayed.append(event.event_id)
        if replayed:
            logger.info("恢复补做完成:%d 条", len(replayed))
        return replayed

    def reclaim_stale(
        self,
        stream: str,
        group: str,
        consumer: str,
        min_idle_ms: int = 60_000,
    ) -> int:
        """认领超时未 ACK 的消息并 ACK 掉(业务已由 journal 重放兜住)。"""
        if self.bus is None:
            return 0
        stale = self.bus.reclaim_stale(stream, group, consumer, min_idle_ms=min_idle_ms)
        for msg_id, _event in stale:
            self.bus.ack(stream, group, msg_id)
        if stale:
            logger.info("认领并清理未 ACK 消息:%d 条", len(stale))
        return len(stale)

    def run(
        self,
        handler: Callable[[Event], None],
        stream: Optional[str] = None,
        group: str = None,
        consumer: str = None,
        types: Optional[Iterable[str]] = None,
    ) -> dict:
        """启动时的完整恢复:先补做 journal,再清理 PEL。"""
        replayed = self.replay(handler, types=types)
        reclaimed = 0
        if self.bus is not None and stream:
            reclaimed = self.reclaim_stale(
                stream, group or "coach-workers", consumer or "recovery"
            )
        return {"replayed": len(replayed), "reclaimed": reclaimed}
