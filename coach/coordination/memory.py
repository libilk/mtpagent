"""进程内总线(work.md §6 之外的新增件,理由见下)。

**为什么要有它:**
1. 测试需要一个不依赖真 Redis 的总线 —— 原先放在 `tests/coach/conftest.py` 里,
   现在收编到正式代码,避免两套实现漂移;
2. 让 `demo.sh` 与 README 的「clone 下来就能跑」**零外部依赖** ——
   评审人不必先装 Docker/Redis 才能看到东西。

**它不是什么(别拿它当生产的替代品):**
- 没有消费组:所有消费者抢同一个队列,单进程内可用,**跨进程无效**;
- 没有持久化:进程退出即丢,所以 `journal`/`recovery` 在它上面形同虚设;
- 没有阻塞语义:`block_ms` 被忽略,`consume` 立刻返回(可能为空)。

真正要验证 journal + recovery + 死信,必须用 `RedisBus`。
"""

from collections import defaultdict
from typing import Any, Dict, List, Tuple

from coach.events.schema import Event


def dlq_name(stream: str) -> str:
    return f"{stream}:dlq"


class InMemoryBus:
    """满足 `coordination.redis.Bus` 协议的进程内实现。"""

    def __init__(self) -> None:
        self.queues: Dict[str, List[Tuple[str, Event]]] = defaultdict(list)
        #: 已 ACK 的 (stream, msg_id),测试用来断言"确实收尾了"
        self.acked: List[Tuple[str, str]] = []
        #: (stream, event, reason)
        self.dlq: List[Tuple[str, Event, str]] = []
        self.groups: set = set()
        self._seq = 0

    def publish(self, stream: str, event: Event) -> str:
        self._seq += 1
        msg_id = f"mem-{self._seq}"
        self.queues[stream].append((msg_id, event))
        return msg_id

    def ensure_group(self, stream: str, group: str) -> None:
        self.groups.add((stream, group))

    def consume(
        self,
        stream: str,
        group: str,
        consumer: str,
        count: int = 10,
        block_ms: int = 1000,
    ) -> List[Tuple[str, Event]]:
        """取走并返回队首的 count 条。block_ms 忽略(没有阻塞语义)。"""
        self.ensure_group(stream, group)
        queue = self.queues[stream]
        taken = queue[:count]
        del queue[:count]
        return taken

    def ack(self, stream: str, group: str, msg_id: str) -> None:
        self.acked.append((stream, msg_id))

    def reclaim_stale(
        self, stream: str, group: str, consumer: str, min_idle_ms: int = 60_000, count: int = 100
    ) -> List[Tuple[str, Event]]:
        """没有 PEL 概念,永远返回空 —— 所以它测不出 recovery 的 PEL 那条路。"""
        return []

    def move_to_dlq(self, stream: str, event: Event, reason: str) -> str:
        self.dlq.append((stream, event, reason))
        return f"dlq-{len(self.dlq)}"

    def ping(self) -> bool:
        return True

    def close(self) -> None:
        pass
