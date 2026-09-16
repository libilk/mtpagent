"""Redis Stream 封装(work.md §2.1 / §3)。

为什么不用 Kafka:这里要解决的是「不丢消息」,而这件事的关键在
journal + recovery,不在中间件有多重(见 mainconten §2.4)。

对外只暴露 Bus 协议需要的方法,worker 依赖协议而非具体实现,
测试可以塞一个内存实现进来,不必起 Redis。
"""

import logging
import time
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import redis

from coach import config
from coach.events.schema import Event

logger = logging.getLogger(__name__)


@runtime_checkable
class Bus(Protocol):
    """worker 依赖的最小接口。"""

    def publish(self, stream: str, event: Event) -> str: ...

    def consume(
        self, stream: str, group: str, consumer: str, count: int = 10, block_ms: int = 1000
    ) -> List[Tuple[str, Event]]: ...

    def ack(self, stream: str, group: str, msg_id: str) -> None: ...


def dlq_name(stream: str) -> str:
    return f"{stream}:dlq"


class RedisBus:
    """Redis Stream 实现:消费组 + 死信。"""

    def __init__(self, url: Optional[str] = None, client: Optional[Any] = None):
        self.url = url or config.REDIS_URL
        # decode_responses=True:字段直接拿 str,省掉到处 decode
        self.client = client or redis.Redis.from_url(self.url, decode_responses=True)
        self._groups: set = set()

    # ---------------- 消费组 ----------------

    def ensure_group(self, stream: str, group: str) -> None:
        """建消费组,已存在则忽略(BUSYGROUP)。幂等,可在启动时反复调。"""
        if (stream, group) in self._groups:
            return
        try:
            self.client.xgroup_create(stream, group, id="0", mkstream=True)
        except redis.ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise
        self._groups.add((stream, group))

    # ---------------- 生产 ----------------

    def publish(self, stream: str, event: Event) -> str:
        msg_id = self.client.xadd(stream, event.to_stream_fields())
        return msg_id if isinstance(msg_id, str) else msg_id.decode("utf-8")

    def move_to_dlq(self, stream: str, event: Event, reason: str) -> str:
        """超过重试上限后挪进死信流,保留失败原因与尝试次数。"""
        fields = event.to_stream_fields()
        fields["reason"] = reason
        fields["failed_at"] = str(time.time())
        msg_id = self.client.xadd(dlq_name(stream), fields)
        return msg_id if isinstance(msg_id, str) else msg_id.decode("utf-8")

    # ---------------- 消费 ----------------

    def consume(
        self,
        stream: str,
        group: str,
        consumer: str,
        count: int = 10,
        block_ms: int = 1000,
    ) -> List[Tuple[str, Event]]:
        """读新消息(xreadgroup 的 '>')。返回 [(msg_id, event)]。"""
        self.ensure_group(stream, group)
        resp = self.client.xreadgroup(
            group, consumer, {stream: ">"}, count=count, block=block_ms
        )
        return self._unpack(resp)

    def reclaim_stale(
        self,
        stream: str,
        group: str,
        consumer: str,
        min_idle_ms: int = 60_000,
        count: int = 100,
    ) -> List[Tuple[str, Event]]:
        """认领别人(或已死进程)留下的未 ACK 消息。重启时配合 recovery 用。"""
        self.ensure_group(stream, group)
        resp = self.client.xautoclaim(
            stream, group, consumer, min_idle_time=min_idle_ms, start_id="0-0", count=count
        )
        # redis-py: (next_cursor, messages) 或 (next_cursor, messages, deleted)
        messages = resp[1] if len(resp) > 1 else []
        return self._unpack([[stream, messages]])

    def ack(self, stream: str, group: str, msg_id: str) -> None:
        self.client.xack(stream, group, msg_id)

    def pending_count(self, stream: str, group: str) -> int:
        info = self.client.xpending(stream, group)
        return info["pending"] if isinstance(info, dict) else info[0]

    def stream_length(self, stream: str) -> int:
        return int(self.client.xlen(stream))

    def read_dlq(self, stream: str, count: int = 100) -> List[Dict[str, Any]]:
        entries = self.client.xrange(dlq_name(stream), count=count)
        out = []
        for msg_id, fields in entries:
            item = dict(fields)
            item["msg_id"] = msg_id
            out.append(item)
        return out

    # ---------------- 杂项 ----------------

    def ping(self) -> bool:
        try:
            return bool(self.client.ping())
        except Exception:  # noqa: BLE001 — 探活用,任何异常都算不通
            logger.warning("Redis 不可达:%s", self.url)
            return False

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _unpack(resp: Any) -> List[Tuple[str, Event]]:
        out: List[Tuple[str, Event]] = []
        for _stream, messages in resp or []:
            for msg_id, fields in messages or []:
                out.append((msg_id, Event.from_stream_fields(fields)))
        return out
