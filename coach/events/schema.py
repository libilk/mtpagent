"""事件信封与类型(work.md §4.3)。

信封字段固定七项,payload 放业务数据。event_id 用 ULID:
既是幂等键,又保证「按 id 排序 = 按时间排序」。
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from coach.domain.ids import new_ulid

ANSWER_SUBMITTED = "answer.submitted"
PROFILE_UPDATED = "profile.updated"
TICK_SCHEDULED = "tick.scheduled"
PLAN_UPDATED = "plan.updated"
PROBLEM_GENERATE = "problem.generate"

EVENT_TYPES = (
    ANSWER_SUBMITTED,
    PROFILE_UPDATED,
    TICK_SCHEDULED,
    PLAN_UPDATED,
    PROBLEM_GENERATE,
)


@dataclass
class Event:
    """事件信封(§4.3)。"""

    event_id: str
    type: str
    ts: float
    learner_id: str
    payload: Dict[str, Any] = field(default_factory=dict)
    trace_id: str = ""
    attempt: int = 1

    def __post_init__(self) -> None:
        if not self.trace_id:
            self.trace_id = self.event_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "type": self.type,
            "ts": self.ts,
            "learner_id": self.learner_id,
            "trace_id": self.trace_id,
            "attempt": self.attempt,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Event":
        return cls(
            event_id=data["event_id"],
            type=data["type"],
            ts=float(data.get("ts", 0.0)),
            learner_id=data.get("learner_id", ""),
            payload=dict(data.get("payload") or {}),
            trace_id=data.get("trace_id") or "",
            attempt=int(data.get("attempt", 1)),
        )

    # Redis Stream 的字段值必须是字符串,所以整个信封序列化成一个 data 字段
    def to_stream_fields(self) -> Dict[str, str]:
        return {"data": json.dumps(self.to_dict(), ensure_ascii=False)}

    @classmethod
    def from_stream_fields(cls, fields: Mapping[str, Any]) -> "Event":
        raw = _field(fields, "data")
        if raw is None:
            raise ValueError(f"消息缺少 data 字段:{fields!r}")
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return cls.from_dict(json.loads(raw))

    def with_attempt(self, attempt: int) -> "Event":
        clone = Event.from_dict(self.to_dict())
        clone.attempt = attempt
        return clone


def _field(fields: Mapping[str, Any], name: str) -> Optional[Any]:
    if name in fields:
        return fields[name]
    return fields.get(name.encode("utf-8"))


def new_event(
    type: str,
    learner_id: str,
    payload: Optional[Dict[str, Any]] = None,
    trace_id: Optional[str] = None,
    attempt: int = 1,
    ts: Optional[float] = None,
) -> Event:
    """造一个事件。trace_id 不传则新开一条链路。"""
    event_id = new_ulid(ts)
    return Event(
        event_id=event_id,
        type=type,
        ts=time.time() if ts is None else ts,
        learner_id=learner_id,
        payload=payload or {},
        trace_id=trace_id or event_id,
        attempt=attempt,
    )
