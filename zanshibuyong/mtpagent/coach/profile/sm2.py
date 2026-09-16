"""SM-2 间隔重复(work.md §5.4,照抄公式)。

为什么用 SM-2 不用 FSRS(§3.2 已定):
参数少、可解释、够用。FSRS 是优化项不是必需项。

状态四个字段:
- EF  ease factor,难度因子,默认 2.5,下限 1.3
- I   间隔天数
- reps 连续通过次数
- lapses 遗忘次数
"""

import math
from dataclasses import dataclass
from typing import Optional

from coach import config


@dataclass
class SM2State:
    ease: float = config.SM2_DEFAULT_EF
    interval_days: float = 0.0
    reps: int = 0
    lapses: int = 0
    last_review: Optional[float] = None
    due_at: Optional[float] = None

    @classmethod
    def from_row(cls, row: Optional[dict]) -> "SM2State":
        """从 profile/store 的 learner_memory 行还原。没有记录就是全新状态。"""
        if not row:
            return cls()
        return cls(
            ease=row.get("ease") if row.get("ease") is not None else config.SM2_DEFAULT_EF,
            interval_days=row.get("interval_days") or 0.0,
            reps=row.get("reps") or 0,
            lapses=row.get("lapses") or 0,
            last_review=row.get("last_review"),
            due_at=row.get("due_at"),
        )


def quality_from_correct(correct: bool) -> int:
    """把二值判分映射成 SM-2 的 0~5 评分。映射理由见 config.py。"""
    return config.SM2_QUALITY_CORRECT if correct else config.SM2_QUALITY_WRONG


def review(state: Optional[SM2State], quality: int, now: float) -> SM2State:
    """一次复习后的新状态。state 为 None 时按全新知识点起算。

    q < 3(失败):reps 归零,I 回到 1 天,lapses +1,EF 不动。
    q ≥ 3(通过):按 reps 推下一间隔,并按公式调 EF(下限 1.3)。
    """
    if not 0 <= quality <= 5:
        raise ValueError(f"评分必须在 0~5 之间,收到 {quality}")

    current = state or SM2State()

    if quality < config.SM2_PASS_SCORE:
        ease = current.ease
        interval = 1.0
        reps = 0
        lapses = current.lapses + 1
    else:
        ease = _next_ease(current.ease, quality)
        reps = current.reps + 1
        interval = _next_interval(current.interval_days, ease, reps)
        lapses = current.lapses

    return SM2State(
        ease=ease,
        interval_days=interval,
        reps=reps,
        lapses=lapses,
        last_review=now,
        due_at=now + interval * config.DAY_SECONDS,
    )


def _next_interval(prev_interval: float, ease: float, reps: int) -> float:
    """reps 是**本次之后**的连续通过次数(对齐经典 SM-2 的 n=1→1, n=2→6)。"""
    if reps == 1:
        return 1.0
    if reps == 2:
        return 6.0
    return float(math.floor(prev_interval * ease))


def _next_ease(ease: float, quality: int) -> float:
    delta = 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)
    return max(config.SM2_MIN_EF, ease + delta)


def is_due(state: Optional[SM2State], now: float) -> bool:
    return bool(state and state.due_at is not None and state.due_at <= now)
