"""ULID 生成(work.md §4.3 事件信封要求 event_id 为 ULID)。

26 位 Crockford Base32:前 10 位毫秒时间戳 + 后 16 位随机,
好处是**按字典序 = 按时间序**,journal 直接按 id 排序就是事件顺序。
"""

import os
import time
from typing import Optional

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def new_ulid(ts: Optional[float] = None) -> str:
    """生成一个 ULID。ts 为 unix 秒,默认取当前时间。"""
    ms = int((time.time() if ts is None else ts) * 1000)
    ts_part = ""
    for _ in range(10):
        ts_part = _CROCKFORD[ms & 31] + ts_part
        ms >>= 5

    rand = int.from_bytes(os.urandom(10), "big")
    rand_part = ""
    for _ in range(16):
        rand_part = _CROCKFORD[rand & 31] + rand_part
        rand >>= 5

    return ts_part + rand_part
