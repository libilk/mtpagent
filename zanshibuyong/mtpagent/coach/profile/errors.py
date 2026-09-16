"""易错模式分析(work.md §6 里列了 `profile/errors.py`,这就是它的内容)。

分工:
- **写**在 `store.py`(`bump_error`:每次答错 +1,纯持久化)
- **读+分析**在这里 —— 光把计数读出来不叫"模式",得能回答
  "这个学生是**某个知识点不会**,还是**某类错误在好多知识点上重复犯**"。

前者的对策是补那个知识点;后者说明是**系统性误解**,补单点没用。
这个区分就是本模块存在的理由。
"""

from typing import Dict, List, Optional

#: 同一个 error_type 至少在这么多个知识点上出现,才算"系统性"
SYSTEMATIC_MIN_KPS = 2


def top_errors(store, learner_id: str, limit: int = 10) -> List[Dict]:
    """错误计数排行(次数降序)。"""
    return store.list_errors(learner_id)[:limit]


def recurring_types(
    store, learner_id: str, min_kps: int = SYSTEMATIC_MIN_KPS
) -> List[Dict]:
    """## 跨越多个知识点重复出现的错误类型 —— 疑似系统性误解。

    例:同一个 `wrong` 类型在 5 个知识点上都出现过 —— 那大概率不是"这 5 个点
    都没学会",而是某类更底层的理解有偏差。
    """
    buckets: Dict[str, Dict] = {}
    for row in store.list_errors(learner_id):
        bucket = buckets.setdefault(
            row["error_type"], {"error_type": row["error_type"], "kp_ids": [], "count": 0}
        )
        bucket["kp_ids"].append(row["kp_id"])
        bucket["count"] += row["count"]

    recurring = [b for b in buckets.values() if len(b["kp_ids"]) >= min_kps]
    recurring.sort(key=lambda b: (-len(b["kp_ids"]), -b["count"], b["error_type"]))
    return recurring


def summary(store, learner_id: str, limit: int = 5) -> Dict:
    """给 `GET /profile` 用的错误模式摘要。"""
    return {
        "top": top_errors(store, learner_id, limit=limit),
        "recurring": recurring_types(store, learner_id),
    }
