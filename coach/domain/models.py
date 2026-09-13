"""领域模型(work.md §4 数据模型的 Python 侧映射)。

约定:
- 时间戳一律用 unix 秒(float),由调用方传入或在构造时取 now。
- JSON 字段(test_cases / kp_ids / payload)在模型里是 Python 对象,
  序列化交给 store 层。
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# 边的类型。**只有一种**(2026-09-13 决定,见 work.md §11.3 / §12):
#
# 原设计照搬了"四类关系"(prerequisite/related/extends/contrasts),但读 WeSmartFlow
# 源码后发现那是错记 —— 它实际是 8 类 + embedding 探针路由,且没有 per-type 分支。
# 更要紧的是:另外三类在我们这里**存进图却没有任何代码读取**,是装饰不是设计。
#
# 所以收窄成只做 PREREQUISITE,把前置关系做深(多跳闭包 + 根因定位)。
# 这是**有意的裁剪,不是遗漏** —— 面试被问"为什么不做多类型"时,答
# "我只做根因定位需要的前置,多类型我评估过、主动砍掉了"。
EDGE_TYPES = ("PREREQUISITE",)

OBSERVATION_KINDS = ("edge", "concept")
PROPOSAL_STATUSES = ("pending", "accepted", "rejected")


def _now() -> float:
    return time.time()


@dataclass
class KnowledgePoint:
    """知识点。id 形如 "algo.dp"。"""

    id: str
    name: str
    subject: str
    difficulty: float = 3.0
    description: Optional[str] = None
    created_at: float = field(default_factory=_now)


@dataclass
class Edge:
    """四类关系之一。from_id 是起点,PREREQUISITE 时 from 是 to 的前置。"""

    from_id: str
    to_id: str
    type: str = "PREREQUISITE"
    weight: float = 1.0
    confidence: float = 1.0
    source: Optional[str] = None

    def key(self) -> tuple[str, str, str]:
        return (self.from_id, self.to_id, self.type)


@dataclass
class Problem:
    """题目。judge_type 决定判分方式,P0 只支持 exact_output。"""

    id: str
    title: str
    source: Optional[str] = None
    difficulty: Optional[float] = None
    judge_type: str = "exact_output"
    test_cases: List[Dict[str, Any]] = field(default_factory=list)
    kp_ids: List[str] = field(default_factory=list)


@dataclass
class Observation:
    """原始观察:LLM 或人工得到的信号,**未经验证**。"""

    id: str
    kind: str
    payload: Dict[str, Any]
    source: Optional[str] = None
    observed_at: float = field(default_factory=_now)


@dataclass
class Proposal:
    """提案:观察经规则校验后的产物,等待 aggregator 决定是否入库。"""

    id: str
    observation_id: Optional[str]
    kind: str
    payload: Dict[str, Any]
    status: str = "pending"
    reason: Optional[str] = None
    created_at: float = field(default_factory=_now)
