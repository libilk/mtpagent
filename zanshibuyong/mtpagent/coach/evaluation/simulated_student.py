"""模拟学生(work.md §7 P5.1)。

**这一份设计的关键在于「不能和 BKT 用同一个模型」**(§11 明确点出的循环论证风险):
如果模拟学生的答题概率就是 BKT 的 slip/guess 公式,那 BKT 必然完美校准,
评测出来的数字只是在夸自己。

所以这里用的是**另一套假设**:
1. 每个知识点有一个潜在能力 `own(kp) ∈ [0,1]`,而不是离散的「掌握/未掌握」。
2. **前置会拖累表现**:`eff(kp) = own(kp) × (0.3 + 0.7 × min(eff(前置)))`。
   —— 一个学生可能"学过"动态规划,但函数调用很烂,于是动态规划也做不出来。
   这条正是图推理有用的前提,而 BKT 假设各知识点相互独立,压根没有这一项。
3. 答题概率是逻辑函数:`p = 0.02 + 0.96 × σ(6·(eff − 0.5))`。
   形式与 BKT 的 `P(1−S) + (1−P)G` 不同,参数也不重合。
4. 练有长进(边际递减),不练会忘(指数衰减)。

这些差异让评测有意义:我们测的是「BKT 能不能逼近一个**不是 BKT** 的真实过程」。
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

# 逻辑函数的陡峭度:越大越接近"会/不会"的硬阈值
SLOPE = 6.0
FLOOR = 0.02   # 完全不会也能蒙对
CEIL = 0.96    # 完全会了也可能失手
#: 前置拖累的软硬程度:0 = 完全被最弱前置决定,1 = 只看自己
PREREQ_DRAG = 0.7
#: 练习增益上限与衰减系数
PRACTICE_GAIN = 0.18
DECAY_PER_DAY = 0.06


@dataclass
class SimulatedStudent:
    """一个模拟学生。`ability` 是**真值**,评测时拿它当 ground truth。"""

    learner_id: str
    ability: Dict[str, float] = field(default_factory=dict)
    #: 每个知识点上次练习的"第几天",用来算遗忘
    last_practiced: Dict[str, int] = field(default_factory=dict)
    #: 真正被埋下的薄弱点(评测要考的答案)
    planted_weakness: Optional[str] = None

    def own(self, kp_id: str) -> float:
        return self.ability.get(kp_id, 0.5)


class StudentSimulator:
    """按上面那套假设生成学生、出答题结果、模拟练习与遗忘。"""

    def __init__(self, knowledge, seed: int = 0):
        self.knowledge = knowledge
        self.rng = random.Random(seed)
        self._eff_cache: Dict[tuple, float] = {}
        self.concept_ids = [c.id for c in knowledge.list_concepts()]

    # ---------------- 造学生 ----------------

    def make_student(
        self,
        learner_id: str,
        weakness: Optional[str] = None,
        weakness_value: float = 0.12,
        base_low: float = 0.45,
        base_high: float = 0.95,
    ) -> SimulatedStudent:
        """造一个学生。`weakness` 指定要埋哪个知识点的短板(根因评测的 ground truth)。"""
        self._eff_cache.clear()
        ability = {
            kp: self.rng.uniform(base_low, base_high) for kp in self.concept_ids
        }
        student = SimulatedStudent(learner_id=learner_id, ability=ability)
        if weakness is not None and weakness in ability:
            ability[weakness] = weakness_value
            student.planted_weakness = weakness
        return student

    # ---------------- 表现与答题 ----------------

    def effective(self, student: SimulatedStudent, kp_id: str, depth: int = 4) -> float:
        """被前置拖累之后的实际能力。递归算,结果缓存。"""
        if depth <= 0:
            return student.own(kp_id)

        key = (student.learner_id, kp_id, depth)
        if key in self._eff_cache:
            return self._eff_cache[key]

        own = student.own(kp_id)
        prereqs = self.knowledge.ancestors(kp_id, depth=1)
        if not prereqs:
            value = own
        else:
            worst = min(self.effective(student, p, depth - 1) for p in prereqs)
            value = own * ((1.0 - PREREQ_DRAG) + PREREQ_DRAG * worst)

        value = min(max(value, 0.0), 1.0)
        self._eff_cache[key] = value
        return value

    def p_correct(self, student: SimulatedStudent, kp_id: str) -> float:
        eff = self.effective(student, kp_id)
        return FLOOR + (CEIL - FLOOR) * _sigmoid(SLOPE * (eff - 0.5))

    def answer(self, student: SimulatedStudent, problem) -> bool:
        """答一道题。题目挂多个知识点时,取最弱的那个决定表现(木桶效应)。"""
        kp_ids = list(problem.kp_ids) or [problem.id]
        p = min(self.p_correct(student, kp) for kp in kp_ids)
        return self.rng.random() < p

    # ---------------- 练习与遗忘 ----------------

    def practice(self, student: SimulatedStudent, kp_id: str, gain: float = PRACTICE_GAIN) -> None:
        """练一次:能力涨一点,越接近上限涨得越少(边际递减)。"""
        current = student.own(kp_id)
        student.ability[kp_id] = min(1.0, current + gain * (1.0 - current))
        self._eff_cache.clear()

    def forget(self, student: SimulatedStudent, day: int, per_day: float = DECAY_PER_DAY) -> None:
        """时间流逝:距上次练习越久,掉得越多。"""
        for kp_id in list(student.ability):
            last = student.last_practiced.get(kp_id)
            if last is None:
                continue
            idle = max(0, day - last)
            if idle <= 0:
                continue
            retained = math.exp(-per_day * idle)
            student.ability[kp_id] *= retained
        self._eff_cache.clear()

    def mark_practiced(self, student: SimulatedStudent, kp_id: str, day: int) -> None:
        student.last_practiced[kp_id] = day

    # ---------------- 轨迹 ----------------

    def trace(
        self,
        student: SimulatedStudent,
        problem_ids: Sequence[str],
        rounds: int = 1,
    ) -> List[dict]:
        """生成答题轨迹:[{problem_id, kp_ids, correct}]。"""
        problems = []
        for pid in problem_ids:
            problem = self.knowledge.get_problem(pid)
            if problem is not None:
                problems.append(problem)

        events = []
        for _ in range(rounds):
            for problem in problems:
                correct = self.answer(student, problem)
                events.append(
                    {
                        "learner_id": student.learner_id,
                        "problem_id": problem.id,
                        "kp_ids": list(problem.kp_ids),
                        "correct": correct,
                    }
                )
        return events


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)
