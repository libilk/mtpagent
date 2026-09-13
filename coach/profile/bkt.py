"""BKT 掌握度更新(work.md §5.3,照抄公式,不要自己改)。

四个参数的含义:
- P_init 初始掌握概率
- T  transit 学完忘了/学会了没体现,状态自行转移的概率
- G  guess   没掌握也蒙对的概率
- S  slip    掌握了也答错的概率
"""

from typing import Dict, Iterable, Tuple

from coach import config

P_INIT = config.BKT_P_INIT
TRANSIT = config.BKT_TRANSIT
GUESS = config.BKT_GUESS
SLIP = config.BKT_SLIP


def update(
    p_known: float,
    correct: bool,
    p_init: float = P_INIT,
    p_transit: float = TRANSIT,
    p_guess: float = GUESS,
    p_slip: float = SLIP,
) -> float:
    """一次观测后的新掌握度。p_known 为 None 时按初始值起算。"""
    p = p_init if p_known is None else float(p_known)
    p = min(max(p, 0.0), 1.0)

    if correct:
        numerator = p * (1.0 - p_slip)
        denominator = numerator + (1.0 - p) * p_guess
    else:
        numerator = p * p_slip
        denominator = numerator + (1.0 - p) * (1.0 - p_guess)

    posterior = numerator / denominator if denominator > 0 else p
    return posterior + (1.0 - posterior) * p_transit


def update_many(
    mastery: Dict[str, float],
    kp_ids: Iterable[str],
    correct: bool,
    p_init: float = P_INIT,
) -> Dict[str, float]:
    """对一组知识点各更新一次,返回新掌握度(不修改入参)。"""
    result = dict(mastery)
    for kp_id in kp_ids:
        result[kp_id] = update(result.get(kp_id), correct, p_init=p_init)
    return result


def update_sequence(p_known: float, observations: Iterable[bool]) -> float:
    """把一串对/错连续喂进去,返回最终掌握度。评测和测试用。"""
    p = p_known
    for correct in observations:
        p = update(p, correct)
    return p


def predict_correct(p_known: float, p_guess: float = GUESS, p_slip: float = SLIP) -> float:
    """给定掌握度,预测「下一题答对」的概率。评测算 AUC/Brier 时用。"""
    p = min(max(float(p_known), 0.0), 1.0)
    return p * (1.0 - p_slip) + (1.0 - p) * p_guess
