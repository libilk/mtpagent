"""评测指标(work.md §7 P5.2)。全是纯函数,好测。

- AUC / Brier:掌握度预测准不准
- 校准曲线:预测"70% 会答对"的人,是不是真的七成答对
- 准确率@1:根因定位的头号答案对不对
"""

import math
from typing import Dict, List, Sequence, Tuple


def _average_ranks(values: Sequence[float]) -> List[float]:
    """并列值取平均名次。AUC 对并列的处理全靠这里。"""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def auc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """ROC-AUC(秩和法)。只有单一类别时返回 nan —— 那种情况 AUC 无定义。"""
    if len(scores) != len(labels):
        raise ValueError("scores 与 labels 长度不一致")
    positives = sum(1 for label in labels if label)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")

    ranks = _average_ranks(list(scores))
    rank_sum = sum(rank for rank, label in zip(ranks, labels) if label)
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def brier(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Brier 分数:均方误差,**越小越好**。"""
    if not scores:
        return float("nan")
    return sum((p - (1.0 if y else 0.0)) ** 2 for p, y in zip(scores, labels)) / len(scores)


def log_loss(scores: Sequence[float], labels: Sequence[bool], eps: float = 1e-12) -> float:
    if not scores:
        return float("nan")
    total = 0.0
    for p, y in zip(scores, labels):
        p = min(max(p, eps), 1.0 - eps)
        total += -math.log(p if y else 1.0 - p)
    return total / len(scores)


def calibration_curve(
    scores: Sequence[float], labels: Sequence[bool], bins: int = 10
) -> List[Dict]:
    """分箱校准:每个箱里「平均预测值」vs「实际正确率」。"""
    buckets: List[List[Tuple[float, bool]]] = [[] for _ in range(bins)]
    for p, y in zip(scores, labels):
        index = min(int(p * bins), bins - 1)
        if index < 0:
            index = 0
        buckets[index].append((p, y))

    curve = []
    for index, bucket in enumerate(buckets):
        if not bucket:
            continue
        mean_predicted = sum(p for p, _ in bucket) / len(bucket)
        observed = sum(1 for _, y in bucket if y) / len(bucket)
        curve.append(
            {
                "bin_low": index / bins,
                "bin_high": (index + 1) / bins,
                "count": len(bucket),
                "mean_predicted": round(mean_predicted, 4),
                "observed_rate": round(observed, 4),
                "gap": round(observed - mean_predicted, 4),
            }
        )
    return curve


def expected_calibration_error(curve: Sequence[Dict], total: int) -> float:
    """ECE:各箱 |实际−预测| 按样本数加权平均。越小越准。"""
    if not total:
        return float("nan")
    return sum(abs(item["gap"]) * item["count"] for item in curve) / total


def top1_accuracy(predicted: Sequence[Sequence[str]], truth: Sequence[str]) -> float:
    """头号答案命中率。predicted 是每个样本的候选列表(有序)。"""
    if not truth:
        return float("nan")
    hits = sum(
        1 for candidates, answer in zip(predicted, truth) if candidates and candidates[0] == answer
    )
    return hits / len(truth)


def recall_at_k(predicted: Sequence[Sequence[str]], truth: Sequence[str]) -> float:
    """真值出现在前 k 个候选里的比例(k 由候选列表长度决定)。"""
    if not truth:
        return float("nan")
    hits = sum(1 for candidates, answer in zip(predicted, truth) if answer in candidates)
    return hits / len(truth)


def mean(values: Sequence[float]) -> float:
    clean = [v for v in values if not math.isnan(v)]
    return sum(clean) / len(clean) if clean else float("nan")
