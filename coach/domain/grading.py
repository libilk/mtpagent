"""判分(work.md P1.9)。**纯规则,不调 LLM** —— 这是 §3.2 已确认的取舍。

P1 只做最简单的一种:exact_output —— 归一化后字符串相等。
约定:多用例的题目,以**最后一个用例**为提交用例(前面的用例是样例)。
"""

from typing import Any

from coach.domain.models import Problem


def normalize(text: Any) -> str:
    """去首尾空白、把内部连续空白压成一个空格。避免格式差异误判。"""
    if text is None:
        return ""
    return " ".join(str(text).strip().split())


def grade(problem: Problem, answer_text: str) -> bool:
    """规则判分。题目不可判时抛 ValueError,由 worker 记为 ungradeable。"""
    if problem.judge_type != "exact_output":
        raise ValueError(f"P1 尚不支持 judge_type={problem.judge_type}")
    if not problem.test_cases:
        raise ValueError(f"题目 {problem.id} 没有 test_cases,无法判分")

    expected = problem.test_cases[-1].get("expected")
    if expected is None:
        raise ValueError(f"题目 {problem.id} 的用例缺少 expected 字段")

    return normalize(answer_text) == normalize(expected)
