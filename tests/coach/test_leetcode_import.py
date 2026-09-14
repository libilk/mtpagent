"""LeetCode 题单导入测试(work.md §11.1 #5)。

**不打网络** —— 抓取函数是注入的,测的是"转换"这一层。转换出错会污染判分,
判分错了 BKT 学到假信号,所以这里断言的份量和判分本身一样重。
"""

import json

import pytest

from coach.knowledge import leetcode_import as lc
from coach.knowledge import problem_bank

PRE_CONTENT = """
<p><strong>Example 1:</strong></p>
<pre>
<strong>Input:</strong> nums = [100,4,200,1,3,2]
<strong>Output:</strong> 4
<strong>Explanation:</strong> The longest consecutive elements sequence is [1, 2, 3, 4].
</pre>
<p><strong>Example 2:</strong></p>
<pre>
<strong>Input:</strong> nums = [0,3,7,2,5,8,4,6,0,1]
<strong>Output:</strong> 9
</pre>
"""

SPAN_CONTENT = """
<p><strong>Example 1:</strong></p>
<p><strong>Input:</strong> <span class="example-io">nums = [3,2,1,5,6,4], k = 2</span></p>
<p><strong>Output:</strong> <span class="example-io">5</span></p>
"""


def _question(params, testcases, content):
    return {
        "metaData": json.dumps({"params": [{"name": n, "type": "integer[]"} for n in params]}),
        "exampleTestcases": testcases,
        "content": content,
    }


FAKE_LISTING = [
    ("哈希", {"titleSlug": "two-sum", "translatedTitle": "两数之和", "questionFrontendId": "1", "difficulty": "EASY"}),
    ("哈希", {"titleSlug": "longest-consecutive-sequence", "translatedTitle": "最长连续序列", "questionFrontendId": "128", "difficulty": "MEDIUM"}),
    ("普通数组", {"titleSlug": "rotate-array", "translatedTitle": "轮转数组", "questionFrontendId": "189", "difficulty": "MEDIUM"}),
    ("堆", {"titleSlug": "kth-largest-element-in-an-array", "translatedTitle": "数组中的第K个最大元素", "questionFrontendId": "215", "difficulty": "MEDIUM"}),
    ("堆", {"titleSlug": "find-median-from-data-stream", "translatedTitle": "数据流的中位数", "questionFrontendId": "295", "difficulty": "HARD"}),
    ("栈", {"titleSlug": "min-stack", "translatedTitle": "最小栈", "questionFrontendId": "155", "difficulty": "MEDIUM"}),
    ("技巧", {"titleSlug": "single-number", "translatedTitle": "只出现一次的数字", "questionFrontendId": "136", "difficulty": "EASY"}),
]

FAKE_DETAILS = {
    "longest-consecutive-sequence": _question(
        ["nums"], "[100,4,200,1,3,2]\n[0,3,7,2,5,8,4,6,0,1]", PRE_CONTENT
    ),
    "rotate-array": _question(["nums", "k"], "[1,2,3,4,5,6,7]\n3", "<p><strong>Output:</strong> [5,6,7,1,2,3,4]</p>"),
    "kth-largest-element-in-an-array": _question(
        ["nums", "k"], "[3,2,1,5,6,4]\n2\n[3,2,3,1,2,4,5,5,6]\n4", SPAN_CONTENT
    ),
    "min-stack": _question([], "", "<p>设计题,没有输入输出</p>"),
}


@pytest.fixture
def fake_fetch():
    def _fetch_plan(slug):
        return FAKE_LISTING

    def _fetch_question(slug):
        return FAKE_DETAILS[slug]

    return _fetch_plan, _fetch_question


class TestExtractOutputs:
    def test_pre_block_format(self):
        assert lc.extract_outputs(PRE_CONTENT) == ["4", "9"]

    def test_span_format(self):
        assert lc.extract_outputs(SPAN_CONTENT) == ["5"]

    def test_chinese_marker(self):
        assert lc.extract_outputs("<p><strong>输出:</strong> 7</p>") == ["7"]

    def test_no_output_returns_empty(self):
        assert lc.extract_outputs("<p>设计题</p>") == []


class TestBuildTestCases:
    def test_pairs_params_with_inputs_and_outputs(self):
        cases = lc.build_test_cases(FAKE_DETAILS["longest-consecutive-sequence"])
        # 提交用例(最后一条)= 最典型的第一条示例
        assert cases == [
            {"input": "nums=[0,3,7,2,5,8,4,6,0,1]", "expected": "9"},
            {"input": "nums=[100,4,200,1,3,2]", "expected": "4"},
        ]

    def test_multi_param_cases_are_grouped_by_param_count(self):
        cases = lc.build_test_cases(FAKE_DETAILS["kth-largest-element-in-an-array"])
        assert cases == [{"input": "nums=[3,2,1,5,6,4], k=2", "expected": "5"}]

    def test_design_problem_yields_no_cases(self):
        assert lc.build_test_cases(FAKE_DETAILS["min-stack"]) == []

    def test_output_count_caps_case_count(self):
        # 3 组输入但只有 1 个 Output —— 宁可少收,不猜配对
        detail = _question(["n"], "1\n2\n3", "<p><strong>Output:</strong> 1</p>")
        assert lc.build_test_cases(detail) == [{"input": "n=1", "expected": "1"}]

    def test_ragged_trailing_input_is_dropped(self):
        detail = _question(["a", "b"], "1\n2\n3", "<p><strong>Output:</strong> x</p><p><strong>Output:</strong> y</p>")
        assert lc.build_test_cases(detail) == [{"input": "a=1, b=2", "expected": "x"}]

    def test_empty_input_case_is_dropped_from_submit_slot(self):
        """判分取最后一条 —— LeetCode 常把 `head=[]` 这种边界样例放最后,得挤掉。"""
        detail = _question(
            ["head"],
            "[1,2,3]\n[]",
            "<p><strong>Output:</strong> [3,2,1]</p><p><strong>Output:</strong> []</p>",
        )
        cases = lc.build_test_cases(detail)
        assert cases[-1] == {"input": "head=[1,2,3]", "expected": "[3,2,1]"}

    def test_all_degenerate_cases_are_kept_rather_than_emptied(self):
        detail = _question(["head"], "[]", "<p><strong>Output:</strong> []</p>")
        assert lc.build_test_cases(detail) == [{"input": "head=[]", "expected": "[]"}]


class TestKpMapping:
    def test_group_default(self):
        assert lc.kp_ids_for("链表", "reverse-linked-list") == ["ds.linked_list"]

    def test_slug_override_beats_group(self):
        assert lc.kp_ids_for("堆", "kth-largest-element-in-an-array") == [
            "algo.quicksort",
            "algo.divide_conquer",
            "algo.sorting",
        ]

    def test_group_without_concept_and_no_override_is_unmapped(self):
        assert lc.kp_ids_for("堆", "find-median-from-data-stream") == []
        assert lc.kp_ids_for("技巧", "single-number") == []


class TestToBank:
    def test_bank_shape_is_accepted_by_parse_bank(self, fake_fetch):
        bank, _ = lc.to_bank("fake", fetch_plan=fake_fetch[0], fetch_question=fake_fetch[1])
        parsed = problem_bank.parse_bank(bank)
        assert {p["id"] for p in parsed} == {"lc.128", "lc.189", "lc.215"}

    def test_skips_multi_answer_design_and_unmapped(self, fake_fetch):
        _, report = lc.to_bank("fake", fetch_plan=fake_fetch[0], fetch_question=fake_fetch[1])
        assert report["total_in_plan"] == 7
        assert report["imported"] == 3
        assert [e["slug"] for e in report["skipped"]["multi_answer"]] == ["two-sum"]
        assert [e["slug"] for e in report["skipped"]["design"]] == ["min-stack"]
        assert {e["slug"] for e in report["skipped"]["unmapped"]} == {
            "find-median-from-data-stream",
            "single-number",
        }

    def test_marks_in_place_problems(self, fake_fetch):
        _, report = lc.to_bank("fake", fetch_plan=fake_fetch[0], fetch_question=fake_fetch[1])
        assert [e["slug"] for e in report["in_place"]] == ["rotate-array"]

    def test_imported_problems_carry_test_cases_and_source(self, fake_fetch):
        bank, _ = lc.to_bank("fake", fetch_plan=fake_fetch[0], fetch_question=fake_fetch[1])
        by_id = {p["id"]: p for p in bank["problems"]}
        assert by_id["lc.128"]["source"] == "leetcode:128"
        assert by_id["lc.128"]["test_cases"][-1]["expected"] == "4"
        assert by_id["lc.215"]["kp_ids"] == ["algo.quicksort", "algo.divide_conquer", "algo.sorting"]

    def test_every_generated_problem_is_gradeable(self, fake_fetch, seeded):
        """★ 生成的每一题都必须真的能判分 —— 挂了非存在的知识点会被 import 丢掉。"""
        bank, _ = lc.to_bank("fake", fetch_plan=fake_fetch[0], fetch_question=fake_fetch[1])
        result = problem_bank.import_problems(seeded, problem_bank.parse_bank(bank))
        assert result["skipped"] == []
        assert result["written"] == 3


class TestFetchFailures:
    def test_question_fetch_error_is_reported_not_raised(self, fake_fetch):
        plan, _ = fake_fetch

        def boom(slug):
            raise lc.LeetCodeError("502")

        _, report = lc.to_bank("fake", fetch_plan=plan, fetch_question=boom)
        # 只有"能映射且不是多解"的题才会去请求:128 / 189 / 215 / 155 四道
        assert report["imported"] == 0
        assert len(report["skipped"]["parse_failed"]) == 4

    def test_plan_not_found_raises(self):
        def missing(slug):
            raise lc.LeetCodeError("题单不存在")

        with pytest.raises(lc.LeetCodeError):
            lc.to_bank("nope", fetch_plan=missing, fetch_question=lambda s: {})
