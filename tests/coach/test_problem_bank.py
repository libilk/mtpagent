"""题库导入测试(work.md §11.1 #5)。

题库不是可有可无的东西:**一个知识点没有题,就永远产生不了作答证据,
也就永远无法被定位为根因**。所以这几条断言的份量和根因定位一样重。
"""

import json

import pytest

from coach.knowledge import builder, problem_bank
from coach.knowledge.governance import Governance


@pytest.fixture
def seeded(knowledge):
    builder.seed_graph(Governance(knowledge))
    return knowledge


class TestParseBank:
    def test_parses_valid_payload(self):
        parsed = problem_bank.parse_bank(
            {
                "problems": [
                    {
                        "id": "p1",
                        "title": "题一",
                        "kp_ids": ["a"],
                        "test_cases": [{"input": "1", "expected": "2"}],
                    }
                ]
            }
        )
        assert len(parsed) == 1
        assert parsed[0]["judge_type"] == "exact_output"

    @pytest.mark.parametrize(
        "bad",
        [
            {"problems": [{"id": "p", "title": "t", "kp_ids": ["a"]}]},               # 缺用例
            {"problems": [{"id": "p", "title": "t", "test_cases": [{"expected": "1"}]}]},  # 缺 kp
            {"problems": [{"id": "p", "kp_ids": ["a"], "test_cases": [{"expected": "1"}]}]},  # 缺 title
            {"problems": [{"id": "p", "title": "t", "kp_ids": ["a"], "test_cases": []}]},  # 空用例
        ],
    )
    def test_skips_dirty_entries_instead_of_raising(self, bad):
        """外部数据脏了不该让整批失败 —— 跳过坏的,留下好的。"""
        assert problem_bank.parse_bank(bad) == []

    def test_requires_problems_array(self):
        with pytest.raises(ValueError):
            problem_bank.parse_bank({"problems": "nope"})


class TestImport:
    def test_written_and_skipped(self, seeded):
        stats = problem_bank.import_problems(
            seeded,
            [
                {"id": "ok", "title": "能进", "kp_ids": ["ds.array"], "test_cases": [{"expected": "1"}]},
                {"id": "bad", "title": "悬空引用", "kp_ids": ["does.not.exist"],
                 "test_cases": [{"expected": "1"}]},
            ],
        )

        assert stats["written"] == 1
        assert seeded.get_problem("ok") is not None
        assert seeded.get_problem("bad") is None
        assert "does.not.exist" in stats["skipped"][0][1]

    def test_reimport_is_idempotent(self, knowledge, seeded):
        payload = [
            {"id": "p1", "title": "题", "kp_ids": ["ds.array"], "test_cases": [{"expected": "1"}]}
        ]
        problem_bank.import_problems(seeded, payload)
        problem_bank.import_problems(seeded, payload)

        assert len(seeded.list_problems()) == 1


class TestExpectedAnswer:
    def test_returns_last_case_expected(self, seeded):
        problem_bank.import_problems(
            seeded,
            [{"id": "p1", "title": "题", "kp_ids": ["ds.array"],
              "test_cases": [{"expected": "样例"}, {"expected": "提交用例"}]}],
        )
        assert problem_bank.expected_answer(seeded, "p1") == "提交用例"

    def test_missing_problem_returns_none(self, seeded):
        assert problem_bank.expected_answer(seeded, "nope") is None


class TestBundledBank:
    def test_bundled_file_loads(self):
        bank = problem_bank.load_from_file(problem_bank.BUNDLED_BANK)
        assert len(bank) >= 19

    def test_bundled_bank_covers_every_concept(self, seeded):
        """★ 每个知识点都要有题 —— 否则根因定位在原理上够不着它。"""
        problem_bank.seed_from_bank(seeded)

        stats = problem_bank.coverage(seeded)

        assert stats["missing"] == [], f"这些知识点没有题:{stats['missing']}"
        assert stats["covered"] == stats["total_concepts"]

    def test_every_bundled_problem_is_gradable(self, seeded):
        """从题库进来的每一道题,拿它的 expected 提交都必须判对。"""
        from coach.domain.grading import grade

        problem_bank.seed_from_bank(seeded)
        for problem in seeded.list_problems():
            expected = problem_bank.expected_answer(seeded, problem.id)
            assert expected is not None, f"{problem.id} 没有可用例"
            assert grade(problem, expected) is True, f"{problem.id} 的正确提交被判错"

    def test_every_bundled_problem_references_real_concepts(self, seeded):
        problem_bank.seed_from_bank(seeded)
        concepts = {c.id for c in seeded.list_concepts()}
        for problem in seeded.list_problems():
            assert set(problem.kp_ids) <= concepts, problem.id


class TestCoverageReport:
    def test_reports_missing(self, seeded):
        problem_bank.import_problems(
            seeded,
            [{"id": "p1", "title": "题", "kp_ids": ["ds.array"], "test_cases": [{"expected": "1"}]}],
        )
        stats = problem_bank.coverage(seeded)

        assert "ds.array" not in stats["missing"]
        assert "algo.dp" in stats["missing"]
        assert stats["total_problems"] == 1
