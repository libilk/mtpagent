"""P5 评测层测试。

评测代码自己出错是最危险的 —— 它会让"数字"变成谎话。所以这一组重点验:
1. 指标函数的数学正确性(用已知答案的构造样本)
2. 模拟学生的确定性,以及"按掌握度概率答题"这个性质真的成立
3. 对照组基线的行为符合设计(而不是碰巧给出好看的数)
"""

import math

import pytest

from coach.evaluation import baselines, metrics
from coach.evaluation.simulated_student import StudentSimulator


class TestMetrics:
    def test_auc_perfect_separation(self):
        assert metrics.auc([0.1, 0.4, 0.6, 0.9], [False, False, True, True]) == pytest.approx(1.0)

    def test_auc_inverted(self):
        assert metrics.auc([0.9, 0.6, 0.4, 0.1], [False, False, True, True]) == pytest.approx(0.0)

    def test_auc_all_ties_is_half(self):
        assert metrics.auc([0.5, 0.5, 0.5, 0.5], [False, True, False, True]) == pytest.approx(0.5)

    def test_auc_single_class_is_nan(self):
        assert math.isnan(metrics.auc([0.1, 0.9], [True, True]))

    def test_auc_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            metrics.auc([0.1], [True, False])

    def test_brier(self):
        assert metrics.brier([1.0, 0.0], [True, False]) == pytest.approx(0.0)
        assert metrics.brier([0.0, 1.0], [True, False]) == pytest.approx(1.0)
        assert metrics.brier([0.5, 0.5], [True, False]) == pytest.approx(0.25)

    def test_log_loss_perfect_is_near_zero(self):
        assert metrics.log_loss([1.0, 0.0], [True, False]) < 1e-6

    def test_calibration_curve_bins(self):
        curve = metrics.calibration_curve([0.05, 0.95], [False, True], bins=10)

        assert len(curve) == 2
        assert curve[0]["observed_rate"] == 0.0
        assert curve[1]["observed_rate"] == 1.0
        assert curve[0]["gap"] == pytest.approx(-0.05)

    def test_ece_zero_when_perfectly_calibrated(self):
        # 预测 1.0 且全对、预测 0.0 且全错 → 无误差
        scores = [1.0] * 5 + [0.0] * 5
        labels = [True] * 5 + [False] * 5
        curve = metrics.calibration_curve(scores, labels, bins=10)
        assert metrics.expected_calibration_error(curve, len(labels)) == pytest.approx(0.0)

    def test_top1_and_recall(self):
        predicted = [["a", "b"], ["c", "a"], ["x"]]
        truth = ["a", "a", "a"]

        assert metrics.top1_accuracy(predicted, truth) == pytest.approx(1 / 3)
        assert metrics.recall_at_k(predicted, truth) == pytest.approx(2 / 3)

    def test_top1_empty_candidates_counts_as_miss(self):
        assert metrics.top1_accuracy([[]], ["a"]) == pytest.approx(0.0)


class TestSimulatedStudent:
    @pytest.fixture
    def seeded_graph(self, knowledge):
        from coach.knowledge import builder
        from coach.knowledge.governance import Governance

        builder.seed_graph(Governance(knowledge))
        builder.seed_problems(knowledge)
        return knowledge

    def test_deterministic_by_seed(self, seeded_graph):
        a = StudentSimulator(seeded_graph, seed=7).make_student("x")
        b = StudentSimulator(seeded_graph, seed=7).make_student("x")

        assert a.ability == b.ability

    def test_different_seeds_differ(self, seeded_graph):
        a = StudentSimulator(seeded_graph, seed=1).make_student("x")
        b = StudentSimulator(seeded_graph, seed=2).make_student("x")

        assert a.ability != b.ability

    def test_plants_weakness(self, seeded_graph):
        sim = StudentSimulator(seeded_graph, seed=0)
        student = sim.make_student("x", weakness="algo.recursion", weakness_value=0.12)

        assert student.own("algo.recursion") == pytest.approx(0.12)
        assert student.planted_weakness == "algo.recursion"

    @staticmethod
    def _set_closure(student, graph, kp_id, value):
        """连前置的递归影响一起设好 —— 只设直接前置是不够的,
        它们自己被更深的前置拖累,回传上来照样把目标拉低。"""
        student.ability[kp_id] = value
        for ancestor in graph.ancestors(kp_id, depth=4):
            student.ability[ancestor] = value

    def test_prerequisite_drags_effective_ability(self, seeded_graph):
        """★ 这是模拟学生与 BKT 的关键分歧:前置弱会拖累下游。"""
        sim = StudentSimulator(seeded_graph, seed=0)
        student = sim.make_student("x")
        self._set_closure(student, seeded_graph, "algo.dp", 0.9)
        # 只把最深的基础打烂
        student.ability["prog.func_call"] = 0.05

        assert sim.effective(student, "algo.dp") < 0.6

    def test_no_prereq_means_effective_equals_own(self, seeded_graph):
        sim = StudentSimulator(seeded_graph, seed=0)
        student = sim.make_student("x")
        student.ability["prog.func_call"] = 0.8

        assert sim.effective(student, "prog.func_call") == pytest.approx(0.8)

    def test_strong_prereqs_keep_effective_high(self, seeded_graph):
        sim = StudentSimulator(seeded_graph, seed=0)
        student = sim.make_student("x")
        self._set_closure(student, seeded_graph, "algo.dp", 0.95)

        assert sim.effective(student, "algo.dp") > 0.8

    def test_practice_raises_ability_with_diminishing_returns(self, seeded_graph):
        sim = StudentSimulator(seeded_graph, seed=0)
        student = sim.make_student("x")
        student.ability["algo.dp"] = 0.2

        sim.practice(student, "algo.dp")
        first_gain = student.own("algo.dp") - 0.2
        sim.practice(student, "algo.dp")
        second_gain = student.own("algo.dp") - 0.2 - first_gain

        assert first_gain > 0
        assert second_gain < first_gain
        assert student.own("algo.dp") <= 1.0

    def test_forget_lowers_ability_after_idle_days(self, seeded_graph):
        sim = StudentSimulator(seeded_graph, seed=0)
        student = sim.make_student("x")
        student.ability["algo.dp"] = 0.9
        sim.mark_practiced(student, "algo.dp", day=0)

        sim.forget(student, day=30)

        assert student.own("algo.dp") < 0.3

    def test_forget_is_noop_without_practice_record(self, seeded_graph):
        """没记过练习时间就不该掉 —— 曾经在这里踩过坑,遗忘整段失效。"""
        sim = StudentSimulator(seeded_graph, seed=0)
        student = sim.make_student("x")
        student.ability["algo.dp"] = 0.9

        sim.forget(student, day=30)

        assert student.own("algo.dp") == pytest.approx(0.9)

    def test_answer_probability_tracks_ability(self, seeded_graph):
        """掌握度高的人答对率应该显著更高 —— 否则整个评测就没有意义。"""
        sim = StudentSimulator(seeded_graph, seed=3)
        problem = seeded_graph.get_problem("lc.70")

        strong = sim.make_student("strong")
        weak = sim.make_student("weak")
        for kp in problem.kp_ids:
            self._set_closure(strong, seeded_graph, kp, 0.95)
            self._set_closure(weak, seeded_graph, kp, 0.05)

        strong_hits = sum(sim.answer(strong, problem) for _ in range(200))
        weak_hits = sum(sim.answer(weak, problem) for _ in range(200))

        assert strong_hits > 170, strong_hits
        assert weak_hits < 30, weak_hits


class TestFlatBaseline:
    @pytest.fixture
    def seeded_graph(self, knowledge):
        from coach.knowledge import builder
        from coach.knowledge.governance import Governance

        builder.seed_graph(Governance(knowledge))
        return knowledge

    def test_similar_ranks_self_first(self, seeded_graph):
        flat = baselines.FlatRetrievalBaseline(seeded_graph)
        assert flat.similar("algo.dp", top_k=3)[0] == "algo.dp"

    def test_similar_can_exclude_self(self, seeded_graph):
        flat = baselines.FlatRetrievalBaseline(seeded_graph)
        assert "algo.dp" not in flat.similar("algo.dp", top_k=5, include_self=False)

    def test_root_causes_respects_threshold(self, seeded_graph):
        flat = baselines.FlatRetrievalBaseline(seeded_graph)
        mastery = {c.id: 0.9 for c in seeded_graph.list_concepts()}

        assert flat.root_causes(mastery, "algo.dp") == []

    def test_root_causes_skips_unobserved_when_told(self, seeded_graph):
        """★ 不把"未知"当成"最弱" —— 否则基线会去挑一堆毫无证据的点。"""
        flat = baselines.FlatRetrievalBaseline(seeded_graph)
        mastery = {"algo.greedy": 0.1}  # 只有这一个有观测值

        with_filter = flat.root_causes(mastery, "algo.dp", observed={"algo.greedy"})
        without_filter = flat.root_causes(mastery, "algo.dp")

        assert [r["kp_id"] for r in with_filter] == ["algo.greedy"]
        # 不过滤时,一堆缺失值(0.0)会挤到前面
        assert len(without_filter) >= 1

    def test_root_causes_sorted_by_mastery_ascending(self, seeded_graph):
        flat = baselines.FlatRetrievalBaseline(seeded_graph)
        mastery = {"algo.greedy": 0.3, "algo.bfs": 0.1}

        result = flat.root_causes(mastery, "algo.dp", observed={"algo.greedy", "algo.bfs"})

        assert [r["mastery"] for r in result] == sorted(r["mastery"] for r in result)


@pytest.fixture(scope="module")
def report():
    """小规模跑通全流程。跑一次给整个模块复用(全流程不便宜)。"""
    from coach.evaluation.run_eval import run_all

    return run_all(n_students=2, seed=0, plan_students=2, plan_steps=3, root_students=2)


class TestRunEvalIntegration:
    """保证四张表都能产出、数字落在合法区间。"""

    def test_all_four_tables_present(self, report):
        assert set(report) >= {
            "mastery_prediction",
            "forgetting",
            "planning_gain",
            "root_cause_full_coverage",
            "root_cause_sparse_coverage",
        }

    def test_mastery_metrics_in_range(self, report):
        table = report["mastery_prediction"]
        assert 0.0 <= table["auc"] <= 1.0
        assert 0.0 <= table["brier"] <= 1.0
        assert table["n_predictions"] > 0

    def test_root_cause_accuracies_in_range(self, report):
        for key in ("root_cause_full_coverage", "root_cause_sparse_coverage"):
            table = report[key]
            assert 0.0 <= table["graph_top1_accuracy"] <= 1.0
            assert 0.0 <= table["flat_top1_accuracy"] <= 1.0
            assert table["n_cases"] > 0

    def test_planning_reports_three_arms(self, report):
        table = report["planning_gain"]
        for arm in ("planned", "planned_practicable", "random"):
            assert arm in table
            assert "wasted_step_ratio" in table[arm]

    def test_markdown_renders(self, report):
        from coach.evaluation.run_eval import render_markdown

        markdown = render_markdown(report)
        assert "根因定位" in markdown
        assert "扁平召回" in markdown

    def test_no_agent_arm_by_default(self, report):
        """默认不跑 agent 臂 —— 确定性那四张表的产出必须和以前一模一样。"""
        assert "root_cause_agent" not in report

        from coach.evaluation.run_eval import render_markdown

        assert "表 5" not in render_markdown(report)


class SubmittingLLM:
    """假 LLM:第一轮就交卷。不联网、不花钱,只用来验证接线。"""

    def __init__(self, kp_id="algo.dp"):
        self.kp_id = kp_id
        self.calls = 0

    def chat_structured(self, messages, tools=None, **kwargs):
        from llm.llm_client import LLMResponse, ToolCall

        self.calls += 1
        return LLMResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id=f"c{self.calls}",
                    name="submit_diagnosis",
                    arguments={"root_causes": [{"kp_id": self.kp_id}], "summary": "假的"},
                )
            ],
        )


def _small_run(**overrides):
    from coach.evaluation.run_eval import run_all

    base = dict(n_students=1, seed=0, plan_students=1, plan_steps=1, root_students=1)
    return run_all(**{**base, **overrides})


@pytest.fixture(scope="module")
def agent_report():
    return _small_run(with_agent=True, agent_students=1, llm=SubmittingLLM())


@pytest.fixture(scope="module")
def agent_skipped_report():
    """要求了 --agent 但没有 LLM —— 这一臂必须如实记成「未运行」,不许编数字。"""
    return _small_run(with_agent=True, agent_students=1, llm=None)


class TestAgentArm:
    def test_skipped_honestly_when_no_llm(self, agent_skipped_report):
        block = agent_skipped_report["root_cause_agent"]
        assert block["skipped"] is True
        assert "未运行" in block["reason"]
        assert "agent_top1_accuracy" not in str(block)

    def test_skipped_markdown_says_not_run(self, agent_skipped_report):
        from coach.evaluation.run_eval import render_markdown

        markdown = render_markdown(agent_skipped_report)
        assert "**未运行**" in markdown
        assert "表 5 的 agent 臂" not in markdown, "没跑就不该出现那条限制说明"

    def test_runs_and_produces_scores(self, agent_report):
        block = agent_report["root_cause_agent"]
        assert "skipped" not in block
        for key in ("full", "sparse"):
            arm = block[key]
            assert 0.0 <= arm["agent_top1_accuracy"] <= 1.0
            assert 0.0 <= arm["agent_recall_at_k"] <= 1.0

    def test_all_five_arms_share_the_same_sample(self, agent_report):
        """★ agent 臂单独跑小样本,但**五个臂必须同一个 n** —— 否则对照不成立。"""
        for key in ("full", "sparse"):
            arm = agent_report["root_cause_agent"][key]
            assert arm["agent"]["cases"] == arm["n_cases"]
            assert arm["agent"]["failures"] == 0

    def test_records_cost_metrics(self, agent_report):
        """步数/工具调用是确定性臂恒为 0 的东西,必须一起报出来。"""
        meta = agent_report["root_cause_agent"]["full"]["agent"]
        assert meta["mean_steps"] >= 1
        assert meta["mean_tool_calls"] >= 1  # 至少调了 submit_diagnosis
        assert meta["terminations"] == {"submitted": meta["cases"]}

    def test_markdown_renders_agent_table_and_limitation(self, agent_report):
        from coach.evaluation.run_eval import render_markdown

        markdown = render_markdown(agent_report)
        assert "表 5 agent 诊断" in markdown
        assert "同样本对照" in markdown
        assert "表 5 的 agent 臂" in markdown, "跑了就该带出那条限制说明"
        assert "不可复现" in markdown

    def test_single_failure_does_not_kill_the_run(self, knowledge):
        """单例失败只计数,不让整轮评测崩 —— 评测崩了比数字难看严重得多。"""
        from coach.evaluation.run_eval import AgentPredictor
        from coach.knowledge import builder
        from coach.knowledge.governance import Governance

        builder.seed_graph(Governance(knowledge))

        class ExplodingLLM:
            def chat_structured(self, *args, **kwargs):
                raise RuntimeError("模拟接口挂了")

        predictor = AgentPredictor(ExplodingLLM(), tag="t")
        assert predictor.predict(knowledge, "u1", "algo.dp", {}, set()) == []

        summary = predictor.summary()
        assert summary["failures"] == 1
        assert summary["cases"] == 0
        assert summary["mean_steps"] is None
