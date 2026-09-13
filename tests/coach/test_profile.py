"""P1/P3 画像测试(work.md §7 P1.10 / P3.5):BKT、SM-2、画像存取、幂等表。"""

import pytest

from coach import config
from coach.profile import bkt, sm2


class TestBKT:
    def test_correct_raises_mastery(self):
        p0 = config.BKT_P_INIT
        p1 = bkt.update(p0, True)
        assert p1 > p0

    def test_wrong_lowers_mastery(self):
        p0 = 0.8
        p1 = bkt.update(p0, False)
        assert p1 < p0
        assert p1 > 0.0

    def test_continuous_correct_is_monotonic(self):
        p = config.BKT_P_INIT
        history = [p]
        for _ in range(20):
            p = bkt.update(p, True)
            history.append(p)
        assert all(b >= a for a, b in zip(history, history[1:])), history

    def test_continuous_wrong_is_monotonic(self):
        p = 0.9
        history = [p]
        for _ in range(20):
            p = bkt.update(p, False)
            history.append(p)
        assert all(b <= a for a, b in zip(history, history[1:])), history

    def test_all_correct_saturates_near_one(self):
        """§5.3 公式在全对时快速收敛到 1.0(约 10 次就饱和)。

        这是照抄公式的必然结果,不是 bug;但它意味着 BKT 会过度自信,
        P5 做校准曲线时要留意(已记入 work.md §11)。
        """
        assert bkt.update_sequence(config.BKT_P_INIT, [True] * 5) > 0.99
        assert bkt.update_sequence(config.BKT_P_INIT, [True] * 50) == pytest.approx(1.0)

    def test_all_wrong_converges_to_nonzero_floor(self):
        """全错也不会掉到 0:存在一个约 0.17 的下界。"""
        p = bkt.update_sequence(0.9, [False] * 50)
        assert 0.1 < p < 0.3

    def test_none_starts_from_p_init(self):
        assert bkt.update(None, True) == pytest.approx(bkt.update(config.BKT_P_INIT, True))

    def test_stays_within_bounds(self):
        for start in (0.0, 0.5, 1.0):
            for correct in (True, False):
                p = bkt.update(start, correct)
                assert 0.0 <= p <= 1.0

    def test_update_many_does_not_mutate_input(self):
        mastery = {"a": 0.5}
        result = bkt.update_many(mastery, ["a", "b"], True)
        assert mastery == {"a": 0.5}
        assert set(result) == {"a", "b"}
        assert result["a"] > 0.5
        # b 之前没记录,按初始值起算
        assert result["b"] == pytest.approx(bkt.update(config.BKT_P_INIT, True))

    def test_predict_correct_increases_with_mastery(self):
        assert bkt.predict_correct(0.1) < bkt.predict_correct(0.9)
        assert 0.0 <= bkt.predict_correct(0.0) <= 1.0
        assert 0.0 <= bkt.predict_correct(1.0) <= 1.0


class TestSM2:
    NOW = 1_700_000_000.0
    DAY = config.DAY_SECONDS

    def test_first_pass_gives_one_day(self):
        state = sm2.review(None, quality=4, now=self.NOW)

        assert state.reps == 1
        assert state.interval_days == 1
        assert state.due_at == pytest.approx(self.NOW + self.DAY)

    def test_second_pass_gives_six_days(self):
        first = sm2.review(None, 4, self.NOW)
        second = sm2.review(first, 4, self.NOW)

        assert second.reps == 2
        assert second.interval_days == 6

    def test_third_pass_multiplies_by_ease(self):
        state = None
        for _ in range(3):
            state = sm2.review(state, 4, self.NOW)

        # q=4 时 EF 不漂移,保持 2.5 → 6 × 2.5 = 15
        assert state.ease == pytest.approx(2.5)
        assert state.interval_days == 15

    def test_failure_resets_reps_and_counts_lapse(self):
        state = sm2.review(None, 4, self.NOW)
        state = sm2.review(state, 4, self.NOW)
        failed = sm2.review(state, quality=1, now=self.NOW)

        assert failed.reps == 0
        assert failed.interval_days == 1
        assert failed.lapses == 1
        assert failed.due_at == pytest.approx(self.NOW + self.DAY)

    def test_failure_does_not_change_ease(self):
        state = sm2.review(None, 4, self.NOW)
        assert sm2.review(state, quality=1, now=self.NOW).ease == pytest.approx(state.ease)

    def test_perfect_recall_raises_ease(self):
        state = sm2.review(None, quality=5, now=self.NOW)
        assert state.ease == pytest.approx(2.6)

    def test_low_pass_lowers_ease(self):
        state = sm2.review(None, quality=3, now=self.NOW)
        assert state.ease == pytest.approx(2.36)

    def test_ease_never_drops_below_floor(self):
        state = None
        for _ in range(30):
            state = sm2.review(state, quality=3, now=self.NOW)
        assert state.ease == pytest.approx(config.SM2_MIN_EF)

    def test_quality_out_of_range_rejected(self):
        with pytest.raises(ValueError):
            sm2.review(None, quality=6, now=self.NOW)

    def test_quality_mapping(self):
        assert sm2.quality_from_correct(True) == config.SM2_QUALITY_CORRECT
        assert sm2.quality_from_correct(False) == config.SM2_QUALITY_WRONG
        assert sm2.quality_from_correct(False) < config.SM2_PASS_SCORE

    def test_from_row_handles_missing_row(self):
        state = sm2.SM2State.from_row(None)
        assert state.ease == config.SM2_DEFAULT_EF
        assert state.reps == 0

    def test_is_due(self):
        state = sm2.review(None, 4, self.NOW)
        assert sm2.is_due(state, self.NOW) is False
        assert sm2.is_due(state, self.NOW + 2 * self.DAY) is True
        assert sm2.is_due(None, self.NOW) is False


class TestProfileStore:
    def test_mastery_defaults_to_p_init(self, profile):
        assert profile.get_mastery("u1", "algo.dp") == config.BKT_P_INIT

    def test_set_and_get_mastery(self, profile):
        profile.set_mastery("u1", "algo.dp", 0.73)
        assert profile.get_mastery("u1", "algo.dp") == pytest.approx(0.73)

    def test_mastery_is_per_learner(self, profile):
        profile.set_mastery("u1", "algo.dp", 0.9)
        assert profile.get_mastery("u2", "algo.dp") == config.BKT_P_INIT

    def test_observed_kp_ids_distinguishes_no_record(self, profile):
        """★ `get_mastery` 对没记录的点返回 0.1,和"考过且确实弱"数值一样 ——
        `observed_kp_ids` 就是用来把这两者分开的。"""
        profile.set_mastery("u1", "a", config.BKT_P_INIT)

        observed = profile.observed_kp_ids("u1", ["a", "b"])

        assert observed == {"a"}
        assert profile.get_mastery("u1", "b") == config.BKT_P_INIT  # 数值一样

    def test_observed_kp_ids_without_filter(self, profile):
        profile.set_mastery("u1", "a", 0.1)
        profile.set_mastery("u1", "b", 0.2)

        assert profile.observed_kp_ids("u1") == {"a", "b"}

    def test_observed_kp_ids_empty_input(self, profile):
        assert profile.observed_kp_ids("u1", []) == set()

    def test_get_mastery_map_fills_missing_with_default(self, profile):
        profile.set_mastery("u1", "a", 0.4)
        result = profile.get_mastery_map("u1", ["a", "b"])
        assert result["a"] == pytest.approx(0.4)
        assert result["b"] == config.BKT_P_INIT

    def test_get_mastery_map_without_ids_returns_all(self, profile):
        profile.set_mastery("u1", "a", 0.4)
        profile.set_mastery("u1", "b", 0.6)
        assert set(profile.get_mastery_map("u1")) == {"a", "b"}

    def test_record_answer_is_idempotent_by_answer_id(self, profile):
        first = profile.record_answer("e1", "u1", "p1", ["a"], True, "8")
        second = profile.record_answer("e1", "u1", "p1", ["a"], True, "8")

        assert first is True
        assert second is False
        assert len(profile.list_answers("u1")) == 1

    def test_list_answers_parses_kp_ids(self, profile):
        profile.record_answer("e1", "u1", "p1", ["a", "b"], False, "wrong")
        answer = profile.list_answers("u1")[0]
        assert answer["kp_ids"] == ["a", "b"]
        assert answer["correct"] is False

    def test_processed_events_idempotency(self, profile):
        assert profile.mark_processed("e1") is True
        assert profile.mark_processed("e1") is False
        assert profile.is_processed("e1") is True
        assert profile.is_processed("e2") is False

    def test_due_reviews_only_returns_past_due(self, profile):
        profile.set_memory("u1", "due", ease=2.5, interval_days=1, reps=1, lapses=0,
                           last_review=0.0, due_at=100.0)
        profile.set_memory("u1", "later", ease=2.5, interval_days=1, reps=1, lapses=0,
                           last_review=0.0, due_at=10_000.0)

        assert profile.due_reviews("u1", now=1000.0) == ["due"]

    def test_bump_error_counts_up(self, profile):
        profile.bump_error("u1", "algo.dp", "wrong")
        profile.bump_error("u1", "algo.dp", "wrong")

        assert profile.list_errors("u1") == [
            {"kp_id": "algo.dp", "error_type": "wrong", "count": 2}
        ]
