"""P4 答题链测试(work.md §7 P4.4)。

核心验收:**中途杀进程,恢复后不重跑已经完成的步骤**,尤其是调 LLM 那一步。
断言用「调用计数」——不是看日志,是数 LLM 被调了几次。

    链:START → grade → update_profile → detect_gap ─┬→ explain → finalize → END
                                                     └──────────→ finalize → END
"""

import pytest

from coach import config
from coach.events import schema as events
from coach.workflow.pipeline import AnswerPipeline, make_checkpointer

LLM_REPLY = "先把函数调用补上。"


class FakeLLM:
    def __init__(self):
        self.calls = []

    def chat(self, messages, **_kwargs):
        self.calls.append(messages)
        return LLM_REPLY


def make_pipeline(knowledge, profile, journal, bus=None, llm=None, checkpointer=None):
    return AnswerPipeline(
        knowledge, profile, journal, llm=llm, bus=bus, checkpointer=checkpointer
    )


def answer_event(problem_id="lc.70", text="8", learner_id="u1"):
    return events.new_event(
        events.ANSWER_SUBMITTED,
        learner_id,
        payload={"problem_id": problem_id, "answer_text": text, "elapsed_ms": 120},
    )


@pytest.fixture
def seeded_problems(knowledge, seeded):
    from coach.knowledge import builder

    builder.seed_problems(knowledge)
    return knowledge


# ---------------------------------------------------------------- 验收:P4.4

class TestCheckpointResume:
    def test_crash_after_explain_does_not_recall_llm(
        self, seeded_problems, profile, journal, monkeypatch
    ):
        """★ P4 验收线:explain 已经跑完、finalize 崩了 → 恢复不重调 LLM。"""
        llm = FakeLLM()
        pipeline = make_pipeline(
            seeded_problems, profile, journal, llm=llm, checkpointer=make_checkpointer()
        )
        event = answer_event()

        # 让 finalize 崩掉(它内部会调 mark_processed)
        def boom(*_args, **_kwargs):
            raise RuntimeError("模拟进程被杀")

        monkeypatch.setattr(profile, "mark_processed", boom)
        with pytest.raises(RuntimeError):
            pipeline.run(event)

        assert len(llm.calls) == 2, "lc.70 挂在两个知识点上,应各生成一次解释"

        # 进程重启:故障排除,用同一个 thread_id 再跑一次
        monkeypatch.undo()
        result = pipeline.run(event)

        assert result["status"] == "updated"
        assert len(llm.calls) == 2, "★ explain 已完成,恢复时不该重调 LLM"
        assert profile.is_processed(event.event_id) is True

    def test_crash_before_explain_reruns_only_that_step(
        self, seeded_problems, profile, journal, monkeypatch
    ):
        """崩在 update_profile → 恢复从那里接着跑,后面正常完成。"""
        llm = FakeLLM()
        pipeline = make_pipeline(
            seeded_problems, profile, journal, llm=llm, checkpointer=make_checkpointer()
        )
        event = answer_event()

        def boom(*_args, **_kwargs):
            raise RuntimeError("模拟进程被杀")

        monkeypatch.setattr(profile, "set_memory", boom)
        with pytest.raises(RuntimeError):
            pipeline.run(event)
        monkeypatch.undo()

        assert profile.list_answers("u1") == [], "事务应整体回滚"

        result = pipeline.run(event)

        assert result["status"] == "updated"
        assert len(llm.calls) == 2  # explain 这次才第一次跑
        assert profile.get_mastery("u1", "algo.dp") > config.BKT_P_INIT

    def test_checkpoint_records_next_step_on_crash(
        self, seeded_problems, profile, journal, monkeypatch
    ):
        """P4.3 交付:检查点确实落盘,并且记下了「下一步该跑谁」。"""
        llm = FakeLLM()
        pipeline = make_pipeline(
            seeded_problems, profile, journal, llm=llm, checkpointer=make_checkpointer()
        )
        event = answer_event()

        def boom(*_args, **_kwargs):
            raise RuntimeError("模拟进程被杀")

        monkeypatch.setattr(profile, "mark_processed", boom)
        with pytest.raises(RuntimeError):
            pipeline.run(event)
        monkeypatch.undo()

        snapshot = pipeline.graph.get_state(
            {"configurable": {"thread_id": event.event_id}}
        )

        assert snapshot.next == ("finalize",)
        assert snapshot.values["status"] == "updated"
        assert snapshot.values["explained"] == ["algo.dp", "algo.recursion"]

    def test_completed_thread_restarts_and_answers_gate_catches_it(
        self, seeded_problems, profile, journal
    ):
        """跑完的线程再跑一次:没有待跑节点,于是重头跑 —— 靠 answers 唯一键兜底。"""
        pipeline = make_pipeline(seeded_problems, profile, journal, checkpointer=make_checkpointer())
        event = answer_event()
        pipeline.run(event)
        mastery_after_first = profile.get_mastery("u1", "algo.dp")

        profile.conn.execute(
            "DELETE FROM processed_events WHERE event_id = ?", (event.event_id,)
        )
        profile.conn.commit()
        second = pipeline.run(event)

        assert second["status"] == "duplicate_answer"
        assert profile.get_mastery("u1", "algo.dp") == mastery_after_first
        assert len(profile.list_answers("u1")) == 1

    def test_without_checkpointer_replay_falls_back_to_answers_gate(
        self, seeded_problems, profile, journal
    ):
        pipeline = make_pipeline(seeded_problems, profile, journal, checkpointer=None)
        event = answer_event()
        pipeline.run(event)
        mastery_after_first = profile.get_mastery("u1", "algo.dp")

        profile.conn.execute(
            "DELETE FROM processed_events WHERE event_id = ?", (event.event_id,)
        )
        profile.conn.commit()
        second = pipeline.run(event)

        assert second["status"] == "duplicate_answer"
        assert profile.get_mastery("u1", "algo.dp") == mastery_after_first

    def test_sqlite_checkpointer_survives_new_pipeline_instance(
        self, seeded_problems, profile, journal, tmp_path, monkeypatch
    ):
        """生产用的 SqliteSaver:换个进程(新的 pipeline 实例)也能接着跑。"""
        cp_path = tmp_path / "checkpoints.db"
        event = answer_event()

        pipeline = make_pipeline(
            seeded_problems, profile, journal, llm=FakeLLM(),
            checkpointer=make_checkpointer(cp_path),
        )

        def boom(*_args, **_kwargs):
            raise RuntimeError("模拟进程被杀")

        monkeypatch.setattr(profile, "mark_processed", boom)
        with pytest.raises(RuntimeError):
            pipeline.run(event)
        monkeypatch.undo()

        # 模拟进程重启:新建一个 pipeline,连同一个检查点库
        llm2 = FakeLLM()
        revived = make_pipeline(
            seeded_problems, profile, journal, llm=llm2,
            checkpointer=make_checkpointer(cp_path),
        )
        result = revived.run(event)

        assert result["status"] == "updated"
        assert llm2.calls == [], "explain 上次已经跑完,重启后不该重调 LLM"
        assert profile.is_processed(event.event_id) is True


# ---------------------------------------------------------------- 节点行为

class TestNodes:
    def test_unknown_problem_short_circuits(self, knowledge, profile, journal):
        pipeline = make_pipeline(knowledge, profile, journal)

        result = pipeline.run(answer_event(problem_id="nope"))

        assert result["status"] == "problem_not_found"
        assert profile.list_answers("u1") == []

    def test_ungradeable_problem_short_circuits(self, knowledge, profile, journal):
        from coach.domain.models import Problem

        knowledge.upsert_problem(
            Problem(id="no-cases", title="无用例", test_cases=[])
        )
        pipeline = make_pipeline(knowledge, profile, journal)

        result = pipeline.run(answer_event(problem_id="no-cases"))

        assert result["status"] == "ungradeable"

    def test_grade_ignores_surrounding_whitespace(self, seeded_problems, profile, journal):
        pipeline = make_pipeline(seeded_problems, profile, journal)
        assert pipeline.run(answer_event(text="  8 \n"))["correct"] is True

    def test_wrong_answer_records_lapse(self, seeded_problems, profile, journal):
        pipeline = make_pipeline(seeded_problems, profile, journal)

        pipeline.run(answer_event(text="999"))

        memory = profile.get_memory("u1", "algo.dp")
        assert memory["lapses"] == 1
        assert memory["reps"] == 0
        assert profile.list_errors("u1")[0]["error_type"] == "wrong"

    def test_publishes_profile_updated(self, seeded_problems, profile, journal, fake_bus):
        pipeline = make_pipeline(seeded_problems, profile, journal, bus=fake_bus)

        pipeline.run(answer_event())

        queue = fake_bus.queues[config.STREAM_PROFILE]
        assert len(queue) == 1
        _, event = queue[0]
        assert event.type == events.PROFILE_UPDATED
        assert event.payload["kp_ids"] == ["algo.dp", "algo.recursion"]
        assert "since" in event.payload

    def test_summary_shape(self, seeded_problems, profile, journal):
        pipeline = make_pipeline(seeded_problems, profile, journal)

        result = pipeline.run(answer_event())

        assert set(result) == {
            "status", "correct", "reason", "mastery", "due_at", "gaps", "explained",
        }
        assert result["gaps"]  # 默认掌握度都是初始值,应该有缺口

    def test_marks_journal_done(self, seeded_problems, profile, journal):
        pipeline = make_pipeline(seeded_problems, profile, journal)
        event = answer_event()

        pipeline.run(event)

        assert journal.is_done(event.event_id) is True
        assert journal.counts()["pending"] == 0


class TestConditionalRouting:
    def test_no_gaps_skips_explain_node(self, seeded_problems, profile, journal):
        """前置都掌握了 → 条件边直接去 finalize,LLM 一次都不调。"""
        for kp_id in ("algo.recursion", "ds.array", "algo.memoization", "prog.func_call"):
            profile.set_mastery("u1", kp_id, 0.9)
        llm = FakeLLM()
        pipeline = make_pipeline(seeded_problems, profile, journal, llm=llm)

        result = pipeline.run(answer_event())

        assert result["gaps"] == []
        assert result["explained"] == []
        assert llm.calls == []

    def test_gaps_route_to_explain(self, seeded_problems, profile, journal):
        llm = FakeLLM()
        pipeline = make_pipeline(seeded_problems, profile, journal, llm=llm)

        result = pipeline.run(answer_event())

        assert result["gaps"]
        assert result["explained"] == ["algo.dp", "algo.recursion"]
        assert len(llm.calls) == 2

    def test_no_llm_configured_still_completes(self, seeded_problems, profile, journal, fake_bus):
        """没配 LLM 时不该崩:跳过解释,其余照常。"""
        pipeline = make_pipeline(seeded_problems, profile, journal, llm=None, bus=fake_bus)

        result = pipeline.run(answer_event())

        assert result["status"] == "updated"
        assert result["explained"] == []


class TestSharedExplanation:
    def test_planner_skips_when_cache_is_fresh(self, seeded_problems, profile, journal):
        """答题链刚写过解释 → planner 不该再调一次 LLM。"""
        from coach.workers.planner_worker import PlannerWorker
        from coach.workflow.query import QueryService

        service = QueryService(seeded_problems, profile)
        llm = FakeLLM()
        pipeline = make_pipeline(seeded_problems, profile, journal, llm=llm)
        event = answer_event()
        pipeline.run(event)

        profile.conn.execute(
            "DELETE FROM processed_events WHERE event_id = ?", (event.event_id,)
        )
        profile.conn.commit()

        came_from_pipeline = len(llm.calls)
        planner = PlannerWorker(
            seeded_problems, profile, journal, llm=llm, service=service
        )
        # 模拟 pipeline 发出来的 profile.updated(带 since)
        profile_event = events.new_event(
            events.PROFILE_UPDATED,
            "u1",
            payload={"kp_ids": ["algo.dp"], "since": event.ts},
        )
        result = planner.process(profile_event)

        assert result["refreshed"] == []
        assert len(llm.calls) == came_from_pipeline, "planner 不该重复调 LLM"

    def test_planner_regenerates_when_cache_is_stale(self, seeded_problems, profile, journal):
        from coach.workers.planner_worker import PlannerWorker
        from coach.workflow.query import QueryService

        service = QueryService(seeded_problems, profile)
        llm = FakeLLM()
        planner = PlannerWorker(seeded_problems, profile, journal, llm=llm, service=service)

        # since 给一个很晚的时间 → 现有缓存都算过期
        profile_event = events.new_event(
            events.PROFILE_UPDATED,
            "u1",
            payload={"kp_ids": ["algo.dp"], "since": 9_999_999_999.0},
        )
        result = planner.process(profile_event)

        assert result["refreshed"] == ["algo.dp"]
        assert len(llm.calls) == 1
