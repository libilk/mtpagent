"""P1 worker 测试(work.md §7 P1.10):幂等、journal 先记后做、重启恢复、重试/死信。

不需要真 Redis:FakeBus 满足 Bus 协议。
"""

import pytest

from coach import config
from coach.coordination.journal import Journal
from coach.coordination.recovery import Recovery
from coach.domain.models import Problem
from coach.events import schema as events
from coach.workers.profile_worker import ProfileWorker


def answer_event(problem_id="lc.70", answer="8", attempt=1, learner_id="u1"):
    return events.new_event(
        events.ANSWER_SUBMITTED,
        learner_id,
        payload={"problem_id": problem_id, "answer_text": answer, "elapsed_ms": 120},
        attempt=attempt,
    )


@pytest.fixture
def worker(profile, knowledge, journal, fake_bus):
    return ProfileWorker(profile, knowledge, journal, bus=fake_bus)


class TestHandleAnswer:
    def test_correct_answer_raises_mastery(self, worker, profile, problem):
        result = worker.process(answer_event(answer="8"))

        assert result["status"] == "updated"
        assert result["correct"] is True
        assert profile.get_mastery("u1", "algo.dp") > config.BKT_P_INIT

    def test_wrong_answer_lowers_mastery_and_records_error(self, worker, profile, problem):
        profile.set_mastery("u1", "algo.dp", 0.9)

        result = worker.process(answer_event(answer="999"))

        assert result["correct"] is False
        assert profile.get_mastery("u1", "algo.dp") < 0.9
        assert profile.list_errors("u1")[0]["error_type"] == "wrong"

    def test_grading_ignores_surrounding_whitespace(self, worker, problem):
        assert worker.process(answer_event(answer="  8 \n"))["correct"] is True

    def test_unknown_problem_is_reported_not_fatal(self, worker):
        result = worker.process(answer_event(problem_id="nope"))
        assert result["status"] == "problem_not_found"

    def test_ungradeable_problem_is_reported(self, worker, knowledge):
        knowledge.upsert_problem(
            Problem(id="no-cases", title="无用例", judge_type="exact_output", test_cases=[])
        )
        result = worker.process(answer_event(problem_id="no-cases"))
        assert result["status"] == "ungradeable"

    def test_unknown_event_type_ignored(self, worker):
        event = events.new_event(events.TICK_SCHEDULED, "u1")
        assert worker.process(event)["status"] == "ignored"

    def test_profile_updated_event_is_published(self, worker, profile, problem, fake_bus):
        worker.process(answer_event())

        published = fake_bus.queues[config.STREAM_PROFILE]
        assert len(published) == 1
        assert published[0][1].type == events.PROFILE_UPDATED
        assert published[0][1].payload["kp_ids"] == ["algo.dp"]


class TestIdempotency:
    def test_same_event_processed_only_once(self, worker, profile, problem):
        event = answer_event()

        first = worker.process(event)
        mastery_after_first = profile.get_mastery("u1", "algo.dp")
        second = worker.process(event)

        assert first["status"] == "updated"
        assert second["status"] == "skipped"
        assert profile.get_mastery("u1", "algo.dp") == mastery_after_first
        assert len(profile.list_answers("u1")) == 1

    def test_answer_id_uniqueness_blocks_double_counting_on_replay(self, worker, profile, problem):
        """第二道闸:即使 processed_events 丢了(模拟恢复重放),画像也不会更新两次。"""
        event = answer_event()
        worker.handle(event)
        mastery_after_first = profile.get_mastery("u1", "algo.dp")

        replay = worker.handle(event)  # 绕过 processed_events 直接重放

        assert replay["status"] == "duplicate_answer"
        assert profile.get_mastery("u1", "algo.dp") == mastery_after_first


class TestAtomicity:
    """★ 一次答题的写操作必须全成或全不成(见 profile/store.py transaction)。"""

    @staticmethod
    def _crash_after_answer_insert(worker, monkeypatch):
        """模拟:答题记录已写、掌握度还没写时进程被杀。"""
        def boom(_learner_id, _kp_id, _memory):
            raise RuntimeError("模拟进程被杀")

        monkeypatch.setattr(worker, "_save_memory", boom)

    def test_partial_failure_rolls_back_everything(self, worker, profile, problem, monkeypatch):
        self._crash_after_answer_insert(worker, monkeypatch)

        with pytest.raises(RuntimeError):
            worker.process(answer_event())

        # 答题记录必须一起回滚,否则恢复重放会以为已经处理过而跳过
        assert profile.list_answers("u1") == []
        assert profile.get_memory("u1", "algo.dp") is None
        assert profile.get_mastery("u1", "algo.dp") == config.BKT_P_INIT

    def test_recovery_after_crash_applies_everything(self, worker, profile, journal, problem, monkeypatch):
        event = answer_event()
        self._crash_after_answer_insert(worker, monkeypatch)
        with pytest.raises(RuntimeError):
            worker.process(event)
        assert profile.get_mastery("u1", "algo.dp") == config.BKT_P_INIT

        # 进程重启:关掉故障,走正常恢复路径
        monkeypatch.undo()
        replayed = Recovery(journal).replay(worker.process)

        assert replayed == [event.event_id]
        assert profile.get_mastery("u1", "algo.dp") > config.BKT_P_INIT
        assert len(profile.list_answers("u1")) == 1
        assert profile.get_memory("u1", "algo.dp")["reps"] == 1

    def test_transaction_commits_on_success(self, profile):
        with profile.transaction():
            profile.set_mastery("u1", "a", 0.5)
            profile.set_mastery("u1", "b", 0.7)

        assert profile.get_mastery("u1", "a") == pytest.approx(0.5)
        assert profile.get_mastery("u1", "b") == pytest.approx(0.7)

    def test_nested_transaction_reuses_outer(self, profile):
        with profile.transaction():
            profile.set_mastery("u1", "a", 0.5)
            with profile.transaction():
                profile.set_mastery("u1", "b", 0.7)

        assert profile.get_mastery("u1", "a") == pytest.approx(0.5)
        assert profile.get_mastery("u1", "b") == pytest.approx(0.7)

    def test_rollback_leaves_no_trace(self, profile):
        with pytest.raises(RuntimeError):
            with profile.transaction():
                profile.set_mastery("u1", "a", 0.5)
                raise RuntimeError("boom")

        assert profile.get_mastery("u1", "a") == config.BKT_P_INIT


class TestJournal:
    def test_record_is_idempotent(self, journal):
        event = answer_event()
        assert journal.record(event) is True
        assert journal.record(event) is False

    def test_pending_lists_unfinished_events(self, journal):
        done = answer_event()
        pending = answer_event()
        journal.record(done)
        journal.record(pending)
        journal.mark_done(done.event_id)

        assert [e.event_id for e in journal.pending()] == [pending.event_id]
        assert journal.counts() == {"total": 2, "pending": 1, "done": 1}

    def test_mark_done_is_idempotent(self, journal):
        event = answer_event()
        journal.record(event)
        journal.mark_done(event.event_id, done_at=100.0)
        journal.mark_done(event.event_id, done_at=200.0)

        row = journal.conn.execute(
            "SELECT done_at FROM journal WHERE event_id = ?", (event.event_id,)
        ).fetchone()
        assert row["done_at"] == 100.0


class TestRecovery:
    def test_replays_event_recorded_before_crash(self, worker, journal, profile, problem):
        """先记后做:journal 有记录但业务没做,重启后必须补做。"""
        event = answer_event()
        journal.record(event)  # 记完了,进程在 ACK 前被杀
        assert journal.counts()["pending"] == 1

        replayed = Recovery(journal).replay(worker.process)

        assert replayed == [event.event_id]
        assert journal.counts()["pending"] == 0
        assert profile.get_mastery("u1", "algo.dp") > config.BKT_P_INIT

    def test_failing_handler_stays_pending(self, journal):
        event = answer_event()
        journal.record(event)

        def broken(_event):
            raise RuntimeError("boom")

        assert Recovery(journal).replay(broken) == []
        assert journal.counts()["pending"] == 1  # 不吞异常,下次接着补

    def test_recovery_is_safe_when_event_already_done(self, worker, journal, problem):
        event = answer_event()
        worker.process(event)
        # 业务已完成,journal 已标 done —— 重放不该再动画像
        assert Recovery(journal).replay(worker.process) == []


class TestRetryAndDLQ:
    @staticmethod
    def _boom(_event):
        raise RuntimeError("模拟业务失败")

    def test_failure_before_limit_is_republished(self, worker, journal, fake_bus, monkeypatch):
        monkeypatch.setattr(worker, "process", self._boom)
        event = answer_event(attempt=1)
        journal.record(event)  # 真实链路里 process 已先记 journal

        worker._process_with_retry("0-1", event)

        assert fake_bus.dlq == []
        republished = fake_bus.queues[config.STREAM_ANSWER]
        assert republished[-1][1].attempt == 2
        assert fake_bus.acked == [(config.STREAM_ANSWER, "0-1")]

    def test_failure_past_limit_goes_to_dlq(self, worker, journal, fake_bus, monkeypatch):
        monkeypatch.setattr(worker, "process", self._boom)
        event = answer_event(attempt=config.MAX_ATTEMPTS)
        journal.record(event)

        worker._process_with_retry("0-9", event)

        assert len(fake_bus.dlq) == 1
        assert fake_bus.queues[config.STREAM_ANSWER] == []
        # 进死信后标 done,避免被 recovery 反复重放
        assert journal.is_done(event.event_id) is True


class TestConsumerLoop:
    def test_run_consumes_published_events(self, worker, profile, problem, fake_bus):
        import asyncio

        async def scenario():
            fake_bus.publish(config.STREAM_ANSWER, answer_event())
            task = asyncio.create_task(worker.run(block_ms=1, count=10))
            await asyncio.sleep(0.05)
            worker.stop()
            await asyncio.wait_for(task, timeout=2.0)

        asyncio.run(scenario())
        assert profile.get_mastery("u1", "algo.dp") > config.BKT_P_INIT
