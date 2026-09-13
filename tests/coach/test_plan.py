"""P3 计划与调度测试(work.md §7 P3.5):到期知识点必须排进计划。

验收线:把某知识点的 due_at 设成过去,调用 /plan 能看到它排进「复习」。
"""

import pytest

from coach import config
from coach.events import schema as events
from coach.workflow.query import QueryService
from coach.workers.scheduler_worker import SchedulerWorker

NOW = 1_700_000_000.0
DAY = config.DAY_SECONDS


@pytest.fixture
def service(knowledge, profile, seeded) -> QueryService:
    return QueryService(knowledge, profile)


@pytest.fixture
def scheduler(knowledge, profile, journal, fake_bus, service) -> SchedulerWorker:
    return SchedulerWorker(knowledge, profile, journal, bus=fake_bus, service=service)


def make_due(profile, learner_id, kp_id, due_at):
    profile.set_memory(
        learner_id, kp_id, ease=2.5, interval_days=1, reps=1, lapses=0,
        last_review=due_at - DAY, due_at=due_at,
    )


def tick_event(learner_id="u1"):
    return events.new_event(events.TICK_SCHEDULED, learner_id, payload={})


class TestDueReviewArrivesInPlan:
    """★ P3 验收线。"""

    def test_overdue_kp_appears_as_review(self, service, profile, seeded):
        make_due(profile, "u1", "ds.array", NOW - DAY)

        plan = service.plan_view("u1", now=NOW)

        reviews = [i for i in plan["items"] if i["action"] == "review"]
        assert [i["kp_id"] for i in reviews] == ["ds.array"]
        assert "到期复习" in reviews[0]["reason"]

    def test_future_due_kp_not_in_plan(self, service, profile, seeded):
        make_due(profile, "u1", "ds.array", NOW + 10 * DAY)

        plan = service.plan_view("u1", now=NOW)

        assert [i for i in plan["items"] if i["action"] == "review"] == []

    def test_review_sorted_by_most_overdue_first(self, service, profile, seeded):
        make_due(profile, "u1", "ds.array", NOW - 1 * DAY)
        make_due(profile, "u1", "ds.stack", NOW - 5 * DAY)

        plan = service.plan_view("u1", now=NOW)

        reviews = [i["kp_id"] for i in plan["items"] if i["action"] == "review"]
        assert reviews == ["ds.stack", "ds.array"]


class TestPlanComposition:
    def test_remedial_and_learn_items_from_goal(self, service, profile, seeded):
        profile.upsert_profile("u1", goal_kp_id="algo.dp", daily_minutes=600)
        profile.set_mastery("u1", "prog.func_call", 0.1)

        plan = service.plan_view("u1", now=NOW)

        actions = {i["action"] for i in plan["items"]}
        assert "remedial" in actions
        assert plan["goal_kp_id"] == "algo.dp"

    def test_no_goal_means_only_reviews(self, service, profile, seeded):
        make_due(profile, "u1", "ds.array", NOW - DAY)

        plan = service.plan_view("u1", now=NOW)

        assert {i["action"] for i in plan["items"]} == {"review"}

    def test_same_kp_never_appears_twice(self, service, profile, seeded):
        profile.upsert_profile("u1", goal_kp_id="algo.dp", daily_minutes=600)
        # 既是到期复习,又是目标的前置缺口
        make_due(profile, "u1", "algo.recursion", NOW - DAY)
        profile.set_mastery("u1", "algo.recursion", 0.1)

        plan = service.plan_view("u1", now=NOW)

        ids = [i["kp_id"] for i in plan["items"]]
        assert len(ids) == len(set(ids))
        # 复习优先级更高
        assert next(i for i in plan["items"] if i["kp_id"] == "algo.recursion")[
            "action"
        ] == "review"

    def test_budget_shortage_still_surfaces_top_item(self, service, profile, seeded):
        """预算是软的:再紧也至少给最紧要的一项,不返回空计划。"""
        profile.upsert_profile("u1", goal_kp_id="algo.dp", daily_minutes=5)
        profile.set_mastery("u1", "prog.func_call", 0.1)

        plan = service.plan_view("u1", now=NOW)

        assert len(plan["items"]) == 1

    def test_budget_caps_item_count(self, service, profile, seeded):
        profile.upsert_profile("u1", goal_kp_id="algo.dp", daily_minutes=60)
        profile.set_mastery("u1", "prog.func_call", 0.1)

        plan = service.plan_view("u1", now=NOW)

        assert len(plan["items"]) > 1
        assert plan["planned_minutes"] <= 60

    def test_daily_minutes_override(self, service, profile, seeded):
        profile.upsert_profile("u1", goal_kp_id="algo.dp", daily_minutes=30)

        plan = service.plan_view("u1", now=NOW, daily_minutes=120)

        assert plan["daily_minutes"] == 120

    def test_plan_has_date_string(self, service, profile, seeded):
        plan = service.plan_view("u1", now=NOW)
        assert len(plan["date"]) == 10 and plan["date"].count("-") == 2


class TestSchedulerWorker:
    def test_tick_publishes_plan_updated(self, scheduler, profile, seeded, fake_bus):
        make_due(profile, "u1", "ds.array", NOW - DAY)

        result = scheduler.process(tick_event())

        assert result["status"] == "ok"
        assert result["due"] == ["ds.array"]
        published = fake_bus.queues[config.STREAM_PLAN]
        assert len(published) == 1
        assert published[0][1].type == events.PLAN_UPDATED

    def test_same_tick_processed_once(self, scheduler, seeded):
        event = tick_event()
        scheduler.process(event)
        assert scheduler.process(event)["status"] == "skipped"

    def test_unrelated_event_ignored(self, scheduler, seeded):
        event = events.new_event(events.ANSWER_SUBMITTED, "u1")
        assert scheduler.process(event)["status"] == "ignored"

    def test_publish_failure_does_not_break_tick(self, scheduler, seeded):
        class BrokenBus:
            def publish(self, *_args, **_kwargs):
                raise RuntimeError("redis down")

        scheduler.bus = BrokenBus()
        assert scheduler.process(tick_event())["status"] == "ok"


class TestWorkerAdvancesDueDate:
    def test_answering_sets_due_at_in_the_future(self, worker, profile, problem):
        worker.process(
            events.new_event(
                events.ANSWER_SUBMITTED,
                "u1",
                payload={"problem_id": "lc.70", "answer_text": "8", "elapsed_ms": 100},
            )
        )

        memory = profile.get_memory("u1", "algo.dp")
        assert memory["reps"] == 1
        assert memory["interval_days"] == 1
        assert memory["due_at"] > memory["last_review"]

    def test_wrong_answer_records_a_lapse(self, worker, profile, problem):
        for answer in ("8", "8", "999"):
            worker.process(
                events.new_event(
                    events.ANSWER_SUBMITTED,
                    "u1",
                    payload={"problem_id": "lc.70", "answer_text": answer, "elapsed_ms": 100},
                )
            )

        memory = profile.get_memory("u1", "algo.dp")
        assert memory["lapses"] == 1
        assert memory["reps"] == 0


class TestPlanOnlyRecommendsPracticable:
    """#2:推荐必须能落地 —— 推一个没有题的知识点,学生无从练起。

    P5 实测:不加这道过滤时,计划 360 步里约 300 步(82%)是白费的。
    """

    def test_skips_kp_without_problems(self, service, profile, knowledge, seeded):
        from coach.domain.models import KnowledgePoint

        # 造一个没有题的知识点,并让它成为目标的唯一根因
        knowledge.upsert_concept(
            KnowledgePoint(id="algo.orphan", name="孤儿点", subject="algorithms")
        )
        profile.upsert_profile("u1", goal_kp_id="algo.dp", daily_minutes=600)
        profile.set_mastery("u1", "algo.orphan", 0.05)

        plan = service.plan_view("u1", now=NOW)

        assert "algo.orphan" not in [i["kp_id"] for i in plan["items"]]

    def test_keeps_kp_with_problems(self, service, profile, seeded):
        profile.upsert_profile("u1", goal_kp_id="algo.dp", daily_minutes=600)
        profile.set_mastery("u1", "ds.array", 0.1)

        plan = service.plan_view("u1", now=NOW)

        assert "ds.array" in [i["kp_id"] for i in plan["items"]]


class TestPlanAdvancesToGoal:
    """#4:前置都备齐之后,计划要能推进**目标本身**。

    `next_to_learn` 只在前置闭包里找候选,**目标自己永远不在里面** ——
    没有这一支,学生补完前置之后计划会变成空的。实测 24 步里 18 步为空。
    """

    def test_goal_appears_when_prereqs_ready(self, service, profile, seeded):
        profile.upsert_profile("u1", goal_kp_id="algo.dp", daily_minutes=600)
        # 前置全部达标,目标自己还没掌握
        for kp in ("algo.recursion", "ds.array", "algo.memoization", "prog.func_call"):
            profile.set_mastery("u1", kp, 0.9)
        profile.set_mastery("u1", "algo.dp", 0.2)

        plan = service.plan_view("u1", now=NOW)

        goal_items = [i for i in plan["items"] if i["kp_id"] == "algo.dp"]
        assert goal_items, "前置都备齐了,计划应该推进目标"
        assert goal_items[0]["action"] == "learn"

    def test_goal_marked_probe_when_prereq_untested(self, service, profile, seeded):
        profile.upsert_profile("u1", goal_kp_id="algo.dp", daily_minutes=600)
        profile.set_mastery("u1", "algo.dp", 0.2)
        # 前置一个都没测过

        plan = service.plan_view("u1", now=NOW)

        goal_items = [i for i in plan["items"] if i["kp_id"] == "algo.dp"]
        assert goal_items and goal_items[0]["action"] == "probe"

    def test_goal_not_offered_when_prereq_confirmed_weak(self, service, profile, seeded):
        profile.upsert_profile("u1", goal_kp_id="algo.dp", daily_minutes=600)
        profile.set_mastery("u1", "algo.dp", 0.2)
        profile.set_mastery("u1", "algo.recursion", 0.05)  # 已确认的弱前置

        plan = service.plan_view("u1", now=NOW)

        goal_items = [i for i in plan["items"] if i["kp_id"] == "algo.dp"]
        assert goal_items == [], "前置确实不会,不该直接推目标"
