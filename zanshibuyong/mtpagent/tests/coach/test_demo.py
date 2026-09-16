"""端到端「真的能跑起来」测试。

把 api + 三个 worker 用内存总线串成一条完整链路,走一遍真实业务:

    POST /answer(交对的答案)
      → profile_worker 判分 / BKT / SM-2
      → 发 profile.updated
      → planner_worker 算根因 + 写人话解释
      → GET /gap       读得到 llm 解释
      → tick → scheduler_worker → plan.updated → GET /plan 有计划

这一组测试是「这道题真能做」的证明:题目库非空、答案能被判分、
三个 worker 的接力不断、API 读得到下游产物。
"""

import pytest
from fastapi.testclient import TestClient

from coach import config
from coach.api import main as api_main
from coach.coordination.recovery import Recovery
from coach.events import schema as events
from coach.knowledge import builder, problem_bank
from coach.knowledge.governance import Governance
from coach.knowledge.store import KnowledgeStore
from coach.profile.store import ProfileStore
from coach.workflow.query import QueryService
from coach.workers.planner_worker import PlannerWorker
from coach.workers.profile_worker import ProfileWorker
from coach.workers.scheduler_worker import SchedulerWorker

LEARNER = "u1"
GOAL = "algo.dp"


class FakeLLM:
    reply = "你卡在动态规划,但根因是先修好函数调用。"

    def __init__(self):
        self.calls = []

    def chat(self, messages, **_kwargs):
        self.calls.append(messages)
        return self.reply


@pytest.fixture
def system(db_conn, knowledge, profile, journal, fake_bus):
    """把整套系统装配到同一个临时库上。"""
    builder.seed_graph(Governance(knowledge))
    builder.seed_problems(knowledge)
    profile.upsert_profile(LEARNER, goal_kp_id=GOAL, daily_minutes=30)

    service = QueryService(knowledge, profile)
    llm = FakeLLM()
    workers = {
        # P4 起,答题链(判分→BKT→SM-2→根因→LLM 解释)在 pipeline 里跑,
        # 所以 llm 要从 profile_worker 传进去,planner 退化为「缓存过期才刷新」
        "profile": ProfileWorker(profile, knowledge, journal, bus=fake_bus, llm=llm),
        "planner": PlannerWorker(
            knowledge, profile, journal, llm=llm, bus=fake_bus, service=service
        ),
        "scheduler": SchedulerWorker(knowledge, profile, journal, bus=fake_bus, service=service),
    }
    client = TestClient(
        api_main.create_app(bus=fake_bus, query_service=service)
    )
    return {
        "client": client,
        "bus": fake_bus,
        "workers": workers,
        "service": service,
        "llm": llm,
        "profile": profile,
        "journal": journal,
    }


def pump(system, worker_name: str) -> int:
    """把某个 worker 对应流上的消息取出来处理掉(模拟常驻消费循环跑一轮)。"""
    worker = system["workers"][worker_name]
    messages = system["bus"].consume(worker.stream, worker.group, worker.consumer)
    for _msg_id, event in messages:
        worker.process(event)
    return len(messages)


def answer(system, problem_id: str, text: str):
    body = {
        "learner_id": LEARNER,
        "problem_id": problem_id,
        "answer_text": text,
        "elapsed_ms": 500,
    }
    response = system["client"].post("/answer", json=body)
    assert response.status_code == 202
    return response.json()


class TestFullLoop:
    def test_answer_flows_through_to_gap_explanation(self, system):
        # 1. 交一个正确答案
        answer(system, "lc.70", "8")
        assert len(system["bus"].queues[config.STREAM_ANSWER]) == 1

        # 2. profile_worker 消费 → 画像更新 → 发 profile.updated
        assert pump(system, "profile") == 1
        assert len(system["bus"].queues[config.STREAM_PROFILE]) == 1

        # 3. 解释已在 pipeline 的 explain 节点里生成
        #    lc.70 挂在 algo.dp 和 algo.recursion 两个知识点上,每个各一次
        assert len(system["llm"].calls) == 2

        # 4. planner 收到 profile.updated,但缓存还新鲜 → 跳过,不重复调 LLM
        assert pump(system, "planner") == 1
        assert len(system["llm"].calls) == 2

        # 4. GET /gap 读得到人话解释
        gap = system["client"].get(f"/gap/{LEARNER}/{GOAL}").json()
        assert gap["explanation_source"] == "llm"
        assert gap["explanation_pending"] is False
        assert gap["root_causes"], "动态规划的前置应该存在缺口"

        # 5. 掌握度确实动了
        assert gap["mastery"] > config.BKT_P_INIT

    def test_tick_produces_plan(self, system):
        system["bus"].publish(
            config.STREAM_TICK, events.new_event(events.TICK_SCHEDULED, LEARNER)
        )

        assert pump(system, "scheduler") == 1
        assert len(system["bus"].queues[config.STREAM_PLAN]) == 1

        plan = system["client"].get(f"/plan/{LEARNER}").json()
        assert plan["items"], "计划不该为空"
        assert plan["goal_kp_id"] == GOAL

    def test_wrong_answer_creates_review_due_item(self, system):
        answer(system, "lc.70", "999")
        pump(system, "profile")

        memory = system["profile"].get_memory(LEARNER, "algo.dp")
        assert memory["reps"] == 0
        assert memory["lapses"] == 1

        # 把到期时间调到过去,计划里应出现复习项
        system["profile"].set_memory(
            LEARNER, "algo.dp", ease=memory["ease"], interval_days=1, reps=0, lapses=1,
            last_review=memory["last_review"], due_at=memory["last_review"] - 86400,
        )
        plan = system["client"].get(f"/plan/{LEARNER}").json()
        assert "algo.dp" in [i["kp_id"] for i in plan["items"] if i["action"] == "review"]

    def test_every_seeded_problem_is_answerable(self, system):
        """题目库非空,且每道题的"正确提交"都能被判对。"""
        knowledge = system["service"].knowledge
        problems = knowledge.list_problems()
        bank = problem_bank.load_from_file(problem_bank.BUNDLED_BANK)
        assert len(problems) == len(bank)

        for problem in problems:
            expected = problem_bank.expected_answer(knowledge, problem.id)
            assert expected is not None, f"{problem.id} 缺正确答案"

            result = system["workers"]["profile"].handle_answer(
                events.new_event(
                    events.ANSWER_SUBMITTED,
                    f"probe-{problem.id}",
                    payload={
                        "problem_id": problem.id,
                        "answer_text": expected,
                        "elapsed_ms": 1,
                    },
                )
            )
            assert result["correct"] is True, f"{problem.id} 的正确提交被判错:{result}"

    def test_full_recovery_after_crash_mid_pipeline(self, system):
        """journal 里留下"已记未完成"的事件,重启后能被逐个补做。"""
        event = answer(system, "lc.70", "8")
        body_event = system["bus"].queues[config.STREAM_ANSWER][0][1]
        system["journal"].record(body_event)  # 记了但没做

        replayed = Recovery(system["journal"]).replay(
            system["workers"]["profile"].process, types=("answer.submitted",)
        )

        assert replayed == [body_event.event_id]
        assert system["profile"].get_mastery(LEARNER, "algo.dp") > config.BKT_P_INIT


class TestRunAllHelpers:
    """verify run_all 这个开发入口的两块新齿轮:tick 生产、建档。"""

    def test_publish_tick_sends_one_event_per_learner(self, fake_bus):
        from coach.workers import run_all

        sent = run_all.publish_tick(fake_bus, ["u1", "u2"])

        assert sent == 2
        queue = fake_bus.queues[config.STREAM_TICK]
        assert [e.learner_id for _mid, e in queue] == ["u1", "u2"]
        assert all(e.type == events.TICK_SCHEDULED for _mid, e in queue)

    def test_init_learner_creates_profile(self, profile):
        from coach.workers import run_all

        run_all.init_learner(profile, "u9", "algo.dp", 45)

        row = profile.get_profile("u9")
        assert row["goal_kp_id"] == "algo.dp"
        assert row["daily_minutes"] == 45
        assert "u9" in profile.list_learners()

    def test_init_learner_is_idempotent(self, profile):
        from coach.workers import run_all

        run_all.init_learner(profile, "u9", "algo.dp", 30)
        run_all.init_learner(profile, "u9", "algo.greedy", 60)

        assert profile.get_profile("u9")["goal_kp_id"] == "algo.greedy"
        assert profile.list_learners().count("u9") == 1

    def test_build_llm_disabled_returns_none(self):
        from coach.workers import run_all

        assert run_all.build_llm(enabled=False) is None

    def test_build_llm_without_key_returns_none(self, monkeypatch):
        from coach.workers import run_all

        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        assert run_all.build_llm(enabled=True) is None


class TestLearnerBootstrap:
    def test_plan_is_empty_without_a_goal(self, db_conn, knowledge, profile, journal, fake_bus):
        builder.seed_graph(Governance(knowledge))
        builder.seed_problems(knowledge)
        profile.upsert_profile("u2")  # 不设 goal

        service = QueryService(knowledge, profile)
        client = TestClient(api_main.create_app(bus=fake_bus, query_service=service))

        plan = client.get("/plan/u2").json()
        assert plan["items"] == []
        assert plan["goal_kp_id"] is None

    def test_gap_still_works_without_a_goal(self, db_conn, knowledge, profile, journal, fake_bus):
        builder.seed_graph(Governance(knowledge))
        service = QueryService(knowledge, profile)
        client = TestClient(api_main.create_app(bus=fake_bus, query_service=service))

        # /gap 是"以某个知识点为目标"的查询,不需要 learner_profile 里有 goal
        body = client.get(f"/gap/{LEARNER}/{GOAL}").json()
        assert body["root_causes"]
