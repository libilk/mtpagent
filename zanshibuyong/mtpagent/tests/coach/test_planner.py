"""P2 planner_worker + 只读门面测试。

核心是验收场景:构造已知掌握度分布,断言根因定位到预期节点。
LLM 用假对象,测试不需要 API key。
"""

import pytest

from coach.events import schema as events
from coach.workflow.query import QueryService
from coach.workers.planner_worker import PlannerWorker


class FakeLLM:
    def __init__(self, reply="先把函数调用补上,递归是它的直接下游。"):
        self.reply = reply
        self.calls = []

    def chat(self, messages, **_kwargs):
        self.calls.append(messages)
        return self.reply


class BoomLLM:
    def chat(self, *_args, **_kwargs):
        raise RuntimeError("模型超时")


@pytest.fixture
def service(knowledge, profile, seeded) -> QueryService:
    return QueryService(knowledge, profile)


@pytest.fixture
def planner(knowledge, profile, journal, fake_bus, service) -> PlannerWorker:
    return PlannerWorker(
        knowledge, profile, journal, llm=FakeLLM(), bus=fake_bus, service=service
    )


def set_mastery(profile, learner_id, values):
    for kp_id, p in values.items():
        profile.set_mastery(learner_id, kp_id, p)


def profile_updated_event(learner_id="u1", kp_ids=("algo.dp",)):
    return events.new_event(
        events.PROFILE_UPDATED, learner_id, payload={"kp_ids": list(kp_ids)}
    )


class TestAcceptanceScenario:
    """§7 P2 验收线:递归 0.9、函数调用 0.2、动态规划 0.3 → 报出函数调用。"""

    @pytest.fixture(autouse=True)
    def _setup(self, profile):
        set_mastery(
            profile,
            "u1",
            {"algo.recursion": 0.9, "prog.func_call": 0.2, "algo.dp": 0.3},
        )

    def test_func_call_is_reported_as_root_cause(self, service):
        view = service.gap_view("u1", "algo.dp")

        reported = [r["kp_id"] for r in view["root_causes"]]
        assert "prog.func_call" in reported, reported
        assert view["mastery"] == pytest.approx(0.3)

    def test_func_call_ranks_first_when_other_prereqs_mastered(self, service, profile):
        set_mastery(profile, "u1", {"ds.array": 0.9, "algo.memoization": 0.9})

        view = service.gap_view("u1", "algo.dp")

        assert view["root_causes"][0]["kp_id"] == "prog.func_call"
        assert view["root_causes"][0]["path"] == [
            "algo.dp",
            "algo.recursion",
            "prog.func_call",
        ]

    def test_explanation_is_template_before_worker_runs(self, service):
        view = service.gap_view("u1", "algo.dp")

        assert view["explanation_source"] == "template"
        assert view["explanation_pending"] is True
        assert "函数调用" in view["explanation"]

    def test_unknown_kp_raises_keyerror(self, service):
        with pytest.raises(KeyError):
            service.gap_view("u1", "does.not.exist")


class TestRefreshExplanation:
    def test_writes_llm_explanation_to_cache(self, planner, service, profile):
        set_mastery(profile, "u1", {"algo.dp": 0.3, "prog.func_call": 0.2})

        assert planner.refresh_explanation("u1", "algo.dp") is True

        view = service.gap_view("u1", "algo.dp")
        assert view["explanation_source"] == "llm"
        assert view["explanation_pending"] is False
        assert view["explanation"] == "先把函数调用补上,递归是它的直接下游。"

    def test_prompt_carries_root_cause_and_path(self, planner, profile):
        set_mastery(
            profile,
            "u1",
            {
                "algo.dp": 0.3,
                "algo.recursion": 0.9,
                "ds.array": 0.9,
                "algo.memoization": 0.9,
                "prog.func_call": 0.2,
            },
        )
        planner.refresh_explanation("u1", "algo.dp")

        user_message = planner.llm.calls[0][-1]["content"]
        assert "函数调用" in user_message
        assert "algo.dp → algo.recursion → prog.func_call" in user_message

    def test_no_gaps_means_no_llm_call(self, planner, profile):
        # 闭包内全部节点都要给高掌握度,否则未记录的点会按初始值算成缺口
        set_mastery(
            profile,
            "u1",
            {
                "algo.recursion": 0.9,
                "ds.array": 0.9,
                "algo.memoization": 0.9,
                "prog.func_call": 0.9,
            },
        )

        assert planner.refresh_explanation("u1", "algo.dp") is False
        assert planner.llm.calls == []

    def test_without_llm_keeps_template(self, knowledge, profile, journal, service):
        worker = PlannerWorker(knowledge, profile, journal, llm=None, service=service)
        set_mastery(profile, "u1", {"algo.dp": 0.3, "prog.func_call": 0.2})

        assert worker.refresh_explanation("u1", "algo.dp") is False
        assert service.gap_view("u1", "algo.dp")["explanation_source"] == "template"

    def test_llm_failure_falls_back_without_crashing(self, knowledge, profile, journal, service):
        worker = PlannerWorker(knowledge, profile, journal, llm=BoomLLM(), service=service)
        set_mastery(profile, "u1", {"algo.dp": 0.3, "prog.func_call": 0.2})

        assert worker.refresh_explanation("u1", "algo.dp") is False
        assert service.gap_view("u1", "algo.dp")["explanation_source"] == "template"

    def test_unknown_kp_is_skipped(self, planner):
        assert planner.refresh_explanation("u1", "nope") is False


class TestPlannerConsumption:
    def test_profile_updated_triggers_refresh(self, planner, profile, service):
        set_mastery(profile, "u1", {"algo.dp": 0.3, "prog.func_call": 0.2})

        result = planner.process(profile_updated_event())

        assert result["status"] == "ok"
        assert result["refreshed"] == ["algo.dp"]
        assert service.gap_view("u1", "algo.dp")["explanation_source"] == "llm"

    def test_same_event_is_processed_once(self, planner, profile):
        set_mastery(profile, "u1", {"algo.dp": 0.3, "prog.func_call": 0.2})
        event = profile_updated_event()

        planner.process(event)
        assert planner.process(event)["status"] == "skipped"
        assert len(planner.llm.calls) == 1

    def test_tick_uses_goal_from_profile_when_no_kp_ids(self, planner, profile):
        set_mastery(profile, "u1", {"algo.dp": 0.3, "prog.func_call": 0.2})
        profile.upsert_profile("u1", goal_kp_id="algo.dp")
        tick = events.new_event(events.TICK_SCHEDULED, "u1", payload={})

        result = planner.process(tick)

        assert result["refreshed"] == ["algo.dp"]

    def test_tick_without_goal_is_a_noop(self, planner, profile):
        tick = events.new_event(events.TICK_SCHEDULED, "u1", payload={})
        assert planner.process(tick) == {"status": "ok", "refreshed": []}

    def test_unrelated_event_ignored(self, planner):
        event = events.new_event(events.ANSWER_SUBMITTED, "u1")
        assert planner.process(event)["status"] == "ignored"


class AgentLLM:
    """假 LLM:一轮就交卷。agent 诊断走 `chat_structured`,解释走 `chat`。"""

    def __init__(self, kp_id="prog.func_call"):
        self.kp_id = kp_id
        self.calls = 0

    def chat(self, messages, **_kwargs):
        return "先把函数调用补上。"

    def chat_structured(self, messages, tools=None, **_kwargs):
        from llm.llm_client import LLMResponse, ToolCall

        self.calls += 1
        return LLMResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id=f"c{self.calls}",
                    name="submit_diagnosis",
                    arguments={"root_causes": [{"kp_id": self.kp_id}], "summary": "先补函数调用。"},
                )
            ],
        )


class TestAgentDiagnosis:
    """agent 诊断的异步生成(M5)。

    ★ 默认必须是关的 —— 一次诊断要好几轮 LLM 调用,挂在每条 profile.updated 上会失控。
    """

    def _seed(self, profile):
        profile.upsert_profile("u1", goal_kp_id="algo.dp")
        set_mastery(profile, "u1", {"algo.dp": 0.3, "prog.func_call": 0.2, "algo.recursion": 0.9})

    def _worker(self, knowledge, profile, journal, fake_bus, service, llm):
        return PlannerWorker(
            knowledge, profile, journal, llm=llm, bus=fake_bus, service=service, agent_diagnosis=True
        )

    def test_off_by_default(self, planner, profile):
        self._seed(profile)

        result = planner.process(profile_updated_event())

        assert "agent_diagnosed" not in result
        assert profile.get_agent_diagnosis("u1", "algo.dp", top_k=3, depth=3) is None

    def test_writes_cache_when_enabled(self, knowledge, profile, journal, fake_bus, service):
        self._seed(profile)
        llm = AgentLLM()
        worker = self._worker(knowledge, profile, journal, fake_bus, service, llm)

        result = worker.process(profile_updated_event())

        assert result["agent_diagnosed"] == "algo.dp"
        assert llm.calls >= 1
        cached = profile.get_agent_diagnosis("u1", "algo.dp", top_k=3, depth=3)
        assert cached["payload"]["root_causes"][0]["kp_id"] == "prog.func_call"
        assert cached["payload"]["explanation_source"] == "agent"

    def test_only_diagnoses_the_goal_kp(self, knowledge, profile, journal, fake_bus, service):
        """payload 里可能有多个 kp,但只对目标跑一次 —— 否则调用次数会失控。"""
        self._seed(profile)
        llm = AgentLLM()
        worker = self._worker(knowledge, profile, journal, fake_bus, service, llm)
        event = events.new_event(
            events.PROFILE_UPDATED, "u1", payload={"kp_ids": ["algo.dp", "ds.array", "algo.recursion"]}
        )

        worker.process(event)

        assert llm.calls == 1, "三个 kp 也只该跑一次 agent 诊断"

    def test_no_goal_is_a_noop(self, knowledge, profile, journal, fake_bus, service):
        set_mastery(profile, "u1", {"algo.dp": 0.3})
        llm = AgentLLM()
        worker = self._worker(knowledge, profile, journal, fake_bus, service, llm)

        result = worker.process(profile_updated_event())

        assert result["agent_diagnosed"] is None
        assert llm.calls == 0

    def test_llm_failure_does_not_break_the_event(self, knowledge, profile, journal, fake_bus, service):
        """诊断失败只该让这一项跳过,不能把整条事件弄挂。"""
        self._seed(profile)

        class BoomAgentLLM(AgentLLM):
            def chat_structured(self, messages, tools=None, **_kwargs):
                raise RuntimeError("模型挂了")

        worker = self._worker(knowledge, profile, journal, fake_bus, service, BoomAgentLLM())

        result = worker.process(profile_updated_event())

        assert result["status"] == "ok"  # 事件本身处理成功
        assert result["agent_diagnosed"] is None
        assert profile.get_agent_diagnosis("u1", "algo.dp", top_k=3, depth=3) is None


class TestGraphView:
    def test_tree_structure_and_depth(self, service, seeded):
        view = service.graph_view("algo.dp", depth=2)

        assert view["node"]["kp_id"] == "algo.dp"
        child_ids = {c["kp_id"] for c in view["prereq_tree"]["children"]}
        assert child_ids == {"algo.recursion", "ds.array", "algo.memoization"}

        recursion = next(
            c for c in view["prereq_tree"]["children"] if c["kp_id"] == "algo.recursion"
        )
        assert {c["kp_id"] for c in recursion["children"]} == {"prog.func_call"}

    def test_marks_gaps_and_mastered(self, service, seeded, profile):
        set_mastery(
            profile,
            "u1",
            {"algo.recursion": 0.9, "ds.array": 0.9, "algo.memoization": 0.9,
             "prog.func_call": 0.1},
        )

        view = service.graph_view("algo.dp", depth=3, learner_id="u1")

        assert [g["kp_id"] for g in view["gaps"]] == ["prog.func_call"]
        assert {m["kp_id"] for m in view["mastered"]} == {
            "algo.recursion",
            "ds.array",
            "algo.memoization",
        }

    def test_without_learner_id_no_marks(self, service, seeded):
        view = service.graph_view("algo.dp", depth=2)

        assert view["mastered"] == []
        assert view["gaps"] == []
        assert view["prereq_tree"]["mastery"] is None

    def test_unknown_kp_raises(self, service, seeded):
        with pytest.raises(KeyError):
            service.graph_view("nope")
