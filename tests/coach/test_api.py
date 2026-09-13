"""P1 API 测试(work.md §7 P1.10):/answer 只入队、不调 LLM、< 50ms、分层不越界。"""

import inspect
import time

import pytest
from fastapi.testclient import TestClient

from coach import config
from coach.api import main as api_main
from coach.api import routes as api_routes
from coach.api import schemas as api_schemas
from coach.events import schema as events
from coach.knowledge.store import KnowledgeStore
from coach.profile.store import ProfileStore
from coach.workflow.query import QueryService

BODY = {
    "learner_id": "u1",
    "problem_id": "lc.70",
    "answer_text": "8",
    "elapsed_ms": 1200,
}


@pytest.fixture
def query_service(db_conn) -> QueryService:
    return QueryService(KnowledgeStore(db_conn), ProfileStore(db_conn))


@pytest.fixture
def client(fake_bus, query_service):
    """注入测试自己的库,避免 create_app 默认去开真实的 data/coach/coach.db。"""
    return TestClient(api_main.create_app(bus=fake_bus, query_service=query_service))


class TestSubmitAnswer:
    def test_returns_202_with_ids(self, client):
        response = client.post("/answer", json=BODY)

        assert response.status_code == 202
        body = response.json()
        assert body["accepted"] is True
        assert len(body["event_id"]) == 26  # ULID
        assert body["trace_id"] == body["event_id"]  # 新链路,trace 即自身

    def test_publishes_answer_submitted_event(self, client, fake_bus):
        client.post("/answer", json=BODY)

        queue = fake_bus.queues[config.STREAM_ANSWER]
        assert len(queue) == 1
        _, event = queue[0]
        assert event.type == events.ANSWER_SUBMITTED
        assert event.learner_id == "u1"
        assert event.payload["problem_id"] == "lc.70"

    @pytest.mark.parametrize(
        "bad",
        [
            {"problem_id": "p", "answer_text": "x"},                    # 缺 learner_id
            {"learner_id": "u1", "answer_text": "x"},                   # 缺 problem_id
            {"learner_id": "", "problem_id": "p", "answer_text": "x"},  # 空 learner_id
            {"learner_id": "u1", "problem_id": "p", "elapsed_ms": -1},  # 负耗时
        ],
    )
    def test_validation_rejects_bad_payload(self, client, bad):
        assert client.post("/answer", json=bad).status_code == 422

    def test_bad_payload_publishes_nothing(self, client, fake_bus):
        client.post("/answer", json={"learner_id": "u1"})
        assert fake_bus.queues[config.STREAM_ANSWER] == []

    def test_latency_under_50ms(self, client):
        client.post("/answer", json=BODY)  # 预热,排除首次建连开销

        start = time.perf_counter()
        response = client.post("/answer", json=BODY)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert response.status_code == 202
        assert elapsed_ms < 50, f"/answer 耗时 {elapsed_ms:.1f}ms,超过 50ms 验收线"


class TestNoLLMInApiLayer:
    def test_llm_methods_are_never_called(self, client, monkeypatch):
        llm_module = pytest.importorskip("llm.llm_client")

        def boom(*_args, **_kwargs):
            raise AssertionError("API 层不允许调用 LLM")

        for name in ("chat", "generate", "chat_structured"):
            monkeypatch.setattr(llm_module.LLM, name, boom)

        assert client.post("/answer", json=BODY).status_code == 202

    def test_api_source_has_no_forbidden_imports(self):
        """硬约束:api/ 不许 import knowledge/ 或 profile/(也顺带禁 llm)。"""
        forbidden = ("coach.knowledge", "coach.profile", "coach.workers", "llm.")
        for module in (api_main, api_routes, api_schemas):
            source = inspect.getsource(module)
            for name in forbidden:
                assert f"import {name}" not in source, f"{module.__name__} 违规 import {name}"


class TestProfileEndpoint:
    @pytest.fixture(autouse=True)
    def _seed(self, seeded, profile):
        profile.upsert_profile("u1", goal_kp_id="algo.dp", daily_minutes=45)
        profile.set_mastery("u1", "algo.dp", 0.3)
        profile.set_mastery("u1", "ds.array", 0.2)
        profile.bump_error("u1", "algo.dp", "wrong")
        profile.bump_error("u1", "ds.array", "wrong")
        profile.set_memory("u1", "ds.array", ease=2.5, interval_days=1, reps=1, lapses=0,
                           last_review=0.0, due_at=time.time() - 3600)

    def test_returns_goal_and_summary(self, client):
        body = client.get("/profile/u1").json()

        assert body["goal"]["kp_id"] == "algo.dp"
        assert body["goal"]["mastery"] == pytest.approx(0.3)
        assert body["daily_minutes"] == 45
        assert body["mastery_summary"]["observed_count"] == 2

    def test_lowest_lists_weak_points(self, client):
        body = client.get("/profile/u1").json()
        assert [x["kp_id"] for x in body["mastery_summary"]["lowest"]] == ["ds.array", "algo.dp"]

    def test_due_now(self, client):
        body = client.get("/profile/u1").json()
        assert [x["kp_id"] for x in body["due_now"]] == ["ds.array"]

    def test_error_patterns_flag_systematic_misunderstanding(self, client):
        """同一个错误类型跨两个知识点 → 疑似系统性误解,不是单点不会。"""
        body = client.get("/profile/u1").json()

        assert len(body["error_patterns"]["top"]) == 2
        assert len(body["error_patterns"]["recurring"]) == 1
        assert set(body["error_patterns"]["recurring"][0]["kp_ids"]) == {"algo.dp", "ds.array"}

    def test_unknown_learner_is_not_an_error(self, client):
        body = client.get("/profile/nobody").json()
        assert body["goal"] is None
        assert body["mastery_summary"]["observed_count"] == 0


class TestHealthAndDocs:
    def test_health_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "redis": True}

    def test_openapi_available(self, client):
        """Swagger 是 P1 指定的唯一入口(不做前端)。"""
        schema = client.get("/openapi.json").json()
        assert "/answer" in schema["paths"]
        assert "/health" in schema["paths"]
        assert "/gap/{learner_id}/{kp_id}" in schema["paths"]
        assert "/graph/{kp_id}" in schema["paths"]
        assert "/profile/{learner_id}" in schema["paths"]

    def test_answer_documented_as_202(self, client):
        schema = client.get("/openapi.json").json()
        assert "202" in schema["paths"]["/answer"]["post"]["responses"]


class TestGapEndpoint:
    """§7 P2 验收:GET /gap 必须报出函数调用。"""

    @pytest.fixture(autouse=True)
    def _seed(self, seeded, profile):
        for kp_id, p in {
            "algo.dp": 0.3,
            "algo.recursion": 0.9,
            "prog.func_call": 0.2,
            "ds.array": 0.9,
            "algo.memoization": 0.9,
        }.items():
            profile.set_mastery("u1", kp_id, p)

    def test_reports_func_call_as_top_root_cause(self, client):
        response = client.get("/gap/u1/algo.dp")

        assert response.status_code == 200
        body = response.json()
        assert body["root_causes"][0]["kp_id"] == "prog.func_call"
        assert body["root_causes"][0]["path"] == [
            "algo.dp",
            "algo.recursion",
            "prog.func_call",
        ]
        assert body["mastery"] == pytest.approx(0.3)

    def test_explanation_pending_until_worker_runs(self, client):
        body = client.get("/gap/u1/algo.dp").json()
        assert body["explanation_source"] == "template"
        assert body["explanation_pending"] is True

    def test_unknown_kp_returns_404(self, client):
        response = client.get("/gap/u1/nope.not.exist")
        assert response.status_code == 404
        assert "nope.not.exist" in response.json()["detail"]

    def test_top_k_is_validated(self, client):
        assert client.get("/gap/u1/algo.dp?top_k=0").status_code == 422
        assert client.get("/gap/u1/algo.dp?top_k=99").status_code == 422


class TestGraphEndpoint:
    @pytest.fixture(autouse=True)
    def _seed(self, seeded, profile):
        profile.set_mastery("u1", "prog.func_call", 0.1)

    def test_returns_prereq_tree(self, client):
        body = client.get("/graph/algo.dp?depth=2").json()

        child_ids = {c["kp_id"] for c in body["prereq_tree"]["children"]}
        assert child_ids == {"algo.recursion", "ds.array", "algo.memoization"}

    def test_marks_gaps_when_learner_given(self, client):
        body = client.get("/graph/algo.dp?depth=3&learner_id=u1").json()

        assert "prog.func_call" in [g["kp_id"] for g in body["gaps"]]

    def test_unknown_kp_returns_404(self, client):
        assert client.get("/graph/nope").status_code == 404


class TestPlanEndpoint:
    @pytest.fixture(autouse=True)
    def _seed(self, seeded, profile):
        profile.upsert_profile("u1", goal_kp_id="algo.dp", daily_minutes=600)
        # 把闭包里其余前置设为已掌握,让 func_call 成为唯一根因(否则 top_k 截断会挤掉它)
        for kp_id in ("algo.recursion", "ds.array", "algo.memoization"):
            profile.set_mastery("u1", kp_id, 0.9)
        profile.set_mastery("u1", "prog.func_call", 0.1)
        profile.set_memory(
            "u1", "ds.array", ease=2.5, interval_days=1, reps=1, lapses=0,
            last_review=time.time() - 86_400, due_at=time.time() - 86_400,
        )

    def test_overdue_kp_is_review(self, client):
        body = client.get("/plan/u1").json()

        reviews = [i for i in body["items"] if i["action"] == "review"]
        assert [i["kp_id"] for i in reviews] == ["ds.array"]

    def test_plan_includes_remedial_for_goal(self, client):
        body = client.get("/plan/u1").json()

        assert "prog.func_call" in [
            i["kp_id"] for i in body["items"] if i["action"] == "remedial"
        ]

    def test_daily_minutes_override(self, client):
        assert client.get("/plan/u1?daily_minutes=60").json()["daily_minutes"] == 60

    def test_daily_minutes_validated(self, client):
        assert client.get("/plan/u1?daily_minutes=1").status_code == 422

    def test_unknown_learner_gets_empty_plan(self, client):
        body = client.get("/plan/nobody").json()
        assert body["items"] == []
