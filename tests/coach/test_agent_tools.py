"""agent 工具层测试(agent 实验 M1)。

**不打网络、不需要 langgraph、不需要真 LLM** —— 工具层是上下文上的纯函数,
这正是它单独成模块的理由(循环那层才依赖 langgraph)。

两条断言份量最重:
1. 工具的输出必须和 `queries.*` **逐字一致** —— 包装层改变了语义的话,
   agent 与确定性路径的对照就不可信了。
2. **坏参数不能让整个循环崩** —— `llm_client` 对坏 JSON 会静默给 `{}`,
   所以"缺必填参数"必须返回 error 而不是抛异常。
"""

import json

import pytest

from coach.domain.models import Edge, KnowledgePoint
from coach.knowledge import problem_bank, queries
from coach.knowledge.governance import Governance
from coach.knowledge.store import KnowledgeStore
from coach.workflow import agent_tools as tools
from coach.workflow.agent_tools import DiagnosisContext


@pytest.fixture
def chain(knowledge) -> KnowledgeStore:
    """base → mid → top,外加同级更弱的 sibling(与 test_queries 同构)。"""
    for kp_id, name in [("base", "基础"), ("mid", "中层"), ("top", "目标"), ("sibling", "同级短板")]:
        knowledge.upsert_concept(KnowledgePoint(id=kp_id, name=name, subject="test"))
    for a, b in [("base", "mid"), ("mid", "top"), ("sibling", "top")]:
        knowledge.upsert_edge(Edge(from_id=a, to_id=b))
    return knowledge


@pytest.fixture
def ctx(chain) -> DiagnosisContext:
    """评测路径的构造方式:掌握度与观测集合都是现成的映射。"""
    return DiagnosisContext.from_maps(
        chain,
        learner_id="u1",
        mastery={"mid": 0.2, "sibling": 0.9, "base": 0.05},
        observed={"mid", "top"},
    )


def call(name, arguments, ctx):
    return json.loads(tools.execute_tool(name, arguments, ctx))


# ---------------------------------------------------------------- 上下文


class TestDiagnosisContext:
    def test_mastery_of_falls_back_to_bkt_init(self, ctx):
        from coach import config

        assert ctx.mastery_of("从未出现过的点") == config.BKT_P_INIT

    def test_is_observed_reflects_the_set(self, ctx):
        assert ctx.is_observed("mid") is True
        assert ctx.is_observed("base") is False

    def test_name_of_returns_kp_id_when_missing(self, ctx):
        assert ctx.name_of("nope") == "nope"

    def test_from_stores_reads_profile(self, chain, profile):
        """生产路径:掌握度/观测集合从 ProfileStore 一次性批量取。"""
        profile.set_mastery("u1", "mid", 0.2)
        built = DiagnosisContext.from_stores(chain, profile, "u1")

        assert built.mastery_of("mid") == pytest.approx(0.2)
        assert built.is_observed("mid") is True
        assert built.is_observed("base") is False  # 没有 learner_mastery 行 = 没观测过


# ---------------------------------------------------------------- 只读工具


class TestClosureTool:
    def test_matches_queries_exactly(self, ctx, chain):
        """包装层不许改变语义 —— 这是对照可信的前提。"""
        assert call("get_prereq_closure", {"kp_id": "top"}, ctx)["prerequisites"] == (
            queries.prereq_closure(chain, "top", depth=3)
        )

    def test_unknown_kp_returns_error_not_raise(self, ctx):
        assert "error" in call("get_prereq_closure", {"kp_id": "no.such"}, ctx)


class TestPathTool:
    def test_matches_queries_exactly(self, ctx, chain):
        got = call("get_shortest_path", {"from_kp_id": "top", "to_kp_id": "base"}, ctx)
        assert got["path"] == queries.shortest_prereq_path(chain, "top", "base", 3)


class TestMasteryTool:
    def test_returns_mastery_and_observed_flag(self, ctx):
        rows = {r["kp_id"]: r for r in call("get_mastery", {"kp_ids": ["mid", "base"]}, ctx)["mastery"]}

        assert rows["mid"] == {"kp_id": "mid", "name": "中层", "mastery": 0.2, "observed": True}
        # ★ 没观测过的点必须显式标出来,否则模型会把"初始值"当成"确实弱"
        assert rows["base"]["observed"] is False

    def test_rejects_non_list(self, ctx):
        assert "error" in call("get_mastery", {"kp_ids": "mid"}, ctx)


class TestObservedTool:
    def test_returns_sorted_ids(self, ctx):
        assert call("get_observed_kp_ids", {}, ctx)["observed_kp_ids"] == ["mid", "top"]


class TestProblemsTool:
    def test_lists_problems_hanging_off_the_kp(self, seeded):
        """用完整种子图 —— 内置题库引用的知识点只在那张图里,chain 太小会被全跳过。"""
        written = problem_bank.seed_from_bank(seeded)
        assert written["written"] > 0, "内置题库应能灌进种子图"

        ctx = DiagnosisContext.from_maps(seeded, "u1", {})
        payload = call("list_problems_for_kp", {"kp_id": "algo.dp"}, ctx)

        assert payload["problems"], "algo.dp 在种子题库里应该有题"
        assert {"id", "title", "difficulty"} <= set(payload["problems"][0])

    def test_unknown_kp_returns_empty_not_error(self, chain):
        ctx = DiagnosisContext.from_maps(chain, "u1", {})
        assert call("list_problems_for_kp", {"kp_id": "nothing.here"}, ctx)["problems"] == []


class TestErrorPatternTool:
    def test_empty_by_default(self, ctx):
        assert call("get_error_patterns", {}, ctx) == {"top": [], "recurring": []}

    def test_returns_provided_summary(self, chain):
        provided = {"top": [{"kp_id": "base", "error_type": "wrong", "count": 3}], "recurring": []}
        ctx = DiagnosisContext.from_maps(chain, "u1", {}, errors=provided)
        assert call("get_error_patterns", {}, ctx) == provided


# ---------------------------------------------------------------- 分发与鲁棒性


class TestDispatch:
    def test_unknown_tool_returns_error(self, ctx):
        assert "error" in call("no_such_tool", {}, ctx)

    @pytest.mark.parametrize(
        "name",
        [
            s["function"]["name"]
            for s in tools.TOOL_SCHEMAS
            if not tools.is_terminal(s["function"]["name"])
            and s["function"]["parameters"].get("required")
        ],
    )
    def test_empty_arguments_return_error_not_raise(self, name, ctx):
        """模拟 llm_client 把坏 JSON 变成 `{}` 之后的处境:有必填参数的工具必须报错。"""
        assert "error" in call(name, {}, ctx)

    @pytest.mark.parametrize("name", ["get_observed_kp_ids", "get_error_patterns"])
    def test_no_arg_tools_accept_empty_arguments(self, name, ctx):
        """这两个工具本来就没有必填参数,`{}` 是合法调用,不该报错。"""
        assert "error" not in call(name, {}, ctx)

    def test_every_schema_is_wellformed_openai_function(self):
        assert tools.TOOL_SCHEMAS
        for schema in tools.TOOL_SCHEMAS:
            assert schema["type"] == "function"
            fn = schema["function"]
            assert fn["name"] and fn["description"]
            assert fn["parameters"]["type"] == "object"

    def test_terminal_tool_is_flagged(self):
        assert tools.is_terminal(tools.TERMINAL_TOOL) is True
        assert tools.is_terminal("get_mastery") is False
        assert tools.schema_for(tools.TERMINAL_TOOL)["function"]["name"] == tools.TERMINAL_TOOL


# ---------------------------------------------------------------- 交卷规整


class TestParseDiagnosis:
    def test_fills_facts_from_graph_not_from_model(self, ctx):
        """模型只负责**排序和理由**;name/depth/mastery/path 由我们从图里补。"""
        result = tools.parse_diagnosis(
            {"root_causes": [{"kp_id": "base", "reason": "最浅且最弱"}], "summary": "先补基础"},
            ctx,
            target_kp="top",
        )

        root = result["root_causes"][0]
        assert root["name"] == "基础"
        assert root["depth"] == 2
        assert root["mastery"] == pytest.approx(0.05)
        assert root["path"] == ["top", "mid", "base"]
        assert root["reason"] == "最浅且最弱"
        assert result["summary"] == "先补基础"

    def test_status_distinguishes_observed_from_untested(self, ctx):
        result = tools.parse_diagnosis(
            {"root_causes": [{"kp_id": "mid"}, {"kp_id": "base"}]}, ctx, target_kp="top"
        )
        assert [r["status"] for r in result["root_causes"]] == ["gap", "unobserved"]

    def test_preserves_model_ordering(self, ctx):
        """★ 顺序就是模型的判断,不能替它重排 —— top1 准确率全靠这个顺序。"""
        result = tools.parse_diagnosis(
            {"root_causes": [{"kp_id": "base"}, {"kp_id": "sibling"}, {"kp_id": "mid"}]},
            ctx,
            target_kp="top",
        )
        assert [r["kp_id"] for r in result["root_causes"]] == ["base", "sibling", "mid"]

    def test_caps_at_top_k(self, ctx):
        result = tools.parse_diagnosis(
            {"root_causes": [{"kp_id": k} for k in ("base", "mid", "sibling", "top")]},
            ctx,
            target_kp="top",
            top_k=2,
        )
        assert len(result["root_causes"]) == 2

    def test_dedups_repeated_kp_ids(self, ctx):
        result = tools.parse_diagnosis(
            {"root_causes": [{"kp_id": "base"}, {"kp_id": "base"}]}, ctx, target_kp="top"
        )
        assert [r["kp_id"] for r in result["root_causes"]] == ["base"]

    @pytest.mark.parametrize("bad", [{}, {"root_causes": []}, {"root_causes": "base"}, {"root_causes": [{}]}])
    def test_malformed_submission_raises_valueerror(self, bad, ctx):
        """交卷参数不合法时**必须抛** —— 调用方据此让循环再走一轮,而不是接受一份空诊断。"""
        with pytest.raises(ValueError):
            tools.parse_diagnosis(bad, ctx, target_kp="top")

    def test_kp_outside_closure_still_reported_with_negative_depth(self, ctx):
        """模型挑了个不是前置的点:如实报告(depth=-1),不假装它在前置里。"""
        result = tools.parse_diagnosis({"root_causes": [{"kp_id": "top"}]}, ctx, target_kp="top")
        assert result["root_causes"][0]["depth"] == -1
