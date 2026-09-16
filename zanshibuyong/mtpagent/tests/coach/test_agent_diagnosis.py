"""agent 诊断循环测试(agent 实验 M2)。

**不打网络** —— 用一个按脚本吐响应的假 LLM,把循环的每条分支都走一遍。

最重的一条是断点恢复:崩溃后重跑**不能重调已经完成的那轮 LLM**。
这是"挂在 LangGraph 上"的全部理由,也是两个对标项目都没有的能力,
所以它必须由测试守住,而不是靠文档声称。
"""

import pytest

from coach.domain.models import Edge, KnowledgePoint
from coach.knowledge.store import KnowledgeStore
from coach.workflow import agent_tools
from coach.workflow.agent_diagnosis import DiagnosisAgent
from coach.workflow.agent_tools import TERMINAL_TOOL, DiagnosisContext
from coach.workflow.pipeline import make_checkpointer
from coach.workflow.query import QueryService
from llm.llm_client import LLMResponse, ToolCall


class ScriptedLLM:
    """按脚本吐响应,并记录每次调用收到的 messages(供调用计数与状态断言)。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat_structured(self, messages, tools=None, **kwargs):
        self.calls.append([dict(m) for m in messages])
        if not self.responses:
            return LLMResponse(content="(脚本已用完)")
        return self.responses.pop(0)


def tool_response(name, arguments, call_id="c1"):
    return LLMResponse(content="", tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)])


@pytest.fixture
def chain(knowledge) -> KnowledgeStore:
    for kp_id, name in [("base", "基础"), ("mid", "中层"), ("top", "目标"), ("sibling", "同级短板")]:
        knowledge.upsert_concept(KnowledgePoint(id=kp_id, name=name, subject="test"))
    for a, b in [("base", "mid"), ("mid", "top"), ("sibling", "top")]:
        knowledge.upsert_edge(Edge(from_id=a, to_id=b))
    return knowledge


@pytest.fixture
def ctx(chain) -> DiagnosisContext:
    return DiagnosisContext.from_maps(
        chain, "u1", {"mid": 0.2, "sibling": 0.9, "base": 0.05}, observed={"mid", "top"}
    )


def submit(kp_id="base", **extra):
    return tool_response(TERMINAL_TOOL, {"root_causes": [{"kp_id": kp_id}], **extra})


class TestTermination:
    def test_terminates_on_submit_diagnosis(self, chain, ctx):
        llm = ScriptedLLM([tool_response("get_prereq_closure", {"kp_id": "top"}), submit("base")])
        result = DiagnosisAgent(chain, llm).run("u1", "top", ctx)

        assert [r["kp_id"] for r in result["root_causes"]] == ["base"]
        assert result["_agent"]["terminated"] == "submitted"
        assert result["_agent"]["steps"] == 2
        assert result["_agent"]["tools_used"] == ["get_prereq_closure", TERMINAL_TOOL]

    def test_model_answering_without_tools_ends_the_loop(self, chain, ctx):
        """模型直接打字、不调工具 —— 收尾且没有诊断,不假装成功。"""
        llm = ScriptedLLM([LLMResponse(content="我觉得是基础不牢")])
        result = DiagnosisAgent(chain, llm).run("u1", "top", ctx)

        assert result["root_causes"] == []
        assert result["_agent"]["terminated"] == "no_submission"

    def test_step_limit_stops_runaway_loop(self, chain, ctx):
        llm = ScriptedLLM([tool_response("get_mastery", {"kp_ids": ["mid"]})] * 20)
        result = DiagnosisAgent(chain, llm, max_steps=3).run("u1", "top", ctx)

        assert result["_agent"]["terminated"] == "step_limit"
        assert result["_agent"]["steps"] == 3
        assert len(llm.calls) == 3, "步数上限必须真的封住 LLM 调用次数"


class TestMalformedArguments:
    def test_bad_submission_gets_error_and_model_retries(self, chain, ctx):
        """★ 交卷参数不合法时不能接受一份空诊断,要回错误让模型重来。"""
        llm = ScriptedLLM([tool_response(TERMINAL_TOOL, {}), submit("mid")])
        result = DiagnosisAgent(chain, llm).run("u1", "top", ctx)

        assert result["_agent"]["terminated"] == "submitted"
        assert [r["kp_id"] for r in result["root_causes"]] == ["mid"]

        # 第二次调用时,模型应当看到上一次那条带 error 的工具消息
        tool_messages = [m for m in llm.calls[1] if m.get("role") == "tool"]
        assert tool_messages and "error" in tool_messages[-1]["content"]

    def test_bad_tool_arguments_return_error_and_loop_continues(self, chain, ctx):
        llm = ScriptedLLM([tool_response("get_prereq_closure", {}), submit("base")])
        result = DiagnosisAgent(chain, llm).run("u1", "top", ctx)

        assert result["_agent"]["terminated"] == "submitted"
        tool_messages = [m for m in llm.calls[1] if m.get("role") == "tool"]
        assert "error" in tool_messages[-1]["content"]


class TestOutputContract:
    def test_matches_gap_view_key_for_key(self, chain, profile, ctx):
        """★ 契约必须逐字节一致 —— 不一致就没法和确定性路径做对照。"""
        llm = ScriptedLLM([submit("base")])
        result = DiagnosisAgent(chain, llm).run("u1", "top", ctx)

        reference = QueryService(chain, profile).gap_view("u1", "top")
        assert set(result) - {"_agent"} == set(reference)

    def test_root_cause_elements_match_the_api_schema(self, chain, ctx):
        llm = ScriptedLLM([submit("base")])
        result = DiagnosisAgent(chain, llm).run("u1", "top", ctx)

        assert set(result["root_causes"][0]) == {
            "kp_id", "name", "depth", "mastery", "status", "path", "reason"
        }

    def test_explanation_source_is_marked_agent(self, chain, ctx):
        llm = ScriptedLLM([submit("base", summary="先把基础补上")])
        result = DiagnosisAgent(chain, llm).run("u1", "top", ctx)

        assert result["explanation"] == "先把基础补上"
        assert result["explanation_source"] == "agent"
        assert result["explanation_pending"] is False


class TestGuards:
    def test_without_llm_raises_pointing_at_deterministic_path(self, chain, ctx):
        with pytest.raises(ValueError, match="确定性路径"):
            DiagnosisAgent(chain, None).run("u1", "top", ctx)

    def test_unknown_kp_raises_keyerror(self, chain, ctx):
        with pytest.raises(KeyError):
            DiagnosisAgent(chain, ScriptedLLM([])).run("u1", "no.such", ctx)

    def test_temperature_is_forced_to_zero(self, chain, ctx):
        """采样抖动是评测不可复现的来源,必须在循环里压住。"""
        seen = {}

        class SpyLLM(ScriptedLLM):
            def chat_structured(self, messages, tools=None, **kwargs):
                seen.update(kwargs)
                return super().chat_structured(messages, tools=tools, **kwargs)

        DiagnosisAgent(chain, SpyLLM([submit("base")])).run("u1", "top", ctx)
        assert seen.get("temperature") == 0.0


class TestCheckpointResume:
    def test_resume_after_crash_does_not_redo_finished_round(self, chain, ctx, tmp_path, monkeypatch):
        """★ 崩溃在 tools 节点时,恢复应当从检查点续上,而不是从头重跑。

        断言方式:看**第二轮 LLM 调用收到的消息**里有没有第一轮的工具结果。
        从头重跑的话,第一次调用只会看到 system+user 两条。
        """
        checkpointer = make_checkpointer(tmp_path / "checkpoints.db")
        llm = ScriptedLLM([tool_response("get_prereq_closure", {"kp_id": "top"}), submit("base")])
        agent = DiagnosisAgent(chain, llm, checkpointer=checkpointer)

        def boom(name, arguments, context):
            raise RuntimeError("模拟进程被杀")

        monkeypatch.setattr(agent_tools, "execute_tool", boom)
        with pytest.raises(RuntimeError):
            agent.run("u1", "top", ctx)

        assert len(llm.calls) == 1, "第一轮 LLM 调用已完成"
        assert len(llm.calls[0]) == 2, "首轮只该看到 system + user"

        monkeypatch.undo()
        result = agent.run("u1", "top", ctx)

        assert result["_agent"]["terminated"] == "submitted"
        assert [r["kp_id"] for r in result["root_causes"]] == ["base"]
        assert len(llm.calls) == 2, "只该新增第二轮,不该重跑第一轮"
        assert any(m.get("role") == "tool" for m in llm.calls[1]), (
            "第二轮应看到第一轮的工具结果 —— 说明状态是从检查点续上的"
        )

    def test_same_learner_and_kp_resumes_same_thread(self, chain, ctx, tmp_path):
        """默认 thread_id 由 (learner, kp) 决定 —— 同一诊断崩溃后重跑会续上。"""
        checkpointer = make_checkpointer(tmp_path / "checkpoints.db")
        first = DiagnosisAgent(chain, ScriptedLLM([submit("base")]), checkpointer=checkpointer)
        result = first.run("u1", "top", ctx)

        assert result["_agent"]["terminated"] == "submitted"
        snapshot = first.graph.get_state({"configurable": {"thread_id": "diag:u1:top"}})
        assert snapshot.values["diagnosis"]["root_causes"][0]["kp_id"] == "base"

    def test_distinct_thread_ids_do_not_share_state(self, chain, ctx, tmp_path):
        """评测要每次全新,所以 thread_id 得能显式区分 —— 否则第二次会直接读上次的结果。"""
        checkpointer = make_checkpointer(tmp_path / "checkpoints.db")
        llm = ScriptedLLM([submit("base"), submit("mid")])
        agent = DiagnosisAgent(chain, llm, checkpointer=checkpointer)

        first = agent.run("u1", "top", ctx, thread_id="run-1")
        second = agent.run("u1", "top", ctx, thread_id="run-2")

        assert first["root_causes"][0]["kp_id"] == "base"
        assert second["root_causes"][0]["kp_id"] == "mid"
        assert len(llm.calls) == 2
