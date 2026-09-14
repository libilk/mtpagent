"""agent 版根因诊断器 —— 手写 ReAct 循环(agent 实验 M2)。

**它和 `queries.root_causes` 的关系:并列,不是替代。**
确定性那条路径一行不动(它的 top-1 0.60 是对照基线),这里只是多给一个由 LLM
自己决定「调哪些工具、按什么顺序排」的诊断器。

**为什么手写而不用 `create_react_agent`:** 那玩意把循环机制藏起来了 ——
而这个模块的**首要目的是能读懂**,不是少写二十行。

**为什么坚持挂在 LangGraph 上:** 白拿已有的 checkpointer。一轮崩了,
前面的工具调用与 LLM 调用不重跑。两个企业级对标项目(Tencent WeSmartFlow /
HKUDS DeepTutor)的 agent 状态**都没有检查点**,这里反而领先。

三条实现约束(都对应一个真实会踩的坑):

1. **★ `DiagnosisContext` 绝不进 state。** checkpointer 会把 state 序列化落盘,
   而上下文里攥着 sqlite 连接 → 直接炸。它通过**节点闭包**注入,
   state 里只放可序列化的东西(消息、步数、工具名)。
2. **消息用普通 dict,不用 `add_messages`。** `chat_structured` 收的就是
   `List[Dict]`;循环是线性的,不需要 reducer 的合并/去重语义,转换反而多一层。
3. **`temperature=0` 在循环里强制。** 生产的 `LLM(temperature=0.7)` 也该得到
   确定性的诊断器。这是唯一能压住采样抖动的地方。

输出契约与 `query.gap_view` **逐字节一致**(8 个 key),另有 `_agent` 一个
下划线开头的元数据槽放步数/工具轨迹 —— 入口层会把它剥掉。
"""

import json
import logging
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from coach import config
from coach.workflow import agent_tools
from coach.workflow.agent_tools import TOOL_SCHEMAS, DiagnosisContext

logger = logging.getLogger(__name__)

#: 循环上限。超了就带着现有的东西收尾,不让它无限跑。
MAX_STEPS = 8

EXPLAIN_SYSTEM_PROMPT = """你是学习诊断助手。学生的某个知识点学不会,你要找出他**真正缺的那个前置基础**。

工作方式:
1. 先用 get_prereq_closure 看这个知识点的前置依赖结构。
2. 用 get_mastery 查这些前置的掌握度。**关键**:返回里的 `observed` 字段表示
   这个点有没有作答证据。`observed=false` 的点掌握度只是初始值,**不代表学生真的弱**;
   优先怀疑 `observed=true` 且掌握度低的点。两者数值可能一样,含义完全不同。
3. 需要时再查依赖路径(get_shortest_path)、有没有题可做(list_problems_for_kp)、
   易错模式(get_error_patterns)。
4. 判断完成后调用 submit_diagnosis 交卷:root_causes 按"是根因的可能性"从高到低排,
   summary 写一段给学生看的人话。

纪律:
- 每个结论都要有工具返回的数据支撑,不要凭常识臆测。
- 不要调用没有必要的工具;数据够了就交卷。
"""


class DiagnosisState(TypedDict, total=False):
    """链上流转的状态。**只放可序列化的东西** —— 见模块顶部的约束 1。"""

    messages: List[Dict[str, Any]]
    steps: int
    tools_used: List[str]
    diagnosis: Dict[str, Any]
    terminated: str


def _route_after_agent(state: DiagnosisState, max_steps: int) -> str:
    """交卷了 / 模型不再调工具 / 步数用尽 → 收尾;否则去执行工具。

    `max_steps` 必须由调用方传进来 —— 曾经把它写成模块常量,
    结果实例上传的 `max_steps` 根本不生效(上限永远是 8),测试抓出来的。
    """
    if state.get("diagnosis"):
        return "end"
    messages = state.get("messages") or []
    if not messages or not messages[-1].get("tool_calls"):
        return "end"
    if state.get("steps", 0) >= max_steps:
        return "end"
    return "tools"


class DiagnosisAgent:
    """一次诊断 = 一个 LangGraph 线程。"""

    def __init__(
        self,
        knowledge,
        llm,
        checkpointer=None,
        max_steps: int = MAX_STEPS,
        temperature: float = 0.0,
    ):
        self.knowledge = knowledge
        self.llm = llm
        self.checkpointer = checkpointer
        self.max_steps = max_steps
        self.temperature = temperature
        self._context: Optional[DiagnosisContext] = None
        self._target = {"kp_id": "", "top_k": config.ROOT_CAUSE_TOP_K, "depth": config.MAX_PREREQ_DEPTH}
        self.graph = self._build()

    # ---------------- 图 ----------------

    def _build(self):
        graph = StateGraph(DiagnosisState)
        graph.add_node("agent", self._agent)
        graph.add_node("tools", self._tools)
        graph.add_edge(START, "agent")
        graph.add_conditional_edges(
            "agent", lambda state: _route_after_agent(state, self.max_steps), {"tools": "tools", "end": END}
        )
        graph.add_edge("tools", "agent")
        return graph.compile(checkpointer=self.checkpointer)

    def _agent(self, state: DiagnosisState) -> Dict[str, Any]:
        """一轮 = 一次带工具声明的 LLM 调用。"""
        messages = list(state.get("messages") or [])
        response = self.llm.chat_structured(messages, tools=TOOL_SCHEMAS, temperature=self.temperature)

        message: Dict[str, Any] = {"role": "assistant", "content": response.content or ""}
        if response.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(call.arguments, ensure_ascii=False)},
                }
                for call in response.tool_calls
            ]
        messages.append(message)

        # 工具轨迹在**请求时**就记下来(含终结工具)—— 它反映的是模型的选择,
        # 而不是我们执行了哪些。这对 M4 的"调了几次工具"指标才有意义。
        update: Dict[str, Any] = {
            "messages": messages,
            "steps": state.get("steps", 0) + 1,
            "tools_used": list(state.get("tools_used") or []) + [call.name for call in response.tool_calls],
        }

        # 终结工具在这里**试解**:解得开就交卷,解不开就当没交,让 tools 节点回一条错误
        for call in response.tool_calls:
            if agent_tools.is_terminal(call.name):
                try:
                    update["diagnosis"] = agent_tools.parse_diagnosis(
                        call.arguments,
                        self._context,
                        target_kp=self._target["kp_id"],
                        top_k=self._target["top_k"],
                        depth=self._target["depth"],
                    )
                except ValueError as exc:
                    logger.warning("交卷参数不合法,让模型重试:%s", exc)
        return update

    def _tools(self, state: DiagnosisState) -> Dict[str, Any]:
        """执行上一轮请求的工具,把结果作为 role=tool 的消息回灌。

        参数校验失败**返回错误而不抛** —— 模型看见错误会自我纠正,循环能继续
        (`llm_client` 对坏 JSON 会静默给 `{}`,所以这层必须挡)。
        """
        messages = list(state.get("messages") or [])
        last = messages[-1] if messages else {}

        for call in last.get("tool_calls") or []:
            name = call["function"]["name"]
            arguments = _loads(call["function"].get("arguments"))
            if agent_tools.is_terminal(name):
                text = (
                    "已收到你的诊断。"
                    if state.get("diagnosis")
                    else json.dumps({"error": "交卷参数不合法(需要非空的 root_causes 数组),请修正后重试。"}, ensure_ascii=False)
                )
            else:
                text = agent_tools.execute_tool(name, arguments, self._context)
            messages.append({"role": "tool", "tool_call_id": call.get("id") or name, "content": text})

        return {"messages": messages}

    # ---------------- 执行 ----------------

    def run(
        self,
        learner_id: str,
        kp_id: str,
        context: DiagnosisContext,
        top_k: int = config.ROOT_CAUSE_TOP_K,
        depth: int = config.MAX_PREREQ_DEPTH,
        thread_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """跑一次诊断,返回与 `gap_view` 同构的 8 个 key(+ `_agent` 元数据)。

        `thread_id` 默认由 (learner_id, kp_id) 决定 —— 也就是说**同一个人同一个知识点
        崩溃后重跑会从检查点续上**。评测若要每次全新,显式传不同的 thread_id。
        """
        if self.llm is None:
            raise ValueError("agent 诊断需要 LLM;llm=None 时请用确定性路径 queries.root_causes")

        concept = self.knowledge.get_concept(kp_id)
        if concept is None:
            raise KeyError(kp_id)

        self._context = context
        self._target = {"kp_id": kp_id, "top_k": top_k, "depth": depth}
        thread_id = thread_id or f"diag:{learner_id}:{kp_id}"
        config_dict = {"configurable": {"thread_id": thread_id}}

        if self.checkpointer is not None:
            snapshot = self.graph.get_state(config_dict)
            if snapshot.next:
                logger.info("从检查点恢复诊断 thread=%s,待跑节点=%s", thread_id, snapshot.next)
                return self._summarize(self.graph.invoke(None, config=config_dict), learner_id, concept)

        state: DiagnosisState = {
            "messages": [
                {"role": "system", "content": EXPLAIN_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"学生 {learner_id} 卡在「{concept.name}」({kp_id})。"
                        f"请找出他真正缺的前置基础,最多给 {top_k} 个。"
                    ),
                },
            ],
            "steps": 0,
            "tools_used": [],
        }
        result = self.graph.invoke(state, config=config_dict)
        return self._summarize(result, learner_id, concept)

    def _summarize(self, state: DiagnosisState, learner_id: str, concept) -> Dict[str, Any]:
        diagnosis = state.get("diagnosis") or {}
        roots = diagnosis.get("root_causes") or []
        context = self._context
        terminated = (
            "submitted"
            if diagnosis
            else ("step_limit" if state.get("steps", 0) >= self.max_steps else "no_submission")
        )
        return {
            # ---- 与 gap_view 逐字节一致的 8 个 key ----
            "learner_id": learner_id,
            "kp_id": concept.id,
            "name": concept.name,
            "mastery": round(context.mastery_of(concept.id), 4),
            "root_causes": roots,
            "explanation": diagnosis.get("summary") or "",
            "explanation_source": "agent",
            "explanation_pending": False,
            # ---- 下划线开头:入口层会剥掉的元数据 ----
            "_agent": {
                "steps": state.get("steps", 0),
                "tools_used": list(state.get("tools_used") or []),
                "terminated": terminated,
            },
        }


def _loads(raw: Any) -> Dict[str, Any]:
    """模型给的参数是 JSON 字符串;解析失败就当空——交给 execute_tool 报缺参数。"""
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def make_context_for(knowledge, profile, learner_id: str) -> DiagnosisContext:
    """生产路径的便捷装配(评测路径用 `DiagnosisContext.from_maps`)。"""
    return DiagnosisContext.from_stores(knowledge, profile, learner_id)


def refresh_agent_diagnosis(
    knowledge,
    profile,
    llm,
    learner_id: str,
    kp_id: str,
    top_k: int = config.ROOT_CAUSE_TOP_K,
    depth: int = config.MAX_PREREQ_DEPTH,
    not_before: Optional[float] = None,
) -> bool:
    """异步生成一份 agent 诊断并落缓存。返回是否真的写了。

    这是 `pipeline.refresh_explanation` 的同一套做法(新鲜度闸门 → 生成 → 落缓存),
    理由也一样:**入口层不许碰 LLM**。

    ⚠️ 一次诊断要好几轮 LLM 调用,所以调用方必须**显式开启**(见 planner_worker 的
    `agent_diagnosis` 开关),不能默认挂在每条事件上。
    """
    if llm is None:
        return False
    if not_before is not None:
        cached = profile.get_agent_diagnosis(learner_id, kp_id, top_k=top_k, depth=depth)
        if cached and cached.get("generated_at") and cached["generated_at"] >= not_before:
            return False

    agent = DiagnosisAgent(knowledge, llm)
    try:
        result = agent.run(
            learner_id,
            kp_id,
            DiagnosisContext.from_stores(knowledge, profile, learner_id),
            top_k=top_k,
            depth=depth,
        )
    except Exception as exc:  # noqa: BLE001 —— 诊断失败不该影响答题链
        logger.warning("agent 诊断失败 learner=%s kp=%s:%s", learner_id, kp_id, exc)
        return False

    payload = {key: value for key, value in result.items() if not key.startswith("_")}
    payload.pop("learner_id", None)
    profile.set_agent_diagnosis(learner_id, kp_id, payload, top_k=top_k, depth=depth)
    return True


def main(argv: Optional[List[str]] = None) -> int:
    """真 LLM 冒烟(M3):一个学生一个知识点,跑一次诊断并把过程打出来。"""
    import argparse
    import sys

    parser = argparse.ArgumentParser(prog="coach.workflow.agent_diagnosis", description="agent 诊断冒烟")
    parser.add_argument("--learner", default="u1")
    parser.add_argument("--goal", default="algo.dp")
    parser.add_argument("--db", default=None)
    parser.add_argument("--no-llm", action="store_true", help="不调真 LLM(只验证装配)")
    args = parser.parse_args(argv)

    from coach.knowledge.schema import open_db
    from coach.knowledge.store import KnowledgeStore
    from coach.profile.store import ProfileStore

    conn = open_db(args.db, check_same_thread=False)
    knowledge, profile = KnowledgeStore(conn), ProfileStore(conn)
    context = make_context_for(knowledge, profile, args.learner)

    llm = None
    if not args.no_llm:
        from llm.llm_client import LLM

        llm = LLM()

    agent = DiagnosisAgent(knowledge, llm)
    try:
        result = agent.run(args.learner, args.goal, context)
    except (ValueError, KeyError) as exc:
        print(f"跑不了:{exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()

    meta = result.pop("_agent")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n步数 {meta['steps']} · 工具轨迹 {meta['tools_used']} · 结束方式 {meta['terminated']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
