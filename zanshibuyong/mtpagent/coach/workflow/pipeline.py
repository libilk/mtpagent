"""答题处理链(work.md §2.2 / P4):LangGraph + checkpointer。

**为什么 P4 才引入 LangGraph**(§6 P4 的叙事):
P0~P3 用「事件 + 各 worker 自己做几步」就够了 —— 短路、好调、无依赖。
但链一长就出问题:进程在链中间被杀,恢复是**从事件头重放**,前面已经算完的
步骤(尤其是调 LLM 生成解释)会被重做一遍。checkpointer 把「每个节点之后的状态」
落盘,恢复时从最后一个成功的节点接着跑,而不是从头再来。

**节点划分(与 §2.2 的 [1]~[6] 有出入,理由见下):**

    START → grade → update_profile → detect_gap ─┬→ explain → finalize → END
                                                 └──────────→ finalize → END

- §2.2 把 [2] BKT 和 [3] SM-2 列为两步,这里合成 `update_profile` 一个节点:
  这一组写操作必须**原子**(见 profile/store.py 的事务),拆成两个节点会让
  "BKT 写了、SM-2 没写"成为可能。
- `explain` 单独成节点,是因为它**贵且非幂等**(调 LLM)。单独成节点的意义是:
  它的产出会先被 checkpoint 落盘,之后才进 finalize —— 这样 finalize 阶段崩了,
  恢复时不会重调 LLM。这正是 P4 要证明的事。
- `detect_gap` 之后是条件边:有缺口才去解释,没缺口直接收尾。
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from coach import config
from coach.domain.grading import grade
from coach.events import schema as events
from coach.events.schema import Event
from coach.profile import bkt, sm2
from coach.profile.store import ProfileStore
from coach.workflow.query import QueryService

logger = logging.getLogger(__name__)

EXPLAIN_SYSTEM_PROMPT = (
    "你是一位算法课老师。请用 2~3 句中文,直接告诉学生他真正薄弱的前置基础是什么,"
    "为什么这个基础会导致他当前学不下去,以及应该先补哪一个。"
    "像跟人说话一样,不要罗列数据,不要用表格,不要复述题目。"
)


class AnswerState(TypedDict, total=False):
    """链上流转的状态。total=False —— 每个节点只需返回自己产出的字段。"""

    # 输入
    event_id: str
    event_ts: float
    learner_id: str
    problem_id: str
    answer_text: str
    elapsed_ms: int
    # 中间产物
    correct: bool
    kp_ids: List[str]
    mastery: Dict[str, float]
    due_at: Dict[str, float]
    gaps: List[Dict[str, Any]]
    explained: List[str]
    # 结果
    status: str
    reason: str


# ---------------------------------------------------------------- 解释生成
# pipeline 的 explain 节点和 planner_worker 共用这一段,避免两处各写一遍。

def explain_gaps_with_llm(llm, view: Dict[str, Any]) -> str:
    lines = [
        f"目标知识点:{view['name']}(当前掌握度 {view['mastery']:.2f})",
        "前置缺口(按重要度排序):",
    ]
    for i, gap in enumerate(view["root_causes"], 1):
        chain = " → ".join(gap.get("path") or [])
        lines.append(
            f"{i}. {gap['name']} 掌握度 {gap['mastery']:.2f}"
            + (f",依赖路径 {chain}" if chain else "")
        )
    return llm.chat(
        [
            {"role": "system", "content": EXPLAIN_SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(lines)},
        ]
    )


def refresh_explanation(
    service: QueryService,
    profile: ProfileStore,
    llm,
    learner_id: str,
    kp_id: str,
    not_before: Optional[float] = None,
) -> bool:
    """重算根因并把解释写进缓存。返回是否真的写了。

    not_before: 缓存生成时间不早于这个时间戳就跳过(避免 pipeline 刚写完、
    planner 又调一次 LLM 重复生成)。
    """
    if not_before is not None:
        cached = profile.get_gap_explanation(learner_id, kp_id)
        if cached and cached.get("generated_at") and cached["generated_at"] >= not_before:
            return False

    try:
        view = service.gap_view(learner_id, kp_id)
    except KeyError:
        logger.warning("知识点不存在,跳过解释:%s", kp_id)
        return False

    if not view["root_causes"] or llm is None:
        return False

    try:
        text = explain_gaps_with_llm(llm, view)
    except Exception:  # noqa: BLE001 — LLM 失败不该拖垮整条链
        logger.exception("生成根因解释失败:%s/%s", learner_id, kp_id)
        return False

    profile.set_gap_explanation(learner_id, kp_id, text)
    return True


# ---------------------------------------------------------------- 节点

class AnswerPipeline:
    """把一次答题的完整处理链编译成图,并负责执行。

    依赖都从构造器注入,节点函数只认这些依赖 —— 所以测试可以塞假总线、
    假 LLM、内存 checkpointer,不需要 Redis 也不要 API key。
    """

    def __init__(
        self,
        knowledge,
        profile: ProfileStore,
        journal,
        llm=None,
        bus=None,
        service: Optional[QueryService] = None,
        checkpointer=None,
    ):
        self.knowledge = knowledge
        self.profile = profile
        self.journal = journal
        self.llm = llm
        self.bus = bus
        self.service = service or QueryService(knowledge, profile)
        self.checkpointer = checkpointer
        self.graph = self._build()

    # ---------------- 图 ----------------

    def _build(self):
        graph = StateGraph(AnswerState)
        graph.add_node("grade", self._grade)
        graph.add_node("update_profile", self._update_profile)
        graph.add_node("detect_gap", self._detect_gap)
        graph.add_node("explain", self._explain)
        graph.add_node("finalize", self._finalize)

        graph.add_edge(START, "grade")
        graph.add_edge("grade", "update_profile")
        graph.add_edge("update_profile", "detect_gap")
        graph.add_conditional_edges(
            "detect_gap",
            _route_after_gap,
            {"explain": "explain", "finalize": "finalize"},
        )
        graph.add_edge("explain", "finalize")
        graph.add_edge("finalize", END)
        return graph.compile(checkpointer=self.checkpointer)

    # ---------------- 执行 ----------------

    def run(self, event: Event) -> Dict[str, Any]:
        """跑一次答题链。

        **恢复语义(踩过坑,写清楚):** langgraph 记住的是「下一步该跑谁」。
        要接着跑,必须 `invoke(None, config=cfg)` —— 传一份新的 state 会让它
        **从头开始**,那 checkpoint 就白存了。所以下面先看有没有未完成的节点。
        """
        if self.profile.is_processed(event.event_id):
            return {"status": "skipped", "reason": "already_processed"}

        self.journal.record(event)  # 先记后做

        config_dict = {"configurable": {"thread_id": event.event_id}}

        if self.checkpointer is not None:
            snapshot = self.graph.get_state(config_dict)
            if snapshot.next:
                logger.info(
                    "从检查点恢复 event=%s,待跑节点=%s", event.event_id, snapshot.next
                )
                return _summarize(self.graph.invoke(None, config=config_dict))

        state: AnswerState = {
            "event_id": event.event_id,
            "event_ts": event.ts,
            "learner_id": event.learner_id,
            "problem_id": event.payload.get("problem_id", ""),
            "answer_text": event.payload.get("answer_text", ""),
            "elapsed_ms": int(event.payload.get("elapsed_ms", 0)),
        }
        return _summarize(self.graph.invoke(state, config=config_dict))

    # ---------------- 节点实现 ----------------

    def _grade(self, state: AnswerState) -> Dict[str, Any]:
        problem = self.knowledge.get_problem(state["problem_id"])
        if problem is None:
            logger.warning("题目不存在:%s", state["problem_id"])
            return {"status": "problem_not_found", "reason": state["problem_id"]}

        try:
            correct = grade(problem, state["answer_text"])
        except ValueError as exc:
            logger.warning("无法判分:%s", exc)
            return {"status": "ungradeable", "reason": str(exc)}

        return {"correct": correct, "kp_ids": list(problem.kp_ids)}

    def _update_profile(self, state: AnswerState) -> Dict[str, Any]:
        """答题记录 + BKT + SM-2,一个事务里做完(要么全成要么全不成)。"""
        if state.get("status") in ("problem_not_found", "ungradeable"):
            return {}

        learner_id = state["learner_id"]
        kp_ids = state.get("kp_ids") or []
        correct = bool(state.get("correct"))
        now = time.time()
        quality = sm2.quality_from_correct(correct)

        with self.profile.transaction():
            inserted = self.profile.record_answer(
                answer_id=state["event_id"],
                learner_id=learner_id,
                problem_id=state["problem_id"],
                kp_ids=kp_ids,
                correct=correct,
                answer_text=state["answer_text"],
                elapsed_ms=state.get("elapsed_ms", 0),
            )
            if not inserted:
                # 第二道幂等闸:这条答题已经计过画像(检查点丢了时的兜底)
                return {"status": "duplicate_answer"}

            mastery = self.profile.get_mastery_map(learner_id, kp_ids)
            due_at: Dict[str, float] = {}
            for kp_id in kp_ids:
                new_p = bkt.update(mastery.get(kp_id), correct)
                mastery[kp_id] = new_p
                self.profile.set_mastery(learner_id, kp_id, new_p)

                memory = sm2.review(
                    sm2.SM2State.from_row(self.profile.get_memory(learner_id, kp_id)),
                    quality,
                    now,
                )
                self.profile.set_memory(
                    learner_id,
                    kp_id,
                    ease=memory.ease,
                    interval_days=memory.interval_days,
                    reps=memory.reps,
                    lapses=memory.lapses,
                    last_review=memory.last_review,
                    due_at=memory.due_at,
                )
                due_at[kp_id] = memory.due_at

                if not correct:
                    self.profile.bump_error(learner_id, kp_id, "wrong")

        return {"mastery": mastery, "due_at": due_at, "status": "updated"}

    def _detect_gap(self, state: AnswerState) -> Dict[str, Any]:
        """对这次答题涉及的知识点各查一次前置缺口。"""
        if state.get("status") != "updated":
            return {"gaps": []}

        learner_id = state["learner_id"]
        gaps: List[Dict[str, Any]] = []
        for kp_id in state.get("kp_ids") or []:
            try:
                view = self.service.gap_view(learner_id, kp_id)
            except KeyError:
                continue
            if view["root_causes"]:
                gaps.append({"kp_id": kp_id, "view": view})
        return {"gaps": gaps}

    def _explain(self, state: AnswerState) -> Dict[str, Any]:
        """★ 贵且非幂等的一步:单独成节点,产出先落检查点,崩了不用重调 LLM。"""
        if not state.get("gaps") or self.llm is None:
            return {"explained": []}

        learner_id = state["learner_id"]
        explained = []
        for gap in state["gaps"]:
            kp_id = gap["kp_id"]
            try:
                text = explain_gaps_with_llm(self.llm, gap["view"])
            except Exception:  # noqa: BLE001
                logger.exception("生成根因解释失败:%s/%s", learner_id, kp_id)
                continue
            self.profile.set_gap_explanation(learner_id, kp_id, text)
            explained.append(kp_id)
        return {"explained": explained}

    def _finalize(self, state: AnswerState) -> Dict[str, Any]:
        event_id = state["event_id"]
        self.profile.mark_processed(event_id)
        self.journal.mark_done(event_id)
        self._publish_profile_updated(state)
        return {"status": state.get("status", "updated")}

    def _publish_profile_updated(self, state: AnswerState) -> None:
        if self.bus is None or state.get("status") != "updated":
            return
        update = events.new_event(
            events.PROFILE_UPDATED,
            learner_id=state["learner_id"],
            payload={
                "kp_ids": list(state.get("kp_ids") or []),
                "mastery": state.get("mastery") or {},
                "source_event": state["event_id"],
                # 原始答题时间:下游据此判断解释缓存够不够新,避免重复调 LLM
                "since": state.get("event_ts"),
            },
            trace_id=state["event_id"],
        )
        try:
            self.bus.publish(config.STREAM_PROFILE, update)
        except Exception:  # noqa: BLE001 — 下游订阅失败不该影响答题处理
            logger.exception("发布 profile.updated 失败")


def _route_after_gap(state: AnswerState) -> str:
    return "explain" if state.get("gaps") else "finalize"


# ---------------------------------------------------------------- checkpointer

def make_checkpointer(path=None):
    """造检查点存储。

    - path 为 None → InMemorySaver(测试用,进程退出即丢)
    - 否则 → SqliteSaver,单独一个库文件

    **为什么检查点单独一个文件**:图/画像/journal 共用一个连接,而 SqliteSaver
    自己会 commit。放在同一个连接上,它的提交会把我们事务里的半成品一起提交掉,
    原子性就没了。分开存互不干扰。
    """
    if path is None:
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver()

    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver

    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    return SqliteSaver(conn)


def _summarize(state: Dict[str, Any]) -> Dict[str, Any]:
    """把最终状态裁成调用方关心的摘要(不要整份 state 漏出去)。"""
    mastery = state.get("mastery") or {}
    return {
        "status": state.get("status", "updated"),
        "correct": state.get("correct"),
        "reason": state.get("reason"),
        "mastery": {k: round(v, 4) for k, v in mastery.items()},
        "due_at": state.get("due_at") or {},
        "gaps": [g["kp_id"] for g in state.get("gaps") or []],
        "explained": state.get("explained") or [],
    }
