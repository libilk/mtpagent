"""只读门面:把「图 + 画像」组合成 API 需要的 JSON 结构。

**为什么需要这一层(一处文档张力,已记录在 work.md §12):**
硬约束说 `api/` 不许 import `coach.knowledge` 或 `coach.profile`,
但 `GET /gap`、`GET /graph` 又必须查图和画像。
折中:算法留在 `knowledge/queries.py`(纯函数,P2.1 的要求),
由本层负责取数据、拼结构,`api/` 只依赖本层。
这样 api 依旧不含业务逻辑,分层约束也字面成立。

本层**不调 LLM**:根因解释优先读 planner_worker 预先算好的缓存,
没有缓存就退化成模板文案。入口层因此永远不碰 LLM,保持 < 50ms。
"""

import datetime as dt
import time
from typing import Dict, List, Optional

from coach import config
from coach.knowledge import queries
from coach.knowledge.schema import open_db
from coach.knowledge.store import KnowledgeStore
from coach.profile import errors, sm2
from coach.profile.store import ProfileStore


class QueryService:
    def __init__(self, knowledge: KnowledgeStore, profile: ProfileStore):
        self.knowledge = knowledge
        self.profile = profile

    # ---------------- 根因定位(★ 核心)----------------

    def gap_view(
        self,
        learner_id: str,
        kp_id: str,
        top_k: int = config.ROOT_CAUSE_TOP_K,
        depth: int = config.MAX_PREREQ_DEPTH,
    ) -> Dict:
        """§5.5 的输出。给不出根因时 root_causes 为空,不报错。"""
        concept = self.knowledge.get_concept(kp_id)
        if concept is None:
            raise KeyError(kp_id)

        closure = queries.prereq_closure(self.knowledge, kp_id, depth=depth)
        mastery = self.profile.get_mastery_map(learner_id, list(closure))
        goal_mastery = self.profile.get_mastery(learner_id, kp_id)
        # ★ 把"有没有证据"一并交给查询层,否则未观测点会挤掉真正的弱项
        observed = self.profile.observed_kp_ids(learner_id, list(closure))

        roots = queries.root_causes(
            self.knowledge,
            mastery,
            kp_id,
            top_k=top_k,
            depth=depth,
            observed=observed,
        )

        cached = self.profile.get_gap_explanation(learner_id, kp_id)
        explanation = (
            cached["explanation"]
            if cached
            else queries.explain_gaps(kp_id, concept.name, goal_mastery, roots)
        )

        return {
            "learner_id": learner_id,
            "kp_id": kp_id,
            "name": concept.name,
            "mastery": goal_mastery,
            "root_causes": roots,
            "explanation": explanation,
            "explanation_source": "llm" if cached else "template",
            "explanation_pending": bool(roots) and cached is None,
        }

    def agent_gap_view(
        self,
        learner_id: str,
        kp_id: str,
        top_k: int = config.ROOT_CAUSE_TOP_K,
        depth: int = config.MAX_PREREQ_DEPTH,
    ) -> Dict:
        """agent 版根因诊断,契约与 `gap_view` **逐字节相同**(同一个 8 键结构)。

        ★ **只读缓存,不调 LLM。** agent 一次诊断要几秒到几十秒、还花钱,
        绝不能放在入口的同步路径上(那条线是 < 50ms)。它由 `planner_worker`
        在答题事件之后异步生成并落 `agent_diagnoses` 表。

        缓存未命中时返回空 `root_causes` + `explanation_pending=True` ——
        注意这里的 pending 指的是**整份诊断还没生成**,不只是解释文案。
        参数(top_k/depth)对不上也算未命中:`agent_diagnoses` 里存了它们,
        拿别的参数算出来的结果会误导人。
        """
        concept = self.knowledge.get_concept(kp_id)
        if concept is None:
            raise KeyError(kp_id)

        cached = self.profile.get_agent_diagnosis(learner_id, kp_id, top_k=top_k, depth=depth)
        if cached is None:
            return {
                "learner_id": learner_id,
                "kp_id": kp_id,
                "name": concept.name,
                "mastery": self.profile.get_mastery(learner_id, kp_id),
                "root_causes": [],
                "explanation": "",
                "explanation_source": "agent",
                "explanation_pending": True,
            }
        return {"learner_id": learner_id, **cached["payload"]}

    def learn_next(
        self,
        learner_id: str,
        goal_kp_id: str,
        top_k: int = config.ROOT_CAUSE_TOP_K,
        depth: int = config.MAX_PREREQ_DEPTH,
    ) -> List[Dict]:
        concept = self.knowledge.get_concept(goal_kp_id)
        if concept is None:
            raise KeyError(goal_kp_id)
        closure = queries.prereq_closure(self.knowledge, goal_kp_id, depth=depth)
        mastery = self.profile.get_mastery_map(learner_id, list(closure))
        return queries.next_to_learn(
            self.knowledge, mastery, goal_kp_id, top_k=top_k, depth=depth
        )

    # ---------------- 每日计划(P3)----------------

    def plan_view(
        self,
        learner_id: str,
        now: Optional[float] = None,
        daily_minutes: Optional[int] = None,
    ) -> Dict:
        """合并三类待办:到期复习 > 根因补救 > 推进新知识点。

        优先级按"时间敏感度"排:
        - review  到期了不复习就会忘,最急
        - remedial 是目标的根因缺口,不补学不动
        - learn   新知识点,有余力才推

        预算是**软的**:超预算就停,但至少留一项 —— 学习计划不该是空的。

        纯计算不调 LLM,所以可以在请求路径上直接跑(入口层保持 < 50ms)。
        """
        now = time.time() if now is None else now
        profile_row = self.profile.get_profile(learner_id) or {}
        budget = daily_minutes or profile_row.get("daily_minutes") or 30
        goal_kp_id = profile_row.get("goal_kp_id")

        items: List[Dict] = []
        seen: set = set()

        items.extend(self._due_items(learner_id, now, seen))
        if goal_kp_id and self.knowledge.get_concept(goal_kp_id):
            items.extend(self._goal_items(learner_id, goal_kp_id, seen))

        planned: List[Dict] = []
        used = 0.0
        for item in items:
            # 超预算就停,但至少留一项 —— 计划不该是空的
            if planned and used + item["est_minutes"] > budget:
                break
            planned.append(item)
            used += item["est_minutes"]

        return {
            "learner_id": learner_id,
            "date": dt.date.fromtimestamp(now).isoformat(),
            "goal_kp_id": goal_kp_id,
            "daily_minutes": budget,
            "planned_minutes": round(used, 1),
            "items": planned,
        }

    def _due_items(self, learner_id: str, now: float, seen: set) -> List[Dict]:
        items = []
        covered = self.knowledge.kp_ids_with_problems()
        for kp_id in self.profile.due_reviews(learner_id, now):
            concept = self.knowledge.get_concept(kp_id)
            if concept is None or kp_id in seen:
                continue
            # ★ 没有题可做就不推 —— 推了学生也无从练起(P5 里 82% 的步数是这么白费的)
            if covered and kp_id not in covered:
                continue
            memory = sm2.SM2State.from_row(self.profile.get_memory(learner_id, kp_id))
            overdue_days = (now - memory.due_at) / config.DAY_SECONDS if memory.due_at else 0.0
            items.append(
                {
                    "kp_id": kp_id,
                    "name": concept.name,
                    "action": "review",
                    "reason": f"到期复习:间隔 {memory.interval_days:.0f} 天,已逾期 {overdue_days:.1f} 天",
                    "est_minutes": config.PLAN_REVIEW_MINUTES,
                }
            )
            seen.add(kp_id)
        return items

    def _goal_items(self, learner_id: str, goal_kp_id: str, seen: set) -> List[Dict]:
        closure = queries.prereq_closure(self.knowledge, goal_kp_id)
        mastery = self.profile.get_mastery_map(learner_id, list(closure))
        observed = self.profile.observed_kp_ids(learner_id, list(closure))
        covered = self.knowledge.kp_ids_with_problems()
        goal_name = self.knowledge.get_concept(goal_kp_id).name

        def practicable(kp_id: str) -> bool:
            return not covered or kp_id in covered

        items = []
        for root in queries.root_causes(
            self.knowledge, mastery, goal_kp_id, observed=observed
        ):
            concept = self.knowledge.get_concept(root["kp_id"])
            if concept is None or root["kp_id"] in seen or not practicable(root["kp_id"]):
                continue
            flag = "已确认薄弱" if root.get("status") == "gap" else "还没测过"
            items.append(
                {
                    "kp_id": root["kp_id"],
                    "name": concept.name,
                    "action": "remedial",
                    "reason": f"「{goal_name}」的根因({flag},掌握度 {root['mastery']:.2f})",
                    "est_minutes": round(
                        concept.difficulty
                        * config.PLAN_LEARN_MINUTES_PER_DIFFICULTY
                        * config.PLAN_REMEDIAL_MULTIPLIER,
                        1,
                    ),
                }
            )
            seen.add(root["kp_id"])

        # ★ 「无缺口 → 推进目标」这一支(§2.2 的 advance 分支)
        # `next_to_learn` 只在前置闭包里找候选,**目标自己永远不在里面** ——
        # 于是前置都备齐之后计划会变成空的,学生不知道该干什么了。
        # 实测:24 步里 18 步是空的,这是计划跑不赢随机的真正原因。
        if goal_kp_id not in seen and practicable(goal_kp_id):
            goal_mastery = self.profile.get_mastery(learner_id, goal_kp_id)
            if goal_mastery < config.GAP_THRESHOLD:
                goal_prereqs = self.knowledge.prerequisite_adjacency().get(goal_kp_id, [])
                blocked = [
                    p
                    for p in goal_prereqs
                    if mastery.get(p, 0.0) < config.GAP_THRESHOLD and p in observed
                ]
                needs_probe = any(
                    mastery.get(p, 0.0) < config.GAP_THRESHOLD and p not in observed
                    for p in goal_prereqs
                )
                if not blocked:
                    goal_concept = self.knowledge.get_concept(goal_kp_id)
                    items.append(
                        {
                            "kp_id": goal_kp_id,
                            "name": goal_concept.name,
                            "action": "probe" if needs_probe else "learn",
                            "reason": (
                                "前置里还有点没测过,先摸个底"
                                if needs_probe
                                else "前置已备齐,可以直接攻目标了"
                            ),
                            "est_minutes": round(
                                goal_concept.difficulty
                                * config.PLAN_LEARN_MINUTES_PER_DIFFICULTY,
                                1,
                            ),
                        }
                    )
                    seen.add(goal_kp_id)

        for candidate in queries.next_to_learn(
            self.knowledge, mastery, goal_kp_id, observed=observed
        ):
            concept = self.knowledge.get_concept(candidate["kp_id"])
            if concept is None or candidate["kp_id"] in seen or not practicable(candidate["kp_id"]):
                continue
            # next_to_learn 会把"前置还没测过"的标成 probe(先测再学)
            is_probe = candidate.get("action") == "probe"
            items.append(
                {
                    "kp_id": candidate["kp_id"],
                    "name": concept.name,
                    "action": "probe" if is_probe else "learn",
                    "reason": "前置里还有点没测过,先摸个底" if is_probe else "前置已备齐,可以推进",
                    "est_minutes": round(
                        concept.difficulty * config.PLAN_LEARN_MINUTES_PER_DIFFICULTY, 1
                    ),
                }
            )
            seen.add(candidate["kp_id"])
        return items

    # ---------------- 学习者档案(P1 的契约欠账,2026-09-13 补上)----------------

    def profile_view(self, learner_id: str, top_k: int = 5) -> Dict:
        """§5.3 契约:`{goal, mastery_summary, due_now, error_patterns}`。

        只读、纯计算,入口层可以放心同步调用。
        """
        row = self.profile.get_profile(learner_id) or {}
        goal_kp_id = row.get("goal_kp_id")

        mastered = self.profile.get_mastery_map(learner_id)
        mastery_summary = {
            "observed_count": len(mastered),
            "mastered_count": sum(1 for v in mastered.values() if v >= config.GAP_THRESHOLD),
            "mean": round(sum(mastered.values()) / len(mastered), 4) if mastered else None,
            "lowest": sorted(
                (
                    {"kp_id": kp, "name": self._name(kp), "mastery": round(v, 4)}
                    for kp, v in mastered.items()
                    if v < config.GAP_THRESHOLD
                ),
                key=lambda item: item["mastery"],
            )[:top_k],
        }

        due_now = [
            {"kp_id": kp, "name": self._name(kp)}
            for kp in self.profile.due_reviews(learner_id)
        ]

        goal = None
        if goal_kp_id:
            concept = self.knowledge.get_concept(goal_kp_id)
            goal = {
                "kp_id": goal_kp_id,
                "name": concept.name if concept else goal_kp_id,
                "mastery": round(self.profile.get_mastery(learner_id, goal_kp_id), 4),
            }

        return {
            "learner_id": learner_id,
            "goal": goal,
            "daily_minutes": row.get("daily_minutes"),
            "mastery_summary": mastery_summary,
            "due_now": due_now,
            "error_patterns": errors.summary(self.profile, learner_id, limit=top_k),
        }

    def _name(self, kp_id: str) -> str:
        concept = self.knowledge.get_concept(kp_id)
        return concept.name if concept else kp_id

    # ---------------- 图谱视图 ----------------

    def graph_view(
        self,
        kp_id: str,
        depth: int = config.MAX_PREREQ_DEPTH,
        learner_id: Optional[str] = None,
    ) -> Dict:
        """前置树 + 已掌握 / 缺口标记。不传 learner_id 就只给结构。"""
        concept = self.knowledge.get_concept(kp_id)
        if concept is None:
            raise KeyError(kp_id)

        closure = queries.prereq_closure(self.knowledge, kp_id, depth=depth)
        mastery: Dict[str, float] = {}
        if learner_id:
            mastery = self.profile.get_mastery_map(learner_id, list(closure))

        threshold = config.GAP_THRESHOLD
        mastered, gaps = [], []
        for kp, d in sorted(closure.items(), key=lambda kv: (kv[1], kv[0])):
            m = mastery.get(kp)
            if m is None:
                continue
            (gaps if m < threshold else mastered).append(
                {"kp_id": kp, "depth": d, "mastery": m}
            )

        return {
            "node": {
                "kp_id": kp_id,
                "name": concept.name,
                "subject": concept.subject,
                "difficulty": concept.difficulty,
                "mastery": mastery.get(kp_id),
            },
            "prereq_tree": self._build_tree(kp_id, depth, closure, mastery, threshold, set()),
            "mastered": mastered,
            "gaps": gaps,
            "learner_id": learner_id,
        }

    def _build_tree(
        self,
        node: str,
        remaining: int,
        closure: Dict[str, int],
        mastery: Dict[str, float],
        threshold: float,
        on_path: set,
    ) -> Dict:
        concept = self.knowledge.get_concept(node)
        m = mastery.get(node)
        entry = {
            "kp_id": node,
            "name": concept.name if concept else node,
            "depth": closure.get(node, 0),
            "mastery": m,
            "is_gap": (m is not None and m < threshold),
            "children": [],
        }
        if remaining <= 0:
            return entry

        on_path = on_path | {node}  # 每个分支各自记路径,菱形依赖允许重复出现
        for prereq in self.knowledge.ancestors(node, depth=1):
            if prereq in closure and prereq not in on_path:
                entry["children"].append(
                    self._build_tree(prereq, remaining - 1, closure, mastery, threshold, on_path)
                )
        return entry

    def close(self) -> None:
        self.knowledge.close()


def build_default_query_service(db_path=None) -> QueryService:
    """装配一个连真实库的只读门面。

    放在 workflow 层而不是 api 层,是为了让 `api/` 只依赖本模块,
    从而满足「api 不许 import knowledge/profile」的硬约束。
    连接用 check_same_thread=False:FastAPI 的同步路由跑在线程池里。
    """
    conn = open_db(db_path, check_same_thread=False)
    return QueryService(KnowledgeStore(conn), ProfileStore(conn))
