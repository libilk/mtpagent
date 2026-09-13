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
from coach.knowledge.schema import connect
from coach.knowledge.store import KnowledgeStore
from coach.profile import sm2
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

        roots = queries.root_causes(
            self.knowledge, mastery, kp_id, top_k=top_k, depth=depth
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
        for kp_id in self.profile.due_reviews(learner_id, now):
            concept = self.knowledge.get_concept(kp_id)
            if concept is None or kp_id in seen:
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

        items = []
        for root in queries.root_causes(self.knowledge, mastery, goal_kp_id):
            concept = self.knowledge.get_concept(root["kp_id"])
            if concept is None or root["kp_id"] in seen:
                continue
            items.append(
                {
                    "kp_id": root["kp_id"],
                    "name": concept.name,
                    "action": "remedial",
                    "reason": f"「{self.knowledge.get_concept(goal_kp_id).name}」的根因缺口(掌握度 {root['mastery']:.2f})",
                    "est_minutes": round(
                        concept.difficulty
                        * config.PLAN_LEARN_MINUTES_PER_DIFFICULTY
                        * config.PLAN_REMEDIAL_MULTIPLIER,
                        1,
                    ),
                }
            )
            seen.add(root["kp_id"])

        for candidate in queries.next_to_learn(self.knowledge, mastery, goal_kp_id):
            concept = self.knowledge.get_concept(candidate["kp_id"])
            if concept is None or candidate["kp_id"] in seen:
                continue
            items.append(
                {
                    "kp_id": candidate["kp_id"],
                    "name": concept.name,
                    "action": "learn",
                    "reason": "前置已备齐,可以推进",
                    "est_minutes": round(
                        concept.difficulty * config.PLAN_LEARN_MINUTES_PER_DIFFICULTY, 1
                    ),
                }
            )
            seen.add(candidate["kp_id"])
        return items

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
    conn = connect(db_path, check_same_thread=False)
    return QueryService(KnowledgeStore(conn), ProfileStore(conn))
