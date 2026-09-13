"""多跳图查询与根因定位(work.md §5.5,**项目核心**)。

这一层是**纯函数**:只接收「图 + 一份掌握度映射」,不认识 ProfileStore。
理由:knowledge/ 不许 import profile/(分层硬约束)。掌握度由上层批量取好传进来。

根因定位的直觉:
    学生做错「动态规划」,根因往往不在动态规划本身,
    而在它的前置里某个更浅、更弱的知识点。
    扁平检索只能推"相似题目",做不到这件事——
    **只有沿前置依赖反向遍历,才能找到真正缺的那个基础**。
"""

from typing import Dict, List, Optional

from coach import config
from coach.knowledge.store import KnowledgeStore


def prereq_closure(
    store: KnowledgeStore, kp_id: str, depth: int = config.MAX_PREREQ_DEPTH
) -> Dict[str, int]:
    """X 的前置闭包:{kp_id: 最短深度}。直接转发 §5.1 的递归 CTE。"""
    return store.ancestors(kp_id, depth=depth)


def detect_gaps(
    store: KnowledgeStore,
    mastery: Dict[str, float],
    kp_id: str,
    depth: int = config.MAX_PREREQ_DEPTH,
    threshold: float = config.GAP_THRESHOLD,
) -> List[Dict]:
    """找出 X 前置闭包里的缺口,按 §5.5 排序。

    Args:
        mastery: {kp_id: p_known};**必须已经包含闭包内全部节点**(缺的按 0 兜底)

    Returns:
        [{"kp_id","name","depth","mastery"}],按 (depth 升序, mastery 升序)。
        最浅且最弱 = 最可能的根因,排在前面。
    """
    closure = prereq_closure(store, kp_id, depth)
    if not closure:
        return []

    names = {}
    for kp in closure:
        concept = store.get_concept(kp)
        names[kp] = concept.name if concept else kp

    gaps = []
    for kp, d in closure.items():
        m = mastery.get(kp, 0.0)
        if m < threshold:
            gaps.append({"kp_id": kp, "name": names[kp], "depth": d, "mastery": m})

    gaps.sort(key=lambda g: (g["depth"], g["mastery"]))
    return gaps


def root_causes(
    store: KnowledgeStore,
    mastery: Dict[str, float],
    kp_id: str,
    top_k: int = config.ROOT_CAUSE_TOP_K,
    depth: int = config.MAX_PREREQ_DEPTH,
    threshold: float = config.GAP_THRESHOLD,
) -> List[Dict]:
    """缺口里最像"根因"的前 top_k 个,每个带上从目标出发的依赖路径。"""
    gaps = detect_gaps(store, mastery, kp_id, depth=depth, threshold=threshold)[:top_k]
    for gap in gaps:
        gap["path"] = shortest_prereq_path(store, kp_id, gap["kp_id"], depth)
    return gaps


def next_to_learn(
    store: KnowledgeStore,
    mastery: Dict[str, float],
    goal_kp_id: str,
    top_k: int = config.ROOT_CAUSE_TOP_K,
    depth: int = config.MAX_PREREQ_DEPTH,
    threshold: float = config.GAP_THRESHOLD,
) -> List[Dict]:
    """「现在该学什么」:目标前置闭包里,前置已备齐、自己还没掌握、离目标最近的节点。

    排序:depth 降序(先从离目标近的教起);同深度按掌握度升序(更弱的优先)。
    """
    closure = prereq_closure(store, goal_kp_id, depth)
    candidates = []
    for kp, d in closure.items():
        if mastery.get(kp, 0.0) >= threshold:
            continue  # 自己已经掌握了
        own_prereqs = store.ancestors(kp, depth=1)
        if any(mastery.get(p, 0.0) < threshold for p in own_prereqs):
            continue  # 自己的前置还没备齐,还不能学
        candidates.append(
            {"kp_id": kp, "depth": d, "mastery": mastery.get(kp, 0.0)}
        )

    candidates.sort(key=lambda c: (-c["depth"], c["mastery"]))
    for item in candidates[:top_k]:
        concept = store.get_concept(item["kp_id"])
        item["name"] = concept.name if concept else item["kp_id"]
    return candidates[:top_k]


def shortest_prereq_path(
    store: KnowledgeStore, from_kp: str, to_kp: str, depth: int = config.MAX_PREREQ_DEPTH
) -> List[str]:
    """从目标知识点往回走到某个前置的最短路径,如 ["algo.dp","algo.recursion","prog.func_call"]。

    沿 PREREQUISITE 反向 BFS(目标 → 它的前置 → 前置的前置)。
    """
    if from_kp == to_kp:
        return [from_kp]

    parent: Dict[str, Optional[str]] = {from_kp: None}
    frontier = [from_kp]
    for _ in range(depth):
        nxt = []
        for node in frontier:
            for prereq in store.ancestors(node, depth=1):
                if prereq in parent:
                    continue
                parent[prereq] = node
                if prereq == to_kp:
                    return _rebuild(parent, to_kp)
                nxt.append(prereq)
        if not nxt:
            break
        frontier = nxt
    return []


def _rebuild(parent: Dict[str, Optional[str]], target: str) -> List[str]:
    path = []
    node: Optional[str] = target
    while node is not None:
        path.append(node)
        node = parent[node]
    return list(reversed(path))


def explain_gaps(
    goal_kp_id: str, goal_name: str, kp_mastery: float, gaps: List[Dict]
) -> str:
    """不调 LLM 的兜底解释。有 LLM 时由 planner_worker 覆盖成人话(§5.5 第 6 步)。"""
    if not gaps:
        return f"「{goal_name}」的前置知识都已具备,问题可能出在这个知识点本身。"

    parts = []
    for gap in gaps:
        chain = " → ".join(gap.get("path") or [])
        where = f"(路径 {chain})" if chain else ""
        parts.append(f"「{gap['name']}」掌握度 {gap['mastery']:.2f}{where}")

    return (
        f"你卡在「{goal_name}」(当前掌握度 {kp_mastery:.2f}),"
        f"但根因很可能在更基础的地方:{';'.join(parts)}。"
        "建议先把这些前置补上,再回到目标知识点。"
    )
