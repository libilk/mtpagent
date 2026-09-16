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
    observed: Optional[set] = None,
) -> List[Dict]:
    """找出 X 前置闭包里的缺口,按 §5.5 排序。

    ★ **两态区分(2026-09-13 修正的核心)**
    闭包里掌握度低的点有两种,数值上可能一样,含义完全不同:
    - `status="gap"`:有作答证据,确实弱 → 这是**真根因**
    - `status="unobserved"`:从没被观测过,掌握度只是初始值 → 这是**没测过**,不是弱

    原先只按 `(depth, mastery)` 排,导致大量从没见过的浅层点(掌握度停在 0.1)
    把真正观测到的弱项挤到后面。P5 评测里这让作答稀疏时的 Top-1 掉到 0.4933,
    **反低于闭包内随机排序的 0.6267**。

    排序改为 `(status_rank, depth, mastery)`:已观测的缺口永远排在未观测点前面。
    (对标 DeepTutor:它的 `new` 与 `learning` 从不共享同一条排序轴。)

    Args:
        mastery: {kp_id: p_known};**应包含闭包内全部节点**(缺的按 0 兜底)
        observed: 有证据的 kp_id 集合。**传 None 则退化为旧行为**(全部当已观测),
                  这是给不关心两态的调用方留的出口。

    Returns:
        [{"kp_id","name","depth","mastery","status"}]
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
        if m >= threshold:
            continue
        seen = observed is None or kp in observed
        gaps.append(
            {
                "kp_id": kp,
                "name": names[kp],
                "depth": d,
                "mastery": m,
                "status": "gap" if seen else "unobserved",
            }
        )

    # 已观测的缺口(depth 浅、掌握度低)在前;未观测的按深浅兜底排在后
    gaps.sort(key=lambda g: (0 if g["status"] == "gap" else 1, g["depth"], g["mastery"]))
    return gaps


def root_causes(
    store: KnowledgeStore,
    mastery: Dict[str, float],
    kp_id: str,
    top_k: int = config.ROOT_CAUSE_TOP_K,
    depth: int = config.MAX_PREREQ_DEPTH,
    threshold: float = config.GAP_THRESHOLD,
    observed: Optional[set] = None,
) -> List[Dict]:
    """缺口里最像"根因"的前 top_k 个,每个带上从目标出发的依赖路径。

    `observed` 传进来时,有证据的缺口会排在前面(见 `detect_gaps` 的说明)。
    """
    gaps = detect_gaps(
        store, mastery, kp_id, depth=depth, threshold=threshold, observed=observed
    )[:top_k]
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
    observed: Optional[set] = None,
) -> List[Dict]:
    """「现在该学什么」:目标前置闭包里,前置已备齐、自己还没掌握、离目标最近的节点。

    排序:depth 降序(先从离目标近的教起);同深度按掌握度升序(更弱的优先)。

    ★ **两态区分(与 detect_gaps 同一类修正)**
    原先一行 `any(mastery.get(p, 0.0) < threshold ...)` 会把**从没测过的**前置
    当成"没准备好",于是全新学生只能学 7 个叶子,`algo.dp` / `mergesort` / `bfs`
    全被卡死 —— 而学生可能**早就会了,只是系统没测过**。

    现在的处理:
    - 前置**已确认不会**(有证据且 < 阈值)→ 真的不能学,跳过;
    - 前置**从没测过** → 不阻塞,但把这一项的 `action` 标成 `probe`(先测一下),
      对标 DeepTutor 的 `probe` 动作。
    """
    closure = prereq_closure(store, goal_kp_id, depth)
    adjacency = store.prerequisite_adjacency()  # 一次查完,避免 N+1
    candidates = []

    for kp, d in closure.items():
        if mastery.get(kp, 0.0) >= threshold:
            continue  # 自己已经掌握了

        own_prereqs = adjacency.get(kp, [])
        blocked = [
            p
            for p in own_prereqs
            if mastery.get(p, 0.0) < threshold and (observed is None or p in observed)
        ]
        if blocked:
            continue  # 前置**已确认不会** —— 这是真的不能学

        needs_probe = any(
            mastery.get(p, 0.0) < threshold and observed is not None and p not in observed
            for p in own_prereqs
        )
        candidates.append(
            {
                "kp_id": kp,
                "depth": d,
                "mastery": mastery.get(kp, 0.0),
                # probe = 前置里有点"还没测过",先测再学更稳
                "action": "probe" if needs_probe else "learn",
            }
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
    """不调 LLM 的兜底解释。有 LLM 时由 pipeline 的 explain 节点覆盖成人话。

    ★ 措辞区分两类:有证据的弱项说得肯定("确实薄弱"),
    没观测过的只说"还没测过",不武断下结论。
    """
    if not gaps:
        return f"「{goal_name}」的前置知识都已具备,问题可能出在这个知识点本身。"

    confirmed, untested = [], []
    for gap in gaps:
        chain = " → ".join(gap.get("path") or [])
        where = f"(路径 {chain})" if chain else ""
        line = f"「{gap['name']}」掌握度 {gap['mastery']:.2f}{where}"
        (untested if gap.get("status") == "unobserved" else confirmed).append(line)

    head = f"你卡在「{goal_name}」(当前掌握度 {kp_mastery:.2f})。"
    body = []
    if confirmed:
        body.append(f"已确认薄弱的前置:{';'.join(confirmed)}。建议先补这些。")
    if untested:
        body.append(f"另外这些前置还没有作答记录,建议先测一下:{';'.join(untested)}。")
    return head + "".join(body)
