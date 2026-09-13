"""建图流程:LLM 抽取 + 治理入库(work.md P0.6 / §4.4)。

关键纪律:**LLM 抽出来的东西一律先进 Observation,再走 propose → aggregate**。
builder 不做任何"直接写边"的捷径,否则治理流程就形同虚设。

两种入口:
- `ingest(...)`    —— 把已经人工确认好的知识点/边喂进治理流程(种子数据走这条)
- `extract_*`      —— 让 LLM 从教材/大纲文本里抽,产出的是**候选**,仍需治理

CLI:
    python -m coach.knowledge.builder --extract-file outline.txt --subject algorithms
    python -m coach.knowledge.builder --build [--seed 15]        # P0.7
"""

import argparse
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from coach.domain.models import EDGE_TYPES
from coach.knowledge.governance import Governance, has_cycle
from coach.knowledge.store import KnowledgeStore

# ---------------------------------------------------------------- 抽取

EXTRACT_SYSTEM_PROMPT = """你是计算机课程的知识工程师。从给定的教材/大纲文本中,
抽取「知识点」以及知识点之间的依赖关系,供学习路径规划使用。

规则:
1. id 用 "域名.短名" 的小写英文形式,如 algo.dp、ds.array;同一概念复用同一 id。
2. 关系四选一:
   - PREREQUISITE:学 to 之前必须先掌握 from(★ 最重要,只标真正必要的前置)
   - RELATED:相关但无先后
   - EXTENDS:to 是 from 的深化/推广
   - CONTRASTS:to 与 from 常被对比(如快排 vs 归并)
3. confidence 是你的确信度 0~1;拿不准就写低一点,下游会过滤。
4. 只抽文本中确有依据的关系,不要凭常识补充文本没提到的内容。
5. 每条关系给出 evidence:原文中的短语或句子片段。
"""

EXTRACT_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "submit_knowledge",
        "description": "提交抽取到的知识点与关系",
        "parameters": {
            "type": "object",
            "properties": {
                "concepts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "name": {"type": "string"},
                            "subject": {"type": "string"},
                            "difficulty": {"type": "number", "description": "1.0~5.0"},
                            "description": {"type": "string"},
                        },
                        "required": ["id", "name", "subject"],
                    },
                },
                "edges": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "from_id": {"type": "string"},
                            "to_id": {"type": "string"},
                            "type": {"type": "string", "enum": list(EDGE_TYPES)},
                            "confidence": {"type": "number"},
                            "evidence": {"type": "string"},
                        },
                        "required": ["from_id", "to_id", "type", "confidence"],
                    },
                },
            },
            "required": ["concepts", "edges"],
        },
    },
}


def _parse_extraction(raw: Dict[str, Any], subject: Optional[str] = None) -> tuple:
    """把 LLM 返回的 dict 规整成 (concepts, edges),容忍缺失字段。"""
    concepts: List[Dict[str, Any]] = []
    for c in raw.get("concepts") or []:
        if not isinstance(c, dict) or not c.get("id") or not c.get("name"):
            continue
        concepts.append(
            {
                "id": str(c["id"]).strip(),
                "name": str(c["name"]).strip(),
                "subject": str(c.get("subject") or subject or "unknown"),
                "difficulty": c.get("difficulty", 3.0),
                "description": c.get("description"),
            }
        )

    edges: List[Dict[str, Any]] = []
    for e in raw.get("edges") or []:
        if not isinstance(e, dict) or not e.get("from_id") or not e.get("to_id"):
            continue
        if e.get("type") not in EDGE_TYPES:
            continue
        edges.append(
            {
                "from_id": str(e["from_id"]).strip(),
                "to_id": str(e["to_id"]).strip(),
                "type": e["type"],
                "confidence": float(e.get("confidence", 1.0)),
                "weight": 1.0,
                "source": "llm_extract",
                "evidence": e.get("evidence"),
            }
        )
    return concepts, edges


def extract(llm, text: str, subject: Optional[str] = None, known_concepts=None) -> tuple:
    """调 LLM 抽取,返回 (concepts, edges) 两个 payload 列表。**不写库**。

    优先取 tool_call 的结构化参数;模型没走工具调用时,退化到解析 content 里的 JSON。

    known_concepts: 图中已有的知识点(id + name)。喂给模型要求**复用已有 id**,
    从源头减少 id 漂移。注意这是把清单整个塞进 prompt,知识点多了会撑爆上下文——
    到那时要换成"按文本检索相关概念"再喂。当前 22 个点没问题。
    """
    hint = f"\n\n所属学科:{subject}" if subject else ""
    if known_concepts:
        listing = "、".join(f"{c.id}={c.name}" for c in known_concepts)
        hint += (
            "\n\n图中已有这些知识点,**请优先复用它们的 id,不要另起新 id**"
            f"(确实不在其中的才新建):\n{listing}"
        )

    messages = [
        {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
        {"role": "user", "content": f"请抽取以下材料中的知识点与关系:{hint}\n\n{text}"},
    ]
    resp = llm.chat_structured(messages, tools=[EXTRACT_TOOL])

    raw: Optional[Dict[str, Any]] = None
    for call in resp.tool_calls:
        if call.name == "submit_knowledge":
            raw = call.arguments
            break
    if raw is None:
        raw = _loads_object(resp.content)
    if raw is None:
        raise ValueError("LLM 未返回可解析的抽取结果(既无 submit_knowledge 工具调用,content 也不是 JSON)")

    return _parse_extraction(raw, subject=subject)


def _loads_object(text: Optional[str]) -> Optional[Dict[str, Any]]:
    """从可能带 ```json 围栏的文本里抠出 JSON 对象。"""
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        s = s.split("```")[1]
        if s.lstrip().lower().startswith("json"):
            s = s.lstrip()[4:]
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


# ---------------------------------------------------------------- 入库

@dataclass
class IngestReport:
    """一次入库的结果统计。被拒的提案会带上原因,便于排查。"""

    concepts_accepted: int = 0
    concepts_rejected: int = 0
    edges_accepted: int = 0
    edges_rejected: int = 0
    rejections: List[tuple] = field(default_factory=list)  # (kind, payload_key, reason)

    def summary(self) -> str:
        return (
            f"知识点 接受 {self.concepts_accepted} / 拒 {self.concepts_rejected};"
            f"关系 接受 {self.edges_accepted} / 拒 {self.edges_rejected}"
        )


def _key(kind: str, payload: Dict[str, Any]) -> str:
    if kind == "edge":
        return f"{payload.get('from_id')}→{payload.get('to_id')}({payload.get('type')})"
    return str(payload.get("id"))


def ingest(
    governance: Governance,
    concepts: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    source: str = "manual",
) -> IngestReport:
    """把 payload 列表喂进 observe → propose → aggregate。先概念后关系。"""
    report = IngestReport()

    for c in concepts:
        proposal = _run(governance, "concept", c, source)
        if proposal.status == "accepted":
            report.concepts_accepted += 1
        else:
            report.concepts_rejected += 1
            report.rejections.append(("concept", _key("concept", c), proposal.reason))

    for e in edges:
        proposal = _run(governance, "edge", e, source)
        if proposal.status == "accepted":
            report.edges_accepted += 1
        else:
            report.edges_rejected += 1
            report.rejections.append(("edge", _key("edge", e), proposal.reason))

    return report


def _run(governance: Governance, kind: str, payload: Dict[str, Any], source: str):
    observation = governance.observe(kind, payload, source=source)
    proposal = governance.propose(observation.id)
    return governance.aggregate(proposal.id)


# ---------------------------------------------------------------- 种子数据(P0.7)

# 来源:mainconten.md §5.1「前置依赖示例(建图时的金标准样例)」。
# 顺序刻意按依赖拓扑排:--seed N 截断时,前 N 个仍是一张连通的子图。
GOLDEN_CONCEPTS: List[Dict[str, Any]] = [
    {"id": "prog.func_call", "name": "函数调用", "subject": "programming", "difficulty": 1.0,
     "description": "调用栈、参数传递、返回值"},
    {"id": "ds.array", "name": "数组", "subject": "data_structures", "difficulty": 1.0,
     "description": "连续内存、下标随机访问"},
    {"id": "ds.linked_list", "name": "链表", "subject": "data_structures", "difficulty": 2.0,
     "description": "指针串接、插入删除 O(1)"},
    {"id": "ds.stack", "name": "栈", "subject": "data_structures", "difficulty": 2.0,
     "description": "后进先出"},
    {"id": "ds.queue", "name": "队列", "subject": "data_structures", "difficulty": 2.0,
     "description": "先进先出"},
    {"id": "ds.hash_table", "name": "哈希表", "subject": "data_structures", "difficulty": 2.5,
     "description": "散列函数与冲突处理"},
    {"id": "algo.recursion", "name": "递归", "subject": "algorithms", "difficulty": 2.5,
     "description": "函数自我调用 + 基线条件"},
    {"id": "algo.sorting", "name": "排序", "subject": "algorithms", "difficulty": 2.5,
     "description": "排序问题与稳定性、复杂度对比"},
    {"id": "algo.binary_search", "name": "二分查找", "subject": "algorithms", "difficulty": 2.5,
     "description": "有序区间折半收缩"},
    {"id": "algo.two_pointers", "name": "双指针", "subject": "algorithms", "difficulty": 2.5,
     "description": "两个下标协同移动"},
    {"id": "algo.divide_conquer", "name": "分治", "subject": "algorithms", "difficulty": 3.0,
     "description": "分解 → 求解 → 合并"},
    {"id": "algo.mergesort", "name": "归并排序", "subject": "algorithms", "difficulty": 3.0,
     "description": "分治 + 有序合并,稳定"},
    {"id": "algo.quicksort", "name": "快速排序", "subject": "algorithms", "difficulty": 3.0,
     "description": "划分 + 分治,原地但不稳定"},
    {"id": "algo.memoization", "name": "记忆化搜索", "subject": "algorithms", "difficulty": 3.0,
     "description": "递归 + 缓存子问题结果"},
    {"id": "algo.dp", "name": "动态规划", "subject": "algorithms", "difficulty": 4.0,
     "description": "状态定义 + 转移方程 + 最优子结构"},
    {"id": "algo.sliding_window", "name": "滑动窗口", "subject": "algorithms", "difficulty": 3.0,
     "description": "双指针维护一个窗口"},
    {"id": "algo.backtracking", "name": "回溯", "subject": "algorithms", "difficulty": 3.5,
     "description": "递归试探 + 撤销选择"},
    {"id": "algo.greedy", "name": "贪心", "subject": "algorithms", "difficulty": 3.0,
     "description": "每步取局部最优"},
    {"id": "algo.dfs", "name": "深度优先搜索", "subject": "algorithms", "difficulty": 3.0,
     "description": "沿一条路走到底再回溯"},
    {"id": "algo.bfs", "name": "广度优先搜索", "subject": "algorithms", "difficulty": 3.0,
     "description": "逐层扩散"},
    {"id": "algo.topological_sort", "name": "拓扑排序", "subject": "algorithms", "difficulty": 3.5,
     "description": "DAG 的线性序"},
    {"id": "algo.dijkstra", "name": "Dijkstra 最短路径", "subject": "algorithms", "difficulty": 4.0,
     "description": "非负权单源最短路"},
]

# (from, to, type, evidence) —— evidence 直接引 mainconten §5.1 的表述
GOLDEN_EDGES: List[Dict[str, Any]] = [
    ("prog.func_call", "algo.recursion", "PREREQUISITE", "函数调用 ──PREREQUISITE──► 递归"),
    ("algo.recursion", "algo.divide_conquer", "PREREQUISITE", "递归 ──PREREQUISITE──► 分治"),
    ("algo.recursion", "algo.backtracking", "PREREQUISITE", "递归 ──PREREQUISITE──► 回溯"),
    ("algo.recursion", "algo.memoization", "PREREQUISITE", "记忆化搜索 = 递归 + 缓存"),
    ("algo.memoization", "algo.dp", "PREREQUISITE", "记忆化是 DP 的递归形式"),
    ("algo.recursion", "algo.dp", "PREREQUISITE", "递归 + 数组 ─PREREQUISITE──► 动态规划"),
    ("ds.array", "algo.dp", "PREREQUISITE", "递归 + 数组 ─PREREQUISITE──► 动态规划"),
    ("ds.array", "algo.two_pointers", "PREREQUISITE", "数组 ──PREREQUISITE──► 双指针"),
    ("algo.two_pointers", "algo.sliding_window", "PREREQUISITE", "双指针 ──PREREQUISITE──► 滑动窗口"),
    ("ds.array", "algo.binary_search", "PREREQUISITE", "数组 ──PREREQUISITE──► 二分查找"),
    ("ds.array", "algo.sorting", "PREREQUISITE", "排序的对象是数组"),
    ("algo.divide_conquer", "algo.mergesort", "PREREQUISITE", "归并排序是分治的典型应用"),
    ("algo.recursion", "algo.mergesort", "PREREQUISITE", "归并排序靠递归拆半"),
    ("algo.divide_conquer", "algo.quicksort", "PREREQUISITE", "快排的划分是分治"),
    ("algo.recursion", "algo.quicksort", "PREREQUISITE", "快排递归处理两侧"),
    ("ds.queue", "algo.bfs", "PREREQUISITE", "队列 ──PREREQUISITE──► BFS"),
    ("ds.stack", "algo.dfs", "PREREQUISITE", "栈 ──PREREQUISITE──► DFS"),
    ("algo.dfs", "algo.topological_sort", "PREREQUISITE", "DFS + 队列 ─PREREQUISITE──► 拓扑排序"),
    ("ds.queue", "algo.topological_sort", "PREREQUISITE", "DFS + 队列 ─PREREQUISITE──► 拓扑排序"),
    ("algo.greedy", "algo.dijkstra", "PREREQUISITE", "贪心 ──PREREQUISITE──► Dijkstra"),
    ("algo.sorting", "algo.binary_search", "RELATED", "排序 ──RELATED──────► 二分查找"),
    ("ds.array", "ds.linked_list", "CONTRASTS", "连续存储 vs 链式存储"),
    ("ds.hash_table", "ds.array", "RELATED", "都依赖下标寻址"),
    ("algo.quicksort", "algo.mergesort", "CONTRASTS", "快排 ──CONTRASTS────► 归并排序"),
    ("algo.dp", "algo.greedy", "CONTRASTS", "动态规划 ──CONTRASTS────► 贪心"),
    ("algo.bfs", "algo.dfs", "CONTRASTS", "逐层扩散 vs 一路到底"),
    ("algo.memoization", "algo.dp", "EXTENDS", "DP 是记忆化的迭代化推广"),
]


def _golden_payloads(limit: Optional[int] = None):
    """取种子概念/边。limit 截断概念数,并丢掉端点不全的边。"""
    concepts = GOLDEN_CONCEPTS[:limit] if limit else list(GOLDEN_CONCEPTS)
    ids = {c["id"] for c in concepts}
    edges = []
    for from_id, to_id, edge_type, evidence in GOLDEN_EDGES:
        if from_id not in ids or to_id not in ids:
            continue
        edges.append(
            {
                "from_id": from_id,
                "to_id": to_id,
                "type": edge_type,
                "confidence": 1.0,
                "weight": 1.0,
                "source": "golden",
                "evidence": evidence,
            }
        )
    return concepts, edges


def seed_graph(governance: Governance, limit: Optional[int] = None) -> IngestReport:
    """把金标准种子数据走治理流程灌进图。可重复执行(重复的会被判重拒绝)。"""
    concepts, edges = _golden_payloads(limit)
    return ingest(governance, concepts, edges, source="golden")


# 题目不经过治理流程(治理是给"LLM 抽的知识"用的,题目来自公开题库,是事实)。
# test_cases 的**最后一条**视为提交用例 —— 学生交的 answer_text 跟它比。
GOLDEN_PROBLEMS: List[Dict[str, Any]] = [
    {
        "id": "lc.1", "title": "两数之和", "source": "leetcode:1", "difficulty": 2.0,
        "kp_ids": ["ds.hash_table", "ds.array"],
        "test_cases": [
            {"input": "nums=[2,7,11,15], target=9", "expected": "[0,1]"},
            {"input": "nums=[3,2,4], target=6", "expected": "[1,2]"},
        ],
    },
    {
        "id": "lc.704", "title": "二分查找", "source": "leetcode:704", "difficulty": 2.0,
        "kp_ids": ["algo.binary_search", "ds.array"],
        "test_cases": [
            {"input": "nums=[-1,0,3,5,9,12], target=9", "expected": "4"},
            {"input": "nums=[-1,0,3,5,9,12], target=2", "expected": "-1"},
        ],
    },
    {
        "id": "lc.206", "title": "反转链表", "source": "leetcode:206", "difficulty": 2.5,
        "kp_ids": ["ds.linked_list"],
        "test_cases": [
            {"input": "head=[1,2,3,4,5]", "expected": "5 4 3 2 1"},
        ],
    },
    {
        "id": "lc.20", "title": "有效的括号", "source": "leetcode:20", "difficulty": 2.0,
        "kp_ids": ["ds.stack"],
        "test_cases": [
            {"input": "s=\"()[]{}\"", "expected": "true"},
            {"input": "s=\"([)]\"", "expected": "false"},
        ],
    },
    {
        "id": "lc.232", "title": "用栈实现队列", "source": "leetcode:232", "difficulty": 2.5,
        "kp_ids": ["ds.queue", "ds.stack"],
        "test_cases": [
            {"input": "push(1);push(2);peek();pop();empty()", "expected": "ok"},
        ],
    },
    {
        "id": "lc.70", "title": "爬楼梯", "source": "leetcode:70", "difficulty": 2.0,
        "kp_ids": ["algo.dp", "algo.recursion"],
        "test_cases": [
            {"input": "n=3", "expected": "3"},
            {"input": "n=5", "expected": "8"},
        ],
    },
    {
        "id": "lc.509", "title": "斐波那契数", "source": "leetcode:509", "difficulty": 1.5,
        "kp_ids": ["algo.recursion", "algo.memoization"],
        "test_cases": [
            {"input": "n=4", "expected": "3"},
            {"input": "n=5", "expected": "5"},
        ],
    },
    {
        "id": "lc.46", "title": "全排列", "source": "leetcode:46", "difficulty": 3.5,
        "kp_ids": ["algo.backtracking", "algo.recursion"],
        "test_cases": [
            {"input": "nums=[1,2,3]", "expected": "6"},
        ],
    },
    {
        "id": "lc.200", "title": "岛屿数量", "source": "leetcode:200", "difficulty": 3.0,
        "kp_ids": ["algo.dfs", "ds.stack"],
        "test_cases": [
            {"input": "grid=3x5 见题面", "expected": "3"},
        ],
    },
    {
        "id": "lc.102", "title": "二叉树的层序遍历", "source": "leetcode:102", "difficulty": 3.0,
        "kp_ids": ["algo.bfs", "ds.queue"],
        "test_cases": [
            {"input": "root=[3,9,20,null,null,15,7]", "expected": "[3,9,20,15,7]"},
        ],
    },
    {
        "id": "lc.53", "title": "最大子数组和", "source": "leetcode:53", "difficulty": 3.0,
        "kp_ids": ["algo.dp", "ds.array"],
        "test_cases": [
            {"input": "nums=[-2,1,-3,4,-1,2,1,-5,4]", "expected": "6"},
        ],
    },
    {
        "id": "lc.455", "title": "分发饼干", "source": "leetcode:455", "difficulty": 2.5,
        "kp_ids": ["algo.greedy", "algo.sorting"],
        "test_cases": [
            {"input": "g=[1,2,3], s=[1,1]", "expected": "1"},
            {"input": "g=[1,2], s=[1,2,3]", "expected": "2"},
        ],
    },
    {
        "id": "lc.209", "title": "长度最小的子数组", "source": "leetcode:209", "difficulty": 3.0,
        "kp_ids": ["algo.sliding_window", "algo.two_pointers"],
        "test_cases": [
            {"input": "target=7, nums=[2,3,1,2,4,3]", "expected": "2"},
        ],
    },
]


def seed_problems(knowledge) -> int:
    """灌题目。引用不到知识点的题目直接跳过,不留悬空引用。返回实际写入数。"""
    from coach.domain.models import Problem

    written = 0
    for payload in GOLDEN_PROBLEMS:
        if not all(knowledge.has_concept(kp) for kp in payload["kp_ids"]):
            continue
        knowledge.upsert_problem(
            Problem(
                id=payload["id"],
                title=payload["title"],
                source=payload.get("source"),
                difficulty=payload.get("difficulty"),
                judge_type=payload.get("judge_type", "exact_output"),
                test_cases=payload["test_cases"],
                kp_ids=payload["kp_ids"],
            )
        )
        written += 1
    return written


def expected_answer(problem_id: str) -> Optional[str]:
    """取某题的"正确提交",方便演示和测试。"""
    for payload in GOLDEN_PROBLEMS:
        if payload["id"] == problem_id:
            return str(payload["test_cases"][-1]["expected"])
    return None


# ---------------------------------------------------------------- CLI

def _cmd_extract(args) -> int:
    from llm.llm_client import LLM

    text = open(args.extract_file, encoding="utf-8").read()
    store = KnowledgeStore.open(args.db)
    try:
        llm = LLM(model_name=args.model)
        concepts, edges = extract(
            llm, text, subject=args.subject, known_concepts=store.list_concepts()
        )
        print(f"抽取到 {len(concepts)} 个知识点、{len(edges)} 条关系")

        if not args.apply:
            print("(dry-run:未写库。加 --apply 才入库)")
            print(json.dumps({"concepts": concepts, "edges": edges}, ensure_ascii=False, indent=2))
            return 0

        report = ingest(Governance(store), concepts, edges, source="llm_extract")
        print(report.summary())
        print("提示:被拒提案里若有「同名知识点已存在:归并到 X」,"
              "说明 LLM 起了别名,图里没有重复节点,这条边已改指规范 id。")
        for kind, key, reason in report.rejections:
            print(f"  [拒] {kind} {key}: {reason}")
    finally:
        store.close()
    return 0


def _cmd_build(args) -> int:
    store = KnowledgeStore.open(args.db)
    try:
        governance = Governance(store)
        report = seed_graph(governance, limit=args.seed)
        print(report.summary())
        for kind, key, reason in report.rejections:
            print(f"  [拒] {kind} {key}: {reason}")

        problems = seed_problems(store)
        print(f"题目写入/更新:{problems} 道")

        pairs = store.prerequisite_pairs()
        print(f"\n图规模:知识点 {len(store.list_concepts())} / 关系 {len(store.list_edges())}"
              f" / 题目 {len(store.list_problems())}")
        print(f"无环校验:{'通过' if not has_cycle(pairs) else '★ 成环(异常,请检查)'}")
        print(f"ancestors('algo.dp', 3) = {store.ancestors('algo.dp', 3)}")
    finally:
        store.close()
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="coach.knowledge.builder", description="建图:LLM 抽取 + 治理入库")
    parser.add_argument("--db", default=None, help="SQLite 路径,默认 data/coach/coach.db")
    parser.add_argument("--model", default="qwen-plus", help="LLM 模型名")

    parser.add_argument("--extract-file", help="从文本文件抽取知识点与关系")
    parser.add_argument("--subject", default=None, help="学科标签,如 algorithms")
    parser.add_argument("--apply", action="store_true", help="抽取结果写库(默认 dry-run,只打印)")

    parser.add_argument("--build", action="store_true", help="建种子图(P0.7)")
    parser.add_argument("--seed", nargs="?", type=int, const=15, default=None, help="种子知识点数量,默认 15")

    args = parser.parse_args(argv)

    if args.build or args.seed is not None:
        return _cmd_build(args)
    if args.extract_file:
        return _cmd_extract(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
