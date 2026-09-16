"""agent 诊断器的工具层。

**为什么会有这个文件:** 两个企业级对标项目(Tencent WeSmartFlow / HKUDS DeepTutor)
读下来是同一个做法 —— **领域内核保持确定性,只把它包成 tool,
让 LLM 决定「什么时候、为什么」去调它**。它们都没把判分/掌握度/复习调度交给 LLM。

这个文件就是那层"包装":`queries.py` 的纯函数原样保留,只是多了一层给 LLM 看的门面。

**三条设计约束(每条都有理由,别随手改):**

1. **工具是上下文上的纯函数,不是 store 上的方法。**
   `queries.py` 早就立了先例 —— 它收 `mastery: dict` 而不是 `ProfileStore`,
   这样 `knowledge/` 永不 import `profile/`。这里照抄:`DiagnosisContext` 有两个构造器,
   生产从 store 装配、评测从现成的映射装配,**同一份工具层两条路径通用**。

2. **★ 绝不把 `detect_gaps` / `root_causes` 暴露成工具。**
   那是确定性版的最终答案。给了它,agent 就退化成"调一次函数复述结果"的套壳,
   和确定性路径的对照也就失去意义。这里只给**原料**,排序由模型自己决定。

3. **★ `DiagnosisContext` 绝不能进 LangGraph 的 state。**
   checkpointer 会把 state 序列化落盘,而这个对象攥着 sqlite 连接。
   必须通过**节点闭包**注入。("白拿 checkpointer" 这个卖点只有守住这条才成立。)

用法(工具层本身不认识 langgraph,可独立测试)::

    ctx = DiagnosisContext.from_stores(knowledge, profile, "u1")
    execute_tool("get_prereq_closure", {"kp_id": "algo.dp"}, ctx)
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set

from coach import config
from coach.knowledge import queries
from coach.knowledge.store import KnowledgeStore
from coach.profile import errors as error_analysis

#: 终结工具:模型调它 = 交卷。循环见到它就停,不再回灌结果。
TERMINAL_TOOL = "submit_diagnosis"

#: 每个工具的必填参数 —— 用来挡"参数解析失败被静默换成 {}"的情况
#: (`llm_client.chat_structured` 遇到坏 JSON 会给 `arguments = {}`,不抛异常)。
_REQUIRED_ARGS: Dict[str, Sequence[str]] = {
    "get_prereq_closure": ("kp_id",),
    "get_shortest_path": ("from_kp_id", "to_kp_id"),
    "get_mastery": ("kp_ids",),
    "list_problems_for_kp": ("kp_id",),
    "submit_diagnosis": ("root_causes",),
}


@dataclass
class DiagnosisContext:
    """诊断所需的**只读**数据。两个构造器对应两条使用路径。"""

    knowledge: KnowledgeStore
    learner_id: str
    mastery: Dict[str, float] = field(default_factory=dict)
    observed: Set[str] = field(default_factory=set)
    errors: Dict[str, Any] = field(default_factory=dict)
    goals: List[str] = field(default_factory=list)
    _problems_by_kp: Optional[Dict[str, List[Dict]]] = field(default=None, repr=False)

    # ---------- 构造 ----------

    @classmethod
    def from_maps(
        cls,
        knowledge: KnowledgeStore,
        learner_id: str,
        mastery: Dict[str, float],
        observed: Optional[Set[str]] = None,
        errors: Optional[Dict[str, Any]] = None,
        goals: Optional[Sequence[str]] = None,
    ) -> "DiagnosisContext":
        """评测路径:掌握度/观测集合已经是现成的映射,直接收下。"""
        return cls(
            knowledge=knowledge,
            learner_id=learner_id,
            mastery=dict(mastery),
            observed=set(observed or ()),
            errors=dict(errors or {}),
            goals=list(goals or ()),
        )

    @classmethod
    def from_stores(cls, knowledge: KnowledgeStore, profile, learner_id: str) -> "DiagnosisContext":
        """生产路径:从 ProfileStore 一次性批量取好,别在工具里逐个查(N+1)。"""
        kp_ids = [c.id for c in knowledge.list_concepts()]
        return cls(
            knowledge=knowledge,
            learner_id=learner_id,
            mastery=profile.get_mastery_map(learner_id, kp_ids),
            observed=profile.observed_kp_ids(learner_id, kp_ids),
            errors=error_analysis.summary(profile, learner_id),
            goals=_learner_goals(profile, learner_id),
        )

    # ---------- 只读访问(工具就建在这些方法上)----------

    def name_of(self, kp_id: str) -> str:
        concept = self.knowledge.get_concept(kp_id)
        return concept.name if concept else kp_id

    def mastery_of(self, kp_id: str) -> float:
        """缺省用 BKT 初始值 —— 和"考过但很弱"数值可能一样,所以必须配 `is_observed` 看。"""
        return float(self.mastery.get(kp_id, config.BKT_P_INIT))

    def is_observed(self, kp_id: str) -> bool:
        return kp_id in self.observed

    def closure(self, kp_id: str, depth: int = config.MAX_PREREQ_DEPTH) -> Dict[str, int]:
        return queries.prereq_closure(self.knowledge, kp_id, depth)

    def path(self, from_kp: str, to_kp: str, depth: int = config.MAX_PREREQ_DEPTH) -> List[str]:
        return queries.shortest_prereq_path(self.knowledge, from_kp, to_kp, depth)

    def problems_for(self, kp_id: str) -> List[Dict]:
        """挂在这个知识点上的题(计划要"可落地",所以 agent 也该看得到有没有题)。"""
        if self._problems_by_kp is None:
            index: Dict[str, List[Dict]] = {}
            for problem in self.knowledge.list_problems():
                for kp in problem.kp_ids:
                    index.setdefault(kp, []).append(
                        {
                            "id": problem.id,
                            "title": problem.title,
                            "difficulty": problem.difficulty,
                        }
                    )
            self._problems_by_kp = index
        return self._problems_by_kp.get(kp_id, [])


def _learner_goals(profile, learner_id: str) -> List[str]:
    """学习者的目标知识点(诊断通常围绕它展开)。没有目标就返回空。"""
    row = profile.get_profile(learner_id) or {}
    goal = row.get("goal_kp_id")
    return [goal] if goal else []


# --------------------------------------------------------------------------
# 工具 schema(写法照抄 builder.py 的 EXTRACT_TOOL)
# --------------------------------------------------------------------------

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_prereq_closure",
            "description": (
                "查某知识点的全部前置(反向依赖闭包),返回 {知识点id: 深度}。"
                "深度越小越基础。这是判断根因的主要依据。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "kp_id": {"type": "string", "description": "知识点 id,如 algo.dp"},
                    "depth": {"type": "integer", "description": "最大深度,默认 3"},
                },
                "required": ["kp_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_shortest_path",
            "description": "查从一个知识点到它某个前置的最短依赖路径,用来向学生解释「为什么是这个根因」。",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_kp_id": {"type": "string", "description": "起点(通常是目标知识点)"},
                    "to_kp_id": {"type": "string", "description": "终点(某个前置)"},
                },
                "required": ["from_kp_id", "to_kp_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_mastery",
            "description": (
                "查这批知识点的掌握度与**是否被观测过**。"
                "注意:没被观测过的点掌握度只是初始值,不代表学生真的弱 —— "
                "`observed=false` 的点应当当作「还没测过」而不是「已确认薄弱」。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "kp_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "要查的知识点 id 列表",
                    }
                },
                "required": ["kp_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_observed_kp_ids",
            "description": "列出这个学生**有作答证据**的全部知识点。不在这里面的点都属于「没测过」。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_problems_for_kp",
            "description": "列出挂在某个知识点下的练习题。没有题的根因点无法出题补救,评估时应考虑这一点。",
            "parameters": {
                "type": "object",
                "properties": {"kp_id": {"type": "string"}},
                "required": ["kp_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_error_patterns",
            "description": (
                "查这个学生的易错模式。`recurring` 里的是**跨多个知识点重复出现的错误类型**,"
                "通常意味着系统性误解而不是单点不会。"
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": TERMINAL_TOOL,
            "description": (
                "★ 交卷。给出你判断的根因排序(第一个是头号根因)和一段给学生看的人话解释。"
                "调用它之后本轮诊断就结束。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "root_causes": {
                        "type": "array",
                        "description": "根因,按可能性从高到低排序;每项至少给 kp_id",
                        "items": {
                            "type": "object",
                            "properties": {
                                "kp_id": {"type": "string"},
                                "reason": {"type": "string", "description": "为什么怀疑它是根因"},
                            },
                            "required": ["kp_id"],
                        },
                    },
                    "summary": {
                        "type": "string",
                        "description": "给学生看的人话解释(两三句)",
                    },
                },
                "required": ["root_causes"],
            },
        },
    },
]


def schema_for(name: str) -> Optional[Dict[str, Any]]:
    for schema in TOOL_SCHEMAS:
        if schema["function"]["name"] == name:
            return schema
    return None


def is_terminal(name: str) -> bool:
    return name == TERMINAL_TOOL


# --------------------------------------------------------------------------
# 执行
# --------------------------------------------------------------------------


def execute_tool(name: str, arguments: Dict[str, Any], ctx: DiagnosisContext) -> str:
    """执行一个**非终结**工具,返回给模型的 JSON 字符串。

    参数不合法时**返回错误而不是抛异常** —— 让模型看见错误并自我纠正,
    比整个循环崩掉有用。`llm_client` 对坏 JSON 会静默给 `{}`,所以这层校验是必须的。
    """
    try:
        missing = [key for key in _REQUIRED_ARGS.get(name, ()) if key not in arguments]
        if missing:
            raise ValueError(f"缺少必填参数 {missing}")
        if name == "get_prereq_closure":
            payload = _tool_closure(ctx, arguments)
        elif name == "get_shortest_path":
            payload = _tool_path(ctx, arguments)
        elif name == "get_mastery":
            payload = _tool_mastery(ctx, arguments)
        elif name == "get_observed_kp_ids":
            payload = {"observed_kp_ids": sorted(ctx.observed)}
        elif name == "list_problems_for_kp":
            payload = {"kp_id": arguments["kp_id"], "problems": ctx.problems_for(arguments["kp_id"])}
        elif name == "get_error_patterns":
            payload = ctx.errors or {"top": [], "recurring": []}
        else:
            raise ValueError(f"未知工具:{name}")
    except (KeyError, TypeError, ValueError) as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False)
    return json.dumps(payload, ensure_ascii=False, default=str)


def _tool_closure(ctx: DiagnosisContext, arguments: Dict[str, Any]) -> Dict:
    kp_id = arguments["kp_id"]
    depth = int(arguments.get("depth") or config.MAX_PREREQ_DEPTH)
    if not ctx.knowledge.has_concept(kp_id):
        raise ValueError(f"知识点不存在:{kp_id}")
    return {"kp_id": kp_id, "prerequisites": ctx.closure(kp_id, depth)}


def _tool_path(ctx: DiagnosisContext, arguments: Dict[str, Any]) -> Dict:
    from_kp, to_kp = arguments["from_kp_id"], arguments["to_kp_id"]
    return {"from": from_kp, "to": to_kp, "path": ctx.path(from_kp, to_kp)}


def _tool_mastery(ctx: DiagnosisContext, arguments: Dict[str, Any]) -> Dict:
    kp_ids = arguments["kp_ids"]
    if not isinstance(kp_ids, list):
        raise ValueError("kp_ids 必须是数组")
    return {
        "mastery": [
            {
                "kp_id": kp,
                "name": ctx.name_of(kp),
                "mastery": round(ctx.mastery_of(kp), 4),
                "observed": ctx.is_observed(kp),
            }
            for kp in kp_ids
        ]
    }


def parse_diagnosis(
    arguments: Dict[str, Any],
    ctx: DiagnosisContext,
    target_kp: str,
    top_k: int = config.ROOT_CAUSE_TOP_K,
    depth: int = config.MAX_PREREQ_DEPTH,
) -> Dict[str, Any]:
    """把终结工具的入参规整成 `root_causes` + `explanation`。

    **只信模型给的顺序和理由**,不信它给的 name/depth/path —— 那些由我们从图里补齐。
    理由:路径和深度是图的事实,让 LLM 复述只会引入噪声;而**排序**才是它真正要负责的判断。
    """
    raw = arguments.get("root_causes")
    if not isinstance(raw, list) or not raw:
        raise ValueError("root_causes 必须是非空数组")

    closure = ctx.closure(target_kp, depth)
    roots: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for item in raw:
        kp_id = item.get("kp_id") if isinstance(item, dict) else None
        if not kp_id or kp_id in seen:
            continue
        seen.add(kp_id)
        roots.append(
            {
                "kp_id": kp_id,
                "name": ctx.name_of(kp_id),
                "depth": closure.get(kp_id, -1),
                "mastery": round(ctx.mastery_of(kp_id), 4),
                "status": "gap" if ctx.is_observed(kp_id) else "unobserved",
                "path": ctx.path(target_kp, kp_id, depth),
                "reason": (item.get("reason") or "") if isinstance(item, dict) else "",
            }
        )
        if len(roots) >= top_k:
            break

    if not roots:
        raise ValueError("root_causes 里没有一个可用的 kp_id")
    return {"root_causes": roots, "summary": str(arguments.get("summary") or "")}
