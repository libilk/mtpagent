"""一键跑出四张评测表(work.md §7 P5.3)。

    .venv/Scripts/python.exe -m coach.evaluation.run_eval --n 20

四张表回答的问题:
    1. 掌握度预测准不准            → AUC / Brier / ECE,并给出平凡基线做参照
    2. 遗忘预测准不准              → 校准曲线 + 「隔久了会怎样」的对照
    3. 规划相比随机有没有增益      → 同样步数下真实能力涨了多少
    4. ★ 图 vs 扁平召回,根因定位谁准 → 本文档最核心的一张

**诚实性约束(§11 明确要求):**
- 模拟学生的生成模型与 BKT **不同**(见 simulated_student.py 的说明),否则是循环论证;
- 每个数字都配一个平凡基线:掌握度预测配"基准率",根因配"扁平召回";
- 结果无论好坏都照实写,包括图没赢过基线的情况。
"""

import argparse
import json
import math
import random
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from coach import config
from coach.coordination.journal import Journal
from coach.evaluation import baselines, metrics
from coach.evaluation.simulated_student import StudentSimulator
from coach.events import schema as events
from coach.knowledge import builder, queries, schema
from coach.knowledge.governance import Governance
from coach.knowledge.store import KnowledgeStore
from coach.profile import bkt
from coach.profile.store import ProfileStore
from coach.workflow.pipeline import AnswerPipeline
from coach.workflow.query import QueryService

RESULTS_DIR = Path(__file__).resolve().parent / "results"
GOAL = "algo.dp"


class Env:
    """一次评测用的环境:一个临时库 + 种子图 + 种子题。"""

    def __init__(self, seed: int = 0):
        self.seed = seed
        tmp = Path(tempfile.mkdtemp(prefix="coach_eval_"))
        self.db_path = tmp / "eval.db"
        self.conn = schema.connect(self.db_path, check_same_thread=False)
        schema.init_schema(self.conn)

        self.knowledge = KnowledgeStore(self.conn)
        builder.seed_graph(Governance(self.knowledge))
        builder.seed_problems(self.knowledge)

        self.profile = ProfileStore(self.conn)
        self.journal = Journal(self.conn)
        self.service = QueryService(self.knowledge, self.profile)
        self.simulator = StudentSimulator(self.knowledge, seed=seed)
        self.pipeline = AnswerPipeline(
            self.knowledge, self.profile, self.journal, llm=None, checkpointer=None
        )
        self.problems = {p.id: p for p in self.knowledge.list_problems()}

    def close(self) -> None:
        self.conn.close()

    def submit(self, learner_id: str, problem_id: str, correct: bool) -> bool:
        """把模拟学生的作答结果送进真实答题链,让画像被正常更新。

        答对就给标准答案,答错给一个明显错的 —— 判分是纯规则比对,这样做
        「学生答对/答错」和「链路判对/判错」严格一致。
        """
        expected = builder.expected_answer(problem_id)
        text = expected if correct else "__definitely_wrong__"
        event = events.new_event(
            events.ANSWER_SUBMITTED,
            learner_id,
            payload={"problem_id": problem_id, "answer_text": text, "elapsed_ms": 100},
        )
        result = self.pipeline.run(event)
        return bool(result.get("correct"))


def problems_for_kp(env: Env, kp_id: str) -> List[str]:
    return [p.id for p in env.problems.values() if kp_id in p.kp_ids]


def new_learner(env: Env, learner_id: str, goal: Optional[str] = GOAL, minutes: int = 600) -> None:
    env.profile.upsert_profile(learner_id, goal_kp_id=goal, daily_minutes=minutes)


# ================================================================ 表 1:掌握度预测

def exp_mastery_prediction(env: Env, n_students: int = 20, rounds: int = 3) -> Dict:
    """BKT 的 p_known 能不能预测下一次答对?配基准率做参照。"""
    scores: List[float] = []
    labels: List[bool] = []

    for index in range(n_students):
        learner = f"mp_{index}"
        new_learner(env, learner)
        student = env.simulator.make_student(learner)
        problem_ids = list(env.problems)

        for _ in range(rounds):
            for problem_id in problem_ids:
                problem = env.problems[problem_id]
                # 预测:题目涉及的每个知识点都按 predict_correct 算,取最弱(木桶)
                mastery = env.profile.get_mastery_map(learner, list(problem.kp_ids))
                predicted = min(
                    bkt.predict_correct(mastery.get(kp, config.BKT_P_INIT))
                    for kp in problem.kp_ids
                )

                correct = env.simulator.answer(student, problem)
                env.submit(learner, problem_id, correct)

                scores.append(predicted)
                labels.append(correct)

    base_rate = sum(1 for y in labels if y) / len(labels)
    curve = metrics.calibration_curve(scores, labels, bins=10)

    return {
        "n_predictions": len(labels),
        "base_rate": round(base_rate, 4),
        "auc": round(metrics.auc(scores, labels), 4),
        "brier": round(metrics.brier(scores, labels), 4),
        "brier_baseline_baserate": round(
            metrics.brier([base_rate] * len(labels), labels), 4
        ),
        "log_loss": round(metrics.log_loss(scores, labels), 4),
        "ece": round(metrics.expected_calibration_error(curve, len(labels)), 4),
        "calibration_curve": curve,
    }


# ================================================================ 表 2:遗忘预测

def exp_forgetting_calibration(env: Env, n_students: int = 20) -> Dict:
    """隔久不练之后,BKT 的预测还准吗?

    BKT 的公式里**没有时间项**,所以理论上它会一直觉得"你还会"。
    这一表就是去验证这个理论推断 —— 结论无论正反都照实写。
    """
    fresh_scores, fresh_labels = [], []
    stale_scores, stale_labels = [], []

    for index in range(n_students):
        problem_ids = list(env.problems)

        # (a) 连续练习,不插入遗忘
        fresh = f"fr_{index}"
        new_learner(env, fresh)
        student_fresh = env.simulator.make_student(fresh)
        for problem_id in problem_ids:
            problem = env.problems[problem_id]
            predicted = min(
                bkt.predict_correct(env.profile.get_mastery(fresh, kp))
                for kp in problem.kp_ids
            )
            correct = env.simulator.answer(student_fresh, problem)
            env.submit(fresh, problem_id, correct)
            fresh_scores.append(predicted)
            fresh_labels.append(correct)

        # (b) 先练一遍,然后让学生"忘掉",再考同一批题
        stale = f"st_{index}"
        new_learner(env, stale)
        student_stale = env.simulator.make_student(stale)
        for problem_id in problem_ids:
            problem = env.problems[problem_id]
            correct = env.simulator.answer(student_stale, problem)
            env.submit(stale, problem_id, correct)
            # ★ 必须记下"练过"的时间,否则 forget() 找不到基准,等于没忘
            for kp in problem.kp_ids:
                env.simulator.mark_practiced(student_stale, kp, day=0)

        # 时间流逝:真实能力衰减,但 BKT 的掌握度不会动
        env.simulator.forget(student_stale, day=30)
        for problem_id in problem_ids:
            problem = env.problems[problem_id]
            predicted = min(
                bkt.predict_correct(env.profile.get_mastery(stale, kp))
                for kp in problem.kp_ids
            )
            correct = env.simulator.answer(student_stale, problem)
            stale_scores.append(predicted)
            stale_labels.append(correct)

    fresh_curve = metrics.calibration_curve(fresh_scores, fresh_labels, bins=10)
    stale_curve = metrics.calibration_curve(stale_scores, stale_labels, bins=10)

    return {
        "fresh": {
            "n": len(fresh_labels),
            "auc": round(metrics.auc(fresh_scores, fresh_labels), 4),
            "brier": round(metrics.brier(fresh_scores, fresh_labels), 4),
            "ece": round(metrics.expected_calibration_error(fresh_curve, len(fresh_labels)), 4),
            "observed_rate": round(sum(fresh_labels) / len(fresh_labels), 4),
            "mean_predicted": round(sum(fresh_scores) / len(fresh_scores), 4),
            "calibration_curve": fresh_curve,
        },
        "stale_after_30_days": {
            "n": len(stale_labels),
            "auc": round(metrics.auc(stale_scores, stale_labels), 4),
            "brier": round(metrics.brier(stale_scores, stale_labels), 4),
            "ece": round(metrics.expected_calibration_error(stale_curve, len(stale_labels)), 4),
            "observed_rate": round(sum(stale_labels) / len(stale_labels), 4),
            "mean_predicted": round(sum(stale_scores) / len(stale_scores), 4),
            "calibration_curve": stale_curve,
        },
    }


# ================================================================ 表 3:规划增益

def _closure_mean_ability(env: Env, student, goal: str) -> float:
    """目标前置闭包里的真实能力均值(ground truth,不是 BKT)。"""
    closure = queries.prereq_closure(env.knowledge, goal)
    ids = list(closure) + [goal]
    return sum(student.own(kp) for kp in ids) / len(ids)


def exp_planning_gain(
    env: Env, n_students: int = 12, steps: int = 24, goal: str = GOAL
) -> Dict:
    """同样步数下,「按计划练」比「随机练」多涨多少。

    两组用**同一个随机种子**造学生,起点分布一致;练习步数相同;
    唯一差别是「下一步练哪个」由谁决定。

    **两个度量都报**,因为它们答案可能不一样:
    - `goal_effective_gain`:目标知识点本身的实际能力(计划优化的就是它)
    - `closure_mean_gain`:闭包内平均能力(摊平了,计划未必占优)
    只报对自己有利的那个就是挑数字,所以两个都写出来。
    """
    policies = ("planned", "planned_practicable", "random")
    results = {key: [] for key in policies}
    wasted = {key: 0 for key in policies}

    def pick_target(policy: str, learner: str, pool: List[str], rng) -> Optional[str]:
        if policy == "random":
            return rng.choice(pool)

        plan = env.service.plan_view(learner)
        items = [i["kp_id"] for i in plan["items"]]
        if not items:
            return rng.choice(pool)

        if policy == "planned":
            return items[0]

        # planned_practicable:跳过"没有题可做"的推荐,这正是在诊断上一行的问题
        for kp in items:
            if problems_for_kp(env, kp):
                return kp
        return rng.choice(pool)

    def run_one(policy: str, index: int) -> Tuple[float, float]:
        learner = f"plan_{policy}_{index}"
        new_learner(env, learner, goal=goal)
        student = env.simulator.make_student(learner)

        goal_before = env.simulator.effective(student, goal)
        mean_before = _closure_mean_ability(env, student, goal)

        pool = list(queries.prereq_closure(env.knowledge, goal)) + [goal]
        rng = random.Random(1000 + index)

        for _ in range(steps):
            target = pick_target(policy, learner, pool, rng)
            if target is None:
                continue

            candidates = problems_for_kp(env, target)
            if not candidates:
                # ★ 诊断计数器:计划推荐了没有题目的知识点,这一步就白费了
                wasted[policy] += 1
                continue

            problem = env.problems[rng.choice(candidates)]
            correct = env.simulator.answer(student, problem)
            env.submit(learner, problem.id, correct)

            # 练一次就长一点;真做对了长得多
            env.simulator.practice(student, target, gain=0.18 if correct else 0.10)
            env.simulator.mark_practiced(student, target, day=0)

        return (
            env.simulator.effective(student, goal) - goal_before,
            _closure_mean_ability(env, student, goal) - mean_before,
        )

    for index in range(n_students):
        for policy in policies:
            results[policy].append(run_one(policy, index))

    summary = {}
    for policy in policies:
        goal_gains = [g for g, _ in results[policy]]
        mean_gains = [m for _, m in results[policy]]
        total_steps = n_students * steps
        summary[policy] = {
            "goal_effective_gain": round(metrics.mean(goal_gains), 4),
            "closure_mean_gain": round(metrics.mean(mean_gains), 4),
            "wasted_steps": wasted[policy],
            "wasted_step_ratio": round(wasted[policy] / total_steps, 4),
            "goal_gains": [round(g, 4) for g in goal_gains],
            "closure_gains": [round(m, 4) for m in mean_gains],
        }

    def lift(policy: str, key: str) -> Optional[float]:
        value = summary[policy][key]
        baseline = summary["random"][key]
        if not baseline:
            return None
        return round((value - baseline) / abs(baseline), 4)

    return {
        "n_students": n_students,
        "steps": steps,
        **summary,
        "planned_goal_lift": lift("planned", "goal_effective_gain"),
        "planned_practicable_goal_lift": lift("planned_practicable", "goal_effective_gain"),
        "planned_closure_lift": lift("planned", "closure_mean_gain"),
        "planned_practicable_closure_lift": lift("planned_practicable", "closure_mean_gain"),
    }


# ================================================================ 表 4:根因定位 ★

def exp_root_cause(
    env: Env, n_students: int = 30, coverage: str = "full", top_k: int = 3
) -> Dict:
    """图遍历 vs 扁平召回,谁更能指出真正的根因。

    ground truth:模拟学生被埋下的那个薄弱知识点 W。
    目标知识点取 W 的后继(保证 W 真的是它的前置)。
    三个方法看到的**掌握度完全相同**(都由同一条答题轨迹学出来),
    差的只是「候选怎么选」和「怎么排序」。

    **三个臂,是为了把"图有没有用"拆干净:**
    | 臂 | 候选集 | 排序 |
    |---|---|---|
    | graph | 前置闭包(小) | depth 升序 + mastery 升序 |
    | closure_random | 前置闭包(同上) | 随机 |
    | flat | 全文相似度 top-N(大) | mastery 升序 |

    `graph` vs `closure_random` 回答「排序逻辑有没有用」;
    `closure_random` vs `flat` 回答「知道前置结构有没有用」。
    只看 graph vs flat 会把「候选集小」误当成「图推理强」。
    """
    flat = baselines.FlatRetrievalBaseline(env.knowledge)
    graph_pred, flat_pred, closure_random_pred = [], [], []
    truths = []
    skipped = 0

    # 只在「有后继」的知识点里埋短板 —— 没人依赖的点谈不上是别人的前置根因
    candidates = [
        c.id
        for c in env.knowledge.list_concepts()
        if env.knowledge.descendants(c.id, depth=3)
    ]
    if not candidates:
        return {"error": "图中没有可作为前置的薄弱点"}

    for index in range(n_students):
        learner = f"rc_{coverage}_{index}"
        new_learner(env, learner, goal=None)

        rng = random.Random(2000 + index)
        weakness = rng.choice(candidates)
        student = env.simulator.make_student(learner, weakness=weakness)

        # 让画像从真实答题里学。
        # ★ 必须把**薄弱点自己**也覆盖到,否则模型手里根本没有"W 很弱"这条证据,
        #   那测出来的就不是检索能力,而是通灵。相关知识点 = 各目标的前置闭包 ∪ 目标 ∪ W
        targets = list(env.knowledge.descendants(weakness, depth=3))
        relevant = {weakness}
        for target in targets:
            relevant |= set(queries.prereq_closure(env.knowledge, target)) | {target}

        if coverage == "full":
            kps = relevant
        else:
            # 稀疏:只答其中一部分(仍含薄弱点),测"证据少时掉多少"
            keep = max(2, int(len(relevant) * 0.4))
            sampled = set(rng.sample(sorted(relevant), k=min(keep, len(relevant))))
            kps = sampled | {weakness}

        for kp in kps:
            for problem_id in problems_for_kp(env, kp):
                problem = env.problems[problem_id]
                correct = env.simulator.answer(student, problem)
                env.submit(learner, problem_id, correct)

        if not targets:
            skipped += 1
            continue

        # ★ 两边看同一份掌握度:全图填满(未观测 = BKT 初始值),这样差别只在"候选怎么选"
        all_ids = [c.id for c in env.knowledge.list_concepts()]
        full_mastery = env.profile.get_mastery_map(learner, all_ids)
        observed = set(env.profile.get_mastery_map(learner))

        for target in targets:
            closure = queries.prereq_closure(env.knowledge, target)
            mastery = {kp: full_mastery.get(kp, config.BKT_P_INIT) for kp in all_ids}

            graph_roots = queries.root_causes(env.knowledge, mastery, target, top_k=top_k)
            flat_roots = flat.root_causes(
                mastery, target, top_k=top_k, observed=observed
            )

            # 同样用前置闭包当候选,但随机排序 —— 隔离掉"候选集大小"这个因素
            closure_ids = sorted(closure)
            shuffled = list(closure_ids)
            rng.shuffle(shuffled)
            closure_random_pred.append(shuffled[:top_k])

            graph_pred.append([r["kp_id"] for r in graph_roots])
            flat_pred.append([r["kp_id"] for r in flat_roots])
            truths.append(weakness)

    if not truths:
        return {"error": "没有可评的样本"}

    return {
        "n_cases": len(truths),
        "n_students": n_students,
        "coverage": coverage,
        "top_k": top_k,
        "skipped_no_successors": skipped,
        "graph_top1_accuracy": round(metrics.top1_accuracy(graph_pred, truths), 4),
        "graph_recall_at_k": round(metrics.recall_at_k(graph_pred, truths), 4),
        "closure_random_top1_accuracy": round(
            metrics.top1_accuracy(closure_random_pred, truths), 4
        ),
        "closure_random_recall_at_k": round(
            metrics.recall_at_k(closure_random_pred, truths), 4
        ),
        "flat_top1_accuracy": round(metrics.top1_accuracy(flat_pred, truths), 4),
        "flat_recall_at_k": round(metrics.recall_at_k(flat_pred, truths), 4),
        "random_guess_top1": round(1.0 / max(1, len(env.knowledge.list_concepts())), 4),
    }


# ================================================================ 输出

def render_markdown(results: Dict) -> str:
    m = results["mastery_prediction"]
    f = results["forgetting"]
    p = results["planning_gain"]
    rc_full = results["root_cause_full_coverage"]
    rc_sparse = results["root_cause_sparse_coverage"]

    lines = [
        "# coach 评测结果",
        "",
        f"> 由 `python -m coach.evaluation.run_eval` 生成,种子 {results['seed']}。",
        "> 完整数字见同目录的 `results.json`。",
        "",
        "## 表 1 掌握度预测(BKT 的 p_known 预测下一次答对)",
        "",
        f"- 样本数:{m['n_predictions']},基准正确率:{m['base_rate']}",
        "",
        "| 指标 | 值 | 平凡基线 |",
        "|---|---|---|",
        f"| AUC | **{m['auc']}** | 0.5(随机) |",
        f"| Brier(越小越好) | **{m['brier']}** | {m['brier_baseline_baserate']}(恒定预测基准率) |",
        f"| LogLoss | {m['log_loss']} | — |",
        f"| ECE(校准误差) | {m['ece']} | — |",
        "",
        "## 表 2 遗忘预测(隔 30 天不练之后)",
        "",
        "| 条件 | AUC | Brier | ECE | 平均预测 | 实际正确率 |",
        "|---|---|---|---|---|---|",
        f"| 连续练习 | {f['fresh']['auc']} | {f['fresh']['brier']} | {f['fresh']['ece']} "
        f"| {f['fresh']['mean_predicted']} | {f['fresh']['observed_rate']} |",
        f"| 隔 30 天 | {f['stale_after_30_days']['auc']} | {f['stale_after_30_days']['brier']} "
        f"| {f['stale_after_30_days']['ece']} | {f['stale_after_30_days']['mean_predicted']} "
        f"| {f['stale_after_30_days']['observed_rate']} |",
        "",
        "## 表 3 规划增益(同样步数,按计划练 vs 随机练)",
        "",
        f"- 学生数 {p['n_students']},每人 {p['steps']} 步",
        "",
        "| 策略 | 目标知识点能力提升 | 闭包内平均提升 | 白费步数 | 白费占比 |",
        "|---|---|---|---|---|",
        f"| 按计划(到期复习 + 根因补救 + 推进) | {p['planned']['goal_effective_gain']} "
        f"| {p['planned']['closure_mean_gain']} | {p['planned']['wasted_steps']} "
        f"| {p['planned']['wasted_step_ratio']} |",
        f"| 按计划(跳过没有题目的推荐) | {p['planned_practicable']['goal_effective_gain']} "
        f"| {p['planned_practicable']['closure_mean_gain']} "
        f"| {p['planned_practicable']['wasted_steps']} "
        f"| {p['planned_practicable']['wasted_step_ratio']} |",
        f"| 随机挑知识点 | **{p['random']['goal_effective_gain']}** "
        f"| {p['random']['closure_mean_gain']} | {p['random']['wasted_steps']} "
        f"| {p['random']['wasted_step_ratio']} |",
        f"| 相对随机提升(按计划) | **{p['planned_goal_lift']}** "
        f"| {p['planned_closure_lift']} | | |",
        f"| 相对随机提升(跳过无题项) | **{p['planned_practicable_goal_lift']}** "
        f"| {p['planned_practicable_closure_lift']} | | |",
        "",
        "> 「白费步数」= 计划推荐的**没有题目**的知识点,学生无从练起,这一步就空转了。",
        "> 第三行是对照:它的数字说明问题出在「推荐能不能落地」,而不在排序逻辑。",
        "",
        "## 表 4 ★ 根因定位:图遍历 vs 扁平召回",
        "",
        "| 条件 | 样本 | 臂 | Top-1 | Recall@k |",
        "|---|---|---|---|---|",
        f"| 作答覆盖充分 | {rc_full['n_cases']} | **图遍历(闭包 + depth/掌握度排序)** "
        f"| **{rc_full['graph_top1_accuracy']}** | {rc_full['graph_recall_at_k']} |",
        f"| | | 闭包内随机排序 | {rc_full['closure_random_top1_accuracy']} "
        f"| {rc_full['closure_random_recall_at_k']} |",
        f"| | | 扁平召回(文本相似度) | {rc_full['flat_top1_accuracy']} "
        f"| {rc_full['flat_recall_at_k']} |",
        f"| 作答稀疏 | {rc_sparse['n_cases']} | **图遍历** "
        f"| **{rc_sparse['graph_top1_accuracy']}** | {rc_sparse['graph_recall_at_k']} |",
        f"| | | 闭包内随机排序 | {rc_sparse['closure_random_top1_accuracy']} "
        f"| {rc_sparse['closure_random_recall_at_k']} |",
        f"| | | 扁平召回(文本相似度) | {rc_sparse['flat_top1_accuracy']} "
        f"| {rc_sparse['flat_recall_at_k']} |",
        "",
        f"> 无脑随机猜的 Top-1 是 {rc_full['random_guess_top1']}(1/知识点数)。",
        "> 「闭包内随机排序」这一臂是关键对照:它和图遍历拿到**同样的候选集**,"
        "只是不排序 —— 于是能把「知道前置结构」和「排序逻辑」两件事分开看。",
        "",
        render_limitations(results),
    ]
    return "\n".join(lines)


def render_limitations(results: Dict) -> str:
    """诚实交代局限。§11 明确要求:结论无论好坏都得把边界写清楚。

    下面几条是**由数字触发**的(条件成立才写),而不是写死的套话。
    """
    m = results["mastery_prediction"]
    f = results["forgetting"]
    p = results["planning_gain"]
    rc_full = results["root_cause_full_coverage"]
    rc_sparse = results["root_cause_sparse_coverage"]

    lines = [
        "## Limitations(必读)",
        "",
        "### 1. 这是模拟学生,不是真人",
        "",
        "模拟学生的答题模型与 BKT **不同**(有前置拖累、逻辑函数、遗忘),所以不是循环论证;",
        "但结论仍然只在「这套假设」里成立。它验证的是**算法能否收敛到一个未知的真实过程**,"
        "不是「学生用了会不会变好」。",
        "",
        "### 2. 扁平基线偏弱,别把差距当成定论",
        "",
        "对照组用的是**字符二元组余弦相似度**,不是真正的向量 embedding。"
        "真 embedding 语义泛化更强,很可能显著缩小差距。要下更硬的结论,"
        "得换成 Chroma + 真实 embedding(`rag_core/chroma_store.py` 已具备,"
        "但那要联网调服务,评测的可复现性会下降)。",
        "",
    ]

    if m["auc"] < 0.6 or m["brier"] > m["brier_baseline_baserate"]:
        lines += [
            "### 3. ★ 表 1 是负面结果:BKT 的掌握度预测基本不可用",
            "",
            f"AUC 只有 {m['auc']}(随机是 0.5),Brier {m['brier']} "
            f"**比恒定预测基准率的 {m['brier_baseline_baserate']} 还差**。",
            "",
            "原因有两个,都在 §11 记过:",
            "- §5.3 的 `P_init = 0.1` 系统性低估,前几次预测必然偏低;",
            "- 连续答对约 10 次后 `p_known` 饱和到 1.0,之后一律预测「会」,失去判别力。",
            "",
            "**这不是实现 bug,是照抄公式的必然结果。**要改善得动 BKT 的参数或公式本身。",
            "",
        ]

    if f["stale_after_30_days"]["ece"] > f["fresh"]["ece"]:
        lines += [
            "### 4. 表 2 印证了「BKT 不建模遗忘」",
            "",
            f"连续练习时 ECE = {f['fresh']['ece']},隔 30 天不练后升到 "
            f"{f['stale_after_30_days']['ece']}。",
            f"隔 30 天后模型平均预测 {f['stale_after_30_days']['mean_predicted']} 的正确率,"
            f"实际只有 {f['stale_after_30_days']['observed_rate']} —— **严重高估**。",
            "",
            "这正是需要把 SM-2 的 `due_at` 和 BKT 的掌握度**分开看**的理由:"
            "一个建模「会不会」,一个建模「忘没忘」。",
            "",
        ]

    if (p["planned_practicable_goal_lift"] is None) or p["planned_practicable_goal_lift"] <= 0.05:
        lines += [
            "### 5. ★ 表 3 是负面结果:计划没跑赢随机",
            "",
            f"修掉「推荐没有题目的知识点」之后,按计划练相对随机是 "
            f"**{p['planned_practicable_goal_lift']}**,基本打平。",
            "",
            "当前计划的价值**没有被这次实验证实**。可能的原因:",
            "- 计划总推「最浅的根因」那个点,反复练同一点,边际收益递减太快;",
            "- 随机策略天然分散练习面,而模拟学生的能力增益是「按点算」的,"
            "分散反而更快抬高闭包均值;",
            "- 本题库只有 13 道题,覆盖不到很多知识点,计划的可选空间被压得很窄。",
            "",
            "要真正证明规划有用,得先解决「推荐必须可落地」(出题能力),再重跑。",
            "",
        ]

    if rc_sparse["graph_top1_accuracy"] < rc_sparse["closure_random_top1_accuracy"]:
        lines += [
            "### 6. ★ 证据稀疏时,排序规则反而是有害的",
            "",
            f"作答充分时,图遍历 {rc_full['graph_top1_accuracy']} > 闭包内随机 "
            f"{rc_full['closure_random_top1_accuracy']},排序有用;",
            f"但作答稀疏时反过来:{rc_sparse['graph_top1_accuracy']} < "
            f"{rc_sparse['closure_random_top1_accuracy']}。",
            "",
            "解释:`root_causes` 先按 depth 升序、再按掌握度升序。当大量前置**从没被观测过**"
            "(掌握度停在初始值 0.1)时,这些「没见过的浅层点」会被排到真正观测到的弱项前面。"
            "证据越稀疏,这个偏差越严重。",
            "",
            "修法方向:区分「未观测」与「已观测且弱」——未观测的点不该当作缺口参与排序。",
            "",
        ]

    if rc_sparse["flat_top1_accuracy"] < 0.15:
        lines += [
            "### 7. 扁平基线的成绩很低,但它确实反映了方法本身",
            "",
            f"扁平召回 Top-1 只有 {rc_full['flat_top1_accuracy']},Recall@k "
            f"{rc_full['flat_recall_at_k']}。",
            "机制是清楚的:它按「和症状文本相似」挑候选,而根因**恰恰常常和症状不像**"
            "(比如「动态规划做不出来」的根因是「函数调用」)。",
            "",
            "但注意第 2 条 —— 换成真 embedding 后这个数字会变,不能只看这一个数就下结论。",
            "",
        ]

    lines += [
        "### 8. 规模小",
        "",
        f"只有 {results['graph']['concepts']} 个知识点、{results['graph']['problems']} 道题、"
        f"{results['n_students']} 个模拟学生。单个数字会有随机波动,"
        "换成别的种子结论方向可能变。`results.json` 里存了原始数组,可复算。",
        "",
    ]
    return "\n".join(lines)


def run_all(
    n_students: int = 20,
    seed: int = 0,
    plan_students: Optional[int] = None,
    plan_steps: int = 24,
    root_students: Optional[int] = None,
) -> Dict:
    """跑四张表。`plan_*` / `root_students` 是为了让测试能用很小的规模跑通。"""
    env = Env(seed=seed)
    try:
        started = time.time()
        plan_n = plan_students if plan_students is not None else max(6, n_students // 2)
        root_n = root_students if root_students is not None else n_students
        results = {
            "seed": seed,
            "n_students": n_students,
            "graph": {
                "concepts": len(env.knowledge.list_concepts()),
                "edges": len(env.knowledge.list_edges()),
                "problems": len(env.knowledge.list_problems()),
            },
            "mastery_prediction": exp_mastery_prediction(env, n_students=n_students),
            "forgetting": exp_forgetting_calibration(env, n_students=n_students),
            "planning_gain": exp_planning_gain(env, n_students=plan_n, steps=plan_steps),
            "root_cause_full_coverage": exp_root_cause(
                env, n_students=root_n, coverage="full"
            ),
            "root_cause_sparse_coverage": exp_root_cause(
                env, n_students=root_n, coverage="sparse"
            ),
            "elapsed_seconds": round(time.time() - started, 2),
        }
        return results
    finally:
        env.close()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="coach.evaluation.run_eval", description="跑四张评测表")
    parser.add_argument("--n", type=int, default=20, help="模拟学生数")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--out", default=str(RESULTS_DIR), help="结果输出目录")
    args = parser.parse_args(argv)

    print(f"开始评测:n={args.n} seed={args.seed}")
    results = run_all(n_students=args.n, seed=args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown = render_markdown(results)
    (out_dir / "RESULTS.md").write_text(markdown, encoding="utf-8")

    print(markdown)
    print(f"\n已写入 {out_dir/'results.json'} 与 {out_dir/'RESULTS.md'}")
    print(f"耗时 {results['elapsed_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
