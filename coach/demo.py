"""两分钟 demo:P6.3 的交付物(work.md §7)。

**为什么是一个脚本而不是一段录像:** 录像看完就完了,脚本可以被复跑、被 diff、
被 CI 跑。README 里放的是它的输出。

它做的事就是走一遍真实链路,一步都不省:

    建图灌题建档 → 起真实 HTTP 服务 → POST /answer(交答案)
      → profile_worker 消费(判分 / BKT / SM-2)→ 发 profile.updated
      → 答题链的下一环算根因、写人话解释
      → GET /gap   看根因
      → tick       发定时事件
      → GET /plan  看今日计划

**零外部依赖**:用进程内总线,不需要 Redis、Docker、LLM key。
所以它在任何 `git clone` 之后都能跑起来 —— 这也是它敢直接进 README 的原因。

用法:
    .venv/Scripts/python.exe -m coach.demo
    .venv/Scripts/python.exe -m coach.demo --learner u1 --goal algo.dp
"""

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

import uvicorn

from coach import config
from coach.api.main import create_app
from coach.coordination.journal import Journal
from coach.coordination.memory import InMemoryBus
from coach.events import schema as events
from coach.knowledge import builder, problem_bank
from coach.knowledge.governance import Governance
from coach.knowledge.schema import open_db
from coach.knowledge.store import KnowledgeStore
from coach.profile.store import ProfileStore
from coach.workflow.query import QueryService
from coach.workers.planner_worker import PlannerWorker
from coach.workers.profile_worker import ProfileWorker
from coach.workers.scheduler_worker import SchedulerWorker

LINE = "─" * 72


def title(text: str) -> None:
    print(f"\n{LINE}\n  {text}\n{LINE}")


def show(label: str, payload) -> None:
    print(f"\n▸ {label}")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------- 起服务

class DemoServer:
    """在后台线程里跑真实的 uvicorn,这样 demo 打的是真 HTTP,不是内部调用。"""

    def __init__(self, app, port: int):
        self._config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
        self._server = uvicorn.Server(self._config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self.base = f"http://127.0.0.1:{port}"

    def __enter__(self):
        self._thread.start()
        deadline = time.time() + 10
        while not self._server.started:
            if time.time() > deadline:
                raise RuntimeError("服务起不来,检查端口是否被占用")
            time.sleep(0.05)
        return self

    def __exit__(self, *exc):
        self._server.should_exit = True
        self._thread.join(timeout=5)

    def get(self, path: str):
        with urllib.request.urlopen(f"{self.base}{path}", timeout=10) as response:
            return json.loads(response.read())

    def post(self, path: str, body: dict):
        request = urllib.request.Request(
            f"{self.base}{path}",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())


# ---------------------------------------------------------------- 主流程

def pump(bus: InMemoryBus, worker, name: str) -> int:
    """让某个 worker 把它那条流上的消息消费掉(模拟常驻消费循环跑一轮)。"""
    messages = bus.consume(worker.stream, worker.group, worker.consumer)
    for _msg_id, event in messages:
        worker.process(event)
    if messages:
        print(f"  · {name} 消费了 {len(messages)} 条消息")
    return len(messages)


def run(learner: str, goal: str, db_path, port: int) -> None:
    print(f"数据库:{db_path}")
    conn = open_db(db_path, check_same_thread=False)
    knowledge = KnowledgeStore(conn)
    profile = ProfileStore(conn)
    journal = Journal(conn)

    # ---- 1. 建图 + 灌题 + 建档
    title("① 建图:22 个知识点,前置关系经过「观察 → 提案 → 聚合」治理流程")
    report = builder.seed_graph(Governance(knowledge))
    problems = builder.seed_problems(knowledge)
    print(f"  {report.summary()}")
    print(f"  题目:{problems} 道")
    print(f"  图规模:知识点 {len(knowledge.list_concepts())} / 前置边 {len(knowledge.list_edges())}")
    print(f"  无环校验:{'通过' if not _has_cycle(knowledge) else '★ 成环'}")

    profile.upsert_profile(learner, goal_kp_id=goal, daily_minutes=30)
    print(f"  学习者 {learner} 已建档,目标 = {goal}")

    # ---- 2. 装配
    service = QueryService(knowledge, profile)
    bus = InMemoryBus()
    profile_worker = ProfileWorker(profile, knowledge, journal, bus=bus)
    planner_worker = PlannerWorker(knowledge, profile, journal, bus=bus, service=service)
    scheduler_worker = SchedulerWorker(knowledge, profile, journal, bus=bus, service=service)

    app = create_app(bus=bus, query_service=service)

    with DemoServer(app, port) as server:
        # ---- 3. 交一个正确答案
        title("② POST /answer —— 交一个正确答案(入口只入队,不判分不调 LLM)")
        started = time.perf_counter()
        status, accepted = server.post(
            "/answer",
            {
                "learner_id": learner,
                "problem_id": "lc.70",
                "answer_text": problem_bank.expected_answer(knowledge, "lc.70"),
                "elapsed_ms": 4200,
            },
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        print(f"\n  HTTP {status} —— 耗时 {elapsed_ms:.1f}ms(判分和 LLM 都还没发生)")
        show("响应", accepted)

        # ---- 4. 异步消费
        title("③ 异步消费:判分 → BKT 掌握度 → SM-2 记忆状态")
        pump(bus, profile_worker, "profile_worker")
        memory = profile.get_memory(learner, "algo.dp")
        print(f"  algo.dp 掌握度:{profile.get_mastery(learner, 'algo.dp'):.3f}")
        print(f"  SM-2 状态:间隔 {memory['interval_days']:.0f} 天,"
              f"下次复习在 {time.strftime('%Y-%m-%d', time.localtime(memory['due_at']))}")

        # ---- 5. 根因
        title("④ 根因定位:沿前置依赖反向遍历,找真正缺的那块基础")
        gap = server.get(f"/gap/{learner}/{goal}")
        print(f"\n  目标「{gap['name']}」当前掌握度 {gap['mastery']:.2f}")
        print("\n  根因候选(按重要度):")
        for index, root in enumerate(gap["root_causes"], 1):
            flag = "已确认薄弱" if root["status"] == "gap" else "还没测过"
            chain = " → ".join(root["path"])
            print(f"    {index}. {root['name']}({flag},掌握度 {root['mastery']:.2f})")
            print(f"       依赖路径:{chain}")
        print(f"\n  解释[{gap['explanation_source']}]:{gap['explanation']}")

        # ---- 6. 交一个错答案
        title("⑤ 再交一个错答案 —— 看掌握度和复习计划怎么变")
        server.post(
            "/answer",
            {
                "learner_id": learner,
                "problem_id": "lc.53",
                "answer_text": "__wrong__",
                "elapsed_ms": 2100,
            },
        )
        pump(bus, profile_worker, "profile_worker")
        pump(bus, planner_worker, "planner_worker")
        errors = profile.list_errors(learner)
        print(f"\n  algo.dp 掌握度:{profile.get_mastery(learner, 'algo.dp'):.3f}(答错会掉)")
        print(f"  易错记录:{errors}")

        # ---- 7. 计划
        title("⑥ 定时触发 → GET /plan —— 今日该练什么")
        bus.publish(config.STREAM_TICK, events.new_event(events.TICK_SCHEDULED, learner))
        pump(bus, scheduler_worker, "scheduler_worker")

        # 把一条到期时间挪到过去,让复习项出现在计划里
        memory = profile.get_memory(learner, "algo.dp")
        profile.set_memory(
            learner, "algo.dp", ease=memory["ease"], interval_days=1, reps=memory["reps"],
            lapses=memory["lapses"], last_review=memory["last_review"],
            due_at=time.time() - 86400,
        )
        plan = server.get(f"/plan/{learner}")
        print(f"\n  日期 {plan['date']},预算 {plan['daily_minutes']} 分钟,"
              f"已排 {plan['planned_minutes']} 分钟")
        for item in plan["items"]:
            print(f"    [{item['action']:8}] {item['name']}(约 {item['est_minutes']:.0f} 分钟)")
            print(f"               {item['reason']}")

        # ---- 8. 收尾
        title("⑦ 收尾:幂等与日志")
        counts = journal.counts()
        print(f"  journal:共 {counts['total']} 条,已完成 {counts['done']},未完成 {counts['pending']}")
        print(f"  同一条消息重复投递不会重复计分(processed_events + answers 唯一键两道闸)")
        print(f"\n  Swagger 文档:{server.base}/docs(服务已停,重跑本 demo 可访问)")


def _has_cycle(knowledge: KnowledgeStore) -> bool:
    from coach.knowledge.governance import has_cycle

    return has_cycle(knowledge.prerequisite_pairs())


def _force_utf8_stdout() -> None:
    """Windows 控制台默认 GBK,打印框线字符/箭头会直接 UnicodeEncodeError 崩掉。

    强制 UTF-8 并允许替换,最坏情况是字符显示成问号,但不会再中断 demo。
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def main(argv: Optional[list] = None) -> int:
    _force_utf8_stdout()

    parser = argparse.ArgumentParser(prog="coach.demo", description="两分钟走完一条完整链路")
    parser.add_argument("--learner", default="demo", help="学习者 id")
    parser.add_argument("--goal", default="algo.dp", help="目标知识点 id")
    parser.add_argument(
        "--db",
        default=str(config.DATA_DIR / "demo.db"),
        help="demo 用单独的库文件,不碰你的 coach.db",
    )
    parser.add_argument("--port", type=int, default=8765, help="demo 服务的端口")
    args = parser.parse_args(argv)

    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    # 每次从干净状态开始,保证 demo 可复跑
    for suffix in ("", "-wal", "-shm"):
        Path(str(args.db) + suffix).unlink(missing_ok=True)

    run(args.learner, args.goal, args.db, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
