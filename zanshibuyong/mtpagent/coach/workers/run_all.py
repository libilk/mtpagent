"""开发期单进程拉起全部 worker + 定时 tick(work.md §6 / §8)。

启动顺序:
    1. 打开共享连接(图 + 画像 + journal 同库)
    2. `--init-learner` 建档(可选)
    3. **先跑 recovery** —— 补做上次进程挂掉时「已记未完成」的事件;
       两个 worker 共用一个 journal,所以各自只认领自己处理的事件类型
    4. 并发跑:各 worker 的消费循环 + 一个定时发布 `tick.scheduled` 的任务

`tick.scheduled` 必须有人发,scheduler_worker 才有活干 —— 所以定时任务在这里,
而不是让 scheduler 自己 poll(那样就绕过了事件链,recovery 也管不到它)。

用法:
    # 建档 + 起 worker(60 秒一次 tick)
    .venv/Scripts/python.exe -m coach.workers.run_all --init-learner u1 --goal algo.dp

    # 只发一次 tick 然后退出(手动触发,不需要常驻)
    .venv/Scripts/python.exe -m coach.workers.run_all --tick-once u1

    # 没有 LLM key 时降级:/gap 会返回模板文案而不是人话
    .venv/Scripts/python.exe -m coach.workers.run_all --no-llm
"""

import argparse
import asyncio
import logging
import os
import signal
from typing import List, Optional

from coach import config
from coach.coordination.journal import Journal
from coach.coordination.redis import RedisBus
from coach.coordination.recovery import Recovery
from coach.events import schema as events
from coach.knowledge import builder
from coach.knowledge.schema import open_db
from coach.knowledge.store import KnowledgeStore
from coach.profile.store import ProfileStore
from coach.workers.planner_worker import PlannerWorker
from coach.workers.profile_worker import ProfileWorker
from coach.workers.scheduler_worker import SchedulerWorker
from coach.workflow.pipeline import make_checkpointer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)
logger = logging.getLogger("coach.run_all")

DEFAULT_TICK_INTERVAL = 60.0


def build_llm(enabled: bool):
    """装配 LLM。没配 key 就返回 None —— planner 会退化成只算根因、不写人话。"""
    if not enabled:
        logger.warning("--no-llm:根因解释将退化为模板文案")
        return None
    if not os.getenv("DASHSCOPE_API_KEY"):
        logger.warning("未设置 DASHSCOPE_API_KEY:根因解释将退化为模板文案")
        return None
    try:
        from llm.llm_client import LLM

        llm = LLM()
        logger.info("LLM 已装配:%s", llm.model_name)
        return llm
    except Exception:  # noqa: BLE001 — 装配失败不该拖垮整个 worker 进程
        logger.exception("LLM 装配失败,降级为模板文案")
        return None


def make_bus(kind: str = "redis"):
    """总线实现。memory 只用于离线 demo / 本地跑通,不具备消费组与持久化。"""
    if kind == "memory":
        from coach.coordination.memory import InMemoryBus

        return InMemoryBus()
    return RedisBus()


def build_workers(
    db_path=None, llm=None, checkpoint_path=None, bus=None, agent_diagnosis: bool = False
) -> tuple:
    """装配全部 worker。连接按 worker 的跨线程用法开(check_same_thread=False)。

    `agent_diagnosis=True` 会让 planner_worker 在答题后额外跑一次 agent 诊断。
    **默认关**:一次诊断是好几轮 LLM 调用(贵且慢),不能默认挂在每条事件上。
    """
    conn = open_db(db_path, check_same_thread=False)
    knowledge = KnowledgeStore(conn)
    profile = ProfileStore(conn)
    journal = Journal(conn)
    bus = bus if bus is not None else RedisBus()

    from coach.workflow.query import QueryService

    service = QueryService(knowledge, profile)
    workers = [
        ProfileWorker(
            profile,
            knowledge,
            journal,
            bus=bus,
            llm=llm,
            checkpointer=make_checkpointer(checkpoint_path),
        ),
        PlannerWorker(
            knowledge,
            profile,
            journal,
            llm=llm,
            bus=bus,
            service=service,
            agent_diagnosis=agent_diagnosis,
        ),
        SchedulerWorker(knowledge, profile, journal, bus=bus, service=service),
    ]
    return workers, bus, conn, profile


def seed(db_path=None, learner_id: Optional[str] = None, goal: Optional[str] = None,
         daily_minutes: int = 30) -> dict:
    """把图、题目、学习者都准备好。只碰 SQLite,不依赖 Redis。

    `--seed-only` 走这条路,给 docker-compose 的一次性初始化用。
    """
    conn = open_db(db_path, check_same_thread=False)
    try:
        knowledge = KnowledgeStore(conn)
        profile = ProfileStore(conn)

        from coach.knowledge.governance import Governance

        report = builder.seed_graph(Governance(knowledge))
        problems = builder.seed_problems(knowledge)
        if learner_id:
            init_learner(profile, learner_id, goal, daily_minutes)

        return {
            "concepts": len(knowledge.list_concepts()),
            "edges": len(knowledge.list_edges()),
            "problems": len(knowledge.list_problems()),
            "seed_report": report.summary(),
            "seeded_problems": problems,
        }
    finally:
        conn.close()


def init_learner(
    profile: ProfileStore, learner_id: str, goal_kp_id: Optional[str], daily_minutes: int
) -> None:
    """建档。没有 goal 的话,`/plan` 只会给出复习项,不会推新知识点。"""
    profile.upsert_profile(
        learner_id, goal_kp_id=goal_kp_id, daily_minutes=daily_minutes
    )
    logger.info(
        "学习者已建档:%s goal=%s 每日 %d 分钟", learner_id, goal_kp_id or "(未设置)", daily_minutes
    )


def publish_tick(bus: RedisBus, learner_ids: List[str]) -> int:
    sent = 0
    for learner_id in learner_ids:
        event = events.new_event(events.TICK_SCHEDULED, learner_id, payload={})
        bus.publish(config.STREAM_TICK, event)
        sent += 1
    return sent


async def tick_loop(bus: RedisBus, profile: ProfileStore, interval: float) -> None:
    """定时给每个已建档学习者发 tick.scheduled。"""
    while True:
        await asyncio.sleep(interval)
        learner_ids = profile.list_learners()
        if not learner_ids:
            logger.info("没有已建档的学习者,tick 跳过")
            continue
        sent = publish_tick(bus, learner_ids)
        logger.info("已发布 tick:%d 个学习者", sent)


async def main_async(args) -> None:
    # `--seed-only`:只把图/题目/学习者准备好就退出。给容器初始化用,不碰 Redis。
    if args.seed_only:
        stats = seed(args.db, args.init_learner, args.goal, args.daily_minutes)
        logger.info("初始化完成:%s", stats)
        return

    llm = build_llm(enabled=not args.no_llm)
    if args.agent_diagnosis and llm is None:
        logger.warning("要求了 --agent-diagnosis 但没有 LLM —— 这一项会静默跳过")
    workers, bus, conn, profile = build_workers(
        args.db,
        llm=llm,
        checkpoint_path=args.checkpoint_db,
        bus=make_bus(args.bus),
        agent_diagnosis=args.agent_diagnosis,
    )

    # 建档只碰 SQLite,不依赖 Redis —— 放在 ping 之前,免得没起 Redis 就建不了档案
    if args.init_learner:
        init_learner(profile, args.init_learner, args.goal, args.daily_minutes)

    if not bus.ping():
        logger.error("连不上 Redis(%s)。先起容器:docker start coach-redis", config.REDIS_URL)
        conn.close()
        return

    if args.tick_once:
        init_learner(profile, args.tick_once, args.goal, args.daily_minutes)
        sent = publish_tick(bus, [args.tick_once])
        logger.info("手动触发 tick:%s(%d 条),等 worker 消费…", args.tick_once, sent)
        await asyncio.sleep(args.tick_once_wait)
        logger.info("journal 统计:%s", workers[0].journal.counts())
        bus.close()
        conn.close()
        return

    recovery = Recovery(journal=workers[0].journal, bus=bus)
    for worker in workers:
        stats = recovery.run(
            handler=worker.process,
            stream=worker.stream,
            group=worker.group,
            consumer=worker.consumer,
            types=worker.handled_types,
        )
        logger.info(
            "%s 启动恢复:补做 %d 条,清理未 ACK %d 条",
            type(worker).__name__,
            stats["replayed"],
            stats["reclaimed"],
        )

    tasks = [asyncio.create_task(w.run()) for w in workers]
    tasks.append(asyncio.create_task(tick_loop(bus, profile, args.tick_interval)))

    loop = asyncio.get_running_loop()
    stop = lambda: [t.cancel() for t in tasks]  # noqa: E731
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop)
        except NotImplementedError:  # Windows 上部分信号不支持
            pass

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        logger.info("收到停止信号")
    finally:
        logger.info("journal 统计:%s", workers[0].journal.counts())
        bus.close()
        conn.close()
        logger.info("worker 已停止")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="coach.workers.run_all", description="拉起全部 worker + 定时 tick")
    parser.add_argument("--db", default=None, help="SQLite 路径,默认 data/coach/coach.db")
    parser.add_argument(
        "--checkpoint-db",
        default=str(config.DATA_DIR / "checkpoints.db"),
        help="LangGraph 检查点库(单独一个文件,避免和主库事务互相干扰)",
    )
    parser.add_argument("--no-llm", action="store_true", help="不装配 LLM(解释退化为模板)")
    parser.add_argument(
        "--agent-diagnosis",
        action="store_true",
        help="答题后额外跑 agent 诊断(LLM 自主调工具)。★ 贵:一次好几轮调用,默认关",
    )
    parser.add_argument(
        "--bus",
        choices=("redis", "memory"),
        default="redis",
        help="总线实现。memory = 进程内、无持久化,只给离线 demo 用",
    )
    parser.add_argument(
        "--seed-only",
        action="store_true",
        help="只建图/灌题/建档然后退出(容器初始化用,不依赖 Redis)",
    )
    parser.add_argument("--init-learner", metavar="LEARNER_ID", help="建档一个学习者")
    parser.add_argument("--goal", default=None, help="目标知识点 id,如 algo.dp")
    parser.add_argument("--daily-minutes", type=int, default=30, help="每日学习预算(分钟)")
    parser.add_argument(
        "--tick-interval", type=float, default=DEFAULT_TICK_INTERVAL, help="定时 tick 间隔秒数"
    )
    parser.add_argument("--tick-once", metavar="LEARNER_ID", help="只发一次 tick 然后退出")
    parser.add_argument("--tick-once-wait", type=float, default=3.0, help="发完 tick 后等多久再退")
    args = parser.parse_args(argv)

    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        logger.info("收到中断,退出")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
