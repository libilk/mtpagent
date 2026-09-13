"""开发期单进程拉起全部 worker(work.md §6 / §8)。

启动顺序很重要:
    1. 打开共享连接(图 + 画像 + journal 同库)
    2. **先跑 recovery** —— 把上次进程挂掉时留下的「已记未完成」事件补做;
       两个 worker 共用一个 journal,所以**各自只认领自己处理的事件类型**
    3. 再并发进入各自的消费循环

用法:
    .venv/Scripts/python.exe -m coach.workers.run_all
"""

import asyncio
import logging
import signal

from coach import config
from coach.coordination.journal import Journal
from coach.coordination.redis import RedisBus
from coach.coordination.recovery import Recovery
from coach.knowledge.schema import connect
from coach.knowledge.store import KnowledgeStore
from coach.profile.store import ProfileStore
from coach.workers.planner_worker import PlannerWorker
from coach.workers.profile_worker import ProfileWorker
from coach.workers.scheduler_worker import SchedulerWorker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)
logger = logging.getLogger("coach.run_all")


def build_workers(db_path=None, llm=None):
    """装配全部 worker。连接按 worker 的跨线程用法开(check_same_thread=False)。"""
    conn = connect(db_path, check_same_thread=False)
    knowledge = KnowledgeStore(conn)
    profile = ProfileStore(conn)
    journal = Journal(conn)
    bus = RedisBus()

    from coach.workflow.query import QueryService

    service = QueryService(knowledge, profile)
    workers = [
        ProfileWorker(profile, knowledge, journal, bus=bus),
        # llm=None 时 planner 只算根因、不写人话解释(GET /gap 会用模板兜底)
        PlannerWorker(knowledge, profile, journal, llm=llm, bus=bus, service=service),
        SchedulerWorker(knowledge, profile, journal, bus=bus, service=service),
    ]
    return workers, bus, conn


async def main_async() -> None:
    workers, bus, conn = build_workers()

    if not bus.ping():
        logger.error("连不上 Redis(%s)。先起容器:docker start coach-redis", config.REDIS_URL)
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

    loop = asyncio.get_running_loop()
    stop = lambda: [w.stop() for w in workers]  # noqa: E731
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop)
        except NotImplementedError:  # Windows 上部分信号不支持
            pass

    try:
        await asyncio.gather(*(w.run() for w in workers))
    finally:
        logger.info("journal 统计:%s", workers[0].journal.counts())
        bus.close()
        conn.close()
        logger.info("worker 已停止")


def main() -> None:
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("收到中断,退出")


if __name__ == "__main__":
    main()
