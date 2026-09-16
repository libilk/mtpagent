"""coach 测试共享夹具。

关键点:**测试不依赖真 Redis**。`InMemoryBus` 满足 coordination.Bus 协议,
worker 只依赖协议,所以业务逻辑可以完整测掉;真 Redis 只在端到端演示时用。
"""

from typing import Dict

import pytest

from coach.coordination.journal import Journal
from coach.coordination.memory import InMemoryBus
from coach.domain.models import Problem
from coach.knowledge import builder, schema
from coach.knowledge.governance import Governance
from coach.knowledge.store import KnowledgeStore
from coach.profile.store import ProfileStore

#: 别名:早先这个假总线就住在本文件里,测试里到处叫 FakeBus,保留名字省得改一堆
FakeBus = InMemoryBus


@pytest.fixture
def db_conn(tmp_path):
    """图与画像共用一个连接(与生产同库同连接的形态一致)。

    check_same_thread=False:worker 用 asyncio.to_thread 把阻塞的 SQLite 调用
    丢到线程池执行,连接必然跨线程——这是生产要求,不是测试的方便。
    """
    conn = schema.connect(tmp_path / "coach.db", check_same_thread=False)
    schema.init_schema(conn)
    yield conn
    conn.close()


@pytest.fixture
def knowledge(db_conn) -> KnowledgeStore:
    return KnowledgeStore(db_conn)


@pytest.fixture
def profile(db_conn) -> ProfileStore:
    return ProfileStore(db_conn)


@pytest.fixture
def journal(db_conn) -> Journal:
    return Journal(db_conn)


@pytest.fixture
def fake_bus() -> FakeBus:
    return FakeBus()


@pytest.fixture
def seeded(knowledge) -> KnowledgeStore:
    """把金标准种子图灌好(22 点 / 27 边)。"""
    builder.seed_graph(Governance(knowledge))
    return knowledge


@pytest.fixture
def worker(profile, knowledge, journal, fake_bus):
    from coach.workers.profile_worker import ProfileWorker

    return ProfileWorker(profile, knowledge, journal, bus=fake_bus)


@pytest.fixture
def problem(knowledge) -> Problem:
    """一道挂在 algo.dp 上的题,供 worker 判分测试使用。"""
    p = Problem(
        id="lc.70",
        title="爬楼梯",
        source="leetcode:70",
        difficulty=2.0,
        judge_type="exact_output",
        test_cases=[{"input": "3", "expected": "3"}, {"input": "5", "expected": "8"}],
        kp_ids=["algo.dp"],
    )
    knowledge.upsert_problem(p)
    return p
