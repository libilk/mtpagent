"""FastAPI 应用装配。

`create_app(bus=..., query_service=...)` 允许注入:
- 生产/演示:RedisBus + 连真实库的 QueryService(默认)
- 测试:内存假总线 + 内存库门面,不需要起 Redis

注意 `api/` 只依赖 **workflow 层的门面**,不直接 import 图或画像——
查询算法在 knowledge/queries.py,门面在 workflow/query.py。
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI

from coach.api.routes import router
from coach.coordination.redis import RedisBus
from coach.workflow.query import build_default_query_service

logger = logging.getLogger(__name__)


def create_app(bus: Optional[object] = None, query_service: Optional[object] = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        for target in (app.state.bus, app.state.query_service):
            close = getattr(target, "close", None)
            if callable(close):
                close()

    app = FastAPI(
        title="coach API",
        version="0.1.0",
        description=(
            "学习根因定位 Agent 的入口。\n\n"
            "本层只做**校验 + 入队 + 查询**,不含任何 LLM 调用。\n\n"
            "- `POST /answer` 入队后立刻返回,处理由 worker 异步完成\n"
            "- `GET /gap` 根因定位:沿前置依赖反向遍历找真正的薄弱基础"
        ),
        lifespan=lifespan,
    )
    app.state.bus = bus if bus is not None else RedisBus()
    # 默认连 data/coach/coach.db;RedisBus 与 SQLite 都是惰性的,导入本模块不会失败
    app.state.query_service = (
        query_service if query_service is not None else build_default_query_service()
    )
    app.include_router(router)
    return app


app = create_app()
