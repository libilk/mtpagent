"""入口层:FastAPI,只做校验 + 入队 + 查询。

硬约束:禁止 import coach.knowledge 或 coach.profile。
"""

from coach.api.main import app, create_app

__all__ = ["app", "create_app"]
