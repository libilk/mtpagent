"""coach 全局配置:路径、算法默认参数、队列名。

数值来源见 work.md §5(BKT §5.3 / SM-2 §5.4),**不要在这里重新调参**。
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "coach"
DB_PATH = DATA_DIR / "coach.db"


def db_path() -> Path:
    """返回默认 SQLite 路径,并确保父目录存在。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DB_PATH


# --- 图遍历 ---
MAX_PREREQ_DEPTH = 3
GAP_THRESHOLD = 0.4          # 掌握度低于此值算缺口(§5.5)
ROOT_CAUSE_TOP_K = 3

# --- 写入治理 ---
MIN_EDGE_CONFIDENCE = 0.6    # 低于此置信度的提案直接拒绝(§4.4)

# --- BKT 掌握度(§5.3)---
BKT_P_INIT = 0.1
BKT_TRANSIT = 0.15
BKT_GUESS = 0.2
BKT_SLIP = 0.1

# --- SM-2 间隔重复(§5.4)---
SM2_DEFAULT_EF = 2.5
SM2_MIN_EF = 1.3
SM2_PASS_SCORE = 3
# ★ 文档没规定 correct/incorrect 怎么映射到 0~5 的评分,这里定一个:
#   答对 = 4(中性),因为二值信号撑不起"完美回忆";答错 = 1(低于及格线)。
#   q=4 时 EF 增量为 0.1 − 1×(0.08+0.02) = 0,难度因子不漂移。
SM2_QUALITY_CORRECT = 4
SM2_QUALITY_WRONG = 1

DAY_SECONDS = 86400.0

# --- 每日计划 ---
PLAN_REVIEW_MINUTES = 5.0      # 复习一项的基础耗时
PLAN_LEARN_MINUTES_PER_DIFFICULTY = 5.0   # 新学一项:难度 × 该值
PLAN_REMEDIAL_MULTIPLIER = 1.2  # 补根因比新学更费时

# --- Redis / 事件 ---
REDIS_URL = "redis://localhost:6379/0"
STREAM_ANSWER = "coach:answer"
STREAM_PROFILE = "coach:profile"
STREAM_TICK = "coach:tick"
STREAM_PLAN = "coach:plan"
GROUP_COACH = "coach-workers"
MAX_ATTEMPTS = 3
