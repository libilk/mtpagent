# -*- coding: utf-8 -*-
"""
LangSmith（链路追踪服务）开关 —— 显式启用，默认关闭
==================================================

【它解决什么问题】
项目原本的链路追踪是自建的：图里每个节点用 get_stream_writer() 推一条进度事件，
经 SSE 到前端 trace 面板（见 langgraph_orchestrator/nodes.py 的 _emit）。
那套是「节点 + 阶段 + 消息」的**扁平事件流** —— 看得到"走到了哪一步"，
看不到"这一步花了多久""这一步的 prompt 和响应长什么样"。

LangSmith 给的是**带耗时归因的 span 树**：一次请求 → 每个节点 → 每次 LLM 调用
→ 每次工具调用，层层嵌套，每层都带耗时和输入输出。

【为什么单独写一个模块，而不是在 .env 里塞几个变量就算了】
因为 LangSmith 会把**完整 prompt、用户原话、工具返回结果**全量上传到云端。
而本项目有一条数据红线 ——「不长期留存投诉原文」（见 study.md §7.1，在代码里的
落地是 tools/scripts/extract_memory_events.py 的 SOURCE_QUOTE_MAX 截断）。
两者直接冲突。所以这里做成**显式开关 + 启动时大声提示**，
而不是"设了 key 就悄悄开始上传"。

【怎么用】
在 nlp/.env 里加三行：
    LANGSMITH_TRACING=true
    LANGSMITH_API_KEY=lsv2_pt_xxxxxxxx
    LANGSMITH_PROJECT=mtpagent-nlp          # 可选，默认 mtpagent-nlp
然后照常启动（python api.py 或 python main.py），去 https://smith.langchain.com 看。

不想上传 prompt / 响应内容、只要耗时结构的话，再加两行：
    LANGSMITH_HIDE_INPUTS=true
    LANGSMITH_HIDE_OUTPUTS=true

【★ 一个必须知道的时序约束】
langsmith 读环境变量用了 @lru_cache（见 .venv/…/langsmith/utils.py 的 get_env_var），
**第一次读之后就锁死了**。所以 enable_langsmith_tracing() 必须在任何 langchain /
langgraph 模块被 import 之前调用 —— 两个入口都是在 load_dotenv() 之后立刻调它，
**位置不能往后挪**。
"""

import os
import logging

logger = logging.getLogger(__name__)

# 默认项目名。LangSmith 网页按这个名字分组，多环境（开发/测试）可以各自覆盖。
DEFAULT_PROJECT = "mtpagent-nlp"

# 开关与 Key 都认两套命名：LANGSMITH_* 是新命名，LANGCHAIN_* 是旧命名。
# langsmith 内部两个命名空间都会查，这里显式列出来是为了让读代码的人知道都支持。
_ENABLE_VARS = ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING")
_KEY_VARS = ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY")


def _first_set(names) -> str:
    """按顺序返回第一个有值的环境变量；都没有就返回空串"""
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def enable_langsmith_tracing() -> bool:
    """按 .env 的配置启用 LangSmith 追踪；没配就安静地什么都不做。

    这里刻意不 import langchain / langgraph —— 本函数要在它们之前跑，
    自己不能反过来把它们拉起来。

    Returns:
        True = 已启用（并已把状态与隐私提示打进日志）；False = 未启用。
    """
    if _first_set(_ENABLE_VARS).lower() != "true":
        # 默认路径不打日志：没配就是没配，不该在每次启动时刷一行噪音。
        return False

    api_key = _first_set(_KEY_VARS)
    if not api_key:
        logger.warning(
            "[LangSmith] LANGSMITH_TRACING=true，但没找到 API Key，追踪未启用。\n"
            "  请在 .env 补一行：LANGSMITH_API_KEY=lsv2_pt_...\n"
            "  Key 获取地址：https://smith.langchain.com（Settings → API Keys）"
        )
        return False

    # 补齐 langsmith 需要的变量。用 setdefault：用户显式设过的值优先，不覆盖。
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_API_KEY", api_key)
    os.environ.setdefault("LANGSMITH_PROJECT", DEFAULT_PROJECT)

    # 隐私提示：把"会上传什么"和"怎么关掉"一次说清，别让人事后才发现。
    hidden = []
    if os.environ.get("LANGSMITH_HIDE_INPUTS", "").lower() == "true":
        hidden.append("输入")
    if os.environ.get("LANGSMITH_HIDE_OUTPUTS", "").lower() == "true":
        hidden.append("输出")

    # 日志里只用 BMP 字符（如 ✓），不用 emoji —— Windows 控制台默认 GBK，
    # 非 BMP 字符会被转义成 \uXXXX 字样，反而看不清。
    logger.warning(
        "[LangSmith] 追踪已启用 -> 项目「%s」\n"
        "  【注意】每次请求的完整 prompt、用户原话、工具返回结果都会上传到 LangSmith 云端。\n"
        "          本项目有「不长期留存投诉原文」的数据红线（study.md §7.1）——\n"
        "          生产环境请先脱敏，或加 LANGSMITH_HIDE_INPUTS=true / LANGSMITH_HIDE_OUTPUTS=true\n"
        "          只保留耗时结构。当前已隐藏的内容：%s",
        os.environ["LANGSMITH_PROJECT"],
        "、".join(hidden) if hidden else "无（内容全量上传）",
    )
    return True
