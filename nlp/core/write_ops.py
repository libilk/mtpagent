# -*- coding: utf-8 -*-
"""
写操作记录器（线程本地）
====================

**解决的问题：** Agent 只返回一个字符串（`handle(query, context) -> str`），
没法在返回值和上下文里携带"我刚才改了数据"这个信号。而编排层需要在
Agent 跑完之后知道这件事 —— 才能触发人工审批。

**为什么用线程本地（threading.local）而不是挂在 self 上：**

Agent 实例在启动时注册一次、之后**被所有请求共享**。如果写成
`self.last_write_op = ...`，两个并发请求会互相覆盖，A 用户提交的退货申请
可能触发 B 用户的审批弹窗。

而 `handle()` 是在节点函数里**同步**调用的，Agent 和节点处在同一个线程中。
线程本地存储正好保证：每个请求各写入自己的副本，互不干扰。

**为什么登记必须发生在「调用线程」，不能写在工具函数内部（项目真踩过的坑）：**

LLM 一轮里同时调多个工具时，这些工具会被线程池（ThreadPoolExecutor）并发执行，
跑在 worker 线程上。若把 `record_write_op` 写在工具实现（如 CRM）内部，记录就落在
**worker 线程**的副本上 —— 节点线程读不到，信号直接丢失。更糟的是线程池的线程会被复用，
残留记录可能被下一个恰好调度到该线程的请求读走，**跨用户串号**，比丢信号更难查。

所以登记被上移到 Agent 的工具分发层（`customer_service_agent._record_write_operations`，
在 `_execute_tool_calls` 里所有工具执行完之后），那里已经回到调用线程。
通用教训：用线程本地 / contextvar 之前，先确认代码有没有开线程池 ——
「同一次逻辑调用」不等于「同一个线程」。

**用法：**

    # 写入侧（Agent 在做真正改数据的操作时）
    # 事实说明：上面这行示例原本写作"Agent / CRM"，但按上文那条坑，
    # 登记点必须落在 Agent 的工具分发层，**不能**下沉到 CRM 工具内部。
    record_write_op("submit_return_request", {"order_id": "SO20260909001", ...})

    # 读取侧（编排层的节点，Agent 跑完之后）
    ops = consume_write_ops()      # 读走并清空

注意 `consume_write_ops()` 是**读走并清空**的语义 —— 不清空的话，
下一次请求会读到上一次残留的记录，误触发审批。
"""

import threading
from typing import Any, Dict, List

# 线程本地存储（threading.local）：每个线程一份副本，互不可见。
# 副本按需创建 —— 线程首次访问 _local.ops 前它并不存在，所以两个函数都用 getattr 取而不是直接读。
_local = threading.local()


def record_write_op(op_type: str, detail: Dict[str, Any] = None) -> None:
    """
    记录一次写操作。

    Args:
        op_type: 操作类型，如 "submit_return_request" / "create_ticket"
        detail:  操作细节，会展示给审批人看（如订单号、退款金额）
    """
    ops = getattr(_local, "ops", None)
    if ops is None:
        ops = []
        _local.ops = ops

    ops.append({"type": op_type, "detail": detail or {}})


def consume_write_ops() -> List[Dict[str, Any]]:
    """
    读走本次调用累积的写操作记录，并清空。

    Returns:
        写操作列表；没有写操作时返回空列表。
    """
    ops = getattr(_local, "ops", None)
    if not ops:
        return []

    _local.ops = []          # 清空，避免残留影响下一次请求
    return ops


def clear_write_ops() -> None:
    """只清空不读取。异常路径上兜底用，防止脏数据影响后续请求。

    实际调用点只有一个：nodes.py 里 Agent 抛异常的 except 分支。
    那条路径上正常读取被跳过了，不补清一次，残留记录会误触下一个请求的审批闸门。
    """
    _local.ops = []
