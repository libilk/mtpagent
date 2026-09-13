"""
从零构建 MCP Server —— 纯手写 stdio transport + JSON-RPC

不依赖 mcp.server.Server，自己处理每一层协议。

MCP 协议分层（从底到顶）：
  第 0 层：stdio 传输 —— 从 stdin 读字节，往 stdout 写字节
  第 1 层：MCP 帧格式 —— Content-Length: N\r\n\r\n{json}
  第 2 层：JSON-RPC 2.0 —— {"jsonrpc":"2.0","method":"...","id":1}
  第 3 层：业务逻辑 —— list_tools / call_tool

运行：
  python learn/mcp_server_raw.py

然后用 mcp_agent.py 连接（需要改一下它连的服务端路径）：
  python learn/mcp_agent.py
"""

import sys
import json
import asyncio
import logging
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="[RAW-MCP] %(message)s",
    stream=sys.stderr,  # stderr 不会被 MCP 协议污染，只有 stdout 用于通信
)
logger = logging.getLogger("raw-mcp")

# ============================================================
# CRM 模拟数据
# ============================================================
_users = {
    "user_001": {"name": "张三", "vip": "gold", "orders": 25},
    "user_002": {"name": "李四", "vip": "silver", "orders": 10},
}
_tickets: dict = {}
_ticket_counter = 0


# ============================================================
# 第 0 层：stdio 传输
# ============================================================
# MCP 走 stdin/stdout。
#   - 服务端从 stdin 读请求
#   - 服务端往 stdout 写响应
#   - 日志打到 stderr（不污染通信通道）
#
# sys.stdin.buffer.read() 是同步阻塞的，但我们用 asyncio 的事件循环来读。

async def read_stdin() -> bytes:
    """从 stdin 读一次（异步）"""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, sys.stdin.buffer.read1, 65536)


def write_stdout(data: bytes):
    """往 stdout 写（同步）"""
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()


# ============================================================
# 第 1 层：MCP 帧格式
# ============================================================
# MCP 不使用简单的 "\n" 分隔。
# 格式：Content-Length: <字节数>\r\n\r\n<JSON正文>
#
# 例子：
#   Content-Length: 87\r\n
#   \r\n
#   {"jsonrpc":"2.0","method":"tools/list","id":1}
#
# 为什么：JSON 里可能有 \n，用长度前缀更可靠。

def pack_message(obj: dict) -> bytes:
    """把 dict 打包成 MCP 帧"""
    body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8")
    return header + body


class FrameReader:
    """
    从字节流中解析 MCP 帧。

    因为 TCP/stdio 是流式的，一次 read 可能拿到半帧或多帧。
    需要维护一个缓冲区，凑齐一帧就解析一帧。
    """

    def __init__(self):
        self._buffer = b""

    def feed(self, data: bytes) -> list[dict]:
        """喂入新字节，返回本次解析出的所有完整 JSON 消息"""
        self._buffer += data
        messages = []

        while True:
            # 1. 找 Content-Length 头
            header_end = self._buffer.find(b"\r\n\r\n")
            if header_end == -1:
                break  # 还没收齐头部

            header = self._buffer[:header_end].decode("utf-8")

            # 2. 解析 Content-Length 值
            content_length = 0
            for line in header.split("\r\n"):
                if line.lower().startswith("content-length:"):
                    content_length = int(line.split(":")[1].strip())
                    break

            if content_length == 0:
                logger.error("未找到 Content-Length")
                self._buffer = b""
                break

            body_start = header_end + 4  # 跳过 \r\n\r\n
            body_end = body_start + content_length

            # 3. 检查是否收齐了整个 body
            if len(self._buffer) < body_end:
                break  # body 还没收全

            body = self._buffer[body_start:body_end]

            # 4. 解析 JSON
            try:
                msg = json.loads(body.decode("utf-8"))
                messages.append(msg)
            except json.JSONDecodeError as e:
                logger.error(f"JSON 解析失败: {e}")

            # 5. 从缓冲区移除已解析的部分
            self._buffer = self._buffer[body_end:]

        return messages


# ============================================================
# 第 2 层：JSON-RPC 2.0
# ============================================================
# MCP 用的就是标准 JSON-RPC 2.0：
#   请求:  {"jsonrpc":"2.0", "method":"tools/list", "id":1}
#   通知:  {"jsonrpc":"2.0", "method":"notifications/..."}  ← 没有 id，不需要响应
#   响应:  {"jsonrpc":"2.0", "result":{...}, "id":1}
#   错误:  {"jsonrpc":"2.0", "error":{"code":-32600, "message":"..."}, "id":1}

def make_response(request_id: Any, result: Any) -> dict:
    """构造 JSON-RPC 成功响应"""
    return {"jsonrpc": "2.0", "result": result, "id": request_id}


def make_error(request_id: Any, code: int, message: str) -> dict:
    """构造 JSON-RPC 错误响应"""
    return {"jsonrpc": "2.0", "error": {"code": code, "message": message}, "id": request_id}


# ============================================================
# JSON-RPC 标准错误码
# ============================================================
PARSE_ERROR = -32700
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


# ============================================================
# 第 3 层：业务逻辑
# ============================================================

# MCP 协议规定的 server info
SERVER_INFO = {
    "protocolVersion": "2024-11-05",
    "capabilities": {
        "tools": {},  # 声明：我支持 tools
    },
    "serverInfo": {
        "name": "raw-crm-server",
        "version": "1.0.0",
    },
}

# 工具定义（和 mcp_server.py 一样，只是这里手写 dict）
TOOLS = [
    {
        "name": "get_user_info",
        "description": "查询用户信息，根据用户ID获取姓名、VIP等级、订单数",
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "用户ID，如 user_001"},
            },
            "required": ["user_id"],
        },
    },
    {
        "name": "create_ticket",
        "description": "创建客服工单",
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "用户ID"},
                "issue": {"type": "string", "description": "问题简述"},
                "priority": {
                    "type": "string",
                    "enum": ["low", "normal", "high", "urgent"],
                    "description": "优先级",
                },
            },
            "required": ["user_id", "issue"],
        },
    },
]


def handle_initialize(params: dict) -> dict:
    """处理 initialize 请求 —— MCP 握手"""
    logger.info(f"客户端初始化: {params.get('clientInfo', {})}")
    return SERVER_INFO


def handle_tools_list(_params: dict) -> dict:
    """列出所有工具"""
    return {"tools": TOOLS}


def handle_tools_call(params: dict) -> dict:
    """执行工具调用"""
    name = params["name"]
    args = params.get("arguments", {})
    logger.info(f"调用工具: {name}({args})")

    try:
        if name == "get_user_info":
            user = _users.get(args["user_id"])
            text = json.dumps(user, ensure_ascii=False) if user else f"未找到用户 {args['user_id']}"

        elif name == "create_ticket":
            global _ticket_counter
            _ticket_counter += 1
            tid = f"T{_ticket_counter:05d}"
            _tickets[tid] = {
                "ticket_id": tid,
                "user_id": args["user_id"],
                "issue": args["issue"],
                "priority": args.get("priority", "normal"),
            }
            text = f"工单已创建: {tid} - {args['issue']}"

        else:
            text = f"未知工具: {name}"

        # MCP call_tool 返回格式：content 是列表，每项有 type + text
        return {"content": [{"type": "text", "text": text}]}

    except Exception as e:
        logger.error(f"工具执行失败: {e}")
        return {"content": [{"type": "text", "text": f"错误: {e}"}], "isError": True}


# ---- 路由表 ----
METHOD_HANDLERS = {
    "initialize": handle_initialize,
    "tools/list": handle_tools_list,
    "tools/call": handle_tools_call,
}

# 初始化完成前唯一允许的方法
ALLOWED_BEFORE_INIT = {"initialize"}


# ============================================================
# 主循环
# ============================================================

async def main():
    """
    主循环：读 stdin → 解析帧 → 处理 JSON-RPC → 写 stdout

    这就是 mcp.server.Server 内部做的事，现在全展开在你面前。
    """
    logger.info("=== 纯手写 MCP Server 启动 (stdio) ===")
    logger.info(f"PID: {os.getpid()}")
    logger.info("等待客户端连接...")

    import os
    reader = FrameReader()
    initialized = False

    while True:
        # 1. 从 stdin 读字节
        chunk = await read_stdin()
        if not chunk:
            logger.info("stdin 关闭，退出")
            break

        # 2. 解析帧 → JSON 消息列表
        messages = reader.feed(chunk)

        for msg in messages:
            logger.info(f"收到: {msg.get('method', '响应')}")

            # 3. 判断是请求还是通知
            method = msg.get("method")
            request_id = msg.get("id")

            if not method:
                continue  # 不是请求，跳过

            # 3a. 初始化前的安全检查
            if not initialized and method not in ALLOWED_BEFORE_INIT:
                response = make_error(request_id, -32002, "未初始化")
                write_stdout(pack_message(response))
                continue

            # 3b. 路由到 handler
            handler = METHOD_HANDLERS.get(method)
            if handler is None:
                response = make_error(request_id, METHOD_NOT_FOUND, f"未知方法: {method}")
                write_stdout(pack_message(response))
                continue

            # 3c. 执行
            try:
                params = msg.get("params", {})
                result = handler(params)
                if method == "initialize":
                    initialized = True
                response = make_response(request_id, result)
            except Exception as e:
                logger.error(f"处理 {method} 失败: {e}")
                response = make_error(request_id, INTERNAL_ERROR, str(e))

            # 4. 写响应到 stdout
            write_stdout(pack_message(response))
            logger.info(f"已响应: {method}")


if __name__ == "__main__":
    asyncio.run(main())
