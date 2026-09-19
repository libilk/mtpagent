# -*- coding: utf-8 -*-
"""
OutputParser 兼容层
===================

向后兼容旧的 parse_llm_json 接口

**别被"兼容层"三个字劝退：本模块仍在主力路径上，不是死代码。**
database / knowledge / document / customer_service 四个 Agent 和
langgraph_orchestrator/nodes.py 都在用它解析 LLM 的文本输出。
（真正没有调用方的是隔壁 function_calling.py。）

它和隔壁 langchain_parser.py 的分工：本模块是**手写的容错解析** —— 面对的是
不保证合法 JSON 的裸文本，尽量从噪声里把 JSON "抠"出来；
langchain_parser 走 Pydantic，格式不符就直接放弃、返回 fallback。
"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def parse_llm_json(response: str, fallback: Any = None) -> Any:
    """
    兼容旧的 JSON 解析接口

    Args:
        response: LLM 响应
        fallback: 解析失败时的默认值（fallback＝降级兜底：主路径解析不出来就退到它）

    Returns:
        解析后的字典，失败时返回 fallback
    """
    if not response or not isinstance(response, str):
        logger.warning(f"Invalid response type: {type(response)}")
        return fallback

    # 去除 markdown 代码块
    # 为什么少不了这一步：模型几乎总把 JSON 包进 ```json 围栏里，直接 json.loads 必失败
    text = response.strip()
    if text.startswith("```"):
        lines = text.split('\n', 1)
        if len(lines) > 1:
            text = lines[1]
        if text.endswith("```"):
            text = text[:-3].strip()

    # 提取 JSON
    json_str = _extract_json(text)
    if not json_str:
        logger.warning(f"No valid JSON found in response")
        return fallback

    # 解析 JSON
    try:
        result = json.loads(json_str)
        return result
    except json.JSONDecodeError as e:
        logger.warning(f"JSON decode error: {e}")
        return fallback


def parse_tool_calls(response: str) -> list:
    """
    解析 LLM 返回的工具调用

    认的是 `{"type": "tool_calls", "tool_calls": [...]}` 这个信封格式 ——
    它并非模型原样吐出的，而是 llm_client._chat_with_openai 把 OpenAI 的
    tool_calls 重新 json.dumps 成的（见 llm/llm_client.py）。
    所以改那个信封时要同步改这里，两处是硬耦合。

    Args:
        response: LLM 响应（可能是 JSON 格式的工具调用，或普通文本）

    Returns:
        工具调用列表，无工具调用时返回空列表
    """
    if not response or not isinstance(response, str):
        return []

    try:
        data = json.loads(response)
        if isinstance(data, dict) and data.get("type") == "tool_calls":
            return data.get("tool_calls", [])
    except (json.JSONDecodeError, ValueError):
        pass

    return []


def _extract_json(text: str) -> str:
    """提取 JSON 字符串

    这里用"先找第一个 { / [，再用栈把括号配平"而不是正则：JSON 允许嵌套 {},
    非贪婪正则会停在第一个 } 上，把半个对象当结果返回。
    """
    # 查找第一个 { 或 [
    start = -1
    for i, char in enumerate(text):
        if char in ('{', '['):
            start = i
            break

    if start == -1:
        return ""

    # 使用平衡括号提取
    stack = []
    end = start
    open_char = text[start]
    close_char = '}' if open_char == '{' else ']'

    for i in range(start, len(text)):
        char = text[i]
        if char == open_char:
            stack.append(char)
        elif char == close_char:
            if stack:
                stack.pop()
            if not stack:
                end = i + 1
                break

    if stack:
        return ""

    return text[start:end]
