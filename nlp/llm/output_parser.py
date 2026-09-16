# -*- coding: utf-8 -*-
"""
OutputParser 兼容层
===================

向后兼容旧的 parse_llm_json 接口
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
        fallback: 解析失败时的默认值

    Returns:
        解析后的字典，失败时返回 fallback
    """
    if not response or not isinstance(response, str):
        logger.warning(f"Invalid response type: {type(response)}")
        return fallback

    # 去除 markdown 代码块
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
    """提取 JSON 字符串"""
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
