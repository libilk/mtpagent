# -*- coding: utf-8 -*-
"""
Agent容错处理器
===============

处理Agent调用中的各种错误
"""

import json
import logging
import re
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class AgentErrorHandler:
    """Agent错误处理器"""

    @staticmethod
    def parse_json_safe(text: str) -> Optional[Dict]:
        """
        安全解析JSON（容错）

        策略：
        1. 直接解析
        2. 提取JSON块（```json...```）
        3. 正则提取{}内容
        4. 修复常见错误（尾部逗号、单引号等）

        Args:
            text: 待解析文本

        Returns:
            解析结果或None
        """
        if not text:
            return None

        # 策略1：直接解析
        try:
            return json.loads(text)
        except:
            pass

        # 策略2：提取JSON代码块
        json_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        match = re.search(json_block_pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                pass

        # 策略3：提取{}内容
        brace_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(brace_pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except:
                pass

        # 策略4：修复常见错误
        try:
            # 移除尾部逗号
            fixed = re.sub(r',(\s*[}\]])', r'\1', text)
            # 单引号转双引号
            fixed = fixed.replace("'", '"')
            return json.loads(fixed)
        except:
            pass

        logger.warning(f"JSON解析失败: {text[:100]}...")
        return None

    @staticmethod
    def validate_agent_response(response: Any, expected_keys: list = None) -> Dict:
        """
        验证Agent响应格式

        Args:
            response: Agent响应
            expected_keys: 期望的键列表

        Returns:
            标准化响应
        """
        # 如果是字符串，尝试解析
        if isinstance(response, str):
            parsed = AgentErrorHandler.parse_json_safe(response)
            if parsed:
                response = parsed
            else:
                # 非 JSON 文本视为有效回答
                return {
                    "success": True,
                    "answer": response
                }

        # 如果不是字典，包装
        if not isinstance(response, dict):
            return {
                "success": False,
                "error": "响应类型错误",
                "raw_response": str(response)
            }

        # 验证必需键
        if expected_keys:
            missing = [k for k in expected_keys if k not in response]
            if missing:
                logger.warning(f"响应缺少键: {missing}")
                response["_missing_keys"] = missing

        return response

    @staticmethod
    def handle_agent_error(error: Exception, agent_id: str, query: str) -> Dict:
        """
        处理Agent执行错误

        Args:
            error: 异常对象
            agent_id: Agent ID
            query: 用户查询

        Returns:
            错误响应
        """
        error_type = type(error).__name__
        error_msg = str(error)

        logger.error(f"Agent {agent_id} 执行失败: {error_type} - {error_msg}")

        return {
            "success": False,
            "error": error_msg,
            "error_type": error_type,
            "agent_id": agent_id,
            "query": query,
            "fallback_available": True
        }
