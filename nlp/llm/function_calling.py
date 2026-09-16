# -*- coding: utf-8 -*-
"""
FunctionCalling 兼容层
=====================

向后兼容旧的 FunctionCalling 接口，内部使用 LangChain Tool
"""

import logging
from typing import Dict, Any, Callable
from llm.langchain_tools import ToolRegistry

logger = logging.getLogger(__name__)


class FunctionCalling(ToolRegistry):
    """向后兼容的 FunctionCalling 类"""

    def register_function(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        function: Callable
    ):
        """
        兼容旧的注册方式

        Args:
            name: 函数名
            description: 函数描述
            parameters: 参数schema（OpenAI格式，会被忽略）
            function: 实际执行的函数
        """
        logger.debug(f"[兼容层] 注册函数: {name}")
        return self.register_tool(
            name=name,
            func=function,
            description=description,
            handle_tool_error=True
        )

    def call_function(self, name: str, input: Any) -> str:
        """
        兼容旧的调用方式

        Args:
            name: 函数名
            input: 函数输入

        Returns:
            函数输出
        """
        return self.call_tool(name, input)
