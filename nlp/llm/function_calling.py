# -*- coding: utf-8 -*-
"""
FunctionCalling 兼容层
=====================

向后兼容旧的 FunctionCalling 接口，内部使用 LangChain Tool

**本文件在 nlp/ 内已无任何调用方**（全仓库只有已归档的 zanshibuyong/ 引用过它），
是一层历史兼容层、现状等同死代码 —— 读 nlp 主流程时不必花时间研究它。
它为什么被淘汰，见下面 register_function 里的说明：兼容签名把 parameters 丢了。

Function Calling（函数调用）= 模型不直接回答，而是吐出"要调哪个函数、参数是什么"，
由调用方去执行 —— 这是 Agent 能"动手办事"的底层机制。
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

        注意"会被忽略"这四个字的分量：本方法没有把 parameters 转交给 register_tool，
        于是下游算不出 args_schema、只能退化成 LangChain 的 Tool（见
        langchain_tools.register_tool 的 ★ 段），结果是**凡是参数多过 1 个的工具，
        经这里注册都会在调用时报 Too many arguments**。
        兼容签名容不下 LangChain 需要的 schema —— 这就是这层接口被弃用的原因。
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
