# -*- coding: utf-8 -*-
"""
LangChain Tool 封装
==================

使用 LangChain 的 StructuredTool + Pydantic Schema 实现工具注册与自动参数校验
"""

import logging
from typing import List, Dict, Any, Callable, Union
from langchain_core.tools import Tool, StructuredTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ToolRegistry:
    """工具注册器（基于 LangChain StructuredTool + Pydantic 自动校验）"""

    def __init__(self):
        self.tools: List[Union[Tool, StructuredTool]] = []

    def register_structured_tool(
        self,
        name: str,
        func: Callable,
        description: str,
        args_schema: type[BaseModel]
    ) -> StructuredTool:
        """
        注册结构化工具（带 Pydantic 自动参数校验）

        Args:
            name: 工具名称
            func: 工具函数
            description: 工具描述
            args_schema: 参数 schema（Pydantic 模型）

        Returns:
            StructuredTool 实例
        """
        tool = StructuredTool.from_function(
            func=func,
            name=name,
            description=description,
            args_schema=args_schema,
            handle_tool_error=True
        )
        self.tools.append(tool)
        logger.info(f"注册工具: {name}")
        return tool

    def register_tool(
        self,
        name: str,
        func: Callable = None,
        description: str = "",
        handle_tool_error: bool = True,
        parameters: dict = None,
        function: Callable = None
    ) -> Tool:
        """
        注册普通工具（兼容其他 Agent 使用）

        推荐使用 register_structured_tool 以获得 Pydantic 自动参数校验。
        """
        actual_func = func or function
        if actual_func is None:
            raise ValueError(f"注册工具 '{name}' 失败：必须提供 func 或 function 参数")

        tool = Tool(
            name=name,
            func=actual_func,
            description=description,
            handle_tool_error=handle_tool_error
        )

        if parameters:
            tool._raw_parameters = parameters

        self.tools.append(tool)
        logger.info(f"注册工具: {name}")
        return tool

    def get_tools(self) -> List[Union[Tool, StructuredTool]]:
        """获取所有工具"""
        return self.tools

    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """
        获取工具 schema（OpenAI Function Calling 格式）

        StructuredTool 通过 Pydantic args_schema 自动生成参数描述。
        普通 Tool 使用注册时提供的 parameters dict。
        """
        schemas = []
        for tool in self.tools:
            schema = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                }
            }

            # 优先使用原始 parameters dict（普通 Tool）
            if hasattr(tool, '_raw_parameters') and tool._raw_parameters:
                schema["function"]["parameters"] = tool._raw_parameters
            # 其次使用 Pydantic args_schema（StructuredTool）
            elif hasattr(tool, 'args_schema') and tool.args_schema:
                # Pydantic V2 使用 model_json_schema()，V1 兜底用 schema()
                if hasattr(tool.args_schema, 'model_json_schema'):
                    schema["function"]["parameters"] = tool.args_schema.model_json_schema()
                else:
                    schema["function"]["parameters"] = tool.args_schema.schema()

            schemas.append(schema)

        return schemas

    def has_tool(self, tool_name: str) -> bool:
        """检查工具是否存在"""
        return any(tool.name == tool_name for tool in self.tools)

    def get_tool_schema(self, tool_name: str) -> Dict[str, Any]:
        """获取指定工具的schema"""
        for tool in self.tools:
            if tool.name == tool_name:
                schema = {"parameters": {}}
                if hasattr(tool, '_raw_parameters') and tool._raw_parameters:
                    schema["parameters"] = tool._raw_parameters
                elif hasattr(tool, 'args_schema') and tool.args_schema:
                    if hasattr(tool.args_schema, 'model_json_schema'):
                        schema["parameters"] = tool.args_schema.model_json_schema()
                    else:
                        schema["parameters"] = tool.args_schema.schema()
                return schema
        return {}

    def get_all_tool_names(self) -> List[str]:
        """获取所有工具名称"""
        return [tool.name for tool in self.tools]

    def call_tool(self, tool_name: str, tool_input: Any) -> str:
        """
        调用工具（统一接口，Pydantic 自动校验参数）

        StructuredTool.invoke() 内部会先执行 args_schema(**tool_input) 进行校验，
        校验通过后才调用实际函数。
        """
        for tool in self.tools:
            if tool.name == tool_name:
                try:
                    if isinstance(tool_input, dict):
                        # 普通 Tool 不支持 dict 输入，空参数时转为空字符串
                        if isinstance(tool, Tool) and not tool_input:
                            result = tool.run("")
                        else:
                            result = tool.invoke(tool_input)
                    else:
                        result = tool.run(tool_input)
                    return str(result)
                except Exception as e:
                    logger.error(f"工具 {tool_name} 执行失败: {e}")
                    return f"工具执行错误: {str(e)}"

        return f"错误：工具 {tool_name} 不存在"
