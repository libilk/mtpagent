# -*- coding: utf-8 -*-
"""
LangChain OutputParser 封装
===========================

使用 LangChain 的 OutputParser 替代手动 JSON 解析

OutputParser（输出解析器）= 按约定把模型吐出的文本转成结构化对象的组件。
本模块正在被 orchestrator/planner.py（解析成 TaskPlan）和
orchestrator/router.py（解析成 RouteResult）使用，不是死代码。

与隔壁 output_parser.py 的分工：本模块走 Pydantic（数据校验库：用类型标注定义
数据结构、运行时自动校验），格式不符即认输、返回 fallback；
output_parser.py 是手写容错版，尽量从噪声文本里"抠"出 JSON。
"""

import logging
from typing import Any, TypeVar, Type
from pydantic import BaseModel, ValidationError
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException

logger = logging.getLogger(__name__)

# 泛型变量：让"传入哪个 Pydantic 模型类、就返回哪个类型"能被类型检查器看懂。
# bound=BaseModel 限定 T 必须是 Pydantic 模型，传别的类型在静态检查阶段就报错。
T = TypeVar('T', bound=BaseModel)


def parse_llm_output(
    response: str,
    pydantic_model: Type[T],
    fallback: Any = None
) -> T | Any:
    """
    使用 LangChain 解析 LLM 输出

    为什么用 PydanticOutputParser，而不是 json.loads 再手搓对象：
    它同时产出两样东西 —— parse() 负责解析，get_format_instructions() 负责
    生成"要模型按什么格式输出"的说明。两者同源于同一个模型类，prompt 里写的
    格式要求与代码里的解析规则不会各写各的、慢慢漂移。

    Args:
        response: LLM 响应
        pydantic_model: Pydantic 模型类
        fallback: 解析失败时的默认值

    Returns:
        解析后的 Pydantic 对象，失败时返回 fallback
    """
    if not response or not isinstance(response, str):
        logger.warning(f"Invalid response type: {type(response)}")
        return fallback

    parser = PydanticOutputParser(pydantic_object=pydantic_model)

    try:
        result = parser.parse(response)
        return result
    # 事实性说明（未改逻辑）：PydanticOutputParser 内部已经把 pydantic 的
    # ValidationError 包成 OutputParserException 再抛（见其 _parse_obj），
    # 所以实际能落进本分支的只有前者；带上 ValidationError 是跨版本防御。
    except (OutputParserException, ValidationError) as e:
        logger.warning(f"LangChain 解析失败: {e}, 使用 fallback")
        return fallback
    except Exception as e:
        logger.error(f"解析出错: {e}")
        return fallback


def get_format_instructions(pydantic_model: Type[BaseModel]) -> str:
    """
    获取格式说明（用于 Prompt）

    Args:
        pydantic_model: Pydantic 模型类

    Returns:
        格式说明字符串
    """
    parser = PydanticOutputParser(pydantic_object=pydantic_model)
    return parser.get_format_instructions()
