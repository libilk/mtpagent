# -*- coding: utf-8 -*-
"""
LangChain OutputParser 封装
===========================

使用 LangChain 的 OutputParser 替代手动 JSON 解析
"""

import logging
from typing import Any, TypeVar, Type
from pydantic import BaseModel, ValidationError
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


def parse_llm_output(
    response: str,
    pydantic_model: Type[T],
    fallback: Any = None
) -> T | Any:
    """
    使用 LangChain 解析 LLM 输出

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
