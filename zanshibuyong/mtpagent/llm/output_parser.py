import json
import logging
from typing import Any, Optional, Type, TypeVar
from pydantic import BaseModel, ValidationError
logger = logging.getLogger(__name__)
T = TypeVar('T', bound=BaseModel)
def parse_llm_json(response: str, fallback: Any = None) -> Any:
    """从 LLM 响应中提取并解析 JSON
        Args:
            response: LLM 的原始响应文本
            fallback: 解析失败时返回的默认值
        Returns:
            解析后的 Python 对象（dict/list），失败时返回 fallback
        """
    if not response or not isinstance(response, str):
        logger.warning(f"Invalid response type: {type(response)}")
        return fallback
    text = response.strip()
    if text.startswith("```"):
        # 找到第一个换行符后的内容
        lines = text.split('\n', 1)
        if len(lines) > 1:
            text = lines[1]
        # 去除结尾的 ```
        if text.endswith("```"):
            text = text[:-3].strip()
        json_str = _extract_balanced_json(text)
        if not json_str:
            logger.warning(f"No valid JSON structure found in response: {text[:100]}...")
            return fallback
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error: {e}. Content: {json_str[:200]}...")
            return fallback

def parse_llm_json_as(
    response: str,
    model_class: Type[T],
    fallback: Optional[T] = None
) -> Optional[T]:
    """从 LLM 响应中提取 JSON 并验证为 Pydantic 模型
    Args:
        response: LLM 的原始响应文本
        model_class: Pydantic 模型类
        fallback: 解析/验证失败时返回的默认值
    Returns:
        验证后的 Pydantic 模型实例，失败时返回 fallback
    """
    data = parse_llm_json(response, fallback=None)
    if data is None:
        return fallback
    try:
        return model_class.model_validate(data)
    except ValidationError as e:
        logger.warning(f"Pydantic validation error for {model_class.__name__}: {e}")
        return fallback

def _extract_balanced_json(text: str) -> Optional[str]:
    """使用平衡括号算法提取 JSON 字符串
    支持提取 {...} 或 [...] 格式的 JSON
    """
    # 尝试提取对象 {...}
    json_str = _extract_balanced_brackets(text, '{', '}')
    if json_str:
        return json_str
    # 尝试提取数组 [...]
    json_str = _extract_balanced_brackets(text, '[', ']')
    return json_str

def _extract_balanced_brackets(text: str, open_char: str, close_char: str) -> Optional[str]:
    """提取平衡的括号内容"""
    start_idx = text.find(open_char)
    if start_idx == -1:
        return None
    depth = 0
    in_string = False
    escape_next = False
    for i in range(start_idx, len(text)):
        char = text[i]
        # 处理字符串内的引号
        if char == '"' and not escape_next:
            in_string = not in_string
        # 处理转义字符
        if char == '\\' and not escape_next:
            escape_next = True
            continue
        else:
            escape_next = False
        # 只在字符串外计数括号
        if not in_string:
            if char == open_char:
                depth += 1
            elif char == close_char:
                depth -= 1
                if depth == 0:
                    return text[start_idx:i+1]
    return None

class RiskAnalysis(BaseModel):
    """风险分析结果"""
    risks: list[dict[str, str]]  # [{"type": "...", "description": "...", "severity": "..."}]
    summary: str


class ContractStructure(BaseModel):
    """合同结构提取结果"""
    sections: list[dict[str, Any]]  # [{"title": "...", "content": "...", "subsections": [...]}]
    metadata: dict[str, Any]


class HistoryComparison(BaseModel):
    """历史对比结果"""
    similarities: list[str]
    differences: list[str]
    recommendations: list[str]


class QualityEvaluation(BaseModel):
    """质量评估结果"""
    score: float  # 0.0-1.0
    reasoning: str
    issues: list[str]


class CompletenessCheck(BaseModel):
    """完整性检查结果"""
    is_complete: bool
    missing_aspects: list[str]
    suggestions: list[str]


class SentimentResult(BaseModel):
    """情感分析结果"""
    sentiment: str  # "positive", "negative", "neutral"
    confidence: float
    keywords: list[str]


class TaskPlan(BaseModel):
    """任务规划结果"""
    steps: list[dict[str, Any]]  # [{"step": 1, "action": "...", "agent": "..."}]
    estimated_complexity: str  # "low", "medium", "high"