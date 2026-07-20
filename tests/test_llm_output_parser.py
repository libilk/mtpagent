import pytest

from llm.output_parser import (
    parse_llm_json,
    parse_llm_json_as,
    QualityEvaluation,
    RiskAnalysis,
    ContractStructure,
    HistoryComparison,
    CompletenessCheck,
    SentimentResult,
    TaskPlan,
)


class TestParseLlmJson:
    def test_json_in_markdown_block(self):
        """纯 JSON 对象必须包裹在 ```...``` 中才能解析"""
        result = parse_llm_json('```\n{"a": 1}\n```')
        assert result == {"a": 1}

    def test_markdown_json_code_block(self):
        result = parse_llm_json('```json\n{"a": 1}\n```')
        assert result == {"a": 1}

    def test_json_array_in_markdown_block(self):
        result = parse_llm_json('```\n[1, 2, 3]\n```')
        assert result == [1, 2, 3]

    def test_nested_json_in_markdown_block(self):
        result = parse_llm_json('```\n{"outer": {"inner": 1}}\n```')
        assert result == {"outer": {"inner": 1}}

    def test_unicode_json_in_markdown_block(self):
        result = parse_llm_json('```\n{"message": "你好世界"}\n```')
        assert result == {"message": "你好世界"}

    def test_empty_string_returns_fallback(self):
        result = parse_llm_json("", fallback="default")
        assert result == "default"

    def test_none_returns_fallback(self):
        result = parse_llm_json(None, fallback="default")
        assert result == "default"

    def test_plain_text_without_markdown(self):
        """不带 ``` 标记的文本直接返回 None（当前实现限制）"""
        result = parse_llm_json('{"a": 1}')
        assert result is None

    def test_plain_text_no_json(self):
        result = parse_llm_json("```\n普通文本\n```", fallback={"error": True})
        assert result == {"error": True}


class TestParseLlmJsonAs:
    def test_valid_quality_evaluation(self):
        data = '```\n{"score": 0.85, "reasoning": "good", "issues": ["minor"]}\n```'
        result = parse_llm_json_as(data, QualityEvaluation)
        assert isinstance(result, QualityEvaluation)
        assert result.score == 0.85
        assert result.reasoning == "good"

    def test_invalid_schema_returns_fallback(self):
        data = '```\n{"wrong_field": "value"}\n```'
        result = parse_llm_json_as(data, QualityEvaluation, fallback=None)
        assert result is None

    def test_invalid_json_returns_fallback(self):
        result = parse_llm_json_as("not json at all", QualityEvaluation, fallback=None)
        assert result is None


class TestPydanticModels:
    def test_risk_analysis_valid(self):
        data = {
            "risks": [{"type": "financial", "description": "成本超支", "severity": "high"}],
            "summary": "ok",
        }
        m = RiskAnalysis.model_validate(data)
        assert m.summary == "ok"

    def test_risk_analysis_invalid(self):
        with pytest.raises(Exception):
            RiskAnalysis.model_validate({"wrong": "field"})

    def test_contract_structure_valid(self):
        data = {
            "sections": [{"title": "第1条", "content": "...", "subsections": []}],
            "metadata": {"parties": "甲乙"},
        }
        m = ContractStructure.model_validate(data)
        assert len(m.sections) == 1

    def test_history_comparison_valid(self):
        data = {
            "similarities": ["价格条款"],
            "differences": ["交付时间"],
            "recommendations": ["重新谈判"],
        }
        m = HistoryComparison.model_validate(data)
        assert "价格条款" in m.similarities

    def test_quality_evaluation_valid(self):
        data = {"score": 0.9, "reasoning": "结构完整", "issues": []}
        m = QualityEvaluation.model_validate(data)
        assert m.score == 0.9

    def test_completeness_check_valid(self):
        data = {
            "is_complete": False,
            "missing_aspects": ["验收标准"],
            "suggestions": ["补充验收条款"],
        }
        m = CompletenessCheck.model_validate(data)
        assert m.is_complete is False

    def test_sentiment_result_valid(self):
        data = {"sentiment": "negative", "confidence": 0.88, "keywords": ["违约", "罚款"]}
        m = SentimentResult.model_validate(data)
        assert m.sentiment == "negative"

    def test_task_plan_valid(self):
        data = {
            "steps": [{"step": 1, "action": "分析合同", "agent": "contract_agent"}],
            "estimated_complexity": "medium",
        }
        m = TaskPlan.model_validate(data)
        assert m.estimated_complexity == "medium"
