import json
import pytest
from unittest.mock import MagicMock

from llm.function_calling import FunctionCalling


class TestFunctionCalling:
    @pytest.fixture
    def fc(self):
        return FunctionCalling()

    def test_register_function(self, fc):
        fc.register_function(
            name="get_weather",
            description="获取天气",
            parameters={"type": "object", "properties": {}, "required": []},
            function=lambda: "sunny",
        )
        assert "get_weather" in fc.functions

    def test_get_tools_schema(self, fc):
        fc.register_function("f1", "desc", {"type": "object"}, lambda: None)
        fc.register_function("f2", "desc2", {"type": "object"}, lambda: None)
        schema = fc.get_tools_schema()
        assert len(schema) == 2
        assert schema[0]["type"] == "function"
        assert schema[0]["function"]["name"] == "f1"

    def test_execute_function_success(self, fc):
        fc.register_function(
            "add",
            "add two numbers",
            {
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
            function=lambda a, b: a + b,
        )
        result = fc.execute_function("add", {"a": 1, "b": 2})
        assert result == 3

    def test_execute_nonexistent_function(self, fc):
        with pytest.raises(ValueError, match="函数不存在"):
            fc.execute_function("missing", {})

    def test_missing_required_param(self, fc):
        fc.register_function(
            "greet",
            "greet",
            {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            function=lambda name: f"Hello {name}",
        )
        with pytest.raises(ValueError, match="缺少必需参数"):
            fc.execute_function("greet", {})

    def test_type_mismatch(self, fc):
        fc.register_function(
            "square",
            "square",
            {
                "type": "object",
                "properties": {"n": {"type": "integer"}},
                "required": ["n"],
            },
            function=lambda n: n * n,
        )
        with pytest.raises(TypeError, match="类型错误"):
            fc.execute_function("square", {"n": "not a number"})

    def test_unknown_param_warning(self, fc):
        fc.register_function(
            "f",
            "f",
            {"type": "object", "properties": {}, "required": []},
            function=lambda **kwargs: "ok",
        )
        result = fc.execute_function("f", {"extra": "value"})
        assert result == "ok"

    def test_create_function_schema(self):
        schema = FunctionCalling.create_function_schema(
            "my_func", "does something", {"type": "object"}
        )
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "my_func"

    def test_parse_tool_calls_valid(self):
        data = json.dumps({"type": "tool_calls", "tool_calls": [{"name": "test"}]})
        result = FunctionCalling.parse_tool_calls(data)
        assert result == [{"name": "test"}]

    def test_parse_tool_calls_not_tool(self):
        result = FunctionCalling.parse_tool_calls("普通文本")
        assert result is None

    def test_format_tool_result(self):
        result = FunctionCalling.format_tool_result("my_tool", {"output": 42})
        parsed = json.loads(result)
        assert parsed["tool"] == "my_tool"
        assert parsed["result"] == {"output": 42}
