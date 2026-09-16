import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)
class FunctionCalling:
    """Function Calling管理器"""

    def __init__(self):
        self.functions = {}

    def register_function(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        function: callable
    ):
        """
        注册函数
        Args:
            name: 函数名
            description: 函数描述
            parameters: 参数schema
            function: 实际执行的函数
        """
        self.functions[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "function": function
        }
        logger.info(f"注册函数: {name}")

    def get_tools_schema(self) -> List[Dict]:
        """
        获取工具schema（OpenAI格式）

        Returns:
            工具列表
        """
        tools = []
        for func_info in self.functions.values():
            tools.append({
                "type": "function",
                "function": {
                    "name": func_info["name"],
                    "description": func_info["description"],
                    "parameters": func_info["parameters"]
                }
            })
        return tools

    def execute_function(self, name: str, arguments: Dict[str, Any]) -> Any:
        """
        执行函数

        Args:
            name: 函数名
            arguments: 参数

        Returns:
            执行结果
        """
        if name not in self.functions:
            raise ValueError(f"函数不存在: {name}")

        func_info = self.functions[name]
        function = func_info["function"]

        try:
            # 参数校验
            self._validate_arguments(func_info["parameters"], arguments)

            # 执行函数
            result = function(**arguments)
            logger.info(f"执行函数成功: {name}")
            return result

        except Exception as e:
            logger.error(f"执行函数失败: {name}, 错误: {e}")
            raise

    def _validate_arguments(self, parameters: Dict, arguments: Dict):
        """
        校验参数

        Args:
            parameters: 参数schema
            arguments: 实际参数
        """
        required = parameters.get("required", [])
        properties = parameters.get("properties", {})

        # 检查必需参数
        for param in required:
            if param not in arguments:
                raise ValueError(f"缺少必需参数: {param}")

        # 检查参数类型
        for param, value in arguments.items():
            if param not in properties:
                logger.warning(f"未知参数: {param}")
                continue

            expected_type = properties[param].get("type")
            if expected_type:
                self._check_type(param, value, expected_type)

    def _check_type(self, param: str, value: Any, expected_type: str):
        """检查参数类型"""
        type_mapping = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict
        }

        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type and not isinstance(value, expected_python_type):
            raise TypeError(
                f"参数 {param} 类型错误: 期望 {expected_type}, "
                f"实际 {type(value).__name__}"
            )

    def create_function_schema(
            name: str,
            description: str,
            parameters: Dict[str, Any]
    ) -> Dict:
        """
        创建函数schema

        Args:
            name: 函数名
            description: 函数描述
            parameters: 参数定义

        Returns:
            函数schema
        """
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters
            }
        }

    def parse_tool_calls(response: str) -> Optional[List[Dict]]:
        """
        解析工具调用

        Args:
            response: LLM响应

        Returns:
            工具调用列表，如果不是工具调用则返回None
        """
        try:
            data = json.loads(response)
            if data.get("type") == "tool_calls":
                return data.get("tool_calls", [])
        except json.JSONDecodeError:
            pass
        return None

    def format_tool_result(tool_name: str, result: Any) -> str:
        """
        格式化工具执行结果

        Args:
            tool_name: 工具名
            result: 执行结果

        Returns:
            格式化的结果字符串
        """
        return json.dumps({
            "tool": tool_name,
            "result": result
        }, ensure_ascii=False, indent=2)


def parse_tool_calls(response: str):
    """模块级便捷函数，代理到 FunctionCalling.parse_tool_calls"""
    return FunctionCalling.parse_tool_calls(response)