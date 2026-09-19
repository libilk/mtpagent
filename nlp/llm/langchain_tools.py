# -*- coding: utf-8 -*-
"""
LangChain Tool 封装
==================

使用 LangChain 的 StructuredTool + Pydantic Schema 实现工具注册与自动参数校验。

**本文件最值钱的一条教训：`parameters` 不只是"给 LLM 看的说明书"。**
它同时驱动另一件事 —— 生成 args_schema（参数结构）做**运行时参数校验**，
而这一步决定了多参数工具到底能不能被调用。读者最容易低估的就是这一点。

历史坑（已修）：本文件早先用 LangChain 的 `Tool` 注册所有工具。但 `Tool` 的源码
（langchain_core/tools/simple.py 的 _to_args_and_kwargs）写死了这段：

    all_args = list(args) + list(kwargs.values())
    if len(all_args) != 1:
        raise ToolException("Too many arguments to single-input tool ...")

也就是说 `Tool` **永远只接受一个入参**，连传了 args_schema 也救不回来。
项目原有工具恰好都只传 1 个参数，坑长期没暴露 —— 而 `create_ticket`（6 个参数，
注册于 agents/customer_service_agent/agent.py）其实一直是坏的，
只要 LLM 一次传 2 个以上参数就炸。后来改用 StructuredTool + Pydantic args_schema 才真正修好。

所以本文件现在有两条注册路径，分界不在"工具重不重要"，而在**有没有声明 parameters**：
有 → 能算出 args_schema → 走 StructuredTool（可多参数、带校验）；
没有 → 退回旧的 Tool 兜底（只适用于单参数/零参数工具）。
"""

import inspect
import logging
from typing import List, Dict, Any, Callable, Optional, Union
from langchain_core.tools import Tool, StructuredTool
from pydantic import BaseModel, Field, create_model

logger = logging.getLogger(__name__)

# JSON Schema 类型 → Python 类型。用于把调用方声明的 parameters 转成 Pydantic 字段。
# 为什么要这张表：parameters 里的 "type" 是给 LLM 看的字符串类型名，
# 而 Pydantic 校验需要真正的 Python 类型对象，两边对不上就没法建模型。
# 注意映射是有损的：array→list、object→dict 只校验"是个列表/字典"，
# 不校验元素类型 —— Pydantic 在这里能帮到的就这一层。
_JSON_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _build_args_schema(tool_name: str, parameters: dict, func: Callable) -> Optional[type]:
    """
    根据调用方声明的 parameters（JSON Schema 片段）构造 Pydantic 模型。

    **为什么需要这一步：** LangChain 的 `Tool` 在没有 args_schema 时，会一律把
    工具当成"单参数工具"（只有一个 input）。此时如果 LLM 一次传了 2 个以上参数，
    invoke() 会直接抛 `Too many arguments to single-input tool`。

    所以凡是参数多于一个的工具，都必须带上 args_schema 才能真正被调用。

    Args:
        tool_name: 工具名，用于生成模型名，便于报错时定位
        parameters: 注册时提供的 JSON Schema 形式参数声明
        func: 实际函数，用来核对声明的参数名是否真的存在

    Returns:
        Pydantic 模型类；没有可用字段时返回 None（保持旧行为）

    模型是"运行时拼出来"的：parameters 要等注册那一刻才拿到，没法提前手写一个类，
    所以末尾用 create_model 动态建类（类名带工具名，Pydantic 报错时能看出是谁）。
    """
    properties = (parameters or {}).get("properties") or {}
    required = set((parameters or {}).get("required") or [])

    # 用函数签名核对声明：声明了但函数里没有的参数会被剔除。
    # 这样即使某处 tool 的声明和签名对不上，也只是少传一个参数，
    # 而不是整个工具调不通 —— 同时打日志把问题暴露出来。
    try:
        sig_names = set(inspect.signature(func).parameters) - {"self"}
    except (TypeError, ValueError):
        sig_names = set(properties)   # 拿不到签名就不核对

    fields = {}
    for prop_name, spec in properties.items():
        if prop_name not in sig_names:
            logger.warning(
                f"工具 '{tool_name}' 声明了参数 '{prop_name}'，"
                f"但函数签名里没有它（实际参数: {sorted(sig_names)}），已忽略"
            )
            continue

        py_type = _JSON_TYPE_MAP.get((spec or {}).get("type"), str)
        desc = (spec or {}).get("description", "")

        if prop_name in required:
            fields[prop_name] = (py_type, Field(..., description=desc))
        elif "default" in (spec or {}):
            fields[prop_name] = (py_type, Field(default=spec["default"], description=desc))
        else:
            # 可选且没给默认值 → 允许 None，避免 Pydantic 报"字段必填"
            fields[prop_name] = (Optional[py_type], Field(default=None, description=desc))

    if not fields:
        return None

    return create_model(f"{tool_name}_Args", **fields)


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
        # handle_tool_error=True：工具内部抛异常时不把异常抛穿整个调用栈，
        # 而是转成一段错误字符串当结果返回 —— 让 LLM 看到失败原因、自己换参数重试。
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

        名字里的"普通"别理解成"次要"：customer_service_agent 与 database_agent
        的工具全部经此注册，它才是本项目的主力注册入口；register_structured_tool
        则被 knowledge_agent / document_agent 使用。两者返回的工具行为一致。

        Args:
            parameters: JSON Schema 形式的参数声明。**强烈建议提供** ——
                它有两个作用：一是决定喂给 LLM 的工具描述，
                二是用来生成 args_schema，让多参数工具真正能被调用。
                **第二个作用才是关键**：没有它就只能退回单参数的 Tool（见下方 ★）。
        """
        actual_func = func or function
        if actual_func is None:
            raise ValueError(f"注册工具 '{name}' 失败：必须提供 func 或 function 参数")

        # 用声明的 parameters 生成 args_schema。
        #
        # ★ 必须用 StructuredTool，不能用 Tool ★
        # Tool 源码在 langchain_core/tools/simple.py（旧版本这个类叫 SimpleTool，
        # 本版本直接叫 Tool 并继承 BaseTool，但下面这段写死的判断没变）：
        #     if len(all_args) != 1:
        #         raise ToolException("Too many arguments to single-input tool ...")
        # 也就是说 Tool **永远只接受一个参数**，哪怕传了 args_schema 也没用。
        # 结果就是：只要 LLM 一次传 2 个以上参数（比如 create_ticket 同时传
        # issue + user_id + priority），调用就会失败。
        #
        # 只有声明了 parameters 的工具才能算出 args_schema，才用 StructuredTool；
        # 没声明的走旧路径，保持兼容。
        args_schema = _build_args_schema(name, parameters, actual_func) if parameters else None

        # ★ parameters 的第二重身份（读者最容易低估的一点）★
        # 它不只是 get_tools_schema() 里那份"给 LLM 看的说明书"，
        # 更是上面 _build_args_schema 的**唯一输入**：没有 parameters 就没有
        # args_schema，没有 args_schema 就没有运行时参数校验，工具也只能是单参数。
        # 换句话说，"给模型看的"和"用来校验的"是同一份声明 —— 漏写它，
        # 坏的往往不是模型理解，而是工具根本调不通。
        #
        # 分支条件值得记牢：判据是"args_schema 能否算出来"，不是"参数有几个"。
        # 边界情况：零参工具即使声明了 parameters（properties 为空），
        # _build_args_schema 也返回 None，照旧走 Tool —— 对零参调用是安全的，
        # 真正危险的是"多参数却漏声明 parameters"。
        if args_schema is not None:
            tool = StructuredTool.from_function(
                func=actual_func,
                name=name,
                description=description,
                args_schema=args_schema,
                handle_tool_error=handle_tool_error,
            )
        else:
            tool = Tool(
                name=name,
                func=actual_func,
                description=description,
                handle_tool_error=handle_tool_error,
            )

        if parameters:
            # 保留原始声明：get_tools_schema() 会优先用它，保证喂给 LLM 的
            # schema 格式不变（Pydantic 生成的 schema 会多出 title 等噪音字段）
            # 注意这里对两条分支都会挂 —— 所以经本方法注册的 StructuredTool
            # 也带 _raw_parameters，get_tools_schema() 里"谁优先"要看下面那段。
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

        事实性说明（未改逻辑）：实际判据不是"工具是哪个类"，而是"有没有
        `_raw_parameters`"——经 register_tool 且声明了 parameters 的工具，
        哪怕是 StructuredTool 也会命中第一支、返回原始声明；只有经
        register_structured_tool 注册的工具（不挂 _raw_parameters）才走
        args_schema 那一支。上面的 docstring 措辞因此略旧，以代码为准。
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

            # 这里是 parameters 的"说明书"身份：原样喂给 LLM。
            # 优先用原始 dict 而非 Pydantic 生成物，是为了避开 title/$defs 等噪音。
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
                        # 没有 args_schema 的原生 Tool 不接受 dict，
                        # 无参调用时退化成传空字符串（保持旧行为）。
                        # 有 args_schema 的走 invoke，由 Pydantic 校验参数。
                        if not tool_input and getattr(tool, "args_schema", None) is None:
                            result = tool.run("")
                        else:
                            result = tool.invoke(tool_input)
                    else:
                        result = tool.run(tool_input)
                    return str(result)
                # 这里刻意吞掉异常、返回错误字符串而非向上抛：工具失败的信息会作为
                # 返回值回到 LLM 的上下文里，让它自己换参数重试；一旦抛出，整个
                # ReAct（推理-行动循环）当场中断。这是 try/except 之外的兜底 ——
                # handle_tool_error=True 已能拦下多数工具内部异常。
                except Exception as e:
                    logger.error(f"工具 {tool_name} 执行失败: {e}")
                    return f"工具执行错误: {str(e)}"

        return f"错误：工具 {tool_name} 不存在"
