"""
从零构建 LangChain Agent（不使用 create_agent）
================================================

Agent 的本质：
  LLM 收到用户问题 → 决定调用哪个工具 → 拿到工具结果 → 再思考 → 最终回复

这个文件从最底层的 "手动解析" 开始，逐步演进到使用 LangChain 标准组件。
每个 Step 都可以独立运行。

运行方式（在项目根目录）：
  python -m core.agent_from_scratch  --step 1
  python -m core.agent_from_scratch  --step 2
  ...
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional

# ============================================================
# 前置：LLM 配置（使用 DashScope / OpenAI 兼容接口）
# ============================================================

# 请替换为你自己的 API Key 和 Base URL
# DashScope:  https://dashscope.aliyuncs.com/compatible-mode/v1
# OpenAI:     https://api.openai.com/v1

LLM_CONFIG = {
    "api_key": os.getenv("DASHSCOPE_API_KEY", os.getenv("OPENAI_API_KEY", "your-api-key")),
    "base_url": os.getenv("OPENAI_BASE_URL", os.getenv(
        "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")),
    "model": os.getenv("LLM_MODEL", "qwen-plus"),
    "temperature": 0.0,
}


# ============================================================
# Mock CRM 工具集（复用项目现有的 MockCRM）
# ============================================================

from core.crm_mock import MockCRM

crm = MockCRM()


def get_user_info(user_id: str) -> str:
    """查询用户信息。参数 user_id: 用户ID，如 user_001"""
    user = crm.get_user_info(user_id)
    if user:
        return json.dumps(user, ensure_ascii=False, indent=2)
    return f"未找到用户: {user_id}"


def create_ticket(user_id: str, issue: str, priority: str = "normal", description: str = "") -> str:
    """创建工单。参数 user_id: 用户ID, issue: 问题简述, priority: low/normal/high/urgent, description: 详细描述"""
    ticket = crm.create_ticket(user_id, issue, priority, description)
    return json.dumps(ticket, ensure_ascii=False, indent=2)


def get_ticket(ticket_id: str) -> str:
    """查询工单。参数 ticket_id: 工单ID，如 T00001"""
    ticket = crm.get_ticket(ticket_id)
    if ticket:
        return json.dumps(ticket, ensure_ascii=False, indent=2)
    return f"未找到工单: {ticket_id}"


# 工具定义列表——这是 Agent 的 "技能清单"
TOOLS_MANUAL: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_user_info",
            "description": "查询用户信息，根据用户ID获取用户的详细资料（姓名、VIP等级、订单数等）",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "用户ID，例如 user_001"}
                },
                "required": ["user_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_ticket",
            "description": "创建客服工单",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "用户ID"},
                    "issue": {"type": "string", "description": "问题简述"},
                    "priority": {
                        "type": "string",
                        "enum": ["low", "normal", "high", "urgent"],
                        "description": "优先级"
                    },
                    "description": {"type": "string", "description": "详细描述"}
                },
                "required": ["user_id", "issue"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_ticket",
            "description": "查询工单详情",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_id": {"type": "string", "description": "工单ID，例如 T00001"}
                },
                "required": ["ticket_id"]
            }
        }
    }
]

# 工具名 → 实际函数的映射
TOOL_FUNCTIONS = {
    "get_user_info": get_user_info,
    "create_ticket": create_ticket,
    "get_ticket": get_ticket,
}

SYSTEM_PROMPT = """你是一个客服助手。你可以使用工具查询用户信息、创建工单、查询工单。

规则：
1. 当用户提到某个用户ID时，先调用 get_user_info 查询该用户的信息
2. 当用户要求创建工单时，调用 create_ticket
3. 回复用户时用中文，简洁友好"""


# ============================================================
# Step 1: 纯手动 —— 自己调用 API，自己解析 tool_calls
# ============================================================

def step1_pure_manual(user_message: str):
    """
    这是最底层的实现。没有任何 LangChain 组件，只有 HTTP 调用。
    目标是看清楚 OpenAI 兼容 API 的 tool calling 机制底层长什么样。

    流程：
      1. 构造 messages（system + user + tools）
      2. 调用 API
      3. 如果返回 tool_calls → 执行工具 → 把结果塞回 messages → 再调 API
      4. 如果返回 content → 结束，这就是最终回复
    """
    from openai import OpenAI

    client = OpenAI(api_key=LLM_CONFIG["api_key"], base_url=LLM_CONFIG["base_url"])

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    max_turns = 5  # 防止无限循环
    for turn in range(max_turns):
        print(f"\n{'='*50}")
        print(f"[Turn {turn + 1}] 调用 LLM...")

        response = client.chat.completions.create(
            model=LLM_CONFIG["model"],
            messages=messages,
            tools=TOOLS_MANUAL,
            temperature=LLM_CONFIG["temperature"],
        )

        choice = response.choices[0]
        msg = choice.message

        # 情况 A：LLM 想调用工具
        if msg.tool_calls:
            print(f"  LLM 决定调用工具: {[tc.function.name for tc in msg.tool_calls]}")

            # 把 LLM 的 tool_calls 消息追加到对话历史
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    }
                    for tc in msg.tool_calls
                ]
            })

            # 执行每个工具调用，把结果以 tool role 追加
            for tc in msg.tool_calls:
                func_name = tc.function.name
                func_args = json.loads(tc.function.arguments)
                print(f"  执行 {func_name}({func_args})")

                fn = TOOL_FUNCTIONS.get(func_name)
                if fn:
                    result = fn(**func_args)
                else:
                    result = f"未知工具: {func_name}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result),
                })

            # 循环继续，让 LLM 看到工具结果后再思考
            continue

        # 情况 B：LLM 直接返回了文本回复（最终答案）
        if msg.content:
            print(f"  LLM 最终回复: {msg.content}")
            return msg.content

        # 情况 C：既没有 tool_calls 也没有 content（罕见）
        print("  LLM 返回为空，退出")
        break

    return "无法完成任务"


# ============================================================
# Step 2: 使用 LangChain 的 ChatModel + Tool 定义（不用 create_agent）
# ============================================================

def step2_langchain_tools(user_message: str):
    """
    用 LangChain 的两个核心组件替代裸 API 调用：
      - ChatOpenAI: 封装了 LLM 调用，统一接口
      - @tool / StructuredTool: 用 Python 函数自动生成 tool schema

    但循环逻辑仍然是我们自己写的。
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.tools import tool
    from langchain_core.messages import (
        SystemMessage, HumanMessage, ToolMessage
    )

    # --- 用 @tool 装饰器定义工具 ---
    # LangChain 会自动从函数签名 + docstring + 类型注解生成 JSON Schema

    @tool
    def get_user_info_lc(user_id: str) -> str:
        """查询用户信息，根据用户ID获取用户的详细资料（姓名、VIP等级、订单数等）"""
        user = crm.get_user_info(user_id)
        if user:
            return json.dumps(user, ensure_ascii=False, indent=2)
        return f"未找到用户: {user_id}"

    @tool
    def create_ticket_lc(
        user_id: str,
        issue: str,
        priority: str = "normal",
        description: str = ""
    ) -> str:
        """创建客服工单"""
        ticket = crm.create_ticket(user_id, issue, priority, description)
        return json.dumps(ticket, ensure_ascii=False, indent=2)

    @tool
    def get_ticket_lc(ticket_id: str) -> str:
        """查询工单详情"""
        ticket = crm.get_ticket(ticket_id)
        if ticket:
            return json.dumps(ticket, ensure_ascii=False, indent=2)
        return f"未找到工单: {ticket_id}"

    tools = [get_user_info_lc, create_ticket_lc, get_ticket_lc]
    tools_map = {t.name: t for t in tools}

    # --- 创建 LLM ---
    llm = ChatOpenAI(
        api_key=LLM_CONFIG["api_key"],
        base_url=LLM_CONFIG["base_url"],
        model=LLM_CONFIG["model"],
        temperature=LLM_CONFIG["temperature"],
    )

    # --- 关键步骤：bind_tools ---
    # 把工具定义绑定到 LLM 上，这样每次调用 LLM 都会自动带上 tools 参数
    llm_with_tools = llm.bind_tools(tools)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]

    max_turns = 5
    for turn in range(max_turns):
        print(f"\n{'='*50}")
        print(f"[Turn {turn + 1}] 调用 LLM...")

        # ai_msg 是 LangChain 的 AIMessage 对象
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)

        # 检查是否有 tool_calls
        if ai_msg.tool_calls:
            print(f"  LLM 决定调用工具: {[tc['name'] for tc in ai_msg.tool_calls]}")

            for tc in ai_msg.tool_calls:
                func_name = tc["name"]
                func_args = tc["args"]
                print(f"  执行 {func_name}({func_args})")

                tool_fn = tools_map[func_name]
                # LangChain tool 的 invoke 接受 dict 作为输入
                result = tool_fn.invoke(func_args)

                # 把工具结果追加为 ToolMessage
                messages.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tc["id"],
                ))

            # 把工具结果给 LLM 后再来一轮
            continue

        # 没有 tool_calls，就是最终回复
        print(f"  LLM 最终回复: {ai_msg.content}")
        return ai_msg.content

    return "无法完成任务"


# ============================================================
# Step 3: 使用 Runnable 链 —— 但还是手写循环
# ============================================================

def step3_runnable_chain(user_message: str):
    """
    Step 2 里 messages 是手动 append 的。Step 3 用 LangChain 的 Runnable
    模式来管理消息流，让代码更声明式。

    关键概念：
      - RunnablePassthrough: 透传数据
      - RunnableLambda: 把普通函数包装成 Runnable
      - 管道操作符 | 串联各步骤
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.tools import tool
    from langchain_core.messages import (
        SystemMessage, HumanMessage, ToolMessage, AIMessage
    )
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough

    @tool
    def get_user_info_lc(user_id: str) -> str:
        """查询用户信息，根据用户ID获取用户的详细资料（姓名、VIP等级、订单数等）"""
        user = crm.get_user_info(user_id)
        if user:
            return json.dumps(user, ensure_ascii=False, indent=2)
        return f"未找到用户: {user_id}"

    @tool
    def create_ticket_lc(
        user_id: str, issue: str, priority: str = "normal", description: str = ""
    ) -> str:
        """创建客服工单"""
        ticket = crm.create_ticket(user_id, issue, priority, description)
        return json.dumps(ticket, ensure_ascii=False, indent=2)

    @tool
    def get_ticket_lc(ticket_id: str) -> str:
        """查询工单详情"""
        ticket = crm.get_ticket(ticket_id)
        if ticket:
            return json.dumps(ticket, ensure_ascii=False, indent=2)
        return f"未找到工单: {ticket_id}"

    tools = [get_user_info_lc, create_ticket_lc, get_ticket_lc]
    tools_map = {t.name: t for t in tools}

    llm = ChatOpenAI(
        api_key=LLM_CONFIG["api_key"],
        base_url=LLM_CONFIG["base_url"],
        model=LLM_CONFIG["model"],
        temperature=LLM_CONFIG["temperature"],
    )
    llm_with_tools = llm.bind_tools(tools)

    # --- 自定义一个函数来执行工具调用 ---
    def execute_tools(ai_msg: AIMessage) -> List[ToolMessage]:
        """接收 LLM 的 AIMessage，执行其中的 tool_calls，返回 ToolMessage 列表"""
        tool_msgs = []
        for tc in ai_msg.tool_calls:
            func_name = tc["name"]
            func_args = tc["args"]
            print(f"  执行 {func_name}({func_args})")
            result = tools_map[func_name].invoke(func_args)
            tool_msgs.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
        return tool_msgs

    # --- 构建可复用的 tool-call 处理链 ---
    # 这段链的意思是：拿到 AIMessage → 执行工具 → 返回 ToolMessage 列表
    tool_exec_chain = RunnableLambda(execute_tools)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]

    max_turns = 5
    for turn in range(max_turns):
        print(f"\n{'='*50}")
        print(f"[Turn {turn + 1}] 调用 LLM...")

        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)

        if not ai_msg.tool_calls:
            print(f"  LLM 最终回复: {ai_msg.content}")
            return ai_msg.content

        print(f"  LLM 决定调用工具: {[tc['name'] for tc in ai_msg.tool_calls]}")

        # 用链执行工具
        tool_msgs = tool_exec_chain.invoke(ai_msg)
        messages.extend(tool_msgs)

    return "无法完成任务"


# ============================================================
# Step 4: AgentExecutor 的替代 —— 手写 Tool calling Agent
# ============================================================

def step4_agent_executor_pattern(user_message: str):
    """
    LangChain 的 create_tool_calling_agent + AgentExecutor 本质就是你
    在 Step 1-3 里写的东西。这里展示如何把它抽象成一个可复用的类，
    和 AgentExecutor 做类似的事情——但不依赖它。

    这个类就是你自己的 "AgentExecutor"，你完全掌控每一行逻辑。
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.tools import tool
    from langchain_core.messages import (
        SystemMessage, HumanMessage, ToolMessage, AIMessage, BaseMessage
    )

    @tool
    def get_user_info_lc(user_id: str) -> str:
        """查询用户信息，根据用户ID获取用户的详细资料（姓名、VIP等级、订单数等）"""
        user = crm.get_user_info(user_id)
        if user:
            return json.dumps(user, ensure_ascii=False, indent=2)
        return f"未找到用户: {user_id}"

    @tool
    def create_ticket_lc(
        user_id: str, issue: str, priority: str = "normal", description: str = ""
    ) -> str:
        """创建客服工单"""
        ticket = crm.create_ticket(user_id, issue, priority, description)
        return json.dumps(ticket, ensure_ascii=False, indent=2)

    @tool
    def get_ticket_lc(ticket_id: str) -> str:
        """查询工单详情"""
        ticket = crm.get_ticket(ticket_id)
        if ticket:
            return json.dumps(ticket, ensure_ascii=False, indent=2)
        return f"未找到工单: {ticket_id}"

    # ========================================
    # 你自己的 Agent 类
    # ========================================
    class SimpleAgent:
        """
        一个极简的 Tool-calling Agent。

        对比 AgentExecutor：
          - AgentExecutor 源码 ~500 行，处理了 early_stopping、max_iterations、
            callbacks、streaming、错误恢复等边界情况
          - SimpleAgent ~60 行，只有核心循环，让你看清楚骨架是什么
        """

        def __init__(self, llm: ChatOpenAI, tools: list, system_prompt: str):
            self.llm = llm
            self.tools = tools
            self.tools_map = {t.name: t for t in tools}
            self.system_prompt = system_prompt
            # bind_tools 是关键——它让 LLM 知道有哪些工具可用
            self.llm_with_tools = llm.bind_tools(tools)

        def invoke(self, user_input: str, max_turns: int = 10) -> str:
            messages: List[BaseMessage] = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=user_input),
            ]

            for turn in range(max_turns):
                ai_msg = self.llm_with_tools.invoke(messages)
                messages.append(ai_msg)

                # 没有工具调用 = 最终答案
                if not ai_msg.tool_calls:
                    return ai_msg.content or ""

                # 执行工具
                for tc in ai_msg.tool_calls:
                    name = tc["name"]
                    args = tc["args"]
                    try:
                        result = self.tools_map[name].invoke(args)
                    except Exception as e:
                        result = f"工具执行失败: {e}"
                    messages.append(ToolMessage(
                        content=str(result), tool_call_id=tc["id"]
                    ))

            return "已达到最大轮次，无法完成任务"

    # --- 使用 ---
    llm = ChatOpenAI(
        api_key=LLM_CONFIG["api_key"],
        base_url=LLM_CONFIG["base_url"],
        model=LLM_CONFIG["model"],
        temperature=LLM_CONFIG["temperature"],
    )

    tools = [get_user_info_lc, create_ticket_lc, get_ticket_lc]
    agent = SimpleAgent(llm, tools, SYSTEM_PROMPT)
    return agent.invoke(user_message)


# ============================================================
# Step 5: 加上记忆（Memory）—— 多轮对话
# ============================================================

def step5_with_memory(user_message: str):
    """
    上面的 Agent 每次调用都是独立的。加上记忆后，Agent 能记住之前的对话。

    LangChain 提供了多种 Memory 实现，但原理很简单：
      本质就是把历史消息存在一个列表里，每次调用时拼到新消息前面。
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.tools import tool
    from langchain_core.messages import (
        SystemMessage, HumanMessage, ToolMessage, AIMessage, BaseMessage
    )

    @tool
    def get_user_info_lc(user_id: str) -> str:
        """查询用户信息"""
        user = crm.get_user_info(user_id)
        if user:
            return json.dumps(user, ensure_ascii=False, indent=2)
        return f"未找到用户: {user_id}"

    @tool
    def create_ticket_lc(
        user_id: str, issue: str, priority: str = "normal", description: str = ""
    ) -> str:
        """创建客服工单"""
        ticket = crm.create_ticket(user_id, issue, priority, description)
        return json.dumps(ticket, ensure_ascii=False, indent=2)

    @tool
    def get_ticket_lc(ticket_id: str) -> str:
        """查询工单详情"""
        ticket = crm.get_ticket(ticket_id)
        if ticket:
            return json.dumps(ticket, ensure_ascii=False, indent=2)
        return f"未找到工单: {ticket_id}"

    class AgentWithMemory:
        """带记忆的 Agent"""

        def __init__(self, llm: ChatOpenAI, tools: list, system_prompt: str):
            self.llm_with_tools = llm.bind_tools(tools)
            self.tools_map = {t.name: t for t in tools}
            self.system_prompt = system_prompt
            # 这就是 "记忆" —— 一个消息列表
            self.history: List[BaseMessage] = []

        def invoke(self, user_input: str, max_turns: int = 10) -> str:
            # 构建本轮的消息列表 = system + 历史 + 新消息
            messages: List[BaseMessage] = [SystemMessage(content=self.system_prompt)]
            messages.extend(self.history)
            messages.append(HumanMessage(content=user_input))

            for _ in range(max_turns):
                ai_msg = self.llm_with_tools.invoke(messages)
                messages.append(ai_msg)

                if not ai_msg.tool_calls:
                    # 把本轮新增的消息存入历史（Human + AI 的完整交互）
                    self.history.append(HumanMessage(content=user_input))
                    self.history.append(ai_msg)
                    # 限制历史长度，防止超出 context window
                    if len(self.history) > 20:
                        self.history = self.history[-20:]
                    return ai_msg.content or ""

                for tc in ai_msg.tool_calls:
                    name = tc["name"]
                    args = tc["args"]
                    try:
                        result = self.tools_map[name].invoke(args)
                    except Exception as e:
                        result = f"工具执行失败: {e}"
                    messages.append(ToolMessage(
                        content=str(result), tool_call_id=tc["id"]
                    ))

            return "已达到最大轮次"

    llm = ChatOpenAI(
        api_key=LLM_CONFIG["api_key"],
        base_url=LLM_CONFIG["base_url"],
        model=LLM_CONFIG["model"],
        temperature=LLM_CONFIG["temperature"],
    )

    tools = [get_user_info_lc, create_ticket_lc, get_ticket_lc]

    # 使用全局实例来保持跨调用的记忆
    global _agent_with_memory
    if "_agent_with_memory" not in globals() or not hasattr(globals(), '_agent_with_memory'):
        _agent_with_memory = AgentWithMemory(llm, tools, SYSTEM_PROMPT)
    else:
        _agent_with_memory = globals()['_agent_with_memory']

    return _agent_with_memory.invoke(user_message)


# ============================================================
# 运行入口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="从零学 Agent")
    parser.add_argument("--step", type=int, default=1, choices=[1, 2, 3, 4, 5],
                        help="Step 1-5，对应不同层次的教学实现")
    parser.add_argument("--message", type=str, default=None,
                        help="用户消息（留空则使用默认示例）")
    args = parser.parse_args()

    # 默认测试用例
    default_messages = {
        1: "用户 user_001 的订单没发货，帮我查一下他的信息并建个工单",
        2: "用户 user_001 的订单没发货，帮我查一下他的信息并建个工单",
        3: "查一下工单 T00001 的状态",
        4: "用户 user_002 反映物流太慢，帮他建一个高优先级的工单",
        5: "查一下 user_001 的信息",  # 可以连续调用，Step 5 会记住上下文
    }

    message = args.message or default_messages.get(args.step, "帮我查一下工单 T00001")

    step_funcs = {
        1: step1_pure_manual,
        2: step2_langchain_tools,
        3: step3_runnable_chain,
        4: step4_agent_executor_pattern,
        5: step5_with_memory,
    }

    print(f"\n{'#'*60}")
    print(f"# Step {args.step}")
    print(f"# 用户: {message}")
    print(f"{'#'*60}")

    result = step_funcs[args.step](message)
    print(f"\n{'='*60}")
    print(f"最终结果:\n{result}")
