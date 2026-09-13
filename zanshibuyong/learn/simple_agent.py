"""
从零构建 Agent —— 最小可运行示例

运行前设置环境变量：
  export DASHSCOPE_API_KEY="your-key"
  export LLM_MODEL="qwen-plus"

运行：
  python learn/simple_agent.py

代码从 main() 开始看，逐行追踪即可。
"""

import json
import os

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)


# ============================================================
# 第一部分：定义工具
# ============================================================
DASHSCOPE_API_KEY=os.getenv("DASHSCOPE_API_KEY")
# 用内存字典模拟数据库，省去文件依赖
_users = {
    "user_001": {"name": "张三", "vip": "gold", "orders": 25},
    "user_002": {"name": "李四", "vip": "silver", "orders": 10},
}
_tickets: dict[str, dict] = {}
_ticket_counter = 0


@tool
def get_user_info(user_id: str) -> str:
    """查询用户信息，参数 user_id 是用户ID，如 user_001"""
    user = _users.get(user_id)
    if user:
        return json.dumps(user, ensure_ascii=False)
    return f"未找到用户 {user_id}"


@tool
def create_ticket(user_id: str, issue: str, priority: str = "normal") -> str:
    """创建工单。priority 可选 low / normal / high / urgent"""
    global _ticket_counter
    _ticket_counter += 1
    tid = f"T{_ticket_counter:05d}"
    _tickets[tid] = {"ticket_id": tid, "user_id": user_id, "issue": issue, "priority": priority}
    return f"工单已创建: {tid} - {issue} (优先级:{priority})"


# 工具列表 + 名字→函数的映射
TOOLS = [get_user_info, create_ticket]
TOOLS_MAP = {t.name: t for t in TOOLS}


# ============================================================
# 第二部分：Agent 类（核心：消息列表 + 循环 + 记忆）
# ============================================================

class Agent:
    """
    Agent 只有三样东西：
      1. llm_with_tools — 绑了工具的 LLM
      2. tools_map       — 工具名→函数的映射
      3. memory          — 历史消息列表（这就是"记忆"）
    """

    def __init__(self):
        # 创建 LLM —— 就是一个无状态的函数
        llm = init_chat_model(
            model=os.getenv("LLM_MODEL", "qwen-plus"),
            api_key=os.getenv("DASHSCOPE_API_KEY", os.getenv("OPENAI_API_KEY")),
            base_url=os.getenv(
                "OPENAI_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            model_provider="openai",
        )

        # bind_tools：让 LLM 知道有哪些工具可以调用
        self.llm_with_tools = llm.bind_tools(TOOLS)
        self.tools_map = TOOLS_MAP
        self.memory: list = []  # 记忆就是消息列表

    def run(self, user_input: str) -> str:
        """处理一条用户消息，返回最终回复"""

        # 拼消息列表：系统提示 + 历史记忆 + 当前输入
        messages = [
            SystemMessage(content="你是客服助手。查询用户用 get_user_info，建工单用 create_ticket。用中文回复。"),
            *self.memory,
            HumanMessage(content=user_input),
        ]

        # 核心循环
        for _ in range(5):  # 最多 5 轮，防止死循环
            ai_msg: AIMessage = self.llm_with_tools.invoke(messages)
            messages.append(ai_msg)

            # 没有 tool_calls → 这是最终回复，结束
            if not ai_msg.tool_calls:
                self._save_to_memory(user_input, ai_msg)
                return ai_msg.content or ""

            # 有 tool_calls → 执行工具，结果追加到消息列表
            for tc in ai_msg.tool_calls:
                fn = self.tools_map[tc["name"]]
                result = fn.invoke(tc["args"])
                messages.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tc["id"],
                ))
            # 循环继续，LLM 下一轮会看到 ToolMessage

        return "达到最大轮次，任务未完成"

    def _save_to_memory(self, user_input: str, ai_msg: AIMessage):
        """保存本轮对话到记忆"""
        self.memory.append(HumanMessage(content=user_input))
        self.memory.append(ai_msg)
        # 只保留最近 10 轮，防止超出 token 上限
        if len(self.memory) > 20:
            self.memory = self.memory[-20:]


# ============================================================
# 第三部分：运行
# ============================================================

def main():
    print("=" * 50)
    print("Agent 就绪，输入 quit 退出")
    print("=" * 50)

    agent = Agent()

    while True:
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break

        reply = agent.run(user_input)
        print(f"\nAgent: {reply}")


if __name__ == "__main__":
    main()
