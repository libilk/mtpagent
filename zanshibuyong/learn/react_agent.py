"""
ReAct Agent —— Reasoning + Acting

和 tool-calling agent 的核心区别：
  1. 不用 bind_tools —— 工具定义写在 prompt 文本里
  2. LLM 输出纯文本 —— 不是 JSON tool_calls
  3. 你自己解析 LLM 的输出 —— 用正则提取 Action / Action Input
  4. 工具结果以 Observation 格式追加到 messages

ReAct 格式：
  Thought: <推理过程>
  Action: <工具名>
  Action Input: <JSON参数>
  Observation: <工具返回结果>
  ...（多轮）...
  Thought: I know the answer
  Final Answer: <最终回复>

运行：
  export DASHSCOPE_API_KEY="your-key"
  python learn/react_agent.py
"""

import json
import os
import re

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# ============================================================
# 工具定义 —— 就是普通函数，没有 @tool 装饰器
# ============================================================
_users = {
    "user_001": {"name": "张三", "vip": "gold", "orders": 25},
    "user_002": {"name": "李四", "vip": "silver", "orders": 10},
}
_tickets: dict = {}
_ticket_counter = 0


def get_user_info(user_id: str) -> str:
    user = _users.get(user_id)
    if user:
        return json.dumps(user, ensure_ascii=False)
    return f"未找到用户 {user_id}"


def create_ticket(user_id: str, issue: str, priority: str = "normal") -> str:
    global _ticket_counter
    _ticket_counter += 1
    tid = f"T{_ticket_counter:05d}"
    _tickets[tid] = {"ticket_id": tid, "user_id": user_id, "issue": issue, "priority": priority}
    return f"工单已创建: {tid} - {issue} (优先级:{priority})"


TOOLS = {
    "get_user_info": get_user_info,
    "create_ticket": create_ticket,
}

# ============================================================
# ReAct Prompt —— 工具定义写在文本里，不是 bind_tools
# ============================================================
REACT_SYSTEM_PROMPT = """你是一个客服助手。你可以使用以下工具：

工具列表：
1. get_user_info - 查询用户信息
   参数：{"user_id": "string, 用户ID, 如 user_001"}

2. create_ticket - 创建客服工单
   参数：{"user_id": "string, 用户ID", "issue": "string, 问题简述", "priority": "string, 优先级 low/normal/high/urgent"}

你必须按以下格式回复。每个步骤必须包含 Thought，如果需要调用工具，必须包含 Action 和 Action Input。如果得到了最终答案，必须包含 Final Answer。

格式：
Thought: 你的思考过程
Action: 工具名称
Action Input: {"参数名": "参数值"}

Observation: 工具返回结果（由系统自动填入）

...重复以上直到得出最终答案...

Thought: 我现在有足够信息回答
Final Answer: 最终回复

严格规则：
1. Thought 必须写在 Action 之前
2. Action Input 必须是单行合法 JSON
3. Final Answer 之前必须有一个 Thought
4. 一次只能调用一个工具"""


# ============================================================
# ReAct Agent
# ============================================================

class ReActAgent:
    """
    ReAct Agent。

    和 tool-calling agent 的区别：
      - 没有 bind_tools
      - 工具执行后，结果以 "Observation: xxx" 的形式追加到 messages
      - 自己解析 LLM 输出，提取 Action 和 Action Input
    """

    def __init__(self):
        llm = init_chat_model(
            model=os.getenv("LLM_MODEL", "qwen-plus"),
            api_key=os.getenv("DASHSCOPE_API_KEY", os.getenv("OPENAI_API_KEY")),
            base_url=os.getenv(
                "OPENAI_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            model_provider="openai",
        )
        self.llm = llm  # 注意：没有 bind_tools
        self.memory: list = []

    def run(self, user_input: str) -> str:
        messages = [
            SystemMessage(content=REACT_SYSTEM_PROMPT),
            *self.memory,
            HumanMessage(content=user_input),
        ]

        for _ in range(10):
            # 1. 调 LLM —— 返回的是文本，不是 tool_calls
            ai_msg: AIMessage = self.llm.invoke(messages)
            messages.append(ai_msg)
            text = ai_msg.content or ""

            print(f"\n[LLM输出]\n{text}\n")  # 观察 ReAct 格式

            # 2. 解析 ———— 看 LLM 是想调工具还是给最终答案
            action, action_input = self._parse_action(text)

            if action and action_input:
                # 2a. LLM 想调工具 → 执行 → 拼接 Observation → 继续循环
                observation = self._execute(action, action_input)
                # 把 Observation 追加到同一个 AIMessage 后面
                # （或作为单独的 HumanMessage，两种写法都可以）
                messages.append(HumanMessage(content=f"Observation: {observation}"))
                continue

            # 2b. 没有 Action → 检查是否有 Final Answer
            final = self._parse_final(text)
            if final:
                self._save_to_memory(user_input, f"Final Answer: {final}")
                return final

            # 2c. 都没有 → 模型可能迷路了，提醒它
            messages.append(HumanMessage(
                content="请按格式回复：先 Thought，如果有工具要调用则写 Action 和 Action Input，否则写 Final Answer。"
            ))

        return "达到最大轮次"

    # ---- 解析 ----

    def _parse_action(self, text: str) -> tuple[str | None, dict | None]:
        """从 LLM 输出中提取 Action 和 Action Input"""
        action_match = re.search(r"Action:\s*(.+?)\s*$", text, re.MULTILINE)
        input_match = re.search(r"Action Input:\s*(.+?)\s*$", text, re.MULTILINE)

        if not action_match or not input_match:
            return None, None

        action = action_match.group(1).strip()
        try:
            action_input = json.loads(input_match.group(1).strip())
        except json.JSONDecodeError:
            return None, None

        return action, action_input

    def _parse_final(self, text: str) -> str | None:
        """从 LLM 输出中提取 Final Answer"""
        match = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    # ---- 执行 ----

    def _execute(self, action: str, action_input: dict) -> str:
        """执行工具，返回结果文本"""
        fn = TOOLS.get(action)
        if fn is None:
            return f"未知工具: {action}。可用工具: {list(TOOLS.keys())}"
        try:
            return str(fn(**action_input))
        except Exception as e:
            return f"工具执行失败: {e}"

    # ---- 记忆 ----

    def _save_to_memory(self, user_input: str, final_answer: str):
        self.memory.append(HumanMessage(content=user_input))
        self.memory.append(AIMessage(content=final_answer))
        if len(self.memory) > 20:
            self.memory = self.memory[-20:]


# ============================================================
# 运行
# ============================================================

def main():
    print("=" * 50)
    print("ReAct Agent 就绪，输入 quit 退出")
    print("=" * 50)

    agent = ReActAgent()

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
