"""agents 包：所有业务 Agent 的实现所在，一个子目录一个 Agent。

这里刻意不做任何导入聚合（本文件为空壳）—— 各 Agent 的依赖差得远
（有的要检索器、有的要多模态 API），统一在 __init__ 里 import 会让
任何一个可选依赖缺失就整个包炸掉。所以一律按需从子模块直接导入，例如
from agents.database_agent.agent import DatabaseAgent。

现役 Agent：knowledge / database / customer_service / vqa / chat。
另有 document_agent、critic_agent 两个"有类但未注册"的，别当现役读。
"""
