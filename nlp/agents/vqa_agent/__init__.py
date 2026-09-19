"""vqa_agent 包：多模态识图 Agent（VQA = 视觉问答）。

只导出 VQAAgent 一个类。注意它和别的 Agent 形态不同：不注册工具、不走 ReAct，
直接把图片连同问题发给千问VL模型拿一段回答（细节见 agent.py 文件头）。
"""
from .agent import VQAAgent

__all__ = ["VQAAgent"]
