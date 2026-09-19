# -*- coding: utf-8 -*-
"""
统一LLM接口
===========

封装Qwen-Max和Qwen-Plus调用，支持Function Calling（函数调用：模型不直接回答，
而是吐出"要调哪个函数、参数是什么"，由调用方去执行）。

**这里的价值是"统一"**：上层（planner / router / 各 Agent）只认这一个类，
换模型只改构造参数，调用方一行不用动 —— 它不关心底下是 Qwen-Max 还是 Qwen-Plus。

**为什么用 `openai` SDK，而不是通义官方的 `dashscope` 包？**
因为 DashScope 开放了 **OpenAI 兼容接口**（看默认 base_url 里的 compatible-mode），
对它来说通义只是个"讲 OpenAI 协议的服务器"。所以本项目**根本没装 dashscope**，
别被 README 满篇的"通义千问"误导。附带好处是与厂商解耦，换别家不用改上层代码。
（openai 是可选依赖：没装时退回手写 requests，见 _chat_with_requests。）

调用形态只有两种：generate()（单轮）与 chat()（多轮）；tools 参数可给任一形态
挂上函数调用能力。

**本类不做"结构化输出"**：全仓库没有 chat_structured 之类的方法。让模型返回 JSON
是靠 prompt 里约定、由调用方解析（llm/langchain_parser.parse_llm_output 带 fallback
兜底），这里拿到的始终是一段裸文本。
"""

import os
import json
import logging
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    logger.warning("openai库未安装，将使用requests库")


class LLM:
    """统一LLM接口"""

    def __init__(
        self,
        model_name: str = "qwen-plus",
        api_key: Optional[str] = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",  # 路径里的 compatible-mode 就是"OpenAI 兼容接口"，接口形状照搬 OpenAI 官方
        max_tokens: int = 1024,
        temperature: float = 0.7,
        timeout: int = 60,
        enable_cache: bool = True,
        redis_client=None
    ):
        """
        初始化LLM

        Args:
            model_name: 模型名称（qwen-max/qwen-plus）
            api_key: API密钥
            base_url: API地址
            max_tokens: 最大生成token数
            temperature: 温度参数（越低输出越确定；判断/解析类调用常传 0.1~0.3）
            timeout: 超时时间（秒）；仅约束单次 HTTP 请求，本类不负责重试
            enable_cache: 是否启用LLM缓存
            redis_client: Redis 客户端实例（可选，用于 LLM 缓存持久化）
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.base_url = base_url
        self.max_tokens = max_tokens  # token（词元）：LLM 处理文本的最小单位，也是计费与长度单位
        self.temperature = temperature
        self.timeout = timeout
        self.enable_cache = enable_cache

        # 宁可启动即失败，也不留到第一次调用时才报 401
        if not self.api_key:
            raise ValueError("API密钥未设置，请设置DASHSCOPE_API_KEY环境变量")

        # 初始化客户端
        if HAS_OPENAI:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=timeout  # openai 路径的超时在这里设；手写 requests 路径另传
            )
        else:
            # fallback（降级兜底）：没装 openai 就把 client 置空，改走 _chat_with_requests
            self.client = None
            logger.warning("将使用requests库调用API")

        # 初始化LLM缓存（支持可选 Redis 持久化）
        # 缓存默认只是**进程内**的；只有传了 redis_client 才会跨进程持久化
        if self.enable_cache:
            from core.unified_cache import LLMCache
            self.llm_cache = LLMCache(max_size=500, ttl=3600, redis_client=redis_client)
            logger.info("LLM缓存已启用")
        else:
            self.llm_cache = None

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> str:
        """
        生成文本

        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词
            tools: 工具列表（Function Calling）
            **kwargs: 其他参数

        Returns:
            生成的文本

        这只是"拼两条消息"的薄封装，多轮对话请直接用 chat。
        **kwargs 会透传给 chat，可在此临时覆盖 max_tokens / temperature 等。
        """
        messages = []
        # role 是 OpenAI 的消息角色约定：system 定人设与规矩，user 是用户提问。
        # 单轮调用只有这两条；多轮对话的历史也在 messages 里，由 chat() 接收。
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        return self.chat(messages, tools=tools, **kwargs)

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> str:
        """
        多轮对话

        Args:
            messages: 消息列表
            tools: 工具列表
            **kwargs: 其他参数

        Returns:
            生成的文本
        """
        # LLM缓存：检查缓存（只缓存无工具调用的普通对话）
        # 为什么排除 tools：工具调用要真去执行、有副作用，且同样的话该调哪个工具并不确定
        if self.enable_cache and self.llm_cache and not tools:
            cached_response = self.llm_cache.get_cached_response(messages, self.model_name)
            if cached_response:
                logger.debug(f"[LLM缓存命中] model={self.model_name}")
                return cached_response

        # 缓存未命中，调用API
        # 两条实现路径产出同一种返回值（纯文本，或 tool_calls 的 JSON 字符串），上层无感
        if HAS_OPENAI:
            response = self._chat_with_openai(messages, tools, **kwargs)
        else:
            response = self._chat_with_requests(messages, tools, **kwargs)

        # LLM缓存：存入缓存（只缓存无工具调用的普通对话）
        if self.enable_cache and self.llm_cache and not tools:
            self.llm_cache.cache_response(messages, response, self.model_name)

        return response

    def _chat_with_openai(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> str:
        """使用openai库调用（返回值始终是 str，不是结构体）"""
        try:
            params = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "temperature": kwargs.get("temperature", self.temperature),
            }

            if tools:
                params["tools"] = tools
                # tool_choice="auto" 是让模型自己决定"直接回答"还是"去调函数"；不是强制调用
                params["tool_choice"] = "auto"

            response = self.client.chat.completions.create(**params)

            # 处理工具调用
            message = response.choices[0].message
            # 关键：这里把工具调用**序列化成 JSON 字符串**返回，并不替调用方执行。
            # 本类只负责"转达模型想调什么"，真正执行工具、把结果回灌给模型是上层的事。
            # （别把返回值当结构化对象用 —— 要 json.loads 自己解。）
            if hasattr(message, 'tool_calls') and message.tool_calls:
                return json.dumps({
                    "type": "tool_calls",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in message.tool_calls
                    ]
                })

            # 模型直接回答时文本在 content；纯工具调用那一轮 content 常为 None，故用 or "" 兜住
            return message.content or ""

        except Exception as e:
            # 本类**没有重试/退避**：一次失败即向上抛。重试是编排层的职责
            # （evaluator 判不合格 → iteration 回退重跑），别以为这里会自愈。
            logger.error(f"LLM调用失败: {e}")
            raise

    def _chat_with_requests(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> str:
        """使用requests库调用（openai 未装时的降级实现，请求形状与 OpenAI 一致）"""
        import requests

        try:
            # /chat/completions 是 OpenAI 的接口路径 —— 兼容接口连路径都一样，这就是"兼容"的含义
            url = f"{self.base_url}/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            data = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "temperature": kwargs.get("temperature", self.temperature),
            }

            if tools:
                data["tools"] = tools
                data["tool_choice"] = "auto"

            response = requests.post(
                url,
                headers=headers,
                json=data,
                timeout=self.timeout  # 手写路径必须自己传超时，否则可能永久挂住
            )
            response.raise_for_status()

            result = response.json()
            message = result["choices"][0]["message"]

            # 处理工具调用
            if "tool_calls" in message:
                # 安全补全：确保每个 tool_call 都有 type 字段
                # （接口有时会省略它；下游若按 OpenAI 标准结构解析会 KeyError）
                for tc in message["tool_calls"]:
                    tc.setdefault("type", "function")
                return json.dumps({
                    "type": "tool_calls",
                    "tool_calls": message["tool_calls"]
                })

            return message.get("content") or ""

        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            raise


def create_llm(model_name: str = "qwen-plus", **kwargs) -> LLM:
    """
    创建LLM实例的工厂函数（factory：把"怎么造对象"收口到一处）

    Args:
        model_name: 模型名称
        **kwargs: 其他参数（原样转给 LLM 构造函数，如 redis_client）

    Returns:
        LLM实例

    上层不必自己 new LLM，统一走这里 —— 将来要加默认参数或做实例复用，只改这一处。
    """
    return LLM(model_name=model_name, **kwargs)
