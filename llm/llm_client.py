import os
import json
import logging
from http.client import responses
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
    has_OpenAi=True
except ImportError:
    has_OpenAi=False
    logger.warning("openAI库未安装")
class LLM:
    def __init__(
    self,
    model_name:str="qwen-plus",
    api_key: Optional[str]=None,
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    max_tokens: int = 1024,
    temperature: float = 0.7,
    timeout: int = 60,
    enable_cache: bool = True
):
     """
    Args:
            model_name: 模型名称（qwen-max/qwen-plus）
            api_key: API密钥
            base_url: API地址
            max_tokens: 最大生成token数
            temperature: 温度参数
            timeout: 超时时间
            enable_cache: 是否启用LLM缓存
    """
     self.model_name = model_name
     self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
     self.base_url = base_url
     self.max_tokens = max_tokens
     self.temperature = temperature
     self.timeout = timeout
     self.enable_cache = enable_cache
     if not self.api_key:
         raise ValueError("API密钥未设置，请设置DASHSCOPE_API_KEY环境变量")
     if has_OpenAi:
         self.client=OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=timeout
            )
     else:
         self.client = None
         logger.warning("将使用requests库调用API")
         if self.enable_cache:
             from core.unified_cache import LLMCache#待完成
             self.llm_cache = LLMCache(max_size=500, ttl=3600)
             logger.info("LLM缓存已启用")
         else:
             self.llm_cache = None

     def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
        ** kwargs
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
         """
         messages=[]
         if system_prompt:
             messages.append({"role": "system", "content": system_prompt})
         messages.append({"role": "user", "content": prompt})
         return self.chat(messages, tools=tools, **kwargs)

     def chat(
     self,
     messages: List[Dict[str, str]],
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
         if self.enable_cache and self.llm_cache and not tools:
             cached_response = self.llm_cache.get_cached_response(messages, self.model_name)
             logger.debug("缓存命中")
             return cached_response
         if has_OpenAi:
            response = self._chat_with_openai(messages, tools, **kwargs)
         else:
            response=self._chat_with_requests(messages, tools, **kwargs)
            # LLM缓存：存入缓存（只缓存无工具调用的普通对话）
         if self.enable_cache and self.llm_cache and not tools:
            self.llm_cache.cache_response(messages, response, self.model_name)
            return response

     def _chat_with_openai(
          self,
          messages: List[Dict[str, str]],
          tools: Optional[List[Dict]] = None,
          **kwargs
     )->str:
         """使用openai库调用"""
         try:
             params = {
                 "model": self.model_name,
                 "messages": messages,
                 "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                 "temperature": kwargs.get("temperature", self.temperature),
             }
             if tools:
                params["tools"] = tools
                params["tool_choice"] = "auto"
             response=self.client.chat.completions.create(**params)
             #处理工具调用
             message = response.choices[0].message
             if hasattr(message, 'tool_calls') and message.tool_calls:
                 return json.dumps({
                     "type": "tool_calls",
                     "tool_calls": [
                         {
                             "id": tc.id,
                             "function": {
                                 "name": tc.function.name,
                                 "arguments": tc.function.arguments
                             }
                         }
                         for tc in message.tool_calls
                     ]
                 })
             return message.content
         except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            raise

     def _chat_with_requests(
             self,
             messages: List[Dict[str, str]],
             tools: Optional[List[Dict]] = None,
             **kwargs
     ) -> str:
         """使用requests库调用"""
         import requests

         try:
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
                 timeout=self.timeout
             )
             response.raise_for_status()

             result = response.json()
             message = result["choices"][0]["message"]

             # 处理工具调用
             if "tool_calls" in message:
                 return json.dumps({
                     "type": "tool_calls",
                     "tool_calls": message["tool_calls"]
                 })

             return message["content"]

         except Exception as e:
             logger.error(f"LLM调用失败: {e}")
             raise

     def create_llm(model_name: str = "qwen-plus", **kwargs) -> LLM:
         """
         创建LLM实例的工厂函数

         Args:
             model_name: 模型名称
             **kwargs: 其他参数

         Returns:
             LLM实例
         """
         return LLM(model_name=model_name, **kwargs)






