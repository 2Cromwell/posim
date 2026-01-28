"""
LLM客户端 - 异步并发调用大模型接口
"""
import asyncio
from typing import Dict, Any, Optional, List
from openai import AsyncOpenAI
from pydantic import BaseModel


class LLMClient(BaseModel):
    """大模型客户端，支持异步并发调用"""
    name: str = 'openai'
    base_url: str = ''
    api_key: str = ''
    model: str = ''
    temperature: float = 0.7
    top_p: float = 0.9
    weight: float = 1.0
    enabled: bool = True
    aclient: Any = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        self.aclient = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    async def async_query(self, messages: List[Dict[str, str]], hyper_params: Optional[Dict[str, Any]] = None) -> str:
        """异步查询大模型"""
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            **(hyper_params or {})
        }
        response = await self.aclient.chat.completions.create(**params)
        return response.choices[0].message.content

    async def async_text_query(self, query: str, system_prompt: str = None, hyper_params: Optional[Dict[str, Any]] = None) -> str:
        """简化的文本查询接口"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": query})
        return await self.async_query(messages, hyper_params)
