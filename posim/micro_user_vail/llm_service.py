# -*- coding: utf-8 -*-
"""LLM服务封装，支持多模型并发调用"""
import asyncio
import json
import re
import logging
import time
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
import httpx

logger = logging.getLogger(__name__)


def extract_json_from_response(text: str) -> Optional[Dict]:
    """从LLM响应中提取JSON，处理各种格式"""
    if not text:
        return None

    cleaned = text.strip()

    # 去除 <think>...</think> 标签（思考模型）
    cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL).strip()

    # 尝试从 markdown code block 提取
    code_block = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', cleaned, re.DOTALL)
    if code_block:
        try:
            return json.loads(code_block.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 尝试直接解析
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 尝试找到第一个 { 和最后一个 }
    first_brace = cleaned.find('{')
    last_brace = cleaned.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(cleaned[first_brace:last_brace + 1])
        except json.JSONDecodeError:
            pass

    return None


class LLMService:
    """统一的LLM服务，支持并发控制"""

    def __init__(self, name: str, base_url: str, api_key: str, model: str,
                 temperature: float = 0.7, top_p: float = 0.9,
                 max_concurrent: int = 30, timeout: int = 120):
        self.name = name
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_concurrent = max_concurrent
        self.timeout = timeout

        timeout_config = httpx.Timeout(timeout=timeout, connect=30)
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_config,
            max_retries=2,
        )
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.stats = {'calls': 0, 'errors': 0, 'total_time': 0.0}

    async def query(self, prompt: str, system_prompt: str = None,
                    temperature: float = None, top_p: float = None,
                    max_retries: int = 3) -> str:
        """单次查询，带重试"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        temp = temperature if temperature is not None else self.temperature
        tp = top_p if top_p is not None else self.top_p

        for attempt in range(max_retries):
            try:
                async with self.semaphore:
                    start = time.time()
                    response = await asyncio.wait_for(
                        self.client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            temperature=temp,
                            top_p=tp,
                        ),
                        timeout=self.timeout + 10
                    )
                    elapsed = time.time() - start
                    self.stats['calls'] += 1
                    self.stats['total_time'] += elapsed
                    return response.choices[0].message.content or ""
            except Exception as e:
                self.stats['errors'] += 1
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    logger.warning(f"[{self.name}] Retry {attempt+1}/{max_retries}: {e}")
                else:
                    logger.error(f"[{self.name}] Failed after {max_retries} retries: {e}")
                    return ""

    async def batch_query(self, prompts: List[Dict[str, Any]],
                          show_progress: bool = False,
                          progress_desc: str = "") -> List[str]:
        """批量并发查询"""
        if not prompts:
            return []

        async def _query_one(idx, p):
            result = await self.query(
                prompt=p['prompt'],
                system_prompt=p.get('system_prompt'),
                temperature=p.get('temperature'),
                top_p=p.get('top_p'),
            )
            return idx, result

        tasks = [_query_one(i, p) for i, p in enumerate(prompts)]
        results = [""] * len(prompts)

        if show_progress:
            from tqdm.asyncio import tqdm_asyncio
            completed = await tqdm_asyncio.gather(*tasks, desc=progress_desc)
        else:
            completed = await asyncio.gather(*tasks)

        for idx, result in completed:
            results[idx] = result

        return results

    async def query_json(self, prompt: str, system_prompt: str = None,
                         temperature: float = None) -> Optional[Dict]:
        """查询并解析JSON"""
        response = await self.query(prompt, system_prompt, temperature)
        return extract_json_from_response(response), response

    def get_stats(self) -> Dict:
        return {
            **self.stats,
            'avg_time': self.stats['total_time'] / max(self.stats['calls'], 1),
            'model': self.model,
        }
