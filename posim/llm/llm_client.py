"""
LLM客户端 - 异步并发调用大模型接口
支持超参数扰动以增强模拟多样性
"""
import asyncio
import logging
import random
from typing import Dict, Any, Optional, List
from openai import AsyncOpenAI
from pydantic import BaseModel
import httpx

logger = logging.getLogger(__name__)

# 默认超时设置（秒）
DEFAULT_TIMEOUT = 120  # 连接+读取总超时
DEFAULT_CONNECT_TIMEOUT = 30  # 连接超时

# 超参数扰动配置
PERTURBATION_CONFIG = {
    'enabled': True,               # 是否启用扰动
    'temperature_range': 0.1,      # temperature扰动范围 ±0.1
    'top_p_range': 0.05,           # top_p扰动范围 ±0.05
    'frequency_penalty_range': 0.2,  # frequency_penalty扰动范围 [0, 0.2]
    'presence_penalty_range': 0.2,   # presence_penalty扰动范围 [0, 0.2]
}


def generate_perturbed_params(base_temperature: float = 0.7, 
                               base_top_p: float = 0.9,
                               perturbation_strength: float = 1.0) -> Dict[str, float]:
    """
    生成扰动后的超参数，增强模拟多样性
    
    Args:
        base_temperature: 基础temperature值
        base_top_p: 基础top_p值
        perturbation_strength: 扰动强度 (0.0-1.0)，0表示无扰动，1表示最大扰动
    
    Returns:
        包含扰动后超参数的字典
    """
    if not PERTURBATION_CONFIG['enabled'] or perturbation_strength <= 0:
        return {}
    
    params = {}
    strength = min(max(perturbation_strength, 0.0), 1.0)
    
    # Temperature扰动：在基础值附近小幅波动
    temp_range = PERTURBATION_CONFIG['temperature_range'] * strength
    temp_delta = random.uniform(-temp_range, temp_range)
    perturbed_temp = max(0.1, min(1.5, base_temperature + temp_delta))
    params['temperature'] = round(perturbed_temp, 3)
    
    # Top_p扰动
    top_p_range = PERTURBATION_CONFIG['top_p_range'] * strength
    top_p_delta = random.uniform(-top_p_range, top_p_range)
    perturbed_top_p = max(0.5, min(1.0, base_top_p + top_p_delta))
    params['top_p'] = round(perturbed_top_p, 3)
    
    # 随机添加frequency_penalty（增加词汇多样性）
    if random.random() < 0.3 * strength:  # 30%概率添加
        freq_penalty = random.uniform(0, PERTURBATION_CONFIG['frequency_penalty_range'] * strength)
        params['frequency_penalty'] = round(freq_penalty, 3)
    
    # 随机添加presence_penalty（鼓励新话题）
    if random.random() < 0.3 * strength:  # 30%概率添加
        pres_penalty = random.uniform(0, PERTURBATION_CONFIG['presence_penalty_range'] * strength)
        params['presence_penalty'] = round(pres_penalty, 3)
    
    return params


class LLMClient(BaseModel):
    """大模型客户端，支持异步并发调用（带超时控制和超参数扰动）"""
    name: str = 'openai'
    base_url: str = ''
    api_key: str = ''
    model: str = ''
    temperature: float = 0.7
    top_p: float = 0.9
    weight: float = 1.0
    enabled: bool = True
    timeout: float = DEFAULT_TIMEOUT  # 请求超时（秒）
    enable_perturbation: bool = True  # 是否启用超参数扰动
    perturbation_strength: float = 0.5  # 扰动强度 (0.0-1.0)
    aclient: Any = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        # 配置带超时的 httpx 客户端
        timeout_config = httpx.Timeout(
            timeout=self.timeout,
            connect=DEFAULT_CONNECT_TIMEOUT
        )
        self.aclient = AsyncOpenAI(
            api_key=self.api_key, 
            base_url=self.base_url,
            timeout=timeout_config,
            max_retries=2  # 自动重试2次
        )

    async def async_query(self, messages: List[Dict[str, str]], hyper_params: Optional[Dict[str, Any]] = None,
                          use_perturbation: bool = None) -> str:
        """
        异步查询大模型（带超时控制和可选的超参数扰动）
        
        Args:
            messages: 消息列表
            hyper_params: 自定义超参数（会覆盖扰动参数）
            use_perturbation: 是否使用扰动，None表示使用实例默认设置
        """
        # 基础参数
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        
        # 应用超参数扰动以增强多样性
        should_perturb = use_perturbation if use_perturbation is not None else self.enable_perturbation
        if should_perturb and self.perturbation_strength > 0:
            perturbed = generate_perturbed_params(
                base_temperature=self.temperature,
                base_top_p=self.top_p,
                perturbation_strength=self.perturbation_strength
            )
            params.update(perturbed)
        
        # 用户指定的超参数优先级最高
        if hyper_params:
            params.update(hyper_params)
        
        try:
            # 使用 asyncio.wait_for 添加额外的超时保护
            response = await asyncio.wait_for(
                self.aclient.chat.completions.create(**params),
                timeout=self.timeout + 10  # 比 httpx 超时多10秒作为兜底
            )
            return response.choices[0].message.content
        except asyncio.TimeoutError:
            logger.warning(f"[{self.name}] LLM request timed out after {self.timeout}s")
            raise
        except Exception as e:
            logger.warning(f"[{self.name}] LLM request failed: {type(e).__name__}: {e}")
            raise

    async def async_text_query(self, query: str, system_prompt: str = None, 
                               hyper_params: Optional[Dict[str, Any]] = None,
                               use_perturbation: bool = None) -> str:
        """简化的文本查询接口"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": query})
        return await self.async_query(messages, hyper_params, use_perturbation)
    
    def set_perturbation(self, enabled: bool = True, strength: float = 0.5):
        """设置超参数扰动配置"""
        self.enable_perturbation = enabled
        self.perturbation_strength = min(max(strength, 0.0), 1.0)
