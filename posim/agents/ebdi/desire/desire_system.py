"""
欲望系统 - 基于信念和环境感知生成欲望集
"""
import json
import re
import logging
from datetime import datetime
from typing import List, Dict
from .desire_types import Desire, DesireType
from posim.prompts.prompt_loader import PromptLoader
from posim.utils.formatters import format_external_events, format_exposed_posts

logger = logging.getLogger(__name__)

# 强度等级到数值的映射
INTENSITY_MAP = {
    '极低': 0.1, '低': 0.3, '中等': 0.5, '高': 0.7, '极高': 0.9
}


class DesireSystem:
    """欲望系统 - 生成带权重的多目标欲望列表"""
    
    def __init__(self, api_pool):
        self.api_pool = api_pool
    
    async def generate_desires(self, belief_text: str, exposed_posts: List[Dict], 
                               external_events: List[Dict], agent_type: str = 'citizen',
                               current_time: str = None, event_background: str = "") -> List[Desire]:
        """基于信念和环境生成欲望集"""
        if current_time is None:
            current_time = datetime.now().strftime('%Y-%m-%dT%H:%M')
        prompt = self._build_desire_prompt(belief_text, exposed_posts, external_events, agent_type, current_time, event_background)
        
        # 调用LLM（无需system prompt，已合并到主提示词中）
        response = await self.api_pool.async_text_query(prompt, "", purpose='desire')
        return self._parse_desires(response)
    
    def _build_desire_prompt(self, belief_text: str, posts: List[Dict], 
                             events: List[Dict], agent_type: str, current_time: str,
                             event_background: str = "") -> str:
        prompts = PromptLoader.get_desire_prompts(agent_type)
        output_format = PromptLoader.get_output_format('desire')
        
        # 使用工具函数格式化信息
        exposed_text = format_exposed_posts(posts, current_time, max_posts=5,
                                           include_stats=True, include_comments=True)
        events_text = format_external_events(events, current_time)
        
        prompt = prompts['desire'].format(
            current_time=current_time,
            belief_text=belief_text,
            exposed_info=exposed_text if exposed_text else "无新信息",
            external_events=events_text,
            event_background=event_background
        )
        return prompt + output_format
    
    def _parse_desires(self, response: str) -> List[Desire]:
        """解析欲望列表（纯中文字段）"""
        desires = []
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                desire_list = data.get('欲望列表', [])
                for d in desire_list:
                    try:
                        # 解析欲望类型（中文）
                        type_cn = d.get('类型', '自我表达')
                        desire_type = DesireType(type_cn)
                        
                        # 解析强度（中文）
                        intensity = d.get('强度', '中等')
                        weight = INTENSITY_MAP.get(intensity, 0.5)
                        
                        # 解析作用对象和描述（中文）
                        target = d.get('作用对象')
                        description = d.get('描述', '')
                        
                        desires.append(Desire(
                            type=desire_type,
                            weight=weight,
                            target=target,
                            description=description
                        ))
                    except (ValueError, KeyError):
                        continue
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse desire response JSON: {e}")
        
        # 如果解析失败，返回默认欲望
        if not desires:
            desires = [Desire(type=DesireType.INFORMATION_SEEKING, weight=0.5, description="默认欲望")]
        return desires
