"""
欲望系统 - 基于信念和环境感知生成欲望集
"""
import json
import re
from datetime import datetime
from typing import List, Dict
from .desire_types import Desire, DesireType, DESIRE_CN
from posim.prompts.prompt_loader import PromptLoader

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
        
        # 构建曝光信息文本（带时间戳，区分原发/转发/转发评论）
        exposed_parts = []
        if posts:
            exposed_parts.append("### 曝光的博文：")
            for i, post in enumerate(posts[:5], 1):
                post_time = post.get('time', '')
                time_prefix = f"[{post_time}] " if post_time else ""
                author = post.get('author', '')
                post_type = post.get('post_type', 'original')
                
                if post_type == 'repost':
                    original_author = post.get('original_author', '')
                    original_content = post.get('original_content', '')[:150]
                    exposed_parts.append(f"{i}. {time_prefix}@{author} 转发了博文（仅转发）：")
                    exposed_parts.append(f"   原博 @{original_author}：{original_content}")
                elif post_type == 'repost_comment':
                    original_author = post.get('original_author', '')
                    original_content = post.get('original_content', '')[:150]
                    repost_comment = post.get('repost_comment', post.get('content', ''))[:100]
                    exposed_parts.append(f"{i}. {time_prefix}@{author} 转发了博文并评论：")
                    exposed_parts.append(f"   原博 @{original_author}：{original_content}")
                    exposed_parts.append(f"   转发评论：{repost_comment}")
                else:
                    content = post.get('content', '')[:200]
                    exposed_parts.append(f"{i}. {time_prefix}@{author} 发表了博文：{content}")
                
                if post.get('comments'):
                    exposed_parts.append(f"   热门评论：{post['comments'][0] if post['comments'] else ''}")
        # 构建外部突发事件文本（3个事件窗口）
        events_parts = []
        if events:
            events_parts.append("### 当前外部突发事件：")
            for evt in events[:3]:
                events_parts.append(f"- [{evt.get('time', '')}] {evt.get('content', '')}")
        
        prompt = prompts['desire'].format(
            current_time=current_time,
            belief_text=belief_text,
            exposed_info="\n".join(exposed_parts) if exposed_parts else "无新信息",
            external_events="\n".join(events_parts) if events_parts else "",
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
            except json.JSONDecodeError:
                pass
        
        # 如果解析失败，返回默认欲望
        if not desires:
            desires = [Desire(type=DesireType.INFORMATION_SEEKING, weight=0.5, description="默认欲望")]
        return desires
