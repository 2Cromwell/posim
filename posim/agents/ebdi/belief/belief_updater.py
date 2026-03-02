"""
信念推断机制 - 基于外部环境和历史记忆推断当前信念状态
引入确认偏差、锚定效应等认知偏见机制
"""
import json
import re
from datetime import datetime
from typing import Dict, Any, List
from .belief_system import BeliefSystem
from posim.prompts.prompt_loader import PromptLoader
from posim.utils.formatters import format_external_events, format_exposed_posts, format_memories


class BeliefUpdater:
    """信念更新器"""
    
    def __init__(self, api_pool):
        self.api_pool = api_pool
    
    async def update_belief(self, belief_system: BeliefSystem, exposed_posts: List[Dict], 
                           external_events: List[Dict], memories: List[Dict], 
                           agent_type: str = 'citizen', current_time: str = None,
                           event_background: str = "") -> BeliefSystem:
        """
        推断当前时刻的信念状态
        Args:
            belief_system: 当前信念系统
            exposed_posts: 曝光的博文列表（含评论）
            external_events: 外部事件列表
            memories: 历史行为记忆
            agent_type: 智能体类型
            current_time: 当前仿真时间
        """
        if current_time is None:
            current_time = datetime.now().strftime('%Y-%m-%dT%H:%M')
        # 构建推断提示词
        prompt = self._build_update_prompt(belief_system, exposed_posts, external_events, memories, agent_type, current_time, event_background)
        
        # 调用LLM推断信念（无需system prompt，已合并到主提示词中）
        response = await self.api_pool.async_text_query(prompt, "", purpose='belief')
        
        # 解析推断结果
        updates = self._parse_update_response(response)
        
        # 应用推断结果
        self._apply_updates(belief_system, updates, current_time)
        
        return belief_system
    
    def _build_update_prompt(self, belief: BeliefSystem, posts: List[Dict], 
                            events: List[Dict], memories: List[Dict], agent_type: str,
                            current_time: str, event_background: str = "") -> str:
        prompts = PromptLoader.get_belief_prompts(agent_type)
        output_format = PromptLoader.get_output_format('belief')
        
        # 使用工具函数格式化信息
        new_info_text = format_exposed_posts(posts, current_time, max_posts=5, 
                                             include_stats=True, include_comments=True)
        events_text = format_external_events(events, current_time)
        memory_text = format_memories(memories)
        
        # 获取身份信息和其他信念状态
        identity_text = belief.identity.to_prompt_text()
        other_beliefs = self._format_other_beliefs(belief)
        
        prompt = prompts['update'].format(
            current_time=current_time,
            identity_text=identity_text,
            belief_text=other_beliefs,
            new_info=new_info_text if new_info_text else "无新信息",
            memories=memory_text if memory_text else "无历史记忆",
            external_events=events_text,
            event_background=event_background
        )
        return prompt + output_format
    
    def _format_other_beliefs(self, belief: BeliefSystem) -> str:
        """格式化除身份以外的信念状态"""
        parts = [
            belief.psychology.to_prompt_text(),
            belief.event.to_prompt_text(),
            belief.emotion.to_prompt_text()
        ]
        return "\n\n".join([p for p in parts if p])
    
    def _parse_update_response(self, response: str) -> Dict[str, Any]:
        """解析LLM返回的更新结果"""
        # 尝试多种模式匹配 JSON
        patterns = [
            r'```json\s*([\s\S]*?)\s*```',  # 标准markdown代码块
            r'```\s*([\s\S]*?)\s*```',       # 无语言标记的代码块
            r'\{[\s\S]*\}',                  # 直接查找JSON对象
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                json_str = match.group(1) if match.lastindex else match.group(0)
                try:
                    return json.loads(json_str.strip())
                except json.JSONDecodeError:
                    continue
        
        # 所有模式都失败，返回空字典
        return {}
    
    # 中文情绪到英文的映射
    EMOTION_MAP = {
        '快乐': 'happiness', '悲伤': 'sadness', '愤怒': 'anger',
        '恐惧': 'fear', '惊讶': 'surprise', '厌恶': 'disgust'
    }
    
    def _apply_updates(self, belief: BeliefSystem, updates: Dict[str, Any], current_time: str = None):
        """应用信念推断结果（纯中文字段）"""
        # 更新事件观点
        event_opinions = updates.get('事件观点', [])
        for op_update in event_opinions:
            subject = op_update.get('主体', '')
            opinion = op_update.get('观点', '')
            if subject and opinion:
                belief.event.update_opinion(subject, opinion, '', current_time)
        
        # 更新情绪向量
        emotion_vector_raw = updates.get('情绪向量', {})
        if emotion_vector_raw:
            # 将中文情绪名转换为英文
            emotion_vector = {}
            for key, value in emotion_vector_raw.items():
                eng_key = self.EMOTION_MAP.get(key, key)
                emotion_vector[eng_key] = value
            belief.emotion.update_from_content(emotion_vector)
        
        # 更新心理认知
        psych_beliefs = updates.get('心理认知', [])
        if psych_beliefs:
            belief.psychology.belief_items = psych_beliefs
