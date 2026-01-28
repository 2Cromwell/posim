"""
表达策略选择器 - 选择情绪、立场、表达风格和叙事策略
"""
import json
import re
from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass
from enum import Enum
from posim.prompts.prompt_loader import PromptLoader

# 强度等级到数值的映射
INTENSITY_MAP = {
    '极低': 0.1, '低': 0.3, '中等': 0.5, '高': 0.7, '极高': 0.9
}


class EmotionTone(Enum):
    HAPPINESS = "happiness"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    NEUTRAL = "neutral"


class Stance(Enum):
    SUPPORT = "support"
    OPPOSE = "oppose"
    NEUTRAL = "neutral"


class ExpressionStyle(Enum):
    RATIONAL = "rational"       # 理性分析
    SARCASTIC = "sarcastic"     # 讽刺嘲讽
    AGGRESSIVE = "aggressive"   # 攻击谩骂
    EMPATHETIC = "empathetic"   # 共情理解
    QUESTIONING = "questioning" # 质疑求证


class NarrativeStrategy(Enum):
    FACT_LIST = "fact_list"           # 事实罗列
    STORYTELLING = "storytelling"     # 讲故事
    LABELING = "labeling"             # 扣帽子
    CALL_TO_ACTION = "call_to_action" # 呼吁行动
    CITE_AUTHORITY = "cite_authority" # 转述权威
    CONSPIRACY = "conspiracy"         # 阴谋质疑


@dataclass
class ExpressionStrategy:
    """表达策略"""
    emotion: EmotionTone
    emotion_intensity: float  # 0-1
    stance: Stance
    stance_intensity: float   # 0-1
    style: ExpressionStyle
    narrative: NarrativeStrategy


class StrategySelector:
    """表达策略选择器"""
    
    def __init__(self, api_pool):
        self.api_pool = api_pool
    
    async def select_strategy(self, belief_text: str, desires: List[Dict], 
                              action_type: str, target_content: str,
                              agent_type: str = 'citizen', current_time: str = None) -> ExpressionStrategy:
        """选择表达策略"""
        # 对于like和repost行为，不需要表达策略
        if action_type in ['like', 'repost']:
            return ExpressionStrategy(
                emotion=EmotionTone.NEUTRAL, emotion_intensity=0.0,
                stance=Stance.NEUTRAL, stance_intensity=0.0,
                style=ExpressionStyle.RATIONAL, narrative=NarrativeStrategy.FACT_LIST
            )
        
        if current_time is None:
            current_time = datetime.now().isoformat()
        prompt = self._build_strategy_prompt(belief_text, desires, action_type, target_content, agent_type, current_time)
        system_prompt = self._get_system_prompt(agent_type)
        
        response = await self.api_pool.async_text_query(prompt, system_prompt, purpose='strategy')
        return self._parse_strategy(response)
    
    def _get_system_prompt(self, agent_type: str) -> str:
        prompts = PromptLoader.get_intention_prompts(agent_type)
        return prompts.get('action_system', '')
    
    def _build_strategy_prompt(self, belief_text: str, desires: List[Dict], 
                                action_type: str, target_content: str, agent_type: str,
                                current_time: str) -> str:
        prompts = PromptLoader.get_intention_prompts(agent_type)
        output_format = PromptLoader.get_output_format('strategy')
        
        # 构建欲望文本
        desires_text = "\n".join([f"- {d.get('type', '')}: {d.get('description', '')}" for d in desires])
        
        prompt = prompts['strategy'].format(
            current_time=current_time,
            belief_text=belief_text,
            desires=desires_text if desires_text else "无明确欲望",
            action_type=action_type,
            target_content=target_content[:200] if target_content else "原创发布"
        )
        return prompt + output_format
    
    def _parse_strategy(self, response: str) -> ExpressionStrategy:
        """解析表达策略"""
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                # 将强度等级转换为数值
                emotion_int = data.get('emotion_intensity', '中等')
                if isinstance(emotion_int, str):
                    emotion_int = INTENSITY_MAP.get(emotion_int, 0.5)
                stance_int = data.get('stance_intensity', '中等')
                if isinstance(stance_int, str):
                    stance_int = INTENSITY_MAP.get(stance_int, 0.5)
                    
                return ExpressionStrategy(
                    emotion=EmotionTone(data.get('emotion', 'neutral')),
                    emotion_intensity=float(emotion_int),
                    stance=Stance(data.get('stance', 'neutral')),
                    stance_intensity=float(stance_int),
                    style=ExpressionStyle(data.get('style', 'rational')),
                    narrative=NarrativeStrategy(data.get('narrative', 'fact_list'))
                )
            except (json.JSONDecodeError, ValueError, KeyError):
                pass
        
        return ExpressionStrategy(
            emotion=EmotionTone.NEUTRAL, emotion_intensity=0.5,
            stance=Stance.NEUTRAL, stance_intensity=0.5,
            style=ExpressionStyle.RATIONAL, narrative=NarrativeStrategy.FACT_LIST
        )
