"""
提示词加载器 - 统一加载各类智能体的提示词模板
"""
from typing import Dict


class PromptLoader:
    """提示词加载器 - 根据智能体类型加载对应的提示词"""
    
    @classmethod
    def get_belief_prompts(cls, agent_type: str) -> Dict[str, str]:
        """获取信念系统提示词"""
        if agent_type == 'kol':
            from .kol_prompts.belief_prompts import KOL_BELIEF_UPDATE_PROMPT
            return {'update': KOL_BELIEF_UPDATE_PROMPT}
        elif agent_type == 'media':
            from .media_prompts.belief_prompts import MEDIA_BELIEF_UPDATE_PROMPT
            return {'update': MEDIA_BELIEF_UPDATE_PROMPT}
        elif agent_type == 'government':
            from .government_prompts.belief_prompts import GOVERNMENT_BELIEF_UPDATE_PROMPT
            return {'update': GOVERNMENT_BELIEF_UPDATE_PROMPT}
        else:
            from .citizen_prompts.belief_prompts import CITIZEN_BELIEF_UPDATE_PROMPT
            return {'update': CITIZEN_BELIEF_UPDATE_PROMPT}
    
    @classmethod
    def get_desire_prompts(cls, agent_type: str) -> Dict[str, str]:
        """获取欲望系统提示词"""
        if agent_type == 'kol':
            from .kol_prompts.desire_prompts import KOL_DESIRE_PROMPT
            return {'desire': KOL_DESIRE_PROMPT}
        elif agent_type == 'media':
            from .media_prompts.desire_prompts import MEDIA_DESIRE_PROMPT
            return {'desire': MEDIA_DESIRE_PROMPT}
        elif agent_type == 'government':
            from .government_prompts.desire_prompts import GOVERNMENT_DESIRE_PROMPT
            return {'desire': GOVERNMENT_DESIRE_PROMPT}
        else:
            from .citizen_prompts.desire_prompts import CITIZEN_DESIRE_PROMPT
            return {'desire': CITIZEN_DESIRE_PROMPT}
    
    @classmethod
    def get_intention_prompts(cls, agent_type: str) -> Dict[str, str]:
        """获取意图系统提示词（统一COT格式）"""
        if agent_type == 'kol':
            from .kol_prompts.intention_prompts import KOL_INTENTION_PROMPT
            return {'intention': KOL_INTENTION_PROMPT}
        elif agent_type == 'media':
            from .media_prompts.intention_prompts import MEDIA_INTENTION_PROMPT
            return {'intention': MEDIA_INTENTION_PROMPT}
        elif agent_type == 'government':
            from .government_prompts.intention_prompts import GOVERNMENT_INTENTION_PROMPT
            return {'intention': GOVERNMENT_INTENTION_PROMPT}
        else:
            from .citizen_prompts.intention_prompts import CITIZEN_INTENTION_PROMPT
            return {'intention': CITIZEN_INTENTION_PROMPT}
    
    @classmethod
    def get_output_format(cls, module: str) -> str:
        """获取JSON输出格式模板（已整合到各提示词文件中，返回空字符串）"""
        return ''
