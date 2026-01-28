"""
普通网民智能体 - 社交媒体普通用户
特点：行为轻量化、情绪化程度较高、易受热点影响
"""
from typing import Dict, Any, List
from .base_agent import BaseAgent, AgentProfile


class CitizenAgent(BaseAgent):
    """普通网民智能体"""
    
    def __init__(self, profile: AgentProfile, belief_data: Dict[str, Any], 
                 api_pool, history_posts: List[Dict] = None, event_background: str = ""):
        profile.agent_type = 'citizen'
        super().__init__(profile, belief_data, api_pool, history_posts, event_background)
    
    def _get_max_actions(self) -> int:
        """普通网民单次最多2个行为"""
        return 5
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], api_pool, event_background: str = "") -> 'CitizenAgent':
        """从字典创建普通网民智能体"""
        profile = AgentProfile(
            user_id=data.get('user_id', ''),
            username=data.get('username', ''),
            agent_type='citizen',
            followers_count=data.get('followers_count', 0),
            following_count=data.get('following_count', 0),
            posts_count=data.get('posts_count', 0),
            verified=data.get('verified', False),
            description=data.get('description', '')
        )
        return cls(
            profile=profile,
            belief_data=data.get('belief', data),
            api_pool=api_pool,
            history_posts=data.get('history_posts', []),
            event_background=event_background
        )
