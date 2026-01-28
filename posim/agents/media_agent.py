"""
媒体智能体 - 新闻媒体账号
特点：专业客观、以发布新闻为主、注重公信力
"""
from typing import Dict, Any, List
from .base_agent import BaseAgent, AgentProfile


class MediaAgent(BaseAgent):
    """媒体智能体"""
    
    def __init__(self, profile: AgentProfile, belief_data: Dict[str, Any], 
                 api_pool, history_posts: List[Dict] = None, event_background: str = ""):
        profile.agent_type = 'media'
        super().__init__(profile, belief_data, api_pool, history_posts, event_background)
    
    def _get_max_actions(self) -> int:
        """媒体单次最多2个行为（通常只发一条新闻）"""
        return 2
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], api_pool, event_background: str = "") -> 'MediaAgent':
        """从字典创建媒体智能体"""
        profile = AgentProfile(
            user_id=data.get('user_id', ''),
            username=data.get('username', ''),
            agent_type='media',
            followers_count=data.get('followers_count', 100000),
            following_count=data.get('following_count', 0),
            posts_count=data.get('posts_count', 0),
            verified=True,
            description=data.get('description', '官方媒体账号')
        )
        return cls(
            profile=profile,
            belief_data=data.get('belief', data),
            api_pool=api_pool,
            history_posts=data.get('history_posts', []),
            event_background=event_background
        )
