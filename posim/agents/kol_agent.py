"""
意见领袖智能体 - KOL/大V
特点：高影响力、注重个人品牌、更倾向发布原创内容
"""
from typing import Dict, Any, List
from .base_agent import BaseAgent, AgentProfile


class KOLAgent(BaseAgent):
    """意见领袖智能体"""
    
    def __init__(self, profile: AgentProfile, belief_data: Dict[str, Any], 
                 api_pool, history_posts: List[Dict] = None, event_background: str = ""):
        profile.agent_type = 'kol'
        super().__init__(profile, belief_data, api_pool, history_posts, event_background)
        # KOL的活跃度得分更高
        self.activity_score = min(1.0, self.activity_score * 1.5)
    
    def _get_max_actions(self) -> int:
        """KOL单次最大行为数（仅作为上限参考，实际由LLM根据事件热度自主决定）"""
        return 10
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], api_pool, event_background: str = "") -> 'KOLAgent':
        """从字典创建KOL智能体"""
        profile = AgentProfile(
            user_id=data.get('user_id', ''),
            username=data.get('username', ''),
            agent_type='kol',
            followers_count=data.get('followers_count', 10000),
            following_count=data.get('following_count', 0),
            posts_count=data.get('posts_count', 0),
            verified=data.get('verified', True),
            description=data.get('description', '')
        )
        return cls(
            profile=profile,
            belief_data=data.get('belief', data),
            api_pool=api_pool,
            history_posts=data.get('history_posts', []),
            event_background=event_background
        )
