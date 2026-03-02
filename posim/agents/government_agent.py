"""
政府智能体 - 政府官方账号
特点：语言正式规范、发布官方通报、引导舆论
"""
from typing import Dict, Any, List
from .base_agent import BaseAgent, AgentProfile


class GovernmentAgent(BaseAgent):
    """政府智能体"""
    
    def __init__(self, profile: AgentProfile, belief_data: Dict[str, Any], 
                 api_pool, history_posts: List[Dict] = None, event_background: str = ""):
        profile.agent_type = 'government'
        super().__init__(profile, belief_data, api_pool, history_posts, event_background)
    
    def _get_max_actions(self) -> int:
        """政府账号单次最大行为数（仅作为上限参考，实际由LLM根据事件热度自主决定）"""
        return 10
    
    async def publish_announcement(self, content: str, current_time: str) -> Dict[str, Any]:
        """发布官方通报（由事件队列触发）"""
        if self.is_banned:
            return None
        
        action_result = {
            'action_type': 'long_post',
            'user_id': self.profile.user_id,
            'username': self.profile.username,
            'content': content,
            'time': current_time,
            'topics': [],
            'is_official': True
        }
        
        # 记录到记忆
        self.memory.add(
            content=f"[{current_time}] 我发布了官方通报：{content[:100]}",
            memory_type='action',
            importance=0.9,
            metadata=action_result
        )
        
        return action_result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], api_pool, event_background: str = "") -> 'GovernmentAgent':
        """从字典创建政府智能体"""
        profile = AgentProfile(
            user_id=data.get('user_id', ''),
            username=data.get('username', ''),
            agent_type='government',
            followers_count=data.get('followers_count', 500000),
            following_count=data.get('following_count', 0),
            posts_count=data.get('posts_count', 0),
            verified=True,
            description=data.get('description', '政府官方账号')
        )
        return cls(
            profile=profile,
            belief_data=data.get('belief', data),
            api_pool=api_pool,
            history_posts=data.get('history_posts', []),
            event_background=event_background
        )
