"""
行为选择器 - 选择行为类别和目标对象
7种通用行为：like, repost, repost_comment, short_comment, long_comment, short_post, long_post
"""
import json
import re
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
from posim.prompts.prompt_loader import PromptLoader


class ActionType(Enum):
    LIKE = "like"
    REPOST = "repost"
    REPOST_COMMENT = "repost_comment"
    SHORT_COMMENT = "short_comment"
    LONG_COMMENT = "long_comment"
    SHORT_POST = "short_post"
    LONG_POST = "long_post"


ACTION_CN = {
    ActionType.LIKE: "点赞",
    ActionType.REPOST: "直接转发",
    ActionType.REPOST_COMMENT: "转发带评论",
    ActionType.SHORT_COMMENT: "短评论",
    ActionType.LONG_COMMENT: "长评论",
    ActionType.SHORT_POST: "短原发",
    ActionType.LONG_POST: "长原发"
}


@dataclass
class ActionDecision:
    """行为决策"""
    action_type: ActionType
    target_post_id: Optional[str] = None  # 目标博文ID（原发时为None）
    target_author: Optional[str] = None
    target_content: Optional[str] = None
    topic: Optional[str] = None  # 话题标签（原发时使用）


class ActionSelector:
    """行为选择器"""
    
    def __init__(self, api_pool):
        self.api_pool = api_pool
    
    async def select_action(self, belief_text: str, desires: List[Dict], 
                           exposed_posts: List[Dict], agent_type: str = 'citizen',
                           current_time: str = None) -> ActionDecision:
        """选择行为类别和目标"""
        if current_time is None:
            current_time = datetime.now().isoformat()
        prompt = self._build_action_prompt(belief_text, desires, exposed_posts, agent_type, current_time)
        system_prompt = self._get_system_prompt(agent_type)
        
        response = await self.api_pool.async_text_query(prompt, system_prompt, purpose='action')
        return self._parse_action(response, exposed_posts)
    
    def _get_system_prompt(self, agent_type: str) -> str:
        prompts = PromptLoader.get_intention_prompts(agent_type)
        return prompts.get('action_system', '')
    
    def _build_action_prompt(self, belief_text: str, desires: List[Dict], 
                             posts: List[Dict], agent_type: str, current_time: str) -> str:
        prompts = PromptLoader.get_intention_prompts(agent_type)
        output_format = PromptLoader.get_output_format('action')
        
        # 构建欲望文本
        desires_text = "\n".join([f"- {d.get('type', '')}: {d.get('description', '')}" for d in desires])
        
        # 构建博文文本（带时间戳）
        posts_parts = []
        for i, post in enumerate(posts[:5], 1):
            post_time = post.get('time', '')
            time_prefix = f"[{post_time}] " if post_time else ""
            posts_parts.append(f"{i}. {time_prefix}@{post.get('author', '')}：{post.get('content', '')[:100]}...")
            posts_parts.append(f"   [点赞:{post.get('likes', 0)} 转发:{post.get('reposts', 0)}]")
        
        prompt = prompts['action'].format(
            current_time=current_time,
            belief_text=belief_text,
            desires=desires_text if desires_text else "无明确欲望",
            exposed_posts="\n".join(posts_parts) if posts_parts else "无可互动博文"
        )
        return prompt + output_format
    
    def _parse_action(self, response: str, posts: List[Dict]) -> ActionDecision:
        """解析行为决策"""
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                action_type = ActionType(data.get('action_type', 'like'))
                target_idx = data.get('target_post_index', 1) - 1
                
                target_post = posts[target_idx] if 0 <= target_idx < len(posts) else (posts[0] if posts else None)
                
                return ActionDecision(
                    action_type=action_type,
                    target_post_id=target_post.get('id') if target_post else None,
                    target_author=target_post.get('author') if target_post else None,
                    target_content=target_post.get('content') if target_post else None,
                    topic=data.get('topic')
                )
            except (json.JSONDecodeError, ValueError, KeyError):
                pass
        
        # 默认点赞第一条
        if posts:
            return ActionDecision(
                action_type=ActionType.LIKE,
                target_post_id=posts[0].get('id'),
                target_author=posts[0].get('author'),
                target_content=posts[0].get('content')
            )
        return ActionDecision(action_type=ActionType.SHORT_POST, topic="#热点话题#")
