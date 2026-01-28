"""
意图系统主类 - 通过单次LLM调用使用三级COT生成行为决策列表
"""
import json
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
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


# 强度等级到数值的映射
INTENSITY_MAP = {
    '极低': 0.1, '低': 0.3, '中等': 0.5, '高': 0.7, '极高': 0.9
}


@dataclass
class IntentionResult:
    """意图执行结果（包含行为、策略、内容）"""
    action_type: str
    target_post_id: Optional[str] = None
    target_post_index: int = 0
    target_author: Optional[str] = None
    target_content: Optional[str] = None
    # 表达策略
    emotion: str = "neutral"
    emotion_intensity: float = 0.5
    stance: str = "neutral"
    stance_intensity: float = 0.5
    style: str = "rational"
    narrative: str = "fact_list"
    # 生成内容
    text: str = ""
    topics: List[str] = field(default_factory=list)
    mentions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'action_type': self.action_type,
            'target_post_id': self.target_post_id,
            'target_post_index': self.target_post_index,
            'target_author': self.target_author,
            'target_content': self.target_content,
            'emotion': self.emotion,
            'emotion_intensity': self.emotion_intensity,
            'stance': self.stance,
            'stance_intensity': self.stance_intensity,
            'style': self.style,
            'narrative': self.narrative,
            'text': self.text,
            'topics': self.topics,
            'mentions': self.mentions
        }


class IntentionSystem:
    """意图系统 - 单次LLM调用，三级COT生成行为决策列表"""
    
    def __init__(self, api_pool):
        self.api_pool = api_pool
    
    async def generate_intentions(self, belief_text: str, desires: List[Dict], 
                                  exposed_posts: List[Dict], 
                                  agent_type: str = 'citizen',
                                  max_actions: int = 3,
                                  current_time: str = None,
                                  event_background: str = "",
                                  external_events: List[Dict] = None) -> List[IntentionResult]:
        """
        通过单次LLM调用生成多个行为决策
        Args:
            belief_text: 信念文本
            desires: 欲望列表
            exposed_posts: 曝光博文
            agent_type: 智能体类型
            max_actions: 最大行为数（仅作参考，实际由LLM决定）
            current_time: 当前仿真时间
        """
        if current_time is None:
            current_time = datetime.now().strftime('%Y-%m-%dT%H:%M')
        
        # 构建提示词
        prompt = self._build_intention_prompt(belief_text, desires, exposed_posts, agent_type, current_time, external_events or [], event_background)
        
        # 单次LLM调用（无需system prompt，已合并到主提示词中）
        response = await self.api_pool.async_text_query(prompt, "", purpose='action')
        
        # 解析结果
        intentions = self._parse_intentions(response, exposed_posts)
        
        return intentions
    
    def _build_intention_prompt(self, belief_text: str, desires: List[Dict], 
                                 posts: List[Dict], agent_type: str, current_time: str,
                                 external_events: List[Dict] = None, event_background: str = "") -> str:
        """构建意图决策提示词"""
        prompts = PromptLoader.get_intention_prompts(agent_type)
        output_format = PromptLoader.get_output_format('intention')
        
        # 构建欲望文本
        desires_text = "\n".join([f"- {d.get('type', '')}: {d.get('description', '')}" for d in desires])
        
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
        if external_events:
            events_parts.append("### 当前外部突发事件：")
            for evt in external_events[:3]:
                events_parts.append(f"- [{evt.get('time', '')}] {evt.get('content', '')}")
        
        prompt = prompts['intention'].format(
            current_time=current_time,
            belief_text=belief_text,
            desires=desires_text if desires_text else "无明确欲望",
            exposed_posts="\n".join(exposed_parts) if exposed_parts else "无可互动博文",
            external_events="\n".join(events_parts) if events_parts else "",
            event_background=event_background
        )
        return prompt + output_format
    
    def _parse_intentions(self, response: str, posts: List[Dict]) -> List[IntentionResult]:
        """解析LLM返回的行为决策列表（支持中文和英文字段）"""
        intentions = []
        
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                # 支持中文和英文字段名
                actions_data = data.get('行为列表', data.get('actions', []))
                
                for action_data in actions_data:
                    intention = self._parse_single_action(action_data, posts)
                    if intention:
                        intentions.append(intention)
            except json.JSONDecodeError:
                pass
        
        # 如果解析失败，返回默认行为
        if not intentions and posts:
            intentions.append(IntentionResult(
                action_type='like',
                target_post_id=posts[0].get('id'),
                target_post_index=1,
                target_author=posts[0].get('author'),
                target_content=posts[0].get('content')
            ))
        
        return intentions
    
    # 中文行为类型到英文的映射
    ACTION_TYPE_MAP = {
        '点赞': 'like', '转发': 'repost', '转发并评论': 'repost_comment',
        '短评论': 'short_comment', '长评论': 'long_comment',
        '短博文': 'short_post', '长博文': 'long_post'
    }
    
    # 中文情绪到英文的映射
    EMOTION_MAP = {
        '愤怒': 'anger', '厌恶': 'disgust', '焦虑': 'anxiety', '悲伤': 'sadness',
        '幸灾乐祸': 'schadenfreude', '兴奋': 'excitement', '中性': 'neutral',
        '快乐': 'happiness', '恐惧': 'fear', '惊讶': 'surprise'
    }
    
    # 中文立场到英文的映射
    STANCE_MAP = {'支持': 'support', '反对': 'oppose', '中立': 'neutral'}
    
    # 中文风格到英文的映射
    STYLE_MAP = {
        '阴阳怪气': 'yinyang', '讽刺': 'sarcastic', '激进': 'aggressive',
        '嘲讽': 'mocking', '情绪宣泄': 'emotional_vent', '质疑': 'questioning',
        '共情': 'empathetic', '冷漠': 'indifferent', '理性': 'rational'
    }
    
    # 中文叙事策略到英文的映射
    NARRATIVE_MAP = {
        '贴标签': 'labeling', '道德绑架': 'moral_kidnapping', '阴谋论': 'conspiracy',
        '转移话题': 'whataboutism', '号召行动': 'call_to_action', '人身攻击': 'personal_attack',
        '质疑事实': 'fact_questioning', '事实陈述': 'fact_list', '叙事': 'storytelling',
        '引用权威': 'cite_authority'
    }
    
    def _parse_single_action(self, action_data: Dict, posts: List[Dict]) -> Optional[IntentionResult]:
        """解析单个行为决策（支持中文和英文字段）"""
        try:
            # 解析行为类型（支持中英文）
            action_type_raw = action_data.get('行为类型', action_data.get('action_type', '点赞'))
            action_type = self.ACTION_TYPE_MAP.get(action_type_raw, action_type_raw)
            
            # 解析目标博文序号
            target_idx = action_data.get('目标博文序号', action_data.get('target_post_index', 1)) - 1
            
            # 获取目标博文信息
            target_post = posts[target_idx] if 0 <= target_idx < len(posts) else None
            
            # 解析策略（点赞/转发无策略）
            strategy = action_data.get('表达策略', action_data.get('strategy', {}))
            
            # 解析情绪
            emotion_raw = strategy.get('情绪类型', strategy.get('emotion', '中性'))
            emotion = self.EMOTION_MAP.get(emotion_raw, emotion_raw)
            
            # 解析情绪强度
            emotion_int = strategy.get('情绪强度', strategy.get('emotion_intensity', '中等'))
            if isinstance(emotion_int, str):
                emotion_int = INTENSITY_MAP.get(emotion_int, 0.5)
            
            # 解析立场
            stance_raw = strategy.get('立场', strategy.get('stance', '中立'))
            stance = self.STANCE_MAP.get(stance_raw, stance_raw)
            
            # 解析立场强度
            stance_int = strategy.get('立场强度', strategy.get('stance_intensity', '中等'))
            if isinstance(stance_int, str):
                stance_int = INTENSITY_MAP.get(stance_int, 0.5)
            
            # 解析风格
            style_raw = strategy.get('表达风格', strategy.get('style', '理性'))
            style = self.STYLE_MAP.get(style_raw, style_raw)
            
            # 解析叙事策略
            narrative_raw = strategy.get('叙事策略', strategy.get('narrative', '事实陈述'))
            narrative = self.NARRATIVE_MAP.get(narrative_raw, narrative_raw)
            
            # 解析内容（点赞/转发无内容）
            content = action_data.get('内容', action_data.get('content', {}))
            text = content.get('文本', content.get('text', ''))
            topics = content.get('话题', content.get('topics', []))
            mentions = content.get('提及用户', content.get('mentions', []))
            
            return IntentionResult(
                action_type=action_type,
                target_post_id=target_post.get('id') if target_post else None,
                target_post_index=action_data.get('目标博文序号', action_data.get('target_post_index', 0)),
                target_author=target_post.get('author') if target_post else None,
                target_content=target_post.get('content') if target_post else None,
                emotion=emotion,
                emotion_intensity=float(emotion_int),
                stance=stance,
                stance_intensity=float(stance_int),
                style=style,
                narrative=narrative,
                text=text,
                topics=topics,
                mentions=mentions
            )
        except (ValueError, KeyError, TypeError):
            return None
