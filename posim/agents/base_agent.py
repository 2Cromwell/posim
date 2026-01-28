"""
基础智能体抽象类 - 定义元认知智能体的通用接口
感知 → 信念 → 欲望 → 意图 → 行为
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from .ebdi.belief.belief_system import BeliefSystem
from .ebdi.belief.belief_updater import BeliefUpdater
from .ebdi.desire.desire_system import DesireSystem
from .ebdi.intention.intention_system import IntentionSystem, IntentionResult
from .ebdi.memory.stream_memory import StreamMemory
from .ebdi.memory.memory_retrieval import MemoryRetrieval


@dataclass
class AgentProfile:
    """智能体基础信息"""
    user_id: str
    username: str
    agent_type: str  # citizen/kol/media/government
    followers_count: int = 0
    following_count: int = 0
    posts_count: int = 0
    verified: bool = False
    description: str = ""


class BaseAgent(ABC):
    """基础智能体抽象类"""
    
    def __init__(self, profile: AgentProfile, belief_data: Dict[str, Any], 
                 api_pool, history_posts: List[Dict] = None, event_background: str = ""):
        self.profile = profile
        self.api_pool = api_pool
        self.agent_type = profile.agent_type
        self.event_background = event_background  # 事件背景，用于注入系统提示词
        
        # 初始化EBDI系统
        belief_data['user_id'] = profile.user_id
        self.belief_system = BeliefSystem.from_dict(belief_data)
        self.belief_updater = BeliefUpdater(api_pool)
        self.desire_system = DesireSystem(api_pool)
        self.intention_system = IntentionSystem(api_pool)
        
        # 初始化记忆系统
        self.memory = StreamMemory(profile.user_id)
        self.memory_retrieval = MemoryRetrieval(api_pool)
        
        # 加载历史博文到记忆
        if history_posts:
            for post in history_posts[-20:]:  # 最近20条
                self.memory.add(
                    content=f"[{post.get('time', '')}] 我发布了：{post.get('content', '')}",
                    memory_type='action',
                    importance=0.6,
                    metadata=post,
                    timestamp=post.get('time', '')
                )
        
        # 活跃度（用于霍克斯过程采样）
        self.activity_score = self._calculate_activity_score()
        
        # 状态标志
        self.is_banned = False
        self.last_action_time = None
    
    def _calculate_activity_score(self) -> float:
        """计算活跃度得分"""
        posts_weight = 0.6
        influence_weight = 0.4
        posts_score = min(1.0, self.profile.posts_count / 1000)
        influence_score = min(1.0, self.profile.followers_count / 100000)
        return posts_weight * posts_score + influence_weight * influence_score
    
    async def perceive_and_act(self, exposed_posts: List[Dict], external_events: List[Dict],
                               current_time: str) -> List[IntentionResult]:
        """
        感知环境并产生行为
        核心认知流程：感知 → 信念更新 → 欲望生成 → 意图决策 → 行为输出
        """
        if self.is_banned:
            return []
        
        # 1. 检索相关记忆
        query = self._build_memory_query(exposed_posts, external_events)
        relevant_memories = self.memory_retrieval.retrieve_by_recency_and_importance(self.memory, top_k=5)
        memories_dict = self.memory.to_dict_list(relevant_memories)
        
        # 2. 信念推断（情绪衰减 + 外部信息影响）
        self.belief_system.decay_emotion()
        await self.belief_updater.update_belief(
            self.belief_system, exposed_posts, external_events, memories_dict, 
            self.agent_type, current_time, self.event_background
        )
        
        # 3. 生成欲望集
        belief_text = self.belief_system.to_prompt_text()
        desires = await self.desire_system.generate_desires(
            belief_text, exposed_posts, external_events, self.agent_type, current_time,
            self.event_background
        )
        desires_dict = [d.to_dict() for d in desires]
        
        # 4. 生成意图和行为
        intentions = await self.intention_system.generate_intentions(
            belief_text, desires_dict, exposed_posts, self.agent_type, 
            max_actions=self._get_max_actions(), current_time=current_time,
            event_background=self.event_background, external_events=external_events
        )
        
        # 5. 记录行为到记忆
        for intention in intentions:
            self._record_action_to_memory(intention, current_time)
        
        self.last_action_time = current_time
        return intentions
    
    def _build_memory_query(self, posts: List[Dict], events: List[Dict]) -> str:
        """构建记忆查询"""
        parts = []
        if posts:
            content = posts[0].get('content', '')
            parts.append(content[:200] if content else '')
        if events:
            event_content = events[0].get('content', '')
            parts.append(event_content[:200] if event_content else '')
        return " ".join(parts) if parts else "最近的社交活动"
    
    def _record_action_to_memory(self, intention: IntentionResult, current_time: str):
        """记录行为到记忆（包含目标描述）"""
        action_desc = f"[{current_time}] 我{self._get_action_desc(intention)}"
        importance = self._get_action_importance(intention.action_type)
        # 添加目标信息到metadata
        metadata = intention.to_dict()
        metadata['time'] = current_time
        target_content = intention.target_content or ''
        metadata['target'] = target_content[:100] if target_content else ''
        self.memory.add(content=action_desc, memory_type='action', importance=importance,
                       metadata=metadata, timestamp=current_time)
                       
    def random_decision(self, exposed_posts: List[Dict], external_events: List[Dict],
                        current_time: str) -> List[IntentionResult]:
        """随机决策（非LLM驱动）- 基于身份的差异化行为分布"""
        import random
        actions = []
        
        # 基于身份的行为分布配置 {action_type: weight}
        # citizen: 主要互动(点赞/评论)，较少原创
        # kol: 原创多，转发评论活跃
        # media: 以原创发布为主
        # government: 主要发布官方信息
        behavior_profiles = {
            'citizen': {
                'like': 0.35, 'repost': 0.20, 'repost_comment': 0.10,
                'short_comment': 0.25, 'long_comment': 0.02,
                'short_post': 0.05, 'long_post': 0.01, 'idle': 0.02
            },
            'kol': {
                'like': 0.15, 'repost': 0.25, 'repost_comment': 0.20,
                'short_comment': 0.15, 'long_comment': 0.05,
                'short_post': 0.12, 'long_post': 0.06, 'idle': 0.02
            },
            'media': {
                'like': 0.05, 'repost': 0.15, 'repost_comment': 0.10,
                'short_comment': 0.05, 'long_comment': 0.05,
                'short_post': 0.35, 'long_post': 0.23, 'idle': 0.02
            },
            'government': {
                'like': 0.02, 'repost': 0.10, 'repost_comment': 0.05,
                'short_comment': 0.03, 'long_comment': 0.05,
                'short_post': 0.30, 'long_post': 0.40, 'idle': 0.05
            }
        }
        
        # 情绪分布（基于外部事件情绪偏向）
        # 默认分布，会根据事件动态调整
        base_emotions = {
            '愤怒': 0.15, '悲伤': 0.10, '惊奇': 0.12, '恐惧': 0.05,
            '喜悦': 0.08, '厌恶': 0.10, '中性': 0.40
        }
        
        # 根据外部事件调整情绪分布
        emotion_weights = base_emotions.copy()
        if external_events:
            # 事件会放大非中性情绪
            event_boost = min(0.3, len(external_events) * 0.1)
            emotion_weights['中性'] = max(0.1, emotion_weights['中性'] - event_boost)
            for emo in ['愤怒', '惊奇', '悲伤']:
                emotion_weights[emo] = emotion_weights.get(emo, 0.1) + event_boost / 3
        
        # 获取当前身份的行为分布
        profile = behavior_profiles.get(self.agent_type, behavior_profiles['citizen'])
        action_types = list(profile.keys())
        weights = list(profile.values())
        
        # 每次激活1-3次决策
        num_decisions = random.choices([1, 2, 3], weights=[0.5, 0.35, 0.15])[0]
        
        for _ in range(num_decisions):
            action_type = random.choices(action_types, weights=weights)[0]
            if action_type == 'idle':
                continue
            
            # 采样情绪
            emotions = list(emotion_weights.keys())
            emo_weights = list(emotion_weights.values())
            emotion = random.choices(emotions, weights=emo_weights)[0]
            
            # 根据情绪调整立场 (-1到1)
            stance_map = {'愤怒': -0.7, '悲伤': -0.3, '惊奇': 0.0, '恐惧': -0.4,
                         '喜悦': 0.5, '厌恶': -0.6, '中性': 0.0}
            base_stance = stance_map.get(emotion, 0.0)
            stance = base_stance + random.uniform(-0.2, 0.2)
            stance = max(-1.0, min(1.0, stance))
            
            # 需要目标帖子的行为（互动类）
            needs_target = action_type in ['like', 'repost', 'repost_comment', 'short_comment', 'long_comment']
            
            if needs_target and exposed_posts:
                # 有目标帖子可供互动
                target_post = random.choice(exposed_posts)
                text = ''
                if action_type in ['short_comment', 'long_comment', 'repost_comment']:
                    comment_templates = {
                        '愤怒': ['太过分了！', '这也太离谱了', '必须严查！', '不能容忍'],
                        '悲伤': ['唉...', '太遗憾了', '心痛', '难过'],
                        '惊奇': ['震惊！', '不敢相信', '真的假的？', '太意外了'],
                        '恐惧': ['细思极恐', '害怕', '太可怕了'],
                        '喜悦': ['哈哈哈', '太好了', '支持！', '点赞'],
                        '厌恶': ['无语', '恶心', '看不下去'],
                        '中性': ['转发', '了解了', '关注中', '看看']
                    }
                    templates = comment_templates.get(emotion, comment_templates['中性'])
                    text = random.choice(templates)
                
                # 安全获取字符串内容，避免索引越界
                content = target_post.get('content', '')
                safe_content = content[:100] if content else ''
                author = target_post.get('author', '')
                
                actions.append(IntentionResult(
                    action_type=action_type,
                    target_post_id=target_post.get('id', ''),
                    target_author=author,
                    target_content=safe_content,
                    text=text,
                    topics=target_post.get('topics', [])[:2] if target_post.get('topics') else [],
                    mentions=[author] if author else [],
                    emotion=emotion, stance=stance
                ))
            else:
                # 无目标帖子或本身是发帖类行为 -> 发帖
                if needs_target:
                    # 降级：互动类变为发帖类
                    action_type = 'short_post' if random.random() < 0.7 else 'long_post'
                # 原创发帖
                post_templates = {
                    '愤怒': ['这件事必须要有个说法！', '强烈谴责这种行为', '请相关部门介入调查'],
                    '悲伤': ['为什么会这样...', '太让人痛心了', '希望事情能有转机'],
                    '惊奇': ['刚看到这个消息，太震惊了', '这个瓜也太大了', '不敢相信这是真的'],
                    '恐惧': ['有点担心事态发展', '希望不要扩大化'],
                    '喜悦': ['好消息！', '终于等到了', '太棒了'],
                    '厌恶': ['又是这种事', '真是够了'],
                    '中性': ['关注此事进展', '持续跟进中', '记录一下']
                }
                templates = post_templates.get(emotion, post_templates['中性'])
                text = random.choice(templates)
                
                actions.append(IntentionResult(
                    action_type=action_type,
                    target_post_id='', target_author='', target_content='',
                    text=text,
                    topics=[], mentions=[],
                    emotion=emotion, stance=stance
                ))
        
        return actions
    
    def _get_action_desc(self, intention: IntentionResult) -> str:
        """获取行为描述"""
        action_type = intention.action_type
        if action_type == 'like':
            return f"点赞了@{intention.target_author}的博文"
        elif action_type == 'repost':
            return f"转发了@{intention.target_author}的博文"
        elif action_type == 'repost_comment':
            text = intention.text or ''
            return f"转发并评论了：{text[:100] if text else ''}"
        elif action_type in ['short_comment', 'long_comment']:
            text = intention.text or ''
            return f"评论了：{text[:100] if text else ''}"
        else:
            text = intention.text or ''
            return f"发布了：{text[:100] if text else ''}"
    
    def _get_action_importance(self, action_type: str) -> float:
        """获取行为重要性"""
        return {'like': 0.3, 'repost': 0.7, 'repost_comment': 0.8, 
                'short_comment': 0.5, 'long_comment': 0.6, 
                'short_post': 0.7, 'long_post': 0.8}.get(action_type, 0.5)
    
    @abstractmethod
    def _get_max_actions(self) -> int:
        """获取单次最大行为数（不同智能体类型不同）"""
        pass
    
    def ban(self):
        """禁言"""
        self.is_banned = True
    
    def unban(self):
        """解禁"""
        self.is_banned = False
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化"""
        return {
            'profile': {
                'user_id': self.profile.user_id,
                'username': self.profile.username,
                'agent_type': self.agent_type,
                'followers_count': self.profile.followers_count,
                'following_count': self.profile.following_count,
                'posts_count': self.profile.posts_count,
                'verified': self.profile.verified,
                'description': self.profile.description
            },
            'belief': self.belief_system.to_dict(),
            'activity_score': self.activity_score,
            'is_banned': self.is_banned
        }
