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
        
        # 构建新信息文本（带时间戳，区分原发/转发/转发评论）
        new_info_parts = []
        if posts:
            new_info_parts.append("### 曝光的博文：")
            for i, post in enumerate(posts[:5], 1):
                post_time = post.get('time', '')
                time_prefix = f"[{post_time}] " if post_time else ""
                author = post.get('author', '未知')
                post_type = post.get('post_type', 'original')
                
                if post_type == 'repost':
                    original_author = post.get('original_author', '')
                    orig_content = post.get('original_content', '')
                    original_content = orig_content[:150] if orig_content else ''
                    new_info_parts.append(f"{i}. {time_prefix}@{author} 转发了博文（仅转发）：")
                    new_info_parts.append(f"   原博 @{original_author}：{original_content}")
                elif post_type == 'repost_comment':
                    original_author = post.get('original_author', '')
                    orig_content = post.get('original_content', '')
                    original_content = orig_content[:150] if orig_content else ''
                    rp_comment = post.get('repost_comment', post.get('content', ''))
                    repost_comment = rp_comment[:100] if rp_comment else ''
                    new_info_parts.append(f"{i}. {time_prefix}@{author} 转发了博文并评论：")
                    new_info_parts.append(f"   原博 @{original_author}：{original_content}")
                    new_info_parts.append(f"   转发评论：{repost_comment}")
                else:
                    content_raw = post.get('content', '')
                    content = content_raw[:200] if content_raw else ''
                    new_info_parts.append(f"{i}. {time_prefix}@{author} 发表了博文：{content}")
                
                new_info_parts.append(f"   [点赞:{post.get('likes', 0)} 转发:{post.get('reposts', 0)} 评论:{post.get('comments_count', 0)}]")
                if post.get('comments'):
                    new_info_parts.append("   热门评论：" + " | ".join(post['comments'][:3]))
        
        # 构建外部突发事件文本
        events_parts = []
        if events:
            events_parts.append("### 当前外部突发事件：")
            for evt in events[:3]:
                events_parts.append(f"- [{evt.get('time', '')}] {evt.get('content', '')}")
        
        # 构建记忆文本（带目标描述）
        memory_text = self._format_memories(memories)
        
        # 获取身份信息和其他信念状态
        identity_text = belief.identity.to_prompt_text()
        other_beliefs = self._format_other_beliefs(belief)
        
        prompt = prompts['update'].format(
            current_time=current_time,
            identity_text=identity_text,
            belief_text=other_beliefs,
            new_info="\n".join(new_info_parts) if new_info_parts else "无新信息",
            memories=memory_text if memory_text else "无历史记忆",
            external_events="\n".join(events_parts) if events_parts else "",
            event_background=event_background
        )
        return prompt + output_format
    
    def _format_memories(self, memories: List[Dict]) -> str:
        """格式化历史行为记忆（包含目标描述）"""
        if not memories:
            return ""
        lines = []
        for mem in memories[:5]:
            time_str = mem.get('time', '')
            action_type = mem.get('action_type', '')
            content = mem.get('content', '')
            target = mem.get('target', '')
            
            # 构建带目标的行为描述
            if action_type in ['repost', 'repost_comment']:
                if target:
                    desc = f"[{time_str}] 我转发了博文《{target[:30]}...》"
                    if content:
                        desc += f"，附带评论：{content[:50]}..."
                else:
                    desc = f"[{time_str}] 我转发了一条博文"
            elif action_type in ['short_comment', 'long_comment']:
                if target:
                    desc = f"[{time_str}] 我评论了博文《{target[:30]}...》：{content[:50]}..."
                else:
                    desc = f"[{time_str}] 我发表了评论：{content[:50]}..."
            elif action_type == 'like':
                if target:
                    desc = f"[{time_str}] 我点赞了博文《{target[:30]}...》"
                else:
                    desc = f"[{time_str}] 我点赞了一条博文"
            elif action_type in ['short_post', 'long_post']:
                desc = f"[{time_str}] 我发布了原创博文：{content[:50]}..."
            else:
                desc = f"[{time_str}] {content[:80]}..."
            lines.append(f"- {desc}")
        return "\n".join(lines)
    
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
