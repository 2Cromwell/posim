"""
社交媒体热搜榜单 - 基于热度机制的话题排名
每15分钟更新一次
"""
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field


@dataclass
class TopicStats:
    """话题统计"""
    topic: str
    mentions: int = 0
    likes: int = 0
    reposts: int = 0
    comments: int = 0
    first_appear: str = ""
    last_update: str = ""
    heat_score: float = 0.0


class HotSearchManager:
    """热搜榜单管理器"""
    
    def __init__(self, config):
        self.update_interval = config.hot_search_update_interval  # 更新间隔（分钟）
        self.max_count = config.hot_search_count
        self.min_mentions = getattr(config, 'hot_search_min_mentions', 10)  # 上榜最小提及次数
        self.display_count = getattr(config, 'hot_search_display_count', 5)  # 传递给智能体的数量
        self.topics: Dict[str, TopicStats] = {}
        self.hot_list: List[Tuple[str, float]] = []  # (topic, score)
        self.last_update_time: str = ""
        
        # 热搜历史记录 [{time, topics: [{topic, mentions, score}]}]
        self.history: List[Dict[str, Any]] = []
        
        # 热度计算权重
        self.mention_weight = 1.0
        self.like_weight = 0.5
        self.repost_weight = 2.0
        self.comment_weight = 1.5
        self.decay_rate = 0.95
    
    def add_topic_mention(self, topic: str, current_time: str, 
                         likes: int = 0, reposts: int = 0, comments: int = 0):
        """记录话题提及"""
        topic = topic.strip('#')
        if not topic:
            return
        
        if topic not in self.topics:
            self.topics[topic] = TopicStats(
                topic=topic,
                first_appear=current_time,
                last_update=current_time
            )
        
        stats = self.topics[topic]
        stats.mentions += 1
        stats.likes += likes
        stats.reposts += reposts
        stats.comments += comments
        stats.last_update = current_time
    
    def update_hot_list(self, current_time: str, force: bool = False) -> List[Tuple[str, float]]:
        """更新热搜榜单"""
        # 检查是否需要更新
        if not force and self.last_update_time and current_time:
            last_dt = datetime.fromisoformat(self.last_update_time)
            current_dt = datetime.fromisoformat(current_time)
            minutes_passed = (current_dt - last_dt).total_seconds() / 60
            if minutes_passed < self.update_interval:
                return self.hot_list
        
        # 计算所有话题的热度得分（需满足最小提及次数）
        scored = []
        for topic, stats in self.topics.items():
            # 检查是否满足最小提及次数阈值
            if stats.mentions < self.min_mentions:
                continue
            
            # 基础热度
            base_score = (
                stats.mentions * self.mention_weight +
                stats.likes * self.like_weight +
                stats.reposts * self.repost_weight +
                stats.comments * self.comment_weight
            )
            
            # 时间衰减
            if stats.last_update and current_time:
                last_dt = datetime.fromisoformat(stats.last_update)
                current_dt = datetime.fromisoformat(current_time)
                hours_ago = (current_dt - last_dt).total_seconds() / 3600
                decay = np.exp(-hours_ago / 12)  # 12小时衰减
            else:
                decay = 0.5
            
            final_score = base_score * decay
            stats.heat_score = final_score
            scored.append((topic, final_score, stats.mentions))
        
        # 排序
        scored.sort(key=lambda x: x[1], reverse=True)
        self.hot_list = [(t, s) for t, s, _ in scored[:self.max_count]]
        self.last_update_time = current_time
        
        # 记录历史
        if scored:
            self.history.append({
                'time': current_time,
                'topics': [{'topic': t, 'mentions': m, 'score': s} 
                          for t, s, m in scored[:self.max_count]]
            })
        
        # 应用衰减
        for stats in self.topics.values():
            stats.mentions = int(stats.mentions * self.decay_rate)
            stats.likes = int(stats.likes * self.decay_rate)
            stats.reposts = int(stats.reposts * self.decay_rate)
            stats.comments = int(stats.comments * self.decay_rate)
        
        return self.hot_list
    
    def get_hot_list(self, count: int = 10) -> List[Dict[str, Any]]:
        """获取热搜榜"""
        result = []
        for i, (topic, score) in enumerate(self.hot_list[:count], 1):
            stats = self.topics.get(topic)
            result.append({
                'rank': i,
                'topic': f"#{topic}#",
                'heat_score': score,
                'mentions': stats.mentions if stats else 0
            })
        return result
    
    def get_top_topics_text(self, count: int = None) -> str:
        """获取热搜话题文本（用于提示词）"""
        count = count or self.display_count
        if not self.hot_list:
            return ""
        topics = [f"#{t}#" for t, _ in self.hot_list[:count]]
        return "、".join(topics)
    
    def get_history(self) -> List[Dict[str, Any]]:
        return self.history
