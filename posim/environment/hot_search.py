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
        self.topics: Dict[str, TopicStats] = {}
        self.hot_list: List[Tuple[str, float]] = []  # (topic, score)
        self.last_update_time: str = ""
        
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
    
    def update_hot_list(self, current_time: str) -> List[Tuple[str, float]]:
        """更新热搜榜单"""
        # 检查是否需要更新
        if self.last_update_time and current_time:
            last_dt = datetime.fromisoformat(self.last_update_time)
            current_dt = datetime.fromisoformat(current_time)
            minutes_passed = (current_dt - last_dt).total_seconds() / 60
            if minutes_passed < self.update_interval:
                return self.hot_list
        
        # 计算所有话题的热度得分
        scored = []
        for topic, stats in self.topics.items():
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
            scored.append((topic, final_score))
        
        # 排序
        scored.sort(key=lambda x: x[1], reverse=True)
        self.hot_list = scored[:self.max_count]
        self.last_update_time = current_time
        
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
    
    def get_topic_stats(self, topic: str) -> TopicStats:
        """获取话题统计"""
        topic = topic.strip('#')
        return self.topics.get(topic)
    
    def clear_cold_topics(self, min_score: float = 1.0):
        """清理冷门话题"""
        to_remove = [t for t, s in self.topics.items() if s.heat_score < min_score]
        for t in to_remove:
            del self.topics[t]
