"""
评估指标计算 - 宏观和微观评估指标
"""
import numpy as np
from typing import List, Dict, Any
from collections import Counter
from datetime import datetime


def calculate_behavior_distribution(actions: List[Dict]) -> Dict[str, float]:
    """计算行为类型分布"""
    if not actions:
        return {}
    action_types = [a.get('action_type', 'unknown') for a in actions]
    counter = Counter(action_types)
    total = len(action_types)
    return {k: v / total for k, v in counter.items()}


def calculate_emotion_distribution(actions: List[Dict]) -> Dict[str, float]:
    """计算情绪分布"""
    if not actions:
        return {}
    emotions = [a.get('emotion', 'neutral') for a in actions]
    counter = Counter(emotions)
    total = len(emotions)
    return {k: v / total for k, v in counter.items()}


def calculate_stance_distribution(actions: List[Dict]) -> Dict[str, float]:
    """计算立场分布"""
    if not actions:
        return {}
    stances = [a.get('stance', 'neutral') for a in actions]
    counter = Counter(stances)
    total = len(stances)
    return {k: v / total for k, v in counter.items()}


def calculate_polarization_index(actions: List[Dict]) -> float:
    """
    计算意见极化指数
    基于立场分布的方差，值越大表示极化程度越高
    """
    stance_dist = calculate_stance_distribution(actions)
    support = stance_dist.get('support', 0)
    oppose = stance_dist.get('oppose', 0)
    neutral = stance_dist.get('neutral', 0)
    
    # 极化指数：支持和反对的比例差距
    if support + oppose == 0:
        return 0.0
    polarization = abs(support - oppose) / (support + oppose)
    # 同时考虑中立比例，中立越少极化越高
    return polarization * (1 - neutral * 0.5)


def calculate_activity_curve(actions: List[Dict], interval_minutes: int = 60) -> List[Dict]:
    """计算活跃度曲线"""
    if not actions:
        return []
    
    # 按时间分组
    time_counts = {}
    for action in actions:
        time_str = action.get('time', '')
        if not time_str:
            continue
        try:
            dt = datetime.fromisoformat(time_str)
            bucket = dt.replace(minute=(dt.minute // interval_minutes) * interval_minutes, second=0, microsecond=0)
            bucket_str = bucket.isoformat()
            time_counts[bucket_str] = time_counts.get(bucket_str, 0) + 1
        except:
            continue
    
    # 排序并返回
    return [{'time': k, 'count': v} for k, v in sorted(time_counts.items())]


def calculate_topic_evolution(actions: List[Dict]) -> Dict[str, List[Dict]]:
    """计算话题演化"""
    topic_timeline = {}
    
    for action in actions:
        topics = action.get('topics', [])
        time_str = action.get('time', '')
        if not time_str:
            continue
        
        for topic in topics:
            topic = topic.strip('#')
            if not topic:
                continue
            if topic not in topic_timeline:
                topic_timeline[topic] = []
            topic_timeline[topic].append({'time': time_str, 'action': action.get('action_type', '')})
    
    return topic_timeline


def calculate_user_engagement(actions: List[Dict]) -> Dict[str, Dict]:
    """计算用户参与度"""
    user_stats = {}
    
    for action in actions:
        user_id = action.get('user_id', '')
        if not user_id:
            continue
        
        if user_id not in user_stats:
            user_stats[user_id] = {
                'action_count': 0,
                'action_types': Counter(),
                'emotions': Counter()
            }
        
        user_stats[user_id]['action_count'] += 1
        user_stats[user_id]['action_types'][action.get('action_type', '')] += 1
        user_stats[user_id]['emotions'][action.get('emotion', '')] += 1
    
    return user_stats


def calculate_cascade_depth(actions: List[Dict]) -> Dict[str, int]:
    """计算传播级联深度"""
    # 简化版本：基于转发链计算
    post_chains = {}
    
    for action in actions:
        if action.get('action_type') in ['repost', 'repost_comment']:
            target = action.get('target_post_id', '')
            if target:
                post_chains[target] = post_chains.get(target, 0) + 1
    
    return post_chains


def calculate_all_metrics(actions: List[Dict]) -> Dict[str, Any]:
    """计算所有评估指标"""
    return {
        'behavior_distribution': calculate_behavior_distribution(actions),
        'emotion_distribution': calculate_emotion_distribution(actions),
        'stance_distribution': calculate_stance_distribution(actions),
        'polarization_index': calculate_polarization_index(actions),
        'activity_curve': calculate_activity_curve(actions),
        'total_actions': len(actions),
        'unique_users': len(set(a.get('user_id', '') for a in actions))
    }
