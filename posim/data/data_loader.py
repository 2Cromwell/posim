"""
数据加载器 - 加载用户数据、事件队列等
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional


class DataLoader:
    """数据加载器"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
    
    def load_users(self, filename: str = "users.json") -> List[Dict[str, Any]]:
        """加载用户数据"""
        filepath = self.data_dir / filename
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_events(self, filename: str = "events.json") -> List[Dict[str, Any]]:
        """加载事件队列"""
        filepath = self.data_dir / filename
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_initial_posts(self, filename: str = "initial_posts.json", 
                           before_time: str = None) -> List[Dict[str, Any]]:
        """加载初始博文（仅加载指定时间之前的博文）
        Args:
            filename: 博文文件名
            before_time: 截止时间，只加载此时间之前的博文（ISO格式）
        """
        filepath = self.data_dir / filename
        if not filepath.exists():
            return []
        with open(filepath, 'r', encoding='utf-8') as f:
            posts = json.load(f)
        
        if before_time:
            posts = [p for p in posts if p.get('time', '') < before_time]
        return posts
    
    def load_social_relations(self, filename: str = "relations.json") -> List[Dict[str, Any]]:
        """加载社交关系"""
        filepath = self.data_dir / filename
        if not filepath.exists():
            return []
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_all(self) -> Dict[str, Any]:
        """加载所有数据"""
        return {
            'users': self.load_users(),
            'events': self.load_events(),
            'initial_posts': self.load_initial_posts(),
            'relations': self.load_social_relations()
        }


def parse_user_data(raw_user: Dict) -> Dict[str, Any]:
    """解析用户数据为智能体格式"""
    # 处理emotion_vector：支持dict或list格式
    emotion_vector = raw_user.get('emotion_vector', {})
    if isinstance(emotion_vector, list):
        # 旧格式：列表 -> 转换为字典
        emotion_keys = ['happiness', 'sadness', 'anger', 'fear', 'surprise', 'disgust']
        emotion_vector = dict(zip(emotion_keys, emotion_vector + [0.0] * (6 - len(emotion_vector))))
    elif not isinstance(emotion_vector, dict):
        emotion_vector = {'happiness': 0.0, 'sadness': 0.0, 'anger': 0.0, 
                          'fear': 0.0, 'surprise': 0.0, 'disgust': 0.0}
    
    return {
        'user_id': raw_user.get('user_id', ''),
        'username': raw_user.get('username', ''),
        'agent_type': raw_user.get('agent_type', 'citizen'),
        'followers_count': raw_user.get('followers_count', 0),
        'following_count': raw_user.get('following_count', 0),
        'posts_count': raw_user.get('posts_count', 0),
        'verified': raw_user.get('verified', False),
        'description': raw_user.get('description', ''),
        'belief': {
            'user_id': raw_user.get('user_id', ''),
            'identity_description': raw_user.get('identity_description', ''),
            'raw_profile': raw_user.get('raw_profile', {}),
            'psychological_beliefs': raw_user.get('psychological_beliefs', []),
            'event_opinions': raw_user.get('event_opinions', []),
            'emotion_vector': emotion_vector,
            'last_update': ''
        },
        'history_posts': raw_user.get('history_posts', [])
    }
