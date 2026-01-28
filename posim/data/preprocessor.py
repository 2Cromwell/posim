"""
数据预处理 - 处理原始数据为仿真可用格式
"""
from typing import List, Dict, Any
import re


def extract_topics(text: str) -> List[str]:
    """从文本中提取话题标签"""
    pattern = r'#([^#]+)#'
    return re.findall(pattern, text)


def extract_mentions(text: str) -> List[str]:
    """从文本中提取@用户"""
    pattern = r'@(\w+)'
    return re.findall(pattern, text)


def classify_post_length(content: str) -> str:
    """分类博文长度"""
    length = len(content)
    if length < 50:
        return 'short'
    elif length < 150:
        return 'medium'
    else:
        return 'long'


def normalize_emotion_vector(emotion_dict: Dict[str, float]) -> List[float]:
    """标准化情绪向量"""
    emotions = ['happiness', 'sadness', 'anger', 'fear', 'surprise', 'disgust']
    vector = [emotion_dict.get(e, 0.0) for e in emotions]
    total = sum(vector)
    if total > 0:
        vector = [v / total for v in vector]
    return vector


def preprocess_posts(posts: List[Dict]) -> List[Dict]:
    """预处理博文列表"""
    processed = []
    for post in posts:
        content = post.get('content', '')
        processed.append({
            **post,
            'topics': extract_topics(content),
            'mentions': extract_mentions(content),
            'length_type': classify_post_length(content)
        })
    return processed


def preprocess_user_history(history_posts: List[Dict]) -> List[Dict]:
    """预处理用户历史博文"""
    return preprocess_posts(history_posts)


def aggregate_user_emotions(posts: List[Dict]) -> Dict[str, float]:
    """从历史博文聚合用户情绪倾向"""
    emotion_counts = {'happiness': 0, 'sadness': 0, 'anger': 0, 
                     'fear': 0, 'surprise': 0, 'disgust': 0}
    
    # 简单的关键词匹配（实际可用更复杂的情感分析）
    keywords = {
        'happiness': ['开心', '高兴', '太好了', '哈哈', '棒'],
        'sadness': ['难过', '伤心', '唉', '遗憾', '可惜'],
        'anger': ['愤怒', '气愤', '可恶', '太过分', '无耻'],
        'fear': ['害怕', '担心', '恐惧', '可怕', '危险'],
        'surprise': ['惊讶', '没想到', '震惊', '意外', '居然'],
        'disgust': ['恶心', '讨厌', '无语', '呵呵', '服了']
    }
    
    for post in posts:
        content = post.get('content', '')
        for emotion, kws in keywords.items():
            for kw in kws:
                if kw in content:
                    emotion_counts[emotion] += 1
    
    return emotion_counts
