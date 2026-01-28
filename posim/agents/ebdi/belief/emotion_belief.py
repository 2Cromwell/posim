"""
情绪激发信念 - 基于Ekman六大基本情感的情绪状态
快乐、悲伤、愤怒、恐惧、惊讶、厌恶
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any
from datetime import datetime


EMOTIONS = ['happiness', 'sadness', 'anger', 'fear', 'surprise', 'disgust']
EMOTION_CN = {'happiness': '快乐', 'sadness': '悲伤', 'anger': '愤怒', 
              'fear': '恐惧', 'surprise': '惊讶', 'disgust': '厌恶'}


@dataclass
class EmotionBelief:
    """情绪激发信念"""
    user_id: str
    emotion_vector: np.ndarray = field(default_factory=lambda: np.zeros(6))  # 六维情绪向量
    last_update: str = ''
    neighbor_emotions: Dict[str, np.ndarray] = field(default_factory=dict)  # 邻居情绪状态
    
    def __post_init__(self):
        if isinstance(self.emotion_vector, dict):
            # 新格式：字典 -> 转换为数组
            self.emotion_vector = np.array([self.emotion_vector.get(e, 0.0) for e in EMOTIONS])
        elif isinstance(self.emotion_vector, list):
            self.emotion_vector = np.array(self.emotion_vector)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmotionBelief':
        """从字典创建情绪信念"""
        emotion_data = data.get('emotion_vector', {})
        if isinstance(emotion_data, dict):
            emotion_vector = np.array([emotion_data.get(e, 0.0) for e in EMOTIONS])
        elif isinstance(emotion_data, list):
            emotion_vector = np.array(emotion_data + [0.0] * (6 - len(emotion_data)))
        else:
            emotion_vector = np.zeros(6)
        return cls(
            user_id=data.get('user_id', ''),
            emotion_vector=emotion_vector,
            last_update=data.get('last_update', '')
        )
    
    def decay(self, decay_rate: float = 0.95):
        """情绪衰减"""
        self.emotion_vector *= decay_rate
    
    def update_from_content(self, emotion_scores: Dict[str, float], current_time: str = None):
        """根据内容分析结果更新情绪"""
        for i, emo in enumerate(EMOTIONS):
            if emo in emotion_scores:
                self.emotion_vector[i] = min(1.0, self.emotion_vector[i] + emotion_scores[emo] * 0.3)
        if current_time:
            self.last_update = current_time
        else:
            self.last_update = datetime.now().strftime('%Y-%m-%dT%H:%M')
    
    def update_from_neighbors(self, neighbor_emotions: List[np.ndarray], influence_rate: float = 0.1, 
                              current_time: str = None):
        """根据邻居情绪更新（情绪传染）"""
        if neighbor_emotions:
            avg_neighbor = np.mean(neighbor_emotions, axis=0)
            self.emotion_vector = (1 - influence_rate) * self.emotion_vector + influence_rate * avg_neighbor
            if current_time:
                self.last_update = current_time
    
    def get_dominant_emotion(self) -> str:
        """获取主导情绪"""
        idx = np.argmax(self.emotion_vector)
        return EMOTIONS[idx] if self.emotion_vector[idx] > 0.1 else 'neutral'
    
    def get_emotion_intensity(self) -> float:
        """获取情绪强度"""
        return float(np.max(self.emotion_vector))
    
    def to_prompt_text(self) -> str:
        """转换为提示词文本 - 直接输出情绪向量"""
        vector_str = ", ".join([f"{EMOTION_CN[e]}:{self.emotion_vector[i]:.2f}" for i, e in enumerate(EMOTIONS)])
        return f"【当前情绪向量】[{vector_str}]（取值0-1，0=无，1=极强）"
