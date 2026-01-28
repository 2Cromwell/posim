"""
信念系统主类 - 整合四类信念
"""
from dataclasses import dataclass
from typing import Dict, Any
from .identity_belief import IdentityBelief
from .psychological_belief import PsychologicalBelief
from .event_belief import EventBelief
from .emotion_belief import EmotionBelief


@dataclass
class BeliefSystem:
    """信念系统 - 整合角色身份、心理认知、事件观点、情绪激发四类信念"""
    user_id: str
    identity: IdentityBelief
    psychology: PsychologicalBelief
    event: EventBelief
    emotion: EmotionBelief
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BeliefSystem':
        """从字典创建完整信念系统"""
        user_id = data.get('user_id', '')
        return cls(
            user_id=user_id,
            identity=IdentityBelief.from_dict(data),
            psychology=PsychologicalBelief.from_dict(data),
            event=EventBelief.from_dict(data),
            emotion=EmotionBelief.from_dict(data)
        )
    
    def decay_emotion(self, rate: float = 0.95):
        """情绪衰减"""
        self.emotion.decay(rate)
    
    def to_prompt_text(self) -> str:
        """将完整信念系统转换为提示词文本"""
        parts = [
            self.identity.to_prompt_text(),
            self.psychology.to_prompt_text(),
            self.event.to_prompt_text(),
            self.emotion.to_prompt_text()
        ]
        return "\n\n".join([p for p in parts if p])
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            'user_id': self.user_id,
            'identity_description': self.identity.description,
            'raw_profile': self.identity.raw_data,
            'psychological_beliefs': self.psychology.belief_items,
            'psychology_weights': self.psychology.psychology_weights,
            'event_opinions': [
                {'time': op.time, 'subject': op.subject, 'opinion': op.opinion, 'reason': op.reason}
                for op in self.event.opinions
            ],
            'emotion_vector': self.emotion.emotion_vector.tolist(),
            'last_update': self.emotion.last_update
        }
