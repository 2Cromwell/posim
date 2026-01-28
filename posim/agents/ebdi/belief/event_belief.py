"""
事件观点信念 - 对特定舆情事件的事实、背景、观点认知
以<时间, 主体, 观点, 原因>形式组织
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any
from datetime import datetime


@dataclass
class EventOpinion:
    """单条事件观点"""
    time: str
    subject: str  # 涉事主体
    opinion: str  # 观点倾向
    reason: str   # 原因


@dataclass
class EventBelief:
    """事件观点信念"""
    user_id: str
    opinions: List[EventOpinion] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EventBelief':
        """从字典创建事件观点信念"""
        opinions = []
        for op in data.get('event_opinions', []):
            opinions.append(EventOpinion(
                time=op.get('time', ''),
                subject=op.get('subject', ''),
                opinion=op.get('opinion', ''),
                reason=op.get('reason', '')
            ))
        return cls(user_id=data.get('user_id', ''), opinions=opinions)
    
    def add_opinion(self, subject: str, opinion: str, reason: str, time: str = None):
        """添加新的观点"""
        if time is None:
            time = datetime.now().strftime('%Y-%m-%dT%H:%M')
        self.opinions.append(EventOpinion(time=time, subject=subject, opinion=opinion, reason=reason))
    
    def update_opinion(self, subject: str, new_opinion: str, new_reason: str, time: str = None):
        """更新对某主体的观点"""
        if time is None:
            time = datetime.now().strftime('%Y-%m-%dT%H:%M')
        for op in self.opinions:
            if op.subject == subject:
                op.opinion = new_opinion
                op.reason = new_reason
                op.time = time
                return
        self.add_opinion(subject, new_opinion, new_reason, time)
    
    def to_prompt_text(self) -> str:
        """转换为提示词文本"""
        if not self.opinions:
            return ""
        lines = ["【我对事件的观点】"]
        for op in self.opinions:
            time_str = f"[{op.time}] " if op.time else ""
            lines.append(f"- {time_str}关于{op.subject}：{op.opinion}")
        return "\n".join(lines)
