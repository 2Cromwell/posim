"""
心理认知信念 - 网民固有的心理认知特征
5大典型心理：自我实现、猎奇探究、减压宣泄、仇官仇富、跟风从众
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class PsychologicalBelief:
    """心理认知信念"""
    user_id: str
    belief_items: List[str] = field(default_factory=list)  # 心理信念条目列表
    
    # 心理类型权重（用于行为倾向计算）
    psychology_weights: Dict[str, float] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PsychologicalBelief':
        """从字典创建心理信念"""
        return cls(
            user_id=data.get('user_id', ''),
            belief_items=data.get('psychological_beliefs', []),
            psychology_weights=data.get('psychology_weights', {})
        )
    
    def to_prompt_text(self) -> str:
        """转换为提示词文本"""
        if not self.belief_items:
            return ""
        items_text = "\n".join([f"- {item}" for item in self.belief_items])
        return f"【我的心理认知】\n{items_text}"
