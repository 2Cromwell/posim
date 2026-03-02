"""欲望类型定义"""
from enum import Enum
from dataclasses import dataclass
from typing import Optional


class DesireType(Enum):
    SELF_EXPRESSION = "自我表达"
    SOCIAL_APPROVAL = "社会认可"
    INFORMATION_SEEKING = "信息获取"
    EMOTIONAL_RELEASE = "情绪宣泄"
    SOCIAL_CONNECTION = "社交联结"
    JUSTICE_ADVOCACY = "正义倡导"
    ENTERTAINMENT = "娱乐消遣"
    INFLUENCE_OTHERS = "影响他人"


@dataclass
class Desire:
    type: DesireType
    weight: float
    target: Optional[str] = None
    description: str = ""

    def to_dict(self):
        return {
            'type': self.type.value,
            'weight': self.weight,
            'target': self.target,
            'description': self.description
        }
