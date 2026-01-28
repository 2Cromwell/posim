"""
欲望类型定义 - 社交媒体用户的行为动机
"""
from enum import Enum
from dataclasses import dataclass
from typing import Optional


class DesireType(Enum):
    """欲望类型枚举"""
    SELF_EXPRESSION = "自我表达"       # 自我表达 - 分享观点、展示态度
    SOCIAL_APPROVAL = "社会认可"       # 社会认可 - 获得点赞、关注
    INFORMATION_SEEKING = "信息获取"  # 信息获取 - 了解真相
    EMOTIONAL_RELEASE = "情绪宣泄"   # 情绪宣泄 - 减压发泄
    SOCIAL_CONNECTION = "社交联结"   # 社交联结 - 与他人互动
    JUSTICE_ADVOCACY = "正义倡导"     # 正义倡导 - 伸张正义、维权
    ENTERTAINMENT = "娱乐消遣"           # 娱乐消遣 - 看热闹
    INFLUENCE_OTHERS = "影响他人"     # 影响他人 - 意见领袖动机


DESIRE_CN = {
    DesireType.SELF_EXPRESSION: "自我表达",
    DesireType.SOCIAL_APPROVAL: "社会认可", 
    DesireType.INFORMATION_SEEKING: "信息获取",
    DesireType.EMOTIONAL_RELEASE: "情绪宣泄",
    DesireType.SOCIAL_CONNECTION: "社交联结",
    DesireType.JUSTICE_ADVOCACY: "正义倡导",
    DesireType.ENTERTAINMENT: "娱乐消遣",
    DesireType.INFLUENCE_OTHERS: "影响他人"
}


@dataclass
class Desire:
    """单个欲望"""
    type: DesireType
    weight: float  # 权重 0-1
    target: Optional[str] = None  # 作用对象
    description: str = ""  # 具体描述
    
    def to_dict(self):
        return {
            'type': self.type.value,
            'weight': self.weight,
            'target': self.target,
            'description': self.description
        }
