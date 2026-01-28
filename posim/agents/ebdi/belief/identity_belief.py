"""
角色身份信念 - 自然语言描述的个体基础认知
包括人口学属性、社交属性、身份属性等，从初始化数据载入
"""
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class IdentityBelief:
    """角色身份信念（自然语言描述）"""
    user_id: str
    description: str  # 自然语言形式的角色身份描述
    
    # 原始结构化数据（仅用于内部计算，不参与LLM推理）
    raw_data: Dict[str, Any] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IdentityBelief':
        """从字典创建身份信念"""
        return cls(
            user_id=data.get('user_id', ''),
            description=data.get('identity_description', ''),
            raw_data=data.get('raw_profile', {})
        )
    
    def to_prompt_text(self) -> str:
        """转换为提示词文本"""
        return f"【我的身份】\n{self.description}"
