"""
流式记忆库 - 存储智能体的历史行为和观察
支持时间衰减和相似性检索
"""
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class MemoryItem:
    """记忆条目"""
    id: str
    timestamp: str
    content: str
    memory_type: str  # action/observation/reflection
    importance: float = 0.5
    embedding: np.ndarray = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class StreamMemory:
    """流式记忆库"""
    
    def __init__(self, user_id: str, max_size: int = 1000, decay_rate: float = 0.995):
        self.user_id = user_id
        self.max_size = max_size
        self.decay_rate = decay_rate
        self.memories: List[MemoryItem] = []
        self._id_counter = 0
    
    def add(self, content: str, memory_type: str = 'action', 
            importance: float = 0.5, embedding: np.ndarray = None,
            metadata: Dict[str, Any] = None, timestamp: str = None) -> MemoryItem:
        """添加记忆"""
        self._id_counter += 1
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M')
        item = MemoryItem(
            id=f"{self.user_id}_{self._id_counter}",
            timestamp=timestamp,
            content=content,
            memory_type=memory_type,
            importance=importance,
            embedding=embedding,
            metadata=metadata or {}
        )
        self.memories.append(item)
        
        # 超出容量时删除最老的低重要性记忆
        if len(self.memories) > self.max_size:
            self.memories.sort(key=lambda x: x.importance, reverse=True)
            self.memories = self.memories[:self.max_size]
        
        return item
    
    def decay_all(self):
        """对所有记忆进行时间衰减"""
        for mem in self.memories:
            mem.importance *= self.decay_rate
    
    def get_recent(self, n: int = 10, memory_type: str = None) -> List[MemoryItem]:
        """获取最近的n条记忆"""
        filtered = self.memories if not memory_type else [m for m in self.memories if m.memory_type == memory_type]
        return sorted(filtered, key=lambda x: x.timestamp, reverse=True)[:n]
    
    def to_dict_list(self, items: List[MemoryItem] = None) -> List[Dict]:
        """转换为字典列表"""
        items = items or self.memories
        return [{'content': m.content, 'type': m.memory_type, 'time': m.timestamp, 
                 'importance': m.importance} for m in items]
    
    def clear(self):
        """清空记忆"""
        self.memories.clear()
        self._id_counter = 0
