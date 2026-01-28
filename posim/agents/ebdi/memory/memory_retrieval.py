"""
记忆检索 - 基于内容相似性检索相关记忆
"""
import numpy as np
from typing import List, Dict, Any, Optional
from .stream_memory import StreamMemory, MemoryItem


class MemoryRetrieval:
    """记忆检索器"""
    
    def __init__(self, api_pool=None):
        self.api_pool = api_pool
        self._current_time: str = None  # 仿真时间缓存
    
    def set_current_time(self, current_time: str):
        """设置当前仿真时间（用于新近性计算）"""
        self._current_time = current_time
    
    def retrieve_by_similarity(self, memory: StreamMemory, query: str, 
                               top_k: int = 5, threshold: float = 0.3,
                               current_time: str = None) -> List[MemoryItem]:
        """基于语义相似度检索记忆"""
        if not memory.memories or not self.api_pool:
            return memory.get_recent(top_k)
        
        # 获取有embedding的记忆
        memories_with_emb = [m for m in memory.memories if m.embedding is not None]
        if not memories_with_emb:
            return memory.get_recent(top_k)
        
        # 编码查询
        query_emb = self.api_pool.encode([query])[0]
        
        # 计算相似度
        scores = []
        for mem in memories_with_emb:
            sim = np.dot(query_emb, mem.embedding) / (np.linalg.norm(query_emb) * np.linalg.norm(mem.embedding) + 1e-8)
            # 结合重要性和新近性
            recency = self._calculate_recency(mem.timestamp, current_time)
            combined_score = 0.5 * sim + 0.3 * mem.importance + 0.2 * recency
            scores.append((mem, combined_score))
        
        # 排序并筛选
        scores.sort(key=lambda x: x[1], reverse=True)
        return [m for m, s in scores[:top_k] if s >= threshold]
    
    def retrieve_by_recency_and_importance(self, memory: StreamMemory, 
                                           top_k: int = 5,
                                           current_time: str = None) -> List[MemoryItem]:
        """基于时间新近性和重要性检索"""
        if not memory.memories:
            return []
        
        scored = []
        for mem in memory.memories:
            recency = self._calculate_recency(mem.timestamp, current_time)
            score = 0.6 * recency + 0.4 * mem.importance
            scored.append((mem, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in scored[:top_k]]
    
    def _calculate_recency(self, timestamp: str, current_time: str = None) -> float:
        """计算新近性得分（0-1），越新越高"""
        from datetime import datetime
        if not timestamp:
            return 0.5
        mem_time = datetime.fromisoformat(timestamp)
        # 优先使用传入的时间，其次使用缓存的仿真时间，最后使用系统时间
        if current_time:
            now = datetime.fromisoformat(current_time)
        elif self._current_time:
            now = datetime.fromisoformat(self._current_time)
        else:
            now = datetime.now()
        hours_ago = (now - mem_time).total_seconds() / 3600
        return np.exp(-hours_ago / 24)  # 24小时衰减
    
    def add_with_embedding(self, memory: StreamMemory, content: str, 
                          memory_type: str = 'action', importance: float = 0.5,
                          metadata: Dict[str, Any] = None) -> MemoryItem:
        """添加带embedding的记忆"""
        embedding = None
        if self.api_pool:
            embedding = self.api_pool.encode([content])[0]
        return memory.add(content, memory_type, importance, embedding, metadata)
