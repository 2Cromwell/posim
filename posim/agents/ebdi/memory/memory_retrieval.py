"""记忆检索 - 基于新近性、重要性和多样性（MMR）"""
import numpy as np
from datetime import datetime
from typing import List
from .stream_memory import StreamMemory, MemoryItem

DIVERSITY_LAMBDA = 0.3   # MMR多样性权重


class MemoryRetrieval:

    def __init__(self, api_pool=None):
        self.api_pool = api_pool
        self._current_time: str = None

    def set_current_time(self, current_time: str):
        self._current_time = current_time

    def retrieve_by_recency_and_importance(self, memory: StreamMemory,
                                           top_k: int = 5,
                                           current_time: str = None) -> List[MemoryItem]:
        if not memory.memories:
            return []
        # 计算基础分数
        scored = []
        for mem in memory.memories:
            recency = self._calculate_recency(mem.timestamp, current_time)
            scored.append((mem, 0.6 * recency + 0.4 * mem.importance))
        scored.sort(key=lambda x: x[1], reverse=True)

        # 候选集取 top_k*3，用 MMR 贪心选出 top_k 保证多样性
        candidates = scored[:top_k * 3]
        if not any(m.embedding is not None for m, _ in candidates):
            return [m for m, _ in candidates[:top_k]]

        selected: List[MemoryItem] = []
        remaining = list(candidates)
        for _ in range(min(top_k, len(remaining))):
            best_idx, best_score = 0, -1e9
            for i, (mem, base) in enumerate(remaining):
                if not selected or mem.embedding is None:
                    mmr = base
                else:
                    max_sim = max(self._cosine(mem.embedding, s.embedding)
                                  for s in selected if s.embedding is not None)
                    mmr = (1 - DIVERSITY_LAMBDA) * base - DIVERSITY_LAMBDA * max_sim
                if mmr > best_score:
                    best_score, best_idx = mmr, i
            selected.append(remaining.pop(best_idx)[0])
        return selected

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0

    def _calculate_recency(self, timestamp: str, current_time: str = None) -> float:
        if not timestamp:
            return 0.5
        mem_time = datetime.fromisoformat(timestamp)
        now_str = current_time or self._current_time
        now = datetime.fromisoformat(now_str) if now_str else datetime.now()
        hours_ago = (now - mem_time).total_seconds() / 3600
        return float(np.exp(-hours_ago / 24))
