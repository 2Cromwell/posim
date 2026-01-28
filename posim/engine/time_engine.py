"""
时间引擎 - 分钟级仿真时间管理
支持事件驱动的高效模拟
"""
from datetime import datetime, timedelta
from typing import Callable, List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class TimeState:
    """时间状态"""
    current_time: datetime
    start_time: datetime
    end_time: datetime
    step: int = 0
    granularity_minutes: int = 1


class TimeEngine:
    """时间引擎"""
    
    def __init__(self, start_time: str, end_time: str, granularity: int = 1):
        """
        Args:
            start_time: 开始时间 (ISO格式)
            end_time: 结束时间
            granularity: 时间粒度（分钟）
        """
        self.state = TimeState(
            current_time=datetime.fromisoformat(start_time),
            start_time=datetime.fromisoformat(start_time),
            end_time=datetime.fromisoformat(end_time),
            granularity_minutes=granularity
        )
        self._callbacks: List[Callable] = []
    
    @property
    def current_time(self) -> datetime:
        return self.state.current_time
    
    @property
    def current_time_str(self) -> str:
        return self.state.current_time.strftime('%Y-%m-%dT%H:%M')
    
    @property
    def elapsed_minutes(self) -> int:
        return int((self.state.current_time - self.state.start_time).total_seconds() / 60)
    
    @property
    def remaining_minutes(self) -> int:
        return int((self.state.end_time - self.state.current_time).total_seconds() / 60)
    
    @property
    def progress(self) -> float:
        total = (self.state.end_time - self.state.start_time).total_seconds()
        elapsed = (self.state.current_time - self.state.start_time).total_seconds()
        return elapsed / total if total > 0 else 1.0
    
    def is_finished(self) -> bool:
        return self.state.current_time >= self.state.end_time
    
    def advance(self, minutes: int = None) -> bool:
        """推进时间"""
        if self.is_finished():
            return False
        
        minutes = minutes or self.state.granularity_minutes
        self.state.current_time += timedelta(minutes=minutes)
        self.state.step += 1
        
        # 不超过结束时间
        if self.state.current_time > self.state.end_time:
            self.state.current_time = self.state.end_time
        
        return True
    
    def jump_to(self, target_time: str) -> bool:
        """跳转到指定时间"""
        target = datetime.fromisoformat(target_time)
        if target < self.state.start_time or target > self.state.end_time:
            return False
        self.state.current_time = target
        return True
    
    def reset(self):
        """重置到起始时间"""
        self.state.current_time = self.state.start_time
        self.state.step = 0
    
    def get_time_window(self, window_minutes: int) -> tuple:
        """获取时间窗口"""
        end = self.state.current_time
        start = end - timedelta(minutes=window_minutes)
        return start.strftime('%Y-%m-%dT%H:%M'), end.strftime('%Y-%m-%dT%H:%M')
    
    def should_update_hot_search(self, interval: int = 15) -> bool:
        """检查是否应该更新热搜"""
        return self.elapsed_minutes % interval == 0
    
    def should_save_checkpoint(self, interval: int = 10) -> bool:
        """检查是否应该保存检查点"""
        return self.elapsed_minutes % interval == 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'current_time': self.current_time_str,
            'start_time': self.state.start_time.strftime('%Y-%m-%dT%H:%M'),
            'end_time': self.state.end_time.strftime('%Y-%m-%dT%H:%M'),
            'step': self.state.step,
            'granularity': self.state.granularity_minutes,
            'progress': self.progress
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TimeEngine':
        engine = cls(data['start_time'], data['end_time'], data.get('granularity', 1))
        engine.state.current_time = datetime.fromisoformat(data['current_time'])
        engine.state.step = data.get('step', 0)
        return engine
