"""
外部环境事件队列 - 管理外部不可预测的事件刺激
事件格式：<time, type, source, content, influence>
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class EventType(Enum):
    """事件类型"""
    NODE_POST = "node_post"      # 单点博文发布事件
    GLOBAL_BROADCAST = "global_broadcast"  # 全局广播事件
    BREAKING_NEWS = "breaking_news"  # 突发新闻


@dataclass
class ExternalEvent:
    """外部事件"""
    time: str
    event_type: EventType
    source: List[str]  # 事件源节点列表
    content: str
    influence: float  # 影响力（用于霍克斯过程）
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'time': self.time,
            'type': self.event_type.value,
            'source': self.source,
            'content': self.content,
            'influence': self.influence,
            'metadata': self.metadata or {}
        }


class EventQueue:
    """事件队列管理器"""
    
    def __init__(self):
        self.events: List[ExternalEvent] = []
        self._event_index = 0
    
    def load_events(self, events_data: List[Dict]):
        """从数据加载事件"""
        for evt in events_data:
            type_str = evt.get('type', 'global_broadcast')
            event_type = EventType.GLOBAL_BROADCAST
            for et in EventType:
                if et.value == type_str:
                    event_type = et
                    break
            
            self.events.append(ExternalEvent(
                time=evt.get('time', ''),
                event_type=event_type,
                source=evt.get('source', []) if isinstance(evt.get('source'), list) else [evt.get('source', '')],
                content=evt.get('content', ''),
                influence=float(evt.get('influence', 1.0)),
                metadata=evt.get('metadata', {})
            ))
        
        # 按时间排序
        self.events.sort(key=lambda x: x.time)
    
    def add_event(self, event: ExternalEvent):
        """添加事件"""
        self.events.append(event)
        self.events.sort(key=lambda x: x.time)
    
    def get_current_events(self, current_time: str, window_size: int = 3) -> List[ExternalEvent]:
        """
        获取当前时间窗口内的事件
        返回当前事件（如果有）以及之前的window_size条事件
        """
        current_events = []
        past_events = []
        
        if not current_time:
            return []
        current_dt = datetime.fromisoformat(current_time)
        
        for evt in self.events:
            if not evt.time:
                continue
            evt_dt = datetime.fromisoformat(evt.time)
            if evt_dt <= current_dt:
                if evt_dt == current_dt or (current_dt - evt_dt).total_seconds() < 60:
                    current_events.append(evt)
                else:
                    past_events.append(evt)
        
        # 返回当前事件和最近的past事件
        result = current_events + past_events[-window_size:]
        return result
    
    def get_events_by_type(self, event_type: EventType, 
                           start_time: str = None, end_time: str = None) -> List[ExternalEvent]:
        """按类型获取事件"""
        filtered = [e for e in self.events if e.event_type == event_type]
        
        if start_time:
            start_dt = datetime.fromisoformat(start_time)
            filtered = [e for e in filtered if datetime.fromisoformat(e.time) >= start_dt]
        
        if end_time:
            end_dt = datetime.fromisoformat(end_time)
            filtered = [e for e in filtered if datetime.fromisoformat(e.time) <= end_dt]
        
        return filtered
    
    def get_node_events(self, node_id: str, current_time: str) -> List[ExternalEvent]:
        """获取指定节点需要处理的事件"""
        result = []
        if not current_time:
            return []
        current_dt = datetime.fromisoformat(current_time)
        
        for evt in self.events:
            if evt.event_type == EventType.NODE_POST and node_id in evt.source:
                if not evt.time:
                    continue
                evt_dt = datetime.fromisoformat(evt.time)
                if evt_dt <= current_dt and (current_dt - evt_dt).total_seconds() < 60:
                    result.append(evt)
        
        return result
    
    def get_total_influence(self, current_time: str) -> float:
        """获取当前时间点的总影响力（用于霍克斯过程）"""
        events = self.get_current_events(current_time)
        return sum(e.influence for e in events)
    
    def to_dict_list(self) -> List[Dict]:
        """转换为字典列表"""
        return [e.to_dict() for e in self.events]
