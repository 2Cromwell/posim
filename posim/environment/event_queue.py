"""外部环境事件队列"""
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class EventType(Enum):
    NODE_POST = "node_post"
    GLOBAL_BROADCAST = "global_broadcast"
    BREAKING_NEWS = "breaking_news"


@dataclass
class ExternalEvent:
    time: str
    event_type: EventType
    source: List[str]
    content: str
    metadata: Dict[str, Any] = None
    influence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'time': self.time, 'type': self.event_type.value,
            'source': self.source, 'content': self.content,
            'metadata': self.metadata or {}, 'influence': self.influence
        }


_EVENT_TYPE_MAP = {et.value: et for et in EventType}


class EventQueue:

    def __init__(self):
        self.events: List[ExternalEvent] = []

    def load_events(self, events_data: List[Dict]):
        for evt in events_data:
            event_type = _EVENT_TYPE_MAP.get(evt.get('type', ''), EventType.GLOBAL_BROADCAST)
            influence = max(0.0, min(1.0, float(evt.get('influence', 1.0))))
            src = evt.get('source', [])
            self.events.append(ExternalEvent(
                time=evt.get('time', ''),
                event_type=event_type,
                source=src if isinstance(src, list) else [src],
                content=evt.get('content', ''),
                metadata=evt.get('metadata', {}),
                influence=influence
            ))
        self.events.sort(key=lambda x: x.time)

    def add_event(self, event: ExternalEvent):
        self.events.append(event)
        self.events.sort(key=lambda x: x.time)

    def get_current_events(self, current_time: str, window_minutes: int = 10) -> List[ExternalEvent]:
        """获取当前时间步窗口内新触发的全局广播事件
        
        只返回事件时间落在 (current_time - window_minutes, current_time] 内的事件，
        避免每一步都返回所有历史事件导致重复注入。
        
        Args:
            current_time: 当前模拟时间 (ISO格式)
            window_minutes: 时间窗口大小（分钟），应与 time_granularity 一致
        """
        if not current_time:
            return []
        cur = datetime.fromisoformat(current_time)
        result = []
        for evt in self.events:
            if not evt.time or evt.event_type != EventType.GLOBAL_BROADCAST:
                continue
            evt_dt = datetime.fromisoformat(evt.time)
            delta_seconds = (cur - evt_dt).total_seconds()
            # 事件时间在 (cur - window, cur] 范围内才触发
            if 0 <= delta_seconds < window_minutes * 60:
                result.append(evt)
        return result

    def get_recent_events(self, current_time: str, count: int = 5) -> List[ExternalEvent]:
        """获取截止到当前时间为止最近 count 条全局广播事件，供智能体作为历史上下文"""
        if not current_time:
            return []
        cur = datetime.fromisoformat(current_time)
        past = [
            evt for evt in self.events
            if evt.event_type == EventType.GLOBAL_BROADCAST and evt.time
            and datetime.fromisoformat(evt.time) <= cur
        ]
        return past[-count:] if len(past) > count else past

    def get_node_events(self, current_time: str, window_minutes: int = 10) -> List[ExternalEvent]:
        """获取当前时间步窗口内新触发的节点事件

        只返回事件时间落在 (current_time - window_minutes, current_time] 内的事件，
        与 get_current_events 逻辑一致，避免重复注入。

        Args:
            current_time: 当前模拟时间 (ISO格式)
            window_minutes: 时间窗口大小（分钟），应与 time_granularity 一致
        """
        if not current_time:
            return []
        cur = datetime.fromisoformat(current_time)
        window_seconds = window_minutes * 60
        return [
            evt for evt in self.events
            if evt.event_type == EventType.NODE_POST and evt.time
            and datetime.fromisoformat(evt.time) <= cur
            and (cur - datetime.fromisoformat(evt.time)).total_seconds() < window_seconds
        ]
