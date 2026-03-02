"""
WebSocket 连接管理器 —— 管理所有前端 WebSocket 连接，广播仿真实时信号
"""
import json
import logging
from typing import Dict, Set, Any
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebSocketManager:
    """管理 WebSocket 连接并广播消息"""

    def __init__(self):
        # simulation_id -> set of connected websockets
        self.connections: Dict[int, Set[WebSocket]] = {}

    async def connect(self, simulation_id: int, websocket: WebSocket):
        await websocket.accept()
        if simulation_id not in self.connections:
            self.connections[simulation_id] = set()
        self.connections[simulation_id].add(websocket)
        logger.info(f"WS 客户端连接: sim={simulation_id}, 总数={len(self.connections[simulation_id])}")

    def disconnect(self, simulation_id: int, websocket: WebSocket):
        if simulation_id in self.connections:
            self.connections[simulation_id].discard(websocket)
            if not self.connections[simulation_id]:
                del self.connections[simulation_id]
        logger.info(f"WS 客户端断开: sim={simulation_id}")

    async def broadcast(self, simulation_id: int, message: Dict[str, Any]):
        """向指定仿真的所有连接广播消息"""
        if simulation_id not in self.connections:
            return
        dead = set()
        msg_str = json.dumps(message, ensure_ascii=False, default=str)
        for ws in self.connections[simulation_id]:
            try:
                await ws.send_text(msg_str)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self.connections[simulation_id].discard(ws)

    async def send_signal(self, simulation_id: int, signal_data: Dict[str, Any]):
        """发送仿真步骤信号"""
        await self.broadcast(simulation_id, {"type": "signal", "data": signal_data})

    async def send_action(self, simulation_id: int, action_data: Dict[str, Any]):
        """发送行为事件"""
        await self.broadcast(simulation_id, {"type": "action", "data": action_data})

    async def send_status(self, simulation_id: int, status: str, progress: float = 0.0,
                          extra: Dict = None):
        """发送状态更新"""
        msg = {"type": "status", "data": {"status": status, "progress": progress}}
        if extra:
            msg["data"].update(extra)
        await self.broadcast(simulation_id, msg)

    async def send_history(self, simulation_id: int, websocket: WebSocket,
                           signals: list):
        """向单个客户端发送历史信号"""
        try:
            await websocket.send_text(json.dumps(
                {"type": "history", "data": signals}, ensure_ascii=False, default=str
            ))
        except Exception:
            pass


# 全局单例
ws_manager = WebSocketManager()
