"""
WebSocket服务器 - 实时推送仿真信号到前端
"""
import asyncio
import json
import logging
from typing import Set
from dataclasses import asdict

logger = logging.getLogger(__name__)

try:
    import websockets
    from websockets.server import serve, WebSocketServerProtocol
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    logger.warning("websockets库未安装，请运行: pip install websockets")


class SimulationWebSocketServer:
    """仿真WebSocket服务器"""
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets库未安装，请运行: pip install websockets")
        
        self.host = host
        self.port = port
        self.clients: Set[WebSocketServerProtocol] = set()
        self.server = None
        self._running = False
        self.signals_history = []
        self.max_history = 1000
    
    async def register(self, websocket: WebSocketServerProtocol):
        """注册新客户端"""
        self.clients.add(websocket)
        logger.info(f"客户端连接: {websocket.remote_address}, 总数: {len(self.clients)}")
        
        # 发送历史数据
        if self.signals_history:
            await websocket.send(json.dumps({'type': 'history', 'data': self.signals_history}, ensure_ascii=False))
    
    async def unregister(self, websocket: WebSocketServerProtocol):
        """注销客户端"""
        self.clients.discard(websocket)
        logger.info(f"客户端断开: {websocket.remote_address}, 总数: {len(self.clients)}")
    
    async def broadcast(self, message: dict):
        """广播消息到所有客户端"""
        if not self.clients:
            return
        
        message_str = json.dumps(message, ensure_ascii=False, default=str)
        
        # 保存到历史
        if message.get('type') == 'signal':
            self.signals_history.append(message.get('data', {}))
            if len(self.signals_history) > self.max_history:
                self.signals_history = self.signals_history[-self.max_history:]
        
        # 发送到所有客户端
        disconnected = set()
        for client in self.clients:
            try:
                await client.send(message_str)
            except (websockets.exceptions.ConnectionClosed, Exception):
                disconnected.add(client)
        
        for client in disconnected:
            self.clients.discard(client)
    
    async def handler(self, websocket: WebSocketServerProtocol, path: str = None):
        """处理客户端连接"""
        await self.register(websocket)
        try:
            async for message in websocket:
                data = json.loads(message)
                if data.get('type') == 'get_history':
                    await websocket.send(json.dumps({'type': 'history', 'data': self.signals_history}, ensure_ascii=False))
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister(websocket)
    
    async def start(self):
        """启动WebSocket服务器"""
        self._running = True
        try:
            async with serve(self.handler, self.host, self.port) as server:
                self.server = server
                logger.info(f"WebSocket服务器启动: ws://{self.host}:{self.port}")
                while self._running:
                    await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            logger.info("WebSocket服务器任务被取消")
        finally:
            self._running = False
            logger.info("WebSocket服务器已停止")
    
    async def stop(self):
        """停止WebSocket服务器"""
        self._running = False
    
    def send_signal(self, signal):
        """发送信号（创建异步任务）"""
        if not self._running or not self.clients:
            return
        
        # 转换为字典
        if hasattr(signal, 'to_dict'):
            signal_dict = signal.to_dict()
        elif hasattr(signal, '__dict__'):
            signal_dict = asdict(signal) if hasattr(signal, '__dataclass_fields__') else signal.__dict__
        else:
            signal_dict = signal
        
        message = {'type': 'signal', 'data': signal_dict}
        
        # 在运行的事件循环中创建任务
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.broadcast(message))
        except RuntimeError:
            # 如果没有运行的事件循环，忽略（服务器可能还没启动）
            pass


class WebSocketSignalCallback:
    """WebSocket信号回调包装器"""
    def __init__(self, server: SimulationWebSocketServer):
        self.server = server
    
    def __call__(self, signal):
        self.server.send_signal(signal)
