"""
后台任务管理器 —— 管理仿真运行的异步任务和 EngineBridge 实例
"""
import asyncio
import logging
from typing import Dict, Optional, Any
from datetime import datetime
from pathlib import Path

from .engine_bridge import EngineBridge
from .websocket_manager import ws_manager

logger = logging.getLogger(__name__)


class SimulationTask:
    """单个仿真运行任务"""

    def __init__(self, simulation_id: int, bridge: EngineBridge):
        self.simulation_id = simulation_id
        self.bridge = bridge
        self.task: Optional[asyncio.Task] = None
        self.status = "pending"
        self.error: Optional[str] = None
        self.result: Optional[Dict[str, Any]] = None

    @property
    def progress(self) -> float:
        return self.bridge.progress

    @property
    def current_step(self) -> int:
        return self.bridge.current_step


class TaskManager:
    """管理所有仿真任务"""

    def __init__(self):
        self.tasks: Dict[int, SimulationTask] = {}

    def get_task(self, simulation_id: int) -> Optional[SimulationTask]:
        return self.tasks.get(simulation_id)

    def get_bridge(self, simulation_id: int) -> Optional[EngineBridge]:
        task = self.tasks.get(simulation_id)
        return task.bridge if task else None

    async def start_simulation(self, simulation_id: int, bridge: EngineBridge,
                               db_session, sim_model) -> SimulationTask:
        """启动仿真异步任务"""
        sim_task = SimulationTask(simulation_id, bridge)
        self.tasks[simulation_id] = sim_task

        async def _run():
            sim_task.status = "running"
            sim_model.status = "running"
            sim_model.started_at = datetime.utcnow()
            db_session.commit()

            await ws_manager.send_status(simulation_id, "running")

            try:
                def progress_cb(progress, step_data):
                    sim_model.progress = progress
                    sim_model.current_step = step_data["step"]
                    sim_model.total_actions = bridge.simulator.stats.get("total_actions", 0)
                    try:
                        db_session.commit()
                    except Exception:
                        pass

                # 设置信号回调 -> WebSocket 推送
                def signal_callback(signal):
                    signal_dict = signal.to_dict() if hasattr(signal, "to_dict") else signal
                    asyncio.ensure_future(ws_manager.send_signal(simulation_id, signal_dict))

                bridge.set_signal_callback(signal_callback)

                result = await bridge.run(progress_callback=progress_cb)
                sim_task.result = result
                sim_task.status = "completed" if not bridge.is_stopped else "stopped"

                sim_model.status = sim_task.status
                sim_model.progress = 1.0 if not bridge.is_stopped else bridge.progress
                sim_model.finished_at = datetime.utcnow()
                sim_model.result_summary = {
                    "steps": result.get("steps", 0),
                    "total_actions": result.get("stats", {}).get("total_actions", 0),
                    "actions_by_type": result.get("stats", {}).get("actions_by_type", {}),
                    "performance": result.get("performance", {}),
                }
                db_session.commit()

                await ws_manager.send_status(simulation_id, sim_task.status, sim_model.progress)
                logger.info(f"仿真 {simulation_id} 完成: {sim_task.status}")

            except Exception as e:
                sim_task.status = "failed"
                sim_task.error = str(e)
                sim_model.status = "failed"
                sim_model.finished_at = datetime.utcnow()
                try:
                    db_session.commit()
                except Exception:
                    pass
                await ws_manager.send_status(simulation_id, "failed", extra={"error": str(e)})
                logger.error(f"仿真 {simulation_id} 失败: {e}", exc_info=True)
            finally:
                bridge.cleanup()

        sim_task.task = asyncio.create_task(_run())
        return sim_task

    def pause_simulation(self, simulation_id: int) -> bool:
        task = self.tasks.get(simulation_id)
        if task and task.status == "running":
            task.bridge.pause()
            task.status = "paused"
            return True
        return False

    def resume_simulation(self, simulation_id: int) -> bool:
        task = self.tasks.get(simulation_id)
        if task and task.status == "paused":
            task.bridge.resume()
            task.status = "running"
            return True
        return False

    def stop_simulation(self, simulation_id: int) -> bool:
        task = self.tasks.get(simulation_id)
        if task and task.status in ("running", "paused"):
            task.bridge.stop()
            task.status = "stopped"
            return True
        return False


# 全局单例
task_manager = TaskManager()
