"""
POSIM 引擎适配层 —— 桥接 posim/ 核心模块与 Web 后端

直接调用 posim 的 ConfigManager, APIPool, Simulator, DataLoader, EvaluationManager
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from posim.config.config_manager import ConfigManager
from posim.config.config_schema import SimulationConfig
from posim.llm.api_pool import APIPool
from posim.engine.simulator import Simulator, StepSignals
from posim.data.data_loader import DataLoader, parse_user_data
from posim.storage.database import SimulationDatabase
from posim.evaluation.evaluator_manager import EvaluationManager

logger = logging.getLogger(__name__)


class EngineBridge:
    """
    将 posim 核心引擎封装为可被 Web 后端调用的服务对象。
    每个仿真运行实例对应一个 EngineBridge。
    """

    def __init__(self):
        self.config_manager: Optional[ConfigManager] = None
        self.api_pool: Optional[APIPool] = None
        self.simulator: Optional[Simulator] = None
        self.sim_db: Optional[SimulationDatabase] = None
        self.output_dir: Optional[Path] = None
        self.all_actions: List[Dict] = []
        self._paused = False
        self._stopped = False

    # ──────────────────── 初始化 ────────────────────

    def init_from_config_dict(self, config_dict: Dict[str, Any], output_dir: str) -> None:
        """从字典配置初始化引擎（Web模式，不依赖 config.json 文件）"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 写入临时 config.json 供 ConfigManager 读取
        tmp_config = self.output_dir / "config.json"
        with open(tmp_config, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)

        self.config_manager = ConfigManager(str(tmp_config))
        self.api_pool = APIPool(self.config_manager.llm, self.config_manager.debug, str(self.output_dir))
        self.simulator = Simulator(self.config_manager, self.api_pool)
        self.sim_db = SimulationDatabase(str(self.output_dir / "simulation.db"))

    def init_from_config_file(self, config_path: str, output_dir: str) -> None:
        """从已有 config.json 文件初始化引擎"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.config_manager = ConfigManager(config_path)
        self.api_pool = APIPool(self.config_manager.llm, self.config_manager.debug, str(self.output_dir))
        self.simulator = Simulator(self.config_manager, self.api_pool)
        self.sim_db = SimulationDatabase(str(self.output_dir / "simulation.db"))

    # ──────────────────── 数据加载 ────────────────────

    def load_data_from_dir(self, data_dir: str) -> Dict[str, int]:
        """从目录加载全部仿真数据，返回各类数据的记录数"""
        loader = DataLoader(data_dir)
        data = loader.load_all()
        users_data = [parse_user_data(u) for u in data["users"]]

        self.simulator.load_agents(users_data)
        self.simulator.load_events(data["events"])
        self.simulator.load_relations(data.get("relations", []))
        self.simulator.load_initial_posts(data["initial_posts"])

        return {
            "users": len(users_data),
            "events": len(data["events"]),
            "initial_posts": len(data["initial_posts"]),
            "relations": len(data.get("relations", [])),
            "agents_loaded": len(self.simulator.agents),
        }

    def load_data_from_json(self, users: list, events: list,
                            initial_posts: list, relations: list = None) -> Dict[str, int]:
        """直接从 JSON 列表加载数据"""
        users_data = [parse_user_data(u) for u in users]
        self.simulator.load_agents(users_data)
        self.simulator.load_events(events)
        self.simulator.load_relations(relations or [])
        self.simulator.load_initial_posts(initial_posts)

        return {
            "users": len(users_data),
            "events": len(events),
            "initial_posts": len(initial_posts),
            "relations": len(relations or []),
            "agents_loaded": len(self.simulator.agents),
        }

    # ──────────────────── 仿真运行 ────────────────────

    def set_signal_callback(self, callback):
        """设置实时信号回调（用于 WebSocket 推送）"""
        self.simulator.signal_callback = callback

    def set_action_callback(self, callback):
        """设置行为回调"""
        self.simulator.action_callback = callback

    async def run(self, progress_callback=None) -> Dict[str, Any]:
        """运行完整仿真（异步）"""
        self.all_actions = []
        self._paused = False
        self._stopped = False

        # 注册回调
        self.simulator.step_callback = lambda step_data: self.sim_db.save_statistics(
            step_data["step"], step_data["time"], step_data["intensity"],
            step_data["activated_count"], step_data["actions_count"], []
        )
        original_action_cb = self.simulator.action_callback

        def combined_action_cb(action):
            self.all_actions.append(action)
            self.sim_db.save_action(action, self.simulator.time_engine.state.step)
            if original_action_cb:
                original_action_cb(action)

        self.simulator.action_callback = combined_action_cb

        # 逐步执行（支持暂停/停止）
        results_list = []
        while not self.simulator.time_engine.is_finished():
            if self._stopped:
                logger.info("仿真被手动停止")
                break
            while self._paused:
                import asyncio
                await asyncio.sleep(0.5)
                if self._stopped:
                    break
            if self._stopped:
                break

            step_result = await self.simulator.run_step()
            results_list.append(step_result)

            if progress_callback:
                progress_callback(self.simulator.time_engine.progress, step_result)

        # 汇总结果
        final = {
            "steps": len(results_list),
            "stats": self.simulator.stats,
            "performance": self.simulator.perf_metrics.get_summary(),
            "final_hot_search": self.simulator.hot_search.get_hot_list(20),
            "hot_search_history": self.simulator.hot_search.get_history(),
            "time_engine": self.simulator.time_engine.to_dict(),
            "stopped": self._stopped,
        }

        # 保存结果文件
        self._save_results(final)
        return final

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def stop(self):
        self._stopped = True
        self._paused = False

    @property
    def is_paused(self):
        return self._paused

    @property
    def is_stopped(self):
        return self._stopped

    @property
    def progress(self) -> float:
        if self.simulator and self.simulator.time_engine:
            return self.simulator.time_engine.progress
        return 0.0

    @property
    def current_step(self) -> int:
        if self.simulator and self.simulator.time_engine:
            return self.simulator.time_engine.state.step
        return 0

    @property
    def total_agents(self) -> int:
        if self.simulator:
            return len(self.simulator.agents)
        return 0

    # ──────────────────── 干预接口 ────────────────────

    def ban_user(self, user_id: str) -> bool:
        if self.simulator and user_id in self.simulator.agents:
            self.simulator.ban_user(user_id)
            return True
        return False

    def unban_user(self, user_id: str) -> bool:
        if self.simulator and user_id in self.simulator.agents:
            self.simulator.unban_user(user_id)
            return True
        return False

    def delete_post(self, post_id: str) -> bool:
        if self.simulator:
            self.simulator.delete_post(post_id)
            return True
        return False

    def inject_event(self, content: str, influence: float = 1.0,
                     event_type: str = "global_broadcast", source: list = None) -> bool:
        if self.simulator:
            self.simulator.inject_event(event_type, content, source or [])
            return True
        return False

    # ──────────────────── 查询接口 ────────────────────

    def get_agent_detail(self, user_id: str) -> Optional[Dict]:
        if not self.simulator or user_id not in self.simulator.agents:
            return None
        agent = self.simulator.agents[user_id]
        return agent.to_dict()

    def get_all_agents_summary(self) -> List[Dict]:
        if not self.simulator:
            return []
        result = []
        for uid, agent in self.simulator.agents.items():
            result.append({
                "user_id": uid,
                "username": agent.profile.username,
                "agent_type": agent.agent_type,
                "followers_count": agent.profile.followers_count,
                "activity_score": agent.activity_score,
                "is_banned": agent.is_banned,
            })
        return result

    def get_signals_history(self) -> List[Dict]:
        if not self.simulator:
            return []
        return [s.to_dict() for s in self.simulator.signals_history]

    def get_hot_search(self, count: int = 20) -> List[Dict]:
        if not self.simulator:
            return []
        return self.simulator.hot_search.get_hot_list(count)

    def get_content_pool(self, limit: int = 50) -> List[Dict]:
        """获取推荐池中的博文"""
        if not self.simulator:
            return []
        pool = self.simulator.recommendation.content_pool
        return pool[-limit:] if len(pool) > limit else pool

    # ──────────────────── 评估接口 ────────────────────

    def run_evaluation(self, real_data_path: str = None,
                       base_data_path: str = None,
                       skip_mechanism: bool = False,
                       skip_calibration: bool = False,
                       skip_llm_evaluation: bool = True) -> Dict[str, Any]:
        """运行评估（同步调用 posim 的 EvaluationManager）"""
        sim_results_dir = str(self.output_dir)
        evaluator = EvaluationManager(
            sim_results_dir=sim_results_dir,
            real_data_path=real_data_path,
            base_data_path=base_data_path,
        )
        evaluator.load_data()
        results = evaluator.run_all(
            skip_mechanism=skip_mechanism,
            skip_calibration=skip_calibration,
            skip_llm_evaluation=skip_llm_evaluation,
        )
        return results

    # ──────────────────── 私有方法 ────────────────────

    def _save_results(self, final: Dict):
        """保存仿真结果文件"""
        if not self.output_dir:
            return
        try:
            with open(self.output_dir / "macro_results.json", "w", encoding="utf-8") as f:
                json.dump({
                    "steps": final["steps"],
                    "stats": final["stats"],
                    "performance": final["performance"],
                    "final_hot_search": final.get("final_hot_search", []),
                    "hot_search_history": final.get("hot_search_history", []),
                    "time_engine": final.get("time_engine", {}),
                }, f, ensure_ascii=False, indent=2, default=str)

            with open(self.output_dir / "micro_results.json", "w", encoding="utf-8") as f:
                json.dump(self.all_actions, f, ensure_ascii=False, indent=2, default=str)

            if self.config_manager:
                with open(self.output_dir / "config.json", "w", encoding="utf-8") as f:
                    json.dump(self.config_manager.raw_config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存结果文件失败: {e}")

    def cleanup(self):
        """清理资源"""
        if self.sim_db:
            self.sim_db.close()
        if self.simulator and hasattr(self.simulator, "social_network"):
            self.simulator.social_network.close()
