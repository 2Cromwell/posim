"""
仿真调度服务 —— 桥接 Web API 与 POSIM 引擎
"""
import json
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime
from sqlalchemy.orm import Session

from ..models.project import Project
from ..models.simulation import Simulation
from ..models.dataset import Dataset
from ..config import DATA_DIR
from ..core.engine_bridge import EngineBridge
from ..core.task_manager import task_manager

logger = logging.getLogger(__name__)


def _build_full_config(project: Project, datasets: List[Dataset]) -> Dict[str, Any]:
    """基于项目配置和数据集构建完整的 posim config dict"""
    sim_cfg = dict(project.simulation_config or {})
    llm_cfg = dict(project.llm_config or {})

    # 数据文件路径
    data_cfg = {}
    for ds in datasets:
        key_map = {
            "users": "users_file",
            "events": "events_file",
            "initial_posts": "initial_posts_file",
            "relations": "relations_file",
        }
        if ds.data_type in key_map and ds.file_path:
            data_cfg[key_map[ds.data_type]] = ds.file_path

    return {
        "simulation": sim_cfg,
        "data": data_cfg,
        "llm": llm_cfg,
        "neo4j": {"enabled": False},
        "output": {"base_dir": "output", "save_all_results": True, "run_evaluation": False},
        "debug": {"enabled": True, "log_level": "INFO", "llm_prompt_sample_rate": 0.05},
    }


def create_simulation(db: Session, project_id: int, title: str = "") -> Optional[Simulation]:
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        return None

    sim = Simulation(
        project_id=project_id,
        title=title or f"{project.event_name}_{datetime.now().strftime('%m%d_%H%M')}",
        status="pending",
    )
    db.add(sim)
    db.commit()
    db.refresh(sim)
    return sim


async def start_simulation(db: Session, simulation_id: int) -> Dict[str, Any]:
    """启动仿真"""
    sim = db.query(Simulation).filter(Simulation.id == simulation_id).first()
    if not sim:
        return {"error": "仿真记录不存在"}
    if sim.status in ("running", "paused"):
        return {"error": f"仿真已处于 {sim.status} 状态"}

    project = db.query(Project).filter(Project.id == sim.project_id).first()
    if not project:
        return {"error": "项目不存在"}

    # 获取项目数据集
    datasets = db.query(Dataset).filter(Dataset.project_id == project.id).all()

    # 构建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = DATA_DIR / f"project_{project.id}" / f"sim_{sim.id}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    sim.output_dir = str(output_dir)

    # 构建配置
    full_config = _build_full_config(project, datasets)
    sim.config_snapshot = full_config

    # 计算总步数估算
    sim_cfg = full_config.get("simulation", {})
    try:
        from datetime import datetime as dt
        start = dt.fromisoformat(sim_cfg.get("start_time", ""))
        end = dt.fromisoformat(sim_cfg.get("end_time", ""))
        granularity = sim_cfg.get("time_granularity", 10)
        total_minutes = (end - start).total_seconds() / 60
        sim.total_steps = int(total_minutes / granularity)
    except Exception:
        sim.total_steps = 0

    db.commit()

    # 初始化引擎
    bridge = EngineBridge()
    bridge.init_from_config_dict(full_config, str(output_dir))

    # 加载数据：优先从数据集文件加载，否则尝试从项目目录加载
    data_loaded = False
    if datasets:
        try:
            users_data, events_data, posts_data, relations_data = [], [], [], []
            for ds in datasets:
                if ds.file_path and Path(ds.file_path).exists():
                    with open(ds.file_path, "r", encoding="utf-8") as f:
                        content = json.load(f)
                    if ds.data_type == "users":
                        users_data = content if isinstance(content, list) else [content]
                    elif ds.data_type == "events":
                        events_data = content if isinstance(content, list) else [content]
                    elif ds.data_type == "initial_posts":
                        posts_data = content if isinstance(content, list) else [content]
                    elif ds.data_type == "relations":
                        relations_data = content if isinstance(content, list) else [content]

            if users_data:
                stats = bridge.load_data_from_json(users_data, events_data, posts_data, relations_data)
                data_loaded = True
                logger.info(f"从数据集加载: {stats}")
        except Exception as e:
            logger.error(f"从数据集加载失败: {e}")

    if not data_loaded:
        # 尝试从样例数据目录加载
        sample_dirs = {
            "tianjiaerhuan": "scripts/tianjiaerhuan/data",
            "wudatushuguan": "scripts/wudatushuguan/data",
            "xibeiyuzhicai": "scripts/xibeiyuzhicai/data",
        }
        event_name = sim_cfg.get("event_name", "")
        from ..config import PROJECT_ROOT
        if event_name in sample_dirs:
            data_path = PROJECT_ROOT / sample_dirs[event_name]
            if data_path.exists():
                stats = bridge.load_data_from_dir(str(data_path))
                data_loaded = True
                logger.info(f"从样例数据加载: {stats}")

    if not data_loaded:
        bridge.cleanup()
        return {"error": "未找到仿真数据，请先上传数据集"}

    # 启动异步任务
    sim_task = await task_manager.start_simulation(simulation_id, bridge, db, sim)
    return {"simulation_id": simulation_id, "status": "running"}


def get_simulation(db: Session, simulation_id: int) -> Optional[Simulation]:
    return db.query(Simulation).filter(Simulation.id == simulation_id).first()


def list_simulations(db: Session, project_id: int = None,
                     skip: int = 0, limit: int = 50) -> List[Simulation]:
    q = db.query(Simulation)
    if project_id:
        q = q.filter(Simulation.project_id == project_id)
    return q.order_by(Simulation.created_at.desc()).offset(skip).limit(limit).all()


def count_simulations(db: Session, project_id: int = None) -> int:
    q = db.query(Simulation)
    if project_id:
        q = q.filter(Simulation.project_id == project_id)
    return q.count()


def control_simulation(simulation_id: int, action: str) -> Dict[str, Any]:
    """暂停/恢复/停止仿真"""
    if action == "pause":
        ok = task_manager.pause_simulation(simulation_id)
    elif action == "resume":
        ok = task_manager.resume_simulation(simulation_id)
    elif action == "stop":
        ok = task_manager.stop_simulation(simulation_id)
    else:
        return {"error": f"未知操作: {action}"}
    return {"success": ok, "action": action}


def get_simulation_live_data(simulation_id: int) -> Dict[str, Any]:
    """获取运行中仿真的实时数据"""
    bridge = task_manager.get_bridge(simulation_id)
    if not bridge:
        return {"error": "仿真未运行"}
    return {
        "signals_history": bridge.get_signals_history(),
        "hot_search": bridge.get_hot_search(),
        "agents_count": bridge.total_agents,
        "progress": bridge.progress,
        "current_step": bridge.current_step,
    }


def get_simulation_results(db: Session, simulation_id: int) -> Dict[str, Any]:
    """获取已完成仿真的结果"""
    sim = get_simulation(db, simulation_id)
    if not sim:
        return {"error": "仿真不存在"}

    result = {"simulation_id": simulation_id, "status": sim.status}

    if sim.output_dir:
        output = Path(sim.output_dir)
        for fname in ["macro_results.json", "micro_results.json"]:
            fpath = output / fname
            if fpath.exists():
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        result[fname.replace(".json", "")] = json.load(f)
                except Exception:
                    pass

    result["result_summary"] = sim.result_summary or {}
    return result
