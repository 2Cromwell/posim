"""
项目管理服务
"""
import json
from typing import Optional, List, Dict, Any
from pathlib import Path
from sqlalchemy.orm import Session

from ..models.project import Project
from ..models.dataset import Dataset
from ..config import settings, DATA_DIR


# 默认仿真参数模板
DEFAULT_SIMULATION_CONFIG = {
    "event_name": "",
    "simulation_title": "default",
    "event_background": "",
    "start_time": "",
    "end_time": "",
    "time_granularity": 10,
    "participant_scale": 0,
    "hawkes_mu": 0.05,
    "hawkes_internal": {"alpha": 0.003, "beta": 0.05},
    "hawkes_external": {"alpha": 0.15, "beta": 0.003},
    "total_scale": 200.0,
    "circadian_strength": 0.3,
    "min_activation_noise": {"enabled": True, "min_rate": 0.0, "max_rate": 0.5},
    "action_weights": {
        "like": 0.01, "repost": 1.0, "repost_comment": 0.5,
        "short_comment": 0.1, "long_comment": 0.2,
        "short_post": 0.2, "long_post": 0.4,
    },
    "use_llm": True,
    "recommend_count": 10,
    "comment_count": 5,
    "homophily_weight": 0.3,
    "popularity_weight": 0.3,
    "recency_weight": 0.2,
    "exploration_rate": 0.2,
    "relation_weight": 0.5,
    "hot_search_update_interval": 15,
    "hot_search_count": 50,
    "hot_search_min_mentions": 10,
    "hot_search_display_count": 5,
    "circadian_curve": {
        "0": 0.4, "1": 0.3, "2": 0.2, "3": 0.2, "4": 0.2, "5": 0.3,
        "6": 0.5, "7": 0.7, "8": 0.9, "9": 1.0, "10": 1.0, "11": 0.95,
        "12": 0.85, "13": 0.8, "14": 0.85, "15": 0.9, "16": 0.95, "17": 1.0,
        "18": 1.1, "19": 1.15, "20": 1.2, "21": 1.2, "22": 1.1, "23": 0.8,
    },
}

DEFAULT_LLM_CONFIG = {
    "max_concurrent_requests": 10,
    "belief_llm_index": [0],
    "desire_llm_index": [0],
    "action_llm_index": [0],
    "strategy_llm_index": [0],
    "content_llm_index": [0],
    "recommendation_llm_index": [0],
    "other_llm_index": [0],
    "use_local_embedding_model": True,
    "local_embedding_model_path": "",
    "embedding_dimension": 512,
    "embedding_device": "cpu",
    "llm_api_configs": [],
}

# 场景参数预设模板
PRESET_TEMPLATES = {
    "突发事件": {
        "description": "突发热点事件，高峰快速衰减",
        "simulation_config": {
            "hawkes_mu": 0.02, "total_scale": 500,
            "hawkes_external": {"alpha": 0.2, "beta": 0.008},
        },
    },
    "持续热点": {
        "description": "持续发酵的舆情事件，衰减较慢",
        "simulation_config": {
            "hawkes_mu": 0.05, "total_scale": 300,
            "hawkes_external": {"alpha": 0.1, "beta": 0.002},
        },
    },
    "政策发布": {
        "description": "政府政策发布引发讨论",
        "simulation_config": {
            "hawkes_mu": 0.03, "total_scale": 200,
            "hawkes_external": {"alpha": 0.12, "beta": 0.005},
        },
    },
}


def create_project(db: Session, data: Dict[str, Any]) -> Project:
    sim_config = {**DEFAULT_SIMULATION_CONFIG, **(data.get("simulation_config") or {})}
    llm_config = {**DEFAULT_LLM_CONFIG, **(data.get("llm_config") or {})}

    if data.get("event_name"):
        sim_config["event_name"] = data["event_name"]
    if data.get("event_background"):
        sim_config["event_background"] = data["event_background"]

    project = Project(
        name=data["name"],
        description=data.get("description", ""),
        event_name=data.get("event_name", ""),
        event_background=data.get("event_background", ""),
        simulation_config=sim_config,
        llm_config=llm_config,
    )
    db.add(project)
    db.commit()
    db.refresh(project)

    # 创建项目数据目录
    project_dir = DATA_DIR / f"project_{project.id}"
    project_dir.mkdir(parents=True, exist_ok=True)
    return project


def get_project(db: Session, project_id: int) -> Optional[Project]:
    return db.query(Project).filter(Project.id == project_id).first()


def list_projects(db: Session, skip: int = 0, limit: int = 50) -> List[Project]:
    return db.query(Project).order_by(Project.updated_at.desc()).offset(skip).limit(limit).all()


def count_projects(db: Session) -> int:
    return db.query(Project).count()


def update_project(db: Session, project_id: int, data: Dict[str, Any]) -> Optional[Project]:
    project = get_project(db, project_id)
    if not project:
        return None
    for key, value in data.items():
        if value is not None and hasattr(project, key):
            setattr(project, key, value)
    db.commit()
    db.refresh(project)
    return project


def delete_project(db: Session, project_id: int) -> bool:
    project = get_project(db, project_id)
    if not project:
        return False
    db.delete(project)
    db.commit()
    return True


def clone_project(db: Session, project_id: int, new_name: str) -> Optional[Project]:
    """克隆项目（用于反事实推演）"""
    original = get_project(db, project_id)
    if not original:
        return None
    clone = Project(
        name=new_name,
        description=f"克隆自: {original.name}\n{original.description}",
        event_name=original.event_name,
        event_background=original.event_background,
        simulation_config=dict(original.simulation_config),
        llm_config=dict(original.llm_config),
    )
    db.add(clone)
    db.commit()
    db.refresh(clone)
    return clone


def get_preset_templates() -> Dict[str, Any]:
    return PRESET_TEMPLATES


def get_sample_datasets() -> Dict[str, Any]:
    """获取内置样例数据集信息"""
    result = {}
    for name, dir_path in settings.SAMPLE_DATA_DIRS.items():
        p = Path(dir_path)
        if p.exists():
            data_dir = p / "data"
            config_file = p / "config.json"
            info = {"path": str(p), "has_data": data_dir.exists(), "has_config": config_file.exists()}
            if config_file.exists():
                try:
                    with open(config_file, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                    info["event_name"] = cfg.get("simulation", {}).get("event_name", name)
                    info["event_background"] = cfg.get("simulation", {}).get("event_background", "")
                except Exception:
                    pass
            result[name] = info
    return result
