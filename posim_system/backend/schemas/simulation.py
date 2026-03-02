"""
仿真运行相关的请求/响应模型
"""
from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class SimulationCreate(BaseModel):
    project_id: int
    title: str = ""


class SimulationControlRequest(BaseModel):
    action: str = Field(..., pattern="^(pause|resume|stop)$")


class SimulationStatusResponse(BaseModel):
    id: int
    project_id: int
    title: str
    status: str
    progress: float
    current_step: int
    total_steps: int
    total_actions: int
    output_dir: str
    result_summary: Dict[str, Any]
    config_snapshot: Dict[str, Any]
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    created_at: datetime

    class Config:
        from_attributes = True


class SimulationListResponse(BaseModel):
    total: int
    items: list[SimulationStatusResponse]


class StepSignalResponse(BaseModel):
    """单步实时信号"""
    step: int
    time: str
    elapsed_minutes: float
    activated_count: int
    total_agents: int
    actions_count: int
    post_count: int
    repost_count: int
    comment_count: int
    like_count: int
    actions_by_type: Dict[str, int]
    hawkes_intensity: float
    hawkes_mu: float
    hawkes_circadian_factor: float
    external_events_triggered: List[str]


class SimulationResultResponse(BaseModel):
    """仿真结果摘要"""
    simulation_id: int
    steps: int
    total_actions: int
    actions_by_type: Dict[str, int]
    intensity_history: List[float]
    active_agents_per_step: List[int]
    actions_per_step: List[int]
    final_hot_search: List[Dict[str, Any]]
    performance: Dict[str, Any]
