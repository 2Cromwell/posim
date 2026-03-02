"""
干预操作相关的请求/响应模型
"""
from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class BanUserRequest(BaseModel):
    user_id: str


class DeletePostRequest(BaseModel):
    post_id: str


class InjectEventRequest(BaseModel):
    content: str
    influence: float = Field(default=1.0, ge=0.0, le=1.0)
    event_type: str = "global_broadcast"
    source: List[str] = []


class UpdateParamRequest(BaseModel):
    param_name: str
    param_value: Any


class InterventionResponse(BaseModel):
    id: int
    simulation_id: int
    intervention_type: str
    target: str
    params: Dict[str, Any]
    step_applied: int
    sim_time: str
    result: str
    created_at: datetime

    class Config:
        from_attributes = True


class InterventionListResponse(BaseModel):
    total: int
    items: list[InterventionResponse]
