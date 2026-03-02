"""
项目相关的请求/响应模型
"""
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: str = ""
    event_name: str = ""
    event_background: str = ""
    simulation_config: Dict[str, Any] = {}
    llm_config: Dict[str, Any] = {}


class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    event_name: Optional[str] = None
    event_background: Optional[str] = None
    status: Optional[str] = None
    simulation_config: Optional[Dict[str, Any]] = None
    llm_config: Optional[Dict[str, Any]] = None


class ProjectResponse(BaseModel):
    id: int
    name: str
    description: str
    event_name: str
    event_background: str
    status: str
    simulation_config: Dict[str, Any]
    llm_config: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ProjectListResponse(BaseModel):
    total: int
    items: list[ProjectResponse]
