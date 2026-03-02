"""
数据集相关的请求/响应模型
"""
from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class DatasetResponse(BaseModel):
    id: int
    project_id: int
    name: str
    data_type: str
    file_path: str
    record_count: int
    summary: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class DatasetListResponse(BaseModel):
    total: int
    items: list[DatasetResponse]


class DataPreviewResponse(BaseModel):
    """数据预览"""
    data_type: str
    record_count: int
    sample_records: List[Dict[str, Any]]
    field_names: List[str]
    summary: Dict[str, Any]


class DataValidationResponse(BaseModel):
    """数据校验结果"""
    valid: bool
    record_count: int
    errors: List[str]
    warnings: List[str]
