"""
分析评估相关的请求/响应模型
"""
from typing import Optional, Dict, Any, List
from pydantic import BaseModel


class EvaluationRequest(BaseModel):
    simulation_id: int
    skip_mechanism: bool = False
    skip_calibration: bool = False
    skip_llm_evaluation: bool = True  # 默认跳过LLM评估（耗时）
    real_data_path: Optional[str] = None


class EvaluationStatusResponse(BaseModel):
    simulation_id: int
    status: str  # pending / running / completed / failed
    progress: float
    results: Dict[str, Any]


class ComparisonRequest(BaseModel):
    simulation_ids: List[int]


class ComparisonResponse(BaseModel):
    simulations: List[Dict[str, Any]]
    comparison_metrics: Dict[str, Any]


class AgentDetailResponse(BaseModel):
    """智能体详情"""
    user_id: str
    username: str
    agent_type: str
    followers_count: int
    activity_score: float
    is_banned: bool
    belief: Dict[str, Any]
    recent_actions: List[Dict[str, Any]]


class TimeSeriesResponse(BaseModel):
    """时间序列数据"""
    times: List[str]
    series: Dict[str, List[float]]


class ExportRequest(BaseModel):
    simulation_id: int
    format: str = "json"  # json / csv
    data_type: str = "all"  # all / micro / macro / agents / config
