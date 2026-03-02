"""
仿真项目 ORM 模型
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, JSON
from ..database import Base


class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(200), nullable=False, index=True)
    description = Column(Text, default="")
    event_name = Column(String(200), default="")
    event_background = Column(Text, default="")
    status = Column(String(50), default="draft")  # draft / ready / archived

    # 仿真参数 JSON (存储完整 simulation config)
    simulation_config = Column(JSON, default=dict)
    # LLM 配置 JSON
    llm_config = Column(JSON, default=dict)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
