"""
仿真运行记录 ORM 模型
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, Float, DateTime, JSON, ForeignKey
from ..database import Base


class Simulation(Base):
    __tablename__ = "simulations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False, index=True)
    title = Column(String(200), default="")
    status = Column(String(50), default="pending")  # pending / running / paused / completed / failed / stopped
    progress = Column(Float, default=0.0)  # 0.0 ~ 1.0

    # 运行统计
    current_step = Column(Integer, default=0)
    total_steps = Column(Integer, default=0)
    total_actions = Column(Integer, default=0)

    # 输出路径
    output_dir = Column(Text, default="")
    db_path = Column(Text, default="")  # simulation.db 路径

    # 最终统计摘要 JSON
    result_summary = Column(JSON, default=dict)
    # 运行时使用的完整配置快照
    config_snapshot = Column(JSON, default=dict)

    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
