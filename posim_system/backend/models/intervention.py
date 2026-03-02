"""
干预记录 ORM 模型
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, Float, DateTime, JSON, ForeignKey
from ..database import Base


class Intervention(Base):
    __tablename__ = "interventions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    simulation_id = Column(Integer, ForeignKey("simulations.id"), nullable=False, index=True)
    intervention_type = Column(String(50), nullable=False)
    # ban_user / unban_user / delete_post / inject_event / update_param
    target = Column(Text, default="")  # 目标ID (user_id / post_id / param_name)
    params = Column(JSON, default=dict)  # 干预参数
    step_applied = Column(Integer, default=0)  # 应用时的仿真步骤
    sim_time = Column(String(100), default="")  # 应用时的仿真时间
    result = Column(Text, default="")  # 结果描述

    created_at = Column(DateTime, default=datetime.utcnow)
