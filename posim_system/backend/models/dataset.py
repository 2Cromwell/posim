"""
数据集 ORM 模型
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, ForeignKey
from ..database import Base


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False, index=True)
    name = Column(String(200), nullable=False)
    data_type = Column(String(50), nullable=False)  # users / events / initial_posts / relations
    file_path = Column(Text, default="")  # 存储的文件路径
    record_count = Column(Integer, default=0)
    # 数据摘要 / 统计信息
    summary = Column(JSON, default=dict)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
