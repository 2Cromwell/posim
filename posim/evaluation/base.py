# -*- coding: utf-8 -*-
"""
评估器基类 - 所有评估模块的基础类
"""
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class BaseEvaluator(ABC):
    """
    评估器基类
    
    所有子评估器需要实现 evaluate() 方法，
    返回包含评估指标和可视化结果的字典。
    """
    
    def __init__(self, output_dir: Path, name: str = "base"):
        """
        Args:
            output_dir: 输出目录（vis_results下的子目录）
            name: 评估器名称
        """
        self.name = name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    def evaluate(self, sim_data: Dict[str, Any], real_data: Optional[Dict[str, Any]] = None,
                 **kwargs) -> Dict[str, Any]:
        """
        执行评估
        
        Args:
            sim_data: 模拟数据（包含 micro_results, macro_results 等）
            real_data: 真实数据（可选，机制验证类评估器不需要）
            **kwargs: 额外参数（如 api_pool, embedding_model 等）
            
        Returns:
            评估结果字典
        """
        raise NotImplementedError
    
    def _save_results(self, results: Dict[str, Any], filename: str = "metrics.json"):
        """保存评估结果到JSON"""
        from .utils import save_json
        save_json(self.output_dir / filename, results)
        self.logger.info(f"[SAVED] {self.output_dir.name}/{filename}")
    
    def _log_section(self, title: str):
        """打印分隔标题"""
        print(f"\n  {'─' * 50}")
        print(f"  📊 {title}")
        print(f"  {'─' * 50}")
