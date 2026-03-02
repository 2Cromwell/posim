# -*- coding: utf-8 -*-
"""
POSIM Evaluation Package - 仿真评估模块

模块结构:
- base: 基础评估器类
- utils: 共享工具函数
- visualization: 可视化配置
- data_loader: 数据加载器（模拟/真实数据）
- evaluator_manager: 评估管理器（总调度）
- mechanism/: 机制验证（仅使用模拟数据）
  - agent_behavior: 智能体行为机制验证（LLM驱动）
  - macro_phenomenon: 宏观现象机制验证
- calibration/: 真实数据校准
  - hotness: 宏观热度曲线校准
  - emotion: 情绪/情感曲线校准
  - topic: 话题演化校准
  - opinion_index: 舆情演化指数
  - network: 网络拓扑结构校准
"""

from .evaluator_manager import EvaluationManager
