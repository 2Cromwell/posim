"""
霍克斯点过程 - 时间驱动的活跃度模拟
λ(t) = μ + Σ α·exp(-β(t-ti))
用于宏观控制智能体活跃规模
"""
import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class HawkesEvent:
    """霍克斯事件"""
    time: float  # 相对时间（分钟）
    influence: float  # 事件影响力
    event_type: str  # external/internal
    source: str = ""


class HawkesProcess:
    """霍克斯点过程模拟器"""
    
    def __init__(self, mu: float = 0.1, alpha: float = 0.5, beta: float = 1.0,
                 action_weights: Dict[str, float] = None, time_granularity: int = 10,
                 activation_scale: float = 1.0, circadian_curve: Dict[int, float] = None,
                 event_influence_scale: float = 1.0):
        """
        Args:
            mu: 基础强度（背景活跃率）
            alpha: 激励强度（事件对活跃度的影响）
            beta: 衰减率（影响力随时间衰减的速度，单位: 1/分钟）
            action_weights: 不同行为的影响力权重
            time_granularity: 时间粒度（分钟），用于调整每步激活数
            activation_scale: 激活系数，乘在participant_scale上
            circadian_curve: 昼夜活跃度曲线（小时->系数）
            event_influence_scale: 外部事件影响力缩放系数，用于放大事件对热度的影响
        """
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.time_granularity = time_granularity
        self.activation_scale = activation_scale
        self.event_influence_scale = event_influence_scale
        self.circadian_curve = circadian_curve or {}
        self.action_weights = action_weights or {
            'like': 0.1, 'repost': 1.0, 'repost_comment': 0.8,
            'short_comment': 0.3, 'long_comment': 0.5,
            'short_post': 0.6, 'long_post': 0.8
        }
        
        self.events: List[HawkesEvent] = []
        self.current_time: float = 0.0
        self.start_hour: int = 0  # 仿真开始时的小时数
    
    def set_start_hour(self, hour: int):
        """设置仿真开始时间的小时数（用于昼夜节律计算）"""
        self.start_hour = hour
    
    def _get_circadian_factor(self, t: float) -> float:
        """获取当前时刻的昼夜节律系数"""
        if not self.circadian_curve:
            return 1.0
        # t是相对分钟数，转换为当前小时
        current_hour = (self.start_hour + int(t // 60)) % 24
        return self.circadian_curve.get(current_hour, 1.0)
    
    def add_external_event(self, time: float, influence: float, source: str = ""):
        """添加外部事件，应用事件影响力缩放系数"""
        scaled_influence = influence * self.event_influence_scale
        self.events.append(HawkesEvent(
            time=time, influence=scaled_influence, event_type='external', source=source
        ))
    
    def add_internal_event(self, time: float, action_type: str, source: str = "", 
                           user_influence: float = 1.0):
        """添加内部事件（智能体行为）
        
        Args:
            time: 事件发生时间
            action_type: 行为类型
            source: 事件源用户ID
            user_influence: 用户影响力因子（基于粉丝数等）
        """
        base_influence = self.action_weights.get(action_type, 0.5)
        # 行为影响力 = 行为基础强度 × 用户影响力因子
        influence = base_influence * user_influence
        self.events.append(HawkesEvent(
            time=time, influence=influence, event_type='internal', source=source
        ))
    
    def get_intensity(self, t: float) -> float:
        """
        计算t时刻的强度λ(t)，应用昼夜节律因子
        λ(t) = circadian_factor * (μ + Σ α·w_i·exp(-β(t-t_i)))
        """
        base_intensity = self.mu
        for evt in self.events:
            if evt.time < t:
                delta = t - evt.time
                base_intensity += self.alpha * evt.influence * np.exp(-self.beta * delta)
        # 应用昼夜节律因子
        circadian_factor = self._get_circadian_factor(t)
        return base_intensity * circadian_factor
    
    def get_expected_activations(self, t: float, total_agents: int) -> int:
        """
        计算t时刻预期激活的智能体数量
        考虑：强度、人数规模、时间粒度、激活系数
        """
        intensity = self.get_intensity(t)
        # 基于时间粒度和激活系数计算期望激活数
        # activation_scale控制整体活跃程度，time_granularity调整每步激活数
        scale = self.activation_scale * (self.time_granularity / 60.0)
        expected = intensity * total_agents * scale
        activated = np.random.poisson(max(0, min(expected, total_agents)))
        return min(activated, total_agents)
    
    def simulate_next_event(self, t_start: float, t_max: float = None) -> float:
        """
        模拟下一个事件发生的时间
        使用Ogata的thinning算法
        """
        t = t_start
        t_max = t_max or t_start + 1440  # 默认最多24小时
        
        while t < t_max:
            lambda_bar = self.get_intensity(t) * 1.5  # 上界
            if lambda_bar <= 0:
                lambda_bar = self.mu
            
            # 指数分布采样
            u = np.random.random()
            w = -np.log(u) / lambda_bar
            t = t + w
            
            if t >= t_max:
                return t_max
            
            # 接受-拒绝
            lambda_t = self.get_intensity(t)
            d = np.random.random()
            if d <= lambda_t / lambda_bar:
                return t
        
        return t_max
    
    def advance_time(self, delta_minutes: float):
        """推进时间"""
        self.current_time += delta_minutes
    
    def clear_old_events(self, max_age_minutes: float = 1440):
        """清理过期事件（默认保留24小时）"""
        cutoff = self.current_time - max_age_minutes
        self.events = [e for e in self.events if e.time >= cutoff]
    
    def get_activity_history(self, window_minutes: int = 60) -> List[float]:
        """获取历史活跃度曲线"""
        start = max(0, self.current_time - window_minutes)
        history = []
        for t in np.arange(start, self.current_time, 1):
            history.append(self.get_intensity(t))
        return history
    
    def fit_from_timestamps(self, timestamps: List[float], 
                           max_iter: int = 100, lr: float = 0.01) -> Dict[str, float]:
        """
        从真实事件时间戳拟合霍克斯参数
        使用最大似然估计
        """
        if len(timestamps) < 10:
            return {'mu': self.mu, 'alpha': self.alpha, 'beta': self.beta}
        
        timestamps = sorted(timestamps)
        T = timestamps[-1] - timestamps[0]
        n = len(timestamps)
        
        mu, alpha, beta = self.mu, self.alpha, self.beta
        
        for _ in range(max_iter):
            # 计算似然和梯度
            ll = n * np.log(mu) - mu * T
            grad_mu = n / mu - T
            grad_alpha = 0
            grad_beta = 0
            
            for i, ti in enumerate(timestamps):
                R = sum(np.exp(-beta * (ti - tj)) for tj in timestamps[:i])
                lambda_i = mu + alpha * R
                
                if lambda_i > 0:
                    ll += np.log(lambda_i)
                    grad_alpha += R / lambda_i
                    grad_beta += -alpha * sum((ti - tj) * np.exp(-beta * (ti - tj)) 
                                              for tj in timestamps[:i]) / lambda_i
            
            # 更新参数
            mu = max(0.01, mu + lr * grad_mu)
            alpha = max(0.01, min(0.99, alpha + lr * grad_alpha))
            beta = max(0.1, beta + lr * grad_beta)
        
        self.mu, self.alpha, self.beta = mu, alpha, beta
        return {'mu': mu, 'alpha': alpha, 'beta': beta}
