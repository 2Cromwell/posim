"""
霍克斯点过程 - 归一化强度的时间驱动活跃度模拟

核心公式:
    λ_raw(t) = μ + Σ α_int × w_i × exp(-β_int × Δt) + Σ α_ext × inf_i × exp(-β_ext × Δt)
    λ_norm(t) = 1 - exp(-λ_raw(t))                          ∈ [0, 1]
    λ(t) = λ_norm(t) × circadian_blend(t)

激活数:
    expected = total_scale × λ(t) × (time_granularity / 60)
    activated = Poisson(expected)

设计要点:
    - 软饱和归一化 1-exp(-x) 使强度自然稳定在 [0,1]，参数微调不会引起剧烈波动
    - total_scale 是唯一控制总量的线性旋钮（含义：峰值时每小时目标激活数）
    - circadian_blend = (1-s) + s×curve(h)，s 控制昼夜影响强度
    - 时间单位统一为分钟
"""
import logging
import numpy as np
from typing import List, Tuple, Dict, Any, Set
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class HawkesEvent:
    """霍克斯事件"""
    time: float         # 相对时间（分钟）
    influence: float    # 事件影响力
    event_type: str     # external / internal
    source: str = ""
    action_type: str = ""
    event_id: str = ""


@dataclass
class IntensityDebugInfo:
    """强度计算调试信息"""
    t: float
    mu: float
    internal_alpha: float
    internal_beta: float
    external_alpha: float
    external_beta: float
    circadian_hour: int
    circadian_factor: float
    raw_circadian: float
    circadian_strength: float
    activation_scale: float  # total_scale 值
    total_events: int
    external_events_count: int
    internal_events_count: int
    internal_contribution: float  # 内部原始累加
    external_contribution: float  # 外部原始累加
    internal_raw: float           # 同 internal_contribution（兼容）
    external_raw: float           # 同 external_contribution（兼容）
    base_intensity_before_circadian: float  # λ_norm（归一化后）
    final_intensity: float                  # λ_norm × circadian
    top_internal_contributions: List[Dict[str, Any]] = field(default_factory=list)
    top_external_contributions: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ActivationDebugInfo:
    """激活数计算调试信息"""
    t: float
    intensity: float
    total_agents: int
    time_granularity: int
    scale_factor: float
    expected_raw: float
    expected_clamped: float
    noise_added: float
    expected_with_noise: float
    poisson_sampled: int
    final_activated: int


class HawkesProcess:
    """归一化霍克斯点过程"""

    def __init__(self, mu=0.05, internal_alpha=0.003, internal_beta=0.05,
                 external_alpha=0.15, external_beta=0.003,
                 total_scale=200.0, circadian_strength=0.3,
                 action_weights=None, time_granularity=10,
                 circadian_curve=None, min_activation_noise=None,
                 # 以下为兼容旧配置的参数，传入时会被忽略
                 internal_weight=None, external_weight=None, activation_scale=None):
        self.mu = mu
        self.internal_alpha = internal_alpha
        self.internal_beta = internal_beta
        self.external_alpha = external_alpha
        self.external_beta = external_beta
        self.total_scale = total_scale
        self.circadian_strength = max(0.0, min(1.0, circadian_strength))
        self.time_granularity = time_granularity
        self.circadian_curve = circadian_curve or {}
        self.action_weights = action_weights or {
            'like': 0.01, 'repost': 1.0, 'repost_comment': 0.5,
            'short_comment': 0.1, 'long_comment': 0.2,
            'short_post': 0.2, 'long_post': 0.4
        }
        self.min_activation_noise = min_activation_noise or {
            'enabled': True, 'min_rate': 0.0, 'max_rate': 0.5
        }

        self.internal_events: List[HawkesEvent] = []
        self.external_events: List[HawkesEvent] = []
        self.added_external_event_ids: Set[str] = set()
        self.current_time: float = 0.0
        self.start_hour: int = 0

    # ── 公共接口 ────────────────────────────────────────────

    def set_start_hour(self, hour: int):
        self.start_hour = hour

    def add_external_event(self, time: float, influence: float,
                           source: str = "", event_id: str = None):
        if event_id is None:
            event_id = f"ext_{time}_{source}"
        if event_id in self.added_external_event_ids:
            return
        self.added_external_event_ids.add(event_id)
        self.external_events.append(HawkesEvent(
            time=time, influence=influence, event_type='external',
            source=source, event_id=event_id
        ))

    def add_internal_event(self, time: float, action_type: str,
                           source: str = "", user_influence: float = 1.0):
        influence = self.action_weights.get(action_type, 0.1) * user_influence
        self.internal_events.append(HawkesEvent(
            time=time, influence=influence, event_type='internal',
            source=source, action_type=action_type,
            event_id=f"int_{time}_{source}_{action_type}"
        ))

    def advance_time(self, delta_minutes: float):
        self.current_time += delta_minutes

    def clear_old_events(self, max_age_minutes: float = 1440):
        cutoff = self.current_time - max_age_minutes
        self.internal_events = [e for e in self.internal_events if e.time >= cutoff]
        self.external_events = [e for e in self.external_events if e.time >= cutoff]
        # 注意：不重置 added_external_event_ids，保留完整的去重记录
        # 否则已清理的旧事件会被 get_current_events (72h窗口) 重新返回并以新时间戳重新注入

    # ── 强度计算 ────────────────────────────────────────────

    def _circadian_blend(self, t: float) -> float:
        """昼夜节律混合: (1-s) + s × curve(h)"""
        if not self.circadian_curve:
            return 1.0
        h = (self.start_hour + int(t // 60)) % 24
        raw = self.circadian_curve.get(h, 1.0)
        return (1.0 - self.circadian_strength) + self.circadian_strength * raw

    def _compute_raw(self, t: float) -> tuple:
        """计算 λ_raw 及内部/外部分项"""
        internal_sum = sum(
            self.internal_alpha * e.influence * np.exp(-self.internal_beta * (t - e.time))
            for e in self.internal_events if e.time < t
        )
        external_sum = sum(
            self.external_alpha * e.influence * np.exp(-self.external_beta * (t - e.time))
            for e in self.external_events if e.time < t
        )
        return self.mu + internal_sum + external_sum, internal_sum, external_sum

    @staticmethod
    def _normalize(raw: float) -> float:
        """软饱和归一化: 1 - exp(-x)，将 [0,∞) 映射到 [0,1)"""
        return 1.0 - np.exp(-max(0.0, raw))

    def _normalize_separated(self, mu: float, int_sum: float, ext_sum: float) -> float:
        """分别归一化内部和外部贡献后叠加，避免互相压制
        
        原方案: 1-exp(-(μ+int+ext)) 导致内部活跃高时外部事件影响被压缩
        新方案: 分别归一化后叠加，确保外部事件始终有独立的影响力
        
        λ_int = 1 - exp(-(μ + int_sum))
        λ_ext = 1 - exp(-ext_sum)
        λ_norm = min(1.0, λ_int + λ_ext)
        """
        int_norm = 1.0 - np.exp(-max(0.0, mu + int_sum))
        ext_norm = 1.0 - np.exp(-max(0.0, ext_sum))
        return min(1.0, int_norm + ext_norm)

    def get_intensity(self, t: float) -> float:
        """返回归一化后的强度 λ(t) ∈ [0,1]"""
        raw, int_sum, ext_sum = self._compute_raw(t)
        norm = self._normalize_separated(self.mu, int_sum, ext_sum)
        return norm * self._circadian_blend(t)

    def get_intensity_with_debug(self, t: float, top_n: int = 3) -> Tuple[float, IntensityDebugInfo]:
        raw, int_sum, ext_sum = self._compute_raw(t)
        norm = self._normalize_separated(self.mu, int_sum, ext_sum)
        h = (self.start_hour + int(t // 60)) % 24
        raw_circ = self.circadian_curve.get(h, 1.0) if self.circadian_curve else 1.0
        circ = self._circadian_blend(t)
        final = norm * circ

        # top 贡献（仅在 debug 需要时计算）
        top_int, top_ext = [], []
        if top_n > 0:
            for e in self.internal_events:
                if e.time < t:
                    c = self.internal_alpha * e.influence * np.exp(-self.internal_beta * (t - e.time))
                    top_int.append({'time': e.time, 'delta_minutes': t - e.time,
                                    'action_type': e.action_type, 'source': e.source,
                                    'influence': e.influence, 'contribution': c})
            for e in self.external_events:
                if e.time < t:
                    c = self.external_alpha * e.influence * np.exp(-self.external_beta * (t - e.time))
                    top_ext.append({'time': e.time, 'delta_minutes': t - e.time,
                                    'source': e.source, 'influence': e.influence,
                                    'contribution': c})
            top_int.sort(key=lambda x: x['contribution'], reverse=True)
            top_ext.sort(key=lambda x: x['contribution'], reverse=True)

        debug = IntensityDebugInfo(
            t=t, mu=self.mu,
            internal_alpha=self.internal_alpha, internal_beta=self.internal_beta,
            external_alpha=self.external_alpha, external_beta=self.external_beta,
            circadian_hour=h, circadian_factor=circ,
            raw_circadian=raw_circ, circadian_strength=self.circadian_strength,
            activation_scale=self.total_scale,
            total_events=len(self.internal_events) + len(self.external_events),
            external_events_count=len([e for e in self.external_events if e.time < t]),
            internal_events_count=len([e for e in self.internal_events if e.time < t]),
            internal_contribution=int_sum, external_contribution=ext_sum,
            internal_raw=int_sum, external_raw=ext_sum,
            base_intensity_before_circadian=norm,
            final_intensity=final,
            top_internal_contributions=top_int[:top_n],
            top_external_contributions=top_ext[:top_n],
        )
        return final, debug

    # ── 激活数计算 ────────────────────────────────────────────

    def get_expected_activations(self, t: float, total_agents: int) -> int:
        """expected = total_scale × λ(t) × (granularity/60)，泊松采样"""
        intensity = self.get_intensity(t)
        expected = self.total_scale * intensity * (self.time_granularity / 60.0)
        if expected <= 0 and self.min_activation_noise.get('enabled', True):
            lo = self.min_activation_noise.get('min_rate', 0.0)
            hi = self.min_activation_noise.get('max_rate', 0.5)
            expected = np.random.uniform(lo, hi) * self.time_granularity
        expected = max(0, min(expected, total_agents))
        return min(np.random.poisson(expected), total_agents)

    def get_expected_activations_with_debug(self, t: float, total_agents: int,
                                            intensity_debug: IntensityDebugInfo = None
                                            ) -> Tuple[int, ActivationDebugInfo]:
        intensity = intensity_debug.final_intensity if intensity_debug else self.get_intensity(t)
        sf = self.time_granularity / 60.0
        raw = self.total_scale * intensity * sf
        clamped = max(0, min(raw, total_agents))

        noise = 0.0
        with_noise = clamped
        if clamped <= 0 and self.min_activation_noise.get('enabled', True):
            lo = self.min_activation_noise.get('min_rate', 0.0)
            hi = self.min_activation_noise.get('max_rate', 0.5)
            noise = np.random.uniform(lo, hi) * self.time_granularity
            with_noise = noise

        sampled = np.random.poisson(with_noise)
        final = min(sampled, total_agents)

        debug = ActivationDebugInfo(
            t=t, intensity=intensity, total_agents=total_agents,
            time_granularity=self.time_granularity, scale_factor=sf,
            expected_raw=raw, expected_clamped=clamped,
            noise_added=noise, expected_with_noise=with_noise,
            poisson_sampled=sampled, final_activated=final
        )
        return final, debug
