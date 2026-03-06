"""
配置Schema定义

霍克斯过程核心公式:
    λ_raw(t) = μ + Σ α_int × w_i × exp(-β_int × Δt) + Σ α_ext × inf_i × exp(-β_ext × Δt)
    λ_norm(t) = 1 - exp(-λ_raw)   ∈ [0, 1]
    expected  = total_scale × λ_norm × circadian × (granularity / 60)

时间单位统一为分钟。
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class HawkesInternalConfig:
    alpha: float = 0.003   # 内部激励强度
    beta: float = 0.05     # 内部衰减率 (1/min)

@dataclass
class HawkesExternalConfig:
    alpha: float = 0.15    # 外部激励强度
    beta: float = 0.003    # 外部衰减率 (1/min)

@dataclass
class MinActivationNoiseConfig:
    enabled: bool = True
    min_rate: float = 0.0
    max_rate: float = 0.5


@dataclass
class SimulationConfig:
    event_name: str
    simulation_title: str
    event_background: str = ""
    start_time: str = ""
    end_time: str = ""
    time_granularity: int = 10         # 分钟/步
    participant_scale: int = 0         # 0=使用全部用户
    use_llm: bool = True
    decision_mode: str = "ebdi"  # "ebdi" | "no_ebdi" | "cot" | "wo_belief" | "wo_desire" | "wo_intention" | "wo_hawkes"
    # 霍克斯过程
    hawkes_mu: float = 0.05
    hawkes_internal: HawkesInternalConfig = field(default_factory=HawkesInternalConfig)
    hawkes_external: HawkesExternalConfig = field(default_factory=HawkesExternalConfig)
    total_scale: float = 200.0         # 峰值时每小时目标激活数（线性旋钮）
    circadian_strength: float = 0.3    # 昼夜节律影响强度 [0,1]
    min_activation_noise: MinActivationNoiseConfig = field(default_factory=MinActivationNoiseConfig)
    # 推荐系统
    recommend_count: int = 5
    comment_count: int = 5
    homophily_weight: float = 0.4
    popularity_weight: float = 0.3
    recency_weight: float = 0.3
    exploration_rate: float = 0.2       # 推荐时随机探索比例，抑制信息茧房
    relation_weight: float = 0.5
    # 热搜
    hot_search_update_interval: int = 15
    hot_search_count: int = 50
    hot_search_min_mentions: int = 10
    hot_search_display_count: int = 5
    # 行为权重
    action_weights: Dict[str, float] = field(default_factory=lambda: {
        'like': 0.01, 'repost': 1.0, 'repost_comment': 0.5,
        'short_comment': 0.1, 'long_comment': 0.2,
        'short_post': 0.2, 'long_post': 0.4
    })
    # 昼夜节律
    circadian_curve: Dict[int, float] = field(default_factory=lambda: {
        "0": 0.4, "1": 0.3, "2": 0.2, "3": 0.2, "4": 0.2, "5": 0.3,
        "6": 0.5, "7": 0.7, "8": 0.9, "9": 1.0, "10": 1.0, "11": 0.95,
        "12": 0.85, "13": 0.8, "14": 0.85, "15": 0.9, "16": 0.95, "17": 1.0,
        "18": 1.1, "19": 1.15, "20": 1.2, "21": 1.2, "22": 1.1, "23": 0.8
    })


@dataclass
class DataConfig:
    users_file: str = "data/users.json"
    events_file: str = "data/events.json"
    initial_posts_file: str = "data/initial_posts.json"
    relations_file: str = "data/relations.json"


@dataclass
class LLMConfig:
    max_concurrent_requests: int = 10
    belief_llm_index: List[int] = field(default_factory=lambda: [0])
    desire_llm_index: List[int] = field(default_factory=lambda: [0])
    action_llm_index: List[int] = field(default_factory=lambda: [0])
    strategy_llm_index: List[int] = field(default_factory=lambda: [0])
    content_llm_index: List[int] = field(default_factory=lambda: [0])
    recommendation_llm_index: List[int] = field(default_factory=lambda: [0])
    other_llm_index: List[int] = field(default_factory=lambda: [0])
    use_local_embedding_model: bool = True
    local_embedding_model_path: str = ""
    embedding_dimension: int = 512
    embedding_device: str = "cpu"
    llm_api_configs: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Neo4jConfig:
    enabled: bool = False
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "password"


@dataclass
class OutputConfig:
    base_dir: str = "output"
    save_all_results: bool = True
    run_evaluation: bool = True


@dataclass
class DebugConfig:
    enabled: bool = False
    log_level: str = "INFO"
    llm_prompt_sample_rate: float = 0.05
