"""
配置Schema定义
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class SimulationConfig:
    """仿真配置"""
    event_name: str
    simulation_title: str
    event_background: str = ""  # 事件背景描述，用于注入系统提示词
    start_time: str = ""
    end_time: str = ""
    time_granularity: int = 10  # 时间粒度（分钟），默认10分钟一轮
    participant_scale: int = 0  # 参与规模，0表示使用全部用户
    use_llm: bool = True  # 是否使用LLM进行决策，False则使用规则驱动
    hawkes_mu: float = 0.1
    hawkes_alpha: float = 0.5
    hawkes_beta: float = 1.0
    recommend_count: int = 5
    comment_count: int = 5
    homophily_weight: float = 0.4
    popularity_weight: float = 0.3
    recency_weight: float = 0.3
    relation_weight: float = 0.5  # 关系推荐权重
    hot_search_update_interval: int = 15
    hot_search_count: int = 50
    action_weights: Dict[str, float] = field(default_factory=lambda: {
        'like': 0.1, 'repost': 1.0, 'repost_comment': 0.8,
        'short_comment': 0.3, 'long_comment': 0.5,
        'short_post': 0.6, 'long_post': 0.8
    })
    circadian_curve: Dict[int, float] = field(default_factory=lambda: {
      "0": 0.4, "1": 0.3, "2": 0.2, "3": 0.2, "4": 0.2, "5": 0.3,
      "6": 0.5, "7": 0.7, "8": 0.9, "9": 1.0, "10": 1.0, "11": 0.95,
      "12": 0.85, "13": 0.8, "14": 0.85, "15": 0.9, "16": 0.95, "17": 1.0,
      "18": 1.1, "19": 1.15, "20": 1.2, "21": 1.2, "22": 1.1, "23": 0.8
    })
    event_influence_scale: float = 1.0



@dataclass
class DataConfig:
    """数据路径配置"""
    users_file: str = "data/users.json"
    events_file: str = "data/events.json"
    initial_posts_file: str = "data/initial_posts.json"
    relations_file: str = "data/relations.json"


@dataclass
class LLMConfig:
    """大模型配置"""
    max_concurrent_requests: int = 10
    # 各模块使用的LLM索引
    belief_llm_index: List[int] = field(default_factory=lambda: [0])
    desire_llm_index: List[int] = field(default_factory=lambda: [0])
    action_llm_index: List[int] = field(default_factory=lambda: [0])
    strategy_llm_index: List[int] = field(default_factory=lambda: [0])
    content_llm_index: List[int] = field(default_factory=lambda: [0])
    recommendation_llm_index: List[int] = field(default_factory=lambda: [0])
    other_llm_index: List[int] = field(default_factory=lambda: [0])
    # Embedding配置
    use_local_embedding_model: bool = True
    local_embedding_model_path: str = ""
    embedding_dimension: int = 512
    embedding_device: str = "cpu"
    llm_api_configs: List[Dict[str, Any]] = field(default_factory=list)


@dataclass  
class Neo4jConfig:
    """Neo4j数据库配置"""
    enabled: bool = False
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "password"


@dataclass
class OutputConfig:
    """输出配置"""
    base_dir: str = "output"
    save_all_results: bool = True
    run_evaluation: bool = True


@dataclass
class DebugConfig:
    """调试配置"""
    enabled: bool = False
    log_level: str = "INFO"
    llm_prompt_sample_rate: float = 0.05  # 5%概率打印LLM提示词
