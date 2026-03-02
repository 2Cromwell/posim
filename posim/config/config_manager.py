"""配置管理器"""
import json
from pathlib import Path
from typing import Dict, Any
from .config_schema import (
    SimulationConfig, Neo4jConfig, OutputConfig, DataConfig, LLMConfig, DebugConfig,
    HawkesInternalConfig, HawkesExternalConfig, MinActivationNoiseConfig
)


class ConfigManager:

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.raw_config = self._load_config()
        self.simulation = self._parse_simulation()
        self.data = self._parse_data()
        self.llm = self._parse_llm()
        self.neo4j = self._parse_neo4j()
        self.output = self._parse_output()
        self.debug = self._parse_debug()

    def _load_config(self) -> Dict[str, Any]:
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _parse_simulation(self) -> SimulationConfig:
        s = self.raw_config.get('simulation', {})

        hi = s.get('hawkes_internal', {})
        he = s.get('hawkes_external', {})
        noise = s.get('min_activation_noise', {})

        return SimulationConfig(
            event_name=s.get('event_name', 'default'),
            simulation_title=s.get('simulation_title', 'default'),
            event_background=s.get('event_background', ''),
            start_time=s.get('start_time', ''),
            end_time=s.get('end_time', ''),
            time_granularity=s.get('time_granularity', 10),
            participant_scale=s.get('participant_scale', 0),
            use_llm=s.get('use_llm', True),
            decision_mode=s.get('decision_mode', 'ebdi'),
            hawkes_mu=s.get('hawkes_mu', 0.05),
            hawkes_internal=HawkesInternalConfig(
                alpha=hi.get('alpha', 0.003),
                beta=hi.get('beta', 0.05),
            ),
            hawkes_external=HawkesExternalConfig(
                alpha=he.get('alpha', 0.15),
                beta=he.get('beta', 0.003),
            ),
            total_scale=s.get('total_scale', 200.0),
            circadian_strength=s.get('circadian_strength', 0.3),
            min_activation_noise=MinActivationNoiseConfig(
                enabled=noise.get('enabled', True),
                min_rate=noise.get('min_rate', 0.0),
                max_rate=noise.get('max_rate', 0.5),
            ),
            recommend_count=s.get('recommend_count', 5),
            comment_count=s.get('comment_count', 5),
            homophily_weight=s.get('homophily_weight', 0.4),
            popularity_weight=s.get('popularity_weight', 0.3),
            recency_weight=s.get('recency_weight', 0.3),
            exploration_rate=s.get('exploration_rate', 0.2),
            relation_weight=s.get('relation_weight', 0.5),
            hot_search_update_interval=s.get('hot_search_update_interval', 15),
            hot_search_count=s.get('hot_search_count', 50),
            hot_search_min_mentions=s.get('hot_search_min_mentions', 10),
            hot_search_display_count=s.get('hot_search_display_count', 5),
            action_weights=s.get('action_weights', {}),
            circadian_curve=s.get('circadian_curve', {}),
        )

    def _parse_data(self) -> DataConfig:
        d = self.raw_config.get('data', {})
        return DataConfig(**{k: d[k] for k in DataConfig.__dataclass_fields__ if k in d})

    def _parse_llm(self) -> LLMConfig:
        l = self.raw_config.get('llm', {})
        return LLMConfig(**{k: l[k] for k in LLMConfig.__dataclass_fields__ if k in l})

    def _parse_neo4j(self) -> Neo4jConfig:
        n = self.raw_config.get('neo4j', {})
        return Neo4jConfig(**{k: n[k] for k in Neo4jConfig.__dataclass_fields__ if k in n})

    def _parse_output(self) -> OutputConfig:
        o = self.raw_config.get('output', {})
        return OutputConfig(**{k: o[k] for k in OutputConfig.__dataclass_fields__ if k in o})

    def _parse_debug(self) -> DebugConfig:
        d = self.raw_config.get('debug', {})
        return DebugConfig(**{k: d[k] for k in DebugConfig.__dataclass_fields__ if k in d})

    def get_data_dir(self) -> Path:
        return self.config_path.parent

    def get_file_path(self, file_key: str) -> Path:
        file_map = {
            'users': self.data.users_file,
            'events': self.data.events_file,
            'initial_posts': self.data.initial_posts_file,
            'relations': self.data.relations_file
        }
        return self.config_path.parent / file_map.get(file_key, '')
