"""
配置管理器 - 加载和管理仿真配置
"""
import json
from pathlib import Path
from typing import Dict, Any
from .config_schema import SimulationConfig, Neo4jConfig, OutputConfig, DataConfig, LLMConfig, DebugConfig


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.raw_config = self._load_config()
        self.simulation = self._parse_simulation_config()
        self.data = self._parse_data_config()
        self.llm = self._parse_llm_config()
        self.neo4j = self._parse_neo4j_config()
        self.output = self._parse_output_config()
        self.debug = self._parse_debug_config()

    def _load_config(self) -> Dict[str, Any]:
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _parse_simulation_config(self) -> SimulationConfig:
        sim = self.raw_config.get('simulation', {})
        return SimulationConfig(
            event_name=sim.get('event_name', 'default_event'),
            simulation_title=sim.get('simulation_title', 'default_sim'),
            event_background=sim.get('event_background', ''),
            start_time=sim.get('start_time', '2024-01-01T00:00:00'),
            end_time=sim.get('end_time', '2024-01-02T00:00:00'),
            time_granularity=sim.get('time_granularity', 1),
            participant_scale=sim.get('participant_scale', 0),
            use_llm=sim.get('use_llm', True),
            hawkes_mu=sim.get('hawkes_mu', 0.1),
            hawkes_alpha=sim.get('hawkes_alpha', 0.5),
            hawkes_beta=sim.get('hawkes_beta', 1.0),
            recommend_count=sim.get('recommend_count', 5),
            comment_count=sim.get('comment_count', 5),
            homophily_weight=sim.get('homophily_weight', 0.4),
            popularity_weight=sim.get('popularity_weight', 0.3),
            recency_weight=sim.get('recency_weight', 0.3),
            relation_weight=sim.get('relation_weight', 0.5),
            hot_search_update_interval=sim.get('hot_search_update_interval', 15),
            hot_search_count=sim.get('hot_search_count', 50),
            action_weights=sim.get('action_weights', {
                'like': 0.1, 'repost': 1.0, 'repost_comment': 0.8,
                'short_comment': 0.3, 'long_comment': 0.5,
                'short_post': 0.6, 'long_post': 0.8
            }),
            circadian_curve=sim.get('circadian_curve', {}),
            event_influence_scale=sim.get('event_influence_scale', 1.0)
        )

    def _parse_data_config(self) -> DataConfig:
        data = self.raw_config.get('data', {})
        return DataConfig(
            users_file=data.get('users_file', 'data/users.json'),
            events_file=data.get('events_file', 'data/events.json'),
            initial_posts_file=data.get('initial_posts_file', 'data/initial_posts.json'),
            relations_file=data.get('relations_file', 'data/relations.json')
        )
    
    def _parse_llm_config(self) -> LLMConfig:
        llm = self.raw_config.get('llm', {})
        return LLMConfig(
            max_concurrent_requests=llm.get('max_concurrent_requests', 10),
            belief_llm_index=llm.get('belief_llm_index', [0]),
            desire_llm_index=llm.get('desire_llm_index', [0]),
            action_llm_index=llm.get('action_llm_index', [0]),
            strategy_llm_index=llm.get('strategy_llm_index', [0]),
            content_llm_index=llm.get('content_llm_index', [0]),
            recommendation_llm_index=llm.get('recommendation_llm_index', [0]),
            other_llm_index=llm.get('other_llm_index', [0]),
            use_local_embedding_model=llm.get('use_local_embedding_model', True),
            local_embedding_model_path=llm.get('local_embedding_model_path', ''),
            embedding_dimension=llm.get('embedding_dimension', 512),
            embedding_device=llm.get('embedding_device', 'cuda'),
            llm_api_configs=llm.get('llm_api_configs', [])
        )

    def _parse_neo4j_config(self) -> Neo4jConfig:
        neo = self.raw_config.get('neo4j', {})
        return Neo4jConfig(
            enabled=neo.get('enabled', False),
            uri=neo.get('uri', 'bolt://localhost:7687'),
            user=neo.get('user', 'neo4j'),
            password=neo.get('password', 'password')
        )

    def _parse_output_config(self) -> OutputConfig:
        out = self.raw_config.get('output', {})
        return OutputConfig(
            base_dir=out.get('base_dir', 'output'),
            save_all_results=out.get('save_all_results', True),
            run_evaluation=out.get('run_evaluation', True)
        )
    
    def _parse_debug_config(self) -> DebugConfig:
        dbg = self.raw_config.get('debug', {})
        return DebugConfig(
            enabled=dbg.get('enabled', False),
            log_level=dbg.get('log_level', 'INFO'),
            llm_prompt_sample_rate=dbg.get('llm_prompt_sample_rate', 0.05)
        )

    def get_data_dir(self) -> Path:
        """获取数据目录"""
        return self.config_path.parent
    
    def get_file_path(self, file_key: str) -> Path:
        """获取数据文件完整路径"""
        file_map = {
            'users': self.data.users_file,
            'events': self.data.events_file,
            'initial_posts': self.data.initial_posts_file,
            'relations': self.data.relations_file
        }
        return self.config_path.parent / file_map.get(file_key, '')
