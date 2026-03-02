# -*- coding: utf-8 -*-
"""实验配置"""
import os
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class DecisionMethod(Enum):
    DIRECT_NOTHINK = "direct_nothink"
    DIRECT_THINK = "direct_think"
    COT = "cot"
    EBDI = "ebdi"


@dataclass
class LLMModelConfig:
    name: str
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.7
    top_p: float = 0.9
    max_concurrent: int = 30
    timeout: int = 120


@dataclass
class ExperimentConfig:
    event_background: str = ""
    num_agents: int = 100
    num_rounds: int = 12
    
    robustness_decision_samples: int = 2
    robustness_repeat_count: int = 5

    methods: List[str] = field(default_factory=lambda: [
        "direct_nothink", "direct_think", "cot", "ebdi"
    ])

    nothink_model: LLMModelConfig = None
    think_model: LLMModelConfig = None
    eval_model: LLMModelConfig = None

    data_dir: str = ""
    output_dir: str = ""
    
    log_level: str = "INFO"

    @classmethod
    def from_project_config(cls, config_path: str, output_base: str = None):
        """从项目config.json加载配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)

        sim_cfg = cfg.get('simulation', {})
        llm_cfg = cfg.get('llm', {})

        enabled_apis = [a for a in llm_cfg.get('llm_api_configs', []) if a.get('enabled', True)]
        if not enabled_apis:
            raise ValueError("No enabled LLM API found in config")

        base_api = enabled_apis[0]
        base_url = base_api['base_url']
        api_key = base_api['api_key']

        nothink = LLMModelConfig(
            name="qwen2.5-14b",
            base_url=base_url,
            api_key=api_key,
            model="Qwen/Qwen2.5-14B-Instruct",
            temperature=0.7,
            top_p=0.9,
            max_concurrent=llm_cfg.get('max_concurrent_requests', 30),
        )
        think = LLMModelConfig(
            name="qwen3-14b",
            base_url=base_url,
            api_key=api_key,
            model="Qwen/Qwen3-14B",
            temperature=0.7,
            top_p=0.9,
            max_concurrent=llm_cfg.get('max_concurrent_requests', 30),
        )
        eval_model = LLMModelConfig(
            name="eval-qwen2.5-14b",
            base_url=base_url,
            api_key=api_key,
            model="Qwen/Qwen2.5-14B-Instruct",
            temperature=0.3,
            top_p=0.9,
            max_concurrent=llm_cfg.get('max_concurrent_requests', 30),
        )

        config_dir = os.path.dirname(os.path.abspath(config_path))
        data_dir = os.path.join(config_dir, cfg.get('data', {}).get('users_file', 'data/users.json'))
        data_dir = os.path.dirname(data_dir)

        if output_base is None:
            output_base = os.path.join(config_dir, 'output', 'micro_validation')

        return cls(
            event_background=sim_cfg.get('event_background', ''),
            nothink_model=nothink,
            think_model=think,
            eval_model=eval_model,
            data_dir=data_dir,
            output_dir=output_base,
        )
