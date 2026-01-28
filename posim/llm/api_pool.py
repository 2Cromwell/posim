"""
API资源池 - 管理多个大模型API，支持负载均衡和并发控制
"""
import asyncio
import random
import logging
import time
import os
import json
from typing import Dict, Any, Optional, List, Union
from collections import defaultdict
from .llm_client import LLMClient

logger = logging.getLogger(__name__)


class APIPool:
    """大模型API资源池，支持负载均衡和并发控制"""
    
    def __init__(self, llm_config: Union[Dict, Any], debug_config: Any = None, output_dir: str = None):
        """
        Args:
            llm_config: LLMConfig对象或字典
            debug_config: DebugConfig对象，用于控制日志打印
        """
        # 支持字典或dataclass
        if hasattr(llm_config, 'max_concurrent_requests'):
            self.config = llm_config
            self.max_concurrent = llm_config.max_concurrent_requests
            # 各模块LLM索引
            self.llm_indices = {
                'belief': llm_config.belief_llm_index,
                'desire': llm_config.desire_llm_index,
                'action': llm_config.action_llm_index,
                'strategy': llm_config.strategy_llm_index,
                'content': llm_config.content_llm_index,
                'recommendation': llm_config.recommendation_llm_index,
                'other': llm_config.other_llm_index
            }
            self.use_local_embedding = llm_config.use_local_embedding_model
            self.local_embedding_path = llm_config.local_embedding_model_path
            self.embedding_dimension = llm_config.embedding_dimension
            self.embedding_device = llm_config.embedding_device.strip()
            self.api_configs = llm_config.llm_api_configs
        else:
            self.config = llm_config
            self.max_concurrent = llm_config.get('max_concurrent_requests', 10)
            # 各模块LLM索引
            self.llm_indices = {
                'belief': llm_config.get('belief_llm_index', [0]),
                'desire': llm_config.get('desire_llm_index', [0]),
                'action': llm_config.get('action_llm_index', [0]),
                'strategy': llm_config.get('strategy_llm_index', [0]),
                'content': llm_config.get('content_llm_index', [0]),
                'recommendation': llm_config.get('recommendation_llm_index', [0]),
                'other': llm_config.get('other_llm_index', [0])
            }
            self.use_local_embedding = llm_config.get('use_local_embedding_model', True)
            self.local_embedding_path = llm_config.get('local_embedding_model_path', '')
            self.embedding_dimension = llm_config.get('embedding_dimension')
            self.embedding_device = llm_config.get('embedding_device', 'cpu').strip()
            self.api_configs = llm_config.get('llm_api_configs', [])
        
        # Debug配置
        self.debug_enabled = debug_config.enabled if debug_config else False
        self.llm_sample_rate = debug_config.llm_prompt_sample_rate if debug_config else 0.5
        self.output_dir = output_dir
        
        # 统计信息
        self._init_stats()
        
        self.clients: List[LLMClient] = []
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        self._init_clients()
        self._embedding_model = None
        logger.info(f"APIPool initialized: max_concurrent={self.max_concurrent}, clients={len(self.clients)}")

    def _init_clients(self):
        for cfg in self.api_configs:
            if cfg.get('enabled', True):
                client = LLMClient(
                    name=cfg.get('name', 'default'),
                    base_url=cfg.get('base_url', ''),
                    api_key=cfg.get('api_key', ''),
                    model=cfg.get('model', ''),
                    temperature=cfg.get('temperature', 0.7),
                    top_p=cfg.get('top_p', 0.9),
                    weight=cfg.get('weight', 1.0),
                    enabled=cfg.get('enabled', True)
                )
                self.clients.append(client)
    
    def _log_llm_call(self, query: str, system_prompt: str, response: str, purpose: str):
        """按概率记录LLM调用"""
        # 按配置的采样率决定是否打印
        if not self.debug_enabled:
            return
        if random.random() > self.llm_sample_rate:
            return
        
        sep = "\n" + "=" * 60 + "\n"
        logger.info(
            f"{sep}"
            f" LLM CALL [{purpose.upper()}] \n"
            f"{'-'*40}\n"
            f" SYSTEM:\n{system_prompt}\n"
            f"{'-'*40}\n"
            f" QUERY:\n{query}\n"
            f"{'-'*40}\n"
            f" RESPONSE:\n{response}"
            f"{sep}"
        )

    def _select_client(self, indices: List[int]) -> LLMClient:
        """基于权重随机选择客户端"""
        valid_clients = [self.clients[i] for i in indices if i < len(self.clients)]
        if not valid_clients:
            valid_clients = self.clients
        weights = [c.weight for c in valid_clients]
        return random.choices(valid_clients, weights=weights, k=1)[0]

    def _get_indices(self, purpose: str) -> List[int]:
        """根据用途获取LLM索引"""
        return self.llm_indices.get(purpose, self.llm_indices.get('other', [0]))

    async def async_query(self, messages: List[Dict[str, str]], purpose: str = 'other', 
                          hyper_params: Optional[Dict[str, Any]] = None) -> str:
        """带并发控制的异步查询"""
        indices = self._get_indices(purpose)
        client = self._select_client(indices)
        async with self.semaphore:
            return await client.async_query(messages, hyper_params)

    async def async_text_query(self, query: str, system_prompt: str = None, purpose: str = 'other',
                               hyper_params: Optional[Dict[str, Any]] = None) -> str:
        """简化的文本查询接口"""
        indices = self._get_indices(purpose)
        client = self._select_client(indices)
        start_time = time.time()
        success = True
        async with self.semaphore:
            try:
                response = await client.async_text_query(query, system_prompt, hyper_params)
            except Exception as e:
                success = False
                response = ""
                logger.warning(f"LLM call failed: {e}")
            
            elapsed = time.time() - start_time
            # 估算token数（简化：按字符数/4估算）
            tokens_in = (len(query) + len(system_prompt or '')) // 4
            tokens_out = len(response) // 4
            self._record_call(purpose, client.name, tokens_in, tokens_out, elapsed, success)
            self._log_llm_call(query, system_prompt or '', response, purpose)
            return response

    async def batch_query(self, queries: List[Dict[str, Any]], purpose: str = 'agent') -> List[str]:
        """批量并发查询"""
        tasks = [self.async_text_query(q['query'], q.get('system_prompt'), purpose, q.get('hyper_params')) 
                 for q in queries]
        return await asyncio.gather(*tasks)

    def get_embedding_model(self):
        """获取本地embedding模型（懒加载）"""
        if self._embedding_model is None and self.use_local_embedding:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(self.local_embedding_path, device=self.embedding_device)
        return self._embedding_model

    def encode(self, texts: List[str]) -> Any:
        """编码文本为向量"""
        model = self.get_embedding_model()
        return model.encode(texts, normalize_embeddings=True)

    def _init_stats(self):
        """初始化统计信息"""
        self.total_stats = {
            'total_calls': 0,
            'total_tokens_in': 0,
            'total_tokens_out': 0,
            'total_time': 0.0,
            'calls_by_purpose': defaultdict(int),
            'calls_by_client': defaultdict(int),
            'errors': 0,
            'start_time': time.time()
        }
        self.step_stats = {
            'calls': 0,
            'tokens_in': 0,
            'tokens_out': 0,
            'time': 0.0,
            'calls_by_purpose': defaultdict(int),
            'errors': 0
        }
        self.step_history = []
    
    def _record_call(self, purpose: str, client_name: str, tokens_in: int, tokens_out: int, 
                     elapsed: float, success: bool = True):
        """记录一次API调用"""
        # 步级统计
        self.step_stats['calls'] += 1
        self.step_stats['tokens_in'] += tokens_in
        self.step_stats['tokens_out'] += tokens_out
        self.step_stats['time'] += elapsed
        self.step_stats['calls_by_purpose'][purpose] += 1
        if not success:
            self.step_stats['errors'] += 1
        
        # 全局统计
        self.total_stats['total_calls'] += 1
        self.total_stats['total_tokens_in'] += tokens_in
        self.total_stats['total_tokens_out'] += tokens_out
        self.total_stats['total_time'] += elapsed
        self.total_stats['calls_by_purpose'][purpose] += 1
        self.total_stats['calls_by_client'][client_name] += 1
        if not success:
            self.total_stats['errors'] += 1

    def log_step_stats(self, step_num: int):
        """记录每步的API调用统计"""
        stats = {
            'step': step_num,
            'calls': self.step_stats['calls'],
            'tokens_in': self.step_stats['tokens_in'],
            'tokens_out': self.step_stats['tokens_out'],
            'time': round(self.step_stats['time'], 2),
            'calls_by_purpose': dict(self.step_stats['calls_by_purpose']),
            'errors': self.step_stats['errors']
        }
        self.step_history.append(stats)
        
        if self.step_stats['calls'] > 0:
            logger.info(f"Step {step_num} LLM: {stats['calls']} calls, "
                        f"{stats['tokens_in']}+{stats['tokens_out']} tokens, "
                        f"{stats['time']:.1f}s, errors={stats['errors']}")

    def log_final_stats(self):
        """记录最终的API调用统计"""
        elapsed = time.time() - self.total_stats['start_time']
        summary = {
            'total_calls': self.total_stats['total_calls'],
            'total_tokens_in': self.total_stats['total_tokens_in'],
            'total_tokens_out': self.total_stats['total_tokens_out'],
            'total_tokens': self.total_stats['total_tokens_in'] + self.total_stats['total_tokens_out'],
            'total_llm_time': round(self.total_stats['total_time'], 2),
            'total_elapsed_time': round(elapsed, 2),
            'errors': self.total_stats['errors'],
            'calls_by_purpose': dict(self.total_stats['calls_by_purpose']),
            'calls_by_client': dict(self.total_stats['calls_by_client']),
            'avg_tokens_per_call': round((self.total_stats['total_tokens_in'] + self.total_stats['total_tokens_out']) / 
                                         max(1, self.total_stats['total_calls']), 1),
            'avg_time_per_call': round(self.total_stats['total_time'] / max(1, self.total_stats['total_calls']), 3)
        }
        
        logger.info(f"\n{'='*60}\nLLM API Summary:\n"
                    f"  Total calls: {summary['total_calls']}\n"
                    f"  Total tokens: {summary['total_tokens']} (in:{summary['total_tokens_in']}, out:{summary['total_tokens_out']})\n"
                    f"  Avg tokens/call: {summary['avg_tokens_per_call']}\n"
                    f"  Total LLM time: {summary['total_llm_time']}s\n"
                    f"  Avg time/call: {summary['avg_time_per_call']}s\n"
                    f"  Errors: {summary['errors']}\n"
                    f"  By purpose: {summary['calls_by_purpose']}\n"
                    f"  By client: {summary['calls_by_client']}\n{'='*60}")
        
        # 保存到文件
        if self.output_dir:
            stats_file = os.path.join(self.output_dir, 'llm_stats.json')
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump({'summary': summary, 'step_history': self.step_history}, f, ensure_ascii=False, indent=2)
            logger.info(f"LLM stats saved to {stats_file}")

    def reset_step_stats(self):
        """重置每步的统计"""
        self.step_stats = {
            'calls': 0,
            'tokens_in': 0,
            'tokens_out': 0,
            'time': 0.0,
            'calls_by_purpose': defaultdict(int),
            'errors': 0
        }
