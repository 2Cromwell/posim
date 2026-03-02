# -*- coding: utf-8 -*-
"""
智能体行为机制验证（LLM驱动）

验证内容:
1. 行为序列稳定性 - 从心理认知视角判断行为逻辑
2. 人格稳定性 - 从行为推断固有属性的一致性（横向验证）
3. 提示稳定性 - 相同提示多次输出的一致性（纵向验证）
"""
import asyncio
import json
import random
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict

from ..base import BaseEvaluator
from ..utils import save_json

logger = logging.getLogger(__name__)

# ===== LLM评估提示词 =====
BEHAVIOR_SEQUENCE_PROMPT = """你是一个社会心理学和认知科学专家。请分析以下社交媒体用户的行为序列,从心理认知视角评估其行为逻辑是否合理。

用户身份: {identity}
心理信念: {beliefs}

行为序列(按时间排列):
{actions}

请从以下维度评估，并给出1-5分评分(1=非常不合理, 5=非常合理)及简要理由:
1. 情绪连贯性: 行为序列中的情绪变化是否符合心理学规律？
2. 立场一致性: 用户在事件中的立场是否保持基本一致或合理演变？
3. 行为合理性: 用户选择的行为类型(发博/转发/评论/点赞)是否符合其身份？
4. 内容逻辑性: 行为内容是否与事件发展逻辑一致？
5. 总体评分: 综合上述维度的总体评分。

请以JSON格式输出:
{{"emotion_coherence": {{"score": X, "reason": "..."}}, "stance_consistency": {{"score": X, "reason": "..."}}, "behavior_rationality": {{"score": X, "reason": "..."}}, "content_logic": {{"score": X, "reason": "..."}}, "overall": {{"score": X, "reason": "..."}}}}
"""

PERSONALITY_INFERENCE_PROMPT = """你是一个人格心理学专家。请根据以下社交媒体行为推断用户的人格特征。

用户行为:
- 行为类型: {action_type}
- 内容: {content}
- 情绪: {emotion}
- 立场: {stance}

请推断该用户在以下人格维度上的得分(1-5分):
1. 开放性(Openness): 对新事物的接受度
2. 尽责性(Conscientiousness): 做事的认真程度
3. 外向性(Extraversion): 社交活跃度
4. 宜人性(Agreeableness): 与人相处的友善度
5. 神经质(Neuroticism): 情绪不稳定程度

请以JSON格式输出:
{{"openness": X, "conscientiousness": X, "extraversion": X, "agreeableness": X, "neuroticism": X}}
"""

PROMPT_STABILITY_TEMPLATE = """你是一个社交媒体用户模拟器。请模拟以下用户在给定场景下的行为。

用户身份: {identity}
心理信念: {beliefs}
事件背景: {event_background}

当前时间点曝光到以下内容:
{exposed_content}

请模拟该用户会采取的行为(发博/转发/评论/点赞/不参与),并给出:
1. 行为类型
2. 内容(如有)
3. 情绪类型
4. 立场

以JSON格式输出:
{{"action_type": "...", "content": "...", "emotion": "...", "stance": "..."}}
"""


class AgentBehaviorEvaluator(BaseEvaluator):
    """智能体行为机制验证"""
    
    def __init__(self, output_dir: Path):
        super().__init__(output_dir / "agent_behavior", name="agent_behavior")
        self.sample_size = 200  # 随机采样用户数
    
    def evaluate(self, sim_data: Dict[str, Any], real_data: Optional[Dict[str, Any]] = None,
                 **kwargs) -> Dict[str, Any]:
        """执行智能体行为机制验证（需要LLM）"""
        self._log_section("智能体行为机制验证")
        
        api_pool = kwargs.get('api_pool')
        users_data = kwargs.get('users_data', [])
        event_background = kwargs.get('event_background', '')
        
        micro_results = sim_data.get('micro_results', [])
        
        if not micro_results:
            print("    ⚠️ 无模拟行为数据，跳过智能体行为验证")
            return {}
        
        # 按用户分组行为
        user_actions = defaultdict(list)
        for action in micro_results:
            uid = action.get('user_id', '')
            if uid:
                user_actions[uid].append(action)
        
        # 构建用户资料映射
        user_profiles = {}
        for u in users_data:
            uid = u.get('user_id', '')
            user_profiles[uid] = {
                'identity': u.get('identity_description', ''),
                'beliefs': u.get('psychological_beliefs', []),
                'username': u.get('username', ''),
                'agent_type': u.get('agent_type', 'citizen')
            }
        
        # 随机采样用户
        available_users = [uid for uid in user_actions if len(user_actions[uid]) >= 2]
        sample_users = random.sample(available_users, min(self.sample_size, len(available_users)))
        
        print(f"    采样 {len(sample_users)} 个用户进行行为机制验证")
        
        results = {}
        
        if api_pool:
            # 使用LLM进行验证
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 已在异步环境中
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self._evaluate_with_llm(
                            sample_users, user_actions, user_profiles,
                            api_pool, event_background
                        )
                    )
                    results = future.result()
            else:
                results = asyncio.run(
                    self._evaluate_with_llm(
                        sample_users, user_actions, user_profiles,
                        api_pool, event_background
                    )
                )
        else:
            # 无LLM时使用统计方法
            results = self._evaluate_statistical(sample_users, user_actions, user_profiles)
        
        self._save_results(results, "agent_behavior_metrics.json")
        self._print_summary(results)
        return results
    
    async def _evaluate_with_llm(self, sample_users: List[str],
                                   user_actions: Dict, user_profiles: Dict,
                                   api_pool, event_background: str) -> Dict:
        """使用LLM进行行为评估"""
        results = {
            'behavior_sequence': [],
            'personality_stability': [],
            'prompt_stability': [],
            'summary': {}
        }
        
        # 1. 行为序列稳定性
        print("    [1/3] 评估行为序列稳定性...")
        seq_queries = []
        for uid in sample_users:
            actions = user_actions[uid]
            profile = user_profiles.get(uid, {})
            actions_text = "\n".join([
                f"  [{a.get('time', '')}] {a.get('action_type', '')} | 情绪:{a.get('emotion', '')} | "
                f"立场:{a.get('stance', '')} | 内容:{(a.get('text', a.get('content', '')))[:80]}"
                for a in sorted(actions, key=lambda x: x.get('time', ''))
            ])
            query = BEHAVIOR_SEQUENCE_PROMPT.format(
                identity=profile.get('identity', '未知用户'),
                beliefs='\n'.join(profile.get('beliefs', ['无'])),
                actions=actions_text
            )
            seq_queries.append({'query': query, 'system_prompt': '请以JSON格式回答。'})
        
        seq_responses = await api_pool.batch_query(seq_queries, purpose='other')
        
        for i, (uid, response) in enumerate(zip(sample_users, seq_responses)):
            try:
                # 尝试解析JSON
                json_str = response.strip()
                if '```' in json_str:
                    json_str = json_str.split('```')[1].replace('json', '').strip()
                scores = json.loads(json_str)
                results['behavior_sequence'].append({
                    'user_id': uid,
                    'username': user_profiles.get(uid, {}).get('username', ''),
                    'scores': scores
                })
            except:
                results['behavior_sequence'].append({
                    'user_id': uid, 'scores': None, 'raw_response': response[:200]
                })
        
        # 2. 人格稳定性（横向验证）
        print("    [2/3] 评估人格稳定性...")
        personality_queries = []
        personality_user_map = []  # (uid, action_idx)
        
        for uid in sample_users[:200]:  # 每用户多次推断
            actions = user_actions[uid]
            for action in actions[:3]:  # 取前3个行为
                query = PERSONALITY_INFERENCE_PROMPT.format(
                    action_type=action.get('action_type', ''),
                    content=(action.get('text', action.get('content', '')))[:200],
                    emotion=action.get('emotion', ''),
                    stance=action.get('stance', '')
                )
                personality_queries.append({'query': query, 'system_prompt': '请以JSON格式回答。'})
                personality_user_map.append(uid)
        
        if personality_queries:
            pers_responses = await api_pool.batch_query(personality_queries, purpose='other')
            
            user_personalities = defaultdict(list)
            for uid, response in zip(personality_user_map, pers_responses):
                try:
                    json_str = response.strip()
                    if '```' in json_str:
                        json_str = json_str.split('```')[1].replace('json', '').strip()
                    pers = json.loads(json_str)
                    user_personalities[uid].append(pers)
                except:
                    pass
            
            # 计算人格稳定性（各维度的标准差）
            for uid, pers_list in user_personalities.items():
                if len(pers_list) >= 2:
                    dims = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
                    stability = {}
                    for dim in dims:
                        values = [p.get(dim, 3) for p in pers_list if dim in p]
                        if values:
                            stability[dim] = {
                                'mean': float(np.mean(values)),
                                'std': float(np.std(values)),
                                'cv': float(np.std(values) / max(np.mean(values), 0.01))
                            }
                    
                    avg_cv = np.mean([v['cv'] for v in stability.values()]) if stability else 1.0
                    results['personality_stability'].append({
                        'user_id': uid,
                        'username': user_profiles.get(uid, {}).get('username', ''),
                        'dimensions': stability,
                        'overall_stability': float(1 - min(avg_cv, 1.0)),
                        'inference_count': len(pers_list)
                    })
        
        # 3. 提示稳定性（纵向验证）
        print("    [3/3] 评估提示稳定性...")
        stability_queries = []
        stability_users = sample_users[:200]
        repeats = 3
        
        for uid in stability_users:
            profile = user_profiles.get(uid, {})
            actions = user_actions[uid]
            # 构造一个固定场景
            exposed = actions[0].get('text', actions[0].get('content', ''))[:200] if actions else '热点事件讨论中'
            query = PROMPT_STABILITY_TEMPLATE.format(
                identity=profile.get('identity', '普通网友'),
                beliefs='\n'.join(profile.get('beliefs', ['关注社会热点'])),
                event_background=event_background[:200],
                exposed_content=exposed
            )
            for _ in range(repeats):
                stability_queries.append({
                    'query': query,
                    'system_prompt': '请以JSON格式回答。',
                    'hyper_params': {'temperature': 0.7}
                })
        
        if stability_queries:
            stab_responses = await api_pool.batch_query(stability_queries, purpose='other')
            
            idx = 0
            for uid in stability_users:
                user_responses = []
                for _ in range(repeats):
                    if idx < len(stab_responses):
                        try:
                            json_str = stab_responses[idx].strip()
                            if '```' in json_str:
                                json_str = json_str.split('```')[1].replace('json', '').strip()
                            parsed = json.loads(json_str)
                            user_responses.append(parsed)
                        except:
                            user_responses.append({'raw': stab_responses[idx][:100]})
                    idx += 1
                
                # 计算一致性
                action_types = [r.get('action_type', '') for r in user_responses if 'action_type' in r]
                emotions = [r.get('emotion', '') for r in user_responses if 'emotion' in r]
                stances = [r.get('stance', '') for r in user_responses if 'stance' in r]
                
                action_consistency = len(set(action_types)) / max(len(action_types), 1) if action_types else 0
                emotion_consistency = len(set(emotions)) / max(len(emotions), 1) if emotions else 0
                stance_consistency = len(set(stances)) / max(len(stances), 1) if stances else 0
                
                # 一致性越高，unique/total越低
                results['prompt_stability'].append({
                    'user_id': uid,
                    'username': user_profiles.get(uid, {}).get('username', ''),
                    'repeats': repeats,
                    'action_type_consistency': float(1 - (action_consistency - 1/max(len(action_types), 1)) if action_types else 0),
                    'emotion_consistency': float(1 - (emotion_consistency - 1/max(len(emotions), 1)) if emotions else 0),
                    'stance_consistency': float(1 - (stance_consistency - 1/max(len(stances), 1)) if stances else 0),
                    'responses': user_responses
                })
        
        # 汇总
        results['summary'] = self._compute_summary(results)
        return results
    
    def _evaluate_statistical(self, sample_users, user_actions, user_profiles) -> Dict:
        """统计方法验证（无LLM时使用）"""
        results = {
            'behavior_sequence': [],
            'personality_stability': [],
            'prompt_stability': [],
            'summary': {}
        }
        
        for uid in sample_users:
            actions = user_actions[uid]
            profile = user_profiles.get(uid, {})
            
            # 统计行为序列的基本指标
            emotions = [a.get('emotion', 'Neutral') for a in actions]
            stances = [a.get('stance', '') for a in actions]
            action_types = [a.get('action_type', '') for a in actions]
            
            # 情绪变化频率
            emotion_changes = sum(1 for i in range(1, len(emotions)) if emotions[i] != emotions[i-1])
            emotion_change_rate = emotion_changes / max(len(emotions) - 1, 1)
            
            # 立场一致性
            unique_stances = len(set(s for s in stances if s))
            stance_consistency = 1.0 / max(unique_stances, 1)
            
            # 行为类型多样性
            type_diversity = len(set(action_types)) / max(len(action_types), 1)
            
            results['behavior_sequence'].append({
                'user_id': uid,
                'username': profile.get('username', ''),
                'action_count': len(actions),
                'emotion_change_rate': float(emotion_change_rate),
                'stance_consistency': float(stance_consistency),
                'type_diversity': float(type_diversity),
                'scores': {
                    'emotion_coherence': {'score': max(1, 5 - emotion_change_rate * 5)},
                    'stance_consistency': {'score': stance_consistency * 5},
                    'behavior_rationality': {'score': 3.0},
                    'content_logic': {'score': 3.0},
                    'overall': {'score': (max(1, 5 - emotion_change_rate * 5) + stance_consistency * 5 + 6) / 4}
                }
            })
        
        results['summary'] = self._compute_summary(results)
        return results
    
    def _compute_summary(self, results: Dict) -> Dict:
        """计算汇总统计"""
        summary = {}
        
        # 行为序列评分汇总
        seq_scores = []
        for entry in results.get('behavior_sequence', []):
            scores = entry.get('scores', {})
            if scores and isinstance(scores, dict):
                overall = scores.get('overall', {})
                if isinstance(overall, dict) and 'score' in overall:
                    seq_scores.append(overall['score'])
        
        if seq_scores:
            summary['behavior_sequence'] = {
                'avg_score': float(np.mean(seq_scores)),
                'std_score': float(np.std(seq_scores)),
                'min_score': float(np.min(seq_scores)),
                'max_score': float(np.max(seq_scores)),
                'sample_count': len(seq_scores)
            }
        
        # 人格稳定性汇总
        stability_scores = [e.get('overall_stability', 0) for e in results.get('personality_stability', [])
                           if 'overall_stability' in e]
        if stability_scores:
            summary['personality_stability'] = {
                'avg_stability': float(np.mean(stability_scores)),
                'std_stability': float(np.std(stability_scores)),
                'sample_count': len(stability_scores)
            }
        
        # 提示稳定性汇总
        prompt_scores = []
        for e in results.get('prompt_stability', []):
            avg = np.mean([e.get('action_type_consistency', 0),
                          e.get('emotion_consistency', 0),
                          e.get('stance_consistency', 0)])
            prompt_scores.append(avg)
        if prompt_scores:
            summary['prompt_stability'] = {
                'avg_consistency': float(np.mean(prompt_scores)),
                'std_consistency': float(np.std(prompt_scores)),
                'sample_count': len(prompt_scores)
            }
        
        return summary
    
    def _print_summary(self, results: Dict):
        """打印评估摘要"""
        summary = results.get('summary', {})
        
        seq = summary.get('behavior_sequence', {})
        if seq:
            print(f"    ✅ 行为序列稳定性: 平均分={seq.get('avg_score', 0):.2f}/5.0 "
                  f"(样本数={seq.get('sample_count', 0)})")
        
        pers = summary.get('personality_stability', {})
        if pers:
            print(f"    ✅ 人格稳定性: 平均稳定度={pers.get('avg_stability', 0):.2%} "
                  f"(样本数={pers.get('sample_count', 0)})")
        
        prompt = summary.get('prompt_stability', {})
        if prompt:
            print(f"    ✅ 提示稳定性: 平均一致度={prompt.get('avg_consistency', 0):.2%} "
                  f"(样本数={prompt.get('sample_count', 0)})")
