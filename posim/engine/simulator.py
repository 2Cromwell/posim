"""
仿真核心引擎 - 整合各模块执行仿真
"""
import asyncio
import random
import logging
import time
import json
import numpy as np
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """性能监控指标"""
    step_times: List[float] = field(default_factory=list)
    llm_times: List[float] = field(default_factory=list)
    agent_execution_times: List[float] = field(default_factory=list)
    emotion_contagion_times: List[float] = field(default_factory=list)
    total_llm_calls: int = 0
    
    def record_step(self, duration: float):
        self.step_times.append(duration)
    
    def record_llm(self, duration: float):
        self.llm_times.append(duration)
        self.total_llm_calls += 1
    
    def record_agent_execution(self, duration: float):
        self.agent_execution_times.append(duration)
    
    def record_emotion_contagion(self, duration: float):
        self.emotion_contagion_times.append(duration)
    
    def get_summary(self) -> Dict[str, Any]:
        def safe_stats(times):
            if not times:
                return {'avg': 0, 'max': 0, 'min': 0, 'total': 0}
            return {
                'avg': sum(times) / len(times),
                'max': max(times),
                'min': min(times),
                'total': sum(times)
            }
        return {
            'step': safe_stats(self.step_times),
            'llm': safe_stats(self.llm_times),
            'agent_execution': safe_stats(self.agent_execution_times),
            'emotion_contagion': safe_stats(self.emotion_contagion_times),
            'total_llm_calls': self.total_llm_calls,
            'total_steps': len(self.step_times)
        }


@dataclass
class StepSignals:
    """每一步的实时信号数据"""
    step: int
    time: str
    elapsed_minutes: float
    # 智能体相关
    activated_count: int
    total_agents: int
    # 行为相关
    actions_count: int
    post_count: int  # 发博数
    repost_count: int  # 转发数
    comment_count: int  # 评论数
    like_count: int  # 点赞数
    actions_by_type: Dict[str, int]
    # 霍克斯强度
    hawkes_intensity: float
    hawkes_mu: float
    hawkes_internal_contribution: float
    hawkes_external_contribution: float
    hawkes_internal_raw: float
    hawkes_external_raw: float
    hawkes_circadian_factor: float
    # 事件统计
    internal_events_count: int
    external_events_count: int
    # 噪声
    noise_added: float
    # 外部事件
    external_events_triggered: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


from .hawkes_process import HawkesProcess, IntensityDebugInfo, ActivationDebugInfo
from .time_engine import TimeEngine
from ..agents.base_agent import BaseAgent
from ..agents.citizen_agent import CitizenAgent
from ..agents.kol_agent import KOLAgent
from ..agents.media_agent import MediaAgent
from ..agents.government_agent import GovernmentAgent
from ..environment.recommendation import RecommendationSystem
from ..environment.hot_search import HotSearchManager
from ..environment.social_network import SocialNetwork
from ..environment.event_queue import EventQueue, EventType, ExternalEvent
from ..config.config_manager import ConfigManager
from ..llm.api_pool import APIPool


class Simulator:
    """仿真核心引擎"""
    
    def __init__(self, config_manager: ConfigManager, api_pool: APIPool):
        self.config = config_manager.simulation
        self.api_pool = api_pool
        self.decision_mode = getattr(self.config, 'decision_mode', 'ebdi')
        
        # 初始化时间引擎
        self.time_engine = TimeEngine(
            self.config.start_time, 
            self.config.end_time,
            self.config.time_granularity
        )
        
        circadian_curve = {int(k): v for k, v in self.config.circadian_curve.items()}
        hi = self.config.hawkes_internal
        he = self.config.hawkes_external
        mn = self.config.min_activation_noise
        
        self.hawkes = HawkesProcess(
            mu=self.config.hawkes_mu,
            internal_alpha=hi.alpha, internal_beta=hi.beta,
            external_alpha=he.alpha, external_beta=he.beta,
            total_scale=self.config.total_scale,
            circadian_strength=self.config.circadian_strength,
            action_weights=self.config.action_weights,
            time_granularity=self.config.time_granularity,
            circadian_curve=circadian_curve,
            min_activation_noise={
                'enabled': mn.enabled, 'min_rate': mn.min_rate, 'max_rate': mn.max_rate
            },
        )
        # 设置仿真开始时间的小时数（用于昼夜节律）
        start_dt = datetime.fromisoformat(self.config.start_time)
        self.hawkes.set_start_hour(start_dt.hour)
        
        # 初始化环境模块
        self.recommendation = RecommendationSystem(api_pool, self.config)
        self.hot_search = HotSearchManager(self.config)
        self.social_network = SocialNetwork(config_manager.neo4j)
        self.event_queue = EventQueue()
        
        # 智能体池
        self.agents: Dict[str, BaseAgent] = {}
        
        # 事件背景（用于注入系统提示词）
        self.event_background = self.config.event_background
        
        # 仿真统计
        self.stats = {
            'total_actions': 0,
            'actions_by_type': {},
            'active_agents_per_step': [],
            'intensity_history': [],
            'actions_per_step': []
        }
        
        # 上一步的行为统计（用于调试日志）
        self.last_step_stats = {
            'step': 0,
            'actions_count': 0,
            'post_count': 0,  # short_post + long_post
            'repost_count': 0,  # repost + repost_comment
            'comment_count': 0,  # short_comment + long_comment
            'like_count': 0,
            'post_repost_comment_sum': 0,  # 发博 + 转发 + 评论的总和
            'actions_by_type': {}
        }
        
        # 性能监控
        self.perf_metrics = PerformanceMetrics()
        
        # 回调函数
        self.step_callback = None
        self.action_callback = None
        
        # 实时信号回调（用于WebSocket推送）
        self.signal_callback: Optional[Callable[[StepSignals], None]] = None
        
        # 信号历史（用于前端展示）
        self.signals_history: List[StepSignals] = []
    
    def load_agents(self, agents_data: List[Dict]):
        """加载智能体，支持参与规模采样"""
        agent_classes = {
            'citizen': CitizenAgent,
            'kol': KOLAgent,
            'media': MediaAgent,
            'government': GovernmentAgent
        }
        
        # 根据 participant_scale 按分布比例采样用户
        participant_scale = getattr(self.config, 'participant_scale', 0)
        if participant_scale > 0 and participant_scale < len(agents_data):
            # 按类型分组
            type_groups = {}
            for d in agents_data:
                agent_type = d.get('agent_type', 'citizen')
                if agent_type not in type_groups:
                    type_groups[agent_type] = []
                type_groups[agent_type].append(d)
            
            # 计算各类型原始比例并按比例采样
            total_original = len(agents_data)
            sampled_agents = []
            remaining_quota = participant_scale
            
            for agent_type, agents_list in type_groups.items():
                # 按原始比例计算目标数量
                ratio = len(agents_list) / total_original
                target_count = max(1, int(participant_scale * ratio))  # 至少保留1个
                target_count = min(target_count, len(agents_list), remaining_quota)
                
                if target_count > 0:
                    sampled = random.sample(agents_list, target_count)
                    sampled_agents.extend(sampled)
                    remaining_quota -= target_count
                    logger.debug(f"  采样 {agent_type}: {len(agents_list)} -> {target_count}")
            
            agents_data = sampled_agents
            logger.info(f"参与规模按比例采样: {len(agents_data)} 用户")
        
        for data in agents_data:
            agent_type = data.get('agent_type', 'citizen')
            cls = agent_classes.get(agent_type, CitizenAgent)
            agent = cls.from_dict(data, self.api_pool, self.event_background)
            self.agents[agent.profile.user_id] = agent
        
        logger.info(f"智能体加载完成: {len(self.agents)} 个")
    
    def load_events(self, events_data: List[Dict]):
        """加载事件队列"""
        self.event_queue.load_events(events_data)
    
    def load_relations(self, relations_data: List[Dict]):
        """加载关注关系数据"""
        # 加载到推荐系统
        self.recommendation.set_relations(relations_data)
        
        # 过滤有效关系
        valid_relations = []
        for rel in relations_data:
            follower_id = rel.get('follower_id', '')
            following_id = rel.get('following_id', '')
            if follower_id and following_id:
                if follower_id in self.agents and following_id in self.agents:
                    valid_relations.append({
                        'follower_id': follower_id,
                        'following_id': following_id
                    })
        
        # 批量加载到社交网络
        self.social_network.add_follows_batch(valid_relations)
        
        logger.info(f"加载关注关系: {len(valid_relations)} 条 (原始 {len(relations_data)} 条)")
    
    def load_initial_posts(self, posts_data: List[Dict]):
        """加载初始博文（过滤超过start_time的博文）"""
        if posts_data:
            start_time = self.time_engine.state.start_time
            valid_posts = []
            filtered_count = 0
            
            for post in posts_data:
                post_time_str = post.get('time', '')
                if post_time_str:
                    try:
                        post_time = datetime.fromisoformat(post_time_str)
                        if post_time <= start_time:
                            valid_posts.append(post)
                        else:
                            filtered_count += 1
                    except ValueError:
                        logger.warning(f"初始博文时间格式错误: {post_time_str}")
                        valid_posts.append(post)
                else:
                    valid_posts.append(post)
            
            if valid_posts:
                self.recommendation.add_posts_batch(valid_posts)
                logger.info(f"批量加载初始博文: {len(valid_posts)} 条 (过滤 {filtered_count} 条超过start_time的博文)")
    
    async def run_step(self) -> Dict[str, Any]:
        """执行一步仿真"""
        step_start = time.time()
        
        current_time = self.time_engine.current_time_str
        elapsed = self.time_engine.elapsed_minutes
        step_num = self.time_engine.state.step
        
        logger.debug(f"\n{'='*50}\n📍 Step {step_num} | Time: {current_time}\n{'='*50}")
        
        # 输出上一轮行为统计（如果不是第一步）
        if step_num > 1 and self.last_step_stats['step'] > 0:
            self._log_last_step_stats()
            
        # 1. 获取当前外部全局事件（仅global_broadcast加入霍克斯过程）
        global_events = self.event_queue.get_current_events(current_time, window_minutes=self.config.time_granularity)
        # 智能体接收历史近5条事件作为上下文，而非仅本步事件
        agent_events = self.event_queue.get_recent_events(current_time, count=5)
        triggered_events = []
        for evt in global_events:
            event_id = f"{evt.time}_{evt.content[:20]}"
            influence = getattr(evt, 'influence', 1.0)
            
            self.hawkes.add_external_event(
                elapsed, 
                influence, 
                evt.source[0] if evt.source else "",
                event_id=event_id
            )
            triggered_events.append(evt.content[:50])
            logger.info(f"📰 External Event (influence={influence}): {evt.content}")
        
        # 2. 处理节点事件（node_post）：直接触发对应智能体执行（与全局事件一致使用 time_granularity）
        node_events = self.event_queue.get_node_events(current_time, window_minutes=self.config.time_granularity)
        for evt in node_events:
            for source_id in evt.source:
                if source_id in self.agents:
                    agent = self.agents[source_id]
                    # 使用外部干预方法直接生成行为
                    intention = agent.external_intervention(evt.to_dict())
                    if intention:
                        self._record_action(intention, current_time, agent)
                        logger.info(f"🎯 Node Event triggered action for {agent.profile.username}: {intention.text[:30]}...")
        
        # 3. 计算当前强度并确定激活数量（使用带调试信息的版本）
        intensity, intensity_debug = self.hawkes.get_intensity_with_debug(elapsed)
        num_activate, activation_debug = self.hawkes.get_expected_activations_with_debug(
            elapsed, len(self.agents), intensity_debug
        )
        
        # 输出详细的强度和激活数计算过程
        self._log_intensity_debug(intensity_debug, activation_debug)
        
        # 4. 采样激活智能体（保证最小激活数以充分利用API端点）
        min_agents = max(1, len(self.api_pool.clients)) if self.config.use_llm else 1
        if 0 < num_activate < min_agents:
            logger.debug(f"   ⬆️ 提升激活数 {num_activate} -> {min_agents} (最小端点利用保证)")
            num_activate = min_agents
        activated_agents = self._sample_agents(num_activate)
        if activated_agents:
            agent_names = [a.profile.username for a in activated_agents[:5]]
            # 计算智能体类型分布
            type_counts = {}
            for a in activated_agents:
                t = a.agent_type
                type_counts[t] = type_counts.get(t, 0) + 1
            total = len(activated_agents)
            type_dist = ', '.join([f"{t}:{c}({c*100/total:.1f}%)" for t, c in sorted(type_counts.items())])
            logger.info(f"👥 Activated {total} agents: {agent_names}{'...' if total > 5 else ''}")
            logger.info(f"   📊 Distribution: {type_dist}")
        
        # 5. 并发执行智能体行为（分阶段执行以最大化API利用率）
        step_actions = []
        agent_exposed_posts = {}  # 收集每个智能体的曝光博文
        if activated_agents:
            exec_start = time.time()
            if self.config.use_llm:
                # LLM模式：分阶段执行，最大化端点利用率
                actions_results, exposed_results = await self._execute_agents_phased(
                    activated_agents, current_time, agent_events
                )
                step_actions = actions_results
                agent_exposed_posts = exposed_results
            else:
                # 非LLM模式：直接并发执行
                tasks = [self._execute_agent(agent, current_time, agent_events) 
                         for agent in activated_agents]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.warning(f"⚠️ Agent execution error: {result}")
                    elif isinstance(result, tuple) and len(result) == 2:
                        actions, exposed_posts = result
                        if isinstance(actions, list):
                            step_actions.extend(actions)
                        agent_exposed_posts[activated_agents[i].profile.user_id] = exposed_posts
                    elif isinstance(result, list):
                        step_actions.extend(result)
            self.perf_metrics.record_agent_execution(time.time() - exec_start)
        
        # 6. 情绪传染（基于曝光博文作者情绪）
        self._apply_emotion_contagion(activated_agents, agent_exposed_posts, current_time)
        
        # 7. 更新统计
        self.stats['total_actions'] += len(step_actions)
        self.stats['active_agents_per_step'].append(len(activated_agents))
        self.stats['intensity_history'].append(intensity)
        self.stats['actions_per_step'].append(len(step_actions))
        
        # 更新上一步的行为统计
        self._update_last_step_stats(step_num, step_actions)
        
        # 8. 每轮更新热搜（强制更新以保持实时性）
        self.hot_search.update_hot_list(current_time, force=True)
        
        # 8.5 定期清理过期事件和博文（每10步清理一次）
        if step_num % 10 == 0:
            self.hawkes.clear_old_events()
            self.recommendation.clear_old_posts(current_time)
        
        # 9. 生成实时信号
        actions_by_type = {}
        for action in step_actions:
            action_type = action.get('action_type', 'unknown')
            actions_by_type[action_type] = actions_by_type.get(action_type, 0) + 1
        
        post_count = actions_by_type.get('short_post', 0) + actions_by_type.get('long_post', 0)
        repost_count = actions_by_type.get('repost', 0) + actions_by_type.get('repost_comment', 0)
        comment_count = actions_by_type.get('short_comment', 0) + actions_by_type.get('long_comment', 0)
        like_count = actions_by_type.get('like', 0)
        
        step_signals = StepSignals(
            step=step_num,
            time=current_time,
            elapsed_minutes=elapsed,
            activated_count=len(activated_agents),
            total_agents=len(self.agents),
            actions_count=len(step_actions),
            post_count=post_count,
            repost_count=repost_count,
            comment_count=comment_count,
            like_count=like_count,
            actions_by_type=actions_by_type,
            hawkes_intensity=intensity_debug.final_intensity,
            hawkes_mu=intensity_debug.mu,
            hawkes_internal_contribution=intensity_debug.internal_contribution,
            hawkes_external_contribution=intensity_debug.external_contribution,
            hawkes_internal_raw=intensity_debug.internal_raw,
            hawkes_external_raw=intensity_debug.external_raw,
            hawkes_circadian_factor=intensity_debug.circadian_factor,
            internal_events_count=intensity_debug.internal_events_count,
            external_events_count=intensity_debug.external_events_count,
            noise_added=activation_debug.noise_added,
            external_events_triggered=triggered_events
        )
        
        # 保存信号历史
        self.signals_history.append(step_signals)
        
        # 调用信号回调（用于WebSocket推送）
        if self.signal_callback:
            try:
                self.signal_callback(step_signals)
            except Exception as e:
                logger.warning(f"信号回调执行失败: {e}")
        
        # 10. 推进时间（time_granularity单位为分钟）
        self.time_engine.advance()
        self.hawkes.advance_time(self.config.time_granularity)
        
        # 记录步骤耗时
        step_duration = time.time() - step_start
        self.perf_metrics.record_step(step_duration)
        logger.debug(f"⏱️ Step {step_num} completed in {step_duration:.3f}s")
        
        # 记录LLM统计并重置
        self.api_pool.log_step_stats(step_num)
        self.api_pool.reset_step_stats()
        
        step_result = {
            'step': self.time_engine.state.step,
            'time': current_time,
            'intensity': intensity,
            'activated_count': len(activated_agents),
            'actions_count': len(step_actions),
            'actions': step_actions,
            'signals': step_signals.to_dict()
        }
        
        if self.step_callback:
            self.step_callback(step_result)
        
        return step_result
    
    def _log_last_step_stats(self):
        """输出上一轮的详细行为统计"""
        stats = self.last_step_stats
        logger.debug(f"\n{'─'*40}")
        logger.debug(f"📊 上一轮(Step {stats['step']})行为统计:")
        logger.debug(f"   总行为数: {stats['actions_count']}")
        logger.debug(f"   ├─ 发博(short_post+long_post): {stats['post_count']}")
        logger.debug(f"   ├─ 转发(repost+repost_comment): {stats['repost_count']}")
        logger.debug(f"   ├─ 评论(short_comment+long_comment): {stats['comment_count']}")
        logger.debug(f"   └─ 点赞(like): {stats['like_count']}")
        logger.debug(f"   发博+转发+评论总和: {stats['post_repost_comment_sum']}")
        if stats['actions_by_type']:
            type_details = ', '.join([f"{k}:{v}" for k, v in sorted(stats['actions_by_type'].items())])
            logger.debug(f"   详细分类: {type_details}")
        logger.debug(f"{'─'*40}\n")
    
    def _log_intensity_debug(self, intensity_debug: IntensityDebugInfo, 
                             activation_debug: ActivationDebugInfo):
        """输出强度和激活数计算详情"""
        d = intensity_debug
        a = activation_debug
        logger.debug(
            f"\n{'─'*40}\n"
            f"Hawkes t={d.t:.1f}min | hour={d.circadian_hour} circ={d.circadian_factor:.3f}\n"
            f"  μ={d.mu} α_int={d.internal_alpha} β_int={d.internal_beta} "
            f"α_ext={d.external_alpha} β_ext={d.external_beta}\n"
            f"  events: ext={d.external_events_count} int={d.internal_events_count}\n"
            f"  int_sum={d.internal_raw:.6f} ext_sum={d.external_raw:.6f}\n"
            f"  λ_norm={d.base_intensity_before_circadian:.6f} → λ_final={d.final_intensity:.6f}\n"
            f"  total_scale={d.activation_scale:.0f} expected={a.expected_raw:.2f} "
            f"→ activated={a.final_activated}"
        )
    def _update_last_step_stats(self, step_num: int, step_actions: List[Dict]):
        """更新上一步的行为统计"""
        actions_by_type = {}
        for action in step_actions:
            action_type = action.get('action_type', 'unknown')
            actions_by_type[action_type] = actions_by_type.get(action_type, 0) + 1
        
        post_count = actions_by_type.get('short_post', 0) + actions_by_type.get('long_post', 0)
        repost_count = actions_by_type.get('repost', 0) + actions_by_type.get('repost_comment', 0)
        comment_count = actions_by_type.get('short_comment', 0) + actions_by_type.get('long_comment', 0)
        like_count = actions_by_type.get('like', 0)
        
        self.last_step_stats = {
            'step': step_num,
            'actions_count': len(step_actions),
            'post_count': post_count,
            'repost_count': repost_count,
            'comment_count': comment_count,
            'like_count': like_count,
            'post_repost_comment_sum': post_count + repost_count + comment_count,
            'actions_by_type': actions_by_type
        }
    
    def _sample_agents(self, count: int) -> List[BaseAgent]:
        """基于活跃度采样智能体"""
        if count <= 0:
            return []
        
        available = [a for a in self.agents.values() if not a.is_banned]
        if not available:
            return []
        
        count = min(count, len(available))
        weights = [a.activity_score for a in available]
        total = sum(weights)
        
        if total == 0:
            return random.sample(available, count)
        
        # 只选择权重大于0的智能体参与加权采样，其余随机补足
        weighted_indices = [i for i, w in enumerate(weights) if w > 0]
        unweighted_indices = [i for i, w in enumerate(weights) if w == 0]
        
        n_weighted = min(count, len(weighted_indices))
        n_unweighted = count - n_weighted

        sampled_indices = []
        if n_weighted > 0:
            weighted_weights = [weights[i] for i in weighted_indices]
            weighted_total = sum(weighted_weights)
            weighted_probs = [w / weighted_total for w in weighted_weights]
            # 若数量等于可用数量，直接全选
            if n_weighted == len(weighted_indices):
                sampled_indices.extend(weighted_indices)
            else:
                sampled = np.random.choice(weighted_indices, size=n_weighted, replace=False, p=weighted_probs)
                sampled_indices.extend(sampled.tolist())
        
        if n_unweighted > 0 and unweighted_indices:
            sampled = random.sample(unweighted_indices, min(n_unweighted, len(unweighted_indices)))
            sampled_indices.extend(sampled)
        
        # 确保采样数量与count一致
        if len(sampled_indices) < count:
            remaining = set(range(len(available))) - set(sampled_indices)
            need = count - len(sampled_indices)
            sampled_indices.extend(random.sample(list(remaining), need))
        
        return [available[i] for i in sampled_indices]
    
    def _apply_emotion_contagion(self, activated_agents: List[BaseAgent], 
                                agent_exposed_posts: Dict[str, List], current_time: str):
        """应用情绪传染 - 基于曝光博文作者的情绪向量"""
        if not activated_agents:
            return
        
        contagion_start = time.time()
        
        for agent in activated_agents:
            if not hasattr(agent, 'belief_system') or not hasattr(agent.belief_system, 'emotion'):
                continue
                
            user_id = agent.profile.user_id
            exposed_posts = agent_exposed_posts.get(user_id, [])
            
            if not exposed_posts:
                continue
            
            # 收集曝光博文作者的情绪向量
            author_emotions = []
            for post in exposed_posts:
                author_id = post.get('author_id')
                if author_id and author_id in self.agents:
                    author_agent = self.agents[author_id]
                    if hasattr(author_agent, 'belief_system') and hasattr(author_agent.belief_system, 'emotion'):
                        author_emotions.append(author_agent.belief_system.emotion.emotion_vector)
            
            # 应用情绪传染
            if author_emotions:
                agent.belief_system.emotion.update_from_neighbors(
                    author_emotions, 
                    influence_rate=0.1,
                    current_time=current_time
                )
        
        self.perf_metrics.record_emotion_contagion(time.time() - contagion_start)
        logger.debug(f"🧠 Emotion contagion applied to {len(activated_agents)} agents")
    
    async def _execute_agents_phased(self, agents: List[BaseAgent], current_time: str,
                                     external_events: List) -> tuple:
        """
        分阶段并发执行智能体行为（最大化API端点利用率）
        
        将3阶段串行的认知流程拆分为并行批次:
        Phase 0: 并行获取所有智能体的推荐博文
        Phase 1: 并行执行所有智能体的信念更新
        Phase 2: 并行执行所有智能体的欲望生成
        Phase 3: 并行执行所有智能体的意图决策
        
        优势: 每个阶段所有智能体同时请求API，充分利用所有端点
        """
        hot_topics = self.hot_search.get_top_topics_text()
        events_dict = [e.to_dict() for e in external_events]
        
        # Phase 0: 并行获取推荐博文
        agent_exposed = {}
        async def fetch_recommendations(agent):
            user_profile = {
                'user_id': agent.profile.user_id,
                'description': agent.profile.description
            }
            recent_posts = [m.content for m in agent.memory.get_recent(10)]
            return self.recommendation.get_recommendations(user_profile, recent_posts, current_time)
        
        rec_tasks = [fetch_recommendations(a) for a in agents]
        rec_results = await asyncio.gather(*rec_tasks, return_exceptions=True)
        for i, result in enumerate(rec_results):
            uid = agents[i].profile.user_id
            agent_exposed[uid] = result if not isinstance(result, Exception) else []
            if isinstance(result, Exception):
                logger.warning(f"⚠️ Recommendation error for {agents[i].profile.username}: {result}")
        
        # Phase 1: 并行信念更新
        async def update_belief(agent):
            exposed = agent_exposed.get(agent.profile.user_id, [])
            relevant_memories = agent.memory_retrieval.retrieve_by_recency_and_importance(agent.memory, top_k=5)
            memories_dict = agent.memory.to_dict_list(relevant_memories)
            agent.belief_system.decay_emotion()
            await agent.belief_updater.update_belief(
                agent.belief_system, exposed, events_dict, memories_dict,
                agent.agent_type, current_time, agent.event_background
            )
        
        belief_tasks = [update_belief(a) for a in agents]
        belief_results = await asyncio.gather(*belief_tasks, return_exceptions=True)
        for i, result in enumerate(belief_results):
            if isinstance(result, Exception):
                logger.warning(f"⚠️ Belief update error for {agents[i].profile.username}: {result}")
        
        # Phase 2: 并行欲望生成
        agent_desires = {}
        async def generate_desire(agent):
            exposed = agent_exposed.get(agent.profile.user_id, [])
            belief_text = agent.belief_system.to_prompt_text()
            desires = await agent.desire_system.generate_desires(
                belief_text, exposed, events_dict, agent.agent_type, current_time,
                agent.event_background
            )
            return [d.to_dict() for d in desires]
        
        desire_tasks = [generate_desire(a) for a in agents]
        desire_results = await asyncio.gather(*desire_tasks, return_exceptions=True)
        for i, result in enumerate(desire_results):
            uid = agents[i].profile.user_id
            if isinstance(result, Exception):
                logger.warning(f"⚠️ Desire error for {agents[i].profile.username}: {result}")
                agent_desires[uid] = []
            else:
                agent_desires[uid] = result
        
        # Phase 3: 并行意图生成
        all_actions = []
        all_exposed = {}
        
        async def generate_intention(agent):
            uid = agent.profile.user_id
            exposed = agent_exposed.get(uid, [])
            desires = agent_desires.get(uid, [])
            belief_text = agent.belief_system.to_prompt_text()
            intentions = await agent.intention_system.generate_intentions(
                belief_text, desires, exposed, agent.agent_type,
                max_actions=agent._get_max_actions(), current_time=current_time,
                event_background=agent.event_background, external_events=events_dict,
                hot_topics=hot_topics
            )
            # 记录行为到记忆
            for intention in intentions:
                agent._record_action_to_memory(intention, current_time)
            agent.last_action_time = current_time
            return intentions
        
        intention_tasks = [generate_intention(a) for a in agents]
        intention_results = await asyncio.gather(*intention_tasks, return_exceptions=True)
        
        for i, result in enumerate(intention_results):
            agent = agents[i]
            uid = agent.profile.user_id
            all_exposed[uid] = agent_exposed.get(uid, [])
            
            if isinstance(result, Exception):
                logger.warning(f"⚠️ Intention error for {agent.profile.username}: {result}")
                continue
            
            for intention in result:
                action = self._process_intention(agent, intention, current_time)
                if action:
                    all_actions.append(action)
                    self._record_action(action, current_time, agent)
                    logger.info(f"   ✅ [{agent.profile.username}] -> {action['action_type']}: {action['content'][:50]}...")
        
        return all_actions, all_exposed
    
    async def _execute_agent(self, agent: BaseAgent, current_time: str, 
                            external_events: List):
        """执行单个智能体的行为，返回(actions, exposed_posts)"""
        logger.debug(f"🤖 Agent [{agent.profile.username}] starting perceive_and_act...")
        
        # 获取当前热搜话题
        hot_topics = self.hot_search.get_top_topics_text()
        
        if not self.config.use_llm:
            # 随机决策时不需要推荐博文，直接传空列表和外部事件
            events_dict = [e.to_dict() for e in external_events]
            intentions = agent.random_decision([], events_dict, current_time)
            # 处理并记录行为（与LLM路径统一）
            actions = []
            for intention in intentions:
                action = self._process_intention(agent, intention, current_time)
                if action:
                    actions.append(action)
                    self._record_action(action, current_time, agent)
            return (actions, [])  # 随机决策时没有exposed_posts
        
        # 获取推荐博文
        user_profile = {
            'user_id': agent.profile.user_id,
            'description': agent.profile.description
        }
        recent_posts = [m.content for m in agent.memory.get_recent(10)]
        exposed_posts = self.recommendation.get_recommendations(user_profile, recent_posts, current_time)
        logger.debug(f"   📥 Exposed to {len(exposed_posts)} posts")
        
        # 转换外部事件格式
        events_dict = [e.to_dict() for e in external_events]
        
        # 执行感知-行为流程（传递热搜话题）
        intentions = await agent.perceive_and_act(exposed_posts, events_dict, current_time, hot_topics)
        logger.debug(f"   🧠 Generated {len(intentions)} intentions")
        
        # 处理行为结果
        actions = []
        for intention in intentions:
            action = self._process_intention(agent, intention, current_time)
            if action:
                actions.append(action)
                self._record_action(action, current_time, agent)
                logger.info(f"   ✅ [{agent.profile.username}] -> {action['action_type']}: {action['content']}...")
        
        return (actions, exposed_posts)
    
    def _process_intention(self, agent: BaseAgent, intention, current_time: str) -> Dict:
        """处理意图结果（包含完整表达策略）"""
        action_type = intention.action_type
        
        action = {
            'user_id': agent.profile.user_id,
            'username': agent.profile.username,
            'agent_type': agent.agent_type,
            'action_type': action_type,
            'target_post_id': intention.target_post_id,
            'target_author': intention.target_author,
            'content': intention.text,
            'text': intention.text,  # 备用字段，兼容评估器
            'topics': intention.topics,
            'mentions': intention.mentions,
            # 完整表达策略
            'emotion': intention.emotion,
            'emotion_intensity': intention.emotion_intensity,
            'stance': intention.stance,
            'stance_intensity': intention.stance_intensity,
            'style': intention.style,
            'narrative': intention.narrative,
            # 表达策略字典（兼容评估器）
            'expression_strategy': {
                'emotion_type': intention.emotion,
                'emotion_intensity': intention.emotion_intensity,
                'stance': intention.stance,
                'stance_intensity': intention.stance_intensity,
                'expression_style': intention.style,
                'narrative_strategy': intention.narrative
            },
            'time': current_time
        }
        
        # 更新推荐系统统计
        if intention.target_post_id:
            self.recommendation.update_post_stats(intention.target_post_id, action_type)
            if action_type in ['short_comment', 'long_comment'] and intention.text:
                self.recommendation.add_comment(intention.target_post_id, intention.text)
        
        # 记录转发/评论关系到社交网络（Neo4j）
        if intention.target_post_id and intention.target_author:
            if action_type in ['repost', 'repost_comment']:
                self.social_network.add_repost(
                    user_id=agent.profile.user_id,
                    post_id=intention.target_post_id,
                    original_author_id=intention.target_author,
                    time=current_time
                )
            elif action_type in ['short_comment', 'long_comment']:
                self.social_network.add_comment(
                    user_id=agent.profile.user_id,
                    post_id=intention.target_post_id,
                    original_author_id=intention.target_author,
                    content=intention.text or '',
                    time=current_time
                )
        
        # 添加新博文到推荐池
        if action_type in ['short_post', 'long_post', 'repost', 'repost_comment']:
            post = {
                'type': 'original' if action_type in ['short_post', 'long_post'] else action_type,
                'author': agent.profile.username,
                'author_id': agent.profile.user_id,
                'content': intention.text,
                'time': current_time,
                'likes': 0,
                'reposts': 0,
                'comments': []
            }
            # 转发类型需要添加原博信息
            if action_type in ['repost', 'repost_comment'] and intention.target_post_id:
                # 从推荐池中查找原博
                for p in self.recommendation.content_pool:
                    if p.get('id') == intention.target_post_id:
                        post['root_author'] = p.get('author', '')
                        post['root_content'] = p.get('content', p.get('root_content', ''))
                        break
            # 将模拟产生的新博文加入推荐池，使后续智能体能看到新内容
            self.recommendation.add_post(post, current_time)
        
        # 更新话题热度
        for topic in intention.topics:
            self.hot_search.add_topic_mention(topic, current_time)
        
        return action
    
    def _record_action(self, action, current_time: str, agent: BaseAgent = None):
        """记录行为（支持Dict或IntentionResult）"""
        # 兼容处理：支持字典或IntentionResult对象
        if hasattr(action, 'to_dict'):
            action_dict = action.to_dict()
        elif isinstance(action, dict):
            action_dict = action
        else:
            logger.warning(f"Unknown action type: {type(action)}")
            return
        
        action_type = action_dict.get('action_type', 'unknown')
        self.stats['actions_by_type'][action_type] = self.stats['actions_by_type'].get(action_type, 0) + 1
        
        # 计算用户影响力因子
        user_influence = 1.0
        if agent:
            followers = agent.profile.followers_count
            user_influence = 1.0 + np.log1p(followers) / 10.0
        
        # 添加到霍克斯过程
        elapsed = self.time_engine.elapsed_minutes
        self.hawkes.add_internal_event(elapsed, action_type, action_dict.get('user_id', ''), user_influence)
        
        if self.action_callback:
            self.action_callback(action_dict)
    
    async def run(self, progress_callback=None) -> Dict[str, Any]:
        """运行完整仿真"""
        results = []
        
        while not self.time_engine.is_finished():
            step_result = await self.run_step()
            results.append(step_result)
            
            if progress_callback:
                progress_callback(self.time_engine.progress, step_result)
        
        # 输出性能摘要
        perf_summary = self.perf_metrics.get_summary()
        logger.info(f"\n{'='*50}\n📊 Performance Summary\n{'='*50}")
        logger.info(f"  Total steps: {perf_summary['total_steps']}")
        logger.info(f"  Avg step time: {perf_summary['step']['avg']:.3f}s")
        logger.info(f"  Max step time: {perf_summary['step']['max']:.3f}s")
        logger.info(f"  Total LLM calls: {perf_summary['total_llm_calls']}")
        if perf_summary['llm']['total'] > 0:
            logger.info(f"  Avg LLM time: {perf_summary['llm']['avg']:.3f}s")
        logger.info(f"  Avg agent exec time: {perf_summary['agent_execution']['avg']:.3f}s")
        
        # 输出LLM详细统计
        self.api_pool.log_final_stats()
        
        return {
            'steps': len(results),
            'stats': self.stats,
            'performance': perf_summary,
            'final_hot_search': self.hot_search.get_hot_list(20),
            'hot_search_history': self.hot_search.get_history(),
            'time_engine': self.time_engine.to_dict()
        }
    
    # 干预接口
    def ban_user(self, user_id: str):
        """禁言用户"""
        if user_id in self.agents:
            self.agents[user_id].ban()
    
    def unban_user(self, user_id: str):
        """解禁用户"""
        if user_id in self.agents:
            self.agents[user_id].unban()
    
    def delete_post(self, post_id: str):
        """删除博文"""
        self.recommendation.content_pool = [
            p for p in self.recommendation.content_pool if p['id'] != post_id
        ]
    
    def inject_event(self, event_type: str, content: str, source: List[str] = None):
        """注入事件"""
        event = ExternalEvent(
            time=self.time_engine.current_time_str,
            event_type=EventType(event_type),
            source=source or [],
            content=content
        )
        self.event_queue.add_event(event)
