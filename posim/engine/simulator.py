"""
仿真核心引擎 - 整合各模块执行仿真
"""
import asyncio
import random
import logging
import time
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field

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

from .hawkes_process import HawkesProcess
from .time_engine import TimeEngine
from ..agents.base_agent import BaseAgent
from ..agents.citizen_agent import CitizenAgent
from ..agents.kol_agent import KOLAgent
from ..agents.media_agent import MediaAgent
from ..agents.government_agent import GovernmentAgent
from ..environment.recommendation import RecommendationSystem
from ..environment.hot_search import HotSearchManager
from ..environment.social_network import SocialNetwork
from ..environment.event_queue import EventQueue, EventType
from ..config.config_manager import ConfigManager
from ..llm.api_pool import APIPool


class Simulator:
    """仿真核心引擎"""
    
    def __init__(self, config_manager: ConfigManager, api_pool: APIPool):
        self.config = config_manager.simulation
        self.api_pool = api_pool
        
        # 初始化时间引擎
        self.time_engine = TimeEngine(
            self.config.start_time, 
            self.config.end_time,
            self.config.time_granularity
        )
        
        circadian_curve = {int(k): v for k, v in self.config.circadian_curve.items()}
        
        # 初始化霍克斯过程
        self.hawkes = HawkesProcess(
            mu=self.config.hawkes_mu,
            alpha=self.config.hawkes_alpha,
            beta=self.config.hawkes_beta,
            action_weights=self.config.action_weights,
            time_granularity=self.config.time_granularity,
            activation_scale=getattr(self.config, 'hawkes_activation_scale', 1.0),
            circadian_curve=circadian_curve,
            event_influence_scale=getattr(self.config, 'event_influence_scale', 1.0)
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
        
        # 性能监控
        self.perf_metrics = PerformanceMetrics()
        
        # 回调函数
        self.step_callback = None
        self.action_callback = None
    
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
        
        # 1. 获取当前外部事件
        external_events = self.event_queue.get_current_events(current_time)
        for evt in external_events:
            self.hawkes.add_external_event(elapsed, evt.influence, evt.source[0] if evt.source else "")
            logger.info(f"📰 External Event: {evt.content} (influence={evt.influence})")
        
        # 2. 处理节点事件（政府/媒体发布）
        await self._process_node_events(current_time)
        
        # 3. 计算当前强度并确定激活数量
        intensity = self.hawkes.get_intensity(elapsed)
        num_activate = self.hawkes.get_expected_activations(elapsed, len(self.agents))
        logger.debug(f"⚡ Hawkes intensity={intensity:.4f}, expected_activations={num_activate}")
        
        # 4. 采样激活智能体
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
        
        # 5. 并发执行智能体行为
        step_actions = []
        agent_exposed_posts = {}  # 收集每个智能体的曝光博文
        if activated_agents:
            exec_start = time.time()
            tasks = [self._execute_agent(agent, current_time, external_events) 
                     for agent in activated_agents]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"⚠️ Agent execution error: {result}")
                elif isinstance(result, tuple) and len(result) == 2:
                    # 期望返回 (actions, exposed_posts)
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
        
        # 8. 更新热搜
        if self.time_engine.should_update_hot_search(self.config.hot_search_update_interval):
            self.hot_search.update_hot_list(current_time)
        
        # 9. 推进时间
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
            'actions': step_actions
        }
        
        if self.step_callback:
            self.step_callback(step_result)
        
        return step_result
    
    async def _process_node_events(self, current_time: str):
        """处理节点事件（优化：先遍历事件再找智能体）"""
        # 获取当前时间窗口内的所有节点事件
        node_events = self.event_queue.get_events_by_type(EventType.NODE_POST)
        
        for evt in node_events:
            # 检查事件时间是否在当前时间窗口内
            evt_dt = datetime.fromisoformat(evt.time)
            cur_dt = datetime.fromisoformat(current_time)
            if not (evt_dt <= cur_dt and (cur_dt - evt_dt).total_seconds() < 60):
                continue
            
            # 遍历事件源节点，找到对应智能体执行
            for source_id in evt.source:
                if source_id in self.agents:
                    agent = self.agents[source_id]
                    if isinstance(agent, GovernmentAgent):
                        result = await agent.publish_announcement(evt.content, current_time)
                        if result:
                            self._record_action(result, current_time, agent)
    
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
            if not hasattr(agent, 'belief') or not hasattr(agent.belief, 'emotion'):
                continue
                
            # 获取该智能体的曝光博文（从已收集的数据中）
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
                    if hasattr(author_agent, 'belief') and hasattr(author_agent.belief, 'emotion'):
                        author_emotions.append(author_agent.belief.emotion.emotion_vector)
            
            # 应用情绪传染
            if author_emotions:
                agent.belief.emotion.update_from_neighbors(
                    author_emotions, 
                    influence_rate=0.1,
                    current_time=current_time
                )
        
        self.perf_metrics.record_emotion_contagion(time.time() - contagion_start)
        logger.debug(f"🧠 Emotion contagion applied to {len(activated_agents)} agents")
    
    async def _execute_agent(self, agent: BaseAgent, current_time: str, 
                            external_events: List):
        """执行单个智能体的行为，返回(actions, exposed_posts)"""
        logger.debug(f"🤖 Agent [{agent.profile.username}] starting perceive_and_act...")
        
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
        recent_posts = [m.content for m in agent.memory.get_recent(5)]
        exposed_posts = self.recommendation.get_recommendations(user_profile, recent_posts, current_time)
        logger.debug(f"   📥 Exposed to {len(exposed_posts)} posts")
        
        # 转换外部事件格式
        events_dict = [e.to_dict() for e in external_events]
        
        # 执行感知-行为流程
        intentions = await agent.perceive_and_act(exposed_posts, events_dict, current_time)
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
        """处理意图结果"""
        action_type = intention.action_type
        
        action = {
            'user_id': agent.profile.user_id,
            'username': agent.profile.username,
            'agent_type': agent.agent_type,
            'action_type': action_type,
            'target_post_id': intention.target_post_id,
            'target_author': intention.target_author,
            'content': intention.text,
            'topics': intention.topics,
            'mentions': intention.mentions,
            'emotion': intention.emotion,
            'stance': intention.stance,
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
                'author': agent.profile.username,
                'author_id': agent.profile.user_id,
                'content': intention.text,
                'time': current_time,
                'likes': 0,
                'reposts': 0,
                'comments': []
            }
            self.recommendation.add_post(post, current_time)
        
        # 更新话题热度
        for topic in intention.topics:
            self.hot_search.add_topic_mention(topic, current_time)
        
        return action
    
    def _record_action(self, action: Dict, current_time: str, agent: BaseAgent = None):
        """记录行为"""
        action_type = action.get('action_type', 'unknown')
        self.stats['actions_by_type'][action_type] = self.stats['actions_by_type'].get(action_type, 0) + 1
        
        # 计算用户影响力因子
        user_influence = 1.0
        if agent:
            # 基于粉丝数计算影响力（对数缩放）
            followers = agent.profile.followers_count
            user_influence = 1.0 + np.log1p(followers) / 10.0  # log(1+followers)/10 作为加成
        
        # 添加到霍克斯过程（行为强度 * 用户影响力）
        elapsed = self.time_engine.elapsed_minutes
        self.hawkes.add_internal_event(elapsed, action_type, action.get('user_id', ''), user_influence)
        
        if self.action_callback:
            self.action_callback(action)
    
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
    
    def inject_event(self, event_type: str, content: str, influence: float, source: List[str] = None):
        """注入事件"""
        from ..environment.event_queue import ExternalEvent, EventType as ET
        event = ExternalEvent(
            time=self.time_engine.current_time_str,
            event_type=ET(event_type),
            source=source or [],
            content=content,
            influence=influence
        )
        self.event_queue.add_event(event)
