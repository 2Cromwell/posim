# -*- coding: utf-8 -*-
"""
评估数据加载器 - 加载模拟结果和真实标注数据
"""
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

from .utils import (
    parse_time, truncate_time, generate_time_range,
    normalize_emotion_label, extract_topics, extract_user_id_from_url,
    classify_emotion_by_keywords, batch_classify_emotions,
    classify_stance, classify_style, EMO_CN_TO_EN,
    INTENSITY_MAP, STANCE_MAP, SENTIMENT_POLARITY_MAP
)

logger = logging.getLogger(__name__)


class SimulationDataLoader:
    """模拟结果数据加载器"""
    
    def __init__(self, results_dir: Path, time_granularity: int = 10):
        self.results_dir = Path(results_dir)
        self.granularity = time_granularity
    
    def load(self) -> Dict[str, Any]:
        """加载模拟结果数据"""
        data = {}
        
        # 加载微观结果
        micro_path = self.results_dir / "micro_results.json"
        if micro_path.exists():
            with open(micro_path, 'r', encoding='utf-8') as f:
                data['micro_results'] = json.load(f)
            logger.info(f"加载模拟微观结果: {len(data['micro_results'])} 条行为")
        else:
            data['micro_results'] = []
            logger.warning(f"微观结果文件不存在: {micro_path}")
        
        # 加载宏观结果
        macro_path = self.results_dir / "macro_results.json"
        if macro_path.exists():
            with open(macro_path, 'r', encoding='utf-8') as f:
                data['macro_results'] = json.load(f)
            logger.info(f"加载模拟宏观结果: {data['macro_results'].get('steps', 0)} 步")
        else:
            data['macro_results'] = {}
        
        # 加载配置
        config_path = self.results_dir / "config.json"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                data['config'] = json.load(f)
        else:
            data['config'] = {}
        
        # 聚合时间序列数据
        data['aggregated'] = self._aggregate_data(data['micro_results'])
        
        return data
    
    def _aggregate_data(self, micro_results: List[Dict]) -> Dict[str, Any]:
        """聚合模拟数据到时间序列"""
        h_buckets = defaultdict(lambda: {'total': 0, 'original': 0, 'repost': 0, 'comment': 0})
        e_buckets = defaultdict(lambda: defaultdict(int))
        sentiment_buckets = defaultdict(lambda: {'positive': 0, 'negative': 0, 'neutral': 0})
        action_buckets = defaultdict(lambda: defaultdict(int))
        topic_buckets = defaultdict(lambda: defaultdict(int))
        
        type_map = {
            'short_post': 'original', 'long_post': 'original',
            'repost': 'repost', 'repost_comment': 'repost',
            'short_comment': 'comment', 'long_comment': 'comment', 'like': 'comment'
        }
        
        # 批量情绪分类
        texts = [a.get('text', a.get('content', '')) for a in micro_results]
        emotions, _ = batch_classify_emotions(texts)
        
        actions_data = []
        for i, a in enumerate(micro_results):
            t = parse_time(a.get('time', ''))
            if not t:
                continue
            t = truncate_time(t, self.granularity)
            action_type = a.get('action_type', 'unknown')
            atype = type_map.get(action_type, 'comment')
            emotion = emotions[i] if i < len(emotions) else 'Neutral'
            
            # 尝试从表达策略获取情绪
            strategy = a.get('expression_strategy', {})
            if strategy:
                emo_raw = strategy.get('emotion_type', '')
                if emo_raw and isinstance(emo_raw, str):
                    emotion = normalize_emotion_label(emo_raw)
            
            h_buckets[t]['total'] += 1
            h_buckets[t][atype] += 1
            e_buckets[t][emotion] += 1
            action_buckets[t][action_type] += 1
            
            # 情感极性
            text = a.get('text', a.get('content', ''))
            try:
                from snownlp import SnowNLP
                sentiment_score = SnowNLP(text).sentiments if text else 0.5
            except:
                sentiment_score = 0.5
            if sentiment_score > 0.6:
                sentiment_buckets[t]['positive'] += 1
            elif sentiment_score < 0.4:
                sentiment_buckets[t]['negative'] += 1
            else:
                sentiment_buckets[t]['neutral'] += 1
            
            # 话题
            for topic in a.get('topics', []):
                topic = topic.strip('#').strip()
                if topic:
                    topic_buckets[topic][t] += 1
            # 也从content中提取
            for topic in extract_topics(text):
                topic_buckets[topic][t] += 1
            
            actions_data.append({
                'time': t, 'action_type': action_type, 'emotion': emotion,
                'stance': classify_stance(text), 'style': classify_style(text),
                'text': text[:200] if text else '',
                'user_id': a.get('user_id', ''),
                'username': a.get('username', ''),
                'agent_type': a.get('agent_type', ''),
                'sentiment_score': sentiment_score,
                'target_post_id': a.get('target_post_id', ''),
                'target_author': a.get('target_author', '')
            })
        
        if not h_buckets:
            return {'times': [], 'hotness': {}, 'emotion': {}, 'sentiment': {},
                    'topics': {}, 'actions': actions_data, 'action_buckets': {}}
        
        times = sorted(h_buckets.keys())
        all_times = generate_time_range(times[0], times[-1], self.granularity)
        
        hotness = {k: [h_buckets[t].get(k, 0) for t in all_times]
                   for k in ['total', 'original', 'repost', 'comment']}
        emotion = {e: [e_buckets[t].get(e, 0) for t in all_times]
                   for e in set(emotions)}
        sentiment = {k: [sentiment_buckets[t].get(k, 0) for t in all_times]
                     for k in ['positive', 'negative', 'neutral']}
        
        topics = {}
        for topic, time_counts in topic_buckets.items():
            topics[topic] = [{'time': t.isoformat(), 'count': time_counts.get(t, 0)} for t in all_times]
        
        action_time_series = {}
        all_action_types = set()
        for t_counts in action_buckets.values():
            all_action_types.update(t_counts.keys())
        for atype in all_action_types:
            action_time_series[atype] = [action_buckets[t].get(atype, 0) for t in all_times]
        
        return {
            'times': all_times,
            'hotness': hotness,
            'emotion': emotion,
            'sentiment': sentiment,
            'topics': topics,
            'actions': actions_data,
            'action_time_series': action_time_series
        }


class RealDataLoader:
    """真实标注数据加载器 - 支持 labels.json 格式"""
    
    def __init__(self, labels_path: str, time_granularity: int = 10,
                 time_start: str = None, time_end: str = None):
        self.labels_path = Path(labels_path)
        self.granularity = time_granularity
        self.time_start = parse_time(time_start) if time_start else None
        self.time_end = parse_time(time_end) if time_end else None
    
    def load(self) -> Dict[str, Any]:
        """加载真实标注数据"""
        if not self.labels_path.exists():
            logger.warning(f"真实数据文件不存在: {self.labels_path}")
            return self._empty_result()
        
        with open(self.labels_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        logger.info(f"加载真实标注数据: {len(raw_data)} 条记录")
        
        # 判断数据格式
        if isinstance(raw_data, list) and raw_data:
            first = raw_data[0]
            if isinstance(first, dict):
                if 'post_id' in first:
                    return self._load_labels_format(raw_data)
                # base_data.json: user_key / user_info, original_posts, repost_posts, comments
                if 'original_posts' in first or 'repost_posts' in first or 'comments' in first:
                    return self._load_users_format(raw_data)
                if 'user_id' in first:
                    return self._load_users_format(raw_data)
        logger.warning("未识别的真实数据格式")
        return self._empty_result()
    
    def _empty_result(self) -> Dict[str, Any]:
        return {
            'actions': [], 'times': [], 'hotness': {}, 'emotion': {},
            'sentiment': {}, 'topics': {}, 'user_actions': {}
        }
    
    def _load_labels_format(self, raw_data: List[Dict]) -> Dict[str, Any]:
        """加载 labels.json 格式的数据"""
        actions = []
        
        for entry in raw_data:
            t = parse_time(entry.get('time', ''))
            if not t:
                continue
            if self.time_start and t < self.time_start:
                continue
            if self.time_end and t > self.time_end:
                continue
            
            post_type = entry.get('type', 'comment')
            content = entry.get('content', '')
            user_id = extract_user_id_from_url(entry.get('url', ''))
            
            # 从标注中提取信息
            annotation = entry.get('llm_annotation', {})
            nlp_sentiment = entry.get('nlp_sentiment', {})
            
            # 情绪类型处理
            emotion_types = annotation.get('emotion_type', [])
            if isinstance(emotion_types, list):
                if emotion_types:
                    primary_emotion = normalize_emotion_label(emotion_types[0])
                else:
                    primary_emotion = 'Neutral'
            else:
                primary_emotion = normalize_emotion_label(str(emotion_types))
            
            # 情绪强度
            intensity_raw = annotation.get('emotion_intensity', '低')
            emotion_intensity = INTENSITY_MAP.get(intensity_raw, 0.5)
            
            # 立场
            stance_raw = annotation.get('stance', '中立')
            stance = stance_raw
            stance_value = STANCE_MAP.get(stance_raw, 0.0)
            
            # 情感极性
            sentiment_polarity = annotation.get('sentiment_polarity', '中性')
            sentiment_en = SENTIMENT_POLARITY_MAP.get(sentiment_polarity, 'neutral')
            
            # NLP情感值
            nlp_score = nlp_sentiment.get('score', 0.5)
            
            # 话题
            topics = extract_topics(content)
            
            actions.append({
                'post_id': entry.get('post_id', ''),
                'user_id': user_id,
                'time': t,
                'type': post_type,
                'content': content,
                'behavior_type': entry.get('behavior_type', ''),
                'emotion': primary_emotion,
                'emotion_types': [normalize_emotion_label(e) for e in (emotion_types if isinstance(emotion_types, list) else [emotion_types])],
                'emotion_intensity': emotion_intensity,
                'stance': stance,
                'stance_value': stance_value,
                'stance_intensity': INTENSITY_MAP.get(annotation.get('stance_intensity', '低'), 0.5),
                'expression_styles': annotation.get('expression_style', []),
                'narrative_strategies': annotation.get('narrative_strategy', []),
                'sentiment_polarity': sentiment_en,
                'nlp_sentiment_score': nlp_score,
                'attitude': annotation.get('attitude', ''),
                'emotionality': annotation.get('emotionality', ''),
                'topics': topics
            })
        
        logger.info(f"真实数据时间过滤后: {len(actions)} 条")
        return self._aggregate_real_data(actions)
    
    def _load_users_format(self, raw_data: List[Dict]) -> Dict[str, Any]:
        """加载 base_data.json 格式（用户级别聚合）"""
        actions = []
        for u in raw_data:
            # 支持 base_data.json (user_info.user_id / user_key) 或 直接 user_id
            user_id = (u.get('user_info') or {}).get('user_id') or u.get('user_key') or u.get('user_id', '')
            for p in u.get('original_posts', []):
                t = parse_time(p.get('time', ''))
                if t:
                    emo_en = normalize_emotion_label(p.get('emotion', '中性'))
                    actions.append({
                        'user_id': user_id, 'time': t, 'type': 'original',
                        'emotion': emo_en, 'content': p.get('content', ''),
                        'nlp_sentiment_score': 0.5, 'sentiment_polarity': 'neutral',
                        'stance': 'Neutral', 'stance_value': 0.0,
                        'emotion_intensity': 0.5, 'topics': extract_topics(p.get('content', ''))
                    })
            for p in u.get('repost_posts', []):
                t = parse_time(p.get('time', ''))
                if t:
                    emo_en = normalize_emotion_label(p.get('emotion', '中性'))
                    actions.append({
                        'user_id': user_id, 'time': t, 'type': 'repost',
                        'emotion': emo_en, 'content': p.get('content', ''),
                        'nlp_sentiment_score': 0.5, 'sentiment_polarity': 'neutral',
                        'stance': 'Neutral', 'stance_value': 0.0,
                        'emotion_intensity': 0.5, 'topics': extract_topics(p.get('content', ''))
                    })
            for c in u.get('comments', []):
                t = parse_time(c.get('time', ''))
                if t:
                    emo_en = normalize_emotion_label(c.get('emotion', '中性'))
                    actions.append({
                        'user_id': user_id, 'time': t, 'type': 'comment',
                        'emotion': emo_en, 'content': c.get('content', ''),
                        'nlp_sentiment_score': 0.5, 'sentiment_polarity': 'neutral',
                        'stance': 'Neutral', 'stance_value': 0.0,
                        'emotion_intensity': 0.5, 'topics': extract_topics(c.get('content', ''))
                    })
        
        # 时间过滤
        if self.time_start:
            actions = [a for a in actions if a['time'] >= self.time_start]
        if self.time_end:
            actions = [a for a in actions if a['time'] <= self.time_end]
        
        logger.info(f"真实数据(用户格式)时间过滤后: {len(actions)} 条")
        return self._aggregate_real_data(actions)
    
    def _aggregate_real_data(self, actions: List[Dict]) -> Dict[str, Any]:
        """聚合真实数据到时间序列"""
        if not actions:
            return self._empty_result()
        
        h_buckets = defaultdict(lambda: {'total': 0, 'original': 0, 'repost': 0, 'comment': 0})
        e_buckets = defaultdict(lambda: defaultdict(int))
        sentiment_buckets = defaultdict(lambda: {'positive': 0, 'negative': 0, 'neutral': 0})
        topic_buckets = defaultdict(lambda: defaultdict(int))
        user_actions = defaultdict(list)
        
        for a in actions:
            t = truncate_time(a['time'], self.granularity)
            h_buckets[t]['total'] += 1
            h_buckets[t][a['type']] += 1
            e_buckets[t][a['emotion']] += 1
            sentiment_buckets[t][a.get('sentiment_polarity', 'neutral')] += 1
            
            for topic in a.get('topics', []):
                topic_buckets[topic][t] += 1
            
            if a.get('user_id'):
                user_actions[a['user_id']].append(a)
        
        times = sorted(h_buckets.keys())
        all_times = generate_time_range(times[0], times[-1], self.granularity)
        
        hotness = {k: [h_buckets[t].get(k, 0) for t in all_times]
                   for k in ['total', 'original', 'repost', 'comment']}
        emotion = {e: [e_buckets[t].get(e, 0) for t in all_times]
                   for e in set(a['emotion'] for a in actions)}
        sentiment = {k: [sentiment_buckets[t].get(k, 0) for t in all_times]
                     for k in ['positive', 'negative', 'neutral']}
        
        topics = {}
        for topic, time_counts in topic_buckets.items():
            topics[topic] = [{'time': t.isoformat(), 'count': time_counts.get(t, 0)} for t in all_times]
        
        return {
            'actions': actions,
            'times': all_times,
            'hotness': hotness,
            'emotion': emotion,
            'sentiment': sentiment,
            'topics': topics,
            'user_actions': dict(user_actions)
        }
