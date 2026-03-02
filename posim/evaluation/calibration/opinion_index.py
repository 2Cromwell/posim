# -*- coding: utf-8 -*-
"""
舆情演化指数

评估舆情演化的综合对抗性指标:
1. 话语对抗性程度（关键词检测）
2. 语义相似程度（Embedding编码计算）
3. 负面情绪程度（指标计算）
4. 综合舆情演化指数
"""
import logging
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional, List
from collections import Counter, defaultdict

from ..base import BaseEvaluator
from ..utils import (
    smooth, normalize, save_json, calculate_jsd,
    calculate_normalized_entropy,
    CONFLICT_KEYWORDS, EMOTION_KEYWORDS
)
from ..visualization import (
    FIG_SIZE, FIG_SIZE_WIDE, FIG_SIZE_TALL, LW, LW_MINOR,
    MARKER_SIZE, FONT_SIZE,
    C_SIM, C_REAL, setup_time_axis, add_grid, add_legend, save_figure
)

logger = logging.getLogger(__name__)


class OpinionIndexEvaluator(BaseEvaluator):
    """舆情演化指数"""
    
    def __init__(self, output_dir: Path):
        super().__init__(output_dir / "opinion_index", name="opinion_index")
    
    def evaluate(self, sim_data: Dict[str, Any], real_data: Optional[Dict[str, Any]] = None,
                 **kwargs) -> Dict[str, Any]:
        """执行舆情演化指数评估"""
        self._log_section("舆情演化指数")
        
        embedding_model = kwargs.get('embedding_model')
        
        sim_agg = sim_data.get('aggregated', {})
        sim_actions = sim_agg.get('actions', [])
        micro_results = sim_data.get('micro_results', [])
        
        sim_texts = [a.get('text', a.get('content', '')) for a in micro_results
                     if a.get('text') or a.get('content')]
        
        has_real = real_data is not None and real_data.get('actions')
        real_actions = real_data.get('actions', []) if has_real else []
        real_texts = [a.get('content', '') for a in real_actions if a.get('content')] if has_real else []
        
        if not sim_texts:
            print("    ⚠️ 无模拟文本数据，跳过")
            return {}
        
        results = {}
        
        # 1. 话语对抗性程度
        print("    [1/8] 分析话语对抗性程度...")
        results['discourse_confrontation'] = self._analyze_confrontation(sim_texts, real_texts, has_real)
        
        # 2. 语义相似程度
        print("    [2/8] 计算语义相似程度...")
        results['semantic_similarity'] = self._analyze_semantic_similarity(
            sim_texts, real_texts, embedding_model, has_real)
        
        # 3. 负面情绪程度
        print("    [3/8] 分析负面情绪程度...")
        results['negative_emotion'] = self._analyze_negative_emotion(
            sim_texts, sim_actions, real_texts, real_actions, has_real)
        
        # 4. 语义多样性
        print("    [4/8] 计算语义多样性...")
        results['semantic_diversity'] = self._analyze_semantic_diversity(
            sim_texts, real_texts, embedding_model, has_real)

        # 5. 立场表达丰富度
        print("    [5/8] 分析立场表达丰富度...")
        results['stance_richness'] = self._analyze_stance_richness(
            micro_results, real_actions, has_real)

        # 6. 情绪传染效率
        print("    [6/8] 分析情绪传染效率...")
        results['emotion_contagion'] = self._analyze_emotion_contagion(
            sim_agg.get('actions', []), micro_results, real_actions, has_real)

        # 7. 内容层面附加指标
        print("    [7/8] 计算内容层面附加指标...")
        results['content_layer_metrics'] = self._compute_content_layer_metrics(
            sim_texts, real_texts, micro_results, real_actions, has_real)

        # 8. 综合舆情演化指数
        print("    [8/8] 计算综合舆情演化指数...")
        results['opinion_evolution_index'] = self._compute_evolution_index(results, has_real)

        self._plot_radar(results, has_real)
        self._plot_semantic_content(results, has_real)

        self._save_results(results, "opinion_index_metrics.json")
        self._print_summary(results)
        return results
    
    def _analyze_confrontation(self, sim_texts, real_texts, has_real):
        """话语对抗性程度分析"""
        metrics = {}
        
        # 模拟数据
        metrics['sim'] = self._compute_confrontation_detail(sim_texts)
        
        # 真实数据
        if has_real and real_texts:
            metrics['real'] = self._compute_confrontation_detail(real_texts)
            
            # 对抗性差异
            sim_score = metrics['sim']['conflict_score']
            real_score = metrics['real']['conflict_score']
            metrics['confrontation_diff'] = float(abs(sim_score - real_score))
            metrics['confrontation_similarity'] = float(max(0, 1 - abs(sim_score - real_score) * 5))
            print(f"      模拟对抗性: {sim_score:.4f}, 真实对抗性: {real_score:.4f}")
        
        # 绘图
        self._plot_confrontation(metrics, has_real)
        
        return metrics
    
    def _compute_confrontation_detail(self, texts):
        """计算对抗性详细指标"""
        total = len(texts) if texts else 1
        conflict_counts = {key: 0 for key in CONFLICT_KEYWORDS.keys()}
        total_keywords = 0
        
        for text in texts:
            if not text:
                continue
            for ctype, keywords in CONFLICT_KEYWORDS.items():
                if any(kw in text for kw in keywords):
                    conflict_counts[ctype] += 1
                    total_keywords += 1
        
        weights = {
            'debate': 0.15, 'attack': 0.30, 'confrontation': 0.25,
            'criticism': 0.15, 'defense': 0.05, 'polarization': 0.10
        }
        
        conflict_score = sum(weights.get(k, 0) * (v / total) for k, v in conflict_counts.items())
        
        return {
            'conflict_score': float(conflict_score),
            'conflict_counts': {k: int(v) for k, v in conflict_counts.items()},
            'conflict_ratios': {k: float(v / total) for k, v in conflict_counts.items()},
            'total_conflict_keywords': total_keywords,
            'total_texts': total
        }
    
    def _plot_confrontation(self, metrics, has_real):
        """绘制对抗性对比图（拆分为独立图）"""
        ctypes = list(CONFLICT_KEYWORDS.keys())
        sim_ratios = [metrics['sim']['conflict_ratios'].get(ct, 0) for ct in ctypes]
        
        # Figure 1: Conflict type distribution
        fig1, ax1 = plt.subplots(figsize=FIG_SIZE)
        if has_real and 'real' in metrics:
            real_ratios = [metrics['real']['conflict_ratios'].get(ct, 0) for ct in ctypes]
            x = np.arange(len(ctypes))
            width = 0.35
            ax1.bar(x - width/2, sim_ratios, width, label='Simulation', color=C_SIM['total'], alpha=0.8)
            ax1.bar(x + width/2, real_ratios, width, label='Real Data', color=C_REAL['total'], alpha=0.8)
            ax1.set_xticks(x)
        else:
            ax1.bar(range(len(ctypes)), sim_ratios, color=C_SIM['total'], alpha=0.8)
            ax1.set_xticks(range(len(ctypes)))
        ax1.set_xticklabels([ct.title() for ct in ctypes], rotation=45, ha='right')
        ax1.set_ylabel('Ratio')
        ax1.set_title('Conflict Type Distribution', fontweight='bold')
        add_legend(ax1)
        add_grid(ax1)
        save_figure(fig1, self.output_dir / 'confrontation_type_distribution.png')
        
        # Figure 2: Overall confrontation score
        fig2, ax2 = plt.subplots(figsize=FIG_SIZE)
        labels = ['Simulation']
        scores = [metrics['sim']['conflict_score']]
        colors = [C_SIM['total']]
        if has_real and 'real' in metrics:
            labels.append('Real Data')
            scores.append(metrics['real']['conflict_score'])
            colors.append(C_REAL['total'])
        bars = ax2.bar(labels, scores, color=colors, alpha=0.8, edgecolor='black')
        for bar, score in zip(bars, scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{score:.4f}', ha='center', va='bottom', fontsize=11)
        ax2.set_ylabel('Conflict Score')
        ax2.set_title('Overall Discourse Confrontation', fontweight='bold')
        add_grid(ax2)
        save_figure(fig2, self.output_dir / 'confrontation_overall_score.png')
    
    def _analyze_semantic_similarity(self, sim_texts, real_texts, embedding_model, has_real):
        """语义相似程度分析"""
        metrics = {}
        
        if not has_real or not embedding_model:
            if not embedding_model:
                print("      ⚠️ 无Embedding模型，跳过语义相似度计算")
            return metrics
        
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            
            # 采样（避免计算量过大）
            n_sample = min(500, len(sim_texts), len(real_texts))
            import random
            sim_sample = random.sample(sim_texts, n_sample) if len(sim_texts) > n_sample else sim_texts
            real_sample = random.sample(real_texts, n_sample) if len(real_texts) > n_sample else real_texts
            
            # 编码
            sim_embeddings = embedding_model.encode(sim_sample, normalize_embeddings=True)
            real_embeddings = embedding_model.encode(real_sample, normalize_embeddings=True)
            
            # 计算相似度矩阵
            sim_matrix = cosine_similarity(sim_embeddings, real_embeddings)
            
            # 统计指标
            all_sims = sim_matrix.flatten()
            metrics['avg_similarity'] = float(np.mean(all_sims))
            metrics['max_similarity'] = float(np.max(all_sims))
            metrics['min_similarity'] = float(np.min(all_sims))
            metrics['median_similarity'] = float(np.median(all_sims))
            metrics['std_similarity'] = float(np.std(all_sims))
            
            # 每条模拟文本与最近真实文本的最大相似度
            max_per_sim = np.max(sim_matrix, axis=1)
            metrics['avg_max_similarity'] = float(np.mean(max_per_sim))
            
            # 质心相似度
            sim_centroid = np.mean(sim_embeddings, axis=0)
            real_centroid = np.mean(real_embeddings, axis=0)
            centroid_sim = float(cosine_similarity([sim_centroid], [real_centroid])[0][0])
            metrics['centroid_similarity'] = centroid_sim
            
            print(f"      语义相似度: avg={metrics['avg_similarity']:.4f}, "
                  f"centroid={centroid_sim:.4f}")
            
            # 绘制相似度分布
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(all_sims, bins=50, alpha=0.7, color=C_SIM['total'], edgecolor='black', density=True)
            ax.axvline(x=metrics['avg_similarity'], color='red', ls='--', lw=1.5,
                      label=f'Mean={metrics["avg_similarity"]:.3f}')
            ax.axvline(x=centroid_sim, color='blue', ls='--', lw=1.5,
                      label=f'Centroid={centroid_sim:.3f}')
            ax.set_xlabel('Cosine Similarity')
            ax.set_ylabel('Density')
            ax.set_title('Semantic Similarity Distribution (Sim vs Real)', fontweight='bold')
            add_legend(ax)
            add_grid(ax)
            save_figure(fig, self.output_dir / 'semantic_similarity.png')
            
        except Exception as e:
            logger.warning(f"语义相似度计算失败: {e}")
            metrics['error'] = str(e)
        
        return metrics
    
    def _analyze_negative_emotion(self, sim_texts, sim_actions, real_texts, real_actions, has_real):
        """负面情绪程度分析"""
        metrics = {}
        
        # 模拟数据
        sim_neg = self._compute_negative_metrics(sim_texts, sim_actions)
        metrics['sim'] = sim_neg
        
        if has_real and real_texts:
            real_neg = self._compute_negative_metrics(real_texts, real_actions)
            metrics['real'] = real_neg
            metrics['negative_diff'] = float(abs(sim_neg['negative_ratio'] - real_neg['negative_ratio']))
            metrics['negative_similarity'] = float(max(0, 1 - abs(sim_neg['negative_ratio'] - real_neg['negative_ratio'])))
            print(f"      模拟负面比例: {sim_neg['negative_ratio']:.2%}, 真实负面比例: {real_neg['negative_ratio']:.2%}")
        
        # Figure 1: Negative emotion indicators
        fig1, ax1 = plt.subplots(figsize=FIG_SIZE)
        labels = ['Negative\nEmotion', 'High\nIntensity', 'Extreme\nStance']
        sim_vals = [sim_neg['negative_ratio'], sim_neg['high_intensity_ratio'], sim_neg['extreme_stance_ratio']]
        
        if has_real and 'real' in metrics:
            real_neg_data = metrics['real']
            real_vals = [real_neg_data['negative_ratio'], real_neg_data['high_intensity_ratio'], real_neg_data['extreme_stance_ratio']]
            x = np.arange(len(labels))
            width = 0.35
            ax1.bar(x - width/2, sim_vals, width, label='Simulation', color=C_SIM['total'], alpha=0.8)
            ax1.bar(x + width/2, real_vals, width, label='Real Data', color=C_REAL['total'], alpha=0.8)
            ax1.set_xticks(x)
        else:
            ax1.bar(range(len(labels)), sim_vals, color=C_SIM['total'], alpha=0.8)
            ax1.set_xticks(range(len(labels)))
        ax1.set_xticklabels(labels)
        ax1.set_ylabel('Ratio')
        ax1.set_title('Negative Emotion Indicators', fontweight='bold')
        add_legend(ax1)
        add_grid(ax1)
        save_figure(fig1, self.output_dir / 'negative_emotion_indicators.png')
        
        # Figure 2: Negative emotion breakdown
        fig2, ax2 = plt.subplots(figsize=FIG_SIZE)
        neg_keywords = ['愤怒', '厌恶', '焦虑', '悲伤']
        neg_emotions_sim = [sim_neg.get('emotion_breakdown', {}).get(e, 0) for e in neg_keywords]
        if has_real and 'real' in metrics:
            neg_emotions_real = [metrics['real'].get('emotion_breakdown', {}).get(e, 0) for e in neg_keywords]
            x = np.arange(len(neg_keywords))
            width = 0.35
            ax2.bar(x - width/2, neg_emotions_sim, width, label='Simulation', color='#d62728', alpha=0.8)
            ax2.bar(x + width/2, neg_emotions_real, width, label='Real Data', color='#8c564b', alpha=0.8)
            ax2.set_xticks(x)
        else:
            ax2.bar(range(len(neg_keywords)), neg_emotions_sim, color='#d62728', alpha=0.8)
            ax2.set_xticks(range(len(neg_keywords)))
        ax2.set_xticklabels(['Anger', 'Disgust', 'Anxiety', 'Sadness'])
        ax2.set_ylabel('Count')
        ax2.set_title('Negative Emotion Breakdown', fontweight='bold')
        add_legend(ax2)
        add_grid(ax2)
        save_figure(fig2, self.output_dir / 'negative_emotion_breakdown.png')
        
        return metrics
    
    def _compute_negative_metrics(self, texts, actions):
        """计算负面情绪指标"""
        total = len(texts) if texts else 1
        
        # 负面关键词
        neg_keywords = []
        for emo in ['Anger', 'Disgust', 'Anxiety', 'Sadness']:
            neg_keywords.extend(EMOTION_KEYWORDS.get(emo, []))
        
        neg_count = sum(1 for t in texts if t and any(kw in t for kw in neg_keywords))
        
        # 情绪分类
        emotion_breakdown = Counter()
        for a in actions:
            emo = a.get('emotion', 'Neutral')
            if emo in ['Anger', 'Disgust', 'Anxiety', 'Sadness']:
                emotion_breakdown[emo] += 1
        
        # 高强度情绪
        high_intensity = 0
        extreme_stance = 0
        for a in actions:
            intensity = a.get('emotion_intensity', 0.5)
            if isinstance(intensity, (int, float)) and intensity > 0.5:
                high_intensity += 1
            
            stance = a.get('stance_value', 0)
            if isinstance(stance, (int, float)) and abs(stance) > 0.5:
                extreme_stance += 1
        
        total_actions = max(len(actions), 1)
        
        return {
            'negative_count': neg_count,
            'negative_ratio': float(neg_count / total),
            'high_intensity_count': high_intensity,
            'high_intensity_ratio': float(high_intensity / total_actions),
            'extreme_stance_count': extreme_stance,
            'extreme_stance_ratio': float(extreme_stance / total_actions),
            'emotion_breakdown': dict(emotion_breakdown)
        }
    
    def _compute_evolution_index(self, results, has_real):
        """计算综合舆情演化指数"""
        index = {}
        
        # 各维度分数
        confrontation = results.get('discourse_confrontation', {})
        semantic = results.get('semantic_similarity', {})
        negative = results.get('negative_emotion', {})
        
        # 维度1: 话语对抗性相似度
        if has_real and 'confrontation_similarity' in confrontation:
            index['confrontation_similarity'] = float(confrontation['confrontation_similarity'])
        
        # 维度2: 语义相似度
        if semantic and 'centroid_similarity' in semantic:
            index['semantic_similarity'] = float(semantic['centroid_similarity'])
        
        # 维度3: 负面情绪相似度
        if has_real and 'negative_similarity' in negative:
            index['negative_emotion_similarity'] = float(negative['negative_similarity'])
        
        # 综合指数（加权平均）
        components = []
        weights = []
        if 'confrontation_similarity' in index:
            components.append(index['confrontation_similarity'])
            weights.append(0.35)
        if 'semantic_similarity' in index:
            components.append(index['semantic_similarity'])
            weights.append(0.35)
        if 'negative_emotion_similarity' in index:
            components.append(index['negative_emotion_similarity'])
            weights.append(0.30)
        
        if components:
            total_weight = sum(weights)
            index['overall_index'] = float(sum(c * w for c, w in zip(components, weights)) / total_weight)
        else:
            index['overall_index'] = 0.0
        
        return index
    
    def _plot_radar(self, results, has_real):
        """绘制综合雷达图"""
        index = results.get('opinion_evolution_index', {})
        
        if not index:
            return
        
        labels = []
        values = []
        for k, v in index.items():
            if k != 'overall_index' and isinstance(v, (int, float)):
                labels.append(k.replace('_', '\n').title())
                values.append(v)
        
        if len(labels) < 3:
            return
        
        values += values[:1]
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        ax.fill(angles, values, color=C_SIM['total'], alpha=0.25)
        ax.plot(angles, values, color=C_SIM['total'], linewidth=2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, size=9)
        ax.set_ylim(0, 1)
        
        overall = index.get('overall_index', 0)
        ax.set_title(f'Opinion Evolution Index\nOverall: {overall:.2%}',
                    fontweight='bold', size=14, pad=20)
        
        save_figure(fig, self.output_dir / 'opinion_evolution_radar.png')
    
    def _analyze_semantic_diversity(self, sim_texts, real_texts, embedding_model, has_real):
        """语义多样性: 计算文本集合语义空间的扩散程度"""
        metrics = {}
        if not embedding_model:
            return metrics

        try:
            from sklearn.metrics.pairwise import cosine_similarity
            import random

            n_sample = min(300, len(sim_texts))
            sim_sample = random.sample(sim_texts, n_sample) if len(sim_texts) > n_sample else sim_texts
            sim_emb = embedding_model.encode(sim_sample, normalize_embeddings=True)

            sim_centroid = np.mean(sim_emb, axis=0)
            sim_dists = 1 - cosine_similarity(sim_emb, [sim_centroid]).flatten()
            metrics['sim_avg_spread'] = float(np.mean(sim_dists))
            metrics['sim_std_spread'] = float(np.std(sim_dists))

            if has_real and real_texts:
                n_sample_r = min(300, len(real_texts))
                real_sample = random.sample(real_texts, n_sample_r) if len(real_texts) > n_sample_r else real_texts
                real_emb = embedding_model.encode(real_sample, normalize_embeddings=True)

                real_centroid = np.mean(real_emb, axis=0)
                real_dists = 1 - cosine_similarity(real_emb, [real_centroid]).flatten()
                metrics['real_avg_spread'] = float(np.mean(real_dists))
                metrics['real_std_spread'] = float(np.std(real_dists))

                centroid_sim = float(cosine_similarity([sim_centroid], [real_centroid])[0][0])
                metrics['centroid_similarity'] = centroid_sim

                ratio = metrics['sim_avg_spread'] / max(metrics['real_avg_spread'], 1e-10)
                metrics['diversity_ratio'] = float(ratio)
                metrics['diversity_similarity'] = float(min(ratio, 1 / max(ratio, 1e-10)))
                print(f"      语义多样性比值: {ratio:.4f}, 质心相似度: {centroid_sim:.4f}")
        except Exception as e:
            logger.warning(f"语义多样性计算失败: {e}")
        return metrics

    def _analyze_stance_richness(self, micro_results, real_actions, has_real):
        """立场表达丰富度: 统计不同立场表达方式和叙事策略的种类数"""
        metrics = {}

        sim_stances = set()
        sim_narratives = set()
        sim_styles = set()
        for a in micro_results:
            es = a.get('expression_strategy', {})
            s = es.get('stance', a.get('stance', ''))
            if s:
                sim_stances.add(str(s))
            n = es.get('narrative_strategy', a.get('narrative', ''))
            if n:
                sim_narratives.add(str(n))
            st = es.get('expression_style', a.get('style', ''))
            if st:
                sim_styles.add(str(st))

        metrics['sim_unique_stances'] = len(sim_stances)
        metrics['sim_unique_narratives'] = len(sim_narratives)
        metrics['sim_unique_styles'] = len(sim_styles)
        metrics['sim_total_richness'] = len(sim_stances) + len(sim_narratives) + len(sim_styles)

        if has_real and real_actions:
            real_stances = set()
            real_narratives = set()
            real_styles = set()
            for a in real_actions:
                s = a.get('stance', '')
                if s:
                    real_stances.add(str(s))
                ns = a.get('narrative_strategies', [])
                if isinstance(ns, list):
                    real_narratives.update(ns)
                elif ns:
                    real_narratives.add(str(ns))
                es = a.get('expression_styles', [])
                if isinstance(es, list):
                    real_styles.update(es)
                elif es:
                    real_styles.add(str(es))

            metrics['real_unique_stances'] = len(real_stances)
            metrics['real_unique_narratives'] = len(real_narratives)
            metrics['real_unique_styles'] = len(real_styles)
            metrics['real_total_richness'] = len(real_stances) + len(real_narratives) + len(real_styles)

            sim_r = metrics['sim_total_richness']
            real_r = metrics['real_total_richness']
            metrics['richness_ratio'] = float(min(sim_r, real_r) / max(sim_r, real_r)) if max(sim_r, real_r) > 0 else 0
            print(f"      立场丰富度 - 模拟: {sim_r}, 真实: {real_r}, 比值: {metrics['richness_ratio']:.4f}")
        return metrics

    def _compute_content_layer_metrics(self, sim_texts, real_texts, micro_results, real_actions, has_real):
        """计算内容层面附加指标（纯指标计算，无可视化）"""
        metrics = {}

        # --- 1. Lexical Diversity (Type-Token Ratio) ---
        def _ttr(texts):
            all_tokens = []
            for t in texts:
                if t:
                    all_tokens.extend(t.strip().split())
            if not all_tokens:
                return 0.0, 0, 0
            unique = set(all_tokens)
            return len(unique) / len(all_tokens), len(unique), len(all_tokens)

        sim_ttr, sim_unique_words, sim_total_words = _ttr(sim_texts)
        metrics['sim_lexical_ttr'] = float(sim_ttr)
        metrics['sim_unique_words'] = sim_unique_words
        metrics['sim_total_words'] = sim_total_words

        if has_real and real_texts:
            real_ttr, real_unique_words, real_total_words = _ttr(real_texts)
            metrics['real_lexical_ttr'] = float(real_ttr)
            metrics['real_unique_words'] = real_unique_words
            metrics['real_total_words'] = real_total_words
            metrics['ttr_diff'] = float(abs(sim_ttr - real_ttr))
            metrics['ttr_similarity'] = float(max(0, 1 - abs(sim_ttr - real_ttr) * 5))

        # --- 2. Content Length Distribution Similarity ---
        sim_lengths = [len(t) for t in sim_texts if t]
        metrics['sim_avg_length'] = float(np.mean(sim_lengths)) if sim_lengths else 0.0
        metrics['sim_std_length'] = float(np.std(sim_lengths)) if sim_lengths else 0.0
        metrics['sim_median_length'] = float(np.median(sim_lengths)) if sim_lengths else 0.0

        if has_real and real_texts:
            real_lengths = [len(t) for t in real_texts if t]
            metrics['real_avg_length'] = float(np.mean(real_lengths)) if real_lengths else 0.0
            metrics['real_std_length'] = float(np.std(real_lengths)) if real_lengths else 0.0
            metrics['real_median_length'] = float(np.median(real_lengths)) if real_lengths else 0.0

            if sim_lengths and real_lengths:
                n_bins = 30
                all_lens = sim_lengths + real_lengths
                bins = np.linspace(0, max(all_lens) + 1, n_bins + 1)
                sim_hist, _ = np.histogram(sim_lengths, bins=bins, density=True)
                real_hist, _ = np.histogram(real_lengths, bins=bins, density=True)
                sim_hist = sim_hist / (sim_hist.sum() + 1e-12)
                real_hist = real_hist / (real_hist.sum() + 1e-12)
                length_jsd = float(calculate_jsd(sim_hist, real_hist))
                metrics['length_distribution_jsd'] = length_jsd
                metrics['length_distribution_similarity'] = float(max(0, 1 - length_jsd))

        # --- 3. Emotion Vocabulary Richness ---
        all_emo_keywords = set()
        for kw_list in EMOTION_KEYWORDS.values():
            all_emo_keywords.update(kw_list)

        def _emotion_vocab_richness(texts):
            found = set()
            for t in texts:
                if t:
                    for kw in all_emo_keywords:
                        if kw in t:
                            found.add(kw)
            return found

        sim_emo_found = _emotion_vocab_richness(sim_texts)
        metrics['sim_emotion_vocab_count'] = len(sim_emo_found)
        metrics['sim_emotion_vocab_ratio'] = float(len(sim_emo_found) / max(len(all_emo_keywords), 1))

        if has_real and real_texts:
            real_emo_found = _emotion_vocab_richness(real_texts)
            metrics['real_emotion_vocab_count'] = len(real_emo_found)
            metrics['real_emotion_vocab_ratio'] = float(len(real_emo_found) / max(len(all_emo_keywords), 1))
            overlap = len(sim_emo_found & real_emo_found)
            union = len(sim_emo_found | real_emo_found)
            metrics['emotion_vocab_jaccard'] = float(overlap / max(union, 1))

        # --- 4. Narrative Strategy Diversity (Shannon Entropy) ---
        def _narrative_entropy(actions, strategy_key='narrative_strategy', nested_key='expression_strategy'):
            counts = Counter()
            for a in actions:
                ns = a.get(nested_key, {})
                if isinstance(ns, dict):
                    val = ns.get(strategy_key, '')
                else:
                    val = a.get(strategy_key, a.get('narrative', ''))
                if val:
                    counts[str(val)] += 1
            total = sum(counts.values())
            if total == 0:
                return 0.0, 0, {}
            probs = [c / total for c in counts.values()]
            entropy = -sum(p * math.log2(p) for p in probs if p > 0)
            return entropy, len(counts), dict(counts)

        sim_entropy, sim_n_strategies, sim_strat_dist = _narrative_entropy(micro_results)
        metrics['sim_narrative_entropy'] = float(sim_entropy)
        metrics['sim_narrative_strategy_count'] = sim_n_strategies
        if sim_n_strategies > 0:
            metrics['sim_narrative_entropy_normalized'] = float(sim_entropy / math.log2(max(sim_n_strategies, 2)))

        if has_real and real_actions:
            real_entropy, real_n_strategies, real_strat_dist = _narrative_entropy(
                real_actions, strategy_key='narrative_strategies', nested_key='_unused_')
            metrics['real_narrative_entropy'] = float(real_entropy)
            metrics['real_narrative_strategy_count'] = real_n_strategies
            if real_n_strategies > 0:
                metrics['real_narrative_entropy_normalized'] = float(real_entropy / math.log2(max(real_n_strategies, 2)))
            if sim_entropy > 0 and real_entropy > 0:
                max_e = max(sim_entropy, real_entropy)
                metrics['narrative_entropy_similarity'] = float(min(sim_entropy, real_entropy) / max_e)

        # --- 5. Topic Focus Index (Herfindahl Index of topic distribution) ---
        def _topic_focus(actions, topic_key='topic'):
            counts = Counter()
            for a in actions:
                t = a.get(topic_key, a.get('stance', ''))
                if t:
                    counts[str(t)] += 1
            total = sum(counts.values())
            if total == 0:
                return 0.0, 0
            shares = [c / total for c in counts.values()]
            hhi = sum(s ** 2 for s in shares)
            return hhi, len(counts)

        sim_hhi, sim_n_topics = _topic_focus(micro_results)
        metrics['sim_topic_hhi'] = float(sim_hhi)
        metrics['sim_topic_count'] = sim_n_topics

        if has_real and real_actions:
            real_hhi, real_n_topics = _topic_focus(real_actions)
            metrics['real_topic_hhi'] = float(real_hhi)
            metrics['real_topic_count'] = real_n_topics
            if sim_hhi > 0 and real_hhi > 0:
                metrics['topic_focus_diff'] = float(abs(sim_hhi - real_hhi))
                metrics['topic_focus_similarity'] = float(max(0, 1 - abs(sim_hhi - real_hhi) * 5))

        return metrics

    def _analyze_emotion_contagion(self, sim_agg_actions, micro_results, real_actions, has_real):
        """情绪传染效率: 高强度帖子后续智能体情绪变化"""
        metrics = {}

        def _compute_contagion(actions, threshold=0.7, window=5):
            sorted_actions = sorted(actions, key=lambda a: a.get('time', ''))
            contagion_effects = []
            for i, a in enumerate(sorted_actions):
                intensity = a.get('emotion_intensity',
                                  a.get('expression_strategy', {}).get('emotion_intensity', 0.5))
                try:
                    intensity = float(intensity)
                except (ValueError, TypeError):
                    intensity = 0.5
                if intensity >= threshold:
                    uid = a.get('user_id', '')
                    subsequent = sorted_actions[i + 1: i + 1 + window]
                    other_intensities = []
                    for sa in subsequent:
                        if sa.get('user_id', '') != uid:
                            si = sa.get('emotion_intensity',
                                        sa.get('expression_strategy', {}).get('emotion_intensity', 0.5))
                            try:
                                other_intensities.append(float(si))
                            except (ValueError, TypeError):
                                pass
                    if other_intensities:
                        contagion_effects.append(float(np.mean(other_intensities)))
            return contagion_effects

        sim_effects = _compute_contagion(micro_results if micro_results else sim_agg_actions)
        if sim_effects:
            metrics['sim_avg_contagion'] = float(np.mean(sim_effects))
            metrics['sim_contagion_count'] = len(sim_effects)

        if has_real and real_actions:
            real_effects = _compute_contagion(real_actions)
            if real_effects:
                metrics['real_avg_contagion'] = float(np.mean(real_effects))
                metrics['real_contagion_count'] = len(real_effects)

            if sim_effects and real_effects:
                diff = abs(metrics['sim_avg_contagion'] - metrics['real_avg_contagion'])
                metrics['contagion_similarity'] = float(max(0, 1 - diff * 2))
                print(f"      情绪传染效率 - 模拟: {metrics['sim_avg_contagion']:.4f}, "
                      f"真实: {metrics['real_avg_contagion']:.4f}")
        return metrics

    def _plot_semantic_content(self, results, has_real):
        """绘制语义内容图（拆分为独立图）"""
        # Figure 1: Semantic diversity
        fig1, ax1 = plt.subplots(figsize=FIG_SIZE)
        sd = results.get('semantic_diversity', {})
        if sd:
            labels = ['Avg Spread', 'Std Spread']
            sv = [sd.get('sim_avg_spread', 0), sd.get('sim_std_spread', 0)]
            if has_real and 'real_avg_spread' in sd:
                rv = [sd.get('real_avg_spread', 0), sd.get('real_std_spread', 0)]
                x = np.arange(len(labels))
                w = 0.35
                ax1.bar(x - w / 2, sv, w, label='Simulation', color=C_SIM['total'], alpha=0.85)
                ax1.bar(x + w / 2, rv, w, label='Real Data', color=C_REAL['total'], alpha=0.85)
                ax1.set_xticks(x)
            else:
                ax1.bar(labels, sv, color=C_SIM['total'], alpha=0.85)
                ax1.set_xticks(range(len(labels)))
            ax1.set_xticklabels(labels)
            add_legend(ax1)
        ax1.set_ylabel('Cosine Distance')
        ax1.set_title('Semantic Diversity', fontweight='bold')
        add_grid(ax1)
        save_figure(fig1, self.output_dir / 'semantic_diversity.png')

        # Figure 2: Stance expression richness
        fig2, ax2 = plt.subplots(figsize=FIG_SIZE)
        sr = results.get('stance_richness', {})
        if sr:
            cats = ['Stances', 'Narratives', 'Styles']
            sv = [sr.get('sim_unique_stances', 0), sr.get('sim_unique_narratives', 0), sr.get('sim_unique_styles', 0)]
            if has_real and 'real_unique_stances' in sr:
                rv = [sr.get('real_unique_stances', 0), sr.get('real_unique_narratives', 0), sr.get('real_unique_styles', 0)]
                x = np.arange(len(cats))
                w = 0.35
                ax2.bar(x - w / 2, sv, w, label='Simulation', color=C_SIM['total'], alpha=0.85)
                ax2.bar(x + w / 2, rv, w, label='Real Data', color=C_REAL['total'], alpha=0.85)
                ax2.set_xticks(x)
            else:
                ax2.bar(cats, sv, color=C_SIM['total'], alpha=0.85)
                ax2.set_xticks(range(len(cats)))
            ax2.set_xticklabels(cats)
            add_legend(ax2)
        ax2.set_ylabel('Unique Count')
        ax2.set_title('Stance Expression Richness', fontweight='bold')
        add_grid(ax2)
        save_figure(fig2, self.output_dir / 'stance_expression_richness.png')

        # Figure 3: Emotion contagion efficiency
        fig3, ax3 = plt.subplots(figsize=FIG_SIZE)
        ec = results.get('emotion_contagion', {})
        if ec:
            labels_ec = ['Simulation']
            vals = [ec.get('sim_avg_contagion', 0)]
            colors = [C_SIM['total']]
            if has_real and 'real_avg_contagion' in ec:
                labels_ec.append('Real Data')
                vals.append(ec.get('real_avg_contagion', 0))
                colors.append(C_REAL['total'])
            bars = ax3.bar(labels_ec, vals, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
            for bar, val in zip(bars, vals):
                ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{val:.4f}', ha='center', fontsize=FONT_SIZE['annotation'])
        ax3.set_ylabel('Avg Intensity After Trigger')
        ax3.set_title('Emotion Contagion Efficiency', fontweight='bold')
        add_grid(ax3)
        save_figure(fig3, self.output_dir / 'emotion_contagion_efficiency.png')

        # Figure 4: Opinion evolution index (horizontal bar)
        fig4, ax4 = plt.subplots(figsize=FIG_SIZE)
        index = results.get('opinion_evolution_index', {})
        if index:
            items = [(k, v) for k, v in index.items()
                     if k != 'overall_index' and isinstance(v, (int, float))]
            if items:
                labels_r = [k.replace('_', '\n').title() for k, v in items]
                vals = [v for k, v in items]
                bars = ax4.barh(labels_r, vals, color='#1f77b4', alpha=0.85, edgecolor='black', linewidth=0.5)
                for bar, val in zip(bars, vals):
                    ax4.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                            f'{val:.3f}', ha='left', va='center', fontsize=FONT_SIZE['annotation'])
                ax4.set_xlim(0, 1.15)
                overall = index.get('overall_index', 0)
                ax4.set_title(f'Opinion Evolution Index ({overall:.3f})', fontweight='bold')
        add_grid(ax4)
        save_figure(fig4, self.output_dir / 'opinion_evolution_index_bar.png')
        logger.info("[SAVED] semantic content figures (4 individual)")

    def _print_summary(self, results):
        """打印摘要"""
        conf = results.get('discourse_confrontation', {})
        if conf:
            print(f"    ✅ 话语对抗性: 模拟={conf.get('sim', {}).get('conflict_score', 0):.4f}")
            if 'confrontation_similarity' in conf:
                print(f"       对抗性相似度: {conf['confrontation_similarity']:.4f}")
        
        sem = results.get('semantic_similarity', {})
        if sem and 'centroid_similarity' in sem:
            print(f"    ✅ 语义质心相似度: {sem['centroid_similarity']:.4f}")
            print(f"       平均语义相似度: {sem.get('avg_similarity', 0):.4f}")
        
        neg = results.get('negative_emotion', {})
        if neg:
            print(f"    ✅ 负面情绪比例: 模拟={neg.get('sim', {}).get('negative_ratio', 0):.2%}")
        
        index = results.get('opinion_evolution_index', {})
        if index:
            print(f"    ✅ 综合舆情演化指数: {index.get('overall_index', 0):.4f}")

        sd = results.get('semantic_diversity', {})
        if sd and 'centroid_similarity' in sd:
            print(f"    ✅ 语义质心相似度: {sd['centroid_similarity']:.4f}")
            print(f"       语义多样性比值: {sd.get('diversity_ratio', 0):.4f}")

        sr = results.get('stance_richness', {})
        if sr and 'richness_ratio' in sr:
            print(f"    ✅ 立场表达丰富度比值: {sr['richness_ratio']:.4f}")

        ec = results.get('emotion_contagion', {})
        if ec and 'contagion_similarity' in ec:
            print(f"    ✅ 情绪传染效率相似度: {ec['contagion_similarity']:.4f}")

        cl = results.get('content_layer_metrics', {})
        if cl:
            print(f"    ✅ 词汇多样性 (TTR): 模拟={cl.get('sim_lexical_ttr', 0):.4f}")
            if 'real_lexical_ttr' in cl:
                print(f"       真实TTR={cl['real_lexical_ttr']:.4f}, 相似度={cl.get('ttr_similarity', 0):.4f}")
            print(f"    ✅ 文本平均长度: 模拟={cl.get('sim_avg_length', 0):.1f}")
            if 'length_distribution_similarity' in cl:
                print(f"       长度分布相似度={cl['length_distribution_similarity']:.4f}")
            print(f"    ✅ 情绪词汇丰富度: 模拟={cl.get('sim_emotion_vocab_count', 0)}种")
            if 'emotion_vocab_jaccard' in cl:
                print(f"       Jaccard相似度={cl['emotion_vocab_jaccard']:.4f}")
            print(f"    ✅ 叙事策略熵: 模拟={cl.get('sim_narrative_entropy', 0):.4f} ({cl.get('sim_narrative_strategy_count', 0)}种)")
            print(f"    ✅ 话题集中度 (HHI): 模拟={cl.get('sim_topic_hhi', 0):.4f} ({cl.get('sim_topic_count', 0)}个话题)")
