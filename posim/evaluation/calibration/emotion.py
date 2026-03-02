# -*- coding: utf-8 -*-
"""
情绪/情感曲线校准

包含:
1. 情绪曲线对比（去除中性的Top3情绪类别，数值/归一化/占比）
2. 情感曲线对比（正面/中立/负面，数值/归一化/占比）
3. 情绪值校准（SnowNLP / TextBlob等库计算情感值曲线）
4. 情感极化程度
5. 情绪煽动程度
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from collections import Counter, defaultdict

from ..base import BaseEvaluator
from ..utils import (
    smooth, normalize, save_json, calculate_curve_similarity,
    calculate_jsd, calculate_entropy, calculate_normalized_entropy,
    calculate_kendall_tau, calculate_ks_test, detect_inflection_points,
    EMOTION_KEYWORDS, INTENSITY_MAP
)
from ..visualization import (
    FIG_SIZE, FIG_SIZE_TALL, FIG_SIZE_WIDE, LW, LW_MINOR, ALPHA,
    MARKER_SIZE, FONT_SIZE,
    C_SIM, C_REAL, C_EMOTION, C_SENTIMENT,
    setup_time_axis, add_grid, add_legend, save_figure
)

logger = logging.getLogger(__name__)


class EmotionCalibrationEvaluator(BaseEvaluator):
    """情绪/情感曲线校准"""
    
    def __init__(self, output_dir: Path):
        super().__init__(output_dir / "emotion_calibration", name="emotion_calibration")
    
    def evaluate(self, sim_data: Dict[str, Any], real_data: Optional[Dict[str, Any]] = None,
                 **kwargs) -> Dict[str, Any]:
        """执行情绪/情感校准"""
        self._log_section("情绪/情感曲线校准")
        
        sim_agg = sim_data.get('aggregated', {})
        sim_times = sim_agg.get('times', [])
        sim_emotion = sim_agg.get('emotion', {})
        sim_sentiment = sim_agg.get('sentiment', {})
        sim_actions = sim_agg.get('actions', [])
        
        if not sim_times:
            print("    ⚠️ 无模拟数据，跳过")
            return {}
        
        results = {}
        has_real = real_data is not None and real_data.get('times')
        
        real_times = real_data.get('times', []) if has_real else []
        real_emotion = real_data.get('emotion', {}) if has_real else {}
        real_sentiment = real_data.get('sentiment', {}) if has_real else {}
        real_actions = real_data.get('actions', []) if has_real else []
        
        # 1. 情绪曲线对比
        print("    [1/5] 情绪曲线对比...")
        results['emotion_curves'] = self._analyze_emotion_curves(
            sim_times, sim_emotion, real_times, real_emotion, has_real)
        
        # 2. 情感曲线对比（正/中/负）
        print("    [2/5] 情感曲线对比...")
        results['sentiment_curves'] = self._analyze_sentiment_curves(
            sim_times, sim_sentiment, real_times, real_sentiment, has_real)
        
        # 3. 情绪值校准
        print("    [3/5] 情绪值校准...")
        results['sentiment_score'] = self._analyze_sentiment_score(
            sim_times, sim_actions, real_times, real_actions, has_real)
        
        # 4. 情感极化程度
        print("    [4/5] 情感极化程度...")
        results['emotion_polarization'] = self._analyze_emotion_polarization(
            sim_actions, real_actions, has_real)
        
        # 5. 情绪煽动程度
        print("    [5/6] 情绪煽动程度...")
        results['emotion_incitement'] = self._analyze_emotion_incitement(
            sim_times, sim_actions, real_times, real_actions, has_real)

        # 6. 情感动态指标
        print("    [6/6] 情感动态指标...")
        results['emotion_dynamics'] = self._analyze_emotion_dynamics(
            sim_times, sim_emotion, sim_sentiment, sim_actions,
            real_times, real_emotion, real_sentiment, real_actions, has_real)

        self._save_results(results, "emotion_calibration_metrics.json")
        self._print_summary(results)
        return results
    
    def _analyze_emotion_curves(self, sim_times, sim_emotion, real_times, real_emotion, has_real):
        """情绪曲线对比（去除中性的Top3）"""
        metrics = {}
        
        # 获取非Neutral的Top3情绪
        sim_totals = {e: sum(v) for e, v in sim_emotion.items() if e != 'Neutral'}
        top3_sim = sorted(sim_totals.keys(), key=lambda x: sim_totals.get(x, 0), reverse=True)[:3]
        
        if has_real:
            real_totals = {e: sum(v) for e, v in real_emotion.items() if e != 'Neutral'}
            top3_real = sorted(real_totals.keys(), key=lambda x: real_totals.get(x, 0), reverse=True)[:3]
            all_top = list(dict.fromkeys(top3_sim + top3_real))[:5]  # 合并取前5
        else:
            all_top = top3_sim
            top3_real = []
        
        metrics['top_emotions_sim'] = top3_sim
        metrics['top_emotions_real'] = top3_real if has_real else []
        metrics['sim_emotion_totals'] = {e: int(sum(sim_emotion.get(e, []))) for e in sim_emotion}
        if has_real:
            metrics['real_emotion_totals'] = {e: int(sum(real_emotion.get(e, []))) for e in real_emotion}
        
        if not all_top:
            return metrics
        
        # 绘制Top3情绪: 数值曲线、归一化曲线、占比曲线
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        # === 3a. 数值曲线 ===
        ax = axes[0]
        for emo in all_top:
            c = C_EMOTION.get(emo, '#888888')
            if emo in sim_emotion:
                ax.plot(sim_times[:len(sim_emotion[emo])],
                       smooth(sim_emotion[emo], target_len=len(sim_emotion[emo])),
                       color=c, lw=LW, label=f'Sim-{emo}')
            if has_real and emo in real_emotion:
                ax.plot(real_times[:len(real_emotion[emo])],
                       smooth(real_emotion[emo], target_len=len(real_emotion[emo])),
                       color=c, lw=LW, ls='--', alpha=0.7, label=f'Real-{emo}')
        ax.set_ylabel('Count')
        ax.set_title('Top Emotion Curves (Absolute Values)', fontweight='bold')
        setup_time_axis(ax)
        add_legend(ax, fontsize=8, ncol=2)
        add_grid(ax)
        
        # === 3b. 归一化曲线 ===
        ax = axes[1]
        for emo in all_top:
            c = C_EMOTION.get(emo, '#888888')
            if emo in sim_emotion:
                norm_vals = normalize(sim_emotion[emo])
                ax.plot(sim_times[:len(norm_vals)], smooth(norm_vals, target_len=len(norm_vals)),
                       color=c, lw=LW, label=f'Sim-{emo}')
            if has_real and emo in real_emotion:
                norm_vals = normalize(real_emotion[emo])
                ax.plot(real_times[:len(norm_vals)], smooth(norm_vals, target_len=len(norm_vals)),
                       color=c, lw=LW, ls='--', alpha=0.7, label=f'Real-{emo}')
        ax.set_ylabel('Normalized Value')
        ax.set_title('Top Emotion Curves (Normalized)', fontweight='bold')
        setup_time_axis(ax)
        add_legend(ax, fontsize=8, ncol=2)
        add_grid(ax)
        
        # === 3c. 各类别占比曲线 ===
        ax = axes[2]
        if sim_emotion:
            n = len(sim_times)
            for emo in all_top:
                if emo in sim_emotion:
                    d = sim_emotion[emo]
                    # 计算每个时间点的占比
                    total_per_t = [sum(sim_emotion[e][t] if t < len(sim_emotion[e]) else 0
                                      for e in sim_emotion if e != 'Neutral')
                                   for t in range(len(d))]
                    ratio = [d[t] / max(total_per_t[t], 1) for t in range(len(d))]
                    c = C_EMOTION.get(emo, '#888888')
                    ax.plot(sim_times[:len(ratio)], smooth(ratio, target_len=len(ratio)),
                           color=c, lw=LW, label=f'Sim-{emo}')
        if has_real and real_emotion:
            for emo in all_top:
                if emo in real_emotion:
                    d = real_emotion[emo]
                    total_per_t = [sum(real_emotion[e][t] if t < len(real_emotion[e]) else 0
                                      for e in real_emotion if e != 'Neutral')
                                   for t in range(len(d))]
                    ratio = [d[t] / max(total_per_t[t], 1) for t in range(len(d))]
                    c = C_EMOTION.get(emo, '#888888')
                    ax.plot(real_times[:len(ratio)], smooth(ratio, target_len=len(ratio)),
                           color=c, lw=LW, ls='--', alpha=0.7, label=f'Real-{emo}')
        ax.set_ylabel('Proportion')
        ax.set_xlabel('Time')
        ax.set_title('Top Emotion Proportion Over Time', fontweight='bold')
        setup_time_axis(ax)
        add_legend(ax, fontsize=8, ncol=2)
        add_grid(ax)
        
        save_figure(fig, self.output_dir / 'emotion_top3_curves.png')
        
        # 计算情绪分布相似度
        if has_real:
            all_emos = set(sim_emotion.keys()) | set(real_emotion.keys())
            sim_vec = np.array([sum(sim_emotion.get(e, [0])) for e in sorted(all_emos)])
            real_vec = np.array([sum(real_emotion.get(e, [0])) for e in sorted(all_emos)])
            if sim_vec.sum() > 0:
                sim_vec = sim_vec / sim_vec.sum()
            if real_vec.sum() > 0:
                real_vec = real_vec / real_vec.sum()
            jsd = calculate_jsd(sim_vec, real_vec)
            metrics['emotion_distribution_jsd'] = float(jsd)
            metrics['emotion_distribution_similarity'] = float(1 - jsd)
            print(f"      情绪分布JSD: {jsd:.4f}, 相似度: {1-jsd:.4f}")
            
            # 各情绪曲线相似度
            metrics['per_emotion_similarity'] = {}
            for emo in all_top:
                if emo in sim_emotion and emo in real_emotion:
                    sim_c = normalize(sim_emotion[emo]).tolist()
                    real_c = normalize(real_emotion[emo]).tolist()
                    metrics['per_emotion_similarity'][emo] = calculate_curve_similarity(sim_c, real_c)
        
        return metrics
    
    def _analyze_sentiment_curves(self, sim_times, sim_sentiment, real_times, real_sentiment, has_real):
        """情感曲线对比（正/中/负）"""
        metrics = {}
        
        # 绘制3行图: 数值、归一化、占比
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        sent_labels = {'positive': 'Positive', 'neutral': 'Neutral', 'negative': 'Negative'}
        
        # === 数值曲线 ===
        ax = axes[0]
        for sent_key, label in sent_labels.items():
            c = C_SENTIMENT.get(sent_key, '#888888')
            if sent_key in sim_sentiment:
                d = sim_sentiment[sent_key]
                ax.plot(sim_times[:len(d)], smooth(d, target_len=len(d)),
                       color=c, lw=LW, label=f'Sim-{label}')
            if has_real and sent_key in real_sentiment:
                d = real_sentiment[sent_key]
                ax.plot(real_times[:len(d)], smooth(d, target_len=len(d)),
                       color=c, lw=LW, ls='--', alpha=0.7, label=f'Real-{label}')
        ax.set_ylabel('Count')
        ax.set_title('Sentiment Curves (Absolute Values)', fontweight='bold')
        setup_time_axis(ax)
        add_legend(ax, fontsize=8, ncol=2)
        add_grid(ax)
        
        # === 归一化曲线 ===
        ax = axes[1]
        for sent_key, label in sent_labels.items():
            c = C_SENTIMENT.get(sent_key, '#888888')
            if sent_key in sim_sentiment:
                d = sim_sentiment[sent_key]
                ax.plot(sim_times[:len(d)], normalize(smooth(d, target_len=len(d))),
                       color=c, lw=LW, label=f'Sim-{label}')
            if has_real and sent_key in real_sentiment:
                d = real_sentiment[sent_key]
                ax.plot(real_times[:len(d)], normalize(smooth(d, target_len=len(d))),
                       color=c, lw=LW, ls='--', alpha=0.7, label=f'Real-{label}')
        ax.set_ylabel('Normalized')
        ax.set_title('Sentiment Curves (Normalized)', fontweight='bold')
        setup_time_axis(ax)
        add_legend(ax, fontsize=8, ncol=2)
        add_grid(ax)
        
        # === 占比曲线 ===
        ax = axes[2]
        for src, times, sentiment, ls_style, prefix in [
            ('sim', sim_times, sim_sentiment, '-', 'Sim'),
            ('real', real_times, real_sentiment, '--', 'Real')
        ]:
            if src == 'real' and not has_real:
                continue
            if not sentiment:
                continue
            for sent_key, label in sent_labels.items():
                if sent_key not in sentiment:
                    continue
                d = sentiment[sent_key]
                total_per_t = [sum(sentiment.get(sk, [0] * len(d))[t] if t < len(sentiment.get(sk, [])) else 0
                                  for sk in sent_labels.keys())
                               for t in range(len(d))]
                ratio = [d[t] / max(total_per_t[t], 1) for t in range(len(d))]
                c = C_SENTIMENT.get(sent_key, '#888888')
                ax.plot(times[:len(ratio)], smooth(ratio, target_len=len(ratio)),
                       color=c, lw=LW, ls=ls_style, label=f'{prefix}-{label}')
        ax.set_ylabel('Proportion')
        ax.set_xlabel('Time')
        ax.set_title('Sentiment Proportion Over Time', fontweight='bold')
        setup_time_axis(ax)
        add_legend(ax, fontsize=8, ncol=2)
        add_grid(ax)
        
        save_figure(fig, self.output_dir / 'sentiment_curves.png')
        
        # 计算统计指标
        for src, sentiment, prefix in [('sim', sim_sentiment, 'sim'), ('real', real_sentiment, 'real')]:
            if src == 'real' and not has_real:
                continue
            totals = {k: sum(v) for k, v in sentiment.items()}
            total_all = sum(totals.values())
            metrics[f'{prefix}_sentiment_distribution'] = {k: int(v) for k, v in totals.items()}
            metrics[f'{prefix}_sentiment_ratios'] = {k: float(v / max(total_all, 1)) for k, v in totals.items()}
        
        if has_real:
            # 情感分布JSD
            sim_vec = np.array([sum(sim_sentiment.get(k, [0])) for k in ['positive', 'neutral', 'negative']])
            real_vec = np.array([sum(real_sentiment.get(k, [0])) for k in ['positive', 'neutral', 'negative']])
            if sim_vec.sum() > 0: sim_vec = sim_vec / sim_vec.sum()
            if real_vec.sum() > 0: real_vec = real_vec / real_vec.sum()
            jsd = calculate_jsd(sim_vec, real_vec)
            metrics['sentiment_jsd'] = float(jsd)
            metrics['sentiment_similarity'] = float(1 - jsd)
            print(f"      情感分布JSD: {jsd:.4f}, 相似度: {1-jsd:.4f}")
        
        return metrics
    
    def _analyze_sentiment_score(self, sim_times, sim_actions, real_times, real_actions, has_real):
        """情绪值校准（使用SnowNLP计算情感分数）"""
        metrics = {}
        
        # 计算模拟数据的情感分数曲线
        sim_scores_by_time = self._compute_sentiment_scores_by_time(sim_actions, sim_times)
        metrics['sim_avg_score'] = float(np.mean([s for scores in sim_scores_by_time.values() for s in scores])) if sim_scores_by_time else 0.5
        
        if has_real:
            real_scores_by_time = self._compute_sentiment_scores_by_time_real(real_actions, real_times)
            metrics['real_avg_score'] = float(np.mean([s for scores in real_scores_by_time.values() for s in scores])) if real_scores_by_time else 0.5
        
        # 绘制情感分数曲线
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        
        if sim_scores_by_time and sim_times:
            sim_curve = [np.mean(sim_scores_by_time.get(t, [0.5])) for t in sim_times]
            ax.plot(sim_times, smooth(sim_curve, target_len=len(sim_times)),
                   color=C_SIM['total'], lw=LW, label='Simulation')
        
        if has_real and real_times:
            real_scores_by_time_computed = self._compute_sentiment_scores_by_time_real(real_actions, real_times)
            real_curve = [np.mean(real_scores_by_time_computed.get(t, [0.5])) for t in real_times]
            ax.plot(real_times, smooth(real_curve, target_len=len(real_times)),
                   color=C_REAL['total'], lw=LW, ls='--', label='Real Data')
            
            # 曲线相似度
            metrics['score_curve_similarity'] = calculate_curve_similarity(
                [np.mean(sim_scores_by_time.get(t, [0.5])) for t in sim_times],
                real_curve
            )
        
        ax.axhline(y=0.5, color='gray', ls=':', lw=1, alpha=0.5)
        ax.set_xlabel('Time')
        ax.set_ylabel('Sentiment Score (0=Neg, 1=Pos)')
        ax.set_title('Sentiment Score Over Time (SnowNLP)', fontweight='bold')
        ax.set_ylim(0, 1)
        setup_time_axis(ax)
        add_legend(ax)
        add_grid(ax)
        save_figure(fig, self.output_dir / 'sentiment_score_curve.png')
        
        return metrics
    
    def _compute_sentiment_scores_by_time(self, actions, times):
        """按时间计算模拟数据的情感分数"""
        from ..utils import truncate_time
        scores_by_time = defaultdict(list)
        try:
            from snownlp import SnowNLP
            for a in actions:
                text = a.get('text', a.get('content', ''))
                t = a.get('time')
                if text and t:
                    try:
                        score = SnowNLP(text).sentiments
                        scores_by_time[t].append(score)
                    except:
                        scores_by_time[t].append(0.5)
        except ImportError:
            logger.warning("SnowNLP未安装，使用默认值0.5")
            for a in actions:
                t = a.get('time')
                if t:
                    scores_by_time[t].append(a.get('sentiment_score', 0.5))
        return scores_by_time
    
    def _compute_sentiment_scores_by_time_real(self, actions, times):
        """按时间计算真实数据的情感分数"""
        from ..utils import truncate_time
        scores_by_time = defaultdict(list)
        
        granularity = 10
        if len(times) >= 2:
            diff = (times[1] - times[0]).total_seconds() / 60
            granularity = int(diff) if diff > 0 else 10
        
        for a in actions:
            t = a.get('time')
            if t:
                t_trunc = truncate_time(t, granularity)
                # 优先使用标注的nlp分数
                score = a.get('nlp_sentiment_score', 0.5)
                if score is not None:
                    scores_by_time[t_trunc].append(float(score))
                else:
                    try:
                        from snownlp import SnowNLP
                        text = a.get('content', '')
                        if text:
                            scores_by_time[t_trunc].append(SnowNLP(text).sentiments)
                    except:
                        scores_by_time[t_trunc].append(0.5)
        return scores_by_time
    
    def _analyze_emotion_polarization(self, sim_actions, real_actions, has_real):
        """情感极化程度分析"""
        metrics = {}
        
        # 模拟数据
        sim_scores = self._get_sentiment_scores(sim_actions)
        if sim_scores:
            metrics['sim_polarization'] = self._compute_polarization_metrics(sim_scores, 'sim')
        
        # 真实数据
        if has_real and real_actions:
            real_scores = self._get_sentiment_scores_real(real_actions)
            if real_scores:
                metrics['real_polarization'] = self._compute_polarization_metrics(real_scores, 'real')
        
        # 绘制极化对比图
        fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE_WIDE)
        
        ax = axes[0]
        if sim_scores:
            ax.hist(sim_scores, bins=20, alpha=0.6, color=C_SIM['total'], label='Simulation',
                   density=True, edgecolor='black')
        if has_real and real_actions:
            real_scores = self._get_sentiment_scores_real(real_actions)
            if real_scores:
                ax.hist(real_scores, bins=20, alpha=0.6, color=C_REAL['total'], label='Real Data',
                       density=True, edgecolor='black')
        ax.set_xlabel('Sentiment Score')
        ax.set_ylabel('Density')
        ax.set_title('Sentiment Score Distribution', fontweight='bold')
        add_legend(ax)
        add_grid(ax)
        
        # 极化指标对比
        ax = axes[1]
        labels = ['Polarization\nIndex', 'Bimodality', 'Extreme\nRatio']
        sim_vals = [
            metrics.get('sim_polarization', {}).get('polarization_index', 0),
            metrics.get('sim_polarization', {}).get('bimodality', 0),
            metrics.get('sim_polarization', {}).get('extreme_ratio', 0)
        ]
        if has_real:
            real_vals = [
                metrics.get('real_polarization', {}).get('polarization_index', 0),
                metrics.get('real_polarization', {}).get('bimodality', 0),
                metrics.get('real_polarization', {}).get('extreme_ratio', 0)
            ]
            x = np.arange(len(labels))
            width = 0.35
            ax.bar(x - width/2, sim_vals, width, label='Simulation', color=C_SIM['total'], alpha=0.8)
            ax.bar(x + width/2, real_vals, width, label='Real Data', color=C_REAL['total'], alpha=0.8)
            ax.set_xticks(x)
        else:
            ax.bar(labels, sim_vals, color=C_SIM['total'], alpha=0.8)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Value')
        ax.set_title('Emotion Polarization Metrics', fontweight='bold')
        add_legend(ax)
        add_grid(ax)
        
        save_figure(fig, self.output_dir / 'emotion_polarization.png')
        
        return metrics
    
    def _compute_polarization_metrics(self, scores: List[float], prefix: str) -> Dict:
        """计算极化指标"""
        arr = np.array(scores)
        mean_score = float(np.mean(arr))
        std_score = float(np.std(arr))
        
        # 极化指数: 情感分数偏离0.5的程度
        polarization_index = float(np.mean(np.abs(arr - 0.5)))
        
        # 双峰性
        try:
            from scipy.stats import kurtosis, skew
            n = len(arr)
            s = skew(arr)
            k = kurtosis(arr)
            bimodality = (s ** 2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
            bimodality = float(min(max(bimodality, 0), 1))
        except:
            bimodality = 0.0
        
        # 极端情感比例
        extreme_ratio = float(np.mean((arr < 0.2) | (arr > 0.8)))
        
        return {
            'mean_score': mean_score,
            'std_score': std_score,
            'polarization_index': polarization_index,
            'bimodality': bimodality,
            'extreme_ratio': extreme_ratio
        }
    
    def _get_sentiment_scores(self, actions):
        """获取模拟数据的情感分数"""
        scores = []
        for a in actions:
            s = a.get('sentiment_score', None)
            if s is not None:
                scores.append(float(s))
        if not scores:
            # 尝试用SnowNLP计算
            try:
                from snownlp import SnowNLP
                for a in actions:
                    text = a.get('text', a.get('content', ''))
                    if text:
                        scores.append(SnowNLP(text).sentiments)
            except ImportError:
                pass
        return scores
    
    def _get_sentiment_scores_real(self, actions):
        """获取真实数据的情感分数"""
        scores = []
        for a in actions:
            s = a.get('nlp_sentiment_score', None)
            if s is not None:
                scores.append(float(s))
            else:
                try:
                    from snownlp import SnowNLP
                    text = a.get('content', '')
                    if text:
                        scores.append(SnowNLP(text).sentiments)
                except:
                    pass
        return scores
    
    def _analyze_emotion_incitement(self, sim_times, sim_actions, real_times, real_actions, has_real):
        """情绪煽动程度分析"""
        metrics = {}
        
        # 煽动关键词
        incitement_keywords = [
            '气死', '太过分', '不能忍', '崩溃', '震惊', '可恶', '愤怒',
            '暴怒', '无耻', '混蛋', '该死', '抵制', '必须严惩', '还有天理吗',
            '人神共愤', '天理不容', '不可原谅', '令人发指', '恶心', '作呕',
            '可恨', '怒火', '痛心', '心寒', '太黑暗', '！！', '？？'
        ]
        
        # 模拟数据煽动分析
        sim_texts = [a.get('text', a.get('content', '')) for a in sim_actions if a.get('text') or a.get('content')]
        sim_incitement = self._count_incitement(sim_texts, incitement_keywords)
        metrics['sim_incitement'] = sim_incitement
        
        if has_real:
            real_texts = [a.get('content', '') for a in real_actions if a.get('content')]
            real_incitement = self._count_incitement(real_texts, incitement_keywords)
            metrics['real_incitement'] = real_incitement
        
        # 绘图
        fig, ax = plt.subplots(figsize=(10, 6))
        labels = list(sim_incitement.get('keyword_frequency', {}).keys())[:10]
        sim_vals = [sim_incitement.get('keyword_frequency', {}).get(k, 0) for k in labels]
        
        if has_real:
            real_vals = [metrics.get('real_incitement', {}).get('keyword_frequency', {}).get(k, 0) for k in labels]
            x = np.arange(len(labels))
            width = 0.35
            ax.bar(x - width/2, sim_vals, width, label='Simulation', color=C_SIM['total'], alpha=0.8)
            ax.bar(x + width/2, real_vals, width, label='Real Data', color=C_REAL['total'], alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels([f'KW{i+1}' for i in range(len(labels))], rotation=45)
        else:
            ax.bar([f'KW{i+1}' for i in range(len(labels))], sim_vals,
                   color=C_SIM['total'], alpha=0.8)
        
        ax.set_ylabel('Frequency')
        ax.set_title(f'Incitement Keyword Frequency (Sim ratio={sim_incitement.get("incitement_ratio", 0):.2%})',
                    fontweight='bold')
        add_legend(ax)
        add_grid(ax)
        save_figure(fig, self.output_dir / 'emotion_incitement.png')
        
        return metrics
    
    def _count_incitement(self, texts, keywords):
        """计算煽动指标"""
        total = len(texts) if texts else 1
        kw_freq = Counter()
        incitement_count = 0
        
        for text in texts:
            if not text:
                continue
            found = False
            for kw in keywords:
                if kw in text:
                    kw_freq[kw] += 1
                    found = True
            if found:
                incitement_count += 1
        
        return {
            'total_texts': total,
            'incitement_count': incitement_count,
            'incitement_ratio': float(incitement_count / max(total, 1)),
            'keyword_frequency': dict(kw_freq.most_common(15))
        }
    
    def _analyze_emotion_dynamics(self, sim_times, sim_emotion, sim_sentiment, sim_actions,
                                   real_times, real_emotion, real_sentiment, real_actions, has_real):
        """情感动态指标: 排序相关、拐点对齐、负面比例时序、强度分布"""
        metrics = {}

        if not has_real:
            return metrics

        # --- 1. 情绪类别排序一致性 (Kendall's tau) ---
        sim_totals = {e: sum(v) for e, v in sim_emotion.items() if e != 'Neutral'}
        real_totals = {e: sum(v) for e, v in real_emotion.items() if e != 'Neutral'}
        all_emotions = sorted(set(sim_totals.keys()) | set(real_totals.keys()))

        if len(all_emotions) >= 3:
            sim_counts = [sim_totals.get(e, 0) for e in all_emotions]
            real_counts = [real_totals.get(e, 0) for e in all_emotions]
            from scipy.stats import rankdata
            sim_ranks = rankdata([-c for c in sim_counts])
            real_ranks = rankdata([-c for c in real_counts])
            kt = calculate_kendall_tau(sim_ranks, real_ranks)
            metrics['emotion_rank_kendall_tau'] = kt['tau']
            metrics['emotion_rank_p_value'] = kt['p_value']
            metrics['emotion_ranking'] = {
                'categories': all_emotions,
                'sim_ranks': sim_ranks.tolist(),
                'real_ranks': real_ranks.tolist()
            }
            print(f"      情绪排序Kendall's tau: {kt['tau']:.4f}")

        # --- 2. 情感极性转折点对齐度 ---
        neg_keys = ['Anger', 'Disgust', 'Anxiety', 'Sadness']

        def _neg_ratio_curve(emotion_dict, times):
            curve = []
            for i in range(len(times)):
                neg_sum = sum(emotion_dict.get(e, [0] * len(times))[i]
                              if i < len(emotion_dict.get(e, [])) else 0 for e in neg_keys)
                total = sum(emotion_dict.get(e, [0] * len(times))[i]
                            if i < len(emotion_dict.get(e, [])) else 0 for e in emotion_dict)
                curve.append(neg_sum / max(total, 1))
            return curve

        if sim_emotion and real_emotion and sim_times and real_times:
            sim_neg_curve = _neg_ratio_curve(sim_emotion, sim_times)
            real_neg_curve = _neg_ratio_curve(real_emotion, real_times)

            sim_inflections = detect_inflection_points(sim_neg_curve)
            real_inflections = detect_inflection_points(real_neg_curve)

            sim_inflection_pos = set()
            for idx in sim_inflections:
                pos = idx / max(len(sim_neg_curve) - 1, 1)
                sim_inflection_pos.add(round(pos, 2))

            real_inflection_pos = set()
            for idx in real_inflections:
                pos = idx / max(len(real_neg_curve) - 1, 1)
                real_inflection_pos.add(round(pos, 2))

            tolerance = 0.1
            matched = 0
            for sp in sim_inflection_pos:
                if any(abs(sp - rp) <= tolerance for rp in real_inflection_pos):
                    matched += 1
            total_inflections = max(len(sim_inflection_pos), 1)
            metrics['inflection_alignment'] = float(matched / total_inflections)
            metrics['sim_inflection_count'] = len(sim_inflection_pos)
            metrics['real_inflection_count'] = len(real_inflection_pos)
            print(f"      拐点对齐度: {metrics['inflection_alignment']:.4f}")

            # --- 3. 负面情绪占比时序相关 ---
            neg_sim = calculate_curve_similarity(sim_neg_curve, real_neg_curve)
            p_val = neg_sim.get('pearson', 0)
            s_val = neg_sim.get('spearman', 0)
            metrics['negative_ratio_pearson'] = 0.0 if (isinstance(p_val, float) and np.isnan(p_val)) else p_val
            metrics['negative_ratio_spearman'] = 0.0 if (isinstance(s_val, float) and np.isnan(s_val)) else s_val
            print(f"      负面比例时序Pearson: {metrics['negative_ratio_pearson']:.4f}")

        # --- 4. 情绪强度分布相似度 ---
        sim_intensities = []
        for a in sim_actions:
            ei = a.get('emotion_intensity', a.get('expression_strategy', {}).get('emotion_intensity'))
            if ei is not None:
                try:
                    sim_intensities.append(float(ei))
                except (ValueError, TypeError):
                    pass
        real_intensities = []
        for a in real_actions:
            ei = a.get('emotion_intensity')
            if ei is not None:
                try:
                    real_intensities.append(float(ei))
                except (ValueError, TypeError):
                    pass

        if sim_intensities and real_intensities:
            ks = calculate_ks_test(sim_intensities, real_intensities)
            metrics['intensity_ks_statistic'] = ks['statistic']
            metrics['intensity_ks_p_value'] = ks['p_value']

            bins = np.linspace(0, 1, 21)
            sh, _ = np.histogram(sim_intensities, bins=bins, density=True)
            rh, _ = np.histogram(real_intensities, bins=bins, density=True)
            sh = sh + 1e-10
            rh = rh + 1e-10
            sh = sh / sh.sum()
            rh = rh / rh.sum()
            jsd = calculate_jsd(sh, rh)
            metrics['intensity_jsd'] = float(jsd)
            metrics['intensity_distribution_similarity'] = float(1 - jsd)
            print(f"      强度分布相似度: {1 - jsd:.4f} (KS={ks['statistic']:.4f})")

        # ---- 绘制情感动态图 ----
        self._plot_emotion_dynamics(sim_times, sim_emotion, real_times, real_emotion,
                                    sim_intensities, real_intensities, metrics, all_emotions)
        return metrics

    def _plot_emotion_dynamics(self, sim_times, sim_emotion, real_times, real_emotion,
                               sim_intensities, real_intensities, metrics, all_emotions):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # (a) 情绪排序对比
        ax = axes[0, 0]
        ranking = metrics.get('emotion_ranking', {})
        cats = ranking.get('categories', all_emotions[:6])
        sr = ranking.get('sim_ranks', list(range(1, len(cats) + 1)))
        rr = ranking.get('real_ranks', list(range(1, len(cats) + 1)))
        x = np.arange(len(cats))
        width = 0.35
        ax.bar(x - width / 2, sr, width, label='Sim Rank', color=C_SIM['total'], alpha=0.85)
        ax.bar(x + width / 2, rr, width, label='Real Rank', color=C_REAL['total'], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(cats, rotation=30, ha='right', fontsize=FONT_SIZE['tick'])
        ax.set_ylabel('Rank (1=Most Frequent)')
        tau = metrics.get('emotion_rank_kendall_tau', 0)
        ax.set_title(f"(a) Emotion Rank Comparison (tau={tau:.3f})", fontweight='bold')
        ax.invert_yaxis()
        add_legend(ax)
        add_grid(ax)

        # (b) 拐点可视化
        ax = axes[0, 1]
        neg_keys = ['Anger', 'Disgust', 'Anxiety', 'Sadness']

        def _neg_r(emo_d, times):
            c = []
            for i in range(len(times)):
                ns = sum(emo_d.get(e, [0] * len(times))[i] if i < len(emo_d.get(e, [])) else 0 for e in neg_keys)
                tt = sum(emo_d.get(e, [0] * len(times))[i] if i < len(emo_d.get(e, [])) else 0 for e in emo_d)
                c.append(ns / max(tt, 1))
            return c

        if sim_times and sim_emotion:
            snc = _neg_r(sim_emotion, sim_times)
            ax.plot(sim_times[:len(snc)], smooth(snc, target_len=len(snc)),
                    color=C_SIM['total'], lw=LW, label='Sim Neg Ratio')
            for ip in detect_inflection_points(snc):
                if ip < len(sim_times):
                    ax.axvline(x=sim_times[ip], color=C_SIM['total'], ls=':', alpha=0.4, lw=0.8)
        if real_times and real_emotion:
            rnc = _neg_r(real_emotion, real_times)
            ax.plot(real_times[:len(rnc)], smooth(rnc, target_len=len(rnc)),
                    color=C_REAL['total'], lw=LW, ls='--', label='Real Neg Ratio')
            for ip in detect_inflection_points(rnc):
                if ip < len(real_times):
                    ax.axvline(x=real_times[ip], color=C_REAL['total'], ls=':', alpha=0.4, lw=0.8)
        ax.set_ylabel('Negative Emotion Ratio')
        ax.set_title('(b) Negative Ratio & Inflection Points', fontweight='bold')
        setup_time_axis(ax)
        add_legend(ax)
        add_grid(ax)

        # (c) 负面比例时序
        ax = axes[1, 0]
        if sim_times and sim_emotion:
            snc = _neg_r(sim_emotion, sim_times)
            ax.plot(sim_times[:len(snc)], smooth(snc, target_len=len(snc)),
                    color=C_SIM['total'], lw=LW, label='Simulation')
        if real_times and real_emotion:
            rnc = _neg_r(real_emotion, real_times)
            ax.plot(real_times[:len(rnc)], smooth(rnc, target_len=len(rnc)),
                    color=C_REAL['total'], lw=LW, ls='--', label='Real Data')
        ax.set_xlabel('Time')
        ax.set_ylabel('Negative Ratio')
        pr = metrics.get('negative_ratio_pearson', 0)
        ax.set_title(f'(c) Negative Ratio Time Series (r={pr:.3f})', fontweight='bold')
        setup_time_axis(ax)
        add_legend(ax)
        add_grid(ax)

        # (d) 情绪强度分布
        ax = axes[1, 1]
        bins = np.linspace(0, 1, 21)
        if sim_intensities:
            ax.hist(sim_intensities, bins=bins, alpha=0.6, color=C_SIM['total'],
                    label='Simulation', density=True, edgecolor='black', linewidth=0.5)
        if real_intensities:
            ax.hist(real_intensities, bins=bins, alpha=0.6, color=C_REAL['total'],
                    label='Real Data', density=True, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Emotion Intensity')
        ax.set_ylabel('Density')
        sim_val = metrics.get('intensity_distribution_similarity', 0)
        ax.set_title(f'(d) Intensity Distribution (sim={sim_val:.3f})', fontweight='bold')
        add_legend(ax)
        add_grid(ax)

        save_figure(fig, self.output_dir / 'emotion_dynamics.png')
        logger.info("[SAVED] emotion_dynamics.png")

    def _print_summary(self, results):
        """打印摘要"""
        emo = results.get('emotion_curves', {})
        if emo:
            print(f"    ✅ Top3情绪(模拟): {emo.get('top_emotions_sim', [])}")
            if emo.get('top_emotions_real'):
                print(f"       Top3情绪(真实): {emo.get('top_emotions_real', [])}")
            if 'emotion_distribution_similarity' in emo:
                print(f"       情绪分布相似度: {emo['emotion_distribution_similarity']:.4f}")
        
        sent = results.get('sentiment_curves', {})
        if sent:
            if 'sentiment_similarity' in sent:
                print(f"    ✅ 情感分布相似度: {sent['sentiment_similarity']:.4f}")
        
        score = results.get('sentiment_score', {})
        if score:
            print(f"    ✅ 模拟平均情感值: {score.get('sim_avg_score', 0.5):.3f}")
            if 'real_avg_score' in score:
                print(f"       真实平均情感值: {score['real_avg_score']:.3f}")
        
        pol = results.get('emotion_polarization', {})
        sim_pol = pol.get('sim_polarization', {})
        if sim_pol:
            print(f"    ✅ 情感极化指数(模拟): {sim_pol.get('polarization_index', 0):.3f}")
        
        inc = results.get('emotion_incitement', {})
        sim_inc = inc.get('sim_incitement', {})
        if sim_inc:
            print(f"    ✅ 情绪煽动比例(模拟): {sim_inc.get('incitement_ratio', 0):.2%}")

        ed = results.get('emotion_dynamics', {})
        if ed:
            if 'emotion_rank_kendall_tau' in ed:
                print(f"    ✅ 情绪排序Kendall's tau: {ed['emotion_rank_kendall_tau']:.4f}")
            if 'inflection_alignment' in ed:
                print(f"    ✅ 拐点对齐度: {ed['inflection_alignment']:.4f}")
            if 'negative_ratio_pearson' in ed:
                v = ed['negative_ratio_pearson']
                v_str = f"{v:.4f}" if not (isinstance(v, float) and np.isnan(v)) else "N/A (constant curve)"
                print(f"    ✅ 负面比例Pearson: {v_str}")
            if 'intensity_distribution_similarity' in ed:
                print(f"    ✅ 强度分布相似度: {ed['intensity_distribution_similarity']:.4f}")
