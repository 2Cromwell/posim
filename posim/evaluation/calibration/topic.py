# -*- coding: utf-8 -*-
"""
话题演化校准

评估模拟与真实的热点话题(#标签)的相似性:
- 话题分布对比
- 话题曲线对比
- 话题重叠度
- 话题演化时序相似性
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional, List
from collections import Counter

from ..base import BaseEvaluator
from ..utils import (
    smooth, normalize, save_json, calculate_curve_similarity,
    calculate_jsd, extract_topics
)
from ..visualization import (
    FIG_SIZE, FIG_SIZE_TALL, FIG_SIZE_WIDE, LW, ALPHA,
    C_SIM, C_REAL, setup_time_axis, add_grid, add_legend, save_figure
)

logger = logging.getLogger(__name__)

# 话题颜色
TOPIC_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


class TopicCalibrationEvaluator(BaseEvaluator):
    """话题演化校准"""
    
    def __init__(self, output_dir: Path):
        super().__init__(output_dir / "topic_calibration", name="topic_calibration")
    
    def evaluate(self, sim_data: Dict[str, Any], real_data: Optional[Dict[str, Any]] = None,
                 **kwargs) -> Dict[str, Any]:
        """执行话题演化校准"""
        self._log_section("话题演化校准")
        
        sim_agg = sim_data.get('aggregated', {})
        sim_times = sim_agg.get('times', [])
        sim_topics = sim_agg.get('topics', {})
        
        if not sim_topics:
            print("    ⚠️ 无模拟话题数据，跳过")
            return {}
        
        results = {}
        has_real = real_data is not None and real_data.get('topics')
        real_times = real_data.get('times', []) if has_real else []
        real_topics = real_data.get('topics', {}) if has_real else {}
        
        # 1. 话题总量统计
        sim_totals = {t: sum(d['count'] for d in data) for t, data in sim_topics.items()}
        real_totals = {t: sum(d['count'] for d in data) for t, data in real_topics.items()} if has_real else {}
        
        results['sim_topic_count'] = len(sim_totals)
        results['sim_top_topics'] = dict(Counter(sim_totals).most_common(10))
        
        if has_real:
            results['real_topic_count'] = len(real_totals)
            results['real_top_topics'] = dict(Counter(real_totals).most_common(10))
        
        # 2. 话题重叠度
        if has_real:
            sim_set = set(sim_totals.keys())
            real_set = set(real_totals.keys())
            overlap = sim_set & real_set
            union = sim_set | real_set
            
            results['topic_overlap'] = {
                'sim_unique': len(sim_set),
                'real_unique': len(real_set),
                'common_topics': list(overlap),
                'common_count': len(overlap),
                'jaccard_similarity': float(len(overlap) / max(len(union), 1)),
                'overlap_ratio_sim': float(len(overlap) / max(len(sim_set), 1)),
                'overlap_ratio_real': float(len(overlap) / max(len(real_set), 1))
            }
            print(f"      话题Jaccard相似度: {results['topic_overlap']['jaccard_similarity']:.4f}")
            print(f"      共同话题: {len(overlap)}/{len(union)}")
        
        # 3. 话题分布相似度
        if has_real:
            all_topics = list(set(sim_totals.keys()) | set(real_totals.keys()))
            sim_vec = np.array([sim_totals.get(t, 0) for t in all_topics], dtype=float)
            real_vec = np.array([real_totals.get(t, 0) for t in all_topics], dtype=float)
            
            if sim_vec.sum() > 0: sim_vec = sim_vec / sim_vec.sum()
            if real_vec.sum() > 0: real_vec = real_vec / real_vec.sum()
            
            jsd = calculate_jsd(sim_vec, real_vec)
            results['topic_distribution_jsd'] = float(jsd)
            results['topic_distribution_similarity'] = float(1 - jsd)
            print(f"      话题分布JSD: {jsd:.4f}, 相似度: {1-jsd:.4f}")
        
        # 4. Top话题曲线对比
        top5_sim = sorted(sim_totals.keys(), key=lambda x: sim_totals[x], reverse=True)[:5]
        if has_real:
            top5_real = sorted(real_totals.keys(), key=lambda x: real_totals[x], reverse=True)[:5]
            all_top = list(dict.fromkeys(top5_sim + top5_real))[:8]
        else:
            all_top = top5_sim
        
        # 计算共同话题的曲线相似度
        if has_real:
            results['per_topic_curve_similarity'] = {}
            common = set(sim_totals.keys()) & set(real_totals.keys())
            for topic in list(common)[:10]:
                sim_curve = [d['count'] for d in sim_topics[topic]]
                real_curve = [d['count'] for d in real_topics[topic]]
                sim_norm = normalize(sim_curve).tolist()
                real_norm = normalize(real_curve).tolist()
                results['per_topic_curve_similarity'][topic] = calculate_curve_similarity(sim_norm, real_norm)
        
        # 5. 绘图
        self._plot_topics(sim_times, sim_topics, sim_totals,
                         real_times, real_topics, real_totals,
                         all_top, has_real, results)
        
        self._save_results(results, "topic_calibration_metrics.json")
        self._print_summary(results)
        return results
    
    def _plot_topics(self, sim_times, sim_topics, sim_totals,
                    real_times, real_topics, real_totals, all_top, has_real, results):
        """绘制话题相关图"""
        # 话题ID映射（避免中文显示问题）
        topic_labels = {t: f'Topic{i+1}' for i, t in enumerate(all_top)}
        
        # 保存映射
        save_json(self.output_dir / 'topic_label_mapping.json', {
            'mapping': {v: k for k, v in topic_labels.items()},
            'reverse': topic_labels
        })
        
        # === 1. 话题演化时序图 ===
        n_topics = len(all_top)
        if n_topics > 0:
            fig, axes = plt.subplots(min(n_topics, 5), 1, figsize=(14, 3 * min(n_topics, 5)))
            if n_topics == 1:
                axes = [axes]
            
            for i, topic in enumerate(all_top[:5]):
                ax = axes[i]
                c = TOPIC_COLORS[i % len(TOPIC_COLORS)]
                
                if topic in sim_topics:
                    sim_curve = [d['count'] for d in sim_topics[topic]]
                    ax.plot(sim_times[:len(sim_curve)],
                           smooth(sim_curve, target_len=len(sim_curve)),
                           color=c, lw=LW, label='Simulation')
                
                if has_real and topic in real_topics:
                    real_curve = [d['count'] for d in real_topics[topic]]
                    ax.plot(real_times[:len(real_curve)],
                           smooth(real_curve, target_len=len(real_curve)),
                           color=c, lw=LW, ls='--', alpha=0.7, label='Real Data')
                
                ax.set_title(f'{topic_labels.get(topic, topic)}', fontweight='bold')
                ax.set_ylabel('Mentions')
                add_legend(ax, loc='upper right')
                add_grid(ax)
                setup_time_axis(ax)
            
            axes[-1].set_xlabel('Time')
            save_figure(fig, self.output_dir / 'topic_evolution.png')
        
        # === 2. 话题分布对比柱状图 ===
        fig, ax = plt.subplots(figsize=FIG_SIZE_WIDE)
        x = np.arange(len(all_top))
        width = 0.35 if has_real else 0.7
        
        sim_vals = [sim_totals.get(t, 0) for t in all_top]
        labels_x = [topic_labels.get(t, t) for t in all_top]
        
        if has_real:
            real_vals = [real_totals.get(t, 0) for t in all_top]
            ax.bar(x - width/2, sim_vals, width, label='Simulation', color=C_SIM['total'], alpha=0.8)
            ax.bar(x + width/2, real_vals, width, label='Real Data', color=C_REAL['total'], alpha=0.8)
        else:
            ax.bar(x, sim_vals, width, label='Simulation', color=C_SIM['total'], alpha=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels_x, rotation=45, ha='right')
        ax.set_ylabel('Total Mentions')
        ax.set_title('Topic Distribution Comparison', fontweight='bold')
        add_legend(ax)
        add_grid(ax)
        save_figure(fig, self.output_dir / 'topic_distribution.png')
        
        # === 3. 话题热力图 ===
        if len(all_top) >= 2 and sim_times:
            fig, ax = plt.subplots(figsize=(14, 6))
            matrix = []
            for topic in all_top[:8]:
                if topic in sim_topics:
                    curve = [d['count'] for d in sim_topics[topic]]
                    norm = normalize(curve)
                    matrix.append(norm[:len(sim_times)])
                else:
                    matrix.append(np.zeros(len(sim_times)))
            
            if matrix:
                sample_step = max(1, len(sim_times) // 20)
                sampled = np.array([row[::sample_step] for row in matrix])
                sampled_labels = [sim_times[i].strftime('%m-%d %H:%M')
                                 for i in range(0, len(sim_times), sample_step)]
                
                im = ax.imshow(sampled, aspect='auto', cmap='YlOrRd')
                ax.set_yticks(range(len(all_top[:8])))
                ax.set_yticklabels([topic_labels.get(t, t) for t in all_top[:8]])
                n_labels = min(10, len(sampled_labels))
                step = max(1, len(sampled_labels) // n_labels)
                ax.set_xticks(range(0, len(sampled_labels), step))
                ax.set_xticklabels([sampled_labels[i] for i in range(0, len(sampled_labels), step)],
                                  rotation=45, ha='right')
                ax.set_xlabel('Time')
                ax.set_ylabel('Topic')
                ax.set_title('Topic Heatmap (Normalized)', fontweight='bold')
                plt.colorbar(im, ax=ax, label='Normalized Mentions')
                save_figure(fig, self.output_dir / 'topic_heatmap.png')
        
        logger.info("[SAVED] 话题校准图")
    
    def _print_summary(self, results):
        """打印摘要"""
        print(f"    ✅ 模拟话题数: {results.get('sim_topic_count', 0)}")
        top = results.get('sim_top_topics', {})
        if top:
            top3 = list(top.items())[:3]
            print(f"       Top3: {', '.join(f'{k}({v})' for k, v in top3)}")
        
        overlap = results.get('topic_overlap', {})
        if overlap:
            print(f"    ✅ 话题重叠: Jaccard={overlap.get('jaccard_similarity', 0):.4f}, "
                  f"共同={overlap.get('common_count', 0)}")
        
        if 'topic_distribution_similarity' in results:
            print(f"    ✅ 话题分布相似度: {results['topic_distribution_similarity']:.4f}")
