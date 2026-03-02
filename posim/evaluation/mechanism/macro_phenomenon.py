# -*- coding: utf-8 -*-
"""
宏观现象机制验证

验证内容:
1. 观点极化与对立现象
2. 信息长尾效应与二八定律
3. 舆情生命周期效应
4. 后真相时代效应
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict, Counter

from ..base import BaseEvaluator
from ..utils import (
    save_json, parse_time, truncate_time, smooth, normalize,
    calculate_gini_coefficient, calculate_entropy, calculate_normalized_entropy,
    classify_emotion_by_keywords, CONFLICT_KEYWORDS
)
from ..visualization import (
    FIG_SIZE, FIG_SIZE_WIDE, FIG_SIZE_TALL, LW, C_SIM, C_EMOTION, C_SENTIMENT,
    setup_time_axis, add_grid, add_legend, save_figure
)

logger = logging.getLogger(__name__)


class MacroPhenomenonEvaluator(BaseEvaluator):
    """宏观现象机制验证"""
    
    def __init__(self, output_dir: Path):
        super().__init__(output_dir / "macro_phenomenon", name="macro_phenomenon")
    
    def evaluate(self, sim_data: Dict[str, Any], real_data: Optional[Dict[str, Any]] = None,
                 **kwargs) -> Dict[str, Any]:
        """执行宏观现象验证"""
        self._log_section("宏观现象机制验证")
        
        micro_results = sim_data.get('micro_results', [])
        macro_results = sim_data.get('macro_results', {})
        aggregated = sim_data.get('aggregated', {})
        
        if not micro_results:
            print("    ⚠️ 无模拟数据，跳过宏观现象验证")
            return {}
        
        results = {}
        
        # 1. 观点极化与对立
        print("    [1/4] 分析观点极化与对立现象...")
        results['opinion_polarization'] = self._analyze_polarization(micro_results, aggregated)
        
        # 2. 信息长尾效应
        print("    [2/4] 分析信息长尾效应与二八定律...")
        results['information_longtail'] = self._analyze_longtail(micro_results)
        
        # 3. 舆情生命周期
        print("    [3/4] 分析舆情生命周期效应...")
        results['lifecycle'] = self._analyze_lifecycle(macro_results, aggregated)
        
        # 4. 后真相效应
        print("    [4/4] 分析后真相时代效应...")
        results['post_truth'] = self._analyze_post_truth(micro_results, aggregated)
        
        self._save_results(results, "macro_phenomenon_metrics.json")
        self._print_summary(results)
        return results
    
    def _analyze_polarization(self, micro_results: List[Dict], aggregated: Dict) -> Dict:
        """分析观点极化与对立现象"""
        metrics = {}
        
        # 立场分布
        stances = []
        stance_values = []
        for a in micro_results:
            stance = a.get('stance', 0)
            if isinstance(stance, (int, float)):
                stance_values.append(float(stance))
                if stance > 0.3:
                    stances.append('Support')
                elif stance < -0.3:
                    stances.append('Oppose')
                else:
                    stances.append('Neutral')
            elif isinstance(stance, str):
                stances.append(stance if stance in ['Support', 'Oppose', 'Neutral'] else 'Neutral')
        
        stance_counts = Counter(stances)
        total = len(stances) if stances else 1
        
        metrics['stance_distribution'] = {k: v for k, v in stance_counts.items()}
        metrics['stance_ratios'] = {k: round(v / total, 4) for k, v in stance_counts.items()}
        
        # 极化指数：支持和反对的比例差与总比例
        support_ratio = stance_counts.get('Support', 0) / total
        oppose_ratio = stance_counts.get('Oppose', 0) / total
        neutral_ratio = stance_counts.get('Neutral', 0) / total
        
        # Esteban-Ray极化指数的简化版本
        polarization_index = 1 - neutral_ratio  # 非中立比例越高越极化
        metrics['polarization_index'] = float(polarization_index)
        
        # 对立强度：支持和反对的占比乘积（越平衡越对立）
        metrics['confrontation_intensity'] = float(4 * support_ratio * oppose_ratio)
        
        # 如果有数值型立场，计算双峰性
        if stance_values:
            metrics['stance_mean'] = float(np.mean(stance_values))
            metrics['stance_std'] = float(np.std(stance_values))
            metrics['stance_bimodality'] = self._compute_bimodality(stance_values)
        
        # 时序极化演化
        times = aggregated.get('times', [])
        if times and micro_results:
            polar_evolution = self._compute_polarization_evolution(micro_results, times, aggregated)
            metrics['temporal_evolution'] = polar_evolution
            self._plot_polarization(metrics, times, polar_evolution)
        
        return metrics
    
    def _compute_bimodality(self, values: List[float]) -> float:
        """计算双峰性系数（Sarle's bimodality coefficient）"""
        if len(values) < 4:
            return 0.0
        from scipy.stats import kurtosis, skew
        n = len(values)
        s = skew(values)
        k = kurtosis(values)
        bc = (s ** 2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
        return float(min(max(bc, 0), 1))
    
    def _compute_polarization_evolution(self, micro_results, times, aggregated) -> List[Dict]:
        """计算极化程度的时序演化"""
        granularity = 10
        if len(times) >= 2:
            diff = (times[1] - times[0]).total_seconds() / 60
            granularity = int(diff)
        
        evolution = []
        time_actions = defaultdict(list)
        for a in micro_results:
            t = parse_time(a.get('time', ''))
            if t:
                t = truncate_time(t, granularity)
                time_actions[t].append(a)
        
        for t in times:
            actions = time_actions.get(t, [])
            if not actions:
                evolution.append({'time': t.isoformat(), 'polarization': 0.0, 'confrontation': 0.0})
                continue
            
            stances = []
            for a in actions:
                s = a.get('stance', 0)
                if isinstance(s, (int, float)):
                    stances.append(s)
            
            if stances:
                pos = sum(1 for s in stances if s > 0.3)
                neg = sum(1 for s in stances if s < -0.3)
                total = len(stances)
                pol = 1 - (total - pos - neg) / max(total, 1)
                conf = 4 * (pos / max(total, 1)) * (neg / max(total, 1))
            else:
                pol, conf = 0.0, 0.0
            
            evolution.append({
                'time': t.isoformat(),
                'polarization': float(pol),
                'confrontation': float(conf)
            })
        
        return evolution
    
    def _plot_polarization(self, metrics: Dict, times, evolution: List[Dict]):
        """绘制极化分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 立场分布饼图
        ax = axes[0, 0]
        dist = metrics.get('stance_distribution', {})
        if dist:
            labels = list(dist.keys())
            values = list(dist.values())
            colors = ['#2ca02c' if l == 'Support' else '#d62728' if l == 'Oppose' else '#7f7f7f' for l in labels]
            ax.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.set_title('Stance Distribution', fontweight='bold')
        
        # 2. 极化指数随时间变化
        ax = axes[0, 1]
        if evolution:
            ev_times = [parse_time(e['time']) for e in evolution]
            pol_vals = [e['polarization'] for e in evolution]
            conf_vals = [e['confrontation'] for e in evolution]
            ax.plot(ev_times, smooth(pol_vals), color='#d62728', lw=LW, label='Polarization')
            ax.plot(ev_times, smooth(conf_vals), color='#ff7f0e', lw=LW, label='Confrontation')
            ax.set_ylabel('Index')
            ax.set_title('Polarization & Confrontation Over Time', fontweight='bold')
            setup_time_axis(ax)
            add_legend(ax)
            add_grid(ax)
        
        # 3. 立场值分布直方图
        ax = axes[1, 0]
        stance_vals = []
        for a in metrics.get('stance_distribution', {}):
            pass
        # 使用汇总指标
        labels = ['Polarization\nIndex', 'Confrontation\nIntensity', 'Bimodality']
        values = [metrics.get('polarization_index', 0),
                  metrics.get('confrontation_intensity', 0),
                  metrics.get('stance_bimodality', 0)]
        bars = ax.bar(labels, values, color=['#d62728', '#ff7f0e', '#9467bd'], alpha=0.8, edgecolor='black')
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=11)
        ax.set_ylabel('Value')
        ax.set_title('Polarization Metrics Summary', fontweight='bold')
        ax.set_ylim(0, max(max(values) * 1.3, 0.1))
        add_grid(ax)
        
        # 4. 情绪对立热力图
        ax = axes[1, 1]
        ratios = metrics.get('stance_ratios', {})
        if ratios:
            cats = list(ratios.keys())
            vals = [ratios[c] for c in cats]
            colors = ['#2ca02c' if c == 'Support' else '#d62728' if c == 'Oppose' else '#7f7f7f' for c in cats]
            ax.barh(cats, vals, color=colors, alpha=0.8, edgecolor='black')
            ax.set_xlabel('Ratio')
            ax.set_title('Stance Ratio Breakdown', fontweight='bold')
            add_grid(ax)
        
        save_figure(fig, self.output_dir / 'opinion_polarization.png')
        logger.info("[SAVED] opinion_polarization.png")
    
    def _analyze_longtail(self, micro_results: List[Dict]) -> Dict:
        """分析信息长尾效应与二八定律"""
        metrics = {}
        
        # 用户行为计数
        user_action_counts = Counter(a.get('user_id', '') for a in micro_results)
        action_counts = sorted(user_action_counts.values(), reverse=True)
        
        if not action_counts:
            return metrics
        
        total_actions = sum(action_counts)
        total_users = len(action_counts)
        
        # 二八定律验证
        top_20_pct_users = max(1, int(total_users * 0.2))
        top_20_actions = sum(action_counts[:top_20_pct_users])
        pareto_ratio = top_20_actions / max(total_actions, 1)
        
        metrics['pareto_analysis'] = {
            'total_users': total_users,
            'total_actions': total_actions,
            'top_20pct_users_count': top_20_pct_users,
            'top_20pct_action_count': top_20_actions,
            'top_20pct_action_ratio': float(pareto_ratio),
            'pareto_index': float(pareto_ratio)  # 越接近0.8越符合二八定律
        }
        
        # 基尼系数
        metrics['gini_coefficient'] = float(calculate_gini_coefficient(action_counts))
        
        # 按用户类型分析
        user_type_actions = defaultdict(lambda: defaultdict(int))
        for a in micro_results:
            uid = a.get('user_id', '')
            atype = a.get('agent_type', 'citizen')
            user_type_actions[atype][uid] += 1
        
        type_influence = {}
        for atype, user_counts in user_type_actions.items():
            counts = list(user_counts.values())
            type_influence[atype] = {
                'user_count': len(counts),
                'total_actions': sum(counts),
                'avg_actions': float(np.mean(counts)),
                'max_actions': int(max(counts)),
                'gini': float(calculate_gini_coefficient(counts))
            }
        metrics['type_influence'] = type_influence
        
        # 绘图
        self._plot_longtail(action_counts, metrics, user_type_actions)
        
        return metrics
    
    def _plot_longtail(self, action_counts, metrics, user_type_actions):
        """绘制长尾效应图"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 用户行为量排序图（长尾分布）
        ax = axes[0, 0]
        ax.bar(range(len(action_counts)), action_counts, color=C_SIM['total'], alpha=0.7)
        ax.set_xlabel('User Rank')
        ax.set_ylabel('Action Count')
        ax.set_title('User Activity Distribution (Long-tail)', fontweight='bold')
        # 标注20%线
        cutoff = max(1, int(len(action_counts) * 0.2))
        ax.axvline(x=cutoff, color='red', linestyle='--', lw=1.5, label=f'Top 20% ({cutoff} users)')
        add_legend(ax)
        add_grid(ax)
        
        # 2. 洛伦兹曲线
        ax = axes[0, 1]
        sorted_counts = sorted(action_counts)
        cumulative = np.cumsum(sorted_counts) / sum(sorted_counts)
        x = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
        ax.plot(x, cumulative, color=C_SIM['total'], lw=LW, label='Lorenz Curve')
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect Equality')
        ax.fill_between(x, cumulative, x, alpha=0.15, color=C_SIM['total'])
        gini = metrics.get('gini_coefficient', 0)
        ax.set_xlabel('Cumulative Share of Users')
        ax.set_ylabel('Cumulative Share of Actions')
        ax.set_title(f'Lorenz Curve (Gini={gini:.3f})', fontweight='bold')
        add_legend(ax)
        add_grid(ax)
        
        # 3. 按用户类型的行为贡献
        ax = axes[1, 0]
        type_inf = metrics.get('type_influence', {})
        if type_inf:
            types = list(type_inf.keys())
            totals = [type_inf[t]['total_actions'] for t in types]
            users = [type_inf[t]['user_count'] for t in types]
            x_pos = np.arange(len(types))
            width = 0.35
            ax.bar(x_pos - width/2, totals, width, label='Actions', color=C_SIM['total'], alpha=0.8)
            ax.bar(x_pos + width/2, users, width, label='Users', color=C_SIM['comment'], alpha=0.8)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(types)
            ax.set_ylabel('Count')
            ax.set_title('Actions & Users by Agent Type', fontweight='bold')
            add_legend(ax)
            add_grid(ax)
        
        # 4. 帕累托图
        ax = axes[1, 1]
        pareto = metrics.get('pareto_analysis', {})
        if pareto:
            labels = ['Top 20%\nUsers', 'Bottom 80%\nUsers']
            action_shares = [pareto.get('top_20pct_action_ratio', 0),
                            1 - pareto.get('top_20pct_action_ratio', 0)]
            user_shares = [0.2, 0.8]
            x_pos = np.arange(len(labels))
            width = 0.35
            ax.bar(x_pos - width/2, action_shares, width, label='Action Share', 
                   color='#d62728', alpha=0.8)
            ax.bar(x_pos + width/2, user_shares, width, label='User Share', 
                   color='#1f77b4', alpha=0.8)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels)
            ax.set_ylabel('Share')
            ax.set_title(f'Pareto Analysis (Top 20% → {pareto.get("top_20pct_action_ratio", 0):.1%} actions)',
                        fontweight='bold')
            ax.set_ylim(0, 1.1)
            add_legend(ax)
            add_grid(ax)
        
        save_figure(fig, self.output_dir / 'information_longtail.png')
        logger.info("[SAVED] information_longtail.png")
    
    def _analyze_lifecycle(self, macro_results: Dict, aggregated: Dict) -> Dict:
        """分析舆情生命周期效应"""
        metrics = {}
        stats = macro_results.get('stats', {})
        
        intensity = stats.get('intensity_history', [])
        actions_per_step = stats.get('actions_per_step', [])
        active_agents = stats.get('active_agents_per_step', [])
        
        times = aggregated.get('times', [])
        hotness = aggregated.get('hotness', {}).get('total', [])
        
        if not hotness and not intensity:
            return metrics
        
        # 生命周期阶段划分
        data = hotness if hotness else actions_per_step
        if data:
            peak_idx = int(np.argmax(data))
            total_len = len(data)
            
            # 阶段划分：潜伏期 -> 爆发期 -> 高峰期 -> 衰退期
            metrics['lifecycle_phases'] = {
                'total_steps': total_len,
                'peak_step': peak_idx,
                'peak_value': float(max(data)),
                'peak_position_ratio': float(peak_idx / max(total_len - 1, 1)),
                'growth_phase': list(range(0, peak_idx)),
                'peak_phase': [peak_idx],
                'decay_phase': list(range(peak_idx + 1, total_len))
            }
            
            # 增长率和衰减率
            if peak_idx > 0:
                growth_data = data[:peak_idx + 1]
                growth_rate = (data[peak_idx] - data[0]) / max(peak_idx, 1)
                metrics['growth_rate'] = float(growth_rate)
            
            if peak_idx < total_len - 1:
                decay_data = data[peak_idx:]
                decay_rate = (data[peak_idx] - data[-1]) / max(total_len - peak_idx - 1, 1)
                metrics['decay_rate'] = float(decay_rate)
            
            # 半衰期（从峰值到降至一半的步数）
            half_value = data[peak_idx] / 2
            half_life = None
            for i in range(peak_idx + 1, total_len):
                if data[i] <= half_value:
                    half_life = i - peak_idx
                    break
            metrics['half_life_steps'] = half_life
        
        # 绘图
        self._plot_lifecycle(times, hotness, intensity, actions_per_step, active_agents, metrics)
        
        return metrics
    
    def _plot_lifecycle(self, times, hotness, intensity, actions_per_step, active_agents, metrics):
        """绘制生命周期图"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 热度生命周期
        ax = axes[0, 0]
        if hotness and times:
            n = min(len(times), len(hotness))
            ax.fill_between(times[:n], smooth(hotness[:n], target_len=n), alpha=0.3, color=C_SIM['total'])
            ax.plot(times[:n], smooth(hotness[:n], target_len=n), color=C_SIM['total'], lw=LW)
            peak_idx = int(np.argmax(hotness))
            if peak_idx < n:
                ax.axvline(x=times[min(peak_idx, n-1)], color='red', linestyle='--', lw=1, alpha=0.7, label='Peak')
            setup_time_axis(ax)
            add_legend(ax)
        ax.set_ylabel('Activity Count')
        ax.set_title('Opinion Lifecycle - Hotness', fontweight='bold')
        add_grid(ax)
        
        # 2. Hawkes强度
        ax = axes[0, 1]
        if intensity:
            ax.plot(intensity, color='#ff7f0e', lw=LW)
            peak = int(np.argmax(intensity))
            ax.axvline(x=peak, color='red', linestyle='--', lw=1, alpha=0.7, label=f'Peak at step {peak}')
            add_legend(ax)
        ax.set_xlabel('Step')
        ax.set_ylabel('Intensity')
        ax.set_title('Hawkes Intensity Lifecycle', fontweight='bold')
        add_grid(ax)
        
        # 3. 每步行为数
        ax = axes[1, 0]
        if actions_per_step:
            ax.bar(range(len(actions_per_step)), actions_per_step, color=C_SIM['comment'], alpha=0.7)
        ax.set_xlabel('Step')
        ax.set_ylabel('Actions')
        ax.set_title('Actions per Step', fontweight='bold')
        add_grid(ax)
        
        # 4. 累积行为与活跃智能体
        ax = axes[1, 1]
        if actions_per_step:
            cumulative = np.cumsum(actions_per_step)
            ax.plot(cumulative, color=C_SIM['total'], lw=LW, label='Cumulative Actions')
            ax2 = ax.twinx()
            if active_agents:
                ax2.plot(active_agents, color='#ff7f0e', lw=LW, ls='--', label='Active Agents')
                ax2.set_ylabel('Active Agents', color='#ff7f0e')
            ax.set_xlabel('Step')
            ax.set_ylabel('Cumulative Actions')
            ax.set_title('Cumulative Actions & Active Agents', fontweight='bold')
            lines1, labels1 = ax.get_legend_handles_labels()
            if active_agents:
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            else:
                ax.legend(loc='upper left')
        add_grid(ax)
        
        save_figure(fig, self.output_dir / 'lifecycle.png')
        logger.info("[SAVED] lifecycle.png")
    
    def _analyze_post_truth(self, micro_results: List[Dict], aggregated: Dict) -> Dict:
        """分析后真相时代效应"""
        metrics = {}
        
        texts = [a.get('text', a.get('content', '')) for a in micro_results if a.get('text') or a.get('content')]
        if not texts:
            return metrics
        
        # 情感vs事实比例分析
        emotional_count = 0
        factual_count = 0
        emotional_keywords = ['气死', '太过分', '不能忍', '崩溃', '震惊', '可恶', '愤怒', 
                             '伤心', '难过', '恶心', '心痛', '😡', '😢', '💢', '生气', '无语']
        factual_keywords = ['根据', '数据', '事实上', '调查', '证据', '报道称', '据悉',
                           '信息显示', '公开资料', '官方', '通报', '声明']
        
        for text in texts:
            if not text:
                continue
            has_emotional = any(kw in text for kw in emotional_keywords)
            has_factual = any(kw in text for kw in factual_keywords)
            if has_emotional:
                emotional_count += 1
            if has_factual:
                factual_count += 1
        
        total = len(texts)
        metrics['emotional_ratio'] = float(emotional_count / max(total, 1))
        metrics['factual_ratio'] = float(factual_count / max(total, 1))
        metrics['emotion_over_fact_ratio'] = float(
            emotional_count / max(factual_count, 1)
        )
        
        # 后真相指数: 情感驱动比例越高越"后真相"
        post_truth_index = metrics['emotional_ratio'] / max(metrics['emotional_ratio'] + metrics['factual_ratio'], 0.01)
        metrics['post_truth_index'] = float(post_truth_index)
        
        # 情绪强度分布
        intensities = []
        for a in micro_results:
            intensity = a.get('emotion_intensity', 0.5)
            if isinstance(intensity, (int, float)):
                intensities.append(float(intensity))
        
        if intensities:
            metrics['avg_emotion_intensity'] = float(np.mean(intensities))
            metrics['high_intensity_ratio'] = float(sum(1 for i in intensities if i > 0.7) / len(intensities))
            metrics['intensity_distribution'] = {
                'low_0_0.3': float(sum(1 for i in intensities if i <= 0.3) / len(intensities)),
                'medium_0.3_0.7': float(sum(1 for i in intensities if 0.3 < i <= 0.7) / len(intensities)),
                'high_0.7_1.0': float(sum(1 for i in intensities if i > 0.7) / len(intensities))
            }
        
        # 立场极端化程度
        stance_vals = []
        for a in micro_results:
            s = a.get('stance', 0)
            if isinstance(s, (int, float)):
                stance_vals.append(abs(float(s)))
        
        if stance_vals:
            metrics['avg_stance_extremity'] = float(np.mean(stance_vals))
            metrics['extreme_stance_ratio'] = float(sum(1 for s in stance_vals if s > 0.7) / len(stance_vals))
        
        # 绘图
        self._plot_post_truth(metrics, aggregated)
        
        return metrics
    
    def _plot_post_truth(self, metrics: Dict, aggregated: Dict):
        """绘制后真相效应分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 情感vs事实比例
        ax = axes[0, 0]
        labels = ['Emotional', 'Factual', 'Other']
        emo_r = metrics.get('emotional_ratio', 0)
        fact_r = metrics.get('factual_ratio', 0)
        other_r = max(0, 1 - emo_r - fact_r)
        values = [emo_r, fact_r, other_r]
        colors = ['#d62728', '#1f77b4', '#7f7f7f']
        ax.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        pti = metrics.get('post_truth_index', 0)
        ax.set_title(f'Emotional vs Factual Content\nPost-Truth Index: {pti:.2f}', fontweight='bold')
        
        # 2. 情绪强度分布
        ax = axes[0, 1]
        intensity_dist = metrics.get('intensity_distribution', {})
        if intensity_dist:
            cats = ['Low\n(0-0.3)', 'Medium\n(0.3-0.7)', 'High\n(0.7-1.0)']
            vals = [intensity_dist.get('low_0_0.3', 0), intensity_dist.get('medium_0.3_0.7', 0),
                    intensity_dist.get('high_0.7_1.0', 0)]
            colors = ['#2ca02c', '#ff7f0e', '#d62728']
            bars = ax.bar(cats, vals, color=colors, alpha=0.8, edgecolor='black')
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:.1%}', ha='center', va='bottom', fontsize=11)
            ax.set_ylabel('Ratio')
            ax.set_title('Emotion Intensity Distribution', fontweight='bold')
            add_grid(ax)
        
        # 3. 后真相综合指标
        ax = axes[1, 0]
        indicator_labels = ['Post-Truth\nIndex', 'Avg Emotion\nIntensity', 'High Intensity\nRatio',
                           'Extreme Stance\nRatio']
        indicator_values = [
            metrics.get('post_truth_index', 0),
            metrics.get('avg_emotion_intensity', 0),
            metrics.get('high_intensity_ratio', 0),
            metrics.get('extreme_stance_ratio', 0)
        ]
        colors = ['#d62728', '#ff7f0e', '#e377c2', '#9467bd']
        bars = ax.bar(indicator_labels, indicator_values, color=colors, alpha=0.8, edgecolor='black')
        for bar, val in zip(bars, indicator_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        ax.set_ylabel('Value')
        ax.set_title('Post-Truth Era Indicators', fontweight='bold')
        ax.set_ylim(0, max(max(indicator_values) * 1.3, 0.1))
        add_grid(ax)
        
        # 4. 情感驱动vs理性讨论比率
        ax = axes[1, 1]
        ratio = metrics.get('emotion_over_fact_ratio', 0)
        ax.barh(['Emotion/Fact\nRatio'], [ratio], color='#d62728', alpha=0.8, edgecolor='black', height=0.4)
        ax.axvline(x=1.0, color='black', linestyle='--', lw=1, label='Balance Line')
        ax.set_xlabel('Ratio')
        ax.set_title(f'Emotion-over-Fact Ratio: {ratio:.2f}', fontweight='bold')
        add_legend(ax)
        add_grid(ax)
        
        save_figure(fig, self.output_dir / 'post_truth.png')
        logger.info("[SAVED] post_truth.png")
    
    def _print_summary(self, results: Dict):
        """打印摘要"""
        polar = results.get('opinion_polarization', {})
        if polar:
            print(f"    ✅ 极化指数: {polar.get('polarization_index', 0):.3f}")
            print(f"       对立强度: {polar.get('confrontation_intensity', 0):.3f}")
            dist = polar.get('stance_distribution', {})
            if dist:
                print(f"       立场分布: {dist}")
        
        lt = results.get('information_longtail', {})
        if lt:
            pareto = lt.get('pareto_analysis', {})
            if pareto:
                print(f"    ✅ 二八定律: Top 20%用户贡献 {pareto.get('top_20pct_action_ratio', 0):.1%} 行为")
            print(f"       基尼系数: {lt.get('gini_coefficient', 0):.3f}")
        
        lc = results.get('lifecycle', {})
        phases = lc.get('lifecycle_phases', {})
        if phases:
            print(f"    ✅ 生命周期: 峰值在第 {phases.get('peak_step', 0)} 步, "
                  f"峰值位置={phases.get('peak_position_ratio', 0):.1%}")
            if lc.get('half_life_steps'):
                print(f"       半衰期: {lc.get('half_life_steps')} 步")
        
        pt = results.get('post_truth', {})
        if pt:
            print(f"    ✅ 后真相指数: {pt.get('post_truth_index', 0):.3f}")
            print(f"       情感/事实比: {pt.get('emotion_over_fact_ratio', 0):.2f}")