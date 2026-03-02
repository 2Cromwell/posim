# -*- coding: utf-8 -*-
"""
增强版舆情生命周期评估器

分析公共舆论事件的生命周期阶段：
  潜伏期 → 增长期 → 高峰期 → 衰退期 → 沉寂期

主要功能:
  - 生命周期阶段检测（阈值法 + 变点检测）
  - 增长/衰减率拟合与 R² 评估
  - 峰值分析（主峰、次峰、显著度、宽度）
  - 半衰期 / 四分之一衰期
  - 活动类型组成演化与香农熵
  - 与真实数据对比（可选）
"""
import logging
import warnings
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from scipy import stats as sp_stats
from scipy.signal import find_peaks

from ..base import BaseEvaluator
from ..utils import (
    save_json, parse_time, truncate_time, smooth, normalize,
    calculate_curve_similarity, calculate_gini_coefficient,
    detect_inflection_points,
)
from ..visualization import (
    FIG_SIZE, FIG_SIZE_WIDE, FIG_SIZE_SQUARE, LW, LW_MINOR, ALPHA,
    MARKER_SIZE, FONT_SIZE,
    C_SIM, C_REAL,
    setup_time_axis, add_grid, add_legend, save_figure,
)

logger = logging.getLogger(__name__)

# Phase colours used across plots
_PHASE_COLORS = {
    'incubation': '#e8e8e8',
    'growth':     '#c6e2c6',
    'peak':       '#ffd6d6',
    'decay':      '#d6d6ff',
    'dormancy':   '#f0f0f0',
}
_PHASE_LABELS = {
    'incubation': 'Incubation',
    'growth':     'Growth',
    'peak':       'Peak',
    'decay':      'Decay',
    'dormancy':   'Dormancy',
}

# Activity‐type colour palette
_COMP_COLORS = {'original': '#ff7f0e', 'repost': '#c49c94', 'comment': '#2ca02c'}


class LifecycleEvaluator(BaseEvaluator):
    """增强版舆情生命周期评估器"""

    def __init__(self, output_dir: Path):
        super().__init__(output_dir / "lifecycle", name="lifecycle")

    # ------------------------------------------------------------------
    # public entry
    # ------------------------------------------------------------------
    def evaluate(
        self,
        sim_data: Dict[str, Any],
        real_data: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        self._log_section("舆情生命周期评估")

        macro_results = sim_data.get('macro_results', {})
        aggregated = sim_data.get('aggregated', {})
        stats = macro_results.get('stats', {})

        intensity_history = stats.get('intensity_history', [])
        actions_per_step = stats.get('actions_per_step', [])
        active_agents = stats.get('active_agents_per_step', [])

        sim_times = aggregated.get('times', [])
        sim_hotness = aggregated.get('hotness', {})
        sim_total = sim_hotness.get('total', [])
        sim_original = sim_hotness.get('original', [])
        sim_repost = sim_hotness.get('repost', [])
        sim_comment = sim_hotness.get('comment', [])

        if not sim_total and not actions_per_step:
            print("    ⚠️ 无活动数据，跳过生命周期评估")
            return {}

        activity = sim_total if sim_total else actions_per_step
        n = len(activity)

        results: Dict[str, Any] = {}

        # 1. Phase detection
        print("    [1/6] 检测生命周期阶段...")
        phases_info = self._detect_phases(activity)
        results['lifecycle_phases'] = phases_info

        # 2. Growth / decay metrics
        print("    [2/6] 计算增长与衰减指标...")
        growth = self._growth_decay_analysis(activity, phases_info)
        results['growth_metrics'] = growth

        # 3. Peak analysis
        print("    [3/6] 峰值分析...")
        peaks = self._peak_analysis(activity)
        results['peak_analysis'] = peaks

        # 4. Half‑life
        print("    [4/6] 半衰期分析...")
        half = self._halflife_analysis(activity, phases_info)
        results['growth_metrics'].update(half)

        # 5. Composition evolution
        print("    [5/6] 活动类型组成演化...")
        comp = self._composition_analysis(sim_original, sim_repost, sim_comment)
        results['activity_composition'] = comp

        # 6. Extra aggregate metrics
        print("    [6/6] 附加统计指标...")
        extras = self._extra_metrics(activity)
        results['growth_metrics'].update(extras)

        # --- comparison with real data ---
        real_total: List[float] = []
        real_times: list = []
        if real_data:
            real_times = real_data.get('times', [])
            real_hot = real_data.get('hotness', {})
            real_total = real_hot.get('total', [])
        has_real = bool(real_total)

        if has_real:
            print("    [+] 与真实数据对比...")
            results['comparison'] = self._compare_with_real(
                activity, sim_times, real_total, real_times, phases_info,
            )

        # ---- Plots (each SEPARATE figure) ---
        print("    [绘图] 生成可视化...")
        self._plot_lifecycle_hotness(sim_times, activity, phases_info, sim_total)
        self._plot_growth_rate(activity, sim_times)
        self._plot_actions_per_step(actions_per_step)
        self._plot_cumulative_activity(actions_per_step, active_agents)
        self._plot_hawkes_intensity(intensity_history)
        self._plot_activity_composition(sim_times, sim_original, sim_repost, sim_comment)
        self._plot_metrics_summary(results)

        if has_real:
            self._plot_lifecycle_comparison(
                sim_times, activity, real_times, real_total, phases_info,
            )
            self._plot_phase_duration_comparison(activity, real_total)

        self._save_results(results, "lifecycle_metrics.json")
        self._print_summary(results)
        return results

    # ==================================================================
    # Analysis helpers
    # ==================================================================

    def _detect_phases(self, activity: List[float]) -> Dict[str, Any]:
        """Threshold‑based five‑phase detection."""
        arr = np.array(activity, dtype=float)
        n = len(arr)
        peak_idx = int(np.argmax(arr))
        peak_val = float(arr[peak_idx])
        threshold = peak_val * 0.10

        # incubation: start → first crossing above threshold
        first_above = 0
        for i in range(n):
            if arr[i] >= threshold:
                first_above = i
                break

        # dormancy: last crossing above threshold → end
        last_above = n - 1
        for i in range(n - 1, -1, -1):
            if arr[i] >= threshold:
                last_above = i
                break

        # peak band: contiguous region >= 80 % of peak
        peak_thresh = peak_val * 0.80
        peak_start = peak_idx
        peak_end = peak_idx
        for i in range(peak_idx, -1, -1):
            if arr[i] >= peak_thresh:
                peak_start = i
            else:
                break
        for i in range(peak_idx, n):
            if arr[i] >= peak_thresh:
                peak_end = i
            else:
                break

        phases = {
            'incubation': {'start': 0, 'end': max(first_above - 1, 0),
                           'duration': max(first_above, 1)},
            'growth':     {'start': first_above, 'end': max(peak_start - 1, first_above),
                           'duration': max(peak_start - first_above, 1)},
            'peak':       {'start': peak_start, 'end': peak_end,
                           'duration': peak_end - peak_start + 1},
            'decay':      {'start': min(peak_end + 1, n - 1), 'end': last_above,
                           'duration': max(last_above - peak_end, 1)},
            'dormancy':   {'start': min(last_above + 1, n - 1), 'end': n - 1,
                           'duration': max(n - 1 - last_above, 1)},
        }

        total_dur = sum(p['duration'] for p in phases.values())
        phase_ratios = [phases[k]['duration'] / max(total_dur, 1)
                        for k in ['incubation', 'growth', 'peak', 'decay', 'dormancy']]

        # derivative‑based change‑point detection
        inflection_pts = detect_inflection_points(list(arr), smooth_w=max(3, n // 20))

        return {
            'total_steps': n,
            'peak_step': peak_idx,
            'peak_value': peak_val,
            'peak_position_ratio': float(peak_idx / max(n - 1, 1)),
            'phases': phases,
            'phase_ratios': [float(r) for r in phase_ratios],
            'inflection_points': inflection_pts,
        }

    def _growth_decay_analysis(self, activity: List[float], phases: Dict) -> Dict[str, Any]:
        arr = np.array(activity, dtype=float)
        peak_idx = phases['peak_step']

        growth_rate = 0.0
        growth_r2 = 0.0
        decay_rate = 0.0
        decay_r2 = 0.0

        # Exponential fit on growth segment
        if peak_idx > 2:
            seg = arr[:peak_idx + 1]
            seg_pos = np.clip(seg, 1e-9, None)
            x = np.arange(len(seg_pos))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    slope, intercept, r, _, _ = sp_stats.linregress(x, np.log(seg_pos))
                    growth_rate = float(slope)
                    growth_r2 = float(r ** 2)
                except Exception:
                    pass

        # Exponential fit on decay segment
        if peak_idx < len(arr) - 3:
            seg = arr[peak_idx:]
            seg_pos = np.clip(seg, 1e-9, None)
            x = np.arange(len(seg_pos))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    slope, intercept, r, _, _ = sp_stats.linregress(x, np.log(seg_pos))
                    decay_rate = float(slope)
                    decay_r2 = float(r ** 2)
                except Exception:
                    pass

        return {
            'growth_rate': growth_rate,
            'growth_r2': growth_r2,
            'decay_rate': decay_rate,
            'decay_r2': decay_r2,
        }

    def _peak_analysis(self, activity: List[float]) -> Dict[str, Any]:
        arr = np.array(activity, dtype=float)
        primary = int(np.argmax(arr))
        primary_val = float(arr[primary])

        smoothed = smooth(list(arr), w=max(3, len(arr) // 15))
        prominence_min = primary_val * 0.15
        try:
            peaks, props = find_peaks(
                smoothed, prominence=prominence_min, distance=max(3, len(arr) // 20),
            )
        except Exception:
            peaks, props = np.array([]), {}

        secondary = [int(p) for p in peaks if p != primary]
        prominences = props.get('prominences', np.array([]))
        widths_result = np.array([])
        if len(peaks):
            try:
                from scipy.signal import peak_widths
                widths_result = peak_widths(smoothed, peaks, rel_height=0.5)[0]
            except Exception:
                pass

        primary_prom = 0.0
        primary_width = 0.0
        if primary in peaks.tolist():
            idx_in_peaks = peaks.tolist().index(primary)
            if idx_in_peaks < len(prominences):
                primary_prom = float(prominences[idx_in_peaks])
            if idx_in_peaks < len(widths_result):
                primary_width = float(widths_result[idx_in_peaks])

        return {
            'primary_peak_idx': primary,
            'secondary_peaks': secondary[:5],
            'peak_prominence': primary_prom,
            'peak_width': primary_width,
        }

    def _halflife_analysis(self, activity: List[float], phases: Dict) -> Dict[str, Any]:
        arr = np.array(activity, dtype=float)
        peak_idx = phases['peak_step']
        peak_val = phases['peak_value']

        half_life = None
        quarter_life = None
        for i in range(peak_idx + 1, len(arr)):
            if half_life is None and arr[i] <= peak_val * 0.5:
                half_life = i - peak_idx
            if quarter_life is None and arr[i] <= peak_val * 0.25:
                quarter_life = i - peak_idx
                break

        return {
            'half_life_steps': int(half_life) if half_life else None,
            'quarter_life_steps': int(quarter_life) if quarter_life else None,
        }

    def _composition_analysis(
        self, original: List, repost: List, comment: List,
    ) -> Dict[str, Any]:
        ori = np.array(original, dtype=float) if original else np.array([])
        rep = np.array(repost, dtype=float) if repost else np.array([])
        com = np.array(comment, dtype=float) if comment else np.array([])

        if ori.size == 0 and rep.size == 0 and com.size == 0:
            return {'original_ratio_growth': 0.0, 'repost_ratio_peak': 0.0,
                    'type_entropy_evolution': []}

        n = max(ori.size, rep.size, com.size)
        if ori.size < n:
            ori = np.pad(ori, (0, n - ori.size))
        if rep.size < n:
            rep = np.pad(rep, (0, n - rep.size))
        if com.size < n:
            com = np.pad(com, (0, n - com.size))

        total = ori + rep + com
        total_safe = np.where(total > 0, total, 1.0)

        ori_ratio = ori / total_safe
        rep_ratio = rep / total_safe
        com_ratio = com / total_safe

        # Shannon entropy at each step
        entropy_evo: List[float] = []
        for i in range(n):
            probs = np.array([ori_ratio[i], rep_ratio[i], com_ratio[i]])
            probs = probs[probs > 0]
            entropy_evo.append(float(-np.sum(probs * np.log2(probs))) if len(probs) else 0.0)

        # Summary ratios at key phases
        growth_end = min(n // 3, n - 1)
        peak_region = slice(max(0, n // 3), min(2 * n // 3, n))

        return {
            'original_ratio_growth': float(np.mean(ori_ratio[:max(growth_end, 1)])),
            'repost_ratio_peak': float(np.mean(rep_ratio[peak_region])),
            'type_entropy_evolution': [round(e, 4) for e in entropy_evo],
        }

    def _extra_metrics(self, activity: List[float]) -> Dict[str, Any]:
        arr = np.array(activity, dtype=float)
        mean_val = float(np.mean(arr)) if arr.size else 1.0
        max_val = float(np.max(arr)) if arr.size else 0.0

        burstiness = max_val / mean_val if mean_val > 0 else 0.0
        cv = float(np.std(arr) / mean_val) if mean_val > 0 else 0.0

        kurtosis_val = 0.0
        if arr.size >= 4:
            kurtosis_val = float(sp_stats.kurtosis(arr, fisher=True))

        # Total active duration (first→last above 5 % of peak)
        threshold = max_val * 0.05
        active_indices = np.where(arr >= threshold)[0]
        duration = int(active_indices[-1] - active_indices[0] + 1) if active_indices.size else 0

        return {
            'burstiness': burstiness,
            'kurtosis': kurtosis_val,
            'coefficient_of_variation': cv,
            'active_duration_steps': duration,
        }

    def _compare_with_real(
        self,
        sim_act: List[float],
        sim_times: list,
        real_act: List[float],
        real_times: list,
        sim_phases: Dict,
    ) -> Dict[str, Any]:
        sim_norm = list(normalize(sim_act))
        real_norm = list(normalize(real_act))
        curve_sim = calculate_curve_similarity(sim_norm, real_norm)

        real_phases = self._detect_phases(real_act)

        # Phase similarity: correlation of phase ratios
        sim_ratios = np.array(sim_phases['phase_ratios'])
        real_ratios = np.array(real_phases['phase_ratios'])
        phase_sim = 1.0 - float(np.sqrt(np.mean((sim_ratios - real_ratios) ** 2)))

        # Peak offset (normalised)
        sim_peak_ratio = sim_phases['peak_position_ratio']
        real_peak_ratio = real_phases['peak_position_ratio']
        peak_offset = abs(sim_peak_ratio - real_peak_ratio)

        # Half‑life ratio
        sim_hl = self._halflife_analysis(sim_act, sim_phases).get('half_life_steps')
        real_hl = self._halflife_analysis(real_act, real_phases).get('half_life_steps')
        hl_ratio = 0.0
        if sim_hl and real_hl:
            hl_ratio = min(sim_hl, real_hl) / max(sim_hl, real_hl)

        return {
            'phase_similarity': float(phase_sim),
            'peak_offset': float(peak_offset),
            'half_life_ratio': float(hl_ratio),
            'curve_similarity': curve_sim,
            'real_phase_ratios': [float(r) for r in real_ratios],
        }

    # ==================================================================
    # Visualisation — every plot is a SEPARATE figure
    # ==================================================================

    def _phase_shade(self, ax, phases_dict: Dict, n: int, x_values=None):
        """Add coloured phase shading to an axis."""
        for phase_name, info in phases_dict.items():
            s, e = info['start'], info['end']
            if x_values is not None and len(x_values) > e:
                ax.axvspan(x_values[s], x_values[min(e, len(x_values) - 1)],
                           alpha=0.12, color=_PHASE_COLORS.get(phase_name, '#eee'),
                           label=_PHASE_LABELS.get(phase_name, phase_name))
            else:
                ax.axvspan(s, e, alpha=0.12,
                           color=_PHASE_COLORS.get(phase_name, '#eee'),
                           label=_PHASE_LABELS.get(phase_name, phase_name))

    # 1. lifecycle_hotness
    def _plot_lifecycle_hotness(self, times, activity, phases_info, raw_total):
        fig, ax = plt.subplots(figsize=FIG_SIZE_WIDE)
        n = len(activity)
        smoothed = smooth(activity, target_len=n)

        use_time = times and len(times) >= n
        x = times[:n] if use_time else list(range(n))

        self._phase_shade(ax, phases_info['phases'], n, x_values=x if use_time else None)

        ax.fill_between(x, smoothed, alpha=ALPHA, color=C_SIM['total'])
        ax.plot(x, smoothed, color=C_SIM['total'], lw=LW, label='Activity (smoothed)')

        if raw_total and len(raw_total) >= n:
            ax.plot(x, np.array(raw_total[:n], dtype=float),
                    color=C_SIM['total'], lw=0.6, alpha=0.35, label='Activity (raw)')

        peak_idx = phases_info['peak_step']
        peak_x = x[peak_idx] if peak_idx < len(x) else peak_idx
        ax.axvline(peak_x, color='red', ls='--', lw=LW_MINOR, alpha=0.7)
        ax.annotate(f'Peak (step {peak_idx})', xy=(peak_x, phases_info['peak_value']),
                    xytext=(15, 15), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=FONT_SIZE['annotation'], color='red')

        if use_time:
            setup_time_axis(ax)
        else:
            ax.set_xlabel('Step')
        ax.set_ylabel('Activity Count')
        ax.set_title('Opinion Lifecycle — Activity Curve with Phases', fontweight='bold')
        add_grid(ax)
        add_legend(ax, loc='upper right', ncol=2)
        save_figure(fig, self.output_dir / 'lifecycle_hotness.png')
        logger.info("[SAVED] lifecycle_hotness.png")

    # 2. lifecycle_growth_rate
    def _plot_growth_rate(self, activity, times):
        arr = np.array(activity, dtype=float)
        n = len(arr)
        if n < 3:
            return

        safe = np.where(arr > 0, arr, 1e-9)
        instant_rate = np.diff(np.log(safe))
        rate_smooth = smooth(list(instant_rate), w=max(3, n // 20), target_len=len(instant_rate))

        fig, ax = plt.subplots(figsize=FIG_SIZE_WIDE)
        use_time = times and len(times) >= n
        x = times[1:n] if use_time and len(times) >= n else list(range(len(instant_rate)))
        x = x[:len(rate_smooth)]

        pos = np.clip(rate_smooth, 0, None)
        neg = np.clip(rate_smooth, None, 0)
        ax.fill_between(x, pos, alpha=0.35, color='#2ca02c', label='Growth')
        ax.fill_between(x, neg, alpha=0.35, color='#d62728', label='Decay')
        ax.plot(x, rate_smooth, color='#333', lw=LW_MINOR)
        ax.axhline(0, color='black', lw=0.8, ls='-')

        if use_time:
            setup_time_axis(ax)
        else:
            ax.set_xlabel('Step')
        ax.set_ylabel('Instantaneous Growth Rate (log‑scale)')
        ax.set_title('Growth Rate Over Time', fontweight='bold')
        add_grid(ax)
        add_legend(ax)
        save_figure(fig, self.output_dir / 'lifecycle_growth_rate.png')
        logger.info("[SAVED] lifecycle_growth_rate.png")

    # 3. actions_per_step (bar chart)
    def _plot_actions_per_step(self, actions_per_step):
        if not actions_per_step:
            return
        fig, ax = plt.subplots(figsize=FIG_SIZE_WIDE)
        n = len(actions_per_step)
        x = np.arange(n)
        bars = ax.bar(x, actions_per_step, color=C_SIM['comment'], alpha=0.78, edgecolor='white', lw=0.3)

        mean_val = np.mean(actions_per_step)
        ax.axhline(mean_val, color='#d62728', ls='--', lw=LW_MINOR, label=f'Mean = {mean_val:.1f}')

        peak = int(np.argmax(actions_per_step))
        bars[peak].set_color('#d62728')
        bars[peak].set_alpha(0.9)

        ax.set_xlabel('Simulation Step')
        ax.set_ylabel('Number of Actions')
        ax.set_title('Actions per Simulation Step', fontweight='bold')
        add_grid(ax)
        add_legend(ax)
        save_figure(fig, self.output_dir / 'actions_per_step.png')
        logger.info("[SAVED] actions_per_step.png")

    # 4. cumulative_activity
    def _plot_cumulative_activity(self, actions_per_step, active_agents):
        if not actions_per_step:
            return
        fig, ax = plt.subplots(figsize=FIG_SIZE_WIDE)
        cumul = np.cumsum(actions_per_step)
        x = np.arange(len(cumul))

        ax.fill_between(x, cumul, alpha=ALPHA, color=C_SIM['total'])
        ax.plot(x, cumul, color=C_SIM['total'], lw=LW, label='Cumulative Actions')

        if active_agents:
            ax2 = ax.twinx()
            n_a = min(len(active_agents), len(x))
            ax2.plot(x[:n_a], active_agents[:n_a], color='#ff7f0e', lw=LW, ls='--', label='Active Agents')
            ax2.set_ylabel('Active Agents', color='#ff7f0e')
            ax2.tick_params(axis='y', labelcolor='#ff7f0e')
            lines2, labels2 = ax2.get_legend_handles_labels()
        else:
            lines2, labels2 = [], []

        ax.set_xlabel('Simulation Step')
        ax.set_ylabel('Cumulative Actions')
        ax.set_title('Cumulative Activity & Active Agents', fontweight='bold')
        add_grid(ax)
        lines1, labels1 = ax.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left',
                  frameon=True, fancybox=False, edgecolor='black',
                  fontsize=FONT_SIZE['legend'])
        save_figure(fig, self.output_dir / 'cumulative_activity.png')
        logger.info("[SAVED] cumulative_activity.png")

    # 5. hawkes_intensity
    def _plot_hawkes_intensity(self, intensity):
        if not intensity:
            return
        fig, ax = plt.subplots(figsize=FIG_SIZE_WIDE)
        arr = np.array(intensity, dtype=float)
        x = np.arange(len(arr))

        ax.fill_between(x, arr, alpha=ALPHA, color='#ff7f0e')
        ax.plot(x, arr, color='#ff7f0e', lw=LW, label='Hawkes Intensity')

        peak = int(np.argmax(arr))
        ax.axvline(peak, color='red', ls='--', lw=LW_MINOR, alpha=0.7)
        ax.annotate(f'Peak λ={arr[peak]:.2f}', xy=(peak, arr[peak]),
                    xytext=(15, 10), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=FONT_SIZE['annotation'], color='red')

        # confidence‑style upper/lower band via rolling std
        w = max(3, len(arr) // 15)
        rolling_std = np.array([np.std(arr[max(0, i - w):i + 1]) for i in range(len(arr))])
        upper = arr + rolling_std
        lower = np.clip(arr - rolling_std, 0, None)
        ax.fill_between(x, lower, upper, alpha=0.12, color='#ff7f0e', label='±1 σ band')

        ax.set_xlabel('Simulation Step')
        ax.set_ylabel('Intensity (λ)')
        ax.set_title('Hawkes Process Intensity', fontweight='bold')
        add_grid(ax)
        add_legend(ax)
        save_figure(fig, self.output_dir / 'hawkes_intensity.png')
        logger.info("[SAVED] hawkes_intensity.png")

    # 6. phase_duration_comparison (real vs sim)
    def _plot_phase_duration_comparison(self, sim_act, real_act):
        sim_phases = self._detect_phases(sim_act)
        real_phases = self._detect_phases(real_act)

        fig, ax = plt.subplots(figsize=FIG_SIZE)
        phase_names = ['incubation', 'growth', 'peak', 'decay', 'dormancy']
        labels = [_PHASE_LABELS[p] for p in phase_names]

        sim_ratios = sim_phases['phase_ratios']
        real_ratios = real_phases['phase_ratios']

        x = np.arange(len(labels))
        w = 0.32
        bars_sim = ax.bar(x - w / 2, sim_ratios, w, label='Simulated',
                          color=C_SIM['total'], alpha=0.85, edgecolor='black', lw=0.5)
        bars_real = ax.bar(x + w / 2, real_ratios, w, label='Real',
                           color=C_REAL['total'], alpha=0.85, edgecolor='black', lw=0.5)

        for bar, val in zip(bars_sim, sim_ratios):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.0%}', ha='center', va='bottom', fontsize=9)
        for bar, val in zip(bars_real, real_ratios):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.0%}', ha='center', va='bottom', fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Phase Duration Ratio')
        ax.set_title('Lifecycle Phase Duration — Sim vs Real', fontweight='bold')
        add_grid(ax)
        add_legend(ax)
        save_figure(fig, self.output_dir / 'phase_duration_comparison.png')
        logger.info("[SAVED] phase_duration_comparison.png")

    # 7. activity_composition (stacked area)
    def _plot_activity_composition(self, times, original, repost, comment):
        ori = np.array(original, dtype=float) if original else np.array([])
        rep = np.array(repost, dtype=float) if repost else np.array([])
        com = np.array(comment, dtype=float) if comment else np.array([])
        if ori.size == 0 and rep.size == 0 and com.size == 0:
            return

        n = max(ori.size, rep.size, com.size)
        if ori.size < n:
            ori = np.pad(ori, (0, n - ori.size))
        if rep.size < n:
            rep = np.pad(rep, (0, n - rep.size))
        if com.size < n:
            com = np.pad(com, (0, n - com.size))

        fig, ax = plt.subplots(figsize=FIG_SIZE_WIDE)
        use_time = times and len(times) >= n
        x = times[:n] if use_time else list(range(n))

        ax.stackplot(
            x, ori, rep, com,
            labels=['Original', 'Repost', 'Comment'],
            colors=[_COMP_COLORS['original'], _COMP_COLORS['repost'], _COMP_COLORS['comment']],
            alpha=0.72,
        )

        if use_time:
            setup_time_axis(ax)
        else:
            ax.set_xlabel('Step')
        ax.set_ylabel('Activity Count')
        ax.set_title('Activity Composition Over Time', fontweight='bold')
        add_grid(ax)
        add_legend(ax, loc='upper left')
        save_figure(fig, self.output_dir / 'activity_composition.png')
        logger.info("[SAVED] activity_composition.png")

    # 8. lifecycle_metrics_summary (horizontal bar)
    def _plot_metrics_summary(self, results: Dict):
        gm = results.get('growth_metrics', {})
        pa = results.get('peak_analysis', {})
        lp = results.get('lifecycle_phases', {})

        labels, values, colors = [], [], []

        def _add(label, val, color='#6eb169'):
            if val is not None:
                labels.append(label)
                values.append(float(val))
                colors.append(color)

        _add('Peak Position Ratio', lp.get('peak_position_ratio'), '#ff7f0e')
        _add('Burstiness', gm.get('burstiness'), '#d62728')
        _add('Growth Rate', gm.get('growth_rate'), '#2ca02c')
        _add('Decay Rate', abs(gm.get('decay_rate', 0)), '#9467bd')
        _add('Growth R²', gm.get('growth_r2'), '#8c564b')
        _add('Decay R²', gm.get('decay_r2'), '#e377c2')
        _add('Coeff of Variation', gm.get('coefficient_of_variation'), '#17becf')
        _add('Kurtosis', gm.get('kurtosis'), '#bcbd22')
        _add('Peak Prominence', pa.get('peak_prominence'), '#1f77b4')
        _add('Peak Width (steps)', pa.get('peak_width'), '#aec7e8')
        hl = gm.get('half_life_steps')
        if hl:
            _add('Half‑Life (steps)', hl, '#ff9896')
        ql = gm.get('quarter_life_steps')
        if ql:
            _add('Quarter‑Life (steps)', ql, '#c5b0d5')

        if not labels:
            return

        fig, ax = plt.subplots(figsize=(9, max(4, len(labels) * 0.45)))
        y = np.arange(len(labels))
        bars = ax.barh(y, values, color=colors, alpha=0.85, edgecolor='black', lw=0.4, height=0.6)
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + max(abs(v) for v in values) * 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}', va='center', fontsize=FONT_SIZE['annotation'])
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel('Value')
        ax.set_title('Lifecycle Metrics Summary', fontweight='bold')
        ax.invert_yaxis()
        add_grid(ax)
        save_figure(fig, self.output_dir / 'lifecycle_metrics_summary.png')
        logger.info("[SAVED] lifecycle_metrics_summary.png")

    # 9. lifecycle_comparison (sim vs real normalised)
    def _plot_lifecycle_comparison(self, sim_times, sim_act, real_times, real_act, phases_info):
        fig, ax = plt.subplots(figsize=FIG_SIZE_WIDE)

        sim_norm = normalize(sim_act)
        real_norm = normalize(real_act)

        n_s = len(sim_norm)
        n_r = len(real_norm)

        use_time_sim = sim_times and len(sim_times) >= n_s
        use_time_real = real_times and len(real_times) >= n_r

        if use_time_sim and use_time_real:
            xs = sim_times[:n_s]
            xr = real_times[:n_r]
        else:
            xs = np.linspace(0, 1, n_s)
            xr = np.linspace(0, 1, n_r)

        sim_smooth = smooth(list(sim_norm), target_len=n_s)
        real_smooth = smooth(list(real_norm), target_len=n_r)

        # Phase shading (sim)
        self._phase_shade(ax, phases_info['phases'], n_s,
                          x_values=xs if use_time_sim else None)

        ax.fill_between(xs, sim_smooth, alpha=ALPHA, color=C_SIM['total'])
        ax.plot(xs, sim_smooth, color=C_SIM['total'], lw=LW, label='Simulated (norm.)')

        ax.fill_between(xr, real_smooth, alpha=ALPHA, color=C_REAL['total'])
        ax.plot(xr, real_smooth, color=C_REAL['total'], lw=LW, ls='--', label='Real (norm.)')

        if use_time_sim:
            setup_time_axis(ax)
        else:
            ax.set_xlabel('Normalised Timeline')
        ax.set_ylabel('Normalised Activity')
        ax.set_title('Lifecycle Comparison — Simulated vs Real', fontweight='bold')
        add_grid(ax)
        add_legend(ax)
        save_figure(fig, self.output_dir / 'lifecycle_comparison.png')
        logger.info("[SAVED] lifecycle_comparison.png")

    # ==================================================================
    # Console summary
    # ==================================================================

    def _print_summary(self, results: Dict):
        lp = results.get('lifecycle_phases', {})
        gm = results.get('growth_metrics', {})
        pa = results.get('peak_analysis', {})
        comp = results.get('comparison', {})

        if lp:
            print(f"    ✅ 总步数: {lp.get('total_steps', 0)}, "
                  f"峰值步: {lp.get('peak_step')}, "
                  f"峰值位置: {lp.get('peak_position_ratio', 0):.1%}")
            ratios = lp.get('phase_ratios', [])
            if ratios:
                names = ['Incub', 'Growth', 'Peak', 'Decay', 'Dorm']
                ratio_str = ' | '.join(f"{n}={r:.0%}" for n, r in zip(names, ratios))
                print(f"       阶段比例: {ratio_str}")
        if gm:
            print(f"    ✅ 增长率: {gm.get('growth_rate', 0):.4f} (R²={gm.get('growth_r2', 0):.3f}), "
                  f"衰减率: {gm.get('decay_rate', 0):.4f} (R²={gm.get('decay_r2', 0):.3f})")
            hl = gm.get('half_life_steps')
            if hl:
                print(f"       半衰期: {hl} 步")
            print(f"       突发性: {gm.get('burstiness', 0):.2f}, "
                  f"峰度: {gm.get('kurtosis', 0):.2f}, "
                  f"CV: {gm.get('coefficient_of_variation', 0):.2f}")
        if pa:
            print(f"    ✅ 主峰索引: {pa.get('primary_peak_idx')}, "
                  f"次峰数: {len(pa.get('secondary_peaks', []))}, "
                  f"显著度: {pa.get('peak_prominence', 0):.2f}, "
                  f"宽度: {pa.get('peak_width', 0):.1f}")
        if comp:
            print(f"    ✅ 对比 → 阶段相似度: {comp.get('phase_similarity', 0):.3f}, "
                  f"峰值偏移: {comp.get('peak_offset', 0):.3f}, "
                  f"半衰期比: {comp.get('half_life_ratio', 0):.3f}")
