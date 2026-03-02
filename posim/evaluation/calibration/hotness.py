# -*- coding: utf-8 -*-
"""
宏观热度曲线校准 + 时序节奏指标

对比模拟和真实数据的热度曲线（总量、原创、转发、评论）
包含：绝对值对比、归一化对比、分类对比、堆叠面积图

时序节奏指标：
1. 日内活跃节奏相似度（Circadian Rhythm Similarity）
2. 峰值定位精度（Peak Location Precision）
3. 生命周期阶段比例相似度（Lifecycle Phase Ratio Similarity）
4. 半衰期比值（Half-Life Ratio）
"""
import logging
import warnings
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional, List
from collections import Counter

from scipy.stats import wasserstein_distance, kendalltau, entropy as sp_entropy

try:
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean as _euclidean
    _HAS_FASTDTW = True
except ImportError:
    _HAS_FASTDTW = False

from ..base import BaseEvaluator
from ..utils import (
    smooth, normalize, calculate_curve_similarity, save_json,
    calculate_jsd, calculate_cosine_similarity_vec
)
from ..visualization import (
    FIG_SIZE, FIG_SIZE_TALL, FIG_SIZE_WIDE, LW, LW_MINOR, ALPHA,
    MARKER_SIZE, FONT_SIZE,
    C_SIM, C_REAL, setup_time_axis, add_grid, add_legend, save_figure
)

logger = logging.getLogger(__name__)


class HotnessCalibrationEvaluator(BaseEvaluator):
    """宏观热度曲线校准 + 时序节奏"""

    def __init__(self, output_dir: Path):
        super().__init__(output_dir / "hotness_calibration", name="hotness_calibration")

    def evaluate(self, sim_data: Dict[str, Any], real_data: Optional[Dict[str, Any]] = None,
                 **kwargs) -> Dict[str, Any]:
        self._log_section("宏观热度曲线校准")

        sim_agg = sim_data.get('aggregated', {})
        sim_times = sim_agg.get('times', [])
        sim_hotness = sim_agg.get('hotness', {})
        sim_actions = sim_agg.get('actions', [])
        micro_results = sim_data.get('micro_results', [])

        if not sim_times or not sim_hotness:
            print("    ⚠️ 无模拟热度数据，跳过")
            return {}

        results = {}
        results['sim_stats'] = self._compute_stats(sim_hotness, 'sim')

        has_real = real_data is not None and real_data.get('times') and real_data.get('hotness')

        if has_real:
            real_times = real_data['times']
            real_hotness = real_data['hotness']
            real_actions = real_data.get('actions', [])

            results['real_stats'] = self._compute_stats(real_hotness, 'real')

            print("    计算曲线相似度指标...")
            results['curve_similarity'] = {}
            for ct in ['total', 'original', 'repost', 'comment']:
                sc = sim_hotness.get(ct, [])
                rc = real_hotness.get(ct, [])
                if sc and rc:
                    sn = normalize(sc).tolist()
                    rn = normalize(rc).tolist()
                    base_abs = calculate_curve_similarity(sc, rc)
                    base_norm = calculate_curve_similarity(sn, rn)
                    ext_abs = self._compute_extended_similarity(sc, rc)
                    ext_norm = self._compute_extended_similarity(sn, rn)
                    base_abs.update(ext_abs)
                    base_norm.update(ext_norm)
                    results['curve_similarity'][ct] = {
                        'absolute': base_abs,
                        'normalized': base_norm,
                    }
                    na = base_abs.get('pearson', 0)
                    nn = base_norm.get('pearson', 0)
                    dtw_v = base_norm.get('dtw_distance', None)
                    cos_v = base_norm.get('cosine_similarity', None)
                    wass_v = base_abs.get('wasserstein_distance', None)
                    dtw_s = f", DTW={dtw_v:.2f}" if dtw_v is not None else ""
                    cos_s = f", Cos={cos_v:.3f}" if cos_v is not None else ""
                    wass_s = f", Wass={wass_v:.2f}" if wass_v is not None else ""
                    print(f"      {ct}: Pearson(abs)={na:.3f}, Pearson(norm)={nn:.3f}"
                          f"{cos_s}{dtw_s}{wass_s}")

            self._plot_comparison(sim_times, sim_hotness, real_times, real_hotness, results)

            # ---- 时序节奏指标 ----
            print("    计算时序节奏指标...")
            results['temporal_rhythm'] = self._analyze_temporal_rhythm(
                sim_times, sim_hotness, sim_actions, micro_results,
                real_times, real_hotness, real_actions)
        else:
            print("    无真实数据，仅绘制模拟热度曲线")
            self._plot_sim_only(sim_times, sim_hotness)

        self._save_results(results, "hotness_calibration_metrics.json")
        return results

    @staticmethod
    def _compute_extended_similarity(curve1: List[float], curve2: List[float]) -> Dict[str, Any]:
        """Compute additional similarity metrics beyond pearson/spearman/rmse/mae."""
        if not curve1 or not curve2:
            return {}

        min_len = min(len(curve1), len(curve2))
        c1 = np.array(curve1[:min_len], dtype=float)
        c2 = np.array(curve2[:min_len], dtype=float)
        result: Dict[str, Any] = {}

        # Cosine similarity (curves treated as vectors)
        result['cosine_similarity'] = float(calculate_cosine_similarity_vec(c1, c2))

        # Kendall's tau rank correlation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                tau, p = kendalltau(c1, c2)
                result['kendall_tau'] = 0.0 if np.isnan(tau) else float(tau)
                result['kendall_tau_p'] = 1.0 if np.isnan(p) else float(p)
            except Exception:
                result['kendall_tau'] = 0.0

        # KL divergence (normalized curves as probability distributions)
        c1_pos = np.maximum(c1, 0)
        c2_pos = np.maximum(c2, 0)
        s1, s2 = c1_pos.sum(), c2_pos.sum()
        if s1 > 0 and s2 > 0:
            p = c1_pos / s1
            q = c2_pos / s2
            eps = 1e-10
            p = np.clip(p, eps, None)
            q = np.clip(q, eps, None)
            result['kl_divergence_sim_to_real'] = float(sp_entropy(p, q))
            result['kl_divergence_real_to_sim'] = float(sp_entropy(q, p))
            result['kl_divergence_symmetric'] = float(
                (sp_entropy(p, q) + sp_entropy(q, p)) / 2
            )

        # Wasserstein / Earth Mover's Distance
        try:
            result['wasserstein_distance'] = float(wasserstein_distance(c1, c2))
        except Exception:
            pass

        # DTW distance
        if _HAS_FASTDTW:
            try:
                dtw_dist, _ = fastdtw(c1.reshape(-1, 1), c2.reshape(-1, 1), dist=_euclidean)
                result['dtw_distance'] = float(dtw_dist)
            except Exception:
                pass
        else:
            try:
                from scipy.spatial.distance import cdist
                d = cdist(c1.reshape(-1, 1), c2.reshape(-1, 1), metric='euclidean')
                n, m = d.shape
                dtw_mat = np.full((n + 1, m + 1), np.inf)
                dtw_mat[0, 0] = 0.0
                for i in range(1, n + 1):
                    for j in range(1, m + 1):
                        dtw_mat[i, j] = d[i - 1, j - 1] + min(
                            dtw_mat[i - 1, j], dtw_mat[i, j - 1], dtw_mat[i - 1, j - 1]
                        )
                result['dtw_distance'] = float(dtw_mat[n, m])
            except Exception:
                pass

        # AUC ratio (area under curve sim / area under curve real)
        _trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
        auc_sim = float(_trapz(c1))
        auc_real = float(_trapz(c2))
        result['auc_sim'] = auc_sim
        result['auc_real'] = auc_real
        if auc_real != 0:
            result['auc_ratio'] = float(auc_sim / auc_real)
        else:
            result['auc_ratio'] = float('inf') if auc_sim > 0 else 1.0

        return result

    def _compute_stats(self, hotness: Dict, prefix: str) -> Dict:
        stats = {}
        for k in ['total', 'original', 'repost', 'comment']:
            data = hotness.get(k, [])
            if data:
                stats[k] = {
                    'total_count': int(sum(data)),
                    'peak': int(max(data)),
                    'peak_index': int(np.argmax(data)),
                    'mean': float(np.mean(data)),
                    'std': float(np.std(data))
                }
        return stats

    # ─────────────────────────────────────────────────────────
    # 时序节奏指标
    # ─────────────────────────────────────────────────────────
    def _analyze_temporal_rhythm(self, sim_times, sim_hotness, sim_actions, micro_results,
                                 real_times, real_hotness, real_actions):
        metrics = {}

        # --- 1. 日内活跃节奏相似度 ---
        sim_hourly = np.zeros(24)
        for t in sim_times:
            h = t.hour
            idx = sim_times.index(t) if t in sim_times else -1
            if idx >= 0:
                total_curve = sim_hotness.get('total', [])
                if idx < len(total_curve):
                    sim_hourly[h] += total_curve[idx]

        real_hourly = np.zeros(24)
        for t in real_times:
            h = t.hour
            idx = real_times.index(t) if t in real_times else -1
            if idx >= 0:
                total_curve = real_hotness.get('total', [])
                if idx < len(total_curve):
                    real_hourly[h] += total_curve[idx]

        if sim_hourly.sum() > 0 and real_hourly.sum() > 0:
            sh_norm = sim_hourly / sim_hourly.sum()
            rh_norm = real_hourly / real_hourly.sum()
            jsd = calculate_jsd(sh_norm, rh_norm)
            from scipy.stats import pearsonr
            try:
                pr, _ = pearsonr(sh_norm, rh_norm)
            except:
                pr = 0.0
            metrics['circadian_jsd'] = float(jsd)
            metrics['circadian_similarity'] = float(1 - jsd)
            metrics['circadian_pearson'] = float(pr)
            print(f"      日内节奏相似度: {1 - jsd:.4f} (JSD={jsd:.4f}, Pearson={pr:.4f})")

        # --- 2. 峰值定位精度 ---
        sim_total = sim_hotness.get('total', [])
        real_total = real_hotness.get('total', [])
        if sim_total and real_total:
            sim_peak_idx = int(np.argmax(sim_total))
            real_peak_idx = int(np.argmax(real_total))
            sim_peak_pos = sim_peak_idx / max(len(sim_total) - 1, 1)
            real_peak_pos = real_peak_idx / max(len(real_total) - 1, 1)
            offset = abs(sim_peak_pos - real_peak_pos)
            metrics['peak_offset_ratio'] = float(offset)
            metrics['peak_precision'] = float(1 - offset)
            metrics['sim_peak_position'] = float(sim_peak_pos)
            metrics['real_peak_position'] = float(real_peak_pos)
            print(f"      峰值精度: {1 - offset:.4f} (偏移={offset:.4f})")

        # --- 3. 生命周期阶段比例相似度 ---
        def _lifecycle_phases(curve):
            if not curve or max(curve) == 0:
                return [0.33, 0.34, 0.33]
            peak_idx = int(np.argmax(curve))
            n = len(curve)
            peak_val = max(curve)
            threshold = peak_val * 0.1

            growth_end = peak_idx
            decay_start = peak_idx
            for i in range(peak_idx, n):
                if curve[i] < threshold:
                    decay_start = i
                    break
            else:
                decay_start = n

            growth_ratio = growth_end / max(n, 1)
            peak_ratio = (decay_start - growth_end) / max(n, 1)
            decay_ratio = (n - decay_start) / max(n, 1)
            return [growth_ratio, peak_ratio, decay_ratio]

        if sim_total and real_total:
            sim_phases = _lifecycle_phases(sim_total)
            real_phases = _lifecycle_phases(real_total)
            cos_sim = calculate_cosine_similarity_vec(sim_phases, real_phases)
            metrics['lifecycle_sim_phases'] = sim_phases
            metrics['lifecycle_real_phases'] = real_phases
            metrics['lifecycle_phase_similarity'] = float(cos_sim)
            print(f"      生命周期阶段相似度: {cos_sim:.4f}")

        # --- 4. 半衰期比值 ---
        def _half_life(curve):
            if not curve or max(curve) == 0:
                return 0
            peak_idx = int(np.argmax(curve))
            peak_val = max(curve)
            half_val = peak_val * 0.5
            for i in range(peak_idx, len(curve)):
                if curve[i] <= half_val:
                    return i - peak_idx
            return len(curve) - peak_idx

        if sim_total and real_total:
            sh = _half_life(sim_total)
            rh = _half_life(real_total)
            ratio = min(sh, rh) / max(sh, rh) if max(sh, rh) > 0 else 1.0
            metrics['sim_half_life_steps'] = int(sh)
            metrics['real_half_life_steps'] = int(rh)
            metrics['half_life_ratio'] = float(ratio)
            print(f"      半衰期比值: {ratio:.4f} (模拟={sh}, 真实={rh})")

        # ---- 绘制时序节奏图 ----
        self._plot_temporal_rhythm(sim_times, sim_hotness, real_times, real_hotness, metrics)
        return metrics

    def _plot_temporal_rhythm(self, sim_times, sim_hotness, real_times, real_hotness, metrics):
        hours = np.arange(24)

        sim_hourly = np.zeros(24)
        for i, t in enumerate(sim_times):
            total_c = sim_hotness.get('total', [])
            if i < len(total_c):
                sim_hourly[t.hour] += total_c[i]
        real_hourly = np.zeros(24)
        for i, t in enumerate(real_times):
            total_c = real_hotness.get('total', [])
            if i < len(total_c):
                real_hourly[t.hour] += total_c[i]

        sim_total = sim_hotness.get('total', [])
        real_total = real_hotness.get('total', [])

        # (a) Circadian Activity Profile
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        if sim_hourly.sum() > 0:
            ax.plot(hours, sim_hourly / sim_hourly.sum(), 'o-', color=C_SIM['total'],
                    lw=LW, markersize=MARKER_SIZE, label='Simulation')
        if real_hourly.sum() > 0:
            ax.plot(hours, real_hourly / real_hourly.sum(), 's--', color=C_REAL['total'],
                    lw=LW, markersize=MARKER_SIZE, label='Real Data')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Activity Proportion')
        ax.set_title('Circadian Activity Profile', fontweight='bold')
        ax.set_xticks(range(0, 24, 3))
        add_legend(ax)
        add_grid(ax)
        save_figure(fig, self.output_dir / 'temporal_circadian.png')

        # (b) Peak Alignment
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        if sim_total:
            x_sim = np.linspace(0, 1, len(sim_total))
            ax.plot(x_sim, normalize(sim_total), color=C_SIM['total'], lw=LW, label='Simulation')
            pi = int(np.argmax(sim_total))
            ax.axvline(x=pi / max(len(sim_total) - 1, 1), color=C_SIM['total'],
                       ls=':', lw=LW_MINOR, alpha=0.7)
        if real_total:
            x_real = np.linspace(0, 1, len(real_total))
            ax.plot(x_real, normalize(real_total), color=C_REAL['total'], lw=LW, ls='--', label='Real Data')
            pi = int(np.argmax(real_total))
            ax.axvline(x=pi / max(len(real_total) - 1, 1), color=C_REAL['total'],
                       ls=':', lw=LW_MINOR, alpha=0.7)
        ax.set_xlabel('Relative Timeline Position')
        ax.set_ylabel('Normalized Activity')
        ax.set_title('Peak Alignment', fontweight='bold')
        add_legend(ax)
        add_grid(ax)
        save_figure(fig, self.output_dir / 'temporal_peak_alignment.png')

        # (c) Lifecycle Phase Ratio
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        phases = ['Growth', 'Peak', 'Decay']
        sp = metrics.get('lifecycle_sim_phases', [0.33, 0.34, 0.33])
        rp = metrics.get('lifecycle_real_phases', [0.33, 0.34, 0.33])
        x = np.arange(len(phases))
        width = 0.35
        ax.bar(x - width / 2, sp, width, label='Simulation', color=C_SIM['total'],
               alpha=0.85, edgecolor='black', linewidth=0.5)
        ax.bar(x + width / 2, rp, width, label='Real Data', color=C_REAL['total'],
               alpha=0.85, edgecolor='black', linewidth=0.5)
        for i, (sv, rv) in enumerate(zip(sp, rp)):
            ax.text(i - width / 2, sv + 0.01, f'{sv:.2f}', ha='center', fontsize=FONT_SIZE['annotation'])
            ax.text(i + width / 2, rv + 0.01, f'{rv:.2f}', ha='center', fontsize=FONT_SIZE['annotation'])
        ax.set_xticks(x)
        ax.set_xticklabels(phases)
        ax.set_ylabel('Proportion of Total Duration')
        ax.set_title('Lifecycle Phase Ratio', fontweight='bold')
        add_legend(ax)
        add_grid(ax)
        save_figure(fig, self.output_dir / 'temporal_lifecycle_phases.png')

        # (d) Half-Life Visualization
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        if sim_total:
            peak_idx = int(np.argmax(sim_total))
            peak_val = max(sim_total)
            tail = sim_total[peak_idx:]
            x_tail = np.arange(len(tail))
            ax.plot(x_tail, smooth(tail, target_len=len(tail)), color=C_SIM['total'], lw=LW, label='Sim Decay')
            ax.axhline(y=peak_val * 0.5, color=C_SIM['total'], ls=':', alpha=0.5)
            sh = metrics.get('sim_half_life_steps', 0)
            ax.axvline(x=sh, color=C_SIM['total'], ls='--', alpha=0.5)
        if real_total:
            peak_idx = int(np.argmax(real_total))
            peak_val = max(real_total)
            tail = real_total[peak_idx:]
            x_tail = np.arange(len(tail))
            ax.plot(x_tail, smooth(tail, target_len=len(tail)), color=C_REAL['total'], lw=LW, ls='--', label='Real Decay')
            rh = metrics.get('real_half_life_steps', 0)
            ax.axvline(x=rh, color=C_REAL['total'], ls='--', alpha=0.5)
        ax.set_xlabel('Steps after Peak')
        ax.set_ylabel('Activity Count')
        ax.set_title('Half-Life Visualization', fontweight='bold')
        add_legend(ax)
        add_grid(ax)
        save_figure(fig, self.output_dir / 'temporal_half_life.png')

        logger.info("[SAVED] temporal rhythm plots (4 individual figures)")

    def _plot_comparison(self, sim_times, sim_hotness, real_times, real_hotness, results):
        sim_n = len(sim_times)

        # 1. 总热度绝对值对比
        fig, ax = plt.subplots(figsize=(8, 8))
        sd = sim_hotness.get('total', [])
        rd = real_hotness.get('total', [])
        ax.plot(sim_times[:len(sd)], smooth(sd, target_len=len(sd)),
                color=C_SIM['total'], lw=LW, label='Simulation')
        ax.fill_between(sim_times[:len(sd)], smooth(sd, target_len=len(sd)),
                        alpha=ALPHA, color=C_SIM['total'])
        ax.plot(real_times[:len(rd)], smooth(rd, target_len=len(rd)),
                color=C_REAL['total'], lw=LW, ls='--', label='Real Data')
        ax.fill_between(real_times[:len(rd)], smooth(rd, target_len=len(rd)),
                        alpha=ALPHA, color=C_REAL['total'])
        ax.set_xlabel('Time')
        ax.set_ylabel('Activity Count')
        ax.set_title('Total Activity Comparison (Absolute)', fontweight='bold')
        setup_time_axis(ax)
        add_legend(ax)
        add_grid(ax)
        save_figure(fig, self.output_dir / 'hotness_total_absolute.png')

        # 2. 归一化对比
        fig, ax = plt.subplots(figsize=(8, 8))
        if sd:
            ax.plot(sim_times[:len(sd)], normalize(smooth(sd, target_len=len(sd))),
                    color=C_SIM['total'], lw=LW, label='Simulation')
        if rd:
            ax.plot(real_times[:len(rd)], normalize(smooth(rd, target_len=len(rd))),
                    color=C_REAL['total'], lw=LW, ls='--', label='Real Data')
        ax.set_xlabel('Time')
        ax.set_ylabel('Normalized Activity')
        ax.set_title('Total Activity (Normalized)', fontweight='bold')
        setup_time_axis(ax)
        add_legend(ax)
        add_grid(ax)
        save_figure(fig, self.output_dir / 'hotness_total_normalized.png')

        # 3. 分类对比 (individual figures per type)
        labels = {'total': 'Total', 'original': 'Original', 'repost': 'Repost', 'comment': 'Comment'}
        for btype in ['original', 'repost', 'comment']:
            fig, ax = plt.subplots(figsize=FIG_SIZE)
            scd = sim_hotness.get(btype, [])
            rcd = real_hotness.get(btype, [])
            if scd:
                ax.plot(sim_times[:len(scd)], smooth(scd, target_len=len(scd)),
                        color=C_SIM[btype], lw=LW, label='Simulation')
            if rcd:
                ax.plot(real_times[:len(rcd)], smooth(rcd, target_len=len(rcd)),
                        color=C_REAL[btype], lw=LW, ls='--', label='Real Data')
            ax.set_xlabel('Time')
            ax.set_ylabel('Count')
            ax.set_title(f'{labels[btype]} Posts', fontweight='bold')
            add_legend(ax, loc='upper right')
            add_grid(ax)
            setup_time_axis(ax)
            save_figure(fig, self.output_dir / f'hotness_{btype}.png')

        # 4. 堆叠面积图
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        stack_data = []
        for k in ['original', 'repost', 'comment']:
            d = sim_hotness.get(k, [])
            stack_data.append(smooth(d, target_len=len(d)) if d else np.zeros(sim_n))
        ax.stackplot(sim_times[:sim_n], [s[:sim_n] for s in stack_data],
                     labels=['Original', 'Repost', 'Comment'],
                     colors=[C_SIM['original'], C_SIM['repost'], C_SIM['comment']], alpha=0.8)
        ax.set_xlabel('Time')
        ax.set_ylabel('Activity Count')
        ax.set_title('Simulated Activity Composition', fontweight='bold')
        setup_time_axis(ax)
        add_legend(ax, loc='upper right')
        add_grid(ax)
        save_figure(fig, self.output_dir / 'hotness_stacked.png')
        logger.info("[SAVED] hotness calibration plots (6 individual figures)")

    def _plot_sim_only(self, sim_times, sim_hotness):
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        d = sim_hotness.get('total', [])
        if d:
            ax.plot(sim_times[:len(d)], smooth(d, target_len=len(d)),
                    color=C_SIM['total'], lw=LW, label='Simulation')
            ax.fill_between(sim_times[:len(d)], smooth(d, target_len=len(d)),
                            alpha=ALPHA, color=C_SIM['total'])
        ax.set_xlabel('Time')
        ax.set_ylabel('Activity Count')
        ax.set_title('Simulation Activity', fontweight='bold')
        setup_time_axis(ax)
        add_legend(ax)
        add_grid(ax)
        save_figure(fig, self.output_dir / 'hotness_total.png')
