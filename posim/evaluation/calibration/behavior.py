# -*- coding: utf-8 -*-
"""
行为分布校准

评估模拟和真实的用户行为分布的一致性:
1. 行为类型分布对比（原创/转发/评论的比例，不含like）
2. 行为类型时间演化对比
3. 用户活跃度分布对比（每用户行为数分布）
4. 行为强度分布（每时间窗口行为数分布）
5. 用户参与度分层对比（头部/中部/尾部用户行为占比）
6. 按角色的行为比例分布（KOL/普通用户/媒体/政府）
7. 评论-原创比（C/O Ratio）时序相关
8. 首次发言时间分布
9. 舆论阵营转移率（Opinion Shift Rate）
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional, List
from collections import Counter, defaultdict

from ..base import BaseEvaluator
from ..utils import (
    smooth, normalize, save_json,
    calculate_curve_similarity, calculate_jsd, calculate_gini_coefficient,
    calculate_entropy, calculate_normalized_entropy,
    calculate_ks_test, calculate_kendall_tau, calculate_cosine_similarity_vec
)
from ..visualization import (
    FIG_SIZE, FIG_SIZE_TALL, FIG_SIZE_WIDE, LW, LW_MINOR, ALPHA,
    MARKER_SIZE, FONT_SIZE,
    C_SIM, C_REAL, STANDARD_TYPE_ORDER, STANDARD_TYPE_COLORS,
    setup_time_axis, add_grid, add_legend, save_figure, create_figure
)

logger = logging.getLogger(__name__)

# 模拟行为类型到标准类型的映射（排除 like）
SIM_TYPE_MAP = {
    'short_post': 'original', 'long_post': 'original',
    'repost': 'repost', 'repost_comment': 'repost',
    'short_comment': 'comment', 'long_comment': 'comment',
}

# 真实行为类型映射
REAL_TYPE_MAP = {
    'original': 'original',
    'repost': 'repost',
    'comment': 'comment'
}

# 详细行为类型中文->英文映射（用于 behavior_type_detailed 图表统一显示英文）
DETAILED_TYPE_TO_ENGLISH = {
    'short_post': 'short_post', 'long_post': 'long_post',
    'short_comment': 'short_comment', 'long_comment': 'long_comment',
    'repost': 'repost', 'repost_comment': 'repost_comment',
    '短博文': 'short_post', '长博文': 'long_post',
    '短评论': 'short_comment', '长评论': 'long_comment',
    '转发': 'repost',
    '转发并评论': 'repost_comment', '转发并短评论': 'repost_comment',
    '转发并长评论': 'repost_long_comment',
}

# 标准显示顺序（英文，用于 behavior_type_detailed 图表）
DETAILED_TYPE_ORDER = [
    'short_post', 'long_post', 'short_comment', 'long_comment',
    'repost', 'repost_comment', 'repost_long_comment'
]

# 角色标签
ROLE_ORDER = ['citizen', 'KOL', 'media', 'government']
ROLE_COLORS = {'citizen': '#1f77b4', 'KOL': '#ff7f0e', 'media': '#2ca02c', 'government': '#d62728'}

# 立场标签
STANCE_LABELS = ['Support', 'Oppose', 'Neutral']


def _to_english_type(raw: str) -> str:
    """将行为类型统一转为英文（用于详细分布图）"""
    if not raw or raw == 'like':
        return ''
    en = DETAILED_TYPE_TO_ENGLISH.get(raw)
    if en:
        return en
    return raw if raw.replace('_', '').replace(' ', '').isascii() else 'other'


def _filter_like(actions: List[Dict]) -> List[Dict]:
    """过滤掉 like 类型的行为"""
    return [a for a in actions if a.get('action_type', '') != 'like']


class BehaviorCalibrationEvaluator(BaseEvaluator):
    """行为分布校准"""

    def __init__(self, output_dir: Path):
        super().__init__(output_dir / "behavior_calibration", name="behavior_calibration")

    def evaluate(self, sim_data: Dict[str, Any], real_data: Optional[Dict[str, Any]] = None,
                 **kwargs) -> Dict[str, Any]:
        """执行行为分布校准"""
        self._log_section("行为分布校准")

        sim_agg = sim_data.get('aggregated', {})
        sim_actions = _filter_like(sim_agg.get('actions', []))
        sim_times = sim_agg.get('times', [])
        micro_results = _filter_like(sim_data.get('micro_results', []))

        if not sim_actions and not micro_results:
            print("    ⚠️ 无模拟数据，跳过")
            return {}

        if not sim_actions:
            sim_actions = micro_results

        results = {}
        has_real = real_data is not None and real_data.get('actions')
        real_actions = real_data.get('actions', []) if has_real else []
        real_times = real_data.get('times', []) if has_real else []

        # 1. 行为类型分布对比
        print("    [1/9] 行为类型分布对比...")
        results['type_distribution'] = self._analyze_type_distribution(
            sim_actions, real_actions, has_real)

        # 2. 行为类型时间演化对比
        print("    [2/9] 行为类型时间演化...")
        results['type_evolution'] = self._analyze_type_evolution(
            sim_actions, sim_times, real_actions, real_times, has_real)

        # 3. 用户活跃度分布对比
        print("    [3/9] 用户活跃度分布...")
        results['user_activity'] = self._analyze_user_activity(
            sim_actions, real_actions, has_real)

        # 4. 行为强度分布
        print("    [4/9] 行为强度分布...")
        results['behavior_intensity'] = self._analyze_behavior_intensity(
            sim_actions, sim_times, real_actions, real_times, has_real)

        # 5. 用户参与度分层对比
        print("    [5/9] 用户参与度分层...")
        results['user_engagement'] = self._analyze_user_engagement(
            sim_actions, real_actions, has_real)

        # 6. 按角色的行为比例分布
        print("    [6/9] 角色行为分布...")
        results['role_distribution'] = self._analyze_role_distribution(
            sim_actions, real_actions, has_real)

        # 7. 评论-原创比时序
        print("    [7/9] 评论-原创比时序...")
        results['co_ratio'] = self._analyze_co_ratio(
            sim_actions, sim_times, real_actions, real_times, has_real)

        # 8. 首次发言时间分布
        print("    [8/9] 首次发言时间分布...")
        results['first_speak'] = self._analyze_first_speak(
            sim_actions, sim_times, real_actions, real_times, has_real)

        # 9. 舆论阵营转移率
        print("    [9/9] 舆论阵营转移率...")
        results['opinion_shift'] = self._analyze_opinion_shift(
            sim_actions, real_actions, has_real)

        self._save_results(results, "behavior_calibration_metrics.json")
        self._print_summary(results)
        return results

    def _get_standard_type(self, action, source='sim'):
        """将行为类型映射到标准类型（排除 like）"""
        if source == 'sim':
            raw_type = action.get('action_type', '')
            return SIM_TYPE_MAP.get(raw_type)
        else:
            raw_type = action.get('type', '')
            return REAL_TYPE_MAP.get(raw_type)

    # ─────────────────────────────────────────────────────────
    # 1. 行为类型分布对比
    # ─────────────────────────────────────────────────────────
    def _analyze_type_distribution(self, sim_actions, real_actions, has_real):
        metrics = {}

        sim_types = Counter()
        sim_detailed = Counter()
        for a in sim_actions:
            std_type = self._get_standard_type(a, 'sim')
            if std_type:
                sim_types[std_type] += 1
            raw = a.get('action_type', 'unknown')
            if raw != 'like':
                en_key = _to_english_type(raw)
                if en_key:
                    sim_detailed[en_key] += 1

        metrics['sim_type_counts'] = dict(sim_types)
        metrics['sim_type_detailed'] = dict(sim_detailed)
        sim_total = max(sum(sim_types.values()), 1)
        metrics['sim_type_ratios'] = {k: float(v / sim_total) for k, v in sim_types.items()}

        real_types = Counter()
        real_detailed = Counter()
        if has_real:
            for a in real_actions:
                std_type = self._get_standard_type(a, 'real')
                if std_type:
                    real_types[std_type] += 1
                raw_bt = a.get('behavior_type', '')
                if raw_bt:
                    en_key = _to_english_type(raw_bt)
                    if en_key:
                        real_detailed[en_key] += 1
            metrics['real_type_counts'] = dict(real_types)
            metrics['real_type_detailed'] = dict(real_detailed)
            real_total = max(sum(real_types.values()), 1)
            metrics['real_type_ratios'] = {k: float(v / real_total) for k, v in real_types.items()}

        # ---- Figure: Sim pie chart ----
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        labels_sim = [t for t in STANDARD_TYPE_ORDER if sim_types.get(t, 0) > 0]
        sizes_sim = [sim_types[k] for k in labels_sim]
        colors_sim = [STANDARD_TYPE_COLORS.get(k, '#888') for k in labels_sim]
        ax.pie(sizes_sim, labels=labels_sim, autopct='%1.1f%%',
               colors=colors_sim, startangle=90, textprops={'fontsize': 9})
        ax.set_title('Simulation Behavior Distribution', fontweight='bold')
        save_figure(fig, self.output_dir / 'behavior_type_pie_sim.png')

        if has_real:
            # ---- Figure: Real pie chart ----
            fig, ax = plt.subplots(figsize=FIG_SIZE)
            labels_real = [t for t in STANDARD_TYPE_ORDER if real_types.get(t, 0) > 0]
            sizes_real = [real_types[k] for k in labels_real]
            colors_real = [STANDARD_TYPE_COLORS.get(k, '#888') for k in labels_real]
            ax.pie(sizes_real, labels=labels_real, autopct='%1.1f%%',
                   colors=colors_real, startangle=90, textprops={'fontsize': 9})
            ax.set_title('Real Behavior Distribution', fontweight='bold')
            save_figure(fig, self.output_dir / 'behavior_type_pie_real.png')

            # ---- Figure: Bar comparison ----
            fig, ax = plt.subplots(figsize=FIG_SIZE)
            all_types = STANDARD_TYPE_ORDER
            x = np.arange(len(all_types))
            width = 0.35
            sim_ratios = [sim_types.get(t, 0) / sim_total for t in all_types]
            real_ratios = [real_types.get(t, 0) / real_total for t in all_types]
            bars1 = ax.bar(x - width / 2, sim_ratios, width, label='Simulation',
                           color=C_SIM['total'], alpha=0.85, edgecolor='black', linewidth=0.5)
            bars2 = ax.bar(x + width / 2, real_ratios, width, label='Real Data',
                           color=C_REAL['total'], alpha=0.85, edgecolor='black', linewidth=0.5)
            for bar, val in zip(bars1, sim_ratios):
                if val > 0.01:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                            f'{val:.1%}', ha='center', va='bottom', fontsize=FONT_SIZE['annotation'])
            for bar, val in zip(bars2, real_ratios):
                if val > 0.01:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                            f'{val:.1%}', ha='center', va='bottom', fontsize=FONT_SIZE['annotation'])
            ax.set_xticks(x)
            ax.set_xticklabels(all_types)
            ax.set_ylabel('Proportion')
            ax.set_title('Behavior Type Distribution Comparison', fontweight='bold')
            add_legend(ax)
            add_grid(ax)
            save_figure(fig, self.output_dir / 'behavior_type_comparison.png')

            sim_vec = np.array([sim_types.get(t, 0) for t in all_types], dtype=float)
            real_vec = np.array([real_types.get(t, 0) for t in all_types], dtype=float)
            if sim_vec.sum() > 0:
                sim_vec = sim_vec / sim_vec.sum()
            if real_vec.sum() > 0:
                real_vec = real_vec / real_vec.sum()

            jsd = calculate_jsd(sim_vec, real_vec)
            metrics['type_distribution_jsd'] = float(jsd)
            metrics['type_distribution_similarity'] = float(1 - jsd)
            print(f"      行为类型分布JSD: {jsd:.4f}, 相似度: {1 - jsd:.4f}")

            # Hellinger distance: H(P,Q) = sqrt(1 - sum(sqrt(pi*qi))) / sqrt(2)
            bc_sum = np.sum(np.sqrt(sim_vec * real_vec))
            hellinger = np.sqrt(max(1.0 - bc_sum, 0.0)) / np.sqrt(2)
            metrics['type_distribution_hellinger'] = float(hellinger)

            # Chi-square statistic: sum((pi - qi)^2 / qi)
            mask = real_vec > 0
            if mask.any():
                chi2 = float(np.sum((sim_vec[mask] - real_vec[mask]) ** 2 / real_vec[mask]))
            else:
                chi2 = 0.0
            metrics['type_distribution_chi_square'] = chi2

            # Total Variation Distance: 0.5 * sum(|pi - qi|)
            tv = 0.5 * float(np.sum(np.abs(sim_vec - real_vec)))
            metrics['type_distribution_tv_distance'] = tv

            # Cosine similarity between distribution vectors
            cosine = calculate_cosine_similarity_vec(sim_vec, real_vec)
            metrics['type_distribution_cosine'] = float(cosine)

            # Overlap coefficient: sum(min(pi, qi))
            overlap = float(np.sum(np.minimum(sim_vec, real_vec)))
            metrics['type_distribution_overlap'] = overlap

            # Bhattacharyya coefficient: sum(sqrt(pi * qi))
            bhattacharyya = float(bc_sum)
            metrics['type_distribution_bhattacharyya'] = bhattacharyya

            print(f"      Hellinger: {hellinger:.4f}, TV: {tv:.4f}, "
                  f"Cosine: {cosine:.4f}, Overlap: {overlap:.4f}, "
                  f"Bhattacharyya: {bhattacharyya:.4f}, Chi2: {chi2:.4f}")
        else:
            # ---- Figure: Sim bar counts (no real data) ----
            fig, ax = plt.subplots(figsize=FIG_SIZE)
            types = [t for t in STANDARD_TYPE_ORDER if sim_types.get(t, 0) > 0]
            counts = [sim_types[t] for t in types]
            colors = [STANDARD_TYPE_COLORS.get(t, '#888') for t in types]
            ax.bar(types, counts, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
            for i, (t, c) in enumerate(zip(types, counts)):
                ax.text(i, c + max(counts) * 0.01, str(c), ha='center', va='bottom', fontsize=9)
            ax.set_ylabel('Count')
            ax.set_title('Simulation Behavior Counts', fontweight='bold')
            add_grid(ax)
            save_figure(fig, self.output_dir / 'behavior_type_counts_sim.png')

        # 详细行为类型分布图（统一英文类型名）
        sim_ordered = [t for t, _ in sim_detailed.most_common()]
        real_only = sorted(k for k in real_detailed if k not in sim_detailed)
        all_detail_types = sim_ordered + real_only

        # ---- Figure: Sim detailed bar ----
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        vals_d = [sim_detailed.get(t, 0) for t in all_detail_types]
        bars = ax.barh(all_detail_types, vals_d, color=C_SIM['total'], alpha=0.85,
                       edgecolor='black', linewidth=0.5)
        for bar, val in zip(bars, vals_d):
            if val > 0:
                ax.text(bar.get_width() + max(vals_d) * 0.02, bar.get_y() + bar.get_height() / 2,
                        str(val), ha='left', va='center', fontsize=FONT_SIZE['annotation'])
        ax.set_xlabel('Count')
        ax.set_title('Simulation Detailed Behavior Types', fontweight='bold')
        add_grid(ax)
        save_figure(fig, self.output_dir / 'behavior_type_detailed_sim.png')

        if has_real:
            # ---- Figure: Real detailed bar ----
            fig, ax = plt.subplots(figsize=FIG_SIZE)
            vals_r = [real_detailed.get(t, 0) for t in all_detail_types]
            bars = ax.barh(all_detail_types, vals_r, color=C_REAL['total'], alpha=0.85,
                           edgecolor='black', linewidth=0.5)
            for bar, val in zip(bars, vals_r):
                if val > 0:
                    ax.text(bar.get_width() + max(max(vals_r), 1) * 0.02,
                            bar.get_y() + bar.get_height() / 2,
                            str(val), ha='left', va='center', fontsize=FONT_SIZE['annotation'])
            ax.set_xlabel('Count')
            ax.set_title('Real Detailed Behavior Types', fontweight='bold')
            add_grid(ax)
            save_figure(fig, self.output_dir / 'behavior_type_detailed_real.png')

        return metrics

    # ─────────────────────────────────────────────────────────
    # 2. 行为类型时间演化
    # ─────────────────────────────────────────────────────────
    def _analyze_type_evolution(self, sim_actions, sim_times, real_actions, real_times, has_real):
        metrics = {}
        std_types = STANDARD_TYPE_ORDER

        sim_type_by_time = defaultdict(lambda: Counter())
        for a in sim_actions:
            t = a.get('time')
            std = self._get_standard_type(a, 'sim')
            if t and std:
                sim_type_by_time[t][std] += 1

        if not sim_times or not sim_type_by_time:
            return metrics

        sim_series = {}
        for st in std_types:
            sim_series[st] = [sim_type_by_time[t].get(st, 0) for t in sim_times]

        # ---- Figure: Sim stackplot ----
        fig, ax = plt.subplots(figsize=FIG_SIZE_WIDE)
        stack_data = [smooth(sim_series.get(st, [0] * len(sim_times)), target_len=len(sim_times))
                      for st in std_types]
        colors = [STANDARD_TYPE_COLORS.get(st, '#888') for st in std_types]
        ax.stackplot(sim_times, stack_data, labels=std_types, colors=colors, alpha=0.8)
        ax.set_ylabel('Count')
        ax.set_title('Simulation Behavior Type Evolution', fontweight='bold')
        setup_time_axis(ax)
        add_legend(ax, loc='upper right')
        add_grid(ax)
        save_figure(fig, self.output_dir / 'behavior_type_evolution_sim.png')

        real_series = {}
        if has_real:
            real_type_by_time = defaultdict(lambda: Counter())
            for a in real_actions:
                t = a.get('time')
                std = self._get_standard_type(a, 'real')
                if t and std:
                    real_type_by_time[t][std] += 1
            for st in std_types:
                real_series[st] = [real_type_by_time[t].get(st, 0) for t in real_times]

            # ---- Figure: Real stackplot ----
            fig, ax = plt.subplots(figsize=FIG_SIZE_WIDE)
            stack_r = [smooth(real_series.get(st, [0] * len(real_times)), target_len=len(real_times))
                       for st in std_types]
            ax.stackplot(real_times, stack_r, labels=std_types, colors=colors, alpha=0.8)
            ax.set_ylabel('Count')
            ax.set_xlabel('Time')
            ax.set_title('Real Behavior Type Evolution', fontweight='bold')
            setup_time_axis(ax)
            add_legend(ax, loc='upper right')
            add_grid(ax)
            save_figure(fig, self.output_dir / 'behavior_type_evolution_real.png')

            metrics['per_type_similarity'] = {}
            for st in std_types:
                sc = normalize(sim_series.get(st, [])).tolist() if sim_series.get(st) else []
                rc = normalize(real_series.get(st, [])).tolist() if real_series.get(st) else []
                if sc and rc:
                    metrics['per_type_similarity'][st] = calculate_curve_similarity(sc, rc)
                    print(f"      {st} 归一化曲线Pearson: {metrics['per_type_similarity'][st].get('pearson', 0):.4f}")

        # 归一化比例对比 — each type as its own figure
        for st in std_types:
            fig, ax = plt.subplots(figsize=FIG_SIZE_WIDE)
            c = STANDARD_TYPE_COLORS.get(st, '#888')
            sc = sim_series.get(st, [])
            if sc:
                ax.plot(sim_times[:len(sc)], smooth(normalize(sc), target_len=len(sc)),
                        color=c, lw=LW, label=f'Sim-{st}')
            if has_real:
                rc = real_series.get(st, [])
                if rc:
                    ax.plot(real_times[:len(rc)], smooth(normalize(rc), target_len=len(rc)),
                            color=c, lw=LW, ls='--', alpha=0.7, label=f'Real-{st}')
            ax.set_ylabel('Normalized')
            ax.set_xlabel('Time')
            ax.set_title(f'{st.capitalize()} Activity (Normalized)', fontweight='bold')
            setup_time_axis(ax)
            add_legend(ax)
            add_grid(ax)
            save_figure(fig, self.output_dir / f'behavior_type_normalized_{st}.png')

        return metrics

    # ─────────────────────────────────────────────────────────
    # 3. 用户活跃度分布
    # ─────────────────────────────────────────────────────────
    def _analyze_user_activity(self, sim_actions, real_actions, has_real):
        metrics = {}

        sim_user_counts = Counter()
        for a in sim_actions:
            uid = a.get('user_id', '')
            if uid:
                sim_user_counts[uid] += 1
        sim_counts = sorted(sim_user_counts.values(), reverse=True)
        metrics['sim_user_count'] = len(sim_user_counts)
        metrics['sim_total_actions'] = sum(sim_counts)
        metrics['sim_avg_actions_per_user'] = float(np.mean(sim_counts)) if sim_counts else 0
        metrics['sim_std_actions_per_user'] = float(np.std(sim_counts)) if sim_counts else 0
        metrics['sim_max_actions_per_user'] = int(max(sim_counts)) if sim_counts else 0
        metrics['sim_activity_gini'] = float(calculate_gini_coefficient(sim_counts))
        metrics['sim_activity_entropy'] = float(calculate_normalized_entropy(Counter(sim_counts)))

        real_counts = []
        if has_real:
            real_user_counts = Counter()
            for a in real_actions:
                uid = a.get('user_id', '')
                if uid:
                    real_user_counts[uid] += 1
            real_counts = sorted(real_user_counts.values(), reverse=True)
            metrics['real_user_count'] = len(real_user_counts)
            metrics['real_total_actions'] = sum(real_counts)
            metrics['real_avg_actions_per_user'] = float(np.mean(real_counts)) if real_counts else 0
            metrics['real_std_actions_per_user'] = float(np.std(real_counts)) if real_counts else 0
            metrics['real_max_actions_per_user'] = int(max(real_counts)) if real_counts else 0
            metrics['real_activity_gini'] = float(calculate_gini_coefficient(real_counts))
            metrics['real_activity_entropy'] = float(calculate_normalized_entropy(Counter(real_counts)))

        # ---- Figure: Histogram ----
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        max_count = max(sim_counts) if sim_counts else 1
        bins = np.arange(0, min(max_count + 2, 50), 1)
        ax.hist(sim_counts, bins=bins, alpha=0.6, color=C_SIM['total'],
                label='Simulation', density=True, edgecolor='black', linewidth=0.5)
        if has_real and real_counts:
            ax.hist(real_counts, bins=bins, alpha=0.6, color=C_REAL['total'],
                    label='Real Data', density=True, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Actions per User')
        ax.set_ylabel('Density')
        ax.set_title('User Activity Distribution', fontweight='bold')
        add_legend(ax)
        add_grid(ax)
        save_figure(fig, self.output_dir / 'user_activity_histogram.png')

        # ---- Figure: CDF ----
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        if sim_counts:
            sim_sorted = np.sort(sim_counts)
            sim_cdf = np.arange(1, len(sim_sorted) + 1) / len(sim_sorted)
            ax.plot(sim_sorted, sim_cdf, color=C_SIM['total'], lw=LW, label='Simulation')
        if has_real and real_counts:
            real_sorted = np.sort(real_counts)
            real_cdf = np.arange(1, len(real_sorted) + 1) / len(real_sorted)
            ax.plot(real_sorted, real_cdf, color=C_REAL['total'], lw=LW, ls='--', label='Real Data')
        ax.set_xlabel('Actions per User')
        ax.set_ylabel('CDF')
        ax.set_title('Cumulative Distribution of User Activity', fontweight='bold')
        add_legend(ax)
        add_grid(ax)
        save_figure(fig, self.output_dir / 'user_activity_cdf.png')

        # ---- Figure: Log-log rank ----
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        if sim_counts:
            ranks_sim = np.arange(1, len(sim_counts) + 1)
            ax.loglog(ranks_sim, sim_counts, 'o-', color=C_SIM['total'],
                      markersize=MARKER_SIZE, lw=LW_MINOR, label='Simulation')
        if has_real and real_counts:
            ranks_real = np.arange(1, len(real_counts) + 1)
            ax.loglog(ranks_real, real_counts, 's--', color=C_REAL['total'],
                      markersize=MARKER_SIZE, lw=LW_MINOR, label='Real Data')
        ax.set_xlabel('Rank')
        ax.set_ylabel('Activity Count')
        ax.set_title('User Activity Rank-Size (Log-Log)', fontweight='bold')
        add_legend(ax)
        add_grid(ax)
        save_figure(fig, self.output_dir / 'user_activity_rank.png')

        # ---- Figure: Metrics bar ----
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        metric_labels = ['Avg\nActions', 'Gini\nCoeff', 'Norm\nEntropy']
        sim_vals = [metrics.get('sim_avg_actions_per_user', 0),
                    metrics.get('sim_activity_gini', 0),
                    metrics.get('sim_activity_entropy', 0)]
        if has_real:
            real_vals = [metrics.get('real_avg_actions_per_user', 0),
                         metrics.get('real_activity_gini', 0),
                         metrics.get('real_activity_entropy', 0)]
            x = np.arange(len(metric_labels))
            width = 0.35
            bars1 = ax.bar(x - width / 2, sim_vals, width, label='Simulation',
                           color=C_SIM['total'], alpha=0.8)
            bars2 = ax.bar(x + width / 2, real_vals, width, label='Real Data',
                           color=C_REAL['total'], alpha=0.8)
            for bar, val in zip(list(bars1) + list(bars2), sim_vals + real_vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=FONT_SIZE['annotation'])
            ax.set_xticks(x)
        else:
            ax.bar(metric_labels, sim_vals, color=C_SIM['total'], alpha=0.8)
            for i, val in enumerate(sim_vals):
                ax.text(i, val + 0.01, f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        ax.set_xticklabels(metric_labels)
        ax.set_ylabel('Value')
        ax.set_title('Activity Distribution Metrics', fontweight='bold')
        add_legend(ax)
        add_grid(ax)
        save_figure(fig, self.output_dir / 'user_activity_metrics.png')

        if has_real and sim_counts and real_counts:
            max_count_all = max(max(sim_counts), max(real_counts))
            bins_edge = np.concatenate([np.arange(0, min(20, max_count_all + 1), 1), [max_count_all + 1]])
            sim_hist, _ = np.histogram(sim_counts, bins=bins_edge, density=True)
            real_hist, _ = np.histogram(real_counts, bins=bins_edge, density=True)
            sim_hist = sim_hist + 1e-10
            real_hist = real_hist + 1e-10
            sim_hist = sim_hist / sim_hist.sum()
            real_hist = real_hist / real_hist.sum()
            jsd = calculate_jsd(sim_hist, real_hist)
            metrics['activity_distribution_jsd'] = float(jsd)
            metrics['activity_distribution_similarity'] = float(1 - jsd)
            metrics['gini_difference'] = float(abs(
                metrics['sim_activity_gini'] - metrics['real_activity_gini']))
            print(f"      活跃度分布JSD: {jsd:.4f}, 相似度: {1 - jsd:.4f}")
            print(f"      基尼系数差异: {metrics['gini_difference']:.4f}")
        return metrics

    # ─────────────────────────────────────────────────────────
    # 4. 行为强度分布
    # ─────────────────────────────────────────────────────────
    def _analyze_behavior_intensity(self, sim_actions, sim_times, real_actions, real_times, has_real):
        metrics = {}
        sim_intensity = Counter()
        for a in sim_actions:
            t = a.get('time')
            if t:
                sim_intensity[t] += 1
        sim_iv = [sim_intensity.get(t, 0) for t in sim_times] if sim_times else list(sim_intensity.values())
        metrics['sim_avg_intensity'] = float(np.mean(sim_iv)) if sim_iv else 0
        metrics['sim_max_intensity'] = int(max(sim_iv)) if sim_iv else 0
        metrics['sim_intensity_std'] = float(np.std(sim_iv)) if sim_iv else 0
        metrics['sim_burstiness'] = float(metrics['sim_max_intensity'] / max(metrics['sim_avg_intensity'], 0.01))

        real_iv = []
        if has_real:
            real_intensity = Counter()
            for a in real_actions:
                t = a.get('time')
                if t:
                    real_intensity[t] += 1
            real_iv = [real_intensity.get(t, 0) for t in real_times] if real_times else list(real_intensity.values())
            metrics['real_avg_intensity'] = float(np.mean(real_iv)) if real_iv else 0
            metrics['real_max_intensity'] = int(max(real_iv)) if real_iv else 0
            metrics['real_intensity_std'] = float(np.std(real_iv)) if real_iv else 0
            metrics['real_burstiness'] = float(metrics['real_max_intensity'] / max(metrics['real_avg_intensity'], 0.01))

        # ---- Figure: Time series ----
        fig, ax = plt.subplots(figsize=FIG_SIZE_WIDE)
        if sim_times and sim_iv:
            ax.plot(sim_times[:len(sim_iv)], smooth(sim_iv, target_len=len(sim_iv)),
                    color=C_SIM['total'], lw=LW, label='Simulation')
        if has_real and real_times and real_iv:
            ax.plot(real_times[:len(real_iv)], smooth(real_iv, target_len=len(real_iv)),
                    color=C_REAL['total'], lw=LW, ls='--', label='Real Data')
        ax.set_xlabel('Time')
        ax.set_ylabel('Actions per Time Window')
        ax.set_title('Behavior Intensity Over Time', fontweight='bold')
        setup_time_axis(ax)
        add_legend(ax)
        add_grid(ax)
        save_figure(fig, self.output_dir / 'behavior_intensity_timeseries.png')

        # ---- Figure: Histogram ----
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        if sim_iv:
            ax.hist(sim_iv, bins=30, alpha=0.6, color=C_SIM['total'],
                    label='Simulation', density=True, edgecolor='black', linewidth=0.5)
        if has_real and real_iv:
            ax.hist(real_iv, bins=30, alpha=0.6, color=C_REAL['total'],
                    label='Real Data', density=True, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Actions per Time Window')
        ax.set_ylabel('Density')
        ax.set_title('Behavior Intensity Distribution', fontweight='bold')
        add_legend(ax)
        add_grid(ax)
        save_figure(fig, self.output_dir / 'behavior_intensity_histogram.png')

        if has_real and sim_iv and real_iv:
            sim_norm = normalize(sim_iv).tolist()
            real_norm = normalize(real_iv).tolist()
            metrics['intensity_curve_similarity'] = calculate_curve_similarity(sim_norm, real_norm)
            print(f"      行为强度曲线Pearson: {metrics['intensity_curve_similarity'].get('pearson', 0):.4f}")

            # Kendall's tau
            min_len = min(len(sim_norm), len(real_norm))
            kendall = calculate_kendall_tau(sim_norm[:min_len], real_norm[:min_len])
            metrics['intensity_kendall_tau'] = kendall
            print(f"      行为强度Kendall tau: {kendall.get('tau', 0):.4f}")

            # Cosine similarity
            cosine = calculate_cosine_similarity_vec(sim_norm[:min_len], real_norm[:min_len])
            metrics['intensity_cosine_similarity'] = float(cosine)
            print(f"      行为强度Cosine: {cosine:.4f}")

            # MAPE (Mean Absolute Percentage Error)
            s_arr = np.array(sim_norm[:min_len], dtype=float)
            r_arr = np.array(real_norm[:min_len], dtype=float)
            mape_mask = np.abs(r_arr) > 1e-10
            if mape_mask.any():
                mape = float(np.mean(np.abs((s_arr[mape_mask] - r_arr[mape_mask]) / r_arr[mape_mask])))
            else:
                mape = 0.0
            metrics['intensity_mape'] = mape
            print(f"      行为强度MAPE: {mape:.4f}")

            # R-squared
            ss_res = float(np.sum((s_arr - r_arr) ** 2))
            ss_tot = float(np.sum((r_arr - np.mean(r_arr)) ** 2))
            r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            metrics['intensity_r_squared'] = float(r_squared)
            print(f"      行为强度R²: {r_squared:.4f}")

        return metrics

    # ─────────────────────────────────────────────────────────
    # 5. 用户参与度分层
    # ─────────────────────────────────────────────────────────
    def _analyze_user_engagement(self, sim_actions, real_actions, has_real):
        metrics = {}

        def compute_layers(actions):
            uc = Counter()
            for a in actions:
                uid = a.get('user_id', '')
                if uid:
                    uc[uid] += 1
            if not uc:
                return {}
            su = sorted(uc.items(), key=lambda x: x[1], reverse=True)
            n = len(su)
            total = sum(c for _, c in su)
            t10 = max(1, int(n * 0.1))
            t20 = max(1, int(n * 0.2))
            mid_end = max(t20 + 1, int(n * 0.8))
            return {
                'n_users': n, 'total_actions': total,
                'top_10pct_action_ratio': float(sum(c for _, c in su[:t10]) / max(total, 1)),
                'top_20pct_action_ratio': float(sum(c for _, c in su[:t20]) / max(total, 1)),
                'mid_60pct_action_ratio': float(sum(c for _, c in su[t20:mid_end]) / max(total, 1)),
                'bottom_20pct_action_ratio': float(sum(c for _, c in su[mid_end:]) / max(total, 1)),
                'avg_actions_per_user': float(total / n),
                'median_actions_per_user': float(np.median([c for _, c in su]))
            }

        metrics['sim_engagement'] = compute_layers(sim_actions)
        if has_real:
            metrics['real_engagement'] = compute_layers(real_actions)

        # ---- Figure: Engagement layers bar ----
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        layer_labels = ['Top 10%\nUsers', 'Top 20%\nUsers', 'Mid 60%\nUsers', 'Bottom 20%\nUsers']
        se = metrics.get('sim_engagement', {})
        sv = [se.get('top_10pct_action_ratio', 0), se.get('top_20pct_action_ratio', 0),
              se.get('mid_60pct_action_ratio', 0), se.get('bottom_20pct_action_ratio', 0)]

        if has_real:
            re = metrics.get('real_engagement', {})
            rv = [re.get('top_10pct_action_ratio', 0), re.get('top_20pct_action_ratio', 0),
                  re.get('mid_60pct_action_ratio', 0), re.get('bottom_20pct_action_ratio', 0)]
            x = np.arange(len(layer_labels))
            width = 0.35
            b1 = ax.bar(x - width / 2, sv, width, label='Simulation',
                        color=C_SIM['total'], alpha=0.85, edgecolor='black', linewidth=0.5)
            b2 = ax.bar(x + width / 2, rv, width, label='Real Data',
                        color=C_REAL['total'], alpha=0.85, edgecolor='black', linewidth=0.5)
            for bar, val in zip(list(b1) + list(b2), sv + rv):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{val:.1%}', ha='center', va='bottom', fontsize=FONT_SIZE['annotation'])
            ax.set_xticks(x)
        else:
            ax.bar(layer_labels, sv, color=C_SIM['total'], alpha=0.85, edgecolor='black', linewidth=0.5)
        ax.set_xticklabels(layer_labels)
        ax.set_ylabel('Action Proportion')
        ax.set_title('User Engagement Layers', fontweight='bold')
        add_legend(ax)
        add_grid(ax)
        save_figure(fig, self.output_dir / 'user_engagement_layers.png')

        # ---- Figure: Lorenz curve ----
        fig, ax = plt.subplots(figsize=FIG_SIZE)

        def plot_lorenz(counts, color, label, ls='-'):
            sc = np.sort(counts)
            cum = np.cumsum(sc)
            cum = cum / cum[-1]
            xl = np.arange(1, len(cum) + 1) / len(cum)
            ax.plot(xl, cum, color=color, lw=LW, ls=ls, label=label)

        suc = Counter()
        for a in sim_actions:
            uid = a.get('user_id', '')
            if uid:
                suc[uid] += 1
        if suc:
            plot_lorenz(list(suc.values()), C_SIM['total'], 'Simulation')
        if has_real:
            ruc = Counter()
            for a in real_actions:
                uid = a.get('user_id', '')
                if uid:
                    ruc[uid] += 1
            if ruc:
                plot_lorenz(list(ruc.values()), C_REAL['total'], 'Real Data', '--')
        ax.plot([0, 1], [0, 1], 'k:', lw=1, alpha=0.5, label='Perfect Equality')
        ax.set_xlabel('Cumulative User Proportion')
        ax.set_ylabel('Cumulative Action Proportion')
        ax.set_title('Lorenz Curve (User Activity)', fontweight='bold')
        add_legend(ax)
        add_grid(ax)
        save_figure(fig, self.output_dir / 'user_engagement_lorenz.png')

        if se:
            print(f"      模拟 - Top10%用户贡献: {se.get('top_10pct_action_ratio', 0):.1%}")
        return metrics

    # ─────────────────────────────────────────────────────────
    # 6. 按角色的行为比例分布
    # ─────────────────────────────────────────────────────────
    def _analyze_role_distribution(self, sim_actions, real_actions, has_real):
        metrics = {}

        sim_roles = Counter()
        for a in sim_actions:
            role = a.get('agent_type', 'citizen')
            sim_roles[role] += 1
        sim_total = max(sum(sim_roles.values()), 1)
        metrics['sim_role_counts'] = dict(sim_roles)
        metrics['sim_role_ratios'] = {r: float(v / sim_total) for r, v in sim_roles.items()}

        real_roles = Counter()
        if has_real:
            for a in real_actions:
                role = a.get('agent_type', a.get('user_type', 'citizen'))
                real_roles[role] += 1
            real_total = max(sum(real_roles.values()), 1)
            metrics['real_role_counts'] = dict(real_roles)
            metrics['real_role_ratios'] = {r: float(v / real_total) for r, v in real_roles.items()}

        all_roles = ROLE_ORDER
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(all_roles))
        width = 0.35

        sv = [sim_roles.get(r, 0) / sim_total for r in all_roles]
        b1 = ax.bar(x - (width / 2 if has_real else 0), sv, width,
                    label='Simulation', color=C_SIM['total'], alpha=0.85, edgecolor='black', linewidth=0.5)
        if has_real:
            rv = [real_roles.get(r, 0) / real_total for r in all_roles]
            b2 = ax.bar(x + width / 2, rv, width,
                        label='Real Data', color=C_REAL['total'], alpha=0.85, edgecolor='black', linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(all_roles)
        ax.set_ylabel('Proportion')
        ax.set_title('Behavior Distribution by Role', fontweight='bold')
        add_legend(ax)
        add_grid(ax)
        save_figure(fig, self.output_dir / 'role_distribution.png')

        if has_real:
            sv_arr = np.array([sim_roles.get(r, 0) for r in all_roles], dtype=float)
            rv_arr = np.array([real_roles.get(r, 0) for r in all_roles], dtype=float)
            if sv_arr.sum() > 0:
                sv_arr = sv_arr / sv_arr.sum()
            if rv_arr.sum() > 0:
                rv_arr = rv_arr / rv_arr.sum()
            jsd = calculate_jsd(sv_arr, rv_arr)
            metrics['role_distribution_jsd'] = float(jsd)
            metrics['role_distribution_similarity'] = float(1 - jsd)
            print(f"      角色分布JSD: {jsd:.4f}, 相似度: {1 - jsd:.4f}")
        return metrics

    # ─────────────────────────────────────────────────────────
    # 7. 评论-原创比 (C/O Ratio) 时序
    # ─────────────────────────────────────────────────────────
    def _analyze_co_ratio(self, sim_actions, sim_times, real_actions, real_times, has_real):
        metrics = {}

        def _compute_co_series(actions, times, source):
            orig_by_t = Counter()
            comm_by_t = Counter()
            for a in actions:
                t = a.get('time')
                std = self._get_standard_type(a, source)
                if t and std == 'original':
                    orig_by_t[t] += 1
                elif t and std == 'comment':
                    comm_by_t[t] += 1
            ratio = []
            for t in times:
                o = orig_by_t.get(t, 0)
                c = comm_by_t.get(t, 0)
                ratio.append(c / max(o, 1))
            return ratio

        sim_co = _compute_co_series(sim_actions, sim_times, 'sim') if sim_times else []
        metrics['sim_avg_co_ratio'] = float(np.mean(sim_co)) if sim_co else 0

        fig, ax = plt.subplots(figsize=FIG_SIZE)
        if sim_co:
            ax.plot(sim_times[:len(sim_co)], smooth(sim_co, target_len=len(sim_co)),
                    color=C_SIM['total'], lw=LW, label='Simulation')
        if has_real and real_times:
            real_co = _compute_co_series(real_actions, real_times, 'real')
            metrics['real_avg_co_ratio'] = float(np.mean(real_co)) if real_co else 0
            ax.plot(real_times[:len(real_co)], smooth(real_co, target_len=len(real_co)),
                    color=C_REAL['total'], lw=LW, ls='--', label='Real Data')
            if sim_co and real_co:
                metrics['co_ratio_similarity'] = calculate_curve_similarity(
                    normalize(sim_co).tolist(), normalize(real_co).tolist())
                print(f"      C/O Ratio Pearson: {metrics['co_ratio_similarity'].get('pearson', 0):.4f}")
        ax.set_xlabel('Time')
        ax.set_ylabel('Comment / Original Ratio')
        ax.set_title('Comment-to-Original Ratio Over Time', fontweight='bold')
        setup_time_axis(ax)
        add_legend(ax)
        add_grid(ax)
        save_figure(fig, self.output_dir / 'co_ratio.png')
        return metrics

    # ─────────────────────────────────────────────────────────
    # 8. 首次发言时间分布
    # ─────────────────────────────────────────────────────────
    def _analyze_first_speak(self, sim_actions, sim_times, real_actions, real_times, has_real):
        metrics = {}

        def _first_speak_positions(actions, times):
            if not times:
                return []
            t_min = min(times)
            t_max = max(times)
            span = (t_max - t_min).total_seconds()
            if span <= 0:
                return []
            first = {}
            for a in actions:
                uid = a.get('user_id', '')
                t = a.get('time')
                if uid and t:
                    if uid not in first or t < first[uid]:
                        first[uid] = t
            positions = [(ft - t_min).total_seconds() / span for ft in first.values()]
            return sorted(positions)

        sim_pos = _first_speak_positions(sim_actions, sim_times)
        metrics['sim_first_speak_count'] = len(sim_pos)

        fig, ax = plt.subplots(figsize=FIG_SIZE)
        if sim_pos:
            cdf_y = np.arange(1, len(sim_pos) + 1) / len(sim_pos)
            ax.plot(sim_pos, cdf_y, color=C_SIM['total'], lw=LW, label='Simulation')

        if has_real and real_times:
            real_pos = _first_speak_positions(real_actions, real_times)
            metrics['real_first_speak_count'] = len(real_pos)
            if real_pos:
                cdf_y_r = np.arange(1, len(real_pos) + 1) / len(real_pos)
                ax.plot(real_pos, cdf_y_r, color=C_REAL['total'], lw=LW, ls='--', label='Real Data')

            if sim_pos and real_pos:
                ks = calculate_ks_test(sim_pos, real_pos)
                metrics['first_speak_ks'] = ks
                metrics['first_speak_similarity'] = float(1 - ks['statistic'])
                print(f"      首次发言KS统计量: {ks['statistic']:.4f}, 相似度: {1 - ks['statistic']:.4f}")

        ax.set_xlabel('Relative Position in Timeline (0=Start, 1=End)')
        ax.set_ylabel('CDF')
        ax.set_title('First-Speak Time CDF', fontweight='bold')
        add_legend(ax)
        add_grid(ax)
        save_figure(fig, self.output_dir / 'first_speak_cdf.png')
        return metrics

    # ─────────────────────────────────────────────────────────
    # 9. 舆论阵营转移率 (Opinion Shift Rate)
    # ─────────────────────────────────────────────────────────
    def _analyze_opinion_shift(self, sim_actions, real_actions, has_real):
        metrics = {}

        def _compute_shifts(actions):
            user_stances = defaultdict(list)
            for a in actions:
                uid = a.get('user_id', '')
                t = a.get('time')
                stance_raw = a.get('stance', a.get('expression_strategy', {}).get('stance', 'Neutral'))
                if isinstance(stance_raw, (int, float)):
                    if stance_raw > 0.3:
                        stance = 'Support'
                    elif stance_raw < -0.3:
                        stance = 'Oppose'
                    else:
                        stance = 'Neutral'
                else:
                    stance = str(stance_raw) if stance_raw else 'Neutral'
                    if stance not in STANCE_LABELS:
                        stance = 'Neutral'
                if uid and t:
                    user_stances[uid].append((t, stance))

            transitions = Counter()
            shifted_users = 0
            total_users = len(user_stances)
            for uid, records in user_stances.items():
                records.sort(key=lambda x: x[0])
                stances_seq = [r[1] for r in records]
                unique_stances = list(dict.fromkeys(stances_seq))
                if len(unique_stances) > 1:
                    shifted_users += 1
                for i in range(len(stances_seq) - 1):
                    if stances_seq[i] != stances_seq[i + 1]:
                        transitions[(stances_seq[i], stances_seq[i + 1])] += 1

            shift_rate = shifted_users / max(total_users, 1)
            return {
                'total_users': total_users,
                'shifted_users': shifted_users,
                'shift_rate': float(shift_rate),
                'transitions': {f'{k[0]}->{k[1]}': v for k, v in transitions.most_common(20)}
            }

        metrics['sim_shift'] = _compute_shifts(sim_actions)
        print(f"      模拟阵营转移率: {metrics['sim_shift']['shift_rate']:.2%}")

        if has_real:
            metrics['real_shift'] = _compute_shifts(real_actions)
            metrics['shift_rate_diff'] = float(abs(
                metrics['sim_shift']['shift_rate'] - metrics['real_shift']['shift_rate']))
            print(f"      真实阵营转移率: {metrics['real_shift']['shift_rate']:.2%}")

        def _plot_transition_matrix(ax, shift_data, title):
            matrix = np.zeros((len(STANCE_LABELS), len(STANCE_LABELS)))
            trans = shift_data.get('transitions', {})
            for key, count in trans.items():
                parts = key.split('->')
                if len(parts) == 2:
                    fr, to = parts
                    if fr in STANCE_LABELS and to in STANCE_LABELS:
                        i = STANCE_LABELS.index(fr)
                        j = STANCE_LABELS.index(to)
                        matrix[i, j] = count
            row_sums = matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            norm_matrix = matrix / row_sums
            im = ax.imshow(norm_matrix, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
            ax.set_xticks(range(len(STANCE_LABELS)))
            ax.set_yticks(range(len(STANCE_LABELS)))
            ax.set_xticklabels(STANCE_LABELS)
            ax.set_yticklabels(STANCE_LABELS)
            ax.set_xlabel('To')
            ax.set_ylabel('From')
            ax.set_title(title, fontweight='bold')
            for i in range(len(STANCE_LABELS)):
                for j in range(len(STANCE_LABELS)):
                    val = norm_matrix[i, j]
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            fontsize=10, color='white' if val > 0.5 else 'black')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # ---- Figure: Sim transition matrix ----
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        _plot_transition_matrix(ax, metrics['sim_shift'], 'Sim Stance Transition')
        save_figure(fig, self.output_dir / 'opinion_shift_matrix_sim.png')

        if has_real:
            # ---- Figure: Real transition matrix ----
            fig, ax = plt.subplots(figsize=FIG_SIZE)
            _plot_transition_matrix(ax, metrics['real_shift'], 'Real Stance Transition')
            save_figure(fig, self.output_dir / 'opinion_shift_matrix_real.png')

        return metrics

    # ─────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────
    def _print_summary(self, results: Dict):
        td = results.get('type_distribution', {})
        if td and 'type_distribution_similarity' in td:
            print(f"    行为类型分布相似度: {td['type_distribution_similarity']:.4f}")
        ua = results.get('user_activity', {})
        if ua and 'activity_distribution_similarity' in ua:
            print(f"    活跃度分布相似度: {ua['activity_distribution_similarity']:.4f}")
        bi = results.get('behavior_intensity', {})
        if bi and bi.get('intensity_curve_similarity'):
            print(f"    行为强度Pearson: {bi['intensity_curve_similarity'].get('pearson', 0):.4f}")
        cr = results.get('co_ratio', {})
        if cr and cr.get('co_ratio_similarity'):
            print(f"    C/O Ratio Pearson: {cr['co_ratio_similarity'].get('pearson', 0):.4f}")
        fs = results.get('first_speak', {})
        if fs and 'first_speak_similarity' in fs:
            print(f"    首次发言分布相似度: {fs['first_speak_similarity']:.4f}")
        os_data = results.get('opinion_shift', {})
        if os_data and os_data.get('sim_shift'):
            print(f"    模拟阵营转移率: {os_data['sim_shift']['shift_rate']:.2%}")
