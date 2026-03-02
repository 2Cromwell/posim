# -*- coding: utf-8 -*-
"""
传播结构机制验证

验证内容:
1. 幂律度分布分析（核心）
2. 同质性与回声室效应
3. 级联效应分析
4. 小世界现象检测
5. 无标度网络属性
"""
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, Counter

from ..base import BaseEvaluator
from ..utils import (
    save_json, parse_time, truncate_time, smooth, normalize,
    calculate_gini_coefficient, calculate_jsd, fit_power_law_exponent
)
from ..visualization import (
    FIG_SIZE, FIG_SIZE_WIDE, FIG_SIZE_SQUARE, LW, LW_MINOR, ALPHA,
    MARKER_SIZE, FONT_SIZE,
    C_SIM, C_REAL,
    setup_time_axis, add_grid, add_legend, save_figure
)

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

logger = logging.getLogger(__name__)

# Palette for standalone mechanism plots
_C_PRIMARY = '#4C72B0'
_C_SECONDARY = '#DD8452'
_C_TERTIARY = '#55A868'
_C_ACCENT = '#C44E52'
_C_MUTED = '#8C8C8C'
_C_FIT = '#937860'


class PropagationStructureEvaluator(BaseEvaluator):
    """传播结构机制验证"""

    def __init__(self, output_dir: Path):
        super().__init__(output_dir / "propagation_structure", name="propagation_structure")

    # ──────────────────────────────────────────────────────────
    # Public entry
    # ──────────────────────────────────────────────────────────
    def evaluate(self, sim_data: Dict[str, Any], real_data: Optional[Dict[str, Any]] = None,
                 **kwargs) -> Dict[str, Any]:
        self._log_section("传播结构机制验证")

        micro_results = sim_data.get('micro_results', [])
        if not micro_results:
            print("    ⚠️ 无模拟数据，跳过传播结构验证")
            return {}

        print("    构建交互网络...")
        G_sim = self._build_interaction_network(micro_results)
        node_count = G_sim.number_of_nodes() if HAS_NX else len(G_sim.get('nodes', set()))
        edge_count = G_sim.number_of_edges() if HAS_NX else len(G_sim.get('edges', []))
        print(f"      模拟网络: {node_count} 节点, {edge_count} 边")

        G_real = None
        base_data_path = kwargs.get('base_data_path')
        if base_data_path and Path(base_data_path).exists():
            print(f"    从base_data.json构建真实网络...")
            G_real = self._build_real_network(base_data_path)
            if HAS_NX:
                print(f"      真实网络: {G_real.number_of_nodes()} 节点, {G_real.number_of_edges()} 边")

        results: Dict[str, Any] = {}

        print("    [1/5] 幂律度分布分析...")
        results['power_law'] = self._analyze_power_law(G_sim, G_real)

        print("    [2/5] 同质性与回声室效应...")
        results['echo_chamber'] = self._analyze_echo_chamber(G_sim, micro_results)

        print("    [3/5] 级联效应分析...")
        results['cascade'] = self._analyze_cascade(micro_results)

        print("    [4/5] 小世界现象检测...")
        results['small_world'] = self._analyze_small_world(G_sim)

        print("    [5/5] 无标度网络属性...")
        results['scale_free'] = self._analyze_scale_free(G_sim)

        self._save_results(results, "propagation_structure_metrics.json")
        self._print_summary(results)
        return results

    # ──────────────────────────────────────────────────────────
    # Network builders
    # ──────────────────────────────────────────────────────────
    def _build_interaction_network(self, micro_results: List[Dict]):
        """Build directed interaction graph from micro_results."""
        if not HAS_NX:
            return self._build_simple_network(micro_results)

        G = nx.DiGraph()
        for a in micro_results:
            uid = a.get('user_id', '')
            if uid:
                G.add_node(uid)

        for a in micro_results:
            action_type = a.get('action_type', '')
            if action_type in ['repost', 'repost_comment', 'short_comment', 'long_comment']:
                src = a.get('user_id', '')
                dst = a.get('target_author', '') or a.get('target_author_id', '')
                if src and dst and src != dst:
                    if G.has_edge(src, dst):
                        G[src][dst]['weight'] = G[src][dst].get('weight', 1) + 1
                    else:
                        G.add_edge(src, dst, weight=1)
        return G

    def _build_simple_network(self, micro_results: List[Dict]) -> Dict:
        nodes = set()
        edges: List[Tuple[str, str]] = []
        for a in micro_results:
            uid = a.get('user_id', '')
            if uid:
                nodes.add(uid)
            action_type = a.get('action_type', '')
            if action_type in ['repost', 'repost_comment', 'short_comment', 'long_comment']:
                src = uid
                dst = a.get('target_author', '') or a.get('target_author_id', '')
                if src and dst and src != dst:
                    edges.append((src, dst))
                    nodes.add(dst)
        return {'nodes': nodes, 'edges': edges, 'type': 'simple'}

    def _build_real_network(self, base_data_path: str):
        """Build real network from base_data.json (same logic as network.py)."""
        import re as _re

        with open(base_data_path, 'r', encoding='utf-8') as f:
            base_data = json.load(f)

        if not HAS_NX:
            return self._build_simple_real_network(base_data)

        G = nx.DiGraph()
        username_to_uid: Dict[str, str] = {}
        for u in base_data:
            ui = u.get('user_info', {})
            uname = ui.get('username', '')
            uid = ui.get('user_id', '')
            if uname and uid:
                username_to_uid[uname] = uid

        for u in base_data:
            uid = u.get('user_info', {}).get('user_id', '')
            if not uid:
                continue
            G.add_node(uid)

            for r in u.get('repost_posts', []):
                target_uid = username_to_uid.get(r.get('root_author', ''), '')
                if target_uid and target_uid != uid:
                    G.add_edge(uid, target_uid)
                for chain_item in r.get('repost_chain', []):
                    chain_uid = username_to_uid.get(chain_item.get('author', ''), '')
                    if chain_uid and chain_uid != uid:
                        G.add_edge(uid, chain_uid)

            for c in u.get('comments', []):
                replied_to = c.get('replied_to_user', '')
                if replied_to:
                    target_uid = username_to_uid.get(replied_to, '')
                    if target_uid and target_uid != uid:
                        G.add_edge(uid, target_uid)
                orig_author = c.get('original_post_author', '')
                if orig_author:
                    target_uid = username_to_uid.get(orig_author, '')
                    if target_uid and target_uid != uid:
                        G.add_edge(uid, target_uid)
        return G

    def _build_simple_real_network(self, base_data: List[Dict]) -> Dict:
        nodes = set()
        edges: List[Tuple[str, str]] = []
        username_to_uid: Dict[str, str] = {}
        for u in base_data:
            ui = u.get('user_info', {})
            uname, uid = ui.get('username', ''), ui.get('user_id', '')
            if uname and uid:
                username_to_uid[uname] = uid
        for u in base_data:
            uid = u.get('user_info', {}).get('user_id', '')
            if not uid:
                continue
            nodes.add(uid)
            for r in u.get('repost_posts', []):
                t = username_to_uid.get(r.get('root_author', ''), '')
                if t and t != uid:
                    edges.append((uid, t)); nodes.add(t)
            for c in u.get('comments', []):
                for field in ('replied_to_user', 'original_post_author'):
                    t = username_to_uid.get(c.get(field, ''), '')
                    if t and t != uid:
                        edges.append((uid, t)); nodes.add(t)
        return {'nodes': nodes, 'edges': edges, 'type': 'simple'}

    # ──────────────────────────────────────────────────────────
    # Helpers: extract degree arrays from nx or simple graph
    # ──────────────────────────────────────────────────────────
    def _get_degrees(self, G) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (in_degrees, out_degrees, total_degrees) arrays."""
        if HAS_NX and isinstance(G, (nx.DiGraph, nx.Graph)):
            in_deg = np.array([d for _, d in G.in_degree()], dtype=float)
            out_deg = np.array([d for _, d in G.out_degree()], dtype=float)
            total_deg = np.array([d for _, d in G.degree()], dtype=float)
        else:
            in_cnt: Counter = Counter()
            out_cnt: Counter = Counter()
            nodes = G.get('nodes', set())
            for s, t in G.get('edges', []):
                out_cnt[s] += 1
                in_cnt[t] += 1
            in_deg = np.array([in_cnt.get(n, 0) for n in nodes], dtype=float)
            out_deg = np.array([out_cnt.get(n, 0) for n in nodes], dtype=float)
            total_deg = in_deg + out_deg
        return in_deg, out_deg, total_deg

    # ──────────────────────────────────────────────────────────
    # 1. Power-law distribution analysis
    # ──────────────────────────────────────────────────────────
    def _analyze_power_law(self, G_sim, G_real) -> Dict[str, Any]:
        in_deg, out_deg, total_deg = self._get_degrees(G_sim)
        if len(total_deg) == 0:
            return {}

        metrics: Dict[str, Any] = {}

        total_exp = fit_power_law_exponent(total_deg)
        in_exp = fit_power_law_exponent(in_deg)
        out_exp = fit_power_law_exponent(out_deg)
        metrics['exponent'] = total_exp
        metrics['in_degree_exponent'] = in_exp
        metrics['out_degree_exponent'] = out_exp

        ks_stat, ks_p = self._ks_power_law_test(total_deg)
        metrics['ks_statistic'] = ks_stat
        metrics['p_value'] = ks_p

        # ── Plot: degree_distribution_loglog ──
        self._plot_degree_loglog(total_deg, in_deg, out_deg)

        # ── Plot: degree_ccdf ──
        self._plot_degree_ccdf(total_deg)

        # ── Plot: power_law_fit ──
        self._plot_power_law_fit(total_deg, total_exp)

        # ── Plot: comparison if real data ──
        if G_real is not None:
            r_in, r_out, r_total = self._get_degrees(G_real)
            if len(r_total) > 0:
                self._plot_degree_comparison(total_deg, r_total)
                metrics['real_exponent'] = fit_power_law_exponent(r_total)

        return metrics

    @staticmethod
    def _ks_power_law_test(degrees: np.ndarray) -> Tuple[float, float]:
        """KS goodness-of-fit against theoretical power-law CDF."""
        from scipy.stats import kstest
        arr = degrees[degrees >= 1]
        if len(arr) < 5:
            return 1.0, 0.0
        x_min = arr.min()
        n = len(arr)
        alpha = 1 + n / np.sum(np.log(arr / x_min))

        def power_law_cdf(x):
            return 1 - (x_min / x) ** (alpha - 1)

        try:
            stat, p = kstest(arr, power_law_cdf)
            return float(stat), float(p)
        except Exception:
            return 1.0, 0.0

    def _plot_degree_loglog(self, total_deg, in_deg, out_deg):
        """Log-log scatter of degree distributions, x limited to 0-100."""
        fig, ax = plt.subplots(figsize=FIG_SIZE)

        for deg_arr, label, color, marker in [
            (total_deg, 'Total', _C_PRIMARY, 'o'),
            (in_deg, 'In-degree', _C_SECONDARY, 's'),
            (out_deg, 'Out-degree', _C_TERTIARY, '^'),
        ]:
            counts = Counter(deg_arr.astype(int))
            xs = sorted(counts.keys())
            ys = [counts[x] for x in xs]
            xs_f = [x for x in xs if 0 < x <= 100]
            ys_f = [counts[x] for x in xs_f]
            if xs_f:
                ax.scatter(xs_f, ys_f, s=MARKER_SIZE * 8, alpha=0.7,
                           label=label, color=color, marker=marker, edgecolors='white', linewidths=0.3)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(0.8, 110)
        ax.set_xlabel('Degree (k)', fontsize=FONT_SIZE['label'])
        ax.set_ylabel('Frequency', fontsize=FONT_SIZE['label'])
        ax.set_title('Degree Distribution (Log-Log)', fontweight='bold')
        add_legend(ax)
        add_grid(ax, alpha=0.25)
        save_figure(fig, self.output_dir / 'degree_distribution_loglog.png')
        logger.info("[SAVED] degree_distribution_loglog.png")

    def _plot_degree_ccdf(self, total_deg):
        """Complementary CDF on log-log scale."""
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        sorted_deg = np.sort(total_deg[total_deg >= 1])
        n = len(sorted_deg)
        if n == 0:
            save_figure(fig, self.output_dir / 'degree_ccdf.png')
            return
        ccdf_y = np.arange(n, 0, -1) / n
        ax.step(sorted_deg, ccdf_y, where='post', color=_C_PRIMARY, lw=LW, label='Empirical CCDF')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Degree (k)', fontsize=FONT_SIZE['label'])
        ax.set_ylabel('P(X ≥ k)', fontsize=FONT_SIZE['label'])
        ax.set_title('Complementary CDF of Degree Distribution', fontweight='bold')
        add_legend(ax)
        add_grid(ax, alpha=0.25)
        save_figure(fig, self.output_dir / 'degree_ccdf.png')
        logger.info("[SAVED] degree_ccdf.png")

    def _plot_power_law_fit(self, total_deg, alpha):
        """Show fitted power-law line overlaid on the empirical distribution."""
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        arr = total_deg[total_deg >= 1].astype(int)
        if len(arr) == 0:
            save_figure(fig, self.output_dir / 'power_law_fit.png')
            return

        counts = Counter(arr)
        xs = np.array(sorted(counts.keys()), dtype=float)
        ys = np.array([counts[int(x)] for x in xs], dtype=float)
        ys_norm = ys / ys.sum()

        ax.scatter(xs, ys_norm, s=MARKER_SIZE * 8, color=_C_PRIMARY,
                   alpha=0.7, zorder=3, label='Empirical', edgecolors='white', linewidths=0.3)

        if alpha > 1:
            x_min = xs.min()
            x_fit = np.logspace(np.log10(x_min), np.log10(xs.max()), 200)
            y_fit = (alpha - 1) / x_min * (x_fit / x_min) ** (-alpha)
            y_fit = y_fit / y_fit.sum() * (x_fit[1] - x_fit[0])
            scale = ys_norm[0] / y_fit[0] if y_fit[0] > 0 else 1.0
            y_fit *= scale
            ax.plot(x_fit, y_fit, color=_C_ACCENT, lw=LW, ls='--',
                    label=f'Power-law fit (α={alpha:.2f})', zorder=4)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Degree (k)', fontsize=FONT_SIZE['label'])
        ax.set_ylabel('P(k)', fontsize=FONT_SIZE['label'])
        ax.set_title('Power-Law Fit on Degree Distribution', fontweight='bold')
        add_legend(ax)
        add_grid(ax, alpha=0.25)
        save_figure(fig, self.output_dir / 'power_law_fit.png')
        logger.info("[SAVED] power_law_fit.png")

    def _plot_degree_comparison(self, sim_deg, real_deg):
        """Overlay comparison of sim vs real degree distributions."""
        fig, ax = plt.subplots(figsize=FIG_SIZE)

        for deg_arr, label, color in [
            (sim_deg, 'Simulation', C_SIM['total']),
            (real_deg, 'Real Data', C_REAL['total']),
        ]:
            counts = Counter(deg_arr[deg_arr >= 1].astype(int))
            xs = sorted(counts.keys())
            ys = [counts[x] for x in xs]
            xs_f = [x for x in xs if x <= 100]
            ys_f = [counts[x] for x in xs_f]
            if xs_f:
                total = sum(ys_f)
                ys_n = [y / total for y in ys_f]
                ax.scatter(xs_f, ys_n, s=MARKER_SIZE * 8, alpha=0.7,
                           label=label, color=color, edgecolors='white', linewidths=0.3)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(0.8, 110)
        ax.set_xlabel('Degree (k)', fontsize=FONT_SIZE['label'])
        ax.set_ylabel('P(k)', fontsize=FONT_SIZE['label'])
        ax.set_title('Degree Distribution Comparison (Sim vs Real)', fontweight='bold')
        add_legend(ax)
        add_grid(ax, alpha=0.25)
        save_figure(fig, self.output_dir / 'degree_distribution_comparison.png')
        logger.info("[SAVED] degree_distribution_comparison.png")

    # ──────────────────────────────────────────────────────────
    # 2. Homophily & Echo Chamber
    # ──────────────────────────────────────────────────────────
    def _analyze_echo_chamber(self, G_sim, micro_results: List[Dict]) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}

        user_stance: Dict[str, str] = {}
        user_emotion: Dict[str, str] = {}
        for a in micro_results:
            uid = a.get('user_id', '')
            if not uid:
                continue
            stance = a.get('stance', '')
            emotion = a.get('emotion', '')
            if isinstance(stance, (int, float)):
                if stance > 0.3:
                    user_stance[uid] = 'Support'
                elif stance < -0.3:
                    user_stance[uid] = 'Oppose'
                else:
                    user_stance[uid] = 'Neutral'
            elif isinstance(stance, str) and stance:
                user_stance[uid] = stance
            if isinstance(emotion, str) and emotion:
                user_emotion[uid] = emotion

        edges = []
        if HAS_NX and isinstance(G_sim, (nx.DiGraph, nx.Graph)):
            edges = list(G_sim.edges())
        elif isinstance(G_sim, dict):
            edges = G_sim.get('edges', [])

        stance_ei, stance_internal, stance_external = self._compute_ei_index(edges, user_stance)
        emotion_ei, emotion_internal, emotion_external = self._compute_ei_index(edges, user_emotion)

        metrics['ei_index'] = stance_ei
        metrics['homophily_by_stance'] = self._homophily_by_group(edges, user_stance)
        metrics['homophily_by_emotion'] = self._homophily_by_group(edges, user_emotion)

        self._plot_ei_index(metrics)
        self._plot_interaction_heatmap(edges, user_stance)

        return metrics

    @staticmethod
    def _compute_ei_index(edges, user_attr: Dict[str, str]) -> Tuple[float, int, int]:
        """E-I Index: (External - Internal) / (External + Internal)"""
        internal = 0
        external = 0
        for src, dst in edges:
            g_src = user_attr.get(src)
            g_dst = user_attr.get(dst)
            if g_src is None or g_dst is None:
                continue
            if g_src == g_dst:
                internal += 1
            else:
                external += 1
        total = internal + external
        if total == 0:
            return 0.0, 0, 0
        ei = (external - internal) / total
        return float(ei), internal, external

    @staticmethod
    def _homophily_by_group(edges, user_attr: Dict[str, str]) -> Dict[str, float]:
        """Per-group homophily: fraction of within-group edges for each group."""
        group_internal: Counter = Counter()
        group_total: Counter = Counter()
        for src, dst in edges:
            g_src = user_attr.get(src)
            g_dst = user_attr.get(dst)
            if g_src is None:
                continue
            group_total[g_src] += 1
            if g_src == g_dst:
                group_internal[g_src] += 1
        result = {}
        for g in group_total:
            result[g] = float(group_internal[g] / max(group_total[g], 1))
        return result

    def _plot_ei_index(self, metrics: Dict):
        """Bar chart of E-I index by attribute group."""
        fig, ax = plt.subplots(figsize=FIG_SIZE)

        groups = []
        values = []

        stance_hom = metrics.get('homophily_by_stance', {})
        for g, v in sorted(stance_hom.items()):
            groups.append(f"Stance:{g}")
            ei_g = 1 - 2 * v  # convert homophily ratio to EI-like scale
            values.append(ei_g)

        emotion_hom = metrics.get('homophily_by_emotion', {})
        for g, v in sorted(emotion_hom.items()):
            groups.append(f"Emo:{g}")
            ei_g = 1 - 2 * v
            values.append(ei_g)

        if not groups:
            ax.text(0.5, 0.5, 'No attribute data', ha='center', va='center',
                    fontsize=14, transform=ax.transAxes)
            save_figure(fig, self.output_dir / 'echo_chamber_ei_index.png')
            return

        colors = [_C_ACCENT if v < 0 else _C_TERTIARY for v in values]
        bars = ax.barh(groups, values, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
        for bar, val in zip(bars, values):
            offset = 0.03 if val >= 0 else -0.03
            ha = 'left' if val >= 0 else 'right'
            ax.text(bar.get_width() + offset, bar.get_y() + bar.get_height() / 2,
                    f'{val:+.2f}', ha=ha, va='center', fontsize=FONT_SIZE['annotation'])

        ax.axvline(x=0, color='black', lw=1, ls='-')
        ax.set_xlim(-1.2, 1.2)
        ax.set_xlabel('E-I Index  (←Homophily | Heterophily→)', fontsize=FONT_SIZE['label'])
        ax.set_title('Echo Chamber E-I Index by Group', fontweight='bold')

        overall_ei = metrics.get('ei_index', 0)
        ax.axvline(x=overall_ei, color=_C_MUTED, lw=LW_MINOR, ls='--',
                   label=f'Overall EI={overall_ei:+.2f}')
        add_legend(ax)
        add_grid(ax, alpha=0.2)
        save_figure(fig, self.output_dir / 'echo_chamber_ei_index.png')
        logger.info("[SAVED] echo_chamber_ei_index.png")

    def _plot_interaction_heatmap(self, edges, user_stance: Dict[str, str]):
        """Heatmap of cross-group interaction counts."""
        fig, ax = plt.subplots(figsize=FIG_SIZE_SQUARE)

        all_groups = sorted(set(user_stance.values()))
        if not all_groups:
            ax.text(0.5, 0.5, 'No stance data', ha='center', va='center',
                    fontsize=14, transform=ax.transAxes)
            save_figure(fig, self.output_dir / 'interaction_heatmap.png')
            return

        n = len(all_groups)
        matrix = np.zeros((n, n))
        g2i = {g: i for i, g in enumerate(all_groups)}

        for src, dst in edges:
            gs = user_stance.get(src)
            gd = user_stance.get(dst)
            if gs is not None and gd is not None:
                matrix[g2i[gs]][g2i[gd]] += 1

        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        norm_matrix = matrix / row_sums

        im = ax.imshow(norm_matrix, cmap='YlOrRd', aspect='auto', vmin=0)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(all_groups, fontsize=FONT_SIZE['tick'])
        ax.set_yticklabels(all_groups, fontsize=FONT_SIZE['tick'])
        ax.set_xlabel('Target Stance', fontsize=FONT_SIZE['label'])
        ax.set_ylabel('Source Stance', fontsize=FONT_SIZE['label'])
        ax.set_title('Cross-Group Interaction Heatmap (Row-normalized)', fontweight='bold')

        for i in range(n):
            for j in range(n):
                val = norm_matrix[i, j]
                txt_color = 'white' if val > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        color=txt_color, fontsize=FONT_SIZE['annotation'])

        fig.colorbar(im, ax=ax, shrink=0.8, label='Row-normalized interaction ratio')
        save_figure(fig, self.output_dir / 'interaction_heatmap.png')
        logger.info("[SAVED] interaction_heatmap.png")

    # ──────────────────────────────────────────────────────────
    # 3. Cascade effect analysis
    # ──────────────────────────────────────────────────────────
    def _analyze_cascade(self, micro_results: List[Dict]) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}

        post_to_author: Dict[str, str] = {}
        for a in micro_results:
            if a.get('action_type', '') in ('short_post', 'long_post'):
                pid = a.get('post_id', a.get('target_post_id', ''))
                if pid:
                    post_to_author[pid] = a.get('user_id', '')

        cascades: Dict[str, List[Dict]] = defaultdict(list)
        for a in micro_results:
            atype = a.get('action_type', '')
            if atype in ('repost', 'repost_comment', 'short_comment', 'long_comment'):
                target_pid = a.get('target_post_id', '')
                if target_pid:
                    cascades[target_pid].append(a)

        if not cascades:
            return metrics

        scales = []
        depths = []
        for root_pid, actions in cascades.items():
            scale = len(actions)
            scales.append(scale)
            depth = self._estimate_cascade_depth(actions)
            depths.append(depth)

        metrics['cascade_count'] = len(scales)
        metrics['avg_scale'] = float(np.mean(scales))
        metrics['max_scale'] = int(max(scales))
        metrics['avg_depth'] = float(np.mean(depths))
        metrics['max_depth'] = int(max(depths))

        if len(scales) >= 5:
            metrics['scale_power_law_exp'] = fit_power_law_exponent(scales)
        else:
            metrics['scale_power_law_exp'] = 0.0

        self._plot_cascade_size(scales)
        self._plot_cascade_depth(depths)

        return metrics

    @staticmethod
    def _estimate_cascade_depth(actions: List[Dict]) -> int:
        """Estimate cascade depth from sorted action timestamps."""
        times = []
        for a in actions:
            t = parse_time(a.get('time', ''))
            if t:
                times.append(t)
        if not times:
            return 1
        times.sort()
        unique_times = sorted(set(times))
        return min(len(unique_times), len(actions))

    def _plot_cascade_size(self, scales: List[int]):
        """Distribution of cascade sizes."""
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        if not scales:
            save_figure(fig, self.output_dir / 'cascade_size_distribution.png')
            return

        counts = Counter(scales)
        xs = sorted(counts.keys())
        ys = [counts[x] for x in xs]

        ax.scatter(xs, ys, s=MARKER_SIZE * 8, color=_C_PRIMARY, alpha=0.8,
                   zorder=3, edgecolors='white', linewidths=0.3)
        ax.fill_between(xs, 0, ys, alpha=0.15, color=_C_PRIMARY, step='mid')

        if max(scales) > 10:
            ax.set_xscale('log')
            ax.set_yscale('log')

        ax.set_xlabel('Cascade Size', fontsize=FONT_SIZE['label'])
        ax.set_ylabel('Frequency', fontsize=FONT_SIZE['label'])
        ax.set_title(f'Cascade Size Distribution (n={len(scales)})', fontweight='bold')

        avg_s = np.mean(scales)
        ax.axvline(x=avg_s, color=_C_ACCENT, ls='--', lw=LW_MINOR,
                   label=f'Mean={avg_s:.1f}')
        add_legend(ax)
        add_grid(ax, alpha=0.25)
        save_figure(fig, self.output_dir / 'cascade_size_distribution.png')
        logger.info("[SAVED] cascade_size_distribution.png")

    def _plot_cascade_depth(self, depths: List[int]):
        """Distribution of cascade depths."""
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        if not depths:
            save_figure(fig, self.output_dir / 'cascade_depth_distribution.png')
            return

        counts = Counter(depths)
        xs = sorted(counts.keys())
        ys = [counts[x] for x in xs]

        ax.bar(xs, ys, color=_C_SECONDARY, alpha=0.85, edgecolor='black', linewidth=0.5)

        ax.set_xlabel('Cascade Depth', fontsize=FONT_SIZE['label'])
        ax.set_ylabel('Frequency', fontsize=FONT_SIZE['label'])
        ax.set_title(f'Cascade Depth Distribution (n={len(depths)})', fontweight='bold')

        avg_d = np.mean(depths)
        ax.axvline(x=avg_d, color=_C_ACCENT, ls='--', lw=LW_MINOR,
                   label=f'Mean={avg_d:.1f}')
        add_legend(ax)
        add_grid(ax, alpha=0.25)
        save_figure(fig, self.output_dir / 'cascade_depth_distribution.png')
        logger.info("[SAVED] cascade_depth_distribution.png")

    # ──────────────────────────────────────────────────────────
    # 4. Small-world phenomenon
    # ──────────────────────────────────────────────────────────
    def _analyze_small_world(self, G_sim) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {
            'clustering_coefficient': 0.0,
            'avg_path_length': 0.0,
            'random_clustering': 0.0,
            'random_path_length': 0.0,
            'small_world_sigma': 0.0,
        }
        if not HAS_NX or not isinstance(G_sim, (nx.DiGraph, nx.Graph)):
            return metrics

        U = G_sim.to_undirected()
        n = U.number_of_nodes()
        m = U.number_of_edges()
        if n < 4 or m == 0:
            return metrics

        C_real = nx.average_clustering(U)
        metrics['clustering_coefficient'] = float(C_real)

        # Average shortest path on largest connected component (sample if large)
        largest_cc = max(nx.connected_components(U), key=len)
        subG = U.subgraph(largest_cc).copy()
        n_sub = subG.number_of_nodes()

        if n_sub < 2:
            return metrics

        if n_sub <= 500:
            L_real = nx.average_shortest_path_length(subG)
        else:
            sample_nodes = list(subG.nodes())
            rng = np.random.default_rng(42)
            sample = rng.choice(sample_nodes, size=min(200, n_sub), replace=False)
            path_lens = []
            for src in sample:
                lengths = nx.single_source_shortest_path_length(subG, src)
                path_lens.extend(lengths.values())
            L_real = float(np.mean(path_lens)) if path_lens else 0.0
        metrics['avg_path_length'] = float(L_real)

        # Erdos-Renyi random graph comparison
        density = 2 * m / (n * (n - 1)) if n > 1 else 0
        p = density
        C_rand = p if p > 0 else 1e-10
        L_rand = np.log(n) / np.log(max(n * p, 2)) if n * p > 1 else float(n)
        metrics['random_clustering'] = float(C_rand)
        metrics['random_path_length'] = float(L_rand)

        if L_rand > 0 and C_rand > 0:
            gamma = C_real / C_rand
            lam = L_real / L_rand if L_rand > 0 else float('inf')
            sigma = gamma / lam if lam > 0 else 0.0
            metrics['small_world_sigma'] = float(sigma)

        self._plot_small_world(metrics)
        return metrics

    def _plot_small_world(self, metrics: Dict):
        """Bar chart comparing clustering and path length with random graph."""
        fig, ax = plt.subplots(figsize=FIG_SIZE)

        labels = ['Clustering\nCoefficient', 'Avg Path\nLength']
        sim_vals = [metrics['clustering_coefficient'], metrics['avg_path_length']]
        rand_vals = [metrics['random_clustering'], metrics['random_path_length']]

        x = np.arange(len(labels))
        width = 0.32
        bars1 = ax.bar(x - width / 2, sim_vals, width, label='Empirical Network',
                        color=_C_PRIMARY, alpha=0.85, edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width / 2, rand_vals, width, label='ER Random Graph',
                        color=_C_MUTED, alpha=0.65, edgecolor='black', linewidth=0.5)

        for bars in (bars1, bars2):
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, h + max(h * 0.02, 0.005),
                            f'{h:.3f}', ha='center', va='bottom', fontsize=FONT_SIZE['annotation'])

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Value', fontsize=FONT_SIZE['label'])
        sigma = metrics.get('small_world_sigma', 0)
        verdict = 'Yes' if sigma > 1.0 else 'No'
        ax.set_title(f'Small-World Test  (σ = {sigma:.2f}, Small-World: {verdict})', fontweight='bold')
        add_legend(ax)
        add_grid(ax, alpha=0.25)
        save_figure(fig, self.output_dir / 'small_world_comparison.png')
        logger.info("[SAVED] small_world_comparison.png")

    # ──────────────────────────────────────────────────────────
    # 5. Scale-free network properties
    # ──────────────────────────────────────────────────────────
    def _analyze_scale_free(self, G_sim) -> Dict[str, Any]:
        _, _, total_deg = self._get_degrees(G_sim)
        if len(total_deg) == 0:
            return {'degree_entropy': 0.0, 'hub_dominance_top10': 0.0, 'hub_dominance_top5pct': 0.0}

        metrics: Dict[str, Any] = {}

        # Degree distribution entropy
        counts = Counter(total_deg.astype(int))
        total = sum(counts.values())
        probs = np.array([c / total for c in counts.values() if c > 0])
        entropy = -np.sum(probs * np.log2(probs + 1e-15))
        metrics['degree_entropy'] = float(entropy)

        # Hub dominance
        sorted_deg = np.sort(total_deg)[::-1]
        total_degree_sum = sorted_deg.sum()
        if total_degree_sum > 0:
            top10 = sorted_deg[:10].sum() / total_degree_sum
            top5pct_count = max(1, int(len(sorted_deg) * 0.05))
            top5pct = sorted_deg[:top5pct_count].sum() / total_degree_sum
        else:
            top10 = 0.0
            top5pct = 0.0
        metrics['hub_dominance_top10'] = float(top10)
        metrics['hub_dominance_top5pct'] = float(top5pct)

        # Preferential attachment proxy: correlation between existing degree and new edge count
        if HAS_NX and isinstance(G_sim, (nx.DiGraph, nx.Graph)):
            edge_list = list(G_sim.edges())
            if len(edge_list) > 20:
                half = len(edge_list) // 2
                early_edges = edge_list[:half]
                late_edges = edge_list[half:]
                early_deg: Counter = Counter()
                for s, t in early_edges:
                    early_deg[s] += 1
                    early_deg[t] += 1
                late_new: Counter = Counter()
                for s, t in late_edges:
                    late_new[s] += 1
                    late_new[t] += 1
                common_nodes = set(early_deg.keys()) & set(late_new.keys())
                if len(common_nodes) > 5:
                    x_vals = [early_deg[n] for n in common_nodes]
                    y_vals = [late_new[n] for n in common_nodes]
                    corr = np.corrcoef(x_vals, y_vals)[0, 1]
                    metrics['preferential_attachment_corr'] = float(corr) if not np.isnan(corr) else 0.0

        self._plot_hub_dominance(sorted_deg)
        return metrics

    def _plot_hub_dominance(self, sorted_deg: np.ndarray):
        """Pareto-style chart of top nodes' cumulative degree share."""
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        n = len(sorted_deg)
        if n == 0:
            save_figure(fig, self.output_dir / 'hub_dominance.png')
            return

        total = sorted_deg.sum()
        if total == 0:
            save_figure(fig, self.output_dir / 'hub_dominance.png')
            return

        cum_share = np.cumsum(sorted_deg) / total
        x_pct = np.arange(1, n + 1) / n * 100

        ax.fill_between(x_pct, 0, cum_share, alpha=0.2, color=_C_PRIMARY)
        ax.plot(x_pct, cum_share, color=_C_PRIMARY, lw=LW, label='Cumulative degree share')
        ax.plot([0, 100], [0, 1], 'k--', lw=1, alpha=0.5, label='Perfect equality')

        # Mark top-5% and top-10 thresholds
        top5_idx = max(0, int(n * 0.05) - 1)
        top10_idx = min(9, n - 1)
        if top5_idx < n:
            ax.axvline(x=x_pct[top5_idx], color=_C_ACCENT, ls=':', lw=LW_MINOR,
                       label=f'Top 5% → {cum_share[top5_idx]:.1%}')
        if top10_idx < n:
            ax.axhline(y=cum_share[top10_idx], color=_C_SECONDARY, ls=':', lw=LW_MINOR,
                       label=f'Top 10 nodes → {cum_share[top10_idx]:.1%}')

        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel('Cumulative % of Nodes (ranked by degree)', fontsize=FONT_SIZE['label'])
        ax.set_ylabel('Cumulative Degree Share', fontsize=FONT_SIZE['label'])
        ax.set_title('Hub Dominance (Degree Concentration)', fontweight='bold')
        add_legend(ax, loc='lower right')
        add_grid(ax, alpha=0.25)
        save_figure(fig, self.output_dir / 'hub_dominance.png')
        logger.info("[SAVED] hub_dominance.png")

    # ──────────────────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────────────────
    def _print_summary(self, results: Dict):
        pl = results.get('power_law', {})
        if pl:
            print(f"    ✅ 幂律指数: α={pl.get('exponent', 0):.3f} "
                  f"(in={pl.get('in_degree_exponent', 0):.3f}, out={pl.get('out_degree_exponent', 0):.3f})")
            print(f"       KS统计量={pl.get('ks_statistic', 0):.4f}, p={pl.get('p_value', 0):.4f}")

        ec = results.get('echo_chamber', {})
        if ec:
            print(f"    ✅ E-I指数: {ec.get('ei_index', 0):+.3f}")

        cas = results.get('cascade', {})
        if cas:
            print(f"    ✅ 级联: 数量={cas.get('cascade_count', 0)}, "
                  f"平均规模={cas.get('avg_scale', 0):.1f}, "
                  f"最大规模={cas.get('max_scale', 0)}")

        sw = results.get('small_world', {})
        if sw:
            sigma = sw.get('small_world_sigma', 0)
            print(f"    ✅ 小世界: σ={sigma:.2f} "
                  f"({'是' if sigma > 1 else '否'}小世界网络)")

        sf = results.get('scale_free', {})
        if sf:
            print(f"    ✅ Hub支配度: Top10={sf.get('hub_dominance_top10', 0):.1%}, "
                  f"Top5%={sf.get('hub_dominance_top5pct', 0):.1%}")
