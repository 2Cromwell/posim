# -*- coding: utf-8 -*-
"""
观点极化机制验证

多维度极化分析:
1. 立场极化 (stance-based) — 从 expression_strategy / stance 字段提取
2. 情感极化 (sentiment-based) — 基于 NLP 情感分值或关键词
3. 情绪极化 (emotion-based) — 基于情绪强度与类型分布

核心指标:
- Esteban-Ray 极化指数 (简化)
- Sarle 双峰系数 (Bimodality Coefficient)
- 标准差离散度 / 非中立比率 / 对抗强度

可视化 (每张图单独 figure):
- stance_distribution.png          水平条形图
- polarization_over_time.png       多指标折线 + fill_between
- sentiment_polarization_over_time.png  情感极化时序
- polarization_metrics_summary.png  指标对比柱状图
- stance_scatter.png                散点图
- stance_violin.png                 小提琴图
- interaction_network_*.png         转发互动网络 (networkx 可用时)
- *_comparison.png                  真实数据对比版本
"""
import logging
import warnings
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, Counter
from datetime import datetime, timedelta

from ..base import BaseEvaluator
from ..utils import (
    save_json, parse_time, truncate_time, smooth, normalize,
    calculate_gini_coefficient, calculate_entropy, calculate_normalized_entropy,
    classify_emotion_by_keywords, CONFLICT_KEYWORDS, EMOTION_KEYWORDS,
    STANCE_MAP, classify_stance, calculate_jsd, calculate_curve_similarity
)
from ..visualization import (
    FIG_SIZE, FIG_SIZE_WIDE, FIG_SIZE_SQUARE, LW, LW_MINOR, ALPHA,
    MARKER_SIZE, FONT_SIZE,
    C_SIM, C_REAL, C_EMOTION, C_SENTIMENT,
    setup_time_axis, add_grid, add_legend, save_figure
)

logger = logging.getLogger(__name__)

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

# ── colour constants ──────────────────────────────────────────────
C_SUPPORT = '#2ca02c'
C_OPPOSE = '#d62728'
C_NEUTRAL = '#7f7f7f'
C_POLAR = '#d62728'
C_CONFRONT = '#ff7f0e'
C_BIMODAL = '#9467bd'
C_DISPERSION = '#1f77b4'
C_NON_NEUTRAL = '#e377c2'

STANCE_THRESHOLD = 0.15


class OpinionPolarizationEvaluator(BaseEvaluator):
    """观点极化机制验证评估器"""

    def __init__(self, output_dir: Path):
        super().__init__(output_dir / "opinion_polarization", name="opinion_polarization")

    # ================================================================
    #  public entry
    # ================================================================
    def evaluate(
        self,
        sim_data: Dict[str, Any],
        real_data: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        self._log_section("观点极化机制验证")

        micro_results = sim_data.get('micro_results', [])
        aggregated = sim_data.get('aggregated', {})

        if not micro_results:
            print("    ⚠️ 无模拟数据，跳过观点极化验证")
            return {}

        # ── extract stance values from every action ──
        sim_records = self._extract_stance_records(micro_results, source='sim')
        if not sim_records:
            print("    ⚠️ 无法提取任何立场值，跳过")
            return {}

        real_records: List[Dict] = []
        if real_data and real_data.get('actions'):
            real_records = self._extract_stance_records_real(real_data['actions'])

        # ── compute metrics ──
        print("    [1/5] 计算极化指标…")
        metrics = self._compute_polarization_metrics(sim_records)

        # ── temporal evolution ──
        print("    [2/5] 计算时序极化演化…")
        times = aggregated.get('times', [])
        granularity = self._detect_granularity(times)
        temporal = self._compute_temporal_evolution(sim_records, times, granularity)
        metrics['temporal_evolution'] = temporal

        real_temporal: List[Dict] = []
        if real_records:
            real_times_sorted = sorted(set(r['time'] for r in real_records))
            real_temporal = self._compute_temporal_evolution(real_records, real_times_sorted, granularity)
            metrics['real_temporal_evolution'] = real_temporal

        # ── comparison metrics ──
        if real_records:
            print("    [3/5] 计算真实数据对比指标…")
            real_metrics = self._compute_polarization_metrics(real_records)
            metrics['real_data_metrics'] = real_metrics
            metrics['comparison'] = self._compare_metrics(metrics, real_metrics)
        else:
            print("    [3/5] 无真实数据，跳过对比")

        # ── network ──
        print("    [4/5] 构建互动网络…")
        if HAS_NETWORKX:
            self._build_and_plot_network(sim_records, times)
        else:
            logger.info("networkx 不可用，跳过网络可视化")

        # ── visualisations ──
        print("    [5/5] 生成可视化…")
        self._plot_stance_distribution(sim_records, real_records)
        self._plot_polarization_over_time(temporal, real_temporal)
        self._plot_sentiment_polarization_over_time(sim_records, real_records, times, granularity)
        self._plot_metrics_summary(metrics, real_records)
        self._plot_stance_scatter(sim_records, real_records)
        self._plot_stance_violin(sim_records, real_records, times, granularity)

        self._save_results(metrics, "opinion_polarization_metrics.json")
        self._print_summary(metrics)
        return metrics

    # ================================================================
    #  stance extraction
    # ================================================================
    def _extract_stance_value(self, action: Dict) -> float:
        """从单条 action 提取数值化立场 (−1 ~ +1)。

        优先级:
        1. expression_strategy.stance → STANCE_MAP
        2. stance 字段 (中文字符串 / 英文字符串 / 数值)
        3. stance_intensity (数值)
        4. 文本关键词 classify_stance
        """
        # 1) expression_strategy
        strategy = action.get('expression_strategy')
        if isinstance(strategy, dict):
            raw = strategy.get('stance', '')
            if isinstance(raw, str) and raw in STANCE_MAP:
                return STANCE_MAP[raw]
            if isinstance(raw, (int, float)):
                return float(np.clip(raw, -1, 1))

        # 2) stance field
        stance_raw = action.get('stance')
        if stance_raw is not None:
            if isinstance(stance_raw, str) and stance_raw in STANCE_MAP:
                return STANCE_MAP[stance_raw]
            if isinstance(stance_raw, (int, float)):
                return float(np.clip(stance_raw, -1, 1))

        # 3) stance_intensity (sometimes used as a numeric stance)
        si = action.get('stance_intensity')
        if isinstance(si, (int, float)) and si != 0:
            return float(np.clip(si, -1, 1))

        # 4) text fallback
        text = action.get('text', action.get('content', ''))
        if text:
            label = classify_stance(text)
            return STANCE_MAP.get(label, 0.0)
        return 0.0

    def _extract_emotion_value(self, action: Dict) -> float:
        """将情绪映射为 sentiment‑like 数值 (−1 ~ +1)。

        Negative emotions → negative values, positive → positive.
        """
        emo = action.get('emotion', '')
        if isinstance(emo, str):
            emo = emo.strip()
        emo_lower = emo.lower() if isinstance(emo, str) else ''

        negative_map = {'anger': -0.8, 'disgust': -0.7, 'anxiety': -0.5, 'sadness': -0.6,
                        '愤怒': -0.8, '厌恶': -0.7, '焦虑': -0.5, '悲伤': -0.6,
                        '恐惧': -0.6, '惊恐': -0.6}
        positive_map = {'excitement': 0.7, 'joy': 0.7, '兴奋': 0.7, '喜悦': 0.7, '惊喜': 0.5}
        neutral_map = {'neutral': 0.0, '中性': 0.0, '中立': 0.0}

        for mapping in (negative_map, positive_map, neutral_map):
            if emo in mapping:
                return mapping[emo]
            if emo_lower in mapping:
                return mapping[emo_lower]

        # intensity-weighted fallback
        intensity = action.get('emotion_intensity', 0.5)
        if isinstance(intensity, str):
            from ..utils import INTENSITY_MAP
            intensity = INTENSITY_MAP.get(intensity, 0.5)
        return 0.0

    def _extract_sentiment_value(self, action: Dict) -> float:
        """Sentiment score in [−1, +1]. Uses NLP score, snownlp, or keyword heuristic."""
        # nlp score (real data often has this)
        nlp = action.get('nlp_sentiment_score')
        if isinstance(nlp, (int, float)):
            return float(np.clip(nlp * 2 - 1, -1, 1))  # [0,1] → [−1,+1]

        # sentiment_polarity label
        pol = action.get('sentiment_polarity', '')
        pol_map = {'positive': 0.6, 'negative': -0.6, 'neutral': 0.0,
                   '正面': 0.6, '负面': -0.6, '中性': 0.0}
        if pol in pol_map:
            return pol_map[pol]

        return 0.0

    def _extract_stance_records(self, micro_results: List[Dict], source: str = 'sim') -> List[Dict]:
        records: List[Dict] = []
        for a in micro_results:
            t = parse_time(a.get('time', ''))
            if not t:
                continue
            sv = self._extract_stance_value(a)
            ev = self._extract_emotion_value(a)
            sev = self._extract_sentiment_value(a)
            records.append({
                'time': t,
                'stance_value': sv,
                'emotion_value': ev,
                'sentiment_value': sev,
                'user_id': a.get('user_id', ''),
                'agent_type': a.get('agent_type', 'citizen'),
                'action_type': a.get('action_type', ''),
                'target_post_id': a.get('target_post_id', ''),
                'target_author': a.get('target_author', ''),
                'source': source,
            })
        return records

    def _extract_stance_records_real(self, actions: List[Dict]) -> List[Dict]:
        records: List[Dict] = []
        for a in actions:
            t = a.get('time')
            if isinstance(t, str):
                t = parse_time(t)
            if not t:
                continue
            # real data: stance_value is often pre-computed
            sv = a.get('stance_value', 0.0)
            if not isinstance(sv, (int, float)):
                raw_stance = a.get('stance', '')
                sv = STANCE_MAP.get(raw_stance, 0.0) if isinstance(raw_stance, str) else 0.0

            ev = self._extract_emotion_value(a)
            sev = self._extract_sentiment_value(a)

            records.append({
                'time': t,
                'stance_value': float(sv),
                'emotion_value': ev,
                'sentiment_value': sev,
                'user_id': a.get('user_id', ''),
                'agent_type': '',
                'action_type': a.get('type', ''),
                'target_post_id': '',
                'target_author': '',
                'source': 'real',
            })
        return records

    # ================================================================
    #  polarization metrics
    # ================================================================
    @staticmethod
    def _classify_stance_label(v: float) -> str:
        if v > STANCE_THRESHOLD:
            return 'Support'
        if v < -STANCE_THRESHOLD:
            return 'Oppose'
        return 'Neutral'

    def _compute_polarization_metrics(self, records: List[Dict]) -> Dict[str, Any]:
        vals = np.array([r['stance_value'] for r in records], dtype=float)
        labels = [self._classify_stance_label(v) for v in vals]
        counts = Counter(labels)
        total = len(labels) or 1

        support_ratio = counts.get('Support', 0) / total
        oppose_ratio = counts.get('Oppose', 0) / total
        neutral_ratio = counts.get('Neutral', 0) / total

        # ── Esteban-Ray (simplified) ──
        er_index = self._esteban_ray(vals, labels)

        # ── Sarle bimodality coefficient ──
        bc = self._bimodality_coefficient(vals)

        # ── dispersion (std) ──
        dispersion = float(np.std(vals)) if len(vals) > 1 else 0.0

        # ── non-neutral ratio ──
        non_neutral = 1.0 - neutral_ratio

        # ── confrontation intensity ──
        confrontation = 4.0 * support_ratio * oppose_ratio

        # emotion-based polarization
        emo_vals = np.array([r['emotion_value'] for r in records], dtype=float)
        emo_dispersion = float(np.std(emo_vals)) if len(emo_vals) > 1 else 0.0
        emo_bc = self._bimodality_coefficient(emo_vals)

        # sentiment-based polarization
        sent_vals = np.array([r['sentiment_value'] for r in records], dtype=float)
        sent_dispersion = float(np.std(sent_vals)) if len(sent_vals) > 1 else 0.0
        sent_bc = self._bimodality_coefficient(sent_vals)

        return {
            'stance_distribution': dict(counts),
            'stance_ratios': {
                'Support': float(support_ratio),
                'Oppose': float(oppose_ratio),
                'Neutral': float(neutral_ratio),
            },
            'esteban_ray_index': er_index,
            'bimodality_coefficient': bc,
            'stance_dispersion': dispersion,
            'non_neutral_ratio': float(non_neutral),
            'confrontation_intensity': float(confrontation),
            'stance_mean': float(np.mean(vals)),
            'stance_median': float(np.median(vals)),
            'emotion_dispersion': emo_dispersion,
            'emotion_bimodality': emo_bc,
            'sentiment_dispersion': sent_dispersion,
            'sentiment_bimodality': sent_bc,
            'total_actions': len(records),
        }

    @staticmethod
    def _esteban_ray(vals: np.ndarray, labels: List[str]) -> float:
        """Simplified Esteban-Ray: ER = K * sum_i sum_j p_i^(1+alpha) * p_j * |y_i - y_j|"""
        groups = defaultdict(list)
        for v, lb in zip(vals, labels):
            groups[lb].append(v)
        if len(groups) < 2:
            return 0.0
        n = len(vals) or 1
        alpha = 1.6  # standard ER alpha
        er = 0.0
        group_keys = list(groups.keys())
        for i, gi in enumerate(group_keys):
            pi = len(groups[gi]) / n
            yi = float(np.mean(groups[gi]))
            for j, gj in enumerate(group_keys):
                pj = len(groups[gj]) / n
                yj = float(np.mean(groups[gj]))
                er += (pi ** (1 + alpha)) * pj * abs(yi - yj)
        return float(min(er, 1.0))

    @staticmethod
    def _bimodality_coefficient(values: np.ndarray) -> float:
        if len(values) < 4:
            return 0.0
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                from scipy.stats import kurtosis, skew
                n = len(values)
                s = skew(values)
                k = kurtosis(values)
                denom = k + 3 * ((n - 1) ** 2) / ((n - 2) * (n - 3))
                if denom == 0:
                    return 0.0
                bc = (s ** 2 + 1) / denom
                return float(np.clip(bc, 0, 1))
            except Exception:
                return 0.0

    # ================================================================
    #  temporal evolution (cumulative window)
    # ================================================================
    @staticmethod
    def _detect_granularity(times: List[datetime]) -> int:
        if len(times) >= 2:
            diff = (times[1] - times[0]).total_seconds() / 60
            return max(int(diff), 1)
        return 10

    def _compute_temporal_evolution(
        self, records: List[Dict], times: List[datetime], granularity: int,
    ) -> List[Dict]:
        if not records:
            return []

        # bucket records
        time_buckets: Dict[datetime, List[Dict]] = defaultdict(list)
        for r in records:
            t_key = truncate_time(r['time'], granularity)
            time_buckets[t_key].append(r)

        if not times:
            times = sorted(time_buckets.keys())
        if not times:
            return []

        cumulative_vals: List[float] = []
        evolution: List[Dict] = []

        for t in times:
            bucket = time_buckets.get(t, [])
            new_vals = [r['stance_value'] for r in bucket]
            cumulative_vals.extend(new_vals)

            if not cumulative_vals:
                evolution.append({
                    'time': t.isoformat(),
                    'polarization': 0.0,
                    'confrontation': 0.0,
                    'bimodality': 0.0,
                    'dispersion': 0.0,
                    'non_neutral': 0.0,
                    'n_actions': 0,
                })
                continue

            arr = np.array(cumulative_vals)
            labels = [self._classify_stance_label(v) for v in arr]
            counts = Counter(labels)
            n = len(arr)
            sr = counts.get('Support', 0) / n
            opr = counts.get('Oppose', 0) / n
            nr = counts.get('Neutral', 0) / n

            evolution.append({
                'time': t.isoformat(),
                'polarization': float(1.0 - nr),
                'confrontation': float(4.0 * sr * opr),
                'bimodality': self._bimodality_coefficient(arr),
                'dispersion': float(np.std(arr)),
                'non_neutral': float(1.0 - nr),
                'support_ratio': float(sr),
                'oppose_ratio': float(opr),
                'n_actions': n,
            })

        return evolution

    # ================================================================
    #  comparison
    # ================================================================
    def _compare_metrics(self, sim_m: Dict, real_m: Dict) -> Dict[str, Any]:
        keys = ['esteban_ray_index', 'bimodality_coefficient', 'stance_dispersion',
                'non_neutral_ratio', 'confrontation_intensity']
        comp: Dict[str, Any] = {}
        for k in keys:
            sv = sim_m.get(k, 0.0)
            rv = real_m.get(k, 0.0)
            comp[k] = {
                'sim': sv, 'real': rv,
                'diff': float(sv - rv),
                'abs_diff': float(abs(sv - rv)),
            }
        # stance distribution JSD
        sim_dist = sim_m.get('stance_ratios', {})
        real_dist = real_m.get('stance_ratios', {})
        all_keys = sorted(set(list(sim_dist.keys()) + list(real_dist.keys())))
        if all_keys:
            p = np.array([sim_dist.get(k, 0.0) for k in all_keys]) + 1e-10
            q = np.array([real_dist.get(k, 0.0) for k in all_keys]) + 1e-10
            p /= p.sum()
            q /= q.sum()
            comp['stance_jsd'] = float(calculate_jsd(p, q))

        return comp

    # ================================================================
    #  network
    # ================================================================
    def _build_and_plot_network(self, records: List[Dict], times: List[datetime]):
        if not HAS_NETWORKX:
            return
        interactions: List[Dict] = []
        for r in records:
            if r.get('target_author') and r.get('user_id'):
                interactions.append(r)
        if len(interactions) < 3:
            logger.info("互动记录不足，跳过网络可视化")
            return

        user_stances: Dict[str, List[float]] = defaultdict(list)
        for r in records:
            if r['user_id']:
                user_stances[r['user_id']].append(r['stance_value'])

        # split into time thirds for snapshots
        sorted_interactions = sorted(interactions, key=lambda x: x['time'])
        n = len(sorted_interactions)
        thirds = [sorted_interactions[:n // 3],
                  sorted_interactions[:2 * n // 3],
                  sorted_interactions]
        labels_snap = ['early', 'mid', 'full']

        for idx, (chunk, label) in enumerate(zip(thirds, labels_snap)):
            if not chunk:
                continue
            G = nx.DiGraph()
            for r in chunk:
                src, tgt = r['user_id'], r['target_author']
                if src and tgt and src != tgt:
                    if G.has_edge(src, tgt):
                        G[src][tgt]['weight'] += 1
                    else:
                        G.add_edge(src, tgt, weight=1)

            if G.number_of_nodes() < 2:
                continue

            fig, ax = plt.subplots(figsize=FIG_SIZE_SQUARE)

            node_colors = []
            for node in G.nodes():
                svs = user_stances.get(node, [0.0])
                mean_s = float(np.mean(svs))
                if mean_s > STANCE_THRESHOLD:
                    node_colors.append(C_SUPPORT)
                elif mean_s < -STANCE_THRESHOLD:
                    node_colors.append(C_OPPOSE)
                else:
                    node_colors.append(C_NEUTRAL)

            degrees = dict(G.degree())
            node_sizes = [max(30, min(300, degrees.get(n, 1) * 15)) for n in G.nodes()]

            try:
                pos = nx.spring_layout(G, k=1.5 / np.sqrt(max(G.number_of_nodes(), 1)),
                                       iterations=50, seed=42)
            except Exception:
                pos = nx.random_layout(G, seed=42)

            nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.15, edge_color='#aaaaaa',
                                   arrows=True, arrowsize=8, width=0.5)
            nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                                   node_size=node_sizes, alpha=0.75, edgecolors='black', linewidths=0.5)

            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=C_SUPPORT, edgecolor='black', label='Support'),
                Patch(facecolor=C_OPPOSE, edgecolor='black', label='Oppose'),
                Patch(facecolor=C_NEUTRAL, edgecolor='black', label='Neutral'),
            ]
            ax.legend(handles=legend_elements, loc='upper left',
                      frameon=True, fancybox=False, edgecolor='black', fontsize=FONT_SIZE['legend'])
            ax.set_title(f'Interaction Network ({label}, {G.number_of_nodes()} nodes, '
                         f'{G.number_of_edges()} edges)', fontweight='bold')
            ax.axis('off')
            save_figure(fig, self.output_dir / f'interaction_network_{label}.png')
            logger.info(f"[SAVED] interaction_network_{label}.png")

    # ================================================================
    #  Plotting helpers
    # ================================================================
    def _plot_stance_distribution(self, sim_records: List[Dict],
                                  real_records: List[Dict]):
        sim_labels = [self._classify_stance_label(r['stance_value']) for r in sim_records]
        sim_counts = Counter(sim_labels)
        categories = ['Support', 'Neutral', 'Oppose']
        sim_vals = [sim_counts.get(c, 0) for c in categories]
        colors = [C_SUPPORT, C_NEUTRAL, C_OPPOSE]

        if real_records:
            real_labels = [self._classify_stance_label(r['stance_value']) for r in real_records]
            real_counts = Counter(real_labels)
            real_vals = [real_counts.get(c, 0) for c in categories]
            self._do_plot_stance_dist(sim_vals, real_vals, categories, colors,
                                     'stance_distribution_comparison.png')
        self._do_plot_stance_dist(sim_vals, None, categories, colors,
                                 'stance_distribution.png')

    def _do_plot_stance_dist(self, sim_vals, real_vals, categories, colors, filename):
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        y_pos = np.arange(len(categories))
        bar_h = 0.35 if real_vals else 0.5

        bars_sim = ax.barh(y_pos + (bar_h / 2 if real_vals else 0), sim_vals,
                           height=bar_h, color=colors, alpha=0.85, edgecolor='black',
                           label='Simulation')
        for bar, val in zip(bars_sim, sim_vals):
            ax.text(bar.get_width() + max(sum(sim_vals) * 0.01, 1), bar.get_y() + bar.get_height() / 2,
                    str(val), ha='left', va='center', fontsize=FONT_SIZE['annotation'], fontweight='bold')

        if real_vals is not None:
            bars_real = ax.barh(y_pos - bar_h / 2, real_vals, height=bar_h,
                                color=colors, alpha=0.40, edgecolor='black',
                                hatch='///', label='Real')
            for bar, val in zip(bars_real, real_vals):
                ax.text(bar.get_width() + max(sum(real_vals) * 0.01, 1),
                        bar.get_y() + bar.get_height() / 2,
                        str(val), ha='left', va='center', fontsize=FONT_SIZE['annotation'])
            add_legend(ax, loc='lower right')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(categories)
        ax.set_xlabel('Action Count')
        ax.set_title('Stance Distribution', fontweight='bold')
        add_grid(ax)
        save_figure(fig, self.output_dir / filename)
        logger.info(f"[SAVED] {filename}")

    # ── polarization over time ────────────────────────────────────
    def _plot_polarization_over_time(self, temporal: List[Dict],
                                     real_temporal: List[Dict]):
        if not temporal:
            return

        if real_temporal:
            self._do_plot_pol_time(temporal, real_temporal,
                                  'polarization_over_time_comparison.png')
        self._do_plot_pol_time(temporal, None, 'polarization_over_time.png')

    def _do_plot_pol_time(self, temporal, real_temporal, filename):
        fig, ax = plt.subplots(figsize=FIG_SIZE_WIDE)
        times_dt = [parse_time(e['time']) for e in temporal]
        pol = smooth([e['polarization'] for e in temporal], w=3)
        conf = smooth([e['confrontation'] for e in temporal], w=3)
        bim = smooth([e['bimodality'] for e in temporal], w=3)
        disp = smooth([e['dispersion'] for e in temporal], w=3)

        n = len(times_dt)
        pol = pol[:n]; conf = conf[:n]; bim = bim[:n]; disp = disp[:n]

        ax.plot(times_dt, pol, color=C_POLAR, lw=LW, label='Non-neutral ratio')
        ax.fill_between(times_dt, np.maximum(pol - 0.1, 0), np.minimum(pol + 0.1, 1.0),
                        color=C_POLAR, alpha=ALPHA)
        ax.plot(times_dt, conf, color=C_CONFRONT, lw=LW, label='Confrontation')
        ax.fill_between(times_dt, np.maximum(conf - 0.05, 0), np.minimum(conf + 0.05, 1.0),
                        color=C_CONFRONT, alpha=ALPHA)
        ax.plot(times_dt, bim, color=C_BIMODAL, lw=LW, ls='--', label='Bimodality')
        ax.plot(times_dt, disp, color=C_DISPERSION, lw=LW, ls=':', label='Dispersion (σ)')

        if real_temporal:
            r_times = [parse_time(e['time']) for e in real_temporal]
            rn = len(r_times)
            r_pol = smooth([e['polarization'] for e in real_temporal], w=3)[:rn]
            r_conf = smooth([e['confrontation'] for e in real_temporal], w=3)[:rn]
            ax.plot(r_times, r_pol, color=C_REAL['total'], lw=LW, ls='--',
                    label='Real non-neutral', marker='o', markersize=MARKER_SIZE - 1)
            ax.plot(r_times, r_conf, color='#b39bb5', lw=LW, ls='--',
                    label='Real confrontation', marker='s', markersize=MARKER_SIZE - 1)

        ax.set_ylabel('Index Value')
        ax.set_xlabel('Time')
        ax.set_title('Polarization Metrics Over Time (Cumulative)', fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        setup_time_axis(ax)
        add_legend(ax, loc='upper left')
        add_grid(ax)
        save_figure(fig, self.output_dir / filename)
        logger.info(f"[SAVED] {filename}")

    # ── sentiment polarization over time ──────────────────────────
    def _plot_sentiment_polarization_over_time(
        self, sim_records, real_records, times, granularity,
    ):
        if not sim_records:
            return

        if real_records:
            self._do_plot_sent_pol(sim_records, real_records, times, granularity,
                                  'sentiment_polarization_over_time_comparison.png')
        self._do_plot_sent_pol(sim_records, None, times, granularity,
                              'sentiment_polarization_over_time.png')

    def _do_plot_sent_pol(self, sim_records, real_records, times, granularity, filename):
        fig, ax = plt.subplots(figsize=FIG_SIZE_WIDE)

        sim_series = self._sentiment_series(sim_records, times, granularity)
        n = len(times) if times else len(sim_series['mean'])
        ts = times[:n] if times else list(range(n))

        mean_arr = smooth(sim_series['mean'][:n], w=3)
        pos_arr = smooth(sim_series['pos_ratio'][:n], w=3)
        neg_arr = smooth(sim_series['neg_ratio'][:n], w=3)

        ax.plot(ts, mean_arr, color=C_SENTIMENT['neutral'], lw=LW, label='Mean sentiment')
        ax.fill_between(ts, np.maximum(mean_arr - 0.15, -1), np.minimum(mean_arr + 0.15, 1),
                        color=C_SENTIMENT['neutral'], alpha=ALPHA)
        ax.plot(ts, pos_arr, color=C_SENTIMENT['positive'], lw=LW_MINOR, ls='--',
                label='Positive ratio')
        ax.plot(ts, -neg_arr, color=C_SENTIMENT['negative'], lw=LW_MINOR, ls='--',
                label='−Negative ratio')

        if real_records:
            real_times_set = sorted(set(r['time'] for r in real_records))
            real_series = self._sentiment_series(real_records, real_times_set, granularity)
            rn = len(real_times_set)
            r_mean = smooth(real_series['mean'][:rn], w=3)
            ax.plot(real_times_set[:len(r_mean)], r_mean, color=C_REAL['total'], lw=LW,
                    ls='-.', label='Real mean sentiment', marker='o', markersize=MARKER_SIZE - 1)

        ax.set_ylabel('Sentiment Value')
        ax.set_xlabel('Time')
        ax.set_title('Sentiment Polarization Over Time', fontweight='bold')
        ax.axhline(y=0, color='black', lw=0.8, ls='-', alpha=0.3)
        if times:
            setup_time_axis(ax)
        add_legend(ax, loc='best')
        add_grid(ax)
        save_figure(fig, self.output_dir / filename)
        logger.info(f"[SAVED] {filename}")

    def _sentiment_series(self, records, times, granularity):
        buckets: Dict[datetime, List[float]] = defaultdict(list)
        for r in records:
            t_key = truncate_time(r['time'], granularity)
            buckets[t_key].append(r['sentiment_value'])
        if not times:
            times = sorted(buckets.keys())
        cumul: List[float] = []
        means, pos_ratios, neg_ratios = [], [], []
        for t in times:
            cumul.extend(buckets.get(t, []))
            if cumul:
                arr = np.array(cumul)
                means.append(float(np.mean(arr)))
                pos_ratios.append(float(np.sum(arr > 0.1) / len(arr)))
                neg_ratios.append(float(np.sum(arr < -0.1) / len(arr)))
            else:
                means.append(0.0)
                pos_ratios.append(0.0)
                neg_ratios.append(0.0)
        return {'mean': means, 'pos_ratio': pos_ratios, 'neg_ratio': neg_ratios}

    # ── metrics summary bar chart ─────────────────────────────────
    def _plot_metrics_summary(self, metrics: Dict, real_records: List[Dict]):
        if real_records:
            self._do_plot_summary(metrics, True, 'polarization_metrics_summary_comparison.png')
        self._do_plot_summary(metrics, False, 'polarization_metrics_summary.png')

    def _do_plot_summary(self, metrics, with_real, filename):
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        labels = ['Esteban-Ray\nIndex', 'Bimodality\nCoeff.', 'Stance\nDispersion',
                  'Non-neutral\nRatio', 'Confrontation\nIntensity']
        keys = ['esteban_ray_index', 'bimodality_coefficient', 'stance_dispersion',
                'non_neutral_ratio', 'confrontation_intensity']
        sim_vals = [metrics.get(k, 0.0) for k in keys]
        colors_list = [C_POLAR, C_BIMODAL, C_DISPERSION, C_NON_NEUTRAL, C_CONFRONT]
        x = np.arange(len(labels))
        width = 0.35 if with_real else 0.5

        bars = ax.bar(x - (width / 2 if with_real else 0), sim_vals, width,
                      color=colors_list, alpha=0.85, edgecolor='black', label='Simulation')
        for bar, val in zip(bars, sim_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=FONT_SIZE['annotation'] - 1,
                    fontweight='bold')

        if with_real:
            real_m = metrics.get('real_data_metrics', {})
            real_vals = [real_m.get(k, 0.0) for k in keys]
            bars_r = ax.bar(x + width / 2, real_vals, width,
                            color=colors_list, alpha=0.40, edgecolor='black',
                            hatch='///', label='Real')
            for bar, val in zip(bars_r, real_vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=FONT_SIZE['annotation'] - 1)
            add_legend(ax, loc='upper right')

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Value')
        ax.set_title('Polarization Metrics Summary', fontweight='bold')
        ax.set_ylim(0, max(max(sim_vals) * 1.35, 0.1))
        add_grid(ax)
        save_figure(fig, self.output_dir / filename)
        logger.info(f"[SAVED] {filename}")

    # ── stance scatter ────────────────────────────────────────────
    def _plot_stance_scatter(self, sim_records, real_records):
        if real_records:
            self._do_plot_scatter(sim_records, real_records, 'stance_scatter_comparison.png')
        self._do_plot_scatter(sim_records, None, 'stance_scatter.png')

    def _do_plot_scatter(self, sim_records, real_records, filename):
        fig, ax = plt.subplots(figsize=FIG_SIZE_WIDE)

        sim_times = [r['time'] for r in sim_records]
        sim_stances = [r['stance_value'] for r in sim_records]
        sim_colors = [C_SUPPORT if v > STANCE_THRESHOLD else C_OPPOSE if v < -STANCE_THRESHOLD
                      else C_NEUTRAL for v in sim_stances]

        ax.scatter(sim_times, sim_stances, c=sim_colors, s=MARKER_SIZE * 3,
                   alpha=0.5, edgecolors='none', label='Sim actions', zorder=2)

        if real_records:
            r_times = [r['time'] for r in real_records]
            r_stances = [r['stance_value'] for r in real_records]
            ax.scatter(r_times, r_stances, c='black', s=MARKER_SIZE * 5, alpha=0.35,
                       marker='x', label='Real actions', zorder=3)

        ax.axhline(y=0, color='black', lw=0.8, ls='-', alpha=0.3)
        ax.axhline(y=STANCE_THRESHOLD, color=C_SUPPORT, lw=0.6, ls=':', alpha=0.5)
        ax.axhline(y=-STANCE_THRESHOLD, color=C_OPPOSE, lw=0.6, ls=':', alpha=0.5)
        ax.set_ylabel('Stance Value')
        ax.set_xlabel('Time')
        ax.set_title('Individual Stance Values Over Time', fontweight='bold')
        ax.set_ylim(-1.15, 1.15)
        setup_time_axis(ax)
        if real_records:
            add_legend(ax, loc='upper right')
        add_grid(ax)
        save_figure(fig, self.output_dir / filename)
        logger.info(f"[SAVED] {filename}")

    # ── stance violin ─────────────────────────────────────────────
    def _plot_stance_violin(self, sim_records, real_records, times, granularity):
        if real_records:
            self._do_plot_violin(sim_records, real_records, times, granularity,
                                'stance_violin_comparison.png')
        self._do_plot_violin(sim_records, None, times, granularity, 'stance_violin.png')

    def _do_plot_violin(self, sim_records, real_records, times, granularity, filename):
        n_periods = min(8, max(3, len(times) // 6)) if times else 4

        sim_sorted = sorted(sim_records, key=lambda x: x['time'])
        chunk_size = max(1, len(sim_sorted) // n_periods)
        sim_chunks = [sim_sorted[i:i + chunk_size] for i in range(0, len(sim_sorted), chunk_size)]
        if len(sim_chunks) > n_periods:
            sim_chunks = sim_chunks[:n_periods]

        period_labels = [f'P{i + 1}' for i in range(len(sim_chunks))]
        sim_data_lists = [[r['stance_value'] for r in ch] for ch in sim_chunks]

        fig, ax = plt.subplots(figsize=FIG_SIZE_WIDE)
        positions = list(range(1, len(sim_data_lists) + 1))

        non_empty = [(p, d) for p, d in zip(positions, sim_data_lists) if len(d) >= 2]
        if not non_empty:
            plt.close(fig)
            return
        pos_ne, data_ne = zip(*non_empty)

        parts = ax.violinplot(data_ne, positions=list(pos_ne), showmeans=True,
                              showmedians=True, widths=0.7)
        for pc in parts['bodies']:
            pc.set_facecolor(C_SIM['total'])
            pc.set_alpha(0.6)
        for key in ('cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxes'):
            if key in parts:
                parts[key].set_color('black')

        if real_records:
            real_sorted = sorted(real_records, key=lambda x: x['time'])
            r_chunk_size = max(1, len(real_sorted) // n_periods)
            real_chunks = [real_sorted[i:i + r_chunk_size]
                           for i in range(0, len(real_sorted), r_chunk_size)]
            if len(real_chunks) > n_periods:
                real_chunks = real_chunks[:n_periods]
            real_data_lists = [[r['stance_value'] for r in ch] for ch in real_chunks]
            r_ne = [(p + 0.0, d) for p, d in zip(range(1, len(real_data_lists) + 1), real_data_lists)
                    if len(d) >= 2]
            if r_ne:
                rp, rd = zip(*r_ne)
                rp_shifted = [p + 0.35 for p in rp]
                r_parts = ax.violinplot(rd, positions=rp_shifted, showmeans=True,
                                        showmedians=True, widths=0.3)
                for pc in r_parts['bodies']:
                    pc.set_facecolor(C_REAL['total'])
                    pc.set_alpha(0.5)

        ax.axhline(y=0, color='black', lw=0.8, ls='-', alpha=0.3)
        ax.axhline(y=STANCE_THRESHOLD, color=C_SUPPORT, lw=0.6, ls=':', alpha=0.5)
        ax.axhline(y=-STANCE_THRESHOLD, color=C_OPPOSE, lw=0.6, ls=':', alpha=0.5)

        ax.set_xticks(positions)
        ax.set_xticklabels(period_labels[:len(positions)])
        ax.set_ylabel('Stance Value')
        ax.set_xlabel('Time Period')
        ax.set_title('Stance Value Distribution by Period (Violin)', fontweight='bold')
        ax.set_ylim(-1.3, 1.3)
        add_grid(ax)

        from matplotlib.patches import Patch
        handles = [Patch(facecolor=C_SIM['total'], alpha=0.6, label='Simulation')]
        if real_records:
            handles.append(Patch(facecolor=C_REAL['total'], alpha=0.5, label='Real'))
        ax.legend(handles=handles, loc='upper right', frameon=True, fancybox=False,
                  edgecolor='black', fontsize=FONT_SIZE['legend'])

        save_figure(fig, self.output_dir / filename)
        logger.info(f"[SAVED] {filename}")

    # ================================================================
    #  summary print
    # ================================================================
    def _print_summary(self, metrics: Dict):
        dist = metrics.get('stance_distribution', {})
        print(f"    ✅ 立场分布: {dist}")
        print(f"       Esteban-Ray 极化指数: {metrics.get('esteban_ray_index', 0):.4f}")
        print(f"       双峰系数: {metrics.get('bimodality_coefficient', 0):.4f}")
        print(f"       立场离散度 (σ): {metrics.get('stance_dispersion', 0):.4f}")
        print(f"       非中立比率: {metrics.get('non_neutral_ratio', 0):.4f}")
        print(f"       对抗强度: {metrics.get('confrontation_intensity', 0):.4f}")
        print(f"       情绪离散度: {metrics.get('emotion_dispersion', 0):.4f}")
        print(f"       情感离散度: {metrics.get('sentiment_dispersion', 0):.4f}")
        if 'comparison' in metrics:
            comp = metrics['comparison']
            jsd = comp.get('stance_jsd', -1)
            if jsd >= 0:
                print(f"    📊 立场分布 JSD (sim vs real): {jsd:.4f}")
            for k in ['esteban_ray_index', 'non_neutral_ratio', 'confrontation_intensity']:
                if k in comp:
                    print(f"       {k}: sim={comp[k]['sim']:.3f}  real={comp[k]['real']:.3f}  "
                          f"Δ={comp[k]['diff']:+.3f}")
