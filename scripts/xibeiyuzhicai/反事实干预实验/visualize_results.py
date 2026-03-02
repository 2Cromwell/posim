"""
PR Strategy Comparison Experiment - Publication-Quality Visualization
Designed to maximize visibility of intervention effects.
"""
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path
from collections import defaultdict

# ============================================================
# Style Configuration
# ============================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '-',
    'grid.linewidth': 0.4,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'lines.linewidth': 2,
})

STRATEGY_LABELS = {
    'baseline':      'Actual Response',
    'swift_apology': 'Swift Apology',
    'transparency':  'Proactive Transparency',
    'dialogue':      'Consumer Dialogue',
    'silence':       'Strategic Silence',
}
STRATEGY_ORDER = ['baseline', 'swift_apology', 'transparency', 'dialogue', 'silence']
ACTIVE_STRATEGIES = ['swift_apology', 'transparency', 'dialogue', 'silence']
STRATEGY_COLORS = {
    'baseline':      '#78909C',
    'swift_apology': '#1565C0',
    'transparency':  '#2E7D32',
    'dialogue':      '#E65100',
    'silence':       '#6A1B9A',
}
STRATEGY_MARKERS = {
    'baseline': 'o', 'swift_apology': 's', 'transparency': '^',
    'dialogue': 'D', 'silence': 'v',
}
STRATEGY_LINESTYLES = {
    'baseline': (0, (4, 3)), 'swift_apology': '-', 'transparency': '--',
    'dialogue': '-.', 'silence': (0, (2, 2)),
}

INTERVENTION_STEPS = [4, 12]
PANEL_LABELS = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']


def load_data(summary_path: str):
    with open(summary_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    agg = defaultdict(list)
    for r in data['results']:
        agg[r['strategy']].append(r)
    return data, agg


def rolling_mean(data, window=3):
    """Small rolling window to reduce noise while preserving sharp changes."""
    arr = np.array(data, dtype=float)
    n = len(arr)
    result = np.zeros(n)
    hw = window // 2
    for i in range(n):
        lo, hi = max(0, i - hw), min(n, i + hw + 1)
        result[i] = np.mean(arr[lo:hi])
    return result


def get_series(agg, strategy, metric):
    """Get averaged time series for a strategy across repeats."""
    all_s = []
    for run in agg[strategy]:
        vals = [s.get(metric, 0) for s in run['step_metrics']]
        all_s.append(vals)
    max_len = max(len(s) for s in all_s)
    padded = []
    for s in all_s:
        if len(s) < max_len:
            s = s + [s[-1]] * (max_len - len(s))
        padded.append(s)
    return np.mean(padded, axis=0)


def add_intervention_zones(ax, steps, alpha=0.08):
    """Add shaded intervention zones and vertical markers."""
    ymin, ymax = ax.get_ylim()
    for istep in INTERVENTION_STEPS:
        ax.axvline(x=istep, color='#37474F', linewidth=1.0,
                   linestyle='--', alpha=0.5, zorder=1)
        ax.axvspan(istep, min(istep + 3, steps[-1]),
                   alpha=alpha, color='#B3E5FC', zorder=0)


def add_intervention_labels(ax, steps):
    """Add clean text labels for intervention points."""
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    for idx, istep in enumerate(INTERVENTION_STEPS):
        label = f'Strategy\nDeployment {idx+1}'
        ax.annotate(label, xy=(istep, ymax - span * 0.03),
                    fontsize=7, color='#37474F', fontweight='bold',
                    ha='center', va='top',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor='#90A4AE', alpha=0.9))


def _save(fig, output_dir, name):
    fig.savefig(output_dir / f'{name}.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'  [OK] {name}')


# ============================================================
# Figure 1 (MAIN): Strategy Effect — Relative to Baseline
# This is the KEY figure showing intervention effects clearly
# ============================================================
def fig1_intervention_effect(agg, output_dir):
    """
    2x2 figure showing strategy effects RELATIVE TO BASELINE.
    By subtracting baseline, shared noise is canceled and intervention
    effects become clearly visible.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = [
        ('negative_emotion_ratio', 'Negative Emotion Ratio\n(Relative to Baseline)', axes[0, 0]),
        ('anger_ratio',            'Anger Ratio\n(Relative to Baseline)',            axes[0, 1]),
        ('mean_emotion_intensity', 'Emotion Intensity\n(Relative to Baseline)',      axes[1, 0]),
        ('aggression_index',       'Aggression Index\n(Relative to Baseline)',       axes[1, 1]),
    ]

    for metric, ylabel, ax in metrics:
        bl = get_series(agg, 'baseline', metric)
        steps = np.arange(1, len(bl) + 1)

        ax.axhline(y=0, color='#78909C', linewidth=2.0, alpha=0.6,
                   linestyle='-', label='Baseline (zero line)', zorder=2)

        for s in ACTIVE_STRATEGIES:
            raw = get_series(agg, s, metric)
            diff_raw = raw - bl[:len(raw)]
            diff_smooth = rolling_mean(diff_raw, window=3)

            ax.plot(steps[:len(diff_raw)], diff_raw,
                    color=STRATEGY_COLORS[s], alpha=0.25, linewidth=0.8)
            ax.plot(steps[:len(diff_smooth)], diff_smooth,
                    color=STRATEGY_COLORS[s], linewidth=2.5,
                    linestyle=STRATEGY_LINESTYLES[s],
                    label=STRATEGY_LABELS[s],
                    marker=STRATEGY_MARKERS[s],
                    markevery=2, markersize=6,
                    markeredgecolor='white', markeredgewidth=0.6)

        ax.set_ylabel(ylabel, fontweight='bold', fontsize=11)
        add_intervention_zones(ax, steps, alpha=0.10)
        add_intervention_labels(ax, steps)

    for pi, (_, _, ax) in enumerate(metrics):
        ax.text(-0.08, 1.05, PANEL_LABELS[pi], transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top')
        if pi >= 2:
            ax.set_xlabel('Simulation Step')

    axes[0, 0].legend(loc='lower left', framealpha=0.95, edgecolor='#cccccc',
                      fancybox=True, borderpad=0.6, fontsize=8.5)

    fig.text(0.5, 0.01,
             'Values below zero = improvement over baseline  |  '
             'Shaded zones = post-strategy-deployment periods',
             ha='center', fontsize=9, style='italic', color='#546E7A')

    fig.suptitle('Strategy Effect on Emotion Dynamics\n(Difference from Actual Response Baseline)',
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    _save(fig, output_dir, 'fig1_intervention_effect')


# ============================================================
# Figure 2: Absolute Values with Intervention Zones
# ============================================================
def fig2_absolute_temporal(agg, output_dir):
    """Absolute NER and Anger over time, raw data + light rolling average."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    for panel_ax, metric, ylabel, title in [
        (ax1, 'negative_emotion_ratio', 'Negative Emotion Ratio', '(a) Negative Emotion Ratio'),
        (ax2, 'anger_ratio', 'Anger Ratio', '(b) Anger Ratio'),
    ]:
        for s in STRATEGY_ORDER:
            raw = get_series(agg, s, metric)
            smoothed = rolling_mean(raw, window=3)
            steps = np.arange(1, len(raw) + 1)

            panel_ax.plot(steps, raw, color=STRATEGY_COLORS[s],
                          alpha=0.2, linewidth=0.8)
            panel_ax.plot(steps, smoothed, color=STRATEGY_COLORS[s],
                          linewidth=2.0, linestyle=STRATEGY_LINESTYLES[s],
                          label=STRATEGY_LABELS[s],
                          marker=STRATEGY_MARKERS[s],
                          markevery=2, markersize=5,
                          markeredgecolor='white', markeredgewidth=0.5)

        panel_ax.set_xlabel('Simulation Step')
        panel_ax.set_ylabel(ylabel, fontweight='bold')
        panel_ax.set_title(title, fontweight='bold')
        add_intervention_zones(panel_ax, steps, alpha=0.10)
        add_intervention_labels(panel_ax, steps)

    ax1.legend(loc='lower left', framealpha=0.95, edgecolor='#cccccc',
               fontsize=8.5)

    fig.suptitle('Temporal Evolution of Emotion Dynamics Under Different PR Strategies',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save(fig, output_dir, 'fig2_absolute_temporal')


# ============================================================
# Figure 3: Cumulative Improvement Score
# ============================================================
def fig3_cumulative_improvement(agg, output_dir):
    """Cumulative improvement score over time — shows accumulated divergence."""
    fig, ax = plt.subplots(figsize=(10, 6))

    bl_ner = get_series(agg, 'baseline', 'negative_emotion_ratio')
    steps = np.arange(1, len(bl_ner) + 1)

    ax.axhline(y=0, color='#78909C', linewidth=2.0, alpha=0.5, linestyle='-')

    for s in ACTIVE_STRATEGIES:
        raw = get_series(agg, s, 'negative_emotion_ratio')
        diff = bl_ner[:len(raw)] - raw
        cumulative = np.cumsum(diff)

        ax.plot(steps[:len(cumulative)], cumulative,
                color=STRATEGY_COLORS[s], linewidth=2.5,
                linestyle='-', label=STRATEGY_LABELS[s],
                marker=STRATEGY_MARKERS[s],
                markevery=2, markersize=7,
                markeredgecolor='white', markeredgewidth=0.6)

        final_val = cumulative[-1]
        ax.annotate(f'{final_val:+.2f}', xy=(steps[len(cumulative)-1], final_val),
                    fontsize=9, fontweight='bold', color=STRATEGY_COLORS[s],
                    xytext=(5, 0), textcoords='offset points', va='center')

    add_intervention_zones(ax, steps, alpha=0.10)
    add_intervention_labels(ax, steps)

    ax.set_xlabel('Simulation Step', fontweight='bold')
    ax.set_ylabel('Cumulative NER Improvement\n(Positive = Better than Baseline)', fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.95, edgecolor='#cccccc', fontsize=9.5)
    ax.set_title('Cumulative Negative Emotion Reduction vs. Actual Response',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save(fig, output_dir, 'fig3_cumulative_improvement')


# ============================================================
# Figure 4: Strategy Comparison Bar Chart (Emotion Only)
# ============================================================
def fig4_strategy_comparison(agg, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    metrics = [
        ('negative_emotion_ratio', 'Negative Emotion Ratio'),
        ('anger_ratio',            'Anger Ratio'),
        ('aggression_index',       'Aggression Index'),
    ]

    x = np.arange(len(STRATEGY_ORDER))
    width = 0.6

    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx]
        vals, errs = [], []
        for s in STRATEGY_ORDER:
            cms = [r['cumulative_metrics'][metric] for r in agg[s]]
            vals.append(np.mean(cms))
            errs.append(np.std(cms))

        bars = ax.bar(x, vals, width, yerr=errs, capsize=4,
                      color=[STRATEGY_COLORS[s] for s in STRATEGY_ORDER],
                      edgecolor='white', linewidth=0.8, alpha=0.85)
        for i, (v, e) in enumerate(zip(vals, errs)):
            ax.text(i, v + e + 0.008, f'{v:.3f}', ha='center', va='bottom',
                    fontsize=8.5, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels([STRATEGY_LABELS[s] for s in STRATEGY_ORDER],
                           rotation=25, ha='right', fontsize=9)
        ax.set_ylabel(title, fontweight='bold')
        ax.set_ylim(0, max(vals) * 1.3)
        ax.text(-0.05, 1.05, PANEL_LABELS[idx], transform=ax.transAxes,
                fontsize=13, fontweight='bold', va='top')

    fig.suptitle('Cumulative Emotion Metrics Comparison Across PR Strategies',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save(fig, output_dir, 'fig4_strategy_comparison')


# ============================================================
# Figure 5: Heatmap (Emotion Metrics Only)
# ============================================================
def fig5_heatmap(agg, output_dir):
    metrics = ['negative_emotion_ratio', 'anger_ratio',
               'mean_emotion_intensity', 'aggression_index', 'positive_ratio']
    metric_labels = ['Negative\nEmotion', 'Anger\nRatio', 'Emotion\nIntensity',
                     'Aggression\nIndex', 'Positive\nRatio']

    baseline_vals = {}
    for m in metrics:
        baseline_vals[m] = np.mean([r['cumulative_metrics'].get(m, 0) for r in agg['baseline']])

    strategies = ['swift_apology', 'transparency', 'dialogue', 'silence']
    strat_labels = [STRATEGY_LABELS[s] for s in strategies]

    data = np.zeros((len(strategies), len(metrics)))
    for i, s in enumerate(strategies):
        for j, m in enumerate(metrics):
            val = np.mean([r['cumulative_metrics'].get(m, 0) for r in agg[s]])
            bv = baseline_vals[m]
            if abs(bv) > 0.001:
                data[i, j] = (val - bv) / abs(bv) * 100
            else:
                data[i, j] = 0

    fig, ax = plt.subplots(figsize=(10, 5))
    vmax = max(abs(data.min()), abs(data.max()), 1)
    im = ax.imshow(data, cmap='RdYlGn_r', aspect='auto', vmin=-vmax, vmax=vmax)

    for i in range(len(strategies)):
        for j in range(len(metrics)):
            val = data[i, j]
            color = 'white' if abs(val) > vmax * 0.55 else 'black'
            ax.text(j, i, f'{val:+.1f}%', ha='center', va='center',
                    fontsize=11, fontweight='bold', color=color)

    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_yticks(np.arange(len(strategies)))
    ax.set_yticklabels(strat_labels, fontsize=11)
    plt.colorbar(im, ax=ax, shrink=0.8, label='Change vs. Baseline (%)')
    ax.set_title('Percentage Change in Emotion Metrics\nRelative to Actual Response (Baseline)',
                 fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    _save(fig, output_dir, 'fig5_heatmap')


# ============================================================
# Figure 6: Radar Chart (Emotion-Focused)
# ============================================================
def fig6_radar(agg, output_dir):
    categories = ['Neg. Emotion\nReduction', 'Anger\nReduction',
                  'Aggression\nReduction', 'Emotion Intensity\nReduction',
                  'Positive Emotion\nIncrease']
    metrics = ['negative_emotion_ratio', 'anger_ratio',
               'aggression_index', 'mean_emotion_intensity', 'positive_ratio']

    baseline_vals = {}
    for m in metrics:
        baseline_vals[m] = np.mean([r['cumulative_metrics'].get(m, 0) for r in agg['baseline']])

    strategies = ACTIVE_STRATEGIES
    n = len(categories)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for s in strategies:
        vals = []
        for m in metrics:
            sv = np.mean([r['cumulative_metrics'].get(m, 0) for r in agg[s]])
            bv = baseline_vals[m]
            if m == 'positive_ratio':
                improvement = (sv - bv) / max(0.01, bv) * 100
            else:
                improvement = (bv - sv) / max(0.01, bv) * 100
            vals.append(max(0, min(100, improvement)))
        vals += vals[:1]

        ax.plot(angles, vals, 'o-', linewidth=2.2, color=STRATEGY_COLORS[s],
                label=STRATEGY_LABELS[s], markersize=7,
                markeredgecolor='white', markeredgewidth=0.5)
        ax.fill(angles, vals, alpha=0.06, color=STRATEGY_COLORS[s])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 80)
    ax.set_yticks([20, 40, 60, 80])
    ax.set_yticklabels(['20%', '40%', '60%', '80%'], fontsize=8, alpha=0.6)
    ax.legend(loc='upper right', bbox_to_anchor=(1.30, 1.12),
              framealpha=0.9, edgecolor='#cccccc')
    ax.set_title('Multi-Dimensional Strategy Effectiveness\n(% Improvement over Baseline)',
                 fontsize=12, fontweight='bold', pad=30)
    plt.tight_layout()
    _save(fig, output_dir, 'fig6_radar_chart')


# ============================================================
# Figure 7: Final Emotion Distribution
# ============================================================
def fig7_emotion_distribution(agg, output_dir):
    emotions = ['happiness', 'sadness', 'anger', 'fear', 'surprise', 'disgust']
    emo_labels = ['Happiness', 'Sadness', 'Anger', 'Fear', 'Surprise', 'Disgust']
    emo_colors = ['#43A047', '#1E88E5', '#E53935', '#8E24AA', '#FB8C00', '#6D4C41']

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(emotions))
    n_strat = len(STRATEGY_ORDER)
    width = 0.15
    offsets = np.arange(n_strat) - n_strat / 2 + 0.5

    for idx, s in enumerate(STRATEGY_ORDER):
        means = []
        for emo in emotions:
            emo_vals = []
            for run in agg[s]:
                fe = run.get('final_emotions', {})
                for uid, ev in fe.items():
                    emo_vals.append(ev.get(emo, 0))
            means.append(np.mean(emo_vals) if emo_vals else 0)

        ax.bar(x + offsets[idx] * width, means, width * 0.9,
               label=STRATEGY_LABELS[s], color=STRATEGY_COLORS[s],
               edgecolor='white', linewidth=0.5, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(emo_labels, fontsize=11)
    ax.set_ylabel('Mean Emotion Intensity', fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9, ncol=2, fontsize=9,
              edgecolor='#cccccc')
    ax.set_title('Final Agent Emotion Distribution Across PR Strategies',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save(fig, output_dir, 'fig7_emotion_distribution')


# ============================================================
# Figure 8: Strategy Scorecard
# ============================================================
def fig8_scorecard(agg, output_dir):
    metrics = {
        'negative_emotion_ratio': ('Negative Emotion', 'lower_better'),
        'anger_ratio':            ('Anger', 'lower_better'),
        'aggression_index':       ('Aggression', 'lower_better'),
        'mean_emotion_intensity': ('Emotion Intensity', 'lower_better'),
        'positive_ratio':         ('Positive Emotion', 'higher_better'),
    }

    baseline_vals = {}
    for m in metrics:
        baseline_vals[m] = np.mean([r['cumulative_metrics'].get(m, 0) for r in agg['baseline']])

    strategies = ACTIVE_STRATEGIES
    strat_labels = [STRATEGY_LABELS[s] for s in strategies]
    m_labels = [v[0] for v in metrics.values()]

    scores = np.zeros((len(strategies), len(metrics)))
    for i, s in enumerate(strategies):
        for j, (m, (_, direction)) in enumerate(metrics.items()):
            sv = np.mean([r['cumulative_metrics'].get(m, 0) for r in agg[s]])
            bv = baseline_vals[m]
            if direction == 'lower_better':
                score = max(0, min(100, (bv - sv) / max(0.01, bv) * 100))
            else:
                score = max(0, min(100, (sv - bv) / max(0.01, bv) * 100))
            scores[i, j] = score

    x = np.arange(len(metrics))
    width = 0.18
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, s in enumerate(strategies):
        offset = (i - len(strategies) / 2 + 0.5) * width
        ax.bar(x + offset, scores[i], width,
               label=strat_labels[i], color=STRATEGY_COLORS[s],
               edgecolor='white', linewidth=0.5, alpha=0.85)
        for k, v in enumerate(scores[i]):
            if v > 3:
                ax.text(x[k] + offset, v + 1, f'{v:.0f}', ha='center',
                        va='bottom', fontsize=7.5, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(m_labels, fontsize=10)
    ax.set_ylabel('Improvement Score (% over Baseline)', fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9, ncol=2, edgecolor='#cccccc')
    ax.set_title('Strategy Effectiveness Scorecard (Higher = Better)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save(fig, output_dir, 'fig8_scorecard')


# ============================================================
# Main
# ============================================================
def main():
    script_dir = Path(__file__).parent
    if len(sys.argv) > 1:
        latest = Path(sys.argv[1])
    else:
        output_dirs = sorted(script_dir.glob('output/experiment_*'))
        if not output_dirs:
            print("No experiment results found!")
            return
        latest = output_dirs[-1]

    summary_path = latest / 'experiment_summary.json'
    print(f"Loading: {summary_path}")
    data, agg = load_data(str(summary_path))

    fig_dir = latest / 'figures'
    fig_dir.mkdir(exist_ok=True)
    print(f"Output: {fig_dir}\n")

    fig1_intervention_effect(agg, fig_dir)
    fig2_absolute_temporal(agg, fig_dir)
    fig3_cumulative_improvement(agg, fig_dir)
    fig4_strategy_comparison(agg, fig_dir)
    fig5_heatmap(agg, fig_dir)
    fig6_radar(agg, fig_dir)
    fig7_emotion_distribution(agg, fig_dir)
    fig8_scorecard(agg, fig_dir)

    print(f"\nAll figures saved to: {fig_dir}")


if __name__ == '__main__':
    main()
