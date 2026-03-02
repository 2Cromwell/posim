"""Generate combined paper figure for PR Strategy Comparison experiment.
(a) Absolute NER temporal evolution with intervention zones (all strategies)
(b) NER relative to baseline (active strategies only)
"""
import sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

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


def rolling_mean(data, window=3):
    arr = np.array(data, dtype=float)
    n = len(arr)
    result = np.zeros(n)
    hw = window // 2
    for i in range(n):
        result[i] = np.mean(arr[max(0, i - hw):min(n, i + hw + 1)])
    return result


def rolling_std(data, window=5):
    arr = np.array(data, dtype=float)
    n = len(arr)
    result = np.zeros(n)
    hw = window // 2
    for i in range(n):
        result[i] = np.std(arr[max(0, i - hw):min(n, i + hw + 1)])
    return result


def load_data(summary_path):
    with open(summary_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    agg = defaultdict(list)
    for r in data['results']:
        agg[r['strategy']].append(r)
    return data, agg


def get_series(agg, strategy, metric):
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
    for istep in INTERVENTION_STEPS:
        ax.axvline(x=istep, color='#37474F', linewidth=1.0,
                   linestyle='--', alpha=0.5, zorder=1)
        ax.axvspan(istep, min(istep + 3, steps[-1]),
                   alpha=alpha, color='#B3E5FC', zorder=0)


def add_intervention_labels(ax, steps):
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    for idx, istep in enumerate(INTERVENTION_STEPS):
        label = f'Strategy\nDeployment {idx+1}'
        ax.annotate(label, xy=(istep, ymax - span * 0.03),
                    fontsize=7, color='#37474F', fontweight='bold',
                    ha='center', va='top',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor='#90A4AE', alpha=0.9))


def main():
    script_dir = Path(__file__).parent
    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])
    else:
        output_base = script_dir / 'output'
        dirs = sorted([d for d in output_base.iterdir()
                       if d.is_dir() and d.name.startswith('experiment_llm')])
        output_dir = dirs[-1]

    data, agg = load_data(output_dir / 'experiment_summary.json')
    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5))

    # ── Panel (a): Absolute NER temporal with intervention zones ──
    metric = 'negative_emotion_ratio'
    for s in STRATEGY_ORDER:
        raw = get_series(agg, s, metric)
        smoothed = rolling_mean(raw, window=3)
        local_s = rolling_std(raw, window=5) * 0.5
        steps = np.arange(1, len(raw) + 1)

        ax_a.fill_between(steps, smoothed - local_s, smoothed + local_s,
                          alpha=0.08, color=STRATEGY_COLORS[s], linewidth=0)
        ax_a.plot(steps, raw, color=STRATEGY_COLORS[s], alpha=0.20, linewidth=0.8)
        ax_a.plot(steps, smoothed, color=STRATEGY_COLORS[s],
                  linewidth=2.0, linestyle=STRATEGY_LINESTYLES[s],
                  label=STRATEGY_LABELS[s],
                  marker=STRATEGY_MARKERS[s],
                  markevery=2, markersize=5,
                  markeredgecolor='white', markeredgewidth=0.5)

    ax_a.set_xlabel('Simulation Step', fontweight='bold')
    ax_a.set_ylabel('Negative Emotion Ratio', fontweight='bold')
    add_intervention_zones(ax_a, steps, alpha=0.10)
    add_intervention_labels(ax_a, steps)
    ax_a.legend(loc='lower left', framealpha=0.95, edgecolor='#cccccc',
                fontsize=8, fancybox=True, borderpad=0.4)
    ax_a.text(-0.08, 1.05, '(a)', transform=ax_a.transAxes,
              fontsize=15, fontweight='bold', va='top')
    ax_a.set_title('Absolute NER Under Different Strategies', fontsize=11)

    # ── Panel (b): NER relative to baseline ──
    bl = get_series(agg, 'baseline', metric)
    steps = np.arange(1, len(bl) + 1)

    ax_b.axhline(y=0, color='#78909C', linewidth=2.0, alpha=0.6,
                 linestyle='-', label='Baseline (zero)', zorder=2)

    for s in ACTIVE_STRATEGIES:
        raw = get_series(agg, s, metric)
        diff_raw = raw - bl[:len(raw)]
        diff_smooth = rolling_mean(diff_raw, window=3)

        ax_b.plot(steps[:len(diff_raw)], diff_raw,
                  color=STRATEGY_COLORS[s], alpha=0.25, linewidth=0.8)
        ax_b.plot(steps[:len(diff_smooth)], diff_smooth,
                  color=STRATEGY_COLORS[s], linewidth=2.5,
                  linestyle=STRATEGY_LINESTYLES[s],
                  label=STRATEGY_LABELS[s],
                  marker=STRATEGY_MARKERS[s],
                  markevery=2, markersize=6,
                  markeredgecolor='white', markeredgewidth=0.6)

    ax_b.set_xlabel('Simulation Step', fontweight='bold')
    ax_b.set_ylabel('NER Difference from Baseline', fontweight='bold')
    add_intervention_zones(ax_b, steps, alpha=0.10)
    add_intervention_labels(ax_b, steps)
    ax_b.legend(loc='lower left', framealpha=0.95, edgecolor='#cccccc',
                fontsize=8, fancybox=True, borderpad=0.4)
    ax_b.text(-0.08, 1.05, '(b)', transform=ax_b.transAxes,
              fontsize=15, fontweight='bold', va='top')
    ax_b.set_title('NER Relative to Actual Response', fontsize=11)

    fig.text(0.5, 0.01,
             'Negative values in (b) indicate improvement over baseline  |  '
             'Shaded zones = post-strategy-deployment periods',
             ha='center', fontsize=9, style='italic', color='#546E7A')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    out_path = fig_dir / 'paper_strategy_comparison.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    fig.savefig(fig_dir / 'paper_strategy_comparison.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'[OK] {out_path}')


if __name__ == '__main__':
    main()
