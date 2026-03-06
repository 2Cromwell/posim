#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized paper figure for PR Strategy Comparison.
- Removed raw noisy lines, only show smoothed curves
- Made baseline more visible with bold dark line
- Cleaner overall style
"""
import sys, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.20,
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
    'baseline':      '#808080',
    'swift_apology': '#3B4892',
    'transparency':  '#631779',
    'dialogue':      '#F00002',
    'silence':       '#2E7D32',
}
STRATEGY_MARKERS = {
    'baseline': 'o', 'swift_apology': 's', 'transparency': '^',
    'dialogue': 'D', 'silence': 'v',
}
STRATEGY_LINESTYLES = {
    'baseline': '-', 'swift_apology': '-', 'transparency': '--',
    'dialogue': '-.', 'silence': (0, (3, 1.5)),
}
STRATEGY_LINEWIDTHS = {
    'baseline': 2.8, 'swift_apology': 2.0, 'transparency': 2.0,
    'dialogue': 2.0, 'silence': 2.0,
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
                    fontsize=9, color='#37474F', fontweight='bold',
                    ha='center', va='top',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor='#90A4AE', alpha=0.9))


def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    exp_base = project_root / 'scripts' / 'xibeiyuzhicai' / '反事实干预实验' / 'output'
    dirs = sorted([d for d in exp_base.iterdir()
                   if d.is_dir() and d.name.startswith('experiment_llm')])
    output_dir = dirs[-1]

    data, agg = load_data(output_dir / 'experiment_summary.json')

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5))

    metric = 'negative_emotion_ratio'

    # Panel (a): Absolute NER — only smoothed lines, no raw noisy lines
    for s in STRATEGY_ORDER:
        raw = get_series(agg, s, metric)
        smoothed = rolling_mean(raw, window=3)
        steps = np.arange(1, len(raw) + 1)

        ax_a.plot(steps, smoothed, color=STRATEGY_COLORS[s],
                  linewidth=STRATEGY_LINEWIDTHS[s],
                  linestyle=STRATEGY_LINESTYLES[s],
                  label=STRATEGY_LABELS[s],
                  marker=STRATEGY_MARKERS[s],
                  markevery=2, markersize=5,
                  markeredgecolor='white', markeredgewidth=0.6,
                  zorder=5 if s == 'baseline' else 3)

    ax_a.set_xlabel('Simulation Step', fontsize=15, fontweight='bold')
    ax_a.set_ylabel('Negative Emotion Ratio', fontsize=15, fontweight='bold')
    add_intervention_zones(ax_a, steps, alpha=0.10)
    add_intervention_labels(ax_a, steps)
    ax_a.legend(loc='lower left', framealpha=0.95, edgecolor='#cccccc',
                fontsize=11, fancybox=True, borderpad=0.4)
    ax_a.text(-0.08, 1.05, '(a)', transform=ax_a.transAxes,
              fontsize=15, fontweight='bold', va='top')
    ax_a.set_title('Absolute NER Under Different Strategies', fontsize=15)

    # Panel (b): NER relative to baseline — only smoothed lines
    bl = get_series(agg, 'baseline', metric)
    steps = np.arange(1, len(bl) + 1)

    ax_b.axhline(y=0, color='#808080', linewidth=2.5, alpha=0.7,
                 linestyle='-', label='Baseline (zero)', zorder=4)

    for s in ACTIVE_STRATEGIES:
        raw = get_series(agg, s, metric)
        diff_raw = raw - bl[:len(raw)]
        diff_smooth = rolling_mean(diff_raw, window=3)

        ax_b.plot(steps[:len(diff_smooth)], diff_smooth,
                  color=STRATEGY_COLORS[s],
                  linewidth=STRATEGY_LINEWIDTHS[s] + 0.3,
                  linestyle=STRATEGY_LINESTYLES[s],
                  label=STRATEGY_LABELS[s],
                  marker=STRATEGY_MARKERS[s],
                  markevery=2, markersize=6,
                  markeredgecolor='white', markeredgewidth=0.6,
                  zorder=3)

    ax_b.set_xlabel('Simulation Step', fontsize=15, fontweight='bold')
    ax_b.set_ylabel('NER Difference from Baseline', fontsize=15, fontweight='bold')
    add_intervention_zones(ax_b, steps, alpha=0.10)
    add_intervention_labels(ax_b, steps)
    ax_b.legend(loc='lower left', framealpha=0.95, edgecolor='#cccccc',
                fontsize=11, fancybox=True, borderpad=0.4)
    ax_b.text(-0.08, 1.05, '(b)', transform=ax_b.transAxes,
              fontsize=15, fontweight='bold', va='top')
    ax_b.set_title('NER Relative to Actual Response', fontsize=15)

    plt.tight_layout()

    out_dir = script_dir / 'els-cas-templates_IPM' / 'figures'
    out_dir.mkdir(exist_ok=True)
    for ext in ['pdf', 'png']:
        out = out_dir / f'paper_strategy_comparison.{ext}'
        fig.savefig(str(out), dpi=300, bbox_inches='tight')
        print(f'  -> {out}')
    plt.close(fig)


if __name__ == '__main__':
    main()
