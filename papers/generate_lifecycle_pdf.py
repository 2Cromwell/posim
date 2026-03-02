#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate fig_lifecycle_paper.pdf for paper inclusion."""
import json, os, warnings
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MICRO = os.path.join(
    PROJECT_ROOT, 'scripts', 'xibeiyuzhicai', 'output',
    'xibeiyuzhicai_baseline_20260223_145442_14B效果不错',
    'simulation_results', 'micro_results.json'
)
OUT_DIR = os.path.join(SCRIPT_DIR, 'figures')

_CJK = ['SimHei', 'Microsoft YaHei', 'STSong', 'Noto Sans CJK SC']
_avail = set(f.name for f in fm.fontManager.ttflist)
_flist = [f for f in _CJK if f in _avail] + ['DejaVu Sans', 'Arial', 'Helvetica']
_serif_list = ['Times New Roman', 'DejaVu Serif']

SIM_START = datetime(2025, 9, 14, 5, 0)
ROUND_MIN = 10
MAX_ROUND = 426

EVENTS = [
    (18, '$E_1$'), (42, '$E_2$'), (62, '$E_3$'), (105, '$E_4$'),
    (162, '$E_5$'), (192, '$E_6$'), (198, '$E_7$'),
]
ECOLS = ['#c0392b', '#e67e22', '#27ae60', '#2980b9', '#8e44ad', '#795548', '#e84393']

PHASES = [
    ('Incubation', 0, 15, '#cfe2f3'),
    ('Outbreak', 15, 42, '#9fc5e8'),
    ('Plateau', 42, 102, '#b6d7a8'),
    ('Revival', 102, 150, '#f4cccc'),
    ('2nd Peak', 150, 204, '#ea9999'),
    ('Decline', 204, 282, '#d9d2e9'),
    ('Long Tail', 282, 426, '#ead1dc'),
]


def main():
    print("Loading data...")
    with open(MICRO, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    rounds = []
    for it in raw:
        try:
            dt = datetime.strptime(it['time'], '%Y-%m-%dT%H:%M')
            r = (dt - SIM_START).total_seconds() / (ROUND_MIN * 60)
            rounds.append(int(r))
        except Exception:
            continue
    df = pd.DataFrame({'round': rounds})
    print(f"  {len(df)} records")

    df['rbin2'] = (df['round'] // 2 * 2).astype(int)
    vol = df.groupby('rbin2').size().reindex(range(0, MAX_ROUND + 2, 2), fill_value=0)

    vol1 = df.groupby('round').size().reindex(range(0, MAX_ROUND + 1), fill_value=0)
    cum_pct = np.cumsum(vol1.values) / vol1.values.sum() * 100

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': _serif_list,
        'font.size': 16,
        'axes.unicode_minus': False,
        'mathtext.default': 'regular',
        'axes.linewidth': 1.2,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
    })

    fig = plt.figure(figsize=(14, 6.2))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 18],
                           hspace=0.0, left=0.07, right=0.91,
                           top=0.94, bottom=0.10)

    ax_top = fig.add_subplot(gs[0])
    ax_top.set_xlim(-5, MAX_ROUND + 6)
    ax_top.set_ylim(0, 1)
    ax_top.axis('off')

    PHASE_DISPLAY = {
        'Incubation': ('Incub.', 'left', 2),
        'Outbreak': ('Outbreak', 'left', 17),
    }
    for name, rs, re, col in PHASES:
        ax_top.axvspan(rs, re, facecolor=col, alpha=0.65, edgecolor='white', lw=1.5)
        display, ha, x_pos = PHASE_DISPLAY.get(name, (name, 'center', (rs + re) / 2))
        fs = 10.5 if (re - rs) < 50 else 12
        ax_top.text(x_pos, 0.5, display, ha=ha, va='center',
                    fontsize=fs, fontweight='bold', color='#333')

    ax1 = fig.add_subplot(gs[1])

    for _, rs, re, col in PHASES:
        ax1.axvspan(rs, re, facecolor=col, alpha=0.10, zorder=0, edgecolor='none')
    for _, rs, _, _ in PHASES[1:]:
        ax1.axvline(x=rs, color='#bbb', linestyle='--', lw=0.5, alpha=0.5, zorder=1)

    ax1.bar(vol.index, vol.values, width=1.6, color='#7bafd4',
            alpha=0.82, edgecolor='none', zorder=2)
    ax1.set_ylabel('Post Volume', fontsize=20, fontweight='bold', labelpad=8)
    ax1.set_xlabel('Simulation Round', fontsize=20, fontweight='bold', labelpad=8)
    ax1.tick_params(axis='both', labelsize=14, width=1.0)
    ax1.set_xlim(-5, MAX_ROUND + 6)
    y_max = vol.values.max()
    ax1.set_ylim(0, y_max * 1.15)
    ax1.grid(axis='y', alpha=0.15, linestyle=':', zorder=0)

    ax2 = ax1.twinx()
    ax2.plot(range(MAX_ROUND + 1), cum_pct, color='#1f77b4', lw=3.0, zorder=4,
             solid_capstyle='round')
    ax2.set_ylabel('Cumulative Posts (%)', fontsize=20, fontweight='bold',
                    color='#1f77b4', labelpad=8)
    ax2.tick_params(axis='y', labelsize=14, colors='#1f77b4', width=1.0)
    ax2.set_ylim(-2, 108)
    ax2.spines['right'].set_color('#1f77b4')

    label_offsets = []
    for (rd, label), col in zip(EVENTS, ECOLS):
        ax1.axvline(x=rd, color=col, linestyle=':', lw=1.5, alpha=0.72, zorder=5)
        y_frac = 0.96
        for prev_rd, prev_y in label_offsets:
            if abs(rd - prev_rd) < 22 and abs(y_frac - prev_y) < 0.065:
                y_frac = prev_y - 0.07
        label_offsets.append((rd, y_frac))
        ax1.text(rd, y_frac, label, ha='center', va='top', fontsize=14,
                 fontweight='bold', color=col,
                 transform=ax1.get_xaxis_transform(),
                 clip_on=False, zorder=6)

    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)

    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ['pdf', 'png']:
        out_path = os.path.join(OUT_DIR, f'fig_lifecycle_paper.{ext}')
        fig.savefig(out_path, dpi=600, bbox_inches='tight', pad_inches=0.05,
                    facecolor='white', edgecolor='none')
        print(f"  -> {out_path}")
    plt.close(fig)


if __name__ == '__main__':
    main()
