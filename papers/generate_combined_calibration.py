#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper Figure: Three-event calibration — hotness curve + behavior distribution.
Layout: 3 rows × 2 columns, width ratio ~2:1, compact with (a)(b)(c) labels.
"""
import json, os, warnings
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.dates as mdates
import numpy as np
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUT_DIR = os.path.join(SCRIPT_DIR, 'figures')

SIM_TYPE_MAP = {
    'short_post': 'original', 'long_post': 'original',
    'repost': 'repost', 'repost_comment': 'repost',
    'short_comment': 'comment', 'long_comment': 'comment',
}
REAL_TYPE_MAP = {'original': 'original', 'repost': 'repost', 'comment': 'comment'}

C_SIM = '#5B9BD5'
C_REAL = '#ED7D31'

EVENTS = [
    {
        'name': 'Luxury-Earring',
        'tag': '(a)',
        'sim_dir': os.path.join(PROJECT_ROOT, 'scripts', 'tianjiaerhuan', 'output',
                                'tianjiaerhuan_baseline_20260221_152957_14B效果好', 'simulation_results'),
        'real_path': os.path.join(PROJECT_ROOT, 'data_process', 'tianjaierhuan', 'output', 'labels.json'),
        'start': '2025-05-16T08:00', 'end': '2025-05-18T06:00', 'granularity': 10,
    },
    {
        'name': 'WHU-Library',
        'tag': '(b)',
        'sim_dir': os.path.join(PROJECT_ROOT, 'scripts', 'wudatushuguan', 'output',
                                'wudatushuguan_baseline_20260221_021403_14B_行为分布好', 'simulation_results'),
        'real_path': os.path.join(PROJECT_ROOT, 'data_process', 'wudatushuguan', 'output', 'labels.json'),
        'start': '2025-07-27T08:00', 'end': '2025-08-04T06:00', 'granularity': 10,
    },
    {
        'name': 'Xibei-Food',
        'tag': '(c)',
        'sim_dir': os.path.join(PROJECT_ROOT, 'scripts', 'xibeiyuzhicai', 'output',
                                'xibeiyuzhicai_baseline_20260223_145442_14B效果不错', 'simulation_results'),
        'real_path': os.path.join(PROJECT_ROOT, 'data_process', 'xibeiyuzhicai', 'output', 'labels.json'),
        'start': '2025-09-14 05:00:00', 'end': '2025-09-17 04:00:00', 'granularity': 10,
    },
]


def parse_time(t_str):
    for fmt in ('%Y-%m-%dT%H:%M', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M'):
        try:
            return datetime.strptime(t_str, fmt)
        except ValueError:
            continue
    return None


def truncate_time(dt, granularity_min):
    mins = dt.hour * 60 + dt.minute
    truncated = (mins // granularity_min) * granularity_min
    return dt.replace(hour=truncated // 60, minute=truncated % 60, second=0, microsecond=0)


def load_sim_data(sim_dir, start_str, end_str, granularity):
    micro_path = os.path.join(sim_dir, 'micro_results.json')
    with open(micro_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    t_start, t_end = parse_time(start_str), parse_time(end_str)
    time_bins = defaultdict(int)
    type_counts = Counter()
    for item in data:
        t = parse_time(item.get('time', ''))
        if not t or t < t_start or t > t_end:
            continue
        time_bins[truncate_time(t, granularity)] += 1
        mapped = SIM_TYPE_MAP.get(item.get('action_type', ''))
        if mapped:
            type_counts[mapped] += 1
    all_times = []
    cur = t_start
    while cur <= t_end:
        all_times.append(cur)
        cur += timedelta(minutes=granularity)
    return all_times, [time_bins.get(t, 0) for t in all_times], type_counts


def load_real_data(real_path, start_str, end_str, granularity):
    with open(real_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    t_start, t_end = parse_time(start_str), parse_time(end_str)
    time_bins = defaultdict(int)
    type_counts = Counter()
    for item in data:
        t = parse_time(item.get('time', ''))
        if not t or t < t_start or t > t_end:
            continue
        time_bins[truncate_time(t, granularity)] += 1
        mapped = REAL_TYPE_MAP.get(item.get('type', ''))
        if mapped:
            type_counts[mapped] += 1
    all_times = []
    cur = t_start
    while cur <= t_end:
        all_times.append(cur)
        cur += timedelta(minutes=granularity)
    return all_times, [time_bins.get(t, 0) for t in all_times], type_counts


def smooth_curve(data, window=5):
    arr = np.array(data, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode='same')


def main():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 7.5,
        'ytick.labelsize': 8,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    fig, axes = plt.subplots(3, 2, figsize=(10, 8.5),
                              gridspec_kw={'width_ratios': [2.2, 1],
                                           'hspace': 0.38, 'wspace': 0.18,
                                           'left': 0.08, 'right': 0.97,
                                           'top': 0.97, 'bottom': 0.06})

    type_order = ['original', 'repost', 'comment']
    bar_colors_sim = ['#5B9BD5', '#5B9BD5', '#5B9BD5']
    bar_colors_real = ['#ED7D31', '#ED7D31', '#ED7D31']

    for row_idx, evt in enumerate(EVENTS):
        print(f"Processing {evt['name']}...")
        sim_times, sim_hot, sim_types = load_sim_data(
            evt['sim_dir'], evt['start'], evt['end'], evt['granularity'])
        real_times, real_hot, real_types = load_real_data(
            evt['real_path'], evt['start'], evt['end'], evt['granularity'])

        ax_hot = axes[row_idx, 0]
        sim_smooth = smooth_curve(sim_hot, window=5)
        real_smooth = smooth_curve(real_hot, window=5)

        ax_hot.fill_between(sim_times, sim_smooth, alpha=0.20, color=C_SIM, linewidth=0)
        ax_hot.plot(sim_times, sim_smooth, color=C_SIM, linewidth=1.6,
                    label='Simulation', solid_capstyle='round')
        ax_hot.plot(real_times[:len(real_smooth)], real_smooth,
                    color=C_REAL, linewidth=1.6, linestyle='--',
                    label='Real Data', solid_capstyle='round')

        ax_hot.set_ylabel('Activity Count', fontsize=9, fontweight='bold')
        if row_idx == 2:
            ax_hot.set_xlabel('Time', fontsize=9, fontweight='bold')

        ax_hot.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax_hot.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=5))
        plt.setp(ax_hot.xaxis.get_majorticklabels(), rotation=25, ha='right', fontsize=6.5)
        ax_hot.grid(axis='y', alpha=0.15, linestyle='-', linewidth=0.3)

        ax_hot.text(0.02, 0.93, f'{evt["tag"]} {evt["name"]}',
                    transform=ax_hot.transAxes, fontsize=10, fontweight='bold', va='top')

        if row_idx == 0:
            ax_hot.legend(loc='upper right', fontsize=7.5, framealpha=0.9,
                          edgecolor='#ddd', handlelength=1.5)

        ax_beh = axes[row_idx, 1]
        sim_total = sum(sim_types.get(t, 0) for t in type_order)
        real_total = sum(real_types.get(t, 0) for t in type_order)
        sim_pcts = [sim_types.get(t, 0) / max(sim_total, 1) for t in type_order]
        real_pcts = [real_types.get(t, 0) / max(real_total, 1) for t in type_order]

        x = np.arange(len(type_order))
        bar_w = 0.30
        bars_sim = ax_beh.bar(x - bar_w / 2, sim_pcts, bar_w,
                              color=C_SIM, alpha=0.85, label='Simulation',
                              edgecolor='white', linewidth=0.5)
        bars_real = ax_beh.bar(x + bar_w / 2, real_pcts, bar_w,
                               color=C_REAL, alpha=0.85, label='Real Data',
                               edgecolor='white', linewidth=0.5)

        for bars in [bars_sim, bars_real]:
            for bar in bars:
                h = bar.get_height()
                ax_beh.text(bar.get_x() + bar.get_width() / 2, h + 0.008,
                            f'{h*100:.1f}%', ha='center', va='bottom', fontsize=6.5)

        ax_beh.set_xticks(x)
        ax_beh.set_xticklabels(type_order, fontsize=8)
        ax_beh.set_ylabel('Proportion', fontsize=9, fontweight='bold')
        ax_beh.set_ylim(0, max(max(sim_pcts), max(real_pcts)) * 1.22)
        ax_beh.grid(axis='y', alpha=0.15, linestyle='-', linewidth=0.3)

        if row_idx == 0:
            ax_beh.legend(loc='upper left', fontsize=7, framealpha=0.9,
                          edgecolor='#ddd', handlelength=1.2)

    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ['pdf', 'png']:
        out_path = os.path.join(OUT_DIR, f'fig_three_event_calibration.{ext}')
        fig.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.04)
        print(f'  -> {out_path}')
    plt.close(fig)
    print('Done.')


if __name__ == '__main__':
    main()
