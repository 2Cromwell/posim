# -*- coding: utf-8 -*-
"""Evaluation Module - Simulation Results Evaluation and Visualization (Publication Quality)"""
import json, logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

# ========== Publication-Quality Visualization Config ==========
FIG_SIZE = (12, 4)  # Single plot
FIG_SIZE_TALL = (12, 10)  # Multi-panel vertical
DPI = 300
LW = 1.8  # Line width
ALPHA = 0.25
SMOOTH_W = 5
FONT_SIZE = {'title': 14, 'label': 12, 'tick': 10, 'legend': 10}

# Color palettes
C_SIM = {'total': '#E74C3C', 'original': '#3498DB', 'repost': '#E67E22', 'comment': '#9B59B6'}
C_REAL = {'total': '#922B21', 'original': '#1A5276', 'repost': '#935116', 'comment': '#6C3483'}
C_EMOTION = {
    '愤怒': '#E74C3C', 'Anger': '#E74C3C',
    '悲伤': '#3498DB', 'Sadness': '#3498DB', 
    '惊奇': '#F39C12', 'Surprise': '#F39C12',
    '恐惧': '#8E44AD', 'Fear': '#8E44AD',
    '喜悦': '#27AE60', 'Joy': '#27AE60',
    '厌恶': '#7F8C8D', 'Disgust': '#7F8C8D',
    '中性': '#BDC3C7', 'Neutral': '#BDC3C7'
}
# Emotion name mapping (Chinese to English)
EMO_EN = {'愤怒': 'Anger', '悲伤': 'Sadness', '惊奇': 'Surprise', '恐惧': 'Fear',
          '喜悦': 'Joy', '厌恶': 'Disgust', '中性': 'Neutral'}

# Publication-quality rcParams
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'SimHei'],
    'axes.unicode_minus': False,
    'axes.linewidth': 1.2,
    'axes.labelsize': FONT_SIZE['label'],
    'axes.titlesize': FONT_SIZE['title'],
    'xtick.labelsize': FONT_SIZE['tick'],
    'ytick.labelsize': FONT_SIZE['tick'],
    'legend.fontsize': FONT_SIZE['legend'],
    'figure.dpi': DPI,
    'savefig.dpi': DPI,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# ========== Utility Functions ==========
def parse_time(s):
    if not s:
        return None
    # Try formats in order of most specific to least specific
    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M"]:
        try:
            parsed = datetime.strptime(s.strip(), fmt)
            return parsed
        except ValueError:
            continue
    return None

def truncate(dt, g): return dt.replace(minute=(dt.minute//g)*g, second=0, microsecond=0)
def smooth(d, w=SMOOTH_W): return np.convolve(d, np.ones(w)/w, mode='same') if w > 1 else np.array(d)
def normalize(d): 
    arr = np.array(d)
    max_val = arr.max()
    return arr / max_val if max_val > 0 else arr
def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f: json.dump(data, f, ensure_ascii=False, indent=2)


class SimulationEvaluator:
    """仿真结果评估器 - 支持与真实数据对比"""
    
    def __init__(self, results_dir: Path, real_data_path: str = None, 
                 time_granularity: int = 60, time_start: str = None, time_end: str = None):
        self.results_dir = Path(results_dir)
        self.vis_dir = self.results_dir.parent / "vis_results"
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        self.granularity = time_granularity
        self.time_start, self.time_end = time_start, time_end
        
        # 加载真实数据（如有）
        self.real_hotness, self.real_emotion, self.real_times = None, None, None
        if real_data_path and Path(real_data_path).exists():
            self._load_real_data(real_data_path)
    
    def _load_real_data(self, path: str):
        """加载并聚合真实数据"""
        with open(path, 'r', encoding='utf-8') as f: users = json.load(f)
        logger.info(f"加载真实数据: {len(users)} 用户")
        
        acts = []
        original_count, repost_count, comment_count = 0, 0, 0
        parse_fail_count = 0
        
        for u in users:
            for p in u.get('original_posts', []):
                original_count += 1
                t = parse_time(p.get('time', ''))
                if t: 
                    acts.append({'type': 'original', 'time': t, 'emotion': p.get('emotion', '中性')})
                else:
                    parse_fail_count += 1
            for p in u.get('repost_posts', []):
                repost_count += 1
                t = parse_time(p.get('time', ''))
                if t: 
                    acts.append({'type': 'repost', 'time': t, 'emotion': p.get('emotion', '中性')})
                else:
                    parse_fail_count += 1
            for c in u.get('comments', []):
                comment_count += 1
                t = parse_time(c.get('time', ''))
                if t: 
                    acts.append({'type': 'comment', 'time': t, 'emotion': c.get('emotion', '中性')})
                else:
                    parse_fail_count += 1
        
        logger.info(f"真实数据统计: 原创={original_count}, 转发={repost_count}, 评论={comment_count}")
        logger.info(f"时间解析失败: {parse_fail_count}, 成功解析: {len(acts)}")
        
        # 时间过滤
        if self.time_start:
            t_s = parse_time(self.time_start)
            if t_s: 
                before_filter = len(acts)
                acts = [a for a in acts if a['time'] >= t_s]
                logger.info(f"时间起始过滤: {before_filter} -> {len(acts)}")
        if self.time_end:
            t_e = parse_time(self.time_end)
            if t_e: 
                before_filter = len(acts)
                acts = [a for a in acts if a['time'] <= t_e]
                logger.info(f"时间结束过滤: {before_filter} -> {len(acts)}")
        
        # 聚合
        h_buckets = defaultdict(lambda: {'total': 0, 'original': 0, 'repost': 0, 'comment': 0})
        e_buckets = defaultdict(lambda: defaultdict(int))
        for a in acts:
            t = truncate(a['time'], self.granularity)
            h_buckets[t]['total'] += 1
            h_buckets[t][a['type']] += 1
            e_buckets[t][a['emotion']] += 1
        
        if not h_buckets: return
        times = sorted(h_buckets.keys())
        all_times, cur = [], times[0]
        while cur <= times[-1]:
            all_times.append(cur)
            cur += timedelta(minutes=self.granularity)
        
        self.real_times = all_times
        self.real_hotness = {k: [h_buckets[t].get(k, 0) for t in all_times] for k in ['total', 'original', 'repost', 'comment']}
        self.real_emotion = {e: [e_buckets[t].get(e, 0) for t in all_times] for e in set(a['emotion'] for a in acts)}
        logger.info(f"真实数据: {len(acts)} 活动, {len(all_times)} 时间点")
    
    def run_evaluation(self, macro_results: Dict, micro_results: List[Dict], 
                       social_network=None) -> Dict[str, Any]:
        """运行评估流程"""
        logger.info("开始仿真评估...")
        
        # 聚合仿真数据
        sim_times, sim_hotness, sim_emotion = self._aggregate_sim_data(micro_results)
        
        # 生成对比可视化
        self._plot_hotness_comparison(sim_times, sim_hotness)
        self._plot_emotion_comparison(sim_times, sim_emotion)
        self._plot_activity_curve(macro_results)
        
        # 保存统计
        results = {
            'total_actions': macro_results.get('stats', {}).get('total_actions', 0),
            'sim_peak': max(sim_hotness.get('total', [])) if sim_hotness.get('total') else 0,
            'real_peak': max(self.real_hotness.get('total', [])) if self.real_hotness and self.real_hotness.get('total') else 0,
        }
        save_json(self.vis_dir / 'evaluation_results.json', results)
        logger.info(f"评估完成: {self.vis_dir}")
        return results
    
    def _aggregate_sim_data(self, micro_results: List[Dict]):
        """聚合仿真数据"""
        h_buckets = defaultdict(lambda: {'total': 0, 'original': 0, 'repost': 0, 'comment': 0})
        e_buckets = defaultdict(lambda: defaultdict(int))
        
        type_map = {'short_post': 'original', 'long_post': 'original',
                    'repost': 'repost', 'repost_comment': 'repost',
                    'short_comment': 'comment', 'long_comment': 'comment', 'like': 'comment'}
        
        for a in micro_results:
            t = parse_time(a.get('time', ''))
            if not t: continue
            t = truncate(t, self.granularity)
            atype = type_map.get(a.get('action_type', ''), 'comment')
            h_buckets[t]['total'] += 1
            h_buckets[t][atype] += 1
            e_buckets[t][a.get('emotion', '中性')] += 1
        
        if not h_buckets: return [], {}, {}
        times = sorted(h_buckets.keys())
        all_times, cur = [], times[0]
        while cur <= times[-1]:
            all_times.append(cur)
            cur += timedelta(minutes=self.granularity)
        
        hotness = {k: [h_buckets[t].get(k, 0) for t in all_times] for k in ['total', 'original', 'repost', 'comment']}
        emotion = {e: [e_buckets[t].get(e, 0) for t in all_times] for e in set(a.get('emotion', '中性') for a in micro_results)}
        return all_times, hotness, emotion
    
    # ========== Hotness Comparison Plots ==========
    def _plot_hotness_comparison(self, sim_times, sim_hotness):
        """Hotness comparison: Total + By category, both absolute and normalized"""
        has_sim = bool(sim_times)
        has_real = bool(self.real_hotness and self.real_times)
        if not has_sim and not has_real:
            logger.warning("[SKIP] cmp_hotness_*.png - No simulation or real data")
            return
        
        labels_en = {'total': 'Total', 'original': 'Original', 'repost': 'Repost', 'comment': 'Comment'}
        
        # === 1. Total Hotness (Absolute) ===
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        if has_sim:
            ax.plot(sim_times, smooth(sim_hotness.get('total', [])), color=C_SIM['total'], lw=LW, label='Simulation')
        if has_real:
            ax.plot(self.real_times, smooth(self.real_hotness.get('total', [])), color=C_REAL['total'], 
                   lw=LW, ls='--', label='Real Data')
        ax.set_xlabel('Time'); ax.set_ylabel('Activity Count')
        ax.set_title('Total Activity Comparison', fontweight='bold')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        plt.xticks(rotation=45); ax.legend(frameon=True, fancybox=False, edgecolor='black')
        ax.grid(alpha=0.3, linestyle=':'); plt.tight_layout()
        plt.savefig(self.vis_dir / 'hotness_total.png'); plt.close()
        logger.info("[SAVED] hotness_total.png")
        
        # === 2. Total Hotness (Normalized) ===
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        if has_sim:
            ax.plot(sim_times, normalize(smooth(sim_hotness.get('total', []))), color=C_SIM['total'], lw=LW, label='Simulation')
        if has_real:
            ax.plot(self.real_times, normalize(smooth(self.real_hotness.get('total', []))), color=C_REAL['total'], 
                   lw=LW, ls='--', label='Real Data')
        ax.set_xlabel('Time'); ax.set_ylabel('Normalized Activity')
        ax.set_title('Total Activity Comparison (Normalized)', fontweight='bold')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        plt.xticks(rotation=45); ax.legend(frameon=True, fancybox=False, edgecolor='black')
        ax.grid(alpha=0.3, linestyle=':'); plt.tight_layout()
        plt.savefig(self.vis_dir / 'hotness_total_norm.png'); plt.close()
        logger.info("[SAVED] hotness_total_norm.png")
        
        # === 3. By Category (Absolute) - Vertical Layout ===
        fig, axes = plt.subplots(3, 1, figsize=FIG_SIZE_TALL)
        for i, btype in enumerate(['original', 'repost', 'comment']):
            ax = axes[i]
            if has_sim:
                sim_data = smooth(sim_hotness.get(btype, []))
                ax.plot(sim_times, sim_data, color=C_SIM[btype], lw=LW, label='Simulation')
            if has_real:
                ax.plot(self.real_times, smooth(self.real_hotness.get(btype, [])), color=C_REAL[btype], 
                       lw=LW, ls='--', label='Real Data')
            ax.set_ylabel('Count')
            ax.set_title(f'{labels_en[btype]} Posts', fontweight='bold')
            ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black')
            ax.grid(alpha=0.3, linestyle=':')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        axes[-1].set_xlabel('Time')
        plt.tight_layout(); plt.savefig(self.vis_dir / 'hotness_by_type.png'); plt.close()
        logger.info("[SAVED] hotness_by_type.png")
        
        # === 4. By Category (Normalized) - Vertical Layout ===
        fig, axes = plt.subplots(3, 1, figsize=FIG_SIZE_TALL)
        for i, btype in enumerate(['original', 'repost', 'comment']):
            ax = axes[i]
            if has_sim:
                sim_data = normalize(smooth(sim_hotness.get(btype, [])))
                ax.plot(sim_times, sim_data, color=C_SIM[btype], lw=LW, label='Simulation')
            if has_real:
                ax.plot(self.real_times, normalize(smooth(self.real_hotness.get(btype, []))), 
                       color=C_REAL[btype], lw=LW, ls='--', label='Real Data')
            ax.set_ylabel('Normalized')
            ax.set_title(f'{labels_en[btype]} Posts (Normalized)', fontweight='bold')
            ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black')
            ax.grid(alpha=0.3, linestyle=':')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        axes[-1].set_xlabel('Time')
        plt.tight_layout(); plt.savefig(self.vis_dir / 'hotness_by_type_norm.png'); plt.close()
        logger.info("[SAVED] hotness_by_type_norm.png")
        
        # === 5. All Categories Overlay (Single Plot) ===
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        for btype in ['original', 'repost', 'comment']:
            if has_sim:
                ax.plot(sim_times, smooth(sim_hotness.get(btype, [])), color=C_SIM[btype], 
                       lw=LW, label=f'Sim-{labels_en[btype]}')
            if has_real:
                ax.plot(self.real_times, smooth(self.real_hotness.get(btype, [])), color=C_REAL[btype], 
                       lw=LW, ls='--', label=f'Real-{labels_en[btype]}')
        ax.set_xlabel('Time'); ax.set_ylabel('Activity Count')
        ax.set_title('Activity by Type Comparison', fontweight='bold')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        plt.xticks(rotation=45); ax.legend(ncol=2, frameon=True, fancybox=False, edgecolor='black')
        ax.grid(alpha=0.3, linestyle=':'); plt.tight_layout()
        plt.savefig(self.vis_dir / 'hotness_overlay.png'); plt.close()
        logger.info("[SAVED] hotness_overlay.png")
        
        # === 6. Stacked Area Chart (Simulation only) ===
        if has_sim and any(sim_hotness.get(k) for k in ['original', 'repost', 'comment']):
            fig, ax = plt.subplots(figsize=FIG_SIZE)
            stack_data = [smooth(sim_hotness.get(k, [])) for k in ['original', 'repost', 'comment']]
            ax.stackplot(sim_times, stack_data, labels=['Original', 'Repost', 'Comment'],
                        colors=[C_SIM['original'], C_SIM['repost'], C_SIM['comment']], alpha=0.8)
            ax.set_xlabel('Time'); ax.set_ylabel('Activity Count')
            ax.set_title('Simulated Activity Composition', fontweight='bold')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            plt.xticks(rotation=45); ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black')
            ax.grid(alpha=0.3, linestyle=':'); plt.tight_layout()
            plt.savefig(self.vis_dir / 'hotness_stacked.png'); plt.close()
            logger.info("[SAVED] hotness_stacked.png")
    
    # ========== Emotion Comparison Plots ==========
    def _plot_emotion_comparison(self, sim_times, sim_emotion):
        """Emotion comparison: TOP3 emotions (excluding Neutral), absolute and normalized"""
        has_sim = bool(sim_times and sim_emotion)
        has_real = bool(self.real_emotion and self.real_times)
        if not has_sim and not has_real:
            logger.warning("[SKIP] emotion_*.png - No simulation or real emotion data")
            return
        
        # Filter out Neutral, get TOP3
        sim_top3 = []
        if has_sim:
            sim_filtered = {k: v for k, v in sim_emotion.items() if k not in ['中性', 'Neutral', 'neutral']}
            if sim_filtered:
                sim_top3 = sorted(sim_filtered.keys(), key=lambda x: sum(sim_filtered[x]), reverse=True)[:3]
        
        real_top3 = []
        if has_real:
            real_filtered = {k: v for k, v in self.real_emotion.items() if k not in ['中性', 'Neutral', 'neutral']}
            if real_filtered:
                real_top3 = sorted(real_filtered.keys(), key=lambda x: sum(real_filtered[x]), reverse=True)[:3]
        
        # Merge TOP3 (union, up to 3)
        all_top3 = list(dict.fromkeys(sim_top3 + real_top3))[:3]
        if not all_top3:
            logger.warning("[SKIP] emotion_*.png - No valid emotion data (only Neutral)")
            return
        
        # === 1. TOP3 Emotions Overlay (Absolute) ===
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        for emo in all_top3:
            c = C_EMOTION.get(emo, '#888')
            emo_en = EMO_EN.get(emo, emo)
            if has_sim and emo in sim_emotion:
                ax.plot(sim_times, smooth(sim_emotion[emo]), color=c, lw=LW, label=f'Sim-{emo_en}')
            if has_real and emo in self.real_emotion:
                ax.plot(self.real_times, smooth(self.real_emotion[emo]), color=c, lw=LW, ls='--', label=f'Real-{emo_en}')
        ax.set_xlabel('Time'); ax.set_ylabel('Count')
        ax.set_title('Top-3 Emotion Comparison (Excl. Neutral)', fontweight='bold')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        plt.xticks(rotation=45); ax.legend(ncol=2, frameon=True, fancybox=False, edgecolor='black')
        ax.grid(alpha=0.3, linestyle=':'); plt.tight_layout()
        plt.savefig(self.vis_dir / 'emotion_top3.png'); plt.close()
        logger.info("[SAVED] emotion_top3.png")
        
        # === 2. TOP3 Emotions Overlay (Normalized) ===
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        for emo in all_top3:
            c = C_EMOTION.get(emo, '#888')
            emo_en = EMO_EN.get(emo, emo)
            if has_sim and emo in sim_emotion:
                ax.plot(sim_times, normalize(smooth(sim_emotion[emo])), color=c, lw=LW, label=f'Sim-{emo_en}')
            if has_real and emo in self.real_emotion:
                ax.plot(self.real_times, normalize(smooth(self.real_emotion[emo])), color=c, lw=LW, ls='--', label=f'Real-{emo_en}')
        ax.set_xlabel('Time'); ax.set_ylabel('Normalized')
        ax.set_title('Top-3 Emotion Comparison (Normalized)', fontweight='bold')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        plt.xticks(rotation=45); ax.legend(ncol=2, frameon=True, fancybox=False, edgecolor='black')
        ax.grid(alpha=0.3, linestyle=':'); plt.tight_layout()
        plt.savefig(self.vis_dir / 'emotion_top3_norm.png'); plt.close()
        logger.info("[SAVED] emotion_top3_norm.png")
        
        # === 3. TOP3 Emotions Facet (Vertical, Absolute) ===
        fig, axes = plt.subplots(len(all_top3), 1, figsize=(12, 3.5*len(all_top3)))
        if len(all_top3) == 1: axes = [axes]
        for i, emo in enumerate(all_top3):
            ax = axes[i]; c = C_EMOTION.get(emo, '#888')
            emo_en = EMO_EN.get(emo, emo)
            if has_sim and emo in sim_emotion:
                sim_data = smooth(sim_emotion[emo])
                ax.plot(sim_times, sim_data, color=c, lw=LW, label='Simulation')
            if has_real and emo in self.real_emotion:
                ax.plot(self.real_times, smooth(self.real_emotion[emo]), color=c, lw=LW, ls='--', label='Real Data')
            ax.set_ylabel('Count')
            ax.set_title(f'{emo_en} Emotion', fontweight='bold')
            ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black')
            ax.grid(alpha=0.3, linestyle=':')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        axes[-1].set_xlabel('Time')
        plt.tight_layout(); plt.savefig(self.vis_dir / 'emotion_facet.png'); plt.close()
        logger.info("[SAVED] emotion_facet.png")
        
        # === 4. TOP3 Emotions Facet (Vertical, Normalized) ===
        fig, axes = plt.subplots(len(all_top3), 1, figsize=(12, 3.5*len(all_top3)))
        if len(all_top3) == 1: axes = [axes]
        for i, emo in enumerate(all_top3):
            ax = axes[i]; c = C_EMOTION.get(emo, '#888')
            emo_en = EMO_EN.get(emo, emo)
            if has_sim and emo in sim_emotion:
                sim_data = normalize(smooth(sim_emotion[emo]))
                ax.plot(sim_times, sim_data, color=c, lw=LW, label='Simulation')
            if has_real and emo in self.real_emotion:
                ax.plot(self.real_times, normalize(smooth(self.real_emotion[emo])), color=c, lw=LW, ls='--', label='Real Data')
            ax.set_ylabel('Normalized')
            ax.set_title(f'{emo_en} Emotion (Normalized)', fontweight='bold')
            ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black')
            ax.grid(alpha=0.3, linestyle=':')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        axes[-1].set_xlabel('Time')
        plt.tight_layout(); plt.savefig(self.vis_dir / 'emotion_facet_norm.png'); plt.close()
        logger.info("[SAVED] emotion_facet_norm.png")
        
        # === 5. Emotion Distribution Pie Chart (if sim data) ===
        if has_sim:
            # Get all emotions except neutral
            emo_totals = {k: sum(v) for k, v in sim_emotion.items() if k not in ['中性', 'Neutral', 'neutral']}
            if emo_totals:
                sorted_emos = sorted(emo_totals.keys(), key=lambda x: emo_totals[x], reverse=True)
                values = [emo_totals[e] for e in sorted_emos]
                labels = [EMO_EN.get(e, e) for e in sorted_emos]
                colors = [C_EMOTION.get(e, '#888') for e in sorted_emos]
                
                fig, ax = plt.subplots(figsize=(8, 8))
                wedges, texts, autotexts = ax.pie(values, labels=labels, colors=colors, autopct='%1.1f%%',
                                                   startangle=90, pctdistance=0.75)
                for autotext in autotexts:
                    autotext.set_fontsize(10)
                ax.set_title('Simulated Emotion Distribution (Excl. Neutral)', fontweight='bold')
                plt.tight_layout(); plt.savefig(self.vis_dir / 'emotion_pie.png'); plt.close()
                logger.info("[SAVED] emotion_pie.png")
        
        # === 6. Emotion Stacked Area (if sim data) ===
        if has_sim and all_top3:
            fig, ax = plt.subplots(figsize=FIG_SIZE)
            stack_data = [smooth(sim_emotion.get(emo, [])) for emo in all_top3]
            stack_labels = [EMO_EN.get(emo, emo) for emo in all_top3]
            stack_colors = [C_EMOTION.get(emo, '#888') for emo in all_top3]
            ax.stackplot(sim_times, stack_data, labels=stack_labels, colors=stack_colors, alpha=0.8)
            ax.set_xlabel('Time'); ax.set_ylabel('Count')
            ax.set_title('Top-3 Emotion Evolution (Stacked)', fontweight='bold')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            plt.xticks(rotation=45); ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black')
            ax.grid(alpha=0.3, linestyle=':'); plt.tight_layout()
            plt.savefig(self.vis_dir / 'emotion_stacked.png'); plt.close()
            logger.info("[SAVED] emotion_stacked.png")
    
    # ========== Activity & Hawkes Process Plots ==========
    def _plot_activity_curve(self, macro_results: Dict):
        """Hawkes intensity and active agent dynamics"""
        stats = macro_results.get('stats', {})
        intensity = stats.get('intensity_history', [])
        active = stats.get('active_agents_per_step', [])
        actions_by_step = stats.get('actions_per_step', [])
        
        if not intensity and not active: return
        
        # === 1. Hawkes Intensity & Active Agents (2-panel) ===
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        if intensity:
            axes[0].plot(intensity, color=C_SIM['total'], lw=LW)
            axes[0].set_xlabel('Step'); axes[0].set_ylabel('Intensity (λ)')
            axes[0].set_title('Hawkes Process Intensity', fontweight='bold')
            axes[0].grid(alpha=0.3, linestyle=':')
        
        if active:
            axes[1].plot(active, color=C_SIM['original'], lw=LW)
            axes[1].set_xlabel('Step'); axes[1].set_ylabel('Active Agents')
            axes[1].set_title('Agent Activity Over Time', fontweight='bold')
            axes[1].grid(alpha=0.3, linestyle=':')
        
        plt.tight_layout(); plt.savefig(self.vis_dir / 'hawkes_activity.png'); plt.close()
        logger.info("[SAVED] hawkes_activity.png")
        
        # === 2. Actions per Step (if available) ===
        if actions_by_step:
            fig, ax = plt.subplots(figsize=FIG_SIZE)
            ax.bar(range(len(actions_by_step)), actions_by_step, color=C_SIM['comment'], alpha=0.7, width=1.0)
            ax.set_xlabel('Step'); ax.set_ylabel('Actions')
            ax.set_title('Actions per Simulation Step', fontweight='bold')
            ax.grid(alpha=0.3, linestyle=':')
            plt.tight_layout(); plt.savefig(self.vis_dir / 'actions_per_step.png'); plt.close()
            logger.info("[SAVED] actions_per_step.png")
        
        # === 3. Cumulative Actions ===
        if actions_by_step:
            cumulative = np.cumsum(actions_by_step)
            fig, ax = plt.subplots(figsize=FIG_SIZE)
            ax.plot(cumulative, color=C_SIM['total'], lw=LW)
            ax.set_xlabel('Step'); ax.set_ylabel('Cumulative Actions')
            ax.set_title('Cumulative Action Count', fontweight='bold')
            ax.grid(alpha=0.3, linestyle=':')
            plt.tight_layout(); plt.savefig(self.vis_dir / 'actions_cumulative.png'); plt.close()
            logger.info("[SAVED] actions_cumulative.png")
        
        # === 4. Action Type Distribution (Bar Chart) ===
        actions_by_type = stats.get('actions_by_type', {})
        if actions_by_type:
            types = list(actions_by_type.keys())
            counts = list(actions_by_type.values())
            colors = [C_SIM.get(t, '#888') for t in types]
            
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.bar(types, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            ax.set_xlabel('Action Type'); ax.set_ylabel('Count')
            ax.set_title('Action Distribution by Type', fontweight='bold')
            ax.grid(alpha=0.3, linestyle=':', axis='y')
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                       f'{count}', ha='center', va='bottom', fontsize=9)
            plt.tight_layout(); plt.savefig(self.vis_dir / 'action_distribution.png'); plt.close()
            logger.info("[SAVED] action_distribution.png")
