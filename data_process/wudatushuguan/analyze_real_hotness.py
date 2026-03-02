# -*- coding: utf-8 -*-
"""真实热度数据分析 - 为霍克斯算法参数调优提供依据"""
import json, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from collections import defaultdict

# ========== 参数配置 ==========
INPUT_FILE = "output/base_data.json"
OUTPUT_DIR = "output/analysis"
TIME_GRANULARITY = 10  # 分钟
TIME_START = "2025-07-17 00:00:00"
TIME_END = "2025-08-06 06:00:00"
SMOOTH_WINDOW = 5
FIG_SIZE, DPI, LW, ALPHA = (14, 5), 150, 1.5, 0.3

# 配色
C_BEHAVIOR = {'total': '#c83e4b', 'original': '#5894c8', 'repost': '#b66e1a', 'comment': '#97c1e7'}
C_EMOTION = {'愤怒': '#c83e4b', '悲伤': '#5894c8', '惊奇': '#b66e1a', '恐惧': '#97c1e7', 
             '喜悦': '#f7ead9', '厌恶': '#c6c6c6', '中性': '#ffd3d4'}

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ========== 工具函数 ==========
def parse_time(s): 
    try: return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    except: return None

def truncate(dt, g): return dt.replace(minute=(dt.minute//g)*g, second=0, microsecond=0)
def smooth(d, w): return np.convolve(d, np.ones(w)/w, mode='same') if w > 1 else d

def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f: json.dump(data, f, ensure_ascii=False, indent=2)

# ========== 数据处理 ==========
def load_and_extract(path, t_start, t_end):
    """加载数据并提取活动"""
    with open(path, 'r', encoding='utf-8') as f: users = json.load(f)
    print(f"[INFO] 加载 {len(users)} 用户")
    
    acts = []
    for u in users:
        for p in u.get('original_posts', []):
            t = parse_time(p.get('time', ''))
            if t: acts.append({'type': 'original', 'time': t, 'emotion': p.get('emotion', '中性')})
        for p in u.get('repost_posts', []):
            t = parse_time(p.get('time', ''))
            if t: acts.append({'type': 'repost', 'time': t, 'emotion': p.get('emotion', '中性')})
        for c in u.get('comments', []):
            t = parse_time(c.get('time', ''))
            if t: acts.append({'type': 'comment', 'time': t, 'emotion': c.get('emotion', '中性')})
    
    # 时间过滤
    if t_start: acts = [a for a in acts if a['time'] >= parse_time(t_start)]
    if t_end: acts = [a for a in acts if a['time'] <= parse_time(t_end)]
    print(f"[INFO] 活动数: {len(acts)}")
    return acts

def aggregate(acts, granularity):
    """聚合热度和情绪数据"""
    h_buckets = defaultdict(lambda: {'total': 0, 'original': 0, 'repost': 0, 'comment': 0})
    e_buckets = defaultdict(lambda: defaultdict(int))
    emotions = set()
    
    for a in acts:
        t = truncate(a['time'], granularity)
        h_buckets[t]['total'] += 1
        h_buckets[t][a['type']] += 1
        e_buckets[t][a['emotion']] += 1
        emotions.add(a['emotion'])
    
    if not h_buckets: return [], {}, {}
    
    # 填充时间序列
    times = sorted(h_buckets.keys())
    t_start, t_end = times[0], times[-1]
    all_times, cur = [], t_start
    while cur <= t_end:
        all_times.append(cur)
        cur += timedelta(minutes=granularity)
    
    hotness = {k: [h_buckets[t].get(k, 0) for t in all_times] for k in ['total', 'original', 'repost', 'comment']}
    emotion = {e: [e_buckets[t].get(e, 0) for t in all_times] for e in emotions}
    
    print(f"[INFO] 时间范围: {t_start} ~ {t_end}, 点数: {len(all_times)}")
    return all_times, hotness, emotion

# ========== 可视化 ==========
def plot_hotness(times, data, out_dir):
    """热度时序图"""
    sm = {k: smooth(v, SMOOTH_WINDOW) for k, v in data.items()}
    
    # 总热度
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    ax.plot(times, sm['total'], color=C_BEHAVIOR['total'], lw=LW, label='总热度')
    ax.fill_between(times, np.array(sm['total'])*0.8, np.array(sm['total'])*1.2, color=C_BEHAVIOR['total'], alpha=ALPHA)
    ax.set_xlabel('时间'); ax.set_ylabel('活动数量'); ax.set_title('舆情总热度时序变化', fontweight='bold')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.xticks(rotation=45); ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(f"{out_dir}/hotness_total.png", dpi=DPI); plt.close()
    
    # 分类对比
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    for k, lb in [('original', '原创'), ('repost', '转发'), ('comment', '评论')]:
        ax.plot(times, sm[k], color=C_BEHAVIOR[k], lw=LW, label=lb)
    ax.set_xlabel('时间'); ax.set_ylabel('数量'); ax.set_title('各类行为热度对比', fontweight='bold')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.xticks(rotation=45); ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(f"{out_dir}/hotness_comparison.png", dpi=DPI); plt.close()
    print(f"[SAVED] hotness_*.png")

def plot_emotion(times, data, out_dir):
    """情绪时序图"""
    if not data: return
    sm = {k: smooth(v, SMOOTH_WINDOW) for k, v in data.items()}
    
    # 各情绪绝对量
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    for e, v in sm.items():
        ax.plot(times, v, color=C_EMOTION.get(e, '#888'), lw=LW, label=e)
    ax.set_xlabel('时间'); ax.set_ylabel('数量'); ax.set_title('各类情绪时序变化', fontweight='bold')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.xticks(rotation=45); ax.legend(ncol=2); ax.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(f"{out_dir}/emotion_absolute.png", dpi=DPI); plt.close()
    
    # 堆叠图
    sorted_e = sorted(sm.keys(), key=lambda x: sum(sm[x]), reverse=True)
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    ax.stackplot(times, [sm[e] for e in sorted_e], labels=sorted_e, 
                 colors=[C_EMOTION.get(e, '#888') for e in sorted_e], alpha=0.8)
    ax.set_xlabel('时间'); ax.set_ylabel('数量'); ax.set_title('情绪分布堆叠图', fontweight='bold')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.xticks(rotation=45); ax.legend(ncol=2); plt.tight_layout()
    plt.savefig(f"{out_dir}/emotion_stacked.png", dpi=DPI); plt.close()
    print(f"[SAVED] emotion_*.png")

def plot_emotion_ratio_top3(times, data, out_dir):
    """TOP3情绪占比变化（移除中性）"""
    # 移除中性情绪
    filtered = {k: v for k, v in data.items() if k != '中性'}
    if not filtered: return
    
    sm = {k: smooth(v, SMOOTH_WINDOW) for k, v in filtered.items()}
    
    # 找TOP3情绪（按总量）
    top3 = sorted(sm.keys(), key=lambda x: sum(sm[x]), reverse=True)[:3]
    
    # 计算每个时间点的占比
    total = np.zeros(len(times))
    for v in sm.values(): total += np.array(v)
    total = np.maximum(total, 1)  # 避免除零
    
    ratios = {e: np.array(sm[e]) / total * 100 for e in top3}
    
    # 绘图
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    for e in top3:
        ax.plot(times, ratios[e], color=C_EMOTION.get(e, '#888'), lw=LW, label=e)
        ax.fill_between(times, ratios[e]*0.9, ratios[e]*1.1, color=C_EMOTION.get(e, '#888'), alpha=ALPHA)
    
    ax.set_xlabel('时间'); ax.set_ylabel('占比 (%)')
    ax.set_title('TOP3情绪占比变化（移除中性）', fontweight='bold')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.xticks(rotation=45); ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(f"{out_dir}/emotion_ratio_top3.png", dpi=DPI); plt.close()
    
    # 保存数据
    ratio_data = {'timestamps': [t.strftime("%Y-%m-%d %H:%M:%S") for t in times],
                  'data': {e: ratios[e].tolist() for e in top3}}
    save_json(f"{out_dir}/emotion_ratio_top3.json", ratio_data)
    print(f"[SAVED] emotion_ratio_top3.*")

# ========== 主函数 ==========
def main():
    print("=" * 50 + "\n真实热度数据分析\n" + "=" * 50)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    in_path = os.path.join(script_dir, INPUT_FILE)
    out_path = os.path.join(script_dir, OUTPUT_DIR)
    os.makedirs(out_path, exist_ok=True)
    
    # 加载和聚合
    acts = load_and_extract(in_path, TIME_START, TIME_END)
    times, hotness, emotion = aggregate(acts, TIME_GRANULARITY)
    if not times: print("[ERROR] 无数据"); return
    
    # 保存JSON
    save_json(f"{out_path}/hotness_timeseries.json", 
              {'timestamps': [t.strftime("%Y-%m-%d %H:%M:%S") for t in times],
               'data': {k: [int(x) for x in v] for k, v in hotness.items()}})
    save_json(f"{out_path}/emotion_timeseries.json",
              {'timestamps': [t.strftime("%Y-%m-%d %H:%M:%S") for t in times],
               'data': {k: [int(x) for x in v] for k, v in emotion.items()}})
    
    # 统计
    stats = {
        'total': sum(hotness['total']),
        'peak': max(hotness['total']),
        'peak_time': times[np.argmax(hotness['total'])].strftime("%Y-%m-%d %H:%M:%S"),
        'by_behavior': {k: sum(hotness[k]) for k in ['original', 'repost', 'comment']},
        'emotion_totals': {k: sum(v) for k, v in emotion.items()}
    }
    save_json(f"{out_path}/statistics.json", stats)
    
    # 可视化
    plot_hotness(times, hotness, out_path)
    plot_emotion(times, emotion, out_path)
    plot_emotion_ratio_top3(times, emotion, out_path)
    
    # 摘要
    print(f"\n总活动: {stats['total']}, 峰值: {stats['peak']} @ {stats['peak_time']}")
    print(f"原创: {stats['by_behavior']['original']}, 转发: {stats['by_behavior']['repost']}, 评论: {stats['by_behavior']['comment']}")
    print(f"\n[DONE] 输出: {out_path}")

if __name__ == '__main__': main()
