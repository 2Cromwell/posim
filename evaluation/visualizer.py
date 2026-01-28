"""
可视化模块 - 生成评估结果图表
"""
import json
from pathlib import Path
from typing import List, Dict, Any


def plot_behavior_distribution(distribution: Dict[str, float], output_path: str):
    """绘制行为分布图"""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    
    labels = list(distribution.keys())
    values = list(distribution.values())
    
    colors = plt.cm.Set3(range(len(labels)))
    plt.bar(labels, values, color=colors)
    plt.xlabel('Action Type')
    plt.ylabel('Proportion')
    plt.title('Behavior Type Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_emotion_evolution(activity_curve: List[Dict], output_path: str):
    """绘制情绪演化图"""
    import matplotlib.pyplot as plt
    
    if not activity_curve:
        return
    
    times = [d['time'] for d in activity_curve]
    counts = [d['count'] for d in activity_curve]
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(times)), counts, marker='o', linewidth=2)
    plt.xlabel('Time Step')
    plt.ylabel('Activity Count')
    plt.title('Activity Evolution Over Time')
    plt.xticks(range(0, len(times), max(1, len(times)//10)), 
               [times[i][-8:-3] for i in range(0, len(times), max(1, len(times)//10))], 
               rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_polarization(stance_dist: Dict[str, float], polarization: float, output_path: str):
    """绘制极化分布图"""
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 立场分布饼图
    labels = list(stance_dist.keys())
    sizes = list(stance_dist.values())
    colors = {'support': '#4CAF50', 'oppose': '#F44336', 'neutral': '#9E9E9E'}
    pie_colors = [colors.get(l, '#2196F3') for l in labels]
    ax1.pie(sizes, labels=labels, colors=pie_colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Stance Distribution')
    
    # 极化指数仪表盘
    ax2.barh(['Polarization'], [polarization], color='#FF9800')
    ax2.set_xlim(0, 1)
    ax2.set_xlabel('Polarization Index')
    ax2.set_title(f'Polarization Index: {polarization:.3f}')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_trending_curve(intensity_history: List[float], output_path: str):
    """绘制热度曲线"""
    import matplotlib.pyplot as plt
    
    if not intensity_history:
        return
    
    plt.figure(figsize=(12, 6))
    plt.plot(intensity_history, linewidth=2, color='#E91E63')
    plt.fill_between(range(len(intensity_history)), intensity_history, alpha=0.3, color='#E91E63')
    plt.xlabel('Time Step (minutes)')
    plt.ylabel('Intensity (λ)')
    plt.title('Hawkes Process Intensity Over Time')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def generate_all_visualizations(metrics: Dict[str, Any], output_dir: str, 
                                intensity_history: List[float] = None):
    """生成所有可视化图表"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 行为分布图
    if metrics.get('behavior_distribution'):
        plot_behavior_distribution(
            metrics['behavior_distribution'],
            str(output_path / 'behavior_distribution.png')
        )
    
    # 活跃度曲线
    if metrics.get('activity_curve'):
        plot_emotion_evolution(
            metrics['activity_curve'],
            str(output_path / 'emotion_evolution.png')
        )
    
    # 极化分布图
    if metrics.get('stance_distribution'):
        plot_polarization(
            metrics['stance_distribution'],
            metrics.get('polarization_index', 0),
            str(output_path / 'polarization.png')
        )
    
    # 热度曲线
    if intensity_history:
        plot_trending_curve(
            intensity_history,
            str(output_path / 'trending_curve.png')
        )
    
    # 保存指标数据
    with open(output_path / 'metrics.json', 'w', encoding='utf-8') as f:
        # 转换不可序列化的数据
        serializable_metrics = {
            k: v for k, v in metrics.items() 
            if not isinstance(v, (type, ))
        }
        json.dump(serializable_metrics, f, ensure_ascii=False, indent=2, default=str)
