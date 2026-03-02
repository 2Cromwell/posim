# -*- coding: utf-8 -*-
"""
共享可视化配置 - 统一的绘图风格和颜色方案

全局风格在此文件统一控制，修改 plt.rcParams.update({...}) 中的参数即可全局生效。
常用调整项:
  - font.sans-serif: 字体优先级列表（CJK 字体在前以支持中文）
  - font.size: 全局基准字号
  - axes.linewidth: 边框线宽
  - axes.labelsize / axes.titlesize: 坐标轴标签 / 标题字号
  - xtick.labelsize / ytick.labelsize: 刻度字号
  - legend.fontsize: 图例字号
  - figure.dpi / savefig.dpi: 分辨率
"""
import warnings
import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings('ignore', message='Glyph .* missing from')

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.dates as mdates

# ========== 中文字体配置（自动检测可用 CJK 字体） ==========
_CJK_FONTS = ['SimHei', 'Microsoft YaHei', 'STSong', 'Noto Sans CJK SC',
              'WenQuanYi Micro Hei', 'PingFang SC', 'Hiragino Sans GB']
_LATIN_FONTS = ['DejaVu Sans', 'Arial', 'Helvetica']

_available_fonts = set(f.name for f in fm.fontManager.ttflist)
_font_list = [f for f in _CJK_FONTS if f in _available_fonts] + _LATIN_FONTS

# ========== 可视化常量 ==========
FIG_SIZE = (8, 6)
FIG_SIZE_TALL = (8, 10)
FIG_SIZE_WIDE = (10, 6)
FIG_SIZE_SQUARE = (8, 8)
DPI = 300
LW = 2.0             # 主数据线宽
LW_MINOR = 1.4       # 辅助线 / 网格线宽
ALPHA = 0.25          # 填充透明度
MARKER_SIZE = 5
FONT_SIZE = {'title': 16, 'label': 14, 'tick': 12, 'legend': 11, 'annotation': 11}
SPINE_LW = 1.8        # 边框线宽

# ========== 颜色方案 ==========
C_SIM = {'total': '#6eb169', 'original': '#cde3d3', 'repost': '#e0c4d6', 'comment': '#397c52'}
C_REAL = {'total': '#2d5a3f', 'original': '#8fb3a3', 'repost': '#b39bb5', 'comment': '#1a4a35'}

C_EMOTION = {
    'Anger': '#d62728', 'Disgust': '#8c564b', 'Anxiety': '#ff7f0e',
    'Sadness': '#1f77b4', 'Excitement': '#2ca02c', 'Neutral': '#7f7f7f'
}

C_ACTION = {
    'short_comment': '#98df8a', 'long_comment': '#c5b0d5',
    'short_post': '#ffbb78', 'long_post': '#ff9896',
    'repost': '#c49c94', 'repost_comment': '#f7b6d2'
}

C_SENTIMENT = {
    'positive': '#2ca02c', 'negative': '#d62728', 'neutral': '#7f7f7f'
}

# 标准类型顺序和颜色（不含 like）
STANDARD_TYPE_ORDER = ['original', 'repost', 'comment']
STANDARD_TYPE_COLORS = {'original': '#ff7f0e', 'repost': '#c49c94', 'comment': '#2ca02c'}

# ============================================================
# Matplotlib 全局配置（封闭式方框 + 大字号 + 原字体风格）
# 修改此处即可全局生效
# ============================================================
plt.rcParams.update({
    # 字体
    'font.family': 'sans-serif',
    'font.sans-serif': _font_list,
    'font.size': 13,
    # 坐标轴 —— 封闭式方框（四边可见，加粗）
    'axes.unicode_minus': False,
    'mathtext.default': 'regular',
    'axes.linewidth': SPINE_LW,
    'axes.labelsize': FONT_SIZE['label'],
    'axes.titlesize': FONT_SIZE['title'],
    'axes.titleweight': 'bold',
    'axes.spines.top': True,
    'axes.spines.right': True,
    'axes.spines.bottom': True,
    'axes.spines.left': True,
    # 刻度
    'xtick.labelsize': FONT_SIZE['tick'],
    'ytick.labelsize': FONT_SIZE['tick'],
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.minor.width': 0.6,
    'ytick.minor.width': 0.6,
    # 图例
    'legend.fontsize': FONT_SIZE['legend'],
    'legend.framealpha': 0.9,
    # 保存
    'figure.dpi': DPI,
    'savefig.dpi': DPI,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.08,
})


def setup_time_axis(ax, rotation=45):
    """统一设置时间轴格式"""
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=rotation, ha='right')


def add_grid(ax, alpha=0.3):
    """添加网格"""
    ax.grid(alpha=alpha, linestyle=':')


def add_legend(ax, **kwargs):
    """添加图例"""
    defaults = {'frameon': True, 'fancybox': False, 'edgecolor': 'black',
                'fontsize': FONT_SIZE['legend']}
    defaults.update(kwargs)
    ax.legend(**defaults)


def save_figure(fig, path, close=True):
    """保存并关闭图表"""
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches='tight', pad_inches=0.05)
    if close:
        plt.close(fig)


def create_figure(nrows=1, ncols=1, figsize=None, **kwargs):
    """标准化创建图表"""
    if figsize is None:
        if nrows == 1 and ncols == 1:
            figsize = FIG_SIZE
        elif nrows > 2:
            figsize = (14, 4 * nrows)
        else:
            figsize = FIG_SIZE_WIDE
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    return fig, axes
