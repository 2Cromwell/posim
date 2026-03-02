# -*- coding: utf-8 -*-
"""
共享工具函数 - 时间解析、数据处理、通用计算
"""
import json
import re
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter

import logging
logger = logging.getLogger(__name__)


# ========== 时间解析 ==========
def parse_time(s: str) -> Optional[datetime]:
    """解析多种时间格式"""
    if not s:
        return None
    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M"]:
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            continue
    return None


def truncate_time(dt: datetime, granularity: int) -> datetime:
    """按粒度截断时间"""
    return dt.replace(minute=(dt.minute // granularity) * granularity, second=0, microsecond=0)


def generate_time_range(start: datetime, end: datetime, granularity: int) -> List[datetime]:
    """生成时间范围内的所有时间点"""
    times = []
    cur = start
    while cur <= end:
        times.append(cur)
        cur += timedelta(minutes=granularity)
    return times


# ========== 数据平滑/归一化 ==========
SMOOTH_W = 5

def smooth(data: List, w: int = SMOOTH_W, target_len: int = None) -> np.ndarray:
    """移动平均平滑"""
    if data is None or (hasattr(data, '__len__') and len(data) == 0):
        return np.array([])
    arr = np.array(data, dtype=float)
    if w > 1 and len(arr) >= w:
        result = np.convolve(arr, np.ones(w) / w, mode='same')
    else:
        result = arr
    if target_len is not None and len(result) != target_len:
        if len(result) > target_len:
            result = result[:target_len]
        else:
            result = np.pad(result, (0, target_len - len(result)), mode='constant', constant_values=0)
    return result


def normalize(data: List) -> np.ndarray:
    """归一化到 [0, 1]"""
    arr = np.array(data, dtype=float)
    max_val = arr.max()
    return arr / max_val if max_val > 0 else arr


# ========== JSON保存 ==========
def save_json(path: Path, data: Any):
    """保存JSON文件"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


# ========== 情绪/立场/风格关键词 ==========
EMOTION_KEYWORDS = {
    'Anger': ['愤怒', '气愤', '生气', '恼火', '愤慨', '怒', '火大', '暴怒', '激愤',
              '可恶', '该死', '混蛋', '无耻', '可恨', '气死', '怒火', '愤恨'],
    'Disgust': ['厌恶', '恶心', '反感', '讨厌', '嫌弃', '鄙视', '唾弃', '作呕',
                '恶臭', '龌龊', '肮脏', '下作', '卑鄙', '恶毒'],
    'Anxiety': ['焦虑', '担心', '忧虑', '不安', '紧张', '担忧', '惶恐', '恐慌',
                '害怕', '恐惧', '惊恐', '慌张', '忐忑', '心慌', '惴惴不安'],
    'Sadness': ['悲伤', '伤心', '难过', '痛心', '心痛', '悲痛', '哀伤', '忧伤',
                '沮丧', '失落', '绝望', '心碎', '惋惜', '遗憾', '可惜'],
    'Excitement': ['兴奋', '激动', '开心', '高兴', '快乐', '喜悦', '欣喜', '欢喜',
                   '振奋', '期待', '惊喜', '喜出望外', '满足', '幸福', '愉快'],
    'Neutral': ['中性', '客观', '理性', '冷静', '平静', '淡定', '无感']
}

EMO_CN_TO_EN = {
    '愤怒': 'Anger', '厌恶': 'Disgust', '焦虑': 'Anxiety',
    '悲伤': 'Sadness', '兴奋': 'Excitement', '中性': 'Neutral',
    '惊奇': 'Excitement', '喜悦': 'Excitement', '恐惧': 'Anxiety'
}

STANCE_KEYWORDS = {
    'Support': ['支持', '赞同', '同意', '认可', '力挺', '点赞', '支援'],
    'Oppose': ['反对', '抵制', '批评', '谴责', '抗议', '拒绝', '否定'],
    'Neutral': ['中立', '观望', '旁观', '不表态', '待定']
}

STYLE_KEYWORDS = {
    'Sarcastic': ['阴阳怪气', '讽刺', '嘲讽', '呵呵', '笑死'],
    'Aggressive': ['激进', '攻击', '人身攻击', '骂', '喷'],
    'Emotional': ['情绪', '宣泄', '发泄', '崩溃', '受不了'],
    'Questioning': ['质疑', '怀疑', '存疑', '疑问', '？'],
    'Empathetic': ['共情', '理解', '同情', '心疼', '可怜'],
    'Indifferent': ['冷漠', '无所谓', '不关心', '漠然']
}

CONFLICT_KEYWORDS = {
    'debate': ['辩论', '争论', '反驳', '质疑', '反对'],
    'attack': ['攻击', '谩骂', '辱骂', '诋毁', '诽谤', '污蔑', '抹黑'],
    'confrontation': ['对抗', '冲突', '对立', '针锋相对', '水火不容'],
    'criticism': ['批评', '谴责', '指责', '抨击', '痛斥'],
    'defense': ['辩护', '维护', '支持', '力挺', '声援'],
    'polarization': ['极端', '偏激', '激进', '极端化', '两极化']
}

# 情绪强度映射
INTENSITY_MAP = {
    '低': 0.2, '中等': 0.5, '高': 0.8, '极高': 1.0,
    'low': 0.2, 'medium': 0.5, 'high': 0.8, 'extreme': 1.0
}

# 立场映射
STANCE_MAP = {
    '支持': 1.0, '反对': -1.0, '中立': 0.0,
    'Support': 1.0, 'Oppose': -1.0, 'Neutral': 0.0
}

# 情感极性映射
SENTIMENT_POLARITY_MAP = {
    '正面': 'positive', '负面': 'negative', '中性': 'neutral',
    'positive': 'positive', 'negative': 'negative', 'neutral': 'neutral'
}


# ========== 分类函数 ==========
def classify_emotion_by_keywords(text: str) -> Tuple[str, Dict[str, float]]:
    """基于关键词的情绪分类"""
    if not text:
        return 'Neutral', {e: 0.0 for e in EMOTION_KEYWORDS}
    scores = {}
    for emotion, keywords in EMOTION_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in text)
        scores[emotion] = count
    total = sum(scores.values())
    if total > 0:
        scores = {e: s / total for e, s in scores.items()}
    else:
        scores = {e: 0.0 for e in EMOTION_KEYWORDS}
        scores['Neutral'] = 1.0
    primary = max(scores.keys(), key=lambda x: scores[x])
    return primary, scores


def batch_classify_emotions(texts: List[str]) -> Tuple[List[str], List[Dict[str, float]]]:
    """批量情绪分类"""
    primaries, all_scores = [], []
    for text in texts:
        primary, scores = classify_emotion_by_keywords(text)
        primaries.append(primary)
        all_scores.append(scores)
    return primaries, all_scores


def normalize_emotion_label(label: str) -> str:
    """将中文情绪标签标准化为英文"""
    if label in EMO_CN_TO_EN:
        return EMO_CN_TO_EN[label]
    if label in EMOTION_KEYWORDS:
        return label
    return 'Neutral'


def classify_stance(text: str) -> str:
    """从文本分类立场"""
    if not text or not isinstance(text, str):
        return 'Neutral'
    for stance, keywords in STANCE_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return stance
    return 'Neutral'


def classify_style(text: str) -> str:
    """从文本分类表达风格"""
    if not text or not isinstance(text, str):
        return 'Other'
    for style, keywords in STYLE_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return style
    return 'Other'


def extract_topics(text: str) -> List[str]:
    """从文本提取话题标签 (#xxx#)"""
    if not text:
        return []
    return [t.strip() for t in re.findall(r'#([^#]+)#', text) if t.strip()]


def extract_user_id_from_url(url: str) -> str:
    """从微博URL提取用户ID"""
    if not url:
        return ''
    match = re.search(r'weibo\.com/(\d+)/', url)
    return match.group(1) if match else ''


# ========== 统计度量 ==========
def calculate_entropy(counts: Dict) -> float:
    """计算信息熵"""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = [c / total for c in counts.values() if c > 0]
    return -sum(p * np.log2(p) for p in probs)


def calculate_normalized_entropy(counts: Dict) -> float:
    """计算归一化信息熵（0-1）"""
    if not counts or len(counts) <= 1:
        return 0.0
    entropy = calculate_entropy(counts)
    max_entropy = np.log2(len(counts))
    return entropy / max_entropy if max_entropy > 0 else 0.0


def calculate_gini_coefficient(values: List[float]) -> float:
    """计算基尼系数"""
    if not values or sum(values) == 0:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    cumulative = np.cumsum(sorted_vals)
    total = cumulative[-1]
    if total == 0:
        return 0.0
    gini = (2 * sum((i + 1) * v for i, v in enumerate(sorted_vals)) - (n + 1) * total) / (n * total)
    return max(0.0, min(1.0, gini))


def calculate_jsd(p: np.ndarray, q: np.ndarray) -> float:
    """计算Jensen-Shannon散度"""
    from scipy.spatial.distance import jensenshannon
    jsd = jensenshannon(p, q)
    return float(jsd) if not np.isnan(jsd) else 0.0


def calculate_curve_similarity(curve1: List[float], curve2: List[float]) -> Dict[str, float]:
    """计算两条曲线的相似度指标（NaN安全）"""
    import warnings
    from scipy.stats import pearsonr, spearmanr
    if not curve1 or not curve2:
        return {'pearson': 0.0, 'spearman': 0.0, 'rmse': float('inf'), 'mae': float('inf')}
    min_len = min(len(curve1), len(curve2))
    c1 = np.array(curve1[:min_len], dtype=float)
    c2 = np.array(curve2[:min_len], dtype=float)
    if np.std(c1) == 0 and np.std(c2) == 0:
        return {'pearson': 1.0, 'spearman': 1.0, 'rmse': 0.0, 'mae': 0.0}
    if np.std(c1) == 0 or np.std(c2) == 0:
        return {'pearson': 0.0, 'spearman': 0.0,
                'rmse': float(np.sqrt(np.mean((c1 - c2) ** 2))),
                'mae': float(np.mean(np.abs(c1 - c2)))}
    metrics = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            r, p = pearsonr(c1, c2)
            metrics['pearson'] = 0.0 if np.isnan(r) else float(r)
            metrics['pearson_p'] = 1.0 if np.isnan(p) else float(p)
        except Exception:
            metrics['pearson'] = 0.0
        try:
            r, p = spearmanr(c1, c2)
            metrics['spearman'] = 0.0 if np.isnan(r) else float(r)
            metrics['spearman_p'] = 1.0 if np.isnan(p) else float(p)
        except Exception:
            metrics['spearman'] = 0.0
    metrics['rmse'] = float(np.sqrt(np.mean((c1 - c2) ** 2)))
    metrics['mae'] = float(np.mean(np.abs(c1 - c2)))
    return metrics


def calculate_kendall_tau(ranks1, ranks2) -> Dict[str, float]:
    """计算 Kendall's tau 排序相关性"""
    from scipy.stats import kendalltau
    try:
        tau, p = kendalltau(ranks1, ranks2)
        return {'tau': float(tau) if not np.isnan(tau) else 0.0,
                'p_value': float(p) if not np.isnan(p) else 1.0}
    except:
        return {'tau': 0.0, 'p_value': 1.0}


def calculate_ks_test(dist1, dist2) -> Dict[str, float]:
    """KS 检验比较两个分布"""
    from scipy.stats import ks_2samp
    try:
        d1 = np.array(dist1, dtype=float)
        d2 = np.array(dist2, dtype=float)
        if len(d1) < 2 or len(d2) < 2:
            return {'statistic': 1.0, 'p_value': 0.0}
        stat, p = ks_2samp(d1, d2)
        return {'statistic': float(stat), 'p_value': float(p)}
    except:
        return {'statistic': 1.0, 'p_value': 0.0}


def detect_inflection_points(curve: List[float], smooth_w: int = 5) -> List[int]:
    """检测曲线的拐点（一阶导数符号变化位置）"""
    if len(curve) < 3:
        return []
    arr = smooth(curve, w=smooth_w) if smooth_w > 1 else np.array(curve, dtype=float)
    deriv = np.diff(arr)
    sign_changes = np.where(np.diff(np.sign(deriv)))[0]
    return sign_changes.tolist()


def calculate_cosine_similarity_vec(vec1, vec2) -> float:
    """计算两个向量的余弦相似度"""
    v1 = np.array(vec1, dtype=float)
    v2 = np.array(vec2, dtype=float)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def fit_power_law_exponent(data: List[float]) -> float:
    """简单 MLE 幂律指数估计 (Clauset et al.)"""
    arr = np.array(data, dtype=float)
    arr = arr[arr >= 1]
    if len(arr) < 5:
        return 0.0
    x_min = arr.min()
    n = len(arr)
    alpha = 1 + n / np.sum(np.log(arr / x_min))
    return float(alpha)
