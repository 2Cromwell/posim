"""
Generate interactive HTML+ECharts visualization for repost and comment networks.
"""
import json
import os
from collections import defaultdict, Counter
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_DIR = os.path.join(
    BASE_DIR,
    "output",
    "wudatushuguan_baseline_20260221_021403_14B_行为分布好",
    "simulation_results",
)
OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "output",
    "wudatushuguan_baseline_20260221_021403_14B_行为分布好",
    "vis_results",
    "network_visualization",
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    with open(os.path.join(SIM_DIR, "micro_results.json"), "r", encoding="utf-8") as f:
        micro = json.load(f)
    with open(os.path.join(BASE_DIR, "data", "users.json"), "r", encoding="utf-8") as f:
        users_raw = json.load(f)
    users_map = {}
    for u in users_raw:
        users_map[u["user_id"]] = {
            "name": u["username"],
            "type": u["agent_type"],
            "followers": u.get("followers_count", 0),
        }
    return micro, users_map


def build_networks(micro, users_map):
    repost_edges = defaultdict(lambda: {"weight": 0, "emotions": [], "times": [], "contents": []})
    comment_edges = defaultdict(lambda: {"weight": 0, "emotions": [], "times": [], "contents": []})

    user_info = {}
    for d in micro:
        if "user_id" not in d:
            continue
        uid = d["user_id"]
        uname = d["username"]
        utype = d.get("agent_type", "citizen")
        if uid not in user_info:
            followers = users_map.get(uid, {}).get("followers", 0)
            user_info[uid] = {"name": uname, "type": utype, "followers": followers}

    target_author_to_id = {}
    for d in micro:
        if "user_id" in d:
            target_author_to_id[d["username"]] = d["user_id"]

    repost_out_degree = Counter()
    repost_in_degree = Counter()
    comment_out_degree = Counter()
    comment_in_degree = Counter()

    post_to_author_id = {}
    for d in micro:
        if "user_id" not in d:
            continue
        if d["action_type"] in ("short_post", "long_post"):
            post_id = d.get("target_post_id")
            if post_id:
                post_to_author_id[post_id] = d["user_id"]

    for d in micro:
        if "user_id" not in d:
            continue
        from_id = d["user_id"]
        target_author = d.get("target_author", "")
        to_id = target_author_to_id.get(target_author)
        if not to_id or from_id == to_id:
            continue

        emotion = d.get("emotion", "neutral")
        time_str = d.get("time", "")
        content = d.get("content", "")[:80]

        if d["action_type"] in ("repost", "repost_comment"):
            key = (from_id, to_id)
            repost_edges[key]["weight"] += 1
            repost_edges[key]["emotions"].append(emotion)
            repost_edges[key]["times"].append(time_str)
            repost_edges[key]["contents"].append(content)
            repost_out_degree[from_id] += 1
            repost_in_degree[to_id] += 1
        elif d["action_type"] in ("long_comment", "short_comment"):
            key = (from_id, to_id)
            comment_edges[key]["weight"] += 1
            comment_edges[key]["emotions"].append(emotion)
            comment_edges[key]["times"].append(time_str)
            comment_edges[key]["contents"].append(content)
            comment_out_degree[from_id] += 1
            comment_in_degree[to_id] += 1

    return (
        repost_edges, comment_edges,
        user_info,
        repost_out_degree, repost_in_degree,
        comment_out_degree, comment_in_degree,
    )


def compute_time_slices(micro):
    times = set()
    for d in micro:
        t = d.get("time", "")
        if t:
            try:
                dt = datetime.strptime(t, "%Y-%m-%dT%H:%M")
                date_str = dt.strftime("%Y-%m-%d")
                times.add(date_str)
            except Exception:
                pass
    return sorted(times)


def prepare_graph_data(edges, user_info, in_degree, out_degree, top_n=200):
    """Prepare nodes and edges for ECharts graph, keeping top_n nodes by total degree."""
    total_degree = Counter()
    for (src, tgt), info in edges.items():
        total_degree[src] += info["weight"]
        total_degree[tgt] += info["weight"]

    top_users = set(uid for uid, _ in total_degree.most_common(top_n))

    type_colors = {
        "citizen": "#5470c6",
        "kol": "#ee6666",
        "media": "#fac858",
        "government": "#91cc75",
    }
    type_labels = {
        "citizen": "普通用户",
        "kol": "意见领袖",
        "media": "媒体",
        "government": "政府",
    }

    nodes = []
    node_ids = set()
    for uid in top_users:
        info = user_info.get(uid, {"name": uid, "type": "citizen", "followers": 0})
        deg = total_degree.get(uid, 1)
        in_d = in_degree.get(uid, 0)
        out_d = out_degree.get(uid, 0)
        node_size = max(5, min(60, deg ** 0.5 * 3))
        nodes.append({
            "id": uid,
            "name": info["name"],
            "symbolSize": round(node_size, 1),
            "category": info["type"],
            "value": deg,
            "in_degree": in_d,
            "out_degree": out_d,
            "followers": info["followers"],
            "itemStyle": {"color": type_colors.get(info["type"], "#5470c6")},
        })
        node_ids.add(uid)

    edge_list = []
    for (src, tgt), info in edges.items():
        if src in node_ids and tgt in node_ids:
            emotion_counter = Counter(info["emotions"])
            dominant_emotion = emotion_counter.most_common(1)[0][0] if emotion_counter else "neutral"

            emotion_colors = {
                "anger": "#ee6666",
                "happiness": "#91cc75",
                "sadness": "#5470c6",
                "fear": "#9a60b4",
                "disgust": "#ea7ccc",
                "surprise": "#fac858",
                "neutral": "#aaaaaa",
            }

            edge_list.append({
                "source": src,
                "target": tgt,
                "value": info["weight"],
                "lineStyle": {
                    "width": max(0.5, min(5, info["weight"] ** 0.5)),
                    "color": emotion_colors.get(dominant_emotion, "#aaaaaa"),
                    "opacity": min(0.8, 0.2 + info["weight"] * 0.05),
                    "curveness": 0.3,
                },
                "emotion": dominant_emotion,
                "sample_content": info["contents"][0] if info["contents"] else "",
            })

    categories = [
        {"name": "citizen", "itemStyle": {"color": "#5470c6"}},
        {"name": "kol", "itemStyle": {"color": "#ee6666"}},
        {"name": "media", "itemStyle": {"color": "#fac858"}},
        {"name": "government", "itemStyle": {"color": "#91cc75"}},
    ]

    return nodes, edge_list, categories


def compute_network_stats(edges, user_info, in_degree, out_degree):
    total_degree = Counter()
    for uid in set(list(in_degree.keys()) + list(out_degree.keys())):
        total_degree[uid] = in_degree.get(uid, 0) + out_degree.get(uid, 0)

    type_counter = Counter()
    for (src, tgt), info in edges.items():
        src_type = user_info.get(src, {}).get("type", "citizen")
        tgt_type = user_info.get(tgt, {}).get("type", "citizen")
        type_counter[f"{src_type}->{tgt_type}"] += info["weight"]

    top_in = in_degree.most_common(10)
    top_out = out_degree.most_common(10)

    stats = {
        "total_edges": sum(info["weight"] for info in edges.values()),
        "unique_edges": len(edges),
        "total_nodes": len(total_degree),
        "top_in_degree": [
            {"name": user_info.get(uid, {}).get("name", uid),
             "type": user_info.get(uid, {}).get("type", "citizen"),
             "value": cnt}
            for uid, cnt in top_in
        ],
        "top_out_degree": [
            {"name": user_info.get(uid, {}).get("name", uid),
             "type": user_info.get(uid, {}).get("type", "citizen"),
             "value": cnt}
            for uid, cnt in top_out
        ],
        "cross_type_flow": [
            {"name": k, "value": v}
            for k, v in sorted(type_counter.items(), key=lambda x: -x[1])
        ],
    }

    emotion_counter = Counter()
    for info in edges.values():
        for e in info["emotions"]:
            emotion_counter[e] += 1
    stats["emotion_distribution"] = [
        {"name": k, "value": v}
        for k, v in emotion_counter.most_common()
    ]

    return stats


def compute_time_series(micro, action_types):
    time_counts = defaultdict(int)
    for d in micro:
        if "user_id" not in d:
            continue
        if d["action_type"] in action_types:
            t = d.get("time", "")
            if t:
                try:
                    dt = datetime.strptime(t, "%Y-%m-%dT%H:%M")
                    hour_key = dt.strftime("%Y-%m-%d %H:00")
                    time_counts[hour_key] += 1
                except Exception:
                    pass
    sorted_keys = sorted(time_counts.keys())
    return sorted_keys, [time_counts[k] for k in sorted_keys]


def generate_html(
    repost_nodes, repost_edges_list, repost_categories,
    comment_nodes, comment_edges_list, comment_categories,
    repost_stats, comment_stats,
    repost_time_keys, repost_time_values,
    comment_time_keys, comment_time_values,
):
    html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>武大图书馆事件 - 转发与评论网络可视化</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js"></script>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
    background: #0f1923;
    color: #e0e6ed;
    overflow-x: hidden;
}
.header {
    text-align: center;
    padding: 30px 20px 15px;
    background: linear-gradient(135deg, #1a2a3a 0%, #0f1923 100%);
    border-bottom: 1px solid #2a3a4a;
}
.header h1 {
    font-size: 28px;
    font-weight: 700;
    background: linear-gradient(90deg, #5470c6, #ee6666, #fac858, #91cc75);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
}
.header p { color: #7a8a9a; font-size: 14px; }
.tabs {
    display: flex;
    justify-content: center;
    gap: 12px;
    padding: 15px;
    background: #141e2b;
}
.tab-btn {
    padding: 10px 28px;
    border: 1px solid #2a3a4a;
    background: transparent;
    color: #7a8a9a;
    border-radius: 8px;
    cursor: pointer;
    font-size: 15px;
    transition: all 0.3s;
}
.tab-btn.active {
    background: linear-gradient(135deg, #1e3a5f, #2a4a6f);
    color: #e0e6ed;
    border-color: #5470c6;
}
.tab-btn:hover { border-color: #5470c6; color: #c0d0e0; }
.tab-content { display: none; }
.tab-content.active { display: block; }
.dashboard {
    display: grid;
    grid-template-columns: 1fr 360px;
    grid-template-rows: auto auto;
    gap: 15px;
    padding: 15px;
    min-height: calc(100vh - 150px);
}
.graph-panel {
    grid-row: 1 / 3;
    background: #141e2b;
    border-radius: 12px;
    border: 1px solid #2a3a4a;
    overflow: hidden;
    position: relative;
}
.graph-container { width: 100%; height: 700px; }
.side-panel {
    display: flex;
    flex-direction: column;
    gap: 15px;
}
.card {
    background: #141e2b;
    border-radius: 12px;
    border: 1px solid #2a3a4a;
    padding: 18px;
}
.card h3 {
    font-size: 14px;
    color: #5470c6;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.card h3::before {
    content: '';
    display: inline-block;
    width: 3px;
    height: 14px;
    background: #5470c6;
    border-radius: 2px;
}
.stat-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
}
.stat-item {
    background: #1a2a3a;
    border-radius: 8px;
    padding: 12px;
    text-align: center;
}
.stat-item .value {
    font-size: 22px;
    font-weight: 700;
    color: #e0e6ed;
}
.stat-item .label {
    font-size: 11px;
    color: #7a8a9a;
    margin-top: 4px;
}
.rank-list { list-style: none; }
.rank-list li {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid #1a2a3a;
    font-size: 13px;
}
.rank-list li:last-child { border-bottom: none; }
.rank-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 20px;
    height: 20px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 700;
    margin-right: 8px;
    background: #1a2a3a;
    color: #7a8a9a;
}
.rank-num.top1 { background: #ee6666; color: #fff; }
.rank-num.top2 { background: #fac858; color: #1a2a3a; }
.rank-num.top3 { background: #91cc75; color: #1a2a3a; }
.rank-name { flex: 1; }
.rank-type {
    font-size: 10px;
    padding: 2px 6px;
    border-radius: 4px;
    margin-left: 6px;
}
.type-citizen { background: rgba(84,112,198,0.2); color: #5470c6; }
.type-kol { background: rgba(238,102,102,0.2); color: #ee6666; }
.type-media { background: rgba(250,200,88,0.2); color: #fac858; }
.type-government { background: rgba(145,204,117,0.2); color: #91cc75; }
.rank-value { font-weight: 600; color: #e0e6ed; min-width: 40px; text-align: right; }
.chart-small { width: 100%; height: 200px; }
.chart-medium { width: 100%; height: 250px; }
.bottom-charts {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 15px;
    padding: 0 15px 15px;
}
.legend-bar {
    display: flex;
    gap: 16px;
    justify-content: center;
    padding: 8px;
    flex-wrap: wrap;
}
.legend-item {
    display: flex;
    align-items: center;
    gap: 5px;
    font-size: 12px;
    color: #7a8a9a;
    cursor: pointer;
}
.legend-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
}
.controls {
    position: absolute;
    top: 12px;
    left: 12px;
    z-index: 10;
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}
.control-btn {
    padding: 5px 12px;
    background: rgba(20,30,43,0.9);
    border: 1px solid #2a3a4a;
    color: #7a8a9a;
    border-radius: 6px;
    cursor: pointer;
    font-size: 12px;
    transition: all 0.2s;
}
.control-btn:hover, .control-btn.active {
    background: rgba(84,112,198,0.3);
    border-color: #5470c6;
    color: #e0e6ed;
}
.tooltip-custom {
    background: rgba(20,30,43,0.95) !important;
    border: 1px solid #2a3a4a !important;
    border-radius: 8px !important;
    padding: 12px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4) !important;
}
</style>
</head>
<body>

<div class="header">
    <h1>武大图书馆事件 · 信息传播网络分析</h1>
    <p>基于模拟数据的转发网络与评论网络交互式可视化 | 展示 Top 200 活跃节点</p>
</div>

<div class="tabs">
    <button class="tab-btn active" onclick="switchTab('repost')">转发网络 (Repost)</button>
    <button class="tab-btn" onclick="switchTab('comment')">评论网络 (Comment)</button>
</div>

<!-- ==================== REPOST TAB ==================== -->
<div id="tab-repost" class="tab-content active">
<div class="dashboard">
    <div class="graph-panel">
        <div class="controls">
            <button class="control-btn active" onclick="toggleLayout('repost','force')">力导向</button>
            <button class="control-btn" onclick="toggleLayout('repost','circular')">环形</button>
            <button class="control-btn" onclick="highlightType('repost','kol')">高亮KOL</button>
            <button class="control-btn" onclick="highlightType('repost','media')">高亮媒体</button>
            <button class="control-btn" onclick="highlightType('repost','government')">高亮政府</button>
            <button class="control-btn" onclick="highlightType('repost','all')">全部显示</button>
        </div>
        <div id="repost-graph" class="graph-container"></div>
        <div class="legend-bar">
            <div class="legend-item"><div class="legend-dot" style="background:#5470c6"></div>普通用户</div>
            <div class="legend-item"><div class="legend-dot" style="background:#ee6666"></div>意见领袖(KOL)</div>
            <div class="legend-item"><div class="legend-dot" style="background:#fac858"></div>媒体</div>
            <div class="legend-item"><div class="legend-dot" style="background:#91cc75"></div>政府</div>
        </div>
    </div>
    <div class="side-panel">
        <div class="card">
            <h3>网络概况</h3>
            <div class="stat-grid">
                <div class="stat-item">
                    <div class="value" id="repost-total-edges">-</div>
                    <div class="label">总转发次数</div>
                </div>
                <div class="stat-item">
                    <div class="value" id="repost-unique-edges">-</div>
                    <div class="label">独立连接数</div>
                </div>
                <div class="stat-item">
                    <div class="value" id="repost-total-nodes">-</div>
                    <div class="label">参与用户数</div>
                </div>
                <div class="stat-item">
                    <div class="value" id="repost-avg-degree">-</div>
                    <div class="label">平均度</div>
                </div>
            </div>
        </div>
        <div class="card">
            <h3>被转发最多 Top 10 (入度)</h3>
            <ul class="rank-list" id="repost-in-rank"></ul>
        </div>
        <div class="card">
            <h3>转发最多 Top 10 (出度)</h3>
            <ul class="rank-list" id="repost-out-rank"></ul>
        </div>
        <div class="card">
            <h3>情绪分布</h3>
            <div id="repost-emotion-chart" class="chart-small"></div>
        </div>
        <div class="card">
            <h3>跨类型信息流向</h3>
            <div id="repost-flow-chart" class="chart-medium"></div>
        </div>
    </div>
</div>
<div class="bottom-charts">
    <div class="card">
        <h3>转发量时序变化 (按小时)</h3>
        <div id="repost-timeline" class="chart-medium"></div>
    </div>
    <div class="card">
        <h3>入度分布 (幂律特征)</h3>
        <div id="repost-degree-dist" class="chart-medium"></div>
    </div>
</div>
</div>

<!-- ==================== COMMENT TAB ==================== -->
<div id="tab-comment" class="tab-content">
<div class="dashboard">
    <div class="graph-panel">
        <div class="controls">
            <button class="control-btn active" onclick="toggleLayout('comment','force')">力导向</button>
            <button class="control-btn" onclick="toggleLayout('comment','circular')">环形</button>
            <button class="control-btn" onclick="highlightType('comment','kol')">高亮KOL</button>
            <button class="control-btn" onclick="highlightType('comment','media')">高亮媒体</button>
            <button class="control-btn" onclick="highlightType('comment','government')">高亮政府</button>
            <button class="control-btn" onclick="highlightType('comment','all')">全部显示</button>
        </div>
        <div id="comment-graph" class="graph-container"></div>
        <div class="legend-bar">
            <div class="legend-item"><div class="legend-dot" style="background:#5470c6"></div>普通用户</div>
            <div class="legend-item"><div class="legend-dot" style="background:#ee6666"></div>意见领袖(KOL)</div>
            <div class="legend-item"><div class="legend-dot" style="background:#fac858"></div>媒体</div>
            <div class="legend-item"><div class="legend-dot" style="background:#91cc75"></div>政府</div>
        </div>
    </div>
    <div class="side-panel">
        <div class="card">
            <h3>网络概况</h3>
            <div class="stat-grid">
                <div class="stat-item">
                    <div class="value" id="comment-total-edges">-</div>
                    <div class="label">总评论次数</div>
                </div>
                <div class="stat-item">
                    <div class="value" id="comment-unique-edges">-</div>
                    <div class="label">独立连接数</div>
                </div>
                <div class="stat-item">
                    <div class="value" id="comment-total-nodes">-</div>
                    <div class="label">参与用户数</div>
                </div>
                <div class="stat-item">
                    <div class="value" id="comment-avg-degree">-</div>
                    <div class="label">平均度</div>
                </div>
            </div>
        </div>
        <div class="card">
            <h3>被评论最多 Top 10 (入度)</h3>
            <ul class="rank-list" id="comment-in-rank"></ul>
        </div>
        <div class="card">
            <h3>评论最多 Top 10 (出度)</h3>
            <ul class="rank-list" id="comment-out-rank"></ul>
        </div>
        <div class="card">
            <h3>情绪分布</h3>
            <div id="comment-emotion-chart" class="chart-small"></div>
        </div>
        <div class="card">
            <h3>跨类型信息流向</h3>
            <div id="comment-flow-chart" class="chart-medium"></div>
        </div>
    </div>
</div>
<div class="bottom-charts">
    <div class="card">
        <h3>评论量时序变化 (按小时)</h3>
        <div id="comment-timeline" class="chart-medium"></div>
    </div>
    <div class="card">
        <h3>入度分布 (幂律特征)</h3>
        <div id="comment-degree-dist" class="chart-medium"></div>
    </div>
</div>
</div>

<script>
const REPOST_NODES = __REPOST_NODES__;
const REPOST_EDGES = __REPOST_EDGES__;
const REPOST_CATEGORIES = __REPOST_CATEGORIES__;
const REPOST_STATS = __REPOST_STATS__;
const REPOST_TIME_KEYS = __REPOST_TIME_KEYS__;
const REPOST_TIME_VALUES = __REPOST_TIME_VALUES__;

const COMMENT_NODES = __COMMENT_NODES__;
const COMMENT_EDGES = __COMMENT_EDGES__;
const COMMENT_CATEGORIES = __COMMENT_CATEGORIES__;
const COMMENT_STATS = __COMMENT_STATS__;
const COMMENT_TIME_KEYS = __COMMENT_TIME_KEYS__;
const COMMENT_TIME_VALUES = __COMMENT_TIME_VALUES__;

const REPOST_IN_DEGREE = __REPOST_IN_DEGREE__;
const COMMENT_IN_DEGREE = __COMMENT_IN_DEGREE__;

const TYPE_LABELS = {citizen:'普通用户',kol:'意见领袖',media:'媒体',government:'政府'};
const EMOTION_LABELS = {anger:'愤怒',happiness:'喜悦',sadness:'悲伤',fear:'恐惧',disgust:'厌恶',surprise:'惊讶',neutral:'中性'};
const EMOTION_COLORS = {anger:'#ee6666',happiness:'#91cc75',sadness:'#5470c6',fear:'#9a60b4',disgust:'#ea7ccc',surprise:'#fac858',neutral:'#aaaaaa'};

let charts = {};

function switchTab(tab) {
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
    document.getElementById('tab-' + tab).classList.add('active');
    event.target.classList.add('active');
    Object.values(charts).forEach(c => c.resize());
}

function renderRankList(containerId, items) {
    const ul = document.getElementById(containerId);
    ul.innerHTML = items.map((item, i) => {
        const topClass = i < 3 ? ' top' + (i+1) : '';
        const typeClass = 'type-' + item.type;
        return '<li>' +
            '<span class="rank-num' + topClass + '">' + (i+1) + '</span>' +
            '<span class="rank-name">' + item.name +
            '<span class="rank-type ' + typeClass + '">' + (TYPE_LABELS[item.type]||item.type) + '</span></span>' +
            '<span class="rank-value">' + item.value + '</span></li>';
    }).join('');
}

function initGraph(containerId, nodes, edges, categories, networkType) {
    const chart = echarts.init(document.getElementById(containerId));
    charts[containerId] = chart;

    const option = {
        backgroundColor: 'transparent',
        tooltip: {
            trigger: 'item',
            className: 'tooltip-custom',
            formatter: function(params) {
                if (params.dataType === 'node') {
                    const d = params.data;
                    return '<div style="font-size:14px;font-weight:700;margin-bottom:6px">' + d.name + '</div>' +
                        '<div style="color:#7a8a9a;font-size:12px">' +
                        '类型: ' + (TYPE_LABELS[d.category]||d.category) + '<br/>' +
                        '总交互度: ' + d.value + '<br/>' +
                        '入度(被' + (networkType==='repost'?'转发':'评论') + '): ' + d.in_degree + '<br/>' +
                        '出度(' + (networkType==='repost'?'转发':'评论') + '他人): ' + d.out_degree + '<br/>' +
                        '粉丝数: ' + (d.followers||0).toLocaleString() + '</div>';
                } else if (params.dataType === 'edge') {
                    const d = params.data;
                    const src = nodes.find(n => n.id === d.source);
                    const tgt = nodes.find(n => n.id === d.target);
                    return '<div style="font-size:13px;margin-bottom:4px">' +
                        (src?src.name:d.source) + ' → ' + (tgt?tgt.name:d.target) + '</div>' +
                        '<div style="color:#7a8a9a;font-size:12px">' +
                        (networkType==='repost'?'转发':'评论') + '次数: ' + d.value + '<br/>' +
                        '主要情绪: ' + (EMOTION_LABELS[d.emotion]||d.emotion) + '<br/>' +
                        (d.sample_content ? '示例: ' + d.sample_content.substring(0,40) + '...' : '') +
                        '</div>';
                }
            }
        },
        series: [{
            type: 'graph',
            layout: 'force',
            data: nodes,
            links: edges,
            categories: categories,
            roam: true,
            draggable: true,
            label: {
                show: false,
                position: 'right',
                formatter: '{b}',
                fontSize: 10,
                color: '#c0d0e0',
            },
            emphasis: {
                focus: 'adjacency',
                label: { show: true, fontSize: 12 },
                lineStyle: { width: 3 },
            },
            force: {
                repulsion: 120,
                gravity: 0.08,
                edgeLength: [40, 200],
                friction: 0.6,
                layoutAnimation: true,
            },
            edgeSymbol: ['none', 'arrow'],
            edgeSymbolSize: [0, 6],
            lineStyle: { curveness: 0.3, opacity: 0.3 },
            scaleLimit: { min: 0.3, max: 5 },
        }],
    };

    chart.setOption(option);

    chart.on('click', function(params) {
        if (params.dataType === 'node') {
            chart.setOption({
                series: [{ emphasis: { focus: 'adjacency' } }]
            });
        }
    });

    return chart;
}

function toggleLayout(network, layout) {
    const containerId = network + '-graph';
    const chart = charts[containerId];
    if (!chart) return;

    const btns = chart.getDom().parentElement.querySelectorAll('.control-btn');
    btns.forEach(b => {
        if (b.textContent === '力导向' || b.textContent === '环形') {
            b.classList.remove('active');
        }
    });
    event.target.classList.add('active');

    if (layout === 'circular') {
        chart.setOption({
            series: [{
                layout: 'circular',
                circular: { rotateLabel: true },
                label: { show: true, fontSize: 8 },
                force: undefined
            }]
        });
    } else {
        chart.setOption({
            series: [{
                layout: 'force',
                label: { show: false },
                force: { repulsion: 120, gravity: 0.08, edgeLength: [40, 200], friction: 0.6 },
            }]
        });
    }
}

function highlightType(network, type) {
    const containerId = network + '-graph';
    const chart = charts[containerId];
    if (!chart) return;

    const nodes = network === 'repost' ? REPOST_NODES : COMMENT_NODES;
    const typeColors = {citizen:'#5470c6', kol:'#ee6666', media:'#fac858', government:'#91cc75'};

    if (type === 'all') {
        const newNodes = nodes.map(n => ({
            ...n,
            itemStyle: { color: typeColors[n.category] || '#5470c6', opacity: 1 },
        }));
        chart.setOption({ series: [{ data: newNodes }] });
    } else {
        const newNodes = nodes.map(n => ({
            ...n,
            itemStyle: {
                color: n.category === type ? typeColors[n.category] : '#333',
                opacity: n.category === type ? 1 : 0.15,
                borderColor: n.category === type ? '#fff' : 'transparent',
                borderWidth: n.category === type ? 1 : 0,
            },
        }));
        chart.setOption({ series: [{ data: newNodes }] });
    }
}

function initEmotionChart(containerId, emotionData) {
    const chart = echarts.init(document.getElementById(containerId));
    charts[containerId] = chart;
    chart.setOption({
        backgroundColor: 'transparent',
        tooltip: { trigger: 'item', formatter: '{b}: {c} ({d}%)' },
        series: [{
            type: 'pie',
            radius: ['35%', '70%'],
            center: ['50%', '50%'],
            data: emotionData.map(d => ({
                name: EMOTION_LABELS[d.name] || d.name,
                value: d.value,
                itemStyle: { color: EMOTION_COLORS[d.name] || '#aaa' }
            })),
            label: { color: '#c0d0e0', fontSize: 10 },
            emphasis: { itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0,0,0,0.5)' } },
        }]
    });
}

function initFlowChart(containerId, flowData) {
    const chart = echarts.init(document.getElementById(containerId));
    charts[containerId] = chart;

    const sankey_nodes_set = new Set();
    const sankey_links = [];
    flowData.forEach(d => {
        const parts = d.name.split('->');
        const src = (TYPE_LABELS[parts[0]]||parts[0]) + '(发起)';
        const tgt = (TYPE_LABELS[parts[1]]||parts[1]) + '(接收)';
        sankey_nodes_set.add(src);
        sankey_nodes_set.add(tgt);
        sankey_links.push({ source: src, target: tgt, value: d.value });
    });

    const typeColorMap = {
        '普通用户(发起)': '#5470c6', '普通用户(接收)': '#5470c6',
        '意见领袖(发起)': '#ee6666', '意见领袖(接收)': '#ee6666',
        '媒体(发起)': '#fac858', '媒体(接收)': '#fac858',
        '政府(发起)': '#91cc75', '政府(接收)': '#91cc75',
    };

    chart.setOption({
        backgroundColor: 'transparent',
        tooltip: { trigger: 'item' },
        series: [{
            type: 'sankey',
            data: Array.from(sankey_nodes_set).map(n => ({
                name: n,
                itemStyle: { color: typeColorMap[n] || '#5470c6' },
            })),
            links: sankey_links,
            lineStyle: { color: 'gradient', opacity: 0.4 },
            emphasis: { focus: 'adjacency' },
            label: { color: '#c0d0e0', fontSize: 11 },
            nodeWidth: 18,
            nodeGap: 12,
        }]
    });
}

function initTimeline(containerId, keys, values, label) {
    const chart = echarts.init(document.getElementById(containerId));
    charts[containerId] = chart;
    chart.setOption({
        backgroundColor: 'transparent',
        tooltip: { trigger: 'axis' },
        xAxis: {
            type: 'category',
            data: keys,
            axisLabel: {
                color: '#7a8a9a', fontSize: 10, rotate: 45,
                formatter: function(v) { return v.substring(5); }
            },
            axisLine: { lineStyle: { color: '#2a3a4a' } },
        },
        yAxis: {
            type: 'value',
            axisLabel: { color: '#7a8a9a', fontSize: 10 },
            splitLine: { lineStyle: { color: '#1a2a3a' } },
        },
        grid: { left: 50, right: 15, top: 15, bottom: 60 },
        series: [{
            type: 'line',
            data: values,
            smooth: true,
            symbol: 'none',
            areaStyle: {
                color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                    { offset: 0, color: 'rgba(84,112,198,0.4)' },
                    { offset: 1, color: 'rgba(84,112,198,0.05)' }
                ])
            },
            lineStyle: { color: '#5470c6', width: 2 },
        }],
        dataZoom: [{ type: 'inside', start: 0, end: 100 }],
    });
}

function initDegreeDist(containerId, inDegreeData) {
    const chart = echarts.init(document.getElementById(containerId));
    charts[containerId] = chart;

    const degreeCount = {};
    inDegreeData.forEach(d => {
        degreeCount[d] = (degreeCount[d] || 0) + 1;
    });
    const scatter = Object.entries(degreeCount)
        .map(([deg, cnt]) => [Math.log10(Number(deg) || 1), Math.log10(cnt)])
        .sort((a, b) => a[0] - b[0]);

    chart.setOption({
        backgroundColor: 'transparent',
        tooltip: {
            trigger: 'item',
            formatter: function(p) {
                return '度: ' + Math.round(Math.pow(10, p.data[0])) + '<br/>频次: ' + Math.round(Math.pow(10, p.data[1]));
            }
        },
        xAxis: {
            name: 'log(入度)',
            nameTextStyle: { color: '#7a8a9a', fontSize: 11 },
            axisLabel: { color: '#7a8a9a', fontSize: 10 },
            axisLine: { lineStyle: { color: '#2a3a4a' } },
            splitLine: { lineStyle: { color: '#1a2a3a' } },
        },
        yAxis: {
            name: 'log(频次)',
            nameTextStyle: { color: '#7a8a9a', fontSize: 11 },
            axisLabel: { color: '#7a8a9a', fontSize: 10 },
            axisLine: { lineStyle: { color: '#2a3a4a' } },
            splitLine: { lineStyle: { color: '#1a2a3a' } },
        },
        grid: { left: 55, right: 15, top: 30, bottom: 40 },
        series: [{
            type: 'scatter',
            data: scatter,
            symbolSize: 8,
            itemStyle: { color: '#5470c6', opacity: 0.7 },
        }],
    });
}

function populateStats(prefix, stats) {
    document.getElementById(prefix + '-total-edges').textContent = stats.total_edges.toLocaleString();
    document.getElementById(prefix + '-unique-edges').textContent = stats.unique_edges.toLocaleString();
    document.getElementById(prefix + '-total-nodes').textContent = stats.total_nodes.toLocaleString();
    const avgDeg = stats.total_nodes > 0 ? (stats.total_edges * 2 / stats.total_nodes).toFixed(1) : '-';
    document.getElementById(prefix + '-avg-degree').textContent = avgDeg;
    renderRankList(prefix + '-in-rank', stats.top_in_degree);
    renderRankList(prefix + '-out-rank', stats.top_out_degree);
}

function init() {
    // Repost network
    initGraph('repost-graph', REPOST_NODES, REPOST_EDGES, REPOST_CATEGORIES, 'repost');
    populateStats('repost', REPOST_STATS);
    initEmotionChart('repost-emotion-chart', REPOST_STATS.emotion_distribution);
    initFlowChart('repost-flow-chart', REPOST_STATS.cross_type_flow);
    initTimeline('repost-timeline', REPOST_TIME_KEYS, REPOST_TIME_VALUES, '转发');
    initDegreeDist('repost-degree-dist', REPOST_IN_DEGREE);

    // Comment network
    initGraph('comment-graph', COMMENT_NODES, COMMENT_EDGES, COMMENT_CATEGORIES, 'comment');
    populateStats('comment', COMMENT_STATS);
    initEmotionChart('comment-emotion-chart', COMMENT_STATS.emotion_distribution);
    initFlowChart('comment-flow-chart', COMMENT_STATS.cross_type_flow);
    initTimeline('comment-timeline', COMMENT_TIME_KEYS, COMMENT_TIME_VALUES, '评论');
    initDegreeDist('comment-degree-dist', COMMENT_IN_DEGREE);
}

window.addEventListener('resize', () => Object.values(charts).forEach(c => c.resize()));
init();
</script>
</body>
</html>"""
    return html


def main():
    print("Loading data...")
    micro, users_map = load_data()

    print("Building networks...")
    (
        repost_edges, comment_edges,
        user_info,
        repost_out_degree, repost_in_degree,
        comment_out_degree, comment_in_degree,
    ) = build_networks(micro, users_map)

    print("Preparing repost graph data...")
    repost_nodes, repost_edges_list, repost_categories = prepare_graph_data(
        repost_edges, user_info, repost_in_degree, repost_out_degree, top_n=200
    )
    print(f"  Repost graph: {len(repost_nodes)} nodes, {len(repost_edges_list)} edges")

    print("Preparing comment graph data...")
    comment_nodes, comment_edges_list, comment_categories = prepare_graph_data(
        comment_edges, user_info, comment_in_degree, comment_out_degree, top_n=200
    )
    print(f"  Comment graph: {len(comment_nodes)} nodes, {len(comment_edges_list)} edges")

    print("Computing stats...")
    repost_stats = compute_network_stats(repost_edges, user_info, repost_in_degree, repost_out_degree)
    comment_stats = compute_network_stats(comment_edges, user_info, comment_in_degree, comment_out_degree)

    print("Computing time series...")
    repost_time_keys, repost_time_values = compute_time_series(micro, ("repost", "repost_comment"))
    comment_time_keys, comment_time_values = compute_time_series(micro, ("long_comment", "short_comment"))

    repost_in_degree_values = list(repost_in_degree.values())
    comment_in_degree_values = list(comment_in_degree.values())

    print("Generating HTML...")
    html = generate_html(
        repost_nodes, repost_edges_list, repost_categories,
        comment_nodes, comment_edges_list, comment_categories,
        repost_stats, comment_stats,
        repost_time_keys, repost_time_values,
        comment_time_keys, comment_time_values,
    )

    html = html.replace("__REPOST_NODES__", json.dumps(repost_nodes, ensure_ascii=False))
    html = html.replace("__REPOST_EDGES__", json.dumps(repost_edges_list, ensure_ascii=False))
    html = html.replace("__REPOST_CATEGORIES__", json.dumps(repost_categories, ensure_ascii=False))
    html = html.replace("__REPOST_STATS__", json.dumps(repost_stats, ensure_ascii=False))
    html = html.replace("__REPOST_TIME_KEYS__", json.dumps(repost_time_keys, ensure_ascii=False))
    html = html.replace("__REPOST_TIME_VALUES__", json.dumps(repost_time_values, ensure_ascii=False))
    html = html.replace("__REPOST_IN_DEGREE__", json.dumps(repost_in_degree_values, ensure_ascii=False))

    html = html.replace("__COMMENT_NODES__", json.dumps(comment_nodes, ensure_ascii=False))
    html = html.replace("__COMMENT_EDGES__", json.dumps(comment_edges_list, ensure_ascii=False))
    html = html.replace("__COMMENT_CATEGORIES__", json.dumps(comment_categories, ensure_ascii=False))
    html = html.replace("__COMMENT_STATS__", json.dumps(comment_stats, ensure_ascii=False))
    html = html.replace("__COMMENT_TIME_KEYS__", json.dumps(comment_time_keys, ensure_ascii=False))
    html = html.replace("__COMMENT_TIME_VALUES__", json.dumps(comment_time_values, ensure_ascii=False))
    html = html.replace("__COMMENT_IN_DEGREE__", json.dumps(comment_in_degree_values, ensure_ascii=False))

    output_path = os.path.join(OUTPUT_DIR, "network_visualization.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nVisualization saved to: {output_path}")
    print("Open this file in a browser to explore the networks!")


if __name__ == "__main__":
    main()
