# -*- coding: utf-8 -*-
"""
从评估结果中提取消融实验所需的指标，输出LaTeX表格行
用法: python extract_ablation_metrics.py <vis_results_dir>
"""
import sys, json
from pathlib import Path

def extract_metrics(vis_dir: str) -> dict:
    vis = Path(vis_dir)
    cal = vis / "calibration"
    
    # 1. 行为层
    beh_file = cal / "behavior_calibration" / "behavior_calibration_metrics.json"
    hot_file = cal / "hotness_calibration" / "hotness_calibration_metrics.json"
    
    jsd = pearson = rmse = None
    if beh_file.exists():
        beh = json.loads(beh_file.read_text(encoding='utf-8'))
        jsd = beh.get("type_distribution", {}).get("type_distribution_jsd")
    if hot_file.exists():
        hot = json.loads(hot_file.read_text(encoding='utf-8'))
        total_norm = hot.get("curve_similarity", {}).get("total", {}).get("normalized", {})
        pearson = total_norm.get("pearson")
        rmse = total_norm.get("rmse")
    
    # 2. 内容层
    emo_file = cal / "emotion_calibration" / "emotion_calibration_metrics.json"
    opi_file = cal / "opinion_index" / "opinion_index_metrics.json"
    
    confr_sim = delta_ttr = delta_s = None
    if opi_file.exists():
        opi = json.loads(opi_file.read_text(encoding='utf-8'))
        # 对抗性相似度
        dc = opi.get("discourse_confrontation", {})
        confr_sim = dc.get("confrontation_similarity")
        # 词汇多样性 - content_layer_metrics中的ttr_diff
        cl = opi.get("content_layer_metrics", {})
        delta_ttr = cl.get("ttr_diff")
        if delta_ttr is None:
            # 尝试从semantic_diversity中手动计算
            sd = opi.get("semantic_diversity", {})
            # TTR not available in semantic_diversity, will be None
    if emo_file.exists():
        emo = json.loads(emo_file.read_text(encoding='utf-8'))
        ss = emo.get("sentiment_score", {})
        sim_avg = ss.get("sim_avg_score")
        real_avg = ss.get("real_avg_score")
        if sim_avg is not None and real_avg is not None:
            delta_s = abs(sim_avg - real_avg)
    
    # 3. 拓扑层
    net_file = cal / "network_calibration" / "network_calibration_metrics.json"
    net_sim = casc_sim = casc_pl_sim = None
    if net_file.exists():
        net = json.loads(net_file.read_text(encoding='utf-8'))
        ns = net.get("network_similarity", {})
        if isinstance(ns, dict):
            net_sim = ns.get("overall_network_similarity")
        else:
            net_sim = ns
        cs = net.get("cascade_structure", {})
        if isinstance(cs, dict):
            casc_sim = cs.get("cascade_scale_similarity")
            casc_pl_sim = cs.get("cascade_power_law_similarity")
    
    # 4. 从 evaluation_report.json 补充缺失值
    report_file = vis / "evaluation_report.json"
    if report_file.exists():
        report = json.loads(report_file.read_text(encoding='utf-8'))
        results = report.get("results", {})
        
        if net_sim is None:
            net_sim = results.get("network_calibration", {}).get("network_similarity")
        if casc_sim is None:
            casc_sim = results.get("network_calibration", {}).get("cascade_scale_similarity")
        if confr_sim is None:
            confr_sim = results.get("opinion_index", {}).get("opinion_evolution_index")
    
    # 计算均值
    def avg_up(vals):
        return sum(vals) / len(vals) if vals else None
    
    # 行为层均值: (1-JSD) + pearson + (1-RMSE) / 3
    beh_vals = []
    if jsd is not None: beh_vals.append(1 - jsd)
    if pearson is not None: beh_vals.append(pearson)
    if rmse is not None: beh_vals.append(1 - rmse)
    beh_avg = avg_up(beh_vals)
    
    # 内容层均值: confr_sim + (1-delta_ttr) + (1-delta_s) / 3
    con_vals = []
    if confr_sim is not None: con_vals.append(confr_sim)
    if delta_ttr is not None: con_vals.append(1 - delta_ttr)
    if delta_s is not None: con_vals.append(1 - delta_s)
    con_avg = avg_up(con_vals)
    
    # 拓扑层均值: (net_sim + casc_sim + casc_pl_sim) / n
    top_vals = []
    if net_sim is not None: top_vals.append(net_sim)
    if casc_sim is not None: top_vals.append(casc_sim)
    if casc_pl_sim is not None: top_vals.append(casc_pl_sim)
    top_avg = avg_up(top_vals)
    
    return {
        "jsd": jsd, "pearson": pearson, "rmse": rmse, "beh_avg": beh_avg,
        "confr_sim": confr_sim, "delta_ttr": delta_ttr, "delta_s": delta_s, "con_avg": con_avg,
        "net_sim": net_sim, "casc_sim": casc_sim, "casc_pl_sim": casc_pl_sim, "top_avg": top_avg
    }

def fmt(v, digits=3):
    if v is None: return "---"
    return f"{v:.{digits}f}"

def to_latex_row(method_name, m):
    return (f"& {method_name:22s} & {fmt(m['jsd'])} & {fmt(m['pearson'])} & {fmt(m['rmse'])} & {fmt(m['beh_avg'])} "
            f"& {fmt(m['confr_sim'])} & {fmt(m['delta_ttr'])} & {fmt(m['delta_s'])} & {fmt(m['con_avg'])} "
            f"& {fmt(m['net_sim'])} & {fmt(m['casc_sim'])} & {fmt(m.get('casc_pl_sim','---'))} & {fmt(m['top_avg'])} \\\\")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_ablation_metrics.py <vis_results_dir> [method_name]")
        sys.exit(1)
    
    vis_dir = sys.argv[1]
    method = sys.argv[2] if len(sys.argv) > 2 else "POSIM w/o EBDI"
    
    m = extract_metrics(vis_dir)
    print("\n=== Raw Metrics ===")
    for k, v in m.items():
        print(f"  {k}: {fmt(v)}")
    
    print(f"\n=== LaTeX Row ===")
    print(to_latex_row(method, m))
