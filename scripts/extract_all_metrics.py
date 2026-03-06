# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.extract_ablation_metrics import extract_metrics, fmt

models = {
    'EBDI': r'e:\PychameProjects\posim\scripts\tianjiaerhuan\output\tianjiaerhuan_baseline_20260221_152957_14B效果好\vis_results',
    'w/o EBDI': r'e:\PychameProjects\posim\scripts\tianjiaerhuan\output\tianjiaerhuan_no_ebdi_20260302_222929\vis_results',
    'CoT': r'e:\PychameProjects\posim\scripts\tianjiaerhuan\output\tianjiaerhuan_cot_20260302_232357\vis_results',
}

header = f"  {'Model':<12} {'confr':>7} {'dTTR':>7} {'dS':>7} {'con_avg':>8} | {'net':>7} {'casc':>7} {'pl':>7} {'top_avg':>8}"
print(header)
print('-' * len(header))

for name, vis in models.items():
    m = extract_metrics(vis)
    confr = fmt(m['confr_sim'])
    dttr = fmt(m['delta_ttr'])
    ds = fmt(m['delta_s'])
    con = fmt(m['con_avg'])
    net = fmt(m['net_sim'])
    casc = fmt(m['casc_sim'])
    pl = fmt(m['casc_pl_sim'])
    top = fmt(m['top_avg'])
    print(f"  {name:<12} {confr:>7} {dttr:>7} {ds:>7} {con:>8} | {net:>7} {casc:>7} {pl:>7} {top:>8}")

print()
# Also print full LaTeX rows
from scripts.extract_ablation_metrics import to_latex_row
for name, vis in models.items():
    m = extract_metrics(vis)
    print(to_latex_row(name, m))
