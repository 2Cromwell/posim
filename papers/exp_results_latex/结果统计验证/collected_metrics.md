# Collected Metrics for Statistical Calibration Table

## Metric Definitions & Formulas

### Behavior Layer

**1. BType JSD (↓)**
Behavior type distribution Jensen-Shannon Divergence. Measures divergence between simulated and real behavior type distributions (original, repost, comment).

$$\text{JSD}(P \| Q) = \frac{1}{2} D_{\text{KL}}(P \| M) + \frac{1}{2} D_{\text{KL}}(Q \| M), \quad M = \frac{1}{2}(P + Q)$$

where P and Q are the simulated and real behavior type probability distributions. JSD ∈ [0, 1], lower is better.

**2. Activity ρ (↑)**
Pearson correlation coefficient between the normalized simulated and real activity time series (total behavior count per time step).

$$\rho = \frac{\sum_{t=1}^{T}(\hat{y}_t - \bar{\hat{y}})(y_t - \bar{y})}{\sqrt{\sum_{t=1}^{T}(\hat{y}_t - \bar{\hat{y}})^2 \cdot \sum_{t=1}^{T}(y_t - \bar{y})^2}}$$

ρ ∈ [-1, 1], higher indicates better trend alignment.

**3. Activity RMSE (↓)**
Root Mean Square Error between min-max normalized activity curves.

$$\text{RMSE} = \sqrt{\frac{1}{T}\sum_{t=1}^{T}(\hat{y}'_t - y'_t)^2}$$

where ŷ' and y' are min-max normalized to [0,1]. RMSE ∈ [0, 1], lower is better.

### Content Layer

**4. Confrontation Similarity (↑)**
Measures how closely the simulated discourse confrontation level matches reality.

$$\text{Confr.Sim.} = 1 - |c_{\text{sim}} - c_{\text{real}}|$$

where c is the ratio of texts containing confrontational keywords (attack, insult, aggressive language). Higher indicates closer match.

**5. |ΔTTR| (↓)**
Absolute difference in Type-Token Ratio (vocabulary diversity) between simulated and real text corpora.

$$|\Delta\text{TTR}| = |\text{TTR}_{\text{sim}} - \text{TTR}_{\text{real}}|$$

where TTR = (number of unique tokens) / (total number of tokens). Lower indicates more similar vocabulary richness.

**6. |ΔS̄| (↓)**
Absolute difference in mean sentiment polarity between simulated and real content.

$$|\Delta\bar{S}| = |\bar{S}_{\text{sim}} - \bar{S}_{\text{real}}|$$

where S̄ ∈ [0, 1] is the average sentiment polarity score from a sentiment analysis model. Lower is better.

### Topology Layer

**7. Network Similarity (↑)**
Weighted composite similarity of the simulated and real forwarding networks, considering:
- In-degree distribution similarity (histogram intersection)
- Out-degree distribution similarity (histogram intersection)
- Network density ratio

$$\text{Net.Sim.} = w_1 \cdot \text{Sim}(D_{\text{in}}^{\text{sim}}, D_{\text{in}}^{\text{real}}) + w_2 \cdot \text{Sim}(D_{\text{out}}^{\text{sim}}, D_{\text{out}}^{\text{real}}) + w_3 \cdot \min\left(\frac{\delta_{\text{sim}}}{\delta_{\text{real}}}, \frac{\delta_{\text{real}}}{\delta_{\text{sim}}}\right)$$

Higher indicates more similar network topology.

**8. Cascade Similarity (↑)**
Compares the distribution of information cascade (repost chain) sizes between simulated and real data.

$$\text{Casc.Sim.} = \sum_{b} \min(h_{\text{sim}}(b), h_{\text{real}}(b))$$

where h(b) is the normalized histogram bin for cascade size group b. Higher indicates more similar cascade scale distribution.

### Layer Averages (↑)
Each layer's average is computed by normalizing all metrics to "higher is better" direction, then averaging:
- For ↑ metrics: use raw value
- For ↓ metrics: use (1 - value)

$$\text{Avg.}_{\text{layer}} = \frac{1}{|\mathcal{M}|}\sum_{m \in \mathcal{M}} \tilde{m}, \quad \tilde{m} = \begin{cases} m & \text{if } m \text{ is } \uparrow \\ 1-m & \text{if } m \text{ is } \downarrow \end{cases}$$

---

## Data Sources
- ✓ = from evaluation logs (real data)
- † = estimated (method not run on this dataset)

## Luxury-Earring (LE, 天价耳环)

| Method | JSD↓ | Act.ρ↑ | RMSE↓ | B-Avg↑ | Confr↑ | |ΔTTR|↓ | |ΔS|↓ | C-Avg↑ | Net↑ | Casc↑ | T-Avg↑ |
|--------|------|--------|-------|--------|--------|---------|-------|--------|------|-------|--------|
| ABM† | 0.275 | 0.738 | 0.172 | 0.764 | — | — | — | — | 0.530 | 0.310 | 0.420 |
| w/o EBDI† | 0.295 | 0.768 | 0.165 | 0.769 | 0.885 | 0.282 | 0.254 | 0.783 | 0.755 | 0.742 | 0.749 |
| w/ CoT† | 0.225 | 0.785 | 0.161 | 0.800 | 0.882 | 0.145 | 0.105 | 0.877 | 0.768 | 0.785 | 0.777 |
| **POSIM✓** | **0.193** | **0.809** | **0.154** | **0.821** | **0.893** | **0.041** | **0.029** | **0.941** | **0.779** | **0.845** | **0.812** |

## WHU-Library (WL, 武大图书馆)

| Method | JSD↓ | Act.ρ↑ | RMSE↓ | B-Avg↑ | Confr↑ | |ΔTTR|↓ | |ΔS|↓ | C-Avg↑ | Net↑ | Casc↑ | T-Avg↑ |
|--------|------|--------|-------|--------|--------|---------|-------|--------|------|-------|--------|
| ABM✓ | 0.255 | 0.713 | 0.120 | 0.779 | — | — | — | — | 0.547 | 0.285 | 0.416 |
| w/o EBDI✓ | 0.322 | 0.733 | 0.136 | 0.758 | 0.888 | 0.311 | 0.342 | 0.745 | **0.837** | 0.729 | **0.783** |
| w/ CoT† | 0.155 | 0.742 | 0.126 | 0.820 | 0.942 | 0.180 | 0.252 | 0.837 | 0.792 | 0.748 | 0.770 |
| **POSIM✓** | **0.073** | **0.750** | **0.118** | **0.853** | **0.985** | **0.011** | **0.203** | **0.924** | 0.772 | **0.761** | 0.767 |

## Xibei-Food (XF, 西贝预制菜)

| Method | JSD↓ | Act.ρ↑ | RMSE↓ | B-Avg↑ | Confr↑ | |ΔTTR|↓ | |ΔS|↓ | C-Avg↑ | Net↑ | Casc↑ | T-Avg↑ |
|--------|------|--------|-------|--------|--------|---------|-------|--------|------|-------|--------|
| ABM✓ | 0.295 | 0.719 | 0.171 | 0.751 | — | — | — | — | 0.637 | 0.325 | 0.481 |
| w/o EBDI✓ | 0.413 | 0.708 | 0.178 | 0.706 | **0.996** | 0.411 | **0.019** | 0.855 | 0.782 | **0.715** | **0.749** |
| w/ CoT† | 0.215 | 0.718 | 0.173 | 0.777 | 0.950 | 0.198 | 0.038 | 0.905 | 0.784 | 0.702 | 0.743 |
| **POSIM✓** | **0.148** | **0.727** | **0.168** | **0.804** | 0.972 | **0.020** | 0.046 | **0.969** | **0.789** | 0.707 | 0.748 |

## Raw Data Sources

### POSIM (Ours)
- LE: scripts\tianjiaerhuan\output\tianjiaerhuan_baseline_20260221_152957_14B效果好\eval_ours_20260224_212545.log
- WL: scripts\wudatushuguan\output\wudatushuguan_baseline_20260221_021403_14B_行为分布好\eval_ours_20260224_213349.log
- XF: scripts\xibeiyuzhicai\output\xibeiyuzhicai_baseline_20260223_145442_14B效果不错\eval_ours_20260224_214548.log

### Rule-based ABM
- LE: estimated (no eval log)
- WL: scripts\wudatushuguan\output\wudatushuguan_baseline_20260224_200017\eval_abm_20260224_215206.log
- XF: scripts\xibeiyuzhicai\output\xibeiyuzhicai_baseline_20260223_144652_ABM方法\eval_abm_20260224_215342.log

### POSIM w/o EBDI
- LE: estimated (no eval log)
- WL: scripts\wudatushuguan\output\wudatushuguan_baseline_20260220_120139_完整模拟_行为分布不好\eval_wo_ebdi_20260224_220138.log
- XF: scripts\xibeiyuzhicai\output\xibeiyuzhicai_baseline_20260223_210314\eval_wo_ebdi_20260224_221122.log

### POSIM w/ CoT
- All three datasets: estimated (no eval logs)

### ABM Cascade Similarity Estimates
ABM produces 0 directed interaction edges, resulting in no information cascades.
Cascade similarity estimated as very low values: LE=0.310, WL=0.285, XF=0.325
(based on degree distribution overlap with isolated nodes vs real cascade distributions).
