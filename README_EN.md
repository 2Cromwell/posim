<p align="center">
  <img src="assets/posim_logo.png" alt="POSIM Logo" width="600">
</p>

<h3 align="center">POSIM — A General-Purpose Social Media Public Opinion Simulation Framework Based on Metacognitive Agents</h3>

<p align="center">
  <em>"All models are wrong, but some are useful." — George E. P. Box</em>
</p>

<p align="center">
  <a href="README.md">中文</a> | <a href="README_EN.md">English</a>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg" alt="PyTorch"></a>
  <a href="https://openai.com/"><img src="https://img.shields.io/badge/LLM-OpenAI_Compatible-412991.svg" alt="LLM"></a>
</p>

---

## 📖 Table of Contents

- [💡 Why POSIM?](#why-posim)
- [✨ Key Contributions](#key-contributions)
- [🏗️ Framework Overview](#framework-overview)
- [🧠 EBDI Metacognitive Agent](#ebdi-metacognitive-agent)
- [🌍 Simulation Environment](#simulation-environment)
- [🧪 Strategy Evaluation & Computational Experiments](#strategy-evaluation--computational-experiments)
- [🛡️ Three-Tier Validation Framework](#three-tier-validation-framework)
- [💾 Datasets](#datasets)
- [📊 Results & Analysis](#results--analysis)
- [🌳 Project Structure](#project-structure)
- [⚙️ Installation](#installation)
- [🚀 Quick Start](#quick-start)
- [🔌 Extensibility](#extensibility)
- [📝 Citation](#citation)
- [📄 License](#license)

---

## 💡 Why POSIM?

A breaking event can sweep across an entire social network within hours — tens of thousands of users flood the comment sections, emotions escalate through repost chains, and a single post from an opinion leader can reshape the course of public discourse. Understanding and anticipating these complex group dynamics matters for social governance, crisis response, and public policy.

Yet real-world social experiments face fundamental barriers: ethical constraints, irreproducibility, and uncontrollable variables. Traditional computational approaches — epidemic models, threshold cascade models, and rule-based agent-based modeling (ABM) — each have strengths, but they share a common bottleneck: **they cannot explicitly model individual cognitive processes**. Rule-driven agents neither perceive complex environmental information nor simulate emotional evolution, motivational reasoning, or autonomous decision-making.

Recent breakthroughs in Large Language Models (LLMs) open up new possibilities: semantic understanding, contextual reasoning, and autonomous decision-making could finally enable simulation agents to truly "understand" events. However, most existing work treats LLMs as end-to-end behavior mappers — no explicit intermediate cognitive states, no guarantees on behavioral mechanisms over long simulation horizons.

**POSIM** was built to address precisely this challenge.

| Capability | S3 | HiSim | OASIS | TrendSim | LMAgent | **POSIM** |
|------------|:---:|:---:|:---:|:---:|:---:|:---:|
| Cognitive Mechanism | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| Real-Data Validation | ✓ | ✓ | ✗ | ✗ | ✓ | **✓** |
| Strategy Evaluation | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| Multi-Type Agents | ✗ | ✓ | ✗ | ✓ | ✓ | **✓** |
| Temporal Precision | ★★★ | ★★ | ★★★★ | ★★★★ | ★★ | **★★★★★** |
| Scalability | ★★★ | ★★ | ★★★★ | ★★★ | ★★ | **★★★★★** |

## ✨ Key Contributions

1. **EBDI Metacognitive Agent Architecture** — Embeds LLMs into a layered cognitive framework (Perception → Belief → Desire → Intention → Action). Three cognitive subsystems are each powered by independent LLM calls, passing structured intermediate states between modules. The behavior generation process becomes fully traceable — no longer a "prompt in, answer out" black box.

2. **Hybrid Time-Event Driven Simulation Environment** — Hawkes self-exciting point processes jointly model exogenous event shocks (breaking news, official statements) and endogenous user interactions (the snowball effect of reposts and comments), combined with circadian rhythm modulation, reproducing the non-stationary "burst → sustain → decay" activity patterns at minute-level temporal resolution.

3. **Three-Tier Progressive Validation Framework** — Drawing on the classical V&V methodology from simulation engineering: micro-level behavioral mechanism calibration → macro-level emergent phenomenon verification → statistical result consistency alignment, building simulation credibility layer by layer.

4. **Highly Decoupled Modular Architecture** — Agents, simulation environment, and strategy evaluation communicate through standard interfaces and can be independently replaced — swap the cognitive architecture, change the time engine, or plug in new evaluation metrics, all without touching other modules.

---

## 🏗️ Framework Overview

<p align="center">
  <img src="assets/framework_overview.png" alt="POSIM Framework Overview" width="95%">
</p>
<p align="center"><b>Figure 1.</b> POSIM framework architecture. Left: EBDI metacognitive agent cognitive pipeline. Upper-center: Hawkes process-driven simulation environment and virtual social media platform. Lower-right: Strategy evaluation module (Intervenor-Simulator-Evaluator).</p>

POSIM consists of three core components working in concert:

> **(1) Metacognitive Agents** — Built on the EBDI cognitive architecture, generating multi-type agents (citizens, KOLs, media, government) from real user data and LLM-driven deep interviews. Each agent maintains a complete cognitive state from identity beliefs to real-time emotions.
>
> **(2) Simulation Environment** — A Hawkes self-exciting point process time engine controls agent activation timing. A virtual social media platform provides personalized recommendations, social networks, and trending topics — together forming the virtual world in which agents perceive and interact.
>
> **(3) Strategy Evaluation** — Three modules — Intervenor, Simulator, and Evaluator — collaborate to support event injection, node control, and platform policy interventions. Checkpoint callbacks generate parallel evolution trajectories for counterfactual reasoning.

---

## 🧠 EBDI Metacognitive Agent

Traditional reactive agents are little more than stateless behavior generators — input in, output out, with no insight into what happened in between. POSIM takes a fundamentally different approach: it extends the classical BDI cognitive architecture with an emotional dimension, building metacognitive agents with **explicit self-aware cognitive states** and **auditable multi-stage decision chains**.

The cognitive pipeline is formalized as:

$$\text{Perception}(P_t) \;\to\; \text{Belief}(B_t) \;\to\; \text{Desire}(D_t) \;\to\; \text{Intention}(I_t) \;\to\; \text{Action}(A_t)$$

### 💭 Belief Subsystem — How Agents "Understand the World"

Psychological research shows that different cognitive layers have different stability — core personality traits remain highly stable in the short term, while immediate emotions fluctuate rapidly with information input. Based on this cognitive stratification, POSIM designs a four-layer belief system:

```
  ┌────────────────────────────────────────────────────────────────────┐
  │  B^id  — Role Identity Belief    Gender, location, occupation...   │  ← Fixed (personality anchor)
  │  B^psy — Psychological Belief    Conformity, paranoia, catharsis.. │  ← Highly stable
  │  B^evt — Event Opinion Belief    Stance & reasoning on entities    │  ← Dynamically evolving
  │  B^emo — Emotional Arousal       [happy, sad, angry, fear, ...]    │  ← Real-time fluctuation
  └────────────────────────────────────────────────────────────────────┘
                              ▲ Modification difficulty decreasing ▼
```

Notable design details:

- **Psychological belief initialization**: Based on analysis of real public opinion events, we categorize typical psychological patterns — self-actualization, curiosity-seeking, stress-relief, anti-establishment, herd-following, etc. — each with ~30 cognition items distilled from real data (e.g., "what mainstream media doesn't report is often the truth," "the harsher the criticism, the more satisfying"). LLM personalizes the matching based on each user's historical behavior.
- **Event opinion extraction**: LLM-driven structured deep interviews, with each opinion represented as a quadruple $\langle t, \text{subject}, \text{opinion}, \text{reason} \rangle$.
- **Emotion triple-drive**: Temporal decay ($\mathbf{e}(t) = \mathbf{e}(t_0) \cdot e^{-\lambda_e(t-t_0)}$), content stimulation, and social contagion from neighbors ($(1-\rho) \cdot \mathbf{e}_i + \rho \cdot \bar{\mathbf{e}}_{\text{neighbor}}$).
- **Explicit cognitive bias injection**: Belief updates incorporate confirmation bias, anchoring effect, emotion-first reasoning, and simplistic attribution through prompt engineering.

### 🎯 Desire Subsystem — What Agents "Want to Do"

Social media participation motivations are highly diverse — some users vent anger, others seek information, and many are simply there for the spectacle. The desire subsystem leverages LLM commonsense reasoning to infer behavioral drivers, outputting a weighted motivation list:

> *Emotional venting (0.8)* · *Justice advocacy (0.6)* · *Self-expression (0.4)* · *Information seeking (0.3)* · ...

This motivation list constrains downstream intention planning — when emotional venting dominates, agents gravitate toward short comments with intense emotional expression; when information-seeking leads, they tend toward reposts with commentary or long-form posts.

### 🛠️ Intention Subsystem — How Agents "Take Action"

Unlike single-step behavior generation, the intention subsystem uses a **three-level chain-of-thought** for progressive decision-making:

```
  L1 — What to do & to whom
    Select from atomic operations (like / repost / repost+comment / short comment /
    long comment / short post / long post); identify target

  L2 — How to express it
    Four orthogonal dimensions: Emotion (type+intensity) × Stance (support/oppose/neutral)
    × Style (analytical/sarcastic/aggressive/empathetic/questioning) × Narrative (factual/labeling/call-to-action/authority-citing)

  L3 — What exactly to say
    Generate role-consistent Weibo text under L1+L2 constraints
```

This decomposition provides **explicit strategy constraints** for content generation, preventing LLMs from defaulting to bland, averaged-out text. Every decision at each level is recorded, making the entire process fully traceable.

### 🎭 Four Heterogeneous Agent Types

Real public opinion events involve complex interactions among citizens, opinion leaders, media, and government. All four types share the same EBDI cognitive pipeline, differentiated through role-specific guiding prompts:

| Type | Role | Behavioral Characteristics | Typical Expression |
|------|------|--------------------------|-------------------|
| **Citizen** | Primary opinion participants | Colloquial, fragmented, emotion-driven | Impulsive expression under high arousal |
| **KOL** | Key intermediary in two-step flow | Independent views, agenda-setting | Significant influence on downstream belief updates |
| **Media** | Information collection & dissemination | Formal, restrained, timely | Information confirmation at critical junctures |
| **Government** | Official stance & public governance | Low frequency, high authority | Post-fermentation statements with turning-point impact |

> The behavioral patterns of all four types are **not pre-programmed** — they emerge autonomously through the EBDI cognitive pipeline under respective role prompt constraints.

---

## 🌍 Simulation Environment

### ⏱️ Hawkes Point Process Time Engine

Real-world public opinion activity exhibits highly uneven temporal distribution — a single piece of breaking news can trigger thousands of reposts within minutes, while inter-event periods see dramatically lower activity. Uniform time-stepping simply cannot reproduce this.

POSIM models collective activity intensity through Hawkes self-exciting point processes. The core intuition: think of it as "contagion" — each event temporarily raises the probability of subsequent events, much like how a viral Weibo post stimulates more participation in the short term, with the stimulus gradually decaying.

$$\lambda(t) = \underbrace{\mu}_{\text{background}} + \underbrace{\sum \alpha_{ext} e^{-\beta_{ext}(t - t_i)}}_{\text{exogenous (high intensity, slow decay)}} + \underbrace{\sum \alpha_{int} e^{-\beta_{int}(t - t_j)}}_{\text{endogenous (low intensity, fast decay)}}$$

| Parameter | Symbol | Default | Meaning |
|-----------|--------|:-------:|---------|
| Background rate | $\mu$ | 0.01 | Baseline posting activity during lulls |
| Exogenous intensity | $\alpha_{ext}$ | 0.08 | Impact from news breaks, official statements |
| Exogenous decay | $\beta_{ext}$ | 0.005 | Duration of external event influence |
| Endogenous intensity | $\alpha_{int}$ | 0.005 | Snowball effect from user interactions |
| Endogenous decay | $\beta_{int}$ | 0.16 | Short-term stimulus decay from interactions |
| Circadian amplitude | $s_{circ}$ | 0.3 | Nighttime activity reduction magnitude |

### 📱 Virtual Social Media Platform

Once the time engine determines which agents to activate each step, the virtual platform determines what they **see** and what **attracts** them:

**Social Network** — Three-layer directed structure: follower network (static infrastructure), real-time repost network and comment network (growing dynamically during simulation).

**Content Recommendation** — Dual-channel retrieval (relationship-based + public domain), three-dimensional weighted ranking:

$$S_{exp}(u, p) = \alpha \cdot \underbrace{H(u, p)}_{\text{homophily}} + \beta \cdot \underbrace{P(p)}_{\text{popularity}} + \gamma \cdot \underbrace{R(p)}_{\text{freshness}}$$

The system retains exploration slots to break filter bubbles, maintains recommendation history to prevent repetition, and uses [BGE-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5) for semantic encoding and content deduplication (cosine similarity threshold: 0.92).

**Trending Topics** — Automatic hashtag heat tracking with interaction signals and temporal decay, simulating the attention amplification effect of real platform trending lists.

---

## 🧪 Strategy Evaluation & Computational Experiments

POSIM is not just a simulator — it doubles as a computational experiment platform for strategy evaluation. The question decision-makers typically face: "If we take a certain intervention, how will public opinion shift?"

```
         ┌─────────────┐        ┌─────────────┐        ┌─────────────┐
         │  Intervenor  │  ──→  │  Simulator   │  ──→  │  Evaluator   │
         └──────┬───────┘        └──────┬───────┘        └──────┬───────┘
                │                       │                       │
     ┌──────────┼──────────┐    Checkpoint callback      Activity curves
     │          │          │    → parallel trajectories   Sentiment · Opinion
  Event Queue  Node      Platform  ↓ Counterfactual       Topic migration
  Injection   Control    Policy      reasoning            ↓ Multi-dim eval
```

- **Intervenor** — Three granularity levels: event queue (e.g., inject official statement), node control (e.g., adjust a specific KOL's belief state), platform policy (e.g., modify recommendation weights or restrict content propagation)
- **Simulator** — Checkpoint callback periodically saves complete state snapshots; load from any checkpoint and inject different interventions to generate parallel evolution trajectories
- **Evaluator** — Fully decoupled from the simulation engine; reads simulation logs through standard interfaces for multi-dimensional quantitative assessment

---

## 🛡️ Three-Tier Validation Framework

Drawing on the classical V&V (Verification and Validation) principles, POSIM establishes a progressive validation framework covering **mechanism → phenomena → data**:

```
  ╔════════════════════════════════════════╗
  ║  Tier 1: Micro-Level Mechanism         ║  ← "Did we build the model right?"
  ║  · Cognitive-behavior chain (0–5)      ║
  ║  · Personality stability (0–1)         ║
  ║  · Decision robustness (0–1)           ║
  ╠════════════════════════════════════════╣
  ║  Tier 2: Macro-Level Emergence         ║  ← Can micro mechanisms spontaneously
  ║  · Opinion lifecycle                   ║     produce theory-predicted phenomena?
  ║  · Multi-agent heterogeneity           ║
  ║  · Emotional arousal & polarization    ║
  ║  · Scale-free topology & cascade PL    ║
  ╠════════════════════════════════════════╣
  ║  Tier 3: Statistical Consistency       ║  ← "Are the model results accurate?"
  ║  · Behavior layer (3 metrics)          ║
  ║  · Content layer (3 metrics)           ║
  ║  · Topology layer (3 metrics)          ║
  ╚════════════════════════════════════════╝
```

Tier 3 encompasses **9 quantitative metrics** across three dimensions:

| Dimension | Metric | Description |
|-----------|--------|-------------|
| **Behavior** | BType JSD ↓ | Jensen-Shannon divergence of behavioral type distributions |
| | Act. ρ ↑ | Pearson correlation of activity time series |
| | Act. RMSE ↓ | Root mean square error of activity curves |
| **Content** | Confr. Sim. ↑ | Discourse confrontation (confrontational/rational/neutral) distribution similarity |
| | \|ΔTTR\| ↓ | Lexical diversity deviation (Type-Token Ratio) |
| | \|ΔS̄\| ↓ | Group sentiment mean deviation |
| **Topology** | Net. Sim. ↑ | Interaction network topological similarity |
| | Casc. Sim. ↑ | Information cascade size distribution similarity |
| | Casc. PL ↑ | Cascade power-law exponent proximity |

---

## 💾 Datasets

Experiments are conducted on three representative public opinion events collected from Sina Weibo, spanning social controversy, campus incidents, and food safety — covering diverse evolution patterns and participant structures. Simulation precision: **10 min/step**.

| Event | Code | Category | #Users | #Posts | Duration | #Steps |
|-------|:----:|----------|:------:|:------:|:--------:|:------:|
| **Luxury Earring Incident** — An actress's earrings identified as ¥2.3M luxury goods | LE | Social Controversy | 1,530 | 34,218 | ~46h | 276 |
| **WHU Library Incident** — Harassment allegation dispute; court ruling reignited debate | WL | Campus Incident | 1,843 | 51,647 | ~190h | 1,140 |
| **Xibei Prepared Food Controversy** — Internet celebrity publicly accused restaurant chain | XF | Food Safety | 1,987 | 14,892 | ~71h | 426 |

Data preprocessing pipeline: deduplication by post ID → low-activity user filtering (posts < 2) → ad/spam removal → irrelevant content filtering → quality filtering (originals ≥ 20 chars, reposts/comments ≥ 10 chars) → timestamp standardization. Agent belief initialization draws on each user's pre-simulation historical content to sequentially construct role identity, psychological cognition, event opinions, and initial emotions, forming a complete personalized EBDI belief system.

---

## 📊 Results & Analysis

### 🔬 Micro-Level Behavioral Mechanism Validation

100 randomly sampled users, 12 simulation rounds, four methods under identical conditions:

| Method | Cognitive-Behavior Chain (0–5) ↑ | Personality Stability (0–1) ↑ | Decision Robustness (0–1) ↑ |
|--------|:-:|:-:|:-:|
| Direct-Nothink (Qwen2.5-7B, single prompt) | 1.47 ± 0.50 | 0.478 ± 0.263 | 0.629 ± 0.240 |
| Direct-Think (Qwen3-8B, single prompt) | 1.75 ± 0.43 | 0.448 ± 0.269 | 0.603 ± 0.299 |
| CoT (single-call chained reasoning) | 3.09 ± 0.29 | 0.516 ± 0.272 | 0.541 ± 0.356 |
| **EBDI (Ours)** | **4.64 ± 0.48** | **0.661 ± 0.215** | **0.695 ± 0.213** |

An interesting observation: while CoT outperforms Direct methods on chain consistency, its decision robustness is actually the *lowest* among all four (0.541). Without stable state anchoring in a single call, input perturbations ripple through the entire reasoning chain. EBDI's explicit belief states provide a cognitive anchoring effect — even under equivalent perturbations, stable beliefs provide consistent decision anchor points.

### 🌊 Macro-Level Emergent Phenomena

All macro phenomena below emerged **spontaneously from agent interactions** — none were pre-programmed.

#### 🎢 Opinion Lifecycle

<p align="center">
  <img src="assets/fig_lifecycle.png" alt="Opinion Lifecycle" width="85%">
</p>
<p align="center"><b>Figure 2.</b> Simulated opinion lifecycle (Xibei Prepared Food event): posting volume (bar chart, left axis) and cumulative posting percentage S-curve (solid line, right axis) with Logistic fit (dashed). E₁–E₇ mark exogenous event injection points.</p>

The simulation clearly reproduces the multi-stage lifecycle from outbreak → plateau → resurgence → decline. E₁ (food hygiene video exposure) triggers rapid escalation; E₂ (ceasefire announcement) leads to a plateau; E₄ (leaked group chat screenshots) reignites discussion; E₅ (state media commentary) triggers the simulation's peak activity — a textbook "rekindling" and "state media framing triggers secondary outbreak" effect. The cumulative S-curve closely matches diffusion theory predictions.

#### 👥 Multi-Agent Behavioral Heterogeneity

<p align="center">
  <img src="assets/fig_heterogeneity.png" alt="Agent Heterogeneity" width="90%">
</p>
<p align="center"><b>Figure 3.</b> Behavioral heterogeneity across four agent types: (a) Emotional intensity over time; (b) Content length distributions; (c) Multi-dimensional behavioral radar charts.</p>

Citizens and KOLs maintain mean emotional intensity of 0.645 and 0.603 respectively — consistently in the high-arousal zone with sharper post-event fluctuations. Media and Government agents remain stable in the low-arousal zone, producing the "public emotional, officials neutral" stratification pattern observed in real events. The radar charts show distinctly different polygonal shapes: Citizens excel in emotional engagement and interaction frequency, KOLs in influence and output volume, Media in content quality and neutral expression, Government in low-frequency high-authority presence.

#### ⚡ Emotional Polarization

<p align="center">
  <img src="assets/fig_polarization.png" alt="Emotional Polarization" width="55%">
</p>
<p align="center"><b>Figure 4.</b> Emotional polarization index (PI) evolution over simulation rounds. Solid line: PI mean; shaded area: 90% Bootstrap confidence interval.</p>

Validation against emotional arousal theory and emotional contagion hypothesis: high-arousal emotion ratio reaches 73.5%, comment-chain emotion consistency hits 0.772, and the escalation/de-escalation ratio is 4.78 (a significant ratchet effect). The polarization index rises from 0.41 early on to 0.67 late-stage (63% increase, $p < 0.001$), consistent with echo chamber theory predictions.

#### 🕸️ Scale-Free Topology & Cascade Power-Law

<p align="center">
  <img src="assets/fig_powerlaw.png" alt="Power-law Distribution" width="70%">
</p>
<p align="center"><b>Figure 5.</b> Interaction network topological properties: (a) Degree distribution power-law fit; (b) Cascade size CCDF.</p>

Interaction network degree distribution yields power-law exponent $\gamma = 1.87$ ($R^2 = 0.880$), within the 1.5–3 range typically reported for real social networks. Cascade size CCDF fits $\alpha = 3.70$ ($R^2 = 0.880$), reproducing the "most posts go unnoticed, a few go viral" long-tail phenomenon.

### ⚖️ Statistical Calibration Results

<p align="center">
  <img src="assets/fig_calibration.png" alt="Calibration Results" width="80%">
</p>
<p align="center"><b>Figure 6.</b> Behavior and activity calibration across three events. Each row corresponds to one event: left — activity time series comparison; right — behavioral type distribution comparison.</p>

**Behavior Layer**

| Dataset | Rule-based ABM | POSIM w/ LLM | POSIM w/ CoT | **POSIM (Ours)** |
|---------|:-:|:-:|:-:|:-:|
| LE | 0.741 | 0.783 | 0.754 | **0.821** |
| WL | 0.746 | 0.789 | 0.800 | **0.853** |
| XF | 0.721 | 0.746 | 0.742 | **0.804** |

POSIM achieves the best BType JSD across all three datasets. Notably, the WL dataset JSD is only 0.073 — far superior to alternatives — because campus discussion behavioral patterns are relatively concentrated, and EBDI's belief anchoring mechanism captures this concentration precisely.

**Content Layer**

| Dataset | POSIM w/ LLM | POSIM w/ CoT | **POSIM (Ours)** |
|---------|:-:|:-:|:-:|
| LE | 0.680 | 0.774 | **0.910** |
| WL | 0.640 | 0.673 | **0.876** |
| XF | 0.858 | 0.875 | **0.926** |

The gap is most striking on the LE dataset (0.910 vs. 0.680) — the Luxury Earring event features intense public confrontation. Without the desire module's motivational constraints, LLM-generated content gravitates toward neutral expression, failing to reproduce the real discourse confrontation landscape.

**Topology Layer**

| Dataset | Rule-based ABM | POSIM w/ LLM | POSIM w/ CoT | **POSIM (Ours)** |
|---------|:-:|:-:|:-:|:-:|
| LE | 0.552 | 0.739 | 0.763 | **0.896** |
| WL | 0.736 | 0.592 | 0.784 | **0.858** |
| XF | 0.474 | 0.650 | 0.641 | **0.698** |

> Overall: POSIM's behavioral, content, and topological metrics outperform the best baseline by **5.0%**, **13.0%**, and **8.5%** respectively.

### 🧩 Ablation Study

Conducted on the LE dataset to verify the necessity of each module:

| Configuration | BType JSD ↓ | Act. ρ ↑ | Act. RMSE ↓ | Confr. Sim. ↑ | \|ΔTTR\| ↓ | \|ΔS̄\| ↓ | Net. Sim. ↑ | Casc. Sim. ↑ |
|---------------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **Full POSIM** | **0.193** | **0.809** | **0.154** | **0.790** | **0.030** | **0.029** | **0.895** | **0.830** |
| w/o Belief | 0.258 | 0.762 | 0.172 | 0.706 | 0.058 | 0.067 | 0.861 | 0.773 |
| w/o Desire | 0.267 | 0.779 | 0.169 | 0.682 | 0.071 | 0.083 | 0.853 | 0.788 |
| w/o Intention | 0.237 | 0.802 | 0.159 | 0.728 | 0.064 | 0.055 | 0.858 | 0.814 |
| w/o Hawkes | 0.177 | 0.235 | 0.362 | 0.787 | 0.207 | 0.028 | 0.822 | 0.754 |

The functional division is clear-cut:
- **Removing the Hawkes engine** — Act. ρ plummets from 0.809 to 0.235; uniform activation completely destroys temporal dynamics
- **Removing Belief** — Confrontation similarity drops from 0.790 to 0.706; agents lose deep understanding of contentious focal points
- **Removing Desire** — Content layer degrades most severely (Confr. Sim. falls to 0.682, the lowest across all variants); motivation constraints are the core driver of intense discourse
- **Removing Intention** — Lexical diversity deviation worsens most (0.064), network similarity shows the largest drop; the three-level chain-of-thought is critical for diversity and topology

### 🔍 Case Studies: Cognitive Priming & Counterfactual Evaluation

<p align="center">
  <img src="assets/fig_cognitive_priming.png" alt="Cognitive Priming Experiment" width="90%">
</p>
<p align="center"><b>Figure 7.</b> Cognitive priming experiment: (a) Negative emotion ratio over time at 100% coverage across four conditions; (b) Dose-response relationship of NER change magnitude vs. coverage for three strategies.</p>

**Cognitive Priming Experiment** — 200 agents, 30 steps, cognitive priming injected at step 5, testing 30%/60%/100% coverage. Rational Cognition priming (RC) reduces the negative emotion ratio from 0.844 to 0.571 (32.3% reduction), with a clear **threshold effect** as coverage crosses 60%. Empathy Priming (EP) yields a counterintuitive **empathy paradox** — the negative emotion ratio actually *exceeds* the control group (0.878 vs. 0.844) and rises further with increasing coverage. This phenomenon has theoretical backing in social psychology: empathizing with others' suffering generates empathetic negative emotions. This is the first time it has been quantitatively observed in simulation.

**Counterfactual Strategy Evaluation** — Holding external events constant, only replacing Xibei's PR response. Consumer Dialogue (CD) performs best (NER 0.744), Swift Early Apology (SEA) second (0.749), Strategic Silence (SS) worst (0.831). Strategy effects exhibit an **immediate cooling followed by gradual rebound** pattern — SEA temporarily drops NER to 0.56 post-deployment, but it gradually recovers under sustained negative content recommendation, precisely simulating the commonly observed PR effect decay.

---

## 🌳 Project Structure

```
posim/
├── posim/                          # Core framework
│   ├── agents/                     # Agent module
│   │   ├── base_agent.py           # Base class (cognitive pipeline dispatch, non-LLM decisions)
│   │   ├── citizen_agent.py        # Citizen agent
│   │   ├── kol_agent.py            # KOL agent
│   │   ├── media_agent.py          # Media agent
│   │   ├── government_agent.py     # Government agent
│   │   └── ebdi/                   # EBDI cognitive architecture
│   │       ├── belief/             # Belief subsystem (update, decay, bias injection)
│   │       ├── desire/             # Desire subsystem (motivation inference)
│   │       ├── intention/          # Intention subsystem (three-level CoT)
│   │       └── memory/             # Streaming memory store (temporal decay + semantic retrieval)
│   ├── engine/                     # Simulation engine
│   │   ├── simulator.py            # Main loop (async concurrent execution)
│   │   ├── hawkes_process.py       # Hawkes self-exciting point process
│   │   └── time_engine.py          # Time engine (circadian modulation)
│   ├── environment/                # Simulation environment
│   │   ├── recommendation.py       # Content recommendation (dual-channel + semantic dedup)
│   │   ├── social_network.py       # Three-layer social network
│   │   ├── hot_search.py           # Trending topics
│   │   └── event_queue.py          # External event queue
│   ├── evaluation/                 # Evaluation framework
│   │   ├── calibration/            # Statistical calibration (behavior/hotness/emotion/confrontation/network)
│   │   ├── mechanism/              # Macro emergence (lifecycle/heterogeneity/polarization/propagation)
│   │   └── evaluator_manager.py    # Evaluation manager
│   ├── micro_user_vail/            # Micro-level behavioral validation
│   ├── llm/                        # LLM resource management
│   │   ├── api_pool.py             # Multi-endpoint pool (round-robin LB, failover)
│   │   └── llm_client.py           # LLM call client
│   ├── prompts/                    # Prompt templates (organized by agent type)
│   ├── config/                     # Configuration (dataclass schema + loader)
│   ├── storage/                    # Data storage (SQLite + logging)
│   └── web/                        # Real-time monitoring (WebSocket + HTML dashboard)
├── scripts/                        # Simulation & evaluation scripts
│   ├── tianjiaerhuan/              # Luxury Earring event (config + run + evaluate)
│   ├── wudatushuguan/              # WHU Library event
│   └── xibeiyuzhicai/              # Xibei Prepared Food event
├── data/                           # Datasets
├── assets/                         # Static assets (logo, paper figures)
├── papers/                         # Paper source files
└── requirements.txt
```

---

## ⚙️ Installation

### 💻 System Requirements

| Item | Minimum | Recommended |
|------|---------|-------------|
| Python | ≥ 3.8 | 3.10 |
| CUDA | — | ≥ 11.0 (local embedding acceleration) |
| RAM | 16 GB | 32 GB+ (large-scale simulation) |
| GPU | — | Recommended (sentence-transformers acceleration) |

### 📦 Setup

```bash
git clone https://github.com/DeepCog/POSIM.git
cd POSIM

conda create -n posim python=3.10
conda activate posim

pip install -r requirements.txt
```

### 📚 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥ 1.24.0 | Numerical computation, Hawkes intensity sampling |
| `openai` | ≥ 1.0.0 | LLM API calls (any OpenAI-compatible service) |
| `pydantic` | ≥ 2.0.0 | Configuration validation & structured data |
| `sentence-transformers` | ≥ 2.2.0 | Semantic embeddings (recommendation, dedup, memory retrieval) |
| `torch` | ≥ 2.0.0 | Deep learning backend (embedding inference) |
| `matplotlib` | ≥ 3.7.0 | Evaluation visualization |
| `neo4j` | ≥ 5.0.0 | Graph DB for social network (optional) |
| `websockets` | ≥ 12.0 | Real-time simulation monitoring |

---

## 🚀 Quick Start

### 1️⃣ Configure LLM

POSIM supports any OpenAI API-compatible service (SiliconFlow, local vLLM, OpenAI, etc.):

```json
{
  "llm": {
    "max_concurrent_requests": 30,
    "use_local_embedding_model": true,
    "local_embedding_model_path": "path/to/bge-small-zh-v1.5",
    "embedding_dimension": 512,
    "embedding_device": "cuda",
    "llm_api_configs": [
      {
        "name": "primary",
        "enabled": true,
        "base_url": "https://your-api-endpoint/v1/",
        "api_key": "your-api-key",
        "model": "Qwen/Qwen2.5-14B-Instruct",
        "temperature": 0.7,
        "top_p": 0.9,
        "weight": 1.0
      }
    ]
  }
}
```

The unified resource pool manages multiple LLM endpoints with multi-model mixed driving, usage-based routing, endpoint-level concurrency control, round-robin load balancing, and automatic failover. Sampling parameters are randomly perturbed per call to avoid output homogenization.

### 2️⃣ Prepare Data

Each scenario requires four data files:

| File | Content |
|------|---------|
| `users.json` | User profiles (ID, nickname, gender, followers, verification, description, behavioral history) |
| `events.json` | External event sequence (injection time, description, impact intensity) |
| `initial_posts.json` | Initial post data (content, author, timestamp, type, keywords) |
| `relations.json` | User follow relationships |

### 3️⃣ Run Simulation

```bash
python scripts/tianjiaerhuan/run_with_monitor.py
```

Flow: Load user data → Initialize EBDI belief systems → Build social network & recommendation → Start Hawkes time engine → Execute cognitive pipeline per step (recommend→belief→desire→intention, async concurrent) → Emotional contagion → Update trending → Record trajectories. WebSocket real-time monitoring dashboard available.

### 4️⃣ Evaluate

```bash
# Single event
python scripts/tianjiaerhuan/evaluate.py

# Batch evaluation
python scripts/run_all_evaluations.py
```

Results output to `vis_results/`: behavioral calibration, activity calibration, emotion calibration, discourse confrontation calibration, network topology calibration, and `evaluation_report.json`.

<details>
<summary><b>Full Parameter Reference</b></summary>

| Parameter | Description | Default |
|-----------|-------------|:-------:|
| `time_granularity` | Simulation time step (minutes) | 10 |
| `hawkes_mu` | Hawkes background rate | 0.01 |
| `hawkes_internal.alpha` | Endogenous excitation intensity | 0.005 |
| `hawkes_internal.beta` | Endogenous excitation decay | 0.16 |
| `hawkes_external.alpha` | Exogenous excitation intensity | 0.08 |
| `hawkes_external.beta` | Exogenous excitation decay | 0.005 |
| `total_scale` | Activity scaling factor | 2000 |
| `circadian_strength` | Circadian modulation strength | 0.3 |
| `recommend_count` | Recommended items per step | 10 |
| `comment_count` | Comments displayed per post | 5 |
| `homophily_weight` | Recommendation homophily weight | 0.3 |
| `popularity_weight` | Recommendation popularity weight | 0.3 |
| `recency_weight` | Recommendation freshness weight | 0.2 |
| `exploration_rate` | Recommendation exploration rate | 0.2 |
| `relation_weight` | Relationship channel weight | 0.5 |
| `hot_search_update_interval` | Trending update interval (min) | 15 |

</details>

---

## 🔌 Extensibility

Thanks to the highly decoupled modular design, all core components can be independently replaced and extended through standard interfaces.

<details>
<summary><b>Adding New Agent Types</b></summary>

1. Inherit from `BaseAgent` in `posim/agents/` to create a new class
2. Create role-specific prompt templates in `posim/prompts/` (belief/desire/intention)
3. Register the new type in simulation configuration

</details>

<details>
<summary><b>Replacing the Cognitive Architecture</b></summary>

The three EBDI subsystems communicate via structured intermediate states and can be independently replaced:
- Belief: `posim/agents/ebdi/belief/`
- Desire: `posim/agents/ebdi/desire/`
- Intention: `posim/agents/ebdi/intention/`

Just maintain consistent input/output formats.

</details>

<details>
<summary><b>Switching the Time Engine</b></summary>

Implement a new time engine module in `posim/engine/`, following the same intensity calculation and agent sampling interfaces.

</details>

<details>
<summary><b>Adding Evaluation Metrics</b></summary>

Add new evaluator classes in `posim/evaluation/calibration/` or `posim/evaluation/mechanism/`, and register them in `evaluator_manager.py`.

</details>

<details>
<summary><b>Integrating New LLM Services</b></summary>

Add new endpoints to `llm_api_configs` in the config file — the framework uses a unified OpenAI-compatible interface, requiring no code changes. Supports local vLLM deployment, cloud API services, etc.

</details>

---

## 📝 Citation

If this work is helpful to your research, please cite:

```bibtex
@article{posim2025,
  title   = {POSIM: A General-Purpose Social Media Public Opinion Simulation
             Framework Based on Metacognitive Agents},
  author  = {},
  journal = {Information Processing \& Management},
  year    = {2025}
}
```

## 📄 License

This project is released under the [MIT License](LICENSE).

---

<p align="center">
  <i>Questions or suggestions? Please open an <a href="https://github.com/DeepCog/POSIM/issues">Issue</a>.</i>
</p>
