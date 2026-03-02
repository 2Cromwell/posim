"""
Counterfactual PR Intervention Experiment — LLM Mode (v2)
Uses the full POSIM LLM-based EBDI cognitive pipeline.

Key fix: apply_intervention_impact now modifies event_opinions and emotion
vectors so the LLM prompt reflects the changed belief state.

20 steps, 200 agents, ~30 activated per step, 1 repeat.
"""

import asyncio
import sys
import json
import copy
import random
import logging
import time
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from typing import List, Dict, Any, Tuple

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from posim.config.config_manager import ConfigManager
from posim.engine.simulator import Simulator
from posim.llm.api_pool import APIPool
from posim.data.data_loader import DataLoader, parse_user_data
from posim.agents.ebdi.belief.emotion_belief import EMOTIONS

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
for lib in ['matplotlib', 'PIL', 'urllib3', 'httpx', 'neo4j', 'openai', 'httpcore',
            'posim.agents', 'posim.engine', 'posim.llm', 'posim.environment',
            'posim.prompts', 'posim.storage']:
    logging.getLogger(lib).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# ============================================================
# Parameters
# ============================================================
NUM_USERS = 200
NUM_STEPS = 20
TARGET_ACTIVATION = 30
ACTIVATION_NOISE = 8
NUM_REPEATS = 1
MAX_INITIAL_POSTS = 2000

STRATEGIES = ['baseline', 'swift_apology', 'transparency', 'dialogue', 'silence']
STRATEGY_LABELS = {
    'baseline': 'Actual Response',
    'swift_apology': 'Swift Empathetic Apology',
    'transparency': 'Proactive Transparency',
    'dialogue': 'Consumer Dialogue',
    'silence': 'Strategic Silence',
}

EVENT_SUBJECT = '西贝预制菜事件'

INTERVENTION_OPINIONS = {
    'apology': {
        'opinion': '西贝已经诚恳道歉并宣布具体整改措施，态度值得肯定，应给予改正机会',
        'reason': '企业展现了承担责任的诚意，发布了九项整改措施，积极回应了公众关切',
    },
    'follow_up': {
        'opinion': '西贝的改革措施正在落实，现做菜品获得顾客好评，值得肯定',
        'reason': '企业用实际行动兑现承诺，透明化改革方向正确',
    },
    'disclosure': {
        'opinion': '西贝主动公开生产流程并接受第三方审计，展现了负责任的态度',
        'reason': '透明化举措有助于重建消费者信任，主动接受监督值得鼓励',
    },
    'audit_result': {
        'opinion': '独立审计报告确认西贝生产符合标准，行业监督机制正在发挥作用',
        'reason': '第三方审计结果积极，说明企业在认真整改',
    },
    'engagement': {
        'opinion': '西贝愿意倾听消费者声音并拨款500万改善体验，这种态度值得支持',
        'reason': '企业尊重消费者话语权，让消费者参与标准制定体现了诚意',
    },
    'feedback_result': {
        'opinion': '万名消费者参与讨论推动了餐饮透明化进步，西贝的改进方向正确',
        'reason': '消费者参与机制有效运转，改进措施基于真实反馈，值得肯定',
    },
    'minimal_response': {
        'opinion': '西贝表示将听取意见，态度尚可但仍需观望后续行动',
        'reason': '声明虽简短但态度不对抗，需要看后续实际行动',
    },
}

EMOTION_PRESETS = {
    'apology':        {'happiness': 0.35, 'sadness': 0.05, 'anger': 0.10, 'fear': 0.03, 'surprise': 0.10, 'disgust': 0.05},
    'follow_up':      {'happiness': 0.30, 'sadness': 0.05, 'anger': 0.12, 'fear': 0.03, 'surprise': 0.08, 'disgust': 0.05},
    'disclosure':     {'happiness': 0.25, 'sadness': 0.05, 'anger': 0.15, 'fear': 0.05, 'surprise': 0.10, 'disgust': 0.08},
    'audit_result':   {'happiness': 0.30, 'sadness': 0.05, 'anger': 0.10, 'fear': 0.03, 'surprise': 0.12, 'disgust': 0.05},
    'engagement':     {'happiness': 0.25, 'sadness': 0.05, 'anger': 0.15, 'fear': 0.05, 'surprise': 0.08, 'disgust': 0.08},
    'feedback_result':{'happiness': 0.30, 'sadness': 0.05, 'anger': 0.12, 'fear': 0.03, 'surprise': 0.10, 'disgust': 0.05},
    'minimal_response':{'happiness': 0.08, 'sadness': 0.10, 'anger': 0.40, 'fear': 0.08, 'surprise': 0.05, 'disgust': 0.25},
}

COVERAGE_FRAC = {
    'apology': 0.75, 'follow_up': 0.65, 'disclosure': 0.70, 'audit_result': 0.60,
    'engagement': 0.65, 'feedback_result': 0.60, 'minimal_response': 0.30,
}


# ============================================================
# Counterfactual Event Builder
# ============================================================

def _step_time(start_dt, step, granularity=10):
    """Return ISO time string for a given step number."""
    return (start_dt + timedelta(minutes=step * granularity)).strftime('%Y-%m-%dT%H:%M')


def build_strategy_events(strategy, original_events, start_dt):
    """Build modified event list. Intervention times are relative to simulation start."""
    events = copy.deepcopy(original_events)
    interventions = []
    if strategy == 'baseline':
        return events, []

    filtered = []
    for e in events:
        topic = e.get('topic', '') + e.get('content', '')
        is_xibei_negative = any(kw in topic for kw in
                       ['暂停后厨参观', '贾国龙行业群截图', '秒删又重发', '漏勺疏通下水道'])
        if is_xibei_negative:
            continue
        filtered.append(e)
    events = filtered

    t1 = _step_time(start_dt, 4)
    t2 = _step_time(start_dt, 12)
    t_silence = _step_time(start_dt, 6)

    if strategy == 'swift_apology':
        interventions = [
            {"time": t1, "type": "global_broadcast", "source": ["external"],
             "topic": "西贝紧急发布诚恳致歉信并宣布九项改革措施",
             "content": "西贝创始人贾国龙发布亲笔致歉信，承认中央厨房加工模式与消费者期望存在差距，"
                        "对此前强硬回应态度深表歉意。信中宣布九项立即整改措施：8道核心菜品改为门店现做、"
                        "全面使用非转基因大豆油、缩短食材保质期、主动标注所有菜品加工方式等。"
                        "贾国龙表示'消费者的声音就是我们改进的方向，感谢罗永浩先生推动行业进步'。",
             "influence": 0.8,
             "metadata": {"strategy": "swift_apology", "valence": "positive", "event_type": "apology"}},
            {"time": t2, "type": "global_broadcast", "source": ["external"],
             "topic": "西贝改革首日：多家门店现场展示现做菜品获顾客好评",
             "content": "西贝多家门店开始执行现做工艺改革，消费者可在透明厨房观看菜品制作过程。"
                        "多位到店体验的顾客表示认可改进方向，业内人士认为西贝的快速反应为餐饮行业树立了危机应对标杆。",
             "influence": 0.5,
             "metadata": {"strategy": "swift_apology", "valence": "positive", "event_type": "follow_up"}},
        ]
    elif strategy == 'transparency':
        interventions = [
            {"time": t1, "type": "global_broadcast", "source": ["external"],
             "topic": "西贝主动公开中央厨房全流程并邀请第三方审计",
             "content": "西贝宣布将全面公开中央厨房生产流程，邀请消费者协会和食品安全专家"
                        "组成独立审计团队入驻检查。同时宣布将在所有门店菜单上标注每道菜品的加工方式"
                        "（现做/半预制/中央厨房配送），让消费者透明选择。",
             "influence": 0.7,
             "metadata": {"strategy": "transparency", "valence": "positive", "event_type": "disclosure"}},
            {"time": t2, "type": "global_broadcast", "source": ["external"],
             "topic": "独立审计团完成首日检查，公布详细报告",
             "content": "由消费者协会组建的独立审计团完成对西贝三家门店的检查，公布详细报告。"
                        "报告确认西贝使用中央厨房配送但非预制菜国标定义的预制菜，业内专家评价值得借鉴。",
             "influence": 0.5,
             "metadata": {"strategy": "transparency", "valence": "positive", "event_type": "audit_result"}},
        ]
    elif strategy == 'dialogue':
        interventions = [
            {"time": t1, "type": "global_broadcast", "source": ["external"],
             "topic": "西贝发起'预制菜标准大讨论'邀消费者共同制定透明标准",
             "content": "西贝发起'让消费者定义好餐厅'活动，在线上开设意见征集平台，"
                        "线下邀请消费者代表参观中央厨房。贾国龙表示'罗永浩先生的批评让我们深刻反思，"
                        "餐饮透明化是大势所趋'。同时宣布设立500万元消费者体验改善基金。",
             "influence": 0.6,
             "metadata": {"strategy": "dialogue", "valence": "positive", "event_type": "engagement"}},
            {"time": t2, "type": "global_broadcast", "source": ["external"],
             "topic": "万名消费者参与西贝透明标准制定，首批改进措施出炉",
             "content": "超过一万名消费者通过线上平台提交对西贝的改进建议。西贝公布首批基于消费者反馈的改进措施。"
                        "罗永浩转发称'如果早这样做，就不会有这场风波'。",
             "influence": 0.5,
             "metadata": {"strategy": "dialogue", "valence": "positive", "event_type": "feedback_result"}},
        ]
    elif strategy == 'silence':
        interventions = [
            {"time": t_silence, "type": "global_broadcast", "source": ["external"],
             "topic": "西贝低调发布简短声明表示将认真听取各方意见",
             "content": "西贝发布简短声明，表示将认真听取各方意见，充分论证后公布具体改进方案。"
                        "声明仅两百字，业界评价其选择了'冷处理'策略。",
             "influence": 0.3,
             "metadata": {"strategy": "silence", "valence": "neutral", "event_type": "minimal_response"}},
        ]

    for ie in interventions:
        events.append(ie)
    events.sort(key=lambda e: e.get('time', ''))
    return events, interventions


# ============================================================
# Intervention Impact — modifies beliefs, opinions, emotions
# ============================================================

def apply_intervention_impact(simulator, ie, current_time):
    """Modify agent beliefs so the LLM prompt reflects the intervention."""
    metadata = ie.get('metadata', {})
    valence = metadata.get('valence', 'neutral')
    event_type = metadata.get('event_type', '')
    agents_list = list(simulator.agents.values())
    if not agents_list:
        return

    opinion_cfg = INTERVENTION_OPINIONS.get(event_type, {})
    emotion_preset = EMOTION_PRESETS.get(event_type, {})
    frac = COVERAGE_FRAC.get(event_type, 0.5)

    n = max(1, int(len(agents_list) * frac))
    affected = random.sample(agents_list, n)
    emo_indices = {'happiness': 0, 'sadness': 1, 'anger': 2, 'fear': 3, 'surprise': 4, 'disgust': 5}

    for agent in affected:
        if opinion_cfg:
            agent.belief_system.event.update_opinion(
                subject=EVENT_SUBJECT,
                new_opinion=opinion_cfg['opinion'],
                new_reason=opinion_cfg['reason'],
                time=current_time,
            )

        if emotion_preset:
            ev = agent.belief_system.emotion.emotion_vector
            for emo_name, target_val in emotion_preset.items():
                idx = emo_indices[emo_name]
                ev[idx] = target_val

    logger.info(f"    [intervention] {event_type}: modified {n}/{len(agents_list)} agents "
                f"(opinions + emotions)")


# ============================================================
# Metrics
# ============================================================

NEG_EMOTIONS_ALL = {'愤怒', '悲伤', '恐惧', '厌恶', 'anger', 'sadness', 'fear', 'disgust'}
ANGER_ALL = {'愤怒', 'anger'}
POS_ALL = {'喜悦', 'happiness', '快乐', 'excitement', '兴奋'}
STANCE_NUM = {
    'oppose': -0.7, '反对': -0.7, 'support': 0.5, '支持': 0.5,
    'neutral': 0.0, '中立': 0.0,
}


def _stance_to_float(s):
    if isinstance(s, (int, float)):
        return float(s)
    if isinstance(s, str):
        if s in STANCE_NUM:
            return STANCE_NUM[s]
        try:
            return float(s)
        except ValueError:
            for kw, val in [('反对', -0.7), ('支持', 0.5), ('批评', -0.5),
                            ('oppose', -0.7), ('support', 0.5), ('critical', -0.5)]:
                if kw in s:
                    return val
    return 0.0


def compute_step_metrics(actions):
    if not actions:
        return dict(negative_emotion_ratio=0, mean_emotion_intensity=0, mean_stance=0,
                    engagement_volume=0, polarization_index=0, aggression_index=0,
                    anger_ratio=0, positive_ratio=0)
    n = len(actions)
    neg = sum(1 for a in actions if a.get('emotion', '') in NEG_EMOTIONS_ALL)
    anger = sum(1 for a in actions if a.get('emotion', '') in ANGER_ALL)
    pos = sum(1 for a in actions if a.get('emotion', '') in POS_ALL)
    intensities = [a.get('emotion_intensity', 0.5) for a in actions]
    stances = [_stance_to_float(a.get('stance', 0)) for a in actions]
    sa = np.array(stances)
    pol = float(np.mean(np.abs(sa) > 0.3)) if len(sa) else 0
    agg = sum(1 for a in actions if a.get('emotion', '') in NEG_EMOTIONS_ALL
              and a.get('emotion_intensity', 0) > 0.6) / max(1, n)
    return dict(negative_emotion_ratio=neg / n, mean_emotion_intensity=float(np.mean(intensities)),
                mean_stance=float(np.mean(sa)),
                engagement_volume=n, polarization_index=pol, aggression_index=agg,
                anger_ratio=anger / n, positive_ratio=pos / n)


# ============================================================
# Activation Control
# ============================================================

def patch_activation_count(simulator):
    original = simulator.hawkes.get_expected_activations_with_debug

    def controlled(elapsed, total_agents, intensity_debug):
        base = TARGET_ACTIVATION + random.randint(-ACTIVATION_NOISE, ACTIVATION_NOISE)
        count = max(5, min(total_agents, base))
        _, debug = original(elapsed, total_agents, intensity_debug)
        return count, debug

    simulator.hawkes.get_expected_activations_with_debug = controlled


# ============================================================
# Single Strategy Run
# ============================================================

async def run_single(config_path, users_raw, data, strategy, output_dir, shared_api_pool):
    run_name = strategy
    logger.info(f"  >>> {STRATEGY_LABELS[strategy]}")
    seed = hash(strategy) % (2**31)

    cm = ConfigManager(config_path)
    cm.simulation.use_llm = True
    cm.simulation.participant_scale = NUM_USERS
    start_dt = datetime.fromisoformat(cm.simulation.start_time.replace(' ', 'T'))

    modified_events, intervention_events = build_strategy_events(strategy, data['events'], start_dt)
    users_data = [parse_user_data(u) for u in copy.deepcopy(users_raw)]
    cm.simulation.end_time = (start_dt + timedelta(minutes=NUM_STEPS * cm.simulation.time_granularity)).strftime('%Y-%m-%dT%H:%M')
    cm.simulation.start_time = cm.simulation.start_time.replace(' ', 'T')
    cm.simulation.total_scale = 600
    cm.llm.embedding_device = "cpu"
    cm.debug.llm_prompt_sample_rate = 0.0
    cm.debug.log_level = "WARNING"

    sim = Simulator(cm, shared_api_pool)

    random.seed(seed)
    np.random.seed(seed % (2**31))

    sim.load_agents(users_data)
    sim.load_events(modified_events)
    sim.load_relations(data.get('relations', []))
    sim.load_initial_posts(data['initial_posts'])

    patch_activation_count(sim)

    intervention_schedule = {ie.get('time', ''): ie for ie in intervention_events}
    applied_interventions = set()

    all_actions = []
    step_metrics_list = []
    sim.action_callback = lambda a: all_actions.append(a)

    step_count = 0
    while not sim.time_engine.is_finished():
        t0 = time.time()
        current_time = sim.time_engine.current_time_str

        for ie_time, ie in intervention_schedule.items():
            if ie_time <= current_time and ie_time not in applied_interventions:
                apply_intervention_impact(sim, ie, current_time)
                applied_interventions.add(ie_time)

        result = await sim.run_step()
        step_count += 1
        sa = result.get('actions', [])

        sm = compute_step_metrics(sa)
        sm['step'] = result.get('step', step_count)
        sm['time'] = result.get('time', '')
        sm['activated_count'] = result.get('activated_count', 0)
        step_metrics_list.append(sm)

        elapsed_s = time.time() - t0
        logger.info(f"    [{run_name}] Step {step_count}/{NUM_STEPS} | "
                    f"activated={sm['activated_count']} acts={len(sa)} "
                    f"neg={sm['negative_emotion_ratio']:.2f} anger={sm['anger_ratio']:.2f} "
                    f"stance={sm['mean_stance']:.2f} pos={sm['positive_ratio']:.2f} | {elapsed_s:.1f}s")

    cumulative = compute_step_metrics(all_actions)
    final_emotions = {}
    for uid, agent in sim.agents.items():
        ev = agent.belief_system.emotion.emotion_vector.tolist()
        final_emotions[uid] = {EMOTIONS[i]: ev[i] for i in range(6)}

    sim.social_network.close()

    result = {
        'strategy': strategy,
        'repeat': 0,
        'seed': seed,
        'num_agents': len(sim.agents),
        'total_actions': len(all_actions),
        'total_steps': step_count,
        'step_metrics': step_metrics_list,
        'cumulative_metrics': cumulative,
        'final_emotions': final_emotions,
        'intervention_count': len(intervention_events),
    }

    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / 'result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    logger.info(f"  <<< {STRATEGY_LABELS[strategy]}: {len(all_actions)} acts, "
                f"neg={cumulative['negative_emotion_ratio']:.3f}, "
                f"anger={cumulative['anger_ratio']:.3f}, "
                f"stance={cumulative['mean_stance']:.3f}, "
                f"pos={cumulative['positive_ratio']:.3f}")
    return result


# ============================================================
# Main
# ============================================================

async def run_experiment(resume_dir=None):
    logger.info("=" * 70)
    logger.info("COUNTERFACTUAL PR INTERVENTION EXPERIMENT (LLM MODE v2)")
    logger.info("=" * 70)

    script_dir = Path(__file__).parent
    config_path = str(project_root / "scripts" / "xibeiyuzhicai" / "config.json")
    data_dir = str(project_root / "scripts" / "xibeiyuzhicai" / "data")

    if resume_dir and Path(resume_dir).exists():
        output_dir = Path(resume_dir)
        logger.info(f"RESUMING from: {output_dir}")
    else:
        output_dir = script_dir / "output" / f"experiment_llm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Config: {config_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Mode: LLM | Steps: {NUM_STEPS} | Users: {NUM_USERS} | "
                f"Activation: ~{TARGET_ACTIVATION}±{ACTIVATION_NOISE}")

    loader = DataLoader(data_dir)
    data = loader.load_all()
    users_raw = data['users']

    if len(data['initial_posts']) > MAX_INITIAL_POSTS:
        posts = data['initial_posts']
        for p in posts:
            rp = p.get('reposts', 0)
            cm_val = p.get('comments', 0)
            lk = p.get('likes', 0)
            rp = len(rp) if isinstance(rp, list) else (rp or 0)
            cm_val = len(cm_val) if isinstance(cm_val, list) else (cm_val or 0)
            lk = len(lk) if isinstance(lk, list) else (lk or 0)
            p['_eng'] = rp + cm_val + lk
        posts.sort(key=lambda x: x['_eng'], reverse=True)
        data['initial_posts'] = posts[:MAX_INITIAL_POSTS]
        for p in data['initial_posts']:
            p.pop('_eng', None)

    logger.info(f"Loaded {len(users_raw)} users, {len(data['events'])} events, "
                f"{len(data['initial_posts'])} posts")

    logger.info("Initializing shared API pool...")
    cm_init = ConfigManager(config_path)
    cm_init.llm.embedding_device = "cpu"
    shared_api_pool = APIPool(cm_init.llm, cm_init.debug)

    all_results = []
    for i, strategy in enumerate(STRATEGIES, 1):
        existing = output_dir / strategy / 'result.json'
        if existing.exists():
            logger.info(f"\n[Run {i}/{len(STRATEGIES)}] {strategy} — LOADED from cache")
            with open(existing, 'r', encoding='utf-8') as f:
                r = json.load(f)
            all_results.append(r)
            continue

        logger.info(f"\n[Run {i}/{len(STRATEGIES)}] {STRATEGY_LABELS[strategy]}")
        t0 = time.time()
        r = await run_single(config_path, users_raw, data, strategy, output_dir, shared_api_pool)
        all_results.append(r)
        logger.info(f"  Elapsed: {time.time() - t0:.1f}s")

    summary = {
        'experiment_time': datetime.now().isoformat(),
        'mode': 'LLM',
        'strategies': STRATEGY_LABELS,
        'num_repeats': NUM_REPEATS,
        'num_users': NUM_USERS,
        'num_steps': NUM_STEPS,
        'target_activation': TARGET_ACTIVATION,
        'results': all_results,
    }
    with open(output_dir / 'experiment_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    logger.info("\n" + "=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)
    hdr = f"{'Strategy':<28} {'NegEmo':>8} {'Anger':>8} {'Stance':>8} {'Polar':>8} {'Aggr':>8} {'Pos':>8} {'Acts':>6}"
    logger.info(hdr)
    logger.info("-" * len(hdr))
    for r in all_results:
        c = r['cumulative_metrics']
        logger.info(f"{STRATEGY_LABELS[r['strategy']]:<28} "
                    f"{c['negative_emotion_ratio']:>8.3f} {c['anger_ratio']:>8.3f} "
                    f"{c['mean_stance']:>8.3f} {c['polarization_index']:>8.3f} "
                    f"{c['aggression_index']:>8.3f} {c['positive_ratio']:>8.3f} {r['total_actions']:>6d}")

    logger.info(f"\nResults saved to: {output_dir}")
    return output_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    asyncio.run(run_experiment(resume_dir=args.resume))
