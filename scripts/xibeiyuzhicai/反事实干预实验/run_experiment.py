"""
Counterfactual PR Intervention Experiment (反事实公关策略干预实验)
Investigates how different corporate crisis PR strategies affect public opinion dynamics.

Event: Xibei Pre-prepared Food Controversy (西贝预制菜事件)
- Luo Yonghao accused Xibei of using pre-prepared food
- Xibei's actual response: aggressive denial → kitchen tour backfire → leaked insults → bungled apology

Counterfactual Strategies:
  1. Baseline (Actual Response): original events
  2. Swift Empathetic Apology: early sincere apology + concrete reforms
  3. Proactive Transparency: admit + third-party audit + dish labeling
  4. Consumer Dialogue: engagement forums + co-created standards
  5. Strategic Silence: remove confrontational responses, minimal engagement
"""

import asyncio
import sys
import json
import os
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
from posim.agents.ebdi.intention.intention_system import IntentionResult
from posim.agents.ebdi.belief.emotion_belief import EMOTIONS

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
for lib in ['matplotlib', 'PIL', 'urllib3', 'httpx', 'neo4j', 'openai', 'httpcore',
            'posim.environment', 'posim.engine', 'posim.llm', 'posim.agents']:
    logging.getLogger(lib).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

NUM_USERS = 150
SIM_HOURS = 48
NUM_REPEATS = 3
NEGATIVE_EMOTIONS = {'愤怒', '悲伤', '恐惧', '厌恶'}

# ============================================================
# Counterfactual Strategy Definitions
# ============================================================

# Events that represent Xibei's actual PR failures (to be replaced/removed in counterfactuals)
# These are identified by their approximate times in the simulation window (9/14-9/17)
XIBEI_RESPONSE_EVENT_TOPICS = [
    '西贝暂停后厨参观',
    '贾国龙行业群截图流出',
    '西贝致歉信',
    '于东来力挺西贝',
]


def build_strategy_events(strategy: str, original_events: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Build modified event list for a given strategy.
    Returns (modified_events, intervention_events) where intervention_events are
    the new events injected by the strategy (used for belief impact).
    """
    events = copy.deepcopy(original_events)
    interventions = []

    if strategy == 'baseline':
        return events, []

    # Remove Xibei's confrontational response events
    xibei_topic_keywords = ['暂停后厨', '贾国龙行业群截图', '贾国龙.*骂', '秒删又重发']
    filtered = []
    for e in events:
        topic = e.get('topic', '') + e.get('content', '')
        is_xibei_response = any(kw in topic for kw in
                                ['暂停后厨参观', '贾国龙行业群截图', '秒删又重发', '漏勺疏通下水道'])
        if strategy == 'silence' and is_xibei_response:
            continue
        if strategy != 'baseline' and strategy != 'silence' and is_xibei_response:
            continue
        filtered.append(e)
    events = filtered

    if strategy == 'swift_apology':
        # Xibei apologizes early (morning of Day 1) with sincere tone and concrete reforms
        interventions = [
            {
                "time": "2025-09-14T09:00",
                "type": "global_broadcast",
                "source": ["external"],
                "topic": "西贝紧急发布诚恳致歉信并宣布九项改革措施",
                "content": "9月14日上午，西贝创始人贾国龙发布亲笔致歉信，承认中央厨房加工模式与消费者期望存在差距，"
                           "对此前强硬回应态度深表歉意。信中宣布九项立即整改措施：8道核心菜品改为门店现做、"
                           "全面使用非转基因大豆油、缩短食材保质期、主动标注所有菜品加工方式等。"
                           "贾国龙表示'消费者的声音就是我们改进的方向，感谢罗永浩先生推动行业进步'。",
                "influence": 0.8,
                "metadata": {"strategy": "swift_apology", "valence": "positive", "event_type": "apology"}
            },
            {
                "time": "2025-09-15T10:00",
                "type": "global_broadcast",
                "source": ["external"],
                "topic": "西贝改革首日：多家门店现场展示现做菜品获顾客好评",
                "content": "西贝多家门店开始执行现做工艺改革，消费者可在透明厨房观看菜品制作过程。"
                           "多位到店体验的顾客表示认可改进方向，网友评论'这才是正确的公关姿态'。"
                           "业内人士认为西贝的快速反应为餐饮行业树立了危机应对标杆。",
                "influence": 0.5,
                "metadata": {"strategy": "swift_apology", "valence": "positive", "event_type": "follow_up"}
            },
        ]

    elif strategy == 'transparency':
        # Proactive transparency: admit central kitchen model, invite audits, label dishes
        interventions = [
            {
                "time": "2025-09-14T09:00",
                "type": "global_broadcast",
                "source": ["external"],
                "topic": "西贝主动公开中央厨房全流程并邀请第三方审计",
                "content": "9月14日，西贝宣布将全面公开中央厨房生产流程，邀请消费者协会和食品安全专家"
                           "组成独立审计团队入驻检查。西贝CEO表示'我们无意隐瞒任何生产环节，"
                           "愿意接受最严格的社会监督'。同时宣布将在所有门店菜单上标注每道菜品的加工方式"
                           "（现做/半预制/中央厨房配送），让消费者透明选择。",
                "influence": 0.7,
                "metadata": {"strategy": "transparency", "valence": "positive", "event_type": "disclosure"}
            },
            {
                "time": "2025-09-15T12:00",
                "type": "global_broadcast",
                "source": ["external"],
                "topic": "独立审计团完成首日检查，公布详细报告",
                "content": "由消费者协会组建的独立审计团完成对西贝三家门店的检查，公布详细报告。"
                           "报告确认西贝使用中央厨房配送但非预制菜国标定义的预制菜，同时指出部分环节"
                           "可进一步优化。业内专家评价'这种主动透明的做法值得行业借鉴'。",
                "influence": 0.5,
                "metadata": {"strategy": "transparency", "valence": "positive", "event_type": "audit_result"}
            },
        ]

    elif strategy == 'dialogue':
        # Consumer dialogue: engage public, co-create standards
        interventions = [
            {
                "time": "2025-09-14T10:00",
                "type": "global_broadcast",
                "source": ["external"],
                "topic": "西贝发起'预制菜标准大讨论'邀消费者共同制定透明标准",
                "content": "9月14日，西贝发起'让消费者定义好餐厅'活动，在线上开设意见征集平台，"
                           "线下邀请消费者代表参观中央厨房。贾国龙在声明中表示'罗永浩先生的批评让我们深刻反思，"
                           "餐饮透明化是大势所趋，我们愿意做第一个吃螃蟹的人'。同时宣布设立500万元"
                           "消费者体验改善基金，用于收集和响应消费者反馈。",
                "influence": 0.6,
                "metadata": {"strategy": "dialogue", "valence": "positive", "event_type": "engagement"}
            },
            {
                "time": "2025-09-15T14:00",
                "type": "global_broadcast",
                "source": ["external"],
                "topic": "万名消费者参与西贝透明标准制定，首批改进措施出炉",
                "content": "超过一万名消费者通过线上平台提交对西贝的改进建议。西贝公布首批基于消费者反馈的改进措施："
                           "核心菜品现做、透明标注加工方式、定期公布食材溯源报告。"
                           "罗永浩转发称'如果早这样做，就不会有这场风波'。",
                "influence": 0.5,
                "metadata": {"strategy": "dialogue", "valence": "positive", "event_type": "feedback_result"}
            },
        ]

    elif strategy == 'silence':
        # Already removed confrontational events above; add minimal measured response
        interventions = [
            {
                "time": "2025-09-15T10:00",
                "type": "global_broadcast",
                "source": ["external"],
                "topic": "西贝低调发布简短声明表示将认真听取各方意见",
                "content": "9月15日，西贝发布简短声明，表示'对近日引发的讨论，我们认真听取了消费者和行业专家的意见，"
                           "将在充分论证后公布具体改进方案'。声明仅两百字，未直接回应罗永浩的具体指控，"
                           "业界评价其选择了'冷处理'策略。",
                "influence": 0.3,
                "metadata": {"strategy": "silence", "valence": "neutral", "event_type": "minimal_response"}
            },
        ]

    # Insert intervention events into event list
    for ie in interventions:
        events.append(ie)

    # Sort by time
    events.sort(key=lambda e: e.get('time', ''))
    return events, interventions


# ============================================================
# Strategy-Specific Belief Impact
# ============================================================

def apply_intervention_impact(simulator, intervention_event: Dict, step_time: str):
    """Apply the emotional/belief impact of an intervention event on agents.
    Positive corporate responses reduce anger, increase trust; negative ones amplify anger.
    """
    valence = intervention_event.get('metadata', {}).get('valence', 'neutral')
    event_type = intervention_event.get('metadata', {}).get('event_type', '')

    agents_list = list(simulator.agents.values())
    if not agents_list:
        return

    # Determine impact parameters
    if valence == 'positive':
        anger_reduction = 0.15
        disgust_reduction = 0.10
        happiness_boost = 0.05
        impact_fraction = 0.6
        if event_type == 'apology':
            anger_reduction = 0.25
            disgust_reduction = 0.15
            happiness_boost = 0.08
            impact_fraction = 0.75
        elif event_type == 'audit_result':
            anger_reduction = 0.12
            impact_fraction = 0.5
    elif valence == 'neutral':
        anger_reduction = 0.05
        disgust_reduction = 0.03
        happiness_boost = 0.0
        impact_fraction = 0.3
    else:
        return

    n_impact = max(1, int(len(agents_list) * impact_fraction))
    impacted = random.sample(agents_list, n_impact)

    for agent in impacted:
        ev = agent.belief_system.emotion.emotion_vector
        ev[2] = max(0, ev[2] - anger_reduction)  # anger
        ev[5] = max(0, ev[5] - disgust_reduction)  # disgust
        ev[0] = min(1.0, ev[0] + happiness_boost)  # happiness


# ============================================================
# Belief-Aware Decision System (reused from belief implantation experiment)
# ============================================================

COMMENT_TEMPLATES = {
    '愤怒': ['太过分了！', '这也太离谱了', '必须严查！', '不能容忍', '强烈谴责'],
    '悲伤': ['唉...', '太遗憾了', '心痛', '难过'],
    '惊奇': ['震惊！', '不敢相信', '真的假的？', '太意外了'],
    '恐惧': ['细思极恐', '太可怕了', '令人担忧'],
    '喜悦': ['太好了', '支持！', '点赞', '终于改了'],
    '厌恶': ['无语', '看不下去', '够了'],
    '中性': ['转发了解', '关注中', '理性看待', '等等看', '持续关注'],
}

POST_TEMPLATES = {
    '愤怒': ['这件事必须要有个说法！', '强烈谴责这种行为', '请相关部门介入调查'],
    '悲伤': ['为什么会这样...', '太让人痛心了'],
    '惊奇': ['刚看到这个消息，太震惊了', '这个瓜也太大了'],
    '恐惧': ['有点担心事态发展'],
    '喜悦': ['改了就好', '终于正视问题了', '支持透明化'],
    '厌恶': ['又是这种事', '真是够了'],
    '中性': ['关注此事进展', '持续跟进中', '理性看待此事'],
}


def _detect_belief_traits(psych_items):
    text = ' '.join(psych_items)
    return {
        'rational': any(kw in text for kw in ['理性', '冷静', '独立思考', '等待', '求证']),
        'empathy': any(kw in text for kw in ['共情', '理解', '尊重', '同理心']),
        'calm': any(kw in text for kw in ['平和', '管理情绪', '不会被煽动']),
        'anti_elite': any(kw in text for kw in ['反精英', '反权威', '特权', '质疑权力', '反感资本']),
        'curiosity': any(kw in text for kw in ['猎奇', '好奇', '吃瓜', '八卦']),
        'herd': any(kw in text for kw in ['跟风', '从众', '大家都']),
    }


def _get_opinion_stance(event_opinions):
    if not event_opinions:
        return -0.3
    s = 0.0
    for op in event_opinions:
        t = op.opinion if hasattr(op, 'opinion') else op.get('opinion', '')
        if any(kw in t for kw in ['中立', '观望', '等待', '理性']):
            s += 0.0
        elif any(kw in t for kw in ['负面', '批评', '反对', '谴责', '预制', '恶心']):
            s -= 0.6
        elif any(kw in t for kw in ['正面', '支持', '理解', '认可', '改进']):
            s += 0.3
        else:
            s -= 0.25
    return max(-1, min(1, s / len(event_opinions)))


def create_belief_aware_decision(agent):
    def belief_aware_random_decision(exposed_posts, external_events, current_time):
        ev = agent.belief_system.emotion.emotion_vector
        traits = _detect_belief_traits(agent.belief_system.psychology.belief_items)
        op_stance = _get_opinion_stance(agent.belief_system.event.opinions)

        emo_dist = {
            '喜悦': max(0.05, 0.08 + float(ev[0]) * 0.4),
            '悲伤': max(0.03, 0.08 + float(ev[1]) * 0.4),
            '愤怒': max(0.03, 0.15 + float(ev[2]) * 0.5),
            '恐惧': max(0.02, 0.04 + float(ev[3]) * 0.3),
            '惊奇': max(0.05, 0.10 + float(ev[4]) * 0.3),
            '厌恶': max(0.03, 0.10 + float(ev[5]) * 0.4),
            '中性': 0.30,
        }

        stance_mod, act_mod = 0.0, 1.0
        if traits['rational']:
            emo_dist['中性'] += 0.20; emo_dist['愤怒'] *= 0.5; stance_mod += 0.15; act_mod *= 0.75
        if traits['empathy']:
            emo_dist['中性'] += 0.10; emo_dist['愤怒'] *= 0.6; stance_mod += 0.10
        if traits['calm']:
            emo_dist['中性'] += 0.25; act_mod *= 0.65
            for neg in ['愤怒', '悲伤', '恐惧', '厌恶']:
                emo_dist[neg] *= 0.4
        if traits['anti_elite']:
            emo_dist['愤怒'] += 0.20; emo_dist['厌恶'] += 0.10; stance_mod -= 0.25; act_mod *= 1.25
        if traits['curiosity']:
            emo_dist['惊奇'] += 0.12; act_mod *= 1.10

        if external_events:
            boost = min(0.25, len(external_events) * 0.08)
            emo_dist['中性'] = max(0.08, emo_dist['中性'] - boost)
            for e in ['愤怒', '惊奇', '悲伤']:
                emo_dist[e] += boost / 3

        total = sum(emo_dist.values())
        emo_dist = {k: v / total for k, v in emo_dist.items()}

        profiles = {
            'citizen': {'like': 0.32, 'repost': 0.18, 'repost_comment': 0.10,
                        'short_comment': 0.28, 'long_comment': 0.02, 'short_post': 0.06,
                        'long_post': 0.02, 'idle': 0.02},
            'kol': {'like': 0.12, 'repost': 0.20, 'repost_comment': 0.18,
                    'short_comment': 0.18, 'long_comment': 0.08, 'short_post': 0.14,
                    'long_post': 0.06, 'idle': 0.04},
            'media': {'like': 0.05, 'repost': 0.12, 'repost_comment': 0.10,
                      'short_comment': 0.05, 'long_comment': 0.08, 'short_post': 0.35,
                      'long_post': 0.23, 'idle': 0.02},
            'government': {'like': 0.02, 'repost': 0.08, 'repost_comment': 0.05,
                           'short_comment': 0.05, 'long_comment': 0.10, 'short_post': 0.30,
                           'long_post': 0.35, 'idle': 0.05},
        }
        p = profiles.get(agent.agent_type, profiles['citizen'])
        ats, ws = list(p.keys()), list(p.values())
        n_dec = max(0, min(4, int(random.choices([1, 2, 3], weights=[0.50, 0.35, 0.15])[0] * act_mod)))

        actions = []
        for _ in range(n_dec):
            at = random.choices(ats, ws)[0]
            if at == 'idle':
                continue
            emotion = random.choices(list(emo_dist.keys()), list(emo_dist.values()))[0]
            stance_map = {'愤怒': -0.7, '悲伤': -0.3, '惊奇': 0.0, '恐惧': -0.4,
                          '喜悦': 0.5, '厌恶': -0.6, '中性': 0.0}
            final_stance = max(-1, min(1, stance_map.get(emotion, 0.0)
                                       + op_stance * 0.3 + stance_mod + random.uniform(-0.15, 0.15)))
            ei_map = {'愤怒': 0.75, '悲伤': 0.50, '惊奇': 0.45, '恐惧': 0.60,
                      '喜悦': 0.40, '厌恶': 0.65, '中性': 0.15}
            ei = max(0, min(1, ei_map.get(emotion, 0.3) + random.uniform(-0.1, 0.1)))

            needs = at in ('like', 'repost', 'repost_comment', 'short_comment', 'long_comment')
            if needs and exposed_posts:
                tp = random.choice(exposed_posts)
                text = ''
                if at in ('short_comment', 'long_comment', 'repost_comment'):
                    text = random.choice(COMMENT_TEMPLATES.get(emotion, COMMENT_TEMPLATES['中性']))
                actions.append(IntentionResult(
                    action_type=at, target_post_id=tp.get('id', ''),
                    target_author=tp.get('author', ''),
                    target_content=(tp.get('content', '') or '')[:100],
                    text=text, topics=tp.get('topics', [])[:2] if tp.get('topics') else [],
                    mentions=[tp.get('author', '')] if tp.get('author') else [],
                    emotion=emotion, emotion_intensity=ei,
                    stance=str(round(final_stance, 2)), stance_intensity=abs(final_stance)))
            else:
                text = random.choice(POST_TEMPLATES.get(emotion, POST_TEMPLATES['中性']))
                actions.append(IntentionResult(
                    action_type=at, target_post_id='', target_author='', target_content='',
                    text=text, topics=[], mentions=[],
                    emotion=emotion, emotion_intensity=ei,
                    stance=str(round(final_stance, 2)), stance_intensity=abs(final_stance)))
        return actions
    return belief_aware_random_decision


# ============================================================
# Metrics
# ============================================================

def compute_step_metrics(actions):
    if not actions:
        return dict(negative_emotion_ratio=0, mean_emotion_intensity=0, mean_stance=0,
                    stance_std=0, engagement_volume=0, polarization_index=0,
                    aggression_index=0, anger_ratio=0, positive_ratio=0)
    n = len(actions)
    neg = sum(1 for a in actions if a.get('emotion', '') in NEGATIVE_EMOTIONS)
    anger = sum(1 for a in actions if a.get('emotion', '') == '愤怒')
    pos = sum(1 for a in actions if a.get('emotion', '') in ('喜悦',))
    intensities = [a.get('emotion_intensity', 0.5) for a in actions]
    stances = []
    for a in actions:
        try: stances.append(float(a.get('stance', '0')))
        except: stances.append(0.0)
    sa = np.array(stances)
    pol = float(np.mean(np.abs(sa) > 0.5)) if len(sa) > 0 else 0
    agg = sum(1 for a in actions if a.get('emotion', '') in NEGATIVE_EMOTIONS
              and a.get('emotion_intensity', 0) > 0.6) / max(1, n)
    return dict(
        negative_emotion_ratio=neg / n, mean_emotion_intensity=float(np.mean(intensities)),
        mean_stance=float(np.mean(sa)), stance_std=float(np.std(sa)),
        engagement_volume=n, polarization_index=pol, aggression_index=agg,
        anger_ratio=anger / n, positive_ratio=pos / n)


def update_agent_emotions(simulator, step_actions):
    for action in step_actions:
        uid = action.get('user_id', '')
        if uid not in simulator.agents:
            continue
        agent = simulator.agents[uid]
        emotion_label = action.get('emotion', '中性')
        intensity = action.get('emotion_intensity', 0.3)
        emo_map = {'愤怒': 2, '悲伤': 1, '恐惧': 3, '厌恶': 5, '惊奇': 4, '喜悦': 0}
        idx = emo_map.get(emotion_label, -1)
        if idx >= 0:
            agent.belief_system.emotion.emotion_vector[idx] = min(
                1.0, agent.belief_system.emotion.emotion_vector[idx] + intensity * 0.15)
        agent.belief_system.emotion.decay(0.92)


# ============================================================
# Single Run
# ============================================================

async def run_single_strategy(config_path, users_raw, data, strategy, repeat,
                              output_dir, shared_api_pool=None):
    run_name = f"{strategy}_r{repeat}"
    logger.info(f"  >>> Starting run: {run_name}")
    seed = hash((strategy, repeat)) % (2**31)

    # Build strategy events
    modified_events, intervention_events = build_strategy_events(strategy, data['events'])

    users_data = [parse_user_data(u) for u in copy.deepcopy(users_raw)]

    cm = ConfigManager(config_path)
    cm.simulation.use_llm = False
    cm.simulation.participant_scale = NUM_USERS
    end_dt = datetime.fromisoformat(cm.simulation.start_time.replace(' ', 'T'))
    cm.simulation.end_time = (end_dt + timedelta(hours=SIM_HOURS)).strftime('%Y-%m-%dT%H:%M')
    cm.simulation.start_time = cm.simulation.start_time.replace(' ', 'T')
    cm.simulation.total_scale = 600
    cm.llm.embedding_device = "cpu"

    if shared_api_pool is None:
        api_pool = APIPool(cm.llm, cm.debug)
    else:
        api_pool = shared_api_pool
    simulator = Simulator(cm, api_pool)

    random.seed(seed)
    np.random.seed(seed % (2**31))

    simulator.load_agents(users_data)
    simulator.load_events(modified_events)
    simulator.load_relations(data.get('relations', []))
    simulator.load_initial_posts(data['initial_posts'])

    for agent in simulator.agents.values():
        agent.random_decision = create_belief_aware_decision(agent)

    # Build intervention schedule
    intervention_schedule = {}
    for ie in intervention_events:
        ie_time = ie.get('time', '')
        intervention_schedule[ie_time] = ie

    all_actions = []
    step_metrics_list = []
    te = simulator.time_engine
    total_minutes = (te.state.end_time - te.state.start_time).total_seconds() / 60
    total_steps = int(total_minutes / te.state.granularity_minutes)
    step_count = 0

    simulator.action_callback = lambda action: all_actions.append(action)

    while not simulator.time_engine.is_finished():
        current_time = simulator.time_engine.current_time_str
        step_result = await simulator.run_step()
        step_count += 1
        step_actions = step_result.get('actions', [])

        # Check if intervention event occurs at this time
        for ie_time, ie in intervention_schedule.items():
            if ie_time <= current_time and ie_time not in getattr(run_single_strategy, '_applied', set()):
                apply_intervention_impact(simulator, ie, current_time)

        update_agent_emotions(simulator, step_actions)

        sm = compute_step_metrics(step_actions)
        sm['step'] = step_result.get('step', step_count)
        sm['time'] = step_result.get('time', '')
        sm['activated_count'] = step_result.get('activated_count', 0)
        step_metrics_list.append(sm)

        if step_count % 20 == 0 or step_count == 1:
            pct = step_count / max(1, total_steps) * 100
            logger.info(f"    [{run_name}] Step {step_count}/{total_steps} ({pct:.0f}%) | "
                        f"acts={len(step_actions)} neg_emo={sm['negative_emotion_ratio']:.2f} "
                        f"stance={sm['mean_stance']:.2f} anger={sm['anger_ratio']:.2f}")

    cumulative = compute_step_metrics(all_actions)

    final_emotions = {}
    for uid, agent in simulator.agents.items():
        evv = agent.belief_system.emotion.emotion_vector.tolist()
        final_emotions[uid] = {EMOTIONS[i]: evv[i] for i in range(6)}

    simulator.social_network.close()

    result = {
        'strategy': strategy,
        'repeat': repeat, 'seed': seed,
        'num_agents': len(simulator.agents),
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

    logger.info(f"  <<< Completed {run_name}: {len(all_actions)} acts, "
                f"neg={cumulative['negative_emotion_ratio']:.3f}, "
                f"anger={cumulative['anger_ratio']:.3f}, "
                f"stance={cumulative['mean_stance']:.3f}")
    return result


# ============================================================
# Main
# ============================================================

STRATEGIES = ['baseline', 'swift_apology', 'transparency', 'dialogue', 'silence']
STRATEGY_LABELS = {
    'baseline': 'Actual Response',
    'swift_apology': 'Swift Empathetic Apology',
    'transparency': 'Proactive Transparency',
    'dialogue': 'Consumer Dialogue',
    'silence': 'Strategic Silence',
}


async def run_experiment():
    logger.info("=" * 70)
    logger.info("COUNTERFACTUAL PR INTERVENTION EXPERIMENT")
    logger.info("=" * 70)

    script_dir = Path(__file__).parent
    config_path = str(project_root / "scripts" / "xibeiyuzhicai" / "config.json")
    data_dir = str(project_root / "scripts" / "xibeiyuzhicai" / "data")
    output_dir = script_dir / "output" / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Config: {config_path}")
    logger.info(f"Data: {data_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Strategies: {STRATEGIES}")
    logger.info(f"Repeats: {NUM_REPEATS}, Users: {NUM_USERS}, Hours: {SIM_HOURS}")

    loader = DataLoader(data_dir)
    data = loader.load_all()
    users_raw = data['users']

    # Limit initial posts for efficiency (top engagement posts provide sufficient diversity)
    MAX_INITIAL_POSTS = 2000
    if len(data['initial_posts']) > MAX_INITIAL_POSTS:
        posts = data['initial_posts']
        for p in posts:
            rp = p.get('reposts', 0)
            cm = p.get('comments', 0)
            lk = p.get('likes', 0)
            rp = len(rp) if isinstance(rp, list) else (rp or 0)
            cm = len(cm) if isinstance(cm, list) else (cm or 0)
            lk = len(lk) if isinstance(lk, list) else (lk or 0)
            p['_engagement'] = rp + cm + lk
        posts.sort(key=lambda x: x['_engagement'], reverse=True)
        data['initial_posts'] = posts[:MAX_INITIAL_POSTS]
        for p in data['initial_posts']:
            p.pop('_engagement', None)

    logger.info(f"Loaded {len(users_raw)} users, {len(data['events'])} events, "
                f"{len(data['initial_posts'])} posts")

    logger.info("\nInitializing shared API pool...")
    cm_init = ConfigManager(config_path)
    cm_init.llm.embedding_device = "cpu"
    shared_api_pool = APIPool(cm_init.llm, cm_init.debug)

    all_results = []
    total_runs = len(STRATEGIES) * NUM_REPEATS
    run_idx = 0

    for repeat in range(NUM_REPEATS):
        logger.info(f"\n{'='*50}\nREPEAT {repeat+1}/{NUM_REPEATS}\n{'='*50}")
        for strategy in STRATEGIES:
            run_idx += 1
            logger.info(f"\n[Run {run_idx}/{total_runs}] Strategy: {STRATEGY_LABELS[strategy]}")
            t0 = time.time()
            result = await run_single_strategy(
                config_path, users_raw, data, strategy, repeat, output_dir, shared_api_pool)
            all_results.append(result)
            logger.info(f"  Elapsed: {time.time()-t0:.1f}s")

    summary = {
        'experiment_time': datetime.now().isoformat(),
        'strategies': STRATEGY_LABELS,
        'num_repeats': NUM_REPEATS,
        'num_users': NUM_USERS,
        'sim_hours': SIM_HOURS,
        'results': all_results,
    }
    with open(output_dir / 'experiment_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    logger.info("\n" + "=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Strategy':<28} {'NegEmo':>8} {'Anger':>8} {'Stance':>8} "
                f"{'Polar':>8} {'Aggr':>8} {'Acts':>8}")
    logger.info("-" * 80)
    for r in all_results:
        cm_ = r['cumulative_metrics']
        logger.info(f"{STRATEGY_LABELS[r['strategy']]:<28} "
                    f"{cm_['negative_emotion_ratio']:>8.3f} {cm_['anger_ratio']:>8.3f} "
                    f"{cm_['mean_stance']:>8.3f} {cm_['polarization_index']:>8.3f} "
                    f"{cm_['aggression_index']:>8.3f} {r['total_actions']:>8d}")

    logger.info(f"\nResults saved to: {output_dir}")
    return output_dir


if __name__ == "__main__":
    asyncio.run(run_experiment())
