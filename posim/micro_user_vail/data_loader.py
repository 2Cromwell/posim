# -*- coding: utf-8 -*-
"""数据加载器"""
import json
import os
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


def load_users(data_dir: str, num_agents: int = 100) -> List[Dict[str, Any]]:
    """加载用户数据，取前num_agents个citizen类型用户"""
    users_file = os.path.join(data_dir, 'users.json')
    with open(users_file, 'r', encoding='utf-8') as f:
        all_users = json.load(f)

    citizens = [u for u in all_users if u.get('agent_type') == 'citizen']
    selected = citizens[:num_agents]
    logger.info(f"Loaded {len(selected)} citizen agents from {len(all_users)} total users")
    return selected


def load_events(data_dir: str) -> Tuple[List[Dict], List[Dict]]:
    """加载事件数据，分别返回 global_broadcast 和 node_post"""
    events_file = os.path.join(data_dir, 'events.json')
    with open(events_file, 'r', encoding='utf-8') as f:
        all_events = json.load(f)

    broadcasts = [e for e in all_events if e.get('type') == 'global_broadcast']
    node_posts = [e for e in all_events if e.get('type') == 'node_post']

    broadcasts.sort(key=lambda x: x.get('time', ''))
    node_posts.sort(key=lambda x: x.get('time', ''))

    logger.info(f"Loaded {len(broadcasts)} global_broadcast events, {len(node_posts)} node_posts")
    return broadcasts, node_posts


def build_round_contexts(broadcasts: List[Dict], node_posts: List[Dict],
                         event_background: str) -> List[Dict[str, Any]]:
    """
    为每个模拟轮次构建上下文信息。
    每轮包含：当前broadcast事件、之前的所有broadcast、该轮可见的node_posts
    """
    rounds = []
    for i, broadcast in enumerate(broadcasts):
        bc_time = broadcast.get('time', '')
        prev_broadcasts = broadcasts[:i]

        if i + 1 < len(broadcasts):
            next_time = broadcasts[i + 1].get('time', '')
            visible_posts = [p for p in node_posts if bc_time <= p.get('time', '') < next_time]
        else:
            visible_posts = [p for p in node_posts if p.get('time', '') >= bc_time]

        exposed_posts_formatted = []
        for j, post in enumerate(visible_posts[:5], 1):
            sp = post.get('source_post', {})
            exposed_posts_formatted.append({
                'index': j,
                'author': sp.get('username', post.get('source', ['未知'])[0]),
                'content': post.get('content', '')[:200],
                'time': post.get('time', ''),
                'emotion': sp.get('emotion', '中性'),
                'reposts': sp.get('reposts', 0),
                'comments': sp.get('comments', 0),
                'likes': sp.get('likes', 0),
            })

        external_events_text = ""
        recent = prev_broadcasts[-3:] + [broadcast]
        for evt in recent:
            external_events_text += f"- [{evt.get('time', '')}] {evt.get('content', '')[:150]}\n"

        exposed_posts_text = ""
        for p in exposed_posts_formatted:
            exposed_posts_text += (
                f"{p['index']}. [{p['time']}] @{p['author']}：{p['content'][:120]}\n"
                f"   [点赞:{p['likes']} 转发:{p['reposts']} 评论:{p['comments']}]\n"
            )

        rounds.append({
            'round_index': i,
            'current_event': broadcast,
            'current_time': bc_time,
            'event_background': event_background,
            'previous_broadcasts': prev_broadcasts,
            'visible_posts': visible_posts,
            'exposed_posts_text': exposed_posts_text if exposed_posts_text else "（暂无可互动的博文）",
            'external_events_text': external_events_text,
            'exposed_posts_formatted': exposed_posts_formatted,
        })

    return rounds


def format_user_profile(user: Dict) -> Dict[str, str]:
    """格式化用户资料为prompt所需的文本"""
    identity = user.get('identity_description', '')
    beliefs = user.get('psychological_beliefs', [])
    tendency = user.get('behavior_tendency', {})
    emotions = user.get('emotion_vector', {})
    opinions = user.get('event_opinions', [])

    beliefs_text = "\n".join(f"- {b}" for b in beliefs) if beliefs else "无"
    tendency_text = ", ".join(f"{k}(权重{v})" for k, v in tendency.items()) if tendency else "无"

    opinions_text = ""
    for op in opinions:
        opinions_text += f"- [{op.get('time', '')}] 关于{op.get('subject', '')}：{op.get('opinion', '')}\n"

    return {
        'identity': identity,
        'beliefs_text': beliefs_text,
        'tendency_text': tendency_text,
        'opinions_text': opinions_text if opinions_text else "暂无",
        'raw_beliefs': beliefs,
        'raw_tendency': tendency,
        'raw_emotions': emotions,
        'raw_opinions': opinions,
    }
