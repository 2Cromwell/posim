# -*- coding: utf-8 -*-
"""
事件检测模块 v3.2 - 基于标签影响力的智能突发事件检测

主要功能：
1. 提取原创博文的话题标签，计算标签传播影响力
2. 排列Top50高影响力标签，包含详细传播指标
3. 让LLM智能筛选事件发展的关键标签
4. 对每个选定标签，生成 global_broadcast（热搜）+ node_post（关键博文）

输出格式保持与原版本一致
"""
import json
import os
import re
import math
import asyncio
from datetime import datetime, timedelta
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple
from openai import AsyncOpenAI

# ==================== 全局配置 ====================
LLM_CONFIG = {
    "base_url": "https://api.siliconflow.cn/v1/",
    "api_key": "sk-rjxdimwpuwtlqapnxjzleqlxlkjzmercjjitczrfkioatbsb",
    "model": "Pro/zai-org/GLM-4.7",
    "temperature": 0.3,
    "max_tokens": 8192,
    "concurrency": 30
}

TAG_CONFIG = {
    "top_n_tags": 50,
    "min_posts_per_tag": 2,
    "max_posts_for_llm": 15,
}

INFLUENCE_WEIGHTS = {
    "post_count": 0.10,
    "total_reposts": 0.25,
    "total_comments": 0.20,
    "total_likes": 0.08,
    "engagement_rate": 0.12,
    "propagation_depth": 0.10,
    "participant_diversity": 0.08,
    "time_concentration": 0.07
}


# ==================== 工具函数 ====================
def parse_time(s: str) -> Optional[datetime]:
    """解析时间字符串"""
    if not s:
        return None
    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M"]:
        try:
            return datetime.strptime(s, fmt)
        except:
            continue
    return None


def extract_tags(content: str) -> List[str]:
    """从内容中提取话题标签（不带#符号）"""
    if not content:
        return []
    tags = re.findall(r'#([^#]+)#', str(content))
    return [t.strip() for t in tags if t.strip() and len(t.strip()) > 1]


def load_json(path: str) -> Any:
    """加载JSON文件"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, path: str, indent: int = 2):
    """保存JSON文件"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def normalize_score(value: float, min_val: float, max_val: float) -> float:
    """归一化得分到0-1范围"""
    if max_val <= min_val:
        return 0.5
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))


def clean_content(content: str, max_length: int = 300) -> str:
    """清理内容"""
    if not content:
        return ""
    content = re.sub(r'http[s]?://\S+', '', content)
    content = re.sub(r'\s+', ' ', content).strip()
    return content[:max_length] if len(content) > max_length else content


def calculate_time_concentration(times: List[datetime]) -> float:
    """计算时间集中度"""
    if len(times) < 2:
        return 0.5
    sorted_times = sorted(times)
    total_span = (sorted_times[-1] - sorted_times[0]).total_seconds()
    if total_span <= 0:
        return 1.0
    intervals = [(sorted_times[i+1] - sorted_times[i]).total_seconds() 
                 for i in range(len(sorted_times)-1)]
    avg_interval = sum(intervals) / len(intervals)
    if avg_interval <= 0:
        return 1.0
    variance = sum((i - avg_interval) ** 2 for i in intervals) / len(intervals)
    cv = math.sqrt(variance) / avg_interval
    return max(0, 1 - cv / 2)


# ==================== 数据提取模块 ====================
class DataExtractor:
    """数据提取器"""
    
    def __init__(self, users_data: List[Dict], t_start: datetime, t_end: datetime):
        self.users_data = users_data
        self.t_start = t_start
        self.t_end = t_end
        self.user_type_map = {}
        self._build_user_type_map()
    
    def _build_user_type_map(self):
        for u in self.users_data:
            user_info = u.get('user_info', u)
            user_id = user_info.get('user_id', '')
            agent_type = user_info.get('agent_type', u.get('agent_type', 'citizen'))
            self.user_type_map[user_id] = agent_type
    
    def extract_all_posts(self) -> List[Dict]:
        """提取所有原创博文"""
        posts = []
        
        for u in self.users_data:
            user_info = u.get('user_info', u)
            user_id = user_info.get('user_id', u.get('user_id', ''))
            username = user_info.get('username', u.get('username', ''))
            followers = user_info.get('followers_count', u.get('followers_count', 0))
            agent_type = user_info.get('agent_type', u.get('agent_type', 'citizen'))
            verified = user_info.get('verified', u.get('verified', False))
            
            for p in u.get('original_posts', []):
                t = parse_time(p.get('time', ''))
                if not t or not (self.t_start <= t <= self.t_end):
                    continue
                
                content = p.get('content', '')
                tags = extract_tags(content)
                if not tags:
                    continue
                
                reposts = p.get('reposts', 0)
                comments = p.get('comments', 0)
                likes = p.get('likes', 0)
                
                posts.append({
                    'time': t,
                    'user_id': user_id,
                    'username': username,
                    'followers': followers,
                    'agent_type': agent_type,
                    'verified': verified,
                    'content': content,
                    'emotion': p.get('emotion', '中性'),
                    'reposts': reposts,
                    'comments': comments,
                    'likes': likes,
                    'influence_score': reposts * 2 + comments * 1.5 + likes * 0.5,
                    'url': p.get('url', ''),
                    'tags': tags
                })
        
        print(f"[INFO] 提取到 {len(posts)} 条带标签的原创博文")
        return sorted(posts, key=lambda x: x['time'])


# ==================== 标签影响力计算 ====================
class TagInfluenceCalculator:
    """标签影响力计算器"""
    
    def __init__(self, posts: List[Dict]):
        self.posts = posts
        self.tag_data = {}
    
    def calculate_all(self) -> Dict:
        """计算所有标签的影响力"""
        tag_data = defaultdict(lambda: {
            'posts': [],
            'total_reposts': 0,
            'total_comments': 0,
            'total_likes': 0,
            'users': set(),
            'user_types': defaultdict(int),
            'verified_count': 0,
            'first_time': None,
            'last_time': None,
            'times': []
        })
        
        for post in self.posts:
            for tag in post['tags']:
                stats = tag_data[tag]
                stats['posts'].append(post)
                stats['total_reposts'] += post.get('reposts', 0)
                stats['total_comments'] += post.get('comments', 0)
                stats['total_likes'] += post.get('likes', 0)
                stats['users'].add(post['user_id'])
                stats['user_types'][post.get('agent_type', 'citizen')] += 1
                if post.get('verified'):
                    stats['verified_count'] += 1
                stats['times'].append(post['time'])
                
                if stats['first_time'] is None or post['time'] < stats['first_time']:
                    stats['first_time'] = post['time']
                if stats['last_time'] is None or post['time'] > stats['last_time']:
                    stats['last_time'] = post['time']
        
        for tag, stats in tag_data.items():
            post_count = len(stats['posts'])
            if post_count < TAG_CONFIG['min_posts_per_tag']:
                continue
            
            stats['participant_diversity'] = len(stats['users']) / max(post_count, 1)
            total_interactions = stats['total_reposts'] + stats['total_comments'] + stats['total_likes']
            stats['engagement_rate'] = total_interactions / max(post_count, 1)
            stats['time_concentration'] = calculate_time_concentration(stats['times'])
            avg_reposts = stats['total_reposts'] / max(post_count, 1)
            stats['propagation_depth'] = min(5, math.log2(avg_reposts + 1) + 1)
            
            if stats['first_time'] and stats['last_time']:
                stats['active_duration_hours'] = (stats['last_time'] - stats['first_time']).total_seconds() / 3600
            else:
                stats['active_duration_hours'] = 0
            
            stats['influence_score'] = self._calculate_score(stats)
            self.tag_data[tag] = stats
        
        return self.tag_data
    
    def _calculate_score(self, stats: Dict) -> float:
        post_count = len(stats['posts'])
        post_score = normalize_score(math.log2(post_count + 1), 0, math.log2(500 + 1))
        reposts_score = normalize_score(stats['total_reposts'], 0, 10000)
        comments_score = normalize_score(stats['total_comments'], 0, 5000)
        likes_score = normalize_score(stats['total_likes'], 0, 20000)
        engagement_score = normalize_score(stats['engagement_rate'], 0, 1000)
        depth_score = normalize_score(stats['propagation_depth'], 0, 5)
        diversity_score = normalize_score(stats['participant_diversity'], 0, 1)
        concentration_score = stats['time_concentration']
        
        score = (
            post_score * INFLUENCE_WEIGHTS['post_count'] +
            reposts_score * INFLUENCE_WEIGHTS['total_reposts'] +
            comments_score * INFLUENCE_WEIGHTS['total_comments'] +
            likes_score * INFLUENCE_WEIGHTS['total_likes'] +
            engagement_score * INFLUENCE_WEIGHTS['engagement_rate'] +
            depth_score * INFLUENCE_WEIGHTS['propagation_depth'] +
            diversity_score * INFLUENCE_WEIGHTS['participant_diversity'] +
            concentration_score * INFLUENCE_WEIGHTS['time_concentration']
        )
        return round(min(1.0, score), 4)
    
    def get_top_tags(self, n: int = 50) -> List[Tuple[str, Dict]]:
        sorted_tags = sorted(self.tag_data.items(), key=lambda x: x[1]['influence_score'], reverse=True)
        return sorted_tags[:n]
    
    def format_tag_info(self, tag: str, stats: Dict, rank: int) -> Dict:
        post_count = len(stats.get('posts', []))
        user_types = stats.get('user_types', {})
        
        return {
            'rank': rank,
            'tag': tag,
            'influence_score': round(stats.get('influence_score', 0), 4),
            'post_count': post_count,
            'total_reposts': stats.get('total_reposts', 0),
            'total_comments': stats.get('total_comments', 0),
            'total_likes': stats.get('total_likes', 0),
            'avg_engagement': round(stats.get('engagement_rate', 0), 1),
            'first_time': stats.get('first_time').strftime('%m-%d %H:%M') if stats.get('first_time') else 'N/A',
            'last_time': stats.get('last_time').strftime('%m-%d %H:%M') if stats.get('last_time') else 'N/A',
            'unique_users': len(stats.get('users', set())),
            'media_posts': user_types.get('media', 0),
            'government_posts': user_types.get('government', 0),
            'kol_posts': user_types.get('kol', 0),
        }


# ==================== LLM调用管理器 ====================
class LLMManager:
    """LLM调用管理器"""
    
    def __init__(self, config: Dict = None):
        self.config = config or LLM_CONFIG
        self.semaphore = asyncio.Semaphore(self.config['concurrency'])
        self.client = None
    
    def _get_client(self) -> AsyncOpenAI:
        if self.client is None:
            self.client = AsyncOpenAI(
                api_key=self.config['api_key'],
                base_url=self.config['base_url']
            )
        return self.client
    
    async def call(self, prompt: str, temperature: float = None, max_tokens: int = None,
                   task_name: str = "", show_io: bool = True) -> Optional[str]:
        """调用LLM"""
        async with self.semaphore:
            try:
                if show_io:
                    print(f"\n{'='*80}")
                    print(f"[LLM INPUT] {task_name}")
                    print('='*80)
                    print(prompt)
                    print('='*80)
                
                client = self._get_client()
                resp = await client.chat.completions.create(
                    model=self.config['model'],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature or self.config['temperature'],
                    max_tokens=max_tokens or self.config['max_tokens']
                )
                response = resp.choices[0].message.content
                
                if show_io:
                    print(f"\n{'='*80}")
                    print(f"[LLM OUTPUT] {task_name}")
                    print('='*80)
                    print(response)
                    print('='*80)
                
                return response
            except Exception as e:
                print(f"[LLM ERROR] {e}")
                return None
    
    @staticmethod
    def parse_json_response(response: str) -> Optional[Dict]:
        """解析LLM的JSON响应"""
        if not response:
            return None
        
        try:
            return json.loads(response)
        except:
            pass
        
        # 尝试提取JSON块
        patterns = [r'```json\s*([\s\S]*?)\s*```', r'```\s*([\s\S]*?)\s*```']
        for pattern in patterns:
            matches = re.findall(pattern, response)
            for match in matches:
                try:
                    text = match if isinstance(match, str) else match[0]
                    start = text.find('{')
                    end = text.rfind('}')
                    if start >= 0 and end > start:
                        return json.loads(text[start:end+1])
                except:
                    continue
        
        # 直接找JSON
        try:
            start = response.find('{')
            end = response.rfind('}')
            if start >= 0 and end > start:
                return json.loads(response[start:end+1])
        except:
            pass
        
        return None


# ==================== 提示词模板 ====================
# 提示词1：筛选关键标签（只筛选，不判断类型）
PROMPT_TAG_SELECTION = """你是一位专业的舆情分析专家，擅长识别舆情事件发展过程中的关键标签。

【事件背景（可参考）】
{event_background}



【数据概况】
- 时间范围：{time_range}
- 总博文数：{total_posts}条
- 有效标签数：{total_tags}个

【Top {tag_count} 高影响力标签】
（按综合影响力得分排序）

{tags_detail}

【任务】
请从上述标签中筛选出**代表事件发展关键节点**的标签。

筛选原则：
1. 选择能代表事件重要进展、转折、爆点的标签
2. 选择传播指标（转发、评论）较高的标签
3. 当前每个标签其实都代表了一个事件，如果有代表同一个事件的标签，可以合并为一组
4. **尽可能多选**，建议筛选 10-20 个标签组
5. 每个标签组代表一个独立的事件节点，同一个标签组内不要有一类事件，比如当事人回应与自证可能有不同当事人回应，但一定不要将其作为一个标签组，应该分为不同事件节点、不同标签组
6. 并不是每个标签都需要，有些不重要或者没有影响力的标签可以丢弃

【输出格式】
```json
{{
    "selected_tags": [
        {{
            "tag_group_id": 1,
            "tags": ["标签1", "标签2"],
            "group_name": "标签组的统一名称",
            "reason": "选择理由"
        }}
    ]
}}
```

【重要】tags数组中的标签名必须与上面列表中的标签名**完全一致**！"""

# 提示词2：为选定标签生成事件（同时生成global_broadcast和node_post）
PROMPT_EVENT_GENERATION = """你是一位专业的舆情分析师，需要基于话题标签下的博文，提取外部刺激事件。

【事件背景（可参考）】
{event_background}



【当前标签组】
标签组名称：{group_name}
包含标签：{tags}
选择理由：{reason}

【重要说明】
- 标签首次出现时间：{first_time}
- 以下博文是该标签首次出现后1小时内的博文（共{post_count}条）
- 请仅基于这些早期博文生成事件，不要假设后续发展

【该标签下的早期博文】（按影响力排序）
{posts_content}

【任务】
基于上述**早期博文**，生成两类事件：

1. **global_broadcast（全局广播/热搜事件）**
   - 类似于微博热搜的信息
   - 用于向所有用户广播的重大进展
   - **注意：内容要简洁，50-100字即可，只描述当前已知信息**

2. **node_post（关键节点博文）**
   - 从上述博文中选择 1-2 条最具影响力的博文
   - 这些博文通常是：突发爆料、官方回应、媒体报道、意见领袖发声等
   - 需要标注博文序号

【输出格式】
```json
{{
    "global_broadcast": {{
        "title": "热搜标题（15-20字，简洁）",
        "content": "热搜简介（50-100字，概括当前事件，突出能够影响事件走向的关键信息）",
        "trigger_time": "{first_time}"
    }},
    "node_posts": [
        {{
            "post_index": 1,
            "reason": "选择理由",
            "event_type": "爆料/官方回应/媒体报道/意见领袖发声/其他"
        }}
    ]
}}
```

【重要】
- 只基于提供的博文内容，不要编造或推测后续发展
- 全局广播内容要**简短精炼**，不超过100字"""


# ==================== 事件生成器 ====================
class EventGenerator:
    """事件生成器"""
    
    def __init__(self, event_background: str, tag_stats: Dict, posts: List[Dict], llm: LLMManager):
        self.event_background = event_background
        self.tag_stats = tag_stats
        self.posts = posts
        self.llm = llm
    
    async def select_tags(self, top_tags_info: List[Dict], time_range: str) -> List[Dict]:
        """让LLM筛选关键标签"""
        # 格式化标签信息
        tags_detail_lines = []
        for info in top_tags_info:
            line = (
                f"{info['rank']:2d}. {info['tag']}\n"
                f"    影响力:{info['influence_score']:.3f} | "
                f"博文:{info['post_count']} | "
                f"转发:{info['total_reposts']} | "
                f"评论:{info['total_comments']} | "
                f"点赞:{info['total_likes']}\n"
                f"    平均互动:{info['avg_engagement']:.0f} | "
                f"独立用户:{info['unique_users']} | "
                f"首发:{info['first_time']} | 最新:{info['last_time']}\n"
            )
            tags_detail_lines.append(line)
        
        prompt = PROMPT_TAG_SELECTION.format(
            event_background=self.event_background,
            time_range=time_range,
            total_posts=len(self.posts),
            total_tags=len(self.tag_stats),
            tag_count=len(top_tags_info),
            tags_detail="\n\n".join(tags_detail_lines)
        )
        
        response = await self.llm.call(prompt, temperature=0.3, task_name="标签筛选")
        result = self.llm.parse_json_response(response)
        
        if result and 'selected_tags' in result:
            return result['selected_tags']
        else:
            print("[WARN] LLM标签筛选失败，使用默认策略")
            # 默认选择前10个标签
            return [
                {"tag_group_id": i+1, "tags": [info['tag']], "group_name": info['tag'], "reason": "默认选择"}
                for i, info in enumerate(top_tags_info[:10])
            ]
    
    def _get_posts_for_tags(self, tags: List[str], max_count: int = 15, time_window_hours: float = 1.0) -> Tuple[List[Dict], datetime]:
        """获取标签相关的博文（避免数据泄露）
        
        策略：找到标签首次出现的时间点，只获取该时间点往后time_window_hours小时内的博文
        
        Returns:
            (博文列表, 首次出现时间)
        """
        tag_set = set(tags)
        
        # 找到所有包含这些标签的博文
        related = [p for p in self.posts if any(t in tag_set for t in p.get('tags', []))]
        
        if not related:
            return [], None
        
        # 按时间排序，找到首次出现时间
        related.sort(key=lambda x: x['time'])
        first_time = related[0]['time']
        
        # 只保留首次出现时间往后time_window_hours小时内的博文
        time_cutoff = first_time + timedelta(hours=time_window_hours)
        filtered = [p for p in related if p['time'] <= time_cutoff]
        
        # 在时间窗口内按影响力排序
        filtered.sort(key=lambda x: x.get('influence_score', 0), reverse=True)
        
        return filtered[:max_count], first_time
    
    def _format_posts(self, posts: List[Dict]) -> str:
        """格式化博文列表"""
        lines = []
        for i, p in enumerate(posts, 1):
            agent_type_cn = {'media': '媒体', 'government': '政府', 'kol': 'KOL', 'citizen': '普通用户'}.get(p.get('agent_type'), '普通用户')
            verified = "✓" if p.get('verified') else ""
            lines.append(
                f"【{i}】[{agent_type_cn}]{verified} @{p['username']} (粉丝:{p.get('followers', 0):,})\n"
                f"    时间: {p['time'].strftime('%Y-%m-%d %H:%M')}\n"
                f"    内容: {clean_content(p.get('content', ''), 250)}\n"
                f"    转发:{p.get('reposts', 0)} | 评论:{p.get('comments', 0)} | 点赞:{p.get('likes', 0)}"
            )
        return "\n\n".join(lines)
    
    async def generate_events_for_tag_group(self, tag_group: Dict) -> List[Dict]:
        """为一个标签组生成事件"""
        tags = tag_group.get('tags', [])
        group_name = tag_group.get('group_name', tags[0] if tags else '')
        reason = tag_group.get('reason', '')
        
        # 获取相关博文（只获取标签首次出现后1小时内的博文，避免数据泄露）
        related_posts, first_time = self._get_posts_for_tags(tags, TAG_CONFIG['max_posts_for_llm'], time_window_hours=1.0)
        
        if not related_posts:
            print(f"[WARN] 标签组 '{group_name}' 没有找到相关博文")
            return []
        
        first_time_str = first_time.strftime('%Y-%m-%dT%H:%M') if first_time else ''
        print(f"  标签首次出现: {first_time_str}, 获取到 {len(related_posts)} 条早期博文")
        
        prompt = PROMPT_EVENT_GENERATION.format(
            event_background=self.event_background,
            group_name=group_name,
            tags=", ".join(tags),
            reason=reason,
            first_time=first_time_str,
            post_count=len(related_posts),
            posts_content=self._format_posts(related_posts)
        )
        
        response = await self.llm.call(prompt, temperature=0.4, task_name=f"事件生成-{group_name}")
        result = self.llm.parse_json_response(response)
        
        events = []
        
        if result:
            # 生成 global_broadcast 事件
            gb = result.get('global_broadcast', {})
            if gb:
                trigger_time = gb.get('trigger_time', '')
                event_time = parse_time(trigger_time) or related_posts[0]['time']
                
                events.append({
                    "time": event_time.strftime("%Y-%m-%dT%H:%M"),
                    "type": "global_broadcast",
                    "source": ["external"],
                    "topic": gb.get('title', group_name),
                    "content": gb.get('content', clean_content(related_posts[0]['content'], 500)),
                    "influence": 0.85,
                    "metadata": {
                        "original_tags": tags,
                        "group_name": group_name,
                        "detection_method": "tag_influence_v3.2",
                        "reason": reason,
                        "core_posts_count": len(related_posts)
                    }
                })
            
            # 生成 node_post 事件
            node_posts = result.get('node_posts', [])
            for np in node_posts:
                idx = np.get('post_index', 1) - 1
                if 0 <= idx < len(related_posts):
                    post = related_posts[idx]
                    events.append({
                        "time": post['time'].strftime("%Y-%m-%dT%H:%M"),
                        "type": "node_post",
                        "source": [post['user_id']],
                        "topic": group_name,
                        "content": clean_content(post['content'], 500),
                        "influence": 0.7,
                        "metadata": {
                            "original_tags": tags,
                            "group_name": group_name,
                            "detection_method": "tag_influence_v3.2",
                            "trigger_type": np.get('event_type', ''),
                            "selection_reason": np.get('reason', '')
                        },
                        "source_post": {
                            "user_id": post['user_id'],
                            "username": post['username'],
                            "agent_type": post['agent_type'],
                            "time": post['time'].strftime("%Y-%m-%dT%H:%M:%S"),
                            "content": post['content'],
                            "url": post.get('url', ''),
                            "emotion": post.get('emotion', ''),
                            "reposts": post.get('reposts', 0),
                            "comments": post.get('comments', 0),
                            "likes": post.get('likes', 0),
                            "tags": post.get('tags', [])
                        }
                    })
        else:
            # 默认策略
            top_post = related_posts[0]
            events.append({
                "time": top_post['time'].strftime("%Y-%m-%dT%H:%M"),
                "type": "global_broadcast",
                "source": ["external"],
                "topic": group_name,
                "content": clean_content(top_post['content'], 500),
                "influence": 0.7,
                "metadata": {
                    "original_tags": tags,
                    "detection_method": "tag_influence_v3.2_default"
                }
            })
        
        return events


# ==================== 主流程 ====================
async def main():
    """主函数"""
    print("=" * 80)
    print("事件检测模块 v3.2 - 基于标签影响力的智能突发事件检测")
    print(f"使用模型: {LLM_CONFIG['model']} | 并发数: {LLM_CONFIG['concurrency']}")
    print("=" * 80)
    
    # 路径配置
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "../config.json")
    config = load_json(config_path)
    
    output_dir = os.path.join(script_dir, "..", config['paths']['output_dir'])
    base_data_path = os.path.join(output_dir, config['paths']['base_data_file'])
    output_events_path = os.path.join(output_dir, "events.json")
    
    # ========== STEP 1: 加载数据 ==========
    print("\n[STEP 1/4] 加载数据...")
    users_data = load_json(base_data_path)
    print(f"[INFO] 加载 {len(users_data)} 条用户数据")
    
    t_start = parse_time(config['filter']['start_time'])
    t_end = parse_time(config['filter']['end_time'])
    time_range = f"{t_start.strftime('%Y-%m-%d %H:%M')} ~ {t_end.strftime('%Y-%m-%d %H:%M')}"
    print(f"[INFO] 时间范围: {time_range}")
    
    # ========== STEP 2: 提取博文和计算标签影响力 ==========
    print("\n[STEP 2/4] 提取博文和计算标签影响力...")
    extractor = DataExtractor(users_data, t_start, t_end)
    posts = extractor.extract_all_posts()
    
    if not posts:
        print("[ERROR] 无符合条件的博文")
        return []
    
    calculator = TagInfluenceCalculator(posts)
    tag_stats = calculator.calculate_all()
    print(f"[INFO] 有效标签数量: {len(tag_stats)}")
    
    # 获取Top标签
    top_tags = calculator.get_top_tags(TAG_CONFIG['top_n_tags'])
    top_tags_info = [calculator.format_tag_info(tag, stats, i+1) for i, (tag, stats) in enumerate(top_tags)]
    
    # 打印Top标签
    print(f"\n[标签影响力 Top{len(top_tags_info)}]")
    for info in top_tags_info:
        print(f"  {info['rank']:2d}. #{info['tag']}# "
              f"(影响力:{info['influence_score']:.3f}, 博文:{info['post_count']}, "
              f"转发:{info['total_reposts']}, 评论:{info['total_comments']})")
    
    # 事件背景
    event_background = f"{config.get('event_name', '')}：{config.get('simulation', {}).get('event_background', '')}"
    
    # ========== STEP 3: LLM筛选关键标签 ==========
    print("\n[STEP 3/4] LLM筛选关键标签...")
    llm = LLMManager()
    generator = EventGenerator(event_background, tag_stats, posts, llm)
    
    selected_tag_groups = await generator.select_tags(top_tags_info, time_range)
    print(f"\n[INFO] 筛选出 {len(selected_tag_groups)} 个标签组")
    
    for tg in selected_tag_groups:
        print(f"  - {tg.get('group_name', '')}: {tg.get('tags', [])}")
    
    # ========== STEP 4: 为每个标签组生成事件 ==========
    print("\n[STEP 4/4] 为每个标签组生成事件...")
    all_events = []
    
    for i, tag_group in enumerate(selected_tag_groups, 1):
        print(f"\n[处理 {i}/{len(selected_tag_groups)}] {tag_group.get('group_name', '')}")
        events = await generator.generate_events_for_tag_group(tag_group)
        all_events.extend(events)
        print(f"  生成 {len(events)} 个事件")
    
    # 按时间排序
    all_events.sort(key=lambda x: x['time'])
    
    # 保存结果
    save_json(all_events, output_events_path)
    print(f"\n[INFO] 事件已保存至: {output_events_path}")
    
    # 打印摘要
    print("\n" + "=" * 80)
    print("事件检测结果摘要")
    print("=" * 80)
    gb_count = sum(1 for e in all_events if e['type'] == 'global_broadcast')
    np_count = sum(1 for e in all_events if e['type'] == 'node_post')
    print(f"总事件数: {len(all_events)}")
    print(f"  - 全局广播 (global_broadcast): {gb_count}")
    print(f"  - 节点发布 (node_post): {np_count}")
    
    print("\n事件列表:")
    for e in all_events:
        print(f"  [{e['time']}] [{e['type']}] {e['topic'][:40]}...")
    
    return all_events


if __name__ == '__main__':
    asyncio.run(main())
