"""
Step4: 生成users.json
- 加入行为动态分布采样到身份信念
- 多API端点并发调用
- 增量写入JSON实现断点恢复
"""
import json
import os
import random
import asyncio
import numpy as np
from datetime import datetime
from openai import AsyncOpenAI
from collections import defaultdict
import threading

# 加载配置
with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

CUTOFF_TIME = datetime.strptime(config['filter']['cutoff_time'], "%Y-%m-%d %H:%M:%S")
PRINT_PROB = config['llm']['print_probability']
CONCURRENCY = config['llm']['concurrency']

# 行为类型及其描述
BEHAVIOR_TYPES = {
    'like': '点赞',
    'repost': '转发',
    'repost_comment': '转发并评论',
    'short_comment': '短评论（20字以内）',
    'long_comment': '长评论（20字以上）',
    'short_post': '短博文（50字以内）',
    'long_post': '长博文（50字以上）'
}

# 行为倾向分布
BEHAVIOR_DISTRIBUTIONS = config['behavior_tendency_distributions']

# 心理认知信念池
PSYCHOLOGICAL_BELIEFS_POOL = config['psychological_beliefs_pool']

# 初始化多个API客户端
api_configs = [c for c in config['llm']['api_configs'] if c.get('enabled', True)]
api_clients = []
for cfg in api_configs:
    client = AsyncOpenAI(
        api_key=cfg['api_key'],
        base_url=cfg['base_url']
    )
    api_clients.append({
        'client': client,
        'model': cfg['model'],
        'temperature': cfg.get('temperature', 0.7),
        'name': cfg['name'],
        'weight': cfg.get('weight', 1.0)
    })

print(f"[INFO] 初始化了 {len(api_clients)} 个API端点")

# API轮询计数器
api_counter = 0
api_lock = threading.Lock()


def get_next_api():
    """轮询获取下一个API客户端"""
    global api_counter
    with api_lock:
        idx = api_counter % len(api_clients)
        api_counter += 1
    return api_clients[idx]


def parse_time(time_str):
    """解析时间字符串"""
    if not time_str:
        return None
    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M"]:
        try:
            return datetime.strptime(time_str, fmt)
        except:
            continue
    return None


def is_before_cutoff(time_str):
    """判断时间是否在截止时间之前"""
    dt = parse_time(time_str)
    return dt is not None and dt < CUTOFF_TIME


def get_pre_cutoff_data(user_data):
    """
    获取用户在截止时间（模拟开始）之前的行为数据，用于推断用户信念。
    仅使用 cutoff_time 之前的数据，避免数据泄露；若无则返回空列表。
    """
    pre_posts = []
    
    for post in user_data.get('original_posts', []):
        if is_before_cutoff(post.get('time', '')):
            pre_posts.append({
                "type": "原创",
                "content": post['content'][:200],
                "time": post['time'],
                "emotion": post.get('emotion', '')
            })
    
    for post in user_data.get('repost_posts', []):
        if is_before_cutoff(post.get('time', '')):
            pre_posts.append({
                "type": "转发",
                "content": post.get('user_content', '')[:100],
                "root_content": post.get('root_content', '')[:100],
                "time": post['time']
            })
    
    for comment in user_data.get('comments', []):
        if is_before_cutoff(comment.get('time', '')):
            pre_posts.append({
                "type": "评论",
                "content": comment.get('content', '')[:150],
                "time": comment['time']
            })
    
    return pre_posts


def sample_behavior_tendency(agent_type):
    """
    根据用户类型从行为倾向分布中采样
    返回采样得到的行为倾向描述
    """
    dist = BEHAVIOR_DISTRIBUTIONS.get(agent_type, BEHAVIOR_DISTRIBUTIONS['citizen'])
    behaviors = list(dist.keys())
    probs = [dist[b] for b in behaviors]
    
    # 采样3-5个主要行为倾向（允许重复以体现权重）
    num_samples = random.randint(3, 5)
    sampled = np.random.choice(behaviors, size=num_samples, p=probs, replace=True)
    
    # 统计频次得到个人化的倾向分布
    tendency_counts = defaultdict(int)
    for b in sampled:
        tendency_counts[b] += 1
    
    # 按频次排序
    sorted_tendencies = sorted(tendency_counts.items(), key=lambda x: -x[1])
    
    # 生成倾向描述
    tendency_desc_parts = []
    for behavior, count in sorted_tendencies:
        intensity = "偶尔" if count == 1 else ("经常" if count == 2 else "非常喜欢")
        tendency_desc_parts.append(f"{intensity}{BEHAVIOR_TYPES[behavior]}")
    
    # 构建完整描述
    primary = sorted_tendencies[0][0] if sorted_tendencies else 'like'
    primary_desc = BEHAVIOR_TYPES[primary]
    
    tendency_info = {
        'distribution': dict(tendency_counts),
        'primary_behavior': primary,
        'description': "、".join(tendency_desc_parts),
        'detailed': f"主要行为倾向是{primary_desc}，" + "、".join(tendency_desc_parts[:3])
    }
    
    return tendency_info


def sample_psychological_beliefs():
    """从心理认知信念池中抽样"""
    beliefs = []
    categories = list(PSYCHOLOGICAL_BELIEFS_POOL.keys())
    num_categories = random.randint(2, 4)
    selected_categories = random.sample(categories, min(num_categories, len(categories)))
    
    for cat in selected_categories:
        belief = random.choice(PSYCHOLOGICAL_BELIEFS_POOL[cat])
        beliefs.append(belief)
    
    return beliefs


def build_inference_prompt(user_info, pre_posts, agent_type, behavior_tendency):
    """构建用于推断用户信念的提示词（含行为动态）"""
    
    user_desc = f"""用户信息：
- 昵称：{user_info['username']}
- 粉丝数：{user_info.get('followers_count', 0)}
- 关注数：{user_info.get('following_count', 0)}
- 发博数：{user_info.get('posts_count', 0)}
- 认证类型：{user_info.get('verified_type', '普通用户')}
- 性别：{user_info.get('gender', '未知')}
- 地区：{user_info.get('location', '未知')}
- 个人简介：{user_info.get('description', '无')}"""

    if pre_posts:
        posts_text = "\n".join([
            f"[{p['type']}] {p.get('time', '')}: {p.get('content', '')[:100]}"
            for p in pre_posts[:10]
        ])
        behavior_section = f"\n\n该用户在武大图书馆事件发酵前的历史行为（截至{CUTOFF_TIME.strftime('%Y年%m月%d日')}）：\n{posts_text}"
    else:
        behavior_section = "\n\n该用户在事件发酵前暂无历史行为数据。"

    # 行为动态倾向描述
    behavior_tendency_section = f"""

该用户的社交媒体行为动态特征（基于分析）：
- 行为倾向：{behavior_tendency['detailed']}
- 参与方式偏好：{behavior_tendency['description']}

行为类型说明：
- 点赞：成本最低的参与方式，表示认同或关注
- 转发：传播信息但不表态
- 转发并评论：传播信息同时表达观点
- 短评论：快速表达简短看法
- 长评论：深入分析或情绪宣泄
- 短博文：简短原创表态
- 长博文：详细阐述个人观点"""

    type_hints = {
        'kol': '该用户是意见领袖（KOL），通常有较大的粉丝基础和社会影响力，倾向于发表有深度的观点。',
        'media': '该用户是媒体账号，通常发布新闻报道，注重客观性和权威性。',
        'government': '该用户是政府官方账号，通常发布政务信息，态度中立权威。',
        'citizen': '该用户是普通网民，可能有各种心理特征和情绪倾向，行为成本意识较强。'
    }

    prompt = f"""你是一个社交媒体用户心理分析专家。请基于以下用户信息，推断该用户的三类信念。

{user_desc}
{behavior_section}
{behavior_tendency_section}

用户类型提示：{type_hints.get(agent_type, type_hints['citizen'])}

请推断以下三类信念：

1. **角色身份信念（identity_description）**：用第一人称描述该用户的社交媒体身份定位、行为特点和表达风格。**必须详细包含其行为动态特征**。约120-200字。

2. **心理认知信念（psychological_beliefs）**：该用户在网络舆情参与中可能持有的深层心理认知，体现其对社会、权力、财富等的固有看法。给出5条以上信念条目。
   参考心理类型：自我实现类、猎奇探究类、减压宣泄类、仇官仇富类、跟风从众类。

3. **事件观点信念（event_opinions）**：如果有相关数据，推断该用户对武大图书馆事件的初始观点。格式为：时间、主体、观点、原因。如果没有相关数据则返回空列表。

请严格按以下JSON格式输出，不要有其他内容：
{{
    "identity_description": "第一人称的身份描述（必须包含行为动态特征）...",
    "psychological_beliefs": ["信念1", "信念2", "信念3"],
    "event_opinions": [
        {{"time": "2025-07-27T12:00", "subject": "武大图书馆事件", "opinion": "观点内容", "reason": "原因"}}
    ]
}}"""

    return prompt


def build_expansion_prompt(user_info, sampled_beliefs, agent_type, behavior_tendency):
    """为无先验数据的用户构建扩展提示词"""
    
    user_desc = f"""用户基本信息：
- 昵称：{user_info['username']}
- 粉丝数：{user_info.get('followers_count', 0)}
- 认证类型：{user_info.get('verified_type', '普通用户')}
- 性别：{user_info.get('gender', '未知')}
- 地区：{user_info.get('location', '未知')}"""

    beliefs_text = "\n".join([f"- {b}" for b in sampled_beliefs])
    
    behavior_tendency_section = f"""
该用户的社交媒体行为动态特征（基于统计采样）：
- 行为倾向：{behavior_tendency['detailed']}
- 参与方式偏好：{behavior_tendency['description']}

行为类型说明：
- 点赞：成本最低，表示认同或关注
- 转发：传播但不表态
- 转发并评论：传播同时表态
- 短评论/长评论：评论区表达观点
- 短博文/长博文：原创发布内容"""

    prompt = f"""你是一个社交媒体用户心理分析专家。请基于以下用户基本信息、抽样的心理信念和行为动态特征，扩展生成完整的用户画像。

{user_desc}

该用户可能持有的心理认知倾向（抽样结果）：
{beliefs_text}
{behavior_tendency_section}

请结合微博网民的典型心理特征，生成：

1. **角色身份信念（identity_description）**：用第一人称描述该用户可能的社交媒体身份定位、行为特点和表达风格。**必须包含其行为动态特征**。约120-200字。要符合普通微博用户的真实感。

2. **心理认知信念（psychological_beliefs）**：基于抽样信念进行扩展，生成5条以上具体的心理认知条目。要体现真实网民的心理特征。

3. **事件观点信念（event_opinions）**：该用户在武大图书馆事件初期可能持有的初始观点。考虑其心理特征和行为倾向。

请严格按以下JSON格式输出：
{{
    "identity_description": "第一人称的身份描述（必须包含行为动态特征）...",
    "psychological_beliefs": ["信念1", "信念2", "信念3"],
    "event_opinions": [
        {{"time": "2025-07-27T14:00", "subject": "武大图书馆事件", "opinion": "观点内容", "reason": "原因"}}
    ]
}}"""

    return prompt


async def call_llm(prompt, user_id):
    """调用大模型API（轮询多端点）"""
    api = get_next_api()
    try:
        response = await api['client'].chat.completions.create(
            model=api['model'],
            messages=[{"role": "user", "content": prompt}],
            temperature=api['temperature']
        )
        result = response.choices[0].message.content
        
        if random.random() < PRINT_PROB:
            print(f"\n{'='*40}")
            print(f"[DEBUG] 用户 {user_id} 的LLM调用 (API: {api['name']})")
            print(f"[INPUT] {prompt[:500]}...")
            print(f"[OUTPUT] {result[:500]}...")
            print(f"{'='*40}\n")
        
        return result
    except Exception as e:
        print(f"[ERROR] LLM调用失败 (用户 {user_id}, API: {api['name']}): {e}")
        return None


def parse_llm_response(response_text):
    """解析LLM响应"""
    if not response_text:
        return None
    
    try:
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        if start >= 0 and end > start:
            json_str = response_text[start:end]
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    
    return None


def load_processed_users(output_file):
    """从JSONL文件加载已处理的用户ID集合和数据"""
    processed = set()
    users = []
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        user = json.loads(line)
                        processed.add(user['user_id'])
                        users.append(user)
                    except:
                        pass
    return processed, users


def append_user_to_jsonl(user_result, output_file):
    """追加写入用户数据到JSONL文件（每行一个JSON）"""
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(user_result, ensure_ascii=False) + '\n')


async def process_user(user_data, influence_data, output_file, processed_users, semaphore, progress_counter, total_users):
    """处理单个用户"""
    async with semaphore:
        user_info = user_data['user_info']
        user_id = user_info['user_id']
        
        # 跳过已处理的用户
        if user_id in processed_users:
            return None
        
        # 获取影响力信息
        influence_info = influence_data.get(user_id, {})
        agent_type = influence_info.get('agent_type', 'citizen')
        influence = influence_info.get('influence', 0.1)
        
        # 采样行为倾向
        behavior_tendency = sample_behavior_tendency(agent_type)
        
        # 获取截止时间之前的行为数据
        pre_posts = get_pre_cutoff_data(user_data)
        
        # 根据是否有先验数据选择不同的推断策略
        if pre_posts:
            prompt = build_inference_prompt(user_info, pre_posts, agent_type, behavior_tendency)
        else:
            sampled_beliefs = sample_psychological_beliefs()
            prompt = build_expansion_prompt(user_info, sampled_beliefs, agent_type, behavior_tendency)
        
        # 调用LLM
        response = await call_llm(prompt, user_id)
        parsed = parse_llm_response(response)
        
        # 构建用户数据结构
        user_result = {
            "user_id": user_id,
            "username": user_info['username'],
            "agent_type": agent_type,
            "followers_count": int(user_info.get('followers_count', 0) or 0),
            "following_count": int(user_info.get('following_count', 0) or 0),
            "posts_count": int(user_info.get('posts_count', 0) or 0),
            "verified": '认证' in str(user_info.get('verified_type', '')),
            "description": str(user_info.get('description', '') or ''),
            "raw_profile": {
                "gender": user_info.get('gender', '未知'),
                "location": user_info.get('location', '未知'),
                "verified_type": str(user_info.get('verified_type', '普通用户') or '普通用户'),
                "register_time": user_info.get('register_time', '')
            },
            "behavior_tendency": behavior_tendency['distribution'],
            "emotion_vector": {
                "happiness": 0.0, "sadness": 0.0, "anger": 0.0,
                "fear": 0.0, "surprise": 0.0, "disgust": 0.0
            },
            "history_posts": []
        }
        
        # 添加LLM推断结果
        if parsed:
            user_result["identity_description"] = parsed.get("identity_description", "")
            user_result["psychological_beliefs"] = parsed.get("psychological_beliefs", [])
            user_result["event_opinions"] = parsed.get("event_opinions", [])
        else:
            default_behavior_desc = f"，平时{behavior_tendency['description']}"
            user_result["identity_description"] = f"我是一名普通微博用户，昵称{user_info['username']}{default_behavior_desc}。"
            user_result["psychological_beliefs"] = sample_psychological_beliefs()
            user_result["event_opinions"] = []
        
        # 追加写入JSONL
        append_user_to_jsonl(user_result, output_file)
        
        # 更新进度
        with progress_counter['lock']:
            progress_counter['count'] += 1
            if progress_counter['count'] % 100 == 0:
                print(f"[PROGRESS] 已处理 {progress_counter['count']}/{total_users} 用户")
        
        return user_result


async def main():
    print("=" * 60)
    print("Step4: 生成users.json (增量写入版)")
    print("=" * 60)
    
    # 加载基础数据
    base_data_path = os.path.join(config['paths']['output_dir'], config['paths']['base_data_file'])
    with open(base_data_path, 'r', encoding='utf-8') as f:
        users_data = json.load(f)
    
    # 加载影响力数据
    influence_path = os.path.join(config['paths']['output_dir'], 'users_influence.json')
    with open(influence_path, 'r', encoding='utf-8') as f:
        influence_list = json.load(f)
    influence_data = {u['user_id']: u for u in influence_list}
    
    print(f"[INFO] 加载了 {len(users_data)} 个用户数据")
    print(f"[INFO] API端点数: {len(api_clients)}")
    print(f"[INFO] 并发数: {CONCURRENCY}")
    print(f"[INFO] 截止时间(cutoff): {CUTOFF_TIME}")
    
    # 诊断：统计有多少用户有截止时间前的数据
    users_with_strict_pre = 0
    users_no_pre = 0
    users_no_data = 0
    for u in users_data:
        all_times = []
        for p in u.get('original_posts', []):
            all_times.append(p.get('time', ''))
        for p in u.get('repost_posts', []):
            all_times.append(p.get('time', ''))
        for c in u.get('comments', []):
            all_times.append(c.get('time', ''))
        
        has_pre = any(is_before_cutoff(t) for t in all_times if t)
        if has_pre:
            users_with_strict_pre += 1
        elif all_times:
            users_no_pre += 1
        else:
            users_no_data += 1
    
    print(f"[DIAG] 有cutoff前历史博文的用户: {users_with_strict_pre} (将使用行为推断信念)")
    print(f"[DIAG] 无cutoff前数据的用户: {users_no_pre} (将使用采样预设信念，避免数据泄露)")
    print(f"[DIAG] 完全无博文数据的用户: {users_no_data}")
    
    # 输出文件（使用.jsonl临时文件）
    final_output = os.path.join(config['paths']['output_dir'], config['paths']['users_file'])
    output_file = final_output + 'l'  # users.jsonl
    
    # 加载已处理的用户
    processed_users, existing_users = load_processed_users(output_file)
    remaining = len(users_data) - len(processed_users)
    print(f"[INFO] 已处理用户: {len(processed_users)}，剩余: {remaining}")
    
    if remaining == 0:
        print("[INFO] 所有用户已处理完成，转换为JSON格式...")
        with open(final_output, 'w', encoding='utf-8') as f:
            json.dump(existing_users, f, ensure_ascii=False, indent=2)
        print(f"[INFO] 已保存到 {final_output}")
        return
    
    # 进度计数器
    progress_counter = {'count': len(processed_users), 'lock': threading.Lock()}
    
    # 信号量控制并发
    semaphore = asyncio.Semaphore(CONCURRENCY)
    
    # 创建所有任务
    tasks = [
        process_user(user, influence_data, output_file, processed_users, 
                    semaphore, progress_counter, len(users_data))
        for user in users_data
    ]
    
    # 并发执行
    await asyncio.gather(*tasks)
    
    print(f"[INFO] 全部处理完成，转换为JSON格式...")
    
    # 从JSONL重新加载并转换为JSON
    _, all_results = load_processed_users(output_file)
    with open(final_output, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 已保存到 {final_output}")
    
    # 统计信息
    type_counts = defaultdict(int)
    for u in all_results:
        type_counts[u['agent_type']] += 1
    
    print(f"[STATS] 总用户数: {len(all_results)}")
    print("[STATS] 用户类型分布:")
    for t, c in type_counts.items():
        print(f"  - {t}: {c}")


if __name__ == "__main__":
    asyncio.run(main())
