"""
Step3: 生成relations.json
- 基于用户影响力综合计算
- 使用Barabási-Albert优先连接模型生成社交网络
- 结合用户影响力进行关注关系生成
"""
import json
import os
import random
import math
import numpy as np
from collections import defaultdict

# 加载配置
with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

random.seed(42)
np.random.seed(42)


def calculate_user_influence(user_info, stats):
    """
    计算用户综合影响力
    
    影响力 = w1 * log(粉丝数+1) + w2 * log(发博数+1) + w3 * 认证加成 + w4 * 互动率
    
    - 粉丝数: 直接反映用户的社交影响范围
    - 发博数: 反映用户活跃程度
    - 认证类型: 认证用户通常更有影响力
    - 互动率: 用户发布内容获得的平均互动
    """
    weights = config['influence_weights']
    
    # 确保数值类型正确
    followers = int(user_info.get('followers_count', 0) or 0)
    posts = int(user_info.get('posts_count', 0) or 0)
    verified_type = str(user_info.get('verified_type', '普通用户') or '普通用户')
    
    # 粉丝数得分（对数归一化）
    followers_score = math.log(followers + 1) / math.log(10000000)  # 假设最大1000万粉丝
    
    # 发博数得分
    posts_score = math.log(posts + 1) / math.log(100000)  # 假设最大10万博文
    
    # 认证加成
    verified_score = 0.0
    if '蓝V' in verified_type or '机构' in verified_type:
        verified_score = 1.0
    elif '橙V' in verified_type or '个人认证' in verified_type:
        verified_score = 0.8
    elif '黄V' in verified_type or '会员' in verified_type:
        verified_score = 0.5
    
    # 互动率得分（基于stats中的数据估算）
    total_activities = stats.get('total_activities', 0)
    engagement_score = min(1.0, math.log(total_activities + 1) / math.log(1000))
    
    # 综合影响力
    influence = (
        weights['followers'] * followers_score +
        weights['posts'] * posts_score +
        weights['verified'] * verified_score +
        weights['engagement'] * engagement_score
    )
    
    return min(1.0, max(0.01, influence))


def determine_agent_type(user_info, influence):
    """
    根据用户信息和影响力确定agent类型
    - kol: 意见领袖（高粉丝、认证用户）
    - media: 媒体（机构认证）
    - government: 政府（政府认证）
    - citizen: 普通网民
    """
    verified_type = str(user_info.get('verified_type', '普通用户') or '普通用户')
    followers = int(user_info.get('followers_count', 0) or 0)
    description = str(user_info.get('description', '') or '')
    
    # 政府账号
    gov_keywords = ['发布', '政府', '公安', '纪委', '官方', '市委', '省委', '政务']
    if any(kw in verified_type or kw in description for kw in gov_keywords):
        return 'government'
    
    # 媒体账号
    media_keywords = ['媒体', '新闻', '日报', '晚报', '电视台', '广播', '频道', '报社']
    if '蓝V' in verified_type or any(kw in verified_type or kw in description for kw in media_keywords):
        return 'media'
    
    # 意见领袖（高影响力个人）
    if followers >= 100000 or (followers >= 50000 and '认证' in verified_type):
        return 'kol'
    
    # 普通网民
    return 'citizen'


def generate_social_network(users_with_influence):
    """
    使用改进的Barabási-Albert模型生成社交网络
    
    核心思想：优先连接 + 影响力权重
    - 新节点倾向于连接高影响力节点
    - 考虑用户类型的异质性（普通用户关注KOL/媒体的概率更高）
    """
    import time
    n = len(users_with_influence)
    relations = []
    
    # 按影响力排序
    sorted_users = sorted(users_with_influence, key=lambda x: x['influence'], reverse=True)
    user_ids = [u['user_id'] for u in sorted_users]
    influences = {u['user_id']: u['influence'] for u in sorted_users}
    agent_types = {u['user_id']: u['agent_type'] for u in sorted_users}
    usernames = {u['user_id']: u['username'] for u in sorted_users}
    
    # 构建初始核心网络（前5%高影响力用户相互连接）
    core_size = max(3, int(n * 0.05))
    core_users = user_ids[:core_size]
    
    print(f"[INFO] 核心用户数: {core_size}")
    
    # 生成时间戳基准（使用事件开始时间，从 config 读取）
    base_timestamp = int(time.mktime(time.strptime(config['filter']['start_time'], "%Y-%m-%d %H:%M:%S")))
    
    # 核心用户之间相互关注（形成核心圈）
    for i, u1 in enumerate(core_users):
        for u2 in core_users[i+1:]:
            if random.random() < 0.6:  # 60%概率相互关注
                timestamp = base_timestamp + random.randint(0, 86400 * 3)  # 3天内
                relations.append({
                    "follower_id": u1,
                    "following_id": u2,
                    "relation_type": "follow",
                    "timestamp": timestamp
                })
                if random.random() < 0.5:
                    timestamp = base_timestamp + random.randint(0, 86400 * 3)
                    relations.append({
                        "follower_id": u2,
                        "following_id": u1,
                        "relation_type": "follow",
                        "timestamp": timestamp
                    })
    
    # 为每个用户生成关注关系
    # 平均每人关注 5-15 个用户
    for idx, user in enumerate(sorted_users):
        user_id = user['user_id']
        user_type = user['agent_type']
        user_influence = user['influence']
        
        # 根据用户类型确定关注数量
        if user_type == 'government':
            follow_count = random.randint(2, 5)  # 政府账号关注较少
        elif user_type == 'media':
            follow_count = random.randint(3, 8)  # 媒体关注适中
        elif user_type == 'kol':
            follow_count = random.randint(5, 15)  # KOL关注较多
        else:
            follow_count = random.randint(3, 12)  # 普通用户
        
        # 候选关注对象（排除自己）
        candidates = [u for u in user_ids if u != user_id]
        
        # 计算关注概率（基于影响力的优先连接）
        probs = []
        for c in candidates:
            c_type = agent_types[c]
            c_influence = influences[c]
            
            # 基础概率与影响力成正比
            prob = c_influence
            
            # 类型偏好调整
            if user_type == 'citizen':
                # 普通用户更倾向于关注KOL和媒体
                if c_type in ['kol', 'media']:
                    prob *= 2.0
                elif c_type == 'government':
                    prob *= 1.5
            elif user_type == 'kol':
                # KOL倾向于关注其他KOL和媒体
                if c_type in ['kol', 'media']:
                    prob *= 1.5
            
            probs.append(prob)
        
        # 归一化概率
        total_prob = sum(probs)
        probs = [p / total_prob for p in probs]
        
        # 采样关注对象
        follow_count = min(follow_count, len(candidates))
        following = np.random.choice(candidates, size=follow_count, replace=False, p=probs)
        
        for f in following:
            # 时间戳：越早注册的用户关注时间越早
            timestamp = base_timestamp + idx * 3600 + random.randint(0, 3600)
            relations.append({
                "follower_id": user_id,
                "following_id": f,
                "relation_type": "follow",
                "timestamp": timestamp
            })
    
    return relations, usernames


def generate_cosmograph_files(relations, users_with_influence, output_dir):
    """
    生成cosmograph可视化所需的TSV文件
    - nodes.tsv: id, label, agent_type, followers_count
    - edges.tsv: source, target, timestamp, relation_type
    """
    import csv
    
    # 创建用户信息字典
    user_info_map = {u['user_id']: u for u in users_with_influence}
    
    # 生成nodes文件
    nodes_file = os.path.join(output_dir, 'follow_nodes.tsv')
    unique_users = set()
    for r in relations:
        unique_users.add(r['follower_id'])
        unique_users.add(r['following_id'])
    
    with open(nodes_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        # 写入表头
        writer.writerow(['id', 'label', 'agent_type', 'followers_count'])
        for user_id in unique_users:
            user_info = user_info_map.get(user_id, {})
            writer.writerow([
                user_id,
                user_info.get('username', user_id),
                user_info.get('agent_type', ''),
                user_info.get('followers_count', 0)
            ])
    print(f"[INFO] 节点文件已保存至: {nodes_file}")
    
    # 生成edges文件
    edges_file = os.path.join(output_dir, 'follow_edges.tsv')
    with open(edges_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        # 写入表头
        writer.writerow(['source', 'target', 'timestamp', 'relation_type'])
        for r in relations:
            writer.writerow([
                r['follower_id'],
                r['following_id'],
                r.get('timestamp', 0),
                r['relation_type']
            ])
    print(f"[INFO] 边文件已保存至: {edges_file}")


def main():
    print("=" * 60)
    print("Step3: 生成relations.json")
    print("=" * 60)
    
    # # 检查输出文件是否已存在
    # output_file = os.path.join(config['paths']['output_dir'], config['paths']['relations_file'])
    # influence_file = os.path.join(config['paths']['output_dir'], 'users_influence.json')
    # if os.path.exists(output_file) and os.path.exists(influence_file):
    #     print(f"[SKIP] 输出文件已存在: {output_file}")
    #     return None
    
    # 加载基础数据
    base_data_path = os.path.join(config['paths']['output_dir'], config['paths']['base_data_file'])
    with open(base_data_path, 'r', encoding='utf-8') as f:
        users_data = json.load(f)
    
    print(f"[INFO] 加载了 {len(users_data)} 个用户数据")
    
    # 计算每个用户的影响力和类型
    users_with_influence = []
    for user in users_data:
        user_info = user['user_info']
        stats = user['stats']
        
        influence = calculate_user_influence(user_info, stats)
        agent_type = determine_agent_type(user_info, influence)
        
        users_with_influence.append({
            "user_id": user_info['user_id'],
            "username": user_info['username'],
            "influence": influence,
            "agent_type": agent_type,
            "followers_count": user_info.get('followers_count', 0),
            "verified_type": user_info.get('verified_type', '普通用户')
        })
    
    # 统计用户类型分布
    type_counts = defaultdict(int)
    for u in users_with_influence:
        type_counts[u['agent_type']] += 1
    
    print("[INFO] 用户类型分布:")
    for t, c in type_counts.items():
        print(f"  - {t}: {c}")
    
    # 生成社交网络
    print("[INFO] 开始生成社交网络...")
    relations, usernames = generate_social_network(users_with_influence)
    
    print(f"[INFO] 生成关注关系数: {len(relations)}")
    print(f"[INFO] 平均每用户关注数: {len(relations) / len(users_with_influence):.2f}")
    
    # 保存关系数据
    output_file = os.path.join(config['paths']['output_dir'], config['paths']['relations_file'])
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(relations, f, ensure_ascii=False, indent=2)
    
    print(f"[INFO] relations.json 已保存至: {output_file}")
    
    # 同时保存用户影响力数据（供后续使用）
    influence_file = os.path.join(config['paths']['output_dir'], 'users_influence.json')
    with open(influence_file, 'w', encoding='utf-8') as f:
        json.dump(users_with_influence, f, ensure_ascii=False, indent=2)
    
    print(f"[INFO] 用户影响力数据已保存至: {influence_file}")
    
    # 生成cosmograph可视化文件
    print("[INFO] 开始生成cosmograph可视化文件...")
    generate_cosmograph_files(relations, users_with_influence, config['paths']['output_dir'])
    
    return relations


if __name__ == "__main__":
    main()
