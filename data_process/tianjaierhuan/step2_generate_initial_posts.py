"""
Step2: 生成initial_posts.json
- 筛选2025.5.16 00:00之前的数据
- 对于评论只保留一级评论
- 对于转发需要构建多级转发链
"""
import json
import os
from datetime import datetime
from collections import defaultdict

# 加载配置
with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

CUTOFF_TIME = datetime.strptime(config['filter']['cutoff_time'], "%Y-%m-%d %H:%M:%S")


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


def build_repost_structure(repost, user_info):
    """构建转发的完整结构"""
    chain = repost.get('repost_chain', [])
    
    # 构建转发层级结构
    repost_levels = []
    for item in chain:
        repost_levels.append({
            "author": item['author'],
            "content": item['content'],
            "level": item['level']
        })
    
    return {
        "type": "repost",
        "author": user_info['username'],
        "author_id": user_info['user_id'],
        "user_content": repost['user_content'],
        "time": repost['time'],
        "root_author": repost['root_author'],
        "root_content": repost['root_content'],
        "root_time": repost['root_time'],
        "repost_chain": repost_levels,
        "url": repost['url'],
        "emotion": repost['emotion']
    }


def build_original_post_structure(post, user_info, comments):
    """构建原创博文结构"""
    # 收集该博文的一级评论
    post_url = post['url']
    related_comments = []
    
    for c in comments:
        # 只保留一级评论
        if c['level'] == 1 and c.get('original_post_url') == post_url:
            related_comments.append(c['content'])
    
    return {
        "type": "original",
        "id": f"post_{user_info['user_id']}_{hash(post['content']) % 100000}",
        "author": user_info['username'],
        "author_id": user_info['user_id'],
        "content": post['content'],
        "time": post['time'],
        "likes": post['likes'],
        "reposts": post['reposts'],
        "comments_count": post['comments'],
        "comments": related_comments[:10],  # 最多保留10条评论
        "emotion": post['emotion'],
        "keywords": post['keywords']
    }


def main():
    print("=" * 60)
    print("Step2: 生成initial_posts.json")
    print("=" * 60)
    
    # # 检查输出文件是否已存在
    # output_file = os.path.join(config['paths']['output_dir'], config['paths']['initial_posts_file'])
    # if os.path.exists(output_file):
    #     print(f"[SKIP] 输出文件已存在: {output_file}")
    #     return None
    
    # 加载基础数据
    base_data_path = os.path.join(config['paths']['output_dir'], config['paths']['base_data_file'])
    with open(base_data_path, 'r', encoding='utf-8') as f:
        users_data = json.load(f)
    
    print(f"[INFO] 加载了 {len(users_data)} 个用户数据")
    print(f"[INFO] 截止时间: {CUTOFF_TIME}")
    
    initial_posts = []
    
    # 用于收集所有用户的评论（用于匹配）
    all_comments = []
    for user in users_data:
        for c in user.get('comments', []):
            if is_before_cutoff(c.get('time', '')):
                all_comments.append(c)
    
    # 处理每个用户的数据
    for user in users_data:
        user_info = user['user_info']
        
        # 处理原创博文
        for post in user.get('original_posts', []):
            if is_before_cutoff(post.get('time', '')):
                post_struct = build_original_post_structure(post, user_info, user.get('comments', []))
                initial_posts.append(post_struct)
        
        # 处理转发博文
        for repost in user.get('repost_posts', []):
            if is_before_cutoff(repost.get('time', '')):
                repost_struct = build_repost_structure(repost, user_info)
                initial_posts.append(repost_struct)
    
    # 按时间排序
    initial_posts.sort(key=lambda x: x.get('time', ''))
    
    print(f"[INFO] 截止时间之前的博文数: {len(initial_posts)}")
    print(f"  - 原创博文: {sum(1 for p in initial_posts if p['type'] == 'original')}")
    print(f"  - 转发博文: {sum(1 for p in initial_posts if p['type'] == 'repost')}")
    
    # 保存数据
    output_file = os.path.join(config['paths']['output_dir'], config['paths']['initial_posts_file'])
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(initial_posts, f, ensure_ascii=False, indent=2)
    
    print(f"[INFO] initial_posts.json 已保存至: {output_file}")
    
    return initial_posts


if __name__ == "__main__":
    main()
