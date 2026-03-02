"""
Step2: 生成initial_posts.json
- 筛选时间范围内的数据
- 对于评论：
  - 一级评论直接保留
  - 高级评论(level>1)：如果其上一级用户(replied_to_user)或根级用户(original_post_author)在用户列表中，也保留
- 对于转发需要构建多级转发链
- 生成cosmograph格式的转发关系TSV文件
"""
import json
import os
import csv
import time
from datetime import datetime
from collections import defaultdict

# 加载配置
with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

CUTOFF_TIME = datetime.strptime(config['filter']['cutoff_time'], "%Y-%m-%d %H:%M:%S")
START_TIME = datetime.strptime(config['filter']['start_time'], "%Y-%m-%d %H:%M:%S")
END_TIME = datetime.strptime(config['filter']['end_time'], "%Y-%m-%d %H:%M:%S")


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


def build_comment_structure(comment, user_info):
    """构建评论的完整结构"""
    return {
        "type": "comment",
        "author": user_info['username'],
        "author_id": user_info['user_id'],
        "content": comment['content'],
        "raw_content": comment.get('raw_content', ''),
        "time": comment['time'],
        "level": comment['level'],
        "replied_to_user": comment.get('replied_to_user'),
        "replied_to_content": comment.get('replied_to_content', ''),
        "original_post_content": comment.get('original_post_content', ''),
        "original_post_url": comment.get('original_post_url', ''),
        "original_post_author": comment.get('original_post_author', ''),
        "url": comment.get('url', ''),
        "sensitivity": comment.get('sensitivity', ''),
        "keywords": comment.get('keywords', '')
    }


def is_valid_comment(comment, all_usernames):
    """
    判断评论是否有效：
    - 一级评论直接有效
    - 高级评论(level>1)：如果其上一级用户或根级用户在用户列表中，也有效
    """
    if comment['level'] == 1:
        return True
    
    # 对于高级评论，检查上级用户或根级用户是否在列表中
    replied_to_user = comment.get('replied_to_user')
    original_post_author = comment.get('original_post_author')
    
    return replied_to_user in all_usernames or original_post_author in all_usernames


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
        "comments": related_comments[:],  
        "emotion": post['emotion'],
        "keywords": post['keywords']
    }


def generate_repost_cosmograph_files(repost_relations, output_dir):
    """
    生成cosmograph可视化所需的转发关系TSV文件
    - nodes.tsv: id, label, type
    - edges.tsv: source, target, timestamp, post_id
    """
    # 生成nodes文件
    nodes_file = os.path.join(output_dir, 'repost_nodes.tsv')
    unique_users = set()
    for r in repost_relations:
        unique_users.add(r['source'])
        unique_users.add(r['target'])
    
    with open(nodes_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['id', 'label', 'type'])
        for user_id in unique_users:
            writer.writerow([user_id, user_id, 'user'])
    print(f"[INFO] 转发节点文件已保存至: {nodes_file}")
    
    # 生成edges文件
    edges_file = os.path.join(output_dir, 'repost_edges.tsv')
    with open(edges_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['source', 'target', 'timestamp', 'post_id'])
        for r in repost_relations:
            writer.writerow([
                r['source'],
                r['target'],
                r.get('timestamp', 0),
                r.get('post_id', '')
            ])
    print(f"[INFO] 转发边文件已保存至: {edges_file}")


def main():
    print("=" * 60)
    print("Step2: 生成initial_posts.json")
    print("=" * 60)
    
    print(f"[INFO] 时间范围: {START_TIME} ~ {END_TIME}")
    print(f"[INFO] 截止时间(用于初始博文): {CUTOFF_TIME}")
    
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
    
    # 收集所有用户名（用于判断评论是否有效）
    all_usernames = set(user['user_info']['username'] for user in users_data)
    print(f"[INFO] 用户名集合大小: {len(all_usernames)}")
    
    initial_posts = []
    comment_stats = {"level1": 0, "higher_level_valid": 0, "higher_level_invalid": 0}
    repost_relations = []  # 存储转发关系用于cosmograph
    
    # 处理每个用户的数据
    for user in users_data:
        user_info = user['user_info']
        user_id = user_info['user_id']
        
        # 处理原创博文
        for post in user.get('original_posts', []):
            post_time = post.get('time', '')
            if is_before_cutoff(post_time):
                post_struct = build_original_post_structure(post, user_info, user.get('comments', []))
                initial_posts.append(post_struct)
        
        # 处理转发博文
        for repost in user.get('repost_posts', []):
            repost_time = repost.get('time', '')
            if is_before_cutoff(repost_time):
                repost_struct = build_repost_structure(repost, user_info)
                initial_posts.append(repost_struct)
                
                # 记录转发关系
                root_author = repost.get('root_author', '')
                if root_author:
                    # 解析时间戳
                    try:
                        dt = datetime.strptime(repost_time, "%Y-%m-%d %H:%M:%S")
                        timestamp = int(time.mktime(dt.timetuple()))
                    except:
                        timestamp = 0
                    
                    repost_relations.append({
                        'source': user_id,
                        'target': root_author,
                        'timestamp': timestamp,
                        'post_id': repost.get('url', '')
                    })
        
        # 处理评论
        for comment in user.get('comments', []):
            if is_before_cutoff(comment.get('time', '')):
                if is_valid_comment(comment, all_usernames):
                    comment_struct = build_comment_structure(comment, user_info)
                    initial_posts.append(comment_struct)
                    # 统计
                    if comment['level'] == 1:
                        comment_stats["level1"] += 1
                    else:
                        comment_stats["higher_level_valid"] += 1
                else:
                    comment_stats["higher_level_invalid"] += 1
    
    # 按时间排序
    initial_posts.sort(key=lambda x: x.get('time', ''))
    
    print(f"[INFO] 截止时间之前的数据统计:")
    print(f"  - 原创博文: {sum(1 for p in initial_posts if p['type'] == 'original')}")
    print(f"  - 转发博文: {sum(1 for p in initial_posts if p['type'] == 'repost')}")
    print(f"  - 评论总数: {comment_stats['level1'] + comment_stats['higher_level_valid']}")
    print(f"    - 一级评论: {comment_stats['level1']}")
    print(f"    - 有效高级评论(上级/根级用户在列表中): {comment_stats['higher_level_valid']}")
    print(f"    - 无效高级评论(已过滤): {comment_stats['higher_level_invalid']}")
    print(f"  - 总数据条数: {len(initial_posts)}")
    
    # 保存数据
    output_file = os.path.join(config['paths']['output_dir'], config['paths']['initial_posts_file'])
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(initial_posts, f, ensure_ascii=False, indent=2)
    
    print(f"[INFO] initial_posts.json 已保存至: {output_file}")
    
    # 生成cosmograph转发关系文件
    print(f"[INFO] 开始生成cosmograph转发关系文件...")
    generate_repost_cosmograph_files(repost_relations, config['paths']['output_dir'])
    print(f"[INFO] 转发关系数: {len(repost_relations)}")
    
    return initial_posts


if __name__ == "__main__":
    main()
