"""
Step1: 数据提取与清洗
- 从MongoDB读取天价耳环事件数据
- 筛选时间范围内的用户和博文
- 筛选总行为数>=min_total_activities的用户
- 提取关键字段并清洗文本
- 构建转发链和评论链
- 保存基础数据到JSON
"""
import json
import re
import os
from datetime import datetime
from pymongo import MongoClient
from collections import defaultdict
from tqdm import tqdm

# 加载配置
with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

# 时间配置
START_TIME = datetime.strptime(config['filter']['start_time'], "%Y-%m-%d %H:%M:%S")
CUTOFF_TIME = datetime.strptime(config['filter']['cutoff_time'], "%Y-%m-%d %H:%M:%S")
END_TIME = datetime.strptime(config['filter']['end_time'], "%Y-%m-%d %H:%M:%S")

# 输出配置
output_dir = config['paths']['output_dir']
output_file = os.path.join(output_dir, config['paths']['base_data_file'])

# MongoDB连接
client = MongoClient(config['database']['mongodb_uri'])
db = client[config['database']['name']]
collection = db[config['database']['collection']]


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


def is_in_time_range(time_str):
    """判断时间是否在[start_time, end_time]范围内"""
    dt = parse_time(time_str)
    if dt is None:
        return False
    return START_TIME <= dt <= END_TIME


def clean_text(text):
    """清洗文本内容"""
    if not text:
        return ""
    # 去除末尾的多余问号
    text = re.sub(r'\?{2,}$', '', text)
    text = re.sub(r'？{2,}$', '', text)
    # 去除多余空白
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def parse_repost_chain(full_content, root_author):
    """
    解析转发链
    格式: 用户内容//@用户A:内容A//@用户B:内容B
    返回转发链列表，从根博文到当前用户
    """
    if not full_content:
        return []
    
    chain = []
    # 按//@分割
    parts = re.split(r'//@', full_content)
    
    if len(parts) == 1:
        # 无转发链，直接转发根博文
        return [{"author": root_author, "content": "", "level": 0}]
    
    # 第一部分是当前用户的转发内容
    current_content = parts[0].strip()
    
    # 后续部分是转发链
    for i, part in enumerate(parts[1:], 1):
        match = re.match(r'^([^:：]+)[:：](.*)$', part.strip())
        if match:
            author = match.group(1).strip()
            content = clean_text(match.group(2).strip())
            chain.append({
                "author": author,
                "content": content,
                "level": len(parts) - i
            })
    
    # 根博文作者
    chain.append({"author": root_author, "content": "", "level": 0})
    
    return chain


def parse_comment_level(comment):
    """
    解析评论层级
    返回: (level, replied_to_user, actual_content)
    level 1: 一级评论（直接评论原博）
    level 2+: 多级评论（回复其他评论）
    """
    original_content = comment.get('原微博内容', '')
    comment_content = comment.get('评论内容', '')
    
    if not original_content:
        # 一级评论
        return 1, None, comment_content
    
    # 检查是否为多级评论（原微博内容中包含@）
    if '@' in original_content:
        level = 3  # 多级评论
    else:
        level = 2  # 二级评论
    
    # 提取回复对象
    match = re.match(r'^回复@([^:：]+)[:：](.*)$', comment_content)
    if match:
        replied_to = match.group(1).strip()
        actual_content = clean_text(match.group(2).strip())
    else:
        replied_to = None
        actual_content = clean_text(comment_content)
    
    return level, replied_to, actual_content


def extract_user_info(user_info):
    """提取用户基本信息"""
    return {
        "user_id": str(user_info.get('作者ID', '')),
        "username": clean_text(user_info.get('原文作者', '').strip("'")),
        "followers_count": user_info.get('粉丝数', 0),
        "following_count": user_info.get('用户关注数', 0),
        "posts_count": user_info.get('微博数', 0),
        "verified_type": user_info.get('认证类型', '普通用户'),
        "gender": user_info.get('性别', '未知'),
        "location": user_info.get('信源地域', '未知'),
        "register_location": user_info.get('用户注册地', '未知'),
        "register_time": user_info.get('注册时间', ''),
        "description": clean_text(user_info.get('用户简介', ''))
    }


def extract_original_post(post):
    """提取原创博文信息"""
    return {
        "type": "original",
        "content": clean_text(post.get('标题／微博内容', '') or post.get('全文内容', '')),
        "time": post.get('日期', ''),
        "reposts": post.get('转发数', 0),
        "comments": post.get('评论数', 0),
        "likes": post.get('点赞数', 0),
        "url": post.get('原文/评论链接', ''),
        "emotion": post.get('微博情绪', '中性'),
        "sensitivity": post.get('信息属性', '非敏感'),
        "keywords": post.get('涉及词', '')
    }


def extract_repost(post):
    """提取转发博文信息"""
    full_content = post.get('标题／微博内容', '') or post.get('全文内容', '')
    root_author = post.get('根微博作者', '')
    root_content = post.get('原微博内容', '')
    
    # 解析转发链
    repost_chain = parse_repost_chain(full_content, root_author)
    
    # 提取当前用户的转发内容
    parts = re.split(r'//@', full_content)
    user_content = clean_text(parts[0]) if parts else ""
    
    return {
        "type": "repost",
        "user_content": user_content,
        "time": post.get('日期', ''),
        "root_author": root_author,
        "root_content": clean_text(root_content),
        "root_time": post.get('根微博发布时间', ''),
        "repost_chain": repost_chain,
        "url": post.get('原文/评论链接', ''),
        "emotion": post.get('微博情绪', '中性'),
        "sensitivity": post.get('信息属性', '非敏感'),
        "keywords": post.get('涉及词', '')
    }


def extract_comment(comment):
    """提取评论信息"""
    level, replied_to, actual_content = parse_comment_level(comment)
    
    return {
        "type": "comment",
        "level": level,
        "content": actual_content,
        "raw_content": clean_text(comment.get('评论内容', '')),
        "time": comment.get('日期', ''),
        "replied_to_user": replied_to,
        "replied_to_content": clean_text(comment.get('原微博内容', '')),
        "original_post_content": clean_text(comment.get('评论原文内容', '')),
        "original_post_url": comment.get('评论原文链接', ''),
        "original_post_author": clean_text(comment.get('被评论博文作者', '').strip("'")),
        "url": comment.get('原文/评论链接', ''),
        "sensitivity": comment.get('信息属性', '非敏感'),
        "keywords": comment.get('涉及词', '')
    }


def process_user_data(doc):
    """处理单个用户数据"""
    user_info = extract_user_info(doc.get('user_info', {}))
    
    # 提取原创博文
    original_posts = []
    for post in doc.get('original_posts', []):
        original_posts.append(extract_original_post(post))
    
    # 提取转发博文
    repost_posts = []
    for post in doc.get('repost_posts', []):
        repost_posts.append(extract_repost(post))
    
    # 提取评论（只保留一级评论用于initial_posts）
    comments = []
    for comment in doc.get('comments', []):
        comments.append(extract_comment(comment))
    
    stats = doc.get('stats', {})
    
    return {
        "user_key": doc.get('user_key', ''),
        "user_info": user_info,
        "original_posts": original_posts,
        "repost_posts": repost_posts,
        "comments": comments,
        "stats": {
            "original_count": stats.get('original_count', 0),
            "repost_count": stats.get('repost_count', 0),
            "comment_count": stats.get('comment_count', 0),
            "total_activities": stats.get('total_activities', 0)
        }
    }


def main():
    print("=" * 60)
    print("Step1: 数据提取与清洗")
    print("=" * 60)
    print(f"[INFO] 时间范围: {START_TIME} ~ {END_TIME}")
    print(f"[INFO] 截止时间(用于初始博文): {CUTOFF_TIME}")
    
    min_activities = config['filter']['min_total_activities']
    
    # 第一步：获取所有用户数据用于统计时间范围内的行为数
    print("[STEP 1/3] 正在加载所有用户数据...")
    all_docs = list(collection.find({}))
    print(f"[INFO] 数据库中总用户数: {len(all_docs)}")
    
    # 第二步：统计每个用户在时间范围内的行为数，筛选满足条件的用户
    print("[STEP 2/3] 正在筛选时间范围内的活跃用户...")
    eligible_user_ids = set()
    
    for doc in tqdm(all_docs, desc="统计用户行为"):
        total_activities_in_range = 0
        
        # 统计时间范围内的原创博文数
        for post in doc.get('original_posts', []):
            if is_in_time_range(post.get('日期', '')):
                total_activities_in_range += 1
        
        # 统计时间范围内的转发博文数
        for post in doc.get('repost_posts', []):
            if is_in_time_range(post.get('日期', '')):
                total_activities_in_range += 1
        
        # 统计时间范围内的评论数
        for comment in doc.get('comments', []):
            if is_in_time_range(comment.get('日期', '')):
                total_activities_in_range += 1
        
        # 更新用户的stats中的total_activities_in_range
        if 'stats' not in doc:
            doc['stats'] = {}
        doc['stats']['total_activities_in_range'] = total_activities_in_range
        
        if total_activities_in_range >= min_activities:
            eligible_user_ids.add(doc.get('user_key'))
    
    print(f"[INFO] 时间范围内行为数>={min_activities}的用户数: {len(eligible_user_ids)}")
    
    # 第三步：处理符合条件的用户数据
    print("[STEP 3/3] 正在处理用户数据...")
    users_data = []
    
    for doc in tqdm(all_docs, desc="处理用户数据"):
        user_key = doc.get('user_key')
        if user_key not in eligible_user_ids:
            continue
        
        user_data = process_user_data(doc)
        # 添加时间范围内的行为统计
        user_data['stats']['total_activities_in_range'] = doc.get('stats', {}).get('total_activities_in_range', 0)
        users_data.append(user_data)
    
    print(f"[INFO] 数据处理完成，共 {len(users_data)} 个用户")
    
    # 统计信息
    total_originals = sum(u['stats']['original_count'] for u in users_data)
    total_reposts = sum(u['stats']['repost_count'] for u in users_data)
    total_comments = sum(u['stats']['comment_count'] for u in users_data)
    
    print(f"[STATS] 原创博文总数: {total_originals}")
    print(f"[STATS] 转发博文总数: {total_reposts}")
    print(f"[STATS] 评论总数: {total_comments}")
    
    # 保存基础数据
    os.makedirs(output_dir, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(users_data, f, ensure_ascii=False, indent=2)
    
    print(f"[INFO] 基础数据已保存至: {output_file}")
    
    client.close()
    return users_data


if __name__ == "__main__":
    main()
