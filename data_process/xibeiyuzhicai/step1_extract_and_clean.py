"""
Step1: 数据提取与清洗
- 从MongoDB读取西贝预制菜事件数据
- 筛选时间范围内的用户和博文
- 对政府/媒体用户降低活跃度门槛（special_role_activity_ratio）
- 博文级过滤：广告/无关内容关键词、长度过滤、去重、纯转发过滤
- 提取关键字段并清洗文本
- 构建转发链和评论链
- 保存基础数据到JSON
"""
import json
import re
import os
import hashlib
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

# 用户筛选配置
MIN_ACTIVITIES = config['filter']['min_total_activities']
SPECIAL_ROLE_RATIO = config['filter'].get('special_role_activity_ratio', 0.25)
SPECIAL_ROLE_MIN = max(1, int(MIN_ACTIVITIES * SPECIAL_ROLE_RATIO))

# 博文过滤配置
POST_FILTER = config['filter'].get('post_filter', {})
POST_FILTER_ENABLED = POST_FILTER.get('enabled', False)
MIN_ORIGINAL_LEN = POST_FILTER.get('min_original_length', 15)
MIN_REPOST_LEN = POST_FILTER.get('min_repost_content_length', 5)
MIN_COMMENT_LEN = POST_FILTER.get('min_comment_length', 5)
REMOVE_DUPLICATES = POST_FILTER.get('remove_duplicates', True)
REMOVE_PURE_REPOST = POST_FILTER.get('remove_pure_repost', True)
AD_KEYWORDS = POST_FILTER.get('ad_keywords', [])
IRRELEVANT_KEYWORDS = POST_FILTER.get('irrelevant_keywords', [])

# 输出配置
output_dir = config['paths']['output_dir']
output_file = os.path.join(output_dir, config['paths']['base_data_file'])

# MongoDB连接
client = MongoClient(config['database']['mongodb_uri'])
db = client[config['database']['name']]
collection = db[config['database']['collection']]


# ===== 用户角色预判断（用于降低政府/媒体门槛） =====
GOV_KEYWORDS = ['发布', '政府', '公安', '纪委', '官方', '市委', '省委', '政务']
MEDIA_KEYWORDS = ['媒体', '新闻', '日报', '晚报', '电视台', '广播', '频道', '报社']


def guess_user_role(doc):
    """根据MongoDB原始数据快速判断用户角色（用于分级筛选）"""
    user_info = doc.get('user_info', {})
    verified_type = str(user_info.get('认证类型', '普通用户') or '普通用户')
    description = str(user_info.get('用户简介', '') or '')
    followers = int(user_info.get('粉丝数', 0) or 0)

    if any(kw in verified_type or kw in description for kw in GOV_KEYWORDS):
        return 'government'
    if '蓝V' in verified_type or any(kw in verified_type or kw in description for kw in MEDIA_KEYWORDS):
        return 'media'
    if followers >= 100000 or (followers >= 50000 and '认证' in verified_type):
        return 'kol'
    return 'citizen'


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


def _build_keyword_pattern():
    """构建广告和无关内容的正则匹配模式"""
    all_kw = AD_KEYWORDS + IRRELEVANT_KEYWORDS
    if not all_kw:
        return None
    escaped = [re.escape(kw) for kw in all_kw]
    return re.compile('|'.join(escaped))


_KW_PATTERN = _build_keyword_pattern() if POST_FILTER_ENABLED else None


def _get_text_for_check(text):
    """获取用于过滤检查的纯净文本（去除URL、话题标签、@引用）"""
    if not text:
        return ""
    t = re.sub(r'https?://\S+', '', text)
    t = re.sub(r'#.*?#', '', t)
    t = re.sub(r'@\S+[\s:：]?', '', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def _content_hash(text):
    """计算文本内容的hash用于去重"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def filter_posts(user_data):
    """
    博文级过滤：
    1. 广告/无关内容关键词过滤
    2. 长度过滤（原创/转发/评论分别设阈值）
    3. 纯转发过滤（转发内容为空的）
    4. 内容去重
    返回过滤后的user_data和过滤统计
    """
    if not POST_FILTER_ENABLED:
        return user_data, {}

    stats = {'original_removed': 0, 'repost_removed': 0, 'comment_removed': 0,
             'ad_removed': 0, 'short_removed': 0, 'dup_removed': 0, 'pure_repost_removed': 0}
    seen_hashes = set()

    def is_ad_or_irrelevant(text):
        if _KW_PATTERN and _KW_PATTERN.search(text):
            return True
        return False

    def is_duplicate(text):
        if not REMOVE_DUPLICATES:
            return False
        h = _content_hash(text)
        if h in seen_hashes:
            return True
        seen_hashes.add(h)
        return False

    # 原创博文：广告/无关、长度、去重
    filtered_originals = []
    for post in user_data.get('original_posts', []):
        content = post.get('content', '')
        if is_ad_or_irrelevant(content):
            stats['ad_removed'] += 1
            stats['original_removed'] += 1
            continue
        check_text = _get_text_for_check(content)
        if len(check_text) < MIN_ORIGINAL_LEN:
            stats['short_removed'] += 1
            stats['original_removed'] += 1
            continue
        if is_duplicate(check_text):
            stats['dup_removed'] += 1
            stats['original_removed'] += 1
            continue
        filtered_originals.append(post)

    # 过滤转发博文
    filtered_reposts = []
    for post in user_data.get('repost_posts', []):
        user_content = post.get('user_content', '')
        root_content = post.get('root_content', '')
        # 纯转发（用户无附加内容）
        if REMOVE_PURE_REPOST and not user_content.strip():
            stats['pure_repost_removed'] += 1
            stats['repost_removed'] += 1
            continue
        combined = user_content + ' ' + root_content
        # if is_ad_or_irrelevant(combined):
        #     stats['ad_removed'] += 1
        #     stats['repost_removed'] += 1
        #     continue
        # check_text = _get_text_for_check(user_content)
        # if len(check_text) < MIN_REPOST_LEN:
        #     stats['short_removed'] += 1
        #     stats['repost_removed'] += 1
        #     continue
        # if is_duplicate(check_text):
        #     stats['dup_removed'] += 1
        #     stats['repost_removed'] += 1
        #     continue
        filtered_reposts.append(post)

    # 过滤评论
    filtered_comments = []
    for comment in user_data.get('comments', []):
        content = comment.get('content', '') or comment.get('raw_content', '')
        if is_ad_or_irrelevant(content):
            stats['ad_removed'] += 1
            stats['comment_removed'] += 1
            continue
        check_text = _get_text_for_check(content)
        if len(check_text) < MIN_COMMENT_LEN:
            stats['short_removed'] += 1
            stats['comment_removed'] += 1
            continue
        if is_duplicate(check_text):
            stats['dup_removed'] += 1
            stats['comment_removed'] += 1
            continue
        filtered_comments.append(comment)

    user_data['original_posts'] = filtered_originals
    user_data['repost_posts'] = filtered_reposts
    user_data['comments'] = filtered_comments
    user_data['stats']['original_count'] = len(filtered_originals)
    user_data['stats']['repost_count'] = len(filtered_reposts)
    user_data['stats']['comment_count'] = len(filtered_comments)
    user_data['stats']['total_activities'] = len(filtered_originals) + len(filtered_reposts) + len(filtered_comments)

    return user_data, stats


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
    print(f"[INFO] 普通用户最低活跃度: {MIN_ACTIVITIES}")
    print(f"[INFO] 政府/媒体最低活跃度: {SPECIAL_ROLE_MIN} (比例={SPECIAL_ROLE_RATIO})")
    if POST_FILTER_ENABLED:
        print(f"[INFO] 博文过滤已启用: 原创>={MIN_ORIGINAL_LEN}字, 转发>={MIN_REPOST_LEN}字, 评论>={MIN_COMMENT_LEN}字")
        print(f"[INFO] 广告关键词: {len(AD_KEYWORDS)}个, 无关关键词: {len(IRRELEVANT_KEYWORDS)}个")
        print(f"[INFO] 去重={REMOVE_DUPLICATES}, 过滤纯转发={REMOVE_PURE_REPOST}")
    
    # 第一步：获取所有用户数据用于统计时间范围内的行为数
    print("[STEP 1/4] 正在加载所有用户数据...")
    all_docs = list(collection.find({}))
    print(f"[INFO] 数据库中总用户数: {len(all_docs)}")
    
    # 第二步：统计每个用户在时间范围内的行为数，分级筛选
    print("[STEP 2/4] 正在筛选时间范围内的活跃用户（分级门槛）...")
    eligible_user_ids = set()
    role_counts = {'government': 0, 'media': 0, 'kol': 0, 'citizen': 0}
    role_eligible = {'government': 0, 'media': 0, 'kol': 0, 'citizen': 0}
    
    for doc in tqdm(all_docs, desc="统计用户行为"):
        total_activities_in_range = 0
        
        for post in doc.get('original_posts', []):
            if is_in_time_range(post.get('日期', '')):
                total_activities_in_range += 1
        
        for post in doc.get('repost_posts', []):
            if is_in_time_range(post.get('日期', '')):
                total_activities_in_range += 1
        
        for comment in doc.get('comments', []):
            if is_in_time_range(comment.get('日期', '')):
                total_activities_in_range += 1
        
        if 'stats' not in doc:
            doc['stats'] = {}
        doc['stats']['total_activities_in_range'] = total_activities_in_range
        
        role = guess_user_role(doc)
        role_counts[role] = role_counts.get(role, 0) + 1
        
        # 对政府/媒体使用更低的门槛
        threshold = SPECIAL_ROLE_MIN if role in ('government', 'media') else MIN_ACTIVITIES
        
        if total_activities_in_range >= threshold:
            eligible_user_ids.add(doc.get('user_key'))
            role_eligible[role] = role_eligible.get(role, 0) + 1
    
    print(f"[INFO] 数据库中角色分布: {role_counts}")
    print(f"[INFO] 筛选后角色分布: {role_eligible}")
    print(f"[INFO] 总筛选用户数: {len(eligible_user_ids)}")
    
    # 第三步：处理符合条件的用户数据
    print("[STEP 3/4] 正在处理用户数据...")
    users_data = []
    
    for doc in tqdm(all_docs, desc="处理用户数据"):
        user_key = doc.get('user_key')
        if user_key not in eligible_user_ids:
            continue
        
        user_data = process_user_data(doc)
        user_data['stats']['total_activities_in_range'] = doc.get('stats', {}).get('total_activities_in_range', 0)
        users_data.append(user_data)
    
    # 过滤前统计
    pre_originals = sum(u['stats']['original_count'] for u in users_data)
    pre_reposts = sum(u['stats']['repost_count'] for u in users_data)
    pre_comments = sum(u['stats']['comment_count'] for u in users_data)
    print(f"[INFO] 过滤前: 原创={pre_originals}, 转发={pre_reposts}, 评论={pre_comments}, 合计={pre_originals+pre_reposts+pre_comments}")
    
    # 第四步：博文级过滤
    print("[STEP 4/4] 正在执行博文级过滤...")
    total_filter_stats = defaultdict(int)
    
    for i, user_data in enumerate(tqdm(users_data, desc="过滤博文")):
        users_data[i], fstats = filter_posts(user_data)
        for k, v in fstats.items():
            total_filter_stats[k] += v
    
    if POST_FILTER_ENABLED:
        print(f"[FILTER] 过滤统计（含原创）:")
        print(f"  - 广告/无关内容移除: {total_filter_stats['ad_removed']}")
        print(f"  - 长度不足移除: {total_filter_stats['short_removed']}")
        print(f"  - 重复内容移除: {total_filter_stats['dup_removed']}")
        print(f"  - 纯转发移除: {total_filter_stats['pure_repost_removed']}")
        print(f"  - 原创移除合计: {total_filter_stats['original_removed']}")
        print(f"  - 转发移除合计: {total_filter_stats['repost_removed']}")
        print(f"  - 评论移除合计: {total_filter_stats['comment_removed']}")
    
    # 过滤后统计
    total_originals = sum(u['stats']['original_count'] for u in users_data)
    total_reposts = sum(u['stats']['repost_count'] for u in users_data)
    total_comments = sum(u['stats']['comment_count'] for u in users_data)
    total_all = total_originals + total_reposts + total_comments
    
    print(f"[INFO] 数据处理完成，共 {len(users_data)} 个用户")
    print(f"[STATS] 过滤后: 原创={total_originals}, 转发={total_reposts}, 评论={total_comments}, 合计={total_all}")
    if pre_originals + pre_reposts + pre_comments > 0:
        reduction = 1 - total_all / (pre_originals + pre_reposts + pre_comments)
        print(f"[STATS] 数据量缩减比例: {reduction:.1%}")
    
    # 保存基础数据
    os.makedirs(output_dir, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(users_data, f, ensure_ascii=False, indent=2)
    
    print(f"[INFO] 基础数据已保存至: {output_file}")
    
    client.close()
    return users_data


if __name__ == "__main__":
    main()
