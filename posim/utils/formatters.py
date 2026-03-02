"""
信息格式化工具 - 封装EBDI系统中的通用格式化逻辑
包含：外部事件格式化、曝光博文格式化、记忆格式化等
"""
from datetime import datetime
from typing import List, Dict, Optional


def calculate_time_diff_str(event_time: str, current_time: str) -> str:
    """
    计算事件距离当前时间的差值，返回中文描述字符串
    
    Args:
        event_time: 事件时间 (ISO格式)
        current_time: 当前时间 (ISO格式)
    
    Returns:
        时间差描述字符串，如 "【3小时20分钟前】"
    """
    if not event_time or not current_time:
        return ""
    
    try:
        evt_dt = datetime.fromisoformat(event_time)
        cur_dt = datetime.fromisoformat(current_time)
        diff_minutes = int((cur_dt - evt_dt).total_seconds() / 60)
        
        if diff_minutes < 0:
            return "（尚未发生）"
        elif diff_minutes == 0:
            return "【刚刚发生】"
        elif diff_minutes < 60:
            return f"【{diff_minutes}分钟前】"
        elif diff_minutes < 1440:  # 24小时内
            hours = diff_minutes // 60
            mins = diff_minutes % 60
            if mins > 0:
                return f"【{hours}小时{mins}分钟前】"
            else:
                return f"【{hours}小时前】"
        else:
            days = diff_minutes // 1440
            hours = (diff_minutes % 1440) // 60
            if hours > 0:
                return f"【{days}天{hours}小时前】"
            else:
                return f"【{days}天前】"
    except ValueError:
        return ""


def format_external_events(events: List[Dict], current_time: str, 
                           max_events: int = 5, include_header: bool = True) -> str:
    """
    格式化外部突发事件列表
    
    Args:
        events: 事件列表 (按时间排序，旧的在前)
        current_time: 当前仿真时间
        max_events: 最多显示的事件数量
        include_header: 是否包含标题
    
    Returns:
        格式化后的事件文本
    """
    if not events:
        return ""
    
    parts = []
    if include_header:
        parts.append("### 【重要的关注点】近期发生的突发事件：")
        parts.append("")
    
    # 取最新的事件（事件列表按时间排序，旧的在前，新的在后）
    recent_events = events[-max_events:] if len(events) > max_events else events
    
    for evt in recent_events:
        evt_time = evt.get('time', '')
        evt_content = evt.get('content', '')
        evt_stage = evt.get('metadata', {}).get('event_stage', '')
        
        time_diff_str = calculate_time_diff_str(evt_time, current_time)
        stage_str = f"[{evt_stage}]" if evt_stage else ""
        parts.append(f"- {time_diff_str} {stage_str} {evt_content}")
    
    return "\n".join(parts)


def format_exposed_posts(posts: List[Dict], current_time: str = None,
                         max_posts: int = 10, include_header: bool = True,
                         include_stats: bool = True, include_comments: bool = True) -> str:
    """
    格式化曝光的博文列表
    
    Args:
        posts: 博文列表
        current_time: 当前时间（用于计算时间差，可选）
        max_posts: 最多显示的博文数量
        include_header: 是否包含标题
        include_stats: 是否包含互动统计（点赞、转发、评论数）
        include_comments: 是否包含热门评论
    
    Returns:
        格式化后的博文文本
    """
    if not posts:
        return ""
    
    parts = []
    if include_header:
        parts.append("### 曝光的博文：")
    
    for i, post in enumerate(posts[:max_posts], 1):
        post_time = post.get('time', '')
        time_prefix = f"[{post_time}] " if post_time else ""
        author = post.get('author', '未知')
        post_type = post.get('type', 'original')
        
        # 根据博文类型格式化（标准字段: root_author, root_content, content）
        root_author = post.get('root_author', '')
        root_content = (post.get('root_content', '') or '')[:150]
        content = (post.get('content', '') or '')[:200]
        
        if post_type in ['repost', 'repost_comment']:
            parts.append(f"{i}. {time_prefix}@{author} 转发了博文：")
            if root_author:
                parts.append(f"   原博 @{root_author}：{root_content}")
            # 转发评论（content 即为转发时的评论）
            if post_type == 'repost_comment' and content and content != '转发微博':
                parts.append(f"   转发评论：{content[:100]}")
        else:
            # 原创博文
            parts.append(f"{i}. {time_prefix}@{author} 发表了博文：{content}")
        
        # 添加互动统计
        if include_stats:
            likes = post.get('likes', 0)
            reposts = post.get('reposts', 0)
            comments_count = post.get('comments_count', len(post.get('comments', [])))
            parts.append(f"   [点赞:{likes} 转发:{reposts} 评论:{comments_count}]")
        
        # 添加热门评论
        if include_comments and post.get('comments'):
            comments = post['comments'][:3]
            if comments:
                parts.append("   热门评论：" + " | ".join(str(c) for c in comments))
    
    return "\n".join(parts)


def format_memories(memories: List[Dict], max_memories: int = 10) -> str:
    """
    格式化历史行为记忆
    
    Args:
        memories: 记忆列表
        max_memories: 最多显示的记忆数量
    
    Returns:
        格式化后的记忆文本
    """
    if not memories:
        return ""
    
    lines = []
    for mem in memories[:max_memories]:
        time_str = mem.get('time', '')
        action_type = mem.get('action_type', '')
        content = mem.get('content', '')
        target = mem.get('target', '')
        
        # 构建带目标的行为描述
        if action_type in ['repost', 'repost_comment']:
            if target:
                desc = f"我转发并评论了：{content[:50]}..."
            else:
                desc = f"我转发了一条博文"
        elif action_type in ['short_comment', 'long_comment']:
            if target:
                desc = f"我评论了博文《{target[:30]}...》：{content[:50]}..."
            else:
                desc = f"我发表了评论：{content[:50]}..."
        elif action_type == 'like':
            if target:
                desc = f"我点赞了博文《{target[:30]}...》"
            else:
                desc = f"我点赞了一条博文"
        elif action_type in ['short_post', 'long_post']:
            desc = f"我发布了原创博文：{content[:50]}..."
        else:
            desc = f"{content[:80]}..."
        
        lines.append(f"- {desc}")
    
    return "\n".join(lines)


def format_hot_topics(hot_topics: str) -> str:
    """
    格式化热搜话题文本
    
    Args:
        hot_topics: 热搜话题字符串
    
    Returns:
        格式化后的热搜话题文本（带标题）
    """
    if not hot_topics:
        return ""
    return f"\n### 当前热搜话题\n{hot_topics}\n"
