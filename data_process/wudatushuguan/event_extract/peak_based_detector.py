# -*- coding: utf-8 -*-
"""
波峰事件检测脚本 - 基于热度曲线波峰进行事件检测

流程：
1. 加载 base_data，按时间颗粒度计算热度曲线
2. 检测波峰（局部最大值）
3. 对每个波峰时段内的博文进行事件检测（复用 event_detector 逻辑）
4. 合并去重后保存为 events.json

运行: python peak_based_detector.py
"""
import asyncio
import json
import os
import sys
from pathlib import Path

# 添加 event_detect_system 以导入 heat_curve、data_loader、influence
_SCRIPT_DIR = Path(__file__).resolve().parent
_EVENT_DETECT_SYSTEM = _SCRIPT_DIR.parent.parent / "systems" / "event_detect_system"
if str(_EVENT_DETECT_SYSTEM) not in sys.path:
    sys.path.insert(0, str(_EVENT_DETECT_SYSTEM))

from core.data_loader import flatten_posts, parse_time
from core.heat_curve import compute_heat_curve
from core.influence import add_influence_to_posts

from datetime import datetime
from event_detector import detect_events_from_posts, load_json, save_json, parse_time as _parse_time


def _parse_event_time(t_str: str) -> datetime | None:
    """解析事件时间字符串"""
    if not t_str:
        return None
    return _parse_time(t_str)


def _merge_events_by_minute(events: list) -> list:
    """
    将相同1分钟时间范围内的事件合并为一条
    时间格式: 2025-07-31T15:03 -> 按分钟分组
    """
    if not events:
        return []

    # 按时间解析并分组（精确到分钟）
    def time_key(evt):
        t = _parse_event_time(evt.get("time", ""))
        if t:
            return t.strftime("%Y-%m-%dT%H:%M")
        return evt.get("time", "")

    groups = {}
    for evt in events:
        key = time_key(evt)
        if key not in groups:
            groups[key] = []
        groups[key].append(evt)

    merged = []
    for key in sorted(groups.keys()):
        group = groups[key]
        if len(group) == 1:
            merged.append(group[0])
            continue

        # 合并多条事件：优先保留 global_broadcast 类型（若有）
        has_gb = any(e.get("type") == "global_broadcast" for e in group)
        first = next((e for e in group if e.get("type") == "global_broadcast"), group[0]).copy()
        first["type"] = "global_broadcast" if has_gb else "node_post"

        sources = set()
        topics = []
        contents = []
        influence_max = first.get("influence", 0)
        merged_source_posts = []
        meta_tags = set()
        meta_groups = []

        for e in group:
            for s in (e.get("source") or []):
                if s:
                    sources.add(s)
            t = e.get("topic", "")
            if t and t not in topics:
                topics.append(t)
            c = e.get("content", "")
            if c and c not in contents:
                contents.append(c)
            influence_max = max(influence_max, e.get("influence", 0))
            if "source_post" in e:
                merged_source_posts.append(e["source_post"])
            for tag in (e.get("metadata") or {}).get("original_tags", []):
                meta_tags.add(tag)
            g = (e.get("metadata") or {}).get("group_name", "")
            if g and g not in meta_groups:
                meta_groups.append(g)

        first["source"] = list(sources) if sources else first.get("source", ["external"])
        first["topic"] = "；".join(topics[:3]) if topics else first.get("topic", "")
        first["content"] = "；".join(contents[:3]) if contents else first.get("content", "")
        first["influence"] = influence_max
        meta = first.get("metadata") or {}
        meta["original_tags"] = list(meta_tags) if meta_tags else meta.get("original_tags", [])
        meta["merged_count"] = len(group)
        if meta_groups:
            meta["group_name"] = "；".join(meta_groups[:3])
        if merged_source_posts:
            meta["merged_source_posts"] = merged_source_posts[:5]  # 最多保留5条
            if "source_post" in first and len(merged_source_posts) > 1:
                del first["source_post"]  # 多条时用 merged_source_posts
        first["metadata"] = meta
        merged.append(first)

    return merged


def _convert_post_to_detector_format(post: dict) -> dict:
    """将 flatten_posts 输出的博文转换为 event_detector 所需格式"""
    t = post.get("time")
    if hasattr(t, "isoformat"):
        time_val = t
    else:
        time_val = parse_time(post.get("time_str", ""))

    return {
        "time": time_val,
        "user_id": post.get("user_id", ""),
        "username": post.get("username", ""),
        "followers": post.get("followers", 0),
        "agent_type": post.get("agent_type", "citizen"),
        "verified": post.get("verified", False),
        "content": post.get("content", "") or post.get("root_content", ""),
        "emotion": post.get("emotion", "中性"),
        "reposts": post.get("reposts", 0),
        "comments": post.get("comments", 0),
        "likes": post.get("likes", 0),
        "influence_score": post.get("influence_score", 0),
        "url": post.get("url", ""),
        "tags": post.get("tags", []),
    }


def _get_event_background(config: dict, script_dir: Path) -> str:
    """获取事件背景描述"""
    event_name = config.get("event_name", "")
    sim_config = config.get("simulation", config)
    event_background = sim_config.get("event_background", "") or config.get("event_background", "")

    if not event_background:
        scripts_config_path = script_dir.parent.parent.parent / "scripts" / "wudatushuguan" / "config.json"
        if scripts_config_path.exists():
            try:
                scripts_cfg = load_json(str(scripts_config_path))
                event_background = scripts_cfg.get("simulation", {}).get("event_background", "")
            except Exception:
                pass

    if event_background and event_name:
        return f"{event_name}：{event_background}"
    return event_name or "舆情事件"


def _get_llm_config(config: dict) -> dict:
    """从配置提取 LLM 参数"""
    llm_cfg = config.get("llm", {})
    if llm_cfg and llm_cfg.get("api_configs"):
        first = next((c for c in llm_cfg["api_configs"] if c.get("enabled")), llm_cfg["api_configs"][0])
        return {
            "base_url": first.get("base_url", "https://api.siliconflow.cn/v1/"),
            "api_key": first.get("api_key", ""),
            "model": first.get("model", "Pro/Qwen/Qwen2.5-7B-Instruct"),
            "temperature": first.get("temperature", 0.3),
            "max_tokens": 8192,
            "concurrency": min(30, llm_cfg.get("concurrency", 30)),
        }
    return {}


async def main():
    print("=" * 80)
    print("波峰事件检测 - 基于热度曲线波峰进行事件检测")
    print("=" * 80)

    script_dir = Path(__file__).resolve().parent
    config_path = script_dir.parent / "config.json"
    config = load_json(str(config_path))

    output_dir = script_dir.parent / config["paths"]["output_dir"]
    base_data_path = output_dir / config["paths"]["base_data_file"]
    output_events_path = output_dir / "events.json"

    if not base_data_path.exists():
        print(f"[ERROR] 数据文件不存在: {base_data_path}")
        return []

    # ========== STEP 1: 加载数据 ==========
    print("\n[STEP 1/4] 加载数据...")
    base_data = load_json(str(base_data_path))
    print(f"[INFO] 加载 {len(base_data)} 条用户数据")

    filter_config = config.get("filter", {})
    filters = {
        "start_time": filter_config.get("start_time", ""),
        "end_time": filter_config.get("end_time", ""),
        "post_types": ["original", "repost", "comment"],
        "user_types": [],
        "tags": [],
    }

    posts = flatten_posts(base_data, filters)
    influence_weights = config.get("influence_weights", {})
    posts = add_influence_to_posts(posts, influence_weights)

    if not posts:
        print("[ERROR] 无符合条件的博文")
        return []

    print(f"[INFO] 扁平化博文 {len(posts)} 条")

    # ========== STEP 2: 计算热度曲线并检测波峰 ==========
    print("\n[STEP 2/4] 计算热度曲线并检测波峰...")
    granularity = "hour"
    result = compute_heat_curve(posts, granularity)
    curve = result.get("curve", [])
    peaks = result.get("peaks", [])[:10]

    print(f"[INFO] 热度曲线 {len(curve)} 个时间桶，检测到 {len(peaks)} 个波峰")
    for i, p in enumerate(peaks[:5], 1):
        print(f"  波峰{i}: {p.get('time')} (博文数: {p.get('count')})")

    if not peaks:
        print("[WARN] 未检测到波峰")
        save_json([], str(output_events_path))
        return []

    # ========== STEP 3: 对每个波峰进行事件检测 ==========
    print("\n[STEP 3/4] 对每个波峰进行事件检测...")
    event_background = _get_event_background(config, script_dir)
    llm_config = _get_llm_config(config)

    all_events = []
    seen_keys = set()

    for i, peak in enumerate(peaks, 1):
        idx = peak.get("index", 0)
        if idx < 0 or idx >= len(curve):
            continue

        bucket = curve[idx]
        peak_posts = bucket.get("posts", [])
        bucket_time = bucket.get("time", "")

        if not peak_posts:
            print(f"  波峰{i} ({bucket_time}): 无博文，跳过")
            continue

        detector_posts = [_convert_post_to_detector_format(p) for p in peak_posts]
        try:
            events = await detect_events_from_posts(
                detector_posts,
                event_background,
                llm_config=llm_config,
                max_tag_groups=10,
            )
        except Exception as e:
            print(f"  波峰{i} ({bucket_time}): 检测失败 - {e}")
            events = []

        for evt in events:
            key = (evt.get("time", ""), (evt.get("content", "") or evt.get("topic", ""))[:50])
            if key not in seen_keys:
                seen_keys.add(key)
                meta = evt.get("metadata") or {}
                meta["detection_method"] = "peak_based"
                meta["peak_time"] = bucket_time
                evt["metadata"] = meta
                all_events.append(evt)

        print(f"  波峰{i} ({bucket_time}): 博文 {len(peak_posts)} 条 -> 检测到 {len(events)} 个事件")

    # ========== STEP 4: 合并1分钟内的事件并保存 ==========
    all_events.sort(key=lambda x: x.get("time", ""))
    merged_events = _merge_events_by_minute(all_events)
    save_json(merged_events, str(output_events_path))

    print("\n[STEP 4/4] 保存结果...")
    print(f"[INFO] 事件已保存至: {output_events_path}")

    # 打印摘要
    print("\n" + "=" * 80)
    print("事件检测结果摘要")
    print("=" * 80)
    print(f"合并前: {len(all_events)} 条 -> 合并后: {len(merged_events)} 条（1分钟内合并）")
    gb_count = sum(1 for e in merged_events if e.get("type") == "global_broadcast")
    np_count = sum(1 for e in merged_events if e.get("type") == "node_post")
    print(f"总事件数: {len(merged_events)}")
    print(f"  - 全局广播 (global_broadcast): {gb_count}")
    print(f"  - 节点发布 (node_post): {np_count}")

    if merged_events:
        print("\n事件列表（前10条）:")
        for e in merged_events[:10]:
            topic = e.get("topic", e.get("content", ""))[:50]
            print(f"  [{e.get('time')}] [{e.get('type')}] {topic}...")

    return merged_events


if __name__ == "__main__":
    asyncio.run(main())
