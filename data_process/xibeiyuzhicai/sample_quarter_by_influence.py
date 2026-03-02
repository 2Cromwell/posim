#!/usr/bin/env python3
"""
从 base_data.json 中按用户抽取「高质量的四分之一」博文（原创+转发+评论），
每个用户只保留其博文数量 1/4 的帖子，按影响力排序取 top 25%。
输出: output/base_data_quarter.json
"""
import json
import math
from pathlib import Path

def influence_score(item: dict, post_type: str) -> float:
    """计算单条博文/转发/评论的影响力分数，用于排序取高质量 1/4。"""
    if post_type == "original":
        reposts = item.get("reposts") or 0
        comments = item.get("comments") or 0
        likes = item.get("likes") or 0
        return 1.0 + reposts + comments + likes
    if post_type == "repost":
        return 1.0
    if post_type == "comment":
        return 1.0
    return 1.0

def run(
    input_path: str = "output/base_data.json",
    output_path: str = "output/base_data_quarter.json",
    quarter_ratio: float = 0.25,
):
    base = Path(__file__).resolve().parent
    input_file = base / input_path
    output_file = base / output_path

    print(f"Loading {input_file} ...")
    with open(input_file, "r", encoding="utf-8") as f:
        users = json.load(f)

    if not isinstance(users, list):
        raise ValueError("base_data.json root must be a list of user objects")

    result = []
    total_before = 0
    total_after = 0

    for u in users:
        user_key = u.get("user_key", "")
        user_info = u.get("user_info", {})
        original_posts = list(u.get("original_posts") or [])
        repost_posts = list(u.get("repost_posts") or [])
        comments = list(u.get("comments") or [])

        # 合并为 (type, list_name, index, score)
        candidates = []
        for i, p in enumerate(original_posts):
            if p.get("type") == "original":
                candidates.append(("original", "original_posts", i, influence_score(p, "original")))
        for i, p in enumerate(repost_posts):
            candidates.append(("repost", "repost_posts", i, influence_score(p, "repost")))
        for i, p in enumerate(comments):
            candidates.append(("comment", "comments", i, influence_score(p, "comment")))

        n = len(candidates)
        total_before += n

        if n == 0:
            result.append({
                "user_key": user_key,
                "user_info": user_info,
                "original_posts": [],
                "repost_posts": [],
                "comments": [],
                "stats": u.get("stats") or {},
            })
            continue

        # 按影响力降序，取前 ceil(n * quarter_ratio) 条（至少 1 条）
        keep_count = max(1, math.ceil(n * quarter_ratio))
        sorted_candidates = sorted(candidates, key=lambda x: -x[3])
        kept = sorted_candidates[:keep_count]

        # 按原列表归属收集被选中的 index
        kept_original = sorted([c[2] for c in kept if c[1] == "original_posts"])
        kept_repost = sorted([c[2] for c in kept if c[1] == "repost_posts"])
        kept_comment = sorted([c[2] for c in kept if c[1] == "comments"])

        new_original = [original_posts[i] for i in kept_original]
        new_repost = [repost_posts[i] for i in kept_repost]
        new_comment = [comments[i] for i in kept_comment]

        total_after += len(new_original) + len(new_repost) + len(new_comment)

        stats = dict(u.get("stats") or {})
        stats["original_count"] = len(new_original)
        stats["repost_count"] = len(new_repost)
        stats["comment_count"] = len(new_comment)
        stats["total_activities"] = len(new_original) + len(new_repost) + len(new_comment)

        result.append({
            "user_key": user_key,
            "user_info": user_info,
            "original_posts": new_original,
            "repost_posts": new_repost,
            "comments": new_comment,
            "stats": stats,
        })

    print(f"Total activities before: {total_before}, after: {total_after} ({100*total_after/max(1,total_before):.1f}%)")
    print(f"Writing {output_file} ...")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("Done.")
    return output_file

if __name__ == "__main__":
    run()
