#!/usr/bin/env python3
"""
从 labels.json 中按用户抽取「高质量的四分之一」条记录。
每个用户只保留其条数的 1/4，按简单质量分排序取 top 25%。
输出: output/labels_quarter.json
"""
import json
import re
import math
from pathlib import Path
from collections import defaultdict

def extract_user_id(url: str) -> str:
    """从微博 URL 提取用户 ID"""
    if not url:
        return ""
    m = re.search(r"weibo\.com/(\d+)/", url)
    return m.group(1) if m else ""


def quality_score(entry: dict) -> float:
    """
    单条 label 的质量分，用于排序取高质量 1/4。
    - 内容越长越有信息量
    - 情感极性越极端（远离 0.5）越有态度
    """
    content = entry.get("content") or ""
    length = len(content)
    nlp = entry.get("nlp_sentiment") or {}
    score = float(nlp.get("score", 0.5))
    sentiment_extremity = abs(score - 0.5)  # 0~0.5
    return length + 100.0 * sentiment_extremity


def run(
    input_path: str = "output/labels.json",
    output_path: str = "output/labels_quarter.json",
    quarter_ratio: float = 0.25,
):
    base = Path(__file__).resolve().parent
    input_file = base / input_path
    output_file = base / output_path

    if not input_file.exists():
        print(f"❌ 文件不存在: {input_file}")
        return None

    print(f"Loading {input_file} ...")
    with open(input_file, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        print("❌ labels.json 根节点应为 list")
        return None

    # 按用户分组
    by_user = defaultdict(list)
    for i, r in enumerate(records):
        uid = extract_user_id(r.get("url", ""))
        by_user[uid].append((i, r))

    result = []
    total_before = 0
    total_after = 0

    for uid, items in by_user.items():
        total_before += len(items)
        # 按质量分降序，取前 ceil(n * 0.25) 条
        scored = [(quality_score(r), r) for _, r in items]
        scored.sort(key=lambda x: -x[0])
        keep_count = max(1, math.ceil(len(items) * quarter_ratio))
        kept = [r for _, r in scored[:keep_count]]
        result.extend(kept)
        total_after += len(kept)

    # 按时间排序，保持与原始顺序一致
    def get_time(r):
        t = r.get("time", "")
        return t if t else ""

    result.sort(key=get_time)

    print(f"Total before: {total_before}, after: {total_after} ({100 * total_after / max(1, total_before):.1f}%)")
    print(f"Writing {output_file} ...")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("Done.")
    return output_file


if __name__ == "__main__":
    run()
