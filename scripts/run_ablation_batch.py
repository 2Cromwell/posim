# -*- coding: utf-8 -*-
"""
批量运行消融实验脚本
运行 WL (武大图书馆) 和 XF (西贝预制菜) 的 no_ebdi + cot 消融仿真

用法:
  python scripts/run_ablation_batch.py                  # 运行全部4个实验
  python scripts/run_ablation_batch.py --dataset wl     # 只运行WL
  python scripts/run_ablation_batch.py --dataset xf     # 只运行XF
  python scripts/run_ablation_batch.py --mode no_ebdi   # 只运行no_ebdi
  python scripts/run_ablation_batch.py --mode cot       # 只运行cot
"""
import subprocess
import sys
import os
import time
import argparse
from pathlib import Path
from datetime import datetime


SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

EXPERIMENTS = [
    {
        "name": "WL_no_ebdi",
        "dataset": "wl",
        "mode": "no_ebdi",
        "script": SCRIPT_DIR / "wudatushuguan" / "run_with_monitor.py",
        "config": SCRIPT_DIR / "wudatushuguan" / "config_no_ebdi.json",
        "cwd": SCRIPT_DIR / "wudatushuguan",
    },
    {
        "name": "WL_cot",
        "dataset": "wl",
        "mode": "cot",
        "script": SCRIPT_DIR / "wudatushuguan" / "run_with_monitor.py",
        "config": SCRIPT_DIR / "wudatushuguan" / "config_cot.json",
        "cwd": SCRIPT_DIR / "wudatushuguan",
    },
    {
        "name": "XF_no_ebdi",
        "dataset": "xf",
        "mode": "no_ebdi",
        "script": SCRIPT_DIR / "xibeiyuzhicai" / "run_with_monitor.py",
        "config": SCRIPT_DIR / "xibeiyuzhicai" / "config_no_ebdi.json",
        "cwd": SCRIPT_DIR / "xibeiyuzhicai",
    },
    {
        "name": "XF_cot",
        "dataset": "xf",
        "mode": "cot",
        "script": SCRIPT_DIR / "xibeiyuzhicai" / "run_with_monitor.py",
        "config": SCRIPT_DIR / "xibeiyuzhicai" / "config_cot.json",
        "cwd": SCRIPT_DIR / "xibeiyuzhicai",
    },
]


def run_experiment(exp: dict) -> bool:
    """运行单个实验，返回是否成功"""
    print(f"\n{'=' * 70}")
    print(f"  实验: {exp['name']}")
    print(f"  配置: {exp['config']}")
    print(f"  工作目录: {exp['cwd']}")
    print(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 70}\n")

    env = os.environ.copy()
    env.update({
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
        "TORCH_CUDA_ARCH_LIST": "9.0",
        "CUDA_LAUNCH_BLOCKING": "1",
        "CUDA_VISIBLE_DEVICES": "0",
    })

    cmd = [
        sys.executable,
        str(exp["script"]),
        str(exp["config"]),
        "--no-websocket",
    ]

    start_time = time.time()
    result = subprocess.run(
        cmd,
        cwd=str(exp["cwd"]),
        env=env,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.time() - start_time

    success = result.returncode == 0
    status = "SUCCESS" if success else f"FAILED (code={result.returncode})"
    print(f"\n  [{status}] {exp['name']} - 耗时 {elapsed/60:.1f} 分钟")
    return success


def main():
    parser = argparse.ArgumentParser(description="批量运行消融实验")
    parser.add_argument("--dataset", choices=["wl", "xf"], default=None,
                        help="只运行指定数据集 (wl=武大图书馆, xf=西贝预制菜)")
    parser.add_argument("--mode", choices=["no_ebdi", "cot"], default=None,
                        help="只运行指定消融模式")
    args = parser.parse_args()

    # 筛选实验
    experiments = EXPERIMENTS
    if args.dataset:
        experiments = [e for e in experiments if e["dataset"] == args.dataset]
    if args.mode:
        experiments = [e for e in experiments if e["mode"] == args.mode]

    if not experiments:
        print("没有匹配的实验")
        sys.exit(1)

    print("=" * 70)
    print("  POSIM 消融实验批量运行")
    print(f"  计划运行 {len(experiments)} 个实验:")
    for exp in experiments:
        print(f"    - {exp['name']}")
    print(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    total_start = time.time()
    results = {}
    for exp in experiments:
        success = run_experiment(exp)
        results[exp["name"]] = success
        if not success:
            print(f"\n⚠️ {exp['name']} 失败，继续下一个实验...")

    # 汇总
    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 70}")
    print("  实验汇总")
    print(f"{'=' * 70}")
    for name, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"  {status} - {name}")
    print(f"\n  总耗时: {total_elapsed/60:.1f} 分钟")
    print(f"  完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # 如果全部成功，提示运行评估
    if all(results.values()):
        print("\n所有实验完成! 接下来可以运行评估:")
        print("  python scripts/wudatushuguan/evaluate.py --skip-llm --skip-mechanism")
        print("  python scripts/xibeiyuzhicai/evaluate.py --skip-llm --skip-mechanism")


if __name__ == "__main__":
    main()
