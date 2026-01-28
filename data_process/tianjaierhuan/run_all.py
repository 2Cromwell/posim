"""
主运行脚本：按顺序执行所有数据处理步骤
"""
import subprocess
import sys
import os

def run_step(script_name, description):
    """运行单个步骤"""
    print(f"\n{'#' * 60}")
    print(f"# {description}")
    print(f"# 脚本: {script_name}")
    print(f"{'#' * 60}\n")
    
    result = subprocess.run([sys.executable, script_name], cwd=os.path.dirname(__file__))
    
    if result.returncode != 0:
        print(f"[ERROR] {script_name} 执行失败，返回码: {result.returncode}")
        return False
    
    return True


def main():
    print("=" * 60)
    print("天价耳环事件数据处理流程")
    print("=" * 60)
    
    steps = [
        ("step1_extract_and_clean.py", "Step1: 数据提取与清洗"),
        ("step2_generate_initial_posts.py", "Step2: 生成initial_posts.json"),
        ("step3_generate_relations.py", "Step3: 生成relations.json"),
        ("step4_generate_users_new.py", "Step4: 生成users.json（大模型推断+行为动态）")
    ]
    
    for script, desc in steps:
        if not run_step(script, desc):
            print(f"[ABORT] 流程在 {script} 处中断")
            return
    
    print("\n" + "=" * 60)
    print("全部处理完成！")
    print("=" * 60)
    print("生成的文件：")
    print("  - output/base_data.json (基础数据)")
    print("  - output/initial_posts.json (初始博文)")
    print("  - output/relations.json (社交关系)")
    print("  - output/users.json (用户信念)")
    print("  - output/users_influence.json (用户影响力)")


if __name__ == "__main__":
    main()
