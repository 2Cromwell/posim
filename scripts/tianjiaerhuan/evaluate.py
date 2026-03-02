# -*- coding: utf-8 -*-
"""
独立评估脚本 - 对模拟结果进行全面评估

用法:
  python evaluate.py                          # 评估最近一次模拟结果
  python evaluate.py <sim_results_dir>        # 评估指定的模拟结果目录
  python evaluate.py --skip-llm               # 跳过需要LLM的评估
  python evaluate.py --skip-mechanism          # 跳过机制验证
  python evaluate.py --skip-calibration        # 跳过真实数据校准
"""
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from posim.evaluation.evaluator_manager import EvaluationManager


def setup_logging(debug: bool = False):
    """配置日志"""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    # 抑制第三方库日志
    for lib in ['matplotlib', 'PIL', 'urllib3', 'httpx', 'httpcore', 'openai']:
        logging.getLogger(lib).setLevel(logging.WARNING)


def find_latest_simulation(output_dir: Path) -> Path:
    """查找最近一次模拟结果目录"""
    if not output_dir.exists():
        return None
    
    # 查找所有包含 simulation_results 的目录
    candidates = []
    for d in output_dir.iterdir():
        if d.is_dir():
            sim_dir = d / "simulation_results"
            if sim_dir.exists() and (sim_dir / "micro_results.json").exists():
                candidates.append(sim_dir)
    
    if not candidates:
        return None
    
    # 按修改时间排序，取最新的
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]


def load_config(config_path: Path) -> dict:
    """加载配置文件"""
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def init_api_pool(config: dict):
    """初始化API池"""
    try:
        from posim.config.config_manager import ConfigManager
        from posim.llm.api_pool import APIPool
        
        config_path = Path(__file__).parent / "config.json"
        if config_path.exists():
            cm = ConfigManager(str(config_path))
            api_pool = APIPool(cm.llm, cm.debug)
            return api_pool
    except Exception as e:
        logging.warning(f"无法初始化API池: {e}")
    return None


def init_embedding_model(config: dict):
    """初始化Embedding模型"""
    llm_config = config.get('llm', {})
    
    if llm_config.get('use_local_embedding_model'):
        model_path = llm_config.get('local_embedding_model_path', '')
        if model_path and Path(model_path).exists():
            try:
                from sentence_transformers import SentenceTransformer
                device = llm_config.get('embedding_device', 'cpu')
                model = SentenceTransformer(model_path, device=device)
                print(f"  ✅ Embedding模型已加载: {model_path}")
                return model
            except Exception as e:
                logging.warning(f"无法加载Embedding模型: {e}")
    return None


def load_users_data(config: dict, script_dir: Path) -> list:
    """加载用户数据"""
    data_config = config.get('data', {})
    users_file = data_config.get('users_file', 'data/users.json')
    users_path = script_dir / users_file
    
    if users_path.exists():
        with open(users_path, 'r', encoding='utf-8') as f:
            users = json.load(f)
        print(f"  ✅ 用户数据已加载: {len(users)} 个用户")
        return users
    return []


def main():
    parser = argparse.ArgumentParser(description='POSIM 仿真评估工具')
    parser.add_argument('sim_dir', nargs='?', default=None,
                       help='模拟结果目录路径（默认选择最近一次）')
    parser.add_argument('--real-data', '-r', default=None,
                       help='真实标注数据路径（labels.json）')
    parser.add_argument('--base-data', '-b', default=None,
                       help='原始数据路径（base_data.json，用于网络拓扑等完整字段分析）')
    parser.add_argument('--output', '-o', default=None,
                       help='评估结果输出目录')
    parser.add_argument('--granularity', '-g', type=int, default=10,
                       help='时间粒度（分钟，默认10）')
    parser.add_argument('--time-start', default=None,
                       help='时间范围起始（格式: YYYY-MM-DD HH:MM）')
    parser.add_argument('--time-end', default=None,
                       help='时间范围结束（格式: YYYY-MM-DD HH:MM）')
    parser.add_argument('--skip-llm', action='store_true',
                       help='跳过需要LLM的评估')
    parser.add_argument('--skip-mechanism', action='store_true',
                       help='跳过机制验证')
    parser.add_argument('--skip-calibration', action='store_true',
                       help='跳过真实数据校准')
    parser.add_argument('--debug', action='store_true',
                       help='启用调试模式')
    
    args = parser.parse_args()
    
    # 配置日志
    setup_logging(args.debug)
    
    script_dir = Path(__file__).parent
    config_path = script_dir / "config.json"
    config = load_config(config_path)
    
    print("=" * 60)
    print("🔬 POSIM 仿真评估工具")
    print("=" * 60)
    
    # 确定模拟结果目录
    if args.sim_dir:
        sim_dir = Path(args.sim_dir)
        if not sim_dir.exists():
            print(f"❌ 指定的目录不存在: {sim_dir}")
            sys.exit(1)
        
        # 检查是否是simulation_results目录（包含micro_results.json）
        if (sim_dir / "micro_results.json").exists():
            # 已经是simulation_results目录
            pass
        elif (sim_dir / "simulation_results" / "micro_results.json").exists():
            # 传入的是父目录，需要进入simulation_results
            sim_dir = sim_dir / "simulation_results"
        else:
            print(f"❌ 目录中未找到 micro_results.json: {sim_dir}")
            print(f"   请确保目录包含 micro_results.json 文件")
            sys.exit(1)
    else:
        output_base = Path(config.get('output', {}).get('base_dir', 'output'))
        if not output_base.is_absolute():
            output_base = script_dir / output_base
        
        sim_dir = find_latest_simulation(output_base)
        if sim_dir is None:
            print(f"❌ 未找到任何模拟结果目录: {output_base}")
            sys.exit(1)
    
    print(f"\n  📂 模拟结果目录: {sim_dir}")
    
    # 确定真实数据路径
    real_data_path = args.real_data
    if not real_data_path:
        # 尝试默认路径
        default_real = project_root / "data_process" / "tianjaierhuan" / "output" / "labels.json"
        if default_real.exists():
            real_data_path = str(default_real)
            print(f"  📂 真实数据: {real_data_path}")
        else:
            print("  ⚠️ 未找到默认真实数据文件")
    else:
        print(f"  📂 真实数据: {real_data_path}")
    
    # 确定原始数据路径（用于网络拓扑等需要完整字段的分析）
    base_data_path = args.base_data
    if not base_data_path:
        default_base = project_root / "data_process" / "tianjaierhuan" / "output" / "base_data.json"
        if default_base.exists():
            base_data_path = str(default_base)
            print(f"  📂 原始数据(网络构建): {base_data_path}")
    else:
        print(f"  📂 原始数据(网络构建): {base_data_path}")
    
    # 时间参数（从配置读取默认值）
    sim_config = config.get('simulation', {})
    time_start = args.time_start or sim_config.get('start_time')
    time_end = args.time_end or sim_config.get('end_time')
    granularity = args.granularity or sim_config.get('time_granularity', 10)
    
    print(f"  ⏱️ 时间范围: {time_start} ~ {time_end}")
    print(f"  ⏱️ 时间粒度: {granularity} 分钟")
    
    # 初始化资源
    print("\n  初始化资源...")
    
    api_pool = None
    if not args.skip_llm:
        api_pool = init_api_pool(config)
    
    embedding_model = init_embedding_model(config)
    users_data = load_users_data(config, script_dir)
    event_background = sim_config.get('event_background', '')
    
    # 创建评估管理器
    manager = EvaluationManager(
        sim_results_dir=str(sim_dir),
        real_data_path=real_data_path,
        base_data_path=base_data_path,
        output_dir=args.output,
        time_granularity=granularity,
        time_start=time_start,
        time_end=time_end
    )
    
    # 运行评估
    results = manager.run_all(
        api_pool=api_pool,
        embedding_model=embedding_model,
        users_data=users_data,
        event_background=event_background,
        skip_mechanism=args.skip_mechanism,
        skip_calibration=args.skip_calibration,
        skip_llm_evaluation=args.skip_llm
    )
    
    print("\n✅ 评估完成！")


if __name__ == "__main__":
    main()
