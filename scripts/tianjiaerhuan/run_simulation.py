"""
天价耳环事件仿真执行脚本
演示如何使用posim包进行舆情仿真
"""
import asyncio
import sys
import logging
from pathlib import Path

# 添加项目根目录到路径（便于开发调试，正式使用时可通过pip安装posim）
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from posim.config.config_manager import ConfigManager
from posim.llm.api_pool import APIPool
from posim.engine.simulator import Simulator
from posim.data.data_loader import DataLoader, parse_user_data
from posim.storage.log_manager import LogManager
from posim.storage.database import SimulationDatabase
from posim.evaluation.evaluator import SimulationEvaluator


def setup_logging(debug_config):
    """配置日志系统"""
    level = logging.DEBUG if debug_config.enabled else logging.INFO
    log_format = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    logging.basicConfig(level=level, format=log_format)
    
    # 抑制第三方库的调试日志
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('neo4j').setLevel(logging.WARNING)
    logging.getLogger('neo4j.io').setLevel(logging.WARNING)
    logging.getLogger('neo4j.pool').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)
    logging.getLogger('openai._base_client').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('httpcore.http11').setLevel(logging.WARNING)
    
    if debug_config.enabled:
        logging.getLogger('posim').setLevel(logging.DEBUG)
        logging.info("Debug mode enabled - detailed logging active")


class FlushingFileHandler(logging.FileHandler):
    """每次写入后立即刷新的文件处理器"""
    def emit(self, record):
        super().emit(record)
        self.flush()


def add_file_handler(log_dir: Path, debug_config):
    """添加文件日志处理器（实时刷新）"""
    log_file = log_dir / "detailed.log"
    level = logging.DEBUG if debug_config.enabled else logging.INFO
    file_handler = FlushingFileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
    logging.getLogger().addHandler(file_handler)
    logging.info(f"Detailed logs will be saved to: {log_file}")


async def main():
    # 配置路径 - 只需要一个config.json
    script_dir = Path(__file__).parent
    config_path = script_dir / "config.json"
    
    print("=" * 60)
    print("POSIM: Social Media Public Opinion Simulator")
    print("=" * 60)
    
    # 加载配置（合并后的单一配置文件）
    print("\n[1/7] Loading configurations...")
    config_manager = ConfigManager(str(config_path))
    
    # 配置日志
    setup_logging(config_manager.debug)
    logger = logging.getLogger(__name__)
    
    # 创建API池（使用合并后的llm配置）
    api_pool = APIPool(config_manager.llm, config_manager.debug)
    
    logger.info(f"Event: {config_manager.simulation.event_name}")
    logger.info(f"Time range: {config_manager.simulation.start_time} ~ {config_manager.simulation.end_time}")
    logger.info(f"Time granularity: {config_manager.simulation.time_granularity} minute(s)")
    logger.info(f"Neo4j enabled: {config_manager.neo4j.enabled}")
    logger.info(f"Debug mode: {config_manager.debug.enabled}")
    
    print(f"  - Event: {config_manager.simulation.event_name}")
    print(f"  - Time range: {config_manager.simulation.start_time} ~ {config_manager.simulation.end_time}")
    print(f"  - Time granularity: {config_manager.simulation.time_granularity} minute(s)")
    
    # 加载数据（使用配置中的路径）
    print("\n[2/7] Loading simulation data...")
    data_dir = config_manager.get_data_dir()
    data_loader = DataLoader(str(data_dir / "data"))
    data = data_loader.load_all()
    
    users_data = [parse_user_data(u) for u in data['users']]
    events_data = data['events']
    initial_posts = data['initial_posts']
    relations_data = data.get('relations', [])
    
    print(f"  - Users loaded: {len(users_data)}")
    print(f"  - Events loaded: {len(events_data)}")
    print(f"  - Initial posts: {len(initial_posts)}")
    print(f"  - Relations loaded: {len(relations_data)}")
    
    # 初始化仿真器
    print("\n[3/6] Initializing simulator...")
    simulator = Simulator(config_manager, api_pool)
    simulator.load_agents(users_data)
    simulator.load_events(events_data)
    simulator.load_initial_posts(initial_posts)
    simulator.load_relations(relations_data)
    
    print(f"  - Agents initialized: {len(simulator.agents)}")
    
    # 初始化日志管理器
    print("\n[4/6] Setting up logging...")
    log_manager = LogManager(
        config_manager.output.base_dir,
        config_manager.simulation.event_name,
        config_manager.simulation.simulation_title
    )
    log_manager.save_config(config_manager.raw_config)
    
    # 添加文件日志处理器，将详细日志写入文件
    add_file_handler(log_manager.results_dir, config_manager.debug)
    
    # 初始化数据库
    db = SimulationDatabase(str(log_manager.results_dir / "simulation.db"))
    
    # 设置回调
    all_actions = []
    
    def on_step(step_data):
        log_manager.log_step(step_data)
        db.save_statistics(
            step_data['step'],
            step_data['time'],
            step_data['intensity'],
            step_data['activated_count'],
            step_data['actions_count'],
            []
        )
    
    def on_action(action):
        all_actions.append(action)
        db.save_action(action, simulator.time_engine.state.step)
    
    simulator.step_callback = on_step
    simulator.action_callback = on_action
    
    # 运行仿真
    print("\n[5/7] Running simulation...")
    print("-" * 40)
    
    def progress_callback(progress, step_data):
        step = step_data['step']
        time = step_data['time'][-8:]
        actions = step_data['actions_count']
        intensity = step_data['intensity']
        bar_len = 30
        filled = int(bar_len * progress)
        bar = '█' * filled + '░' * (bar_len - filled)
        print(f"\r  [{bar}] {progress*100:5.1f}% | Step {step:4d} | {time} | λ={intensity:.3f} | Actions: {actions}", end='')
    
    results = await simulator.run(progress_callback)
    print("\n" + "-" * 40)
    
    # 保存结果
    print("\n[6/7] Saving results...")
    log_manager.save_macro_results(results)
    log_manager.save_micro_results(all_actions)
    log_manager.save_statistics(results['stats'])
    logger.info(f"Results saved: {len(all_actions)} actions recorded")
    
    # 运行评估（如果配置开启）
    if config_manager.output.run_evaluation:
        print("\n[7/7] Running evaluation...")
        # 真实数据路径（优先使用data_process输出的base_data.json）
        real_data_path = script_dir.parent.parent / "data_process" / "tianjaierhuan" / "output" / "base_data.json"
        if not real_data_path.exists():
            real_data_path = script_dir / "data" / "base_data.json"
        evaluator = SimulationEvaluator(
            log_manager.results_dir,
            real_data_path=str(real_data_path) if real_data_path.exists() else None,
            time_granularity=config_manager.simulation.time_granularity,
            time_start=config_manager.simulation.start_time,
            time_end=config_manager.simulation.end_time
        )
        eval_results = evaluator.run_evaluation(
            results, all_actions, simulator.social_network
        )
        logger.info(f"Evaluation completed: sim_peak={eval_results.get('sim_peak', 0)}, real_peak={eval_results.get('real_peak', 0)}")
    else:
        print("\n[7/7] Skipping evaluation (disabled in config)")
    
    summary = f"""
POSIM Simulation Summary
========================
Event: {config_manager.simulation.event_name}
Title: {config_manager.simulation.simulation_title}
Duration: {config_manager.simulation.start_time} ~ {config_manager.simulation.end_time}

Results:
- Total Steps: {results['steps']}
- Total Actions: {results['stats']['total_actions']}
- Actions by Type:
"""
    for action_type, count in results['stats'].get('actions_by_type', {}).items():
        summary += f"  - {action_type}: {count}\n"
    
    summary += f"\nOutput Directory: {log_manager.output_path}"
    
    log_manager.save_summary(summary)
    print(summary)
    
    # 清理
    db.close()
    simulator.social_network.close()
    log_manager.close()
    
    print("\n✓ Simulation completed successfully!")
    print(f"  Results saved to: {log_manager.output_path}")


if __name__ == "__main__":
    asyncio.run(main())
