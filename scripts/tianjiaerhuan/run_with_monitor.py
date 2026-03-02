"""
带实时监控的仿真运行脚本
"""
import asyncio
import sys
import json
import logging
import subprocess
import os
from pathlib import Path
from datetime import datetime

os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from posim.config.config_manager import ConfigManager
from posim.llm.api_pool import APIPool
from posim.engine.simulator import Simulator
from posim.data.data_loader import DataLoader, parse_user_data
from posim.storage.database import SimulationDatabase

try:
    from posim.web.websocket_server import SimulationWebSocketServer, WebSocketSignalCallback
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False


def setup_logging(debug_config):
    """配置日志"""
    level = logging.DEBUG if debug_config.enabled else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    for lib in ['matplotlib', 'PIL', 'urllib3', 'httpx', 'neo4j', 'openai', 'httpcore']:
        logging.getLogger(lib).setLevel(logging.WARNING)
    if debug_config.enabled:
        logging.getLogger('posim').setLevel(logging.DEBUG)


class FlushingFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


def add_file_handler(log_dir: Path, debug_config):
    """添加文件日志"""
    log_file = log_dir / "detailed.log"
    handler = FlushingFileHandler(log_file, encoding='utf-8')
    handler.setLevel(logging.DEBUG if debug_config.enabled else logging.INFO)
    handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
    logging.getLogger().addHandler(handler)


async def run_simulation_with_monitor(config_path: str, enable_websocket: bool = True):
    """运行带实时监控的仿真"""
    config_manager = ConfigManager(config_path)
    setup_logging(config_manager.debug)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("POSIM: Social Media Public Opinion Simulator (With Monitor)")
    logger.info("=" * 60)

    # 1. 加载配置
    logger.info("[1/9] Loading configurations...")
    logger.info(f"Event: {config_manager.simulation.event_name}")
    logger.info(f"Time range: {config_manager.simulation.start_time} ~ {config_manager.simulation.end_time}")
    logger.info(f"Time granularity: {config_manager.simulation.time_granularity} minute(s)")

    # 2. 加载数据
    logger.info("[2/9] Loading simulation data...")
    data_dir = config_manager.get_data_dir()
    data = DataLoader(str(data_dir / "data")).load_all()
    users_data = [parse_user_data(u) for u in data['users']]
    logger.info(f"Users: {len(users_data)}, Events: {len(data['events'])}, Posts: {len(data['initial_posts'])}, Relations: {len(data.get('relations', []))}")

    # 3. 初始化输出目录
    logger.info("[3/9] Setting up output directory...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config_manager.output.base_dir) / f"{config_manager.simulation.event_name}_{config_manager.simulation.simulation_title}_{timestamp}" / "simulation_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    add_file_handler(output_dir, config_manager.debug)

    # 4. 初始化数据库
    logger.info("[4/9] Initializing database...")
    db = SimulationDatabase(str(output_dir / "simulation.db"))

    # 5. 初始化API池
    logger.info("[5/9] Initializing API pool...")
    api_pool = APIPool(config_manager.llm, config_manager.debug)

    # 6. 初始化仿真器
    logger.info("[6/9] Initializing simulator...")
    simulator = Simulator(config_manager, api_pool)
    simulator.load_agents(users_data)
    simulator.load_events(data['events'])
    simulator.load_relations(data.get('relations', []))
    simulator.load_initial_posts(data['initial_posts'])
    logger.info(f"Agents initialized: {len(simulator.agents)}")

    # 7. 启动WebSocket服务器
    ws_server = None
    ws_task = None
    if enable_websocket and WEBSOCKET_AVAILABLE:
        logger.info("[7/9] Starting WebSocket server...")
        ws_server = SimulationWebSocketServer(host="localhost", port=8765)
        simulator.signal_callback = WebSocketSignalCallback(ws_server)
        ws_task = asyncio.create_task(ws_server.start())
        logger.info(f"WebSocket server: ws://localhost:8765")
        logger.info(f"Monitor page: file:///{project_root}/posim/web/monitor.html")
        await asyncio.sleep(0.5)
        ws_server.send_signal({'type': 'test', 'message': 'Server ready'})
    else:
        logger.info("[7/9] WebSocket disabled")

    # 8. 设置回调
    all_actions = []
    simulator.step_callback = lambda step_data: db.save_statistics(
        step_data['step'], step_data['time'], step_data['intensity'],
        step_data['activated_count'], step_data['actions_count'], []
    )
    simulator.action_callback = lambda action: all_actions.append(action) or db.save_action(action, simulator.time_engine.state.step)

    # 9. 运行仿真
    logger.info("[8/9] Running simulation...")
    logger.info("-" * 60)
    
    def progress_callback(progress, step_data):
        step = step_data['step']
        time = step_data['time'][-8:]
        actions = step_data['actions_count']
        intensity = step_data['intensity']
        activated = step_data['activated_count']
        bar = '█' * int(30 * progress) + '░' * (30 - int(30 * progress))
        logger.info(f"[{bar}] {progress*100:5.1f}% | Step {step:4d}| {time} | λ={intensity:.4f} | 激活:{activated:3d} | 行为:{actions:3d}")
    
    results = await simulator.run(progress_callback)
    logger.info("-" * 60)

    logger.info("=" * 60)
    logger.info("Simulation completed!")
    logger.info("=" * 60)
    logger.info(f"Total Steps: {results['steps']}")
    logger.info(f"Total Actions: {results['stats']['total_actions']}")
    for action_type, count in results['stats'].get('actions_by_type', {}).items():
        logger.info(f"  - {action_type}: {count}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 60)

    # 保存结果
    logger.info("Saving simulation results...")
    macro_results = {
        'steps': results['steps'],
        'stats': results['stats'],
        'performance': results['performance'],
        'final_hot_search': results.get('final_hot_search', []),
        'hot_search_history': results.get('hot_search_history', []),
        'time_engine': results.get('time_engine', {})
    }
    with open(output_dir / "macro_results.json", 'w', encoding='utf-8') as f:
        json.dump(macro_results, f, ensure_ascii=False, indent=2, default=str)
    with open(output_dir / "micro_results.json", 'w', encoding='utf-8') as f:
        json.dump(all_actions, f, ensure_ascii=False, indent=2, default=str)
    with open(output_dir / "config.json", 'w', encoding='utf-8') as f:
        json.dump(config_manager.raw_config, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved: macro_results.json, micro_results.json, config.json")

    # 运行评估
    if config_manager.output.run_evaluation:
        logger.info("[9/9] Running evaluation...")
        eval_script = Path(__file__).parent / "evaluate.py"
        if eval_script.exists():
            real_data_path = project_root / "data_process" / "tianjaierhuan" / "output" / "labels.json"
            base_data_path = project_root / "data_process" / "tianjaierhuan" / "output" / "base_data.json"
            # 确保使用绝对路径
            output_dir_abs = output_dir.resolve()
            cmd = [sys.executable, str(eval_script), str(output_dir_abs)]
            if real_data_path.exists():
                cmd.extend(['--real-data', str(real_data_path.resolve())])
            if base_data_path.exists():
                cmd.extend(['--base-data', str(base_data_path.resolve())])
            
            env = os.environ.copy()
            env.update({'PYTHONIOENCODING': 'utf-8', 'PYTHONUNBUFFERED': '1'})
            
            logger.info(f"Executing: {' '.join(cmd)}")
            logger.info("-" * 60)
            result = subprocess.run(cmd, cwd=str(project_root), env=env, encoding='utf-8', errors='replace')
            logger.info("-" * 60)
            
            if result.returncode == 0:
                logger.info("Evaluation completed successfully")
            else:
                logger.error(f"Evaluation failed with return code: {result.returncode}")
        else:
            logger.warning(f"Evaluation script not found: {eval_script}")
    else:
        logger.info("[9/9] Skipping evaluation (disabled in config)")

    # 清理
    if ws_server:
        logger.info("Closing WebSocket server...")
        ws_server._running = False
        if ws_task and not ws_task.done():
            ws_task.cancel()
            try:
                await ws_task
            except asyncio.CancelledError:
                pass
        await ws_server.stop()
    
    db.close()
    simulator.social_network.close()
    logger.info("Cleanup completed")


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    logger = logging.getLogger(__name__)
    
    script_dir = Path(__file__).parent
    config_path = script_dir / "config.json"
    if len(sys.argv) > 1 and not sys.argv[1].startswith('--'):
        config_path = Path(sys.argv[1])
    
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    enable_websocket = "--no-websocket" not in sys.argv
    asyncio.run(run_simulation_with_monitor(str(config_path), enable_websocket))


if __name__ == "__main__":
    main()
