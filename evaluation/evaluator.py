"""
评估器 - 评估仿真结果
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional
from .metrics import calculate_all_metrics
from .visualizer import generate_all_visualizations


class SimulationEvaluator:
    """仿真结果评估器"""
    
    def __init__(self, output_base_dir: str = "output"):
        self.output_base_dir = Path(output_base_dir)
    
    def find_latest_simulation(self, event_name: str) -> Optional[Path]:
        """找到指定事件的最新仿真结果"""
        if not self.output_base_dir.exists():
            return None
        
        matching = [d for d in self.output_base_dir.iterdir() 
                   if d.is_dir() and d.name.startswith(event_name)]
        
        if not matching:
            return None
        
        # 按名称排序（包含时间戳）取最新
        return sorted(matching, key=lambda x: x.name)[-1]
    
    def load_simulation_results(self, sim_dir: Path) -> Dict[str, Any]:
        """加载仿真结果"""
        results_dir = sim_dir / "simulation_results"
        
        data = {}
        
        # 加载微观结果
        micro_path = results_dir / "micro_results.json"
        if micro_path.exists():
            with open(micro_path, 'r', encoding='utf-8') as f:
                data['actions'] = json.load(f)
        
        # 加载宏观结果
        macro_path = results_dir / "macro_results.json"
        if macro_path.exists():
            with open(macro_path, 'r', encoding='utf-8') as f:
                data['macro'] = json.load(f)
        
        # 加载配置
        config_path = results_dir / "config.json"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                data['config'] = json.load(f)
        
        return data
    
    def evaluate(self, event_name: str, simulation_title: str = None) -> Dict[str, Any]:
        """
        评估仿真结果
        Args:
            event_name: 事件名称
            simulation_title: 仿真标题（可选，默认最新）
        """
        # 找到仿真目录
        if simulation_title:
            sim_dir = self.output_base_dir / f"{event_name}_{simulation_title}"
            if not sim_dir.exists():
                # 尝试模糊匹配
                matching = [d for d in self.output_base_dir.iterdir() 
                           if d.is_dir() and event_name in d.name and simulation_title in d.name]
                if matching:
                    sim_dir = matching[0]
                else:
                    raise FileNotFoundError(f"Simulation not found: {event_name}_{simulation_title}")
        else:
            sim_dir = self.find_latest_simulation(event_name)
            if not sim_dir:
                raise FileNotFoundError(f"No simulation found for event: {event_name}")
        
        # 加载数据
        data = self.load_simulation_results(sim_dir)
        actions = data.get('actions', [])
        macro = data.get('macro', {})
        
        # 计算指标
        metrics = calculate_all_metrics(actions)
        
        # 生成可视化
        vis_dir = sim_dir / "vis_results"
        intensity_history = macro.get('stats', {}).get('intensity_history', [])
        generate_all_visualizations(metrics, str(vis_dir), intensity_history)
        
        # 返回评估结果
        return {
            'simulation_dir': str(sim_dir),
            'metrics': metrics,
            'macro_stats': macro.get('stats', {}),
            'visualization_dir': str(vis_dir)
        }
    
    def compare_simulations(self, sim_dirs: list) -> Dict[str, Any]:
        """比较多次仿真结果"""
        results = []
        for sim_dir in sim_dirs:
            data = self.load_simulation_results(Path(sim_dir))
            actions = data.get('actions', [])
            metrics = calculate_all_metrics(actions)
            results.append({
                'dir': sim_dir,
                'metrics': metrics
            })
        
        return {'comparisons': results}


def main():
    """命令行入口"""
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate POSIM simulation results')
    parser.add_argument('--event', '-e', required=True, help='Event name')
    parser.add_argument('--title', '-t', default=None, help='Simulation title')
    parser.add_argument('--output', '-o', default='output', help='Output base directory')
    
    args = parser.parse_args()
    
    evaluator = SimulationEvaluator(args.output)
    results = evaluator.evaluate(args.event, args.title)
    
    print(f"Evaluation completed for: {results['simulation_dir']}")
    print(f"Metrics: {json.dumps(results['metrics'], indent=2, default=str)}")
    print(f"Visualizations saved to: {results['visualization_dir']}")


if __name__ == '__main__':
    main()
