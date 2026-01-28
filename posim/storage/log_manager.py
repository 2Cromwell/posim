"""
日志管理器 - 仿真日志记录与回退支持
"""
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime


class LogManager:
    """仿真日志管理器"""
    
    def __init__(self, output_dir: str, event_name: str, simulation_title: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir) / f"{event_name}_{simulation_title}_{timestamp}"
        self.results_dir = self.output_dir / "simulation_results"
        self.vis_dir = self.output_dir / "vis_results"
        
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        
        self._log_file = open(self.results_dir / "simulation.log", 'w', encoding='utf-8')
        self._step_logs: List[Dict] = []
    
    def log(self, message: str, level: str = "INFO"):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] [{level}] {message}\n"
        self._log_file.write(log_line)
        self._log_file.flush()
    
    def log_step(self, step_data: Dict):
        """记录每步数据"""
        self._step_logs.append(step_data)
        self.log(f"Step {step_data.get('step', 0)}: {step_data.get('actions_count', 0)} actions")
    
    def save_config(self, config: Dict):
        """保存配置"""
        with open(self.results_dir / "config.json", 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    
    def save_macro_results(self, results: Dict):
        """保存宏观结果"""
        with open(self.results_dir / "macro_results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    def save_micro_results(self, results: List[Dict]):
        """保存微观结果"""
        with open(self.results_dir / "micro_results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    def save_summary(self, summary: str):
        """保存摘要"""
        with open(self.results_dir / "summary.txt", 'w', encoding='utf-8') as f:
            f.write(summary)
    
    def save_statistics(self, stats: Dict):
        """保存统计数据"""
        with open(self.results_dir / "statistics.txt", 'w', encoding='utf-8') as f:
            f.write("=== POSIM Simulation Statistics ===\n\n")
            f.write(f"Total Actions: {stats.get('total_actions', 0)}\n")
            f.write(f"Actions by Type:\n")
            for action_type, count in stats.get('actions_by_type', {}).items():
                f.write(f"  - {action_type}: {count}\n")
    
    def get_step_log(self, step: int) -> Optional[Dict]:
        """获取指定步骤的日志"""
        for log in self._step_logs:
            if log.get('step') == step:
                return log
        return None
    
    def get_all_step_logs(self) -> List[Dict]:
        """获取所有步骤日志"""
        return self._step_logs
    
    def close(self):
        """关闭日志"""
        self._log_file.close()
    
    @property
    def output_path(self) -> Path:
        return self.output_dir
