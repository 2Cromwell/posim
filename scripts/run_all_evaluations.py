# -*- coding: utf-8 -*-
"""
Run evaluations on all experiment groups.

Usage:
    python scripts/run_all_evaluations.py                 # run all groups
    python scripts/run_all_evaluations.py ours             # run one group
    python scripts/run_all_evaluations.py ours tianjiaerhuan  # run one dataset in one group
"""
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from posim.evaluation.evaluator_manager import EvaluationManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
for lib in ['matplotlib', 'PIL', 'urllib3', 'httpx', 'httpcore', 'openai']:
    logging.getLogger(lib).setLevel(logging.WARNING)


# ──────────────────────────────────────────────────────────────
# Experiment groups  (from papers/exp_results_latex/实验结果映射.md)
# ──────────────────────────────────────────────────────────────
EXPERIMENT_GROUPS = {
    'ours': {
        'label': 'POSIM (Ours)',
        'datasets': [
            {
                'name': 'tianjiaerhuan',
                'sim_dir': 'scripts/tianjiaerhuan/output/tianjiaerhuan_baseline_20260221_152957_14B效果好/simulation_results',
                'real_data': 'data_process/tianjaierhuan/output/labels.json',
                'base_data': 'data_process/tianjaierhuan/output/base_data.json',
                'config': 'scripts/tianjiaerhuan/config.json',
            },
            {
                'name': 'wudatushuguan',
                'sim_dir': 'scripts/wudatushuguan/output/wudatushuguan_baseline_20260221_021403_14B_行为分布好/simulation_results',
                'real_data': 'data_process/wudatushuguan/output/labels.json',
                'base_data': 'data_process/wudatushuguan/output/base_data.json',
                'config': 'scripts/wudatushuguan/config.json',
            },
            {
                'name': 'xibeiyuzhicai',
                'sim_dir': 'scripts/xibeiyuzhicai/output/xibeiyuzhicai_baseline_20260223_145442_14B效果不错/simulation_results',
                'real_data': 'data_process/xibeiyuzhicai/output/labels.json',
                'base_data': 'data_process/xibeiyuzhicai/output/base_data.json',
                'config': 'scripts/xibeiyuzhicai/config.json',
            },
        ],
    },
    'abm': {
        'label': 'Rule-based ABM',
        'datasets': [
            {
                'name': 'tianjiaerhuan',
                'sim_dir': 'scripts/tianjiaerhuan/output/tianjiaerhuan_baseline_20260214_013248/simulation_results',
                'real_data': 'data_process/tianjaierhuan/output/labels.json',
                'base_data': 'data_process/tianjaierhuan/output/base_data.json',
                'config': 'scripts/tianjiaerhuan/config.json',
            },
            {
                'name': 'wudatushuguan',
                'sim_dir': 'scripts/wudatushuguan/output/wudatushuguan_baseline_20260224_200017/simulation_results',
                'real_data': 'data_process/wudatushuguan/output/labels.json',
                'base_data': 'data_process/wudatushuguan/output/base_data.json',
                'config': 'scripts/wudatushuguan/config.json',
            },
            {
                'name': 'xibeiyuzhicai',
                'sim_dir': 'scripts/xibeiyuzhicai/output/xibeiyuzhicai_baseline_20260223_144652_ABM方法/simulation_results',
                'real_data': 'data_process/xibeiyuzhicai/output/labels.json',
                'base_data': 'data_process/xibeiyuzhicai/output/base_data.json',
                'config': 'scripts/xibeiyuzhicai/config.json',
            },
        ],
    },
    'wo_ebdi': {
        'label': 'Ours w/o EBDI',
        'datasets': [
            {
                'name': 'tianjiaerhuan',
                'sim_dir': 'scripts/tianjiaerhuan/output/tianjiaerhuan_baseline_20260215_004906/simulation_results',
                'real_data': 'data_process/tianjaierhuan/output/labels.json',
                'base_data': 'data_process/tianjaierhuan/output/base_data.json',
                'config': 'scripts/tianjiaerhuan/config.json',
            },
            {
                'name': 'wudatushuguan',
                'sim_dir': 'scripts/wudatushuguan/output/wudatushuguan_baseline_20260220_120139_完整模拟_行为分布不好/simulation_results',
                'real_data': 'data_process/wudatushuguan/output/labels.json',
                'base_data': 'data_process/wudatushuguan/output/base_data.json',
                'config': 'scripts/wudatushuguan/config.json',
            },
            {
                'name': 'xibeiyuzhicai',
                'sim_dir': 'scripts/xibeiyuzhicai/output/xibeiyuzhicai_baseline_20260223_210314/simulation_results',
                'real_data': 'data_process/xibeiyuzhicai/output/labels.json',
                'base_data': 'data_process/xibeiyuzhicai/output/base_data.json',
                'config': 'scripts/xibeiyuzhicai/config.json',
            },
        ],
    },
    'w_cot': {
        'label': 'Ours w/ CoT',
        'datasets': [
            {
                'name': 'tianjiaerhuan',
                'sim_dir': 'scripts/tianjiaerhuan/output/tianjiaerhuan_baseline_20260219_213300_备选/simulation_results',
                'real_data': 'data_process/tianjaierhuan/output/labels.json',
                'base_data': 'data_process/tianjaierhuan/output/base_data.json',
                'config': 'scripts/tianjiaerhuan/config.json',
            },
        ],
    },
}


class TeeWriter:
    """Write to both stdout and a log file simultaneously."""
    def __init__(self, log_file, original_stdout):
        self.log_file = log_file
        self.original_stdout = original_stdout

    def write(self, text):
        self.original_stdout.write(text)
        self.log_file.write(text)

    def flush(self):
        self.original_stdout.flush()
        self.log_file.flush()


def _load_embedding_model(config):
    llm_config = config.get('llm', {})
    if not llm_config.get('use_local_embedding_model'):
        return None
    model_path = llm_config.get('local_embedding_model_path', '')
    if not model_path or not Path(model_path).exists():
        return None
    try:
        from sentence_transformers import SentenceTransformer
        device = llm_config.get('embedding_device', 'cpu')
        model = SentenceTransformer(model_path, device=device)
        print(f"  Embedding model loaded: {model_path}")
        return model
    except Exception as e:
        print(f"  Warning: Could not load embedding model: {e}")
        return None


def _get_experiment_dir(ds):
    """sim_dir points to .../simulation_results, the experiment dir is its parent."""
    return project_root / ds['sim_dir'] / '..'


def _get_log_path(ds, group_key):
    exp_dir = _get_experiment_dir(ds).resolve()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return exp_dir / f'eval_{group_key}_{timestamp}.log'


def run_dataset(ds, group_key, group_label, embedding_model=None):
    log_path = _get_log_path(ds, group_key)
    original_stdout = sys.stdout

    with open(log_path, 'w', encoding='utf-8') as log_file:
        tee = TeeWriter(log_file, original_stdout)
        sys.stdout = tee

        print(f"\n{'='*60}")
        print(f"  [{group_label}] Evaluating: {ds['name']}")
        print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Log:  {log_path}")
        print(f"{'='*60}")

        config = {}
        config_path = project_root / ds['config']
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

        sim_config = config.get('simulation', {})

        if embedding_model is None:
            embedding_model = _load_embedding_model(config)

        manager = EvaluationManager(
            sim_results_dir=str(project_root / ds['sim_dir']),
            real_data_path=str(project_root / ds['real_data']),
            base_data_path=str(project_root / ds['base_data']),
            time_granularity=sim_config.get('time_granularity', 10),
            time_start=sim_config.get('start_time'),
            time_end=sim_config.get('end_time'),
        )

        results = manager.run_all(
            embedding_model=embedding_model,
            skip_llm_evaluation=True,
            skip_mechanism=False,
            skip_calibration=False,
        )

        sys.stdout = original_stdout

    print(f"  Log saved: {log_path}")
    return results, embedding_model


def run_group(group_key):
    group = EXPERIMENT_GROUPS[group_key]
    label = group['label']

    print(f"\n{'#'*70}")
    print(f"  Experiment Group: {label}  ({group_key})")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}")

    embedding_model = None
    all_results = {}

    for ds in group['datasets']:
        try:
            results, embedding_model = run_dataset(ds, group_key, label, embedding_model)
            all_results[ds['name']] = results
            print(f"  Done: {ds['name']}")
        except Exception as e:
            print(f"\n  ERROR on {ds['name']}: {e}")
            import traceback
            traceback.print_exc()

    return all_results


def main():
    args = sys.argv[1:]

    if len(args) == 0:
        for gk in EXPERIMENT_GROUPS:
            run_group(gk)

    elif len(args) == 1:
        gk = args[0]
        if gk in EXPERIMENT_GROUPS:
            run_group(gk)
        else:
            print(f"Unknown group: {gk}")
            print(f"Available: {', '.join(EXPERIMENT_GROUPS.keys())}")

    elif len(args) == 2:
        gk, ds_name = args
        if gk not in EXPERIMENT_GROUPS:
            print(f"Unknown group: {gk}")
            print(f"Available: {', '.join(EXPERIMENT_GROUPS.keys())}")
            return
        group = EXPERIMENT_GROUPS[gk]
        for ds in group['datasets']:
            if ds['name'] == ds_name:
                try:
                    run_dataset(ds, gk, group['label'])
                    print(f"  Done: {ds_name}")
                except Exception as e:
                    print(f"\n  ERROR on {ds_name}: {e}")
                    import traceback
                    traceback.print_exc()
                break
        else:
            print(f"Unknown dataset '{ds_name}' in group '{gk}'")
            print(f"Available: {', '.join(d['name'] for d in group['datasets'])}")

    else:
        print("Usage:")
        print("  python scripts/run_all_evaluations.py                    # all groups")
        print("  python scripts/run_all_evaluations.py <group>            # one group")
        print("  python scripts/run_all_evaluations.py <group> <dataset>  # one dataset")
        print(f"\nGroups: {', '.join(EXPERIMENT_GROUPS.keys())}")


if __name__ == '__main__':
    main()
