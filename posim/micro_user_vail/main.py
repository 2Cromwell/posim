# -*- coding: utf-8 -*-
"""
微观行为机制验证 - 主入口

每完成一个方法的一项验证即保存到磁盘，再次运行（--resume）时自动检测并跳过已有结果。

Usage:
    # 首次运行
    python -m posim.micro_user_vail.main --config scripts/tianjiaerhuan/config.json

    # 断点续跑：指定已有的 run 目录，自动跳过已完成的模拟/验证
    python -m posim.micro_user_vail.main --config scripts/tianjiaerhuan/config.json --resume run_20260224_192239

    # 只跑部分方法
    python -m posim.micro_user_vail.main --config scripts/tianjiaerhuan/config.json --methods direct_nothink cot

    # 只跑模拟 / 只跑验证
    python -m posim.micro_user_vail.main --config scripts/tianjiaerhuan/config.json --skip-validation
    python -m posim.micro_user_vail.main --config scripts/tianjiaerhuan/config.json --skip-simulation --resume run_xxx
"""
import argparse
import asyncio
import json
import logging
import os
import sys
import time
import copy
from datetime import datetime
from typing import List, Dict, Any

from .config import ExperimentConfig, DecisionMethod
from .data_loader import load_users, load_events, build_round_contexts, format_user_profile
from .llm_service import LLMService
from .simulation import SimulationEngine, AgentState, DecisionResult, save_simulation_results
from .validation import (
    BehaviorConsistencyValidator, PersonalityStabilityValidator,
    PromptRobustnessValidator, BehaviorHeterogeneityValidator,
    save_validation_results
)


def setup_logging(output_dir: str, log_level: str = "INFO"):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'experiment.log')
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding='utf-8'),
    ]
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=handlers,
    )
    return logging.getLogger(__name__)


def create_llm_services(config: ExperimentConfig):
    nothink_llm = LLMService(
        name=config.nothink_model.name,
        base_url=config.nothink_model.base_url,
        api_key=config.nothink_model.api_key,
        model=config.nothink_model.model,
        temperature=config.nothink_model.temperature,
        top_p=config.nothink_model.top_p,
        max_concurrent=config.nothink_model.max_concurrent,
        timeout=config.nothink_model.timeout,
    )
    think_llm = LLMService(
        name=config.think_model.name,
        base_url=config.think_model.base_url,
        api_key=config.think_model.api_key,
        model=config.think_model.model,
        temperature=config.think_model.temperature,
        top_p=config.think_model.top_p,
        max_concurrent=config.think_model.max_concurrent,
        timeout=config.think_model.timeout,
    )
    eval_llm = LLMService(
        name=config.eval_model.name,
        base_url=config.eval_model.base_url,
        api_key=config.eval_model.api_key,
        model=config.eval_model.model,
        temperature=0.3,
        top_p=config.eval_model.top_p,
        max_concurrent=config.eval_model.max_concurrent,
        timeout=config.eval_model.timeout,
    )
    return nothink_llm, think_llm, eval_llm


def initialize_agents(users: List[Dict], method: str) -> List[AgentState]:
    """为指定方法初始化智能体状态（每种方法独立状态）"""
    agents = []
    for user in users:
        profile = format_user_profile(user)
        agent = AgentState(
            user_id=user['user_id'],
            profile=profile,
            raw_user=user,
            current_beliefs=list(user.get('psychological_beliefs', [])),
            current_opinions=list(user.get('event_opinions', [])),
            current_emotions=dict(user.get('emotion_vector', {})),
        )
        agents.append(agent)
    return agents


def _load_simulation_method(method: str, output_dir: str,
                            num_rounds: int) -> List[List[DecisionResult]]:
    """从磁盘加载某个方法已保存的模拟结果"""
    method_dir = os.path.join(output_dir, 'simulation', method)
    all_decisions_file = os.path.join(method_dir, 'all_decisions.json')
    if not os.path.exists(all_decisions_file):
        return None
    with open(all_decisions_file, 'r', encoding='utf-8') as f:
        decisions_data = json.load(f)
    if not decisions_data:
        return None
    round_results_dict = {}
    for d in decisions_data:
        r_idx = d['round_index']
        if r_idx not in round_results_dict:
            round_results_dict[r_idx] = []
        dr = DecisionResult(
            agent_id=d['agent_id'],
            round_index=r_idx,
            method=d['method'],
            current_time=d.get('current_time', ''),
            actions=d.get('actions', []),
            prompt_used=d.get('prompt_used', ''),
            success=d.get('success', True),
            belief_state=d.get('belief_state'),
            desire_state=d.get('desire_state'),
            cot_analysis=d.get('cot_analysis'),
        )
        round_results_dict[r_idx].append(dr)
    return [round_results_dict.get(i, []) for i in range(num_rounds)]


async def run_simulation(config: ExperimentConfig, methods: List[str],
                         users: List[Dict], round_contexts: List[Dict],
                         nothink_llm: LLMService, think_llm: LLMService,
                         logger) -> Dict[str, Any]:
    """运行所有方法的模拟，已有结果的方法自动跳过"""
    engine = SimulationEngine(nothink_llm, think_llm)
    all_results = {}
    all_agents = {}

    for method in methods:
        agents = initialize_agents(users, method)
        all_agents[method] = agents

        # 检查是否已有该方法的模拟结果
        cached = _load_simulation_method(method, config.output_dir, len(round_contexts))
        if cached is not None:
            logger.info(f"\n[{method}] Simulation results found on disk, skipping.")
            all_results[method] = cached
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Starting simulation: method={method}")
        logger.info(f"{'='*60}")

        start_time = time.time()

        def progress_cb(r_idx, total, m):
            elapsed = time.time() - start_time
            avg_per_round = elapsed / max(r_idx, 1)
            remaining = avg_per_round * (total - r_idx)
            print(f"\r  [{m}] Round {r_idx+1}/{total} "
                  f"| Elapsed: {elapsed:.0f}s | ETA: {remaining:.0f}s", end="", flush=True)

        round_results = await engine.run_method(method, agents, round_contexts, progress_cb)
        print()

        elapsed = time.time() - start_time
        total_actions = sum(
            sum(1 for dr in rr if dr.actions) for rr in round_results
        )
        total_individual_actions = sum(
            sum(len(dr.actions) for dr in rr) for rr in round_results
        )
        success_rate = sum(
            sum(1 for dr in rr if dr.success) for rr in round_results
        ) / max(sum(len(rr) for rr in round_results), 1)

        logger.info(f"[{method}] Completed in {elapsed:.1f}s")
        logger.info(f"  Total agent-rounds with actions: {total_actions}")
        logger.info(f"  Total individual actions: {total_individual_actions}")
        logger.info(f"  LLM success rate: {success_rate:.1%}")

        all_results[method] = round_results

        # 立即保存该方法的模拟结果
        save_simulation_results({method: round_results}, config.output_dir)
        logger.info(f"  [SAVED] Simulation results for {method}")

        if method == "direct_think":
            stats = think_llm.get_stats()
        else:
            stats = nothink_llm.get_stats()
        logger.info(f"  LLM calls: {stats['calls']}, errors: {stats['errors']}, "
                     f"avg_time: {stats['avg_time']:.2f}s")

    return all_results, all_agents


def _check_validation_cache(output_dir: str, val_type: str, method: str):
    """检查某项验证结果是否已存在"""
    filepath = os.path.join(output_dir, 'validation', val_type, f'{method}.json')
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if data and not data.get('error'):
                return data
        except (json.JSONDecodeError, IOError):
            pass
    return None


def _save_single_validation(result: Dict, output_dir: str,
                            val_type: str, method: str):
    """保存单个方法的单项验证结果"""
    val_dir = os.path.join(output_dir, 'validation', val_type)
    os.makedirs(val_dir, exist_ok=True)
    filepath = os.path.join(val_dir, f'{method}.json')
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)


async def run_validation(config: ExperimentConfig, methods: List[str],
                         all_results: Dict, all_agents: Dict,
                         round_contexts: List[Dict],
                         nothink_llm: LLMService, think_llm: LLMService,
                         eval_llm: LLMService, logger) -> Dict[str, Any]:
    """运行所有方法的验证，每完成一项立即保存，已有结果自动跳过"""
    behavior_validator = BehaviorConsistencyValidator(eval_llm)
    personality_validator = PersonalityStabilityValidator(eval_llm)
    heterogeneity_validator = BehaviorHeterogeneityValidator()

    all_behavior = {}
    all_personality = {}
    all_robustness = {}
    all_heterogeneity = {}

    for method in methods:
        agents = all_agents[method]
        round_results = all_results[method]

        logger.info(f"\n{'='*60}")
        logger.info(f"Starting validation: method={method}")
        logger.info(f"{'='*60}")

        # 1. Behavior Consistency
        cached = _check_validation_cache(config.output_dir, 'behavior_consistency', method)
        if cached:
            logger.info(f"  [1/4] Behavior Consistency: CACHED, skipping.")
            bc_result = cached
        else:
            logger.info(f"  [1/4] Behavior Consistency...")
            start = time.time()
            bc_result = await behavior_validator.validate(method, agents, round_results)
            logger.info(f"  Completed in {time.time()-start:.1f}s")
            _save_single_validation(bc_result, config.output_dir, 'behavior_consistency', method)
            logger.info(f"  [SAVED] behavior_consistency/{method}.json")

        dim_stats = bc_result.get('dimension_stats', {})
        for dim, stats in dim_stats.items():
            if isinstance(stats, dict) and 'mean' in stats:
                logger.info(f"    {dim}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, "
                             f"median={stats.get('median', 0):.1f}, "
                             f"[{stats.get('min', 0):.0f}-{stats.get('max', 0):.0f}]")
        all_behavior[method] = bc_result

        # 2. Personality Stability
        cached = _check_validation_cache(config.output_dir, 'personality_stability', method)
        if cached:
            logger.info(f"  [2/4] Personality Stability: CACHED, skipping.")
            ps_result = cached
        else:
            logger.info(f"  [2/4] Personality Stability...")
            start = time.time()
            ps_result = await personality_validator.validate(method, agents, round_results)
            logger.info(f"  Completed in {time.time()-start:.1f}s")
            _save_single_validation(ps_result, config.output_dir, 'personality_stability', method)
            logger.info(f"  [SAVED] personality_stability/{method}.json")

        dim_stats = ps_result.get('dimension_stats', {})
        for dim, stats in dim_stats.items():
            if isinstance(stats, dict) and 'mean' in stats:
                logger.info(f"    {dim}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
        all_personality[method] = ps_result

        # 3. Prompt Robustness (LLM-evaluated cognitive robustness)
        cached = _check_validation_cache(config.output_dir, 'prompt_robustness', method)
        if cached:
            logger.info(f"  [3/4] Prompt Robustness: CACHED, skipping.")
            pr_result = cached
        else:
            logger.info(f"  [3/4] Prompt Robustness...")
            start = time.time()
            rob_llm = think_llm if method == "direct_think" else nothink_llm
            rob_validator = PromptRobustnessValidator(rob_llm, eval_llm)
            pr_result = await rob_validator.validate(
                method, agents, round_results, round_contexts,
                num_sample_decisions=config.robustness_decision_samples,
                num_repeats=config.robustness_repeat_count,
            )
            logger.info(f"  Completed in {time.time()-start:.1f}s")
            _save_single_validation(pr_result, config.output_dir, 'prompt_robustness', method)
            logger.info(f"  [SAVED] prompt_robustness/{method}.json")

        dim_stats = pr_result.get('dimension_stats', {})
        for dim, stats in dim_stats.items():
            if isinstance(stats, dict) and 'mean' in stats:
                logger.info(f"    {dim}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
        all_robustness[method] = pr_result

        # 4. Behavior Heterogeneity (computed from data, no LLM)
        cached = _check_validation_cache(config.output_dir, 'behavior_heterogeneity', method)
        if cached:
            logger.info(f"  [4/4] Behavior Heterogeneity: CACHED, skipping.")
            het_result = cached
        else:
            logger.info(f"  [4/4] Behavior Heterogeneity...")
            het_result = heterogeneity_validator.validate(method, agents, round_results)
            _save_single_validation(het_result, config.output_dir, 'behavior_heterogeneity', method)
            logger.info(f"  [SAVED] behavior_heterogeneity/{method}.json")

        metrics = het_result.get('metrics', {})
        for name, data in metrics.items():
            if isinstance(data, dict) and 'value' in data:
                logger.info(f"    {name}: {data['value']:.4f}")
        all_heterogeneity[method] = het_result

    return all_behavior, all_personality, all_robustness, all_heterogeneity


def generate_comparison_report(all_behavior: Dict, all_personality: Dict,
                               all_robustness: Dict, all_heterogeneity: Dict,
                               output_dir: str, experiment_info: Dict):
    """生成方法对比报告"""
    report = {
        'experiment_info': experiment_info,
        'comparison': {},
    }

    methods = set(
        list(all_behavior.keys()) + list(all_personality.keys()) +
        list(all_robustness.keys()) + list(all_heterogeneity.keys())
    )

    def _extract_full_stats(stats: dict) -> dict:
        """提取完整统计量"""
        return {
            'mean': stats.get('mean', 0),
            'std': stats.get('std', 0),
            'median': stats.get('median', 0),
            'min': stats.get('min', 0),
            'max': stats.get('max', 0),
            'q25': stats.get('q25', 0),
            'q75': stats.get('q75', 0),
            'count': stats.get('count', 0),
            'ci95_lower': stats.get('ci95_lower', 0),
            'ci95_upper': stats.get('ci95_upper', 0),
        }

    for method in sorted(methods):
        method_report = {}

        bc = all_behavior.get(method, {})
        bc_dims = bc.get('dimension_stats', {})
        if bc_dims:
            method_report['behavior_consistency'] = {
                dim: _extract_full_stats(stats)
                for dim, stats in bc_dims.items() if 'mean' in stats
            }

        ps = all_personality.get(method, {})
        ps_dims = ps.get('dimension_stats', {})
        if ps_dims:
            method_report['personality_stability'] = {
                dim: _extract_full_stats(stats)
                for dim, stats in ps_dims.items() if 'mean' in stats
            }

        pr = all_robustness.get(method, {})
        pr_dims = pr.get('dimension_stats', {})
        if pr_dims:
            method_report['prompt_robustness'] = {
                dim: _extract_full_stats(stats)
                for dim, stats in pr_dims.items() if 'mean' in stats
            }

        het = all_heterogeneity.get(method, {})
        het_metrics = het.get('metrics', {})
        if het_metrics:
            method_report['behavior_heterogeneity'] = {
                name: data for name, data in het_metrics.items()
            }
            method_report['behavior_heterogeneity']['overall'] = het.get('overall_heterogeneity', 0)

        report['comparison'][method] = method_report

    report_file = os.path.join(output_dir, 'comparison_report.json')
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print_comparison_table(report)
    return report


def print_comparison_table(report: Dict):
    """打印对比表格"""
    comparison = report.get('comparison', {})
    methods = sorted(comparison.keys())

    print("\n" + "=" * 80)
    print("MICRO BEHAVIOR MECHANISM VALIDATION - COMPARISON REPORT")
    print("=" * 80)

    # 1. Behavior consistency (0-5 Likert scale)
    print("\n--- Cognitive-Behavioral Chain Consistency (0-5 Likert scale) ---")
    dims = ['cognitive_chain_completeness', 'emotion_dynamics_rationality',
            'motivation_behavior_alignment', 'information_integration', 'overall']
    header = f"{'Method':<20}" + "".join(f"{d[:15]:<18}" for d in dims)
    print(header)
    print("-" * len(header))
    for method in methods:
        bc = comparison[method].get('behavior_consistency', {})
        row = f"{method:<20}"
        for dim in dims:
            stats = bc.get(dim, {})
            mean = stats.get('mean', 0)
            std = stats.get('std', 0)
            row += f"{mean:.2f}±{std:.2f}      "
        print(row)

    # 2. Personality stability
    print("\n--- Personality Stability (0-1 scale) ---")
    dims = ['identity_similarity', 'belief_similarity', 'tendency_similarity', 'overall_similarity']
    header = f"{'Method':<20}" + "".join(f"{d[:18]:<22}" for d in dims)
    print(header)
    print("-" * len(header))
    for method in methods:
        ps = comparison[method].get('personality_stability', {})
        row = f"{method:<20}"
        for dim in dims:
            stats = ps.get(dim, {})
            mean = stats.get('mean', 0)
            std = stats.get('std', 0)
            row += f"{mean:.3f}±{std:.3f}         "
        print(row)

    # 3. Robustness (LLM-evaluated)
    print("\n--- Cognitive Robustness (LLM-evaluated, 0-1 scale) ---")
    dims = ['core_decision_stability', 'cognitive_anchoring',
            'adaptive_rationality', 'overall_robustness']
    header = f"{'Method':<20}" + "".join(f"{d[:18]:<22}" for d in dims)
    print(header)
    print("-" * len(header))
    for method in methods:
        pr = comparison[method].get('prompt_robustness', {})
        row = f"{method:<20}"
        for d in dims:
            stats = pr.get(d, {})
            mean = stats.get('mean', 0)
            std = stats.get('std', 0)
            row += f"{mean:.3f}±{std:.3f}         "
        print(row)

    # 4. Heterogeneity
    print("\n--- Behavior Heterogeneity (higher = more diverse) ---")
    het_metrics = ['action_type_entropy', 'stance_diversity', 'emotion_diversity',
                   'participation_gini', 'content_lexical_diversity', 'temporal_variability', 'overall']
    header = f"{'Method':<20}" + "".join(f"{m[:14]:<16}" for m in het_metrics)
    print(header)
    print("-" * len(header))
    for method in methods:
        het = comparison[method].get('behavior_heterogeneity', {})
        row = f"{method:<20}"
        for m in het_metrics:
            val = het.get(m, 0)
            if isinstance(val, dict):
                val = val.get('value', 0)
            row += f"{val:.4f}         "
        print(row)

    print("\n" + "=" * 80)


def _resolve_output_dir(config: ExperimentConfig, resume_dir: str = None) -> str:
    """确定输出目录：resume已有目录或创建新目录"""
    if resume_dir:
        if os.path.isabs(resume_dir):
            return resume_dir
        base = os.path.dirname(config.output_dir.rstrip('/\\'))
        candidate = os.path.join(base, resume_dir)
        if os.path.isdir(candidate):
            return candidate
        if os.path.isdir(resume_dir):
            return os.path.abspath(resume_dir)
        raise FileNotFoundError(f"Resume directory not found: {resume_dir}")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(config.output_dir, f'run_{timestamp}')


async def main_async(args):
    config = ExperimentConfig.from_project_config(args.config, args.output)

    if args.num_agents:
        config.num_agents = args.num_agents
    if args.methods:
        config.methods = args.methods

    config.output_dir = _resolve_output_dir(config, getattr(args, 'resume', None))
    os.makedirs(config.output_dir, exist_ok=True)

    is_resume = getattr(args, 'resume', None) is not None

    logger = setup_logging(config.output_dir, config.log_level)
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Resume mode: {is_resume}")
    logger.info(f"Methods: {config.methods}")
    logger.info(f"Agents: {config.num_agents}")

    experiment_info = {
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'methods': config.methods,
        'num_agents': config.num_agents,
        'event_background': config.event_background,
        'nothink_model': config.nothink_model.model,
        'think_model': config.think_model.model,
        'robustness_samples': config.robustness_decision_samples,
        'robustness_repeats': config.robustness_repeat_count,
    }

    # Load data
    logger.info("Loading data...")
    users = load_users(config.data_dir, config.num_agents)
    broadcasts, node_posts = load_events(config.data_dir)
    round_contexts = build_round_contexts(broadcasts, node_posts, config.event_background)
    config.num_rounds = len(round_contexts)

    experiment_info['num_rounds'] = config.num_rounds
    experiment_info['num_loaded_agents'] = len(users)

    logger.info(f"Loaded {len(users)} agents, {len(round_contexts)} rounds")

    nothink_llm, think_llm, eval_llm = create_llm_services(config)

    all_results = {}
    all_agents = {}

    # Phase 1: Simulation (with per-method checkpoint)
    if not args.skip_simulation:
        logger.info("\n" + "#" * 60)
        logger.info("PHASE 1: SIMULATION")
        logger.info("#" * 60)

        sim_start = time.time()
        all_results, all_agents = await run_simulation(
            config, config.methods, users, round_contexts,
            nothink_llm, think_llm, logger
        )
        sim_elapsed = time.time() - sim_start
        experiment_info['simulation_time'] = sim_elapsed
        logger.info(f"\nSimulation phase completed in {sim_elapsed:.1f}s")
    else:
        logger.info("Skipping simulation phase, will load from disk as needed.")

    # 确保所有方法都有数据（从磁盘补充）
    for method in config.methods:
        if method not in all_agents:
            all_agents[method] = initialize_agents(users, method)
        if method not in all_results:
            cached = _load_simulation_method(method, config.output_dir, config.num_rounds)
            if cached:
                all_results[method] = cached
                logger.info(f"[{method}] Loaded simulation from disk.")
            else:
                logger.warning(f"[{method}] No simulation data available, skipping.")

    # Phase 2: Validation (with per-method-per-metric checkpoint)
    if not args.skip_validation and all_results:
        logger.info("\n" + "#" * 60)
        logger.info("PHASE 2: VALIDATION")
        logger.info("#" * 60)

        methods_with_data = [m for m in config.methods if m in all_results]
        val_start = time.time()
        all_behavior, all_personality, all_robustness, all_heterogeneity = await run_validation(
            config, methods_with_data, all_results, all_agents, round_contexts,
            nothink_llm, think_llm, eval_llm, logger
        )
        val_elapsed = time.time() - val_start
        experiment_info['validation_time'] = val_elapsed
        logger.info(f"\nValidation phase completed in {val_elapsed:.1f}s")

        save_validation_results(all_behavior, config.output_dir, 'behavior_consistency')
        save_validation_results(all_personality, config.output_dir, 'personality_stability')
        save_validation_results(all_robustness, config.output_dir, 'prompt_robustness')
        save_validation_results(all_heterogeneity, config.output_dir, 'behavior_heterogeneity')

        report = generate_comparison_report(
            all_behavior, all_personality, all_robustness, all_heterogeneity,
            config.output_dir, experiment_info
        )
    else:
        logger.info("Skipping validation phase.")

    # Save experiment info
    info_file = os.path.join(config.output_dir, 'experiment_info.json')
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(experiment_info, f, ensure_ascii=False, indent=2, default=str)

    for name, llm in [('nothink', nothink_llm), ('think', think_llm), ('eval', eval_llm)]:
        stats = llm.get_stats()
        logger.info(f"LLM [{name}]: calls={stats['calls']}, errors={stats['errors']}, "
                     f"avg_time={stats['avg_time']:.2f}s, total_time={stats['total_time']:.1f}s")

    logger.info(f"\nAll results saved to: {config.output_dir}")
    return config.output_dir


def main():
    parser = argparse.ArgumentParser(description='Micro Behavior Mechanism Validation')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to project config.json')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: auto from config)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from existing run directory (absolute path or run_xxx name)')
    parser.add_argument('--methods', nargs='+', type=str, default=None,
                        choices=['direct_nothink', 'direct_think', 'cot', 'ebdi'],
                        help='Methods to run (default: all)')
    parser.add_argument('--num-agents', type=int, default=None,
                        help='Number of agents (default: 100)')
    parser.add_argument('--skip-simulation', action='store_true',
                        help='Skip simulation phase')
    parser.add_argument('--skip-validation', action='store_true',
                        help='Skip validation phase')

    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == '__main__':
    main()
