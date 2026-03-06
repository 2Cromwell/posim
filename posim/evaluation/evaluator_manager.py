# -*- coding: utf-8 -*-
"""
评估管理器 - 统一调度所有评估模块

负责:
1. 加载模拟数据和真实数据
2. 按类别调度评估器
3. 汇总评估结果
4. 生成综合报告
"""
import json
import logging
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from .data_loader import SimulationDataLoader, RealDataLoader
from .utils import save_json

# 评估器导入
from .mechanism.agent_behavior import AgentBehaviorEvaluator
from .mechanism.macro_phenomenon import MacroPhenomenonEvaluator
from .mechanism.opinion_polarization import OpinionPolarizationEvaluator
from .mechanism.lifecycle import LifecycleEvaluator
from .mechanism.propagation_structure import PropagationStructureEvaluator
from .calibration.hotness import HotnessCalibrationEvaluator
from .calibration.emotion import EmotionCalibrationEvaluator
from .calibration.topic import TopicCalibrationEvaluator
from .calibration.opinion_index import OpinionIndexEvaluator
from .calibration.network import NetworkCalibrationEvaluator
from .calibration.behavior import BehaviorCalibrationEvaluator

logger = logging.getLogger(__name__)


class EvaluationManager:
    """
    评估管理器 - 总调度器
    
    输出目录结构:
    vis_results/
    ├── mechanism/              # 机制验证
    │   ├── agent_behavior/     # 智能体行为机制
    │   └── macro_phenomenon/   # 宏观现象机制
    ├── calibration/            # 真实数据校准
    │   ├── hotness_calibration/    # 热度曲线校准
    │   ├── emotion_calibration/    # 情绪情感校准
    │   ├── topic_calibration/      # 话题演化校准
    │   ├── opinion_index/          # 舆情演化指数
    │   ├── network_calibration/    # 网络拓扑校准
    │   └── behavior_calibration/   # 行为分布校准
    └── evaluation_report.json  # 综合评估报告
    """
    
    def __init__(self, sim_results_dir: str, 
                 real_data_path: Optional[str] = None,
                 base_data_path: Optional[str] = None,
                 output_dir: Optional[str] = None,
                 time_granularity: int = 10,
                 time_start: Optional[str] = None,
                 time_end: Optional[str] = None):
        """
        Args:
            sim_results_dir: 模拟结果目录（包含 micro_results.json, macro_results.json 等）
            real_data_path: 真实标注数据路径（labels.json 或 base_data.json）
            base_data_path: 原始数据路径（base_data.json，用于网络拓扑等需要完整字段的分析）
            output_dir: 输出目录，默认为 sim_results_dir 的父目录下的 vis_results
            time_granularity: 时间聚合粒度（分钟）
            time_start: 时间过滤起始
            time_end: 时间过滤结束
        """
        self.sim_results_dir = Path(sim_results_dir)
        self.real_data_path = real_data_path
        self.base_data_path = base_data_path
        self.time_granularity = time_granularity
        self.time_start = time_start
        self.time_end = time_end
        
        # 输出目录
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.sim_results_dir.parent / "vis_results"
        
        # 子目录
        self.mechanism_dir = self.output_dir / "mechanism"
        self.calibration_dir = self.output_dir / "calibration"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据
        self.sim_data = None
        self.real_data = None
        
        # 评估结果
        self.results = {}
    
    def load_data(self):
        """加载所有数据"""
        print("\n" + "=" * 60)
        print("📦 加载评估数据")
        print("=" * 60)
        
        # 加载模拟数据
        print(f"\n  加载模拟数据: {self.sim_results_dir}")
        sim_loader = SimulationDataLoader(self.sim_results_dir, self.time_granularity)
        self.sim_data = sim_loader.load()
        
        micro_count = len(self.sim_data.get('micro_results', []))
        print(f"  ✅ 模拟微观行为: {micro_count} 条")
        print(f"  ✅ 时间粒度: {self.time_granularity} 分钟")
        
        agg = self.sim_data.get('aggregated', {})
        times = agg.get('times', [])
        if times:
            print(f"  ✅ 时间范围: {times[0]} ~ {times[-1]}")
            print(f"  ✅ 时间点数: {len(times)}")
        
        # 加载真实数据
        if self.real_data_path and Path(self.real_data_path).exists():
            print(f"\n  加载真实数据: {self.real_data_path}")
            real_loader = RealDataLoader(
                self.real_data_path,
                self.time_granularity,
                self.time_start,
                self.time_end
            )
            self.real_data = real_loader.load()
            
            real_actions = self.real_data.get('actions', [])
            real_times = self.real_data.get('times', [])
            print(f"  ✅ 真实数据行为: {len(real_actions)} 条")
            if real_times:
                print(f"  ✅ 真实时间范围: {real_times[0]} ~ {real_times[-1]}")
        else:
            print("\n  ⚠️ 未提供真实数据路径或文件不存在，将仅进行机制验证")
            self.real_data = None
    
    def run_all(self, api_pool=None, embedding_model=None,
                users_data: List[Dict] = None,
                event_background: str = '',
                skip_mechanism: bool = False,
                skip_calibration: bool = False,
                skip_llm_evaluation: bool = False) -> Dict[str, Any]:
        """
        运行所有评估
        
        Args:
            api_pool: LLM API池（用于智能体行为验证）
            embedding_model: Embedding模型（用于语义相似度计算）
            users_data: 用户数据（用于智能体行为验证）
            event_background: 事件背景描述
            skip_mechanism: 跳过机制验证
            skip_calibration: 跳过真实数据校准
            skip_llm_evaluation: 跳过需要LLM的评估
        """
        if self.sim_data is None:
            self.load_data()
        
        start_time = time.time()
        
        print("\n" + "=" * 60)
        print("🔬 开始评估")
        print("=" * 60)
        
        # ===== 一、机制验证 =====
        if not skip_mechanism:
            print("\n" + "━" * 60)
            print("  ▶ 一、机制验证（仅使用模拟数据）")
            print("━" * 60)
            
            # 1. 智能体行为机制
            if not skip_llm_evaluation:
                try:
                    evaluator = AgentBehaviorEvaluator(self.mechanism_dir)
                    self.results['agent_behavior'] = evaluator.evaluate(
                        self.sim_data,
                        api_pool=api_pool,
                        users_data=users_data or [],
                        event_background=event_background
                    )
                except Exception as e:
                    logger.error(f"智能体行为验证失败: {e}", exc_info=True)
                    print(f"    ❌ 智能体行为验证失败: {e}")
            else:
                print("\n    [SKIP] 智能体行为验证（需要LLM，已跳过）")
            
            # 2. 观点极化现象
            try:
                evaluator = OpinionPolarizationEvaluator(self.mechanism_dir)
                self.results['opinion_polarization'] = evaluator.evaluate(
                    self.sim_data, self.real_data)
            except Exception as e:
                logger.error(f"观点极化验证失败: {e}", exc_info=True)
                print(f"    ❌ 观点极化验证失败: {e}")
            
            # 3. 舆情生命周期
            try:
                evaluator = LifecycleEvaluator(self.mechanism_dir)
                self.results['lifecycle'] = evaluator.evaluate(
                    self.sim_data, self.real_data)
            except Exception as e:
                logger.error(f"生命周期验证失败: {e}", exc_info=True)
                print(f"    ❌ 生命周期验证失败: {e}")
            
            # 4. 传播结构
            try:
                evaluator = PropagationStructureEvaluator(self.mechanism_dir)
                self.results['propagation_structure'] = evaluator.evaluate(
                    self.sim_data, self.real_data,
                    base_data_path=self.base_data_path)
            except Exception as e:
                logger.error(f"传播结构验证失败: {e}", exc_info=True)
                print(f"    ❌ 传播结构验证失败: {e}")
            
            # 5. 宏观现象机制（保留兼容）
            try:
                evaluator = MacroPhenomenonEvaluator(self.mechanism_dir)
                self.results['macro_phenomenon'] = evaluator.evaluate(self.sim_data)
            except Exception as e:
                logger.error(f"宏观现象验证失败: {e}", exc_info=True)
                print(f"    ❌ 宏观现象验证失败: {e}")
        else:
            print("\n  [SKIP] 机制验证（已跳过）")
        
        # ===== 二、真实数据校准 =====
        if not skip_calibration:
            print("\n" + "━" * 60)
            print("  ▶ 二、真实数据校准")
            print("━" * 60)
            
            if self.real_data is None:
                print("\n    ⚠️ 无真实数据，仅生成模拟数据分析")
            
            # 1. 热度曲线校准
            try:
                evaluator = HotnessCalibrationEvaluator(self.calibration_dir)
                self.results['hotness_calibration'] = evaluator.evaluate(
                    self.sim_data, self.real_data)
            except Exception as e:
                logger.error(f"热度校准失败: {e}", exc_info=True)
                print(f"    ❌ 热度校准失败: {e}")
            
            # 2. 情绪情感校准
            try:
                evaluator = EmotionCalibrationEvaluator(self.calibration_dir)
                self.results['emotion_calibration'] = evaluator.evaluate(
                    self.sim_data, self.real_data)
            except Exception as e:
                logger.error(f"情绪校准失败: {e}", exc_info=True)
                print(f"    ❌ 情绪校准失败: {e}")
            
            # 3. 话题演化校准
            try:
                evaluator = TopicCalibrationEvaluator(self.calibration_dir)
                self.results['topic_calibration'] = evaluator.evaluate(
                    self.sim_data, self.real_data)
            except Exception as e:
                logger.error(f"话题校准失败: {e}", exc_info=True)
                print(f"    ❌ 话题校准失败: {e}")
            
            # 4. 舆情演化指数
            try:
                evaluator = OpinionIndexEvaluator(self.calibration_dir)
                self.results['opinion_index'] = evaluator.evaluate(
                    self.sim_data, self.real_data,
                    embedding_model=embedding_model)
            except Exception as e:
                logger.error(f"舆情指数失败: {e}", exc_info=True)
                print(f"    ❌ 舆情指数失败: {e}")
            
            # 5. 网络拓扑结构
            try:
                evaluator = NetworkCalibrationEvaluator(self.calibration_dir)
                self.results['network_calibration'] = evaluator.evaluate(
                    self.sim_data, self.real_data,
                    base_data_path=self.base_data_path,
                    sim_results_dir=str(self.sim_results_dir))
            except Exception as e:
                logger.error(f"网络校准失败: {e}", exc_info=True)
                print(f"    ❌ 网络校准失败: {e}")
            
            # 6. 行为分布校准
            try:
                evaluator = BehaviorCalibrationEvaluator(self.calibration_dir)
                self.results['behavior_calibration'] = evaluator.evaluate(
                    self.sim_data, self.real_data)
            except Exception as e:
                logger.error(f"行为校准失败: {e}", exc_info=True)
                print(f"    ❌ 行为校准失败: {e}")
        else:
            print("\n  [SKIP] 真实数据校准（已跳过）")
        
        # ===== 生成综合报告 =====
        elapsed = time.time() - start_time
        self._generate_report(elapsed)
        
        return self.results
    
    def run_mechanism_only(self, **kwargs) -> Dict[str, Any]:
        """仅运行机制验证"""
        return self.run_all(skip_calibration=True, **kwargs)
    
    def run_calibration_only(self, **kwargs) -> Dict[str, Any]:
        """仅运行真实数据校准"""
        return self.run_all(skip_mechanism=True, **kwargs)
    
    def _generate_report(self, elapsed_time: float):
        """生成综合评估报告"""
        print("\n" + "=" * 60)
        print("📋 综合评估报告")
        print("=" * 60)
        
        report = {
            'metadata': {
                'sim_results_dir': str(self.sim_results_dir),
                'real_data_path': str(self.real_data_path) if self.real_data_path else None,
                'output_dir': str(self.output_dir),
                'time_granularity': self.time_granularity,
                'evaluation_time': datetime.now().isoformat(),
                'elapsed_seconds': round(elapsed_time, 2)
            },
            'data_summary': {
                'sim_actions': len(self.sim_data.get('micro_results', [])) if self.sim_data else 0,
                'real_actions': len(self.real_data.get('actions', [])) if self.real_data else 0,
                'has_real_data': self.real_data is not None
            },
            'results': {}
        }
        
        # 汇总各模块结果摘要
        for module_name, module_results in self.results.items():
            if isinstance(module_results, dict):
                # 提取关键指标
                report['results'][module_name] = self._extract_key_metrics(module_name, module_results)
        
        # 计算综合得分（如果有真实数据校准）
        if self.real_data:
            overall_scores = self._compute_overall_score()
            report['overall_score'] = overall_scores
            
            print(f"\n  综合评估得分:")
            for k, v in overall_scores.items():
                if isinstance(v, (int, float)):
                    print(f"    {k}: {v:.4f}")
        
        # 保存报告
        save_json(self.output_dir / 'evaluation_report.json', report)
        
        print(f"\n  ⏱️ 评估耗时: {elapsed_time:.1f}秒")
        print(f"  📁 结果目录: {self.output_dir}")
        print(f"  📄 综合报告: evaluation_report.json")
        print("=" * 60)
    
    def _extract_key_metrics(self, module_name: str, results: Dict) -> Dict:
        """提取各模块的关键指标"""
        key_metrics = {}

        if module_name == 'agent_behavior':
            summary = results.get('summary', {})
            key_metrics['behavior_avg_score'] = summary.get('behavior_sequence', {}).get('avg_score', None)
            key_metrics['personality_stability'] = summary.get('personality_stability', {}).get('avg_stability', None)
            key_metrics['prompt_consistency'] = summary.get('prompt_stability', {}).get('avg_consistency', None)

        elif module_name == 'opinion_polarization':
            key_metrics['polarization_esteban_ray'] = results.get('esteban_ray_index', None)
            key_metrics['polarization_bimodality'] = results.get('bimodality_coefficient', None)
            key_metrics['polarization_dispersion'] = results.get('stance_dispersion', None)
            key_metrics['non_neutral_ratio'] = results.get('non_neutral_ratio', None)
            key_metrics['confrontation_intensity'] = results.get('confrontation_intensity', None)

        elif module_name == 'lifecycle':
            phases = results.get('lifecycle_phases', {})
            key_metrics['peak_step'] = phases.get('peak_step', None)
            key_metrics['peak_position_ratio'] = phases.get('peak_position_ratio', None)
            gm = results.get('growth_metrics', {})
            key_metrics['growth_rate'] = gm.get('growth_rate', None)
            key_metrics['decay_rate'] = gm.get('decay_rate', None)
            key_metrics['half_life_steps'] = gm.get('half_life_steps', None)
            key_metrics['burstiness'] = gm.get('burstiness', None)

        elif module_name == 'propagation_structure':
            pl = results.get('power_law', {})
            key_metrics['power_law_exponent'] = pl.get('exponent', None)
            key_metrics['power_law_ks'] = pl.get('ks_statistic', None)
            ec = results.get('echo_chamber', {})
            key_metrics['ei_index'] = ec.get('ei_index', None)
            sw = results.get('small_world', {})
            key_metrics['small_world_sigma'] = sw.get('small_world_sigma', None)
            sf = results.get('scale_free', {})
            key_metrics['hub_dominance_top10'] = sf.get('hub_dominance_top10', None)

        elif module_name == 'macro_phenomenon':
            polar = results.get('opinion_polarization', {})
            key_metrics['polarization_index'] = polar.get('polarization_index', None)
            key_metrics['confrontation_intensity'] = polar.get('confrontation_intensity', None)
            lt = results.get('information_longtail', {})
            pareto = lt.get('pareto_analysis', {})
            key_metrics['pareto_ratio'] = pareto.get('top_20pct_action_ratio', None)
            key_metrics['gini_coefficient'] = lt.get('gini_coefficient', None)
            pt = results.get('post_truth', {})
            key_metrics['post_truth_index'] = pt.get('post_truth_index', None)

        elif module_name == 'hotness_calibration':
            curve_sim = results.get('curve_similarity', {}).get('total', {})
            norm = curve_sim.get('normalized', {})
            key_metrics['hotness_pearson'] = norm.get('pearson', None)
            key_metrics['hotness_spearman'] = norm.get('spearman', None)
            # 时序节奏指标
            tr = results.get('temporal_rhythm', {})
            key_metrics['circadian_similarity'] = tr.get('circadian_similarity', None)
            key_metrics['peak_precision'] = tr.get('peak_precision', None)
            key_metrics['lifecycle_phase_similarity'] = tr.get('lifecycle_phase_similarity', None)
            key_metrics['half_life_ratio'] = tr.get('half_life_ratio', None)

        elif module_name == 'emotion_calibration':
            emo = results.get('emotion_curves', {})
            key_metrics['emotion_distribution_similarity'] = emo.get('emotion_distribution_similarity', None)
            sent = results.get('sentiment_curves', {})
            key_metrics['sentiment_similarity'] = sent.get('sentiment_similarity', None)
            pol = results.get('emotion_polarization', {})
            key_metrics['sim_polarization'] = pol.get('sim_polarization', {}).get('polarization_index', None)
            # 情感动态指标
            ed = results.get('emotion_dynamics', {})
            key_metrics['emotion_rank_kendall_tau'] = ed.get('emotion_rank_kendall_tau', None)
            key_metrics['inflection_alignment'] = ed.get('inflection_alignment', None)
            key_metrics['negative_ratio_pearson'] = ed.get('negative_ratio_pearson', None)
            key_metrics['intensity_distribution_similarity'] = ed.get('intensity_distribution_similarity', None)

        elif module_name == 'topic_calibration':
            key_metrics['topic_jaccard'] = results.get('topic_overlap', {}).get('jaccard_similarity', None)
            key_metrics['topic_distribution_similarity'] = results.get('topic_distribution_similarity', None)

        elif module_name == 'opinion_index':
            index = results.get('opinion_evolution_index', {})
            key_metrics['opinion_evolution_index'] = index.get('overall_index', None)
            key_metrics['semantic_similarity'] = results.get('semantic_similarity', {}).get('centroid_similarity', None)
            # 语义内容指标
            sd = results.get('semantic_diversity', {})
            key_metrics['semantic_centroid_similarity'] = sd.get('centroid_similarity', None)
            key_metrics['semantic_diversity_ratio'] = sd.get('diversity_ratio', None)
            sr = results.get('stance_richness', {})
            key_metrics['stance_richness_ratio'] = sr.get('richness_ratio', None)
            ec = results.get('emotion_contagion', {})
            key_metrics['emotion_contagion_similarity'] = ec.get('contagion_similarity', None)

        elif module_name == 'network_calibration':
            net_sim = results.get('network_similarity', {})
            key_metrics['network_similarity'] = net_sim.get('overall_network_similarity', None)
            # 高级网络指标
            dd = results.get('degree_distributions', {})
            key_metrics['in_degree_similarity'] = dd.get('in_degree_similarity', None)
            key_metrics['out_degree_similarity'] = dd.get('out_degree_similarity', None)
            rec = results.get('reciprocity', {})
            key_metrics['reciprocity_similarity'] = rec.get('reciprocity_similarity', None)
            cs = results.get('cascade_structure', {})
            key_metrics['cascade_scale_similarity'] = cs.get('cascade_scale_similarity', None)
            kn = results.get('key_node_influence', {})
            key_metrics['key_node_influence_similarity'] = kn.get('key_node_overall_similarity', None)

        elif module_name == 'behavior_calibration':
            td = results.get('type_distribution', {})
            key_metrics['behavior_type_similarity'] = td.get('type_distribution_similarity', None)
            ua = results.get('user_activity', {})
            key_metrics['activity_distribution_similarity'] = ua.get('activity_distribution_similarity', None)
            bi = results.get('behavior_intensity', {})
            if bi.get('intensity_curve_similarity'):
                key_metrics['intensity_pearson'] = bi['intensity_curve_similarity'].get('pearson', None)
            # 行为模式指标
            rd = results.get('role_distribution', {})
            key_metrics['role_distribution_similarity'] = rd.get('role_distribution_similarity', None)
            cr = results.get('co_ratio', {})
            if cr.get('co_ratio_similarity'):
                key_metrics['co_ratio_pearson'] = cr['co_ratio_similarity'].get('pearson', None)
            fs = results.get('first_speak', {})
            key_metrics['first_speak_similarity'] = fs.get('first_speak_similarity', None)
            os_data = results.get('opinion_shift', {})
            key_metrics['opinion_shift_rate'] = os_data.get('sim_shift', {}).get('shift_rate', None)

        return key_metrics
    
    def _compute_overall_score(self) -> Dict[str, float]:
        """计算综合评估得分（含全部新增指标）"""
        scores = {}
        weights = {}

        # ---- 热度曲线 ----
        hotness = self.results.get('hotness_calibration', {})
        total_sim = hotness.get('curve_similarity', {}).get('total', {}).get('normalized', {})
        if total_sim and 'pearson' in total_sim:
            scores['hotness_score'] = float(max(0, total_sim['pearson']))
            weights['hotness_score'] = 0.12

        # 时序节奏
        tr = hotness.get('temporal_rhythm', {})
        temporal_parts = [
            tr.get('circadian_similarity'),
            tr.get('peak_precision'),
            tr.get('lifecycle_phase_similarity'),
            tr.get('half_life_ratio')
        ]
        temporal_valid = [v for v in temporal_parts if v is not None]
        if temporal_valid:
            scores['temporal_rhythm_score'] = float(np.mean(temporal_valid))
            weights['temporal_rhythm_score'] = 0.08

        # ---- 情绪分布 ----
        emotion = self.results.get('emotion_calibration', {})
        emo_sim = emotion.get('emotion_curves', {}).get('emotion_distribution_similarity')
        if emo_sim is not None:
            scores['emotion_score'] = float(emo_sim)
            weights['emotion_score'] = 0.10

        sent_sim = emotion.get('sentiment_curves', {}).get('sentiment_similarity')
        if sent_sim is not None:
            scores['sentiment_score'] = float(sent_sim)
            weights['sentiment_score'] = 0.06

        # 情感动态
        ed = emotion.get('emotion_dynamics', {})
        emo_dyn_parts = [
            ed.get('emotion_rank_kendall_tau'),
            ed.get('inflection_alignment'),
            ed.get('negative_ratio_pearson'),
            ed.get('intensity_distribution_similarity')
        ]
        emo_dyn_valid = [v for v in emo_dyn_parts
                         if v is not None and not (isinstance(v, float) and np.isnan(v))]
        if emo_dyn_valid:
            scores['emotion_dynamics_score'] = float(np.mean([max(0, v) for v in emo_dyn_valid]))
            weights['emotion_dynamics_score'] = 0.06

        # ---- 话题 ----
        topic = self.results.get('topic_calibration', {})
        topic_sim = topic.get('topic_distribution_similarity')
        if topic_sim is not None:
            scores['topic_score'] = float(topic_sim)
            weights['topic_score'] = 0.10

        topic_jaccard = topic.get('topic_overlap', {}).get('jaccard_similarity')
        if topic_jaccard is not None:
            scores['topic_overlap_score'] = float(topic_jaccard)
            weights['topic_overlap_score'] = 0.05

        # ---- 舆情指数 ----
        opinion = self.results.get('opinion_index', {})
        opinion_idx = opinion.get('opinion_evolution_index', {}).get('overall_index')
        if opinion_idx is not None:
            scores['opinion_index_score'] = float(opinion_idx)
            weights['opinion_index_score'] = 0.10

        # 语义内容
        sem_parts = []
        sd = opinion.get('semantic_diversity', {})
        if 'centroid_similarity' in sd:
            sem_parts.append(sd['centroid_similarity'])
        sr = opinion.get('stance_richness', {})
        if 'richness_ratio' in sr:
            sem_parts.append(sr['richness_ratio'])
        ec = opinion.get('emotion_contagion', {})
        if 'contagion_similarity' in ec:
            sem_parts.append(ec['contagion_similarity'])
        if sem_parts:
            scores['semantic_content_score'] = float(np.mean(sem_parts))
            weights['semantic_content_score'] = 0.07

        # ---- 网络 ----
        network = self.results.get('network_calibration', {})
        net_sim = network.get('network_similarity', {}).get('overall_network_similarity')
        if net_sim is not None:
            scores['network_score'] = float(net_sim)
            weights['network_score'] = 0.06

        # 高级网络
        net_adv_parts = []
        dd = network.get('degree_distributions', {})
        if 'in_degree_similarity' in dd:
            net_adv_parts.append(dd['in_degree_similarity'])
        if 'out_degree_similarity' in dd:
            net_adv_parts.append(dd['out_degree_similarity'])
        rec = network.get('reciprocity', {})
        if 'reciprocity_similarity' in rec:
            net_adv_parts.append(rec['reciprocity_similarity'])
        kn = network.get('key_node_influence', {})
        if 'key_node_overall_similarity' in kn:
            net_adv_parts.append(kn['key_node_overall_similarity'])
        if net_adv_parts:
            scores['network_advanced_score'] = float(np.mean(net_adv_parts))
            weights['network_advanced_score'] = 0.05

        # ---- 行为分布 ----
        behavior = self.results.get('behavior_calibration', {})
        beh_type_sim = behavior.get('type_distribution', {}).get('type_distribution_similarity')
        beh_act_sim = behavior.get('user_activity', {}).get('activity_distribution_similarity')
        if beh_type_sim is not None:
            scores['behavior_type_score'] = float(beh_type_sim)
            weights['behavior_type_score'] = 0.05
        if beh_act_sim is not None:
            scores['behavior_activity_score'] = float(beh_act_sim)
            weights['behavior_activity_score'] = 0.04

        # 行为模式
        beh_pattern_parts = []
        rd = behavior.get('role_distribution', {})
        if 'role_distribution_similarity' in rd:
            beh_pattern_parts.append(rd['role_distribution_similarity'])
        cr = behavior.get('co_ratio', {})
        if cr.get('co_ratio_similarity', {}).get('pearson') is not None:
            beh_pattern_parts.append(max(0, cr['co_ratio_similarity']['pearson']))
        fs = behavior.get('first_speak', {})
        if 'first_speak_similarity' in fs:
            beh_pattern_parts.append(fs['first_speak_similarity'])
        if beh_pattern_parts:
            scores['behavior_pattern_score'] = float(np.mean(beh_pattern_parts))
            weights['behavior_pattern_score'] = 0.06

        # ---- 加权综合 ----
        if scores and weights:
            total_weight = sum(weights.values())
            overall = sum(scores.get(k, 0) * weights.get(k, 0) for k in scores) / max(total_weight, 0.01)
            scores['overall_score'] = float(overall)
            scores['weights'] = weights

        return scores
