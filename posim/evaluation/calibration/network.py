# -*- coding: utf-8 -*-
"""
网络拓扑结构校准

计算模拟和真实的转发网络的各类结构指标相似性:
- 度分布
- 聚类系数
- 连通分量
- 中心性分析
- 网络密度
"""
import json
import logging
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional, List
from collections import defaultdict, Counter

from ..base import BaseEvaluator
from ..utils import (
    save_json, calculate_jsd, calculate_gini_coefficient,
    fit_power_law_exponent
)
from ..visualization import (
    FIG_SIZE, FIG_SIZE_TALL, FIG_SIZE_WIDE, LW, LW_MINOR,
    MARKER_SIZE, FONT_SIZE,
    C_SIM, C_REAL, add_grid, add_legend, save_figure
)

logger = logging.getLogger(__name__)


class NetworkCalibrationEvaluator(BaseEvaluator):
    """网络拓扑结构校准"""
    
    def __init__(self, output_dir: Path):
        super().__init__(output_dir / "network_calibration", name="network_calibration")
    
    def evaluate(self, sim_data: Dict[str, Any], real_data: Optional[Dict[str, Any]] = None,
                 **kwargs) -> Dict[str, Any]:
        """执行网络拓扑校准
        
        kwargs:
            base_data_path: base_data.json路径，用于构建真实转发网络（含完整的转发链信息）
        """
        self._log_section("网络拓扑结构校准")
        
        micro_results = sim_data.get('micro_results', [])
        
        if not micro_results:
            print("    ⚠️ 无模拟数据，跳过")
            return {}
        
        results = {}
        
        # 确定初始帖子数量（用于过滤初始化数据的交互）
        initial_posts_count = kwargs.get('initial_posts_count', 0)
        if not initial_posts_count:
            # 尝试从sim_results_dir路径推断initial_posts.json位置
            sim_dir = kwargs.get('sim_results_dir', '')
            if sim_dir:
                init_path = Path(sim_dir).parent.parent.parent / 'data' / 'initial_posts.json'
                if init_path.exists():
                    try:
                        import json as _json
                        with open(init_path, 'r', encoding='utf-8') as _f:
                            initial_posts_count = len(_json.load(_f))
                        logger.info(f"从 {init_path} 获取初始帖子数量: {initial_posts_count}")
                    except Exception as e:
                        logger.warning(f"读取initial_posts.json失败: {e}")
        
        # 计算内容来源链（用于网络和级联的过滤）
        valid_targets = self._compute_provenance(micro_results, initial_posts_count)
        
        # 构建模拟转发网络
        print("    构建模拟转发网络...")
        sim_graph = self._build_repost_network(micro_results, source='sim',
                                                initial_posts_count=initial_posts_count,
                                                valid_targets=valid_targets)
        sim_metrics = self._compute_network_metrics(sim_graph)
        results['sim_network'] = sim_metrics
        print(f"      节点数: {sim_metrics.get('node_count', 0)}, 边数: {sim_metrics.get('edge_count', 0)}")
        
        # 构建真实转发网络 - 优先从base_data.json构建（包含完整转发链信息）
        base_data_path = kwargs.get('base_data_path')
        has_real = False
        
        if base_data_path and Path(base_data_path).exists():
            print(f"    从base_data.json构建真实转发网络...")
            real_graph = self._build_repost_network_from_base_data(base_data_path)
            real_metrics = self._compute_network_metrics(real_graph)
            results['real_network'] = real_metrics
            has_real = True
            print(f"      节点数: {real_metrics.get('node_count', 0)}, 边数: {real_metrics.get('edge_count', 0)}")
        elif real_data is not None and real_data.get('actions'):
            print("    从labels数据构建真实转发网络...")
            real_graph = self._build_repost_network_real(real_data.get('actions', []))
            real_metrics = self._compute_network_metrics(real_graph)
            results['real_network'] = real_metrics
            has_real = True
            print(f"      节点数: {real_metrics.get('node_count', 0)}, 边数: {real_metrics.get('edge_count', 0)}")
        
        real_graph_ref = None
        if has_real:
            real_graph_ref = real_graph
            print("    计算网络结构相似度...")
            results['network_similarity'] = self._compute_network_similarity(sim_metrics, real_metrics)

            print("    计算高级网络指标...")
            results['degree_distributions'] = self._analyze_degree_distributions(
                sim_graph, real_graph_ref)
            results['reciprocity'] = self._analyze_reciprocity(
                sim_graph, real_graph_ref)
            results['key_node_influence'] = self._analyze_key_node_influence(
                sim_metrics, real_metrics)

        print("    分析信息级联结构...")
        results['cascade_structure'] = self._analyze_cascade_structure(
            micro_results, real_data.get('actions', []) if real_data else [],
            base_data_path, has_real,
            initial_posts_count=initial_posts_count,
            valid_targets=valid_targets)

        self._plot_network(results, has_real)
        if has_real:
            self._plot_network_advanced(results, has_real)

        self._save_results(results, "network_calibration_metrics.json")
        self._print_summary(results)
        return results
    
    def _compute_provenance(self, micro_results, initial_posts_count):
        """计算内容来源链，返回有效目标帖子ID集合
        
        通过重建 post_id 分配追踪每个模拟帖子的内容来源：
        - 模拟原创帖(short_post/long_post) → 'original'
        - 转发初始帖子 → 'initial_derived'（含传递链）
        - 转发模拟原创内容 → 'sim_derived'
        
        Returns: set of valid target post_ids (original + sim_derived)
        """
        if not initial_posts_count:
            return set()
        
        N = initial_posts_count
        POST_CREATING = {'short_post', 'long_post', 'repost', 'repost_comment'}
        
        post_actions = []
        for i, a in enumerate(micro_results):
            if a.get('action_type', '') in POST_CREATING:
                post_actions.append((i, a))
        post_actions.sort(key=lambda x: (x[1].get('time', ''), x[0]))
        
        provenance = {}
        next_id = N + 1
        
        for idx, action in post_actions:
            pid = f"post_{next_id}"
            at = action.get('action_type', '')
            
            if at in ('short_post', 'long_post'):
                provenance[pid] = 'original'
            elif at in ('repost', 'repost_comment'):
                target = action.get('target_post_id', '')
                if target and target.startswith('post_'):
                    try:
                        tnum = int(target.split('_')[1])
                    except (ValueError, IndexError):
                        provenance[pid] = 'unknown'
                        next_id += 1
                        continue
                    if tnum <= N:
                        provenance[pid] = 'initial_derived'
                    elif target in provenance:
                        provenance[pid] = ('initial_derived'
                                           if provenance[target] == 'initial_derived'
                                           else 'sim_derived')
                    else:
                        provenance[pid] = 'unknown'
                else:
                    provenance[pid] = 'unknown'
            
            next_id += 1
        
        valid = {pid for pid, prov in provenance.items()
                 if prov in ('original', 'sim_derived')}
        
        from collections import Counter
        prov_counts = Counter(provenance.values())
        logger.info(f"内容来源链: original={prov_counts.get('original',0)}, "
                    f"sim_derived={prov_counts.get('sim_derived',0)}, "
                    f"initial_derived={prov_counts.get('initial_derived',0)}, "
                    f"unknown={prov_counts.get('unknown',0)}, "
                    f"有效目标帖数={len(valid)}")
        return valid
    
    def _build_repost_network(self, micro_results, source='sim', initial_posts_count=0,
                              valid_targets=None):
        """从模拟结果构建转发网络（基于内容来源链追踪）
        
        当 initial_posts_count > 0 时，仅保留指向 valid_targets 中帖子的交互，
        完全排除初始化数据及其衍生内容的交互。
        """
        try:
            import networkx as nx
        except ImportError:
            logger.warning("networkx未安装，使用简化网络分析")
            return self._build_simple_network(micro_results, source)
        
        # 收集模拟智能体的 user_id 和 username 集合
        sim_agent_ids = set()
        name_to_id = {}
        for a in micro_results:
            uid = a.get('user_id', '')
            uname = a.get('username', '')
            if uid:
                sim_agent_ids.add(uid)
            if uid and uname:
                name_to_id[uname] = uid
        
        # 使用传入的 valid_targets 或空集合
        if valid_targets is None:
            valid_targets = set()
        
        # ---- 构建网络 ----
        G = nx.DiGraph()
        filtered_initial = 0
        filtered_derived = 0
        kept_edges = 0
        
        for a in micro_results:
            action_type = a.get('action_type', '')
            if action_type in ['repost', 'repost_comment', 'short_comment', 'long_comment']:
                src = a.get('user_id', '')
                if not src:
                    continue
                
                target_pid = a.get('target_post_id', '')
                
                if initial_posts_count > 0 and target_pid:
                    if target_pid.startswith('post_'):
                        try:
                            num = int(target_pid.split('_')[1])
                        except (ValueError, IndexError):
                            continue
                        if num <= initial_posts_count:
                            filtered_initial += 1
                            continue
                    # 检查是否为初始化内容的衍生帖子
                    if target_pid not in valid_targets:
                        filtered_derived += 1
                        continue
                
                # 确定目标节点（使用target_author的user_id）
                target_author = a.get('target_author', '')
                target_uid = a.get('target_author_id', '')
                dst = target_uid or name_to_id.get(target_author, target_author)
                
                if src and dst and src != dst:
                    G.add_edge(src, dst)
                    kept_edges += 1
        
        # 添加所有模拟智能体节点
        for uid in sim_agent_ids:
            if not G.has_node(uid):
                G.add_node(uid)
        
        if initial_posts_count > 0:
            logger.info(f"网络构建(来源链过滤): "
                        f"过滤 {filtered_initial} 条→初始帖, "
                        f"{filtered_derived} 条→初始衍生帖, "
                        f"保留 {G.number_of_edges()} 条边 "
                        f"(有效目标帖数: {len(valid_targets)})")
        
        return G
    
    def _build_repost_network_from_base_data(self, base_data_path: str):
        """从base_data.json构建真实转发网络（包含完整的转发链和评论回复信息）"""
        try:
            import networkx as nx
        except ImportError:
            logger.warning("networkx未安装，使用简化网络分析")
            return self._build_simple_network_from_base_data(base_data_path)
        
        with open(base_data_path, 'r', encoding='utf-8') as f:
            base_data = json.load(f)
        
        G = nx.DiGraph()
        
        # 构建 username -> user_id 映射
        username_to_uid = {}
        for u in base_data:
            ui = u.get('user_info', {})
            uname = ui.get('username', '')
            uid = ui.get('user_id', '')
            if uname and uid:
                username_to_uid[uname] = uid
        
        # 同时从URL提取user_id作为补充
        def extract_uid_from_url(url):
            if not url:
                return ''
            match = re.search(r'weibo\.com/(\d+)/', url)
            return match.group(1) if match else ''
        
        edge_count = 0
        for u in base_data:
            ui = u.get('user_info', {})
            uid = ui.get('user_id', '')
            if not uid:
                continue
            
            G.add_node(uid)
            
            # 转发关系：user -> root_author
            for r in u.get('repost_posts', []):
                root_author = r.get('root_author', '')
                target_uid = username_to_uid.get(root_author, '')
                if target_uid and target_uid != uid:
                    G.add_edge(uid, target_uid)
                    edge_count += 1
                
                # 转发链中的中间节点
                for chain_item in r.get('repost_chain', []):
                    chain_author = chain_item.get('author', '')
                    chain_uid = username_to_uid.get(chain_author, '')
                    if chain_uid and chain_uid != uid:
                        G.add_edge(uid, chain_uid)
                        edge_count += 1
            
            # 评论关系
            for c in u.get('comments', []):
                # 评论回复关系：user -> replied_to_user
                replied_to = c.get('replied_to_user', '')
                if replied_to:
                    target_uid = username_to_uid.get(replied_to, '')
                    if target_uid and target_uid != uid:
                        G.add_edge(uid, target_uid)
                        edge_count += 1
                
                # 评论-原帖作者关系：user -> original_post_author
                orig_author = c.get('original_post_author', '')
                if orig_author:
                    target_uid = username_to_uid.get(orig_author, '')
                    if target_uid and target_uid != uid:
                        G.add_edge(uid, target_uid)
                        edge_count += 1
        
        logger.info(f"从base_data.json构建网络: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边 (原始边数{edge_count})")
        return G
    
    def _build_simple_network_from_base_data(self, base_data_path: str):
        """简化版base_data网络构建（不使用networkx）"""
        with open(base_data_path, 'r', encoding='utf-8') as f:
            base_data = json.load(f)
        
        username_to_uid = {}
        for u in base_data:
            ui = u.get('user_info', {})
            uname = ui.get('username', '')
            uid = ui.get('user_id', '')
            if uname and uid:
                username_to_uid[uname] = uid
        
        nodes = set()
        edges = []
        
        for u in base_data:
            uid = u.get('user_info', {}).get('user_id', '')
            if not uid:
                continue
            nodes.add(uid)
            
            for r in u.get('repost_posts', []):
                target_uid = username_to_uid.get(r.get('root_author', ''), '')
                if target_uid and target_uid != uid:
                    edges.append((uid, target_uid))
                    nodes.add(target_uid)
            
            for c in u.get('comments', []):
                replied_to = c.get('replied_to_user', '')
                if replied_to:
                    target_uid = username_to_uid.get(replied_to, '')
                    if target_uid and target_uid != uid:
                        edges.append((uid, target_uid))
                        nodes.add(target_uid)
                orig_author = c.get('original_post_author', '')
                if orig_author:
                    target_uid = username_to_uid.get(orig_author, '')
                    if target_uid and target_uid != uid:
                        edges.append((uid, target_uid))
                        nodes.add(target_uid)
        
        return {'nodes': nodes, 'edges': edges, 'type': 'simple'}
    
    def _build_repost_network_real(self, actions):
        """从真实数据(labels格式)构建转发网络"""
        try:
            import networkx as nx
        except ImportError:
            return self._build_simple_network_real(actions)
        
        G = nx.DiGraph()
        
        for a in actions:
            atype = a.get('type', '')
            user_id = a.get('user_id', '')
            
            if not user_id:
                continue
            
            G.add_node(user_id)
            
            if atype in ['repost', 'comment']:
                # 尝试获取目标用户
                target = a.get('target_user_id', a.get('parent_user_id', ''))
                if target and target != user_id:
                    G.add_edge(user_id, target)
        
        return G
    
    def _build_simple_network(self, data, source='sim'):
        """简化网络（不使用networkx）"""
        nodes = set()
        edges = []
        
        for a in data:
            src = a.get('user_id', '')
            if src:
                nodes.add(src)
            action_type = a.get('action_type', '')
            if action_type in ['repost', 'repost_comment', 'short_comment', 'long_comment']:
                dst = a.get('target_author', '') or a.get('target_author_id', '')
                if src and dst and src != dst:
                    edges.append((src, dst))
                    nodes.add(dst)
        
        return {'nodes': nodes, 'edges': edges, 'type': 'simple'}
    
    def _build_simple_network_real(self, actions):
        """简化真实网络"""
        nodes = set()
        edges = []
        
        for a in actions:
            uid = a.get('user_id', '')
            if uid:
                nodes.add(uid)
            if a.get('type') in ['repost', 'comment']:
                target = a.get('target_user_id', '')
                if uid and target and uid != target:
                    edges.append((uid, target))
                    nodes.add(target)
        
        return {'nodes': nodes, 'edges': edges, 'type': 'simple'}
    
    def _compute_network_metrics(self, graph) -> Dict:
        """计算网络指标"""
        try:
            import networkx as nx
            if isinstance(graph, nx.DiGraph) or isinstance(graph, nx.Graph):
                return self._compute_nx_metrics(graph)
        except ImportError:
            pass
        
        # 简化版本
        if isinstance(graph, dict) and graph.get('type') == 'simple':
            return self._compute_simple_metrics(graph)
        
        return {}
    
    def _compute_nx_metrics(self, G) -> Dict:
        """使用networkx计算网络指标"""
        import networkx as nx
        
        metrics = {
            'node_count': G.number_of_nodes(),
            'edge_count': G.number_of_edges(),
            'density': float(nx.density(G)),
        }
        
        if G.number_of_nodes() == 0:
            return metrics
        
        # 度分布
        in_degrees = [d for _, d in G.in_degree()]
        out_degrees = [d for _, d in G.out_degree()]
        degrees = [d for _, d in G.degree()]
        
        metrics['degree_stats'] = {
            'avg_degree': float(np.mean(degrees)),
            'max_degree': int(max(degrees)) if degrees else 0,
            'avg_in_degree': float(np.mean(in_degrees)),
            'avg_out_degree': float(np.mean(out_degrees)),
            'degree_gini': float(calculate_gini_coefficient(sorted(degrees)))
        }
        
        # 度分布
        degree_dist = Counter(degrees)
        metrics['degree_distribution'] = {str(k): v for k, v in sorted(degree_dist.items())}
        
        # 聚类系数
        try:
            undirected = G.to_undirected()
            metrics['clustering_coefficient'] = float(nx.average_clustering(undirected))
        except:
            metrics['clustering_coefficient'] = 0.0
        
        # 连通分量
        try:
            undirected = G.to_undirected()
            components = list(nx.connected_components(undirected))
            metrics['connected_components'] = len(components)
            metrics['largest_component_size'] = len(max(components, key=len)) if components else 0
            metrics['largest_component_ratio'] = float(
                metrics['largest_component_size'] / max(G.number_of_nodes(), 1)
            )
        except:
            metrics['connected_components'] = 0
        
        # 中心性（采样计算，避免大图太慢）
        try:
            if G.number_of_nodes() <= 1000:
                bc = nx.betweenness_centrality(G)
                metrics['betweenness_centrality'] = {
                    'avg': float(np.mean(list(bc.values()))),
                    'max': float(max(bc.values())),
                    'gini': float(calculate_gini_coefficient(sorted(bc.values())))
                }
            else:
                # 采样计算
                bc = nx.betweenness_centrality(G, k=min(100, G.number_of_nodes()))
                metrics['betweenness_centrality'] = {
                    'avg': float(np.mean(list(bc.values()))),
                    'max': float(max(bc.values())),
                    'gini': float(calculate_gini_coefficient(sorted(bc.values())))
                }
        except:
            pass
        
        # PageRank
        try:
            pr = nx.pagerank(G)
            pr_values = sorted(pr.values(), reverse=True)
            metrics['pagerank'] = {
                'max': float(max(pr_values)),
                'avg': float(np.mean(pr_values)),
                'gini': float(calculate_gini_coefficient(sorted(pr_values))),
                'top10_share': float(sum(pr_values[:10]) / max(sum(pr_values), 1e-10))
            }
        except:
            pass
        
        return metrics
    
    def _compute_simple_metrics(self, graph) -> Dict:
        """简化版网络指标"""
        nodes = graph.get('nodes', set())
        edges = graph.get('edges', [])
        
        n = len(nodes)
        m = len(edges)
        
        metrics = {
            'node_count': n,
            'edge_count': m,
            'density': float(m / max(n * (n - 1), 1)) if n > 1 else 0
        }
        
        # 度分布
        in_deg = Counter()
        out_deg = Counter()
        for src, dst in edges:
            out_deg[src] += 1
            in_deg[dst] += 1
        
        all_degrees = [in_deg.get(node, 0) + out_deg.get(node, 0) for node in nodes]
        
        if all_degrees:
            metrics['degree_stats'] = {
                'avg_degree': float(np.mean(all_degrees)),
                'max_degree': int(max(all_degrees)),
                'degree_gini': float(calculate_gini_coefficient(sorted(all_degrees)))
            }
        
        return metrics
    
    def _compute_network_similarity(self, sim_metrics, real_metrics) -> Dict:
        """计算网络结构相似度（基于结构性指标，避免量依赖）"""
        similarity = {}
        
        # ---- 结构性指标（不受交互量影响） ----
        
        # 1. 聚类系数相似度
        sim_cc = sim_metrics.get('clustering_coefficient', 0)
        real_cc = real_metrics.get('clustering_coefficient', 0)
        similarity['clustering_similarity'] = float(max(0, 1 - abs(sim_cc - real_cc)))
        
        # 2. 度分布不等性相似度（基尼系数）
        sim_gini = sim_metrics.get('degree_stats', {}).get('degree_gini', 0)
        real_gini = real_metrics.get('degree_stats', {}).get('degree_gini', 0)
        similarity['degree_gini_similarity'] = float(max(0, 1 - abs(sim_gini - real_gini)))
        
        # 3. 最大连通分量比例相似度
        sim_lcc = sim_metrics.get('largest_component_ratio', 0)
        real_lcc = real_metrics.get('largest_component_ratio', 0)
        similarity['lcc_similarity'] = float(max(0, 1 - abs(sim_lcc - real_lcc)))
        
        # 4. 幂律指数相似度（尺度不变特征）
        sim_dd = sim_metrics.get('degree_distribution', {})
        real_dd = real_metrics.get('degree_distribution', {})
        sim_pl = self._fit_degree_power_law(sim_dd, sim_metrics.get('node_count', 0))
        real_pl = self._fit_degree_power_law(real_dd, real_metrics.get('node_count', 0))
        if sim_pl is not None and real_pl is not None:
            max_pl = max(abs(sim_pl), abs(real_pl), 1e-10)
            similarity['power_law_exponent_similarity'] = float(max(0, 1 - abs(sim_pl - real_pl) / max_pl))
            similarity['sim_power_law_exp'] = float(sim_pl)
            similarity['real_power_law_exp'] = float(real_pl)
        
        # ---- 保留量指标供参考（不计入综合分） ----
        sim_density = sim_metrics.get('density', 0)
        real_density = real_metrics.get('density', 0)
        similarity['density_diff'] = float(abs(sim_density - real_density))
        similarity['density_similarity'] = float(max(0, 1 - abs(sim_density - real_density) * 100))
        
        sim_avg_deg = sim_metrics.get('degree_stats', {}).get('avg_degree', 0)
        real_avg_deg = real_metrics.get('degree_stats', {}).get('avg_degree', 0)
        max_deg = max(sim_avg_deg, real_avg_deg, 1)
        similarity['avg_degree_similarity'] = float(1 - abs(sim_avg_deg - real_avg_deg) / max_deg)
        
        # ---- 综合相似度（仅基于结构性指标） ----
        structural_components = [
            similarity.get('clustering_similarity', 0),
            similarity.get('degree_gini_similarity', 0),
            similarity.get('lcc_similarity', 0),
        ]
        if 'power_law_exponent_similarity' in similarity:
            structural_components.append(similarity['power_law_exponent_similarity'])
        
        valid = [c for c in structural_components if c > 0]
        similarity['overall_network_similarity'] = float(np.mean(valid)) if valid else 0.0
        
        return similarity
    
    def _fit_degree_power_law(self, degree_dist: Dict, node_count: int) -> Optional[float]:
        """从度分布拟合幂律指数"""
        from ..utils import fit_power_law_exponent
        if not degree_dist or node_count < 10:
            return None
        degrees = []
        for k, v in degree_dist.items():
            try:
                deg = int(k)
                if deg > 0:
                    degrees.extend([deg] * int(v))
            except (ValueError, TypeError):
                continue
        if len(degrees) < 10:
            return None
        return fit_power_law_exponent(degrees)
    
    def _plot_network(self, results, has_real):
        """绘制网络指标对比图"""
        sim_net = results.get('sim_network', {})
        real_net = results.get('real_network', {})
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 基本指标对比
        ax = axes[0, 0]
        metrics_labels = ['Node\nCount', 'Edge\nCount', 'Density\n(x100)']
        sim_vals = [
            sim_net.get('node_count', 0),
            sim_net.get('edge_count', 0),
            sim_net.get('density', 0) * 100
        ]
        
        if has_real:
            real_vals = [
                real_net.get('node_count', 0),
                real_net.get('edge_count', 0),
                real_net.get('density', 0) * 100
            ]
            x = np.arange(len(metrics_labels))
            width = 0.35
            ax.bar(x - width/2, sim_vals, width, label='Simulation', color=C_SIM['total'], alpha=0.8)
            ax.bar(x + width/2, real_vals, width, label='Real Data', color=C_REAL['total'], alpha=0.8)
            ax.set_xticks(x)
        else:
            ax.bar(range(len(metrics_labels)), sim_vals, color=C_SIM['total'], alpha=0.8)
        ax.set_xticklabels(metrics_labels)
        ax.set_ylabel('Value')
        ax.set_title('Network Basic Metrics', fontweight='bold')
        add_legend(ax)
        add_grid(ax)
        
        # 2. 度分布
        ax = axes[0, 1]
        sim_deg_dist = sim_net.get('degree_distribution', {})
        if sim_deg_dist:
            degs = sorted([int(k) for k in sim_deg_dist.keys()])
            counts = [sim_deg_dist.get(str(d), 0) for d in degs]
            ax.bar(degs, counts, color=C_SIM['total'], alpha=0.7, label='Simulation')
        
        if has_real:
            real_deg_dist = real_net.get('degree_distribution', {})
            if real_deg_dist:
                degs_r = sorted([int(k) for k in real_deg_dist.keys()])
                counts_r = [real_deg_dist.get(str(d), 0) for d in degs_r]
                ax.bar([d + 0.3 for d in degs_r], counts_r, 0.3,
                       color=C_REAL['total'], alpha=0.7, label='Real Data')
        
        ax.set_xlabel('Degree')
        ax.set_ylabel('Count')
        ax.set_title('Degree Distribution', fontweight='bold')
        add_legend(ax)
        add_grid(ax)
        
        # 3. 结构指标雷达图
        ax = axes[1, 0]
        labels_radar = ['Density', 'Avg Degree', 'Clustering', 'Degree Gini', 'LCC Ratio']
        sim_radar = [
            min(sim_net.get('density', 0) * 10, 1),
            min(sim_net.get('degree_stats', {}).get('avg_degree', 0) / 10, 1),
            sim_net.get('clustering_coefficient', 0),
            sim_net.get('degree_stats', {}).get('degree_gini', 0),
            sim_net.get('largest_component_ratio', 0)
        ]
        
        if has_real:
            real_radar = [
                min(real_net.get('density', 0) * 10, 1),
                min(real_net.get('degree_stats', {}).get('avg_degree', 0) / 10, 1),
                real_net.get('clustering_coefficient', 0),
                real_net.get('degree_stats', {}).get('degree_gini', 0),
                real_net.get('largest_component_ratio', 0)
            ]
        
        x = np.arange(len(labels_radar))
        width = 0.35 if has_real else 0.7
        ax.bar(x - (width/2 if has_real else 0), sim_radar, width,
               label='Simulation', color=C_SIM['total'], alpha=0.8)
        if has_real:
            ax.bar(x + width/2, real_radar, width,
                   label='Real Data', color=C_REAL['total'], alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_radar, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Value (Normalized)')
        ax.set_title('Structural Metrics Comparison', fontweight='bold')
        add_legend(ax)
        add_grid(ax)
        
        # 4. 相似度汇总
        ax = axes[1, 1]
        net_sim = results.get('network_similarity', {})
        if net_sim:
            sim_labels = ['Clustering', 'Degree Gini', 'LCC', 'Power Law', 'Overall']
            sim_values = [
                net_sim.get('clustering_similarity', 0),
                net_sim.get('degree_gini_similarity', 0),
                net_sim.get('lcc_similarity', 0),
                net_sim.get('power_law_exponent_similarity', 0),
                net_sim.get('overall_network_similarity', 0)
            ]
            colors = ['#1f77b4'] * 4 + ['#d62728']
            bars = ax.barh(sim_labels, sim_values, color=colors, alpha=0.8, edgecolor='black')
            for bar, val in zip(bars, sim_values):
                ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                       f'{val:.3f}', ha='left', va='center', fontsize=10)
            ax.set_xlim(0, 1.15)
            ax.set_xlabel('Similarity')
            ax.set_title('Network Structure Similarity', fontweight='bold')
            add_grid(ax)
        else:
            ax.text(0.5, 0.5, 'No real data\nfor comparison', ha='center', va='center',
                   fontsize=14, transform=ax.transAxes)
            ax.set_title('Network Similarity (N/A)', fontweight='bold')
        
        save_figure(fig, self.output_dir / 'network_topology.png')
        logger.info("[SAVED] network_topology.png")
    
    # ─────────────────────────────────────────────────────────
    # 高级网络指标
    # ─────────────────────────────────────────────────────────
    def _analyze_degree_distributions(self, sim_graph, real_graph):
        """入度/出度分布相似度（分别计算）"""
        metrics = {}
        try:
            import networkx as nx
            if not isinstance(sim_graph, (nx.DiGraph, nx.Graph)):
                return metrics
            sim_in = [d for _, d in sim_graph.in_degree()]
            sim_out = [d for _, d in sim_graph.out_degree()]

            if real_graph is not None and isinstance(real_graph, (nx.DiGraph, nx.Graph)):
                real_in = [d for _, d in real_graph.in_degree()]
                real_out = [d for _, d in real_graph.out_degree()]

                for name, sd, rd in [('in_degree', sim_in, real_in), ('out_degree', sim_out, real_out)]:
                    max_d = max(max(sd, default=0), max(rd, default=0)) + 1
                    bins = np.arange(0, min(max_d + 1, 100), 1)
                    sh, _ = np.histogram(sd, bins=bins, density=True)
                    rh, _ = np.histogram(rd, bins=bins, density=True)
                    sh = sh + 1e-10
                    rh = rh + 1e-10
                    sh /= sh.sum()
                    rh /= rh.sum()
                    jsd = calculate_jsd(sh, rh)
                    metrics[f'{name}_jsd'] = float(jsd)
                    metrics[f'{name}_similarity'] = float(1 - jsd)
                    print(f"      {name} 分布相似度: {1 - jsd:.4f}")
        except ImportError:
            pass
        return metrics

    def _analyze_reciprocity(self, sim_graph, real_graph):
        """互动互惠率"""
        metrics = {}
        try:
            import networkx as nx
            for name, G in [('sim', sim_graph), ('real', real_graph)]:
                if G is None or not isinstance(G, (nx.DiGraph, nx.Graph)):
                    continue
                if G.number_of_edges() == 0:
                    metrics[f'{name}_reciprocity'] = 0.0
                    continue
                reciprocated = sum(1 for u, v in G.edges() if G.has_edge(v, u))
                metrics[f'{name}_reciprocity'] = float(reciprocated / max(G.number_of_edges(), 1))

            if 'sim_reciprocity' in metrics and 'real_reciprocity' in metrics:
                diff = abs(metrics['sim_reciprocity'] - metrics['real_reciprocity'])
                metrics['reciprocity_similarity'] = float(max(0, 1 - diff * 5))
                print(f"      互惠率 - 模拟: {metrics['sim_reciprocity']:.4f}, "
                      f"真实: {metrics['real_reciprocity']:.4f}")
        except ImportError:
            pass
        return metrics

    def _analyze_key_node_influence(self, sim_metrics, real_metrics):
        """关键节点影响力分布对比 (PageRank / Betweenness)"""
        metrics = {}
        for metric_name in ['pagerank', 'betweenness_centrality']:
            sm = sim_metrics.get(metric_name, {})
            rm = real_metrics.get(metric_name, {})
            if sm and rm:
                for sub_key in ['avg', 'max', 'gini']:
                    sv = sm.get(sub_key, 0)
                    rv = rm.get(sub_key, 0)
                    max_val = max(abs(sv), abs(rv), 1e-10)
                    sim_val = max(0, 1 - abs(sv - rv) / max_val)
                    metrics[f'{metric_name}_{sub_key}_similarity'] = float(sim_val)
        if metrics:
            vals = [v for v in metrics.values() if isinstance(v, float)]
            metrics['key_node_overall_similarity'] = float(np.mean(vals)) if vals else 0
            print(f"      关键节点影响力综合相似度: {metrics['key_node_overall_similarity']:.4f}")
        return metrics

    def _analyze_cascade_structure(self, micro_results, real_actions, base_data_path, has_real,
                                    initial_posts_count=0, valid_targets=None):
        """信息级联深度/宽度/规模分析
        
        当提供 valid_targets 时，仅统计指向有效目标帖子的级联，
        排除初始化数据及其衍生内容的级联。
        """
        metrics = {}

        def _build_cascades(actions, source='sim', filter_targets=None, N=0):
            """从动作列表构建级联树
            filter_targets: 仅保留这些 target_post_id 的级联（sim模式下使用）
            N: 初始帖子数量阈值
            """
            post_authors = {}
            cascades = defaultdict(list)
            for a in actions:
                atype = a.get('action_type', '') if source == 'sim' else a.get('type', '')
                uid = a.get('user_id', '')
                pid = a.get('post_id', a.get('target_post_id', ''))
                if source == 'sim' and atype in ('short_post', 'long_post'):
                    post_authors[pid] = uid
                elif atype in ('repost', 'repost_comment', 'comment', 'short_comment', 'long_comment'):
                    target_pid = a.get('target_post_id', pid)
                    if target_pid:
                        # 来源链过滤：排除初始帖子和初始衍生帖子
                        if source == 'sim' and filter_targets is not None and N > 0:
                            if target_pid.startswith('post_'):
                                try:
                                    num = int(target_pid.split('_')[1])
                                except (ValueError, IndexError):
                                    continue
                                if num <= N:
                                    continue  # 初始帖子
                            if target_pid not in filter_targets:
                                continue  # 初始衍生帖子
                        cascades[target_pid].append(uid)
            depths = []
            widths = []
            scales = []
            for root_pid, participants in cascades.items():
                scale = len(participants)
                depth = min(scale, 10)
                width = scale
                depths.append(depth)
                widths.append(width)
                scales.append(scale)
            return depths, widths, scales

        sim_depths, sim_widths, sim_scales = _build_cascades(
            micro_results, 'sim',
            filter_targets=valid_targets,
            N=initial_posts_count)
        metrics['sim_cascade_count'] = len(sim_scales)
        if sim_scales:
            metrics['sim_avg_cascade_scale'] = float(np.mean(sim_scales))
            metrics['sim_max_cascade_scale'] = int(max(sim_scales))
            if len(sim_scales) >= 5:
                metrics['sim_cascade_power_law_exp'] = fit_power_law_exponent(sim_scales)

        if has_real and base_data_path and Path(base_data_path).exists():
            real_depths, real_widths, real_scales = self._build_cascades_from_base_data(base_data_path)
            metrics['real_cascade_count'] = len(real_scales)
            if real_scales:
                metrics['real_avg_cascade_scale'] = float(np.mean(real_scales))
                metrics['real_max_cascade_scale'] = int(max(real_scales))
                if len(real_scales) >= 5:
                    metrics['real_cascade_power_law_exp'] = fit_power_law_exponent(real_scales)

            if sim_scales and real_scales:
                max_s = max(max(sim_scales, default=0), max(real_scales, default=0)) + 1
                bins = np.arange(0, min(max_s + 1, 100), 1)
                sh, _ = np.histogram(sim_scales, bins=bins, density=True)
                rh, _ = np.histogram(real_scales, bins=bins, density=True)
                sh = sh + 1e-10
                rh = rh + 1e-10
                sh /= sh.sum()
                rh /= rh.sum()
                jsd = calculate_jsd(sh, rh)
                metrics['cascade_scale_jsd'] = float(jsd)
                metrics['cascade_scale_similarity'] = float(1 - jsd)
                print(f"      级联规模分布相似度: {1 - jsd:.4f}")
                
                # 级联幂律指数相似度（尺度不变结构特征）
                sim_pl = metrics.get('sim_cascade_power_law_exp')
                real_pl = metrics.get('real_cascade_power_law_exp')
                if sim_pl is not None and real_pl is not None:
                    max_pl = max(abs(sim_pl), abs(real_pl), 1e-10)
                    pl_sim = float(max(0, 1 - abs(sim_pl - real_pl) / max_pl))
                    metrics['cascade_power_law_similarity'] = pl_sim
                    print(f"      级联幂律指数相似度: {pl_sim:.4f} "
                          f"(sim={sim_pl:.3f}, real={real_pl:.3f})")
        elif has_real and real_actions:
            real_depths, real_widths, real_scales = _build_cascades(real_actions, 'real')
            metrics['real_cascade_count'] = len(real_scales)

        return metrics

    def _build_cascades_from_base_data(self, base_data_path):
        """从base_data构建级联"""
        depths = []
        widths = []
        scales = []
        try:
            with open(base_data_path, 'r', encoding='utf-8') as f:
                base_data = json.load(f)
            cascade_map = defaultdict(list)
            for u in base_data:
                uid = u.get('user_info', {}).get('user_id', '')
                for r in u.get('repost_posts', []):
                    chain = r.get('repost_chain', [])
                    root = r.get('root_author', '')
                    cascade_key = root or 'unknown'
                    cascade_map[cascade_key].append(uid)
                    depth = len(chain) + 1
                    depths.append(depth)
            for key, participants in cascade_map.items():
                scale = len(participants)
                scales.append(scale)
                widths.append(scale)
        except Exception as e:
            logger.warning(f"级联分析失败: {e}")
        return depths, widths, scales

    def _plot_network_advanced(self, results, has_real):
        """绘制高级网络指标图"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # (a) 入度/出度分布相似度
        ax = axes[0, 0]
        dd = results.get('degree_distributions', {})
        if dd:
            labels = ['In-Degree', 'Out-Degree']
            sims = [dd.get('in_degree_similarity', 0), dd.get('out_degree_similarity', 0)]
            colors = ['#1f77b4', '#ff7f0e']
            bars = ax.bar(labels, sims, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
            for bar, val in zip(bars, sims):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', fontsize=FONT_SIZE['annotation'])
            ax.set_ylim(0, 1.1)
        ax.set_ylabel('Similarity (1-JSD)')
        ax.set_title('(a) Degree Distribution Similarity', fontweight='bold')
        add_grid(ax)

        # (b) 互惠率
        ax = axes[0, 1]
        rec = results.get('reciprocity', {})
        if rec:
            labels = ['Simulation', 'Real Data']
            vals = [rec.get('sim_reciprocity', 0), rec.get('real_reciprocity', 0)]
            colors = [C_SIM['total'], C_REAL['total']]
            bars = ax.bar(labels, vals, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f'{val:.4f}', ha='center', fontsize=FONT_SIZE['annotation'])
        ax.set_ylabel('Reciprocity Rate')
        ax.set_title('(b) Interaction Reciprocity', fontweight='bold')
        add_grid(ax)

        # (c) 级联规模分布
        ax = axes[1, 0]
        cs = results.get('cascade_structure', {})
        sim_s = cs.get('sim_avg_cascade_scale', 0)
        real_s = cs.get('real_avg_cascade_scale', 0)
        labels = ['Simulation', 'Real Data']
        vals = [sim_s, real_s]
        ax.bar(labels, vals, color=[C_SIM['total'], C_REAL['total']], alpha=0.85, edgecolor='black', linewidth=0.5)
        ax.set_ylabel('Average Cascade Scale')
        ax.set_title('(c) Average Cascade Scale', fontweight='bold')
        add_grid(ax)

        # (d) 关键节点影响力
        ax = axes[1, 1]
        kn = results.get('key_node_influence', {})
        if kn:
            metric_names = []
            metric_vals = []
            for k, v in kn.items():
                if k.endswith('_similarity') and k != 'key_node_overall_similarity':
                    metric_names.append(k.replace('_similarity', '').replace('_', '\n'))
                    metric_vals.append(v)
            if metric_names:
                bars = ax.barh(metric_names, metric_vals, color='#1f77b4', alpha=0.85, edgecolor='black', linewidth=0.5)
                for bar, val in zip(bars, metric_vals):
                    ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                            f'{val:.3f}', ha='left', va='center', fontsize=FONT_SIZE['annotation'])
                ax.set_xlim(0, 1.15)
        ax.set_xlabel('Similarity')
        ax.set_title('(d) Key Node Influence Similarity', fontweight='bold')
        add_grid(ax)

        save_figure(fig, self.output_dir / 'network_advanced.png')
        logger.info("[SAVED] network_advanced.png")

    def _print_summary(self, results):
        """打印摘要"""
        sim_net = results.get('sim_network', {})
        if sim_net:
            print(f"    ✅ 模拟网络: {sim_net.get('node_count', 0)} 节点, "
                  f"{sim_net.get('edge_count', 0)} 边, "
                  f"密度={sim_net.get('density', 0):.6f}")
            deg = sim_net.get('degree_stats', {})
            if deg:
                print(f"       平均度={deg.get('avg_degree', 0):.2f}, "
                      f"最大度={deg.get('max_degree', 0)}, "
                      f"基尼={deg.get('degree_gini', 0):.3f}")
        
        net_sim = results.get('network_similarity', {})
        if net_sim:
            print(f"    ✅ 网络结构相似度: {net_sim.get('overall_network_similarity', 0):.4f}")

        dd = results.get('degree_distributions', {})
        if dd:
            print(f"    ✅ 入度分布相似度: {dd.get('in_degree_similarity', 0):.4f}")
            print(f"    ✅ 出度分布相似度: {dd.get('out_degree_similarity', 0):.4f}")

        rec = results.get('reciprocity', {})
        if 'sim_reciprocity' in rec:
            print(f"    ✅ 互惠率(模拟): {rec['sim_reciprocity']:.4f}")

        cs = results.get('cascade_structure', {})
        if cs:
            print(f"    ✅ 级联数: 模拟={cs.get('sim_cascade_count', 0)}")
            if 'cascade_scale_similarity' in cs:
                print(f"       级联规模相似度: {cs['cascade_scale_similarity']:.4f}")

        kn = results.get('key_node_influence', {})
        if 'key_node_overall_similarity' in kn:
            print(f"    ✅ 关键节点影响力相似度: {kn['key_node_overall_similarity']:.4f}")
