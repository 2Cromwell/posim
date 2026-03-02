# -*- coding: utf-8 -*-
"""
验证引擎：3种验证指标的实现

1. 行为序列一致性 - LLM评估行为序列的心理逻辑合理性
2. 人格稳定性 - 从行为反向推断人格并与初始设定比较
3. 提示鲁棒性 - 相同场景多次决策的输出稳定性
"""
import asyncio
import json
import logging
import os
import random
import time
import numpy as np
from typing import List, Dict, Any, Optional
from collections import defaultdict

from .llm_service import LLMService, extract_json_from_response
from .simulation import DecisionResult, AgentState, _format_action_history
from . import prompts

logger = logging.getLogger(__name__)


def _compute_stats(values: List[float]) -> Dict[str, float]:
    """计算详细统计量"""
    if not values:
        return {'count': 0}
    arr = np.array(values)
    result = {
        'count': len(arr),
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'median': float(np.median(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'q25': float(np.percentile(arr, 25)),
        'q75': float(np.percentile(arr, 75)),
    }
    if len(arr) > 1:
        se = float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
        result['ci95_lower'] = result['mean'] - 1.96 * se
        result['ci95_upper'] = result['mean'] + 1.96 * se
    return result


def _format_actions_for_eval(decisions: List[DecisionResult]) -> str:
    """格式化智能体的行为序列用于评估（仅最终行为）"""
    lines = []
    for dr in decisions:
        for act in dr.actions:
            atype = act.get('action_type', 'none')
            emotion = act.get('emotion', '中性')
            intensity = act.get('emotion_intensity', '低')
            stance = act.get('stance', '')
            content = act.get('content', '')[:100]
            line = f"[{dr.current_time}] {atype} | 情绪:{emotion}({intensity})"
            if stance:
                line += f" | 立场:{stance}"
            if content:
                line += f" | 内容:{content}"
            lines.append(line)
    if not lines:
        lines.append("（该智能体未产生任何行为）")
    return "\n".join(lines)


def _format_cognitive_chain(method: str, decisions: List[DecisionResult]) -> str:
    """按方法类型格式化完整认知-行为链（含方法标签）"""
    sections = []

    if method == "ebdi":
        sections.append("【认知架构：EBDI多步认知模型 — 信念更新/欲望推理/行为意图 各模块独立建模、独立推理】\n")
        for dr in decisions:
            round_section = f"=== 第{dr.round_index+1}轮 [{dr.current_time}] ===\n"

            bs = dr.belief_state
            if bs and isinstance(bs, dict):
                round_section += "【独立模块1：信念更新】\n"
                for belief in bs.get('心理认知', []):
                    round_section += f"  - {belief}\n"
                for op in bs.get('事件观点', []):
                    if isinstance(op, dict):
                        round_section += f"  - 关于{op.get('主体', '')}：{op.get('观点', '')}\n"
                emotions = bs.get('情绪向量', {})
                if emotions:
                    top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
                    round_section += f"  情绪向量：{', '.join(f'{k}={v}' for k, v in top_emotions)}\n"

            ds = dr.desire_state
            if ds and isinstance(ds, dict):
                round_section += "【独立模块2：欲望推理】\n"
                for d in ds.get('欲望列表', []):
                    if isinstance(d, dict):
                        round_section += f"  - {d.get('类型', '')}({d.get('强度', '')}): {d.get('描述', '')}\n"

            round_section += "【独立模块3：意图→行为决策】\n"
            if dr.actions:
                for act in dr.actions:
                    atype = act.get('action_type', 'none')
                    emotion = act.get('emotion', '中性')
                    stance = act.get('stance', '')
                    content = act.get('content', '')[:100]
                    round_section += f"  {atype} | 情绪:{emotion}"
                    if stance:
                        round_section += f" | 立场:{stance}"
                    if content:
                        round_section += f" | 内容:{content}"
                    round_section += "\n"
            else:
                round_section += "  （不行动）\n"

            sections.append(round_section)

    elif method == "cot":
        sections.append("【认知架构：单步思维链推理（Chain-of-Thought） — 信念/欲望/行为在单次调用中一并生成】\n")
        for dr in decisions:
            round_section = f"--- 第{dr.round_index+1}轮 [{dr.current_time}] ---\n"

            cot = dr.cot_analysis
            if cot and isinstance(cot, dict):
                ba = cot.get('信念分析', {})
                if isinstance(ba, dict):
                    round_section += "信念分析：\n"
                    if ba.get('事件认知'):
                        round_section += f"  事件认知：{ba['事件认知']}\n"
                    emotions = ba.get('情绪状态', {})
                    if emotions and isinstance(emotions, dict):
                        top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
                        round_section += f"  情绪：{', '.join(f'{k}={v}' for k, v in top_emotions)}\n"
                    if ba.get('立场态度'):
                        round_section += f"  立场：{ba['立场态度']}\n"

                dr_info = cot.get('欲望推理', {})
                if isinstance(dr_info, dict) and dr_info.get('主要欲望'):
                    round_section += f"欲望推理：{dr_info.get('主要欲望', '')}({dr_info.get('欲望强度', '')})\n"

            round_section += "行为输出：\n"
            if dr.actions:
                for act in dr.actions:
                    atype = act.get('action_type', 'none')
                    emotion = act.get('emotion', '中性')
                    stance = act.get('stance', '')
                    content = act.get('content', '')[:100]
                    round_section += f"  {atype} | 情绪:{emotion}"
                    if stance:
                        round_section += f" | 立场:{stance}"
                    if content:
                        round_section += f" | 内容:{content}"
                    round_section += "\n"
            else:
                round_section += "  （不行动）\n"

            sections.append(round_section)

    else:
        sections.append("【认知架构：直接行为决策 — 无显式认知推理过程，仅输出最终行为】\n")
        sections.append("以下为该用户的行为记录（无认知过程数据）：\n")
        for dr in decisions:
            if dr.actions:
                for act in dr.actions:
                    atype = act.get('action_type', 'none')
                    emotion = act.get('emotion', '中性')
                    stance = act.get('stance', '')
                    content = act.get('content', '')[:80]
                    line = f"第{dr.round_index+1}轮[{dr.current_time}]: {atype} | 情绪:{emotion}"
                    if stance:
                        line += f" | 立场:{stance}"
                    if content:
                        line += f" | {content}"
                    sections.append(line)
            else:
                sections.append(f"第{dr.round_index+1}轮[{dr.current_time}]: （不行动）")

    if len(sections) <= 2:
        return "（该智能体未产生任何认知-行为记录）"
    return "\n".join(sections)


class BehaviorConsistencyValidator:
    """行为序列一致性验证 — 评估完整认知-行为链"""

    def __init__(self, eval_llm: LLMService):
        self.eval_llm = eval_llm

    async def validate(self, method: str,
                       agents: List[AgentState],
                       all_round_results: List[List[DecisionResult]],
                       progress_callback=None) -> Dict[str, Any]:
        """验证行为序列一致性（含认知链评估）"""
        logger.info(f"[{method}] Starting behavior consistency validation...")

        agent_decisions = self._organize_by_agent(agents, all_round_results)

        batch = []
        valid_agents = []
        for agent in agents:
            decisions = agent_decisions.get(agent.user_id, [])
            if not decisions or not any(d.actions for d in decisions):
                continue

            cognitive_chain = _format_cognitive_chain(method, decisions)
            beliefs_text = "\n".join(f"- {b}" for b in agent.raw_user.get('psychological_beliefs', []))

            prompt = prompts.BEHAVIOR_CONSISTENCY_EVAL_PROMPT.format(
                identity=agent.profile['identity'],
                beliefs=beliefs_text if beliefs_text else "无",
                cognitive_chain=cognitive_chain,
            )
            batch.append({
                'prompt': prompt,
                'system_prompt': '你是一个社会心理学和认知科学专家。请严格以JSON格式回答。',
                'temperature': 0.3,
            })
            valid_agents.append(agent)

        if not batch:
            return {'error': 'No valid agents with actions'}

        logger.info(f"[{method}] Evaluating {len(batch)} agents' cognitive-behavior chains...")
        responses = await self.eval_llm.batch_query(batch)

        results = []
        dimensions = ['cognitive_chain_completeness', 'emotion_dynamics_rationality',
                       'motivation_behavior_alignment', 'information_integration', 'overall']
        dim_scores = {d: [] for d in dimensions}

        for agent, response in zip(valid_agents, responses):
            parsed = extract_json_from_response(response)
            entry = {
                'agent_id': agent.user_id,
                'username': agent.raw_user.get('username', ''),
            }
            if parsed:
                scores = {}
                for dim in dimensions:
                    dim_data = parsed.get(dim, {})
                    score_val = None
                    reason = ''
                    if isinstance(dim_data, dict):
                        raw_score = dim_data.get('score', dim_data.get('评分', 0))
                        reason = str(dim_data.get('reason', dim_data.get('理由', '')))
                        try:
                            score_val = float(raw_score)
                        except (ValueError, TypeError):
                            score_val = None
                    elif isinstance(dim_data, (int, float)):
                        score_val = float(dim_data)
                    if score_val is not None:
                        score_val = min(max(score_val, 0), 5)
                        scores[dim] = {'score': score_val, 'reason': reason}
                        dim_scores[dim].append(score_val)
                entry['scores'] = scores
                entry['valid'] = bool(scores)
            else:
                entry['valid'] = False
                entry['raw_response'] = response[:300]
            results.append(entry)

        summary = {dim: _compute_stats(scores) for dim, scores in dim_scores.items()}
        valid_count = sum(1 for r in results if r.get('valid'))

        return {
            'method': method,
            'total_agents': len(agents),
            'evaluated_agents': len(valid_agents),
            'valid_evaluations': valid_count,
            'dimension_stats': summary,
            'details': results,
        }

    def _organize_by_agent(self, agents, all_round_results):
        agent_decisions = defaultdict(list)
        for round_results in all_round_results:
            for dr in round_results:
                agent_decisions[dr.agent_id].append(dr)
        return agent_decisions


def _format_user_data_for_personality(method: str, decisions: List[DecisionResult]) -> str:
    """按方法类型格式化用于人格推断的数据"""
    sections = []

    if method == "ebdi":
        sections.append("### 该用户的认知-行为完整数据\n")

        # 信念演化轨迹
        belief_evolution = []
        emotion_trajectory = []
        desire_history = []
        for dr in decisions:
            if dr.belief_state and isinstance(dr.belief_state, dict):
                beliefs = dr.belief_state.get('心理认知', [])
                opinions = dr.belief_state.get('事件观点', [])
                emotions = dr.belief_state.get('情绪向量', {})
                belief_evolution.append(f"[第{dr.round_index+1}轮] 认知：{'、'.join(beliefs[:3]) if beliefs else '无'}")
                if opinions:
                    for op in opinions[:2]:
                        if isinstance(op, dict):
                            belief_evolution.append(f"  观点：关于{op.get('主体', '')}—{op.get('观点', '')}")
                if emotions:
                    top = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
                    emotion_trajectory.append(f"[第{dr.round_index+1}轮] {', '.join(f'{k}={v}' for k, v in top)}")

            if dr.desire_state and isinstance(dr.desire_state, dict):
                for d in dr.desire_state.get('欲望列表', []):
                    if isinstance(d, dict):
                        desire_history.append(f"[第{dr.round_index+1}轮] {d.get('类型', '')}({d.get('强度', '')}): {d.get('描述', '')}")

        if belief_evolution:
            sections.append("**信念演化轨迹：**")
            sections.append("\n".join(belief_evolution[:24]))
        if emotion_trajectory:
            sections.append("\n**情绪变化轨迹：**")
            sections.append("\n".join(emotion_trajectory[:12]))
        if desire_history:
            sections.append("\n**欲望/动机记录：**")
            sections.append("\n".join(desire_history[:12]))

        # 行为序列
        sections.append("\n**行为序列：**")
        sections.append(_format_actions_for_eval(decisions))

    elif method == "cot":
        sections.append("### 该用户的思维链分析和行为数据\n")

        cot_traces = []
        for dr in decisions:
            if dr.cot_analysis and isinstance(dr.cot_analysis, dict):
                ba = dr.cot_analysis.get('信念分析', {})
                dr_info = dr.cot_analysis.get('欲望推理', {})
                trace = f"[第{dr.round_index+1}轮]"
                if isinstance(ba, dict) and ba.get('立场态度'):
                    trace += f" 立场：{ba['立场态度']}"
                if isinstance(dr_info, dict) and dr_info.get('主要欲望'):
                    trace += f" | 动机：{dr_info['主要欲望']}"
                cot_traces.append(trace)

        if cot_traces:
            sections.append("**认知推理摘要：**")
            sections.append("\n".join(cot_traces[:12]))

        sections.append("\n**行为序列：**")
        sections.append(_format_actions_for_eval(decisions))

    else:
        sections.append("### 该用户的行为数据\n")
        sections.append("**行为序列：**")
        sections.append(_format_actions_for_eval(decisions))

    return "\n".join(sections)


class PersonalityStabilityValidator:
    """人格稳定性验证 — 利用全部可用认知数据推断人格"""

    def __init__(self, eval_llm: LLMService):
        self.eval_llm = eval_llm

    async def validate(self, method: str,
                       agents: List[AgentState],
                       all_round_results: List[List[DecisionResult]],
                       progress_callback=None) -> Dict[str, Any]:
        """验证人格稳定性"""
        logger.info(f"[{method}] Starting personality stability validation...")

        agent_decisions = defaultdict(list)
        for round_results in all_round_results:
            for dr in round_results:
                agent_decisions[dr.agent_id].append(dr)

        # Step 1: 从全部可用数据推断人格
        inference_batch = []
        valid_agents = []
        for agent in agents:
            decisions = agent_decisions.get(agent.user_id, [])
            if not decisions or not any(d.actions for d in decisions):
                continue

            user_data = _format_user_data_for_personality(method, decisions)
            prompt = prompts.PERSONALITY_INFERENCE_PROMPT.format(user_data=user_data)
            inference_batch.append({
                'prompt': prompt,
                'system_prompt': '你是一个人格心理学专家。请严格以JSON格式回答。',
                'temperature': 0.3,
            })
            valid_agents.append(agent)

        if not inference_batch:
            return {'error': 'No valid agents'}

        logger.info(f"[{method}] Inferring personality for {len(inference_batch)} agents...")
        inference_responses = await self.eval_llm.batch_query(inference_batch)

        inferred_personalities = []
        for response in inference_responses:
            parsed = extract_json_from_response(response)
            inferred_personalities.append(parsed)

        # Step 2: 比较相似度
        similarity_batch = []
        similarity_agents = []
        for agent, inferred in zip(valid_agents, inferred_personalities):
            if not inferred:
                continue

            original_beliefs = "\n".join(f"- {b}" for b in agent.raw_user.get('psychological_beliefs', []))
            original_tendency = ", ".join(
                f"{k}:{v}" for k, v in agent.raw_user.get('behavior_tendency', {}).items()
            )
            inferred_beliefs = "\n".join(
                f"- {b}" for b in inferred.get('心理认知', [])
            ) if inferred.get('心理认知') else "无"
            inferred_tendency = ", ".join(
                f"{k}:{v}" for k, v in inferred.get('行为倾向', {}).items()
            ) if inferred.get('行为倾向') else "无"

            prompt = prompts.PERSONALITY_SIMILARITY_PROMPT.format(
                original_identity=agent.profile['identity'],
                original_beliefs=original_beliefs if original_beliefs else "无",
                original_tendency=original_tendency if original_tendency else "无",
                inferred_identity=inferred.get('身份描述', '未推断'),
                inferred_beliefs=inferred_beliefs,
                inferred_tendency=inferred_tendency,
            )
            similarity_batch.append({
                'prompt': prompt,
                'system_prompt': '你是一个人格心理学专家。请严格以JSON格式回答。',
                'temperature': 0.3,
            })
            similarity_agents.append((agent, inferred))

        if not similarity_batch:
            return {'error': 'No valid personality inferences'}

        logger.info(f"[{method}] Evaluating personality similarity for {len(similarity_batch)} agents...")
        similarity_responses = await self.eval_llm.batch_query(similarity_batch)

        results = []
        sim_dimensions = ['identity_similarity', 'belief_similarity',
                          'tendency_similarity', 'overall_similarity']
        dim_scores = {d: [] for d in sim_dimensions}

        for (agent, inferred), response in zip(similarity_agents, similarity_responses):
            parsed = extract_json_from_response(response)
            entry = {
                'agent_id': agent.user_id,
                'username': agent.raw_user.get('username', ''),
                'inferred_personality': inferred,
            }
            if parsed:
                scores = {}
                for dim in sim_dimensions:
                    dim_data = parsed.get(dim, {})
                    if isinstance(dim_data, dict):
                        score = min(max(float(dim_data.get('score', 0)), 0), 1)
                        scores[dim] = {'score': score, 'reason': dim_data.get('reason', '')}
                        dim_scores[dim].append(score)
                    elif isinstance(dim_data, (int, float)):
                        score = min(max(float(dim_data), 0), 1)
                        scores[dim] = {'score': score}
                        dim_scores[dim].append(score)
                entry['similarity_scores'] = scores
                entry['valid'] = True
            else:
                entry['valid'] = False
                entry['raw_response'] = response[:300]
            results.append(entry)

        summary = {dim: _compute_stats(scores) for dim, scores in dim_scores.items()}
        valid_count = sum(1 for r in results if r.get('valid'))

        return {
            'method': method,
            'total_agents': len(agents),
            'inferred_agents': len(valid_agents),
            'similarity_evaluated': len(similarity_agents),
            'valid_evaluations': valid_count,
            'dimension_stats': summary,
            'details': results,
        }


def _reconstruct_prompt(method: str, agent: AgentState, ctx: Dict,
                        action_history: List[Dict]) -> str:
    """当prompt_used缺失时，从agent profile和round context重建prompt"""
    prev_text = _format_action_history(action_history) if action_history else "（这是第一轮，暂无历史行为）"

    if method in ("direct_nothink", "direct_think"):
        return prompts.DIRECT_DECISION_PROMPT.format(
            event_background=ctx['event_background'],
            identity=agent.profile['identity'],
            beliefs_text=agent.profile['beliefs_text'],
            tendency_text=agent.profile['tendency_text'],
            opinions_text=agent.profile['opinions_text'],
            current_time=ctx['current_time'],
            external_events_text=ctx['external_events_text'],
            exposed_posts_text=ctx['exposed_posts_text'],
            previous_actions_text=prev_text,
        )
    elif method == "cot":
        return prompts.COT_DECISION_PROMPT.format(
            event_background=ctx['event_background'],
            identity=agent.profile['identity'],
            beliefs_text=agent.profile['beliefs_text'],
            tendency_text=agent.profile['tendency_text'],
            opinions_text=agent.profile['opinions_text'],
            current_time=ctx['current_time'],
            external_events_text=ctx['external_events_text'],
            exposed_posts_text=ctx['exposed_posts_text'],
            previous_actions_text=prev_text,
        )
    else:
        return prompts.DIRECT_DECISION_PROMPT.format(
            event_background=ctx['event_background'],
            identity=agent.profile['identity'],
            beliefs_text=agent.profile['beliefs_text'],
            tendency_text=agent.profile['tendency_text'],
            opinions_text=agent.profile['opinions_text'],
            current_time=ctx['current_time'],
            external_events_text=ctx['external_events_text'],
            exposed_posts_text=ctx['exposed_posts_text'],
            previous_actions_text=prev_text,
        )


class PromptRobustnessValidator:
    """提示鲁棒性验证 — 基于LLM评估的认知鲁棒性测试

    对输入进行可控语义扰动，然后用LLM评估器评估决策过程的
    认知稳定性，重点考察方法是否具有稳定的认知锚定机制。
    """

    def __init__(self, decision_llm: LLMService, eval_llm: LLMService):
        self.decision_llm = decision_llm
        self.eval_llm = eval_llm

    def _generate_perturbed_prompts(self, base_prompt: str, num_variants: int) -> List[str]:
        """生成语义扰动版 prompt"""
        variants = []

        noise_phrases = [
            "\n（系统提示：请注意最新热搜动态）",
            "\n（旁白：有人发了一条不相关的搞笑视频）",
            "\n（提醒：你有3条未读通知）",
            "\n（插播：某明星官宣恋情引发热议）",
            "\n（热搜插播：#某地暴雨预警#）",
            "\n（系统消息：你的关注列表有新动态）",
        ]

        posts_marker = "### 可互动的博文"
        next_marker = "###"

        posts_start = base_prompt.find(posts_marker)
        if posts_start == -1:
            for _ in range(num_variants):
                noise = random.choice(noise_phrases)
                variants.append(base_prompt + noise)
            return variants[:num_variants]

        after_header = base_prompt.find("\n", posts_start)
        next_section = base_prompt.find(next_marker, after_header + 1)
        if next_section == -1:
            end_marker = base_prompt.find("---", after_header)
            if end_marker == -1:
                end_marker = len(base_prompt)
            next_section = end_marker

        posts_block = base_prompt[after_header:next_section]
        before_posts = base_prompt[:after_header]
        after_posts = base_prompt[next_section:]

        post_items = [p.strip() for p in posts_block.strip().split("\n\n") if p.strip()]
        if not post_items:
            post_items = [p.strip() for p in posts_block.strip().split("\n") if p.strip() and len(p.strip()) > 10]

        for _ in range(num_variants):
            perturbed_items = list(post_items)
            random.shuffle(perturbed_items)
            if len(perturbed_items) >= 3:
                perturbed_items.pop(random.randint(0, len(perturbed_items) - 1))
            perturbed_block = "\n\n".join(perturbed_items)
            noise = random.choice(noise_phrases) if random.random() < 0.5 else ""
            variant = before_posts + "\n" + perturbed_block + "\n" + after_posts + noise
            variants.append(variant)

        return variants[:num_variants]

    def _format_decision_summary(self, parsed: Dict) -> str:
        """格式化单个决策响应的摘要"""
        if not parsed:
            return "（解析失败）"
        actions = parsed.get('行为列表', [])
        if not actions:
            return "（不行动）"
        lines = []
        cot = parsed.get('信念分析', {})
        if cot and isinstance(cot, dict):
            lines.append(f"  [推理] 认知:{cot.get('事件认知', 'N/A')[:60]}, 立场:{cot.get('立场态度', 'N/A')}")
        for act in actions:
            atype = act.get('行为类型', 'unknown')
            strategy = act.get('表达策略', {})
            emotion = strategy.get('情绪类型', act.get('情绪类型', ''))
            stance = strategy.get('立场', act.get('立场', ''))
            content_obj = act.get('内容', {})
            content = content_obj.get('文本', '') if isinstance(content_obj, dict) else str(content_obj)
            line = f"  {atype}"
            if emotion:
                line += f" | 情绪:{emotion}"
            if stance:
                line += f" | 立场:{stance}"
            if content:
                line += f" | 内容:{content[:60]}"
            lines.append(line)
        return "\n".join(lines)

    def _get_method_label(self, method: str) -> str:
        labels = {
            'ebdi': 'EBDI多步认知架构（信念/欲望/意图独立建模）',
            'cot': '单步思维链推理（Chain-of-Thought）',
            'direct_think': '直接决策（含思考但无结构化认知过程）',
            'direct_nothink': '直接决策（无认知推理过程）',
        }
        return labels.get(method, '未知方法')

    async def validate(self, method: str,
                       agents: List[AgentState],
                       all_round_results: List[List[DecisionResult]],
                       round_contexts: List[Dict],
                       num_sample_decisions: int = 2,
                       num_repeats: int = 5,
                       progress_callback=None) -> Dict[str, Any]:
        """
        认知鲁棒性验证：扰动 + LLM评估

        1. 对每个智能体采样决策点，生成扰动变体并执行
        2. 用LLM评估器评估认知稳定性（非机械匹配）
        """
        logger.info(f"[{method}] Starting perturbation robustness validation "
                     f"({num_sample_decisions} decisions × {num_repeats} variants)...")

        agent_map = {a.user_id: a for a in agents}
        agent_decisions = defaultdict(list)
        for round_results in all_round_results:
            for dr in round_results:
                agent_decisions[dr.agent_id].append(dr)

        needs_reconstruct = any(
            not dr.prompt_used
            for drs in agent_decisions.values()
            for dr in drs
        )
        if needs_reconstruct and round_contexts:
            logger.info(f"[{method}] Reconstructing prompts from profile/context...")
            ctx_by_round = {ctx['round_index']: ctx for ctx in round_contexts}
            agent_history_at_round = defaultdict(list)
            for round_results in all_round_results:
                for dr in round_results:
                    if not dr.prompt_used:
                        ctx = ctx_by_round.get(dr.round_index)
                        agent = agent_map.get(dr.agent_id)
                        if ctx and agent:
                            history = list(agent_history_at_round[dr.agent_id])
                            dr.prompt_used = _reconstruct_prompt(method, agent, ctx, history)
                    if dr.actions:
                        agent_history_at_round[dr.agent_id].extend(dr.actions)

        valid_decisions = defaultdict(list)
        for aid, drs in agent_decisions.items():
            for dr in drs:
                if dr.prompt_used:
                    valid_decisions[aid].append(dr)

        robustness_tasks = []
        for agent in agents:
            decisions = valid_decisions.get(agent.user_id, [])
            if not decisions:
                continue
            sampled = random.sample(decisions, min(num_sample_decisions, len(decisions)))
            for dr in sampled:
                perturbed_prompts = self._generate_perturbed_prompts(dr.prompt_used, num_repeats)
                robustness_tasks.append({
                    'agent': agent,
                    'original_decision': dr,
                    'prompts': perturbed_prompts,
                })

        if not robustness_tasks:
            return {'error': 'No valid prompts for robustness testing'}

        total_sim_queries = sum(len(t['prompts']) for t in robustness_tasks)
        logger.info(f"[{method}] Running {total_sim_queries} perturbation simulation queries...")

        # Step 1: Run perturbation simulations
        all_queries = []
        task_map = []
        for t_idx, task in enumerate(robustness_tasks):
            for p in task['prompts']:
                all_queries.append({
                    'prompt': p,
                    'system_prompt': '你是一个微博用户模拟器。请以JSON格式回答。',
                })
                task_map.append(t_idx)

        sim_responses = await self.decision_llm.batch_query(all_queries)

        task_responses = [[] for _ in range(len(robustness_tasks))]
        for q_idx, response in enumerate(sim_responses):
            t_idx = task_map[q_idx]
            parsed = extract_json_from_response(response)
            task_responses[t_idx].append({
                'parsed': parsed,
                'raw': response[:1000],
            })

        # Step 2: LLM evaluation of cognitive robustness
        logger.info(f"[{method}] Running {len(robustness_tasks)} cognitive robustness evaluations...")

        eval_batch = []
        valid_task_indices = []
        for t_idx, task in enumerate(robustness_tasks):
            reps = task_responses[t_idx]
            agent = task['agent']
            original_dr = task['original_decision']

            original_text = _format_cognitive_chain(method, [original_dr])

            perturbed_texts = []
            for i, rep in enumerate(reps, 1):
                summary = self._format_decision_summary(rep['parsed'])
                perturbed_texts.append(f"变体{i}:\n{summary}")

            if not perturbed_texts:
                continue

            prompt = prompts.ROBUSTNESS_EVAL_PROMPT.format(
                identity=agent.profile['identity'],
                method_label=self._get_method_label(method),
                original_decision=original_text,
                num_variants=len(perturbed_texts),
                perturbed_decisions="\n\n".join(perturbed_texts),
            )
            eval_batch.append({
                'prompt': prompt,
                'system_prompt': '你是一个认知心理学专家。请严格以JSON格式回答。',
                'temperature': 0.3,
            })
            valid_task_indices.append(t_idx)

        eval_responses = await self.eval_llm.batch_query(eval_batch)

        # Step 3: Parse and aggregate
        dimensions = ['core_decision_stability', 'cognitive_anchoring',
                       'adaptive_rationality', 'overall_robustness']
        dim_scores = {d: [] for d in dimensions}
        agent_robustness = defaultdict(list)
        task_details = []

        for eval_idx, response in enumerate(eval_responses):
            t_idx = valid_task_indices[eval_idx]
            task = robustness_tasks[t_idx]
            agent_id = task['agent'].user_id
            parsed = extract_json_from_response(response)

            entry = {
                'agent_id': agent_id,
                'round_index': task['original_decision'].round_index,
                'num_variants': len(task_responses[t_idx]),
            }

            if parsed:
                scores = {}
                for dim in dimensions:
                    dim_data = parsed.get(dim, {})
                    if isinstance(dim_data, dict):
                        score = min(max(float(dim_data.get('score', 0)), 0), 1)
                        scores[dim] = score
                        dim_scores[dim].append(score)
                    elif isinstance(dim_data, (int, float)):
                        score = min(max(float(dim_data), 0), 1)
                        scores[dim] = score
                        dim_scores[dim].append(score)
                entry['scores'] = scores
                entry['valid'] = True
            else:
                entry['valid'] = False
                entry['raw_response'] = response[:300]

            task_details.append(entry)
            agent_robustness[agent_id].append(entry)

        summary = {dim: _compute_stats(scores) for dim, scores in dim_scores.items()}
        valid_count = sum(1 for e in task_details if e.get('valid'))

        return {
            'method': method,
            'total_agents': len(agents),
            'tested_agents': len(agent_robustness),
            'total_tasks': len(robustness_tasks),
            'total_sim_queries': total_sim_queries,
            'total_eval_queries': len(eval_batch),
            'valid_evaluations': valid_count,
            'num_sample_decisions': num_sample_decisions,
            'num_variants': num_repeats,
            'dimension_stats': summary,
            'task_details': task_details,
        }


class BehaviorHeterogeneityValidator:
    """智能体行为异质性验证 — 测量群体行为多样性（宏观涌现的关键）

    计算多组异质性指标，不需要LLM调用，直接从模拟数据计算。
    高异质性意味着智能体群体展现了多样化的行为模式，
    是产生宏观涌现现象的前提条件。
    """

    def validate(self, method: str,
                 agents: List[AgentState],
                 all_round_results: List[List[DecisionResult]]) -> Dict[str, Any]:
        """计算多组行为异质性指标"""
        from collections import Counter
        import math

        agent_decisions = defaultdict(list)
        for round_results in all_round_results:
            for dr in round_results:
                agent_decisions[dr.agent_id].append(dr)

        results = {}

        # Group 1: Action Type Entropy (per-round averaged)
        round_entropies = []
        for round_results in all_round_results:
            action_types = []
            for dr in round_results:
                for act in dr.actions:
                    action_types.append(act.get('action_type', 'none'))
            if action_types:
                round_entropies.append(self._shannon_entropy(action_types))
        results['action_type_entropy'] = {
            'value': float(np.mean(round_entropies)) if round_entropies else 0,
            'per_round': [float(e) for e in round_entropies],
            'description': '行为类型分布的Shannon熵（越高=行为类型越多样）',
        }

        # Group 2: Stance Diversity (per-round averaged)
        round_stance_entropies = []
        for round_results in all_round_results:
            stances = []
            for dr in round_results:
                for act in dr.actions:
                    s = act.get('stance', '')
                    if s:
                        normalized = self._normalize_stance(s)
                        stances.append(normalized)
            if stances:
                round_stance_entropies.append(self._shannon_entropy(stances))
        results['stance_diversity'] = {
            'value': float(np.mean(round_stance_entropies)) if round_stance_entropies else 0,
            'per_round': [float(e) for e in round_stance_entropies],
            'description': '立场分布的Shannon熵（越高=立场越多元）',
        }

        # Group 3: Emotion Profile Diversity (cross-agent)
        agent_emotion_profiles = {}
        for aid, drs in agent_decisions.items():
            emotions = []
            for dr in drs:
                for act in dr.actions:
                    e = act.get('emotion', 'neutral')
                    emotions.append(e)
            if emotions:
                agent_emotion_profiles[aid] = dict(Counter(emotions))
        if len(agent_emotion_profiles) >= 2:
            diversities = []
            aids = list(agent_emotion_profiles.keys())
            sample_size = min(len(aids), 50)
            sampled_aids = random.sample(aids, sample_size)
            for i in range(len(sampled_aids)):
                for j in range(i + 1, len(sampled_aids)):
                    div = self._distribution_distance(
                        agent_emotion_profiles[sampled_aids[i]],
                        agent_emotion_profiles[sampled_aids[j]]
                    )
                    diversities.append(div)
            results['emotion_diversity'] = {
                'value': float(np.mean(diversities)) if diversities else 0,
                'description': '智能体间情绪分布差异（越高=情绪反应越异质）',
            }
        else:
            results['emotion_diversity'] = {'value': 0, 'description': '数据不足'}

        # Group 4: Participation Inequality (Gini coefficient)
        agent_action_counts = []
        for aid, drs in agent_decisions.items():
            total_actions = sum(len(dr.actions) for dr in drs)
            agent_action_counts.append(total_actions)
        if agent_action_counts:
            results['participation_gini'] = {
                'value': float(self._gini_coefficient(agent_action_counts)),
                'description': '参与度基尼系数（越高=参与度分化越大，部分活跃部分沉默）',
            }
        else:
            results['participation_gini'] = {'value': 0, 'description': '数据不足'}

        # Group 5: Content Lexical Diversity (Type-Token Ratio)
        all_contents = []
        for round_results in all_round_results:
            for dr in round_results:
                for act in dr.actions:
                    content = act.get('content', '')
                    if content and len(content) > 5:
                        all_contents.append(content)
        if all_contents:
            all_chars = list("".join(all_contents))
            unique_chars = set(all_chars)
            ttr = len(unique_chars) / max(len(all_chars), 1)
            bigrams = [all_chars[i] + all_chars[i+1] for i in range(len(all_chars)-1)]
            unique_bigrams = set(bigrams)
            bigram_ttr = len(unique_bigrams) / max(len(bigrams), 1)
            results['content_lexical_diversity'] = {
                'value': float(bigram_ttr),
                'char_ttr': float(ttr),
                'bigram_ttr': float(bigram_ttr),
                'total_contents': len(all_contents),
                'description': '内容词汇多样性（bigram TTR，越高=生成内容越多样）',
            }
        else:
            results['content_lexical_diversity'] = {'value': 0, 'description': '无内容数据'}

        # Group 6: Temporal Behavioral Variability
        agent_temporal = {}
        for aid, drs in agent_decisions.items():
            round_actions = defaultdict(int)
            for dr in drs:
                round_actions[dr.round_index] = len(dr.actions)
            if round_actions:
                values = list(round_actions.values())
                agent_temporal[aid] = float(np.std(values)) if len(values) > 1 else 0
        if agent_temporal:
            temporal_values = list(agent_temporal.values())
            results['temporal_variability'] = {
                'value': float(np.mean(temporal_values)),
                'std': float(np.std(temporal_values)),
                'description': '时序行为变异性（越高=智能体行为随时间变化越大）',
            }
        else:
            results['temporal_variability'] = {'value': 0, 'description': '数据不足'}

        summary_value = float(np.mean([
            results['action_type_entropy']['value'],
            results['stance_diversity']['value'],
            results['emotion_diversity']['value'],
            results['participation_gini']['value'],
            results['content_lexical_diversity']['value'],
            results['temporal_variability']['value'],
        ]))

        return {
            'method': method,
            'total_agents': len(agents),
            'metrics': results,
            'overall_heterogeneity': summary_value,
        }

    def _shannon_entropy(self, values: list) -> float:
        from collections import Counter
        counter = Counter(values)
        total = sum(counter.values())
        probs = [c / total for c in counter.values()]
        return -sum(p * np.log2(p + 1e-10) for p in probs)

    def _normalize_stance(self, stance: str) -> str:
        s = stance.lower().strip()
        if '支持' in s:
            return 'support'
        elif '反对' in s:
            return 'oppose'
        elif '中立' in s:
            return 'neutral'
        return s

    def _distribution_distance(self, dist1: dict, dist2: dict) -> float:
        all_keys = set(list(dist1.keys()) + list(dist2.keys()))
        total1 = sum(dist1.values())
        total2 = sum(dist2.values())
        if total1 == 0 or total2 == 0:
            return 0
        distance = 0
        for key in all_keys:
            p1 = dist1.get(key, 0) / total1
            p2 = dist2.get(key, 0) / total2
            distance += abs(p1 - p2)
        return distance / 2

    def _gini_coefficient(self, values: list) -> float:
        arr = np.array(sorted(values), dtype=float)
        n = len(arr)
        if n == 0 or arr.sum() == 0:
            return 0
        index = np.arange(1, n + 1)
        return float((2 * np.sum(index * arr) - (n + 1) * np.sum(arr)) / (n * np.sum(arr)))


def save_validation_results(all_validation: Dict[str, Dict],
                            output_dir: str,
                            validation_type: str):
    """保存验证结果"""
    val_dir = os.path.join(output_dir, 'validation', validation_type)
    os.makedirs(val_dir, exist_ok=True)

    for method, result in all_validation.items():
        filepath = os.path.join(val_dir, f'{method}.json')
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"Saved {validation_type}/{method}.json")

    comparison = {}
    for method, result in all_validation.items():
        stats = result.get('dimension_stats') or result.get('metric_stats', {})
        comparison[method] = {}
        for dim, dim_stats in stats.items():
            if isinstance(dim_stats, dict) and 'mean' in dim_stats:
                comparison[method][dim] = {
                    'mean': dim_stats['mean'],
                    'std': dim_stats.get('std', 0),
                }

    comp_file = os.path.join(val_dir, 'comparison.json')
    with open(comp_file, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
