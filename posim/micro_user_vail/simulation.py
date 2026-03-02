# -*- coding: utf-8 -*-
"""
模拟引擎：4种决策方法的实现与运行
"""
import asyncio
import json
import logging
import os
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict

from .llm_service import LLMService, extract_json_from_response
from .data_loader import format_user_profile
from . import prompts

logger = logging.getLogger(__name__)


@dataclass
class ActionRecord:
    """标准化的行为记录"""
    action_type: str = "none"
    target_index: int = 0
    emotion: str = "中性"
    emotion_intensity: str = "低"
    stance: str = ""
    stance_intensity: str = "低"
    content: str = ""
    style: str = ""
    narrative: str = ""

    def to_summary(self) -> str:
        parts = [f"行为:{self.action_type}"]
        if self.emotion and self.emotion != "中性":
            parts.append(f"情绪:{self.emotion}({self.emotion_intensity})")
        if self.stance:
            parts.append(f"立场:{self.stance}")
        if self.content:
            parts.append(f"内容:{self.content[:60]}")
        return " | ".join(parts)


@dataclass
class DecisionResult:
    """一次完整决策的结果"""
    agent_id: str = ""
    round_index: int = 0
    method: str = ""
    current_time: str = ""
    actions: List[Dict] = field(default_factory=list)
    raw_response: str = ""
    belief_state: Optional[Dict] = None
    desire_state: Optional[Dict] = None
    cot_analysis: Optional[Dict] = None
    prompt_used: str = ""
    success: bool = True
    error: str = ""


@dataclass
class AgentState:
    """智能体在模拟过程中的状态"""
    user_id: str
    profile: Dict[str, str]
    raw_user: Dict
    action_history: List[DecisionResult] = field(default_factory=list)
    current_beliefs: List[str] = field(default_factory=list)
    current_opinions: List[Dict] = field(default_factory=list)
    current_emotions: Dict[str, float] = field(default_factory=dict)


def _parse_actions_from_response(parsed: Dict) -> List[Dict]:
    """从解析的JSON中提取标准化的行为列表"""
    actions = []
    raw_list = parsed.get('行为列表', [])
    if not isinstance(raw_list, list):
        return actions

    for item in raw_list:
        strategy = item.get('表达策略', {})
        content_obj = item.get('内容', {})

        action = {
            'action_type': item.get('行为类型', 'none'),
            'target_index': item.get('目标博文序号', 0),
            'emotion': strategy.get('情绪类型', item.get('情绪类型', '中性')),
            'emotion_intensity': strategy.get('情绪强度', item.get('情绪强度', '低')),
            'stance': strategy.get('立场', item.get('立场', '')),
            'stance_intensity': strategy.get('立场强度', item.get('立场强度', '低')),
            'content': '',
        }

        if isinstance(content_obj, dict):
            action['content'] = content_obj.get('文本', content_obj.get('text', ''))
        elif isinstance(content_obj, str):
            action['content'] = content_obj
        elif isinstance(item.get('内容'), str):
            action['content'] = item['内容']

        if not action['content'] and 'content' in item:
            action['content'] = str(item['content'])

        actions.append(action)

    return actions


def _format_action_history(history: List[DecisionResult], max_items: int = 10) -> str:
    """格式化历史行为记录"""
    if not history:
        return "（暂无历史行为）"

    lines = []
    recent = history[-max_items:]
    for dr in recent:
        for act in dr.actions:
            action_type = act.get('action_type', 'none')
            content = act.get('content', '')[:60]
            emotion = act.get('emotion', '')
            stance = act.get('stance', '')
            line = f"- [{dr.current_time}] {action_type}"
            if emotion:
                line += f" 情绪:{emotion}"
            if stance:
                line += f" 立场:{stance}"
            if content:
                line += f" 内容:{content}"
            lines.append(line)

    return "\n".join(lines) if lines else "（暂无历史行为）"


class SimulationEngine:
    """模拟引擎"""

    def __init__(self, nothink_llm: LLMService, think_llm: LLMService):
        self.nothink_llm = nothink_llm
        self.think_llm = think_llm

    async def run_method(self, method: str, agents: List[AgentState],
                         round_contexts: List[Dict],
                         progress_callback=None) -> List[List[DecisionResult]]:
        """
        运行指定方法的完整模拟

        Returns:
            all_results[round][agent] = DecisionResult
        """
        all_round_results = []
        total_rounds = len(round_contexts)

        for r_idx, ctx in enumerate(round_contexts):
            if progress_callback:
                progress_callback(r_idx, total_rounds, method)

            round_results = await self._run_single_round(method, agents, ctx)
            all_round_results.append(round_results)

            for agent, result in zip(agents, round_results):
                agent.action_history.append(result)
                if method == "ebdi" and result.belief_state:
                    bs = result.belief_state
                    if '心理认知' in bs:
                        agent.current_beliefs = bs['心理认知']
                    if '事件观点' in bs:
                        agent.current_opinions = bs['事件观点']
                    if '情绪向量' in bs:
                        agent.current_emotions = bs['情绪向量']
                elif method == "cot" and result.cot_analysis:
                    ca = result.cot_analysis
                    belief = ca.get('信念分析', {})
                    if '情绪状态' in belief:
                        agent.current_emotions = belief['情绪状态']

            logger.info(
                f"[{method}] Round {r_idx+1}/{total_rounds}: "
                f"{sum(1 for r in round_results if r.actions)} agents acted"
            )

        return all_round_results

    async def _run_single_round(self, method: str, agents: List[AgentState],
                                ctx: Dict) -> List[DecisionResult]:
        """运行单轮模拟"""
        if method == "direct_nothink":
            return await self._run_direct(self.nothink_llm, agents, ctx, method)
        elif method == "direct_think":
            return await self._run_direct(self.think_llm, agents, ctx, method)
        elif method == "cot":
            return await self._run_cot(self.nothink_llm, agents, ctx)
        elif method == "ebdi":
            return await self._run_ebdi(self.nothink_llm, agents, ctx)
        else:
            raise ValueError(f"Unknown method: {method}")

    async def _run_direct(self, llm: LLMService, agents: List[AgentState],
                          ctx: Dict, method: str) -> List[DecisionResult]:
        """直接提示方法"""
        batch = []
        for agent in agents:
            prompt = prompts.DIRECT_DECISION_PROMPT.format(
                event_background=ctx['event_background'],
                identity=agent.profile['identity'],
                beliefs_text=agent.profile['beliefs_text'],
                tendency_text=agent.profile['tendency_text'],
                opinions_text=agent.profile['opinions_text'],
                current_time=ctx['current_time'],
                external_events_text=ctx['external_events_text'],
                exposed_posts_text=ctx['exposed_posts_text'],
                previous_actions_text=_format_action_history(agent.action_history),
            )
            batch.append({
                'prompt': prompt,
                'system_prompt': '你是一个微博用户模拟器。请以JSON格式回答。',
            })

        responses = await llm.batch_query(batch)

        results = []
        for agent, response, query in zip(agents, responses, batch):
            parsed = extract_json_from_response(response)
            actions = _parse_actions_from_response(parsed) if parsed else []
            results.append(DecisionResult(
                agent_id=agent.user_id,
                round_index=ctx['round_index'],
                method=method,
                current_time=ctx['current_time'],
                actions=actions,
                raw_response=response[:2000],
                prompt_used=query['prompt'],
                success=bool(parsed),
                error="" if parsed else "JSON parse failed",
            ))
        return results

    async def _run_cot(self, llm: LLMService, agents: List[AgentState],
                       ctx: Dict) -> List[DecisionResult]:
        """CoT思维链方法"""
        batch = []
        for agent in agents:
            prompt = prompts.COT_DECISION_PROMPT.format(
                event_background=ctx['event_background'],
                identity=agent.profile['identity'],
                beliefs_text=agent.profile['beliefs_text'],
                tendency_text=agent.profile['tendency_text'],
                opinions_text=agent.profile['opinions_text'],
                current_time=ctx['current_time'],
                external_events_text=ctx['external_events_text'],
                exposed_posts_text=ctx['exposed_posts_text'],
                previous_actions_text=_format_action_history(agent.action_history),
            )
            batch.append({
                'prompt': prompt,
                'system_prompt': '你是一个微博用户模拟器。请按照认知推理链进行思考，以JSON格式回答。',
            })

        responses = await llm.batch_query(batch)

        results = []
        for agent, response, query in zip(agents, responses, batch):
            parsed = extract_json_from_response(response)
            actions = _parse_actions_from_response(parsed) if parsed else []
            cot_analysis = None
            if parsed:
                cot_analysis = {
                    '信念分析': parsed.get('信念分析', {}),
                    '欲望推理': parsed.get('欲望推理', {}),
                }
            results.append(DecisionResult(
                agent_id=agent.user_id,
                round_index=ctx['round_index'],
                method="cot",
                current_time=ctx['current_time'],
                actions=actions,
                raw_response=response[:2000],
                cot_analysis=cot_analysis,
                prompt_used=query['prompt'],
                success=bool(parsed),
                error="" if parsed else "JSON parse failed",
            ))
        return results

    async def _run_ebdi(self, llm: LLMService, agents: List[AgentState],
                        ctx: Dict) -> List[DecisionResult]:
        """EBDI三阶段方法"""
        # Stage 1: Belief Update
        belief_batch = []
        for agent in agents:
            current_event = ctx['current_event']
            new_info = f"[{current_event.get('time', '')}] {current_event.get('content', '')}"

            beliefs_text = "\n".join(f"- {b}" for b in agent.current_beliefs) \
                if agent.current_beliefs else agent.profile['beliefs_text']
            opinions_text = ""
            for op in (agent.current_opinions or agent.raw_user.get('event_opinions', [])):
                if isinstance(op, dict):
                    opinions_text += f"- 关于{op.get('主体', op.get('subject', ''))}：{op.get('观点', op.get('opinion', ''))}\n"

            ext_section = f"### 近期发生的突发事件\n{ctx['external_events_text']}" \
                if ctx['external_events_text'] else ""

            prompt = prompts.EBDI_BELIEF_PROMPT.format(
                event_background=ctx['event_background'],
                current_time=ctx['current_time'],
                identity=agent.profile['identity'],
                beliefs_text=beliefs_text if beliefs_text else "暂无",
                opinions_text=opinions_text if opinions_text else "暂无",
                new_info=new_info,
                previous_actions_text=_format_action_history(agent.action_history, max_items=5),
                external_events_section=ext_section,
            )
            belief_batch.append({
                'prompt': prompt,
                'system_prompt': '请以JSON格式输出信念状态。',
            })

        belief_responses = await llm.batch_query(belief_batch)

        belief_states = []
        for response in belief_responses:
            parsed = extract_json_from_response(response)
            belief_states.append(parsed or {})

        # Stage 2: Desire Generation
        desire_batch = []
        for agent, bs in zip(agents, belief_states):
            beliefs_text = "\n".join(f"- {b}" for b in bs.get('心理认知', agent.current_beliefs)) \
                if bs.get('心理认知') or agent.current_beliefs else agent.profile['beliefs_text']

            opinions_text = ""
            for op in bs.get('事件观点', agent.current_opinions or []):
                if isinstance(op, dict):
                    opinions_text += f"- 关于{op.get('主体', '')}：{op.get('观点', '')}\n"

            emotions = bs.get('情绪向量', agent.current_emotions or {})
            emotion_text = ", ".join(f"{k}:{v}" for k, v in emotions.items()) if emotions else "平静"

            ext_section = f"### 近期发生的突发事件\n{ctx['external_events_text']}" \
                if ctx['external_events_text'] else ""

            prompt = prompts.EBDI_DESIRE_PROMPT.format(
                event_background=ctx['event_background'],
                current_time=ctx['current_time'],
                identity=agent.profile['identity'],
                beliefs_text=beliefs_text if beliefs_text else "暂无",
                opinions_text=opinions_text if opinions_text else "暂无",
                emotion_text=emotion_text,
                exposed_info=ctx['exposed_posts_text'],
                external_events_section=ext_section,
            )
            desire_batch.append({
                'prompt': prompt,
                'system_prompt': '请以JSON格式输出欲望列表。',
            })

        desire_responses = await llm.batch_query(desire_batch)

        desire_states = []
        for response in desire_responses:
            parsed = extract_json_from_response(response)
            desire_states.append(parsed or {})

        # Stage 3: Intention/Action Generation
        intention_batch = []
        intention_prompts_used = []
        for agent, bs, ds in zip(agents, belief_states, desire_states):
            beliefs_text = "\n".join(f"- {b}" for b in bs.get('心理认知', agent.current_beliefs)) \
                if bs.get('心理认知') or agent.current_beliefs else agent.profile['beliefs_text']

            opinions_text = ""
            for op in bs.get('事件观点', agent.current_opinions or []):
                if isinstance(op, dict):
                    opinions_text += f"- 关于{op.get('主体', '')}：{op.get('观点', '')}\n"

            emotions = bs.get('情绪向量', agent.current_emotions or {})
            emotion_text = ", ".join(f"{k}:{v}" for k, v in emotions.items()) if emotions else "平静"

            desires_text = ""
            for d in ds.get('欲望列表', []):
                if isinstance(d, dict):
                    desires_text += f"- {d.get('类型', '')}({d.get('强度', '')}): {d.get('描述', '')}\n"
            if not desires_text:
                desires_text = "- 浏览信息（低）"

            ext_section = f"### 近期发生的突发事件\n{ctx['external_events_text']}" \
                if ctx['external_events_text'] else ""

            prompt = prompts.EBDI_INTENTION_PROMPT.format(
                event_background=ctx['event_background'],
                current_time=ctx['current_time'],
                identity=agent.profile['identity'],
                beliefs_text=beliefs_text if beliefs_text else "暂无",
                opinions_text=opinions_text if opinions_text else "暂无",
                emotion_text=emotion_text,
                desires_text=desires_text,
                exposed_posts_text=ctx['exposed_posts_text'],
                external_events_section=ext_section,
            )
            intention_batch.append({
                'prompt': prompt,
                'system_prompt': '你是一个微博用户模拟器。请以JSON格式输出行为决策。',
            })
            intention_prompts_used.append(prompt)

        intention_responses = await llm.batch_query(intention_batch)

        results = []
        for i, (agent, response) in enumerate(zip(agents, intention_responses)):
            parsed = extract_json_from_response(response)
            actions = _parse_actions_from_response(parsed) if parsed else []
            results.append(DecisionResult(
                agent_id=agent.user_id,
                round_index=ctx['round_index'],
                method="ebdi",
                current_time=ctx['current_time'],
                actions=actions,
                raw_response=response[:2000],
                belief_state=belief_states[i],
                desire_state=desire_states[i],
                prompt_used=intention_prompts_used[i],
                success=bool(parsed),
                error="" if parsed else "JSON parse failed",
            ))

        return results


def save_simulation_results(all_results: Dict[str, List[List[DecisionResult]]],
                            output_dir: str):
    """保存所有模拟结果"""
    for method, round_results in all_results.items():
        method_dir = os.path.join(output_dir, 'simulation', method)
        os.makedirs(method_dir, exist_ok=True)

        all_decisions = []
        for r_idx, round_result in enumerate(round_results):
            round_data = []
            for dr in round_result:
                record = {
                    'agent_id': dr.agent_id,
                    'round_index': dr.round_index,
                    'method': dr.method,
                    'current_time': dr.current_time,
                    'actions': dr.actions,
                    'success': dr.success,
                    'error': dr.error,
                    'prompt_used': dr.prompt_used[:3000] if dr.prompt_used else '',
                }
                if dr.belief_state:
                    record['belief_state'] = dr.belief_state
                if dr.desire_state:
                    record['desire_state'] = dr.desire_state
                if dr.cot_analysis:
                    record['cot_analysis'] = dr.cot_analysis
                round_data.append(record)
                all_decisions.append(record)

            round_file = os.path.join(method_dir, f'round_{r_idx:02d}.json')
            with open(round_file, 'w', encoding='utf-8') as f:
                json.dump(round_data, f, ensure_ascii=False, indent=2)

        summary_file = os.path.join(method_dir, 'all_decisions.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(all_decisions, f, ensure_ascii=False, indent=2)

        # 按智能体组织
        agent_decisions = {}
        for decision in all_decisions:
            aid = decision['agent_id']
            if aid not in agent_decisions:
                agent_decisions[aid] = []
            agent_decisions[aid].append(decision)

        agent_file = os.path.join(method_dir, 'by_agent.json')
        with open(agent_file, 'w', encoding='utf-8') as f:
            json.dump(agent_decisions, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {method}: {len(all_decisions)} decisions across {len(round_results)} rounds")
