"""Hypothetical Minds Agent Implementation
==========================================

基于 Theory of Mind (ToM) 的多智能体决策框架。

核心机制 (based on "Hypothetical Minds: Scaffolding Theory of Mind
for Multi-Agent Tasks with Large Language Models"):
"""

import json
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from simulation.models.agents.LLMAgent import LLMAgent
from simulation.models.cognitive.experiment_logger import ExperimentLogger


class MentalModel:
    """单个对手的心智模型 (Theory of Mind)

    维护对该对手的以下推断：
    - inferred_goal: 推断的战略目标
    - strategy_tendency: 策略倾向描述
    - action_history: 历史动作序列
    - action_freq: 动作频率统计
    - predictability: 可预测性评分
    """

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.inferred_goal: str = "unknown"
        self.strategy_tendency: str = "unknown"
        self.action_history: List[str] = []
        self.action_freq: Dict[str, int] = defaultdict(int)
        self.predictability: float = 0.5
        self.last_update_step: int = 0

    def observe_action(self, action: str, step: int) -> None:
        """观察对手动作并更新心智模型"""
        self.action_history.append(action)
        self.action_freq[action] += 1
        self.last_update_step = step

        # 更新可预测性：如果对手行为模式越一致，可预测性越高
        if len(self.action_history) >= 3:
            recent = self.action_history[-5:]
            most_common = max(set(recent), key=recent.count)
            self.predictability = recent.count(most_common) / len(recent)

    def predict_action(self, available_actions: List[str]) -> Tuple[str, float]:
        """基于历史频率预测对手最可能的动作"""
        if not self.action_freq:
            # 无先验时均匀分布
            return available_actions[0] if available_actions else "HOLD", 0.5

        # 从观察到的动作中选频率最高的
        best_action = max(self.action_freq, key=self.action_freq.get)
        total = sum(self.action_freq.values())
        confidence = self.action_freq[best_action] / total if total > 0 else 0.5
        return best_action, confidence

    def get_action_distribution(self, available_actions: List[str]) -> Dict[str, float]:
        """获取对手动作的概率分布（Laplace 平滑）"""
        total = sum(self.action_freq.values()) + len(available_actions)  # Laplace
        dist = {}
        for a in available_actions:
            dist[a] = (self.action_freq.get(a, 0) + 1) / total
        return dist

    def to_summary(self) -> str:
        """生成心智模型摘要（用于 LLM prompt）"""
        recent = self.action_history[-5:] if self.action_history else []
        top_actions = sorted(self.action_freq.items(), key=lambda x: -x[1])[:3]
        return (
            f"Agent={self.agent_id} "
            f"Goal={self.inferred_goal} "
            f"Tendency={self.strategy_tendency} "
            f"Recent={recent} "
            f"TopActions={top_actions} "
            f"Predictability={self.predictability:.2f}"
        )


class HypotheticalMindsAgent(LLMAgent):
    """Hypothetical Minds Agent — 基于心智理论的多智能体决策框架

    Architecture:
        1. ToM Module: 为每个对手维护 MentalModel
        2. Hypothetical Generator: 生成候选动作
        3. Mental Simulator: 模拟对手响应
        4. Outcome Evaluator: 评估预期结果
        5. Action Selector: 选择最优动作
    """

    def __init__(
        self,
        country_name: str,
        other_countries: List[str],
        game_attributes: Dict[str, int],
        experiment_logger: ExperimentLogger,
        max_simulations: int = 3,
    ):
        super().__init__(
            agent_name=f"HM_{country_name}",
            has_chat_history=False,
            llm_model='qwen3-max',
            online_track=False,
            json_format=True,
        )

        self.country_name = country_name
        self.other_countries = other_countries
        self.game_attributes = game_attributes.copy()
        self.experiment_logger = experiment_logger
        self.max_simulations = max_simulations

        # ToM Module: 每个对手一个心智模型
        self.mental_models: Dict[str, MentalModel] = {
            c: MentalModel(c) for c in other_countries
        }

        # 决策记录
        self.action_history: List[str] = []
        self.declaration_history: List[str] = []
        self.round_counter = 0
        self.simulation_traces: List[Dict] = []

        # 行动空间
        self.available_actions = [
            "外交谈判", "和平协议", "军事演习", "区域封锁", "武器部署",
            "经济制裁", "情报侦察", "撤回行动", "最后通牒", "宣战", "核打击"
        ]

    def cognitive_game_decide(self, world_info: str) -> Dict[str, Any]:
        """Hypothetical Minds 决策流程

        1. Update ToM models from latest observations
        2. Generate candidate actions
        3. For each candidate: simulate opponent responses via ToM
        4. Evaluate outcomes and select best action
        """
        self.round_counter += 1

        # Step 1: 更新心智模型（基于 world_info 中的对手行为）
        self._update_tom_from_context(world_info)

        # Step 2: 生成候选动作
        candidates = self._generate_candidates(world_info)

        # Step 3 & 4: 对每个候选动作进行心智模拟并评估
        best_action, reasoning = self._simulate_and_evaluate(
            world_info, candidates
        )

        # Step 5: 生成声明
        declaration = self._generate_declaration(best_action, reasoning, world_info)

        self.action_history.append(best_action)
        self.declaration_history.append(declaration)

        return {
            'action': best_action,
            'declaration': declaration,
            'reasoning_result': {
                'tom_summaries': {
                    c: m.to_summary() for c, m in self.mental_models.items()
                },
                'candidates_evaluated': len(candidates),
                'reasoning': reasoning,
                'method': 'hypothetical_minds',
                'final_satisfaction_score': 0.7,
                'reasoning_depth': self.max_simulations,
            },
            'satisfaction_score': 0.7,
            'reasoning_depth': self.max_simulations,
        }

    def _update_tom_from_context(self, world_info: str) -> None:
        """从上下文中更新对手心智模型"""
        prompt = (
            "Analyze the following situation to infer each agent's "
            "likely goal and strategy tendency.\n\n"
            f"Situation:\n{world_info[:500]}\n\n"
            f"Agents to analyze: {', '.join(self.other_countries)}\n\n"
            "Return JSON:\n"
            '{{"agent_analyses": [{{"agent": "name", '
            '"inferred_goal": "...", "strategy_tendency": "..."}}, ...]}}'
        )
        try:
            resp = self.get_response(prompt, flag_debug_print=False)
            if isinstance(resp, dict):
                for analysis in resp.get("agent_analyses", []):
                    agent_id = analysis.get("agent", "")
                    if agent_id in self.mental_models:
                        mm = self.mental_models[agent_id]
                        mm.inferred_goal = analysis.get(
                            "inferred_goal", mm.inferred_goal
                        )
                        mm.strategy_tendency = analysis.get(
                            "strategy_tendency", mm.strategy_tendency
                        )
        except Exception:
            pass

    def _generate_candidates(self, world_info: str) -> List[str]:
        """使用 LLM 生成候选动作"""
        prompt = (
            f"You are {self.country_name}. Given the current situation, "
            f"select 3 promising candidate actions.\n\n"
            f"Situation:\n{world_info[:400]}\n\n"
            f"Available actions: {', '.join(self.available_actions)}\n\n"
            "Return JSON:\n"
            '{{"candidates": ["action1", "action2", "action3"]}}'
        )
        try:
            resp = self.get_response(prompt, flag_debug_print=False)
            if isinstance(resp, dict):
                candidates = resp.get("candidates", [])
                valid = [c for c in candidates if c in self.available_actions]
                if valid:
                    return valid[:self.max_simulations]
        except Exception:
            pass
        # 默认返回前3个动作
        return self.available_actions[:self.max_simulations]

    def _simulate_and_evaluate(
        self, world_info: str, candidates: List[str],
    ) -> Tuple[str, str]:
        """核心：对每个候选动作进行心智模拟并评估

        利用 ToM 模型预测对手对我方每个候选动作的响应，
        然后评估每个候选动作的预期效用。
        """
        # 构建对手心智模型摘要
        tom_context = "\n".join(
            m.to_summary() for m in self.mental_models.values()
        )

        evaluations: List[Dict[str, Any]] = []
        for action in candidates:
            # 心智模拟：预测对手响应
            sim_prompt = (
                f"You are evaluating action '{action}' for {self.country_name}.\n\n"
                f"Current situation:\n{world_info[:300]}\n\n"
                f"Theory of Mind models of other agents:\n{tom_context}\n\n"
                "For this action, predict:\n"
                "1. How each opponent will likely respond\n"
                "2. The expected outcome (positive/negative/neutral)\n"
                "3. A utility score from 0 to 1\n\n"
                "Return JSON:\n"
                '{{"predicted_responses": {{"agent": "response", ...}}, '
                '"expected_outcome": "...", "utility": 0.0-1.0, '
                '"reasoning": "..."}}'
            )
            try:
                resp = self.get_response(sim_prompt, flag_debug_print=False)
                if isinstance(resp, dict):
                    evaluations.append({
                        "action": action,
                        "utility": float(resp.get("utility", 0.5)),
                        "outcome": resp.get("expected_outcome", "neutral"),
                        "reasoning": resp.get("reasoning", ""),
                        "predicted_responses": resp.get("predicted_responses", {}),
                    })
                    continue
            except Exception:
                pass
            evaluations.append({
                "action": action,
                "utility": 0.5,
                "outcome": "neutral",
                "reasoning": "Simulation fallback",
                "predicted_responses": {},
            })

        # 选择效用最高的动作
        if not evaluations:
            return candidates[0] if candidates else "外交谈判", "Default selection"

        best = max(evaluations, key=lambda e: e["utility"])
        self.simulation_traces.append({
            "round": self.round_counter,
            "evaluations": evaluations,
            "selected": best["action"],
        })

        return best["action"], best["reasoning"]

    def _generate_declaration(
        self, action: str, reasoning: str, world_info: str,
    ) -> str:
        """生成行动声明"""
        prompt = (
            f"Generate a brief diplomatic declaration for action '{action}'.\n"
            f"Reasoning: {reasoning}\n\n"
            'Return JSON: {{"declaration": "..."}}'
        )
        try:
            resp = self.get_response(prompt, flag_debug_print=False)
            if isinstance(resp, dict):
                return resp.get("declaration", f"{self.country_name}决定{action}")
        except Exception:
            pass
        return f"{self.country_name}决定{action}，以维护国家利益。"

    def learn_from_interaction(
        self,
        own_action: str,
        world_feedback: str,
        other_actions: Dict[str, str],
        world_memory: Any = None,
    ) -> None:
        """从交互中学习 — 更新对手心智模型"""
        for agent_id, action in other_actions.items():
            if agent_id in self.mental_models:
                self.mental_models[agent_id].observe_action(
                    action, self.round_counter
                )

    def get_cognition_statistics(self) -> Dict[str, Any]:
        return {
            "framework": "HypotheticalMinds",
            "total_decisions": len(self.action_history),
            "tom_models": {
                c: {
                    "predictability": m.predictability,
                    "observations": len(m.action_history),
                    "inferred_goal": m.inferred_goal,
                }
                for c, m in self.mental_models.items()
            },
            "simulation_traces": len(self.simulation_traces),
        }

    @property
    def action(self):
        return self.action_history

    @property
    def declaration(self):
        return self.declaration_history
