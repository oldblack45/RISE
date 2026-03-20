"""
LATSAgent Implementation
基于 Language Agent Tree Search 的智能体决策方法

核心机制：
1. 扩展（Expand）：LLM 生成候选行动节点
2. 选择（Select）：UCB 选择需要进一步模拟的节点
3. 模拟（Simulate）：LLM 进行短视界后果评估
4. 回传（Backpropagate）：将评估价值回传给节点
5. 反思（Reflect）：依据回合反馈更新世界模型
"""

import json
import random
from typing import Any, Dict, List, Optional, Tuple
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from simulation.models.agents.LLMAgent import LLMAgent
from simulation.models.cognitive.experiment_logger import ExperimentLogger


class SearchNode:
    """LATS 树节点（动作级）。"""

    def __init__(self, action: str, thought: str = "", prior: float = 0.0):
        self.action = action
        self.thought = thought
        self.prior = max(0.0, min(1.0, float(prior)))
        self.visits = 0
        self.value_sum = 0.0
        self.last_next_state = ""

    @property
    def mean_value(self) -> float:
        if self.visits <= 0:
            return 0.0
        return self.value_sum / self.visits


class LATSAgent(LLMAgent):
    """LATSAgent - 基于语言树搜索的国家智能体。"""

    def __init__(
        self,
        country_name: str,
        other_countries: List[str],
        game_attributes: Dict[str, int],
        experiment_logger: ExperimentLogger,
        tree_budget: int = 6,
    ):
        super().__init__(
            agent_name=f"LATS_{country_name}",
            has_chat_history=False,
            llm_model="qwen3-max",
            online_track=False,
            json_format=True,
        )

        self.country_name = country_name
        self.other_countries = other_countries
        self.game_attributes = game_attributes.copy()
        self.experiment_logger = experiment_logger

        self.tree_budget = max(4, int(tree_budget))
        self.action_history: List[str] = []
        self.declaration_history: List[str] = []
        self.round_counter = 0
        self.last_reward = 0.0
        self._planner_notes: List[str] = []
        self._current_node: Optional[SearchNode] = None
        self._world_model_state = (
            f"I am {country_name}. Maintain concise strategic state and use LATS "
            "to select robust actions for national survival and advantage."
        )

        self.available_actions = [
            "外交谈判", "和平协议", "军事演习", "区域封锁", "武器部署",
            "经济制裁", "情报侦察", "撤回行动", "最后通牒", "宣战", "核打击"
        ]

    @staticmethod
    def _ucb_score(node: SearchNode, total_visits: int, c: float = 1.2) -> float:
        if node.visits <= 0:
            return 1e9 + node.prior
        value = node.mean_value
        explore = c * ((total_visits + 1) ** 0.5) / ((node.visits + 1) ** 0.5)
        return value + explore + 0.1 * node.prior

    def _build_search_tree(self, world_info: str) -> List[SearchNode]:
        notes = "\n".join(self._planner_notes[-3:]) if self._planner_notes else "(none)"
        prompt = f"""你是LATS规划器，请在当前局势下扩展候选行动节点。

国家：{self.country_name}
当前世界模型：{self._world_model_state}
当前局势：{world_info[:600]}
最近行动：{', '.join(self.action_history[-3:]) if self.action_history else '无'}
规划备注：{notes}
可选行动：{', '.join(self.available_actions)}

请返回JSON：
{{
  "candidates": [
    {{"action": "行动名", "thought": "简要战术意图", "prior": 0.0-1.0}}
  ]
}}

要求：
1) 给出3-5个候选；
2) action必须来自可选行动；
3) 候选尽量多样。
"""
        try:
            result = self.get_response(prompt, flag_debug_print=False)
        except Exception as e:
            self.experiment_logger.log_print(f"LATS扩展阶段失败: {e}", level="WARNING")
            result = None

        nodes: List[SearchNode] = []
        if isinstance(result, dict):
            raw = result.get("candidates", [])
            if isinstance(raw, list):
                for item in raw:
                    if not isinstance(item, dict):
                        continue
                    action = str(item.get("action", "")).strip()
                    if action not in self.available_actions:
                        continue
                    try:
                        prior = float(item.get("prior", 0.0))
                    except Exception:
                        prior = 0.0
                    nodes.append(SearchNode(
                        action=action,
                        thought=str(item.get("thought", "")).strip(),
                        prior=prior,
                    ))

        if not nodes:
            fallback = ["外交谈判", "情报侦察", "军事演习"]
            for act in fallback:
                nodes.append(SearchNode(act, thought="fallback", prior=1.0 / len(fallback)))

        uniq: Dict[str, SearchNode] = {}
        for node in nodes:
            if node.action not in uniq:
                uniq[node.action] = node
        return list(uniq.values())

    def _simulate_node(self, world_info: str, node: SearchNode) -> float:
        prompt = f"""你是LATS模拟器，请评估候选行动的一步后果。

国家：{self.country_name}
候选行动：{node.action}
行动意图：{node.thought}
当前世界模型：{self._world_model_state}
当前局势：{world_info[:600]}

返回JSON：
{{
  "immediate_value": 0.0-1.0,
  "long_term_value": 0.0-1.0,
  "risk": 0.0-1.0,
  "next_state_summary": "下一状态简述"
}}
"""
        try:
            sim = self.get_response(prompt, flag_debug_print=False)
        except Exception as e:
            self.experiment_logger.log_print(f"LATS模拟阶段失败: {e}", level="WARNING")
            sim = None

        if not isinstance(sim, dict):
            return 0.5

        try:
            immediate = float(sim.get("immediate_value", 0.5))
        except Exception:
            immediate = 0.5
        try:
            long_term = float(sim.get("long_term_value", 0.5))
        except Exception:
            long_term = 0.5
        try:
            risk = float(sim.get("risk", 0.5))
        except Exception:
            risk = 0.5

        immediate = max(0.0, min(1.0, immediate))
        long_term = max(0.0, min(1.0, long_term))
        risk = max(0.0, min(1.0, risk))
        node.last_next_state = str(sim.get("next_state_summary", "")).strip()

        return max(0.0, min(1.0, 0.55 * immediate + 0.45 * long_term - 0.35 * risk))

    def _run_lats_search(self, world_info: str) -> SearchNode:
        nodes = self._build_search_tree(world_info)
        for _ in range(self.tree_budget):
            total = sum(n.visits for n in nodes)
            node = max(nodes, key=lambda n: self._ucb_score(n, total))
            value = self._simulate_node(world_info, node)
            node.visits += 1
            node.value_sum += value

        return max(nodes, key=lambda n: n.mean_value)

    def cognitive_game_decide(self, world_info: str) -> Dict[str, Any]:
        """基于 LATS 的决策流程。"""
        self.round_counter += 1
        self._current_node = self._run_lats_search(world_info)
        action = self._current_node.action
        reasoning = self._current_node.thought or "LATS tree search selected this action"
        declaration = self._generate_declaration(action, reasoning)

        self.action_history.append(action)
        self.declaration_history.append(declaration)

        if self._current_node.last_next_state:
            self._world_model_state = self._current_node.last_next_state

        return {
            "action": action,
            "declaration": declaration,
            "reasoning_result": {
                "method": "lats_agent",
                "search_budget": self.tree_budget,
                "selected_node": {
                    "action": self._current_node.action,
                    "visits": self._current_node.visits,
                    "mean_value": self._current_node.mean_value,
                },
                "reasoning": reasoning,
                "final_satisfaction_score": self._current_node.mean_value,
                "reasoning_depth": self._current_node.visits,
            },
            "satisfaction_score": self._current_node.mean_value,
            "reasoning_depth": self._current_node.visits,
        }

    def _generate_declaration(self, action: str, reasoning: str) -> str:
        prompt = f"""为以下行动生成简短外交声明。

国家：{self.country_name}
行动：{action}
理由：{reasoning}

要求：1-2句话，简洁有力。
返回JSON：{{"declaration": "声明内容"}}
"""
        try:
            response = self.get_response(prompt, flag_debug_print=False)
            if isinstance(response, dict):
                return str(response.get("declaration", f"{self.country_name}决定{action}"))
        except Exception as e:
            self.experiment_logger.log_print(f"LATS生成声明失败: {e}", level="WARNING")
        return f"{self.country_name}决定{action}，以维护国家利益。"

    def learn_from_interaction(
        self,
        own_action: str,
        world_feedback: str,
        other_actions: Dict[str, str],
        world_memory: Any = None,
    ):
        """根据回合反馈进行反思与世界模型更新。"""
        reward = self._calculate_reward(world_feedback)
        self.last_reward = reward

        prompt = f"""你是LATS反思器，请根据反馈更新策略笔记与世界模型。

国家：{self.country_name}
本方行动：{own_action}
反馈：{world_feedback[:300]}
对手行动：{json.dumps(other_actions, ensure_ascii=False)}
当前世界模型：{self._world_model_state}

返回JSON：
{{
  "planner_note": "一句可执行改进建议",
  "world_model_state": "更新后的简洁世界模型"
}}
"""
        try:
            result = self.get_response(prompt, flag_debug_print=False)
            if isinstance(result, dict):
                note = str(result.get("planner_note", "")).strip()
                if note:
                    self._planner_notes.append(note)
                    self._planner_notes = self._planner_notes[-10:]
                world_model_state = str(result.get("world_model_state", "")).strip()
                if world_model_state:
                    self._world_model_state = world_model_state
        except Exception as e:
            self.experiment_logger.log_print(f"LATS反思失败: {e}", level="WARNING")

    def _calculate_reward(self, feedback: str) -> float:
        reward = 0.5
        feedback_lower = feedback.lower()
        positive_keywords = ["成功", "缓和", "合作", "支持", "胜利", "优势", "收益", "稳定"]
        negative_keywords = ["失败", "紧张", "对抗", "损失", "危机", "风险", "威胁", "冲突"]

        for kw in positive_keywords:
            if kw in feedback_lower:
                reward += 0.1
        for kw in negative_keywords:
            if kw in feedback_lower:
                reward -= 0.1
        return max(0.0, min(1.0, reward))

    def get_cognition_statistics(self) -> Dict[str, Any]:
        return {
            "framework": "LATSAgent",
            "total_decisions": len(self.action_history),
            "search_budget": self.tree_budget,
            "last_reward": self.last_reward,
            "planner_notes": self._planner_notes[-5:],
            "avg_score": self._current_node.mean_value if self._current_node else 0.5,
        }

    def export_cognition_report(self):
        report = {
            "agent_name": self.country_name,
            "framework": "LATSAgent",
            "world_model_state": self._world_model_state,
            "planner_notes": self._planner_notes,
            "decision_history": {
                "actions": self.action_history[-20:],
                "declarations": self.declaration_history[-10:],
            },
            "statistics": self.get_cognition_statistics(),
        }
        try:
            from pathlib import Path

            report_path = Path(f"{self.country_name}_lats_report.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.experiment_logger.log_print(f"导出LATS报告失败: {e}", level="WARNING")

        return report

    @property
    def action(self):
        return self.action_history

    @property
    def declaration(self):
        return self.declaration_history

    def get_best_strategy(self) -> Optional[Dict[str, Any]]:
        if not self._current_node:
            return None
        return {
            "action": self._current_node.action,
            "thought": self._current_node.thought,
            "mean_value": self._current_node.mean_value,
            "visits": self._current_node.visits,
        }
