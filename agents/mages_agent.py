from __future__ import annotations

import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

from simulation.models.agents.LLMAgent import LLMAgent
try:
    from simulation.models.cognitive.experiment_logger import ExperimentLogger  # type: ignore
except Exception:  # 实验记录器可选
    ExperimentLogger = Any  # type: ignore


@dataclass
class EvolutionaryCounts:
    """可指数衰减的频次统计，用于世界模型与对手画像。"""

    adaptation_rate: float = 0.65
    table: Dict[str, Dict[str, float]] = field(default_factory=lambda: defaultdict(dict))

    def update(self, key: str, label: str) -> None:
        bucket = self.table[key]
        if bucket:
            for feedback in list(bucket.keys()):
                bucket[feedback] *= self.adaptation_rate
                if bucket[feedback] < 1e-6:
                    del bucket[feedback]
        bucket[label] = bucket.get(label, 0.0) + 1.0

    def distribution(self, key: str, candidate_space: Optional[List[str]] = None) -> Dict[str, float]:
        bucket = self.table.get(key, {})
        labels = candidate_space or list(bucket.keys()) or ["unknown"]
        total = sum(bucket.values()) + len(labels)
        if total == 0:
            return {label: 1.0 / len(labels) for label in labels}
        return {label: (bucket.get(label, 0.0) + 1.0) / total for label in labels}


@dataclass
class PredictionRecord:
    target: str
    predicted_action: str
    support_evidence: str
    confidence: float


class MAGESAgent(LLMAgent):
    """MAGES Agent: Observe-Orient-Decide-Act 循环的通用实现。"""

    def __init__(
        self,
        country_name: str,
        other_countries: List[str],
        game_attributes: Optional[Dict[str, Any]] = None,
        experiment_logger: Optional[Any] = None,
        meta_goal: str = "Ensure survival and maximize national interest",
        adaptation_rate: float = 0.65,
        use_llm_reasoner: bool = False,
    ):
        super().__init__(
            agent_name=f"MAGES_{country_name}",
            has_chat_history=False,
            online_track=False,
            json_format=True,
        )
        self.country_name = country_name
        self.other_countries = other_countries
        self.game_attributes = game_attributes or {}
        self.experiment_logger = experiment_logger
        self.meta_goal = meta_goal
        self.adapter = adaptation_rate
        self.use_llm_reasoner = use_llm_reasoner

        self.world_model = EvolutionaryCounts(adaptation_rate)
        self.profile_tables: Dict[str, EvolutionaryCounts] = {
            opponent: EvolutionaryCounts(adaptation_rate)
            for opponent in other_countries
        }
        self.profile_notes: Dict[str, Deque[str]] = {
            opponent: deque(maxlen=5)
            for opponent in other_countries
        }
        self.opponent_latent_strategy: Dict[str, str] = {opponent: "" for opponent in other_countries}
        self.latest_experience: Optional[str] = None
        self.prediction_cache: Dict[int, List[PredictionRecord]] = {}
        self.current_strategy = "Calibrated deterrence with proportional responses"
        self.current_strategy_semantic: str = ""
        self.meta_state: Dict[str, Any] = {
            "round": 0,
            "phase": "Init",
            "tension": 0.0,
        }
        self.available_actions = [
            "HOLD",
            "MOVE",
            "ATTACK",
            "SUPPORT_ATTACK",
            "SUPPORT_DEFEND",
            "RETREAT",
        ]
        self.last_concrete_orders: List[str] = []

    # ------------------------------------------------------------------ Observe
    def observe(self, round_context: Dict[str, Any]) -> Dict[str, Any]:
        self.meta_state.update(
            {
                "round": round_context.get("round", self.meta_state.get("round", 0) + 1),
                "phase": round_context.get("phase", "Unknown"),
                "tension": round_context.get("tension", self.meta_state.get("tension", 0.0)),
            }
        )
        last_orders: Dict[str, str] = round_context.get("last_orders", {})
        last_feedback: Dict[str, str] = round_context.get("last_feedback", {})
        if last_feedback:
            feedback_label = last_feedback.get(self.country_name, "neutral")
            last_action = last_orders.get(self.country_name, "HOLD")
            self.world_model.update(last_action, feedback_label)
            self.latest_experience = self._llm_summarize_experience(last_action, feedback_label, round_context)
            self._log_cognition("world_model", {
                "action": last_action,
                "feedback": feedback_label,
                "distribution": self.world_model.distribution(last_action, self.available_actions),
                "experience": self.latest_experience,
            })

        # 对手经验/策略：一次性批量 LLM 调用（避免逐对手多次调用）
        opponents_to_update: List[str] = [o for o in self.other_countries if o in last_orders]
        for opponent in opponents_to_update:
            action = last_orders[opponent]
            self.profile_tables[opponent].update("GLOBAL", action)
            summary = f"R{self.meta_state['round']-1}: {opponent} executed {action}"
            self.profile_notes[opponent].append(summary)

        if opponents_to_update:
            batch = self._llm_batch_infer_opponents(opponents_to_update, last_orders, last_feedback)
            for opponent in opponents_to_update:
                rec = batch.get(opponent)
                if not isinstance(rec, dict):
                    continue
                strategy = rec.get("strategy")
                experience = rec.get("experience")
                if isinstance(strategy, str) and strategy.strip():
                    self.opponent_latent_strategy[opponent] = strategy.strip()
                if isinstance(experience, str) and experience.strip():
                    # 作为对手画像的“经验片段”写入 notes，供后续证据与摘要使用
                    self.profile_notes[opponent].append(f"LLM经验: {experience.strip()}")
        return {
            "last_orders": last_orders,
            "last_feedback": last_feedback,
            "tension": self.meta_state.get("tension", 0.0),
        }

    def _llm_batch_infer_opponents(
        self,
        opponents: List[str],
        last_orders: Dict[str, str],
        last_feedback: Dict[str, str],
    ) -> Dict[str, Dict[str, str]]:
        """一次 LLM 调用推断多个对手的(经验, 策略)。

        返回：{opponent: {"experience": "...", "strategy": "..."}}
        失败时返回空 dict（不再降级为逐对手多次调用）。
        """
        try:
            opponent_rows: List[Dict[str, Any]] = []
            for opponent in opponents:
                opponent_rows.append({
                    "opponent": opponent,
                    "last_action": last_orders.get(opponent, ""),
                    "feedback": last_feedback.get(opponent, ""),
                    "recent_notes": list(self.profile_notes.get(opponent, deque()))[-3:],
                })

            prompt = (
                "你是 Diplomacy(无外交) 场景的博弈分析师。\n"
                "请基于‘上一回合’观测信息，一次性为多个对手生成：\n"
                "1) 对手经验 experience：一句话总结对手上一回合行为的得失/意图（<=30字）\n"
                "2) 隐含策略 strategy：一句话推断其短期策略倾向（<=30字）\n\n"
                "提示：\n"
                "- 区分‘保守发育’(HOLD/SUPPORT)与‘积极扩张’(MOVE)。\n"
                "- 若对手上一轮保持(HOLD)但周围有空地，可能是在积蓄力量，需警惕突袭。\n"
                "- 若对手频繁移动(MOVE)，需警惕其进攻意图，标记为‘aggressive’。\n"
                "- Reflexion类对手会根据反馈调整，若上一轮失败，本轮可能改变策略。\n"
                "- 请在策略描述中包含关键词：aggressive, defensive, expansionist, conservative, unpredictable。\n\n"
                "你必须严格输出 JSON，格式如下：\n"
                "{{\"opponents\": {{\"France\": {{\"experience\": \"...\", \"strategy\": \"...\"}}, ...}}}}\n"
                "不要输出多余字段，不要输出除 JSON 外的任何文本。\n\n"
                "我方国家: {me}\n"
                "对手列表与观测：{rows}"
            )

            resp = self.get_response(prompt, input_param_dict={
                "me": self.country_name,
                "rows": opponent_rows,
            }, flag_debug_print=False)

            if not isinstance(resp, dict):
                return {}
            raw = resp.get("opponents")
            if not isinstance(raw, dict):
                return {}

            out: Dict[str, Dict[str, str]] = {}
            for opponent, rec in raw.items():
                if not isinstance(opponent, str) or not isinstance(rec, dict):
                    continue
                exp = rec.get("experience")
                strat = rec.get("strategy")
                out[opponent] = {
                    "experience": str(exp) if isinstance(exp, str) else "",
                    "strategy": str(strat) if isinstance(strat, str) else "",
                }
            return out
        except Exception:
            return {}

    # ------------------------------------------------------------------ Orient
    def orient(self, round_context: Dict[str, Any]) -> Dict[str, Any]:
        personas = round_context.get("personas", {})
        predictions: List[PredictionRecord] = []
        for opponent in self.other_countries:
            persona_hint = personas.get(opponent, "Unknown")
            if (not persona_hint or persona_hint == "Unknown") and self.opponent_latent_strategy.get(opponent):
                persona_hint = self.opponent_latent_strategy[opponent]
            
            dist = self.profile_tables[opponent].distribution("GLOBAL", self.available_actions)
            bias = self._persona_bias(persona_hint)
            scored = {
                action: dist.get(action, 1.0 / len(self.available_actions)) + bias.get(action, 0.0)
                for action in self.available_actions
            }
            predicted_action = max(scored.items(), key=lambda item: item[1])[0]
            confidence = self._normalize_confidence(scored[predicted_action], scored.values())
            evidence = ", ".join(list(self.profile_notes[opponent])[-2:])
            predictions.append(
                PredictionRecord(
                    target=opponent,
                    predicted_action=predicted_action,
                    support_evidence=evidence or "Insufficient history",
                    confidence=confidence,
                )
            )
        reflection = self._goal_anchored_reflection(round_context)
        strategy_text = self._llm_evolve_strategy(reflection, predictions)
        pruned_actions = self._llm_prune_actions(strategy_text, self.available_actions)
        candidate_actions = pruned_actions or self._semantic_action_filter(reflection)

        predictions_map: Dict[str, str] = {rec.target: rec.predicted_action for rec in predictions}
        predicted_states: Dict[str, str] = {}
        for action in candidate_actions:
            predicted_feedback = self._predict_feedback_label(action)
            predicted_reaction = predictions_map.get(self._pick_primary_opponent(), "HOLD")
            env_exp = self.latest_experience or ""
            opp_strategy = self.opponent_latent_strategy.get(self._pick_primary_opponent(), "")
            opp_exp = "; ".join(list(self.profile_notes.get(self._pick_primary_opponent(), deque()))[-2:])
            predicted_states[action] = self._llm_synthesize_state(
                action,
                predicted_feedback,
                predicted_reaction,
                env_exp,
                opp_strategy,
                opp_exp,
            )

        orient_state = {
            "predictions": predictions,
            "reflection": reflection,
            "strategy": strategy_text,
            "candidate_actions": candidate_actions,
            "predicted_states": predicted_states,
        }
        self.prediction_cache[round_context.get("round", 0)] = predictions
        return orient_state

    # ------------------------------------------------------------------ Decide
    def decide(self, round_context: Dict[str, Any], orient_state: Dict[str, Any]) -> Dict[str, Any]:
        candidate_actions = orient_state.get("candidate_actions", self.available_actions)
        predictions = {rec.target: rec.predicted_action for rec in orient_state.get("predictions", [])}
        predicted_states = orient_state.get("predicted_states", {})
        strategy_text = orient_state.get("strategy", self.current_strategy)

        llm_scores = self._llm_score_actions(strategy_text, predicted_states)
        scores: Dict[str, float] = {}
        for action in candidate_actions:
            align = llm_scores.get(action, {}).get("align", self._strategy_alignment(action))
            dev = llm_scores.get(action, {}).get("deviation", 0.5)
            scores[action] = align * (1.0 - dev)
        best_action = max(scores.items(), key=lambda kv: kv[1])[0]
        decision: Dict[str, Any] = {
            "selected_action": best_action,
            "utility": scores[best_action],
            "scores": scores,
            "predictions": predictions,
            "llm_scores": llm_scores,
        }

        # --- Tactical layer: choose concrete legal orders inside OODA (if provided) ---
        legal_orders_by_loc = round_context.get("legal_orders_by_loc")
        game_state = round_context.get("game_state")
        if isinstance(legal_orders_by_loc, dict) and isinstance(game_state, dict):
            try:
                concrete = self.propose_legal_orders(
                    round_context=round_context,
                    game_state=game_state,
                    legal_orders_by_loc=legal_orders_by_loc,
                    high_level_action=best_action,
                    strategy_text=strategy_text,
                )
                if isinstance(concrete, list):
                    decision["concrete_orders"] = concrete
            except Exception as e:
                decision["concrete_orders_error"] = repr(e)
        return decision

    # ------------------------------------------------------------------ Act
    def act(self, decision: Dict[str, Any]) -> str:
        chosen = decision.get("selected_action", "HOLD")
        self.current_strategy = self._update_current_strategy(decision)
        concrete = decision.get("concrete_orders")
        if isinstance(concrete, list):
            self.last_concrete_orders = [str(o) for o in concrete if isinstance(o, str)]
        return chosen

    def get_last_concrete_orders(self) -> List[str]:
        return list(self.last_concrete_orders)

    # ------------------------------------------------------------------ High-level API
    def run_cycle(self, round_context: Dict[str, Any]) -> Dict[str, Any]:
        observation = self.observe(round_context)
        orient_state = self.orient(round_context)
        decision = self.decide(round_context, orient_state)
        action = self.act(decision)
        record = {
            "action": action,
            "observation": observation,
            "orient": orient_state,
            "decision": decision,
        }
        self._log_cognition("decision", record)
        return record

    # ------------------------------------------------------------------ Concrete orders (Diplomacy)
    _DIPLOMACY_KNOWLEDGE: str = (
        "Diplomacy(标准地图)要点：\n"
        "- 每回合需要为每个可下单地点(通常=每个我方单位所在地)提交 1 条指令。\n"
        "- 合法指令以引擎提供的 legal orders 为准；必须逐条原样选取，不能自造指令。\n"
        "- 常见指令：H(保持)、- (移动)、S(支援移动/支援保持)、C(海军运输)。\n"
        "- 目标优先级：【最高优先级】争夺中立补给中心(SC)以扩大国力；保住本土；避免无谓对撞。\n"
        "- 英格兰常见思路：优先确保北海/英吉利海峡影响力，巩固本土后再争夺挪威/比利时等。\n"
    )

    def propose_legal_orders(
        self,
        round_context: Dict[str, Any],
        game_state: Dict[str, Any],
        legal_orders_by_loc: Dict[str, List[str]],
        high_level_action: str,
        strategy_text: str,
    ) -> List[str]:
        """为当前回合生成逐地点的具体 orders（必须从 legal_orders_by_loc 中选择）。"""
        power = self.country_name.upper()
        my_units = (game_state.get("units", {}) or {}).get(power, [])
        my_centers = (game_state.get("centers", {}) or {}).get(power, [])
        all_units = game_state.get("units", {}) or {}
        enemy_units = {k: v for k, v in all_units.items() if k != power}

        def _hold_for(loc: str) -> Optional[str]:
            opts = legal_orders_by_loc.get(loc, [])
            for o in opts:
                if isinstance(o, str) and o.strip().endswith(" H"):
                    return o
            return opts[0] if opts else None

        prompt = (
            "你是 Diplomacy(标准地图) 的英国(ENGLAND)战术指挥官。\n"
            "你必须只从给定的合法 orders 列表中挑选指令，不能发明新指令。\n\n"
            "背景知识(用于理解局势与取舍)：\n{knowledge}\n"
            "元目标: {goal}\n"
            "回合信息: round={round} phase={phase} tension={tension}\n"
            "高层意图(抽象动作): {high_level_action}\n"
            "动态策略: {strategy}\n\n"
            "我方单位: {my_units}\n"
            "我方补给中心: {my_centers}\n"
            "敌方单位(摘要): {enemy_units}\n"
            "上回合动作(抽象): {last_orders}\n"
            "上回合反馈: {last_feedback}\n\n"
            "可下单地点 -> 合法 orders 列表(只能从中选)：\n{legal_orders}\n\n"
            "任务：为每个可下单地点选择 1 条最合适的 order。\n"
            "硬约束：每个地点恰好 1 条；输出必须是合法 order 的原样字符串。\n"
            "输出严格 JSON：{{\"orders_by_loc\": {{\"LON\": \"...\", \"EDI\": \"...\"}}, \"rationale\": \"<=200字\"}}"
        )

        resp = self.get_response(
            prompt,
            input_param_dict={
                "knowledge": self._DIPLOMACY_KNOWLEDGE,
                "goal": self.meta_goal,
                "round": round_context.get("round"),
                "phase": round_context.get("phase"),
                "tension": round_context.get("tension"),
                "high_level_action": high_level_action,
                "strategy": strategy_text or self.current_strategy,
                "my_units": my_units,
                "my_centers": my_centers,
                "enemy_units": enemy_units,
                "last_orders": round_context.get("last_orders", {}),
                "last_feedback": round_context.get("last_feedback", {}),
                "legal_orders": legal_orders_by_loc,
            },
            flag_debug_print=False,
        )

        chosen_by_loc: Dict[str, str] = {}
        if isinstance(resp, dict):
            raw = resp.get("orders_by_loc")
            if isinstance(raw, dict):
                for loc, order in raw.items():
                    if isinstance(loc, str) and isinstance(order, str):
                        chosen_by_loc[loc.strip().upper()] = order.strip()

        final_orders: List[str] = []
        for loc in (legal_orders_by_loc.keys() or []):
            loc_u = str(loc).upper()
            cand = chosen_by_loc.get(loc_u)
            legal_list = legal_orders_by_loc.get(loc_u, legal_orders_by_loc.get(loc, [])) or []
            if cand and cand in legal_list:
                final_orders.append(cand)
                continue
            hold = _hold_for(loc_u)
            if hold:
                final_orders.append(hold)

        # 如果 legal_orders_by_loc 为空，回退为全 HOLD（尽量不影响运行）
        if not final_orders and my_units:
            for unit in my_units:
                parts = str(unit).split()
                loc = parts[1] if len(parts) > 1 else ""
                hold = _hold_for(loc)
                if hold:
                    final_orders.append(hold)
        return final_orders

    def learn_from_interaction(
        self,
        action: str,
        world_feedback: str,
        other_actions: Dict[str, str],
        world_memory: Any = None,
    ) -> None:
        feedback_label = self._classify_feedback(world_feedback)
        self.world_model.update(action, feedback_label)
        for opponent, op_action in other_actions.items():
            if opponent not in self.profile_tables:
                continue
            self.profile_tables[opponent].update("GLOBAL", op_action)
            self.profile_notes[opponent].append(
                f"R{self.meta_state.get('round', 0)} actual {opponent}:{op_action}"
            )
        self._log_cognition("learn", {
            "action": action,
            "feedback": feedback_label,
            "other_actions": other_actions,
        })

    # ------------------------------------------------------------------ Helpers
    def _persona_bias(self, persona: str) -> Dict[str, float]:
        if not persona:
            return {}
        persona = persona.lower()
        if "aggressive" in persona or "expansion" in persona:
            return {"ATTACK": 0.4, "SUPPORT_ATTACK": 0.25}
        if "turtle" in persona or "defender" in persona or "defensive" in persona or "conservative" in persona:
            return {"HOLD": 0.5, "SUPPORT_DEFEND": 0.35}
        if "random" in persona or "chaotic" in persona or "unpredictable" in persona:
            return {action: random.uniform(-0.1, 0.1) for action in self.available_actions}
        if "economic" in persona or "builder" in persona:
            return {"MOVE": 0.3, "SUPPORT_DEFEND": 0.2}
        return {}

    def _normalize_confidence(self, best_value: float, values) -> float:
        min_v = min(values)
        max_v = max(values)
        if math.isclose(max_v, min_v):
            return 0.5
        return (best_value - min_v) / (max_v - min_v)

    def _goal_anchored_reflection(self, round_context: Dict[str, Any]) -> str:
        last_feedback = round_context.get("last_feedback", {}).get(self.country_name)
        round_no = round_context.get("round", 0)

        # Early game aggression injection: Force expansion if no gains yet
        if round_no <= 6 and (not last_feedback or "gain" not in str(last_feedback).lower()):
            return "Early game phase: Imperative to secure neutral Supply Centers (SCs) immediately to avoid stagnation."

        if not last_feedback:
            return f"Meta-goal {self.meta_goal} remains intact; maintain deterrence baseline."
        if "loss" in last_feedback.lower():
            return "Detected attrition against core interests; pivot to defensive deterrence."
        if "gain" in last_feedback.lower():
            return "Gains recorded; cautiously extend influence with limited aggression."
        return "Environment neutral; preserve optionality and monitor opponent pivots."

    def _semantic_action_filter(self, reflection: str) -> List[str]:
        if "defensive" in reflection.lower() or "attrition" in reflection.lower():
            return ["HOLD", "SUPPORT_DEFEND", "RETREAT"]
        if "extend" in reflection.lower() or "gain" in reflection.lower():
            return ["MOVE", "ATTACK", "SUPPORT_ATTACK"]
        return self.available_actions

    def _strategy_alignment(self, action: str) -> float:
        is_defensive = "defensive" in self.current_strategy.lower() or "deterrence" in self.current_strategy.lower()

        if action == "HOLD":
            return 0.7 if is_defensive else 0.3
        if action == "SUPPORT_DEFEND":
            return 0.8 if is_defensive else 0.4
        if action == "MOVE":
            return 0.9  # Always favor movement/expansion by default
        if action in ("ATTACK", "SUPPORT_ATTACK"):
            return 0.7
        return 0.4

    # ----------------------------- LLM 辅助模块 -----------------------------

    def _llm_summarize_experience(self, action: str, feedback: str, round_context: Dict[str, Any]) -> Optional[str]:
        try:
            prompt = (
                "你是国家决策助手，请把当前回合的因果链总结为一段自然语言经验。\n"
                "已知我方动作: {action}\n"
                "环境反馈: {feedback}\n"
                "额外上下文: {context}\n"
                "请输出 JSON，如 {{\"experience\": \"...\"}}."
            )
            resp = self.get_response(prompt, input_param_dict={
                "action": action,
                "feedback": feedback,
                "context": round_context,
            }, flag_debug_print=False)
            return resp.get("experience") if isinstance(resp, dict) else None
        except Exception:
            return None

    def _llm_infer_opponent_strategy(self, opponent: str, reaction: str, feedback: str) -> Optional[str]:
        try:
            prompt = (
                "你是博弈分析师，请根据对手的反应推断其隐含策略。\n"
                "对手: {opponent}\n"
                "观测反应 r: {reaction}\n"
                "环境反馈(如有): {feedback}\n"
                "返回 JSON: {{\"strategy\": \"...\"}}."
            )
            resp = self.get_response(prompt, input_param_dict={
                "opponent": opponent,
                "reaction": reaction,
                "feedback": feedback,
            }, flag_debug_print=False)
            return resp.get("strategy") if isinstance(resp, dict) else None
        except Exception:
            return None

    def _llm_evolve_strategy(self, reflection: str, predictions: List[PredictionRecord]) -> str:
        try:
            prompt = (
                "你是战略元认知模块，请基于元目标生成本轮动态策略 St。\n"
                "元目标 G_static: {goal}\n"
                "反思: {reflection}\n"
                "对手预测: {predictions}\n"
                "输出 JSON: {{\"strategy\":\"...\", \"target_outcome\":\"...\"}}."
            )
            resp = self.get_response(prompt, input_param_dict={
                "goal": self.meta_goal,
                "reflection": reflection,
                "predictions": [p.__dict__ for p in predictions],
            }, flag_debug_print=False)
            strategy = resp.get("strategy") if isinstance(resp, dict) else None
            target = resp.get("target_outcome") if isinstance(resp, dict) else None
            if strategy:
                self.current_strategy = strategy
            if target:
                self.current_strategy_semantic = target
            return strategy or self.current_strategy
        except Exception:
            return self.current_strategy

    def _llm_prune_actions(self, strategy: str, actions: List[str]) -> List[str]:
        try:
            prompt = (
                "根据当前策略筛选动作集，只保留符合策略的动作。最多筛选三个动作\n"
                "策略: {strategy}\n"
                "全量动作: {actions}\n"
                "返回 JSON: {{\"actions\": [\"HOLD\", ...]}}")
            resp = self.get_response(prompt, input_param_dict={
                "strategy": strategy,
                "actions": actions,
            }, flag_debug_print=False)
            pruned = resp.get("actions") if isinstance(resp, dict) else None
            if pruned and isinstance(pruned, list):
                return [a for a in pruned if a in actions]
            return []
        except Exception:
            return []

    def _predict_feedback_label(self, action: str) -> str:
        dist = self.world_model.distribution(action, ["positive", "neutral", "negative"])
        return max(dist.items(), key=lambda kv: kv[1])[0]

    def _pick_primary_opponent(self) -> str:
        return self.other_countries[0] if self.other_countries else "Unknown"

    def _llm_synthesize_state(
        self,
        action: str,
        predicted_feedback: str,
        predicted_reaction: str,
        env_experience: str,
        opp_strategy: str,
        opp_experience: str,
    ) -> str:
        try:
            prompt = (
                "你是未来情景合成器，请基于统计预测与历史经验构造具象化的未来图景 S_pred。\n"
                "候选动作 a: {action}\n"
                "预测环境反馈 f~: {predicted_feedback}\n"
                "预测对手反制 r~: {predicted_reaction}\n"
                "环境经验 e_env: {env_exp}\n"
                "对手策略 s: {opp_strategy}\n"
                "对手经验 e_opp: {opp_exp}\n"
                "返回 JSON: {{\"state\": \"...\"}}."
            )
            resp = self.get_response(prompt, input_param_dict={
                "action": action,
                "predicted_feedback": predicted_feedback,
                "predicted_reaction": predicted_reaction,
                "env_exp": env_experience,
                "opp_strategy": opp_strategy,
                "opp_exp": opp_experience,
            }, flag_debug_print=False)
            return resp.get("state") if isinstance(resp, dict) else ""
        except Exception:
            return ""

    def _llm_score_actions(self, strategy: str, predicted_states: Dict[str, str]) -> Dict[str, Dict[str, float]]:
        try:
            prompt = (
                "请为每个候选动作打分，输出 align(0-1) 与 deviation(0-1)。\n"
                "策略 St: {strategy}\n"
                "预测状态: {states}\n"
                "返回 JSON 形如 {{\"HOLD\": {{\"align\":0.6, \"deviation\":0.2}}, ...}}"
            )
            resp = self.get_response(prompt, input_param_dict={
                "strategy": strategy,
                "states": predicted_states,
            }, flag_debug_print=False)
            if isinstance(resp, dict):
                out: Dict[str, Dict[str, float]] = {}
                for k, v in resp.items():
                    if isinstance(v, dict):
                        align = float(v.get("align", 0.5))
                        dev = float(v.get("deviation", 0.5))
                        out[k] = {"align": max(0.0, min(1.0, align)), "deviation": max(0.0, min(1.0, dev))}
                return out
            return {}
        except Exception:
            return {}

    def _risk_eval(self, action: str, predictions: Dict[str, str]) -> float:
        threat = sum(1 for act in predictions.values() if act in ("ATTACK", "SUPPORT_ATTACK"))
        threat_ratio = threat / max(1, len(predictions))
        if action == "ATTACK":
            return 0.5 + 0.5 * threat_ratio
        if action == "MOVE":
            return 0.3 * (0.5 + threat_ratio)
        if action == "HOLD":
            return 0.2 * threat_ratio
        if "SUPPORT" in action:
            return 0.35 * threat_ratio
        return 0.4

    def _update_current_strategy(self, decision: Dict[str, Any]) -> str:
        action = decision.get("selected_action", "HOLD")
        if action in ("ATTACK", "SUPPORT_ATTACK"):
            return "Limited offensive surge with escalation guardrails"
        if action in ("HOLD", "SUPPORT_DEFEND"):
            return "Layered deterrence and territorial integrity focus"
        if action == "MOVE":
            return "Flexible redeployment to balance fronts"
        return self.current_strategy

    def _classify_feedback(self, feedback: str) -> str:
        if not feedback:
            return "neutral"
        normalized = feedback.lower()
        if any(word in normalized for word in ["gain", "capture", "success"]):
            return "positive"
        if any(word in normalized for word in ["loss", "retreat", "defeat"]):
            return "negative"
        return "neutral"

    def _log_cognition(self, label: str, payload: Dict[str, Any]) -> None:
        if not self.experiment_logger:
            return
        try:
            self.experiment_logger.log_cognition_update(
                agent_name=self.country_name,
                cognition_type=label,
                data=payload,
            )
        except Exception:
            pass

    # 兼容旧接口
    def cognitive_game_decide(self, world_info: str) -> Dict[str, Any]:
        context = {"world_info": world_info, "round": self.meta_state.get("round", 0) + 1}
        result = self.run_cycle(context)
        return {
            "action": result["action"],
            "declaration": self.current_strategy,
            "reasoning_result": result,
            "satisfaction_score": result["decision"].get("utility", 0.5),
            "reasoning_depth": 3,
        }
