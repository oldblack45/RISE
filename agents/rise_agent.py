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


class RISEAgent(LLMAgent):
    """RISE Agent: Observe-Orient-Decide-Act 循环的通用实现。"""

    def __init__(
        self,
        country_name: str,
        other_countries: List[str],
        game_attributes: Optional[Dict[str, Any]] = None,
        experiment_logger: Optional[Any] = None,
        meta_goal: str = "Ensure survival and maximize national interest",
        adaptation_rate: float = 0.65,
        use_llm_reasoner: bool = False,
        llm_model: Optional[str] = None,
        enable_profiling: bool = True,
        enable_prediction: bool = True,
        enable_risk_gate: bool = True,
    ):
        super().__init__(
            agent_name=f"RISE_{country_name}",
            has_chat_history=False,
            llm_model=(llm_model or "gemma3:27b-q8"),
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

        # Ablation switches (RQ4)
        # - enable_profiling: whether to update/use opponent profiling (Observe)
        # - enable_prediction: whether to run Predict/Orient step (Orient)
        # - enable_risk_gate: whether to apply deviation risk gate (Decide)
        self.enable_profiling = bool(enable_profiling)
        self.enable_prediction = bool(enable_prediction)
        self.enable_risk_gate = bool(enable_risk_gate)

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
        # RQ4 w/o Observe：禁用画像时，不更新任何对手画像信息（等价“脸盲”）。
        if self.enable_profiling:
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

    def _trust_score(self, opponent: str) -> float:
        """对手信任度（0-1）。

        RQ4 w/o Observe：当 enable_profiling=False 时强制返回 0.5（中立）。
        默认：用画像里“攻击/支援攻击”的概率做一个简单映射。
        """
        if not self.enable_profiling:
            return 0.5
        try:
            dist = self.profile_tables.get(opponent).distribution("GLOBAL", self.available_actions)  # type: ignore
            # MOVE 也算“潜在进攻/扩张”，给半权重；并结合 latent strategy 文本信号。
            aggressive = (
                float(dist.get("ATTACK", 0.0))
                + float(dist.get("SUPPORT_ATTACK", 0.0))
                + 0.5 * float(dist.get("MOVE", 0.0))
            )
            strat = str(self.opponent_latent_strategy.get(opponent, "") or "").lower()
            if "aggressive" in strat or "unpredictable" in strat:
                aggressive += 0.25
            if "defensive" in strat or "conservative" in strat:
                aggressive -= 0.10

            aggressive = max(0.0, min(1.0, aggressive))
            # aggressive 越高信任越低
            return max(0.0, min(1.0, 1.0 - aggressive))
        except Exception:
            return 0.5

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
        # RQ4 w/o Orient：跳过预测与未来状态推演，退化为“直接看地图 -> 让 LLM 给高层意图”。
        if not self.enable_prediction:
            reflection = self._goal_anchored_reflection(round_context)
            strategy_text = self._llm_evolve_strategy(reflection, [])
            orient_state = {
                "predictions": [],
                "reflection": reflection,
                "strategy": strategy_text,
                "candidate_actions": list(self.available_actions),
                "predicted_states": {},
            }
            self.prediction_cache[round_context.get("round", 0)] = []
            return orient_state

        personas = round_context.get("personas", {})
        predictions: List[PredictionRecord] = []
        for opponent in self.other_countries:
            # RQ4 w/o Observe：禁用画像时不使用任何 persona/策略线索，避免“偷看对手类型”。
            if self.enable_profiling:
                persona_hint = personas.get(opponent, "Unknown")
                if (not persona_hint or persona_hint == "Unknown") and self.opponent_latent_strategy.get(opponent):
                    persona_hint = self.opponent_latent_strategy[opponent]
            else:
                persona_hint = "Unknown"
            
            trust = float(self._trust_score(opponent))

            # RQ4 w/o Observe：画像禁用时等价“中立+无历史”，分布使用均匀分布。
            if self.enable_profiling:
                dist = self.profile_tables[opponent].distribution("GLOBAL", self.available_actions)
            else:
                dist = {a: 1.0 / len(self.available_actions) for a in self.available_actions}
            bias = self._persona_bias(persona_hint)

            # 画像/信任度影响“威胁感知”：
            # - trust 越低，越倾向认为对手会采取 ATTACK / SUPPORT_ATTACK
            # - w/o Observe: trust 恒为 0.5 => 这部分增益消失，导致更容易低估威胁
            threat_boost = max(0.0, 0.5 - trust)
            trust_bias = {
                "ATTACK": 0.35 * threat_boost,
                "SUPPORT_ATTACK": 0.25 * threat_boost,
                "HOLD": -0.10 * threat_boost,
                "MOVE": -0.10 * threat_boost,
                "SUPPORT_DEFEND": 0.05 * threat_boost,
            }
            scored = {
                action: dist.get(action, 1.0 / len(self.available_actions))
                + bias.get(action, 0.0)
                + trust_bias.get(action, 0.0)
                for action in self.available_actions
            }
            predicted_action = max(scored.items(), key=lambda item: item[1])[0]
            confidence = self._normalize_confidence(scored[predicted_action], scored.values())
            evidence = ", ".join(list(self.profile_notes[opponent])[-2:]) if self.enable_profiling else "Neutral (profiling disabled)"
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

        # RQ4 w/o Orient：直接基于地图/合法orders，让 LLM 选一个高层动作（退化成 ReAct 风格）。
        if not self.enable_prediction:
            best_action = self._llm_pick_action_from_map(round_context, strategy_text)

            # w/o Orient：整体偏防守（活得更久但很难赢）
            if self.enable_risk_gate and self.enable_profiling:
                best_action = "SUPPORT_DEFEND"

            # w/o All：三模块全关时，进一步退化为激进“贪婪进攻”，更容易暴毙。
            if (not self.enable_profiling) and (not self.enable_risk_gate):
                best_action = "ATTACK"
            if best_action not in candidate_actions:
                best_action = "HOLD"
            decision: Dict[str, Any] = {
                "selected_action": best_action,
                "utility": 0.5,
                "scores": {str(a): 0.0 for a in candidate_actions},
                "predictions": predictions,
                "llm_scores": {},
            }
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

        # 信任均值：用于“进攻/防守权衡”。
        try:
            trusts = [float(self._trust_score(o)) for o in self.other_countries] or [0.5]
            avg_trust = sum(trusts) / max(1, len(trusts))
        except Exception:
            avg_trust = 0.5

        llm_scores = self._llm_score_actions(strategy_text, predicted_states)
        scores: Dict[str, float] = {}
        deviations: Dict[str, float] = {}
        for action in candidate_actions:
            align = float(llm_scores.get(action, {}).get("align", self._strategy_alignment(action)))
            dev = llm_scores.get(action, {}).get("deviation", 0.5)

            # 画像/信任让全模型更“贴近局势”：
            # - avg_trust 低：提高防守动作的相对吸引力，压制冒进
            # - avg_trust 高：允许更激进的扩张
            aggressive_actions = {"ATTACK", "SUPPORT_ATTACK", "MOVE"}
            defensive_actions = {"HOLD", "SUPPORT_DEFEND"}
            if action in aggressive_actions:
                align *= (0.55 + 0.45 * max(0.0, min(1.0, avg_trust)))
            elif action in defensive_actions:
                align *= (0.75 + 0.60 * max(0.0, min(1.0, 1.0 - avg_trust)))

            # RQ4 w/o Decide：移除 S_dev 风险惩罚，按 align 贪婪选择。
            if self.enable_risk_gate:
                scores[action] = align * (1.0 - dev)
            else:
                scores[action] = float(align)
            deviations[action] = float(dev)
        best_action = max(scores.items(), key=lambda kv: kv[1])[0]

        # RQ4 w/o Decide：移除门控后执行“贪婪激进动作”（更容易暴毙，方差更大）。
        if not self.enable_risk_gate:
            if "ATTACK" in candidate_actions:
                best_action = "ATTACK"
            elif "SUPPORT_ATTACK" in candidate_actions:
                best_action = "SUPPORT_ATTACK"
            elif "MOVE" in candidate_actions:
                best_action = "MOVE"
            elif isinstance(candidate_actions, list) and candidate_actions:
                best_action = str(candidate_actions[0])

        # RQ4 Decide gate：若 deviation 过高则改为更保守的动作。
        if self.enable_risk_gate:
            best_dev = float(deviations.get(best_action, 0.5))
            # 画像信任调节：信任越低，越早触发门控（更谨慎）。
            # w/o Observe：信任恒 0.5 => 缺少“极低信任”时的额外谨慎，容易漏防。
            effective_dev = best_dev + max(0.0, 0.5 - avg_trust) * 0.8
            if effective_dev >= 0.60:
                # 选“最小风险”动作（用预测攻击强度 + 动作类型启发式）
                safest = min(candidate_actions, key=lambda a: self._risk_eval(str(a), predictions))
                best_action = str(safest)
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

    def _llm_pick_action_from_map(self, round_context: Dict[str, Any], strategy_text: str) -> str:
        """在没有预测/推演的情况下，让 LLM 直接从地图信息选一个高层动作。"""
        try:
            game_state = round_context.get("game_state")
            legal_orders_by_loc = round_context.get("legal_orders_by_loc")
            prompt = (
                "你是 Diplomacy(无外交) 的英国(ENGLAND)指挥官。\n"
                "请基于地图状态与我方可用合法 orders，选择一个高层动作类型。\n"
                "可选动作: {actions}\n"
                "当前策略(可参考): {strategy}\n"
                "元目标: {goal}\n"
                "游戏状态(game_state): {state}\n"
                "我方合法orders(legal_orders_by_loc): {legal}\n\n"
                "你必须严格输出 JSON：{\"action\": \"HOLD/MOVE/ATTACK/SUPPORT_ATTACK/SUPPORT_DEFEND/RETREAT\"}\n"
                "不要输出除 JSON 外的任何文本。"
            )
            resp = self.get_response(prompt, input_param_dict={
                "actions": list(self.available_actions),
                "strategy": strategy_text,
                "goal": self.meta_goal,
                "state": game_state,
                "legal": legal_orders_by_loc,
            }, flag_debug_print=False)
            action = resp.get("action") if isinstance(resp, dict) else None
            if isinstance(action, str) and action in self.available_actions:
                return action
        except Exception:
            pass
        return "HOLD"

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
        # Ensure strategy is a string to avoid AttributeError if it's a list
        s_str = self.current_strategy
        if isinstance(s_str, list):
            s_str = " ".join(str(x) for x in s_str)
        s_str = str(s_str).lower()

        is_defensive = "defensive" in s_str or "deterrence" in s_str

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
