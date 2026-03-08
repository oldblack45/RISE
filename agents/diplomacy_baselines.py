from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from simulation.models.agents.LLMAgent import LLMAgent, set_llm_log_context


ACTIONS: List[str] = [
    "HOLD",
    "MOVE",
    "ATTACK",
    "SUPPORT_ATTACK",
    "SUPPORT_DEFEND",
    "RETREAT",
]


def _escape_curly(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")


def _dump(obj: Any) -> str:
    try:
        return _escape_curly(json.dumps(obj, ensure_ascii=False, sort_keys=True))
    except Exception:
        return _escape_curly(str(obj))


def _safe_compact_state(state: Dict[str, Any], country: str, max_items: int = 12) -> str:
    """把 diplomacy.Game.get_state() 压缩成适合塞进 prompt 的短文本。"""
    key = country.upper()
    units = state.get("units", {}).get(key, [])
    centers = state.get("centers", {}).get(key, [])
    phase = state.get("name", "")
    year = state.get("year", "")
    me_units = ", ".join(units[:max_items]) if isinstance(units, list) else str(units)
    me_centers = ", ".join(centers[:max_items]) if isinstance(centers, list) else str(centers)
    return _escape_curly(
        f"Phase={phase} Year={year}\n"
        f"MyUnits({len(units) if isinstance(units, list) else '?'})={me_units}\n"
        f"MySC({len(centers) if isinstance(centers, list) else '?'})={me_centers}"
    )


def _coerce_action(value: Any) -> str:
    if not isinstance(value, str):
        return "HOLD"
    value = value.strip().upper()
    return value if value in ACTIONS else "HOLD"


@dataclass
class BaselineMemory:
    last_actions: List[str] = field(default_factory=list)
    reflections: List[str] = field(default_factory=list)
    world_model_state: str = ""


class _LLMBaselineBase:
    """Diplomacy Baseline 抽象基类：统一 LLM 调用、动作规范化、记忆字段。"""

    def __init__(
        self,
        name: str,
        baseline_type: str,
        llm_model: Optional[str] = None,
    ):
        self.name = name
        self.baseline_type = baseline_type
        self.persona = baseline_type  # 用于 tournament 日志字段复用
        self.last_order: str = "HOLD"
        self.memory = BaselineMemory()
        self._llm = LLMAgent(
            agent_name=f"Baseline_{baseline_type}_{name}",
            has_chat_history=False,
            llm_model=(llm_model or "gemma3:27b-q8"),
            online_track=False,
            json_format=True,
        )

    async def propose_orders(self, game_view: Dict[str, Any]) -> str:
        try:
            set_llm_log_context({
                "scenario": "diplomacy",
                "role": "baseline",
                "baseline_type": self.baseline_type,
                "country": self.name,
                "round": game_view.get("round"),
                "phase": (game_view.get("state") or {}).get("name"),
                "stage": "propose_orders",
            })
        except Exception:
            pass
        return await asyncio.to_thread(self._propose_orders_sync, game_view)

    def post_round_update(
        self,
        game_view: Dict[str, Any],
        feedback_label: str,
        sc_delta: int,
    ) -> None:
        """回合结算后更新记忆（可选实现）。"""
        _ = (game_view, feedback_label, sc_delta)

    # --- internal
    def _propose_orders_sync(self, game_view: Dict[str, Any]) -> str:
        raise NotImplementedError


class ReActBaselineAgent(_LLMBaselineBase):
    """ReAct：短上下文（1-2 回合）+ Few-shot Thought/Action，战术敏锐但战略短视。"""

    def __init__(self, name: str, llm_model: Optional[str] = None):
        super().__init__(name=name, baseline_type="ReAct", llm_model=llm_model)

    def _propose_orders_sync(self, game_view: Dict[str, Any]) -> str:
        state = game_view.get("state") or {}
        last_orders = game_view.get("last_orders") or {}
        last_feedback = game_view.get("last_feedback") or {}
        tension = float(game_view.get("tension", 0.5))

        # 短历史窗口：只保留最近 2 条自己的抽象动作
        recent = (self.memory.last_actions or [])[-2:]
        recent_text = " | ".join(recent) if recent else "(none)"

        observation = (
            "PlaintextObservation:\n"
            f"You are {self.name}.\n"
            f"{_safe_compact_state(state, self.name)}\n"
            f"Tension={tension:.2f}\n"
            f"LastTurnOrders={_dump(last_orders)}\n"
            f"LastTurnFeedback={_dump(last_feedback)}\n"
            f"RecentSelfActions={recent_text}\n"
        )

        fewshot = (
            "Example 1\n"
            "PlaintextObservation: Phase=S1902M Year=1902 MyUnits=F LON, A LVP MySC=LON, LVP\n"
            "Thought: 1) Threat: Germany active. 2) Opportunity: secure tempo. 3) Choose defensive support.\n"
            "Action: {{\"abstract_action\":\"SUPPORT_DEFEND\", \"daide\":\"(HLD)\"}}\n\n"
            "Example 2\n"
            "PlaintextObservation: Phase=F1903M Year=1903 MyUnits=F NTH MySC=LON,LVP,EDI\n"
            "Thought: 1) Threat low. 2) Opportunity: expand. 3) Probe with move.\n"
            "Action: {{\"abstract_action\":\"MOVE\", \"daide\":\"(MTO ...)\"}}\n"
        )

        user_template = (
            f"{fewshot}\n"
            f"{observation}\n"
            "Now follow the exact format:\n"
            "Thought: <your concise tactical reasoning, max 5 lines>\n"
            "Action: <a JSON object with keys abstract_action and daide>\n"
            f"abstract_action must be one of: {ACTIONS}.\n"
        )

        # 用 JSON parser 解析：让模型输出整个 response 为 JSON 时不方便保留 Thought，因此这里让 JSON 只出现在 Action 行。
        # LLMAgent(json_format=True) 会强制追加 JSON 输出要求，因此我们直接让最终输出 JSON：{thought, abstract_action, daide}
        system_template = (
            "You are a Diplomacy baseline agent implementing ReAct (Reason+Act). "
            "You are tactical, short-horizon, and only use the latest observation."
        )
        result = self._llm.get_response(
            user_template=(
                "Return a JSON object with keys: thought, abstract_action, daide. "
                "Do not include extra keys.\n" + user_template
            ),
            new_system_template=system_template,
            input_param_dict={},
            is_first_call=False,
            flag_debug_print=False,
        )
        if isinstance(result, Exception):
            raise result

        thought = result.get("thought", "") if isinstance(result, dict) else ""
        action = _coerce_action(result.get("abstract_action") if isinstance(result, dict) else None)
        _ = thought  # 当前版本不落盘 thought，避免污染日志

        self.last_order = action
        self.memory.last_actions.append(action)
        return action


class ReflexionBaselineAgent(_LLMBaselineBase):
    """Reflexion：Actor 产生动作；若触发器命中则 Reflector 写一句教训进长时记忆。"""

    def __init__(self, name: str, llm_model: Optional[str] = None):
        super().__init__(name=name, baseline_type="Reflexion", llm_model=llm_model)

    def _propose_orders_sync(self, game_view: Dict[str, Any]) -> str:
        state = game_view.get("state") or {}
        last_orders = game_view.get("last_orders") or {}
        last_feedback = game_view.get("last_feedback") or {}
        tension = float(game_view.get("tension", 0.5))

        lessons = (self.memory.reflections or [])[-3:]
        lessons_text = "\n".join([f"- {x}" for x in lessons]) if lessons else "- (none)"
        lessons_text = _escape_curly(lessons_text)

        system_template = (
            "You are a Diplomacy baseline agent implementing Reflexion (verbal RL).\n"
            "You act, then learn from mistakes via short self-reflections.\n"
            "[Lessons from the past]\n"
            f"{lessons_text}"
        )

        user_template = (
            f"Current situation for {self.name}:\n"
            f"{_safe_compact_state(state, self.name)}\n"
            f"Tension={tension:.2f}\n"
            f"LastTurnOrders={_dump(last_orders)}\n"
            f"LastTurnFeedback={_dump(last_feedback)}\n"
            f"Choose one abstract_action from: {ACTIONS}.\n"
            "Return JSON: {{\"abstract_action\":..., \"rationale\":...}}"
        )

        result = self._llm.get_response(
            user_template=user_template,
            new_system_template=system_template,
            input_param_dict={},
            is_first_call=False,
            flag_debug_print=False,
        )
        if isinstance(result, Exception):
            raise result

        action = _coerce_action(result.get("abstract_action") if isinstance(result, dict) else None)
        self.last_order = action
        self.memory.last_actions.append(action)
        return action

    def post_round_update(self, game_view: Dict[str, Any], feedback_label: str, sc_delta: int) -> None:
        try:
            set_llm_log_context({
                "scenario": "diplomacy",
                "role": "baseline",
                "baseline_type": self.baseline_type,
                "country": self.name,
                "round": game_view.get("round"),
                "phase": (game_view.get("state") or {}).get("name"),
                "stage": "post_round_update",
                "feedback": feedback_label,
                "sc_delta": sc_delta,
            })
        except Exception:
            pass
        # 触发器：SC 减少 或 进攻/推进未带来收益（近似：攻击类/移动但反馈非 gain）
        last_action = self.last_order
        aggressive = last_action in {"ATTACK", "SUPPORT_ATTACK", "MOVE"}
        failed_attack = aggressive and feedback_label != "gain"
        trigger = (sc_delta < 0) or failed_attack
        if not trigger:
            return

        state = game_view.get("state") or {}
        last_orders = game_view.get("last_orders") or {}

        system_template = "You are a Reflector. Write one actionable lesson in one sentence."
        user_template = (
            f"You are {self.name}. Your last abstract_action was {last_action}.\n"
            f"Outcome feedback_label={feedback_label}, sc_delta={sc_delta}.\n"
            f"State summary: {_safe_compact_state(state, self.name)}\n"
            f"Other powers last orders: {_dump(last_orders)}\n"
            "Analyze why it failed (force, deception, timing) and output JSON: {{\"reflection\": \"...\"}}."
        )

        result = self._llm.get_response(
            user_template=user_template,
            new_system_template=system_template,
            input_param_dict={},
            is_first_call=False,
            flag_debug_print=False,
        )
        if isinstance(result, Exception):
            raise result

        reflection = ""
        if isinstance(result, dict):
            reflection = str(result.get("reflection", "")).strip()
        if reflection:
            self.memory.reflections.append(reflection)


class EvoBaselineAgent(_LLMBaselineBase):
    """EvoAgent：持续世界模型（叙事摘要）+ Updater + Planner。"""

    def __init__(self, name: str, llm_model: Optional[str] = None):
        super().__init__(name=name, baseline_type="EvoAgent", llm_model=llm_model)
        self.memory.world_model_state = (
            f"It is 1901. I am {name}. My goal is to maximize my national interest and survive. "
            "I will maintain a concise global situation summary and update it each round."
        )

    def _propose_orders_sync(self, game_view: Dict[str, Any]) -> str:
        state = game_view.get("state") or {}
        tension = float(game_view.get("tension", 0.5))

        system_template = (
            "You are a Diplomacy baseline agent with a continual world model (EvoAgent).\n"
            "You do NOT read raw logs; you rely on the world model summary."
        )

        user_template = (
            "[World Model State]\n"
            f"{_escape_curly(self.memory.world_model_state)}\n\n"
            "[Local Snapshot]\n"
            f"{_safe_compact_state(state, self.name)}\n"
            f"Tension={tension:.2f}\n\n"
            "Plan this turn and output JSON: {{\"abstract_action\":..., \"plan\":...}}. "
            f"abstract_action must be one of: {ACTIONS}."
        )

        result = self._llm.get_response(
            user_template=user_template,
            new_system_template=system_template,
            input_param_dict={},
            is_first_call=False,
            flag_debug_print=False,
        )
        if isinstance(result, Exception):
            raise result

        action = _coerce_action(result.get("abstract_action") if isinstance(result, dict) else None)
        self.last_order = action
        self.memory.last_actions.append(action)
        return action

    def post_round_update(self, game_view: Dict[str, Any], feedback_label: str, sc_delta: int) -> None:
        try:
            set_llm_log_context({
                "scenario": "diplomacy",
                "role": "baseline",
                "baseline_type": self.baseline_type,
                "country": self.name,
                "round": game_view.get("round"),
                "phase": (game_view.get("state") or {}).get("name"),
                "stage": "post_round_update",
                "feedback": feedback_label,
                "sc_delta": sc_delta,
            })
        except Exception:
            pass
        state = game_view.get("state") or {}
        last_orders = game_view.get("last_orders") or {}
        last_feedback = game_view.get("last_feedback") or {}

        new_events = (
            f"This round feedback for me: {feedback_label} (sc_delta={sc_delta}). "
            f"Orders: {_dump(last_orders)}. Feedback: {_dump(last_feedback)}.\n"
            f"Local snapshot: {_safe_compact_state(state, self.name)}"
        )

        system_template = "You are a world-model updater. Keep the state concise, factual, and updated."
        user_template = (
            "Here is the old world model state:\n"
            f"[Old State]\n{_escape_curly(self.memory.world_model_state)}\n\n"
            "Here are the new events:\n"
            f"[New Events]\n{new_events}\n\n"
            "Update the world model: remove obsolete info, add new strategic situation. "
            "Return JSON: {{\"world_model_state\": \"...\"}}."
        )

        result = self._llm.get_response(
            user_template=user_template,
            new_system_template=system_template,
            input_param_dict={},
            is_first_call=False,
            flag_debug_print=False,
        )
        if isinstance(result, Exception):
            raise result

        if isinstance(result, dict):
            updated = str(result.get("world_model_state", "")).strip()
            if updated:
                self.memory.world_model_state = updated


class HypotheticalMindsBaselineAgent(_LLMBaselineBase):
    """Hypothetical Minds: Theory-of-Mind 驱动的基线。

    核心：为每个对手维护心智模型 (Mental Model)，对候选动作进行
    对手响应模拟 (Mental Simulation)，选择预期效用最高的动作。
    """

    def __init__(self, name: str, llm_model: Optional[str] = None):
        super().__init__(name=name, baseline_type="HypotheticalMinds", llm_model=llm_model)
        # ToM: 每个对手的行动历史和推断
        self._tom: Dict[str, Dict[str, Any]] = {}  # country -> {history, goal, tendency}

    def _ensure_tom(self, country: str) -> Dict[str, Any]:
        if country not in self._tom:
            self._tom[country] = {
                "history": [],
                "goal": "unknown",
                "tendency": "unknown",
            }
        return self._tom[country]

    def _propose_orders_sync(self, game_view: Dict[str, Any]) -> str:
        state = game_view.get("state") or {}
        last_orders = game_view.get("last_orders") or {}
        last_feedback = game_view.get("last_feedback") or {}
        tension = float(game_view.get("tension", 0.5))

        # 更新 ToM 模型
        for country, action in last_orders.items():
            if country != self.name and isinstance(action, str):
                tom = self._ensure_tom(country)
                tom["history"].append(action)
                tom["history"] = tom["history"][-8:]  # 窗口

        # 构建 ToM 摘要
        tom_lines = []
        for country, model in self._tom.items():
            recent = model["history"][-4:]
            tom_lines.append(
                f"{country}: recent={recent}, "
                f"goal={model['goal']}, tendency={model['tendency']}"
            )
        tom_text = _escape_curly("\n".join(tom_lines)) if tom_lines else "(no ToM data yet)"

        system_template = (
            "You are a Diplomacy baseline agent implementing Hypothetical Minds. "
            "You build Theory-of-Mind models of opponents and mentally simulate "
            "their responses to your candidate actions before choosing."
        )

        user_template = (
            f"You are {self.name}.\n"
            f"{_safe_compact_state(state, self.name)}\n"
            f"Tension={tension:.2f}\n"
            f"LastTurnOrders={_dump(last_orders)}\n"
            f"LastTurnFeedback={_dump(last_feedback)}\n"
            f"\n[Theory of Mind Models]\n{tom_text}\n\n"
            "Step 1: Update your mental models of opponents (inferred goals and tendencies).\n"
            "Step 2: Consider 2-3 candidate actions and mentally simulate opponent responses.\n"
            "Step 3: Pick the action with the best expected outcome.\n\n"
            f"abstract_action must be one of: {ACTIONS}.\n"
            "Return JSON:\n"
            "{{\"tom_updates\": [{{\"agent\": \"...\", \"inferred_goal\": \"...\", "
            "\"tendency\": \"...\"}}], "
            "\"candidate_evaluations\": [{{\"action\": \"...\", \"predicted_responses\": "
            "{{\"country\": \"action\"}}, \"expected_utility\": 0.0}}], "
            "\"abstract_action\": \"...\", \"rationale\": \"...\"}}"
        )

        result = self._llm.get_response(
            user_template=user_template,
            new_system_template=system_template,
            input_param_dict={},
            is_first_call=False,
            flag_debug_print=False,
        )
        if isinstance(result, Exception):
            raise result

        # 应用 ToM 更新
        if isinstance(result, dict):
            for update in result.get("tom_updates", []):
                agent_id = update.get("agent", "")
                if agent_id:
                    tom = self._ensure_tom(agent_id)
                    tom["goal"] = update.get("inferred_goal", tom["goal"])
                    tom["tendency"] = update.get("tendency", tom["tendency"])

        action = _coerce_action(result.get("abstract_action") if isinstance(result, dict) else None)
        self.last_order = action
        self.memory.last_actions.append(action)
        return action

    def post_round_update(self, game_view: Dict[str, Any], feedback_label: str, sc_delta: int) -> None:
        try:
            set_llm_log_context({
                "scenario": "diplomacy",
                "role": "baseline",
                "baseline_type": self.baseline_type,
                "country": self.name,
                "round": game_view.get("round"),
                "phase": (game_view.get("state") or {}).get("name"),
                "stage": "post_round_update",
                "feedback": feedback_label,
                "sc_delta": sc_delta,
            })
        except Exception:
            pass
        # 仅在 SC 变化时进行 ToM 反思
        if sc_delta == 0:
            return

        last_orders = game_view.get("last_orders") or {}
        for country, action in last_orders.items():
            if country != self.name and isinstance(action, str):
                tom = self._ensure_tom(country)
                tom["history"].append(action)
                tom["history"] = tom["history"][-8:]
