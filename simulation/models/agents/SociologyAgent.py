"""SociologyAgent – Bridge between Agent frameworks and SocialInvolution Rider
=============================================================================

Provides mixin classes that ``simulation.SocialInvolution.entity.rider.Rider``
can inherit from.  Each mixin instantiates a different decision engine:

    * ``RiderLLMAgent``            – RISE Agent (OODA + BFS Expectimax)
    * ``RiderReActAgent``          – ReAct baseline
    * ``RiderEvoAgent``            – EvoAgent baseline
    * ``RiderHypotheticalMinds``   – Hypothetical Minds baseline (ToM)
    * ``RiderGreedyHeuristic``     – Greedy heuristic baseline (no LLM)
"""

from __future__ import annotations

import json
import math
import random
from typing import Any, Dict, List, Optional, Tuple


class RiderLLMAgent:
    """Mixin providing ``decide_time()`` and ``take_order()`` for riders
    in the SocialInvolution simulation.

    Usage in ``Rider.__init__``::

        RiderLLMAgent.__init__(self, role_param_dict)

    The actual RISE engines are lazily initialised on first call so that
    the rider's ``id`` / ``rider_num`` are available.
    """

    def __init__(self, role_param_dict: Optional[Dict[str, Any]] = None):
        self._role_params: Dict[str, Any] = role_param_dict or {}
        self._rise_initialized: bool = False

        # Will be populated by _ensure_rise_init
        self._rise_time: Any  = None
        self._rise_order: Any = None

        # Track last feedback for world-model learning between calls
        self._prev_money: float = 0.0
        self._prev_rank: int = 0

    # ------------------------------------------------------------------
    #  Lazy initialisation
    # ------------------------------------------------------------------
    def _ensure_rise_init(self) -> None:
        """Create RISE engines once the rider entity attributes are ready."""
        if self._rise_initialized:
            return

        # Deferred import to avoid circular dependency at module load time
        from agents.rise_agent import RISEAgent, SocialInvolutionAdapter

        rider_id: int   = getattr(self, "id", 0)
        rider_num: int  = getattr(self, "rider_num", 100)
        llm_model: str  = (self._role_params.get("llm_model", "gemma3:27b-q8")
                           if self._role_params else "gemma3:27b-q8")

        # Limit peer count for efficiency (LLM prompt size / entropy calc)
        other_ids = [i for i in range(rider_num) if i != rider_id][:10]
        peer_names = [f"rider_{rid}" for rid in other_ids]

        # ---- Work-time RISE engine ------------------------------------
        self._rise_time = RISEAgent(
            country_name=f"rider_{rider_id}",
            other_countries=peer_names,
            meta_goal=(
                "Maximise daily income while maintaining sustainable work "
                "hours. Avoid exhaustion and adapt schedule to competition."
            ),
            llm_model=llm_model,
            scenario="social_involution",
            search_depth=1,      # shallow tree for frequent decisions
            top_k=2,
            prob_threshold=0.05,
            risk_threshold=0.75,
        )
        self._rise_time.adapter = SocialInvolutionAdapter(
            rider_id, other_ids, "time",
        )

        # ---- Order-selection RISE engine ------------------------------
        self._rise_order = RISEAgent(
            country_name=f"rider_{rider_id}",
            other_countries=peer_names,
            meta_goal=(
                "Maximise income per delivery by selecting profitable and "
                "nearby orders.  Skip unprofitable offers."
            ),
            llm_model=llm_model,
            scenario="social_involution",
            search_depth=1,
            top_k=2,
            prob_threshold=0.05,
            risk_threshold=0.75,
        )
        self._rise_order.adapter = SocialInvolutionAdapter(
            rider_id, other_ids, "order",
        )

        self._rise_initialized = True

    # ------------------------------------------------------------------
    #  decide_time  –  called by Rider.decide_work_time() each new day
    # ------------------------------------------------------------------
    def decide_time(
        self, runner_step: int, info: Dict[str, Any],
    ) -> Tuple[int, int]:
        """Return ``(go_work_time, get_off_work_time)`` using RISE.

        ``info`` typically contains::

            {
                "before_go_work_time": 8,
                "before_get_off_work_time": 18,
                "rider_num": 100,
                "order_rank": 5,
                "dis_rank": 12,
                "money_rank": 3,
            }
        """
        self._ensure_rise_init()

        # --- build feedback from delta between calls ---
        current_money = float(getattr(self, "money", 0))
        current_rank  = int(info.get("money_rank", 0))
        if current_money > self._prev_money:
            fb = "income_up"
        elif current_money < self._prev_money:
            fb = "income_down"
        else:
            fb = "income_stable"

        # Learn from the previous cycle
        if self._rise_time.current_step > 0:
            self._rise_time._update_world_model(
                action=self._rise_time.meta_state.get("_last_action", "NORMAL"),
                feedback=fb,
                peer_reactions={},   # no direct peer observation for schedule
            )
        self._prev_money = current_money
        self._prev_rank  = current_rank

        # --- build round context ---
        context: Dict[str, Any] = {
            "round":    runner_step,
            "phase":    "work_schedule",
            "tension":  0.5,
            "rank":     info.get("order_rank", 0),
            "money_rank": info.get("money_rank", 0),
            "dis_rank":   info.get("dis_rank", 0),
            "before_go":  info.get("before_go_work_time", 8),
            "before_off": info.get("before_get_off_work_time", 18),
            "rider_num":  info.get("rider_num", 100),
        }

        result = self._rise_time.run_cycle(context)
        action = result["action"]
        self._rise_time.meta_state["_last_action"] = action

        from agents.rise_agent import SocialInvolutionAdapter
        time_map = SocialInvolutionAdapter.TIME_MAP
        go, off = time_map.get(action, (8, 18))
        return go, off

    # ------------------------------------------------------------------
    #  take_order  –  called by Rider.choose_order() each eligible step
    # ------------------------------------------------------------------
    def take_order(
        self, runner_step: int, info: Dict[str, Any],
    ) -> List[int]:
        """Return list of ``order_id``s to accept.

        ``info`` typically contains::

            {
                "order_list": [(id, {order_info}), ...] or dict,
                "now_location": (x, y),
                "accept_count": int,
            }
        """
        self._ensure_rise_init()

        order_list = info.get("order_list", [])
        if not order_list:
            return []

        now_loc  = info.get("now_location", (0, 0))
        max_take = int(info.get("accept_count", 1))

        # Build context for RISE
        # Compact order descriptions for the LLM
        order_descs: List[Dict[str, Any]] = []
        raw_items = (list(order_list.items())
                     if isinstance(order_list, dict) else list(order_list))
        for item in raw_items[:10]:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                oid  = item[0]
                data = item[1] if isinstance(item[1], dict) else {}
            elif isinstance(item, dict):
                oid  = item.get("order_id", 0)
                data = item
            else:
                continue
            order_descs.append({
                "order_id": oid,
                "money":    data.get("money", 0),
                "pickup":   data.get("pickup_location", (0, 0)),
                "delivery": data.get("delivery_location", (0, 0)),
            })

        context: Dict[str, Any] = {
            "round":      runner_step,
            "phase":      "order_selection",
            "tension":    0.5,
            "location":   now_loc,
            "max_take":   max_take,
            "orders":     order_descs,
        }

        result = self._rise_order.run_cycle(context)
        action = result["action"]

        if action == "SKIP":
            return []

        return self._resolve_orders(action, raw_items, now_loc, max_take)

    # ------------------------------------------------------------------
    #  Order resolution  –  map high-level strategy → concrete order IDs
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_orders(
        strategy: str,
        raw_items: list,
        location: Tuple,
        max_take: int,
    ) -> List[int]:
        """Deterministic heuristic to pick specific orders given strategy."""

        def _dist(a: Tuple, b: Tuple) -> float:
            try:
                return math.sqrt(
                    (float(a[0]) - float(b[0])) ** 2
                    + (float(a[1]) - float(b[1])) ** 2
                )
            except Exception:
                return 999.0

        def _extract(item) -> Tuple[int, Dict]:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                oid  = item[0] if isinstance(item[0], int) else (
                    item[1].get("order_id", 0) if isinstance(item[1], dict) else 0
                )
                data = item[1] if isinstance(item[1], dict) else {}
                return int(oid), data
            return 0, {}

        parsed = [_extract(it) for it in raw_items]
        parsed = [(oid, d) for oid, d in parsed if oid > 0 or d]

        if strategy == "TAKE_HIGHEST_VALUE":
            parsed.sort(key=lambda x: -float(x[1].get("money", 0)))
        elif strategy == "TAKE_NEAREST":
            parsed.sort(key=lambda x: _dist(
                location, x[1].get("pickup_location", (0, 0))
            ))
        elif strategy == "TAKE_BALANCED":
            # Score = money / (1 + distance)
            def _score(item):
                d = item[1] if isinstance(item[1], dict) else {}
                money = float(d.get("money", 1))
                dist  = _dist(location, d.get("pickup_location", (0, 0)))
                return money / (1.0 + dist)
            parsed.sort(key=lambda x: -_score(x))
        elif strategy == "TAKE_VOLUME":
            pass  # keep original order — take as many as possible
        else:
            pass

        selected: List[int] = []
        for oid, _ in parsed[:max_take]:
            if isinstance(oid, int) and oid not in selected:
                selected.append(oid)
        return selected


# ======================================================================
#  Time action mapping (shared across all rider mixins)
# ======================================================================
_TIME_ACTIONS = ["EARLY_LONG", "EARLY_NORMAL", "NORMAL",
                 "NORMAL_LATE", "LATE_LONG", "SHORT"]
_TIME_MAP: Dict[str, Tuple[int, int]] = {
    "EARLY_LONG":   (6, 20),
    "EARLY_NORMAL": (6, 18),
    "NORMAL":       (8, 18),
    "NORMAL_LATE":  (8, 20),
    "LATE_LONG":    (10, 22),
    "SHORT":        (10, 16),
}
_ORDER_ACTIONS = ["TAKE_HIGHEST_VALUE", "TAKE_NEAREST",
                  "TAKE_BALANCED", "TAKE_VOLUME", "SKIP"]


def _parse_order_list(info: Dict[str, Any]) -> Tuple[list, Tuple, int]:
    """Extract order_list, location, max_take from info dict."""
    order_list = info.get("order_list", [])
    now_loc = info.get("now_location", (0, 0))
    max_take = int(info.get("accept_count", 1))
    raw_items = (list(order_list.items())
                 if isinstance(order_list, dict) else list(order_list))
    return raw_items, now_loc, max_take


# ======================================================================
#  ReAct Rider Mixin
# ======================================================================

class RiderReActAgent:
    """ReAct baseline for SocialInvolution riders.

    Short-horizon reasoning: observe current state, think, act.
    No world model or opponent modeling.
    """

    def __init__(self, role_param_dict: Optional[Dict[str, Any]] = None):
        self._role_params: Dict[str, Any] = role_param_dict or {}
        self._react_llm: Any = None
        self._react_initialized: bool = False
        self._prev_money_react: float = 0.0

    def _ensure_react_init(self) -> None:
        if self._react_initialized:
            return
        from simulation.models.agents.LLMAgent import LLMAgent
        llm_model = (self._role_params.get("llm_model", "gemma3:27b-q8")
                     if self._role_params else "gemma3:27b-q8")
        rider_id = getattr(self, "id", 0)
        self._react_llm = LLMAgent(
            agent_name=f"ReAct_rider_{rider_id}",
            has_chat_history=False,
            llm_model=llm_model,
            online_track=False,
            json_format=True,
        )
        self._react_initialized = True

    def decide_time(self, runner_step: int, info: Dict[str, Any]) -> Tuple[int, int]:
        self._ensure_react_init()
        prompt = (
            f"You are a delivery rider deciding work hours.\n"
            f"Current schedule: {info.get('before_go_work_time',8)}:00 to "
            f"{info.get('before_get_off_work_time',18)}:00\n"
            f"Your rank: orders={info.get('order_rank',0)}, "
            f"money={info.get('money_rank',0)}, distance={info.get('dis_rank',0)}\n"
            f"Total riders: {info.get('rider_num',100)}\n\n"
            f"Choose a work schedule. Options: {list(_TIME_MAP.keys())}\n"
            "Return JSON: {{\"schedule\": \"OPTION_NAME\", \"reasoning\": \"...\"}}"
        )
        try:
            resp = self._react_llm.get_response(prompt, flag_debug_print=False)
            if isinstance(resp, dict):
                action = resp.get("schedule", "NORMAL")
                if action in _TIME_MAP:
                    return _TIME_MAP[action]
        except Exception:
            pass
        return 8, 18

    def take_order(self, runner_step: int, info: Dict[str, Any]) -> List[int]:
        self._ensure_react_init()
        raw_items, now_loc, max_take = _parse_order_list(info)
        if not raw_items:
            return []

        prompt = (
            f"You are a delivery rider at location {now_loc}. "
            f"Choose an order selection strategy.\n"
            f"Options: {_ORDER_ACTIONS}\n"
            "Return JSON: {{\"strategy\": \"OPTION_NAME\"}}"
        )
        try:
            resp = self._react_llm.get_response(prompt, flag_debug_print=False)
            if isinstance(resp, dict):
                strategy = resp.get("strategy", "TAKE_BALANCED")
                if strategy in _ORDER_ACTIONS:
                    return RiderLLMAgent._resolve_orders(strategy, raw_items, now_loc, max_take)
        except Exception:
            pass
        return RiderLLMAgent._resolve_orders("TAKE_BALANCED", raw_items, now_loc, max_take)


# ======================================================================
#  EvoAgent Rider Mixin
# ======================================================================

class RiderEvoAgent:
    """EvoAgent baseline for SocialInvolution riders.

    Maintains a population of strategy configurations and evolves
    them based on performance feedback.
    """

    def __init__(self, role_param_dict: Optional[Dict[str, Any]] = None):
        self._role_params: Dict[str, Any] = role_param_dict or {}
        self._evo_llm: Any = None
        self._evo_initialized: bool = False
        # Strategy population: list of {schedule, order_strategy, fitness}
        self._strategies: List[Dict[str, Any]] = []
        self._current_strategy: Optional[Dict[str, Any]] = None
        self._evo_generation: int = 0
        self._prev_money_evo: float = 0.0

    def _ensure_evo_init(self) -> None:
        if self._evo_initialized:
            return
        from simulation.models.agents.LLMAgent import LLMAgent
        llm_model = (self._role_params.get("llm_model", "gemma3:27b-q8")
                     if self._role_params else "gemma3:27b-q8")
        rider_id = getattr(self, "id", 0)
        self._evo_llm = LLMAgent(
            agent_name=f"Evo_rider_{rider_id}",
            has_chat_history=False,
            llm_model=llm_model,
            online_track=False,
            json_format=True,
        )
        # Initialize population
        self._strategies = [
            {"schedule": "EARLY_LONG", "order_strategy": "TAKE_VOLUME", "fitness": 0.5},
            {"schedule": "NORMAL", "order_strategy": "TAKE_BALANCED", "fitness": 0.5},
            {"schedule": "NORMAL_LATE", "order_strategy": "TAKE_HIGHEST_VALUE", "fitness": 0.5},
            {"schedule": "SHORT", "order_strategy": "TAKE_NEAREST", "fitness": 0.5},
            {"schedule": "EARLY_NORMAL", "order_strategy": "TAKE_BALANCED", "fitness": 0.5},
        ]
        self._evo_initialized = True

    def _select_strategy(self) -> Dict[str, Any]:
        total = sum(s["fitness"] for s in self._strategies)
        if total <= 0:
            return random.choice(self._strategies)
        r = random.random() * total
        cumul = 0.0
        for s in self._strategies:
            cumul += s["fitness"]
            if cumul >= r:
                return s
        return self._strategies[-1]

    def _evolve(self) -> None:
        self._evo_generation += 1
        self._strategies.sort(key=lambda s: s["fitness"], reverse=True)
        elite = self._strategies[:3]
        # Mutate top strategy
        if elite:
            mutant = dict(elite[0])
            mutant["schedule"] = random.choice(_TIME_ACTIONS)
            mutant["order_strategy"] = random.choice(_ORDER_ACTIONS[:-1])  # exclude SKIP
            mutant["fitness"] = elite[0]["fitness"] * 0.8
            self._strategies = elite + [mutant, dict(elite[-1])]

    def decide_time(self, runner_step: int, info: Dict[str, Any]) -> Tuple[int, int]:
        self._ensure_evo_init()
        # Update fitness based on income change
        cur_money = float(getattr(self, "money", 0))
        if self._current_strategy and cur_money != self._prev_money_evo:
            delta = cur_money - self._prev_money_evo
            reward = min(1.0, max(0.0, 0.5 + delta / 100.0))
            self._current_strategy["fitness"] = (
                0.3 * reward + 0.7 * self._current_strategy["fitness"]
            )
        self._prev_money_evo = cur_money

        # Evolve every 3 days
        if runner_step > 0 and self._evo_generation < runner_step // getattr(self, "one_day", 480) // 3:
            self._evolve()

        self._current_strategy = self._select_strategy()
        sched = self._current_strategy.get("schedule", "NORMAL")
        return _TIME_MAP.get(sched, (8, 18))

    def take_order(self, runner_step: int, info: Dict[str, Any]) -> List[int]:
        self._ensure_evo_init()
        raw_items, now_loc, max_take = _parse_order_list(info)
        if not raw_items:
            return []
        strategy = (self._current_strategy or {}).get("order_strategy", "TAKE_BALANCED")
        if strategy == "SKIP":
            return []
        return RiderLLMAgent._resolve_orders(strategy, raw_items, now_loc, max_take)


# ======================================================================
#  Hypothetical Minds Rider Mixin
# ======================================================================

class RiderHypotheticalMinds:
    """Hypothetical Minds baseline for SocialInvolution riders.

    Uses Theory of Mind to model competitor riders and predict their
    impact on order availability and earnings.
    """

    def __init__(self, role_param_dict: Optional[Dict[str, Any]] = None):
        self._role_params: Dict[str, Any] = role_param_dict or {}
        self._hm_llm: Any = None
        self._hm_initialized: bool = False
        # ToM: simplified model of competitor behavior
        self._competitor_model: Dict[str, Any] = {
            "avg_start": 8, "avg_end": 18,
            "popular_strategy": "TAKE_BALANCED",
            "competition_level": "moderate",
        }

    def _ensure_hm_init(self) -> None:
        if self._hm_initialized:
            return
        from simulation.models.agents.LLMAgent import LLMAgent
        llm_model = (self._role_params.get("llm_model", "gemma3:27b-q8")
                     if self._role_params else "gemma3:27b-q8")
        rider_id = getattr(self, "id", 0)
        self._hm_llm = LLMAgent(
            agent_name=f"HM_rider_{rider_id}",
            has_chat_history=False,
            llm_model=llm_model,
            online_track=False,
            json_format=True,
        )
        self._hm_initialized = True

    def decide_time(self, runner_step: int, info: Dict[str, Any]) -> Tuple[int, int]:
        self._ensure_hm_init()
        comp = self._competitor_model
        prompt = (
            f"You are a delivery rider deciding work hours using Theory of Mind.\n"
            f"Current schedule: {info.get('before_go_work_time',8)}:00 - "
            f"{info.get('before_get_off_work_time',18)}:00\n"
            f"Your rank: orders={info.get('order_rank',0)}, "
            f"money={info.get('money_rank',0)}\n"
            f"Competitor model: avg_start={comp['avg_start']}, "
            f"avg_end={comp['avg_end']}, competition={comp['competition_level']}\n\n"
            f"Mentally simulate: if most competitors work {comp['avg_start']}-"
            f"{comp['avg_end']}, what schedule avoids peak competition?\n"
            f"Options: {list(_TIME_MAP.keys())}\n"
            "Return JSON: {{\"schedule\": \"OPTION\", \"predicted_competition\": \"low/medium/high\"}}"
        )
        try:
            resp = self._hm_llm.get_response(prompt, flag_debug_print=False)
            if isinstance(resp, dict):
                action = resp.get("schedule", "NORMAL")
                comp_level = resp.get("predicted_competition", "moderate")
                self._competitor_model["competition_level"] = comp_level
                if action in _TIME_MAP:
                    return _TIME_MAP[action]
        except Exception:
            pass
        return 8, 18

    def take_order(self, runner_step: int, info: Dict[str, Any]) -> List[int]:
        self._ensure_hm_init()
        raw_items, now_loc, max_take = _parse_order_list(info)
        if not raw_items:
            return []

        comp = self._competitor_model
        prompt = (
            f"You are a delivery rider at {now_loc}. "
            f"Competitors mostly use {comp['popular_strategy']} strategy. "
            f"Competition level: {comp['competition_level']}.\n"
            f"Mentally simulate competitor choices, then pick a counter-strategy.\n"
            f"Options: {_ORDER_ACTIONS}\n"
            "Return JSON: {{\"strategy\": \"OPTION\"}}"
        )
        try:
            resp = self._hm_llm.get_response(prompt, flag_debug_print=False)
            if isinstance(resp, dict):
                strategy = resp.get("strategy", "TAKE_BALANCED")
                if strategy in _ORDER_ACTIONS:
                    if strategy == "SKIP":
                        return []
                    return RiderLLMAgent._resolve_orders(strategy, raw_items, now_loc, max_take)
        except Exception:
            pass
        return RiderLLMAgent._resolve_orders("TAKE_BALANCED", raw_items, now_loc, max_take)


# ======================================================================
#  Greedy Heuristic Rider Mixin (no LLM)
# ======================================================================

class RiderGreedyHeuristic:
    """Greedy heuristic baseline — no LLM calls.

    Time decision: always works the longest hours.
    Order decision: always picks highest-value orders.
    """

    def __init__(self, role_param_dict: Optional[Dict[str, Any]] = None):
        pass

    def decide_time(self, runner_step: int, info: Dict[str, Any]) -> Tuple[int, int]:
        return 6, 20  # EARLY_LONG — maximize work time

    def take_order(self, runner_step: int, info: Dict[str, Any]) -> List[int]:
        raw_items, now_loc, max_take = _parse_order_list(info)
        if not raw_items:
            return []
        return RiderLLMAgent._resolve_orders("TAKE_HIGHEST_VALUE", raw_items, now_loc, max_take)
