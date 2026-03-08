"""SociologyAgent – Bridge between SAGE Agent and SocialInvolution Rider
=======================================================================

Provides the ``RiderLLMAgent`` mixin class that the existing
``simulation.SocialInvolution.entity.rider.Rider`` imports.

The mixin instantiates two SAGE engines internally:
    * ``_sage_time``  – for daily work-hour decisions  (decide_time)
    * ``_sage_order`` – for order-selection decisions   (take_order)

Both engines use the BFS Expectimax architecture with a
``SocialInvolutionAdapter`` tailored to the delivery-platform scenario.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Tuple


class RiderLLMAgent:
    """Mixin providing ``decide_time()`` and ``take_order()`` for riders
    in the SocialInvolution simulation.

    Usage in ``Rider.__init__``::

        RiderLLMAgent.__init__(self, role_param_dict)

    The actual SAGE engines are lazily initialised on first call so that
    the rider's ``id`` / ``rider_num`` are available.
    """

    def __init__(self, role_param_dict: Optional[Dict[str, Any]] = None):
        self._role_params: Dict[str, Any] = role_param_dict or {}
        self._sage_initialized: bool = False

        # Will be populated by _ensure_sage_init
        self._sage_time: Any  = None
        self._sage_order: Any = None

        # Track last feedback for world-model learning between calls
        self._prev_money: float = 0.0
        self._prev_rank: int = 0

    # ------------------------------------------------------------------
    #  Lazy initialisation
    # ------------------------------------------------------------------
    def _ensure_sage_init(self) -> None:
        """Create SAGE engines once the rider entity attributes are ready."""
        if self._sage_initialized:
            return

        # Deferred import to avoid circular dependency at module load time
        from agents.sage_agent import SAGEAgent, SocialInvolutionAdapter

        rider_id: int   = getattr(self, "id", 0)
        rider_num: int  = getattr(self, "rider_num", 100)
        llm_model: str  = (self._role_params.get("llm_model", "gemma3:27b-q8")
                           if self._role_params else "gemma3:27b-q8")

        # Limit peer count for efficiency (LLM prompt size / entropy calc)
        other_ids = [i for i in range(rider_num) if i != rider_id][:10]
        peer_names = [f"rider_{rid}" for rid in other_ids]

        # ---- Work-time SAGE engine ------------------------------------
        self._sage_time = SAGEAgent(
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
        self._sage_time.adapter = SocialInvolutionAdapter(
            rider_id, other_ids, "time",
        )

        # ---- Order-selection SAGE engine ------------------------------
        self._sage_order = SAGEAgent(
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
        self._sage_order.adapter = SocialInvolutionAdapter(
            rider_id, other_ids, "order",
        )

        self._sage_initialized = True

    # ------------------------------------------------------------------
    #  decide_time  –  called by Rider.decide_work_time() each new day
    # ------------------------------------------------------------------
    def decide_time(
        self, runner_step: int, info: Dict[str, Any],
    ) -> Tuple[int, int]:
        """Return ``(go_work_time, get_off_work_time)`` using SAGE.

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
        self._ensure_sage_init()

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
        if self._sage_time.current_step > 0:
            self._sage_time._update_world_model(
                action=self._sage_time.meta_state.get("_last_action", "NORMAL"),
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

        result = self._sage_time.run_cycle(context)
        action = result["action"]
        self._sage_time.meta_state["_last_action"] = action

        from agents.sage_agent import SocialInvolutionAdapter
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
        self._ensure_sage_init()

        order_list = info.get("order_list", [])
        if not order_list:
            return []

        now_loc  = info.get("now_location", (0, 0))
        max_take = int(info.get("accept_count", 1))

        # Build context for SAGE
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

        result = self._sage_order.run_cycle(context)
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
