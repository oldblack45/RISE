"""RISE Agent – From Hindsight to Foresight via Structured Cognitive Evolution
=============================================================================

A closed-loop agent decision-making framework for non-stationary social
simulations that decouples internal cognitive reasoning from the external
dynamic social environment.

Four-stage closed-loop pipeline (see paper §3):

Stage 1 – World Model Construction  (§3.2)
    * Maintain W_t = {(a, f, r, e)}^{t-1} from historical interactions
    * Uniform uninformative prior at t = 0 via Laplace smoothing

Stage 2 – Candidate Action Pruning  (§3.2 Eq.1)
    * A_cand = LLM_filter(A_raw, W_t, G_meta)
    * Compress action space by eliminating low-value actions

Stage 3 – Hypothetical Reasoning via BFS  (§3.3 Eq.2–6)
    * Layer-by-layer tree expansion with Top-K branching
    * Batched LLM risk assessment (circuit-breaker)
    * Batched LLM actor for next actions
    * Leaf utility evaluation and expectimax back-propagation

Stage 4 – Dynamic Belief Calibration  (§3.4 Eq.7–11)
    * Execute a*, observe (f_obs, r_obs)
    * Semantic summary: e_obs = LLM_sum(a*, f_obs, r_obs)
    * Frequency update via soft-matching (threshold τ)
    * Laplace-smoothed probability recalibration

Supports two scenarios via pluggable adapters:
    * Diplomacy  (backward-compatible with tournament.py)
    * SocialInvolution  (rider delivery platform)
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from simulation.models.agents.LLMAgent import LLMAgent

try:
    from simulation.models.cognitive.experiment_logger import ExperimentLogger  # type: ignore
except Exception:
    ExperimentLogger = Any  # type: ignore


# ============================================================================
#  §1  Core Data Structures
# ============================================================================

@dataclass
class InteractionTuple:
    """A single interaction record (a, f, r, e) in the World Model."""
    action: str
    feedback: str
    peer_reactions: Dict[str, str]   # peer_id → reaction
    experience: str                   # semantic summary e_obs
    step: int


@dataclass
class PredictionRecord:
    """Peer behaviour prediction (for RQ2 reporting)."""
    target: str
    predicted_action: str
    support_evidence: str
    confidence: float


@dataclass
class TreeNode:
    """Node in the BFS hypothetical reasoning tree (§3.3)."""
    node_id: int
    parent_id: Optional[int]
    root_action: str            # origin candidate action at depth 0
    depth: int
    action: str                 # current action at this node
    world_state: Dict[str, Any]
    branch_prob: float          # P_joint for this branch
    cumulative_prob: float      # product of branch probs root → here
    risk_score: float  = 0.0
    utility: float     = 0.0
    children: List[int] = field(default_factory=list)
    pruned: bool       = False


# ============================================================================
#  §2  World Model  W_t = {(a, f, r, e)}^{t-1}   (§3.2 & §3.4)
# ============================================================================

class ProbabilityTracker:
    """Frequency-based probability tracker with Laplace smoothing.

    Maintains:
      N_t(f|a)        – environment feedback frequencies
      C_t^{(i)}(r|a)  – per-peer reaction frequencies

    Updates via soft-matching (Eq.8–9) and Laplace-smoothed
    distribution estimation (Eq.10–11).  When no existing bin matches
    (similarity < τ), a new discrete state bin is dynamically
    instantiated to break cognitive inertia.
    """

    def __init__(self, similarity_threshold: float = 0.8):
        self.tau = similarity_threshold
        # N_t(f|a) → env_counts[action][feedback] = count
        self.env_counts: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        # C_t^{(i)}(r|a) → peer_counts[peer][action][reaction] = count
        self.peer_counts: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(float))
        )

    # ---- Similarity (approximation of Φ-based cosine) -----------------
    @staticmethod
    def _soft_match(s1: str, s2: str) -> float:
        """Approximate cos(Φ(s1), Φ(s2)) via word-level Jaccard similarity."""
        s1, s2 = s1.lower().strip(), s2.lower().strip()
        if s1 == s2:
            return 1.0
        w1, w2 = set(s1.split()), set(s2.split())
        if not w1 or not w2:
            return 0.0
        return len(w1 & w2) / len(w1 | w2)

    # ---- Frequency update for environment  Eq (8) ---------------------
    def update_env(self, action: str, feedback_obs: str) -> None:
        """N_{t+1}(f|a*) = N_t(f|a*) + I[sim(f, f_obs) >= τ]"""
        bucket = self.env_counts[action]
        matched = False
        for existing in list(bucket.keys()):
            if self._soft_match(existing, feedback_obs) >= self.tau:
                bucket[existing] += 1.0
                matched = True
        if not matched:
            bucket[feedback_obs] = 1.0   # new discrete state bin

    # ---- Frequency update for peers  Eq (9) ---------------------------
    def update_peer(self, peer: str, action: str, reaction_obs: str) -> None:
        """C_{t+1}^{(i)}(r|a*) = C_t^{(i)}(r|a*) + I[sim(r, r_obs) >= τ]"""
        bucket = self.peer_counts[peer][action]
        matched = False
        for existing in list(bucket.keys()):
            if self._soft_match(existing, reaction_obs) >= self.tau:
                bucket[existing] += 1.0
                matched = True
        if not matched:
            bucket[reaction_obs] = 1.0   # new discrete state bin

    # ---- Laplace-smoothed distributions  Eq (10) ----------------------
    def get_env_distribution(self, action: str,
                              feedback_space: List[str]) -> Dict[str, float]:
        """P_{t+1}(f|a) = (N(f|a)+1) / Σ_{f'}(N(f'|a)+1)"""
        bucket = dict(self.env_counts.get(action, {}))
        all_bins = sorted(set(feedback_space) | set(bucket.keys()) or {"unknown"})
        total = sum(bucket.get(f, 0.0) + 1.0 for f in all_bins)
        if total == 0:
            return {f: 1.0 / max(1, len(all_bins)) for f in all_bins}
        return {f: (bucket.get(f, 0.0) + 1.0) / total for f in all_bins}

    # ---- Laplace-smoothed distributions  Eq (11) ----------------------
    def get_peer_distribution(self, peer: str, action: str,
                               reaction_space: List[str]) -> Dict[str, float]:
        """P_{t+1}^{(i)}(r|a) = (C^{(i)}(r|a)+1) / Σ_{r'}(C^{(i)}(r'|a)+1)"""
        bucket = dict(self.peer_counts.get(peer, {}).get(action, {}))
        all_bins = sorted(set(reaction_space) | set(bucket.keys()) or {"unknown"})
        total = sum(bucket.get(r, 0.0) + 1.0 for r in all_bins)
        if total == 0:
            return {r: 1.0 / max(1, len(all_bins)) for r in all_bins}
        return {r: (bucket.get(r, 0.0) + 1.0) / total for r in all_bins}


class WorldModel:
    """Structured World Model W_t = {(a, f, r, e)}^{t-1}.

    At t=0, the uniform uninformative prior P_0(f,r|a) is naturally
    obtained via Laplace smoothing over empty frequency tables.
    """

    def __init__(self, tau: float = 0.8):
        self.history: List[InteractionTuple] = []
        self.prob: ProbabilityTracker = ProbabilityTracker(tau)

    def add_interaction(self, interaction: InteractionTuple) -> None:
        """Append tuple to W_t and update frequency tables (Eq.8–9)."""
        self.history.append(interaction)
        self.prob.update_env(interaction.action, interaction.feedback)
        for peer, reaction in interaction.peer_reactions.items():
            self.prob.update_peer(peer, interaction.action, reaction)

    def get_summary(self, max_entries: int = 8) -> List[Dict[str, Any]]:
        """Return recent interaction tuples for LLM context."""
        return [
            {
                "step": t.step,
                "action": t.action,
                "feedback": t.feedback,
                "peer_reactions": t.peer_reactions,
                "experience": t.experience,
            }
            for t in self.history[-max_entries:]
        ]


# ============================================================================
#  §3  Scenario Adapters
# ============================================================================

from abc import ABC, abstractmethod  # noqa: E402


class ScenarioAdapter(ABC):
    """Domain-agnostic interface – each scenario provides a concrete adapter."""

    @abstractmethod
    def get_raw_actions(self, context: Optional[Dict] = None) -> List[str]: ...

    @abstractmethod
    def get_reaction_space(self) -> List[str]: ...

    @abstractmethod
    def get_feedback_space(self) -> List[str]: ...

    @abstractmethod
    def get_peers(self) -> List[str]: ...

    @abstractmethod
    def compute_env_transition(
        self, ws: Dict, action: str, fb: str, pr: Dict[str, str],
    ) -> Dict: ...

    @abstractmethod
    def get_meta_goal_prompt(self) -> str: ...

    @abstractmethod
    def format_state_for_llm(self, ctx: Dict) -> str: ...


# ---------------------------------------------------------------------------
#  Diplomacy Adapter
# ---------------------------------------------------------------------------

class DiplomacyAdapter(ScenarioAdapter):
    ACTIONS  = ["HOLD", "MOVE", "ATTACK", "SUPPORT_ATTACK",
                "SUPPORT_DEFEND", "RETREAT"]
    FEEDBACK = ["gain", "loss", "stable"]

    def __init__(self, country_name: str, other_countries: List[str]):
        self.country_name    = country_name
        self.other_countries = other_countries

    def get_raw_actions(self, context=None):
        return list(self.ACTIONS)

    def get_reaction_space(self):
        return list(self.ACTIONS)

    def get_feedback_space(self):
        return list(self.FEEDBACK)

    def get_peers(self):
        return list(self.other_countries)

    def compute_env_transition(self, ws, action, fb, pr):
        nw = dict(ws)
        nw["last_action"]       = action
        nw["last_feedback"]     = fb
        nw["last_peer_actions"] = pr
        sc = int(nw.get("supply_centers", 0))
        if fb == "gain":
            nw["supply_centers"] = sc + 1
        elif fb == "loss":
            nw["supply_centers"] = max(0, sc - 1)
        return nw

    def get_meta_goal_prompt(self):
        return (
            f"You are {self.country_name} in a strategic Diplomacy "
            f"(no-press) game.  Meta-goal: Maximise supply-center control "
            f"and ensure national survival."
        )

    def format_state_for_llm(self, ctx):
        parts = [f"Round={ctx.get('round','?')} Phase={ctx.get('phase','?')}"]
        gs = ctx.get("game_state") or {}
        if isinstance(gs, dict):
            key = (ctx.get("_country") or "ENGLAND").upper()
            u = gs.get("units",   {}).get(key, [])
            c = gs.get("centers", {}).get(key, [])
            parts.append(f"MyUnits({len(u)})={u[:8]}")
            parts.append(f"MySC({len(c)})={c[:8]}")
        parts.append(f"Tension={ctx.get('tension', 0.5):.2f}")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
#  SocialInvolution Adapter
# ---------------------------------------------------------------------------

class SocialInvolutionAdapter(ScenarioAdapter):
    # Work-time discrete action space
    TIME_ACTIONS = ["EARLY_LONG", "EARLY_NORMAL", "NORMAL",
                    "NORMAL_LATE", "LATE_LONG", "SHORT"]
    TIME_MAP: Dict[str, Tuple[int, int]] = {
        "EARLY_LONG":   (6, 20),
        "EARLY_NORMAL": (6, 18),
        "NORMAL":       (8, 18),
        "NORMAL_LATE":  (8, 20),
        "LATE_LONG":    (10, 22),
        "SHORT":        (10, 16),
    }

    # Order-selection discrete action space
    ORDER_ACTIONS = ["TAKE_HIGHEST_VALUE", "TAKE_NEAREST",
                     "TAKE_BALANCED", "TAKE_VOLUME", "SKIP"]

    FEEDBACK = ["income_up", "income_stable", "income_down"]
    REACTION = ["compete_aggressive", "compete_moderate", "passive"]

    def __init__(self, rider_id: int, peer_ids: List[int],
                 decision_type: str = "time"):
        self.rider_id      = rider_id
        self.peer_ids      = [str(p) for p in peer_ids]
        self.decision_type = decision_type   # "time" | "order"

    def get_raw_actions(self, context=None):
        return list(self.TIME_ACTIONS if self.decision_type == "time"
                    else self.ORDER_ACTIONS)

    def get_reaction_space(self):
        return list(self.REACTION)

    def get_feedback_space(self):
        return list(self.FEEDBACK)

    def get_peers(self):
        return list(self.peer_ids)

    def compute_env_transition(self, ws, action, fb, pr):
        nw = dict(ws)
        nw["last_action"]    = action
        nw["last_feedback"]  = fb
        income = float(nw.get("income", 0))
        if fb == "income_up":
            nw["income"] = income + 10
        elif fb == "income_down":
            nw["income"] = max(0.0, income - 5)
        return nw

    def get_meta_goal_prompt(self):
        return (
            f"You are rider_{self.rider_id} on a food-delivery platform.  "
            f"Meta-goal: Maximise daily income while maintaining sustainable "
            f"working conditions."
        )

    def format_state_for_llm(self, ctx):
        safe: Dict[str, Any] = {}
        for k, v in (ctx or {}).items():
            try:
                json.dumps(v); safe[k] = v
            except (TypeError, ValueError):
                safe[k] = str(v)[:200]
        return json.dumps(safe, ensure_ascii=False, default=str)[:600]


# ============================================================================
#  §4  RISE Agent – Core Decision Engine
# ============================================================================

class RISEAgent(LLMAgent):
    """RISE Agent – BFS Expectimax decision engine.

    Implements Algorithm 1: Decision Loop via BFS-based Hypothetical
    Reasoning with four-stage closed-loop architecture.

    Backward-compatible with the Diplomacy tournament runner::

        observe() → orient() → decide() → act() → learn_from_interaction()

    Also exposes a generic ``run_cycle()`` entry point.
    """

    # ------------------------------------------------------------------
    #  Constructor
    # ------------------------------------------------------------------
    def __init__(
        self,
        country_name: str,
        other_countries: List[str],
        game_attributes: Optional[Dict[str, Any]] = None,
        experiment_logger: Optional[Any] = None,
        meta_goal: str = "Ensure survival and maximize national interest",
        adaptation_rate: float = 0.3,
        use_llm_reasoner: bool = False,
        llm_model: Optional[str] = None,
        # Ablation switches
        enable_profiling: bool = True,
        enable_prediction: bool = True,
        enable_risk_gate: bool = True,
        # Scenario selection
        scenario: str = "diplomacy",
        # BFS hyper-parameters
        search_depth: int = 2,
        top_k: int = 3,
        prob_threshold: float = 0.05,
        risk_threshold: float = 0.7,
    ):
        super().__init__(
            agent_name=f"RISE_{country_name}",
            has_chat_history=False,
            llm_model=llm_model or "gemma3:27b-q8",
            online_track=False,
            json_format=True,
        )
        self.country_name    = country_name
        self.other_countries = list(other_countries)
        self.game_attributes = game_attributes or {}
        self.experiment_logger = experiment_logger
        self.meta_goal       = meta_goal

        # Ablation switches
        self.enable_profiling  = bool(enable_profiling)
        self.enable_prediction = bool(enable_prediction)
        self.enable_risk_gate  = bool(enable_risk_gate)

        # BFS hyper-parameters (§3.3)
        self.search_depth   = search_depth
        self.top_k          = top_k
        self.prob_threshold = prob_threshold
        self.risk_threshold = risk_threshold

        # Scenario adapter --------------------------------------------------
        if scenario == "diplomacy":
            self.adapter: ScenarioAdapter = DiplomacyAdapter(
                country_name, other_countries
            )
        elif scenario == "social_involution":
            self.adapter = SocialInvolutionAdapter(0, [], "time")
        else:
            self.adapter = DiplomacyAdapter(country_name, other_countries)

        # World Model W_t (§3.2) -------------------------------------------
        self.world_model = WorldModel(tau=0.8)

        # Internal bookkeeping ---------------------------------------------
        self.current_step = 0
        self.current_strategy  = meta_goal
        self.last_concrete_orders: List[str] = []
        self.prediction_cache: Dict[int, List[PredictionRecord]] = {}

        # Backward-compat aliases
        self.available_actions = self.adapter.get_raw_actions()
        self.meta_state: Dict[str, Any] = {
            "round": 0, "phase": "Init", "tension": 0.0,
        }

    # ====================================================================
    #  Stage 2 – Candidate Action Pruning  (§3.2 Eq.1)
    # ====================================================================

    def _filter_candidate_actions(
        self, context: Dict[str, Any],
    ) -> Tuple[List[str], str]:
        """Eq (1): A_cand = LLM_filter(A_raw, W_t, G_meta)

        Guided by the global meta-goal G_meta and current world state W_t,
        the LLM acts as a heuristic filter to eliminate low-value actions.

        Returns (A_cand, strategy_text).
        """
        raw = self.adapter.get_raw_actions(context)
        summary = self.world_model.get_summary()
        state_desc = self.adapter.format_state_for_llm(
            {**context, "_country": self.country_name}
        )

        prompt = (
            "You are a strategic action filter.  Given the agent's world "
            "model history, current state, and meta-goal, select a pruned "
            "subset of promising candidate actions.  Eliminate actions that "
            "lack execution feasibility or deviate from the current context.\n\n"
            "Meta-Goal: {meta_goal}\n"
            "World Model (recent interactions):\n{world_summary}\n"
            "Current State:\n{state_desc}\n"
            "Available Raw Actions: {actions}\n\n"
            "Select only feasible, high-value actions.  Also provide a "
            "one-sentence strategic guideline for the current situation.\n\n"
            "Output JSON:\n"
            "{{\"candidate_actions\": [\"action1\", ...], "
            "\"strategy\": \"one-sentence strategic guideline\"}}"
        )

        try:
            resp = self.get_response(prompt, input_param_dict={
                "meta_goal": self.meta_goal,
                "world_summary": summary,
                "state_desc": state_desc,
                "actions": raw,
            }, flag_debug_print=False)
            if isinstance(resp, dict):
                cands = resp.get("candidate_actions", [])
                strategy = str(resp.get("strategy", ""))
                valid = [a for a in cands if a in raw]
                if valid:
                    return valid, strategy or self.meta_goal
        except Exception:
            pass

        return raw, self.meta_goal

    # ====================================================================
    #  Stage 3 – Hypothetical Reasoning via BFS  (§3.3)
    # ====================================================================

    def _get_top_k_outcomes(
        self, action: str,
    ) -> List[Tuple[str, Dict[str, str], float]]:
        """Top-K (feedback, peer_reactions, P_joint) via Eq (2).

        P_joint(f,r|a) = P_t(f|a) · Π_{i∈N} P_t^{(i)}(r^{(i)}|a)
        """
        fb_space = self.adapter.get_feedback_space()
        rx_space = self.adapter.get_reaction_space()
        peers    = self.adapter.get_peers()

        env_dist = self.world_model.prob.get_env_distribution(action, fb_space)

        outcomes: List[Tuple[str, Dict[str, str], float]] = []
        for f, p_f in env_dist.items():
            if p_f < self.prob_threshold:
                continue
            pr: Dict[str, str] = {}
            p_prod = 1.0
            for peer in peers:
                if self.enable_profiling:
                    pd = self.world_model.prob.get_peer_distribution(
                        peer, action, rx_space,
                    )
                else:
                    pd = {r: 1.0 / len(rx_space) for r in rx_space}
                best_r, best_p = max(pd.items(), key=lambda x: x[1])
                pr[peer] = best_r
                p_prod *= best_p

            p_joint = p_f * p_prod
            if p_joint >= self.prob_threshold:
                outcomes.append((f, pr, p_joint))

        outcomes.sort(key=lambda x: x[2], reverse=True)
        return outcomes[: self.top_k]

    # ------------- BFS core (Algorithm 1) ------------------------------
    def _expectimax_search(
        self,
        A_cand: List[str],
        strategy: str,
        context: Dict[str, Any],
    ) -> Tuple[str, float, Dict[str, float]]:
        """Algorithm 1 – BFS-based Hypothetical Reasoning.

        Returns (best_action, utility, {action: utility}).
        """
        if not A_cand:
            return "HOLD", 0.0, {}

        # Compact world state for tree simulation --------------------------
        gs = context.get("game_state") or {}
        sc_count = 0
        if isinstance(gs, dict):
            key = self.country_name.upper()
            sc_list = gs.get("centers", {}).get(key, [])
            sc_count = len(sc_list) if isinstance(sc_list, list) else 0

        cur_ws: Dict[str, Any] = {
            "step": self.current_step,
            "supply_centers": sc_count,
            "tension": context.get("tension", 0.5),
        }

        # Tree bookkeeping -------------------------------------------------
        nodes: Dict[int, TreeNode] = {}
        _ctr = [0]
        root_ids: Dict[str, int] = {}

        def _mk(**kw) -> TreeNode:
            nid = _ctr[0]; _ctr[0] += 1
            n = TreeNode(node_id=nid, **kw)
            nodes[nid] = n
            return n

        # L_0: one node per candidate action (Alg.1 line 4) ---------------
        layer: List[TreeNode] = []
        for a in A_cand:
            n = _mk(
                parent_id=None, root_action=a, depth=0, action=a,
                world_state=dict(cur_ws), branch_prob=1.0, cumulative_prob=1.0,
            )
            root_ids[a] = n.node_id
            layer.append(n)

        # Depth iteration (Alg.1 lines 5–9) -------------------------------
        for d in range(self.search_depth):
            expanded: List[TreeNode] = []
            for node in layer:
                if node.pruned:
                    continue
                # Expand via Top-K outcomes weighted by P_joint (Eq.2)
                for f, pr, p_j in self._get_top_k_outcomes(node.action):
                    ws = self.adapter.compute_env_transition(
                        node.world_state, node.action, f, pr,
                    )
                    child = _mk(
                        parent_id=node.node_id, root_action=node.root_action,
                        depth=d + 1, action=node.action, world_state=ws,
                        branch_prob=p_j,
                        cumulative_prob=node.cumulative_prob * p_j,
                    )
                    node.children.append(child.node_id)
                    expanded.append(child)

            if not expanded:
                break

            # Batched risk assessment – Eq (3): L_safe = LLM_risk(L_d, G_meta)
            safe = list(expanded)
            if self.enable_risk_gate:
                risks = self._llm_batch_risk(expanded, strategy)
                for nd, rsk in zip(expanded, risks):
                    nd.risk_score = rsk
                    if rsk >= self.risk_threshold:
                        nd.pruned = True
                safe = [nd for nd in expanded if not nd.pruned]
                if not safe:
                    break

            # Batched actor – Eq (4): a_next = LLM_actor(L_safe, A_raw)
            if d < self.search_depth - 1 and safe:
                acts = self._llm_batch_actor(safe, strategy)
                for nd, na in zip(safe, acts):
                    nd.action = na

            layer = safe

        # Leaf utility evaluation – Eq (5): U_leaf = LLM_eval(L_leaf, G_meta)
        alive_leaves = [
            n for n in nodes.values()
            if (not n.children or n.depth >= self.search_depth)
               and not n.pruned and n.depth > 0
        ]
        if alive_leaves:
            utilities = self._llm_batch_eval(alive_leaves, strategy)
            for nd, u in zip(alive_leaves, utilities):
                nd.utility = u

        # Expectimax back-propagation – Eq (6)
        max_d = max((n.depth for n in nodes.values()), default=0)
        for d in range(max_d, -1, -1):
            for nd in nodes.values():
                if nd.depth != d or not nd.children:
                    continue
                s = 0.0
                for cid in nd.children:
                    ch = nodes[cid]
                    if ch.pruned:
                        continue
                    s += ch.branch_prob * ch.utility
                nd.utility = s

        # a* = argmax_{a ∈ A_cand} U(a)
        scores: Dict[str, float] = {
            a: nodes[nid].utility for a, nid in root_ids.items()
        }
        best = max(scores, key=scores.get) if scores else A_cand[0]  # type: ignore[arg-type]
        return best, scores.get(best, 0.0), scores

    # ---- Batched LLM helpers (constant API calls per layer) -----------

    def _llm_batch_risk(self, layer: List[TreeNode],
                         strategy: str) -> List[float]:
        """Eq (3): Batched risk assessment L_safe = LLM_risk(L_d, G_meta)."""
        if not layer:
            return []
        descs = [
            {"id": i, "action": n.action, "depth": n.depth,
             "feedback": n.world_state.get("last_feedback", "?"),
             "sc": n.world_state.get("supply_centers", "?")}
            for i, n in enumerate(layer)
        ]
        prompt = (
            "You are a strategic risk assessor.  Evaluate the risk of each "
            "hypothetical state against the meta-goal.\n\n"
            "Meta-Goal: {meta_goal}\nStrategy: {strategy}\n\n"
            "States:\n{states}\n\n"
            "Output JSON: {{\"risks\": [float, ...]}}\n"
            "Each value in [0, 1].  Array length MUST be {count}."
        )
        try:
            resp = self.get_response(prompt, input_param_dict={
                "meta_goal": self.meta_goal,
                "strategy": strategy,
                "states": descs,
                "count": len(layer),
            }, flag_debug_print=False)
            if isinstance(resp, dict):
                arr = resp.get("risks", [])
                if isinstance(arr, list) and len(arr) == len(layer):
                    return [max(0.0, min(1.0, float(r))) for r in arr]
        except Exception:
            pass
        return [0.3] * len(layer)

    def _llm_batch_actor(self, layer: List[TreeNode],
                          strategy: str) -> List[str]:
        """Eq (4): Batched next-action assignment a_next = LLM_actor(L_safe, A_raw)."""
        if not layer:
            return []
        avail = self.adapter.get_raw_actions()
        descs = [
            {"id": i, "action": n.action,
             "feedback": n.world_state.get("last_feedback", "?"),
             "sc": n.world_state.get("supply_centers", "?")}
            for i, n in enumerate(layer)
        ]
        prompt = (
            "For each state, select the best next action.\n\n"
            "Meta-Goal: {meta_goal}\nStrategy: {strategy}\n"
            "Actions: {actions}\n\nStates:\n{states}\n\n"
            "Output JSON: {{\"actions\": [str, ...]}}\n"
            "Array length MUST be {count}."
        )
        try:
            resp = self.get_response(prompt, input_param_dict={
                "meta_goal": self.meta_goal,
                "strategy": strategy,
                "actions": avail,
                "states": descs,
                "count": len(layer),
            }, flag_debug_print=False)
            if isinstance(resp, dict):
                arr = resp.get("actions", [])
                if isinstance(arr, list) and len(arr) == len(layer):
                    return [a if a in avail else avail[0] for a in arr]
        except Exception:
            pass
        return [n.action for n in layer]

    def _llm_batch_eval(self, leaves: List[TreeNode],
                         strategy: str) -> List[float]:
        """Eq (5): U_leaf = LLM_eval(L_leaf, G_meta)

        Returns a continuous utility scalar ∈ [0, 1] per leaf.
        """
        if not leaves:
            return []
        descs = [
            {"id": i, "action": n.action, "depth": n.depth,
             "feedback": n.world_state.get("last_feedback", "?"),
             "sc": n.world_state.get("supply_centers", "?"),
             "peers": n.world_state.get("last_peer_actions", {})}
            for i, n in enumerate(leaves)
        ]
        prompt = (
            "Evaluate each terminal state's alignment with the meta-goal.  "
            "Output a single utility score ∈ [0, 1] for each state where "
            "1 = perfectly aligned and 0 = worst outcome.\n\n"
            "Meta-Goal: {meta_goal}\nStrategy: {strategy}\n\n"
            "States:\n{states}\n\n"
            "Output JSON: {{\"utilities\": [float, ...]}}\n"
            "Array length MUST be {count}."
        )
        try:
            resp = self.get_response(prompt, input_param_dict={
                "meta_goal": self.meta_goal,
                "strategy": strategy,
                "states": descs,
                "count": len(leaves),
            }, flag_debug_print=False)
            if isinstance(resp, dict):
                arr = resp.get("utilities", [])
                if isinstance(arr, list) and len(arr) == len(leaves):
                    return [max(0.0, min(1.0, float(u))) for u in arr]
        except Exception:
            pass
        return [0.5] * len(leaves)

    # ====================================================================
    #  Stage 4 – Dynamic Interaction & Belief Calibration  (§3.4)
    # ====================================================================

    def _summarize_experience(self, action: str, feedback: str,
                               peer_reactions: Dict[str, str]) -> str:
        """Eq (7): e_obs = LLM_sum(a*, f_obs, r_obs)"""
        prompt = (
            "Summarise this interaction into a concise strategic lesson "
            "(max 30 words):\n"
            "Action: {action}\nFeedback: {feedback}\n"
            "Peer Reactions: {reactions}\n\n"
            "Output JSON: {{\"experience\": \"...\"}}"
        )
        try:
            resp = self.get_response(prompt, input_param_dict={
                "action": action, "feedback": feedback,
                "reactions": peer_reactions,
            }, flag_debug_print=False)
            if isinstance(resp, dict):
                return str(resp.get("experience", ""))
        except Exception:
            pass
        return f"Executed {action}, got {feedback}"

    def _update_world_model(
        self, action: str, feedback: str, peer_reactions: Dict[str, str],
    ) -> None:
        """Update W_{t+1} and recalibrate beliefs (Eq.7–11)."""
        exp = self._summarize_experience(action, feedback, peer_reactions)
        tup = InteractionTuple(
            action=action, feedback=feedback,
            peer_reactions=dict(peer_reactions),
            experience=exp, step=self.current_step,
        )
        self.world_model.add_interaction(tup)

    # ====================================================================
    #  Peer Predictions (for RQ2 reporting)
    # ====================================================================

    def _generate_predictions(self) -> List[PredictionRecord]:
        records: List[PredictionRecord] = []
        rx_space = self.adapter.get_reaction_space()
        raw_acts = self.adapter.get_raw_actions()

        for peer in self.adapter.get_peers():
            best_r, best_p = "", 0.0
            for a in raw_acts:
                dist = self.world_model.prob.get_peer_distribution(
                    peer, a, rx_space,
                )
                for r, p in dist.items():
                    if p > best_p:
                        best_p, best_r = p, r

            recent = [
                t for t in self.world_model.history[-5:]
                if peer in t.peer_reactions
            ]
            evidence = "; ".join(
                f"Step {t.step}: {peer}={t.peer_reactions[peer]}"
                for t in recent[-2:]
            ) or "Insufficient history"

            records.append(PredictionRecord(
                target=peer,
                predicted_action=best_r or "HOLD",
                support_evidence=evidence,
                confidence=best_p if best_p > 0 else 0.5,
            ))
        return records

    # ====================================================================
    #  Diplomacy backward-compatible 4-step API
    # ====================================================================

    def observe(self, round_context: Dict[str, Any]) -> Dict[str, Any]:
        """Update meta-state from context."""
        self.meta_state.update({
            "round":   round_context.get("round",
                                          self.meta_state.get("round", 0) + 1),
            "phase":   round_context.get("phase", "Unknown"),
            "tension": round_context.get("tension",
                                          self.meta_state.get("tension", 0.0)),
        })
        self.current_step = int(self.meta_state["round"])
        return {
            "last_orders":  round_context.get("last_orders", {}),
            "last_feedback": round_context.get("last_feedback", {}),
            "tension":       self.meta_state.get("tension", 0.0),
        }

    def orient(self, round_context: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 2: Candidate action pruning + peer predictions."""
        cands, strategy = self._filter_candidate_actions(round_context)
        self.current_strategy = strategy

        if self.enable_profiling:
            preds = self._generate_predictions()
        else:
            preds = [
                PredictionRecord(p, "HOLD", "Profiling disabled", 0.5)
                for p in self.other_countries
            ]
        self.prediction_cache[round_context.get("round", 0)] = preds

        return {
            "predictions":       preds,
            "candidate_actions": cands,
            "strategy":          strategy,
            "strategy_vision":   self.meta_goal,
            "predicted_states":  {},
        }

    def decide(self, round_context: Dict[str, Any],
               orient_state: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 3: BFS hypothetical reasoning → select best action."""
        cand = orient_state.get("candidate_actions", self.available_actions)
        strat = orient_state.get("strategy", self.current_strategy)
        preds = {r.target: r.predicted_action
                 for r in orient_state.get("predictions", [])}

        if not self.enable_prediction:
            best = self._fallback_decide(round_context, strat, cand)
            dec: Dict[str, Any] = {
                "selected_action": best, "utility": 0.5,
                "scores": {a: 0.0 for a in cand},
                "predictions": preds, "llm_scores": {},
            }
        else:
            best, util, scores = self._expectimax_search(
                cand, strat, round_context,
            )
            dec = {
                "selected_action": best, "utility": util,
                "scores": scores,
                "predictions": preds, "llm_scores": {},
            }

        # Concrete orders (Diplomacy)
        legal = round_context.get("legal_orders_by_loc")
        gs    = round_context.get("game_state")
        if isinstance(legal, dict) and isinstance(gs, dict):
            try:
                concrete = self.propose_legal_orders(
                    round_context=round_context, game_state=gs,
                    legal_orders_by_loc=legal,
                    high_level_action=best, strategy_text=strat,
                )
                if isinstance(concrete, list):
                    dec["concrete_orders"] = concrete
            except Exception as e:
                dec["concrete_orders_error"] = repr(e)
        return dec

    def _fallback_decide(self, ctx: Dict, strat: str,
                          cands: List[str]) -> str:
        """Simple single-LLM-call fallback when prediction is disabled."""
        try:
            prompt = (
                "Choose the single best action.\n"
                "Strategy: {strategy}\nGoal: {goal}\n"
                "Actions: {actions}\nState: {state}\n\n"
                "Output JSON: {{\"action\": \"...\"}}"
            )
            resp = self.get_response(prompt, input_param_dict={
                "strategy": strat, "goal": self.meta_goal,
                "actions": cands,
                "state": self.adapter.format_state_for_llm(ctx),
            }, flag_debug_print=False)
            if isinstance(resp, dict):
                a = resp.get("action", "")
                if a in cands:
                    return a
        except Exception:
            pass
        return cands[0] if cands else "HOLD"

    def act(self, decision: Dict[str, Any]) -> str:
        chosen = decision.get("selected_action", "HOLD")
        concrete = decision.get("concrete_orders")
        if isinstance(concrete, list):
            self.last_concrete_orders = [
                str(o) for o in concrete if isinstance(o, str)
            ]
        return chosen

    def get_last_concrete_orders(self) -> List[str]:
        return list(self.last_concrete_orders)

    def run_cycle(self, round_context: Dict[str, Any]) -> Dict[str, Any]:
        """Full decision loop (Algorithm 1)."""
        obs = self.observe(round_context)
        ori = self.orient(round_context)
        dec = self.decide(round_context, ori)
        action = self.act(dec)
        record = {
            "action": action, "observation": obs,
            "orient": ori, "decision": dec,
        }
        self._log_cognition("decision", record)
        return record

    def learn_from_interaction(
        self, action: str, world_feedback: str,
        other_actions: Dict[str, str], world_memory: Any = None,
    ) -> None:
        """Stage 4: Update world model with observed interaction."""
        fb = self._classify_feedback(world_feedback)
        pr = {c: a for c, a in other_actions.items()
              if c in self.other_countries}
        self._update_world_model(action, fb, pr)

    @staticmethod
    def _classify_feedback(feedback: str) -> str:
        if not feedback:
            return "stable"
        s = feedback.lower()
        if any(w in s for w in ("gain", "capture", "success", "up")):
            return "gain"
        if any(w in s for w in ("loss", "retreat", "defeat", "down")):
            return "loss"
        return "stable"

    # ====================================================================
    #  Diplomacy concrete-order generation
    # ====================================================================

    _DIPLOMACY_KNOWLEDGE: str = (
        "Diplomacy(标准地图)要点：\n"
        "- 每回合需要为每个可下单地点提交 1 条指令。\n"
        "- 合法指令以引擎提供的 legal orders 为准；必须逐条原样选取。\n"
        "- 常见指令：H(保持)、- (移动)、S(支援)、C(运输)。\n"
        "- 目标优先级：争夺中立补给中心(SC)；保住本土；避免无谓对撞。\n"
    )

    def propose_legal_orders(
        self,
        round_context: Dict[str, Any],
        game_state: Dict[str, Any],
        legal_orders_by_loc: Dict[str, List[str]],
        high_level_action: str,
        strategy_text: str,
    ) -> List[str]:
        power      = self.country_name.upper()
        my_units   = (game_state.get("units",   {}) or {}).get(power, [])
        my_centers = (game_state.get("centers", {}) or {}).get(power, [])
        all_units  = game_state.get("units", {}) or {}
        enemy      = {k: v for k, v in all_units.items() if k != power}

        def _hold(loc: str) -> Optional[str]:
            for o in legal_orders_by_loc.get(loc, []):
                if isinstance(o, str) and o.strip().endswith(" H"):
                    return o
            opts = legal_orders_by_loc.get(loc, [])
            return opts[0] if opts else None

        prompt = (
            "You are the tactical commander for {power} in Diplomacy.\n"
            "You MUST ONLY select from the given legal orders.\n\n"
            "Background:\n{knowledge}\n"
            "Goal: {goal}\nRound={round} Phase={phase}\n"
            "Intent: {action}\nStrategy: {strategy}\n\n"
            "My units: {my_units}\nMy SC: {my_centers}\n"
            "Enemy: {enemy}\n\n"
            "Legal orders:\n{legal}\n\n"
            "Pick ONE order per location.\n"
            "Output JSON: {{\"orders_by_loc\": {{\"LOC\": \"order\"}}, "
            "\"rationale\": \"brief\"}}"
        )
        resp = self.get_response(prompt, input_param_dict={
            "power": power, "knowledge": self._DIPLOMACY_KNOWLEDGE,
            "goal": self.meta_goal,
            "round": round_context.get("round"),
            "phase": round_context.get("phase"),
            "action": high_level_action, "strategy": strategy_text,
            "my_units": my_units, "my_centers": my_centers,
            "enemy": enemy, "legal": legal_orders_by_loc,
        }, flag_debug_print=False)

        chosen: Dict[str, str] = {}
        if isinstance(resp, dict):
            raw = resp.get("orders_by_loc")
            if isinstance(raw, dict):
                for loc, order in raw.items():
                    if isinstance(loc, str) and isinstance(order, str):
                        chosen[loc.strip().upper()] = order.strip()

        final: List[str] = []
        for loc in legal_orders_by_loc.keys():
            lu = str(loc).upper()
            c = chosen.get(lu)
            ll = legal_orders_by_loc.get(lu,
                    legal_orders_by_loc.get(loc, [])) or []
            if c and c in ll:
                final.append(c)
            else:
                h = _hold(lu)
                if h:
                    final.append(h)

        if not final and my_units:
            for unit in my_units:
                parts = str(unit).split()
                loc = parts[1] if len(parts) > 1 else ""
                h = _hold(loc)
                if h:
                    final.append(h)
        return final

    # ====================================================================
    #  Legacy compatibility
    # ====================================================================

    def cognitive_game_decide(self, world_info: str) -> Dict[str, Any]:
        ctx = {"world_info": world_info,
               "round": self.meta_state.get("round", 0) + 1}
        result = self.run_cycle(ctx)
        return {
            "action": result["action"],
            "declaration": self.current_strategy,
            "reasoning_result": result,
            "satisfaction_score": result["decision"].get("utility", 0.5),
            "reasoning_depth": self.search_depth,
        }

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
