"""Grand Unified Diplomacy Tournament runner.

This module orchestrates Diplomacy (standard/no-press) tournaments where
SAGE (England) competes against a roster of baseline agents. The runner logs
RQ2 (prediction accuracy) and RQ3 (game performance) metrics from the same
experiments to support joint cognitive + game-theoretic evaluation.
"""
from __future__ import annotations

import asyncio
import csv
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from diplomacy import Game

from agents.sage_agent import SAGEAgent
from agents.diplomacy_baselines import EvoBaselineAgent, ReActBaselineAgent, ReflexionBaselineAgent
from simulation.models.agents.LLMAgent import reset_llm_call_stats, set_llm_log_context, snapshot_llm_call_stats

AGGRESSIVE_ACTIONS = {"ATTACK", "SUPPORT_ATTACK"}

# ---------------------------------------------------------------------------
# Baseline personas and architectures -------------------------------------------------

BASELINE_TYPES = ["ReAct", "Reflexion", "EvoAgent"]
PLAYERS = ["England", "France", "Germany", "Italy", "Austria", "Russia", "Turkey"]


@dataclass
class TournamentConfig:
    games: int = 20
    max_year: int = 1910
    rounds_per_game: int = 20
    output_dir: Path = Path("experiments/diplomacy_tournament")
    # Model
    llm_model: Optional[str] = None
    # RQ4 Ablation
    configuration: str = ""
    enable_profiling: bool = True
    enable_prediction: bool = True
    enable_risk_gate: bool = True
    # If provided, RQ4 rows will be appended to this CSV (can be shared across variants).
    # Leave None to disable RQ4 logging for legacy runs.
    rq4_path: Optional[Path] = None


# ---------------------------------------------------------------------------
# Baseline agent skeleton ---------------------------------------------------

BaselineAgent = Any


# ---------------------------------------------------------------------------
# Tournament runner ----------------------------------------------------------

class DiplomacyTournamentRunner:
    def __init__(self, config: TournamentConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.rq2_path = self.config.output_dir / "RQ2_Evolution.csv"
        self.rq3_path = self.config.output_dir / "RQ3_Performance.csv"
        self.turn_log_path = self.config.output_dir / "Turn_Log.csv"
        self.rq4_path = self.config.rq4_path
        self._init_csv_logs()

    def _init_csv_logs(self) -> None:
        with open(self.rq2_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["GameID", "Round", "Phase", "Target_Country", "Target_Persona", "Prediction_Accuracy"])
        with open(self.rq3_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["GameID", "SAGE_Final_SC", "SAGE_Survived", "Winner_Architecture"])
        with open(self.turn_log_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "GameID",
                "Round",
                "PhaseName",
                "PhaseType",
                "Year",
                "Country",
                "Action",
                "Tension",
                "Feedback",
                "SupplyCenters",
            ])

        # RQ4: append-friendly (shared file across variants)
        if self.rq4_path is not None:
            try:
                if not self.rq4_path.exists():
                    self.rq4_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(self.rq4_path, "w", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow(["Configuration", "GameID", "Final_SC", "Survived", "Is_Winner"])
            except Exception:
                pass

    def _append_warning(self, game_id: int, round_no: int, country: str, baseline_type: str, stage: str, error: str) -> None:
        msg = f"[LLM警告] 第{game_id}局 第{round_no}轮 {country}({baseline_type}) 阶段={stage} 错误={error}"
        print(msg)

    def _sys_log(self, message: str) -> None:
        print(f"[系统] {message}")

    async def run(self) -> None:
        for gid in range(1, self.config.games + 1):
            await self._run_single_game(gid)

    async def _run_single_game(self, game_id: int) -> None:
        game = Game(map_name="standard")
        self._sys_log(f"第{game_id}局开始：地图=standard 参赛方数量={len(PLAYERS)}")
        sage = SAGEAgent(
            country_name="England",
            other_countries=[p for p in PLAYERS if p != "England"],
            game_attributes={"max_rounds": self.config.rounds_per_game},
            experiment_logger=None,
            meta_goal="Maximize national interest and ensure survival",
            llm_model=self.config.llm_model,
            enable_profiling=self.config.enable_profiling,
            enable_prediction=self.config.enable_prediction,
            enable_risk_gate=self.config.enable_risk_gate,
        )
        baselines = self._spawn_baselines()
        try:
            roster = {c: getattr(a, "baseline_type", "Unknown") for c, a in baselines.items()}
        except Exception:
            roster = {}
        if roster:
            self._sys_log(f"对手基线配置：{roster}")
        last_orders: Dict[str, str] = {}
        last_concrete_orders: Dict[str, List[str]] = {}
        last_feedback: Dict[str, str] = {}
        prev_sc = self._count_supply_centers(game)
        tension = 0.5

        round_no = 0
        while round_no < self.config.rounds_per_game:
            turn_start_ts = time.perf_counter()
            state = self._ensure_movement_phase(game)
            year = state.get("year", 1901)
            if year > self.config.max_year:
                break
            round_no += 1

            # 每轮开始重置 LLM 调用统计
            try:
                reset_llm_call_stats(game_id, round_no)
            except Exception:
                pass

            phase_name = str(state.get("name", ""))
            phase_type = str(state.get("phase_type", ""))
            self._sys_log(f"第{game_id}局 第{round_no}轮开始：阶段={phase_name} 类型={phase_type} 年份={year} 紧张度={tension:.2f}")
            phase_label = self._phase_label(state.get("name", ""))
            orders = await self._gather_orders(
                game,
                sage,
                baselines,
                game_id,
                round_no,
                last_orders,
                last_concrete_orders,
                last_feedback,
                tension,
                phase_label,
            )

            england_used_concrete = bool((orders.get("concrete") or {}).get("England"))
            self._sys_log(
                f"第{game_id}局 第{round_no}轮下单摘要：英国动作={orders['abstract'].get('England','HOLD')} "
                f"是否使用具体orders={england_used_concrete} 其他国家动作={{{', '.join([f'{k}:{v}' for k,v in orders['abstract'].items() if k!='England'])}}}"
            )
            self._submit_orders_to_game(game, orders["abstract"], orders["concrete"])
            new_sc = self._count_supply_centers(game)
            last_feedback = self._derive_feedback(prev_sc, new_sc)
            prev_sc = new_sc
            last_orders = orders["abstract"]
            last_concrete_orders = orders["concrete"]
            tension = self._update_tension(orders["abstract"], tension)

            try:
                self._sys_log(f"第{game_id}局 第{round_no}轮结算：补给中心={new_sc} 反馈={last_feedback}")
            except Exception:
                pass
            self._append_turn_log(
                game_id,
                round_no,
                game.get_state(),
                orders["abstract"],
                tension,
                last_feedback,
                new_sc,
            )
            state = game.get_state()
            await self._post_round_updates(
                sage,
                baselines,
                orders["abstract"],
                state,
                last_feedback,
                tension,
                prev_sc,
                new_sc,
                game_id,
                round_no,
            )

            # 每轮结束输出 LLM 调用统计（总次数 + 每个 agent）
            try:
                stats = snapshot_llm_call_stats(game_id, round_no)
                by_agent = stats.get("by_agent", {}) if isinstance(stats, dict) else {}
                total = int(stats.get("total", 0)) if isinstance(stats, dict) else 0
                parts = [f"总计={total}"]
                for name, cnt in sorted(by_agent.items(), key=lambda kv: (-int(kv[1]), str(kv[0]))):
                    parts.append(f"{name}={int(cnt)}")
                self._sys_log(f"第{game_id}局 第{round_no}轮 LLM调用次数统计：" + "，".join(parts))
            except Exception as e:
                self._sys_log(f"第{game_id}局 第{round_no}轮 LLM调用次数统计失败：{repr(e)}")

            # 每轮耗时
            try:
                turn_cost = time.perf_counter() - turn_start_ts
                self._sys_log(f"第{game_id}局 第{round_no}轮运行耗时：{turn_cost:.2f}秒")
            except Exception:
                pass

            if getattr(game, "is_game_done", False):
                break

        await self._log_rq3(game_id, game, baselines)
        self._append_rq4(game_id, game, baselines)
        self._sys_log(f"第{game_id}局结束：最终补给中心={self._count_supply_centers(game)}")

    def _append_rq4(self, game_id: int, game: Any, baselines: Dict[str, BaselineAgent]) -> None:
        if self.rq4_path is None:
            return
        sc_map = self._count_supply_centers(game)
        england_sc = int(sc_map.get("England", 0))
        survived = england_sc > 0
        winner = max(sc_map.items(), key=lambda item: item[1])[0] if sc_map else "Unknown"
        is_winner = winner == "England"
        try:
            with open(self.rq4_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    self.config.configuration,
                    game_id,
                    england_sc,
                    bool(survived),
                    bool(is_winner),
                ])
        except Exception:
            pass

    def _spawn_baselines(self) -> Dict[str, BaselineAgent]:
        agents: Dict[str, BaselineAgent] = {}
        for player in PLAYERS:
            if player == "England":
                continue
            baseline_type = random.choice(BASELINE_TYPES)
            if baseline_type == "ReAct":
                agents[player] = ReActBaselineAgent(player, llm_model=self.config.llm_model)
            elif baseline_type == "Reflexion":
                agents[player] = ReflexionBaselineAgent(player, llm_model=self.config.llm_model)
            else:
                agents[player] = EvoBaselineAgent(player, llm_model=self.config.llm_model)
        return agents

    async def _gather_orders(
        self,
        game: Any,
        sage: SAGEAgent,
        baselines: Dict[str, BaselineAgent],
        game_id: int,
        round_no: int,
        last_orders: Dict[str, str],
        last_concrete_orders: Dict[str, List[str]],
        last_feedback: Dict[str, str],
        tension: float,
        phase_label: str,
    ) -> Dict[str, Dict[str, Any]]:
        legal_by_loc: Dict[str, List[str]] = {}
        try:
            legal_by_loc = self._legal_orders_by_location(game, "ENGLAND")
        except Exception:
            legal_by_loc = {}

        context = {
            "round": round_no,
            "phase": phase_label,
            "last_orders": last_orders,
            "last_concrete_orders": last_concrete_orders,
            "last_feedback": last_feedback,
            "personas": {country: agent.persona for country, agent in baselines.items()},
            "tension": tension,
            "history": [],
            "game_state": game.get_state(),
            "legal_orders_by_loc": legal_by_loc,
        }
        try:
            set_llm_log_context({
                "scenario": "diplomacy",
                "role": "sage",
                "country": "England",
                "game_id": game_id,
                "round": round_no,
                "phase": phase_label,
                "stage": "observe",
            })
        except Exception:
            pass
        sage.observe(context)

        try:
            set_llm_log_context({
                "scenario": "diplomacy",
                "role": "sage",
                "country": "England",
                "game_id": game_id,
                "round": round_no,
                "phase": phase_label,
                "stage": "orient",
            })
        except Exception:
            pass
        orient_state = sage.orient(context)
        predictions = orient_state.get("predictions", [])
        candidate_actions = orient_state.get("candidate_actions", sage.available_actions)

        state = game.get_state()
        baseline_coros = [
            agent.propose_orders({
                "game_id": game_id,
                "round": round_no,
                "tension": tension,
                "state": state,
                "last_orders": last_orders,
                "last_feedback": last_feedback,
            })
            for agent in baselines.values()
        ]
        baseline_future = asyncio.gather(*baseline_coros, return_exceptions=True)
        try:
            set_llm_log_context({
                "scenario": "diplomacy",
                "role": "sage",
                "country": "England",
                "game_id": game_id,
                "round": round_no,
                "phase": phase_label,
                "stage": "decide",
            })
        except Exception:
            pass
         # 传入完整 orient_state：decide 需要 strategy / predicted_states 等信息
        decision_future = asyncio.to_thread(sage.decide, context, orient_state)
        baseline_orders, decision = await asyncio.gather(baseline_future, decision_future)
        england_action = sage.act(decision)

        england_concrete_orders: List[str] = []
        if isinstance(decision, dict) and decision.get("concrete_orders_error"):
            self._append_warning(game_id, round_no, "England", "SAGE", "concrete_orders", str(decision.get("concrete_orders_error")))
        if isinstance(decision, dict) and isinstance(decision.get("concrete_orders"), list):
            england_concrete_orders = [str(o) for o in decision.get("concrete_orders") if isinstance(o, str)]
        elif getattr(sage, "get_last_concrete_orders", None):
            england_concrete_orders = list(sage.get_last_concrete_orders())

        abstract: Dict[str, str] = {}
        concrete: Dict[str, List[str]] = {}
        for (country, agent), action in zip(baselines.items(), baseline_orders):
            baseline_type = str(getattr(agent, "baseline_type", "Unknown"))
            if isinstance(action, Exception):
                self._append_warning(game_id, round_no, country, baseline_type, "propose_orders", repr(action))
                chosen = "HOLD"
            elif isinstance(action, str):
                chosen = action
            else:
                self._append_warning(game_id, round_no, country, baseline_type, "propose_orders", f"non-str return: {type(action)}")
                chosen = "HOLD"
            abstract[country] = chosen
            predicted = next((rec.predicted_action for rec in predictions if rec.target == country), "HOLD")
            accuracy = 1 if chosen == predicted else 0
            target_persona = getattr(agent, "persona", getattr(agent, "baseline_type", "Unknown"))
            self._append_rq2(game_id, round_no, context["phase"], country, str(target_persona), accuracy)

        abstract["England"] = england_action
        if england_concrete_orders:
            concrete["England"] = england_concrete_orders
        return {"abstract": abstract, "concrete": concrete}

    async def _post_round_updates(
        self,
        sage: SAGEAgent,
        baselines: Dict[str, BaselineAgent],
        orders: Dict[str, str],
        state: Dict[str, Any],
        feedback: Dict[str, str],
        tension: float,
        prev_sc: Dict[str, int],
        new_sc: Dict[str, int],
        game_id: int,
        round_no: int,
    ) -> None:
        england_feedback = feedback.get("England", "stable")
        feedback_text = f"SC trend: {england_feedback}; tension={tension:.2f}"
        other_actions = {country: action for country, action in orders.items() if country != "England"}
        await asyncio.to_thread(sage.learn_from_interaction, orders["England"], feedback_text, other_actions, state)

        game_view = {
            "tension": tension,
            "state": state,
            "last_orders": orders,
            "last_feedback": feedback,
            "prev_sc": prev_sc,
            "new_sc": new_sc,
        }
        for country, agent in baselines.items():
            agent.last_order = orders.get(country, agent.last_order)
            try:
                prev = prev_sc.get(country, 0)
                cur = new_sc.get(country, prev)
                sc_delta = int(cur) - int(prev)
                fb = feedback.get(country, "stable")
                agent.post_round_update(game_view, fb, sc_delta)
            except Exception as e:
                baseline_type = str(getattr(agent, "baseline_type", "Unknown"))
                self._append_warning(game_id, round_no, country, baseline_type, "post_round_update", repr(e))

    def _append_rq2(self, game_id: int, round_no: int, phase: str, country: str, persona: str, acc: int) -> None:
        with open(self.rq2_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([game_id, round_no, phase, country, persona, acc])

    def _append_turn_log(
        self,
        game_id: int,
        round_no: int,
        state: Dict[str, Any],
        orders: Dict[str, str],
        tension: float,
        feedback: Dict[str, str],
        supply_centers: Dict[str, int],
    ) -> None:
        phase_name = state.get("name", "")
        phase_type = state.get("phase_type", "")
        year = state.get("year", "")
        with open(self.turn_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for country in PLAYERS:
                action = orders.get(country, "HOLD")
                fb = feedback.get(country, "")
                sc = supply_centers.get(country, "")
                writer.writerow([
                    game_id,
                    round_no,
                    phase_name,
                    phase_type,
                    year,
                    country,
                    action,
                    f"{tension:.2f}",
                    fb,
                    sc,
                ])

    async def _log_rq3(self, game_id: int, game: Any, baselines: Dict[str, BaselineAgent]) -> None:
        sc_map = self._count_supply_centers(game)
        england_sc = sc_map.get("England", 0)
        survived = england_sc > 0
        winner = max(sc_map.items(), key=lambda item: item[1])[0]
        winner_arch = "SAGE" if winner == "England" else getattr(baselines.get(winner), "baseline_type", "Unknown")
        with open(self.rq3_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([game_id, england_sc, survived, winner_arch])

    def _derive_feedback(self, prev_sc: Dict[str, int], new_sc: Dict[str, int]) -> Dict[str, str]:
        feedback: Dict[str, str] = {}
        for player in PLAYERS:
            prev = prev_sc.get(player, 0)
            cur = new_sc.get(player, prev)
            delta = cur - prev
            if delta > 0:
                feedback[player] = "gain"
            elif delta < 0:
                feedback[player] = "loss"
            else:
                feedback[player] = "stable"
        return feedback

    def _update_tension(self, orders: Dict[str, str], prev_tension: float) -> float:
        if not orders:
            return prev_tension
        aggressive_ratio = sum(1 for action in orders.values() if action in AGGRESSIVE_ACTIONS) / len(orders)
        return 0.6 * prev_tension + 0.4 * aggressive_ratio

    def _count_supply_centers(self, game: Game) -> Dict[str, int]:
        state = game.get_state()
        centers = state.get("centers", {})
        return {power: len(centers.get(power.upper(), [])) for power in PLAYERS}

    def _phase_label(self, phase_code: str) -> str:
        if phase_code.startswith("S"):
            return "S"
        if phase_code.startswith("F"):
            return "F"
        return "?"

    def _ensure_movement_phase(self, game: Game) -> Dict[str, Any]:
        attempts = 0
        while True:
            state = game.get_state()
            phase_name = state.get("name", "")
            phase_type = state.get("phase_type")
            # 一些状态缺少 phase_type，但名称以 "M" 结尾代表行动阶段
            if phase_type == "M" or str(phase_name).endswith("M"):
                return state
            print(f"[Diplomacy] Resolving non-movement phase {phase_name} ({phase_type}), attempt {attempts+1}")
            self._resolve_non_movement_phase(game, state)
            attempts += 1
            if attempts > 20:
                raise RuntimeError("Failed to reach movement phase after multiple resolutions")

    def _resolve_non_movement_phase(self, game: Game, state: Dict[str, Any]) -> None:
        phase_type = state.get("phase_type")
        if phase_type == "R":
            retreats = state.get("retreats", {})
            for country, units in retreats.items():
                orders = [f"{unit} D" for unit in units.keys()]
                if orders:
                    game.set_orders(country, orders)
        elif phase_type == "A":
            # 调整阶段：为所有可下单的地点生成“建陆军”指令，确保合法推进
            orderable = state.get("orderable_locations", {})
            for country, locations in orderable.items():
                orders = [f"B {loc} A" for loc in locations]
                if orders:
                    game.set_orders(country, orders)
        # 处理阶段推进
        game.process()

    def _submit_orders_to_game(self, game: Game, abstract_orders: Dict[str, str], concrete_overrides: Dict[str, List[str]]) -> None:
        state = game.get_state()
        for country in PLAYERS:
            override = concrete_overrides.get(country)
            if override:
                orders = override
            else:
                action = abstract_orders.get(country, "HOLD")
                orders = self._build_orders_for_country(game, state, country, action)
            if orders:
                game.set_orders(country.upper(), orders)
        game.process()

    def _legal_orders_by_location(self, game: Game, power: str) -> Dict[str, List[str]]:
        """Return orderable location -> legal orders for a power (e.g., ENGLAND)."""
        try:
            orderable = game.get_orderable_locations().get(power.upper(), [])
        except Exception:
            orderable = []
        try:
            all_possible = game.get_all_possible_orders()  # location -> [orders]
        except Exception:
            all_possible = {}
        legal: Dict[str, List[str]] = {}
        for loc in orderable:
            opts = all_possible.get(loc, [])
            if isinstance(opts, list) and opts:
                legal[loc] = list(opts)
            else:
                legal[loc] = []
        return legal

    def _build_orders_for_country(self, game: Game, state: Dict[str, Any], country: str, action: str) -> List[str]:
        """Map abstract_action -> concrete legal orders (deterministic, no randomness).

        NOTE: We intentionally select from diplomacy engine provided legal orders to
        avoid illegal moves and remove random neighbor drifting.
        """

        power = country.upper()
        legal_by_loc = self._legal_orders_by_location(game, power)
        if not legal_by_loc:
            return []

        my_centers = set((state.get("centers", {}) or {}).get(power, []) or [])

        # Try to get a stable list of all supply centers.
        all_scs: set[str] = set()
        try:
            scs = getattr(getattr(game, "map", None), "scs", None)
            if isinstance(scs, (list, tuple, set)):
                all_scs |= {str(x) for x in scs}
        except Exception:
            pass
        try:
            centers = state.get("centers", {}) or {}
            for _, sc_list in centers.items():
                if isinstance(sc_list, list):
                    all_scs |= {str(x) for x in sc_list}
        except Exception:
            pass

        target_scs = all_scs - my_centers if all_scs else set()

        def _parse_move_dest(order: str) -> Optional[str]:
            if " - " not in order:
                return None
            # e.g., "A LVP - WAL" / "F EDI - NTH" / convoy variants still end with dest token
            try:
                return order.split(" - ")[-1].strip().split()[-1]
            except Exception:
                return None

        def _is_hold(order: str) -> bool:
            # common forms: "A PAR H", "F LON H"
            return order.strip().endswith(" H")

        def _is_move(order: str) -> bool:
            return " - " in order

        def _is_support(order: str) -> bool:
            # e.g., "A PAR S A MAR - BUR" or "A PAR S A MAR H"
            return " S " in order

        def _prefer_order(candidates: List[str]) -> Optional[str]:
            if not candidates:
                return None
            # deterministic: sort then pick best by scoring
            best: Optional[str] = None
            best_score: Tuple[int, str] = (-10**9, "")
            for o in sorted({str(x) for x in candidates}):
                score = 0
                if action == "HOLD":
                    score += 1000 if _is_hold(o) else 0
                elif action in {"MOVE", "ATTACK"}:
                    if _is_move(o):
                        score += 500
                        dest = _parse_move_dest(o)
                        if dest and dest in target_scs:
                            score += 200
                    if _is_hold(o):
                        score += 10
                elif action == "SUPPORT_ATTACK":
                    if _is_support(o) and _is_move(o):
                        score += 600
                    elif _is_support(o):
                        score += 200
                    elif _is_move(o):
                        score += 50
                    if _is_hold(o):
                        score += 10
                elif action == "SUPPORT_DEFEND":
                    # prefer support-hold, then any support
                    if _is_support(o) and _is_hold(o):
                        score += 600
                    elif _is_support(o):
                        score += 300
                    elif _is_hold(o):
                        score += 50
                else:
                    # RETREAT or unknown -> be conservative
                    score += 1000 if _is_hold(o) else 0

                key = (score, o)
                if key > best_score:
                    best_score = key
                    best = o
            return best

        chosen_orders: List[str] = []
        for loc in sorted(legal_by_loc.keys()):
            opts = legal_by_loc.get(loc, [])
            picked = _prefer_order(opts)
            if picked:
                chosen_orders.append(picked)
        return chosen_orders


async def run_tournament(config: TournamentConfig) -> None:
    runner = DiplomacyTournamentRunner(config)
    await runner.run()

# DEFAULT_SAGE_LLM_MODEL = "qwen3-max"
