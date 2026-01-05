"""一键运行 Diplomacy 锦标赛。

无需命令行参数/环境变量，直接改代码即可。

本文件提供两个“运行入口”：
    - RQ3/常规：单配置跑锦标赛（保留旧入口行为）
    - RQ4/消融：Full + 3 个消融变体，每个变体跑若干局，并汇总输出 RQ4_Ablation.csv

通过 RUN_MODE 选择运行模式。
"""

import asyncio
from pathlib import Path
from datetime import datetime
import os
import sys

from simulation.diplomacy.tournament import TournamentConfig, run_tournament


# 直接在此修改配置
RUN_MODE = "RQ3_MODELS"  # "RQ3" / "RQ4" / "RQ3_MODELS"

# RQ3（常规）
GAMES = 20

# RQ4（消融）
GAMES_PER_VARIANT = 10
ROUNDS_PER_GAME = 20
MAX_YEAR = 1910
# 每次运行生成独立目录，避免相互覆盖
_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(f"experiments/diplomacy_tournament_{_ts}")


MODEL_GROUPS = [
    "gpt-4o",
    "gpt-5",
    "glm-4.5",
]


def _safe_dir_name(name: str) -> str:
    out = []
    for ch in str(name):
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "model"


# RQ4 variants (Full + Ablations)
VARIANTS = [
    (
        "Full_Model",
        dict(enable_profiling=True, enable_prediction=True, enable_risk_gate=True),
    ),
    (
        "w/o_Observe",
        dict(enable_profiling=False, enable_prediction=True, enable_risk_gate=True),
    ),
    (
        "w/o_Orient",
        dict(enable_profiling=True, enable_prediction=False, enable_risk_gate=True),
    ),
    (
        "w/o_Decide",
        dict(enable_profiling=True, enable_prediction=True, enable_risk_gate=False),
    ),
    (
        "w/o_All",
        dict(enable_profiling=False, enable_prediction=False, enable_risk_gate=False),
    ),
]


class _TeeIO:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
                # VS Code 终端在自定义 stdout 对象下可能出现缓冲，写入后立刻 flush。
                s.flush()
            except Exception:
                pass

    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass


async def _run_rq3_single() -> None:
    cfg = TournamentConfig(
        games=GAMES,
        rounds_per_game=ROUNDS_PER_GAME,
        max_year=MAX_YEAR,
        output_dir=OUTPUT_DIR,
    )
    await run_tournament(cfg)


async def _run_rq3_models() -> None:
    for model_name in MODEL_GROUPS:
        model_dir = OUTPUT_DIR / f"RQ3_model_{_safe_dir_name(model_name)}"
        cfg = TournamentConfig(
            games=GAMES,
            rounds_per_game=ROUNDS_PER_GAME,
            max_year=MAX_YEAR,
            output_dir=model_dir,
            llm_model=model_name,
        )
        print(f"\n[系统] 开始 RQ3 模型组={model_name} 局数={GAMES} 输出目录={model_dir}")
        await run_tournament(cfg)


async def _run_rq4_all_variants() -> None:
    shared_rq4 = OUTPUT_DIR / "RQ4_Ablation.csv"
    for variant_name, flags in VARIANTS:
        variant_dir = OUTPUT_DIR / variant_name
        cfg = TournamentConfig(
            games=GAMES_PER_VARIANT,
            rounds_per_game=ROUNDS_PER_GAME,
            max_year=MAX_YEAR,
            output_dir=variant_dir,
            configuration=variant_name,
            rq4_path=shared_rq4,
            **flags,
        )
        print(f"\n[系统] 开始 RQ4 配置={variant_name} 局数={GAMES_PER_VARIANT} 输出目录={variant_dir}")
        await run_tournament(cfg)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # 让 LLMAgent 将 trace 直接打印到控制台（由 console_output.log 捕获）
    os.environ["MAGES_TRACE"] = "1"

    log_path = OUTPUT_DIR / "console_output.log"
    with open(log_path, "w", encoding="utf-8") as f:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = _TeeIO(old_out, f)
        sys.stderr = _TeeIO(old_err, f)
        try:
            if RUN_MODE.upper() == "RQ3":
                asyncio.run(_run_rq3_single())
            elif RUN_MODE.upper() == "RQ3_MODELS":
                asyncio.run(_run_rq3_models())
            else:
                asyncio.run(_run_rq4_all_variants())
        finally:
            sys.stdout = old_out
            sys.stderr = old_err


if __name__ == "__main__":
    main()
