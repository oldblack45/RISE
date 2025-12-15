"""
一键运行 Diplomacy 锦标赛（直接改代码即可，无需命令行参数/环境变量）。

修改以下常量即可：
  GAMES           比赛局数
  ROUNDS_PER_GAME 每局最大回合数
  MAX_YEAR        最晚结束年份
  OUTPUT_DIR      输出目录
"""

import asyncio
from pathlib import Path
from datetime import datetime
import os
import sys

from simulation.diplomacy.tournament import TournamentConfig, run_tournament


# 直接在此修改配置
GAMES = 1
ROUNDS_PER_GAME = 20
MAX_YEAR = 1910
# 每次运行生成独立目录，避免相互覆盖
_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(f"experiments/diplomacy_tournament_{_ts}")


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
            cfg = TournamentConfig(
                games=GAMES,
                rounds_per_game=ROUNDS_PER_GAME,
                max_year=MAX_YEAR,
                output_dir=OUTPUT_DIR,
            )
            asyncio.run(run_tournament(cfg))
        finally:
            sys.stdout = old_out
            sys.stderr = old_err


if __name__ == "__main__":
    main()
