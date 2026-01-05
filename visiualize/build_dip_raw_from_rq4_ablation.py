import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class DipSummary:
    win_rate: float
    survival_rate: float
    avg_sc: float

    def as_list(self, ndigits: int | None = 3) -> List[float]:
        if ndigits is None:
            return [self.win_rate, self.survival_rate, self.avg_sc]
        return [round(self.win_rate, ndigits), round(self.survival_rate, ndigits), round(self.avg_sc, ndigits)]


def _parse_bool(value: str) -> bool:
    v = value.strip().lower()
    if v in {"true", "1", "yes", "y", "t"}:
        return True
    if v in {"false", "0", "no", "n", "f"}:
        return False
    raise ValueError(f"无法解析布尔值: {value!r}")


def load_rq4_ablation_rows(csv_path: str | Path) -> List[dict]:
    csv_path = Path(csv_path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"Configuration", "Final_SC", "Survived", "Is_Winner"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV 缺少必要列: {sorted(missing)}，实际列为: {reader.fieldnames}")

        rows: List[dict] = []
        for row in reader:
            rows.append(
                {
                    "Configuration": (row.get("Configuration") or "").strip(),
                    "Final_SC": float((row.get("Final_SC") or "0").strip()),
                    "Survived": _parse_bool(row.get("Survived") or "False"),
                    "Is_Winner": _parse_bool(row.get("Is_Winner") or "False"),
                }
            )
        return rows


def compute_dip_raw(rows: Iterable[dict]) -> Dict[str, List[float]]:
    """输出与 radar_2_chart.py 里 dip_raw 一致的结构：

    dip_raw = {
        'Full_Model': [WinRate, SurvivalRate, AvgSC],
        ...
    }
    """

    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        key = r["Configuration"]
        if not key:
            continue
        grouped[key].append(r)

    dip_raw: Dict[str, List[float]] = {}
    for config, items in grouped.items():
        n = len(items)
        if n == 0:
            continue
        win_rate = sum(1 for x in items if x["Is_Winner"]) / n
        survival_rate = sum(1 for x in items if x["Survived"]) / n
        avg_sc = sum(float(x["Final_SC"]) for x in items) / n
        dip_raw[config] = [win_rate, survival_rate, avg_sc]

    return dip_raw


def compute_dip_summary(rows: Iterable[dict]) -> Dict[str, DipSummary]:
    dip_raw = compute_dip_raw(rows)
    return {
        k: DipSummary(win_rate=v[0], survival_rate=v[1], avg_sc=v[2])
        for k, v in dip_raw.items()
    }


def main() -> None:
    # 默认读取同目录的 RQ4_Ablation.csv
    csv_path = Path(__file__).with_name("RQ4_Ablation.csv")
    rows = load_rq4_ablation_rows(csv_path)
    dip_raw = compute_dip_raw(rows)

    # 为了和你 radar_2_chart.py 里的硬编码更接近，这里对胜率/存活率保留 1 位小数，SC 保留 1 位小数
    pretty: Dict[str, List[float]] = {
        k: [round(v[0], 1), round(v[1], 1), round(v[2], 1)] for k, v in dip_raw.items()
    }

    print("dip_raw =")
    print(json.dumps(pretty, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
