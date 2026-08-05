"""Time one rung on a subset, to size the full run before committing to it.

    python -u scripts/time_rung.py <classical|gbm|neural|foundation> [n_series]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from forecast_ladder.protocol import PROTOCOL  # noqa: E402
from forecast_ladder.rungs import (  # noqa: E402
    rung_classical,
    rung_foundation,
    rung_gbm,
    rung_neural,
)

RUNGS = {
    "classical": rung_classical,
    "gbm": rung_gbm,
    "neural": rung_neural,
    "foundation": rung_foundation,
}


def main() -> int:
    name = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    panel = pd.read_parquet(ROOT / "data" / "sample_panel.parquet")
    total_series = panel["unique_id"].nunique()
    ids = sorted(panel["unique_id"].unique())[:n]
    sub = panel[panel["unique_id"].isin(ids)].reset_index(drop=True)

    print(f"rung={name}  series={n}  rows={len(sub):,}", flush=True)
    res = RUNGS[name](sub, PROTOCOL)
    per_series = res.seconds / n
    print(
        f"\n{name}: {res.seconds:.1f}s for {n} series ({per_series:.2f}s/series)\n"
        f"  models: {sorted(res.forecasts['model'].unique())}\n"
        f"  rows:   {len(res.forecasts):,}\n"
        f"  projected for {total_series} series: "
        f"{per_series * total_series / 60:.1f} min\n"
        f"  note: {res.notes}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
