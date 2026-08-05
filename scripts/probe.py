"""Smoke-test every rung on a handful of series and report wall-clock cost.

    python scripts/probe.py [n_series]

Purpose is not results. It is to fail fast on a library API before committing hours to the
full run, and to get a per-series cost estimate so the full run's duration is known in
advance rather than discovered.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from forecast_ladder.protocol import PROTOCOL  # noqa: E402
from forecast_ladder.runner import aggregate, run_seasonal_naive, score  # noqa: E402
from forecast_ladder.rungs import (  # noqa: E402
    rung_classical,
    rung_foundation,
    rung_gbm,
    rung_neural,
)


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    panel = pd.read_parquet(ROOT / "data" / "sample_panel.parquet")
    ids = sorted(panel["unique_id"].unique())[:n]
    sub = panel[panel["unique_id"].isin(ids)].reset_index(drop=True)
    print(f"probe on {sub['unique_id'].nunique()} series, {len(sub):,} rows")
    print(f"protocol: {PROTOCOL.describe()}\n")

    all_fc = []

    # Rung 0: this project's own floor.
    import time

    t0 = time.perf_counter()
    fc0 = run_seasonal_naive(sub, PROTOCOL)
    print(f"{'SeasonalNaive (own)':28s} {time.perf_counter() - t0:8.1f}s  {len(fc0):>7,} rows")
    all_fc.append(fc0)

    for label, fn in [
        ("classical", rung_classical),
        ("gbm", rung_gbm),
        ("neural", rung_neural),
        ("foundation", rung_foundation),
    ]:
        try:
            res = fn(sub, PROTOCOL)
            models = ", ".join(sorted(res.forecasts["model"].unique()))
            print(f"{label:28s} {res.seconds:8.1f}s  {len(res.forecasts):>7,} rows  [{models}]")
            all_fc.append(res.forecasts)
        except Exception:
            print(f"{label:28s}  FAILED")
            traceback.print_exc(limit=6)
            return 1

    fc = pd.concat(all_fc, ignore_index=True)
    per_fold = score(fc, sub, PROTOCOL)
    table = aggregate(per_fold, PROTOCOL)
    print("\n--- probe results (too few series to mean anything, shape check only) ---")
    cols = ["model", "series_folds", "mase_dropped", "mase_mean", "mase_median",
            "pinball_mean", "coverage_mean"]
    print(table[cols].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
