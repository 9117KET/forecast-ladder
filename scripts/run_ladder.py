"""Run the rungs and write raw forecasts to disk. Compute only, no scoring.

    python -u scripts/run_ladder.py                      # every rung
    python -u scripts/run_ladder.py --rungs naive,gbm    # a subset
    python -u scripts/run_ladder.py --series 50          # a smaller sample, for a dry run

Each rung writes `results/raw/<rung>.parquet` and appends its wall-clock cost to
`results/raw/timings.json`, so a rung can be re-run on its own without repeating the others
and a crash in one does not lose the rest. Scoring happens in `scripts/analyse.py`, which
reads these files. Keeping the two apart is deliberate: the compute is expensive and the
analysis is not, so the analysis should be re-runnable without paying for the forecasts
again.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from forecast_ladder.protocol import PROTOCOL  # noqa: E402
from forecast_ladder.runner import run_seasonal_naive  # noqa: E402
from forecast_ladder.rungs import (  # noqa: E402
    RungResult,
    rung_classical,
    rung_foundation,
    rung_gbm,
    rung_neural,
)

RAW = ROOT / "results" / "raw"


def _naive(panel, protocol) -> RungResult:
    t0 = time.perf_counter()
    fc = run_seasonal_naive(panel, protocol)
    return RungResult(
        fc,
        time.perf_counter() - t0,
        "This project's own implementation, tested in tests/test_baselines.py. "
        "Intervals from empirical residual quantiles at the matching seasonal lag.",
    )


RUNGS = {
    "naive": _naive,
    "classical": rung_classical,
    "gbm": rung_gbm,
    "neural": rung_neural,
    "foundation": rung_foundation,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rungs", default=",".join(RUNGS))
    ap.add_argument("--series", type=int, default=0, help="0 means the full frozen sample")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(ROOT / "data" / "sample_panel.parquet")
    if args.series:
        ids = sorted(panel["unique_id"].unique())[: args.series]
        panel = panel[panel["unique_id"].isin(ids)].reset_index(drop=True)

    print(f"panel: {panel['unique_id'].nunique()} series, {len(panel):,} rows")
    print(f"protocol: {PROTOCOL.describe()}\n", flush=True)

    timings_path = RAW / "timings.json"
    timings = json.loads(timings_path.read_text()) if timings_path.exists() else {}

    for name in [r.strip() for r in args.rungs.split(",") if r.strip()]:
        if name not in RUNGS:
            print(f"unknown rung {name!r}, expected one of {sorted(RUNGS)}")
            return 2
        print(f"--- {name} ---", flush=True)
        res = RUNGS[name](panel, PROTOCOL)
        out = RAW / f"{name}.parquet"
        res.forecasts.to_parquet(out, index=False)
        models = sorted(res.forecasts["model"].unique())
        timings[name] = {
            "seconds": round(res.seconds, 2),
            "n_series": int(panel["unique_id"].nunique()),
            "models": models,
            "rows": int(len(res.forecasts)),
            "notes": res.notes,
        }
        timings_path.write_text(json.dumps(timings, indent=2))
        print(
            f"{name}: {res.seconds:.1f}s, {len(res.forecasts):,} rows, models {models}",
            flush=True,
        )
        print(f"  wrote {out}\n", flush=True)

    print("done. next: python -u scripts/analyse.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
