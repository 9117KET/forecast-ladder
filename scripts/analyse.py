"""Score the raw forecasts, derive the finding, cost it, and publish the tables.

    python -u scripts/analyse.py

Reads every `results/raw/*.parquet` written by `run_ladder.py`. Writes to
`results/published/`:

    ladder.csv                one row per model: MASE, pinball, coverage, cost
    per_fold.csv              one row per model, series and fold
    beats_floor.csv           per model and series: did it beat the naive, by how much
    win_rate_by_zero_share.csv    win rate against intermittency quartile
    win_rate_by_volume.csv        win rate against volume quartile
    floor_profile.csv         what the series nothing beats have in common
    economics.csv             newsvendor cost per model
    sensitivity.csv           does the ranking survive the cost assumptions moving
    headline.json             the numbers every write-up in this repo quotes

Nothing in this script chooses a metric or a threshold. Those are fixed in
`forecast_ladder.protocol` and were written before any rung ran.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from forecast_ladder.analysis import (  # noqa: E402
    FLOOR_MODEL,
    beats_floor,
    floor_holds_profile,
    headline_numbers,
    win_rate_by_group,
)
from forecast_ladder.economics import DEFAULT_COSTS, cost_table, sensitivity  # noqa: E402
from forecast_ladder.protocol import PROTOCOL  # noqa: E402
from forecast_ladder.runner import aggregate, score  # noqa: E402

RAW = ROOT / "results" / "raw"
PUB = ROOT / "results" / "published"


def main() -> int:
    PUB.mkdir(parents=True, exist_ok=True)
    files = sorted(RAW.glob("*.parquet"))
    if not files:
        print(f"no raw forecasts in {RAW}: run scripts/run_ladder.py first")
        return 1

    panel = pd.read_parquet(ROOT / "data" / "sample_panel.parquet")
    features = pd.read_csv(PUB / "sample_series.csv")

    fc = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    # Only score series the raw forecasts and the panel agree on, so a dry run on a subset
    # does not silently mix with a full run's output.
    ids = sorted(set(fc["unique_id"]) & set(panel["unique_id"]))
    fc = fc[fc["unique_id"].isin(ids)].reset_index(drop=True)
    panel = panel[panel["unique_id"].isin(ids)].reset_index(drop=True)
    features = features[features["unique_id"].isin(ids)].reset_index(drop=True)

    print(f"scoring {fc['model'].nunique()} models on {len(ids)} series")
    print(f"protocol: {PROTOCOL.describe()}\n", flush=True)

    per_fold = score(fc, panel, PROTOCOL)
    table = aggregate(per_fold, PROTOCOL)

    beats = beats_floor(per_fold, FLOOR_MODEL)
    win_by_model = beats.groupby("model", observed=True)["beats_floor"].mean()
    table["pct_beating_naive_floor"] = table["model"].map(win_by_model * 100.0)

    econ = cost_table(fc, panel, DEFAULT_COSTS)
    table = table.merge(
        econ[["model", "eur_per_series_day_at_cr", "distribution_value_per_series_day"]],
        on="model",
        how="left",
    )

    timings_path = RAW / "timings.json"
    if timings_path.exists():
        timings = json.loads(timings_path.read_text())
        secs = {m: t["seconds"] for t in timings.values() for m in t["models"]}
        # Rungs that produced several models share that rung's cost; recorded as the rung
        # cost rather than divided, because the fits are not separable in wall-clock terms.
        table["rung_seconds"] = table["model"].map(secs)

    print("--- the ladder ---")
    cols = [
        "model", "mase_mean", "mase_median", "pct_beating_naive_floor",
        "pinball_mean", "coverage_mean", "eur_per_series_day_at_cr", "rung_seconds",
    ]
    cols = [c for c in cols if c in table.columns]
    print(table[cols].round(4).to_string(index=False), flush=True)

    print(f"\ncoverage target: {PROTOCOL.nominal_interval:.0%}")
    print(f"mase dropped (undefined scale): {int(table['mase_dropped'].max())} series-folds")

    profile = floor_holds_profile(beats, features)
    print("\n--- where the floor holds, against where it does not ---")
    print(profile.round(3).to_string(index=False), flush=True)

    win_zero = win_rate_by_group(beats, features, "zero_share")
    win_vol = win_rate_by_group(beats, features, "mean_demand")
    print("\n--- win rate against the floor, by intermittency quartile ---")
    print(
        win_zero.pivot(index="bucket", columns="model", values="win_rate")
        .round(3)
        .to_string(),
        flush=True,
    )

    sens = sensitivity(fc, panel)
    best_counts = sens[sens["is_best"]].groupby("model").size()
    print("\n--- cost-assumption sensitivity: times each model is cheapest across the grid ---")
    print(best_counts.to_string(), flush=True)

    head = headline_numbers(per_fold, beats, features, FLOOR_MODEL)
    head["protocol"] = PROTOCOL.describe()
    head["cost_model"] = DEFAULT_COSTS.describe()
    head["cost_model_source_note"] = DEFAULT_COSTS.source_note
    head["cheapest_model_eur_per_series_day"] = float(econ.iloc[0]["eur_per_series_day_at_cr"])
    head["cheapest_model_by_cost"] = str(econ.iloc[0]["model"])
    head["sensitivity_best_counts"] = {str(k): int(v) for k, v in best_counts.items()}

    per_fold.to_csv(PUB / "per_fold.csv", index=False)
    table.to_csv(PUB / "ladder.csv", index=False)
    beats.to_csv(PUB / "beats_floor.csv", index=False)
    win_zero.to_csv(PUB / "win_rate_by_zero_share.csv", index=False)
    win_vol.to_csv(PUB / "win_rate_by_volume.csv", index=False)
    profile.to_csv(PUB / "floor_profile.csv", index=False)
    econ.to_csv(PUB / "economics.csv", index=False)
    sens.to_csv(PUB / "sensitivity.csv", index=False)
    (PUB / "headline.json").write_text(json.dumps(head, indent=2))

    print("\n--- headline ---")
    print(json.dumps(head, indent=2))
    print(f"\nwrote 9 files to {PUB}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
