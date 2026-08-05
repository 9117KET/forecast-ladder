"""Cache the sampled panel as parquet so later runs skip the 47-million-row load.

    python scripts/prepare_panel.py

Reads the frozen sample from results/published/sample_series.csv, pulls those series out of
the cached M5 panel, and writes data/sample_panel.parquet (gitignored, a few MB).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from forecast_ladder.data import (  # noqa: E402
    HISTORY_DAYS,
    filter_to_sample,
    load_m5,
    trim_to_recent,
)
from forecast_ladder.protocol import PROTOCOL  # noqa: E402


def main() -> int:
    sample_path = ROOT / "results" / "published" / "sample_series.csv"
    if not sample_path.exists():
        print(f"missing {sample_path}: run scripts/download_data.py first")
        return 1

    sample = pd.read_csv(sample_path)
    print(f"sample: {len(sample)} series")

    panel = load_m5(ROOT / "data")
    sub = filter_to_sample(panel, sample)

    print(f"trimming to the most recent {HISTORY_DAYS} days (see data.HISTORY_DAYS)")
    before = len(sub)
    sub = trim_to_recent(sub, HISTORY_DAYS)
    print(f"  {before:,} rows -> {len(sub):,} rows")

    # Every series must be a contiguous daily run: the Nixtla cross-validation helpers and
    # the seasonal lags both assume no gaps, and a silent gap would shift every lag.
    gaps = []
    for uid, g in sub.groupby("unique_id", observed=True):
        d = g["ds"].sort_values()
        if len(d) != (d.max() - d.min()).days + 1:
            gaps.append(uid)
    if gaps:
        print(f"WARNING: {len(gaps)} series have gaps in their daily index, e.g. {gaps[:3]}")
    else:
        print("all series are contiguous daily runs")

    lengths = sub.groupby("unique_id", observed=True).size()
    print(
        f"rows: {len(sub):,}  series: {sub['unique_id'].nunique()}  "
        f"length min/median/max: {lengths.min()}/{int(lengths.median())}/{lengths.max()}"
    )
    print(f"dates: {sub['ds'].min().date()} to {sub['ds'].max().date()}")
    print(f"eligible under protocol (needs {PROTOCOL.min_series_length()}): "
          f"{int((lengths >= PROTOCOL.min_series_length()).sum())}")

    out = ROOT / "data" / "sample_panel.parquet"
    sub.to_parquet(out, index=False)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
