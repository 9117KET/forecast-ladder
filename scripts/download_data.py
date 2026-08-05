"""Fetch M5 and freeze the evaluation sample. Run once, before anything else.

    python scripts/download_data.py

Writes:
    data/                                  the M5 cache, gitignored, roughly 500 MB
    results/published/sample_series.csv    the 300 chosen series and their descriptors
    results/published/series_features.csv  descriptors for every eligible series

The sample file is committed. That is the point of it: anyone can check which series the
published numbers were computed on, and that the selection rule was applied rather than
described.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from forecast_ladder.data import (  # noqa: E402
    SAMPLE_SEED,
    SAMPLE_SIZE,
    load_m5,
    series_features,
    stratified_sample,
)
from forecast_ladder.protocol import PROTOCOL  # noqa: E402


def main() -> int:
    out_dir = ROOT / "results" / "published"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"protocol: {PROTOCOL.describe()}")
    print("fetching M5 (cached in data/ after the first run)")
    panel = load_m5(ROOT / "data")
    print(
        f"  panel: {panel['unique_id'].nunique():,} series, "
        f"{len(panel):,} rows, {panel['ds'].min().date()} to {panel['ds'].max().date()}"
    )

    min_len = PROTOCOL.min_series_length()
    print(f"describing series and dropping any shorter than {min_len} observations")
    feats = series_features(panel, min_length=min_len)
    print(f"  eligible on length: {len(feats):,} series")

    sample = stratified_sample(feats, n=SAMPLE_SIZE, seed=SAMPLE_SEED)
    print(f"  sampled: {len(sample)} series, seed {SAMPLE_SEED}")

    counts = sample.groupby(["volume_stratum", "intermittency_stratum"], observed=True).size()
    print("\nsample composition (volume x intermittency):")
    print(counts.to_string())

    print("\nsample descriptors:")
    desc = sample[["mean_demand", "zero_share", "adi", "cv2"]].describe().round(3)
    print(desc.to_string())

    feats.to_csv(out_dir / "series_features.csv", index=False)
    sample.to_csv(out_dir / "sample_series.csv", index=False)
    print(f"\nwrote {out_dir / 'sample_series.csv'}")
    print(f"wrote {out_dir / 'series_features.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
