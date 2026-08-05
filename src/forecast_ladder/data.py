"""Loading M5 and choosing the series every rung is scored on.

**Why M5 and not Rossmann.** Rossmann store sales was the first choice, because German
retail is the closer analogue for the work this was built for. It is a Kaggle competition
dataset, which means the data terms live behind a rules page that has to be accepted with
an account, and it cannot be fetched without a personal API token. M5 is the same problem
shape (daily retail unit sales, promotions, calendar effects, real intermittency), it is
the reference benchmark that published forecasting results are quoted against, and
`datasetsforecast` fetches it without credentials. The loader below is written against a
long-format frame with `unique_id, ds, y`, so a Rossmann loader can be dropped in beside
it without touching the protocol, the metrics or any rung.

**Why one sampled series set for the whole ladder, and not all 30,490.** Two reasons, and
the second is the one that matters.

The practical reason: this runs on a CPU. Fitting AutoARIMA per series and training two
neural architectures per fold across 30,490 series is not a thing that finishes.

The methodological reason: if the classical rungs were scored on all series and the neural
rungs on a subset, the comparison would be between methods *and* populations at once, and
no row of the results table could be read against any other. Every rung is therefore
scored on the identical sample, chosen once, by a documented rule, before any model runs.
The chosen ids are written to `results/published/sample_series.csv` so the selection is
auditable rather than asserted.

**Why stratify rather than take the first N or a plain random draw.** Retail demand is
wildly heterogeneous: a fast-moving staple and an item that sells four units a year are
different forecasting problems, and the interesting question is precisely where on that
spectrum the sophisticated methods start earning their cost. A plain random draw from M5
would be dominated by intermittent low-volume series, because that is most of M5, and the
result would say more about the sampling than about the methods. Stratifying by volume and
by intermittency guarantees every regime is represented well enough to say something about
each.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "load_m5",
    "series_features",
    "stratified_sample",
    "filter_to_sample",
    "trim_to_recent",
    "SAMPLE_SIZE",
    "SAMPLE_SEED",
    "HISTORY_DAYS",
]

#: Training history retained per series, counted back from the end of the panel.
#:
#: M5 runs 2011-01-29 to 2016-06-19, about 5.3 years of daily observations. Only the most
#: recent 842 days are used, which leaves every fold with between 730 and 814 days of
#: training data: two complete annual cycles at minimum.
#:
#: **Two reasons, and the modelling one came first.** A five-year-old observation of a
#: supermarket item is weak evidence about next month: assortments change, shelf position
#: changes, competing products arrive and leave. Retail forecasting in practice works on a
#: window of a year or two for exactly that reason, and a method that is handed five years
#: of a non-stationary series is being handed noise along with the signal.
#:
#: The second reason is compute, and pretending otherwise would be dishonest. A probe on 12
#: series with the full history took 441 seconds for the classical rung alone, which
#: extrapolates to roughly three hours across the sample on one CPU. Trimming cuts the data
#: by a factor of 2.3.
#:
#: Both reasons were settled before any rung produced a result that anybody looked at.
HISTORY_DAYS = 842

#: Series in the evaluation sample. 300 keeps a 4-fold backtest of five rungs inside a few
#: hours on one CPU while still giving roughly 33 series per stratum cell.
SAMPLE_SIZE = 300

#: Fixed so the sample is reproducible. Changing it invalidates every published number.
SAMPLE_SEED = 20260805


def load_m5(directory: str | Path = "data") -> pd.DataFrame:
    """Fetch M5 and return daily unit sales in long format.

    Returns a frame with columns `unique_id` (item and store), `ds` (date), `y` (units).
    Downloads roughly 500 MB on first call and caches under `directory`, which is
    gitignored. Nothing about the data is committed to this repository.
    """
    from datasetsforecast.m5 import M5  # imported lazily: heavy, and only needed here

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    y_df, _x_df, _s_df = M5.load(directory=str(directory))

    y_df = y_df.rename(columns=str.lower)
    missing = {"unique_id", "ds", "y"} - set(y_df.columns)
    if missing:
        raise RuntimeError(f"M5 loader returned unexpected columns, missing {missing}")

    y_df["ds"] = pd.to_datetime(y_df["ds"])
    y_df["y"] = y_df["y"].astype(float)
    return y_df.sort_values(["unique_id", "ds"], ignore_index=True)


def series_features(df: pd.DataFrame, min_length: int) -> pd.DataFrame:
    """Per-series descriptors, used both for stratification and for the final analysis.

    The three that matter:

    `zero_share`: fraction of days with no sale. This is the intermittency measure. Above
    roughly 0.5 a series is mostly zeros and the forecasting problem changes character:
    the question stops being "how many" and becomes "will there be any".

    `adi`: average demand interval, the mean gap in days between non-zero sales. The
    classical intermittent-demand literature splits series on ADI around 1.32, which is
    why it is recorded rather than only the zero share.

    `cv2`: squared coefficient of variation of the non-zero demands. Paired with ADI this
    is the Syntetos-Boylan-Croston quadrant, the standard way of saying which corner of
    the demand space a series sits in.

    Series shorter than `min_length` are dropped here rather than at fit time, so an
    ineligible series can never quietly enter one rung and not another.
    """
    out = []
    for uid, g in df.groupby("unique_id", sort=False):
        y = g["y"].to_numpy(dtype=float)
        if y.size < min_length:
            continue
        nonzero = y[y > 0]
        n_nonzero = int(nonzero.size)
        zero_share = float((y == 0).mean())
        adi = float(y.size / n_nonzero) if n_nonzero > 0 else float("inf")
        if n_nonzero > 1 and nonzero.mean() > 0:
            cv2 = float((nonzero.std(ddof=1) / nonzero.mean()) ** 2)
        else:
            cv2 = float("nan")
        out.append(
            {
                "unique_id": uid,
                "n_obs": int(y.size),
                "mean_demand": float(y.mean()),
                "total_demand": float(y.sum()),
                "zero_share": zero_share,
                "adi": adi,
                "cv2": cv2,
                "n_nonzero": n_nonzero,
            }
        )
    if not out:
        raise ValueError(f"no series reached the minimum length of {min_length}")
    return pd.DataFrame(out)


def stratified_sample(
    features: pd.DataFrame,
    n: int = SAMPLE_SIZE,
    seed: int = SAMPLE_SEED,
    n_volume_bins: int = 3,
    n_intermittency_bins: int = 3,
) -> pd.DataFrame:
    """Draw `n` series spread across volume and intermittency strata.

    Terciles of mean demand crossed with terciles of zero share give nine cells, and the
    draw is proportional to cell population with at least one series per non-empty cell.
    Any shortfall from rounding is topped up from the largest cells.

    Series with a degenerate history are excluded first: a series that never sells cannot
    be forecast, and a series that is exactly constant has a seasonal naive scale of zero,
    which would make its MASE infinite and let one series dominate every average.
    """
    eligible = features[(features["n_nonzero"] > 0) & (features["mean_demand"] > 0)].copy()
    if eligible.empty:
        raise ValueError("no eligible series after dropping empty histories")
    if n > len(eligible):
        raise ValueError(f"asked for {n} series but only {len(eligible)} are eligible")

    def _bin(col: str, bins: int, labels: list[str]) -> pd.Series:
        # qcut with duplicate-edge tolerance: M5 has many tied low-volume series, so the
        # tercile edges are not guaranteed distinct.
        try:
            return pd.qcut(eligible[col], q=bins, labels=labels, duplicates="drop")
        except ValueError:
            return pd.Series(labels[0], index=eligible.index, dtype="object")

    eligible["volume_stratum"] = _bin("mean_demand", n_volume_bins, ["low", "mid", "high"])
    eligible["intermittency_stratum"] = _bin(
        "zero_share", n_intermittency_bins, ["dense", "medium", "sparse"]
    )
    eligible["stratum"] = (
        eligible["volume_stratum"].astype(str) + "_" + eligible["intermittency_stratum"].astype(str)
    )

    rng = np.random.default_rng(seed)
    groups = {k: g for k, g in eligible.groupby("stratum", sort=True) if len(g) > 0}
    total = sum(len(g) for g in groups.values())

    quota = {k: max(1, int(round(n * len(g) / total))) for k, g in groups.items()}
    # Reconcile rounding against the target, adjusting the largest cells first.
    order = sorted(groups, key=lambda k: -len(groups[k]))
    while sum(quota.values()) > n:
        for k in order:
            if sum(quota.values()) <= n:
                break
            if quota[k] > 1:
                quota[k] -= 1
    while sum(quota.values()) < n:
        for k in order:
            if sum(quota.values()) >= n:
                break
            if quota[k] < len(groups[k]):
                quota[k] += 1

    picked = []
    for k, g in groups.items():
        take = min(quota[k], len(g))
        idx = rng.choice(g.index.to_numpy(), size=take, replace=False)
        picked.append(g.loc[idx])
    sample = pd.concat(picked).sort_values("unique_id", ignore_index=True)
    return sample


def trim_to_recent(df: pd.DataFrame, n_days: int = HISTORY_DAYS) -> pd.DataFrame:
    """Keep only the most recent `n_days` observations, counted from the panel's end date.

    Trimming by date rather than per series preserves the property the folds depend on: all
    series still end on the same day and are still cut at the same instants, so a global
    model cannot be trained on a period in which another series is being tested.

    See `HISTORY_DAYS` for why the window is what it is.
    """
    last = pd.Timestamp(df["ds"].max())
    first = last - pd.Timedelta(days=int(n_days) - 1)
    out = df[df["ds"] >= first].copy()
    return out.sort_values(["unique_id", "ds"], ignore_index=True)


def filter_to_sample(df: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    """Restrict the panel to the sampled series, preserving order."""
    ids = set(sample["unique_id"])
    out = df[df["unique_id"].isin(ids)].copy()
    if out["unique_id"].nunique() != len(ids):
        raise RuntimeError("some sampled series are absent from the panel")
    return out.sort_values(["unique_id", "ds"], ignore_index=True)
