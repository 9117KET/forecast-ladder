"""One scoring path, shared by every rung.

The libraries on the ladder each have their own idea of what a forecast frame looks like.
`statsforecast`, `mlforecast` and `neuralforecast` return a cross-validation frame with the
model name as a column and interval bounds suffixed onto it; Chronos returns raw sample
paths from a tensor. If each rung were scored by the code that produced it, the comparison
would be between four scoring implementations as much as between five methods.

So every rung is normalised into one long frame:

    model, unique_id, cutoff, ds, y, point, q10, q50, q90

and `score()` is the only function in the project that computes a metric on a fold. The
training scale for MASE is recomputed here from the panel, per series and per fold, from
data strictly before the cutoff, no matter which library produced the forecast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .baselines import seasonal_naive, seasonal_naive_quantiles
from .metrics import (
    interval_coverage,
    mae,
    mase,
    pinball_loss,
    seasonal_naive_scale,
    wape,
)
from .protocol import PROTOCOL, Protocol

__all__ = [
    "FORECAST_COLUMNS",
    "fold_cutoff_dates",
    "run_seasonal_naive",
    "normalise_cv_frame",
    "score",
    "aggregate",
]

FORECAST_COLUMNS = ["model", "unique_id", "cutoff", "ds", "point", "q10", "q50", "q90"]


def fold_cutoff_dates(panel: pd.DataFrame, protocol: Protocol = PROTOCOL) -> list[pd.Timestamp]:
    """Cutoff dates for each fold, aligned across all series.

    Every M5 series ends on the same date (they start at different dates because leading
    zeros before an item was stocked are trimmed), so the folds can be defined once by date
    and every series is cut at the same instant. That alignment is what makes a global
    model like LightGBM or a neural network legitimate here: no series is being trained on
    a period in which another is being tested.

    The cutoff is the last date included in training. Returned earliest first, matching
    `statsforecast.cross_validation`, whose folds these must line up with exactly.
    """
    last = pd.Timestamp(panel["ds"].max())
    h, step, w = protocol.horizon, protocol.effective_step, protocol.n_windows
    return [last - pd.Timedelta(days=(w - 1 - i) * step + h) for i in range(w)]


def run_seasonal_naive(
    panel: pd.DataFrame, protocol: Protocol = PROTOCOL, model_name: str = "SeasonalNaive"
) -> pd.DataFrame:
    """Rung 0, run through this project's own implementation rather than a library's.

    The floor decides every MASE in the results, so it is the one method that is not
    delegated. `baselines.py` holds the forecast and its empirical-residual quantiles, and
    `tests/test_baselines.py` checks both against hand-computed values.
    """
    cutoffs = fold_cutoff_dates(panel, protocol)
    m, h = protocol.season_length, protocol.horizon
    q_lo, q_hi = protocol.interval_quantiles
    rows = []

    for uid, g in panel.groupby("unique_id", sort=False, observed=True):
        g = g.sort_values("ds")
        ds_all = g["ds"].to_numpy()
        y_all = g["y"].to_numpy(dtype=float)
        for cutoff in cutoffs:
            train_mask = ds_all <= np.datetime64(cutoff)
            y_train = y_all[train_mask]
            if y_train.size < m + 1:
                continue
            test_mask = (ds_all > np.datetime64(cutoff)) & (
                ds_all <= np.datetime64(cutoff + pd.Timedelta(days=h))
            )
            if test_mask.sum() != h:
                continue
            point = seasonal_naive(y_train, h, m)
            qs = seasonal_naive_quantiles(
                y_train, h, m, quantile_levels=(q_lo, 0.5, q_hi)
            )
            rows.append(
                pd.DataFrame(
                    {
                        "model": model_name,
                        "unique_id": uid,
                        "cutoff": cutoff,
                        "ds": ds_all[test_mask],
                        "point": point,
                        "q10": qs[q_lo],
                        "q50": qs[0.5],
                        "q90": qs[q_hi],
                    }
                )
            )
    if not rows:
        raise RuntimeError("seasonal naive produced no forecasts")
    return pd.concat(rows, ignore_index=True)


def normalise_cv_frame(
    cv: pd.DataFrame,
    model_name: str,
    point_col: str,
    lo_col: str | None = None,
    hi_col: str | None = None,
    median_col: str | None = None,
    clip_at_zero: bool = True,
) -> pd.DataFrame:
    """Reshape a library cross-validation frame into the common format.

    Args:
        cv: frame with at least `unique_id`, `ds`, `cutoff` and the named columns.
        point_col: the model's point forecast column.
        lo_col, hi_col: interval bounds for the protocol's nominal level. When absent, the
            point forecast is repeated at every quantile, which makes the interval
            degenerate and drives coverage toward zero. That is the correct outcome: a
            method that was not asked for a distribution should not be credited with one.
        median_col: use when the library reports a separate median, as neural models with
            quantile losses do. Defaults to the point column.
        clip_at_zero: unit sales cannot be negative. Applied after the library's own
            output so its arithmetic is untouched.
    """
    needed = {"unique_id", "ds", "cutoff", point_col}
    missing = needed - set(cv.columns)
    if missing:
        raise ValueError(f"{model_name}: cross-validation frame missing {sorted(missing)}")

    out = pd.DataFrame(
        {
            "model": model_name,
            "unique_id": cv["unique_id"].to_numpy(),
            "cutoff": pd.to_datetime(cv["cutoff"]).to_numpy(),
            "ds": pd.to_datetime(cv["ds"]).to_numpy(),
            "point": cv[point_col].to_numpy(dtype=float),
        }
    )
    out["q50"] = cv[median_col].to_numpy(dtype=float) if median_col else out["point"]
    out["q10"] = cv[lo_col].to_numpy(dtype=float) if lo_col else out["point"]
    out["q90"] = cv[hi_col].to_numpy(dtype=float) if hi_col else out["point"]

    if clip_at_zero:
        for c in ("point", "q10", "q50", "q90"):
            out[c] = out[c].clip(lower=0.0)
    # An interval whose bounds crossed (which conformal methods can produce on short or
    # degenerate series) is repaired by sorting, not by dropping the fold.
    lo = np.minimum(out["q10"], out["q90"])
    hi = np.maximum(out["q10"], out["q90"])
    out["q10"], out["q90"] = lo, hi
    return out[FORECAST_COLUMNS]


def score(
    forecasts: pd.DataFrame,
    panel: pd.DataFrame,
    protocol: Protocol = PROTOCOL,
) -> pd.DataFrame:
    """Per (model, series, fold) metrics. The only place a metric is computed.

    MASE scales are computed here from the panel, per series and per fold, using only
    observations at or before the cutoff. A rung cannot influence its own denominator.

    Series whose training window has a zero seasonal-naive scale (a flat or all-zero
    history in that fold) yield NaN MASE and are dropped by `aggregate`, which reports how
    many were dropped rather than letting them vanish.
    """
    required = set(FORECAST_COLUMNS)
    missing = required - set(forecasts.columns)
    if missing:
        raise ValueError(f"forecast frame missing {sorted(missing)}")

    panel = panel.copy()
    panel["ds"] = pd.to_datetime(panel["ds"])
    actual = panel.set_index(["unique_id", "ds"])["y"]

    f = forecasts.copy()
    f["ds"] = pd.to_datetime(f["ds"])
    f["cutoff"] = pd.to_datetime(f["cutoff"])
    f["y"] = actual.reindex(pd.MultiIndex.from_arrays([f["unique_id"], f["ds"]])).to_numpy()
    if f["y"].isna().any():
        n = int(f["y"].isna().sum())
        raise ValueError(f"{n} forecast rows have no matching actual in the panel")

    # Training scale per (series, fold), computed once and reused for every model.
    scales: dict[tuple[str, pd.Timestamp], float] = {}
    by_series = {uid: g.sort_values("ds") for uid, g in panel.groupby("unique_id", observed=True)}
    for (uid, cutoff) in f[["unique_id", "cutoff"]].drop_duplicates().itertuples(index=False):
        g = by_series[uid]
        y_train = g.loc[g["ds"] <= cutoff, "y"].to_numpy(dtype=float)
        scales[(uid, cutoff)] = seasonal_naive_scale(y_train, protocol.season_length)

    q_lo, q_hi = protocol.interval_quantiles
    rows = []
    for (model, uid, cutoff), g in f.groupby(["model", "unique_id", "cutoff"], observed=True):
        g = g.sort_values("ds")
        y = g["y"].to_numpy(dtype=float)
        p = g["point"].to_numpy(dtype=float)
        scale = scales[(uid, cutoff)]
        rows.append(
            {
                "model": model,
                "unique_id": uid,
                "cutoff": cutoff,
                "n": int(y.size),
                "scale": scale,
                "mae": mae(y, p),
                "wape": wape(y, p),
                "mase": mase(y, p, scale),
                "pinball": pinball_loss(
                    y,
                    {
                        q_lo: g["q10"].to_numpy(dtype=float),
                        0.5: g["q50"].to_numpy(dtype=float),
                        q_hi: g["q90"].to_numpy(dtype=float),
                    },
                ),
                "coverage": interval_coverage(
                    y, g["q10"].to_numpy(dtype=float), g["q90"].to_numpy(dtype=float)
                ),
                "mean_actual": float(y.mean()),
            }
        )
    return pd.DataFrame(rows)


def aggregate(per_fold: pd.DataFrame, protocol: Protocol = PROTOCOL) -> pd.DataFrame:
    """Collapse per-fold scores to one row per model.

    MASE is averaged over the series-folds where it is defined, and the count of dropped
    series-folds is carried in the output so the table can never imply a wider basis than
    it has. Median MASE is reported next to the mean because the mean of a ratio over
    heterogeneous series is sensitive to a handful of extreme values, and on intermittent
    demand there are always a handful.
    """
    out = []
    for model, g in per_fold.groupby("model", observed=True):
        usable = g["mase"].notna()
        out.append(
            {
                "model": model,
                "series_folds": int(len(g)),
                "mase_defined_on": int(usable.sum()),
                "mase_dropped": int((~usable).sum()),
                "mase_mean": float(g.loc[usable, "mase"].mean()),
                "mase_median": float(g.loc[usable, "mase"].median()),
                "pct_beating_naive_floor": float("nan"),  # filled by compare_to_floor
                "pinball_mean": float(g["pinball"].mean()),
                "coverage_mean": float(g["coverage"].mean()),
                "coverage_target": protocol.nominal_interval,
                "mae_mean": float(g["mae"].mean()),
                "wape_mean": float(g["wape"].mean()),
            }
        )
    return pd.DataFrame(out).sort_values("mase_mean", ignore_index=True)
