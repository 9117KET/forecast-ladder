"""Where does the complexity pay, and where does the floor hold.

The results table says which model has the lowest average MASE. That is the least
interesting thing the backtest knows. The useful question is per series: for which items is
a sophisticated method worth running, and what is it about those items that decides it.

Everything here is a table or a comparison of group means. No model is fitted to explain the
results, deliberately: an explanation that needs its own model to be seen is not an
explanation anyone can act on, and the whole point of this section is to produce a rule a
planner could apply.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "FLOOR_MODEL",
    "attach_features",
    "beats_floor",
    "win_rate_by_group",
    "floor_holds_profile",
    "headline_numbers",
]

#: The floor every other rung is measured against. This project's own implementation, not
#: the library's, because it is the one that is tested here.
FLOOR_MODEL = "SeasonalNaive"


def attach_features(per_fold: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Join per-series descriptors onto per-fold scores."""
    keep = [
        c
        for c in ["unique_id", "mean_demand", "zero_share", "adi", "cv2", "n_obs",
                  "volume_stratum", "intermittency_stratum"]
        if c in features.columns
    ]
    return per_fold.merge(features[keep], on="unique_id", how="left", validate="many_to_one")


def beats_floor(per_fold: pd.DataFrame, floor_model: str = FLOOR_MODEL) -> pd.DataFrame:
    """Per (model, series): does it beat the floor, averaged over that series' folds.

    Averaging MASE over a series' folds before comparing, rather than comparing fold by
    fold, is deliberate. The question a planner faces is which method to run for an item,
    not which method to run for an item in one particular month, so the unit of decision is
    the series.
    """
    per_series = (
        per_fold.groupby(["model", "unique_id"], observed=True)["mase"].mean().reset_index()
    )
    floor = per_series[per_series["model"] == floor_model].set_index("unique_id")["mase"]
    if floor.empty:
        raise ValueError(f"floor model {floor_model!r} not present in the scores")

    out = per_series[per_series["model"] != floor_model].copy()
    out["floor_mase"] = out["unique_id"].map(floor)
    out = out.dropna(subset=["mase", "floor_mase"])
    out["beats_floor"] = out["mase"] < out["floor_mase"]
    out["improvement"] = 1.0 - out["mase"] / out["floor_mase"]
    return out.reset_index(drop=True)


def win_rate_by_group(
    beats: pd.DataFrame,
    features: pd.DataFrame,
    group_col: str,
    n_bins: int = 4,
) -> pd.DataFrame:
    """Share of series each model beats the floor on, cut by one series property.

    Continuous properties are cut into quartiles. Quartiles rather than a threshold on
    purpose: a threshold chosen after seeing the data is how a finding turns into a
    coincidence, and a monotone pattern across quartiles is much harder to produce by
    accident than a gap either side of one cut point.
    """
    b = beats.merge(
        features[["unique_id", group_col]], on="unique_id", how="left", validate="many_to_one"
    )
    if pd.api.types.is_numeric_dtype(b[group_col]):
        try:
            b["bucket"] = pd.qcut(b[group_col], q=n_bins, duplicates="drop")
        except ValueError:
            b["bucket"] = b[group_col]
    else:
        b["bucket"] = b[group_col]

    g = (
        b.groupby(["model", "bucket"], observed=True)
        .agg(
            n_series=("unique_id", "nunique"),
            win_rate=("beats_floor", "mean"),
            median_improvement=("improvement", "median"),
        )
        .reset_index()
    )
    return g


def floor_holds_profile(beats: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Compare the series nothing beats against the series something beats.

    "Nothing beats" means every non-floor model on the ladder had a worse average MASE than
    the floor on that series. That is a strong statement and it is the one worth profiling,
    because it identifies items where the correct engineering decision is to run the
    one-line baseline and spend the effort elsewhere.
    """
    per_series_any = (
        beats.groupby("unique_id", observed=True)["beats_floor"].any().rename("any_beats_floor")
    )
    prof = features.merge(per_series_any, on="unique_id", how="inner")
    cols = [c for c in ["mean_demand", "zero_share", "adi", "cv2", "n_obs"] if c in prof.columns]

    rows = []
    for col in cols:
        held = prof.loc[~prof["any_beats_floor"], col].dropna()
        lost = prof.loc[prof["any_beats_floor"], col].dropna()
        rows.append(
            {
                "feature": col,
                "n_floor_holds": int(held.size),
                "n_floor_beaten": int(lost.size),
                "median_where_floor_holds": float(held.median()) if held.size else np.nan,
                "median_where_floor_beaten": float(lost.median()) if lost.size else np.nan,
                "ratio": (
                    float(held.median() / lost.median())
                    if lost.size and float(lost.median()) != 0.0
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def headline_numbers(
    per_fold: pd.DataFrame,
    beats: pd.DataFrame,
    features: pd.DataFrame,
    floor_model: str = FLOOR_MODEL,
) -> dict:
    """The handful of numbers the README, the case study and the interview answer all quote.

    Returned as a dict and written to JSON by the run script, so every document in the
    project cites the same figures from one file instead of each transcribing them.
    """
    n_series = int(per_fold["unique_id"].nunique())
    models = sorted(m for m in per_fold["model"].unique() if m != floor_model)

    per_series_any = beats.groupby("unique_id", observed=True)["beats_floor"].any()
    floor_holds = per_series_any[~per_series_any].index
    n_floor_holds = int(len(floor_holds))

    agg = (
        per_fold[per_fold["mase"].notna()]
        .groupby("model", observed=True)["mase"]
        .agg(["mean", "median"])
        .sort_values("mean")
    )
    best_model = str(agg.index[0])
    floor_mean = float(agg.loc[floor_model, "mean"]) if floor_model in agg.index else float("nan")

    prof = features.set_index("unique_id")
    held_zero = float(prof.loc[floor_holds, "zero_share"].median()) if n_floor_holds else np.nan
    beaten_ids = per_series_any[per_series_any].index
    beaten_zero = (
        float(prof.loc[beaten_ids, "zero_share"].median()) if len(beaten_ids) else np.nan
    )

    return {
        "n_series": n_series,
        "n_models_above_floor": len(models),
        "models": models,
        "floor_model": floor_model,
        "best_model": best_model,
        "best_mase_mean": float(agg.iloc[0]["mean"]),
        "floor_mase_mean": floor_mean,
        "n_series_where_floor_holds": n_floor_holds,
        "pct_series_where_floor_holds": (
            100.0 * n_floor_holds / n_series if n_series else float("nan")
        ),
        "median_zero_share_where_floor_holds": held_zero,
        "median_zero_share_where_floor_beaten": beaten_zero,
        "win_rate_by_model": {
            str(m): float(g["beats_floor"].mean())
            for m, g in beats.groupby("model", observed=True)
        },
    }
