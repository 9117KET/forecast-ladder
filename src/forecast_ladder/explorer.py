"""Loading the published run so it can be explored rather than only read.

Everything in this repository up to here is batch: scripts write tables, a human reads
them. This module is the layer that lets somebody interrogate the same run interactively,
and `app.py` is a thin Streamlit shell over it.

It is a separate module from `app.py`, and not inline in it, for one reason: a Streamlit
script cannot be imported without starting a server, so anything living inside one is
untestable by construction. Every function here is a plain pandas function with a return
value, tested in `tests/test_explorer.py`, and the UI holds no logic of its own.

**What it reads, and why that is enough.** `results/raw/*.parquet` holds the forecast every
rung produced, per series, fold and day, and `data/sample_panel.parquet` holds the actuals
they are scored against. Both are committed, so a fresh clone can re-derive every published
number without the 500 MB download or the two hours of compute that produced them. Nothing
here recomputes a forecast; the expensive half of the project stays where it was.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import FLOOR_MODEL, beats_floor
from .protocol import PROTOCOL, Protocol
from .runner import aggregate, score

__all__ = [
    "repo_root",
    "DataUnavailable",
    "load_panel",
    "load_sample_series",
    "load_published",
    "load_headline",
    "load_timings",
    "load_raw_forecasts",
    "available_models",
    "series_catalogue",
    "series_history",
    "series_fold_view",
    "per_series_scores",
    "run_floor_backtest",
    "model_display_order",
]


class DataUnavailable(RuntimeError):
    """A required committed artefact is missing, with the command that regenerates it."""


def repo_root() -> Path:
    """The repository root, resolved from this file rather than the working directory.

    Streamlit Community Cloud runs the app from the repo root but a local `streamlit run`
    can be launched from anywhere, so nothing here may depend on the process CWD.
    """
    return Path(__file__).resolve().parents[2]


def _require(path: Path, how: str) -> Path:
    if not path.exists():
        raise DataUnavailable(f"missing {path}\n  regenerate with: {how}")
    return path


@lru_cache(maxsize=1)
def load_panel() -> pd.DataFrame:
    """The 300-series actuals panel: `unique_id`, `ds`, `y`."""
    path = _require(
        repo_root() / "data" / "sample_panel.parquet",
        "python scripts/download_data.py && python scripts/prepare_panel.py",
    )
    df = pd.read_parquet(path)
    df["ds"] = pd.to_datetime(df["ds"])
    return df.sort_values(["unique_id", "ds"], ignore_index=True)


@lru_cache(maxsize=1)
def load_sample_series() -> pd.DataFrame:
    """Per-series descriptors for the frozen sample, one row per series."""
    path = _require(
        repo_root() / "results" / "published" / "sample_series.csv",
        "python scripts/download_data.py",
    )
    return pd.read_csv(path)


@lru_cache(maxsize=None)
def load_published(name: str) -> pd.DataFrame:
    """One table from `results/published/`, by stem (`ladder`, `per_fold`, ...)."""
    path = _require(
        repo_root() / "results" / "published" / f"{name}.csv",
        "python scripts/analyse.py",
    )
    df = pd.read_csv(path)
    if "cutoff" in df.columns:
        df["cutoff"] = pd.to_datetime(df["cutoff"])
    return df


@lru_cache(maxsize=1)
def load_headline() -> dict:
    """`headline.json`: the numbers every write-up in the repo quotes."""
    path = _require(
        repo_root() / "results" / "published" / "headline.json",
        "python scripts/analyse.py",
    )
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_timings() -> dict:
    """Wall-clock cost per rung, or an empty dict when the raw run is absent."""
    path = repo_root() / "results" / "raw" / "timings.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_raw_forecasts() -> pd.DataFrame:
    """Every rung's forecast, concatenated: the input `scripts/analyse.py` scores.

    This is what makes the app more than a picture of a CSV. With the raw forecasts in
    hand, a cost assumption can be moved and the whole comparison re-costed live, because
    the newsvendor cost is a function of the forecast quantiles and the actuals, not of
    anything the models would have to be re-run to produce.
    """
    raw = repo_root() / "results" / "raw"
    files = sorted(raw.glob("*.parquet"))
    if not files:
        raise DataUnavailable(
            "no forecasts in results/raw/\n  regenerate with: python scripts/run_ladder.py"
        )
    fc = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    fc["ds"] = pd.to_datetime(fc["ds"])
    fc["cutoff"] = pd.to_datetime(fc["cutoff"])
    return fc


def model_display_order(models) -> list[str]:
    """Models ranked by published mean MASE, with any unknown model appended alphabetically.

    Used so every chart and table in the app lists models in the same order, and that order
    is the finding rather than whatever order a groupby happened to produce.
    """
    models = list(dict.fromkeys(models))
    try:
        ladder = load_published("ladder")
        rank = {m: i for i, m in enumerate(ladder.sort_values("mase_mean")["model"])}
    except DataUnavailable:
        rank = {}
    known = sorted([m for m in models if m in rank], key=lambda m: rank[m])
    return known + sorted(m for m in models if m not in rank)


def available_models() -> list[str]:
    """Model names present in the raw forecasts, in published-MASE order."""
    return model_display_order(load_raw_forecasts()["model"].unique())


def series_catalogue() -> pd.DataFrame:
    """One row per sampled series: descriptors, plus which model actually won it.

    `best_model` is the model with the lowest mean MASE across that series' folds, and
    `floor_holds` marks the series no rung beat the seasonal naive on. Those two columns
    are the reason to browse series at all: the ladder table says the neural rungs win on
    average, and this says on which items that average is made and on which it is not.
    """
    features = load_sample_series().copy()
    per_fold = load_published("per_fold")

    per_series = (
        per_fold.groupby(["unique_id", "model"], observed=True)["mase"].mean().reset_index()
    )
    ranked = per_series.dropna(subset=["mase"]).sort_values("mase")
    best = ranked.groupby("unique_id", observed=True).first()
    features["best_model"] = features["unique_id"].map(best["model"])
    features["best_mase"] = features["unique_id"].map(best["mase"])

    floor = per_series[per_series["model"] == FLOOR_MODEL].set_index("unique_id")["mase"]
    features["floor_mase"] = features["unique_id"].map(floor)

    beaten = beats_floor(per_fold, FLOOR_MODEL).groupby("unique_id")["beats_floor"].any()
    features["floor_holds"] = ~features["unique_id"].map(beaten).fillna(False)
    return features


def series_history(unique_id: str) -> pd.DataFrame:
    """The actuals for one series: `ds`, `y`."""
    panel = load_panel()
    g = panel[panel["unique_id"] == unique_id]
    if g.empty:
        raise KeyError(f"series {unique_id!r} is not in the sample panel")
    return g[["ds", "y"]].reset_index(drop=True)


def series_fold_view(
    unique_id: str,
    cutoff,
    models: list[str] | None = None,
    context_days: int = 84,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Actuals and forecasts around one fold of one series, ready to plot.

    Returns `(context, forecasts)`. `context` is the actuals from `context_days` before the
    cutoff through the end of the test window, so the plot shows what the model had to work
    with and not only what it produced. `forecasts` is long-format, one row per model and
    day, carrying the point forecast and the 80 percent interval bounds.
    """
    cutoff = pd.Timestamp(cutoff)
    fc = load_raw_forecasts()
    sel = fc[(fc["unique_id"] == unique_id) & (fc["cutoff"] == cutoff)]
    if models is not None:
        sel = sel[sel["model"].isin(list(models))]
    sel = sel.sort_values(["model", "ds"]).reset_index(drop=True)

    hist = series_history(unique_id)
    start = cutoff - pd.Timedelta(days=int(context_days))
    end = sel["ds"].max() if not sel.empty else cutoff + pd.Timedelta(days=PROTOCOL.horizon)
    context = hist[(hist["ds"] >= start) & (hist["ds"] <= end)].copy()
    context["period"] = np.where(context["ds"] <= cutoff, "train", "test")
    return context.reset_index(drop=True), sel


def per_series_scores(unique_id: str) -> pd.DataFrame:
    """Published per-fold scores for one series, one row per model and fold."""
    per_fold = load_published("per_fold")
    out = per_fold[per_fold["unique_id"] == unique_id].copy()
    if out.empty:
        raise KeyError(f"series {unique_id!r} has no published scores")
    order = model_display_order(out["model"].unique())
    out["model"] = pd.Categorical(out["model"], categories=order, ordered=True)
    return out.sort_values(["model", "cutoff"], ignore_index=True)


def run_floor_backtest(
    unique_ids: list[str],
    protocol: Protocol = PROTOCOL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run and score the seasonal naive floor live, under any protocol.

    This is the one thing in the app that computes rather than reads. It exists because the
    protocol is the project's central claim, and a claim you can only read is weaker than
    one you can move: change the horizon or the number of folds here and the floor is
    re-run and re-scored end to end, through the same `runner.score` every published number
    came from.

    Only the floor is re-runnable, and deliberately so. The other rungs need torch,
    LightGBM and a pretrained transformer, and shipping those into a web app to re-fit a
    model on demand would trade the honest part of this project for a demo. Returns
    `(per_fold, summary)`.

    Raises:
        ValueError: if the series are too short for the requested protocol. That is the
            protocol's own eligibility rule surfacing, not an error to work around.
    """
    from .runner import run_seasonal_naive

    unique_ids = list(unique_ids)
    if not unique_ids:
        raise ValueError("select at least one series")
    panel = load_panel()
    sub = panel[panel["unique_id"].isin(unique_ids)].reset_index(drop=True)
    if sub.empty:
        raise ValueError("none of the requested series are in the sample panel")

    needed = protocol.total_test_span() + protocol.season_length + 1
    shortest = int(sub.groupby("unique_id", observed=True).size().min())
    if shortest < needed:
        raise ValueError(
            f"protocol needs at least {needed} observations per series "
            f"({protocol.n_windows} folds x {protocol.horizon}d plus a season of history), "
            f"but the shortest selected series has {shortest}"
        )

    fc = run_seasonal_naive(sub, protocol)
    per_fold = score(fc, sub, protocol)
    summary = aggregate(per_fold, protocol)
    return per_fold, summary
