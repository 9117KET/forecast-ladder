"""The five rungs. Each one returns forecasts in the common format and nothing else.

No rung computes a metric. No rung sees the protocol's evaluation code. Each is handed the
panel and the frozen protocol and must produce
`model, unique_id, cutoff, ds, point, q10, q50, q90`, which `runner.score` then judges.

**On fairness between rungs.** Two things are held constant and both matter.

Every rung is cut at the same dates, because `cross_validation` in all three Nixtla
libraries defines folds the way `runner.fold_cutoff_dates` does, and the Chronos loop is
written against the same function. So no method ever sees a period another is being tested
on.

Every rung is asked for the same thing: a point forecast and an 80 percent interval. Where
a method has no native notion of a predictive distribution, it gets conformal intervals
(LightGBM) or empirical-residual intervals (the naive), rather than being scored on an
interval it never produced. What is deliberately *not* equalised is tuning effort, and that
asymmetry favours the complex end of the ladder: the neural models get a fixed step budget
and default architectures, the gradient-boosted model gets hand-chosen features, and the
classical models get whatever `AutoETS` and `AutoARIMA` select on their own. If a simple
method still wins under those conditions, the finding is not an artefact of neglect.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .protocol import PROTOCOL, Protocol
from .runner import fold_cutoff_dates, normalise_cv_frame

__all__ = [
    "RungResult",
    "rung_classical",
    "rung_gbm",
    "rung_neural",
    "rung_foundation",
]


@dataclass
class RungResult:
    """Forecasts plus the wall-clock cost of producing them.

    The seconds are part of the answer, not bookkeeping. "Does the complexity pay" is a
    question about cost as well as accuracy, and a method that needs two orders of
    magnitude more compute to draw level with a one-line baseline has answered it.
    """

    forecasts: pd.DataFrame
    seconds: float
    notes: str = ""


def _cv_kwargs(protocol: Protocol) -> dict:
    return {
        "h": protocol.horizon,
        "step_size": protocol.effective_step,
        "n_windows": protocol.n_windows,
    }


def rung_classical(panel: pd.DataFrame, protocol: Protocol = PROTOCOL) -> RungResult:
    """Rungs 0b and 1: the library seasonal naive as a cross-check, then AutoETS and AutoARIMA.

    The library `SeasonalNaive` is included on purpose even though `baselines.py` already
    implements one. If this project's own floor and a well-used library's floor disagree on
    point accuracy, the bug is in this project, and it is better to find that out from a
    results table than not at all. Their intervals will differ, because the library assumes
    a distribution and this project uses empirical residuals.
    """
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA, AutoETS, SeasonalNaive

    m = protocol.season_length
    level = [int(round(protocol.nominal_interval * 100))]

    # AutoARIMA's search space is bounded, and this is a real constraint worth naming.
    # Unbounded, it took 441 seconds on 12 series in a probe, which is roughly three hours
    # across the sample: a stepwise search over seasonal orders with full maximum-likelihood
    # estimation, repeated per series per fold. The bounds below (first differences only,
    # at most two non-seasonal and one seasonal term each side, conditional-sum-of-squares
    # approximation for the search) are the standard way to make seasonal ARIMA tractable on
    # daily data, and they are wide enough to contain the orders that are actually selected
    # on weekly-seasonal retail series. It remains a bound, and if AutoARIMA loses, part of
    # the reason may be that it was not allowed to look further.
    arima = AutoARIMA(
        season_length=m,
        max_p=2,
        max_q=2,
        max_P=1,
        max_Q=1,
        max_d=1,
        max_D=1,
        approximation=True,
        stepwise=True,
        nmodels=20,
    )
    sf = StatsForecast(
        models=[
            SeasonalNaive(season_length=m),
            AutoETS(season_length=m),
            arima,
        ],
        freq="D",
        n_jobs=-1,
    )
    t0 = time.perf_counter()
    cv = sf.cross_validation(df=panel, level=level, **_cv_kwargs(protocol))
    seconds = time.perf_counter() - t0

    cv = cv.reset_index() if cv.index.name == "unique_id" else cv
    lo, hi = f"-lo-{level[0]}", f"-hi-{level[0]}"
    frames = []
    for name, label in [
        ("SeasonalNaive", "SeasonalNaive (library)"),
        ("AutoETS", "AutoETS"),
        ("AutoARIMA", "AutoARIMA"),
    ]:
        frames.append(
            normalise_cv_frame(
                cv,
                model_name=label,
                point_col=name,
                lo_col=f"{name}{lo}" if f"{name}{lo}" in cv.columns else None,
                hi_col=f"{name}{hi}" if f"{name}{hi}" in cv.columns else None,
            )
        )
    return RungResult(
        pd.concat(frames, ignore_index=True),
        seconds,
        "AutoETS and AutoARIMA select their own orders per series; intervals are the "
        "library's parametric ones.",
    )


def rung_gbm(panel: pd.DataFrame, protocol: Protocol = PROTOCOL) -> RungResult:
    """Rung 2: one global LightGBM over lag and calendar features.

    This is the rung that wins most published retail comparisons, and the M5 competition
    was won with a version of it, so it is the real benchmark rather than the neural tier.

    Three deliberate choices:

    A *global* model, one LightGBM fitted across all series, not one per series. That is
    both the industry pattern and the only thing that makes gradient boosting competitive
    on short intermittent series, because a single series of 1,900 mostly-zero days does
    not contain enough signal to fit a tree ensemble.

    Lags at 1, 7, 14, 21 and 28 days plus rolling means. The weekly multiples are the
    whole point: the seasonality that ETS models explicitly has to be handed to a tree as
    a feature, because a tree has no notion of a period.

    Conformal intervals rather than quantile regression, so the interval is calibrated on
    held-out residuals instead of assuming a shape.
    """
    import lightgbm as lgb
    from mlforecast import MLForecast
    from mlforecast.utils import PredictionIntervals
    from utilsforecast.feature_engineering import time_features  # noqa: F401  (import check)

    level = [int(round(protocol.nominal_interval * 100))]
    model = lgb.LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=40,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        verbosity=-1,
        random_state=protocol.horizon,  # any fixed value; kept deterministic
        n_jobs=-1,
    )

    fcst = MLForecast(
        models={"LightGBM": model},
        freq="D",
        lags=[1, 7, 14, 21, 28],
        date_features=["dayofweek", "day", "month", "week"],
    )
    t0 = time.perf_counter()
    cv = fcst.cross_validation(
        df=panel,
        level=level,
        prediction_intervals=PredictionIntervals(n_windows=2, h=protocol.horizon),
        **_cv_kwargs(protocol),
    )
    seconds = time.perf_counter() - t0

    lo, hi = f"LightGBM-lo-{level[0]}", f"LightGBM-hi-{level[0]}"
    out = normalise_cv_frame(
        cv,
        model_name="LightGBM",
        point_col="LightGBM",
        lo_col=lo if lo in cv.columns else None,
        hi_col=hi if hi in cv.columns else None,
    )
    return RungResult(
        out,
        seconds,
        "One global model across all series; lags 1/7/14/21/28 plus calendar features; "
        "conformal prediction intervals.",
    )


def rung_neural(
    panel: pd.DataFrame,
    protocol: Protocol = PROTOCOL,
    max_steps: int = 400,
) -> RungResult:
    """Rung 3: N-HiTS and PatchTST, trained with a quantile loss.

    Both are global models trained across the sample, both are given the same input window
    (four horizons of history) and the same step budget, and both are trained with a
    multi-quantile loss so the 80 percent interval comes out of the model's own objective
    rather than being bolted on afterwards.

    `max_steps` is the honest constraint: this runs on a CPU, and the budget is fixed in
    advance rather than raised until the models win. It is recorded in the results so a
    reader can see what the neural tier was and was not given.
    """
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import MQLoss
    from neuralforecast.models import NHITS, PatchTST

    h = protocol.horizon
    level = [int(round(protocol.nominal_interval * 100))]
    input_size = 4 * h

    common = dict(
        h=h,
        input_size=input_size,
        loss=MQLoss(level=level),
        max_steps=max_steps,
        val_check_steps=max_steps,  # no early stopping: the budget is the budget
        enable_progress_bar=False,
        logger=False,
        accelerator="cpu",
        random_seed=1,
    )
    models = [
        NHITS(**common, scaler_type="robust"),
        PatchTST(**common, scaler_type="robust", patch_len=16, stride=8),
    ]
    nf = NeuralForecast(models=models, freq="D")

    t0 = time.perf_counter()
    cv = nf.cross_validation(
        df=panel,
        n_windows=protocol.n_windows,
        step_size=protocol.effective_step,
    )
    seconds = time.perf_counter() - t0
    cv = cv.reset_index() if cv.index.name == "unique_id" else cv

    frames = []
    for name in ("NHITS", "PatchTST"):
        median = f"{name}-median"
        lo, hi = f"{name}-lo-{level[0]}", f"{name}-hi-{level[0]}"
        point_col = median if median in cv.columns else name
        frames.append(
            normalise_cv_frame(
                cv,
                model_name=name,
                point_col=point_col,
                lo_col=lo if lo in cv.columns else None,
                hi_col=hi if hi in cv.columns else None,
                median_col=median if median in cv.columns else None,
            )
        )
    return RungResult(
        pd.concat(frames, ignore_index=True),
        seconds,
        f"Global models, input window {input_size}d, multi-quantile loss, "
        f"fixed budget of {max_steps} steps on CPU, no early stopping.",
    )


def rung_foundation(
    panel: pd.DataFrame,
    protocol: Protocol = PROTOCOL,
    model_id: str = "amazon/chronos-bolt-small",
    context_length: int = 512,
    batch_size: int = 32,
) -> RungResult:
    """Rung 4: Chronos, zero-shot. No training on this data at all.

    This is the rung worth being clear about, because "zero-shot forecasting" sounds like a
    contradiction. Chronos is a transformer pretrained on a large corpus of time series
    from many domains. At inference it is handed the history of a series it has never seen
    and emits a forecast directly, with no fitting step: the pattern-matching that a
    classical model performs by estimating parameters per series, Chronos performs in its
    forward pass using what it learned from other series. Chronos-Bolt in particular
    predicts a set of quantiles directly, so the interval comes from the model rather than
    from sampling.

    The loop is written by hand rather than through a wrapper so that the context handed to
    the model is provably the training window and nothing after it.
    """
    import torch
    from chronos import BaseChronosPipeline

    h, m = protocol.horizon, protocol.season_length
    q_lo, q_hi = protocol.interval_quantiles
    cutoffs = fold_cutoff_dates(panel, protocol)

    pipe = BaseChronosPipeline.from_pretrained(
        model_id, device_map="cpu", torch_dtype=torch.float32
    )

    series = {uid: g.sort_values("ds") for uid, g in panel.groupby("unique_id", observed=True)}

    jobs: list[tuple[str, pd.Timestamp, np.ndarray, np.ndarray]] = []
    for uid, g in series.items():
        ds_all = g["ds"].to_numpy()
        y_all = g["y"].to_numpy(dtype=float)
        for cutoff in cutoffs:
            train_mask = ds_all <= np.datetime64(cutoff)
            y_train = y_all[train_mask][-context_length:]
            if y_train.size < m + 1:
                continue
            test_mask = (ds_all > np.datetime64(cutoff)) & (
                ds_all <= np.datetime64(cutoff + pd.Timedelta(days=h))
            )
            if test_mask.sum() != h:
                continue
            jobs.append((uid, cutoff, y_train, ds_all[test_mask]))

    t0 = time.perf_counter()
    rows = []
    for start in range(0, len(jobs), batch_size):
        chunk = jobs[start : start + batch_size]
        contexts = [torch.tensor(y, dtype=torch.float32) for _, _, y, _ in chunk]
        quantiles, _mean = pipe.predict_quantiles(
            context=contexts,
            prediction_length=h,
            quantile_levels=[q_lo, 0.5, q_hi],
        )
        q = quantiles.detach().cpu().numpy()  # (batch, h, n_quantiles)
        for j, (uid, cutoff, _y, ds_test) in enumerate(chunk):
            rows.append(
                pd.DataFrame(
                    {
                        "model": "Chronos-Bolt (zero-shot)",
                        "unique_id": uid,
                        "cutoff": cutoff,
                        "ds": ds_test,
                        "point": q[j, :, 1],
                        "q10": q[j, :, 0],
                        "q50": q[j, :, 1],
                        "q90": q[j, :, 2],
                    }
                )
            )
    seconds = time.perf_counter() - t0

    out = pd.concat(rows, ignore_index=True)
    for c in ("point", "q10", "q50", "q90"):
        out[c] = out[c].clip(lower=0.0)
    return RungResult(
        out,
        seconds,
        f"{model_id}, zero-shot, no training on this data; context {context_length}d; "
        "quantiles emitted directly by the model.",
    )
