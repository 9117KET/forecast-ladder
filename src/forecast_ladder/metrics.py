"""Forecast accuracy metrics, implemented here rather than imported.

Every metric in this file is available in `utilsforecast` and in half a dozen other
packages. They are written out by hand anyway, for one reason: a metric you cannot
derive is a metric you cannot defend when somebody asks why the number moved. Each
function below is tested against a value computed by hand in `tests/test_metrics.py`.

Three choices worth stating, because they are the ones that decide whether a
comparison means anything:

**MASE, not MAPE.** MAPE divides by the actual, so a single zero-demand day makes it
infinite and a near-zero day makes it enormous. Retail demand is full of both. MAPE
also rewards under-forecasting: you can only be 100 percent wrong on the low side and
unboundedly wrong on the high side, so a model that systematically forecasts low scores
better. MASE divides by a fixed number computed once from the training data, so it is
scale-free, defined at zero, and symmetric.

**The MASE denominator comes from the training window, never the test window.** This is
the part that is easy to get wrong and that changes the answer. Scaling by in-sample
naive error means the denominator is a property of the series, so MASE is comparable
across series. Scaling by test-window error would make the denominator a property of the
split, and two models evaluated on different splits would no longer be comparable.

**Pinball loss for the distribution, coverage as the audit.** Pinball loss scores a whole
set of quantiles, so it rewards a forecast that knows how uncertain it is. Coverage is
the separate check that the intervals are honest: if the nominal 80 percent interval
contains the actual 60 percent of the time, the model is overconfident, and pinball loss
alone will not tell you that in a way anyone can act on.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "seasonal_naive_scale",
    "mase",
    "mae",
    "rmse",
    "pinball_loss",
    "interval_coverage",
    "wape",
]


def _as_1d(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("empty array")
    return arr


def seasonal_naive_scale(y_train, season_length: int) -> float:
    """Mean absolute one-season-back difference of the training series.

    This is the denominator of MASE: the average error a seasonal naive forecast would
    have made in sample. Computed on the training window only.

        scale = mean(|y[t] - y[t - m]|) for t = m .. n-1

    Returns NaN when the series is too short to have a single seasonal difference, or
    when the series is exactly constant (scale 0), because dividing by it would produce
    an infinity that would then silently dominate any average over series. A caller that
    averages MASE across series must drop those series and say how many it dropped.
    """
    y = _as_1d(y_train)
    m = int(season_length)
    if m < 1:
        raise ValueError("season_length must be >= 1")
    if y.size <= m:
        return float("nan")
    diffs = np.abs(y[m:] - y[:-m])
    scale = float(diffs.mean())
    return scale if scale > 0 else float("nan")


def mae(y_true, y_pred) -> float:
    """Mean absolute error."""
    y, f = _as_1d(y_true), _as_1d(y_pred)
    if y.shape != f.shape:
        raise ValueError(f"shape mismatch: {y.shape} vs {f.shape}")
    return float(np.abs(y - f).mean())


def rmse(y_true, y_pred) -> float:
    """Root mean squared error. Reported alongside MAE only to show they disagree."""
    y, f = _as_1d(y_true), _as_1d(y_pred)
    if y.shape != f.shape:
        raise ValueError(f"shape mismatch: {y.shape} vs {f.shape}")
    return float(np.sqrt(((y - f) ** 2).mean()))


def wape(y_true, y_pred) -> float:
    """Weighted absolute percentage error: sum|error| / sum|actual|.

    Kept because it is what a business reads most naturally ("we were 12 percent out
    over the quarter"), and because it is defined when individual actuals are zero as
    long as the total is not. It is still scale-dependent, so it is not used to compare
    across series.
    """
    y, f = _as_1d(y_true), _as_1d(y_pred)
    if y.shape != f.shape:
        raise ValueError(f"shape mismatch: {y.shape} vs {f.shape}")
    denom = float(np.abs(y).sum())
    if denom == 0:
        return float("nan")
    return float(np.abs(y - f).sum() / denom)


def mase(y_true, y_pred, scale: float) -> float:
    """Mean absolute scaled error, given a scale from `seasonal_naive_scale`.

    MASE = MAE(test) / scale, where scale is the in-sample seasonal naive MAE.

    Reading it: 1.0 means this forecast is as good as a seasonal naive was in sample.
    Below 1.0 is better, above 1.0 is worse. A model that cannot get below 1.0 has not
    earned the compute it cost.

    The scale is passed in rather than computed here so that a caller cannot accidentally
    scale by the test window. It has to fetch the training scale deliberately.
    """
    if scale is None or not np.isfinite(scale) or scale <= 0:
        return float("nan")
    return mae(y_true, y_pred) / float(scale)


def pinball_loss(y_true, quantile_forecasts: dict[float, np.ndarray]) -> float:
    """Average pinball (quantile) loss over the supplied quantiles and time steps.

    For a single quantile level q and forecast f_q:

        L = q * (y - f_q)        when y >= f_q   (penalty for forecasting too low)
        L = (1 - q) * (f_q - y)  when y <  f_q   (penalty for forecasting too high)

    The asymmetry is the whole point. At q = 0.9 being under the actual costs nine times
    what being over it costs, so the loss is minimised by a forecast that sits near the
    90th percentile of the predictive distribution. Minimising pinball loss across a set
    of quantiles therefore scores the shape of the distribution, not just its centre.

    Args:
        y_true: actuals, length h.
        quantile_forecasts: {quantile level in (0, 1): forecast array of length h}.
    """
    y = _as_1d(y_true)
    if not quantile_forecasts:
        raise ValueError("no quantile forecasts supplied")

    losses = []
    for q, f in quantile_forecasts.items():
        q = float(q)
        if not 0.0 < q < 1.0:
            raise ValueError(f"quantile must be in (0, 1), got {q}")
        f = _as_1d(f)
        if f.shape != y.shape:
            raise ValueError(f"shape mismatch at q={q}: {f.shape} vs {y.shape}")
        diff = y - f
        losses.append(np.where(diff >= 0, q * diff, (q - 1.0) * diff))
    return float(np.mean(np.concatenate(losses)))


def interval_coverage(y_true, lower, upper) -> float:
    """Fraction of actuals falling inside [lower, upper], inclusive.

    Compare against the nominal level of the interval. A nominal 80 percent interval
    should cover close to 0.80. Materially below means the model is overconfident, which
    is the failure mode that matters: an interval nobody can trust is worse than no
    interval, because somebody will size a buffer from it.
    """
    y, lo, hi = _as_1d(y_true), _as_1d(lower), _as_1d(upper)
    if not (y.shape == lo.shape == hi.shape):
        raise ValueError(f"shape mismatch: {y.shape}, {lo.shape}, {hi.shape}")
    if np.any(hi < lo):
        raise ValueError("upper bound below lower bound")
    inside = (y >= lo) & (y <= hi)
    return float(inside.mean())
