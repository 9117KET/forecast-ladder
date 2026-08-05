"""Rung 0: the seasonal naive floor, point and probabilistic.

The floor exists so that every other rung has something to beat. It is written out here
rather than taken from a library because it is the one method in the project that has to
be beyond question: if the floor is wrong, every MASE in the results is wrong.

**The point forecast.** For step `i` ahead, repeat the observation from
`k = floor(i / m) + 1` seasons back. For daily data with weekly seasonality, next Tuesday
is forecast as last Tuesday, and the Tuesday four weeks out is forecast as the Tuesday
four weeks before the cutoff.

**The intervals, and why they widen.** Quantiles come from the empirical distribution of
the method's own in-sample errors at the matching seasonal lag. Step 3 relies on a
one-week-old anchor, step 24 relies on a four-week-old one, so the error distributions are
computed separately per lag and the intervals widen with horizon for a stated reason
rather than by a fitted variance assumption. This is a real method, not a placeholder: an
empirical-residual interval makes no distributional assumption at all, which on
intermittent retail demand is a feature, because the errors are not remotely Gaussian.
"""

from __future__ import annotations

import numpy as np

__all__ = ["seasonal_naive", "seasonal_naive_quantiles", "seasonal_lag_for_step"]


def seasonal_lag_for_step(step_index: int, season_length: int) -> int:
    """Seasonal lag the naive forecast uses for a given 0-based step ahead."""
    if step_index < 0:
        raise ValueError("step_index must be >= 0")
    if season_length < 1:
        raise ValueError("season_length must be >= 1")
    seasons_back = step_index // season_length + 1
    return season_length * seasons_back


def seasonal_naive(y_train, horizon: int, season_length: int) -> np.ndarray:
    """Point forecast: repeat the matching observation from the most recent full season."""
    y = np.asarray(y_train, dtype=float).ravel()
    m, h = int(season_length), int(horizon)
    if m < 1 or h < 1:
        raise ValueError("season_length and horizon must be >= 1")
    if y.size < m:
        raise ValueError(
            f"need at least {m} training observations for a seasonal naive, got {y.size}"
        )
    n = y.size
    out = np.empty(h, dtype=float)
    for i in range(h):
        lag = seasonal_lag_for_step(i, m)
        src = n + i - lag
        # For horizons beyond the available history the anchor would fall before the
        # series starts; fall back to the most recent complete season.
        if src < 0:
            src = n - m + (i % m)
        out[i] = y[src]
    return out


def seasonal_naive_quantiles(
    y_train,
    horizon: int,
    season_length: int,
    quantile_levels=(0.1, 0.5, 0.9),
    min_residuals: int = 10,
) -> dict[float, np.ndarray]:
    """Probabilistic forecast from the empirical distribution of in-sample errors.

    For each distinct seasonal lag the horizon touches, collect
    `e = y[t] - y[t - lag]` over the training window, then add the requested empirical
    quantiles of `e` to the point forecast for the steps that use that lag.

    Where a lag has fewer than `min_residuals` observations to work with, the residuals
    from the shortest available lag are reused and that is a documented limitation: the
    interval at long horizons is then too narrow, not too wide, which is the direction
    that matters and the direction to be honest about.
    """
    y = np.asarray(y_train, dtype=float).ravel()
    m, h = int(season_length), int(horizon)
    point = seasonal_naive(y, h, m)

    lags = sorted({seasonal_lag_for_step(i, m) for i in range(h)})
    residuals: dict[int, np.ndarray] = {}
    for lag in lags:
        if y.size > lag:
            e = y[lag:] - y[:-lag]
            if e.size >= min_residuals:
                residuals[lag] = e
    if not residuals:
        # Nothing to estimate spread from: return the point forecast at every quantile
        # and let the coverage number expose it.
        return {float(q): point.copy() for q in quantile_levels}

    shortest = min(residuals)
    out: dict[float, np.ndarray] = {}
    for q in quantile_levels:
        q = float(q)
        fq = np.empty(h, dtype=float)
        for i in range(h):
            lag = seasonal_lag_for_step(i, m)
            e = residuals.get(lag, residuals[shortest])
            fq[i] = point[i] + float(np.quantile(e, q))
        out[q] = fq

    # Demand cannot be negative, and an interval that allows it is not describing the
    # quantity being forecast. Clipping is applied after the quantiles are computed so
    # monotonicity across quantiles is preserved.
    for q in out:
        out[q] = np.maximum(out[q], 0.0)
    return out
