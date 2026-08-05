"""The evaluation protocol, declared once, before any model exists.

This module is deliberately the second file in the project and it was written before a
single forecasting method was run. The reason is not tidiness. If the models come first,
the protocol gets adjusted, a fold at a time and a metric at a time, until it agrees with
results that already exist, and nobody involved ever notices themselves doing it.

Everything that could be tuned to flatter a model is fixed here as a constant:

**Rolling origin, not one holdout.** A single train/test split measures one draw from one
period. If the last 28 days of the sample happened to contain an unusual promotion, a
single split scores the model on that promotion. Several origins, each forecasting forward
from a different cutoff, measure the method instead of the calendar.

**Origins move by a whole horizon.** Overlapping test windows would reuse the same actuals
in more than one fold, which makes the folds correlated and the average look more precise
than it is.

**Horizon 28 days, weekly seasonality.** 28 is the M5 competition horizon, which makes the
numbers here comparable to published work rather than to nothing. It is also 4 complete
weekly cycles, so every day of the week is forecast 4 times per fold.

**Quantiles fixed in advance.** The 10th and 90th percentiles bracket a nominal 80 percent
interval, and 0.5 is the median. Choosing the interval after seeing which one the model
happened to cover correctly would make the coverage number meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Protocol", "PROTOCOL", "rolling_origin_cutoffs", "train_test_slices"]


@dataclass(frozen=True)
class Protocol:
    """A frozen evaluation configuration. Frozen so a run cannot mutate it midway."""

    horizon: int = 28
    n_windows: int = 4
    season_length: int = 7
    quantile_levels: tuple[float, ...] = (0.1, 0.5, 0.9)
    nominal_interval: float = 0.8
    step: int | None = None  # None means step by a whole horizon, which is the default

    #: Minimum training observations before a series is eligible at all. Two full years
    #: plus the test span, so that a yearly pattern has been seen twice and a model has a
    #: chance of learning it rather than being asked to invent it.
    min_train_obs: int = 730

    @property
    def effective_step(self) -> int:
        return self.horizon if self.step is None else int(self.step)

    @property
    def interval_quantiles(self) -> tuple[float, float]:
        """The two quantile levels that bracket `nominal_interval`."""
        tail = (1.0 - self.nominal_interval) / 2.0
        return (round(tail, 6), round(1.0 - tail, 6))

    def total_test_span(self) -> int:
        """Observations reserved for evaluation across all folds."""
        return self.horizon + (self.n_windows - 1) * self.effective_step

    def min_series_length(self) -> int:
        return self.min_train_obs + self.total_test_span()

    def describe(self) -> str:
        lo, hi = self.interval_quantiles
        return (
            f"rolling origin, {self.n_windows} folds, horizon {self.horizon}d, "
            f"step {self.effective_step}d, season {self.season_length}d, "
            f"quantiles {sorted(self.quantile_levels)}, "
            f"nominal interval {self.nominal_interval:.0%} = [q{lo}, q{hi}], "
            f"min series length {self.min_series_length()}"
        )


#: The protocol used for every rung of the ladder. Changing this invalidates every
#: number in results/published/, so change it by editing here and re-running everything,
#: never by passing a different object for one model.
PROTOCOL = Protocol()


def rolling_origin_cutoffs(
    n_obs: int,
    horizon: int,
    n_windows: int,
    step: int | None = None,
) -> list[int]:
    """Cutoff indices for a rolling-origin backtest.

    A cutoff is the number of observations available for training in that fold, so fold
    `i` trains on `y[:cutoff]` and is scored against `y[cutoff:cutoff + horizon]`. The
    last fold ends exactly at the final observation, so no data is left unused at the end.

    Returns cutoffs in increasing order (earliest origin first).

    Raises:
        ValueError: if the series is too short to yield `n_windows` non-overlapping folds.
    """
    if horizon < 1 or n_windows < 1:
        raise ValueError("horizon and n_windows must be >= 1")
    step = horizon if step is None else int(step)
    if step < 1:
        raise ValueError("step must be >= 1")

    last_cutoff = n_obs - horizon
    first_cutoff = last_cutoff - (n_windows - 1) * step
    if first_cutoff <= 0:
        raise ValueError(
            f"series of length {n_obs} cannot support {n_windows} folds at horizon "
            f"{horizon} with step {step}: the earliest fold would have no training data"
        )
    return [first_cutoff + i * step for i in range(n_windows)]


def train_test_slices(
    y: np.ndarray,
    protocol: Protocol = PROTOCOL,
) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """Split one series into (train, test, cutoff) tuples, one per fold.

    The train array is everything strictly before the cutoff and the test array is the
    `horizon` observations from the cutoff onward. Nothing after the test window is
    handed back, so a model cannot see the future of its own fold even by accident.
    """
    y = np.asarray(y, dtype=float).ravel()
    cutoffs = rolling_origin_cutoffs(
        n_obs=y.size,
        horizon=protocol.horizon,
        n_windows=protocol.n_windows,
        step=protocol.effective_step,
    )
    out = []
    for c in cutoffs:
        out.append((y[:c], y[c : c + protocol.horizon], c))
    return out
