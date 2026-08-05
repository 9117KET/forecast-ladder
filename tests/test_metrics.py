"""Metric tests against values computed by hand.

Every expected number in this file was worked out on paper, not by running the code and
recording what it printed. That distinction is the only thing that makes a test of a
metric worth having.
"""

import math

import numpy as np
import pytest

from forecast_ladder.metrics import (
    interval_coverage,
    mae,
    mase,
    pinball_loss,
    rmse,
    seasonal_naive_scale,
    wape,
)


class TestSeasonalNaiveScale:
    def test_arithmetic_series_scale_equals_seasonal_step(self):
        # y = 1..14, m = 7. Every seasonal difference y[t] - y[t-7] is exactly 7,
        # so the mean absolute seasonal difference is 7.
        y = np.arange(1, 15, dtype=float)
        assert seasonal_naive_scale(y, 7) == pytest.approx(7.0)

    def test_known_mixed_values(self):
        # y = [10, 12, 11, 15], m = 2.
        # diffs: |11-10| = 1, |15-12| = 3. Mean = 2.0
        y = np.array([10.0, 12.0, 11.0, 15.0])
        assert seasonal_naive_scale(y, 2) == pytest.approx(2.0)

    def test_constant_series_is_nan_not_zero(self):
        # A flat series has zero seasonal error. Returning 0 would make every MASE
        # computed from it infinite, and one such series would then dominate any
        # average. NaN forces the caller to drop it and say so.
        assert math.isnan(seasonal_naive_scale(np.ones(20), 7))

    def test_series_too_short_is_nan(self):
        assert math.isnan(seasonal_naive_scale(np.arange(5, dtype=float), 7))
        # Exactly m observations is still too short: no seasonal difference exists.
        assert math.isnan(seasonal_naive_scale(np.arange(7, dtype=float), 7))

    def test_rejects_bad_season_length(self):
        with pytest.raises(ValueError):
            seasonal_naive_scale(np.arange(10, dtype=float), 0)


class TestPointMetrics:
    def test_mae_hand_computed(self):
        # errors 1 and 2, mean 1.5
        assert mae([2.0, 4.0], [1.0, 2.0]) == pytest.approx(1.5)

    def test_rmse_hand_computed(self):
        # errors 3 and 4, mean square (9 + 16) / 2 = 12.5, root = 3.5355...
        assert rmse([0.0, 0.0], [3.0, 4.0]) == pytest.approx(math.sqrt(12.5))

    def test_rmse_exceeds_mae_when_errors_are_uneven(self):
        # Worth asserting because it is the reason both are reported: RMSE punishes the
        # single large miss that MAE averages away.
        y, f = [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 8.0]
        assert rmse(y, f) > mae(y, f)

    def test_wape_hand_computed(self):
        # sum|error| = 2 + 4 = 6, sum|actual| = 20, so 0.3
        assert wape([10.0, 10.0], [8.0, 14.0]) == pytest.approx(0.3)

    def test_wape_undefined_when_total_actual_is_zero(self):
        assert math.isnan(wape([0.0, 0.0], [1.0, 2.0]))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            mae([1.0, 2.0], [1.0])


class TestMase:
    def test_hand_computed(self):
        # MAE = 1.5 (above), scale = 3.0, so MASE = 0.5
        assert mase([2.0, 4.0], [1.0, 2.0], scale=3.0) == pytest.approx(0.5)

    def test_perfect_forecast_is_zero(self):
        assert mase([3.0, 5.0], [3.0, 5.0], scale=2.0) == pytest.approx(0.0)

    def test_one_means_as_good_as_in_sample_naive(self):
        # MAE = 2.0 and scale = 2.0 means this forecast is exactly as accurate as the
        # seasonal naive was in sample. This is the number the whole project is read
        # against, so it gets its own test.
        assert mase([0.0, 0.0], [2.0, 2.0], scale=2.0) == pytest.approx(1.0)

    def test_unusable_scale_propagates_as_nan(self):
        for bad in (float("nan"), 0.0, -1.0, None):
            assert math.isnan(mase([1.0], [2.0], scale=bad))


class TestPinballLoss:
    def test_under_forecast_at_high_quantile_is_penalised_heavily(self):
        # q = 0.9, actual 10, forecast 8. Actual is above the forecast, so the loss is
        # q * (y - f) = 0.9 * 2 = 1.8
        assert pinball_loss([10.0], {0.9: np.array([8.0])}) == pytest.approx(1.8)

    def test_over_forecast_at_high_quantile_is_penalised_lightly(self):
        # q = 0.9, actual 8, forecast 10. Actual below forecast, so the loss is
        # (1 - q) * (f - y) = 0.1 * 2 = 0.2
        assert pinball_loss([8.0], {0.9: np.array([10.0])}) == pytest.approx(0.2)

    def test_asymmetry_reverses_at_the_low_quantile(self):
        # At q = 0.1 the penalties swap: being over is what costs.
        under = pinball_loss([10.0], {0.1: np.array([8.0])})  # 0.1 * 2 = 0.2
        over = pinball_loss([8.0], {0.1: np.array([10.0])})  # 0.9 * 2 = 1.8
        assert under == pytest.approx(0.2)
        assert over == pytest.approx(1.8)

    def test_median_is_symmetric_and_is_half_the_absolute_error(self):
        # At q = 0.5 pinball loss is exactly MAE / 2, in both directions.
        assert pinball_loss([10.0], {0.5: np.array([8.0])}) == pytest.approx(1.0)
        assert pinball_loss([8.0], {0.5: np.array([10.0])}) == pytest.approx(1.0)

    def test_averages_over_quantiles_and_time(self):
        # q=0.5 with error 2 gives 1.0; q=0.9 under-forecast by 2 gives 1.8.
        # Mean over both quantiles = (1.0 + 1.8) / 2 = 1.4
        got = pinball_loss(
            [10.0], {0.5: np.array([8.0]), 0.9: np.array([8.0])}
        )
        assert got == pytest.approx(1.4)

    def test_rejects_quantile_outside_open_unit_interval(self):
        for bad in (0.0, 1.0, -0.1, 1.5):
            with pytest.raises(ValueError):
                pinball_loss([1.0], {bad: np.array([1.0])})

    def test_requires_at_least_one_quantile(self):
        with pytest.raises(ValueError):
            pinball_loss([1.0], {})


class TestIntervalCoverage:
    def test_hand_computed(self):
        # actuals 1,2,3,4 against [0, 2]: 1 and 2 are inside, 3 and 4 are not.
        assert interval_coverage([1, 2, 3, 4], [0] * 4, [2] * 4) == pytest.approx(0.5)

    def test_bounds_are_inclusive(self):
        assert interval_coverage([2.0], [2.0], [2.0]) == pytest.approx(1.0)

    def test_full_and_zero_coverage(self):
        assert interval_coverage([5.0], [0.0], [10.0]) == pytest.approx(1.0)
        assert interval_coverage([50.0], [0.0], [10.0]) == pytest.approx(0.0)

    def test_inverted_interval_raises(self):
        with pytest.raises(ValueError):
            interval_coverage([1.0], [5.0], [0.0])
