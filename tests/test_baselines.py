"""Seasonal naive tests. This is the floor, so it gets checked hardest.

Every MASE in the published results divides by a number derived from this method. If the
floor is wrong, the whole comparison is wrong in a way that would look like a result.
"""

import numpy as np
import pytest

from forecast_ladder.baselines import (
    seasonal_lag_for_step,
    seasonal_naive,
    seasonal_naive_quantiles,
)


class TestSeasonalLag:
    def test_first_season_uses_one_season_back(self):
        # Steps 0 to 6 of a weekly series all read from last week.
        for i in range(7):
            assert seasonal_lag_for_step(i, 7) == 7

    def test_second_season_uses_two_seasons_back(self):
        for i in range(7, 14):
            assert seasonal_lag_for_step(i, 7) == 14

    def test_fourth_week_of_a_28_day_horizon(self):
        assert seasonal_lag_for_step(21, 7) == 28
        assert seasonal_lag_for_step(27, 7) == 28

    def test_rejects_bad_input(self):
        with pytest.raises(ValueError):
            seasonal_lag_for_step(-1, 7)
        with pytest.raises(ValueError):
            seasonal_lag_for_step(0, 0)


class TestSeasonalNaivePoint:
    def test_one_season_ahead_repeats_the_last_season(self):
        # y = 0..13, so the final week is [7, 8, 9, 10, 11, 12, 13].
        # Forecasting 7 days ahead must return exactly that week.
        y = np.arange(14, dtype=float)
        got = seasonal_naive(y, horizon=7, season_length=7)
        np.testing.assert_array_equal(got, np.array([7, 8, 9, 10, 11, 12, 13], dtype=float))

    def test_two_seasons_ahead_repeats_the_same_week_twice(self):
        # Step 7 reads 14 back from the origin, which is the same observation step 0 read
        # 7 back. The second forecast week is therefore identical to the first.
        y = np.arange(14, dtype=float)
        got = seasonal_naive(y, horizon=14, season_length=7)
        expected = np.tile(np.array([7, 8, 9, 10, 11, 12, 13], dtype=float), 2)
        np.testing.assert_array_equal(got, expected)

    def test_perfectly_seasonal_series_is_forecast_exactly(self):
        season = np.array([3, 1, 4, 1, 5, 9, 2], dtype=float)
        y = np.tile(season, 20)
        got = seasonal_naive(y, horizon=28, season_length=7)
        np.testing.assert_array_equal(got, np.tile(season, 4))

    def test_weekday_alignment_is_preserved_across_a_28_day_horizon(self):
        # The point of a seasonal naive on daily retail data: every forecast day must be
        # anchored on the same day of week. Encode the weekday as the value and check.
        y = np.tile(np.arange(7, dtype=float), 30)  # value == weekday index
        got = seasonal_naive(y, horizon=28, season_length=7)
        expected_weekdays = np.arange(28) % 7
        np.testing.assert_array_equal(got, expected_weekdays.astype(float))

    def test_too_short_history_raises(self):
        with pytest.raises(ValueError, match="at least 7"):
            seasonal_naive(np.arange(6, dtype=float), horizon=7, season_length=7)

    def test_rejects_bad_horizon(self):
        with pytest.raises(ValueError):
            seasonal_naive(np.arange(20, dtype=float), horizon=0, season_length=7)


class TestSeasonalNaiveQuantiles:
    def test_perfectly_seasonal_series_gives_zero_width_intervals(self):
        # Residuals are all exactly zero, so every quantile collapses onto the point
        # forecast. Any spread here would be invented.
        y = np.tile(np.array([3, 1, 4, 1, 5, 9, 2], dtype=float), 30)
        qs = seasonal_naive_quantiles(y, horizon=28, season_length=7)
        point = seasonal_naive(y, 28, 7)
        for q, f in qs.items():
            np.testing.assert_allclose(f, point, atol=1e-12)

    def test_quantiles_are_ordered(self):
        rng = np.random.default_rng(0)
        y = np.abs(np.tile(np.arange(7, dtype=float), 60) + rng.normal(0, 2, 420))
        qs = seasonal_naive_quantiles(y, horizon=28, season_length=7)
        assert np.all(qs[0.1] <= qs[0.5] + 1e-9)
        assert np.all(qs[0.5] <= qs[0.9] + 1e-9)

    def test_intervals_never_go_negative(self):
        # Demand cannot be negative, and a lower bound below zero would make an 80 percent
        # interval look wider than the quantity it describes can actually be.
        rng = np.random.default_rng(1)
        y = np.clip(rng.poisson(0.4, 500).astype(float), 0, None)
        qs = seasonal_naive_quantiles(y, horizon=28, season_length=7)
        for f in qs.values():
            assert np.all(f >= 0.0)

    def test_intervals_widen_with_horizon_on_a_random_walk(self):
        # A random walk is the case where multi-step uncertainty genuinely grows: the
        # variance of y[t] - y[t-L] scales with L, so a four-week-old anchor is a worse
        # anchor than a one-week-old one and the week-4 interval must be wider than the
        # week-1 interval. This is the property that makes the horizon-dependent residual
        # pooling worth the extra code.
        #
        # A deterministic trend would NOT show this: y = t + weekly_pattern has
        # y[t] - y[t-7] exactly equal to 7 at every t, so the residual distribution is a
        # single point and the correct interval width is zero at every horizon. That is
        # what an earlier version of this test got wrong.
        rng = np.random.default_rng(7)
        y = np.clip(100.0 + np.cumsum(rng.normal(0.0, 1.0, 800)), 0.0, None)
        qs = seasonal_naive_quantiles(y, horizon=28, season_length=7)
        width = qs[0.9] - qs[0.1]
        assert width[24:28].mean() > width[0:4].mean()

    def test_deterministic_trend_gives_zero_width_but_shifted_centres(self):
        # The companion to the test above, kept because the behaviour is surprising and
        # worth pinning: the seasonal naive is biased by exactly 7 per week on this series,
        # the residual distribution is degenerate, so the interval has zero width and the
        # quantiles sit on the bias-corrected value rather than on the raw point forecast.
        y = np.arange(400, dtype=float) + np.tile(np.arange(7, dtype=float), 400 // 7 + 1)[:400]
        qs = seasonal_naive_quantiles(y, horizon=28, season_length=7)
        point = seasonal_naive(y, 28, 7)
        np.testing.assert_allclose(qs[0.9] - qs[0.1], np.zeros(28), atol=1e-9)
        # Week 1 is corrected by one seasonal step of the trend, week 4 by four.
        np.testing.assert_allclose(qs[0.5][:7] - point[:7], np.full(7, 7.0), atol=1e-9)
        np.testing.assert_allclose(qs[0.5][21:28] - point[21:28], np.full(7, 28.0), atol=1e-9)

    def test_returns_requested_levels(self):
        y = np.tile(np.arange(7, dtype=float), 60)
        qs = seasonal_naive_quantiles(y, 28, 7, quantile_levels=(0.05, 0.5, 0.95))
        assert set(qs) == {0.05, 0.5, 0.95}
