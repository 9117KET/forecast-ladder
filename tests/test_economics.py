"""Economics tests, including a direct check of the identity the business case rests on.

The claim that the cost-optimal order is the critical-ratio quantile is the load-bearing
idea in `economics.py`. `test_cost_is_minimised_at_the_critical_ratio` verifies it by brute
force rather than by citation, which is the only way to be sure the implementation and the
textbook agree.
"""

import numpy as np
import pandas as pd
import pytest

from forecast_ladder.economics import (
    DEFAULT_COSTS,
    CostModel,
    cost_table,
    critical_ratio,
    newsvendor_cost,
    order_quantity,
    sensitivity,
)


class TestCriticalRatio:
    def test_hand_computed(self):
        # Losing a sale costs 3, holding a surplus unit costs 1, so order the 75th
        # percentile: 3 / (3 + 1).
        assert critical_ratio(overstock_cost=1.0, stockout_cost=3.0) == pytest.approx(0.75)

    def test_symmetric_costs_give_the_median(self):
        assert critical_ratio(2.0, 2.0) == pytest.approx(0.5)

    def test_expensive_surplus_pushes_the_order_below_the_median(self):
        assert critical_ratio(overstock_cost=3.0, stockout_cost=1.0) == pytest.approx(0.25)

    def test_rejects_nonpositive_total(self):
        with pytest.raises(ValueError):
            critical_ratio(0.0, 0.0)


class TestCostModel:
    def test_derived_quantities_are_hand_computed(self):
        c = CostModel()  # price 4.50, margin 28%, holding 25%/yr, write-off 30%, 28d
        assert c.unit_cost == pytest.approx(3.24)  # 4.50 * 0.72
        assert c.stockout_cost == pytest.approx(1.26)  # 4.50 * 0.28
        assert c.carrying_cost == pytest.approx(3.24 * 0.25 * 28 / 365)
        assert c.write_off_cost == pytest.approx(0.972)  # 3.24 * 0.30
        assert c.overstock_cost == pytest.approx(c.carrying_cost + 0.972)
        # 1.26 / (1.26 + 1.0341) = 0.5492
        assert c.critical_ratio == pytest.approx(0.5492, abs=1e-3)

    def test_critical_ratio_lands_where_a_grocer_would_recognise_it(self):
        # The whole reason write-off is in the model: carrying cost alone gives a critical
        # ratio of 0.998, meaning "order the 99.8th percentile of everything". This asserts
        # the model does not do that.
        assert 0.3 < DEFAULT_COSTS.critical_ratio < 0.8

    def test_carrying_cost_alone_would_be_degenerate(self):
        # Pinning the failure mode that motivated the design, so nobody reintroduces it.
        c = CostModel(write_off_fraction=0.0, review_period_days=1)
        assert c.critical_ratio > 0.99

    def test_write_off_fraction_moves_the_ratio_the_most(self):
        low = CostModel(write_off_fraction=0.10).critical_ratio
        high = CostModel(write_off_fraction=0.60).critical_ratio
        assert low > high  # cheaper surplus means order more
        assert low - high > 0.25  # and it moves the ratio a lot

    def test_frozen(self):
        with pytest.raises(Exception):
            DEFAULT_COSTS.unit_price = 9.99  # type: ignore[misc]

    def test_describe_carries_the_numbers(self):
        text = DEFAULT_COSTS.describe()
        assert "critical ratio" in text and "write-off" in text


class TestNewsvendorCost:
    def test_perfect_order_costs_nothing(self):
        assert newsvendor_cost([5.0, 3.0], [5.0, 3.0], 1.0, 3.0) == pytest.approx(0.0)

    def test_surplus_is_charged_the_overstock_cost(self):
        # Ordered 7 against demand 5: two surplus units at 1.0 each.
        assert newsvendor_cost([5.0], [7.0], 1.0, 3.0) == pytest.approx(2.0)

    def test_shortfall_is_charged_the_stockout_cost(self):
        # Ordered 3 against demand 5: two units short at 3.0 each.
        assert newsvendor_cost([5.0], [3.0], 1.0, 3.0) == pytest.approx(6.0)

    def test_sums_over_the_horizon(self):
        # Day 1: 2 surplus at 1.0 = 2.0. Day 2: 2 short at 3.0 = 6.0. Total 8.0.
        assert newsvendor_cost([5.0, 5.0], [7.0, 3.0], 1.0, 3.0) == pytest.approx(8.0)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            newsvendor_cost([1.0, 2.0], [1.0], 1.0, 1.0)

    def test_cost_is_minimised_at_the_critical_ratio(self):
        # The identity, verified by brute force. Demand is Poisson(6). If the true
        # predictive distribution is supplied as quantiles, the order that minimises
        # newsvendor cost should be the critical-ratio quantile of that distribution.
        rng = np.random.default_rng(11)
        demand = rng.poisson(6.0, 40_000).astype(float)

        overstock, stockout = 1.0, 3.0
        cr = critical_ratio(overstock, stockout)  # 0.75

        # Search integer order quantities and find the empirical minimiser.
        candidates = np.arange(0, 20, dtype=float)
        costs = [
            newsvendor_cost(demand, np.full_like(demand, q), overstock, stockout)
            for q in candidates
        ]
        empirical_best = float(candidates[int(np.argmin(costs))])

        # The theoretical answer is the 75th percentile of the demand distribution.
        theoretical = float(np.quantile(demand, cr))
        assert abs(empirical_best - theoretical) <= 1.0

    def test_ordering_at_the_median_is_worse_when_costs_are_asymmetric(self):
        # The practical consequence: handing a planner a point forecast (the median) costs
        # real money when stockouts hurt more than surplus does.
        rng = np.random.default_rng(12)
        demand = rng.poisson(6.0, 40_000).astype(float)
        overstock, stockout = 1.0, 3.0
        cr = critical_ratio(overstock, stockout)

        at_median = newsvendor_cost(
            demand, np.full_like(demand, np.quantile(demand, 0.5)), overstock, stockout
        )
        at_cr = newsvendor_cost(
            demand, np.full_like(demand, np.quantile(demand, cr)), overstock, stockout
        )
        assert at_cr < at_median


class TestOrderQuantity:
    def _qs(self):
        return {0.1: np.array([0.0]), 0.5: np.array([10.0]), 0.9: np.array([20.0])}

    def test_interpolates_between_available_quantiles(self):
        # Target 0.7 sits halfway between 0.5 and 0.9, so halfway between 10 and 20.
        got = order_quantity(self._qs(), 0.7)
        assert got[0] == pytest.approx(15.0)

    def test_exact_hit_returns_that_quantile(self):
        assert order_quantity(self._qs(), 0.5)[0] == pytest.approx(10.0)

    def test_clamps_above_the_highest_available_quantile(self):
        # Deliberately does not extrapolate: only three quantiles are forecast, so a
        # critical ratio of 0.99 is answered with the 90th percentile and the resulting
        # conservatism applies equally to every model.
        assert order_quantity(self._qs(), 0.99)[0] == pytest.approx(20.0)

    def test_clamps_below_the_lowest(self):
        assert order_quantity(self._qs(), 0.01)[0] == pytest.approx(0.0)

    def test_rejects_out_of_range_target(self):
        for bad in (0.0, 1.0, -0.5, 2.0):
            with pytest.raises(ValueError):
                order_quantity(self._qs(), bad)


def _tiny_case():
    """One series, four days, two models: a perfect one and one that always orders zero."""
    ds = pd.date_range("2020-02-01", periods=4)
    panel = pd.DataFrame({"unique_id": "a", "ds": ds, "y": [4.0, 4.0, 4.0, 4.0]})
    rows = []
    for model, val in [("Perfect", 4.0), ("AlwaysZero", 0.0)]:
        rows.append(
            pd.DataFrame(
                {
                    "model": model,
                    "unique_id": "a",
                    "cutoff": pd.Timestamp("2020-01-31"),
                    "ds": ds,
                    "point": val,
                    "q10": val,
                    "q50": val,
                    "q90": val,
                }
            )
        )
    return panel, pd.concat(rows, ignore_index=True)


class TestCostTable:
    def test_perfect_forecast_costs_nothing_and_ranks_first(self):
        panel, fc = _tiny_case()
        t = cost_table(fc, panel, DEFAULT_COSTS)
        assert t.iloc[0]["model"] == "Perfect"
        assert t.iloc[0]["cost_at_critical_ratio"] == pytest.approx(0.0)

    def test_always_zero_is_charged_the_full_shortfall(self):
        panel, fc = _tiny_case()
        t = cost_table(fc, panel, DEFAULT_COSTS).set_index("model")
        # 4 days x 4 units short x stockout cost 1.26
        assert t.loc["AlwaysZero", "cost_at_critical_ratio"] == pytest.approx(4 * 4 * 1.26)

    def test_reports_per_series_day_and_the_value_of_the_distribution(self):
        panel, fc = _tiny_case()
        t = cost_table(fc, panel, DEFAULT_COSTS)
        assert "eur_per_series_day_at_cr" in t.columns
        assert "distribution_value_per_series_day" in t.columns
        # With degenerate intervals (every quantile equal) the distribution adds nothing,
        # so the value of having one must come out as exactly zero rather than as noise.
        assert t["distribution_value_per_series_day"].abs().max() == pytest.approx(0.0)

    def test_missing_actual_raises(self):
        panel, fc = _tiny_case()
        with pytest.raises(ValueError, match="without a matching actual"):
            cost_table(fc, panel.iloc[:2], DEFAULT_COSTS)


class TestSensitivity:
    def test_grid_covers_every_combination_and_flags_one_winner_each(self):
        panel, fc = _tiny_case()
        s = sensitivity(fc, panel, write_off_fractions=(0.1, 0.6), margins=(0.2, 0.4))
        assert len(s) == 2 * 2 * 2  # 2 write-offs x 2 margins x 2 models
        winners = s[s["is_best"]].groupby(["write_off_fraction", "gross_margin"]).size()
        assert (winners == 1).all()

    def test_critical_ratio_falls_as_write_off_rises(self):
        panel, fc = _tiny_case()
        s = sensitivity(fc, panel, write_off_fractions=(0.1, 0.6), margins=(0.28,))
        cr = s.groupby("write_off_fraction")["critical_ratio"].first()
        assert cr.loc[0.1] > cr.loc[0.6]
