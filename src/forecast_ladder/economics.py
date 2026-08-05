"""Turning forecast error into money, and the identity that connects the two.

A forecasting comparison that stops at MASE has not answered the question anyone is paying
for. The question is what to order, and the cost of getting it wrong is asymmetric: too much
stock ties up cash and eventually gets written off, too little loses a sale and sometimes a
customer.

**The newsvendor cost.** Ordering `q` against demand `y`:

    cost = overstock_cost * max(0, q - y) + stockout_cost * max(0, y - q)

**The identity worth knowing, and it is why the probabilistic forecast is not a garnish.**
The order quantity that minimises expected cost is not the mean forecast and not the median.
It is the quantile

    q* = stockout_cost / (stockout_cost + overstock_cost)

known as the critical ratio. If losing a sale costs three times what a surplus unit costs,
the right order is the 75th percentile of the demand distribution, not the middle of it. So
a point forecast, however accurate, cannot answer the business question on its own: you need
the distribution, and you need it calibrated at the specific quantile the cost ratio picks
out. That is also, up to a constant, exactly what pinball loss at `q*` measures, which means
the statistical metric and the euro figure are the same quantity in different units. It is
the cleanest argument in this project for why the probabilistic side of the comparison is
not decoration.

**Every number below is an assumption and is labelled as one.** M5 is Walmart US store-item
unit sales; no cost or price data ships with it. `CostModel` explains why surplus is costed
as carrying plus write-off rather than carrying alone, which is the choice that decides
whether the critical ratio lands somewhere a grocer would recognise. `sensitivity()` exists
because the honest way to present a case built on assumed costs is to show how far the answer
moves when the assumptions do.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = [
    "CostModel",
    "DEFAULT_COSTS",
    "critical_ratio",
    "order_quantity",
    "newsvendor_cost",
    "cost_table",
    "sensitivity",
]


@dataclass(frozen=True)
class CostModel:
    """Unit economics for one item over one replenishment decision.

    Every field is an assumption, not a measurement, and one of them dominates the answer.

    **Why surplus stock is not costed as pure carrying cost, which is the mistake that
    breaks this kind of model.** A first pass costed over-ordering at the daily holding rate
    alone: 25 percent of unit cost per year is about 0.2 cents per unit per day, against
    roughly 1.26 euro of lost margin per missed sale. That makes the critical ratio 0.998,
    so the cost-optimal order is the 99.8th percentile of demand, and the entire business
    case collapses into "order enormous quantities of everything", which no grocer does.

    The error is treating a repeated stocking decision as if surplus were free to carry
    indefinitely. In grocery and household goods, surplus is not carried indefinitely: a
    share of it is marked down or thrown away. So the cost of over-ordering is a carrying
    cost over the review period *plus* a write-off on the fraction that never sells at full
    price, and `write_off_fraction` is the single assumption this whole section is most
    sensitive to. It is the first thing `sensitivity()` varies.

    Attributes:
        unit_price: assumed average selling price per unit, in euros.
        gross_margin: assumed gross margin as a fraction of price. Grocery and household
            goods sit around 0.25 to 0.30.
        annual_holding_rate: assumed annual carrying cost as a fraction of unit cost,
            covering capital and space. 0.25 is the standard textbook figure.
        write_off_fraction: assumed share of surplus units that are eventually marked down
            or wasted rather than sold at full price. The dominant assumption.
        review_period_days: days a surplus unit is carried before the next replenishment
            decision, used to scale the carrying cost. Set to the forecast horizon.
    """

    unit_price: float = 4.50
    gross_margin: float = 0.28
    annual_holding_rate: float = 0.25
    write_off_fraction: float = 0.30
    review_period_days: int = 28
    source_note: str = (
        "Assumed, not measured. M5 ships unit sales with no cost or price data. Price and "
        "margin are grocery-retail rules of thumb, the holding rate is the standard 25 "
        "percent annual figure, and the write-off fraction is a judgement. The write-off "
        "fraction is the assumption the conclusion is most sensitive to: see sensitivity()."
    )

    @property
    def unit_cost(self) -> float:
        """Cost of goods per unit, implied by price and margin."""
        return self.unit_price * (1.0 - self.gross_margin)

    @property
    def stockout_cost(self) -> float:
        """Cost of being one unit short: the lost gross margin on a sale not made.

        This is the conservative reading. It counts the margin forgone and nothing else, so
        it excludes any substitution effect (the customer buys something else, so the loss
        is smaller) and any goodwill effect (the customer shops elsewhere next week, so the
        loss is larger). Those pull in opposite directions and neither is measurable here.
        """
        return self.unit_price * self.gross_margin

    @property
    def carrying_cost(self) -> float:
        """Cost of carrying one surplus unit across the review period."""
        return self.unit_cost * self.annual_holding_rate * self.review_period_days / 365.0

    @property
    def write_off_cost(self) -> float:
        """Expected write-off on one surplus unit."""
        return self.unit_cost * self.write_off_fraction

    @property
    def overstock_cost(self) -> float:
        """Total cost of one unit ordered and not sold."""
        return self.carrying_cost + self.write_off_cost

    @property
    def critical_ratio(self) -> float:
        return critical_ratio(self.overstock_cost, self.stockout_cost)

    def describe(self) -> str:
        return (
            f"price EUR {self.unit_price:.2f}, margin {self.gross_margin:.0%}, "
            f"unit cost EUR {self.unit_cost:.2f}; "
            f"stockout cost EUR {self.stockout_cost:.2f}/unit; "
            f"overstock cost EUR {self.overstock_cost:.2f}/unit "
            f"(carrying EUR {self.carrying_cost:.3f} over {self.review_period_days}d "
            f"+ write-off EUR {self.write_off_cost:.2f} at {self.write_off_fraction:.0%}); "
            f"critical ratio {self.critical_ratio:.3f}"
        )


DEFAULT_COSTS = CostModel()


def critical_ratio(overstock_cost: float, stockout_cost: float) -> float:
    """The cost-optimal service level: stockout / (stockout + overstock)."""
    total = overstock_cost + stockout_cost
    if total <= 0:
        raise ValueError("costs must be positive")
    return float(stockout_cost / total)


def order_quantity(
    quantile_forecasts: dict[float, np.ndarray],
    target_quantile: float,
) -> np.ndarray:
    """Order quantity at `target_quantile`, interpolated from the quantiles available.

    Only three quantiles are forecast in this project (10, 50, 90), so a critical ratio of,
    say, 0.996 cannot be read off directly. Linear interpolation between the two nearest
    available quantiles is used, and beyond the highest available quantile the value is held
    flat at that quantile rather than extrapolated.

    **This is a real limitation and it biases against every model equally.** A critical
    ratio far above 0.9 means the cost-optimal order sits in a tail this project never
    forecast, so every model's order is truncated at its own 90th percentile. The comparison
    between models stays fair; the absolute euro figures are conservative for all of them.
    """
    if not 0.0 < target_quantile < 1.0:
        raise ValueError("target_quantile must be in (0, 1)")
    levels = sorted(quantile_forecasts)
    if target_quantile <= levels[0]:
        return np.asarray(quantile_forecasts[levels[0]], dtype=float).copy()
    if target_quantile >= levels[-1]:
        return np.asarray(quantile_forecasts[levels[-1]], dtype=float).copy()
    for lo, hi in zip(levels, levels[1:]):
        if lo <= target_quantile <= hi:
            w = (target_quantile - lo) / (hi - lo)
            a = np.asarray(quantile_forecasts[lo], dtype=float)
            b = np.asarray(quantile_forecasts[hi], dtype=float)
            return a + w * (b - a)
    raise AssertionError("unreachable")


def newsvendor_cost(
    y_true, order_qty, overstock_cost: float, stockout_cost: float
) -> float:
    """Total asymmetric cost of ordering `order_qty` against demand `y_true`.

        cost = overstock_cost * sum(max(0, order - demand))
             + stockout_cost  * sum(max(0, demand - order))

    Public because it is the quantity the whole business case rests on, and because the
    identity it implies (minimised at the critical-ratio quantile) is worth being able to
    verify directly rather than taking on trust. See `tests/test_economics.py`.
    """
    y = np.asarray(y_true, dtype=float).ravel()
    q = np.asarray(order_qty, dtype=float).ravel()
    if y.shape != q.shape:
        raise ValueError(f"shape mismatch: {y.shape} vs {q.shape}")
    surplus = np.maximum(0.0, q - y)
    shortfall = np.maximum(0.0, y - q)
    return float((overstock_cost * surplus + stockout_cost * shortfall).sum())


def cost_table(
    forecasts: pd.DataFrame,
    panel: pd.DataFrame,
    costs: CostModel = DEFAULT_COSTS,
    use_critical_ratio: bool = True,
) -> pd.DataFrame:
    """Newsvendor cost per model, summed over the sample and reported per series-day.

    Two ordering policies are costed, because the difference between them is the point:

    `order_at_critical_ratio`: order the cost-optimal quantile, which needs the predictive
    distribution.

    `order_at_median`: order the median forecast, which is what happens when a point
    forecast is handed to a planner with no distribution attached.

    The gap between those two columns is the euro value of having a calibrated
    distribution at all, separately from the value of having a more accurate point forecast.
    """
    panel = panel.copy()
    panel["ds"] = pd.to_datetime(panel["ds"])
    actual = panel.set_index(["unique_id", "ds"])["y"]

    f = forecasts.copy()
    f["ds"] = pd.to_datetime(f["ds"])
    f["y"] = actual.reindex(pd.MultiIndex.from_arrays([f["unique_id"], f["ds"]])).to_numpy()
    if f["y"].isna().any():
        raise ValueError("forecast rows without a matching actual")

    cr = costs.critical_ratio
    h_cost, s_cost = costs.overstock_cost, costs.stockout_cost

    rows = []
    for model, g in f.groupby("model", observed=True):
        y = g["y"].to_numpy(dtype=float)
        qs = {
            0.1: g["q10"].to_numpy(dtype=float),
            0.5: g["q50"].to_numpy(dtype=float),
            0.9: g["q90"].to_numpy(dtype=float),
        }
        order_cr = order_quantity(qs, cr) if use_critical_ratio else qs[0.5]
        rows.append(
            {
                "model": model,
                "series_days": int(y.size),
                "cost_at_critical_ratio": newsvendor_cost(y, order_cr, h_cost, s_cost),
                "cost_at_median": newsvendor_cost(y, qs[0.5], h_cost, s_cost),
                "cost_perfect_foresight": 0.0,
            }
        )
    out = pd.DataFrame(rows)
    out["eur_per_series_day_at_cr"] = out["cost_at_critical_ratio"] / out["series_days"]
    out["eur_per_series_day_at_median"] = out["cost_at_median"] / out["series_days"]
    out["distribution_value_per_series_day"] = (
        out["eur_per_series_day_at_median"] - out["eur_per_series_day_at_cr"]
    )
    return out.sort_values("cost_at_critical_ratio", ignore_index=True)


def sensitivity(
    forecasts: pd.DataFrame,
    panel: pd.DataFrame,
    write_off_fractions: tuple[float, ...] = (0.10, 0.30, 0.60),
    margins: tuple[float, ...] = (0.15, 0.28, 0.40),
) -> pd.DataFrame:
    """Re-cost the whole comparison across a grid of the two assumptions that drive it.

    `write_off_fraction` comes first because it moves the critical ratio most: at 10 percent
    the optimal service level is around 0.77, at 60 percent it falls to about 0.39, which is
    the difference between ordering above the median and ordering below it.

    The purpose is to answer, before anyone asks, whether the model ranking is a property of
    the forecasts or of the cost assumptions. If the ranking is stable across the grid, the
    conclusion survives the assumptions being wrong. If it flips, that is the finding and it
    needs saying out loud rather than burying.
    """
    rows = []
    for wof in write_off_fractions:
        for margin in margins:
            costs = CostModel(gross_margin=margin, write_off_fraction=wof)
            t = cost_table(forecasts, panel, costs)
            best = t.iloc[0]
            for _, r in t.iterrows():
                rows.append(
                    {
                        "write_off_fraction": wof,
                        "gross_margin": margin,
                        "critical_ratio": costs.critical_ratio,
                        "model": r["model"],
                        "eur_per_series_day": r["eur_per_series_day_at_cr"],
                        "is_best": r["model"] == best["model"],
                    }
                )
    return pd.DataFrame(rows)
