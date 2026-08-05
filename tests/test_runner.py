"""Runner tests: fold alignment, leak-free scoring, and honest aggregation."""

import numpy as np
import pandas as pd
import pytest

from forecast_ladder.protocol import Protocol
from forecast_ladder.runner import (
    aggregate,
    fold_cutoff_dates,
    normalise_cv_frame,
    run_seasonal_naive,
    score,
)

PROTO = Protocol(horizon=28, n_windows=4, season_length=7, min_train_obs=100)


def synthetic_panel(n_series: int = 3, n_obs: int = 500, seed: int = 0) -> pd.DataFrame:
    """A weekly-seasonal panel with a shared end date, like M5 after zero-trimming."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-06", periods=n_obs, freq="D")  # starts on a Monday
    frames = []
    for s in range(n_series):
        season = np.array([5, 4, 4, 6, 8, 12, 3], dtype=float) * (s + 1)
        y = np.tile(season, n_obs // 7 + 1)[:n_obs] + rng.normal(0, 1, n_obs)
        frames.append(
            pd.DataFrame({"unique_id": f"s{s}", "ds": dates, "y": np.clip(y, 0, None)})
        )
    return pd.concat(frames, ignore_index=True)


class TestFoldAlignment:
    def test_cutoffs_are_evenly_spaced_and_end_at_the_last_horizon(self):
        panel = synthetic_panel()
        last = panel["ds"].max()
        cutoffs = fold_cutoff_dates(panel, PROTO)
        assert len(cutoffs) == 4
        # Final fold tests the last 28 days, so its cutoff is 28 days before the end.
        assert cutoffs[-1] == last - pd.Timedelta(days=28)
        # Earliest fold sits three steps further back.
        assert cutoffs[0] == last - pd.Timedelta(days=112)
        assert list(np.diff(cutoffs)) == [pd.Timedelta(days=28)] * 3

    def test_cutoffs_are_identical_for_every_series(self):
        # A global model trained across series must not be tested on one series in a
        # period it was trained on for another. Shared cutoffs are what rule that out.
        panel = synthetic_panel(n_series=5)
        fc = run_seasonal_naive(panel, PROTO)
        per_series = fc.groupby("unique_id")["cutoff"].apply(lambda s: sorted(set(s)))
        first = per_series.iloc[0]
        assert all(v == first for v in per_series)


class TestSeasonalNaiveRun:
    def test_shape_is_folds_times_horizon_per_series(self):
        panel = synthetic_panel(n_series=3)
        fc = run_seasonal_naive(panel, PROTO)
        assert len(fc) == 3 * 4 * 28
        assert set(fc["model"]) == {"SeasonalNaive"}

    def test_forecast_dates_never_include_the_cutoff_or_earlier(self):
        panel = synthetic_panel()
        fc = run_seasonal_naive(panel, PROTO)
        assert (fc["ds"] > fc["cutoff"]).all()

    def test_each_fold_covers_exactly_the_horizon(self):
        panel = synthetic_panel()
        fc = run_seasonal_naive(panel, PROTO)
        counts = fc.groupby(["unique_id", "cutoff"]).size().unique()
        assert list(counts) == [28]


class TestScore:
    def test_perfect_forecast_scores_zero_mase(self):
        panel = synthetic_panel(n_series=2)
        fc = run_seasonal_naive(panel, PROTO)
        # Replace the forecast with the truth and the error must vanish.
        truth = panel.set_index(["unique_id", "ds"])["y"]
        idx = pd.MultiIndex.from_arrays([fc["unique_id"], fc["ds"]])
        for col in ("point", "q10", "q50", "q90"):
            fc[col] = truth.reindex(idx).to_numpy()
        got = score(fc, panel, PROTO)
        assert got["mase"].max() == pytest.approx(0.0)
        assert got["coverage"].min() == pytest.approx(1.0)

    def test_mase_scale_uses_only_pre_cutoff_data(self):
        # Corrupting the panel strictly after a fold's cutoff must not change that fold's
        # scale. This is the leakage test that matters for MASE specifically.
        panel = synthetic_panel(n_series=1, n_obs=400)
        fc = run_seasonal_naive(panel, PROTO)
        base = score(fc, panel, PROTO).sort_values("cutoff", ignore_index=True)

        first_cutoff = base.loc[0, "cutoff"]
        tampered = panel.copy()
        after = tampered["ds"] > first_cutoff
        tampered.loc[after, "y"] = tampered.loc[after, "y"] * 1000.0
        tampered_scores = score(fc, tampered, PROTO).sort_values("cutoff", ignore_index=True)

        assert tampered_scores.loc[0, "scale"] == pytest.approx(base.loc[0, "scale"])

    def test_one_row_per_model_series_fold(self):
        panel = synthetic_panel(n_series=3)
        fc = run_seasonal_naive(panel, PROTO)
        got = score(fc, panel, PROTO)
        assert len(got) == 3 * 4
        assert got.groupby(["model", "unique_id", "cutoff"]).size().max() == 1

    def test_missing_actual_raises_rather_than_scoring_a_gap(self):
        panel = synthetic_panel(n_series=1)
        fc = run_seasonal_naive(panel, PROTO)
        holed = panel.iloc[:-5]  # drop the tail the last fold is scored against
        with pytest.raises(ValueError, match="no matching actual"):
            score(fc, holed, PROTO)

    def test_rejects_frame_missing_required_columns(self):
        panel = synthetic_panel(n_series=1)
        fc = run_seasonal_naive(panel, PROTO).drop(columns=["q90"])
        with pytest.raises(ValueError, match="missing"):
            score(fc, panel, PROTO)


class TestNormaliseCvFrame:
    def _cv(self):
        return pd.DataFrame(
            {
                "unique_id": ["a"] * 3,
                "ds": pd.date_range("2020-02-01", periods=3),
                "cutoff": [pd.Timestamp("2020-01-31")] * 3,
                "MyModel": [1.0, -2.0, 3.0],
                "MyModel-lo-80": [-5.0, -6.0, 1.0],
                "MyModel-hi-80": [4.0, 2.0, 5.0],
            }
        )

    def test_maps_columns_and_clips_negatives(self):
        out = normalise_cv_frame(
            self._cv(), "MyModel", "MyModel", "MyModel-lo-80", "MyModel-hi-80"
        )
        assert list(out["point"]) == [1.0, 0.0, 3.0]
        assert list(out["q10"]) == [0.0, 0.0, 1.0]
        assert set(out["model"]) == {"MyModel"}

    def test_without_intervals_the_quantiles_collapse_onto_the_point(self):
        # A method that was never asked for a distribution must not be credited with one.
        out = normalise_cv_frame(self._cv(), "MyModel", "MyModel")
        assert (out["q10"] == out["point"]).all()
        assert (out["q90"] == out["point"]).all()

    def test_crossed_bounds_are_sorted_not_dropped(self):
        cv = self._cv()
        cv["MyModel-lo-80"], cv["MyModel-hi-80"] = cv["MyModel-hi-80"], cv["MyModel-lo-80"]
        out = normalise_cv_frame(
            cv, "MyModel", "MyModel", "MyModel-lo-80", "MyModel-hi-80"
        )
        assert (out["q10"] <= out["q90"]).all()

    def test_missing_column_raises_with_the_model_named(self):
        with pytest.raises(ValueError, match="MyModel"):
            normalise_cv_frame(self._cv(), "MyModel", "NotAColumn")


class TestAggregate:
    def test_reports_dropped_series_folds_rather_than_hiding_them(self):
        per_fold = pd.DataFrame(
            {
                "model": ["A"] * 4,
                "unique_id": ["s0", "s1", "s2", "s3"],
                "cutoff": [pd.Timestamp("2020-01-01")] * 4,
                "n": [28] * 4,
                "scale": [1.0, 1.0, np.nan, np.nan],
                "mae": [1.0, 3.0, 2.0, 2.0],
                "wape": [0.1] * 4,
                "mase": [1.0, 3.0, np.nan, np.nan],
                "pinball": [0.5] * 4,
                "coverage": [0.8] * 4,
                "mean_actual": [5.0] * 4,
            }
        )
        got = aggregate(per_fold)
        row = got.iloc[0]
        assert row["series_folds"] == 4
        assert row["mase_defined_on"] == 2
        assert row["mase_dropped"] == 2
        assert row["mase_mean"] == pytest.approx(2.0)
        assert row["mase_median"] == pytest.approx(2.0)

    def test_sorted_best_first(self):
        per_fold = pd.DataFrame(
            {
                "model": ["Bad", "Good"],
                "unique_id": ["s0", "s0"],
                "cutoff": [pd.Timestamp("2020-01-01")] * 2,
                "n": [28, 28],
                "scale": [1.0, 1.0],
                "mae": [5.0, 1.0],
                "wape": [0.5, 0.1],
                "mase": [5.0, 1.0],
                "pinball": [2.0, 0.4],
                "coverage": [0.5, 0.8],
                "mean_actual": [5.0, 5.0],
            }
        )
        got = aggregate(per_fold)
        assert list(got["model"]) == ["Good", "Bad"]
