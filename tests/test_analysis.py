"""Analysis tests. These guard the finding, so they check the arithmetic of the claim itself."""

import numpy as np
import pandas as pd
import pytest

from forecast_ladder.analysis import (
    beats_floor,
    floor_holds_profile,
    headline_numbers,
    win_rate_by_group,
)


def per_fold_fixture() -> pd.DataFrame:
    """Three series, two folds, a floor and two challengers, with a designed outcome.

    s0: the floor wins outright, nothing beats it.
    s1: Good beats the floor, Bad does not.
    s2: both challengers beat the floor.
    """
    rows = []
    design = {
        # series: (floor mase, Good mase, Bad mase)
        "s0": (1.0, 1.2, 1.5),
        "s1": (1.0, 0.8, 1.3),
        "s2": (1.0, 0.7, 0.9),
    }
    for uid, (floor, good, bad) in design.items():
        for model, m in [("SeasonalNaive", floor), ("Good", good), ("Bad", bad)]:
            for fold in range(2):
                rows.append(
                    {
                        "model": model,
                        "unique_id": uid,
                        "cutoff": pd.Timestamp("2020-01-01") + pd.Timedelta(days=28 * fold),
                        "n": 28,
                        "scale": 1.0,
                        "mae": m,
                        "wape": 0.1,
                        "mase": m,
                        "pinball": 0.5,
                        "coverage": 0.8,
                        "mean_actual": 5.0,
                    }
                )
    return pd.DataFrame(rows)


def features_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unique_id": ["s0", "s1", "s2"],
            "mean_demand": [0.1, 1.0, 5.0],
            "zero_share": [0.9, 0.5, 0.1],
            "adi": [10.0, 2.0, 1.1],
            "cv2": [1.0, 0.5, 0.2],
            "n_obs": [842, 842, 842],
        }
    )


class TestBeatsFloor:
    def test_flags_match_the_design(self):
        b = beats_floor(per_fold_fixture()).set_index(["model", "unique_id"])["beats_floor"]
        assert b.loc[("Good", "s0")] is np.False_ or not b.loc[("Good", "s0")]
        assert b.loc[("Good", "s1")]
        assert b.loc[("Good", "s2")]
        assert not b.loc[("Bad", "s1")]
        assert b.loc[("Bad", "s2")]

    def test_floor_is_excluded_from_its_own_comparison(self):
        b = beats_floor(per_fold_fixture())
        assert "SeasonalNaive" not in set(b["model"])

    def test_improvement_is_a_fraction_of_the_floor(self):
        b = beats_floor(per_fold_fixture()).set_index(["model", "unique_id"])["improvement"]
        # Good on s2 scores 0.7 against a floor of 1.0, so a 30 percent improvement.
        assert b.loc[("Good", "s2")] == pytest.approx(0.30)
        # Bad on s0 scores 1.5 against 1.0, so a negative improvement.
        assert b.loc[("Bad", "s0")] == pytest.approx(-0.50)

    def test_averages_folds_before_comparing(self):
        # The unit of decision is the series, not the series-fold: a planner picks one
        # method per item. Two folds with different MASE must collapse to their mean first.
        pf = per_fold_fixture()
        pf.loc[(pf["model"] == "Good") & (pf["unique_id"] == "s1"), "mase"] = [0.6, 1.0]
        b = beats_floor(pf).set_index(["model", "unique_id"])["mase"]
        assert b.loc[("Good", "s1")] == pytest.approx(0.8)

    def test_missing_floor_raises(self):
        pf = per_fold_fixture()
        pf = pf[pf["model"] != "SeasonalNaive"]
        with pytest.raises(ValueError, match="not present"):
            beats_floor(pf)


class TestWinRateByGroup:
    def test_groups_by_a_categorical_property(self):
        feats = features_fixture()
        feats["stratum"] = ["sparse", "medium", "dense"]
        w = win_rate_by_group(beats_floor(per_fold_fixture()), feats, "stratum")
        got = w.set_index(["model", "bucket"])["win_rate"]
        assert got.loc[("Good", "sparse")] == pytest.approx(0.0)
        assert got.loc[("Good", "dense")] == pytest.approx(1.0)

    def test_bins_a_continuous_property_without_crashing_on_few_series(self):
        w = win_rate_by_group(
            beats_floor(per_fold_fixture()), features_fixture(), "zero_share", n_bins=4
        )
        assert {"model", "bucket", "n_series", "win_rate"} <= set(w.columns)
        assert w["win_rate"].between(0.0, 1.0).all()


class TestFloorHoldsProfile:
    def test_identifies_the_series_nothing_beats(self):
        prof = floor_holds_profile(beats_floor(per_fold_fixture()), features_fixture())
        row = prof.set_index("feature").loc["zero_share"]
        # Only s0 has no challenger beating the floor.
        assert row["n_floor_holds"] == 1
        assert row["n_floor_beaten"] == 2
        # s0 is the most intermittent series in the fixture.
        assert row["median_where_floor_holds"] > row["median_where_floor_beaten"]

    def test_covers_every_numeric_feature_present(self):
        prof = floor_holds_profile(beats_floor(per_fold_fixture()), features_fixture())
        assert set(prof["feature"]) == {"mean_demand", "zero_share", "adi", "cv2", "n_obs"}


class TestHeadlineNumbers:
    def test_reports_the_numbers_the_writeups_quote(self):
        pf = per_fold_fixture()
        h = headline_numbers(pf, beats_floor(pf), features_fixture())
        assert h["n_series"] == 3
        assert h["floor_model"] == "SeasonalNaive"
        assert h["n_series_where_floor_holds"] == 1
        assert h["pct_series_where_floor_holds"] == pytest.approx(100 / 3)
        assert h["best_model"] == "Good"  # lowest mean MASE in the fixture
        assert set(h["win_rate_by_model"]) == {"Good", "Bad"}
        assert h["win_rate_by_model"]["Good"] == pytest.approx(2 / 3)

    def test_intermittency_contrast_is_carried(self):
        pf = per_fold_fixture()
        h = headline_numbers(pf, beats_floor(pf), features_fixture())
        # The series the floor holds on is more intermittent than the ones it loses on.
        assert (
            h["median_zero_share_where_floor_holds"]
            > h["median_zero_share_where_floor_beaten"]
        )
