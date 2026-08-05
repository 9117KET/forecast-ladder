"""Protocol tests: the folds have to be non-overlapping, exhaustive at the end, and leak-free."""

import numpy as np
import pytest

from forecast_ladder.protocol import (
    PROTOCOL,
    Protocol,
    rolling_origin_cutoffs,
    train_test_slices,
)


class TestCutoffs:
    def test_hand_computed(self):
        # 100 observations, horizon 10, 3 folds stepping by 10.
        # Last fold must end at observation 100, so its cutoff is 90.
        # Earlier folds step back by 10: 80 and 70.
        assert rolling_origin_cutoffs(100, horizon=10, n_windows=3, step=10) == [70, 80, 90]

    def test_last_fold_ends_exactly_at_the_final_observation(self):
        n, h = 500, 28
        cutoffs = rolling_origin_cutoffs(n, horizon=h, n_windows=4)
        assert cutoffs[-1] + h == n

    def test_default_step_is_one_horizon_so_test_windows_do_not_overlap(self):
        h = 28
        cutoffs = rolling_origin_cutoffs(500, horizon=h, n_windows=4)
        gaps = np.diff(cutoffs)
        assert all(g >= h for g in gaps)

    def test_single_fold_is_a_plain_holdout(self):
        assert rolling_origin_cutoffs(100, horizon=10, n_windows=1) == [90]

    def test_too_short_series_raises_rather_than_silently_shrinking(self):
        # 30 observations cannot give 4 folds at horizon 28: the earliest origin would
        # have no training data at all. Failing loudly is the point.
        with pytest.raises(ValueError, match="cannot support"):
            rolling_origin_cutoffs(30, horizon=28, n_windows=4)

    def test_rejects_nonsense_arguments(self):
        with pytest.raises(ValueError):
            rolling_origin_cutoffs(100, horizon=0, n_windows=3)
        with pytest.raises(ValueError):
            rolling_origin_cutoffs(100, horizon=10, n_windows=0)
        with pytest.raises(ValueError):
            rolling_origin_cutoffs(100, horizon=10, n_windows=3, step=0)


class TestSlices:
    def test_shapes_and_alignment(self):
        y = np.arange(500, dtype=float)
        p = Protocol(horizon=28, n_windows=4, min_train_obs=100)
        slices = train_test_slices(y, p)
        assert len(slices) == 4
        for train, test, cutoff in slices:
            assert train.size == cutoff
            assert test.size == p.horizon
            # The test window is exactly the observations following the cutoff.
            np.testing.assert_array_equal(test, y[cutoff : cutoff + p.horizon])

    def test_no_leakage_train_ends_where_test_begins(self):
        y = np.arange(500, dtype=float)
        for train, test, cutoff in train_test_slices(y, Protocol(min_train_obs=100)):
            assert train[-1] < test[0]
            assert train[-1] == y[cutoff - 1]
            # Nothing from the test window can appear in training.
            assert not set(test.tolist()) & set(train.tolist())

    def test_test_windows_are_disjoint_across_folds(self):
        y = np.arange(500, dtype=float)
        seen: set[float] = set()
        for _, test, _ in train_test_slices(y, Protocol(min_train_obs=100)):
            vals = set(test.tolist())
            assert not (vals & seen), "test windows overlap between folds"
            seen |= vals


class TestFrozenProtocol:
    def test_defaults_match_what_the_writeup_claims(self):
        # These four numbers are quoted in the README and the case study. If one of them
        # changes, this test fails and the write-up gets corrected with it.
        assert PROTOCOL.horizon == 28
        assert PROTOCOL.n_windows == 4
        assert PROTOCOL.season_length == 7
        assert PROTOCOL.nominal_interval == 0.8

    def test_interval_quantiles_bracket_the_nominal_level(self):
        assert PROTOCOL.interval_quantiles == (0.1, 0.9)
        assert Protocol(nominal_interval=0.9).interval_quantiles == (0.05, 0.95)

    def test_step_defaults_to_the_horizon(self):
        assert PROTOCOL.effective_step == PROTOCOL.horizon
        assert Protocol(horizon=28, step=7).effective_step == 7

    def test_span_and_minimum_length_arithmetic(self):
        # 4 folds, horizon 28, step 28: the evaluation span is 28 * 4 = 112 days.
        assert PROTOCOL.total_test_span() == 112
        assert PROTOCOL.min_series_length() == PROTOCOL.min_train_obs + 112

    def test_protocol_cannot_be_mutated_mid_run(self):
        with pytest.raises(Exception):
            PROTOCOL.horizon = 7  # type: ignore[misc]

    def test_describe_is_populated(self):
        text = PROTOCOL.describe()
        assert "rolling origin" in text and "horizon 28d" in text
