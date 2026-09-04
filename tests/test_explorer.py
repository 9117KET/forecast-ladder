"""Tests for the layer the app is built on.

The app itself is a Streamlit script and cannot be imported without starting a server, so
the way to keep it honest is to put every decision it makes into `explorer.py` and test
that. These tests are the reason `app.py` contains no logic.

They run against the committed artefacts, which is the point: if `data/sample_panel.parquet`
or `results/raw/*.parquet` ever fall out of the repository, or the published tables and the
raw forecasts stop describing the same run, these fail rather than the app failing in front
of somebody. That is the check worth having here, because a broken deployed page is
discovered by a reader and a broken test is discovered by a commit.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from forecast_ladder.analysis import FLOOR_MODEL
from forecast_ladder.explorer import (
    DataUnavailable,
    available_models,
    load_headline,
    load_panel,
    load_published,
    load_raw_forecasts,
    load_sample_series,
    load_timings,
    mase_colour,
    model_display_order,
    per_series_scores,
    repo_root,
    run_floor_backtest,
    series_catalogue,
    series_fold_view,
    series_history,
)
from forecast_ladder.protocol import PROTOCOL, Protocol

pytestmark = pytest.mark.skipif(
    not (repo_root() / "data" / "sample_panel.parquet").exists(),
    reason="committed sample panel absent; run scripts/prepare_panel.py",
)


# --------------------------------------------------------------------------------------
# The artefacts the app cannot start without
# --------------------------------------------------------------------------------------


def test_panel_is_the_frozen_sample():
    panel = load_panel()
    assert set(panel.columns) >= {"unique_id", "ds", "y"}
    assert panel["unique_id"].nunique() == len(load_sample_series())
    assert panel["y"].notna().all()
    assert pd.api.types.is_datetime64_any_dtype(panel["ds"])


def test_every_series_is_a_contiguous_daily_run():
    """A silent gap would shift every seasonal lag, so it is checked, not assumed."""
    panel = load_panel()
    for _uid, g in panel.groupby("unique_id", observed=True):
        d = g["ds"].sort_values()
        assert len(d) == (d.max() - d.min()).days + 1


def test_every_series_is_long_enough_for_the_protocol():
    lengths = load_panel().groupby("unique_id", observed=True).size()
    assert int(lengths.min()) >= PROTOCOL.total_test_span() + PROTOCOL.season_length + 1


def test_raw_forecasts_cover_the_whole_panel_at_every_fold():
    fc = load_raw_forecasts()
    panel = load_panel()
    assert set(fc.columns) >= {"model", "unique_id", "cutoff", "ds", "point", "q10", "q50", "q90"}
    assert set(fc["unique_id"]) == set(panel["unique_id"])
    assert fc["cutoff"].nunique() == PROTOCOL.n_windows
    # One row per model, series, fold and day, with nothing missing and nothing doubled.
    expected = (
        fc["model"].nunique()
        * panel["unique_id"].nunique()
        * PROTOCOL.n_windows
        * PROTOCOL.horizon
    )
    assert len(fc) == expected
    assert not fc.duplicated(["model", "unique_id", "cutoff", "ds"]).any()


def test_forecast_quantiles_are_ordered_and_non_negative():
    fc = load_raw_forecasts()
    assert (fc["q10"] <= fc["q90"]).all()
    assert (fc[["point", "q10", "q50", "q90"]] >= 0).all().all()


def test_every_forecast_day_has_an_actual():
    """The join `runner.score` depends on. If it fails, every metric is scored on a subset."""
    fc = load_raw_forecasts()
    actual = load_panel().set_index(["unique_id", "ds"])["y"]
    matched = actual.reindex(pd.MultiIndex.from_arrays([fc["unique_id"], fc["ds"]]))
    assert matched.notna().all()


def test_published_tables_and_raw_forecasts_describe_the_same_run():
    """The committed tables must be the output of the committed forecasts, not of some
    earlier run that happened to be on disk when analyse.py was called."""
    fc = load_raw_forecasts()
    ladder = load_published("ladder")
    head = load_headline()
    assert set(ladder["model"]) == set(fc["model"].unique())
    assert head["n_series"] == fc["unique_id"].nunique()
    assert head["best_model"] == ladder.sort_values("mase_mean").iloc[0]["model"]
    assert FLOOR_MODEL in set(ladder["model"])


def test_missing_artefact_names_the_command_that_makes_it():
    with pytest.raises(DataUnavailable) as err:
        load_published("no_such_table")
    assert "scripts/analyse.py" in str(err.value)


# --------------------------------------------------------------------------------------
# Derived views
# --------------------------------------------------------------------------------------


def test_model_display_order_is_by_published_mase():
    ladder = load_published("ladder").sort_values("mase_mean")
    assert model_display_order(reversed(list(ladder["model"]))) == list(ladder["model"])


def test_model_display_order_appends_unknown_models_without_dropping_any():
    known = list(load_published("ladder").sort_values("mase_mean")["model"])
    out = model_display_order(["ZZZ", *known, "AAA", "ZZZ"])
    assert out == known + ["AAA", "ZZZ"]


def test_catalogue_has_one_row_per_series_and_agrees_with_the_headline():
    cat = series_catalogue()
    assert len(cat) == len(load_sample_series())
    assert not cat["unique_id"].duplicated().any()
    assert int(cat["floor_holds"].sum()) == load_headline()["n_series_where_floor_holds"]


def test_catalogue_best_model_really_is_the_lowest_mase_on_that_series():
    cat = series_catalogue().set_index("unique_id")
    means = load_published("per_fold").groupby(["unique_id", "model"], observed=True)["mase"].mean()
    for uid in cat.index:
        assert cat.loc[uid, "best_mase"] == pytest.approx(means.loc[uid].min())
        assert means.loc[uid, cat.loc[uid, "best_model"]] == pytest.approx(means.loc[uid].min())


def test_the_two_naive_implementations_score_identically_on_every_series():
    """The cross-check the README reports, asserted rather than eyeballed.

    This project implements the seasonal naive rather than importing one, because every
    MASE in the results divides by a number derived from it. `statsforecast`'s own is run
    alongside for exactly this comparison. They agree to the last bit on all 300 series.
    """
    means = (
        load_published("per_fold")
        .groupby(["unique_id", "model"], observed=True)["mase"]
        .mean()
        .unstack("model")
    )
    assert (means[FLOOR_MODEL] == means["SeasonalNaive (library)"]).all()


def test_catalogue_breaks_ties_toward_the_cheaper_rung():
    """Ties at the lowest MASE are real and `idxmin` would resolve them by row order.

    The rule is lowest MASE, then lowest wall-clock, then name. It has to be deterministic,
    and it should not credit a 700-second transformer on a series where a seven-line
    baseline scored exactly the same.
    """
    cat = series_catalogue().set_index("unique_id")
    means = (
        load_published("per_fold")
        .groupby(["unique_id", "model"], observed=True)["mase"]
        .mean()
        .unstack("model")
    )
    seconds = {
        m: float(r["seconds"]) for r in load_timings().values() for m in r.get("models", [])
    }

    tied_seen = 0
    for uid in cat.index:
        row = means.loc[uid]
        winners = sorted(row[row == row.min()].index)
        if len(winners) > 1:
            tied_seen += 1
        cheapest = min(seconds.get(m, float("inf")) for m in winners)
        chosen = cat.loc[uid, "best_model"]
        assert chosen in winners
        assert seconds.get(chosen, float("inf")) == cheapest

    assert tied_seen >= 10, f"expected the known exact ties, saw {tied_seen}"
    # The library naive can only ever draw with this project's, which is cheaper, so it
    # never wins a series.
    assert "SeasonalNaive (library)" not in set(cat["best_model"])


def test_floor_holds_means_no_model_beat_the_floor_there():
    cat = series_catalogue().set_index("unique_id")
    means = (
        load_published("per_fold")
        .groupby(["unique_id", "model"], observed=True)["mase"]
        .mean()
        .unstack("model")
    )
    for uid in cat.index[cat["floor_holds"]]:
        row = means.loc[uid].drop(FLOOR_MODEL)
        assert (row >= means.loc[uid, FLOOR_MODEL]).all()


def test_series_history_returns_that_series_only():
    uid = load_panel()["unique_id"].iloc[0]
    hist = series_history(uid)
    assert list(hist.columns) == ["ds", "y"]
    assert hist["ds"].is_monotonic_increasing


def test_series_history_rejects_an_unknown_series():
    with pytest.raises(KeyError):
        series_history("NOT_A_SERIES")


def test_fold_view_splits_context_at_the_cutoff():
    fc = load_raw_forecasts()
    uid = fc["unique_id"].iloc[0]
    cutoff = sorted(fc["cutoff"].unique())[1]
    ctx, sel = series_fold_view(uid, cutoff)

    assert (ctx.loc[ctx["period"] == "train", "ds"] <= cutoff).all()
    assert (ctx.loc[ctx["period"] == "test", "ds"] > cutoff).all()
    # The test span is exactly the horizon, and the forecasts land inside it.
    assert int((ctx["period"] == "test").sum()) == PROTOCOL.horizon
    assert sel["ds"].min() == cutoff + pd.Timedelta(days=1)
    assert len(sel) == sel["model"].nunique() * PROTOCOL.horizon


def test_fold_view_honours_the_model_filter():
    fc = load_raw_forecasts()
    uid = fc["unique_id"].iloc[0]
    cutoff = sorted(fc["cutoff"].unique())[0]
    _ctx, sel = series_fold_view(uid, cutoff, models=[FLOOR_MODEL])
    assert set(sel["model"]) == {FLOOR_MODEL}


def test_per_series_scores_are_ordered_by_published_rank():
    uid = load_panel()["unique_id"].iloc[0]
    scores = per_series_scores(uid)
    assert len(scores) == scores["model"].nunique() * PROTOCOL.n_windows
    assert list(scores["model"].cat.categories) == model_display_order(scores["model"].unique())


def test_available_models_matches_the_raw_forecasts():
    assert sorted(available_models()) == sorted(load_raw_forecasts()["model"].unique())


# --------------------------------------------------------------------------------------
# The one thing the app recomputes
# --------------------------------------------------------------------------------------


def test_live_floor_reproduces_the_published_floor_under_the_published_protocol():
    """The check that makes the interactive backtest worth having.

    Re-running the floor on a handful of series under the frozen protocol has to land on
    the numbers `scripts/analyse.py` already published for those series. If it does not,
    the app is showing a second, quietly different implementation of the project's own
    baseline, which is exactly the failure this repository is built to avoid.
    """
    ids = sorted(load_panel()["unique_id"].unique())[:12]
    per_fold, _summary = run_floor_backtest(ids)

    published = load_published("per_fold")
    published = published[
        (published["model"] == FLOOR_MODEL) & (published["unique_id"].isin(ids))
    ]
    merged = per_fold.merge(
        published, on=["unique_id", "cutoff"], suffixes=("_live", "_pub"), validate="one_to_one"
    )
    assert len(merged) == len(ids) * PROTOCOL.n_windows
    for col in ("mase", "pinball", "coverage", "mae", "scale"):
        np.testing.assert_allclose(
            merged[f"{col}_live"], merged[f"{col}_pub"], rtol=1e-9, atol=1e-12
        )


def test_live_floor_respects_a_protocol_it_was_handed():
    ids = sorted(load_panel()["unique_id"].unique())[:4]
    proto = Protocol(horizon=14, n_windows=2, season_length=7)
    per_fold, summary = run_floor_backtest(ids, proto)
    assert per_fold["cutoff"].nunique() == 2
    assert (per_fold["n"] == 14).all()
    assert len(per_fold) == len(ids) * 2
    assert summary["coverage_target"].iloc[0] == proto.nominal_interval


def test_live_floor_refuses_a_protocol_the_data_cannot_support():
    ids = sorted(load_panel()["unique_id"].unique())[:2]
    with pytest.raises(ValueError, match="at least"):
        run_floor_backtest(ids, Protocol(horizon=28, n_windows=40))


def test_live_floor_refuses_an_empty_selection():
    with pytest.raises(ValueError, match="at least one series"):
        run_floor_backtest([])


def test_live_floor_refuses_series_outside_the_sample():
    with pytest.raises(ValueError, match="none of the requested series"):
        run_floor_backtest(["NOT_A_SERIES"])


# --------------------------------------------------------------------------------------
# Table colouring
# --------------------------------------------------------------------------------------


def test_mase_colour_is_anchored_on_the_floor_not_on_the_data():
    """1.0 is the seasonal naive. It must read as neutral wherever it appears."""
    at_floor = mase_colour(1.0)
    assert "26, 152, 80" in at_floor or "215, 48, 39" in at_floor
    # Faintest possible tint at the anchor, so a cell at the floor never looks like a win.
    assert "0.150" in at_floor


def test_mase_colour_is_green_below_the_floor_and_red_above_it():
    assert "26, 152, 80" in mase_colour(0.5)
    assert "215, 48, 39" in mase_colour(1.5)


def test_mase_colour_deepens_with_distance_from_the_floor():
    def alpha(css):
        return float(css.rsplit(",", 1)[1].strip(" )"))

    assert alpha(mase_colour(0.9)) < alpha(mase_colour(0.6)) < alpha(mase_colour(0.1))
    assert alpha(mase_colour(1.1)) < alpha(mase_colour(1.4)) < alpha(mase_colour(1.9))


def test_mase_colour_saturates_rather_than_running_past_full_opacity():
    """Intermittent series produce MASE in the tens. The scale must not overflow."""
    for extreme in (5.0, 50.0, 1e6):
        css = mase_colour(extreme)
        assert float(css.rsplit(",", 1)[1].strip(" )")) == pytest.approx(0.80)


def test_mase_colour_returns_no_style_for_undefined_values():
    """MASE is NaN where the training window was flat. A NaN must not be coloured as good."""
    for missing in (None, float("nan"), np.nan):
        assert mase_colour(missing) == ""


def test_colouring_a_table_works_with_matplotlib_uninstallable():
    """The reason `mase_colour` exists: `Styler.background_gradient` requires matplotlib,
    and that was this app's only use for it. Dropping it took matplotlib, kiwisolver,
    contourpy and fonttools out of the deployment — and it was matplotlib's transitive
    kiwisolver that had no wheel on the Python the host provisioned.

    So the property under test is that colouring a table still works when matplotlib
    cannot be imported at all. It runs in a subprocess with an import hook that raises on
    matplotlib, rather than asserting matplotlib is absent from `sys.modules`: pandas
    imports it opportunistically when it happens to be installed, so that assertion would
    pass or fail on what else is in the developer's virtualenv rather than on this code.
    """
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent(
        """
        import sys

        class Blocked:
            def find_module(self, name, path=None):
                return self.find_spec(name, path)

            def find_spec(self, name, path=None, target=None):
                if name == "matplotlib" or name.startswith("matplotlib."):
                    raise ImportError("matplotlib is blocked for this test")
                return None

        sys.meta_path.insert(0, Blocked())

        import pandas as pd
        from forecast_ladder.explorer import mase_colour

        df = pd.DataFrame({"a": [0.4, 1.0, 2.6], "b": [float("nan"), 0.95, 12.0]})
        html = df.style.map(mase_colour).to_html()
        assert "background-color" in html, html
        assert "rgba(26, 152, 80" in html, "expected a green cell below the floor"
        assert "rgba(215, 48, 39" in html, "expected a red cell above the floor"
        print("ok")
        """
    )
    env = {**os.environ, "PYTHONPATH": str(repo_root() / "src")}
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


def test_app_uses_no_api_that_needs_an_undeclared_dependency():
    """A guard written after the deployed app crashed on exactly this.

    `Styler.background_gradient` imports matplotlib lazily, from inside pandas, at render
    time. So nothing catches it: the import graph of `app.py` is clean, the tests pass, and
    the failure appears only when a reader opens the page on a machine without matplotlib.
    That is precisely what happened on Streamlit Community Cloud.

    The lesson is not "pin matplotlib" but that a lazily imported optional dependency is
    invisible to every other check in this repository, so it gets its own explicit one.
    `use_container_width` is here too: it is past its documented removal date in Streamlit,
    and on 1.59+ it also made the layered forecast chart log SVG errors.
    """
    app = (repo_root() / "app.py").read_text(encoding="utf-8")
    banned = {
        "background_gradient": "needs matplotlib, which is not a dependency; "
        "use explorer.mase_colour",
        "use_container_width": "removed from Streamlit; use width='stretch'",
        "text_gradient": "needs matplotlib, which is not a dependency",
        "pyplot": "matplotlib is not a dependency of this app",
    }
    found = {name: why for name, why in banned.items() if name in app}
    assert not found, "app.py uses " + "; ".join(f"{k} ({v})" for k, v in found.items())
