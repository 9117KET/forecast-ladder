"""forecast-ladder, as something you can interrogate rather than only read.

    streamlit run app.py

The repository is a batch comparison and stays one: nothing here re-fits AutoARIMA or a
transformer on demand. What it does is put the run's own artefacts behind controls, so the
claims in the README can be checked instead of taken:

- the ladder table, with the compute cost next to the accuracy it bought
- win rate against the floor by intermittency, which is where the finding actually lives
- one series at a time, every model's forecast against the actual, on any fold
- the seasonal naive floor re-run live under a protocol you choose
- the newsvendor case re-costed live when you move the assumption it rests on

All of the logic is in `forecast_ladder.explorer` and is tested. This file is layout.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from forecast_ladder import explorer as ex  # noqa: E402
from forecast_ladder.economics import CostModel, cost_table, sensitivity  # noqa: E402
from forecast_ladder.protocol import PROTOCOL, Protocol  # noqa: E402

st.set_page_config(
    page_title="forecast-ladder",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

FLOOR = "SeasonalNaive"
BAND = alt.Scale(scheme="tableau10")


# --------------------------------------------------------------------------------------
# Cached loaders. Streamlit reruns this script top to bottom on every widget change, so
# anything touching disk has to be cached or the app re-reads 4.5 MB of parquet per click.
# --------------------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def ladder() -> pd.DataFrame:
    return ex.load_published("ladder")


@st.cache_data(show_spinner=False)
def headline() -> dict:
    return ex.load_headline()


@st.cache_data(show_spinner=False)
def catalogue() -> pd.DataFrame:
    return ex.series_catalogue()


@st.cache_data(show_spinner=False)
def raw_forecasts() -> pd.DataFrame:
    return ex.load_raw_forecasts()


@st.cache_data(show_spinner=False)
def panel() -> pd.DataFrame:
    return ex.load_panel()


@st.cache_data(show_spinner=False)
def published(name: str) -> pd.DataFrame:
    return ex.load_published(name)


@st.cache_data(show_spinner="re-costing every model...")
def live_costs(price, margin, holding, write_off, review) -> tuple[pd.DataFrame, str, float]:
    costs = CostModel(
        unit_price=price,
        gross_margin=margin,
        annual_holding_rate=holding,
        write_off_fraction=write_off,
        review_period_days=int(review),
    )
    return cost_table(raw_forecasts(), panel(), costs), costs.describe(), costs.critical_ratio


@st.cache_data(show_spinner="running the floor...")
def floor_backtest(ids: tuple[str, ...], horizon: int, n_windows: int, season: int):
    proto = Protocol(horizon=int(horizon), n_windows=int(n_windows), season_length=int(season))
    return ex.run_floor_backtest(list(ids), proto)


@st.cache_data(show_spinner="re-costing across the assumption grid...")
def live_sensitivity() -> pd.DataFrame:
    return sensitivity(raw_forecasts(), panel())


def fail(err: Exception) -> None:
    """Show a missing-artefact error with the command that produces it, and stop."""
    st.error(str(err))
    st.caption(
        "This app reads committed artefacts only. If you are running from a fresh clone "
        "and see this, the files above are missing from the checkout."
    )
    st.stop()


# --------------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------------

with st.sidebar:
    st.title("forecast-ladder")
    st.caption(
        "A fixed-protocol comparison of forecasting methods on real retail demand, "
        "from a seven-line seasonal naive up to a pretrained transformer."
    )
    try:
        head = headline()
    except ex.DataUnavailable as err:
        fail(err)

    st.metric("Series", head["n_series"])
    st.metric("Best mean MASE", f"{head['best_mase_mean']:.3f}", help=head["best_model"])
    st.metric(
        "Floor holds on",
        f"{head['n_series_where_floor_holds']} series",
        help="Series no rung on the ladder beat the seasonal naive on.",
    )
    st.divider()
    st.markdown("**The protocol**, frozen before any model ran:")
    st.code(PROTOCOL.describe().replace(", ", ",\n"), language=None)
    st.divider()
    st.markdown(
        "[Repository](https://github.com/9117KET/forecast-ladder) · "
        "[Walkthrough](https://github.com/9117KET/forecast-ladder/blob/main/docs/WALKTHROUGH.md)"
    )

st.title("Where does forecasting complexity stop paying for itself?")
st.markdown(
    "Five rungs, 300 M5 retail series, four rolling origins, a 28-day horizon, one "
    "protocol written down before any model ran. **Nothing on this page is recomputed "
    "except where it says so** — the tables are the run's own output, and the two live "
    "sections re-run the floor and re-cost the case through the same code that produced "
    "them."
)

tab_ladder, tab_pays, tab_series, tab_run, tab_money = st.tabs(
    [
        "The ladder",
        "Where complexity pays",
        "One series at a time",
        "Run the floor yourself",
        "The business case",
    ]
)


# --------------------------------------------------------------------------------------
# 1. The ladder
# --------------------------------------------------------------------------------------

with tab_ladder:
    try:
        lad = ladder().copy()
    except ex.DataUnavailable as err:
        fail(err)

    order = list(lad.sort_values("mase_mean")["model"])
    best, floor_mase = lad.iloc[0], head["floor_mase_mean"]

    # Every delta here is set to "off" (neutral grey). Streamlit's default colours a
    # positive delta green, and on a metric where lower is better that arrow says the
    # opposite of what the number means.
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best model", best["model"], f"MASE {best['mase_mean']:.3f}", delta_color="off")
    c2.metric(
        "Against the floor",
        f"{100 * (1 - best['mase_mean'] / floor_mase):.1f}% better",
        help=f"Seasonal naive floor: MASE {floor_mase:.3f}",
    )
    zero_shot = lad[lad["model"].str.contains("Chronos")]
    if not zero_shot.empty:
        z = zero_shot.iloc[0]
        c3.metric(
            "Zero-shot, no fitting",
            f"MASE {z['mase_mean']:.3f}",
            f"{100 * (z['mase_mean'] / best['mase_mean'] - 1):.1f}% off the best",
            delta_color="off",
        )
        if pd.notna(z.get("rung_seconds")) and pd.notna(best.get("rung_seconds")):
            c4.metric(
                "...for this much compute",
                f"{z['rung_seconds']:.0f} s",
                f"{best['rung_seconds'] / z['rung_seconds']:.0f}x less than the best",
                delta_color="off",
            )

    st.subheader("Point accuracy")
    st.caption(
        "MASE 1.0 is the seasonal naive floor. Below 1.0 the method earned its compute; "
        "above 1.0 it did not. Mean and median are both shown because the mean of a ratio "
        "over heterogeneous series is moved by a handful of intermittent ones."
    )
    mase_long = lad.melt(
        id_vars="model", value_vars=["mase_mean", "mase_median"], var_name="stat", value_name="mase"
    )
    st.altair_chart(
        alt.Chart(mase_long)
        .mark_bar()
        .encode(
            y=alt.Y("model:N", sort=order, title=None),
            x=alt.X("mase:Q", title="MASE"),
            yOffset="stat:N",
            color=alt.Color("stat:N", title=None, scale=alt.Scale(scheme="tableau20")),
            tooltip=["model", "stat", alt.Tooltip("mase:Q", format=".3f")],
        )
        .properties(height=320),
        use_container_width=True,
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Does it know how uncertain it is?")
        st.caption(
            "Coverage of the nominal 80 percent interval. Below the line the model is "
            "overconfident, and every safety buffer sized from it will be too small."
        )
        cov = lad[["model", "coverage_mean"]].copy()
        base = alt.Chart(cov).encode(y=alt.Y("model:N", sort=order, title=None))
        st.altair_chart(
            (
                base.mark_bar(color="#4c78a8").encode(
                    x=alt.X("coverage_mean:Q", title="coverage", scale=alt.Scale(domain=[0, 1])),
                    tooltip=["model", alt.Tooltip("coverage_mean:Q", format=".3f")],
                )
                + alt.Chart(pd.DataFrame({"t": [0.8]}))
                .mark_rule(color="#e45756", strokeDash=[4, 4], size=2)
                .encode(x="t:Q")
            ).properties(height=300),
            use_container_width=True,
        )
    with right:
        st.subheader("What the compute bought")
        st.caption(
            "Wall-clock seconds for the whole rung against its mean MASE, on one CPU. "
            "Down and to the left is better. The horizontal spread is the finding."
        )
        if "rung_seconds" in lad.columns and lad["rung_seconds"].notna().any():
            sc = lad.dropna(subset=["rung_seconds"])
            st.altair_chart(
                alt.Chart(sc)
                .mark_circle(size=200, opacity=0.85)
                .encode(
                    x=alt.X(
                        "rung_seconds:Q",
                        title="rung wall-clock (s, log scale)",
                        scale=alt.Scale(type="log"),
                    ),
                    y=alt.Y("mase_mean:Q", title="MASE", scale=alt.Scale(zero=False)),
                    color=alt.Color("model:N", sort=order, scale=BAND, legend=None),
                    tooltip=[
                        "model",
                        alt.Tooltip("mase_mean:Q", format=".3f"),
                        alt.Tooltip("rung_seconds:Q", format=".0f"),
                    ],
                )
                .properties(height=300),
                use_container_width=True,
            )
        else:
            st.info("No timings recorded: results/raw/timings.json is absent.")

    st.subheader("The full table")
    show = lad.rename(
        columns={
            "mase_mean": "MASE mean",
            "mase_median": "MASE median",
            "pct_beating_naive_floor": "beats floor %",
            "pinball_mean": "pinball",
            "coverage_mean": "coverage",
            "eur_per_series_day_at_cr": "EUR / series-day",
            "rung_seconds": "runtime (s)",
        }
    )
    cols = [
        c
        for c in [
            "model", "MASE mean", "MASE median", "beats floor %", "pinball",
            "coverage", "EUR / series-day", "runtime (s)", "mase_dropped",
        ]
        if c in show.columns
    ]
    st.dataframe(show[cols], use_container_width=True, hide_index=True)
    st.caption(
        "`mase_dropped` counts series-folds where the training window had a zero "
        "seasonal-naive scale, so MASE is undefined and the row was excluded rather than "
        "quietly averaged in."
    )


# --------------------------------------------------------------------------------------
# 2. Where complexity pays
# --------------------------------------------------------------------------------------

with tab_pays:
    st.subheader("Win rate against the floor, by how intermittent the series is")
    st.caption(
        "Each cell is the share of series in that quartile where the model's mean MASE "
        "beat the seasonal naive's. Read it left to right: this is the whole finding."
    )
    try:
        win_zero = published("win_rate_by_zero_share")
        win_vol = published("win_rate_by_volume")
        profile = published("floor_profile")
    except ex.DataUnavailable as err:
        fail(err)

    which = st.radio(
        "Cut the sample by",
        ["Intermittency (share of zero-demand days)", "Volume (mean daily demand)"],
        horizontal=True,
        label_visibility="collapsed",
    )
    win = win_zero if which.startswith("Intermittency") else win_vol
    order = ex.model_display_order(win["model"].unique())

    st.altair_chart(
        alt.Chart(win)
        .mark_rect()
        .encode(
            x=alt.X("bucket:N", title=which.split(" (")[0] + " quartile, low to high"),
            y=alt.Y("model:N", sort=order, title=None),
            color=alt.Color(
                "win_rate:Q",
                title="win rate",
                scale=alt.Scale(scheme="redyellowgreen", domain=[0, 1]),
            ),
            tooltip=[
                "model", "bucket", "n_series",
                alt.Tooltip("win_rate:Q", format=".1%"),
                alt.Tooltip("median_improvement:Q", format=".1%"),
            ],
        )
        .properties(height=40 * win["model"].nunique()),
        use_container_width=True,
    )

    st.altair_chart(
        alt.Chart(win)
        .mark_line(point=True)
        .encode(
            x=alt.X("bucket:N", title=None),
            y=alt.Y("win_rate:Q", title="win rate", axis=alt.Axis(format="%")),
            color=alt.Color("model:N", sort=order, scale=BAND, title=None),
            tooltip=["model", "bucket", alt.Tooltip("win_rate:Q", format=".1%")],
        )
        .properties(height=320),
        use_container_width=True,
    )

    st.subheader("The series where nothing beat the floor")
    cat = catalogue()
    n_holds = int(cat["floor_holds"].sum())
    st.markdown(
        f"On **{n_holds} of {len(cat)} series ({n_holds / len(cat):.1%})** every rung on "
        "the ladder had a worse mean MASE than the seven-line baseline. They are not "
        "random: they are the sparsest items in the sample."
    )
    st.dataframe(
        profile.rename(
            columns={
                "median_where_floor_holds": "median where floor holds",
                "median_where_floor_beaten": "median where floor beaten",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.altair_chart(
        alt.Chart(cat)
        .mark_circle(size=70, opacity=0.6)
        .encode(
            x=alt.X("zero_share:Q", title="share of zero-demand days"),
            y=alt.Y("mean_demand:Q", title="mean daily demand (log)", scale=alt.Scale(type="log")),
            color=alt.Color(
                "floor_holds:N",
                title="nothing beat the floor",
                scale=alt.Scale(domain=[False, True], range=["#b8c4d0", "#e45756"]),
            ),
            tooltip=[
                "unique_id",
                alt.Tooltip("zero_share:Q", format=".2f"),
                alt.Tooltip("mean_demand:Q", format=".2f"),
                alt.Tooltip("adi:Q", format=".2f", title="avg demand interval"),
                "best_model",
            ],
        )
        .properties(height=380),
        use_container_width=True,
    )
    st.caption(
        "Every sampled series. Red is a series where the correct engineering decision is "
        "to run the one-line baseline and spend the effort somewhere else."
    )


# --------------------------------------------------------------------------------------
# 3. One series at a time
# --------------------------------------------------------------------------------------

with tab_series:
    st.subheader("Every model's forecast against the actual, on one fold")
    st.caption(
        "An average over 300 series hides what the models are actually doing. This is the "
        "raw forecast each rung wrote, plotted against what happened."
    )
    cat = catalogue()

    f1, f2, f3 = st.columns(3)
    with f1:
        vol = st.multiselect(
            "Volume stratum", sorted(cat["volume_stratum"].unique()), placeholder="any"
        )
    with f2:
        inter = st.multiselect(
            "Intermittency stratum",
            sorted(cat["intermittency_stratum"].unique()),
            placeholder="any",
        )
    with f3:
        only_floor = st.checkbox(
            "Only series where nothing beat the floor",
            help="The 11 hardest series in the sample.",
        )

    sub = cat
    if vol:
        sub = sub[sub["volume_stratum"].isin(vol)]
    if inter:
        sub = sub[sub["intermittency_stratum"].isin(inter)]
    if only_floor:
        sub = sub[sub["floor_holds"]]

    if sub.empty:
        st.warning("No series match those filters.")
        st.stop()

    uid = st.selectbox(
        f"Series ({len(sub)} match)",
        sorted(sub["unique_id"]),
        format_func=lambda s: f"{s}  —  won by {cat.set_index('unique_id').loc[s, 'best_model']}",
    )
    row = cat.set_index("unique_id").loc[uid]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Mean daily demand", f"{row['mean_demand']:.2f}")
    m2.metric("Zero-demand days", f"{row['zero_share']:.0%}")
    m3.metric("Avg gap between sales", f"{row['adi']:.1f} d")
    m4.metric(
        "Best model here",
        row["best_model"],
        f"MASE {row['best_mase']:.2f} vs floor {row['floor_mase']:.2f}",
        delta_color="off",
    )

    all_models = ex.available_models()
    cutoffs = sorted(raw_forecasts()["cutoff"].unique())
    s1, s2 = st.columns([1, 2])
    with s1:
        cutoff = st.selectbox(
            "Fold (cutoff = last day of training)",
            cutoffs,
            index=len(cutoffs) - 1,
            format_func=lambda d: pd.Timestamp(d).date().isoformat(),
        )
    with s2:
        chosen = st.multiselect(
            "Models", all_models, default=[all_models[0], FLOOR], max_selections=4
        )
    show_band = st.checkbox("Show the 80 percent interval", value=True)

    ctx, sel = ex.series_fold_view(uid, cutoff, chosen or None)
    if sel.empty:
        st.warning("No forecasts for that combination.")
    else:
        order = ex.model_display_order(sel["model"].unique())
        actual = (
            alt.Chart(ctx)
            .mark_line(color="#333", size=1.6, point=alt.OverlayMarkDef(size=18, color="#333"))
            .encode(
                x=alt.X("ds:T", title=None),
                y=alt.Y("y:Q", title="units sold"),
                tooltip=["ds:T", "y:Q"],
            )
        )
        cutline = (
            alt.Chart(pd.DataFrame({"ds": [pd.Timestamp(cutoff)]}))
            .mark_rule(color="#e45756", strokeDash=[5, 5], size=2)
            .encode(x="ds:T")
        )
        layers = [actual, cutline]
        if show_band and len(order) <= 2:
            layers.append(
                alt.Chart(sel)
                .mark_area(opacity=0.18)
                .encode(
                    x="ds:T",
                    y=alt.Y("q10:Q", title="units sold"),
                    y2="q90:Q",
                    color=alt.Color("model:N", sort=order, scale=BAND, title=None),
                )
            )
        layers.append(
            alt.Chart(sel)
            .mark_line(size=2)
            .encode(
                x="ds:T",
                y=alt.Y("point:Q", title="units sold"),
                color=alt.Color("model:N", sort=order, scale=BAND, title=None),
                tooltip=[
                    "model", "ds:T",
                    alt.Tooltip("point:Q", format=".2f"),
                    alt.Tooltip("q10:Q", format=".2f"),
                    alt.Tooltip("q90:Q", format=".2f"),
                ],
            )
        )
        st.altair_chart(
            alt.layer(*layers).resolve_scale(y="shared").properties(height=400),
            use_container_width=True,
        )
        st.caption(
            "Black is the actual. The dashed line is the cutoff: everything to its left is "
            "what the model trained on, everything to its right it had never seen. "
            + (
                "Intervals are drawn for one or two models at a time; they overlap into mud "
                "beyond that."
                if len(order) > 2
                else ""
            )
        )

    st.markdown("**Published scores for this series**, one row per model and fold.")
    scores = ex.per_series_scores(uid)
    pivot = scores.pivot_table(
        index="model", columns="cutoff", values="mase", observed=True
    ).round(3)
    pivot.columns = [pd.Timestamp(c).date().isoformat() for c in pivot.columns]
    pivot["mean"] = pivot.mean(axis=1).round(3)
    st.dataframe(
        pivot.style.background_gradient(cmap="RdYlGn_r", axis=None),
        use_container_width=True,
    )


# --------------------------------------------------------------------------------------
# 4. Run the floor yourself
# --------------------------------------------------------------------------------------

with tab_run:
    st.subheader("Move the protocol and watch the floor move")
    st.markdown(
        "The seasonal naive is the one rung this app re-runs, because it is the one this "
        "repository implements rather than imports, and because it is the denominator of "
        "every MASE above. Change the settings and it is re-forecast and re-scored end to "
        "end through `runner.score`, the same function every published number came from. "
        "The other rungs need torch, LightGBM and a pretrained transformer; re-fitting "
        "those on a page load would be a demo, not a comparison."
    )

    cat = catalogue()
    c1, c2, c3, c4 = st.columns(4)
    horizon = c1.number_input("Horizon (days)", 7, 56, PROTOCOL.horizon, step=7)
    n_windows = c2.number_input("Folds", 1, 6, PROTOCOL.n_windows)
    season = c3.number_input("Season length (days)", 2, 28, PROTOCOL.season_length)
    n_series = c4.number_input("Series", 1, len(cat), 50, step=25)

    proto = Protocol(
        horizon=int(horizon), n_windows=int(n_windows), season_length=int(season)
    )
    st.code(proto.describe().replace(", ", ",\n"), language=None)

    pick = st.radio(
        "Which series",
        ["Spread across the strata", "The most intermittent", "The densest"],
        horizontal=True,
    )
    if pick == "The most intermittent":
        chosen_ids = cat.nlargest(int(n_series), "zero_share")["unique_id"]
    elif pick == "The densest":
        chosen_ids = cat.nsmallest(int(n_series), "zero_share")["unique_id"]
    else:
        chosen_ids = cat.sample(int(n_series), random_state=0)["unique_id"]

    if st.button("Run the floor", type="primary"):
        try:
            per_fold, summary = floor_backtest(
                tuple(sorted(chosen_ids)), int(horizon), int(n_windows), int(season)
            )
        except ValueError as err:
            st.error(str(err))
        else:
            r = summary.iloc[0]
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("MASE mean", f"{r['mase_mean']:.3f}")
            k2.metric("MASE median", f"{r['mase_median']:.3f}")
            k3.metric(
                "Coverage",
                f"{r['coverage_mean']:.3f}",
                f"{r['coverage_mean'] - r['coverage_target']:+.3f} vs nominal",
                delta_color="off",
            )
            k4.metric("Series-folds scored", int(r["mase_defined_on"]))

            if r["mase_dropped"]:
                st.warning(
                    f"{int(r['mase_dropped'])} series-folds had an undefined MASE (a flat "
                    "or all-zero training window) and were excluded, not averaged in as "
                    "zero."
                )
            st.dataframe(summary.round(4), use_container_width=True, hide_index=True)

            st.caption(
                "MASE across the selected series, one point per series-fold. The floor "
                "scores 1.0 against itself only on average and in sample; per fold it "
                "spreads widely, which is why the comparison uses 1200 of these and not "
                "one number."
            )
            st.altair_chart(
                alt.Chart(per_fold.dropna(subset=["mase"]))
                .mark_bar(color="#4c78a8")
                .encode(
                    x=alt.X("mase:Q", bin=alt.Bin(maxbins=50), title="MASE"),
                    y=alt.Y("count():Q", title="series-folds"),
                )
                .properties(height=260),
                use_container_width=True,
            )
            st.download_button(
                "Download these per-fold scores (CSV)",
                per_fold.to_csv(index=False).encode("utf-8"),
                file_name=f"floor_h{horizon}_w{n_windows}_m{season}.csv",
                mime="text/csv",
            )
    else:
        st.info("Set the protocol above, then run it. A 50-series run takes a few seconds.")


# --------------------------------------------------------------------------------------
# 5. The business case
# --------------------------------------------------------------------------------------

with tab_money:
    st.subheader("What the forecast is worth, and how much that depends on what you assume")
    st.markdown(
        "M5 ships unit sales and no prices, so **every euro below is an assumption**. The "
        "point of putting them behind sliders is that you can find out for yourself how "
        "far the ranking moves when the assumptions do. Ordering `q` against demand `y` "
        "costs `overstock x max(0, q-y) + stockout x max(0, y-q)`, and the cost-minimising "
        "order is the **critical-ratio quantile** of the predictive distribution, not its "
        "middle — which is the argument for forecasting a distribution at all."
    )

    c1, c2, c3 = st.columns(3)
    price = c1.slider("Unit price (EUR)", 0.5, 20.0, 4.50, 0.25)
    margin = c2.slider("Gross margin", 0.05, 0.60, 0.28, 0.01)
    holding = c3.slider("Annual holding rate", 0.0, 0.60, 0.25, 0.05)
    c4, c5 = st.columns(2)
    write_off = c4.slider(
        "Write-off fraction of surplus",
        0.0,
        0.90,
        0.30,
        0.05,
        help="The share of surplus stock marked down or wasted. The assumption the whole "
        "case is most sensitive to, which is why it is first.",
    )
    review = c5.slider("Review period (days)", 7, 56, 28, 7)

    try:
        econ, described, cr = live_costs(price, margin, holding, write_off, review)
    except ex.DataUnavailable as err:
        fail(err)

    st.metric(
        "Critical ratio (the cost-optimal service level)",
        f"{cr:.3f}",
        help="stockout / (stockout + overstock). Order this quantile of the forecast "
        "distribution, not the median.",
    )
    st.code(described.replace("; ", ";\n"), language=None)

    if cr > 0.9:
        st.warning(
            "The critical ratio is above the highest quantile this project forecast "
            "(0.90), so every model's order is truncated at its own 90th percentile. The "
            "comparison between models stays fair; the absolute euros are conservative for "
            "all of them equally."
        )

    order = ex.model_display_order(econ["model"].unique())
    st.altair_chart(
        alt.Chart(econ)
        .mark_bar()
        .encode(
            y=alt.Y("model:N", sort=order, title=None),
            x=alt.X("eur_per_series_day_at_cr:Q", title="EUR per series-day (lower is better)"),
            color=alt.Color("model:N", sort=order, scale=BAND, legend=None),
            tooltip=[
                "model",
                alt.Tooltip("eur_per_series_day_at_cr:Q", format=".4f", title="at critical ratio"),
                alt.Tooltip("eur_per_series_day_at_median:Q", format=".4f", title="at median"),
            ],
        )
        .properties(height=320),
        use_container_width=True,
    )

    st.dataframe(
        econ[
            [
                "model", "series_days", "eur_per_series_day_at_cr",
                "eur_per_series_day_at_median", "distribution_value_per_series_day",
            ]
        ].round(4),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "`distribution_value_per_series_day` is what ordering at the critical ratio saved "
        "over ordering at the median. It is **negative for every model** in the published "
        "run, and that is finding four: the newsvendor identity holds only when the "
        "predictive distribution is calibrated, and most of these over-cover. A quantile "
        "you cannot trust is worse than a point forecast, because somebody will size a "
        "buffer from it."
    )

    st.subheader("Does the ranking survive the assumptions moving?")
    if st.button("Re-cost across the assumption grid"):
        sens = live_sensitivity()
        wins = sens[sens["is_best"]].groupby("model").size().rename("times cheapest")
        g1, g2 = st.columns([1, 2])
        with g1:
            st.dataframe(wins.reset_index(), use_container_width=True, hide_index=True)
        with g2:
            st.altair_chart(
                alt.Chart(sens)
                .mark_rect()
                .encode(
                    x=alt.X("write_off_fraction:O", title="write-off fraction"),
                    y=alt.Y("gross_margin:O", title="gross margin"),
                    color=alt.Color(
                        "eur_per_series_day:Q", title="EUR/series-day", scale=alt.Scale(scheme="viridis")
                    ),
                    facet=alt.Facet("model:N", columns=4, sort=order, title=None),
                    tooltip=[
                        "model",
                        alt.Tooltip("critical_ratio:Q", format=".3f"),
                        alt.Tooltip("eur_per_series_day:Q", format=".4f"),
                    ],
                )
                .properties(width=110, height=110),
                use_container_width=True,
            )
        st.caption(
            "Nine cost worlds, from a 10 percent write-off at a 15 percent margin to a 60 "
            "percent write-off at 40. If one model is cheapest in most of them, the "
            "conclusion is a property of the forecasts rather than of the assumptions."
        )
    else:
        st.info(
            "Re-costs all eight models across nine combinations of write-off fraction and "
            "gross margin. Takes a few seconds."
        )

st.divider()
st.caption(
    "Not a claim about forecasting in general: one dataset, one domain, one horizon, 300 "
    "of M5's 30,490 series. Not a full M5 benchmark and not comparable to its leaderboard. "
    "Every euro figure is an assumption, labelled as one. "
    "Code MIT; M5 data © the M Open Forecasting Center, used for research and benchmarking."
)
