# forecast-ladder

**A forecasting comparison on real retail demand where the question is not which model wins,
but where model complexity stops paying for itself.**

Five rungs, from a seven-line seasonal naive up to a pretrained transformer forecasting
zero-shot, all scored under one protocol that was written down before any of them ran.

[GitHub](https://github.com/9117KET/forecast-ladder) · Built by
[Kinlo Ephriam Tangiri](https://www.kinloephraim.com/) · Walkthrough:
[`docs/WALKTHROUGH.md`](docs/WALKTHROUGH.md)

---

## What this is

A forecasting team can get another repository that fits a model and reports an accuracy number
anywhere. What is harder to find, and what this is, is a comparison run to a protocol strict
enough that its conclusions survive being questioned:

- **Rolling-origin backtesting**, four non-overlapping folds, not one holdout split.
- **A seasonal naive floor** that every other rung has to beat, and the floor is implemented
  here rather than imported, because every MASE in the results divides by a number derived
  from it.
- **MASE** as the headline metric, scaled on the training window, never the test window.
- **Pinball loss and interval coverage**, so a forecast is judged on whether it knows how
  uncertain it is, not only on where it points.
- **A costed business case** that converts forecast error into euros through the newsvendor
  problem, with the assumptions labelled and a sensitivity grid over the one that dominates.

The ladder:

| Rung | Method | What it adds |
| --- | --- | --- |
| 0 | Seasonal naive | The floor. Last Tuesday, as this Tuesday |
| 1 | ETS, AutoARIMA | Classical statistical structure, fitted per series |
| 2 | LightGBM | One global gradient-boosted model on lag and calendar features |
| 3 | N-HiTS, PatchTST | Deep learning: multi-rate sampling, and a patched transformer |
| 4 | Chronos-Bolt | A pretrained foundation model, zero-shot, no fitting at all |

## What this is not

- **Not a claim about forecasting in general.** One dataset, one domain, one horizon.
- **Not a full M5 benchmark.** 300 of 30,490 series, stratified and frozen before any run.
  The M5 leaderboard is not comparable to anything here.
- **Not a tuned-to-death comparison.** Every rung got a fixed, documented budget. The neural
  models in particular would improve with more compute than one CPU.
- **Not German retail.** Rossmann was the first choice and could not be fetched without a
  personal Kaggle token. See [`data/README.md`](data/README.md) for why, and for how little
  would need to change to add it.
- **Not measured economics.** M5 ships no prices or costs. Every euro figure rests on stated
  assumptions, and the sensitivity grid exists because of that.
- **Not deployed.** There is nothing to click. This is a batch comparison; the deliverable is
  the finding and the code that produced it.

---

## The protocol

Declared in `src/forecast_ladder/protocol.py`, which was the second file written in this
repository and predates every model in it.

| Setting | Value | Why |
| --- | --- | --- |
| Horizon | 28 days | The M5 horizon, so the numbers sit against published work. Also four complete weekly cycles |
| Folds | 4, rolling origin | Measures the method, not one month of calendar |
| Step | 28 days | A full horizon, so test windows never overlap and folds stay independent |
| Season | 7 days | Weekly demand cycle |
| Quantiles | 0.1, 0.5, 0.9 | A nominal 80 percent interval, fixed in advance |
| Training history | Most recent 842 days | Two full annual cycles minimum per fold. See `data.HISTORY_DAYS` |
| Series | 300, stratified, frozen | Volume crossed with intermittency, ids committed |

**Why the order of work matters.** The evaluation harness is the first module and it has tests
against values computed by hand. If the models had come first, the protocol would have been
adjusted a fold and a metric at a time until it agreed with results already on screen, and
nobody doing it would have noticed.

---

## Results

Written by `scripts/analyse.py` into `results/published/`. `headline.json` is the single source
every figure in this README, the walkthrough and the case study is quoted from.

<!-- RESULTS:START -->
_Run `python -u scripts/run_ladder.py` then `python -u scripts/analyse.py` to populate this
section._
<!-- RESULTS:END -->

| File | Contents |
| --- | --- |
| `ladder.csv` | One row per model: MASE, pinball, coverage, cost, wall-clock |
| `per_fold.csv` | One row per model, series and fold |
| `beats_floor.csv` | Per model and series: did it beat the naive, and by how much |
| `win_rate_by_zero_share.csv` | Win rate against intermittency quartile |
| `win_rate_by_volume.csv` | Win rate against volume quartile |
| `floor_profile.csv` | What the series nothing beats have in common |
| `economics.csv` | Newsvendor cost per model |
| `sensitivity.csv` | Whether the ranking survives the cost assumptions moving |
| `sample_series.csv` | The 300 selected series and their descriptors |
| `headline.json` | The numbers every write-up quotes |

---

## Running it

Python 3.10 or newer. Built and run on 3.12.6, Windows, CPU only.

```bash
git clone https://github.com/9117KET/forecast-ladder
cd forecast-ladder

# torch first, into the base interpreter, then a venv that reuses it.
# Installing torch into a deeply nested .venv path on Windows fails with WinError 206
# (path too long), because torch ships dist-info paths hundreds of characters deep.
pip install torch==2.2.2
python -m venv .venv --system-site-packages
.venv/Scripts/python -m pip install -r requirements.txt   # Linux/macOS: .venv/bin/python

# 1. fetch M5 and freeze the 300-series sample (~500 MB download, once)
.venv/Scripts/python -u scripts/download_data.py

# 2. cache the sampled, trimmed panel as parquet
.venv/Scripts/python -u scripts/prepare_panel.py

# 3. run the rungs; each writes results/raw/<rung>.parquet as it finishes
.venv/Scripts/python -u scripts/run_ladder.py

# 4. score, analyse, cost, publish
.venv/Scripts/python -u scripts/analyse.py
```

Useful variants:

```bash
# a single rung, or a dry run on fewer series
python -u scripts/run_ladder.py --rungs naive,gbm
python -u scripts/run_ladder.py --series 25

# size a rung before committing to the full run
python -u scripts/time_rung.py classical 12
```

`run_ladder.py` writes each rung to disk as it completes, so a rung can be re-run alone and a
failure in one does not lose the others. Scoring is a separate script on purpose: the compute
is expensive and the analysis is not.

### Tests

```bash
.venv/Scripts/python -m pytest tests/ -q
```

**101 tests.** They cover the metrics against hand-computed values (including the asymmetry of
pinball loss in both directions), the fold arithmetic and three separate leakage checks, the
seasonal naive's weekday alignment and interval widening, the normalisation of each library's
output, and a brute-force verification that newsvendor cost really is minimised at the
critical-ratio quantile.

Expected numbers in the test suite were worked out on paper, not captured from a run. That
distinction is the only thing that makes a test of a metric worth having.

---

## Layout

```
src/forecast_ladder/
  protocol.py     the evaluation rules, frozen, written before any model
  metrics.py      MASE, pinball loss, coverage, implemented not imported
  baselines.py    the seasonal naive floor, point and probabilistic
  runner.py       one scoring path shared by every rung
  rungs.py        the five rungs; none of them computes a metric
  data.py         M5 loading, series descriptors, the stratified sample
  economics.py    newsvendor cost and the critical-ratio identity
  analysis.py     where the complexity pays and where the floor holds
scripts/
  download_data.py  fetch M5, freeze the sample
  prepare_panel.py  cache the sampled panel
  run_ladder.py     run the rungs, write raw forecasts
  analyse.py        score, analyse, cost, publish
  time_rung.py      size one rung before the full run
docs/WALKTHROUGH.md  how to defend every choice in here
```

---

## What I would do next

1. **Add Rossmann** and check whether the finding survives on a second dataset. Small change:
   one loader, everything else untouched.
2. **Add Croston's method and a specialised intermittent-demand model** as an intermediate
   rung. On a sample that averages 60 percent zero days, those methods exist for exactly these
   series, and leaving them out is the biggest gap in the ladder. ADI and the squared
   coefficient of variation are already recorded per series, which is the standard basis for
   deciding where they apply.
3. **Put the backtest in CI** so a change in accuracy fails a build rather than being noticed a
   quarter later, and serve the winning configuration behind an API.

Not doing these yet is a scope decision, not an oversight: the point of this build was one
honest comparison, and each of the three above is a project rather than an afternoon.

---

## Licence

Code: MIT, see [`LICENSE`](LICENSE). The M5 data is not redistributed here; see
[`data/README.md`](data/README.md).
