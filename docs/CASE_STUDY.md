# Case study: where forecasting complexity stops paying

**Kinlo Ephriam Tangiri, August 2026.** Code and results:
[github.com/9117KET/forecast-ladder](https://github.com/9117KET/forecast-ladder).
Every figure here is read from `results/published/headline.json` and `ladder.csv`.

---

## The problem, and who has it

A retailer with tens of thousands of item-store combinations has to decide, for each one, how
much to order. The forecast behind that decision can be a one-line rule or a pretrained
transformer, and the cost of those two options differs by orders of magnitude in compute,
engineering time and maintenance.

The question nobody answers before choosing is which items justify which. Published
comparisons report an average accuracy across a benchmark and declare a winner, which is not
a decision anybody can act on: the average conceals that the winner may lose on half the
catalogue.

## What was built

A comparison of five method families on 300 real store-item demand series from the M5
competition data, over a 4-fold rolling-origin backtest at a 28-day horizon, all scored under
one protocol fixed before any model ran.

The ladder runs from a seasonal naive (this Tuesday equals last Tuesday), through ETS and
AutoARIMA fitted per series, to a global LightGBM on lag and calendar features, to N-HiTS and
PatchTST trained with a quantile loss, to Chronos-Bolt used zero-shot with no fitting on this
data at all. Accuracy is MASE scaled on the training window; the probabilistic side is pinball
loss plus empirical coverage of a nominal 80 percent interval; and the whole comparison is then
re-expressed in euros through the newsvendor problem.

## Results

| Model | MASE mean | MASE median | Beats naive | Coverage (target 80%) | Runtime | EUR / series-day |
| --- | --- | --- | --- | --- | --- | --- |
| Chronos-Bolt, zero-shot | **1.040** | **0.790** | **95.0%** | **79.7%** | **37 s** | **1.195** |
| AutoARIMA | 1.215 | 0.942 | 77.0% | 86.2% | 488 s | 1.306 |
| AutoETS | 1.216 | 0.931 | 77.7% | 85.7% | 488 s | 1.286 |
| Seasonal naive | 1.383 | 1.073 | floor | 85.5% | 13 s | 1.523 |
| LightGBM | 2.327 | 1.190 | 50.7% | 57.6% | 55 s | 1.424 |

Three findings, in order of how much they change a decision.

**1. A model that had never seen this data beat every model fitted to it.** Chronos-Bolt was
best on point accuracy, best calibrated (79.7 percent coverage against an 80 percent target,
where every fitted method over-covered), and cheapest to run of everything except the naive:
37 seconds against 488 for the classical rung. It has no fitting step. The compute was paid
once, by somebody else, during pretraining.

**2. The gradient-boosted model, which is the method that won M5, came last.** Its mean MASE of
2.33 is worse than doing nothing clever, and its 57.6 percent coverage means its intervals are
badly overconfident: an 80 percent interval that contains the actual 58 percent of the time
will under-size every safety buffer computed from it. It was fitted globally with lags at 1, 7,
14, 21 and 28 days and calendar features, which is the standard recipe.

**3. Whether complexity pays is decided by intermittency, monotonically.** Win rate against the
naive floor, by quartile of the share of zero-demand days:

| Zero-day share | AutoETS | AutoARIMA | LightGBM | Chronos-Bolt |
| --- | --- | --- | --- | --- |
| 0.04 to 0.43 (dense) | 93.3% | 89.3% | 93.3% | 97.3% |
| 0.43 to 0.63 | 88.0% | 85.3% | 74.7% | 96.0% |
| 0.63 to 0.78 | 69.3% | 69.3% | 28.0% | 97.3% |
| 0.78 to 0.97 (sparse) | 60.0% | 64.0% | **6.7%** | 89.3% |

LightGBM goes from beating the baseline on 93 percent of dense series to 6.7 percent of sparse
ones. The classical methods degrade too, from roughly 90 percent to roughly 62. Only the
foundation model holds up.

On 12 of the 300 series (4.0 percent) nothing on the ladder beat the naive at all. Those series
are the slow, sparse tail: median zero-day share 0.86 against 0.62 elsewhere, and a median gap
between sales of 8.5 days against 2.6.

## The business case

Forecast error is not what a retailer pays. The decision is the order quantity, and the cost is
asymmetric: surplus is carried and partly written off, shortfall loses the margin on a sale.

    cost = overstock_cost x max(0, order - demand) + stockout_cost x max(0, demand - order)

**The identity that makes the probabilistic forecast load-bearing.** The cost-minimising order
is neither the mean nor the median forecast. It is the critical-ratio quantile

    q* = stockout_cost / (stockout_cost + overstock_cost)

so a point forecast cannot answer the question at all, however accurate. Minimising pinball loss
at `q*` is, up to a constant, the same as minimising this cost: the statistical metric and the
euro figure are one quantity in two units.

### Assumptions, all of them labelled

| Quantity | Value | Status |
| --- | --- | --- |
| Unit price | EUR 4.50 | **Assumed.** M5 ships no price data |
| Gross margin | 28% | **Assumed.** Grocery and household goods rule of thumb |
| Unit cost | EUR 3.24 | Derived from price and margin |
| Stockout cost | EUR 1.26 / unit | Derived: lost gross margin. Conservative, excludes goodwill and substitution |
| Annual carrying rate | 25% | **Assumed.** Standard textbook figure |
| Write-off fraction of surplus | 30% | **Assumed. This is the dominant assumption** |
| Overstock cost | EUR 1.03 / unit | Derived: EUR 0.062 carrying over 28 days plus EUR 0.97 write-off |
| Critical ratio | 0.549 | Derived |

**The assumption the whole case is most sensitive to is the write-off fraction**, and the reason
is worth stating because getting it wrong the first time produced nonsense. Costing surplus at
carrying cost alone gives a critical ratio of 0.998, meaning "order the 99.8th percentile of
everything", which no grocer does. The error was treating a repeated stocking decision as if
surplus could be carried indefinitely at the cost of capital. Once a write-off share is included
the ratio lands at 0.549, which is a service level a planner would recognise.

### What it is worth

Against the seasonal naive, the best model saves **EUR 0.327 per series-day**
(1.523 against 1.195). Across the 300-series sample and its 33,600 forecast series-days that is
about **EUR 11,000**, and the ratio scales: on a 30,000-series catalogue it is roughly
**EUR 3.6 million a year**, on these assumptions.

That number should be read with its assumptions attached and with one more caveat: it is the
gap between the best method and doing nothing, not a realisable saving, because it assumes the
order quantity is set from the forecast with no other constraint, no minimum order size, no
shelf capacity and no supplier lead time.

**Sensitivity.** Re-costing the whole comparison across a 3 x 3 grid of write-off fraction (10,
30, 60 percent) and gross margin (15, 28, 40 percent), Chronos-Bolt is cheapest in 8 of the 9
cells and AutoETS in 1. The ranking is a property of the forecasts, not of the cost assumptions.

### The result that argues against the theory

`economics.csv` reports the cost of ordering at the critical ratio *and* at the median, and for
every model on the ladder **ordering at the critical ratio came out slightly more expensive**,
by between EUR 0.007 and EUR 0.109 per series-day.

That is not a contradiction of the identity, it is a demonstration of its precondition. The
identity holds when the forecast distribution is calibrated. Most of these models over-cover:
the naive's nominal 80 percent interval actually contains the actual 85.5 percent of the time,
the library naive's 90.5 percent. An over-wide interval puts q90 too high, so interpolating
upward from the median over-orders and costs money. The penalty is smallest, by a factor of
seven, for the one model whose coverage is nearly exact.

**The practical reading: a quantile you cannot trust is worse than a point forecast, because
somebody will size a buffer from it.** Calibration is not a diagnostic to report at the end, it
is the thing that decides whether the distribution is usable at all.

## Limitations

Written here rather than left for a reader to find.

- **One dataset, one domain, one horizon.** M5 is Walmart US grocery and household goods.
  Rossmann was the first choice, for German retail, and needs a personal Kaggle token; the
  loaders are written so adding it is a small change.
- **300 of 30,490 series**, stratified across volume and intermittency, frozen with ids
  committed before any model ran. Not comparable to the M5 leaderboard.
- **The most recent 842 days**, not the full 5.3 years. A five-year-old observation of a
  supermarket item is weak evidence about next month, and trimming also cut the compute by 2.3
  times. Both reasons are real.
- **AutoARIMA's search space is bounded** for tractability. If it lost, part of the reason may
  be that it was not allowed to look further.
- **The neural tier had a fixed step budget on one CPU** with no early stopping.
- **No intermittent-demand specialist was tested.** On a sample averaging 60 percent zero days,
  Croston's method and its variants exist for exactly these series, and their absence is the
  largest gap in the ladder.
- **Only three quantiles were forecast**, so a critical ratio above 0.9 would sit in a tail this
  project never estimated. Every model is truncated at its own q90, which biases the euro
  figures conservatively and equally.
- **Every cost is assumed. None is measured.**

## What I would do next, and why I have not

Add Rossmann, so the finding is checked on a second dataset rather than asserted from one. Add
Croston and a specialised intermittent method, which is where the sparse tail should be
attacked. Then put the backtest in CI so an accuracy regression fails a build.

Each of those is a project rather than an afternoon, and the purpose of this build was one
comparison done properly rather than three done partly.
