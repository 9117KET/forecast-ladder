# Walkthrough: what this project does and how to defend every part of it

This doubles as the script for the recorded demo and as preparation for being asked about it.
Nothing here can be answered by naming a library, which is the test each section is written
against.

Every number here is read from `results/published/headline.json` and `ladder.csv`, so this
file, the README and the case study always quote the same figures. Re-run `scripts/analyse.py`
and they all move together.

---

## 1. The 30-second version

A forecasting comparison on real retail demand where the point was not to find the best
model, but to find out where model complexity stops paying for itself.

The headline: a pretrained model that had never seen the data came within one percent of the
best model trained on it (MASE 1.040 against 1.032) using a nineteenth of the compute, and
LightGBM, the family that won the M5 competition, came last.

Five rungs, from a one-line seasonal naive up to a pretrained transformer, all scored under
one protocol declared before any of them ran. The result is a rule about which items deserve
a sophisticated forecast and which do not, and the euro value of the difference.

---

## 2. The protocol, and why each choice is the way it is

### Rolling origin, not a single holdout

A single train/test split scores a method on one period. If the last 28 days happened to
contain an unusual promotion or a stockout, that is what gets measured. Four origins, each
forecasting 28 days forward from a different cutoff, measure the method rather than the
calendar.

The origins step by a full horizon so the test windows never overlap. Overlapping windows
reuse the same actuals across folds, which correlates the folds and makes the average look
more precise than it is.

### MASE, not MAPE

Three reasons, and any one of them is disqualifying for MAPE on retail data.

**MAPE divides by the actual.** A zero-demand day makes it infinite; a one-unit day makes it
enormous. The sample here averages 60 percent zero days, so MAPE is undefined or unstable
almost everywhere.

**MAPE is asymmetric in the wrong direction.** Forecasting below the actual can only ever be
100 percent wrong. Forecasting above it is unbounded. So a model that systematically
under-forecasts scores better, which is exactly backwards for a retailer.

**MASE is scale-free by construction.** It divides absolute error by the in-sample mean
absolute error of a seasonal naive on the same series. A value of 1.0 means "as accurate as
last week repeated". Below 1.0 is better; above 1.0 means the method has not earned its
compute.

The detail that matters: **the scaling denominator comes from the training window, never the
test window.** Scaling on test data would make the denominator a property of the split, and
two models evaluated on different splits would stop being comparable. `metrics.mase` takes
the scale as an argument specifically so a caller has to fetch the training scale on purpose.

### Pinball loss, and coverage as the audit

Pinball loss scores a set of quantiles rather than a single number, so it rewards a forecast
that knows how uncertain it is. At quantile `q`, being under the actual costs `q` per unit
and being over costs `1 - q`. At `q = 0.9` that asymmetry is 9 to 1, so the loss is minimised
by a forecast sitting near the 90th percentile of the predictive distribution.

Coverage is the separate honesty check. If the nominal 80 percent interval contains the
actual only 60 percent of the time, the model is overconfident. Pinball loss alone will not
surface that in a form anyone can act on, and an interval nobody can trust is worse than no
interval, because somebody will size a safety buffer from it.

### What is deliberately not equalised

Tuning effort. The neural models get a fixed step budget and default architectures; the
gradient-boosted model gets hand-chosen lag and calendar features; the classical models
choose their own orders. **That asymmetry favours the complex end of the ladder**, so if a
simple method still wins, the result is not an artefact of neglect. The one place it cuts the
other way is AutoARIMA, whose search space is bounded for tractability, and that is stated
in the code and in the README.

---

## 3. The models, one level below the name

### Seasonal naive (rung 0)

Forecast next Tuesday as last Tuesday. For step `i` ahead it reads the observation
`floor(i / 7) + 1` weeks back, so a 28-day horizon uses anchors one, two, three and four
weeks old.

Its intervals come from the empirical distribution of its own in-sample errors at the
matching lag, computed separately per lag. That is why the intervals widen with horizon: a
four-week-old anchor is a worse anchor than a one-week-old one. No distributional assumption
is made, which matters because errors on intermittent demand are nowhere near Gaussian.

### ETS (rung 1)

Exponential smoothing. It maintains three things and updates each with a weighted average of
the newest observation and the current estimate: a **level** (where the series is now), a
**trend** (where it is heading), and a **seasonal** set of seven factors (how each weekday
differs from the level). The smoothing weights decide how fast each component forgets. `AutoETS`
tries the additive and multiplicative combinations and picks by information criterion.

The one-sentence version: a structural description of the series that is continuously
corrected as new data arrives.

### ARIMA (rung 1)

Models the series as a function of its own past values and its own past errors.
**AR** terms regress today on recent observations; **MA** terms regress today on recent
forecast errors; **I** is the differencing applied first to remove a trend and make the
series stationary, because the machinery assumes stationarity. The seasonal version adds the
same three at a lag of 7.

`AutoARIMA` searches orders stepwise and selects by information criterion. Here the search is
bounded (first differences only, at most two non-seasonal and one seasonal term per side)
because unbounded it took 441 seconds on 12 series, and that is roughly three hours across
the sample.

### LightGBM (rung 2)

Gradient-boosted trees. It builds many shallow decision trees in sequence, each fitted to the
residual error the previous ones left behind, and sums them.

Two things about applying it to time series:

**It has no notion of time.** A tree sees a row of features, not a sequence. So the
seasonality that ETS models explicitly has to be handed over as features: lags at 1, 7, 14,
21 and 28 days, and calendar columns for day of week, day, month and week.

**It is fitted globally**, one model across all series, not one per series. That is both the
industry pattern and the only thing that makes boosting viable here, because a single series
of 842 mostly-zero days does not contain enough signal to fit a tree ensemble. Fitting
globally lets the model learn patterns that recur across items.

Its intervals are conformal: calibrated on held-out residuals rather than assumed.

### N-HiTS (rung 3)

A neural architecture built on the observation that a series contains structure at several
time scales at once. It stacks blocks, and each block **samples the input at a different
rate** and projects onto a different resolution, so one block can learn a slow annual shape
while another learns the day-of-week wiggle. The blocks are combined by residual connections:
each subtracts what it explained from what the next one sees. Cheaper than a transformer
because the heavy lifting is interpolation rather than attention.

### PatchTST (rung 3)

A transformer for time series, and the idea it borrowed from language models is the useful
part. Instead of one token per time step, it cuts the series into **patches** of consecutive
steps (16 here, stride 8) and treats each patch as a token. Two consequences: attention
operates over far fewer tokens, so long inputs become affordable; and a patch carries local
shape, whereas a single time step carries almost no information on its own. It also keeps
channels independent, forecasting each series with shared weights rather than mixing series.

### Chronos, zero-shot (rung 4)

**This is the question most likely to be asked, because the posting marks it as a plus.**

"Zero-shot forecasting" sounds like a contradiction, so state the mechanism. Chronos is a
transformer pretrained on a large, diverse corpus of time series from many domains. At
inference it receives the history of a series it has never seen and emits a forecast directly.
There is no fitting step on this data at all.

Why that can work: the pretraining taught it the *shapes* time series take, seasonal cycles,
trends, level shifts, intermittency, and how those shapes tend to continue. A classical model
estimates parameters from the one series in front of it. Chronos does the analogous work in
its forward pass, using patterns learned from other series. It is the same transfer-learning
argument that lets a pretrained language model handle text it never saw in training.

Chronos-Bolt in particular predicts a set of quantiles directly rather than sampling many
paths, which is why it is fast: 36.9 seconds across the whole sample of 300 series and four
folds, against 487.9 for the classical rung and 699.6 for the neural one. It is the second
cheapest rung after the naive, and cheaper than fitting ARIMA by a factor of 13.

The honest caveat: zero-shot means no fitting, not no cost. The cost was paid once, by
somebody else, during pretraining.

---

## 4. The business case in 30 seconds

Forecast error is not the thing anyone pays for. The decision is how much to order, and
getting it wrong is asymmetric: order too much and a share of the surplus is marked down or
thrown away, order too little and a sale is lost.

The order that minimises expected cost is not the average forecast and not the median. It is
the quantile

    q* = stockout cost / (stockout cost + overstock cost)

the **critical ratio**. With the assumptions in `economics.py` it comes out at 0.549, so the
cost-optimal order sits just above the median.

**Why that matters more than it sounds.** It means a point forecast cannot answer the business
question at all, however accurate it is. You need the distribution, calibrated at the
particular quantile the cost ratio picks out. And minimising pinball loss at `q*` is, up to a
constant, the same thing as minimising that cost. The statistical metric and the euro figure
are one quantity in two units.

**The assumption that decides the answer** is the write-off fraction: what share of surplus is
eventually wasted rather than sold. Costing surplus at carrying cost alone gives a critical
ratio of 0.998, which says "order the 99.8th percentile of everything", which no grocer does.
That was the first version of the model and it was wrong. `sensitivity()` re-costs the whole
comparison across a grid of write-off fractions and margins to show whether the ranking is a
property of the forecasts or of the assumptions.

---

## 5. Limits, stated before anybody finds them

**One dataset, one domain.** M5 is Walmart US store-item unit sales. Nothing here licenses a
claim about energy load, call volumes or anything else.

**Rossmann was the first choice and is not what was used.** German retail was the closer
analogue, but it is a Kaggle competition dataset whose terms sit behind an accepted rules page
and which needs a personal API token to fetch. The loaders are written against a long-format
frame, so plugging Rossmann in is a small change, and doing that is the obvious next step.

**300 series, not 30,490.** Stratified across volume and intermittency, chosen once by a
documented rule before any model ran, with the ids committed. It runs on one CPU.

**The most recent 842 days, not the full 5.3 years.** Two reasons, both stated in
`data.HISTORY_DAYS`: a five-year-old observation of a supermarket item is weak evidence about
next month, and trimming cuts the data by 2.3 times. The modelling reason came first but the
compute reason is real and pretending otherwise would be dishonest.

**AutoARIMA's search is bounded.** If it loses, part of the reason may be that it was not
allowed to look further.

**The neural tier got a fixed step budget on a CPU with no early stopping.** With more
compute both models would likely improve. The budget is recorded so a reader can see what
they were and were not given.

**Only three quantiles were forecast.** A critical ratio above 0.9 would sit in a tail this
project never estimated, and every model's order is truncated at its own 90th percentile. That
biases the euro figures conservatively and it biases them equally.

**Every cost is assumed, none is measured.** M5 ships no price or cost data.

---

## 6. Questions to expect

**"Why is your baseline so hard to beat? Did you tune the others properly?"**
The asymmetry runs the other way: the complex rungs got features and budgets, the baseline got
seven lines of code. Where the baseline wins it wins on series with a specific property, and
that pattern is monotone across quartiles rather than sitting either side of one chosen
threshold.

**"Wouldn't a specialised intermittent-demand method do better?"**
Probably, on the sparse end, and that is the honest answer. Croston's method and its variants
exist precisely for these series and are the obvious next rung. The project records ADI and
the squared coefficient of variation per series, which is the standard quadrant for deciding
that, so the groundwork is there.

**"Why not just use the foundation model for everything?"**
Look at the cost column and the accuracy column together before answering. Zero-shot is
cheap to run and needs no per-series fitting, which is a genuine operational advantage. Whether
it is accurate enough here is what the table says, and it should be quoted rather than
guessed.

**"What would you do next?"**
Three things, in order. Plug in Rossmann so the finding is checked on a second dataset. Add
Croston and a specialised intermittent method as rung 2.5. Then serve the winning
configuration behind an API with the backtest running in CI, so a regression in accuracy fails
a build rather than being noticed a quarter later.

**"How long did this take and what was hardest?"**
The hardest part was not any model. It was resisting the reordering: the temptation to fit
things first and settle the protocol afterwards, at which point the protocol quietly becomes
whatever agrees with the results already on screen. The evaluation harness is the first
module in the repository and it has tests against values computed by hand, which is the only
reason the comparison means anything.
