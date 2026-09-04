# Data

**The raw M5 download is not committed.** `scripts/download_data.py` fetches it, and it is
roughly 500 MB. One derived file is committed: `sample_panel.parquet`, the 300 sampled series
trimmed to the most recent 842 days, about 300 KB. See "What is committed and why" below.

## Source

**M5 Forecasting Accuracy competition data.** Daily unit sales for 30,490 store-item series
from 10 Walmart stores in California, Texas and Wisconsin, 2011-01-29 to 2016-06-19.

Fetched through [`datasetsforecast`](https://github.com/Nixtla/datasetsforecast) (Nixtla),
which mirrors the competition files and needs no credentials. The M5 data was released for the
2020 competition run by the M Open Forecasting Center at the University of Nicosia and is
widely redistributed for research and benchmarking.

The loader trims leading zeros before an item was first stocked, so series start on different
dates and all end on the same date.

## Why not Rossmann

Rossmann Store Sales was the first choice for this project, because German retail is a closer
analogue to the work it was built for than US grocery. It was not used, for two reasons:

1. It is a Kaggle competition dataset. The data terms sit behind a rules page that has to be
   accepted with an account, and confirming what portfolio use is permitted needs that page
   read rather than assumed.
2. Fetching it requires a personal Kaggle API token, so a reader could not reproduce this
   repository from a clean clone without one.

M5 is the same problem shape, it is the reference benchmark published forecasting results are
quoted against, and it fetches without credentials.

Every loader in `forecast_ladder.data` works on a long-format frame with `unique_id`, `ds` and
`y`, so a Rossmann loader can be added next to `load_m5` without touching the protocol, the
metrics or any rung. Doing that, and checking whether the finding survives on a second
dataset, is the first item on the "what next" list.

## What gets written here

| Path | What it is |
| --- | --- |
| `m5/` | The `datasetsforecast` cache, roughly 500 MB after decompression |
| `sample_panel.parquet` | The 300 sampled series, trimmed to the most recent 842 days, 300 KB. Already in the repository; `scripts/prepare_panel.py` rewrites it |

## What is committed and why

| Committed | Size | Why |
| --- | --- | --- |
| `sample_panel.parquet` | 300 KB | The 300 sampled series, trimmed to 842 days. Without it a fresh clone cannot run the app or the tests without a 500 MB download |
| `results/raw/*.parquet` | 4.5 MB | Every rung's forecasts. Without them `scripts/analyse.py` is not reproducible without two hours of CPU |
| `results/published/sample_series.csv` | 40 KB | Which 300 series were selected, and their descriptors |
| `results/published/series_features.csv` | 3 MB | Descriptors for all 26,834 eligible series, so the selection rule can be verified rather than taken on trust |

Not committed: the M5 download itself, under `m5/`.

**On redistributing the sample.** `sample_panel.parquet` is 300 of M5's 30,490 series, about
one percent, and it is derived rather than the competition files. The M5 data was released by
the M Open Forecasting Center for research and benchmarking and is mirrored in that spirit by
`datasetsforecast` and by a long list of published forecasting repositories. Committing a one
percent slice is the same use. If you are reusing it, cite the M5 competition, not this
repository. The full data is a `scripts/download_data.py` away and no part of this project
depends on the committed slice being treated as a source.
