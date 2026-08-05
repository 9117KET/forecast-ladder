# Data

**Nothing in this directory is committed.** `scripts/download_data.py` fetches it. The
`.gitignore` allows only this file and `.gitkeep`.

## Source

**M5 Forecasting Accuracy competition data.** Daily unit sales for 30,490 store-item series
from 10 Walmart stores in California, Texas and Wisconsin, 2011-01-29 to 2016-06-19.

Fetched through [`datasetsforecast`](https://github.com/Nixtla/datasetsforecast) (Nixtla),
which mirrors the competition files and needs no credentials. The M5 data was released for the
2020 competition run by the M Open Forecasting Center at the University of Nicosia and is
widely redistributed for research and benchmarking; this project redistributes none of it and
fetches it at run time.

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
| `sample_panel.parquet` | The 300 sampled series, trimmed to the most recent 842 days, a few MB |

## The sample is committed, the data is not

`results/published/sample_series.csv` records which 300 series were selected and their
descriptors, and `results/published/series_features.csv` records the descriptors for all
26,834 eligible series. That is enough for anyone to verify the selection rule was applied
rather than described, without this repository redistributing the underlying data.
