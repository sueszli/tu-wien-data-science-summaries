# Daily precipitation forecasting

This project explores one-day-ahead precipitation forecasts for Central European river basins in [LamaH-CE](https://doi.org/10.5281/zenodo.5153305). It sits alongside the AI/ML and climate change notes as related work.

The benchmark compares the training mean, persistence, and lag-one linear regression. Each basin is scored separately. The first 70% of its days are used for fitting. On each held-out day the two lagged models can use the previous day's observed precipitation, but they are not refit. The output contains MAE and RMSE in the input precipitation units.

Run the example from this directory with Python 3 and no extra packages:

```sh
python benchmark.py
```

The bundled example is synthetic and is only a smoke test. For real data, extract the LamaH-CE daily files separately and pass one or more basin CSV files:

```sh
python benchmark.py /path/to/ID_*.csv
```

Inputs can also use comma-separated `date,prec` columns and an optional `basin_id` column. Each basin needs consecutive dates and nonnegative precipitation values.

The original benchmark logged scores under different basin IDs while selecting the same basin repeatedly. This version does not carry over those scores or claim to reproduce the original model ranking. It also leaves out the assignment materials, raw data, and model checkpoints.
