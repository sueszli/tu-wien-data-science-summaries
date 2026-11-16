import argparse
import csv
from collections import defaultdict
from datetime import date, timedelta
from math import isfinite, sqrt
from pathlib import Path
from statistics import fmean


def read_series(paths):
    series = defaultdict(list)
    for path in paths:
        with path.open(newline="") as source:
            separator = ";" if ";" in source.readline() else ","
            source.seek(0)
            for row in csv.DictReader(source, delimiter=separator):
                day = (
                    date.fromisoformat(row["date"])
                    if "date" in row
                    else date(int(row["YYYY"]), int(row["MM"]), int(row["DD"]))
                )
                rain = float(row["prec"])
                if not isfinite(rain) or rain < 0:
                    raise ValueError(f"{path}: expected finite, non-negative precipitation on {day}")
                series[row.get("basin_id") or path.stem].append((day, rain))
    return series


def forecast(points):
    points = sorted(points)
    if len(points) < 5 or any(
        right[0] - left[0] != timedelta(days=1)
        for left, right in zip(points, points[1:])
    ):
        raise ValueError("Each basin needs at least five consecutive daily observations.")
    values = [value for _, value in points]
    split = int(len(values) * 0.7)
    train, actual = values[:split], values[split:]
    x, y = train[:-1], train[1:]
    x_mean, y_mean = fmean(x), fmean(y)
    variance = sum((v - x_mean) ** 2 for v in x)
    slope = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y)) / variance if variance else 0.0
    previous = values[split - 1 : -1]
    predictions = {
        "train_mean": [fmean(train)] * len(actual),
        "persistence": previous,
        "lag1_linear": [max(0.0, y_mean + slope * (v - x_mean)) for v in previous],
    }
    return {
        model: (fmean(abs(a - p) for a, p in zip(actual, pred)),
                sqrt(fmean((a - p) ** 2 for a, p in zip(actual, pred))))
        for model, pred in predictions.items()
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate one-day precipitation baselines.")
    parser.add_argument("csv", nargs="*", type=Path, default=[Path(__file__).with_name("example.csv")])
    args = parser.parse_args()
    series = read_series(args.csv)
    if not series:
        parser.error("No observations found in the input CSV files.")
    print("basin,model,mae,rmse")
    for basin, points in sorted(series.items()):
        try:
            metrics = forecast(points)
        except ValueError as exc:
            parser.error(f"{basin}: {exc}")
        for model, (mae, rmse) in metrics.items():
            print(f"{basin},{model},{mae:.4f},{rmse:.4f}")


if __name__ == "__main__":
    main()
