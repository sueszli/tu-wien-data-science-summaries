import csv
from pathlib import Path


def main():
    path = Path(__file__).with_name("results.csv")
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    best = max(rows, key=lambda row: float(row["validation_f1"]))
    print(f"{len(rows)} runs; best validation F1: {float(best['validation_f1']):.6f}")
    print(f"Features: {best['features']}; regularization: {best['regularization']}; "
          f"iteration cap: {best['max_iterations']}")


if __name__ == "__main__":
    main()
