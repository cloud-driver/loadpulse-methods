"""Read a USER-SUPPLIED aligned CSV and run forecast MCS.

Schema: timestamp,actual,calendar_ols,seasonal_naive,load_lag_model
Run: python -m examples.read_csv_example aligned_predictions.csv --block-length 24
Timestamp strings must be unique and already chronological. This example
requires ISO-8601 UTC timestamps, one forecast horizon and a regular time grid.
"""
import argparse
import csv
from datetime import datetime, timedelta
import numpy as np
from loadpulse_stats import loss_matrix, model_confidence_set


def load_prediction_csv(path, model_columns):
    """Return aligned UTC timestamps, actual:(n,), predictions:(n,k)."""
    with open(path, encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if len(rows) < 3:
        raise ValueError("CSV needs at least three observations.")
    time = [datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00")) for row in rows]
    if any(t.tzinfo is None or t.utcoffset() != timedelta(0) for t in time):
        raise ValueError("Use timezone-aware UTC ISO-8601 timestamps.")
    intervals = [time[i+1]-time[i] for i in range(len(time)-1)]
    if intervals[0] <= timedelta(0) or any(delta != intervals[0] for delta in intervals):
        raise ValueError("Timestamps must be strictly increasing on a regular UTC grid.")
    actual = np.asarray([float(row["actual"]) for row in rows])
    predictions = np.asarray([[float(row[name]) for name in model_columns] for row in rows])
    return time, actual, predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    parser.add_argument("--models", nargs="+", default=["calendar_ols", "seasonal_naive", "load_lag_model"])
    parser.add_argument("--block-length", type=int, required=True,
                        help="Preplanned block length in observation rows")
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()
    _, actual, forecasts = load_prediction_csv(args.csv_path, args.models)
    result = model_confidence_set(loss_matrix(actual, forecasts), model_names=args.models,
                block_length=args.block_length, n_resamples=args.resamples, random_state=42)
    print("MCS included:", result.included)
    print("MCS p-values:", result.pvalues)
