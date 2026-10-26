"""Check Bike Sharing locally, then predict its unlabeled competition test rows.

Extract the three original CSVs into --data-dir. Nothing is submitted to Kaggle.
The holdout is blocked within each month; it is not a future-only forecast.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

ROOT = Path(__file__).resolve().parent
SOURCE_URL = "https://www.kaggle.com/competitions/bike-sharing-demand/data"
SOURCE_FILES = ["train.csv", "test.csv", "sampleSubmission.csv"]
CALENDAR_FEATURES = ["calendar_hour", "calendar_weekday", "calendar_month", "calendar_year"]
REMOVED_COLUMNS = ["datetime", "casual", "registered"]
METADATA = {
    "task": "regression", "target": "count", "id_columns": [], "metric": "rmse",
    "feature_types": {"season": "categorical", "weather": "categorical",
                      **{name: "numeric" for name in CALENDAR_FEATURES}},
}


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def file_hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def prepare_table(data):
    """Derive calendar values row by row; do not learn from labels or other rows."""
    dates = pd.to_datetime(data["datetime"], format="%Y-%m-%d %H:%M:%S", errors="raise")
    if dates.isna().any() or dates.duplicated().any():
        raise ValueError("Dates must be present and unique within each source table.")
    if set(CALENDAR_FEATURES) & set(data.columns):
        raise ValueError("Calendar feature names already occur in the source table.")
    result = data.drop(columns=[name for name in REMOVED_COLUMNS if name in data]).copy()
    result["calendar_hour"] = dates.dt.hour
    result["calendar_weekday"] = dates.dt.dayofweek  # Monday is 0.
    result["calendar_month"] = dates.dt.month
    result["calendar_year"] = dates.dt.year
    return result, dates


def regression_scores(actual, predicted, include_log=True):
    scores = {
        "rmse": float(root_mean_squared_error(actual, predicted)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
    }
    if include_log:
        if (actual < 0).any() or (predicted < 0).any():
            raise ValueError("RMSLE requires nonnegative counts and predictions.")
        scores["rmsle"] = float(root_mean_squared_error(np.log1p(actual), np.log1p(predicted)))
    return scores


def clip_counts(raw):
    """Negative rental counts are impossible; keep fractional predictions."""
    predicted = np.maximum(raw, 0.0)
    return predicted, {
        "negative_predictions_clipped": int((raw < 0).sum()),
        "raw_minimum_prediction": float(raw.min()), "raw_maximum_prediction": float(raw.max()),
        "minimum_prediction": float(predicted.min()), "maximum_prediction": float(predicted.max()),
        "postprocessing": "clip negative predictions to zero; no integer rounding",
    }


def run_model(name, train, test, folder, time_limit):
    inputs, outputs = folder / "input", folder / "output"
    inputs.mkdir(parents=True, exist_ok=True)
    train.to_csv(inputs / "train.csv", index=False)
    test.to_csv(inputs / "test.csv", index=False)
    save_json(inputs / "metadata.json", METADATA)
    command = [
        sys.executable, str(ROOT / "main.py"), "--train", str(inputs / "train.csv"),
        "--test", str(inputs / "test.csv"), "--metadata", str(inputs / "metadata.json"),
        "--output", str(outputs), "--time-limit", str(time_limit), "--seed", "42",
    ]
    started = time.monotonic()
    run = subprocess.run(command, capture_output=True, text=True, cwd=ROOT,
                         timeout=time_limit + 20)
    seconds = time.monotonic() - started
    (folder / "run.log").write_text(run.stdout + run.stderr, encoding="utf-8")
    if run.returncode:
        raise RuntimeError(f"{name} failed; see {folder / 'run.log'}")
    table = pd.read_csv(outputs / "predictions.csv")
    if list(table.columns) != ["prediction"] or len(table) != len(test):
        raise ValueError(f"Wrong prediction columns or row count for {name}.")
    raw = pd.to_numeric(table["prediction"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(raw).all():
        raise ValueError(f"Non-finite predictions for {name}.")
    info = json.loads((outputs / "run_info.json").read_text(encoding="utf-8"))
    if set([*REMOVED_COLUMNS, "count"]) & set(info["model_feature_columns"]):
        raise ValueError("Timestamps, target components, and count must not be model features.")
    return raw, {
        "training_rows": len(train), "prediction_rows": len(test),
        "wall_seconds": round(seconds, 4), "reported_seconds": info["elapsed_seconds"],
        "selected_model": info["model"]["selected_model"],
        "model_deadline_reached": info["model"]["time_limit_reached"],
        "emergency_fallback": info["model"]["emergency_fallback"],
        "model_features": len(info["model_feature_columns"]),
        "selection_metric": "rmse", "selection_predictions": "raw, before clipping",
        "target_transform": "none",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output/bike_sharing_checks"))
    parser.add_argument("--time-limit", type=float, default=50.0)
    args = parser.parse_args()
    if not np.isfinite(args.time_limit) or not 1 <= args.time_limit <= 55:
        parser.error("--time-limit must be between 1 and 55 seconds.")
    missing = [name for name in SOURCE_FILES if not (args.data_dir / name).is_file()]
    if missing:
        parser.error(f"Extract the Bike Sharing ZIP into --data-dir. Missing: {missing}")

    source_train = pd.read_csv(args.data_dir / "train.csv")
    source_test = pd.read_csv(args.data_dir / "test.csv")
    sample = pd.read_csv(args.data_dir / "sampleSubmission.csv", usecols=["datetime"])
    sample_header = pd.read_csv(args.data_dir / "sampleSubmission.csv", nrows=0).columns.tolist()
    if not {"count", "casual", "registered"}.issubset(source_train) or "count" in source_test:
        raise ValueError("Expected labeled train.csv with count components and unlabeled test.csv.")
    train, train_dates = prepare_table(source_train)
    test, test_dates = prepare_table(source_test)
    if set(train.columns) - {"count"} != set(test.columns):
        raise ValueError("Prepared training and competition-test features do not match.")
    if not train_dates.dt.day.between(1, 19).all() or not test_dates.dt.day.ge(20).all():
        raise ValueError("Expected training days 1-19 and competition-test days 20 onward.")
    if set(train_dates) & set(test_dates):
        raise ValueError("Training and competition-test timestamps must be disjoint.")
    if sample_header != ["datetime", "count"] or not sample["datetime"].equals(source_test["datetime"]):
        raise ValueError("The sample submission must match test.csv timestamps and row order.")
    target = pd.to_numeric(train["count"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(target).all() or (target < 0).any():
        raise ValueError("Training counts must be finite and nonnegative.")
    held_out = train_dates.dt.day.ge(16)
    outer_train, outer_test = train.loc[~held_out], train.loc[held_out]
    months = train_dates.dt.to_period("M")
    if min(len(outer_train), len(outer_test)) < 2 or set(months[held_out]) != set(months[~held_out]):
        raise ValueError("Each month must contain both training and holdout rows.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "dataset": "bike_sharing", "task": "regression", "source_url": SOURCE_URL,
        "source_files": {name: {"sha256": file_hash(args.data_dir / name)} for name in SOURCE_FILES},
        "source_training_rows": len(train), "source_test_rows": len(test),
        "seed": 42, "time_limit_seconds_per_run": args.time_limit,
        "preparation": {
            "removed_columns": REMOVED_COLUMNS, "calendar_features": CALENDAR_FEATURES,
            "categorical_codes": ["season", "weather"], "calendar_encoding": "numeric; Monday=0",
            "binary_numeric_features": ["holiday", "workingday"],
            "note": "Calendar extraction is rowwise; no fitted statistics or target components are used.",
        },
        "evaluation_note": "Within-month blocked holdout, not future-only forecasting or a leaderboard score.",
        "outer_training_days": "1-15 of each month", "outer_holdout_days": "16-19 of each month",
        "internal_validation": "The unchanged core uses a random split of the outer training rows.",
        "log_error_definition": "RMSLE = RMSE(log1p(actual count), log1p(nonnegative prediction))",
        "postprocessing_rule": "Clip negative predictions to zero for scoring and export; no rounding.",
        "sample_submission_usage": "Column names and datetime alignment only; its count values are never read.",
    }

    holdout_folder = output_dir / "holdout"
    (holdout_folder / "input").mkdir(parents=True, exist_ok=True)
    labels_path = holdout_folder / "input" / "labels.csv"
    source_train.loc[held_out, ["datetime", "count"]].to_csv(labels_path, index=False)
    raw, holdout = run_model(
        "Bike Sharing holdout", outer_train, outer_test.drop(columns="count"),
        holdout_folder, args.time_limit,
    )
    predicted, clipping = clip_counts(raw)
    labels = pd.read_csv(labels_path)  # The model never receives these outer labels.
    if not np.array_equal(labels["datetime"], source_train.loc[held_out, "datetime"]):
        raise ValueError("Holdout labels and prediction rows are misaligned.")
    actual = labels["count"].to_numpy(dtype=float)
    baseline = np.repeat(outer_train["count"].mean(), len(actual))
    holdout.update(regression_scores(actual, predicted))
    holdout.update(clipping)
    holdout.update({
        "evaluation": "within-month blocked holdout: train days 1-15, score days 16-19",
        "reported_metrics_use": "nonnegative clipped predictions",
        "raw_metrics": regression_scores(actual, raw, include_log=False),
        "baseline": {"method": "training mean", **regression_scores(actual, baseline)},
    })
    results["holdout"] = holdout
    save_json(output_dir / "metrics.json", results)
    print(json.dumps({"holdout": holdout}, allow_nan=False), flush=True)

    raw, full = run_model(
        "Bike Sharing competition predictions", train, test,
        output_dir / "full_training_prediction", args.time_limit,
    )
    predicted, clipping = clip_counts(raw)
    prediction_file = output_dir / "bike_sharing_predictions.csv"
    exported = pd.DataFrame({"datetime": source_test["datetime"].to_numpy(), "count": predicted})
    exported.to_csv(prediction_file, index=False)
    if list(exported.columns) != sample_header or not exported["datetime"].equals(sample["datetime"]):
        raise ValueError("Exported predictions do not match the sample's row format.")
    full.update(clipping)
    full.update({
        "evaluation": "unlabeled prediction run; no score available", "scored": False,
        "predictions_file": prediction_file.name, "predictions_sha256": file_hash(prediction_file),
        "prediction_columns": sample_header, "original_test_order_preserved": True,
        "submitted_to_kaggle": False,
    })
    results["full_training_prediction"] = full
    save_json(output_dir / "metrics.json", results)
    print(json.dumps({"full_training_prediction": full}, allow_nan=False), flush=True)
    print(f"Saved local results and predictions to {output_dir}")


if __name__ == "__main__":
    main()
