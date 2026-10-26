"""Check House Prices locally, then predict its unlabeled competition test rows.

Extract the four original files into --data-dir. Nothing is submitted to Kaggle.
The holdout score is a local check, not a leaderboard score.
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
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
SOURCE_URL = "https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data"
SOURCE_FILES = ["train.csv", "test.csv", "data_description.txt", "sample_submission.csv"]
NA_CATEGORIES = [
    "Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "PoolQC", "Fence", "MiscFeature",
]
METADATA = {
    "task": "regression", "target": "SalePrice", "id_columns": ["Id"],
    "feature_types": {"MSSubClass": "categorical"}, "metric": "rmse",
}


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def file_hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_house_table(path):
    """Preserve category codes described in data_description.txt, without fitting."""
    data = pd.read_csv(path, keep_default_na=False, na_values=[""])
    for column in NA_CATEGORIES:
        # A safe string survives main.py's own CSV reader. Keep conflicting
        # records as given rather than inferring repairs from other columns.
        data[column] = data[column].replace("NA", "__NA_category__")
    data["MasVnrType"] = data["MasVnrType"].replace("None", "__no_masonry__")
    for column in data.select_dtypes(include="object"):
        data[column] = data[column].mask(data[column].eq("NA"), np.nan)
    return data


def regression_scores(actual, predicted):
    nonpositive = int((predicted <= 0).sum())
    return {
        "rmse": float(root_mean_squared_error(actual, predicted)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
        "log_rmse": (float(root_mean_squared_error(np.log(actual), np.log(predicted)))
                     if nonpositive == 0 else None),
        "nonpositive_predictions": nonpositive,
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
    predictions = pd.to_numeric(table["prediction"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(predictions).all():
        raise ValueError(f"Non-finite predictions for {name}.")
    info = json.loads((outputs / "run_info.json").read_text(encoding="utf-8"))
    if "Id" in info["model_feature_columns"] or "SalePrice" in info["model_feature_columns"]:
        raise ValueError("The identifier and target must not be model features.")
    return predictions, {
        "training_rows": len(train), "prediction_rows": len(test),
        "wall_seconds": round(seconds, 4), "reported_seconds": info["elapsed_seconds"],
        "selected_model": info["model"]["selected_model"],
        "model_deadline_reached": info["model"]["time_limit_reached"],
        "emergency_fallback": info["model"]["emergency_fallback"],
        "model_features": len(info["model_feature_columns"]),
        "selection_metric": "rmse", "target_transform": "none",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output/house_prices_checks"))
    parser.add_argument("--time-limit", type=float, default=50.0)
    args = parser.parse_args()
    if not np.isfinite(args.time_limit) or not 1 <= args.time_limit <= 55:
        parser.error("--time-limit must be between 1 and 55 seconds.")
    missing = [name for name in SOURCE_FILES if not (args.data_dir / name).is_file()]
    if missing:
        parser.error(f"Extract the House Prices ZIP into --data-dir. Missing: {missing}")

    train = read_house_table(args.data_dir / "train.csv")
    test = read_house_table(args.data_dir / "test.csv")
    sample = pd.read_csv(args.data_dir / "sample_submission.csv", usecols=["Id"])
    sample_header = pd.read_csv(args.data_dir / "sample_submission.csv", nrows=0).columns.tolist()
    if "SalePrice" not in train or "SalePrice" in test:
        raise ValueError("Expected labeled train.csv and unlabeled test.csv.")
    if set(train.columns) - {"SalePrice"} != set(test.columns):
        raise ValueError("Training and competition-test features do not match.")
    if len(train) < 10 or test.empty:
        raise ValueError("Expected a nonempty competition test set and at least ten training rows.")
    for table in (train, test):
        if table["Id"].isna().any() or table["Id"].duplicated().any():
            raise ValueError("House IDs must be present and unique within each table.")
    if set(train["Id"]) & set(test["Id"]):
        raise ValueError("Training and competition-test IDs must be disjoint.")
    if sample_header != ["Id", "SalePrice"] or not sample["Id"].equals(test["Id"]):
        raise ValueError("The sample submission must match test.csv IDs and row order.")
    target = pd.to_numeric(train["SalePrice"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(target).all() or (target <= 0).any():
        raise ValueError("Training SalePrice values must be finite and positive.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "dataset": "house_prices", "task": "regression", "source_url": SOURCE_URL,
        "source_files": {name: {"sha256": file_hash(args.data_dir / name)}
                         for name in SOURCE_FILES},
        "source_training_rows": len(train), "source_test_rows": len(test),
        "seed": 42, "time_limit_seconds_per_run": args.time_limit,
        "preparation": {
            "excluded_id_columns": ["Id"], "categorical_codes": ["MSSubClass"],
            "preserved_NA_category_columns": NA_CATEGORIES,
            "NA_category_token": "__NA_category__",
            "masonry_None_token": "__no_masonry__",
            "other_NA_values": "missing; handled by the training-only pipeline",
            "note": "Rowwise code preservation only; no inferred repairs or fitted statistics.",
        },
        "evaluation_note": "Local holdout scores are not Kaggle leaderboard results.",
        "log_error_definition": "RMSE(log(actual SalePrice), log(predicted SalePrice)); natural log, no +1",
        "nonpositive_prediction_policy": "Report count and leave log RMSE null; do not clip silently.",
        "sample_submission_usage": "Column names and Id alignment only; its SalePrice values are never read.",
    }

    outer_train, outer_test = train_test_split(train, test_size=0.2, random_state=42)
    holdout_folder = output_dir / "holdout"
    (holdout_folder / "input").mkdir(parents=True, exist_ok=True)
    labels_path = holdout_folder / "input" / "labels.csv"
    outer_test[["Id", "SalePrice"]].to_csv(labels_path, index=False)
    predicted, holdout = run_model(
        "House Prices holdout", outer_train, outer_test.drop(columns="SalePrice"),
        holdout_folder, args.time_limit,
    )
    # Only the evaluator opens the outer labels after predictions have been saved.
    labels = pd.read_csv(labels_path)
    if not np.array_equal(labels["Id"].to_numpy(), outer_test["Id"].to_numpy()):
        raise ValueError("Holdout labels and prediction rows are misaligned.")
    actual = labels["SalePrice"].to_numpy(dtype=float)
    baseline = np.repeat(outer_train["SalePrice"].mean(), len(actual))
    holdout.update(regression_scores(actual, predicted))
    holdout.update({
        "evaluation": "random 80/20 outer holdout; separate internal model selection split",
        "baseline": {"method": "training mean", **regression_scores(actual, baseline)},
    })
    results["holdout"] = holdout
    save_json(output_dir / "metrics.json", results)
    print(json.dumps({"holdout": holdout}, allow_nan=False), flush=True)

    predicted, full = run_model(
        "House Prices competition predictions", train, test,
        output_dir / "full_training_prediction", args.time_limit,
    )
    prediction_file = output_dir / "house_prices_predictions.csv"
    exported = pd.DataFrame({"Id": test["Id"].to_numpy(), "SalePrice": predicted})
    exported.to_csv(prediction_file, index=False)
    if list(exported.columns) != sample_header or not exported["Id"].equals(sample["Id"]):
        raise ValueError("Exported predictions do not match the sample's row format.")
    full.update({
        "evaluation": "unlabeled prediction run; no score available", "scored": False,
        "predictions_file": prediction_file.name, "predictions_sha256": file_hash(prediction_file),
        "prediction_columns": sample_header, "original_test_order_preserved": True,
        "nonpositive_predictions": int((predicted <= 0).sum()),
        "minimum_prediction": float(predicted.min()), "maximum_prediction": float(predicted.max()),
        "submitted_to_kaggle": False,
    })
    results["full_training_prediction"] = full
    save_json(output_dir / "metrics.json", results)
    print(json.dumps({"full_training_prediction": full}, allow_nan=False), flush=True)
    print(f"Saved local results and predictions to {output_dir}")


if __name__ == "__main__":
    main()
