"""Check three public datasets linked in the hackathon brief.

Download the original CSVs separately; they are not bundled in this project.
The results are local checks, not private assessment or Kaggle leaderboard scores.
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
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score, silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
SOURCES = {
    "telco": (
        "WA_Fn-UseC_-Telco-Customer-Churn.csv",
        "https://www.kaggle.com/datasets/blastchar/telco-customer-churn",
    ),
    "ai4i": (
        "ai4i2020.csv",
        "https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020",
    ),
    "mall": (
        "Mall_Customers.csv",
        "https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python",
    ),
}
FAILURE_TYPES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
MALL_FEATURES = ["Annual Income (k$)", "Spending Score (1-100)"]


def find_csv(directory, filename):
    matches = sorted(path for path in directory.rglob(filename) if path.is_file())
    if len(matches) != 1:
        raise ValueError(f"Expected one {filename} below {directory}; found {len(matches)}.")
    return matches[0]


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def classification_scores(actual, predicted, negative, positive):
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "positive_precision": float(precision_score(
            actual, predicted, pos_label=positive, zero_division=0)),
        "positive_recall": float(recall_score(
            actual, predicted, pos_label=positive, zero_division=0)),
        "positive_f1": float(f1_score(
            actual, predicted, pos_label=positive, zero_division=0)),
        "confusion_matrix": confusion_matrix(
            actual, predicted, labels=[negative, positive]).tolist(),
    }


def run_case(name, args, output_dir):
    filename, source_url = SOURCES[name]
    source = find_csv(args.data_dir, filename)
    with source.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    data = pd.read_csv(source)
    folder = output_dir / name
    inputs, outputs = folder / "input", folder / "output"
    inputs.mkdir(parents=True, exist_ok=True)
    metadata = {"task": "classification", "id_columns": [], "feature_types": {}}
    removed_columns = []

    if name == "telco":
        target, negative, positive = "Churn", "No", "Yes"
        metadata["id_columns"] = ["customerID"]
        metadata["feature_types"] = {
            "SeniorCitizen": "categorical", "TotalCharges": "numeric",
        }
    elif name == "ai4i":
        target, negative, positive = "Machine failure", 0, 1
        # These columns describe the outcome, so keeping them would leak the label.
        data = data.drop(columns=FAILURE_TYPES)
        removed_columns = FAILURE_TYPES.copy()
        metadata["id_columns"] = ["UDI", "Product ID"]
        metadata["feature_types"] = {"Type": "categorical"}
    else:
        target = None
        removed_columns = [column for column in data if column not in MALL_FEATURES]
        data = data[MALL_FEATURES].apply(pd.to_numeric, errors="raise")
        if not np.isfinite(data.to_numpy()).all():
            raise ValueError("The Mall check expects finite income and spending values.")
        if not 2 <= args.mall_clusters < len(data):
            raise ValueError("--mall-clusters must be at least 2 and less than the row count.")
        metadata = {
            "task": "clustering", "n_clusters": args.mall_clusters, "id_columns": [],
            "feature_types": {column: "numeric" for column in MALL_FEATURES},
        }

    if target is not None:
        if not data[target].isin([negative, positive]).all():
            raise ValueError(f"Unexpected or missing labels in {name}.{target}.")
        train, test = train_test_split(
            data, test_size=0.2, random_state=42, stratify=data[target],
        )
        metadata["target"] = target
        test.drop(columns=[target]).to_csv(inputs / "test.csv", index=False)
        test[[target]].rename(columns={target: "label"}).to_csv(
            inputs / "labels.csv", index=False)
        prediction_rows = len(test)
    else:
        train = data
        prediction_rows = len(train)
    train.to_csv(inputs / "train.csv", index=False)
    save_json(inputs / "metadata.json", metadata)

    command = [
        sys.executable, str(ROOT / "main.py"), "--train", str(inputs / "train.csv"),
        "--metadata", str(inputs / "metadata.json"), "--output", str(outputs),
        "--time-limit", str(args.time_limit), "--seed", "42",
    ]
    if target is not None:
        command.extend(["--test", str(inputs / "test.csv")])
    started = time.monotonic()
    run = subprocess.run(command, capture_output=True, text=True, cwd=ROOT,
                         timeout=args.time_limit + 20)
    seconds = time.monotonic() - started
    (folder / "run.log").write_text(run.stdout + run.stderr, encoding="utf-8")
    if run.returncode:
        raise RuntimeError(f"{name} failed; see {folder / 'run.log'}")

    predicted = pd.read_csv(outputs / "predictions.csv").iloc[:, 0]
    if len(predicted) != prediction_rows or predicted.isna().any():
        raise ValueError(f"Invalid prediction count or missing predictions for {name}.")
    info = json.loads((outputs / "run_info.json").read_text(encoding="utf-8"))
    item = {
        "dataset": name, "task": metadata["task"], "source_filename": source.name,
        "source_url": source_url, "source_sha256": digest, "source_rows": len(data),
        "training_rows": len(train), "prediction_rows": prediction_rows,
        "seed": 42, "time_limit_seconds": args.time_limit,
        "wall_seconds": round(seconds, 4), "reported_seconds": info["elapsed_seconds"],
        "selected_model": info["model"]["selected_model"],
        "model_deadline_reached": info["model"]["time_limit_reached"],
        "emergency_fallback": info["model"]["emergency_fallback"],
        "removed_columns": removed_columns, "excluded_id_columns": metadata["id_columns"],
    }
    if target is not None:
        if not predicted.isin([negative, positive]).all():
            raise ValueError(f"Predicted labels outside the expected classes for {name}.")
        # The outer holdout labels are only read for scoring after the program exits.
        actual = pd.read_csv(inputs / "labels.csv")["label"]
        baseline = np.repeat(train[target].mode().iloc[0], len(actual))
        item.update(classification_scores(actual, predicted, negative, positive))
        item.update({
            "target": target, "evaluation": "stratified 80/20 holdout",
            "positive_label": positive, "confusion_matrix_labels": [negative, positive],
            "confusion_matrix_orientation": "rows actual, columns predicted",
            "positive_training_rows": int((train[target] == positive).sum()),
            "positive_holdout_rows": int((actual == positive).sum()),
            "majority_baseline": classification_scores(actual, baseline, negative, positive),
        })
    else:
        values = pd.to_numeric(predicted, errors="raise").to_numpy()
        if (not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all()
                or (values < 0).any() or (values >= args.mall_clusters).any()):
            raise ValueError("Mall cluster labels must be integers from 0 to k - 1.")
        labels = values.astype(int)
        groups_found = len(np.unique(labels))
        # Use the same two standardized numeric features used by the model.
        scaled = StandardScaler().fit_transform(train[MALL_FEATURES])
        silhouette = (float(silhouette_score(scaled, labels))
                      if 1 < groups_found < len(train) else None)
        item.update({
            "evaluation": "internal silhouette; no ground-truth cluster labels",
            "features_used": MALL_FEATURES, "requested_clusters": args.mall_clusters,
            "cluster_count_note": "Our example choice; the brief does not give Mall k.",
            "groups_found": groups_found,
            "group_counts": {str(group): int((labels == group).sum())
                             for group in range(args.mall_clusters)},
            "silhouette": silhouette,
            "silhouette_space": "income and spending, each standardized with StandardScaler",
        })
    return item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cases", choices=list(SOURCES), nargs="+", default=list(SOURCES))
    parser.add_argument("--output-dir", type=Path, default=Path("output/linked_checks"))
    parser.add_argument("--time-limit", type=float, default=50.0)
    parser.add_argument("--mall-clusters", type=int, default=5)
    args = parser.parse_args()
    if not args.data_dir.is_dir():
        parser.error("--data-dir must contain the separately downloaded CSV files.")
    if not np.isfinite(args.time_limit) or not 1 <= args.time_limit <= 55:
        parser.error("--time-limit must be between 1 and 55 seconds.")
    if len(args.cases) != len(set(args.cases)):
        parser.error("List each case at most once.")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    save_json(output_dir / "metrics.json", results)
    for name in args.cases:
        item = run_case(name, args, output_dir)
        results.append(item)
        save_json(output_dir / "metrics.json", results)
        print(json.dumps(item, allow_nan=False), flush=True)
    print(f"Saved local benchmark results to {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
