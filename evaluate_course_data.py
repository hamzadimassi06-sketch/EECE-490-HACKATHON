"""Run four checks using the course CSV files, which are not bundled here.

These are local benchmarks, not official hackathon scores. Labels are saved
separately and are read for scoring only after main.py has finished.
"""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, adjusted_rand_score, balanced_accuracy_score,
    mean_absolute_error, r2_score, root_mean_squared_error,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent


def find_csv(directory, name):
    exact = directory / name
    if exact.is_file():
        return exact
    # Also accept attachment names such as 13-iris.csv and 10-bodyPerformance.csv.
    matches = [path for path in directory.iterdir() if path.is_file()
               and re.sub(r"^\d+[-_ ]+", "", path.name).lower() == name.lower()]
    if len(matches) != 1:
        raise ValueError(f"Expected one {name} in {directory}; found {len(matches)}.")
    return matches[0]


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output/course_checks"))
    parser.add_argument("--time-limit", type=float, default=50.0)
    args = parser.parse_args()
    if not args.data_dir.is_dir():
        parser.error("--data-dir must contain the original course CSV files.")
    if not np.isfinite(args.time_limit) or not 1 <= args.time_limit <= 55:
        parser.error("--time-limit must be between 1 and 55 seconds.")

    iris = pd.read_csv(find_csv(args.data_dir, "iris.csv"), header=None, names=[
        "sepal_length", "sepal_width", "petal_length", "petal_width", "species",
    ])
    body = pd.read_csv(find_csv(args.data_dir, "bodyPerformance.csv"))
    life = pd.read_csv(find_csv(args.data_dir, "life_exp_data.csv"))
    life_target = "Life expectancy "  # The space belongs to the supplied CSV header.
    dropped_life_rows = int(life[life_target].isna().sum())
    life = life.dropna(subset=[life_target])
    if len(iris) != 150:
        raise ValueError("Expected the supplied headerless iris.csv with 150 rows.")

    cases = [
        ("iris_classification", iris, "classification", "species", {}),
        ("body_performance", body, "classification", "class", {"gender": "categorical"}),
        ("life_expectancy", life, "regression", life_target,
         {"Country": "categorical", "Status": "categorical"}),
        ("iris_clustering", iris, "clustering", "species", {}),
    ]
    output_dir = args.output_dir.resolve()
    results = []
    for name, data, task, target, feature_types in cases:
        folder = output_dir / name
        inputs, outputs = folder / "input", folder / "output"
        inputs.mkdir(parents=True, exist_ok=True)
        metadata = {"task": task, "feature_types": feature_types, "id_columns": []}
        if task == "clustering":
            train = data.drop(columns=[target])
            actual = data[target]
            metadata["n_clusters"] = 3
        else:
            train, test = train_test_split(
                data, test_size=0.2, random_state=42,
                stratify=data[target] if task == "classification" else None,
            )
            actual = test[target]
            test.drop(columns=[target]).to_csv(inputs / "test.csv", index=False)
            metadata["target"] = target
        train.to_csv(inputs / "train.csv", index=False)
        actual.to_frame("label").to_csv(inputs / "labels.csv", index=False)
        save_json(inputs / "metadata.json", metadata)
        command = [
            sys.executable, str(ROOT / "main.py"), "--train", str(inputs / "train.csv"),
            "--metadata", str(inputs / "metadata.json"), "--output", str(outputs),
            "--time-limit", str(args.time_limit),
        ]
        if task != "clustering":
            command.extend(["--test", str(inputs / "test.csv")])
        started = time.monotonic()
        run = subprocess.run(command, capture_output=True, text=True, cwd=ROOT,
                             timeout=args.time_limit + 20)
        seconds = time.monotonic() - started
        (folder / "run.log").write_text(run.stdout + run.stderr, encoding="utf-8")
        if run.returncode:
            raise RuntimeError(f"{name} failed; see {folder / 'run.log'}")

        actual = pd.read_csv(inputs / "labels.csv")["label"]
        predicted = pd.read_csv(outputs / "predictions.csv").iloc[:, 0]
        info = json.loads((outputs / "run_info.json").read_text(encoding="utf-8"))
        item = {"dataset": name, "task": task, "training_rows": len(train),
                "prediction_rows": len(actual), "wall_seconds": round(seconds, 4),
                "reported_seconds": info["elapsed_seconds"],
                "selected_model": info["model"]["selected_model"],
                "model_deadline_reached": info["model"]["time_limit_reached"]}
        if task == "classification":
            baseline = np.repeat(train[target].mode().iloc[0], len(actual))
            item.update(accuracy=float(accuracy_score(actual, predicted)),
                        balanced_accuracy=float(balanced_accuracy_score(actual, predicted)),
                        baseline_accuracy=float(accuracy_score(actual, baseline)),
                        baseline_balanced_accuracy=float(balanced_accuracy_score(actual, baseline)))
        elif task == "regression":
            baseline = np.repeat(train[target].mean(), len(actual))
            item.update(rmse=float(root_mean_squared_error(actual, predicted)),
                        mae=float(mean_absolute_error(actual, predicted)),
                        r2=float(r2_score(actual, predicted)),
                        baseline_rmse=float(root_mean_squared_error(actual, baseline)),
                        dropped_missing_target_rows=dropped_life_rows)
        else:
            item.update(adjusted_rand_index=float(adjusted_rand_score(actual, predicted)),
                        groups_found=int(predicted.nunique()))
        results.append(item)
        save_json(output_dir / "metrics.json", results)
        print(json.dumps(item), flush=True)
    print(f"Saved local benchmark results to {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
