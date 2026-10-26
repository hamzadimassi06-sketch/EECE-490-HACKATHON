"""Run the three synthetic examples and score their outputs separately."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import pandas as pd
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score, root_mean_squared_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-limit", type=float, default=50)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    results = []
    for task in ["classification", "regression", "clustering"]:
        example = root / "examples" / task
        output = root / "output" / task
        command = [
            sys.executable, str(root / "main.py"),
            "--train", str(example / ("data.csv" if task == "clustering" else "train.csv")),
            "--metadata", str(example / "metadata.json"), "--output", str(output),
            "--time-limit", str(args.time_limit),
        ]
        if task != "clustering":
            command.extend(["--test", str(example / "test.csv")])
        started = time.monotonic()
        subprocess.run(command, check=True, timeout=args.time_limit + 15)
        elapsed = time.monotonic() - started
        predicted = pd.read_csv(output / "predictions.csv").iloc[:, 0]
        reference = pd.read_csv(example / "reference_labels.csv").iloc[:, 0]
        if len(predicted) != len(reference):
            raise ValueError(f"Wrong prediction count for {task}")
        if task == "classification":
            metric, value = "balanced_accuracy", balanced_accuracy_score(reference, predicted)
        elif task == "regression":
            metric, value = "rmse", root_mean_squared_error(reference, predicted)
        else:
            metric, value = "adjusted_rand_index", adjusted_rand_score(reference, predicted)
        info = json.loads((output / "run_info.json").read_text())
        results.append({
            "task": task, "selected_model": info["model"]["selected_model"],
            "metric": metric, "value": float(value),
            "wall_seconds": round(elapsed, 4), "prediction_rows": len(predicted),
            "model_deadline_reached": info["model"]["time_limit_reached"],
        })
    destination = root / "output" / "example_results.json"
    destination.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(json.dumps(results, indent=2))
    print("These are local example results, not official hackathon scores.")


if __name__ == "__main__":
    main()
