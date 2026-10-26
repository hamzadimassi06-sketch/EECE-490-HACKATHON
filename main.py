"""Read one dataset, make predictions, and write its data profile."""

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

# Set these before importing numpy/sklearn, including in a spawned worker.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"


def read_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--task", choices=["classification", "regression", "clustering"])
    parser.add_argument("--target", help="Name of the label column, if present")
    parser.add_argument("--clusters", type=int, help="Required number of clusters")
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--time-limit", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_metadata(path):
    if path is None:
        return {}
    with path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError("Metadata must be a JSON object.")
    allowed = {"task", "target", "n_clusters", "feature_types", "id_columns", "metric"}
    unknown = set(metadata) - allowed
    if unknown:
        raise ValueError(f"Unknown metadata fields: {sorted(unknown)}")
    feature_types = metadata.get("feature_types", {})
    if not isinstance(feature_types, dict):
        raise ValueError("feature_types must map column names to numeric or categorical.")
    for name, kind in feature_types.items():
        if not isinstance(name, str) or kind not in {"numeric", "categorical"}:
            raise ValueError(f"Invalid feature type for {name!r}: {kind!r}")
    ids = metadata.get("id_columns", [])
    if not isinstance(ids, list) or not all(isinstance(name, str) for name in ids):
        raise ValueError("id_columns must be a list of column names.")
    return metadata


def read_table(path, feature_types):
    import pandas as pd

    # pandas renames duplicate headers silently. Catch them before reading.
    with path.open(newline="", encoding="utf-8-sig") as handle:
        header = next(csv.reader(handle), [])
    if not header or any(not name.strip() for name in header):
        raise ValueError(f"{path.name}: CSV needs a nonempty header for every column.")
    if len(set(header)) != len(header):
        raise ValueError(f"{path.name}: duplicate CSV column names are not supported.")
    # Preserve category codes such as 001 when metadata declares them.
    dtypes = {name: str for name, kind in feature_types.items()
              if kind == "categorical" and name in header}
    return pd.read_csv(path, encoding="utf-8-sig", dtype=dtypes, low_memory=False)


def write_json(path, value):
    # allow_nan=False catches invalid JSON instead of quietly writing NaN.
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def run(args, started):
    import numpy as np
    import pandas as pd
    from data_profile import make_profile
    from models import solve_dataset
    from preprocessing import missing_mask

    if not math.isfinite(args.time_limit) or not 1 <= args.time_limit <= 55:
        raise ValueError("--time-limit must be between 1 and 55 seconds.")
    if not 0 <= args.seed < 2**32:
        raise ValueError("--seed must be between 0 and 4294967295.")

    metadata = read_metadata(args.metadata)
    task = args.task or metadata.get("task")
    target = args.target if args.target is not None else metadata.get("target")
    n_clusters = args.clusters if args.clusters is not None else metadata.get("n_clusters")
    feature_types = metadata.get("feature_types", {})
    id_columns = metadata.get("id_columns", [])
    metric = metadata.get("metric")
    notes = []

    if task not in {"classification", "regression", "clustering"}:
        raise ValueError("Provide task in metadata or with --task.")
    if target is not None and not isinstance(target, str):
        raise ValueError("target must be a column name.")
    if task != "clustering" and (not target or args.test is None):
        raise ValueError("Classification and regression need a target and --test CSV.")
    if task == "clustering" and (isinstance(n_clusters, bool)
            or not isinstance(n_clusters, int) or n_clusters < 1):
        raise ValueError("Clustering needs a positive integer n_clusters or --clusters.")
    metrics = {
        "classification": {None, "balanced_accuracy", "accuracy", "f1_macro"},
        "regression": {None, "rmse", "mae", "r2"},
        "clustering": {None, "silhouette"},
    }
    if metric not in metrics[task]:
        raise ValueError(f"Unsupported metric {metric!r} for {task}.")

    train = read_table(args.train, feature_types)
    if train.empty:
        raise ValueError("The training table must contain at least one row.")
    unknown_types = set(feature_types) - set(train.columns)
    if unknown_types:
        raise ValueError(f"feature_types names missing from train: {sorted(unknown_types)}")
    missing_ids = set(id_columns) - set(train.columns)
    if missing_ids:
        raise ValueError(f"ID columns missing from train: {sorted(missing_ids)}")
    if target and target not in train.columns:
        raise ValueError(f"Target column {target!r} is missing from train.")
    if task == "clustering" and n_clusters > len(train):
        raise ValueError("n_clusters cannot exceed the number of training rows.")

    y = None
    if task != "clustering":
        y = train[target].copy()
        if missing_mask(y).any():
            raise ValueError("Training labels contain missing values; labels cannot be imputed.")
        if task == "regression":
            y = pd.to_numeric(y, errors="raise")
            if not np.isfinite(y.to_numpy(dtype=float)).all():
                raise ValueError("Regression labels must be finite numbers.")
        elif pd.api.types.is_numeric_dtype(y) and not np.isfinite(y.to_numpy(dtype=float)).all():
            raise ValueError("Classification labels must be finite values.")

    # Profile the raw features, including IDs, before choosing model columns.
    raw_features = train.drop(columns=[target] if target else [])
    raw_types = {name: kind for name, kind in feature_types.items() if name != target}
    profile = make_profile(raw_features, y=y, task=task, feature_types=raw_types)
    excluded = list(dict.fromkeys([*id_columns, *([target] if target else [])]))
    X_train = train.drop(columns=excluded)
    model_types = {name: kind for name, kind in feature_types.items() if name in X_train.columns}

    prediction_source = args.test if args.test is not None else args.train
    test = read_table(args.test, feature_types) if args.test is not None else train
    if len(test) == 0:
        raise ValueError("The prediction table must contain at least one row.")
    missing = set(X_train.columns) - set(test.columns)
    if missing:
        raise ValueError(f"Prediction table is missing feature columns: {sorted(missing)}")
    extra = [name for name in test.columns if name not in X_train.columns]
    if args.test is not None and extra:
        notes.append(f"Ignored non-feature columns in test: {extra}")
    if task == "clustering" and target:
        notes.append(f"Excluded {target!r}; clustering never reads its label values.")
    X_predict = test.loc[:, X_train.columns].copy()

    # Reserve time for saving the outputs. The worker can be stopped mid-fit.
    remaining = max(0.0, args.time_limit - (time.monotonic() - started) - 1.5)
    predictions, model_report = solve_dataset(
        X_train, y, X_predict, task=task, feature_types=model_types,
        n_clusters=n_clusters, random_state=args.seed, time_budget=remaining, metric=metric,
    )
    predictions = np.asarray(predictions)
    if predictions.ndim != 1 or len(predictions) != len(test):
        raise RuntimeError("Model returned the wrong prediction shape.")
    if task == "regression" and not np.isfinite(predictions.astype(float)).all():
        raise RuntimeError("Model returned non-finite regression predictions.")
    if task == "classification" and not pd.Series(predictions).isin(y.unique()).all():
        raise RuntimeError("Model returned labels that were not in the training set.")
    if task == "clustering":
        if not np.issubdtype(predictions.dtype, np.integer):
            raise RuntimeError("Cluster labels must be integers.")
        if ((predictions < 0) | (predictions >= n_clusters)).any():
            raise RuntimeError("Cluster labels are outside the requested range.")

    profile["schema_version"] = "student-v1"
    profile["scope"] = "training features before cleaning, with target summarized separately"
    profile["excluded_model_columns"] = excluded
    profile["model_feature_columns"] = list(X_train.columns)
    profile["prediction_rows"] = len(test)
    report = {
        "interface_version": "student-v1",
        "task": task,
        "target": target,
        "n_clusters": n_clusters if task == "clustering" else None,
        "seed": args.seed,
        "time_limit_seconds": args.time_limit,
        "training_file": args.train.name,
        "prediction_file": prediction_source.name,
        "training_rows": len(train),
        "prediction_rows": len(test),
        "model_feature_columns": list(X_train.columns),
        "notes": notes,
        "model": model_report,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    column = "cluster" if task == "clustering" else "prediction"
    prediction_path = args.output / "predictions.csv"
    temporary = prediction_path.with_suffix(".csv.tmp")
    pd.DataFrame({column: predictions}).to_csv(temporary, index=False)
    temporary.replace(prediction_path)
    write_json(args.output / "profile.json", profile)
    report["elapsed_seconds"] = round(time.monotonic() - started, 4)
    report["time_limit_exceeded"] = report["elapsed_seconds"] > args.time_limit
    write_json(args.output / "run_info.json", report)
    print(f"Finished {task}: {len(predictions)} predictions in {report['elapsed_seconds']:.2f}s.")
    print(f"Outputs: {args.output.resolve()}")


def main():
    started = time.monotonic()
    args = read_arguments()
    try:
        run(args, started)
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
