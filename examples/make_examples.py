"""Create small synthetic examples. These are not the official practice datasets."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_classification, make_regression
from sklearn.model_selection import train_test_split


def save_example(root, task, X, y, metadata):
    folder = root / task
    folder.mkdir(parents=True, exist_ok=True)
    if task == "clustering":
        X.to_csv(folder / "data.csv", index=False)
        pd.DataFrame({"label": y}).to_csv(folder / "reference_labels.csv", index=False)
    else:
        train_idx, test_idx = train_test_split(
            np.arange(len(X)), test_size=0.2, random_state=21,
            stratify=y if task == "classification" else None,
        )
        train = X.iloc[train_idx].copy()
        train["target"] = y[train_idx]
        train.to_csv(folder / "train.csv", index=False)
        X.iloc[test_idx].to_csv(folder / "test.csv", index=False)
        pd.DataFrame({"target": y[test_idx]}).to_csv(folder / "reference_labels.csv", index=False)
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main():
    root = Path(__file__).resolve().parent
    rng = np.random.default_rng(21)

    values, y = make_classification(
        n_samples=600, n_features=8, n_informative=5, n_redundant=1,
        weights=[0.8, 0.2], class_sep=1.3, random_state=21,
    )
    X = pd.DataFrame(values, columns=[f"x{i}" for i in range(8)])
    X["group"] = np.where(X.x0 > 0, "A", "B")
    X["copy_x1"] = X["x1"]
    X["constant"] = 5
    X["empty"] = np.nan
    X.loc[rng.choice(len(X), 35, replace=False), "x2"] = np.nan
    save_example(root, "classification", X, y, {
        "task": "classification", "target": "target",
        "feature_types": {"group": "categorical"}, "metric": "balanced_accuracy",
    })

    values, y = make_regression(n_samples=600, n_features=6, n_informative=4,
                                noise=10, random_state=21)
    X = pd.DataFrame(values, columns=[f"x{i}" for i in range(6)])
    X["area"] = rng.choice(["north", "south", "east"], len(X))
    X["copy_x1"] = X["x1"]
    X["constant"] = 1
    # Different units are deliberate. The scaler is fitted on training rows only.
    X["x0"] *= 1000
    X.loc[rng.choice(len(X), 30, replace=False), "x2"] = np.nan
    save_example(root, "regression", X, y, {
        "task": "regression", "target": "target",
        "feature_types": {"area": "categorical"}, "metric": "rmse",
    })

    values, y = make_blobs(n_samples=450, centers=3, n_features=3,
                           cluster_std=0.8, random_state=21)
    X = pd.DataFrame(values, columns=["x", "y", "z"])
    X["x"] *= 100
    X["constant"] = 7
    save_example(root, "clustering", X, y, {
        "task": "clustering", "n_clusters": 3,
        "feature_types": {"x": "numeric", "y": "numeric", "z": "numeric"},
    })
    print(f"Created three synthetic examples in {root}")


if __name__ == "__main__":
    main()
