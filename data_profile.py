"""Create a JSON-friendly description of the input table."""

import numpy as np
import pandas as pd

from preprocessing import (
    find_duplicate_columns,
    infer_feature_types,
    missing_mask,
    normalise_table,
)


def _value(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _number(value):
    return float(value) if pd.notna(value) and np.isfinite(value) else None


def _numeric_summary(column):
    present = column.dropna()
    if present.empty:
        return {name: None for name in ["min", "max", "mean", "median", "std"]}
    return {
        "min": _number(present.min()),
        "max": _number(present.max()),
        "mean": _number(present.mean()),
        "median": _number(present.median()),
        "std": _number(present.std(ddof=0)),
    }


def make_profile(X, y=None, task=None, feature_types=None):
    """Profile features from the training table, without using evaluation labels."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError("The profile expects a pandas DataFrame.")
    if X.columns.duplicated().any():
        raise ValueError("Feature column names must be unique.")
    kinds = infer_feature_types(X, feature_types)
    clean = normalise_table(X, kinds)
    n_rows = len(X)
    columns = {}
    empty_columns = []
    constant_columns = []
    id_candidates = []

    for name in X.columns:
        raw = X[name]
        column = clean[name]
        missing = int(missing_mask(raw).sum())
        unusable = int(column.isna().sum())
        unique = int(column.nunique(dropna=True))
        info = {
            "type": kinds[name],
            "type_source": "provided" if name in (feature_types or {}) else "inferred",
            "original_dtype": str(raw.dtype),
            "missing_count": missing,
            "missing_fraction": missing / n_rows if n_rows else 0.0,
            "unusable_count": unusable,
            "unique_nonmissing": unique,
        }
        if unique == 0:
            empty_columns.append(name)
        elif column.nunique(dropna=False) <= 1:
            constant_columns.append(name)

        if kinds[name] == "numeric":
            info["statistics"] = _numeric_summary(column)
            info["invalid_or_nonfinite_count"] = max(0, unusable - missing)
        else:
            counts = column.value_counts(dropna=True)
            info["most_common"] = [
                {"value": _value(value), "count": int(count)}
                for value, count in counts.head(10).items()
            ]
            info["categories_omitted"] = max(0, unique - 10)

        # A unique column is only a hint. For example, price can also be unique.
        lower_name = str(name).lower()
        looks_named_id = lower_name == "id" or lower_name.endswith("_id") or lower_name.startswith("id_")
        if n_rows >= 3 and unique == n_rows and looks_named_id:
            id_candidates.append({"column": name, "reason": "ID-like name and one value per row"})
        columns[name] = info

    profile = {
        "schema_version": "student-v1",
        "task": task,
        "n_rows": n_rows,
        "n_features": X.shape[1],
        "feature_types": kinds,
        "numeric_columns": [name for name in X if kinds[name] == "numeric"],
        "categorical_columns": [name for name in X if kinds[name] == "categorical"],
        "missing_values_total": sum(info["missing_count"] for info in columns.values()),
        "unusable_values_total": sum(info["unusable_count"] for info in columns.values()),
        "all_missing_columns": empty_columns,
        "constant_columns": constant_columns,
        "duplicate_columns": find_duplicate_columns(clean, kinds),
        "duplicate_rows": int(clean.duplicated().sum()) if clean.shape[1] else max(0, n_rows - 1),
        "id_candidates": id_candidates,
        "columns": columns,
        "notes": [
            "Missing counts include nulls and blank strings.",
            "Unusable counts also include invalid numeric values and infinities.",
            "Constant and duplicate columns are checked after type conversion.",
            "ID candidates are reported only; ID columns are removed only when declared.",
            "Feature relevance and nonlinear relationships cannot be established from these summaries alone.",
        ],
    }

    if y is not None:
        target = pd.Series(y)
        if len(target) != n_rows:
            raise ValueError("Target length must match the number of training rows.")
        present = target.loc[~missing_mask(target)]
        target_info = {"missing_count": int(len(target) - len(present))}
        if task == "classification":
            counts = present.value_counts()
            target_info.update({
                "n_classes": len(counts),
                "class_counts": [
                    {"class": _value(value), "count": int(count), "fraction": int(count) / len(present)}
                    for value, count in counts.items()
                ],
                "majority_fraction": float(counts.max() / len(present)) if len(counts) else None,
                "imbalance_ratio": float(counts.max() / counts.min()) if len(counts) else None,
            })
        elif task == "regression":
            numeric = pd.to_numeric(present, errors="coerce").astype(float)
            numeric = numeric.where(np.isfinite(numeric), np.nan)
            target_info["invalid_or_nonfinite_count"] = int(numeric.isna().sum())
            target_info["statistics"] = _numeric_summary(numeric)
        profile["target"] = target_info
    return profile
