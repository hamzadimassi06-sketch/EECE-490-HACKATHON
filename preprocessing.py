"""Cleaning and encoding shared by the three tasks."""

import hashlib

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted


def missing_mask(series):
    """CSV blanks count as missing, including cells containing only spaces."""
    blank = series.map(lambda value: isinstance(value, str) and not value.strip())
    return series.isna() | blank


def infer_feature_types(X, feature_types=None):
    """Use the supplied types first, then infer any columns left unspecified."""
    feature_types = feature_types or {}
    numeric_names = {"numeric", "numerical", "number", "continuous", "integer", "int", "float"}
    categorical_names = {"categorical", "category", "nominal", "ordinal", "string", "object", "bool", "boolean"}
    result = {}
    for name in X.columns:
        if name in feature_types:
            kind = str(feature_types[name]).lower()
            if kind in numeric_names:
                result[name] = "numeric"
            elif kind in categorical_names:
                result[name] = "categorical"
            else:
                raise ValueError(f"Unknown feature type for {name}: {feature_types[name]}")
            continue

        column = X[name]
        if pd.api.types.is_bool_dtype(column):
            result[name] = "categorical"
        elif pd.api.types.is_numeric_dtype(column):
            result[name] = "numeric"
        else:
            present = column.loc[~missing_mask(column)]
            converted = pd.to_numeric(present, errors="coerce")
            # A mostly numeric column can contain a small number of typing errors.
            fraction = converted.notna().mean() if len(present) else 0
            result[name] = "numeric" if fraction >= 0.95 else "categorical"
    return result


def normalise_table(X, feature_types):
    """Convert each column consistently without learning any statistics."""
    result = pd.DataFrame(index=X.index)
    for name, kind in feature_types.items():
        column = X[name]
        if kind == "numeric":
            values = pd.to_numeric(column, errors="coerce").astype(float)
            result[name] = values.where(np.isfinite(values), np.nan)
        else:
            def clean_category(value):
                if pd.isna(value):
                    return np.nan
                if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                    return np.nan
                if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_)):
                    # A missing value can make pandas read integer codes as floats.
                    # Keep code 1.0 in training equal to code 1 in prediction data.
                    if float(value).is_integer():
                        return str(int(value))
                text = str(value).strip()
                return text if text else np.nan

            # Mixed numbers and strings are converted to strings before encoding.
            result[name] = column.map(clean_category).astype(object)
    return result


def find_duplicate_columns(X, feature_types=None):
    """Hash first, then compare matching columns so hash collisions are harmless."""
    groups = {}
    duplicates = {}
    for name in X.columns:
        hashed = pd.util.hash_pandas_object(X[name], index=False).to_numpy()
        digest = hashlib.sha256(hashed.tobytes()).digest()
        kind = feature_types.get(name) if feature_types else str(X[name].dtype)
        key = (kind, digest)
        candidates = groups.setdefault(key, [])
        match = next((other for other in candidates if X[name].equals(X[other])), None)
        if match is None:
            candidates.append(name)
        else:
            duplicates[name] = match
    return duplicates


class TableCleaner(TransformerMixin, BaseEstimator):
    """Learn which columns to keep from the training fold only."""

    def __init__(self, feature_types=None):
        self.feature_types = feature_types

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("The preprocessor expects a pandas DataFrame.")
        if X.columns.duplicated().any():
            raise ValueError("Feature column names must be unique.")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.feature_types_ = infer_feature_types(X, self.feature_types)
        clean = normalise_table(X, self.feature_types_)
        self.empty_columns_ = [name for name in clean if clean[name].isna().all()]
        self.constant_columns_ = [
            name for name in clean
            if name not in self.empty_columns_ and clean[name].nunique(dropna=False) <= 1
        ]
        self.duplicate_columns_ = find_duplicate_columns(clean, self.feature_types_)
        drop = set(self.empty_columns_) | set(self.constant_columns_) | set(self.duplicate_columns_)
        self.dropped_columns_ = [name for name in X.columns if name in drop]
        self.kept_columns_ = [name for name in X.columns if name not in drop]
        return self

    def transform(self, X):
        check_is_fitted(self, "kept_columns_")
        missing = [name for name in self.feature_names_in_ if name not in X.columns]
        if missing:
            raise ValueError(f"Prediction data is missing feature columns: {missing}")
        if not self.kept_columns_:
            # A dummy input still lets the model predict a mean or majority class.
            return pd.DataFrame({"__constant_input__": np.zeros(len(X))}, index=X.index)
        kinds = {name: self.feature_types_[name] for name in self.kept_columns_}
        return normalise_table(X, kinds)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "kept_columns_")
        return np.asarray(self.kept_columns_ or ["__constant_input__"], dtype=object)


def make_preprocessor(feature_types=None, scale_numeric=True, encoding="onehot"):
    """Return a fresh transformer; fit it inside the model pipeline."""
    if encoding not in {"onehot", "ordinal"}:
        raise ValueError("encoding must be 'onehot' or 'ordinal'")

    numeric_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    numeric = Pipeline(numeric_steps)

    if encoding == "onehot":
        encoder = OneHotEncoder(
            handle_unknown="ignore", max_categories=40, sparse_output=True
        )
    else:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent", add_indicator=True)),
        ("encode", encoder),
    ])

    columns = ColumnTransformer(
        transformers=[
            ("numeric", numeric, make_column_selector(dtype_include=np.number)),
            ("categorical", categorical, make_column_selector(dtype_exclude=np.number)),
        ],
        # A mostly numeric table is faster as a dense matrix. Keep sparse
        # storage for wide one-hot tables that actually contain mostly zeros.
        sparse_threshold=0.3 if encoding == "onehot" else 0.0,
    )
    return Pipeline([("clean", TableCleaner(feature_types)), ("columns", columns)])
