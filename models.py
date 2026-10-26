"""Try a few models, compare them, and keep the best completed result."""

import multiprocessing as mp
import queue
import sys
import time
import warnings

import numpy as np
import pandas as pd


MAX_TRAIN_ROWS = 20_000
MAX_SILHOUETTE_ROWS = 800


def _sample_indices(y, count, limit, task, rng):
    """Keep the run small enough, while retaining every training class."""
    if count <= limit:
        return np.arange(count)
    if task != "classification":
        return np.sort(rng.choice(count, limit, replace=False))
    codes, _ = pd.factorize(y, sort=False)
    first = [rng.choice(np.flatnonzero(codes == code)) for code in np.unique(codes)]
    # Having more classes than the cap is unusual; retaining classes takes priority.
    limit = max(limit, len(first))
    remaining = np.setdiff1d(np.arange(count), first)
    extra = rng.choice(remaining, min(limit - len(first), len(remaining)), replace=False)
    return np.sort(np.concatenate([first, extra]).astype(int))


def _split_indices(y, task, rng):
    """Use about 20 percent for validation; singleton classes stay in training."""
    count = len(y)
    if count < 6:
        return np.arange(count), np.array([], dtype=int)
    if task == "regression":
        order = rng.permutation(count)
        size = max(2, int(round(count * 0.2)))
        return order[size:], order[:size]
    validation = []
    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        if len(indices) > 1:
            size = min(len(indices) - 1, max(1, int(round(len(indices) * 0.2))))
            validation.extend(rng.choice(indices, size, replace=False).tolist())
    validation = np.asarray(sorted(validation), dtype=int)
    training = np.setdiff1d(np.arange(count), validation)
    return training, validation


def _metrics(actual, predicted, task):
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, f1_score,
        mean_absolute_error, mean_squared_error, r2_score,
    )

    if task == "classification":
        result = {
            "accuracy": accuracy_score(actual, predicted),
            "balanced_accuracy": balanced_accuracy_score(actual, predicted),
            "f1_macro": f1_score(actual, predicted, average="macro", zero_division=0),
        }
    else:
        result = {
            "rmse": np.sqrt(mean_squared_error(actual, predicted)),
            "mae": mean_absolute_error(actual, predicted),
            "r2": r2_score(actual, predicted) if len(actual) >= 2 else None,
        }
    return {name: float(value) if value is not None and np.isfinite(value) else None
            for name, value in result.items()}


def _score(metrics, metric):
    value = metrics.get(metric)
    if value is None:
        return -np.inf
    return -value if metric in {"rmse", "mae"} else value


def _candidates(task, feature_types, seed, imbalanced):
    from sklearn.ensemble import (
        ExtraTreesClassifier, ExtraTreesRegressor,
        HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    )
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import Pipeline
    from preprocessing import make_preprocessor

    if task == "classification":
        weights = "balanced" if imbalanced else None
        choices = [
            ("logistic_regression", "onehot", True,
             LogisticRegression(max_iter=300, class_weight=weights, random_state=seed)),
            ("extra_trees", "onehot", False,
             ExtraTreesClassifier(n_estimators=100, max_features=0.8,
                                  min_samples_leaf=1, class_weight=weights,
                                  n_jobs=1, random_state=seed)),
            ("hist_gradient_boosting", "ordinal", False,
             HistGradientBoostingClassifier(max_iter=140, max_leaf_nodes=31,
                                            l2_regularization=1.0, class_weight=weights,
                                            categorical_features=None, random_state=seed)),
        ]
    else:
        choices = [
            ("ridge_regression", "onehot", True, Ridge(alpha=1.0, solver="lsqr")),
            ("extra_trees", "onehot", False,
             ExtraTreesRegressor(n_estimators=100, max_features=1.0,
                                 min_samples_leaf=2, n_jobs=1, random_state=seed)),
            ("hist_gradient_boosting", "ordinal", False,
             HistGradientBoostingRegressor(max_iter=180, max_leaf_nodes=31,
                                           l2_regularization=1.0,
                                           categorical_features=None, random_state=seed)),
        ]
    return [
        (name, Pipeline([
            ("prepare", make_preprocessor(feature_types, scale, encoding)),
            ("model", model),
        ]))
        for name, encoding, scale, model in choices
    ]


def _supervised_worker(X, y, X_predict, task, types, seed, metric, messages):
    from sklearn.base import clone

    rng = np.random.default_rng(seed)
    indices = _sample_indices(y, len(X), MAX_TRAIN_ROWS, task, rng)
    X_fit = X.iloc[indices].reset_index(drop=True)
    y_fit = np.asarray(y)[indices]
    labels = None
    if task == "classification":
        y_fit, labels = pd.factorize(y_fit, sort=False)
        counts = np.bincount(y_fit)
        imbalanced = counts.max() / counts.min() >= 1.5
    else:
        y_fit = y_fit.astype(float)
        imbalanced = False
    train, valid = _split_indices(y_fit, task, rng)
    notes = []
    if task == "classification" and len(valid):
        excluded = len(np.setdiff1d(np.unique(y_fit), np.unique(y_fit[valid])))
        if excluded:
            notes.append(f"{excluded} singleton classes remain in training and have no validation rows.")
    if not len(valid):
        notes.append("Too few usable rows for a validation split; the first model that fits is used.")
    setup = {"search_rows_used": len(X_fit), "validation_rows": len(valid), "notes": notes}
    candidates = []
    best_score = -np.inf
    if len(valid):
        if task == "classification":
            constant = np.bincount(y_fit[train]).argmax()
        else:
            constant = np.median(y_fit[train]) if metric == "mae" else np.mean(y_fit[train])
        baseline = _metrics(y_fit[valid], np.repeat(constant, len(valid)), task)
        best_score = _score(baseline, metric)
        setup["baseline_validation_metrics"] = baseline
        setup["validation_metrics"] = baseline
        setup["emergency_fallback"] = False
    messages.put({"kind": "report", "report": setup})

    for name, pipeline in _candidates(task, types, seed, imbalanced):
        started = time.monotonic()
        # The ordinal matrix has at most twice the original column count.
        if name == "hist_gradient_boosting" and len(X_fit) * max(1, X.shape[1]) > 4_000_000:
            candidates.append({"model": name, "status": "skipped", "reason": "Dense matrix size limit."})
            messages.put({"kind": "report", "report": {"candidates": list(candidates)}})
            continue
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                pipeline.fit(X_fit.iloc[train], y_fit[train])
                validation = _metrics(y_fit[valid], pipeline.predict(X_fit.iloc[valid]), task) if len(valid) else {}
                candidate_score = _score(validation, metric) if len(valid) else 0.0
                entry = {"model": name, "status": "evaluated", "validation_metrics": validation}
                # Refit only after choosing by validation, using the training table alone.
                if candidate_score > best_score:
                    fitted = clone(pipeline).fit(X_fit, y_fit)
                    predictions = fitted.predict(X_predict)
                    if labels is not None:
                        predictions = np.asarray(labels)[np.asarray(predictions, dtype=int)]
                    if task == "regression" and not np.isfinite(predictions).all():
                        raise ValueError("The model produced non-finite predictions.")
                    entry["status"] = "completed"
                    entry["seconds"] = round(time.monotonic() - started, 4)
                    if caught:
                        entry["warnings"] = list(dict.fromkeys(str(w.message) for w in caught))[:3]
                    candidates.append(entry)
                    clean = fitted.named_steps["prepare"].named_steps["clean"]
                    messages.put({
                        "kind": "prediction", "predictions": predictions,
                        "report": {
                            "selected_model": name, "validation_metrics": validation,
                            "candidates": list(candidates),
                            "dropped_feature_columns": list(clean.dropped_columns_),
                            "train_rows_used": len(X_fit),
                            "sampled_for_speed": len(X_fit) < len(X),
                            "emergency_fallback": False,
                        },
                    })
                    best_score = candidate_score
                    if not len(valid):
                        break
                    continue
                entry["seconds"] = round(time.monotonic() - started, 4)
                if caught:
                    entry["warnings"] = list(dict.fromkeys(str(w.message) for w in caught))[:3]
                candidates.append(entry)
        except Exception as error:
            candidates.append({"model": name, "status": "failed", "reason": str(error)[:400]})
        messages.put({"kind": "report", "report": {"candidates": list(candidates)}})


def _clustering_worker(X, X_predict, types, k, seed, messages):
    from scipy import sparse
    from sklearn.cluster import KMeans, MiniBatchKMeans
    from sklearn.decomposition import PCA, TruncatedSVD
    from sklearn.metrics import silhouette_score
    from sklearn.mixture import GaussianMixture
    from preprocessing import make_preprocessor

    rng = np.random.default_rng(seed)
    limit = max(MAX_TRAIN_ROWS, k)
    indices = _sample_indices(None, len(X), limit, "clustering", rng)
    X_fit = X.iloc[indices]
    prepare = make_preprocessor(types, scale_numeric=True)
    data = prepare.fit_transform(X_fit)
    prediction_data = prepare.transform(X_predict)
    notes = []
    if data.shape[1] > 60 and min(data.shape) > 2:
        components = min(40, data.shape[0] - 1, data.shape[1] - 1)
        if sparse.issparse(data):
            reduce = TruncatedSVD(n_components=components, n_iter=3, random_state=seed)
        else:
            reduce = PCA(n_components=components, svd_solver="randomized", random_state=seed)
        data = reduce.fit_transform(data)
        prediction_data = reduce.transform(prediction_data)
        notes.append(f"Clustering features reduced to {components} dimensions for speed.")
    messages.put({"kind": "report", "report": {
        "search_rows_used": len(X_fit),
        "validation_rows": 0, "notes": notes,
        "clustering_dimensions": int(data.shape[1]),
        "dropped_feature_columns": list(prepare.named_steps["clean"].dropped_columns_),
    }})

    if len(X_fit) <= 10_000 and k <= 50:
        choices = [("kmeans", KMeans(n_clusters=k, n_init=5, max_iter=200, random_state=seed))]
    else:
        choices = [("minibatch_kmeans", MiniBatchKMeans(
            n_clusters=k, n_init=3, batch_size=max(1024, k), max_iter=100, random_state=seed))]
    if k <= 30 and len(X_fit) >= max(2 * k, 10) and data.shape[1] <= 60:
        choices.append(("gaussian_mixture", GaussianMixture(
            n_components=k, covariance_type="full" if data.shape[1] <= 15 else "diag",
            max_iter=100, n_init=1, reg_covar=1e-4, random_state=seed,
        )))
    sample = rng.choice(len(X_fit), min(MAX_SILHOUETTE_ROWS, len(X_fit)), replace=False)
    best_score = -np.inf
    has_model = False
    candidates = []
    for name, model in choices:
        started = time.monotonic()
        try:
            # GMM needs dense input, but dimensions and row count are bounded here.
            train_data, test_data = data, prediction_data
            if name == "gaussian_mixture" and sparse.issparse(data):
                train_data, test_data = data.toarray(), prediction_data.toarray()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model.fit(train_data)
                training_labels = model.predict(train_data)
                sampled_labels = training_labels[sample]
                distinct = len(np.unique(sampled_labels))
                silhouette = None
                if 1 < distinct < len(sample):
                    silhouette = float(silhouette_score(data[sample], sampled_labels))
                    if not np.isfinite(silhouette):
                        silhouette = None
                score = silhouette if silhouette is not None else -np.inf
                entry = {
                    "model": name, "status": "evaluated",
                    "validation_metrics": {"silhouette": silhouette},
                    "training_groups_found": int(len(np.unique(training_labels))),
                    "seconds": round(time.monotonic() - started, 4),
                }
                if caught:
                    entry["warnings"] = list(dict.fromkeys(str(w.message) for w in caught))[:3]
                if not has_model or score > best_score:
                    predictions = model.predict(test_data).astype(int)
                    entry["status"] = "completed"
                    candidates.append(entry)
                    messages.put({"kind": "prediction", "predictions": predictions, "report": {
                        "selected_model": name, "validation_metrics": {"silhouette": silhouette},
                        "silhouette_sample_rows": len(sample),
                        "training_groups_found": int(len(np.unique(training_labels))),
                        "prediction_groups_found": int(len(np.unique(predictions))),
                        "train_rows_used": len(X_fit),
                        "sampled_for_speed": len(X_fit) < len(X),
                        "candidates": list(candidates), "emergency_fallback": False,
                    }})
                    best_score, has_model = score, True
                    continue
                candidates.append(entry)
        except Exception as error:
            candidates.append({"model": name, "status": "failed", "reason": str(error)[:400]})
        messages.put({"kind": "report", "report": {"candidates": list(candidates)}})


def _worker(X, y, X_predict, task, types, k, seed, metric, messages):
    from threadpoolctl import threadpool_limits

    try:
        with threadpool_limits(limits=1):
            if task == "clustering":
                _clustering_worker(X, X_predict, types, k, seed, messages)
            else:
                _supervised_worker(X, y, X_predict, task, types, seed, metric, messages)
    except Exception as error:
        messages.put({"kind": "report", "report": {"worker_error": str(error)[:500]}})
    finally:
        messages.put({"kind": "done"})


def _fallback(X, y, X_predict, task, k, metric):
    """There is always a valid output if the training process runs out of time."""
    if task == "classification":
        common = pd.Series(y).value_counts().index[0]
        return np.repeat(common, len(X_predict)), "majority_class"
    if task == "regression":
        center = np.median(y) if metric == "mae" else np.mean(y)
        return np.repeat(float(center), len(X_predict)), "median" if metric == "mae" else "mean"
    # Quantile bins provide a cheap, explicit emergency grouping. They are only
    # used until an actual clustering model finishes, and may produce fewer than k groups.
    for column in X.columns[:20]:
        values = pd.to_numeric(X[column].iloc[:5000], errors="coerce")
        finite = values[np.isfinite(values)]
        if finite.nunique() >= 2:
            edges = np.unique(np.quantile(finite, np.linspace(0, 1, k + 1)[1:-1]))
            new = pd.to_numeric(X_predict[column], errors="coerce").to_numpy(dtype=float)
            new = np.where(np.isfinite(new), new, float(finite.median()))
            return np.searchsorted(edges, new, side="right"), "emergency_quantile_bins"
    return np.zeros(len(X_predict), dtype=int), "emergency_single_group"


def solve_dataset(X_train, y_train, X_predict, task, feature_types=None,
                  n_clusters=None, random_state=42, time_budget=45.0, metric=None):
    """Return predictions and a report; stop unfinished fitting at the deadline.

    Use a script with an ``if __name__ == '__main__'`` guard. Linux can reuse the
    imported libraries with fork; other systems use multiprocessing's spawn method.
    """
    started = time.monotonic()
    if task not in {"classification", "regression", "clustering"}:
        raise ValueError("Unknown task.")
    if len(X_train) == 0:
        raise ValueError("Training data is empty.")
    defaults = {"classification": "balanced_accuracy", "regression": "rmse", "clustering": "silhouette"}
    metric = metric or defaults[task]
    allowed = {"classification": {"balanced_accuracy", "accuracy", "f1_macro"},
               "regression": {"rmse", "mae", "r2"}, "clustering": {"silhouette"}}
    if metric not in allowed[task]:
        raise ValueError(f"Metric {metric!r} is unsupported for {task}.")
    if task == "clustering" and (not isinstance(n_clusters, (int, np.integer)) or not 1 <= n_clusters <= len(X_train)):
        raise ValueError("n_clusters must be an integer between 1 and the training row count.")
    if task != "clustering" and (y_train is None or len(y_train) != len(X_train)):
        raise ValueError("Training targets must match the training rows.")
    if task != "clustering" and pd.isna(np.asarray(y_train)).any():
        raise ValueError("Training targets must not contain missing values.")
    if task == "regression" and not np.isfinite(np.asarray(y_train, dtype=float)).all():
        raise ValueError("Regression targets must be finite numbers.")
    if not np.isfinite(time_budget) or time_budget < 0:
        raise ValueError("time_budget must be a nonnegative finite number.")

    predictions, name = _fallback(X_train, y_train, X_predict, task, n_clusters, metric)
    report = {
        "selected_model": name, "selection_metric": metric,
        "validation_metrics": {}, "candidates": [],
        "train_rows_available": len(X_train), "train_rows_used": len(X_train),
        "sampled_for_speed": False, "validation_rows": 0,
        "time_limit_reached": False, "emergency_fallback": True,
        "notes": [], "model_time_budget_seconds": float(time_budget),
    }
    if task == "clustering":
        report["requested_clusters"] = int(n_clusters)
        report["prediction_groups_found"] = int(len(np.unique(predictions)))
    constant_target = task != "clustering" and pd.Series(y_train).nunique() == 1
    if constant_target or (task == "clustering" and n_clusters == 1) or len(X_predict) == 0:
        if task == "clustering" and n_clusters == 1:
            predictions = np.zeros(len(X_predict), dtype=int)
            report["selected_model"] = "one_cluster"
        report["emergency_fallback"] = False
        report["notes"].append("A constant target, one requested cluster, or no prediction rows needs no model search.")
        report["modeling_seconds"] = round(time.monotonic() - started, 4)
        return predictions, report

    if time.monotonic() >= started + time_budget:
        report["time_limit_reached"] = True
        report["notes"].append("No model time remained, so the emergency baseline was returned.")
        report["modeling_seconds"] = round(time.monotonic() - started, 4)
        return predictions, report

    method = "fork" if sys.platform.startswith("linux") and "fork" in mp.get_all_start_methods() else "spawn"
    report["process_start_method"] = method
    context = mp.get_context(method)
    messages = context.Queue()
    process = context.Process(target=_worker, args=(
        X_train, y_train, X_predict, task, feature_types, n_clusters,
        random_state, metric, messages,
    ))
    process.daemon = True
    deadline = started + time_budget
    try:
        process.start()
        finished = False
        while time.monotonic() < deadline:
            try:
                message = messages.get(timeout=min(0.1, max(0.001, deadline - time.monotonic())))
            except queue.Empty:
                if not process.is_alive():
                    break
                continue
            if message["kind"] == "done":
                finished = True
                break
            if message["kind"] == "prediction":
                predictions = message["predictions"]
            report.update(message.get("report", {}))
        report["time_limit_reached"] = not finished and time.monotonic() >= deadline
    except Exception as error:
        report["worker_error"] = str(error)[:500]
    finally:
        if process.pid is not None:
            if process.is_alive():
                process.terminate()
            process.join(timeout=0.2)
            if process.is_alive():
                process.kill()
                process.join(timeout=0.2)
        messages.close()
        messages.cancel_join_thread()
    if report["time_limit_reached"]:
        report["notes"].append("The model deadline stopped unfinished work; the best completed predictions were kept.")
    if report["emergency_fallback"]:
        report["notes"].append("No candidate improved the baseline and completed its full fit before stopping.")
    report["modeling_seconds"] = round(time.monotonic() - started, 4)
    return np.asarray(predictions), report
