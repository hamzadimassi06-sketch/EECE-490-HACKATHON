"""Small checks for leakage, awkward tables, and the command-line output."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from data_profile import make_profile
from models import solve_dataset
from preprocessing import make_preprocessor


def dense(values):
    return values.toarray() if hasattr(values, "toarray") else np.asarray(values)


def strict_json(value):
    """NaN and infinity are not valid numbers in a standard JSON file."""
    return json.loads(json.dumps(value, allow_nan=False))


class PreprocessingTests(unittest.TestCase):
    def test_prediction_rows_do_not_change_training_statistics(self):
        train = pd.DataFrame({
            "reading": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
            "copy": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
            "group": ["A", "B", "A", None, "B", "A"],
            "constant": [9] * 6,
            "empty": [None] * 6,
        })
        original = train.copy(deep=True)
        transformer = make_preprocessor()
        before = dense(transformer.fit_transform(train))
        prediction_rows = pd.DataFrame({
            "reading": [1e12, np.nan],
            "copy": [1e12, np.nan],
            "group": ["new category", None],
            "constant": [-100, 200],
            "empty": ["first value", None],
        })
        transformed = dense(transformer.transform(prediction_rows))
        after = dense(transformer.transform(train))

        np.testing.assert_allclose(before, after, rtol=0, atol=0)
        self.assertTrue(np.isfinite(transformed).all())
        self.assertEqual(transformed.shape, (2, before.shape[1]))
        pd.testing.assert_frame_equal(train, original)

        # Column order may change in the CSV; row order must stay intact.
        reversed_columns = train.iloc[[4, 0, 2], ::-1]
        np.testing.assert_allclose(
            dense(transformer.transform(reversed_columns)), before[[4, 0, 2]]
        )

    def test_declared_types_override_storage_types(self):
        train = pd.DataFrame({
            "amount": ["1", "2", "3", "4", "5", "6"],
            "code": [10, 20, 30, 10, 20, 30],
        })
        transformer = make_preprocessor(
            feature_types={"amount": "numeric", "code": "categorical"}
        )
        transformer.fit(train)
        values = dense(transformer.transform(pd.DataFrame({
            "amount": ["1", "2", "3"], "code": [10, 10, 10]
        })))
        # Numeric strings should keep their numerical spacing after scaling.
        np.testing.assert_allclose(values[1], (values[0] + values[2]) / 2)
        self.assertGreater(np.linalg.norm(values[2] - values[0]), 0)

        codes = dense(transformer.transform(pd.DataFrame({
            "amount": ["2", "2", "2"], "code": [10, 20, 30]
        })))
        # Category 30 should not be twice as far from category 10 as 20 is.
        self.assertAlmostEqual(
            np.linalg.norm(codes[1] - codes[0]),
            np.linalg.norm(codes[2] - codes[0]),
        )
        self.assertGreater(np.linalg.norm(codes[1] - codes[0]), 0)

    def test_all_missing_and_constant_columns_still_give_usable_input(self):
        train = pd.DataFrame({"empty": [np.nan] * 4, "same": [5] * 4})
        for encoding in ("onehot", "ordinal"):
            with self.subTest(encoding=encoding):
                transformer = make_preprocessor(encoding=encoding)
                transformed = dense(transformer.fit_transform(train))
                self.assertEqual(transformed.shape[0], 4)
                self.assertGreater(transformed.shape[1], 0)
                self.assertTrue(np.isfinite(transformed).all())
                self.assertTrue(np.isfinite(dense(
                    transformer.transform(train.iloc[:1])
                )).all())

    def test_missing_feature_is_reported_instead_of_silently_filled(self):
        train = pd.DataFrame({"a": [1, 2, 3], "b": [4, 2, 7]})
        transformer = make_preprocessor().fit(train)
        with self.assertRaises(ValueError):
            transformer.transform(pd.DataFrame({"a": [1]}))


class ProfileTests(unittest.TestCase):
    def test_profile_keeps_raw_data_problems_visible(self):
        data = pd.DataFrame({
            "temperature": [1.0, np.nan, 3.0, np.inf],
            "temperature_copy": [1.0, np.nan, 3.0, np.inf],
            "empty": [None] * 4,
            "constant": [7] * 4,
            "category": ["A", " ", "B", "A"],
        })
        result = strict_json(make_profile(
            data, pd.Series(["yes", "no", "no", "no"]), "classification"
        ))
        self.assertEqual(result["n_rows"], 4)
        self.assertEqual(result["n_features"], 5)
        self.assertEqual(result["columns"]["temperature"]["missing_count"], 1)
        self.assertEqual(
            result["columns"]["temperature"]["invalid_or_nonfinite_count"], 1
        )
        self.assertEqual(result["columns"]["category"]["missing_count"], 1)
        self.assertIn("empty", result["all_missing_columns"])
        self.assertIn("constant", result["constant_columns"])
        self.assertEqual(result["duplicate_columns"]["temperature_copy"], "temperature")
        self.assertEqual(result["target"]["n_classes"], 2)
        self.assertAlmostEqual(result["target"]["majority_fraction"], 0.75)


class ModelTests(unittest.TestCase):
    def test_exhausted_budget_still_returns_valid_predictions(self):
        data = pd.DataFrame({"x": np.arange(30)})
        predicted, report = solve_dataset(
            data, pd.Series(["usual"] * 20 + ["rare"] * 10),
            data.iloc[[2, 1, 5]], "classification", time_budget=0.0,
        )
        self.assertEqual(list(predicted), ["usual"] * 3)
        self.assertTrue(report["time_limit_reached"])
        strict_json(report)

    def test_training_deadline_returns_a_completed_output(self):
        rng = np.random.default_rng(12)
        data = pd.DataFrame(rng.normal(size=(6000, 80)))
        labels = (data[0] + data[1] > 0).astype(int)
        started = time.monotonic()
        predicted, report = solve_dataset(
            data, labels, data.iloc[:7], "classification", time_budget=0.5,
        )
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 5.0)
        self.assertEqual(len(predicted), 7)
        self.assertTrue(set(predicted).issubset({0, 1}))
        self.assertTrue(report["time_limit_reached"])
        strict_json(report)

    def test_one_class_and_a_rare_class_do_not_break_validation(self):
        data = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]})
        predicted, report = solve_dataset(
            data, pd.Series(["only"] * 6), data.iloc[[4, 0]],
            "classification", time_budget=8.0,
        )
        self.assertEqual(list(predicted), ["only", "only"])
        strict_json(report)

        predicted, report = solve_dataset(
            data.iloc[:3], pd.Series(["rare", "usual", "usual"]),
            data.iloc[[2, 0]], "classification", time_budget=8.0,
        )
        self.assertEqual(len(predicted), 2)
        self.assertTrue(set(predicted).issubset({"rare", "usual"}))
        strict_json(report)

    def test_constant_regression_target_is_finite(self):
        data = pd.DataFrame({"x": [1, 2, 3, 4, 5], "empty": [None] * 5})
        predicted, report = solve_dataset(
            data, pd.Series([12.5] * 5), data.iloc[[4, 1, 4]],
            "regression", time_budget=8.0,
        )
        np.testing.assert_allclose(predicted, [12.5] * 3)
        self.assertTrue(np.isfinite(predicted).all())
        strict_json(report)

    def test_clustering_uses_the_requested_number_of_groups(self):
        data = pd.DataFrame({
            "x": [-10.0, -10.2, -9.8, 10.0, 10.2, 9.8],
            "y": [-10.0, -9.8, -10.2, 10.0, 9.8, 10.2],
        })
        predicted, report = solve_dataset(
            data, None, data, "clustering", n_clusters=2, time_budget=8.0,
        )
        self.assertEqual(len(predicted), len(data))
        self.assertEqual(len(set(predicted)), 2)
        self.assertEqual(len(set(predicted[:3])), 1)
        self.assertEqual(len(set(predicted[3:])), 1)
        strict_json(report)


class CommandLineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def invoke(self, train, metadata, test=None, name="run", success=True):
        folder = self.directory / name
        folder.mkdir()
        train_path = folder / "train.csv"
        if isinstance(train, str):
            train_path.write_text(train, encoding="utf-8")
        else:
            train.to_csv(train_path, index=False)
        metadata_path = folder / "metadata.json"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        output = folder / "output"
        command = [
            sys.executable, str(ROOT / "main.py"), "--train", str(train_path),
            "--metadata", str(metadata_path), "--output", str(output),
            "--time-limit", "8",
        ]
        if test is not None:
            test_path = folder / "test.csv"
            test.to_csv(test_path, index=False)
            command.extend(["--test", str(test_path)])
        process = subprocess.run(
            command, capture_output=True, text=True, timeout=25, cwd=ROOT,
        )
        if not success:
            self.assertNotEqual(process.returncode, 0, process.stdout)
            return process
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        predictions = pd.read_csv(output / "predictions.csv")
        with (output / "profile.json").open(encoding="utf-8") as handle:
            profile = strict_json(json.load(handle))
        with (output / "run_info.json").open(encoding="utf-8") as handle:
            run_info = strict_json(json.load(handle))
        return predictions, profile, run_info

    def test_classification_ignores_ids_and_evaluation_labels(self):
        x = np.linspace(-3, 3, 60)
        train = pd.DataFrame({
            "row_id": np.arange(60), "signal": x,
            "kind": np.where(x > 0, "upper", "lower"),
            "target": np.where(x > 0, "yes", "no"),
        })
        test = pd.DataFrame({
            "row_id": [100, 101, 102, 103],
            "signal": [2.0, -2.0, 2.0, -1.5],
            "kind": ["upper", "lower", "upper", "unseen"],
            "target": ["no", "yes", "no", "yes"],
        })
        metadata = {
            "task": "classification", "target": "target", "id_columns": ["row_id"],
            "feature_types": {"signal": "numeric", "kind": "categorical"},
        }
        first, profile, _ = self.invoke(train, metadata, test, name="first")
        altered = test.copy()
        altered["row_id"] = [-900, 8_000_000, 0, 5]
        altered["target"] = ["yes", "no", "yes", "no"]
        second, _, _ = self.invoke(train, metadata, altered, name="second")

        self.assertEqual(list(first.columns), ["prediction"])
        self.assertEqual(len(first), len(test))
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first.iloc[0, 0], first.iloc[2, 0])
        self.assertNotIn("row_id", profile["model_feature_columns"])
        self.assertNotIn("target", profile["model_feature_columns"])
        self.assertNotIn("target", profile["columns"])
        self.assertTrue(set(first["prediction"]).issubset({"yes", "no"}))

    def test_regression_outputs_numbers_in_prediction_order(self):
        x = np.linspace(-3, 3, 60)
        train = pd.DataFrame({"x": x, "cost": 2 * x + 5})
        test = pd.DataFrame({"x": [2.0, -1.0, 2.0, 0.5]})
        predictions, _, _ = self.invoke(
            train, {"task": "regression", "target": "cost"}, test
        )
        self.assertEqual(list(predictions.columns), ["prediction"])
        self.assertEqual(len(predictions), len(test))
        self.assertTrue(np.isfinite(predictions["prediction"]).all())
        self.assertAlmostEqual(predictions.iloc[0, 0], predictions.iloc[2, 0])

    def test_clustering_without_a_test_file_returns_one_label_per_row(self):
        data = pd.DataFrame({
            "x": [-9.0, -10.0, -11.0, 9.0, 10.0, 11.0],
            "y": [-10.0, -9.0, -11.0, 10.0, 9.0, 11.0],
        })
        predictions, profile, _ = self.invoke(
            data, {"task": "clustering", "n_clusters": 2}
        )
        self.assertEqual(list(predictions.columns), ["cluster"])
        self.assertEqual(len(predictions), len(data))
        self.assertEqual(predictions["cluster"].nunique(), 2)
        self.assertEqual(profile["n_rows"], len(data))

    def test_duplicate_csv_headers_are_rejected(self):
        self.invoke(
            "x,x,target\n1,3,no\n2,4,yes\n3,5,yes\n",
            {"task": "classification", "target": "target"},
            pd.DataFrame({"x": [1, 2]}), success=False,
        )

    def test_missing_test_feature_is_rejected(self):
        self.invoke(
            pd.DataFrame({"x": [1, 2, 3], "z": [7, 9, 8], "target": [2, 4, 6]}),
            {"task": "regression", "target": "target"},
            pd.DataFrame({"x": [1, 2]}), success=False,
        )


if __name__ == "__main__":
    unittest.main()
