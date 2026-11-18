"""Small fast checks for leakage, reproducibility, and exact course inputs."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.run_experiments import Config, build_models, read_splits, run_bootstrap

ROOT = Path(__file__).resolve().parents[1]


class CourseworkTests(unittest.TestCase):
    def test_original_student_data_and_no_identifier_leakage(self):
        train, valid, features = read_splits(ROOT / "data" / "student")
        self.assertEqual(len(train), 341)
        self.assertEqual(len(valid), 114)
        self.assertEqual(len(features), 30)
        self.assertNotIn("sample_id", features)
        self.assertNotIn("malignant", features)
        self.assertEqual(int(train.malignant.sum()), 127)
        self.assertEqual(int(valid.malignant.sum()), 43)

    def test_lr_svm_normalize_within_pipeline(self):
        models = build_models(Config())
        for model in (models["Logistic regression"], models["RBF-SVM"]):
            self.assertIsInstance(model, Pipeline)
            self.assertEqual(model.steps[0][0], "standardscaler")

    def test_bootstrap_is_reproducible(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            params = Config(bootstrap_repeats=12)
            x = run_bootstrap(341, params, Path(a))
            y = run_bootstrap(341, params, Path(b))
            self.assertEqual(x, y)
            xfile = pd.read_csv(Path(a) / "C_bootstrap_1000.csv")
            self.assertEqual(len(xfile), 12)
            np.testing.assert_allclose(xfile.unique_fraction + xfile.oob_fraction, 1)

    def test_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            original = ROOT / "data" / "student"
            train = pd.read_csv(original / "train.csv")
            valid = pd.read_csv(original / "validation.csv")
            valid.loc[0, "sample_id"] = train.loc[0, "sample_id"]
            train.to_csv(root / "train.csv", index=False)
            valid.to_csv(root / "validation.csv", index=False)
            with self.assertRaisesRegex(ValueError, "share sample_id"):
                read_splits(root)


if __name__ == "__main__":
    unittest.main()
