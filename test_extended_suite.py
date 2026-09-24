"""Comprehensive Test Suite for Extended ML-Project-01 Modules."""

import unittest
import numpy as np
import pandas as pd
from data_sanitizer import DataSanitizer
from feature_clusterer import FeatureClusterer
from model_calibrator import ModelCalibrator
from target_encoder import OutOfFoldTargetEncoder
from subgroup_auditor import SubgroupAuditor
from conformal_predictor import ConformalPredictor
from feature_hash_encoder import FeatureHashEncoder
from streaming_learner import StreamingOnlineLearner
from tabular_synthesizer import TabularCopulaSynthesizer
from shap_approximator import ShapleyApproximator
from drift_alerter import DriftAlerter, AlertPayloadBuilder
from model_lineage import LineageTracker
from sample_weigher import SampleWeigher
from feature_cache import FeatureStoreCache
from multiclass_evaluator import MulticlassEvaluator
from dataset_fingerprint import DatasetFingerprinter


class TestExtendedModules(unittest.TestCase):
    """Unit tests for the extended module suite."""

    def test_data_sanitizer(self):
        sanitizer = DataSanitizer()
        text = "Contact me at user@example.com or phone 123-456-7890."
        clean = sanitizer.sanitize_text(text)
        self.assertNotIn("user@example.com", clean)
        self.assertIn("[EMAIL_[REDACTED]]", clean)

    def test_conformal_predictor(self):
        conformal = ConformalPredictor(coverage_level=0.90)
        y_val_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        y_val_pred = np.array([10.5, 19.8, 30.2, 39.5, 50.1])
        conformal.calibrate(y_val_true, y_val_pred)
        intervals = conformal.predict_intervals(np.array([25.0, 35.0]))
        self.assertIn("lower_bound", intervals.columns)
        self.assertIn("upper_bound", intervals.columns)

    def test_feature_hash_encoder(self):
        df = pd.DataFrame({"city": ["London", "Paris", "Tokyo", "London"]})
        hasher = FeatureHashEncoder(n_features=8)
        transformed = hasher.fit_transform(df)
        self.assertEqual(transformed.shape[1], 8)
        self.assertIn("hash_feat_0", transformed.columns)

    def test_subgroup_auditor(self):
        df = pd.DataFrame({"cohort": ["A", "A", "B", "B"] * 10})
        y_true = pd.Series([10.0, 12.0, 20.0, 22.0] * 10)
        y_pred = np.array([10.1, 11.9, 25.0, 26.0] * 10)
        audit = SubgroupAuditor.audit_categorical_subgroups(df, "cohort", y_true, y_pred)
        self.assertEqual(len(audit), 2)

    def test_drift_alerter(self):
        alerter = DriftAlerter(platform="slack")
        record = alerter.send_alert("Model Drift Warning", severity="WARNING", details={"psi": 0.35})
        self.assertEqual(record["severity"], "WARNING")
        self.assertEqual(len(alerter.alert_history), 1)

    def test_feature_store_cache(self):
        cache = FeatureStoreCache(default_ttl_seconds=60)
        cache.put("user_123", {"feature_1": 1.5, "feature_2": 2.5})
        cached_feat = cache.get("user_123")
        self.assertIsNotNone(cached_feat)
        self.assertEqual(cached_feat["feature_1"], 1.5)

    def test_multiclass_evaluator(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 2, 2])
        report = MulticlassEvaluator.evaluate(y_true, y_pred)
        self.assertIn("accuracy", report)
        self.assertIn("f1_macro", report)

    def test_dataset_fingerprinter(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        fp = DatasetFingerprinter.generate_fingerprint(df)
        self.assertIn("content_sha256", fp)
        self.assertEqual(fp["shape"]["rows"], 3)

    def test_mase_loss(self):
        from loss_functions import mase_loss
        y_t = np.array([10.0, 12.0, 15.0])
        y_p = np.array([10.5, 12.2, 14.8])
        loss = mase_loss(y_t, y_p)
        self.assertGreaterEqual(loss, 0.0)

    def test_polynomial_decay(self):
        from learning_rate_scheduler import LRScheduler
        schedule = LRScheduler.generate_schedule("polynomial", initial_lr=0.01, total_epochs=10, power=2.0)
        self.assertEqual(len(schedule), 10)
        self.assertLessEqual(schedule[-1], schedule[0])


if __name__ == "__main__":
    unittest.main()
