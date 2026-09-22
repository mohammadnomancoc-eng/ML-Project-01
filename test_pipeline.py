"""Comprehensive Test Suite for ML-Project-01 Core and Advanced Modules."""

import unittest
import numpy as np
import pandas as pd
from data_loader import DataLoader
from preprocessor import DataPreprocessor
from model_trainer import ModelTrainer
from evaluator import ModelEvaluator
from model_serializer import ModelSerializer
from data_sampler import DataSampler
from missing_value_handler import MissingValueHandler
from leakage_detector import DataLeakageDetector
from concept_drift_detector import ConceptDriftDetector
from threshold_optimizer import ThresholdOptimizer
from adversarial_tester import AdversarialRobustnessTester
from model_registry import ModelRegistry
from interaction_generator import FeatureInteractionGenerator
from anomaly_scorer import EnsembleAnomalyScorer
from pipeline_profiler import PipelineProfiler
from data_contract import DataContract, ColumnContract


class TestMLPipeline(unittest.TestCase):
    """Unit tests for ML components."""

    def test_data_loader_synthetic(self):
        df = DataLoader.generate_synthetic_dataset(n_samples=100, n_features=4, save_to_disk=False)
        self.assertEqual(len(df), 100)
        self.assertIn("target", df.columns)
        self.assertIn("region", df.columns)

    def test_data_sampler(self):
        df = DataLoader.generate_synthetic_dataset(n_samples=100, n_features=4, save_to_disk=False)
        sampled = DataSampler.reservoir_sample(df, sample_size=30, random_state=42)
        self.assertEqual(len(sampled), 30)

        stratified = DataSampler.stratified_quantile_sample(df, "target", n_bins=4, fraction=0.5)
        self.assertGreater(len(stratified), 0)

    def test_missing_value_handler(self):
        df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0], "b": ["x", "y", np.nan, "x"]})
        handler = MissingValueHandler(strategy="knn", n_neighbors=2)
        imputed = handler.fit_transform(df, add_indicators=True)
        self.assertEqual(imputed["a"].isnull().sum(), 0)
        self.assertEqual(imputed["b"].isnull().sum(), 0)
        self.assertIn("a_is_missing", imputed.columns)

    def test_leakage_detector(self):
        train = pd.DataFrame({"f1": [1, 2, 3], "target": [10, 20, 30]})
        test = pd.DataFrame({"f1": [2, 4, 5], "target": [20, 40, 50]})
        audit = DataLeakageDetector.audit_dataset(train, test, target_column="target")
        self.assertIn("is_clean", audit)
        self.assertEqual(audit["overlap_audit"]["duplicate_count"], 1)

    def test_concept_drift_detector(self):
        detector = ConceptDriftDetector(threshold=5.0)
        y_true = np.ones(50) * 10
        y_pred = np.ones(50) * 10
        result = detector.evaluate_batch(y_true, y_pred)
        self.assertEqual(result["drift_event_count"], 0)

    def test_threshold_optimizer(self):
        y_true = np.array([0, 1, 0, 1, 1, 0])
        y_probs = np.array([0.1, 0.9, 0.4, 0.8, 0.6, 0.2])
        res = ThresholdOptimizer.optimize_fbeta(y_true, y_probs, beta=1.0)
        self.assertGreaterEqual(res["best_score"], 0.5)

    def test_feature_interactions(self):
        df = pd.DataFrame({"feat_1": [2.0, 4.0], "feat_2": [3.0, 5.0]})
        gen = FeatureInteractionGenerator(include_products=True, include_ratios=True)
        transformed = gen.fit_transform(df)
        self.assertIn("feat_1_x_feat_2", transformed.columns)
        self.assertIn("feat_1_div_feat_2", transformed.columns)

    def test_data_contract(self):
        contract = DataContract([
            ColumnContract(name="feature_1", dtype="numeric", min_value=0.0),
            ColumnContract(name="region", dtype="string", allowed_values=["North", "South", "East", "West"]),
        ])
        df = pd.DataFrame({"feature_1": [5.0, 10.0], "region": ["North", "South"]})
        validation = contract.validate(df)
        self.assertTrue(validation["is_valid"])

    def test_pipeline_profiler(self):
        profiler = PipelineProfiler()
        with profiler.track_stage("unit_test_stage"):
            _ = [x**2 for x in range(1000)]
        report = profiler.get_summary_report()
        self.assertIn("unit_test_stage", report)


if __name__ == "__main__":
    unittest.main()
