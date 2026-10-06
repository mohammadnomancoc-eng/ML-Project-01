"""Unit tests for ML-Project-01 v2.2 additions."""

import unittest
import numpy as np
import pandas as pd
from data_quality_auditor import DataQualityAuditor
from distribution_drift import DistributionDriftEstimator
from data_slice_miner import DataSliceMiner
from residual_analyzer import ResidualDiagnosticAnalyzer
from cloud_cost_optimizer import CloudCostOptimizer


class TestExtendedSuiteV3(unittest.TestCase):
    """Tests for v2.2 modules."""

    def test_data_quality_auditor(self):
        df = pd.DataFrame({"num": [1.0, 2.0, np.nan, 4.0], "cat": ["A", "B", "A", "B"]})
        res = DataQualityAuditor.audit_quality(df)
        self.assertIn("completeness_score", res)
        self.assertEqual(res["total_rows"], 4)

    def test_distribution_drift(self):
        ref = pd.DataFrame({"val": np.random.normal(0, 1, 100)})
        curr = pd.DataFrame({"val": np.random.normal(0, 1, 100)})
        estimator = DistributionDriftEstimator(drift_threshold=0.5)
        res = estimator.calculate_feature_drift(ref, curr, ["val"])
        self.assertIn("drift_detected", res)

    def test_data_slice_miner(self):
        df = pd.DataFrame({"category": ["X", "X", "Y", "Y"] * 10, "num": range(40)})
        y_true = pd.Series(np.ones(40) * 10)
        y_pred = np.ones(40) * 12
        slices = DataSliceMiner.mine_worst_slices(df, y_true, y_pred, min_cohort_size=5)
        self.assertGreaterEqual(len(slices), 1)

    def test_residual_diagnostics(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.9, 5.1])
        diag = ResidualDiagnosticAnalyzer.audit_residuals(y_true, y_pred)
        self.assertIn("durbin_watson_stat", diag)
        self.assertIn("is_homoscedastic", diag)

    def test_cloud_cost_optimizer(self):
        recommendation = CloudCostOptimizer.recommend_instance(peak_qps_target=50.0)
        self.assertIn("recommended_instance", recommendation)
        self.assertGreater(recommendation["estimated_monthly_cost_usd"], 0)

    def test_psi_and_cost_projection(self):
        from cost_analyzer import BusinessCostAnalyzer
        ref = np.random.normal(0, 1, 100)
        curr = np.random.normal(0.1, 1, 100)
        psi = DistributionDriftEstimator.calculate_population_stability_index(ref, curr)
        self.assertIsInstance(psi, float)

        proj = BusinessCostAnalyzer.project_annual_cloud_cost(100.0, 0.10)
        self.assertEqual(proj["base_annual_cost_usd"], 1200.0)
        self.assertEqual(proj["projected_next_year_usd"], 1320.0)

    def test_data_cleaner_and_multiregion(self):
        from data_cleaner import DataCleaner
        clean_text = DataCleaner.strip_html_tags("<p>Hello <b>World</b></p>")
        self.assertEqual(clean_text, "Hello World")

        reg_costs = CloudCostOptimizer.estimate_multiregion_cost(100.0, ["us-east-1", "eu-west-1"])
        self.assertIn("total_multiregion_usd", reg_costs)

    def test_v2_6_helpers(self):
        from data_sampler import DataSampler
        from feature_hash_encoder import FeatureHashEncoder
        from model_calibrator import ModelCalibrator
        from threshold_optimizer import ThresholdOptimizer

        orig = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        samp = pd.DataFrame({"a": [1, 2]})
        stats = DataSampler.get_sampling_stats(orig, samp)
        self.assertEqual(stats["original_rows"], 5)
        self.assertEqual(stats["sampled_rows"], 2)

        hasher = FeatureHashEncoder(n_features=8)
        c_rate = hasher.estimate_collision_rate(["feat_a", "feat_b", "feat_c"])
        self.assertIsInstance(c_rate, float)

        brier = ModelCalibrator.brier_score(np.array([1, 0, 1]), np.array([0.9, 0.1, 0.8]))
        self.assertLess(brier, 0.1)

        curve = ThresholdOptimizer.generate_cost_curve(np.array([1, 0, 1]), np.array([0.9, 0.1, 0.8]), n_points=5)
        self.assertEqual(len(curve), 5)

    def test_v2_7_helpers(self):
        from dataset_fingerprint import DatasetFingerprinter
        from leakage_detector import DataLeakageDetector
        from sample_weigher import SampleWeigher
        from tabular_synthesizer import TabularCopulaSynthesizer

        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "target": [10.0, 20.0, 30.0, 40.0, 50.0]})
        q_fp = DatasetFingerprinter.compute_quantile_fingerprint(df)
        self.assertIn("x", q_fp)

        dup_leak = DataLeakageDetector.detect_duplicate_target_leakage(df, "target")
        self.assertEqual(len(dup_leak), 0)

        w = np.array([0.01, 1.0, 100.0])
        w_clipped = SampleWeigher.clip_extreme_weights(w, 0.1, 10.0)
        self.assertAlmostEqual(np.mean(w_clipped), 1.0, places=3)

        sim = TabularCopulaSynthesizer.get_correlation_similarity_score(df, df)
        self.assertEqual(sim, 1.0)


if __name__ == "__main__":
    unittest.main()



