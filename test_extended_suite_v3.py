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


if __name__ == "__main__":
    unittest.main()
