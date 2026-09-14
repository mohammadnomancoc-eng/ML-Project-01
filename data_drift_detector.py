"""Data Drift and Distribution Shift Detection Module."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from scipy.stats import ks_2samp, wasserstein_distance
from logger import get_logger

logger = get_logger("DriftDetector")


class DataDriftDetector:
    """Detects statistical covariate shift between baseline and current data distributions."""

    def __init__(self, alpha: float = 0.05, psi_threshold: float = 0.2):
        self.alpha = alpha
        self.psi_threshold = psi_threshold

    @staticmethod
    def calculate_psi(baseline: np.ndarray, current: np.ndarray, num_buckets: int = 10) -> float:
        """Calculates Population Stability Index (PSI) between two distributions."""
        quantiles = np.linspace(0, 100, num_buckets + 1)
        bins = np.percentile(baseline, quantiles)
        bins[0] = -np.inf
        bins[-1] = np.inf

        base_counts = np.histogram(baseline, bins=bins)[0]
        curr_counts = np.histogram(current, bins=bins)[0]

        base_pct = np.where(base_counts == 0, 1e-4, base_counts) / len(baseline)
        curr_pct = np.where(curr_counts == 0, 1e-4, curr_counts) / len(current)

        psi = np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct))
        return float(psi)

    def detect_drift(self, baseline_df: pd.DataFrame, current_df: pd.DataFrame) -> Dict[str, Any]:
        """Runs KS-test and PSI on all continuous features."""
        logger.info("Running statistical data drift analysis...")
        numeric_cols = baseline_df.select_dtypes(include=[np.number]).columns

        feature_reports = {}
        drifted_features = []

        for col in numeric_cols:
            if col not in current_df.columns:
                continue

            base_vals = baseline_df[col].dropna().values
            curr_vals = current_df[col].dropna().values

            # KS test
            ks_stat, p_value = ks_2samp(base_vals, curr_vals)
            # PSI
            psi_val = self.calculate_psi(base_vals, curr_vals)
            # Wasserstein distance
            w_dist = wasserstein_distance(base_vals, curr_vals)

            is_drifted = (p_value < self.alpha) or (psi_val > self.psi_threshold)
            if is_drifted:
                drifted_features.append(col)

            feature_reports[col] = {
                "ks_statistic": round(float(ks_stat), 4),
                "p_value": round(float(p_value), 5),
                "psi": round(float(psi_val), 4),
                "wasserstein_distance": round(float(w_dist), 4),
                "is_drifted": is_drifted,
            }

        report = {
            "total_features_checked": len(numeric_cols),
            "drifted_features_count": len(drifted_features),
            "drifted_features": drifted_features,
            "overall_drift_detected": len(drifted_features) > 0,
            "details": feature_reports,
        }

        logger.info(f"Drift check complete: {len(drifted_features)}/{len(numeric_cols)} features shifted.")
        return report
